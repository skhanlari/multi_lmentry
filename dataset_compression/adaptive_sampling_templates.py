"""
Adaptive sampling per prompt template.

This module performs compression on each of the 3 prompt templates separately:
- Template 1: example IDs 0-999 (indices 0-999)
- Template 2: example IDs 1000-1999 (indices 1000-1999)
- Template 3: example IDs 2000-2999 (indices 2000-2999)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    RATIO_GRID,
    REFERENCE_MODEL,
    SPEARMAN_THRESHOLD,
    PEARSON_THRESHOLD,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    N_EXAMPLES,
)

from .samplers import TextEmbedder

from .eval_utils import (
    load_predictions,
    extract_example_ids,
    extract_prompts,
)

from .adaptive_sampling import (
    compute_full_scores,
    evaluate_sampling_random,
    evaluate_sampling_cluster_dedup,
    evaluate_sampling_difficulty,
    select_best_overall,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

TEMPLATE_RANGES = {
    "template_1": (0, 1000),
    "template_2": (1000, 2000),
    "template_3": (2000, 3000),
}

TEMPLATE_TECHNIQUES = ("random", "cluster_dedup", "difficulty_stratified")

OUTPUT_DIR_TEMPLATES = Path("outputs_compression_templates")


@dataclass
class TemplateCompressionConfig:
    
    predictions_root: Path
    output_dir: Path = OUTPUT_DIR_TEMPLATES
    language: str = "en"
    task_names: Optional[List[str]] = None
    ratios: Tuple[float, ...] = RATIO_GRID
    techniques: Tuple[str, ...] = TEMPLATE_TECHNIQUES
    spearman_thr: float = SPEARMAN_THRESHOLD
    pearson_thr: float = PEARSON_THRESHOLD
    similarity_threshold: float = SIMILARITY_THRESHOLD
    embedding_model: str = EMBEDDING_MODEL
    device: str = "cpu"
    embed_batch_size: int = 32
    reference_model: str = REFERENCE_MODEL


def extract_task_data_full(
    predictions_root: Path,
    language: str,
    task_name: str,
) -> Tuple[List[str], Dict[str, str], List[str]]:

    task_dir = predictions_root / language / task_name
    
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    ref_path = task_dir / f"{REFERENCE_MODEL}.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference model predictions not found: {ref_path}")
    
    predictions = load_predictions(str(ref_path))
    example_ids = extract_example_ids(predictions)
    prompts = extract_prompts(predictions)
    
    model_names = [f.stem for f in task_dir.glob("*.json")]
    
    return example_ids, prompts, model_names


def extract_template_subset(
    example_ids: List[str],
    prompts: Dict[str, str],
    template_name: str,
) -> Tuple[List[str], Dict[str, str]]:

    start_idx, end_idx = TEMPLATE_RANGES[template_name]
    
    template_ids = example_ids[start_idx:end_idx]
    template_prompts = {eid: prompts.get(eid, "") for eid in template_ids}
    
    return template_ids, template_prompts


def run_adaptive_sampling_templates(cfg: TemplateCompressionConfig) -> Path:

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    lang_dir = cfg.predictions_root / cfg.language
    if not lang_dir.exists():
        raise FileNotFoundError(f"Language directory not found: {lang_dir}")
    
    if cfg.task_names is None:
        task_names = [d.name for d in lang_dir.iterdir() if d.is_dir()]
    else:
        task_names = cfg.task_names
    
    if not task_names:
        raise ValueError(f"No tasks found for language: {cfg.language}")
    
    logger.info(f"Starting template-wise adaptive sampling for {cfg.language}")
    logger.info(f"Tasks: {len(task_names)}")
    logger.info(f"Templates: {list(TEMPLATE_RANGES.keys())}")
    logger.info(f"Techniques: {cfg.techniques}")
    logger.info(f"Ratios: {len(cfg.ratios)} ({min(cfg.ratios):.0%} - {max(cfg.ratios):.0%})")
    
    embedder = TextEmbedder(cfg.embedding_model, device=cfg.device)

    results: Dict[str, Any] = {
        "language": cfg.language,
        "predictions_root": str(cfg.predictions_root),
        "reference_model": cfg.reference_model,
        "ratios": list(cfg.ratios),
        "techniques": list(cfg.techniques),
        "templates": list(TEMPLATE_RANGES.keys()),
        "template_ranges": {k: list(v) for k, v in TEMPLATE_RANGES.items()},
        "thresholds": {
            "spearman": cfg.spearman_thr,
            "pearson": cfg.pearson_thr,
        },
        "similarity_threshold": cfg.similarity_threshold,
        "embedding_model": cfg.embedding_model,
        "tasks": {},
    }
    
    # Process each task
    for task_idx, task_name in enumerate(sorted(task_names), 1):
        logger.info(f"\n[{task_idx}/{len(task_names)}] Task: {task_name}")
        
        try:
            # Extract full task data
            all_example_ids, all_prompts, model_names = extract_task_data_full(
                cfg.predictions_root, cfg.language, task_name
            )
            
            logger.info(f"  Total examples: {len(all_example_ids)}, Models: {len(model_names)}")
            
            # Skip tasks that don't have exactly 3000 examples
            if len(all_example_ids) != N_EXAMPLES:
                logger.info(f"  Skipping task: {len(all_example_ids)} examples (expected {N_EXAMPLES})")
                results["tasks"][task_name] = {
                    "skipped": True,
                    "reason": f"Task has {len(all_example_ids)} examples, expected {N_EXAMPLES}",
                    "n_examples": len(all_example_ids),
                }
                continue
            
            task_result = {
                "n_examples_total": len(all_example_ids),
                "n_models": len(model_names),
                "templates": {},
            }
            
            # Process each template
            for template_name in TEMPLATE_RANGES.keys():
                logger.info(f"\n  Template: {template_name}")
                
                # Extract template subset
                template_ids, template_prompts = extract_template_subset(
                    all_example_ids, all_prompts, template_name
                )
                
                logger.info(f"    Examples: {len(template_ids)}")
                
                # Compute full scores for this template subset
                full_scores = compute_full_scores(
                    cfg.predictions_root, cfg.language, task_name,
                    template_ids, model_names
                )
                
                technique_results = {}
                
                # Random sampling
                if "random" in cfg.techniques:
                    logger.info("    Technique: random")
                    technique_results["random"] = evaluate_sampling_random(
                        cfg.predictions_root, cfg.language, task_name,
                        template_ids, model_names, full_scores, cfg.ratios
                    )
                
                # Cluster dedup
                if "cluster_dedup" in cfg.techniques:
                    logger.info("    Technique: cluster_dedup")
                    technique_results["cluster_dedup"] = evaluate_sampling_cluster_dedup(
                        cfg.predictions_root, cfg.language, task_name,
                        template_ids, template_prompts, model_names, full_scores,
                        cfg.ratios, embedder, cfg.similarity_threshold
                    )
                
                # Difficulty stratified
                if "difficulty_stratified" in cfg.techniques:
                    logger.info("    Technique: difficulty_stratified")
                    technique_results["difficulty_stratified"] = evaluate_sampling_difficulty(
                        cfg.predictions_root,
                        cfg.language,
                        task_name,
                        template_ids,
                        model_names,
                        full_scores,
                        cfg.ratios,
                        mode="stratified",
                    )
                
                # Select best configuration
                best_config = select_best_overall(
                    technique_results,
                    cfg.spearman_thr,
                    cfg.pearson_thr,
                )
                
                if 'technique' in best_config and best_config.get('ratio') is not None:
                    log_msg = (
                        f"    ✔ Best: {best_config['technique']} @ "
                        f"{int(best_config['ratio']*100)}%, "
                        f"Spearman={best_config['spearman']:.3f}, "
                        f"Pearson={best_config['pearson']:.3f}"
                    )
                    if best_config.get('bin_info') and best_config.get('seeds'):
                        first_seed = list(best_config['seeds'].keys())[0]
                        bin_data = best_config['bin_info'].get(first_seed, {})
                        bin_counts = {k: len(v) for k, v in bin_data.items() if v}
                        if bin_counts:
                            log_msg += f" | Bins: {bin_counts}"
                    logger.info(log_msg)
                else:
                    logger.info("    ✔ No valid best configuration found.")
                
                # Extract selected IDs
                if best_config.get("seeds"):
                    selected_seed = list(best_config["seeds"].keys())[0]
                    selected_ids = list(best_config["seeds"].values())[0]
                else:
                    selected_seed = None
                    selected_ids = []
                
                selected_bin_info = None
                if best_config.get("bin_info") and selected_seed is not None:
                    selected_bin_info = best_config["bin_info"].get(selected_seed)
                
                # Build template result
                template_result = {
                    "n_examples": len(template_ids),
                    "example_id_range": TEMPLATE_RANGES[template_name],
                    "full_scores": full_scores,
                    "techniques": {},
                    "best": {
                        "ratio": best_config.get("ratio"),
                        "technique": best_config.get("technique"),
                        "spearman": best_config.get("spearman"),
                        "pearson": best_config.get("pearson"),
                        "n_selected": best_config.get("n_selected"),
                        "selection_method": best_config.get("selection_method"),
                        "seed": selected_seed,
                        "selected_ids": selected_ids,
                        "bin_counts": {k: len(v) for k, v in selected_bin_info.items()} if selected_bin_info else None,
                        "bin_ids": selected_bin_info,
                    },
                }
                
                # Store technique results
                for tech_name, tech_result in technique_results.items():
                    tech_data = {"ratios": {}}
                    if tech_result.n_clusters is not None:
                        tech_data["n_clusters"] = tech_result.n_clusters
                    
                    for ratio, ratio_result in tech_result.ratio_results.items():
                        tech_data["ratios"][str(ratio)] = {
                            "n_selected": ratio_result.n_selected,
                            "spearman_aggregated": ratio_result.spearman_aggregated,
                            "pearson_aggregated": ratio_result.pearson_aggregated,
                            "spearman_per_seed": {
                                str(k): v for k, v in ratio_result.spearman_per_seed.items()
                            },
                            "pearson_per_seed": {
                                str(k): v for k, v in ratio_result.pearson_per_seed.items()
                            },
                        }
                    
                    template_result["techniques"][tech_name] = tech_data
                
                task_result["templates"][template_name] = template_result
            
            results["tasks"][task_name] = task_result
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results["tasks"][task_name] = {"error": str(e)}
    
    # Save results
    output_file = cfg.output_dir / f"adaptive_results_templates_0.9_{cfg.language}.json"
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    logger.info(f"\nSaved results to {output_file}")
    logger.info("Template-wise adaptive sampling completed successfully")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Run template-wise adaptive sampling for dataset compression"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        required=True,
        help="Language code (e.g., en, es, de)"
    )
    parser.add_argument(
        "--predictions-root", "-p",
        type=Path,
        default=Path("predictions"),
        help="Root directory for predictions (default: predictions)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=OUTPUT_DIR_TEMPLATES,
        help=f"Output directory (default: {OUTPUT_DIR_TEMPLATES})"
    )
    parser.add_argument(
        "--tasks", "-t",
        type=str,
        nargs="+",
        default=None,
        help="Specific tasks to process (default: all)"
    )
    parser.add_argument(
        "--spearman-thr",
        type=float,
        default=SPEARMAN_THRESHOLD,
        help=f"Spearman correlation threshold (default: {SPEARMAN_THRESHOLD})"
    )
    parser.add_argument(
        "--pearson-thr",
        type=float,
        default=PEARSON_THRESHOLD,
        help=f"Pearson correlation threshold (default: {PEARSON_THRESHOLD})"
    )
    parser.add_argument(
        "--similarity-thr",
        type=float,
        default=SIMILARITY_THRESHOLD,
        help=f"Cosine similarity threshold for clustering (default: {SIMILARITY_THRESHOLD})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embeddings (default: cpu)"
    )
    
    args = parser.parse_args()
    
    cfg = TemplateCompressionConfig(
        predictions_root=args.predictions_root,
        output_dir=args.output_dir,
        language=args.language,
        task_names=args.tasks,
        spearman_thr=args.spearman_thr,
        pearson_thr=args.pearson_thr,
        similarity_threshold=args.similarity_thr,
        device=args.device,
    )
    
    run_adaptive_sampling_templates(cfg)


if __name__ == "__main__":
    main()
