from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from .samplers import sample_random_multi_seed, sample_cluster_dedup_multi_seed

import numpy as np

from .config import (
    CompressionConfig,
    RATIO_GRID,
    SAMPLING_TECHNIQUES,
    REFERENCE_MODEL,
    SPEARMAN_THRESHOLD,
    PEARSON_THRESHOLD,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    get_num_seeds,
    RANDOM_SEEDS,
)
from .samplers import (
    TextEmbedder,
    SamplerConfig,
    sample_random_multi_seed,
    sample_cluster_dedup_multi_seed,
)
from .eval_utils import (
    load_predictions,
    extract_example_ids,
    extract_prompts,
    extract_scores,
    compute_model_accuracy,
    spearman_correlation,
    pearson_correlation,
    aggregate_correlations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

Json = Dict[str, Any]


@dataclass
class RatioResult:
    """Results for a single ratio."""
    ratio: float
    n_selected: int
    seeds: Dict[int, List[str]]
    spearman_per_seed: Dict[int, float]
    pearson_per_seed: Dict[int, float]
    spearman_aggregated: float
    pearson_aggregated: float
    bin_info: Optional[Dict[int, Dict[str, List[str]]]] = None  # For stratified: seed -> {bin_name -> ids}


@dataclass  
class TechniqueResult:
    """Results for a sampling technique across all ratios."""
    technique: str
    ratio_results: Dict[float, RatioResult]
    # For cluster_dedup: store representative indices
    representatives: Optional[List[int]] = None
    n_clusters: Optional[int] = None


@dataclass
class TaskResult:
    """Results for a single task."""
    task_name: str
    n_examples: int
    full_scores: Dict[str, float]  # model -> accuracy
    technique_results: Dict[str, TechniqueResult]
    best_config: Dict[str, Any]

# Data Extraction
# ----------------------------------------------------------------------
def extract_task_data(
    predictions_root: Path,
    language: str,
    task_name: str,
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """
    Extract example IDs and prompts from reference model predictions.
    
    - Load predictions/<language>/<task>/ALIA-40b.json
    - Extract all example IDs and corresponding prompts
    
    Args:
        predictions_root: Root directory for predictions
        language: Language code
        task_name: Task name
    
    Returns:
        Tuple of (example_ids, prompts_dict, model_names)
    """
    task_dir = predictions_root / language / task_name
    
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    ref_path = task_dir / f"{REFERENCE_MODEL}.json"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference model predictions not found: {ref_path}")
    
    predictions = load_predictions(str(ref_path))
    example_ids = extract_example_ids(predictions)
    prompts = extract_prompts(predictions)
    
    # Get all model names
    model_names = [f.stem for f in task_dir.glob("*.json")]
    
    logger.info(f"  Extracted {len(example_ids)} examples, {len(model_names)} models")
    
    return example_ids, prompts, model_names


# ----------------------------------------------------------------------
# Sampling and Evaluation
# ----------------------------------------------------------------------
def compute_full_scores(
    predictions_root: Path,
    language: str,
    task_name: str,
    example_ids: Sequence[str],
    model_names: Sequence[str],
) -> Dict[str, float]:
    """
    Compute full dataset scores for all models.
    
    Args:
        predictions_root: Root directory for predictions
        language: Language code
        task_name: Task name
        example_ids: All example IDs
        model_names: List of model names
    
    Returns:
        Dict mapping model_name -> accuracy
    """
    task_dir = predictions_root / language / task_name
    
    scores = {}
    for model in model_names:
        pred_path = task_dir / f"{model}.json"
        if not pred_path.exists():
            scores[model] = float("nan")
            continue
        
        predictions = load_predictions(str(pred_path))
        scores[model] = compute_model_accuracy(predictions, example_ids)
    
    return scores


def compute_subset_scores(
    predictions_root: Path,
    language: str,
    task_name: str,
    subset_ids: Sequence[str],
    model_names: Sequence[str],
) -> Dict[str, float]:
    """Compute subset scores for all models."""
    task_dir = predictions_root / language / task_name
    
    scores = {}
    for model in model_names:
        pred_path = task_dir / f"{model}.json"
        if not pred_path.exists():
            scores[model] = float("nan")
            continue
        
        predictions = load_predictions(str(pred_path))
        scores[model] = compute_model_accuracy(predictions, subset_ids)
    
    return scores

def evaluate_sampling_random(
    predictions_root: Path,
    language: str,
    task_name: str,
    example_ids: Sequence[str],
    model_names: Sequence[str],
    full_scores: Dict[str, float],
    ratios: Sequence[float],
) -> TechniqueResult:
    """
    Evaluate random sampling.
    
    For each ratio:
    - Sample with multiple seeds (10 for r≤10%, 5 for r>10%)
    - Compute Spearman/Pearson for each seed
    - Aggregate correlations (median)
    """
    ratio_results = {}
    
    for ratio in ratios:
        logger.info(f"    Random @ {int(ratio*100)}%")

        seed_samples = sample_random_multi_seed(list(example_ids), ratio)
        
        spearman_per_seed = {}
        pearson_per_seed = {}
        
        for seed, selected_ids in seed_samples.items():
            subset_scores = compute_subset_scores(
                predictions_root, language, task_name,
                selected_ids, model_names
            )
            
            spearman = spearman_correlation(subset_scores, full_scores)
            pearson = pearson_correlation(subset_scores, full_scores)
            
            spearman_per_seed[seed] = spearman
            pearson_per_seed[seed] = pearson

        spearman_agg = aggregate_correlations(list(spearman_per_seed.values()))
        pearson_agg = aggregate_correlations(list(pearson_per_seed.values()))
        
        logger.info(
            f"      Spearman={spearman_agg:.3f} | Pearson={pearson_agg:.3f} | "
            f"n={len(list(seed_samples.values())[0])}"
        )
        
        ratio_results[ratio] = RatioResult(
            ratio=ratio,
            n_selected=len(list(seed_samples.values())[0]),
            seeds=seed_samples,
            spearman_per_seed=spearman_per_seed,
            pearson_per_seed=pearson_per_seed,
            spearman_aggregated=spearman_agg,
            pearson_aggregated=pearson_agg,
        )
    
    return TechniqueResult(
        technique="random",
        ratio_results=ratio_results,
    )


def evaluate_sampling_cluster_dedup(
    predictions_root: Path,
    language: str,
    task_name: str,
    example_ids: Sequence[str],
    prompts: Dict[str, str],
    model_names: Sequence[str],
    full_scores: Dict[str, float],
    ratios: Sequence[float],
    embedder: TextEmbedder,
    similarity_threshold: float,
) -> TechniqueResult:
    """
    Evaluate clustering-based sampling with deduplication.
    
    1. Encode prompts with multilingual embeddings
    2. Cluster using similarity threshold
    3. Retain one representative per cluster
    4. Apply random sampling on deduplicated pool (per ratio)
    """
    ratio_results = {}
    
    # Get prompt texts in order
    prompt_texts = [prompts.get(eid, "") for eid in example_ids]
    
    precomputed_dedup = None
    representatives = None
    n_clusters = None
    dedup_ids = None
    
    for ratio in ratios:
        logger.info(f"    Cluster_dedup @ {int(ratio*100)}%")

        seed_samples, representatives, dedup_ids = sample_cluster_dedup_multi_seed(
            list(example_ids),
            prompt_texts,
            ratio,
            embedder,
            similarity_threshold,
            precomputed_dedup=precomputed_dedup,
        )

        if precomputed_dedup is None:
            precomputed_dedup = (dedup_ids, representatives)
            n_clusters = len(representatives)
        
        spearman_per_seed = {}
        pearson_per_seed = {}
        
        for seed, selected_ids in seed_samples.items():
            subset_scores = compute_subset_scores(
                predictions_root, language, task_name,
                selected_ids, model_names
            )
            
            spearman = spearman_correlation(subset_scores, full_scores)
            pearson = pearson_correlation(subset_scores, full_scores)
            
            spearman_per_seed[seed] = spearman
            pearson_per_seed[seed] = pearson
        
        # Aggregate
        spearman_agg = aggregate_correlations(list(spearman_per_seed.values()))
        pearson_agg = aggregate_correlations(list(pearson_per_seed.values()))
        
        logger.info(
            f"      Spearman={spearman_agg:.3f} | Pearson={pearson_agg:.3f} | "
            f"n={len(list(seed_samples.values())[0])}"
        )
        
        ratio_results[ratio] = RatioResult(
            ratio=ratio,
            n_selected=len(list(seed_samples.values())[0]),
            seeds=seed_samples,
            spearman_per_seed=spearman_per_seed,
            pearson_per_seed=pearson_per_seed,
            spearman_aggregated=spearman_agg,
            pearson_aggregated=pearson_agg,
        )
    
    return TechniqueResult(
        technique="cluster_dedup",
        ratio_results=ratio_results,
        representatives=representatives,
        n_clusters=n_clusters,
    )


def evaluate_sampling_difficulty(
    predictions_root: Path,
    language: str,
    task_name: str,
    example_ids: Sequence[str],
    model_names: Sequence[str],
    full_scores: Dict[str, float],
    ratios: Sequence[float],
    mode: str = "mid",
) -> TechniqueResult:

    from .samplers import sample_difficulty_variance_multi_seed
    from .eval_utils import load_predictions

    ratio_results = {}

    task_dir = predictions_root / language / task_name
    predictions_by_model = {}

    for model in model_names:
        pred_path = task_dir / f"{model}.json"
        if pred_path.exists():
            predictions_by_model[model] = load_predictions(str(pred_path))

    for ratio in ratios:
        logger.info(f"    Difficulty-{mode} @ {int(ratio*100)}%")

        seed_samples, bin_info = sample_difficulty_variance_multi_seed(
            example_ids,
            predictions_by_model,
            ratio,
            mode=mode,
        )

        spearman_per_seed = {}
        pearson_per_seed = {}

        for seed, selected_ids in seed_samples.items():
            subset_scores = compute_subset_scores(
                predictions_root, language, task_name,
                selected_ids, model_names
            )

            spearman = spearman_correlation(subset_scores, full_scores)
            pearson = pearson_correlation(subset_scores, full_scores)

            spearman_per_seed[seed] = spearman
            pearson_per_seed[seed] = pearson

        spearman_agg = aggregate_correlations(list(spearman_per_seed.values()))
        pearson_agg = aggregate_correlations(list(pearson_per_seed.values()))

        ratio_results[ratio] = RatioResult(
            ratio=ratio,
            n_selected=len(list(seed_samples.values())[0]),
            seeds=seed_samples,
            spearman_per_seed=spearman_per_seed,
            pearson_per_seed=pearson_per_seed,
            spearman_aggregated=spearman_agg,
            pearson_aggregated=pearson_agg,
            bin_info=bin_info,
        )

    return TechniqueResult(
        technique=f"difficulty_{mode}",
        ratio_results=ratio_results,
    )






# Subset Selection
# ----------------------------------------------------------------------
def select_best_ratio(
    technique_result: TechniqueResult,
    spearman_thr: float = SPEARMAN_THRESHOLD,
    pearson_thr: float = PEARSON_THRESHOLD,
    relaxed_pearson_thr: float = 0.85,
) -> Dict[str, Any]:
    """
    Select the best ratio based on stability condition.
    
    Selection criteria:
    1. Traverse ratios in increasing order
    2. Select smallest ratio r* where Spearman ≥ 0.90 AND Pearson ≥ 0.90
       for TWO CONSECUTIVE ratios (stability condition)
    
    Fallback if thresholds never reached:
    - Select ratio that maximizes Spearman
    - Subject to Pearson ≥ relaxed threshold
    - Among ties, choose smallest ratio
    
    Args:
        technique_result: Results for a sampling technique
        spearman_thr: Spearman threshold (default 0.90)
        pearson_thr: Pearson threshold (default 0.90)
        relaxed_pearson_thr: Relaxed Pearson threshold for fallback
    
    Returns:
        Dict with best configuration details
    """
    ratios = sorted(technique_result.ratio_results.keys())
    
    prev_meets = False
    
    for i, ratio in enumerate(ratios):
        result = technique_result.ratio_results[ratio]
        spearman = result.spearman_aggregated
        pearson = result.pearson_aggregated
        
        meets_threshold = (
            not np.isnan(spearman) and spearman >= spearman_thr and
            not np.isnan(pearson) and pearson >= pearson_thr
        )
        
        if meets_threshold and prev_meets:
            # Found two consecutive ratios meeting thresholds
            # Return the previous ratio (smaller one)
            prev_ratio = ratios[i - 1]
            prev_result = technique_result.ratio_results[prev_ratio]
            
            return {
                "ratio": prev_ratio,
                "technique": technique_result.technique,
                "spearman": prev_result.spearman_aggregated,
                "pearson": prev_result.pearson_aggregated,
                "n_selected": prev_result.n_selected,
                "selection_method": "stability_condition",
                "seeds": prev_result.seeds,
                "bin_info": prev_result.bin_info,
            }
        
        prev_meets = meets_threshold
    
    # Fallback: max Spearman with relaxed Pearson constraint
    logger.warning("  Stability condition not met, using fallback selection")
    
    best_ratio = None
    best_spearman = -1.0
    
    for ratio in ratios:
        result = technique_result.ratio_results[ratio]
        spearman = result.spearman_aggregated
        pearson = result.pearson_aggregated
        
        if np.isnan(spearman) or np.isnan(pearson):
            continue
        
        if pearson >= relaxed_pearson_thr:
            if spearman > best_spearman or (
                spearman == best_spearman and 
                (best_ratio is None or ratio < best_ratio)
            ):
                best_spearman = spearman
                best_ratio = ratio
    
    if best_ratio is None:
        best_ratio = ratios[-1] if ratios else None
    
    if best_ratio is not None:
        result = technique_result.ratio_results[best_ratio]
        return {
            "ratio": best_ratio,
            "technique": technique_result.technique,
            "spearman": result.spearman_aggregated,
            "pearson": result.pearson_aggregated,
            "n_selected": result.n_selected,
            "selection_method": "fallback_max_spearman",
            "seeds": result.seeds,
            "bin_info": result.bin_info,
        }
    
    return {
        "ratio": None,
        "technique": technique_result.technique,
        "spearman": float("nan"),
        "pearson": float("nan"),
        "n_selected": 0,
        "selection_method": "none",
        "seeds": {},
        "bin_info": None,
    }


def select_best_overall(
    technique_results: Dict[str, TechniqueResult],
    spearman_thr: float = SPEARMAN_THRESHOLD,
    pearson_thr: float = PEARSON_THRESHOLD,
) -> Dict[str, Any]:
    """
    Select the best configuration across all techniques.
    
    Priority:
    1. Met stability condition with smallest ratio
    2. Best fallback (max Spearman with smallest ratio)
    """
    best_selections = {}
    for technique, result in technique_results.items():
        best_selections[technique] = select_best_ratio(
            result, spearman_thr, pearson_thr
        )
    
    # Find overall best
    stability_selections = {
        t: s for t, s in best_selections.items()
        if s["selection_method"] == "stability_condition"
    }
    
    if stability_selections:
        # Choose smallest ratio among stability selections
        best = min(
            stability_selections.values(),
            key=lambda s: (s["ratio"], -s["spearman"])
        )
    else:
        # All are fallback, choose max Spearman with smallest ratio
        valid = [s for s in best_selections.values() if s["ratio"] is not None]
        if valid:
            best = max(
                valid,
                key=lambda s: (s["spearman"], -s["ratio"])
            )
        else:
            # Handle empty selections
            logger.warning("No valid selections found across techniques.")
            return {"selection_method": "none", "ratio": None, "spearman": None, "pearson": None}
    
    return best


def run_adaptive_sampling(cfg: CompressionConfig) -> Path:

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
    
    logger.info(f"Starting adaptive sampling for {cfg.language}")
    logger.info(f"Tasks: {len(task_names)}")
    logger.info(f"Ratios: {len(cfg.ratios)} ({min(cfg.ratios):.0%} - {max(cfg.ratios):.0%})")
    logger.info(f"Techniques: {cfg.techniques}")
    
    # Initialize embedder
    embedder = TextEmbedder(cfg.embedding_model, device=cfg.device)
    
    # Results container
    results: Dict[str, Any] = {
        "language": cfg.language,
        "predictions_root": str(cfg.predictions_root),
        "reference_model": cfg.reference_model,
        "ratios": list(cfg.ratios),
        "techniques": list(cfg.techniques),
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
            example_ids, prompts, model_names = extract_task_data(
                cfg.predictions_root, cfg.language, task_name
            )
            
            full_scores = compute_full_scores(
                cfg.predictions_root, cfg.language, task_name,
                example_ids, model_names
            )
            
            technique_results = {}
            
            if "random" in cfg.techniques:
                logger.info("  Technique: random")
                technique_results["random"] = evaluate_sampling_random(
                    cfg.predictions_root, cfg.language, task_name,
                    example_ids, model_names, full_scores, cfg.ratios
                )
            
            if "cluster_dedup" in cfg.techniques:
                logger.info("  Technique: cluster_dedup")
                technique_results["cluster_dedup"] = evaluate_sampling_cluster_dedup(
                    cfg.predictions_root, cfg.language, task_name,
                    example_ids, prompts, model_names, full_scores,
                    cfg.ratios, embedder, cfg.similarity_threshold
                )

            if "difficulty_easy" in cfg.techniques:
                logger.info("  Technique: difficulty_easy")
                technique_results["difficulty_easy"] = evaluate_sampling_difficulty(
                    cfg.predictions_root,
                    cfg.language,
                    task_name,
                    example_ids,
                    model_names,
                    full_scores,
                    cfg.ratios,
                    mode="easy",
                )

            if "difficulty_mid" in cfg.techniques:
                logger.info("  Technique: difficulty_mid")
                technique_results["difficulty_mid"] = evaluate_sampling_difficulty(
                    cfg.predictions_root,
                    cfg.language,
                    task_name,
                    example_ids,
                    model_names,
                    full_scores,
                    cfg.ratios,
                    mode="mid",
                )

            if "difficulty_hard" in cfg.techniques:
                logger.info("  Technique: difficulty_hard")
                technique_results["difficulty_hard"] = evaluate_sampling_difficulty(
                    cfg.predictions_root,
                    cfg.language,
                    task_name,
                    example_ids,
                    model_names,
                    full_scores,
                    cfg.ratios,
                    mode="hard",
                )

            if "difficulty_stratified" in cfg.techniques:
                logger.info("  Technique: difficulty_stratified")
                technique_results["difficulty_stratified"] = evaluate_sampling_difficulty(
                    cfg.predictions_root,
                    cfg.language,
                    task_name,
                    example_ids,
                    model_names,
                    full_scores,
                    cfg.ratios,
                    mode="stratified",
                )

            # Select best configuration across ALL evaluated techniques
            best_config = select_best_overall(
                technique_results,
                cfg.spearman_thr,
                cfg.pearson_thr,
            )

            if 'technique' in best_config:
                log_msg = (
                    f"  ✔ Best: {best_config['technique']} @ "
                    f"{int(best_config['ratio']*100)}%, "
                    f"Spearman={best_config['spearman']:.3f}, "
                    f"Pearson={best_config['pearson']:.3f}"
                )
                # Add bin info for stratified technique
                if best_config.get('bin_info') and best_config['seeds']:
                    first_seed = list(best_config['seeds'].keys())[0]
                    bin_data = best_config['bin_info'].get(first_seed, {})
                    bin_counts = {k: len(v) for k, v in bin_data.items() if v}
                    if bin_counts:
                        log_msg += f" | Bins: {bin_counts}"
                logger.info(log_msg)
            else:
                logger.info("  ✔ No valid best configuration found.")

            if best_config["seeds"]:
                selected_seed = list(best_config["seeds"].keys())[0]
                selected_ids = list(best_config["seeds"].values())[0]
            else:
                selected_seed = None
                selected_ids = []

            # Extract bin info for stratified technique
            selected_bin_info = None
            if best_config.get("bin_info") and selected_seed is not None:
                selected_bin_info = best_config["bin_info"].get(selected_seed)

            task_result = {
                "n_examples": len(example_ids),
                "n_models": len(model_names),
                "full_scores": full_scores,
                "techniques": {},
                "best": {
                    "ratio": best_config["ratio"],
                    "technique": best_config["technique"],
                    "spearman": best_config["spearman"],
                    "pearson": best_config["pearson"],
                    "n_selected": best_config["n_selected"],
                    "selection_method": best_config["selection_method"],
                    "selected_seed": selected_seed,
                    "selected_ids": selected_ids,
                    "bin_counts": {k: len(v) for k, v in selected_bin_info.items()} if selected_bin_info else None,
                    "bin_ids": selected_bin_info,
                },
            }

            for tech_name, tech_result in technique_results.items():
                tech_data = {
                    "ratios": {}
                }
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
                
                task_result["techniques"][tech_name] = tech_data
            
            results["tasks"][task_name] = task_result
            
        except Exception as e:
            logger.error(f"Error processing task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results["tasks"][task_name] = {"error": str(e)}

    output_file = cfg.output_dir / f"adaptive_results_{cfg.language}.json"
    output_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    logger.info(f"\nSaved results to {output_file}")
    logger.info("Adaptive sampling completed successfully")
    
    return output_file
