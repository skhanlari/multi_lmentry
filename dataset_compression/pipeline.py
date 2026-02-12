from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

from .config import (
    CompressionConfig,
    RATIO_GRID,
    SAMPLING_TECHNIQUES,
    SPEARMAN_THRESHOLD,
    PEARSON_THRESHOLD,
    SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL,
    REFERENCE_MODEL,
)
from .adaptive_sampling import run_adaptive_sampling

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_compression_pipeline(
    predictions_root: Path,
    output_root: Path,
    language: str,
    task_names: Optional[List[str]] = None,
    ratios: tuple = RATIO_GRID,
    techniques: tuple = SAMPLING_TECHNIQUES,
    spearman_thr: float = SPEARMAN_THRESHOLD,
    pearson_thr: float = PEARSON_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    embedding_model: str = EMBEDDING_MODEL,
    device: str = "cpu",
) -> Path:

    logger.info("=" * 70)
    logger.info(f"DATASET COMPRESSION PIPELINE - {language.upper()}")
    logger.info("=" * 70)
    
    # Create configuration
    cfg = CompressionConfig(
        predictions_root=predictions_root,
        output_dir=output_root,
        language=language,
        task_names=task_names,
        ratios=ratios,
        techniques=techniques,
        spearman_thr=spearman_thr,
        pearson_thr=pearson_thr,
        similarity_threshold=similarity_threshold,
        embedding_model=embedding_model,
        device=device,
    )
    
    # Log configuration
    logger.info(f"\nConfiguration:")
    logger.info(f"  Predictions: {predictions_root}")
    logger.info(f"  Output: {output_root}")
    logger.info(f"  Reference model: {REFERENCE_MODEL}")
    logger.info(f"  Ratios: {len(ratios)} ({min(ratios):.0%} - {max(ratios):.0%})")
    logger.info(f"  Techniques: {techniques}")
    logger.info(f"  Thresholds: Spearman≥{spearman_thr}, Pearson≥{pearson_thr}")
    logger.info(f"  Similarity threshold: {similarity_threshold}")
    logger.info(f"  Embedding model: {embedding_model}")
    logger.info(f"  Device: {device}")
    
    # Run adaptive sampling
    logger.info("\n" + "-" * 70)
    logger.info("Running adaptive sampling...")
    logger.info("-" * 70)
    
    results_path = run_adaptive_sampling(cfg)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETED")
    logger.info("=" * 70)
    
    _print_summary(results_path)
    
    return results_path


def run_multi_language_pipeline(
    predictions_root: Path,
    output_root: Path,
    languages: Optional[List[str]] = None,
    task_names: Optional[List[str]] = None,
    ratios: tuple = RATIO_GRID,
    techniques: tuple = SAMPLING_TECHNIQUES,
    spearman_thr: float = SPEARMAN_THRESHOLD,
    pearson_thr: float = PEARSON_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    embedding_model: str = EMBEDDING_MODEL,
    device: str = "cpu",
) -> List[Path]:
    
    # Discover languages if not provided
    if languages is None:
        languages = [
            d.name for d in predictions_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
    
    if not languages:
        raise ValueError(f"No languages found in {predictions_root}")
    
    logger.info("=" * 70)
    logger.info("MULTI-LANGUAGE COMPRESSION PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Languages: {languages}")
    
    results_paths = []
    
    for lang in sorted(languages):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# Processing language: {lang}")
        logger.info(f"{'#' * 70}")
        
        try:
            result_path = run_compression_pipeline(
                predictions_root=predictions_root,
                output_root=output_root,
                language=lang,
                task_names=task_names,
                ratios=ratios,
                techniques=techniques,
                spearman_thr=spearman_thr,
                pearson_thr=pearson_thr,
                similarity_threshold=similarity_threshold,
                embedding_model=embedding_model,
                device=device,
            )
            results_paths.append(result_path)
        except Exception as e:
            logger.error(f"Failed to process language {lang}: {e}")
            import traceback
            traceback.print_exc()
    
    # Combined summary
    logger.info("\n" + "=" * 70)
    logger.info("ALL LANGUAGES COMPLETED")
    logger.info("=" * 70)
    
    for path in results_paths:
        _print_summary(path)
    
    return results_paths


def _print_summary(results_path: Path) -> None:
    """Print a summary of results."""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        language = results.get("language", "unknown")
        tasks = results.get("tasks", {})
        
        logger.info(f"\n--- Summary for {language} ---")
        logger.info(f"Total tasks: {len(tasks)}")
        
        successful = 0
        total_compression = []
        
        for task_name, task_data in tasks.items():
            if "error" in task_data:
                continue
            best = task_data.get("best", {})
            ratio = best.get("ratio")
            if ratio:
                successful += 1
                total_compression.append(ratio)
        
        logger.info(f"Successful: {successful}/{len(tasks)}")
        
        if total_compression:
            avg_ratio = sum(total_compression) / len(total_compression)
            logger.info(f"Average compression ratio: {avg_ratio:.1%}")
            logger.info(f"Min ratio: {min(total_compression):.1%}")
            logger.info(f"Max ratio: {max(total_compression):.1%}")
        
        logger.info(f"Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Failed to print summary: {e}")


def extract_compressed_sets(
    output_root: Path,
    predictions_root: Path = Path("predictions"),
    languages: Optional[List[str]] = None,
) -> None:
    """
    Extract best selected examples from adaptive results and save to compressed_sets.
    
    For each language and task, reads the best selected_ids from adaptive_results,
    looks up the inputs from predictions/<lang>/<task>/ALIA-40b.json, and saves to compressed_sets/<lang>/<task>.json.

    """
    output_root = Path(output_root)
    predictions_root = Path(predictions_root)
    compressed_sets_dir = output_root / "compressed_sets"
    
    # Discover languages from adaptive_results files if not provided
    if languages is None:
        results_files = list(output_root.glob("adaptive_results_*.json"))
        languages = [f.stem.replace("adaptive_results_", "") for f in results_files]
    
    if not languages:
        logger.error(f"No adaptive_results files found in {output_root}")
        return
    
    logger.info("=" * 70)
    logger.info("EXTRACTING COMPRESSED SETS")
    logger.info("=" * 70)
    logger.info(f"Languages: {languages}")
    
    for lang in sorted(languages):
        logger.info(f"\n--- Processing {lang} ---")
        
        # Load adaptive results
        results_file = output_root / f"adaptive_results_{lang}.json"
        if not results_file.exists():
            logger.warning(f"Results file not found: {results_file}")
            continue
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        tasks = results.get("tasks", {})
        
        # Create output directory
        lang_output_dir = compressed_sets_dir / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_count = 0
        for task_name, task_data in tasks.items():
            if "error" in task_data:
                continue
            
            best = task_data.get("best", {})
            selected_ids = best.get("selected_ids", [])
            
            if not selected_ids:
                logger.debug(f"No selected_ids for {task_name}")
                continue
            
            # Load ALIA-40b.json from predictions folder
            alia_file = predictions_root / lang / task_name / "ALIA-40b.json"
            if not alia_file.exists():
                logger.warning(f"ALIA-40b.json not found: {alia_file}")
                continue
            
            with open(alia_file, 'r', encoding='utf-8') as f:
                alia_content = json.load(f)
            
            # Extract only the selected examples with id and input
            compressed_examples = []
            for eid in selected_ids:
                if eid in alia_content:
                    compressed_examples.append({
                        "id": eid,
                        "input": alia_content[eid].get("input")
                    })
            
            # Save
            out_path = lang_output_dir / f"{task_name}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(compressed_examples, f, ensure_ascii=False, indent=2)
            
            saved_count += 1
            logger.info(f"Saved {len(compressed_examples)} examples to {out_path}")
        
        logger.info(f"Saved {saved_count} tasks for {lang}")
    
    logger.info("\n" + "=" * 70)
    logger.info("EXTRACTION COMPLETED")
    logger.info("=" * 70)


def extract_compressed_sets_difficulty(
    output_root: Path,
    predictions_root: Path,
    language: str,
):

    results_file = (
        output_root
        / "adaptive_results_difficulty"
        / f"adaptive_results_diff_{language}.json"
    )

    compressed_dir = (
        output_root
        / "compressed_sets_difficulty"
        / language
    )

    compressed_dir.mkdir(parents=True, exist_ok=True)

    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    for task_name, task_data in results["tasks"].items():
        selected_ids = task_data["best"].get("selected_ids", [])
        if not selected_ids:
            continue

        alia_file = predictions_root / language / task_name / "ALIA-40b.json"
        with open(alia_file, "r", encoding="utf-8") as f:
            alia_content = json.load(f)

        compressed = []
        for eid in selected_ids:
            if eid in alia_content:
                compressed.append({
                    "id": eid,
                    "input": alia_content[eid].get("input")
                })

        out_path = compressed_dir / f"{task_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(compressed, f, ensure_ascii=False, indent=2)




def main():
    parser = argparse.ArgumentParser(
        description="Dataset compression pipeline based on structure.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run for single language
  python -m dataset_compression.pipeline \\
    --predictions predictions/ \\
    --output outputs_compression/ \\
    --language en

  # Run for multiple languages
  python -m dataset_compression.pipeline \\
    --predictions predictions/ \\
    --output outputs_compression/ \\
    --languages en es ca de

  # Run with GPU
  python -m dataset_compression.pipeline \\
    --predictions predictions/ \\
    --output outputs_compression/ \\
    --language en \\
    --device cuda

  # Custom ratios
  python -m dataset_compression.pipeline \\
    --predictions predictions/ \\
    --output outputs_compression/ \\
    --language en \\
    --ratios 0.05 0.10 0.20 0.30

  # Extract compressed sets from existing results (all languages)
  python -m dataset_compression.pipeline \\
    --output outputs_compression/ \\
    --extract-only

  # Extract compressed sets for specific languages
  python -m dataset_compression.pipeline \\
    --output outputs_compression/ \\
    --languages en it \\
    --extract-only

Configuration (from structure.md):
  - Reference model: ALIA-40b
  - Ratio grid: 1-100% (19 values)
  - Techniques: random, cluster_dedup
  - Similarity threshold: 0.90
  - Correlation thresholds: Spearman≥0.90, Pearson≥0.90
  - Multi-seed: 10 seeds for r≤10%, 5 seeds for r>10%
        """
    )
    
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Root directory with model predictions (predictions/<language>/<task>/)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results"
    )
    
    # Language configuration
    parser.add_argument(
        "--language",
        type=str,
        help="Single language to process"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        help="Multiple languages to process (overrides --language)"
    )
    
    # Task filtering
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Specific tasks to process (default: all)"
    )
    
    # Sampling parameters
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        help="Compression ratios to try (default: structure.md grid)"
    )
    parser.add_argument(
        "--techniques",
        type=str,
        nargs="+",
        choices=[
            "random",
            "cluster_dedup",
            "difficulty_easy",
            "difficulty_mid",
            "difficulty_hard",
            "difficulty_stratified",
        ],
        default=list(SAMPLING_TECHNIQUES),
        help="Sampling techniques"
    )

    
    # Thresholds
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
    
    # Model and device
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Multilingual embedding model (default: {EMBEDDING_MODEL})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for embeddings (default: cpu)"
    )
    
    # Extract-only mode
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract compressed sets from existing adaptive results (no recomputation)"
    )
    
    args = parser.parse_args()
    
    # Handle extract-only mode
    if args.extract_only:
        if not args.output:
            parser.error("--output is required for --extract-only")
        languages = args.languages if args.languages else ([args.language] if args.language else None)
        extract_compressed_sets(
            output_root=args.output,
            languages=languages,
        )
        return

    if not args.language and not args.languages:
        parser.error("Either --language or --languages must be specified")
    
    if not args.predictions:
        parser.error("--predictions is required")
    
    ratios = tuple(args.ratios) if args.ratios else RATIO_GRID
    
    if args.languages:
        run_multi_language_pipeline(
            predictions_root=args.predictions,
            output_root=args.output,
            languages=args.languages,
            task_names=args.tasks,
            ratios=ratios,
            techniques=tuple(args.techniques),
            spearman_thr=args.spearman_thr,
            pearson_thr=args.pearson_thr,
            similarity_threshold=args.similarity_thr,
            embedding_model=args.embedding_model,
            device=args.device,
        )
    else:
        run_compression_pipeline(
            predictions_root=args.predictions,
            output_root=args.output,
            language=args.language,
            task_names=args.tasks,
            ratios=ratios,
            techniques=tuple(args.techniques),
            spearman_thr=args.spearman_thr,
            pearson_thr=args.pearson_thr,
            similarity_threshold=args.similarity_thr,
            embedding_model=args.embedding_model,
            device=args.device,
        )


if __name__ == "__main__":
    main()
