from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

Json = Dict[str, Any]


@dataclass
class CompressionApplicationConfig:
    
    tasks_root: Path            # lmentry/tasks/
    adaptive_results: Path      # Path to adaptive results JSON
    source_language: str        # Language of adaptive results
    target_languages: List[str] # Languages to apply compression to
    
    output_root: Path           # outputs_compression/compressed_sets/
    
    overwrite: bool = False

# Helper functions
# ----------------------------------------------------------------------
def _load_adaptive_results(path: Path) -> Dict[str, Any]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load adaptive results: {e}")
        raise


def _extract_best_indices(adaptive_results: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Extract best indices for each task from adaptive results.
    
    Returns:
        Dict mapping task_name -> list of indices
    """
    best_indices = {}
    
    for task_name, task_data in adaptive_results.get("tasks", {}).items():
        if "error" in task_data:
            logger.warning(f"Skipping {task_name} (error in adaptive results)")
            continue
        
        best = task_data.get("best", {})
        indices = best.get("selected_ids", [])
        
        if indices:
            best_indices[task_name] = list(indices)
        else:
            logger.warning(f"No indices found for {task_name}")
    
    return best_indices


def _load_task_json(tasks_root: Path, language: str, task_name: str) -> Dict[str, Any]:
    task_path = tasks_root / language / f"{task_name}.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")
    
    # Try UTF-8 first, then Latin-1 for non-English files
    try:
        return json.loads(task_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(task_path.read_text(encoding="latin-1"))


def _save_compressed_dataset_json(
    task_json: Dict[str, Any],
    indices: List[int],
    output_path: Path,
) -> None:
    from .samplers import _get_examples
    examples = _get_examples(task_json)
    
    # Extract selected examples
    selected_examples = []
    valid_indices = []
    for idx in indices:
        if idx < len(examples):
            selected_examples.append(examples[idx])
            valid_indices.append(idx)
    
    compressed = dict(task_json)
    
    # Store as list or dict based on original format
    if isinstance(task_json.get("examples"), dict):
        orig_examples = task_json["examples"]
        sorted_keys = sorted(orig_examples.keys(), key=lambda x: int(x) if x.isdigit() else x)
        
        # Map: for each valid index, get the original key and selected example
        compressed["examples"] = {
            sorted_keys[idx]: orig_examples[sorted_keys[idx]]
            for idx in valid_indices if idx < len(sorted_keys)
        }
    else:
        compressed["examples"] = selected_examples
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(compressed, f, ensure_ascii=False, indent=2)
    
    logger.debug(f"Saved {len(selected_examples)} examples to {output_path}")

# Compression using separate adaptive results for each language.
# ----------------------------------------------------------------------
def apply_compression_per_language(
    tasks_root: Path,
    adaptive_results_dir: Path,
    target_languages: List[str],
    output_root: Path,
    overwrite: bool = False,
) -> None:

    logger.info("Applying per-language compression")

    for lang in target_languages:
        logger.info(f"\nProcessing language: {lang}")

        # Look for adaptive results for this language
        adaptive_results_path = adaptive_results_dir / f"adaptive_results_{lang}.json"

        if not adaptive_results_path.exists():
            logger.warning(f"No adaptive results found for {lang}, skipping")
            continue

        adaptive_results = _load_adaptive_results(adaptive_results_path)

        # Extract best indices for each task
        best_indices = _extract_best_indices(adaptive_results)

        if not best_indices:
            logger.warning(f"No valid indices found in adaptive results for {lang}")
            continue

        logger.info(f"Found best indices for {len(best_indices)} tasks")

        lang_output_dir = output_root / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each task
        for task_name, indices in best_indices.items():
            output_path = lang_output_dir / f"{task_name}.json"

            # Skip if exists and not overwriting
            if output_path.exists() and not overwrite:
                logger.info(f"  Skipping {task_name} (already exists)")
                continue

            try:
                task_json = _load_task_json(tasks_root, lang, task_name)

                from .samplers import _get_examples
                examples = _get_examples(task_json)
                n_examples = len(examples)
                valid_indices = [i for i in indices if 0 <= i < n_examples]

                if len(valid_indices) != len(indices):
                    logger.warning(
                        f"  {task_name}: {len(indices) - len(valid_indices)} "
                        f"invalid indices (total examples: {n_examples})"
                    )

                if not valid_indices:
                    logger.warning(f"  Skipping {task_name} (no valid indices)")
                    continue

                _save_compressed_dataset_json(task_json, valid_indices, output_path)

                logger.info(
                    f"  ✔ {task_name}: {len(valid_indices)}/{n_examples} examples "
                    f"({len(valid_indices)/n_examples*100:.1f}%)"
                )

            except Exception as e:
                logger.error(f"  Error processing {task_name}: {e}")

    logger.info("\nPer-language compression completed")


def print_compression_statistics(
    adaptive_results_path: Path,
    output_root: Path,
) -> None:
    adaptive_results = _load_adaptive_results(adaptive_results_path)
    best_indices = _extract_best_indices(adaptive_results)
    
    print("\n" + "=" * 70)
    print("COMPRESSION STATISTICS")
    print("=" * 70)
    print(f"Source: {adaptive_results_path}")
    print(f"Language: {adaptive_results.get('language', 'unknown')}")
    print(f"\nTasks: {len(best_indices)}")

    ratios = []
    techniques = {}
    
    for task_name, task_data in adaptive_results.get("tasks", {}).items():
        if "error" in task_data:
            continue
        
        best = task_data.get("best", {})
        
        if "ratio" in best and best["ratio"] is not None:
            ratios.append(best["ratio"])
        
        if "technique" in best and best["technique"]:
            tech = best["technique"]
            techniques[tech] = techniques.get(tech, 0) + 1
    
    if ratios:
        print(f"\nCompression ratios:")
        print(f"  Min: {min(ratios)*100:.1f}%")
        print(f"  Max: {max(ratios)*100:.1f}%")
        print(f"  Mean: {np.mean(ratios)*100:.1f}%")
        print(f"  Median: {np.median(ratios)*100:.1f}%")
    
    if techniques:
        print(f"\nTechniques used:")
        for tech, count in sorted(techniques.items(), key=lambda x: -x[1]):
            print(f"  {tech}: {count} tasks ({count/len(best_indices)*100:.1f}%)")
    
    if output_root.exists():
        lang_dirs = [d for d in output_root.iterdir() if d.is_dir()]
        print(f"\nOutput languages: {len(lang_dirs)}")
        for lang_dir in sorted(lang_dirs):
            n_files = len(list(lang_dir.glob("*.json*")))
            print(f"  {lang_dir.name}: {n_files} tasks")
    
    print("=" * 70 + "\n")


import numpy as np
