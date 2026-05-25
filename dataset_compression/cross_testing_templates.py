from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr, pearsonr

from dataset_compression.eval_utils import (
    load_predictions,
    extract_example_ids,
    extract_prompts,
    compute_model_accuracy,
)

from dataset_compression.adaptive_sampling import (
    evaluate_sampling_random,
    select_best_overall,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


REFERENCE_MODEL = "ALIA-40b"

TEMPLATE_RANGES = {
    "template_1": (0, 1000),
    "template_2": (1000, 2000),
    "template_3": (2000, 3000),
}

RATIO_GRID = (
    0.01, 0.02, 0.03, 0.05, 0.07,
    0.10, 0.12, 0.15, 0.18, 0.20,
    0.25, 0.30, 0.35, 0.40, 0.50,
    0.60, 0.70, 0.80, 1.00,
)


def compute_scores(pred_dir: Path, models: List[str], ids: List[str]) -> Dict[str, float]:
    scores = {}
    for m in models:
        file = pred_dir / f"{m}.json"
        if not file.exists():
            scores[m] = np.nan
            continue
        preds = load_predictions(str(file))
        scores[m] = compute_model_accuracy(preds, ids)
    return scores


def compute_correlations(a: Dict[str, float], b: Dict[str, float]):
    models = sorted(set(a) & set(b))
    x = [a[m] for m in models]
    y = [b[m] for m in models]

    valid = [(i, j) for i, j in zip(x, y) if not (np.isnan(i) or np.isnan(j))]
    if len(valid) < 2:
        return np.nan, np.nan

    x, y = zip(*valid)
    return spearmanr(x, y)[0], pearsonr(x, y)[0]


def compute_mse_list(a: Dict[str, float], b: Dict[str, float]) -> float:
    vals = [
        (a[m] - b[m]) ** 2
        for m in a
        if m in b and not (np.isnan(a[m]) or np.isnan(b[m]))
    ]
    return float(np.mean(vals)) if vals else np.nan


def compute_mse_single(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return float((a - b) ** 2)


def build_ratio_curve_from_technique_result(
    task_dir: Path,
    train_models: List[str],
    test_model: str,
    full_scores_train: Dict[str, float],
    full_test: float,
    technique_result,
    selected_seed: int = 42,
) -> Dict[str, dict]:
    """
    Build ratio-wise train/test metrics using one fixed seed (default=42).
    """
    ratio_curve = {}

    for ratio, ratio_result in sorted(technique_result.ratio_results.items()):
        if selected_seed not in ratio_result.seeds:
            logger.warning(
                f"Seed {selected_seed} not found for ratio={ratio}. "
                f"Available seeds: {list(ratio_result.seeds.keys())}"
            )
            continue

        selected_ids = ratio_result.seeds[selected_seed]

        # TRAIN metrics on train models
        subset_scores_train = {}
        for m in train_models:
            pred_path = task_dir / f"{m}.json"
            if not pred_path.exists():
                subset_scores_train[m] = np.nan
                continue
            preds = load_predictions(str(pred_path))
            subset_scores_train[m] = compute_model_accuracy(preds, selected_ids)

        train_mse = compute_mse_list(subset_scores_train, full_scores_train)

        # TEST metric on held-out model
        pred_path = task_dir / f"{test_model}.json"
        if not pred_path.exists():
            subset_test = np.nan
        else:
            preds = load_predictions(str(pred_path))
            subset_test = compute_model_accuracy(preds, selected_ids)

        test_mse = compute_mse_single(subset_test, full_test)

        ratio_curve[str(ratio)] = {
            "subset_size": len(selected_ids),
            "selected_seed": selected_seed,
            "train_spearman": ratio_result.spearman_per_seed.get(selected_seed, np.nan),
            "train_pearson": ratio_result.pearson_per_seed.get(selected_seed, np.nan),
            "train_spearman_aggregated": ratio_result.spearman_aggregated,
            "train_pearson_aggregated": ratio_result.pearson_aggregated,
            "train_mse": train_mse,
            "test_mse": test_mse,
        }

    return ratio_curve


def run_cross_testing_for_language(predictions_root: Path, output_file: Path, language: str):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        results = json.load(open(output_file, "r", encoding="utf-8"))
    else:
        results = {}

    if language in results:
        logger.info(f"Language {language} already exists -> overwriting")

    results[language] = {}

    language_dir = predictions_root / language
    if not language_dir.exists():
        raise ValueError(f"Language folder not found: {language_dir}")

    logger.info(f"\n=== Processing language: {language} ===")

    for task_dir in language_dir.iterdir():
        if not task_dir.is_dir():
            continue

        task_name = task_dir.name
        logger.info(f"\nTask: {task_name}")

        ref_file = task_dir / f"{REFERENCE_MODEL}.json"
        if not ref_file.exists():
            continue

        ref_preds = load_predictions(str(ref_file))
        example_ids = extract_example_ids(ref_preds)

        if len(example_ids) != 3000:
            logger.info(f"Skipping {task_name}: {len(example_ids)} examples")
            continue

        _ = extract_prompts(ref_preds)
        model_names = [f.stem for f in task_dir.glob("*.json")]

        results[language][task_name] = {}

        for template_name, (start, end) in TEMPLATE_RANGES.items():
            logger.info(f"  Template: {template_name}")
            template_ids = example_ids[start:end]

            results[language][task_name][template_name] = {}

            for test_model in model_names:
                logger.info(f"    Test model: {test_model}")

                train_models = [m for m in model_names if m != test_model]

                # Train full scores on full template
                full_scores_train = compute_scores(task_dir, train_models, template_ids)

                # Evaluate ratios using only train models
                technique_result = evaluate_sampling_random(
                    predictions_root=predictions_root,
                    language=language,
                    task_name=task_name,
                    example_ids=template_ids,
                    model_names=train_models,
                    full_scores=full_scores_train,
                    ratios=RATIO_GRID,
                )

                # Select best ratio using train models only
                best_config = select_best_overall({"random": technique_result})

                # Since you use one fixed seed, prefer seed 42 explicitly
                selected_seed = 42
                if best_config.get("seeds") and selected_seed in best_config["seeds"]:
                    selected_ids = best_config["seeds"][selected_seed]
                elif best_config.get("seeds"):
                    selected_seed = list(best_config["seeds"].keys())[0]
                    selected_ids = best_config["seeds"][selected_seed]
                else:
                    selected_ids = []

                # Final TRAIN metrics at chosen ratio
                subset_scores_train = compute_scores(task_dir, train_models, selected_ids)

                spearman, pearson = compute_correlations(
                    subset_scores_train,
                    full_scores_train,
                )

                mse_train = compute_mse_list(
                    subset_scores_train,
                    full_scores_train,
                )

                # Final TEST metric at chosen ratio
                full_test = compute_scores(task_dir, [test_model], template_ids)[test_model]
                subset_test = compute_scores(task_dir, [test_model], selected_ids)[test_model]
                mse_test = compute_mse_single(subset_test, full_test)

                # Ratio-wise train/test curve
                ratio_curve = build_ratio_curve_from_technique_result(
                    task_dir=task_dir,
                    train_models=train_models,
                    test_model=test_model,
                    full_scores_train=full_scores_train,
                    full_test=full_test,
                    technique_result=technique_result,
                    selected_seed=selected_seed,
                )

                results[language][task_name][template_name][test_model] = {
                    "train": {
                        "spearman": spearman,
                        "pearson": pearson,
                        "mse": mse_train,
                    },
                    "test": {
                        "mse": mse_test,
                    },
                    "ratio": best_config.get("ratio"),
                    "subset_size": len(selected_ids),
                    "selected_seed": selected_seed,
                    "ratio_curve": ratio_curve,
                }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSaved results for language: {language}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--language", required=True, help="Language to run (e.g., en)")
    parser.add_argument("--predictions-root", type=Path, default=Path("predictions"))
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("outputs_cross_testing/cross_testing_templates.json"),
    )

    args = parser.parse_args()

    run_cross_testing_for_language(
        predictions_root=args.predictions_root,
        output_file=args.output_file.parent / f"cross_testing_{args.language}.json",
        language=args.language,
    )


if __name__ == "__main__":
    main()