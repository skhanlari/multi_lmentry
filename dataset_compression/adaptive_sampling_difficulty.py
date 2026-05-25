from pathlib import Path
import json
import logging

from .config import CompressionConfig
from .adaptive_sampling import (
    extract_task_data,
    compute_full_scores,
    select_best_overall,
)
from .adaptive_sampling import evaluate_sampling_difficulty

logger = logging.getLogger(__name__)


def run_difficulty_only(cfg: CompressionConfig, mode="mid"):

    output_dir = cfg.output_dir / "adaptive_results_difficulty"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "language": cfg.language,
        "technique": f"difficulty_{mode}",
        "tasks": {}
    }

    lang_dir = cfg.predictions_root / cfg.language
    task_names = [d.name for d in lang_dir.iterdir() if d.is_dir()]

    for task_name in sorted(task_names):

        example_ids, prompts, model_names = extract_task_data(
            cfg.predictions_root, cfg.language, task_name
        )

        full_scores = compute_full_scores(
            cfg.predictions_root,
            cfg.language,
            task_name,
            example_ids,
            model_names
        )

        tech_result = evaluate_sampling_difficulty(
            cfg.predictions_root,
            cfg.language,
            task_name,
            example_ids,
            model_names,
            full_scores,
            cfg.ratios,
            mode=mode
        )

        best = select_best_overall(
            {tech_result.technique: tech_result},
            cfg.spearman_thr,
            cfg.pearson_thr
        )

        results["tasks"][task_name] = {
            "best": best,
            "n_examples": len(example_ids),
        }

    out_file = output_dir / f"adaptive_results_diff_{cfg.language}.json"

    out_file.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return out_file
