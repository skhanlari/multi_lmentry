from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

from .config import REFERENCE_MODEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def apply_template_compression(
    adaptive_results_file: Path,
    predictions_root: Path,
    output_root: Path,
):

    if not adaptive_results_file.exists():
        raise FileNotFoundError(
            f"Adaptive results file not found: {adaptive_results_file}"
        )

    with open(adaptive_results_file, "r", encoding="utf-8") as f:
        results: Dict[str, Any] = json.load(f)

    language = results.get("language")

    if not language:
        raise ValueError("Language not found in adaptive results")

    tasks = results.get("tasks", {})

    compressed_root = output_root/"compressed_templates"/"threshold_0.95"

    logger.info(f"Writing compressed templates to: {compressed_root}")

    total_tasks = 0
    total_templates = 0

    for task_name, task_data in tasks.items():

        template_results = task_data.get("templates")

        if template_results is None:

            logger.warning(f"Skipping task without templates: {task_name}")

            continue

        total_tasks += 1

        logger.info(f"Processing task: {task_name}")

        alia_file = (
            predictions_root
            / language
            / task_name
            / f"{REFERENCE_MODEL}.json"
        )

        if not alia_file.exists():

            logger.warning(f"Missing reference model file: {alia_file}")

            continue

        with open(alia_file, "r", encoding="utf-8") as f:
            alia_content = json.load(f)

        for template_name, template_data in template_results.items():

            best = template_data.get("best", {})

            selected_ids = best.get("selected_ids", [])

            if not selected_ids:

                logger.warning(
                    f"No selected IDs for {task_name} {template_name}"
                )

                continue

            compressed_examples = []

            for eid in selected_ids:

                if eid in alia_content:

                    compressed_examples.append({
                        "id": eid,
                        "input": alia_content[eid].get("input"),
                    })

            output_dir = (
                compressed_root
                / language
                / task_name
            )

            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{template_name}.json"

            with open(output_file, "w", encoding="utf-8") as f:

                json.dump(
                    compressed_examples,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            logger.info(
                f"Saved {len(compressed_examples)} examples → {output_file}"
            )

            total_templates += 1

    logger.info("")
    logger.info("========================================")
    logger.info("Template compression completed")
    logger.info(f"Tasks processed: {total_tasks}")
    logger.info(f"Templates saved: {total_templates}")
    logger.info(f"Output folder: {compressed_root}")
    logger.info("========================================")


# CLI ENTRY POINT
def main():

    import argparse

    parser = argparse.ArgumentParser(
        description="Apply template-based compression and save compressed datasets"
    )

    parser.add_argument(
        "--adaptive-results",
        required=True,
        type=Path,
        help="Path to adaptive_results_templates_0.90_<language>.json",
    )

    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("predictions"),
        help="Predictions root folder",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs_compression_templates"),
        help="Root folder where compressed_templates will be created",
    )

    args = parser.parse_args()

    apply_template_compression(
        adaptive_results_file=args.adaptive_results,
        predictions_root=args.predictions_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
