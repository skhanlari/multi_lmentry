import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = Path("outputs_compression")
DIFF_DIR = BASE_DIR / "adaptive_results_difficulty"

LANGUAGES = ["en", "it", "ko", "es", "de", "pt_br", "ca", "eu", "gl"]

TECHNIQUES = [
    "random",
    "cluster_dedup",
    "difficulty_easy",
    "difficulty_mid",
    "difficulty_hard",
    "difficulty_stratified",
]

FIXED_RATIOS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]
SPEARMAN_THRESHOLD = 0.90


def load_json_if_exists(path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def merge_task_techniques(task_main, task_diff):

    merged = {}

    if task_main:
        merged.update(task_main.get("techniques", {}))

    if task_diff:
        merged.update(task_diff.get("techniques", {}))

    return merged


def collect_global_means():

    results = {
        ratio: {tech: [] for tech in TECHNIQUES}
        for ratio in FIXED_RATIOS
    }

    for lang in LANGUAGES:

        main_path = BASE_DIR / f"adaptive_results_{lang}.json"
        diff_path = DIFF_DIR / f"adaptive_results_{lang}.json"

        main_data = load_json_if_exists(main_path)
        diff_data = load_json_if_exists(diff_path)

        if main_data is None and diff_data is None:
            print(f"Skipping {lang} (no files found)")
            continue

        tasks_main = main_data.get("tasks", {}) if main_data else {}
        tasks_diff = diff_data.get("tasks", {}) if diff_data else {}

        all_tasks = set(tasks_main.keys()) | set(tasks_diff.keys())

        for task in all_tasks:

            task_main = tasks_main.get(task)
            task_diff = tasks_diff.get(task)

            if (task_main and "error" in task_main) or \
               (task_diff and "error" in task_diff):
                continue

            techniques_data = merge_task_techniques(task_main, task_diff)

            for technique in TECHNIQUES:

                if technique not in techniques_data:
                    continue

                ratio_data = techniques_data[technique].get("ratios", {})

                for ratio in FIXED_RATIOS:
                    r_str = str(ratio)

                    if r_str in ratio_data:
                        spearman = ratio_data[r_str]["spearman_aggregated"]

                        if not np.isnan(spearman):
                            results[ratio][technique].append(spearman)

    mean_results = {
        ratio: {
            tech: np.mean(values) if values else np.nan
            for tech, values in tech_dict.items()
        }
        for ratio, tech_dict in results.items()
    }

    return mean_results


def plot_bar_comparison():

    mean_results = collect_global_means()

    n_ratios = len(FIXED_RATIOS)
    n_tech = len(TECHNIQUES)

    x = np.arange(n_ratios)
    width = 0.12

    plt.figure(figsize=(13, 6))

    for i, technique in enumerate(TECHNIQUES):

        values = [
            mean_results[ratio].get(technique, np.nan)
            for ratio in FIXED_RATIOS
        ]

        plt.bar(
            x + i * width,
            values,
            width,
            label=technique,
        )

    plt.axhline(
        SPEARMAN_THRESHOLD,
        linestyle="--",
        color="black",
        linewidth=2,
        label="Spearman ≥ 0.90"
    )

    plt.xticks(
        x + width * (n_tech - 1) / 2,
        [f"{int(r*100)}%" for r in FIXED_RATIOS]
    )

    plt.ylabel("Mean Spearman (All languages + tasks)")
    plt.xlabel("Compression ratio")
    plt.title("Global Comparison – All Sampling Techniques")
    plt.ylim(0.6, 1.02)

    plt.legend(loc="lower right", fontsize=9)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = Path("dataset_compression") / "plots" / "overall"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_dir / "bar_chart_comparison.pdf",
        bbox_inches="tight"
    )

    print("Saved bar_chart_comparison.pdf")

    plt.close()


if __name__ == "__main__":
    plot_bar_comparison()

# run: python -m dataset_compression.plots.overall.bar_chart_comparison