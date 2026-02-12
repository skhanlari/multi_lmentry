import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE_MAIN = Path("outputs_compression")
BASE_DIFF = BASE_MAIN / "adaptive_results_difficulty"

LANGUAGES = ["en", "it", "ko", "es", "de", "pt_br", "ca", "eu", "gl"]

ALL_TECHNIQUES = [
    "random",
    "cluster_dedup",
    "difficulty_easy",
    "difficulty_mid",
    "difficulty_hard",
    "difficulty_stratified",
]

tech_colors = {
    "random": "#1f77b4",
    "cluster_dedup": "#ff7f0e",
    "difficulty_easy": "#2ca02c",
    "difficulty_mid": "#d62728",
    "difficulty_hard": "#9467bd",
    "difficulty_stratified": "#8c564b",
}

plt.figure(figsize=(12, 6))

global_collector = {tech: {} for tech in ALL_TECHNIQUES}

for lang in LANGUAGES:


    main_path = BASE_MAIN / f"adaptive_results_{lang}.json"
    if main_path.exists():
        with open(main_path, "r", encoding="utf-8") as f:
            main_data = json.load(f)
    else:
        main_data = {"tasks": {}}

    diff_path = BASE_DIFF / f"adaptive_results_{lang}.json"
    if diff_path.exists():
        with open(diff_path, "r", encoding="utf-8") as f:
            diff_data = json.load(f)
    else:
        diff_data = {"tasks": {}}

    tasks = {}
    tasks.update(main_data["tasks"])
    for task, val in diff_data["tasks"].items():
        if task not in tasks:
            tasks[task] = val
        else:

            if "techniques" not in tasks[task]:
                tasks[task]["techniques"] = {}
            tasks[task]["techniques"].update(val.get("techniques", {}))

    for task_data in tasks.values():
        if "error" in task_data:
            continue

        for tech in ALL_TECHNIQUES:
            tech_data = task_data["techniques"].get(tech)
            if tech_data is None:
                continue

            for ratio_str, ratio_data in tech_data["ratios"].items():
                ratio = float(ratio_str)
                spearman = ratio_data["spearman_aggregated"]

                if not np.isnan(spearman):
                    global_collector[tech].setdefault(ratio, []).append(spearman)

for tech in ALL_TECHNIQUES:
    if not global_collector[tech]:
        continue

    ratios = sorted(global_collector[tech].keys())
    means = [np.mean(global_collector[tech][r]) for r in ratios]

    plt.plot(
        [r * 100 for r in ratios],
        means,
        label=tech,
        color=tech_colors.get(tech, "gray"),
        linewidth=3,
    )

plt.axhline(0.90, linestyle="--", color="black", linewidth=2)

plt.xlabel("Compression ratio (%)")
plt.ylabel("Mean Spearman (All languages + tasks)")
plt.title("Global Comparison – All Sampling Techniques")
plt.ylim(0.6, 1.01)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_dir = Path("dataset_compression") / "plots" / "overall"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(output_dir / "line_chart_comparison.pdf")
plt.close()

print("Saved line_chart_comparison.pdf")

# run: python -m dataset_compression.plots.overall.line_chart_comparison