import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path


BASE_DIR = Path("outputs_compression/adaptive_results_difficulty")

DESIRED_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]


DIFFICULTY_TECHNIQUES = [
    "difficulty_easy",
    "difficulty_mid",
    "difficulty_hard",
    "difficulty_stratified",
]


def plot_task_ratio_heatmap(language: str, technique: str):
    """
    Plot heatmap of Spearman correlation per task and ratio.
    Supports random, cluster_dedup, and difficulty techniques.
    """

    # Choose correct results file
    if technique.startswith("difficulty"):
        path = BASE_DIR / f"adaptive_results_{language}.json"

    if not path.exists():
        print(f"File not found: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    matrix = []
    available_ratios = None

    for task_name, task_data in data["tasks"].items():
        if "error" in task_data:
            continue

        if technique not in task_data["techniques"]:
            continue

        ratio_data = task_data["techniques"][technique]["ratios"]

        ratios = sorted(
            float(r) for r in ratio_data.keys()
            if float(r) in DESIRED_RATIOS
        )

        if not ratios:
            continue

        values = [
            ratio_data[str(r)]["spearman_aggregated"]
            for r in ratios
        ]

        tasks.append(task_name)
        matrix.append(values)
        available_ratios = ratios

    if not matrix:
        print(f"No data found for {language} – {technique}")
        return

    matrix = np.array(matrix)

    plt.figure(figsize=(10, max(6, len(tasks) * 0.35)))

    sns.heatmap(
        matrix,
        xticklabels=[f"{int(r*100)}%" for r in available_ratios],
        yticklabels=tasks,
        cmap="viridis",
        vmin=0.90,
        vmax=1.0,
        cbar_kws={"label": "Spearman"}
    )

    plt.title(f"Task × Ratio Spearman – {language.upper()} ({technique})")
    plt.xlabel("Compression ratio")
    plt.ylabel("Task")
    plt.tight_layout()

    output_dir = Path("dataset_compression/plots/heatmap/heatmap_difficulty")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_dir / f"heatmap_{language}_{technique}.png")
    plt.close()

    print(f"Saved heatmap for {language} – {technique}")



if __name__ == "__main__":

    plot_task_ratio_heatmap("en", "difficulty_easy")
    plot_task_ratio_heatmap("en", "difficulty_mid")
    plot_task_ratio_heatmap("en", "difficulty_hard")
    plot_task_ratio_heatmap("en", "difficulty_stratified")


# run: python -m dataset_compression.plots.heatmap.heatmap_difficulty.heatmap_difficulty