import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from dataset_compression.plots.stacked.random_clustering.ratio_perTask import BASE_DIR


DESIRED_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]


def plot_task_ratio_heatmap(language: str, technique: str):
    path = BASE_DIR / f"adaptive_results_{language}.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tasks = []
    matrix = []

    for task_name, task_data in data["tasks"].items():
        if "error" in task_data:
            continue
        if technique not in task_data["techniques"]:
            continue

        ratio_data = task_data["techniques"][technique]["ratios"]

        available_ratios = sorted(
            float(r) for r in ratio_data.keys()
            if float(r) in DESIRED_RATIOS
        )

        if not available_ratios:
            continue

        values = [
            ratio_data[str(r)]["spearman_aggregated"]
            for r in available_ratios
        ]

        tasks.append(task_name)
        matrix.append(values)

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

    plt.savefig(f"dataset_compression/plots/heatmap/random_clustering/heatmap_{language}_{technique}.png")
    print("Saved heatmap.")


if __name__ == "__main__":
    plot_task_ratio_heatmap("ko", "random")
    plot_task_ratio_heatmap("ko", "cluster_dedup")


# run: python -m dataset_compression.plots.heatmap.random_clustering.heatmap