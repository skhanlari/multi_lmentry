import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

BASE_DIR = Path("outputs_compression")

LANGUAGES = ["en", "it", "de", "es", "ca", "gl", "eu", "ko", "pt_br"]

def plot_technique_dominance_all_languages():
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
    axes = axes.flatten()

    for ax, lang in zip(axes, LANGUAGES):
        path = BASE_DIR / f"adaptive_results_{lang}.json"

        if not path.exists():
            ax.set_title(f"{lang.upper()} (missing)")
            ax.axis("off")
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        techniques = []

        for task_name, task_data in data["tasks"].items():
            if "error" in task_data:
                continue
            best = task_data.get("best", {})
            if "technique" in best:
                techniques.append(best["technique"])

        counter = Counter(techniques)

        labels = list(counter.keys())
        values = list(counter.values())

        ax.bar(labels, values, color=["tab:blue", "tab:orange"][:len(labels)])
        ax.set_title(lang.upper())
        ax.set_ylim(0, max(values) + 2)

        for i, v in enumerate(values):
            ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)

    fig.suptitle("Technique dominance across languages", fontsize=16)
    fig.text(0.04, 0.5, "Number of tasks", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    
    plt.savefig(f"dataset_compression/plots/boxplot/random_clustering/best_technique_dominance.png")
    print("Saved boxplot.")
    #plt.show()

if __name__ == "__main__":
    plot_technique_dominance_all_languages()

# run: python -m dataset_compression.plots.boxplot.random_clustering.allbest_tech