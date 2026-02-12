import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

BASE_DIR = Path("outputs_compression/adaptive_results_difficulty")

LANGUAGES = ["en", "it", "de", "es", "ca", "gl", "eu", "ko", "pt_br"]

DIFFICULTY_TECHNIQUES = [
    "difficulty_easy",
    "difficulty_mid",
    "difficulty_hard",
    "difficulty_stratified",
]

TECHNIQUE_COLORS = {
    "difficulty_easy": "#2ca02c",
    "difficulty_mid": "#d62728",
    "difficulty_hard": "#9467bd",
    "difficulty_stratified": "#8c564b",
}


def plot_difficulty_dominance_all_languages():

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

        for task_data in data["tasks"].values():
            if "error" in task_data:
                continue

            best = task_data.get("best", {})
            tech = best.get("technique")

            if tech in DIFFICULTY_TECHNIQUES:
                techniques.append(tech)

        counter = Counter(techniques)

        if not counter:
            ax.set_title(f"{lang.upper()} (no data)")
            ax.axis("off")
            continue

        labels = sorted(counter.keys())
        values = [counter[l] for l in labels]
        colors = [TECHNIQUE_COLORS[l] for l in labels]

        bars = ax.bar(labels, values, color=colors, edgecolor="black")

        ax.set_title(lang.upper())
        ax.set_ylim(0, max(values) + 2)
        ax.tick_params(axis="x", rotation=45)

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.3,
                f"{int(height)}",
                ha="center",
                fontsize=9,
            )

    fig.suptitle("Difficulty technique dominance across languages", fontsize=16)
    fig.text(0.04, 0.5, "Number of tasks", va="center", rotation="vertical")

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])

    output_path = Path("dataset_compression/plots/boxplot/difficulty/difficulty_technique_dominance.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    # plt.show()


if __name__ == "__main__":
    plot_difficulty_dominance_all_languages()


# run: python -m dataset_compression.plots.boxplot.difficulty.allbest_tech