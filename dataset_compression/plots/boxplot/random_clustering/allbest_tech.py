import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter


BASE_DIR = Path("outputs_compression")

LANGUAGES = ["en", "it", "de", "es", "ca", "gl", "eu", "ko", "pt_br"]

TECHNIQUES = ["random", "cluster_dedup"]

TECH_COLORS = {
    "random": "#1f77b4",
    "cluster_dedup": "#ff7f0e",
}


def collect_counts():
    """
    Returns:
        dict[language][technique] = count
    """
    results = {lang: {tech: 0 for tech in TECHNIQUES}
               for lang in LANGUAGES}

    for lang in LANGUAGES:
        path = BASE_DIR / f"adaptive_results_{lang}.json"

        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        techniques = []

        for task_data in data["tasks"].values():
            if "error" in task_data:
                continue

            best = task_data.get("best", {})
            tech = best.get("technique")

            if tech in TECHNIQUES:
                techniques.append(tech)

        counter = Counter(techniques)

        for tech in TECHNIQUES:
            results[lang][tech] = counter.get(tech, 0)

    return results


def plot_grouped_technique_dominance():

    counts = collect_counts()

    x = np.arange(len(LANGUAGES))
    width = 0.30  # width of bars

    plt.figure(figsize=(13, 6))

    for i, technique in enumerate(TECHNIQUES):

        values = [counts[lang][technique] for lang in LANGUAGES]

        offset = (i - 0.5) * width

        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=technique,
            color=TECH_COLORS[technique],
            edgecolor="black",
        )

        # Add numbers above bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.3,
                    f"{int(height)}",
                    ha="center",
                    fontsize=9,
                )

    plt.xticks(x, [l.upper() for l in LANGUAGES])
    plt.ylabel("Number of tasks")
    plt.xlabel("Language")
    plt.title("Technique dominance across languages (Random vs Clustering)")

    plt.legend(title="Technique")

    plt.ylim(0, max(
        max(counts[lang].values()) for lang in LANGUAGES
    ) + 5)

    plt.tight_layout()

    output_path = Path(
        "dataset_compression/plots/boxplot/random_clustering/technique_dominance_grouped.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    plt.close()


if __name__ == "__main__":
    plot_grouped_technique_dominance()


# run: python -m dataset_compression.plots.boxplot.random_clustering.allbest_tech