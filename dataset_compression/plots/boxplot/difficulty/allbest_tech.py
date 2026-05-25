import json
import matplotlib.pyplot as plt
import numpy as np
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


def collect_counts():

    results = {lang: {tech: 0 for tech in DIFFICULTY_TECHNIQUES}
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

            if tech in DIFFICULTY_TECHNIQUES:
                techniques.append(tech)

        counter = Counter(techniques)

        for tech in DIFFICULTY_TECHNIQUES:
            results[lang][tech] = counter.get(tech, 0)

    return results


def plot_grouped_difficulty_dominance():

    counts = collect_counts()

    x = np.arange(len(LANGUAGES))
    width = 0.18

    plt.figure(figsize=(14, 6))

    for i, technique in enumerate(DIFFICULTY_TECHNIQUES):

        values = [counts[lang][technique] for lang in LANGUAGES]

        offset = (i - 1.5) * width

        bars = plt.bar(
            x + offset,
            values,
            width=width,
            label=technique.replace("difficulty_", ""),
            color=TECHNIQUE_COLORS[technique],
            edgecolor="black",
        )

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
    plt.title("Difficulty technique dominance across languages")

    plt.legend(title="Technique")

    plt.ylim(0, max(
        max(counts[lang].values()) for lang in LANGUAGES
    ) + 5)

    plt.tight_layout()

    output_path = Path(
        "dataset_compression/plots/boxplot/difficulty/difficulty_technique_dominance_grouped.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    plt.close()


if __name__ == "__main__":
    plot_grouped_difficulty_dominance()
