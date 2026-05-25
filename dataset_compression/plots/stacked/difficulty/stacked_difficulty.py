import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path("outputs_compression/adaptive_results_difficulty")

LANGUAGES = ["en", "de", "es", "it", "pt_br", "ca", "gl", "eu", "ko"]

DIFFICULTY_TECHNIQUES = [
    "difficulty_easy",
    "difficulty_mid",
    "difficulty_hard",
    "difficulty_stratified",
]

TECH_COLORS = {
    "difficulty_easy": "#1b9e77",
    "difficulty_mid": "#d95f02",
    "difficulty_hard": "#7570b3",
    "difficulty_stratified": "#e7298a",
}


def collect_technique_distribution():

    distributions = {
        lang: defaultdict(int) for lang in LANGUAGES
    }

    for lang in LANGUAGES:

        path = BASE_DIR / f"adaptive_results_{lang}.json"
        if not path.exists():
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for task_data in data["tasks"].values():
            if "error" in task_data:
                continue

            best = task_data.get("best", {})
            technique = best.get("technique")

            if technique in DIFFICULTY_TECHNIQUES:
                distributions[lang][technique] += 1

    return distributions


def plot_stacked_difficulty_techniques():

    distributions = collect_technique_distribution()

    x = np.arange(len(LANGUAGES))
    bottom = np.zeros(len(LANGUAGES))

    plt.figure(figsize=(12, 6))

    for technique in DIFFICULTY_TECHNIQUES:

        values = [
            distributions[lang].get(technique, 0)
            for lang in LANGUAGES
        ]

        plt.bar(
            x,
            values,
            bottom=bottom,
            label=technique.replace("difficulty_", ""),
            color=TECH_COLORS[technique],
            edgecolor="black",
            linewidth=0.6,
        )

        bottom += np.array(values)

    plt.xticks(x, LANGUAGES, rotation=45)
    plt.ylabel("Number of tasks")
    plt.xlabel("Language")
    plt.title("Technique selection distribution (Difficulty methods)")

    plt.legend(title="Technique", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()

    output_path = Path("dataset_compression/plots/stacked/difficulty/stacked_difficulty_technique_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    plt.close()


if __name__ == "__main__":
    plot_stacked_difficulty_techniques()
