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


# ---------------------------------------------------------
# Ratio grouping
# ---------------------------------------------------------
def group_ratio(r):
    percent = int(r * 100)

    if percent <= 2:
        return "1–2%"
    elif percent <= 7:
        return "3–7%"
    elif percent <= 15:
        return "10–15%"
    elif percent <= 30:
        return "20–30%"
    elif percent <= 60:
        return "40–60%"
    else:
        return "70–100%"


# ---------------------------------------------------------
# Collect distribution
# ---------------------------------------------------------
def collect_ratio_distribution():

    distributions = defaultdict(
        lambda: {lang: defaultdict(int) for lang in LANGUAGES}
    )

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
            ratio = best.get("ratio")
            technique = best.get("technique")

            if ratio is None or technique not in DIFFICULTY_TECHNIQUES:
                continue

            group = group_ratio(ratio)
            distributions[technique][lang][group] += 1

    return distributions


# ---------------------------------------------------------
# Plot stacked
# ---------------------------------------------------------
def plot_stacked_ratio_distribution_difficulty():

    distributions = collect_ratio_distribution()

    all_groups = ["1–2%", "3–7%", "10–15%", "20–30%", "40–60%", "70–100%"]

    colors = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
    ]

    techniques = [t for t in DIFFICULTY_TECHNIQUES if t in distributions]

    fig, axes = plt.subplots(
        1, len(techniques),
        figsize=(4 * len(techniques), 6),
        sharey=True
    )

    if len(techniques) == 1:
        axes = [axes]

    for ax, technique in zip(axes, techniques):

        bottom = np.zeros(len(LANGUAGES))

        for i, group in enumerate(all_groups):
            values = [
                distributions[technique][lang].get(group, 0)
                for lang in LANGUAGES
            ]

            ax.bar(
                LANGUAGES,
                values,
                bottom=bottom,
                label=group,
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
            )

            bottom += np.array(values)

        ax.set_title(technique.replace("difficulty_", ""), fontsize=13)
        ax.tick_params(axis="x", rotation=45)

    axes[-1].legend(
        title="Compression ratio group",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.suptitle(
        "Selected compression ratio distribution (Difficulty techniques)",
        fontsize=15
    )

    plt.subplots_adjust(right=0.85)

    output_path = Path("dataset_compression/plots/stacked/difficulty/stacked_difficulty_ratio_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=300)
    print(f"Saved figure to {output_path}")

    # plt.show()


if __name__ == "__main__":
    plot_stacked_ratio_distribution_difficulty()


# run: python -m dataset_compression.plots.stacked.difficulty.stacked_difficulty