import json
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path("outputs_compression_templates")

THRESHOLDS = ["0.9", "0.95", "0.99"]

TEMPLATES = ["template_1", "template_2", "template_3"]

LANGUAGES = ["en", "it", "de", "es", "ca", "gl", "eu", "pt_br", "ko"]

TECH_COLORS = {
    "random": "#1f77b4",
    "cluster_dedup": "#d62728",
    "difficulty_stratified": "#2ca02c",
}


def load_threshold_results(threshold):

    results = {}

    for lang in LANGUAGES:

        file = BASE_DIR / f"adaptive_results_templates_{threshold}_{lang}.json"

        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                results[lang] = json.load(f)

    return results


def extract_template_spearman(results, template):

    spearman_by_lang = {}
    technique_by_lang = {}

    for lang, data in results.items():

        values = []
        techniques = []

        for task, task_data in data["tasks"].items():

            # Check if 'templates' key exists in task_data
            if "templates" not in task_data:
                continue

            template_data = task_data["templates"].get(template)

            if not template_data:
                continue

            best = template_data["best"]

            if best["spearman"] is not None:

                values.append(best["spearman"])
                techniques.append(best["technique"])

        if values:

            spearman_by_lang[lang] = sum(values) / len(values)

            # most frequent technique
            tech_counts = defaultdict(int)

            for t in techniques:
                tech_counts[t] += 1

            technique_by_lang[lang] = max(tech_counts, key=tech_counts.get)

    return spearman_by_lang, technique_by_lang


def plot_threshold(threshold):

    results = load_threshold_results(threshold)

    threshold_value = float(threshold)

    for template in TEMPLATES:

        spearman_by_lang, technique_by_lang = extract_template_spearman(
            results, template
        )

        langs = [l for l in LANGUAGES if l in spearman_by_lang]

        values = [spearman_by_lang[l] for l in langs]

        colors = [
            TECH_COLORS.get(technique_by_lang[l], "gray")
            for l in langs
        ]

        plt.figure(figsize=(12, 6))

        plt.bar(langs, values, color=colors)

        # ADD THIS LINE → threshold line
        plt.axhline(
            y=threshold_value,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Threshold {threshold}"
        )

        plt.title(f"Threshold {threshold} | {template}")
        plt.xlabel("Language")
        plt.ylabel("Spearman Correlation")

        plt.ylim(0.85, 1.01)

        legend_elements = [
            plt.Line2D([0], [0], color=color, lw=6, label=tech)
            for tech, color in TECH_COLORS.items()
        ]

        # Add threshold to legend
        legend_elements.append(
            plt.Line2D([0], [0], color="black", lw=2, linestyle="--", label=f"threshold {threshold}")
        )

        plt.legend(handles=legend_elements, title="Technique")

        plt.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        out_file = BASE_DIR / f"plot_threshold_{threshold}_{template}.png"

        plt.savefig(out_file, dpi=300)

        print("Saved:", out_file)

        plt.show()



def main():

    for threshold in THRESHOLDS:

        print("\nProcessing threshold", threshold)

        plot_threshold(threshold)


if __name__ == "__main__":
    main()
