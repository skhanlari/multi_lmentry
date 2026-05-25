import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import dataset_compression

adaptive_dir = Path("outputs_compression")
languages = ["en", "it", "ko", "es", "de", "pt_br", "ca", "eu", "gl"]

techniques = {
    "random": "-",          
    "cluster_dedup": "--",  
}

spearman_threshold = 0.90

colors = plt.cm.tab10.colors
lang_color = {
    lang: colors[i % len(colors)]
    for i, lang in enumerate(languages)
}

plt.figure(figsize=(12, 6))

for lang in languages:
    path = adaptive_dir / f"adaptive_results_{lang}.json"
    if not path.exists():
        print(f"Skipping {lang} (file not found)")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for technique, linestyle in techniques.items():
        # ratio -> list of Spearman values (across tasks)
        collector = {}

        for task_name, task_data in data["tasks"].items():
            if "error" in task_data:
                continue

            tech_data = task_data["techniques"].get(technique)
            if tech_data is None:
                continue

            for ratio_str, ratio_data in tech_data["ratios"].items():
                ratio = float(ratio_str)
                spearman = ratio_data["spearman_aggregated"]

                if not np.isnan(spearman):
                    collector.setdefault(ratio, []).append(spearman)

        if not collector:
            continue

        ratios = sorted(collector.keys())
        mean_spearman = [np.mean(collector[r]) for r in ratios]

        plt.plot(
            [r * 100 for r in ratios],
            mean_spearman,
            color=lang_color[lang],
            linestyle=linestyle,
            linewidth=2,
        )

plt.axhline(
    spearman_threshold,
    linestyle=":",
    color="black",
    linewidth=2.5,
    alpha=0.9,
    zorder=5,
)

plt.xlabel("Compression ratio (%)")
plt.ylabel("Spearman correlation")
plt.title("Sampling – Spearman correlation across languages")
plt.ylim(0.6, 1.01)
plt.grid(True, alpha=0.3)

lang_handles = [
    Line2D([0], [0], color=lang_color[lang], lw=3)
    for lang in languages
]
lang_legend = plt.legend(
    lang_handles,
    languages,
    title="Language",
    loc="lower right",
)

tech_handles = [
    Line2D([0], [0], color="black", linestyle=ls, lw=3)
    for ls in techniques.values()
]
tech_legend = plt.legend(
    tech_handles,
    techniques.keys(),
    title="Sampling technique",
    loc="lower center",
    ncol=2,
)

plt.gca().add_artist(lang_legend)

plt.tight_layout()

#plt.show()
# To save instead:
plots_dir = Path("dataset_compression") / "plots" / "overall"
plots_dir.mkdir(exist_ok=True)

plt.savefig(
    plots_dir / "overall_perTechnique_perLanguages.pdf",
    bbox_inches="tight"
)
plt.close()

# run: python -m dataset_compression.plots.overall.overall