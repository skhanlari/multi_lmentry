import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
import os
import csv
from collections import defaultdict
import statistics
import numpy as np


MODELS_CLOSED = {
    "Llama-3.2-1B-Instruct": 1,
    "Qwen2.5-1.5B-Instruct": 1.5,
    "Llama-3.2-3B-Instruct": 3,
    "Qwen2.5-3B-Instruct": 3,
    "Llama-3.1-8B-Instruct": 8,
    "Qwen2.5-14B-Instruct": 14,
    "phi-4": 14,
}

MODELS_OPEN = {
    "EuroLLM-1.7B-Instruct": 1.7,
    "salamandra-2b-instruct": 2,
    "occiglot-7b-es-en-instruct": 7,
    "occiglot-7b-eu5-instruct": 7,
    "salamandra-7b-instruct": 7,
    "EuroLLM-9B-Instruct": 9,
}


PLOT_PARAM_LANG = {
    "closed": {"color": 'blue', "linestyle": 'dashed', "marker": 'o'},
    "open": {"color": 'red', "linestyle": 'dashed', "marker": '.'},
}

CODE_TO_LANG = {
    "en": "English",
    "es": "Spanish",
    "it": "Italian",
    "pt_br": "Portuguese (Brazil)",
    "ko": "Korean",
    "all": "All Languages"
}


def main(args):
    input_dir = args.input_dir
    lang = args.lang # all, en, it, ...

    model_lmentry_closed = defaultdict(list)
    model_lmentry_open = defaultdict(list)
    
    for (root,dirs,files) in os.walk(input_dir,topdown=True):
        for file in files:
            folder_lang = root.split("/")[-1]
            if lang != "all" and lang != folder_lang:
                continue
            if "lmentry" in file:
                with open(os.path.join(root, file), "r") as f:                
                    csvreader = csv.reader(f, delimiter=',')
                    for row in csvreader:
                        if "ALIA" in row[0]:
                            continue

                        if row[0] in MODELS_CLOSED:
                            model_lmentry_closed[row[0]].append(float(row[1]))

                        if row[0] in MODELS_OPEN:
                            model_lmentry_open[row[0]].append(float(row[1]))
    
    for model, values in model_lmentry_closed.items():
        model_lmentry_closed[model] = statistics.mean(values)

    for model, values in model_lmentry_open.items():
        model_lmentry_open[model] = statistics.mean(values)

    data = [list(model_lmentry_closed.values()), list(model_lmentry_open.values())]

    # -------

    # modern styling
    sns.set(style="white", font="DejaVu Sans", font_scale=1.3)
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # elegant color palette
    violin_colors = ['#4C78A8', '#F58518'] # muted blue & orange
    text_colors = ['#4C78A8', '#F58518']

    # create figure
    fig, ax = plt.subplots(figsize=(9, 7))

    # Violin plot
    parts = ax.violinplot(data, positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(violin_colors[i])
        pc.set_edgecolor('none')
        pc.set_alpha(0.3)

    from adjustText import adjust_text


    texts = []
    for i, group in enumerate(data, start=1):
        x_vals = [i] * len(group)
        ax.scatter(
            x_vals, group, alpha=0.9, color='#222222',
            edgecolors='white', linewidth=0.6, s=40, zorder=3
        )
        for j, (x_val, y_val) in enumerate(zip(x_vals, group)):
            label = list(model_lmentry_closed.keys())[j] if i == 1 else list(model_lmentry_open.keys())[j]
            label = label.replace("-Instruct", "").replace("-instruct", "")
            text = ax.text(
                x_val + 0.06, y_val, label, fontsize=14,
                color=text_colors[i - 1], va='center', ha='left', fontweight="bold"
            )
            texts.append(text)

    adjust_text(texts, ax=ax, expand_points=(1.2, 1.4), arrowprops=None)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(['Open-Weights', 'Open-Data'])
    ax.set_xlabel('Type of Models', fontsize=18, labelpad=1)
    ax.set_ylabel('LMentry Score (%)', fontsize=18, labelpad=14)
    
    # ax.set_title(f'Averaged LMentry Scores - {CODE_TO_LANG[lang]}', fontsize=18, weight='bold', pad=5)


    # Customization
    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.xaxis.grid(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)


    ax.tick_params(axis='both', which='major', labelsize=18)

    # Tight layout
    plt.tight_layout()

    # ----

    plt.savefig(f"statistics/plot/close_v_open_{lang}.pdf")



if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing the input data.")
    parser.add_argument("--lang", type=str, required=True, help="Lang to plot, all will plot average across all langs")
    args = parser.parse_args()
    main(args)