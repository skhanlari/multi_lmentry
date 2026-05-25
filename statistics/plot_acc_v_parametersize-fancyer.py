import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import csv
from collections import defaultdict
import seaborn as sns
import statistics

MODELS_SIZE = {
    "SmolLM2-360M-Instruct": 0.36,
    "Llama-3.2-1B-Instruct": 1,
    "Qwen2.5-1.5B-Instruct": 1.5,
    "EuroLLM-1.7B-Instruct": 1.7,
    "SmolLM2-1.7B-Instruct": 1.7,
    "salamandra-2b-instruct": 2,
    "Llama-3.2-3B-Instruct": 3,
    "Qwen2.5-3B-Instruct": 3,
    "occiglot-7b-es-en-instruct": 7,
    "occiglot-7b-eu5-instruct": 7,
    "salamandra-7b-instruct": 7,
    "Llama-3.1-8B-Instruct": 8,
    "LLaMAntino-3-ANITA-8B-Inst-DPO-ITA": 8,
    "Latxa-Llama-3.1-8B-Instruct": 8,
    "EuroLLM-9B-Instruct": 9,
    "Qwen2.5-14B-Instruct": 14,
    "phi-4": 14,
}

XSMODELS = {
    "SmolLM2-360M-Instruct": 0.36,
    "Llama-3.2-1B-Instruct": 1,
    "Qwen2.5-1.5B-Instruct": 1.5,
    "EuroLLM-1.7B-Instruct": 1.7,
    "SmolLM2-1.7B-Instruct": 1.7,
}

SMODELS = {
    "salamandra-2b-instruct": 2,
    "Llama-3.2-3B-Instruct": 3,
    "Qwen2.5-3B-Instruct": 3,
}

MMODELS = {
    "occiglot-7b-es-en-instruct": 7,
    "occiglot-7b-eu5-instruct": 7,
    "salamandra-7b-instruct": 7,
    "Llama-3.1-8B-Instruct": 8,
    "LLaMAntino-3-ANITA-8B-Inst-DPO-ITA": 8,
    "Latxa-Llama-3.1-8B-Instruct": 8,
    "EuroLLM-9B-Instruct": 9,
}

LMODELS = {
    "Qwen2.5-14B-Instruct": 14,
    "phi-4": 14,
}


LANGS1 = ["en", "de", "it", "es"] # high-mid resource

LANGS2 = ["ca", "gl", "eu", "ko", "pt_br"] # mid-low resource

CODE_TO_LANG = {
    "en": "English",
    "de": "German",
    "it": "Italian",
    "ca": "Catalan",
    "gl": "Gallician",
    "eu": "Basque",
    "ko": "Korean",
    "pt_br": "Portugal (Brazilian)",
    "es": "Spanish"
}

PLOT_PARAM_LANG = {
    "en": {"color": 'blue', "linestyle": 'dashed', "marker": 'o'},
    "it": {"color": 'red', "linestyle": 'dashed', "marker": '.'},
    "de": {"color": 'green', "linestyle": 'dashed', "marker": 'x'},
    "es": {"color": 'darkcyan', "linestyle": 'dashed', "marker": 'v'},
    "gl": {"color": 'purple', "linestyle": 'dashed', "marker": '^'},
    "eu": {"color": 'black', "linestyle": 'dashed', "marker": '<'},
    "ca": {"color": 'orange', "linestyle": 'dashed', "marker": '>'},
    "pt_br": {"color": 'magenta', "linestyle": 'dashed', "marker": '1'},
    "ko": {"color": 'grey', "linestyle": 'dashed', "marker": '2'},
}


def main(args):
    input_dir = args.input_dir

    model_lmentry = defaultdict(dict)
    model_accuracies = defaultdict(dict)
    
    for (root,dirs,files) in os.walk(input_dir,topdown=True):
        for file in files:
            if "lmentry" in file:
                lang = root.split("/")[-1]
                print(root)
                print(lang)
                with open(os.path.join(root, file), "r") as f:                
                    csvreader = csv.reader(f, delimiter=',')
                    for row in csvreader:
                        if "ALIA" in row[0]:
                            continue
                        model_lmentry[lang][row[0]] = row[1]
                        model_accuracies[lang][row[0]] = row[2]

    for lang, values in model_accuracies.items():
        if not lang in LANGS1:
            continue
        compressed_values = [
            statistics.mean([float(values[k]) for k in XSMODELS.keys()]),
            statistics.mean([float(values[k]) for k in SMODELS.keys()]),
            statistics.mean([float(values[k]) for k in MMODELS.keys()]),
            statistics.mean([float(values[k]) for k in LMODELS.keys()]),
        ]

        plt.plot(
            [1, 3, 8, 14],
            compressed_values,
            label=CODE_TO_LANG[lang], 
            **PLOT_PARAM_LANG[lang]
        )
    
    # Customizations
    
    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid", font_scale=1.2)

    ax = plt.gca()  # Get current axes

    # Set light grey background
    # ax.set_facecolor('#fafafa')  # Hex code for light grey

    # Show grid only for vertical (y-axis) lines
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.7)
    ax.xaxis.grid(False)

    plt.legend(fontsize=12)
    plt.xticks([1, 3, 8, 14], ["XS", "S", "M", "L"])
    
    # labels
    plt.xlabel('Model Size', fontsize=16)
    plt.ylabel('Accuracy Score (%)', fontsize=16)
    plt.title("Averaged Accuracy Scores", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig("statistics/plot/accuracy_v_params_lang1.pdf")

    plt.cla()

    for lang, values in model_accuracies.items():
        if not lang in LANGS2:
            continue
        compressed_values = [
            statistics.mean([float(values[k]) for k in XSMODELS.keys()]),
            statistics.mean([float(values[k]) for k in SMODELS.keys()]),
            statistics.mean([float(values[k]) for k in MMODELS.keys()]),
            statistics.mean([float(values[k]) for k in LMODELS.keys()]),
        ]

        plt.plot(
            [1, 3, 8, 14],
            compressed_values,
            label=CODE_TO_LANG[lang], 
            **PLOT_PARAM_LANG[lang]
        )

    # Customizations

    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid", font_scale=1.2)

    ax = plt.gca()  # Get current axes

    # Set light grey background
    ax.set_facecolor('#fafafa')  # Hex code for light grey

    # Show grid only for vertical (y-axis) lines
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.7)
    ax.xaxis.grid(False)

    plt.legend(fontsize=12)
    plt.xticks([1, 3, 8, 14], ["XS", "S", "M", "L"])
    
    # labels
    plt.xlabel('Model Size', fontsize=16)
    plt.ylabel('Accuracy Score (%)', fontsize=16)
    plt.title("Averaged Accuracy Scores", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    plt.savefig("statistics/plot/accuracy_v_params_lang2.pdf")

    plt.cla()

    # modern styling
    sns.set(style="white", font="DejaVu Sans", font_scale=1.3)
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # create figure
    fig, ax = plt.subplots(figsize=(9, 6))

    for lang, values in model_lmentry.items():
        if not lang in LANGS1:
            continue
        compressed_values = [
            statistics.mean([float(values[k]) for k in XSMODELS.keys()]),
            statistics.mean([float(values[k]) for k in SMODELS.keys()]),
            statistics.mean([float(values[k]) for k in MMODELS.keys()]),
            statistics.mean([float(values[k]) for k in LMODELS.keys()]),
        ]


        plt.plot(
            [1, 3, 8, 14],
            compressed_values,
            label=CODE_TO_LANG[lang], 
            **PLOT_PARAM_LANG[lang]
        )

    # Customizations

    # Set Seaborn style for better aesthetics
    sns.set(style="whitegrid", font_scale=1.2)

    ax = plt.gca()  # Get current axes

    # Set light grey background
    # ax.set_facecolor('#fafafa')  # Hex code for light grey

    # Show grid only for vertical (y-axis) lines
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.7)
    ax.xaxis.grid(False)

    plt.legend(fontsize=12)
    plt.xticks([1, 3, 8, 14], ["XS", "S", "M", "L"])
    
    # labels
    plt.xlabel('Model Size', fontsize=16)
    plt.ylabel('LMentry Score (%)', fontsize=16)
    #plt.title("Averaged LMentry Scores", fontsize=18, weight='bold', pad=5)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.xaxis.grid(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig("statistics/plot/lmentry_v_params_lang1.pdf")

    plt.cla()

    # modern styling
    sns.set(style="white", font="DejaVu Sans", font_scale=1.3)
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8

    # create figure
    fig, ax = plt.subplots(figsize=(9, 6))


    for lang, values in model_lmentry.items():
        if not lang in LANGS2:
            continue
        compressed_values = [
            statistics.mean([float(values[k]) for k in XSMODELS.keys()]),
            statistics.mean([float(values[k]) for k in SMODELS.keys()]),
            statistics.mean([float(values[k]) for k in MMODELS.keys()]),
            statistics.mean([float(values[k]) for k in LMODELS.keys()]),
        ]

        plt.plot(
            [1, 3, 8, 14],
            compressed_values,
            label=CODE_TO_LANG[lang], 
            **PLOT_PARAM_LANG[lang]
        )

    # Customizations


    # Set light grey background
    # ax.set_facecolor('#fafafa')  # Hex code for light grey

    # Show grid only for vertical (y-axis) lines
    ax.yaxis.grid(True, linestyle='--', color='grey', alpha=0.7)
    ax.xaxis.grid(False)

    plt.legend(fontsize=12)
    plt.xticks([1, 3, 8, 14], ["XS", "S", "M", "L"])
    
    # labels
    plt.xlabel('Model Size', fontsize=16)
    plt.ylabel('LMentry Score (%)', fontsize=16)
    # plt.title("Averaged LMentry Scores", fontsize=18, weight='bold', pad=5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_facecolor('white')
    ax.yaxis.grid(True, linestyle='--', color='#CCCCCC', alpha=0.7)
    ax.xaxis.grid(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)


    plt.tight_layout()

    plt.savefig("statistics/plot/lmentry_v_params_lang2.pdf")



if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing the input data.")
    args = parser.parse_args()
    main(args)