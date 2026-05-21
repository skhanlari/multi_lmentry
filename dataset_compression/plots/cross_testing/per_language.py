import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("outputs_cross_testing")
OUTPUT_DIR = Path("dataset_compression/plots/cross_testing/plots_per_language")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

records = []

# =========================
# LOAD DATA
# =========================
for file in INPUT_DIR.glob("*.json"):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) != 1:
        print(f"WARNING: unexpected structure in {file}")
        continue

    language = list(data.keys())[0]

    for task, task_data in data[language].items():
        for template, template_data in task_data.items():
            for model, metrics in template_data.items():

                train = metrics.get("train", {})
                test = metrics.get("test", {})

                records.append({
                    "language": language,
                    "task": task,
                    "template": template,
                    "model": model,
                    "train_spearman": train.get("spearman", np.nan),
                    "train_pearson": train.get("pearson", np.nan),
                    "train_mse": train.get("mse", np.nan),
                    "test_mse": test.get("mse", np.nan),
                    "ratio": metrics.get("ratio", np.nan),
                    "subset_size": metrics.get("subset_size", np.nan),
                })

df = pd.DataFrame(records)

print("Loaded data:", df.shape)

languages = sorted(df["language"].dropna().unique())

# =========================
# PER-LANGUAGE PLOTS
# =========================
for lang in languages:

    df_lang = df[df["language"] == lang]

    print(f"\nProcessing language: {lang}")

    # 1. Train vs Test MSE
    plt.figure()
    plt.scatter(df_lang["train_mse"], df_lang["test_mse"])
    plt.xlabel("Train MSE")
    plt.ylabel("Test MSE")
    plt.title(f"Train vs Test MSE - {lang}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{lang}_train_vs_test_mse.png")
    plt.close()

    # 2. Ranking vs Generalization
    plt.figure()
    plt.scatter(df_lang["train_spearman"], df_lang["test_mse"])
    plt.xlabel("Train Spearman")
    plt.ylabel("Test MSE")
    plt.title(f"Ranking vs Generalization - {lang}")
    plt.grid()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{lang}_ranking_vs_generalization.png")
    plt.close()

    # 3. Template Comparison
    template_group = df_lang.groupby("template").mean(numeric_only=True)

    plt.figure()
    template_group["test_mse"].plot(kind="bar")
    plt.title(f"Test MSE per Template - {lang}")
    plt.ylabel("MSE")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{lang}_template_mse.png")
    plt.close()

    # 4. Model Comparison
    model_group = df_lang.groupby("model").mean(numeric_only=True)

    plt.figure()
    model_group["test_mse"].sort_values().plot(kind="bar")
    plt.title(f"Test MSE per Model - {lang}")
    plt.ylabel("MSE")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{lang}_model_mse.png")
    plt.close()


# =========================
# 🔥 GLOBAL HEATMAP (LANGUAGE × TEMPLATE)
# =========================
print("\nGenerating heatmap...")

heatmap_data = (
    df.groupby(["language", "template"])["test_mse"]
    .mean()
    .unstack()
)

heatmap_data = heatmap_data.sort_index()

plt.figure(figsize=(8, 6))

img = plt.imshow(heatmap_data, aspect='auto')

plt.colorbar(img, label="Average Test MSE")

plt.xticks(range(len(heatmap_data.columns)), heatmap_data.columns)
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)

# 🔥 add values inside cells (VERY useful)
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.iloc[i, j]
        plt.text(j, i, f"{val:.4f}",
                 ha="center", va="center", color="white")

plt.title("Test MSE Heatmap (Language × Template)")
plt.xlabel("Template")
plt.ylabel("Language")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heatmap_language_template.png")
plt.close()

print("\nAll plots (including heatmap) saved in:", OUTPUT_DIR)