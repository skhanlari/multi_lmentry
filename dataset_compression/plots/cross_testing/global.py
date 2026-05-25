import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("outputs_cross_testing")

records = []

# =========================
# LOAD DATA (ROBUST)
# =========================
for file in INPUT_DIR.glob("*.json"):

    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🔥 safest way: get language from JSON itself
    if len(data) != 1:
        print(f"WARNING: unexpected structure in {file}")
        continue

    language = list(data.keys())[0]

    for task, task_data in data[language].items():
        for template, template_data in task_data.items():
            for model, metrics in template_data.items():

                try:
                    records.append({
                        "language": language,
                        "task": task,
                        "template": template,
                        "model": model,
                        "train_spearman": metrics["train"]["spearman"],
                        "train_pearson": metrics["train"]["pearson"],
                        "train_mse": metrics["train"]["mse"],
                        "test_mse": metrics["test"]["mse"],
                        "ratio": metrics.get("ratio", np.nan),
                        "subset_size": metrics.get("subset_size", np.nan),
                    })
                except KeyError as e:
                    print(f"Skipping {language}-{task}-{template}-{model} → missing {e}")
                    continue

# Create DataFrame
df = pd.DataFrame(records)

print("\nData loaded:", df.shape)

# =========================
# 1. AVG TEST MSE PER LANGUAGE
# =========================
lang_group = df.groupby("language").mean(numeric_only=True)

plt.figure()
lang_group["test_mse"].sort_values().plot(kind="bar")
plt.title("Average Test MSE per Language")
plt.ylabel("MSE")
plt.xlabel("Language")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 2. TRAIN vs TEST MSE
# =========================
plt.figure()
plt.scatter(df["train_mse"], df["test_mse"])
plt.xlabel("Train MSE")
plt.ylabel("Test MSE")
plt.title("Train vs Test MSE")
plt.grid()
plt.tight_layout()
plt.show()

# =========================
# 3. TEMPLATE COMPARISON
# =========================
template_group = df.groupby("template").mean(numeric_only=True)

plt.figure()
template_group["test_mse"].plot(kind="bar")
plt.title("Average Test MSE per Template")
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

# =========================
# 4. RATIO DISTRIBUTION
# =========================
plt.figure()
df["ratio"].dropna().hist(bins=20)
plt.title("Distribution of Selected Ratios")
plt.xlabel("Ratio")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =========================
# 5. SPEARMAN DISTRIBUTION
# =========================
plt.figure()
df["train_spearman"].hist(bins=20)
plt.title("Train Spearman Distribution")
plt.xlabel("Spearman")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# =========================
# 6. MODEL GENERALIZATION
# =========================
model_group = df.groupby("model").mean(numeric_only=True)

plt.figure()
model_group["test_mse"].sort_values().plot(kind="bar")
plt.title("Average Test MSE per Model")
plt.xticks(rotation=90)
plt.ylabel("MSE")
plt.tight_layout()
plt.show()

# =========================
# 7. BEST INSIGHT PLOT
# =========================
plt.figure()
plt.scatter(df["train_spearman"], df["test_mse"])
plt.xlabel("Train Spearman")
plt.ylabel("Test MSE")
plt.title("Ranking Preservation vs Generalization")
plt.grid()
plt.tight_layout()
plt.show()