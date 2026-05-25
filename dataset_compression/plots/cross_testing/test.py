import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_DIR = Path("outputs_cross_testing")

records = []

for file in INPUT_DIR.glob("cross_testing_*.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) != 1:
        print(f"WARNING: unexpected structure in {file}")
        continue

    language = list(data.keys())[0]

    for task, task_data in data[language].items():
        for template, template_data in task_data.items():
            for model, metrics in template_data.items():
                ratio_curve = metrics.get("ratio_curve", {})

                for ratio_str, vals in ratio_curve.items():
                    records.append({
                        "language": language,
                        "task": task,
                        "template": template,
                        "model": model,
                        "ratio": float(ratio_str),
                        "train_spearman": vals.get("train_spearman"),
                        "train_pearson": vals.get("train_pearson"),
                        "train_mse": vals.get("train_mse"),
                        "test_mse": vals.get("test_mse"),
                    })

df = pd.DataFrame(records)
print("Loaded:", df.shape)

avg_df = df.groupby("ratio", as_index=False).mean(numeric_only=True).sort_values("ratio")

plt.figure(figsize=(8, 5))
plt.plot(avg_df["ratio"], avg_df["train_spearman"], marker="o")
plt.xlabel("Compression Ratio")
plt.ylabel("Train Spearman")
plt.title("Train Ranking Preservation vs Compression Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(avg_df["ratio"], avg_df["test_mse"], marker="o")
plt.xlabel("Compression Ratio")
plt.ylabel("Test MSE")
plt.title("Generalization vs Compression Ratio")
plt.grid(True)
plt.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(avg_df["ratio"], avg_df["train_spearman"], marker="o")
ax1.set_xlabel("Compression Ratio")
ax1.set_ylabel("Train Spearman")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(avg_df["ratio"], avg_df["test_mse"], marker="s")
ax2.set_ylabel("Test MSE")

plt.title("Ranking Preservation and Generalization vs Compression Ratio")
plt.tight_layout()
plt.show()