import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("en_random_500_judge_scores.csv")
OUT_DIR = Path("judge_plots/en")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

judge_cols = [c for c in df.columns if c.startswith("judge_")]
eval_cols = ["regex_scorer"] + judge_cols

# Clean labels
df["human_label"] = df["human_label"].astype(int)

for col in eval_cols:
    df[col] = df[col].astype(int)


def short_name(col):
    return (
        col.replace("judge_", "")
           .replace("meta_llama_Llama_3_1_8B_Instruct", "Llama-3.1-8B")
           .replace("Qwen_Qwen2_5_7B_Instruct", "Qwen2.5-7B")
           .replace("microsoft_Phi_3_mini_4k_instruct", "Phi-3-mini")
           .replace("regex_scorer", "Regex")
    )


# 1. Accuracy vs human label
accuracies = {
    short_name(col): (df[col] == df["human_label"]).mean()
    for col in eval_cols
}

plt.figure(figsize=(8, 5))
plt.bar(accuracies.keys(), accuracies.values())
plt.ylabel("Accuracy vs human_label")
plt.ylim(0, 1)
plt.title("Evaluator Accuracy Compared with Human Labels")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_vs_human.png", dpi=300)
plt.show()


# 2. Number of 0/1 labels per judge
counts = []

for col in eval_cols:
    value_counts = df[col].value_counts().to_dict()
    counts.append({
        "evaluator": short_name(col),
        "label_0": value_counts.get(0, 0),
        "label_1": value_counts.get(1, 0),
    })

counts_df = pd.DataFrame(counts)

plt.figure(figsize=(8, 5))
x = range(len(counts_df))
plt.bar(x, counts_df["label_0"], label="0 = incorrect")
plt.bar(x, counts_df["label_1"], bottom=counts_df["label_0"], label="1 = correct")
plt.xticks(x, counts_df["evaluator"], rotation=30, ha="right")
plt.ylabel("Number of samples")
plt.title("Distribution of Evaluator Labels")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "label_distribution.png", dpi=300)
plt.show()


# 3. Accuracy by task
task_acc = df.groupby("task").apply(
    lambda g: pd.Series({
        short_name(col): (g[col] == g["human_label"]).mean()
        for col in eval_cols
    })
)

task_acc.plot(kind="bar", figsize=(14, 6))
plt.ylabel("Accuracy vs human_label")
plt.ylim(0, 1)
plt.title("Evaluator Accuracy by Task")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "accuracy_by_task.png", dpi=300)
plt.show()


# 4. Agreement matrix between evaluators
agreement = pd.DataFrame(index=[short_name(c) for c in eval_cols],
                         columns=[short_name(c) for c in eval_cols])

for c1 in eval_cols:
    for c2 in eval_cols:
        agreement.loc[short_name(c1), short_name(c2)] = (df[c1] == df[c2]).mean()

agreement = agreement.astype(float)

plt.figure(figsize=(7, 6))
plt.imshow(agreement, aspect="auto")
plt.colorbar(label="Agreement")
plt.xticks(range(len(agreement.columns)), agreement.columns, rotation=45, ha="right")
plt.yticks(range(len(agreement.index)), agreement.index)
plt.title("Agreement Matrix Between Evaluators")

for i in range(len(agreement.index)):
    for j in range(len(agreement.columns)):
        plt.text(j, i, f"{agreement.iloc[i, j]:.2f}", ha="center", va="center")

plt.tight_layout()
plt.savefig(OUT_DIR / "agreement_matrix.png", dpi=300)
plt.show()


print("Saved plots in:", OUT_DIR)