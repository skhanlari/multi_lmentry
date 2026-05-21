import json
import random
import csv
from pathlib import Path
from collections import defaultdict

PREDICTIONS_DIR = Path("predictions/en")
OUTPUT_CSV = Path("en_random_500_for_annotation.csv")

LANGUAGE = "en"
TOTAL_SAMPLES = 500
SEED = 42

random.seed(SEED)

groups = defaultdict(list)

# Load all JSON files grouped by (task, model)
for json_file in PREDICTIONS_DIR.rglob("*.json"):
    task = json_file.parent.name
    model = json_file.stem

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for example_id, item in data.items():
        groups[(task, model)].append({
            "language": LANGUAGE,
            "task": task,
            "model": model,
            "example_id": example_id,
            "input": item.get("input", ""),
            "prediction": item.get("prediction", ""),
            "regex_scorer": item.get("score", ""),
            "human_label": "",
            "judge_label": "",
        })

num_groups = len(groups)

if num_groups == 0:
    raise ValueError(f"No JSON prediction files found in {PREDICTIONS_DIR}")

samples_per_group = TOTAL_SAMPLES // num_groups
remaining = TOTAL_SAMPLES % num_groups

print(f"Found {num_groups} task/model groups")
print(f"Base samples per group: {samples_per_group}")
print(f"Extra samples to distribute: {remaining}")

sampled_rows = []

group_items = list(groups.items())
random.shuffle(group_items)

# Stratified random sampling from each (task, model)
for idx, ((task, model), rows) in enumerate(group_items):
    n = samples_per_group

    if idx < remaining:
        n += 1

    n = min(n, len(rows))

    chosen = random.sample(rows, n)
    sampled_rows.extend(chosen)

    print(f"{task} | {model} -> {n} samples")

# Final shuffle so rows are mixed in CSV
random.shuffle(sampled_rows)

with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "language",
            "task",
            "model",
            "example_id",
            "input",
            "prediction",
            "regex_scorer",
            "human_label",
            "judge_label",
        ],
    )

    writer.writeheader()
    writer.writerows(sampled_rows)

print(f"\nSaved {len(sampled_rows)} samples to:")
print(OUTPUT_CSV)