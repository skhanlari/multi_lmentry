import pandas as pd
from pathlib import Path
import json

INPUT_FILE = Path("persian_dataset") / "train.tsv"

OUTPUT_DIR = Path("persian_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "fa_1000_sentences.csv"
OUTPUT_JSON = OUTPUT_DIR / "fa_1000_sentences.json"

NUM_SENTENCES = 1000

print(f"Loading dataset from: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str)

print("Columns in dataset:", list(df.columns))

if "text" in df.columns:
    col = "text"
elif "sentence" in df.columns:
    col = "sentence"
elif "normalized_sentence" in df.columns:
    col = "normalized_sentence"
else:
    raise KeyError(f"No sentence column found. Available columns: {list(df.columns)}")

print("Using column:", col)

sentences = (
    df[col]
    .dropna()
    .astype(str)
    .str.strip()
)

sentences = sentences.drop_duplicates()
sentences = sentences[sentences.str.len() > 3]

selected = sentences.head(NUM_SENTENCES).tolist()

print(f"Selected {len(selected)} sentences")

csv_data = pd.DataFrame({
    "id": [f"{i+1}" for i in range(len(selected))],
    "input": selected
})

csv_data.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"Saved CSV → {OUTPUT_CSV}")

json_data = csv_data.to_dict(orient="records")

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=2)

print(f"Saved JSON → {OUTPUT_JSON}")

print("Done.")