from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path("outputs_compression/adaptive_results_difficulty")
LANGUAGES = ["en", "it", "es", "de", "ca", "gl", "eu", "pt_br", "ko"]
OUTPUT_DIR = Path("dataset_compression/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_best_configs(json_path: Path, language: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for task, task_data in data["tasks"].items():
        if "error" in task_data:
            continue

        best = task_data.get("best", {})
        rows.append({
            "language": language,
            "task": task,
            "technique": best.get("technique"),
            "ratio": best.get("ratio"),
            "seed": best.get("selected_seed"),
            "spearman": best.get("spearman"),
            "pearson": best.get("pearson"),
            "n_selected": best.get("n_selected"),
        })

    return rows


def export_csvs_per_language():
    for lang in LANGUAGES:
        path = BASE_DIR / f"adaptive_results_{lang}.json"
        if not path.exists():
            print(f"⚠ skipping {lang} (file not found)")
            continue

        rows = load_best_configs(path, lang)
        if not rows:
            print(f"⚠ skipping {lang} (no valid tasks)")
            continue

        df = pd.DataFrame(rows)

        # Sort for readability
        df = df.sort_values(
            by=["spearman", "ratio"],
            ascending=[False, True]
        )

        out_path = OUTPUT_DIR / f"compression_best_configs_difficulty_{lang}.csv"
        df.to_csv(out_path, index=False)

        print(f"✔ Saved {out_path}")


if __name__ == "__main__":
    export_csvs_per_language()

# run: python -m dataset_compression.plots.csv.csv_difficulty.csv_difficulty