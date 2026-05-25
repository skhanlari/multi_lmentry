#!/usr/bin/env python3

import argparse
import json
import re
from glob import glob
from pathlib import Path


UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LMentry task JSON files into JSONL files matching the Hugging Face dataset format."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input JSON files, directories, or glob patterns (for example: data/ca/*.json)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional output root directory. If provided, files under --source-root keep their relative path "
            "beneath this directory."
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data"),
        help="Base directory used to preserve relative paths when --output-root is set (default: data)",
    )
    return parser.parse_args()


def resolve_input_files(inputs: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()

    for raw_input in inputs:
        matches = [Path(match) for match in glob(raw_input)]
        if not matches:
            matches = [Path(raw_input)]

        for match in matches:
            if match.is_dir():
                candidates = sorted(match.glob("*.json"))
            else:
                candidates = [match]

            for candidate in candidates:
                if candidate.suffix != ".json":
                    continue
                candidate = candidate.resolve()
                if candidate not in seen:
                    resolved.append(candidate)
                    seen.add(candidate)

    return resolved


def extract_canary(settings: dict) -> str:
    canary = settings.get("canary", "")
    match = UUID_RE.search(canary)
    return match.group(0) if match else canary


def iter_examples(examples):
    if isinstance(examples, dict):
        return examples.items()
    if isinstance(examples, list):
        return ((str(index), example) for index, example in enumerate(examples, start=1))
    raise ValueError("Expected `examples` to be a dictionary or list")


def build_output_path(input_path: Path, output_root: Path | None, source_root: Path) -> Path:
    if output_root is None:
        return input_path.with_suffix(".jsonl")

    try:
        relative_path = input_path.relative_to(source_root.resolve())
    except ValueError:
        relative_path = Path(input_path.name)

    return output_root / relative_path.with_suffix(".jsonl")


def convert_file(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    settings = payload.get("settings", {})
    canary = extract_canary(settings)
    examples = payload.get("examples")
    if examples is None:
        raise ValueError(f"Missing `examples` in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_id, example in iter_examples(examples):
            record = {
                "id": str(example_id),
                "input": example["input"],
                "metadata": json.dumps(example.get("metadata", {}), ensure_ascii=False),
                "canary": canary,
            }
            output_file.write(json.dumps(record, ensure_ascii=False))
            output_file.write("\n")


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    input_files = resolve_input_files(args.inputs)

    if not input_files:
        raise SystemExit("No input JSON files found")

    for input_path in input_files:
        output_path = build_output_path(input_path, args.output_root, source_root)
        convert_file(input_path, output_path)
        print(f"Converted {input_path} -> {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
