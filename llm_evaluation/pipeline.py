import gc
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vllm import LLM, SamplingParams


JUDGE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3-mini-4k-instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]

PREDICTIONS_DIR = Path("predictions")
GOLD_DIR = Path("gold_annotations")
OUTPUT_DIR = Path("evaluated_outputs")

MAX_NEW_TOKENS = 32
BATCH_SIZE = 128

# vLLM engine knobs
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
TRUST_REMOTE_CODE = True

SYSTEM_PROMPT = "You are a strict multilingual evaluator."


def build_judge_prompt(task: str, language: str, input_text: str, prediction: str) -> str:
    return f"""
You are evaluating a model answer for a benchmark task.

Language: {language}
Task: {task}

Original prompt shown to the model:
{input_text}

Model answer:
{prediction}

Decide whether the model answer correctly follows the task instruction.

Return exactly one of:
<label>1</label>
<label>0</label>

Where:
1 = correct
0 = incorrect
""".strip()


def load_vllm_model(model_name: str) -> Tuple[object, LLM]:
    print(f"Loading judge model with vLLM: {model_name}")

    llm = LLM(
        model=model_name,
        trust_remote_code=TRUST_REMOTE_CODE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    )
    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"System: {SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"


def parse_label(text: str) -> int:
    generated = text.strip().lower()

    if "<label>1</label>" in generated:
        return 1
    if "<label>0</label>" in generated:
        return 0

    if re.search(r"\b1\b", generated):
        return 1
    if re.search(r"\b0\b", generated):
        return 0

    return 0


def chunk_list(items: List, batch_size: int) -> List[List]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def load_gold(language: str, task: str) -> Optional[Dict]:
    gold_file = GOLD_DIR / language / f"{task}.json"
    if not gold_file.exists():
        return None

    with open(gold_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gold_label(gold_data: Optional[Dict], sample_id: str) -> Optional[int]:
    if gold_data is None or sample_id not in gold_data:
        return None

    value = gold_data[sample_id]

    if isinstance(value, dict):
        if "gold" in value:
            return int(value["gold"])
        if "label" in value:
            return int(value["label"])
        return None

    return int(value)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_")


def sort_sample_items(data: Dict):
    try:
        return sorted(data.items(), key=lambda x: int(x[0]))
    except (ValueError, TypeError):
        return sorted(data.items(), key=lambda x: x[0])


def build_requests_for_file(
    tokenizer,
    language: str,
    task: str,
    source_model: str,
    data: Dict,
) -> List[Dict]:
    requests = []

    for sample_id, sample in sort_sample_items(data):
        input_text = sample.get("input", "")
        prediction = sample.get("prediction", "")
        pred_label = sample.get("score", None)

        judge_prompt = build_judge_prompt(
            task=task,
            language=language,
            input_text=input_text,
            prediction=prediction,
        )
        prompt_text = build_chat_text(tokenizer, judge_prompt)

        requests.append(
            {
                "sample_id": sample_id,
                "language": language,
                "task": task,
                "source_model": source_model,
                "input": input_text,
                "prediction": prediction,
                "predicted_label": pred_label,
                "prompt_text": prompt_text,
            }
        )

    return requests


def run_batched_judging(
    llm: LLM,
    requests: List[Dict],
    max_new_tokens: int,
    batch_size: int,
) -> Dict[str, Dict]:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    results = {}

    for batch in chunk_list(requests, batch_size):
        prompts = [item["prompt_text"] for item in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for req, out in zip(batch, outputs):
            if out.outputs:
                generated_text = out.outputs[0].text
            else:
                generated_text = ""

            judge_label = parse_label(generated_text)

            results[req["sample_id"]] = {
                "language": req["language"],
                "task": req["task"],
                "source_model": req["source_model"],
                "input": req["input"],
                "prediction": req["prediction"],
                "judge_label": judge_label,
                "raw_judge_output": generated_text,
            }

    return results


def save_json(data: Dict, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    json_files = sorted(PREDICTIONS_DIR.glob("*/*/*.json"))

    if not json_files:
        print("No prediction files found")
        return

    print(f"Found {len(json_files)} prediction files.")

    # Gold evaluation does not depend on judge model, so compute it once.
    gold_done = set()

    for judge_name in JUDGE_MODELS:
        print(f"\n{'=' * 80}")
        print(f"Starting judge model: {judge_name}")
        print(f"{'=' * 80}")

        tokenizer, llm = load_vllm_model(judge_name)
        judge_dir_name = safe_model_name(judge_name)

        try:
            for json_file in json_files:
                language = json_file.parent.parent.name
                task = json_file.parent.name
                source_model = json_file.stem

                print(f"\nProcessing {language}/{task}/{source_model} with {judge_name}")

                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                requests = build_requests_for_file(
                    tokenizer=tokenizer,
                    language=language,
                    task=task,
                    source_model=source_model,
                    data=data,
                )

                judge_results = run_batched_judging(
                    llm=llm,
                    requests=requests,
                    max_new_tokens=MAX_NEW_TOKENS,
                    batch_size=BATCH_SIZE,
                )

                # Add judge model name after inference
                for sample_id in judge_results:
                    judge_results[sample_id]["judge_model"] = judge_name

                judge_output_file = (
                    OUTPUT_DIR
                    / "llm_judge"
                    / judge_dir_name
                    / language
                    / task
                    / f"{source_model}.json"
                )
                save_json(judge_results, judge_output_file)
                print("Saved judge output:", judge_output_file)

                # Gold output only once per source file
                gold_key = str(json_file.resolve())
                if gold_key not in gold_done:
                    gold_data = load_gold(language, task)
                    gold_results = {}

                    for sample_id, sample in sort_sample_items(data):
                        input_text = sample.get("input", "")
                        prediction = sample.get("prediction", "")
                        pred_label = sample.get("score", None)
                        gold_label = get_gold_label(gold_data, sample_id)

                        if gold_label is not None and pred_label is not None:
                            gold_correct = 1 if int(pred_label) == int(gold_label) else 0
                        else:
                            gold_correct = None

                        gold_results[sample_id] = {
                            "language": language,
                            "task": task,
                            "source_model": source_model,
                            "input": input_text,
                            "prediction": prediction,
                            "gold_label": gold_label,
                            "predicted_label": pred_label,
                            "correct": gold_correct,
                        }

                    gold_output_file = (
                        OUTPUT_DIR
                        / "gold_eval"
                        / language
                        / task
                        / f"{source_model}.json"
                    )
                    save_json(gold_results, gold_output_file)
                    print("Saved gold output:", gold_output_file)

                    gold_done.add(gold_key)

        finally:
            # Release model before loading the next one
            del llm
            del tokenizer
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()