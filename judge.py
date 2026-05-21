import gc
import re
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams


JUDGE_MODELS = [
    #"Qwen/Qwen2.5-7B-Instruct",
     "google/gemma-2-2b-it",
    #"microsoft/Phi-3-mini-4k-instruct",
    #"meta-llama/Llama-3.1-8B-Instruct",
]

INPUT_CSV = Path("fa_random_500_for_annotation.csv")
OUTPUT_CSV = Path("fa_random_500_with_judge_scores.csv")

MAX_NEW_TOKENS = 32
BATCH_SIZE = 128

TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.90
TRUST_REMOTE_CODE = True

SYSTEM_PROMPT = "You are a strict multilingual evaluator."


def safe_model_col(model_name: str) -> str:
    return "judge_" + model_name.replace("/", "_").replace("-", "_").replace(".", "_")


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


def load_vllm_model(model_name: str):
    print(f"Loading judge model with vLLM: {model_name}")

    llm = LLM(
    model=model_name,
    tokenizer=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=4096,
    )

    tokenizer = llm.get_tokenizer()
    return tokenizer, llm


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\n{prompt}",
        }
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"

def parse_label(text: str) -> int:
    generated = str(text).strip().lower()

    if "<label>1</label>" in generated:
        return 1
    if "<label>0</label>" in generated:
        return 0

    if re.search(r"\b1\b", generated):
        return 1
    if re.search(r"\b0\b", generated):
        return 0

    return 0


def chunk_list(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def build_requests_from_csv(tokenizer, df: pd.DataFrame):
    requests = []

    for idx, row in df.iterrows():
        language = str(row.get("language", ""))
        task = str(row.get("task", ""))
        input_text = str(row.get("input", ""))
        prediction = str(row.get("prediction", ""))

        judge_prompt = build_judge_prompt(
            task=task,
            language=language,
            input_text=input_text,
            prediction=prediction,
        )

        prompt_text = build_chat_text(tokenizer, judge_prompt)

        requests.append({
            "row_idx": idx,
            "prompt_text": prompt_text,
        })

    return requests


def run_batched_judging(llm, requests, max_new_tokens, batch_size):
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    labels = {}

    for batch in chunk_list(requests, batch_size):
        prompts = [item["prompt_text"] for item in batch]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for req, out in zip(batch, outputs):
            generated_text = out.outputs[0].text if out.outputs else ""
            labels[req["row_idx"]] = parse_label(generated_text)

    return labels


def main():
    df = pd.read_csv(INPUT_CSV)

    required_cols = ["language", "task", "model", "example_id", "input", "prediction"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    for judge_name in JUDGE_MODELS:
        print(f"\n{'=' * 80}")
        print(f"Starting judge model: {judge_name}")
        print(f"{'=' * 80}")

        judge_col = safe_model_col(judge_name)

        tokenizer, llm = load_vllm_model(judge_name)

        try:
            requests = build_requests_from_csv(tokenizer, df)

            labels = run_batched_judging(
                llm=llm,
                requests=requests,
                max_new_tokens=MAX_NEW_TOKENS,
                batch_size=BATCH_SIZE,
            )

            df[judge_col] = df.index.map(labels)

            df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"Saved intermediate result with column: {judge_col}")

        finally:
            del llm
            del tokenizer
            gc.collect()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"\nEvaluation completed. Final CSV saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()