import os
import re
import yaml
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Optional, List
from transformers import AutoTokenizer
import json

MAX_BATCH_TOKENS = 1000000

SYSTEM_INSTRUCTION = "You are an effective LLM-based judge. You will be presented with simple questions and corresponding answers. Your task is to concisely evaluate and score the answers."

MAP_LANG = {
    "it": "Italian",
    "en": "English",
    "ca": "Catalan",
    "eu": "Basque",
    "de": "German",
    "pt_br": "Portuguese",
    "ko": "Korean",
    "es": "Spanish",
    "gl": "Gallician"
}

def create_gpt_prompt(
        question: str,
        model_output: str, 
        language: str,
    ) -> List[dict]:

    user_content = (
        f"The following is an elementary question (Question) with an answer (Model Output) in {language} language\n\n"
        f'Question:\n{question}\n\n'
        f'Model Output:\n{model_output}\n\n'
        "Your task is to understand if the Model Output properly answer to the given Question."
        "If the Model Output is correct answer with 1 otherwise with 0.\n\n"
        "Please put your boolean evaluation in \\boxed\{\}. Do not generate explainations for your evaluation."
    )
    prompt_messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content}
    ]
    return prompt_messages


def parse_sample(model_input:str, model_output: str) -> str:
    pass


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def clean_input(input):
    cleaned = None
    if "<|start_header_id|>user<|end_header_id|>\n\n" in input:
        cleaned = input.split("<|start_header_id|>user<|end_header_id|>\n\n")[1].split("<|eot_id|>")[0]
    elif "<|im_start|>user\n" in input:
        cleaned = input.split("<|im_start|>user\n")[1].split("<|im_end|>")[0]

    return cleaned

def collect_examples_per_lang(input_path):
    all_samples = defaultdict(list)
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if ".json" in file:
                if "Llama-3_2-1B-Instruct" in file: # annotate only for a single output, not output used.
                    task = root.split("/")[-1]
                    with open(os.path.join(root,file), "r") as f:
                        data = json.load(f)
                        for id, sample in data.items():
                            all_samples[task].append({
                                "id": id,
                                "model_name": model_name,
                                "question": clean_input(sample["input"]),
                                "model_output": sample["prediction"],
                                "task": task
                            })
    return all_samples


def create_batch(
    config_path: str,
    input_path: str,
    output_dir: str,
    limit_samples: Optional[int] = None,
):
    
    config = load_config(config_path=config_path)

    judge_model_name = config["model_name"]
    generation_config = config["generation_config"]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ## collect all the examples per language
    all_samples = collect_examples_per_lang(input_path)

    language = input_path.split("/")[-1]

    print("## Language:", language)
    
    ## Using a tokenizer to estimate the upper bound of maximum tokens in a batch.
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

    sample_buffer = []
    token_buffer, start, end = 0, 0, 0
    for i, line in tqdm(enumerate(all_samples), total=len(all_samples)):
        question = line["question"]
        model_output = line["model_output"]
        messages = create_gpt_prompt(
            question=question,
            model_output=model_output,
            language=language
        )
        request_id = f'request--{language}--{line["task"]}--{line["model_name"]}--{line["id"]}'

        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        token_buffer += len(input_ids)

        request = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": judge_model_name,
                "messages": messages,
                "temperature": generation_config["temperature"],
                "max_tokens": generation_config["max_tokens"]
            }
        }

        sample_buffer.append(request)
        end = i

        if token_buffer > MAX_BATCH_TOKENS:
            output_path = os.path.join(
                output_dir, 
                f"batch_api_data_{language}_range_{start}_{end}.jsonl"
            )
            with jsonlines.open(output_path, "w") as fout:
                for sample in sample_buffer:
                    fout.write(sample)
            sample_buffer = []
            token_buffer = 0
            start = end
            
        if i == limit_samples:
            output_path = os.path.join(
                output_dir, 
                f"batch_api_data_{language}_range_{start}_{end}.jsonl"
            )
            with jsonlines.open(output_path, "w") as fout:
                for sample in sample_buffer:
                    fout.write(sample)
            sample_buffer = []
            token_buffer = 0
            start = end
            break

    output_path = os.path.join(
        output_dir, 
        f"batch_api_data_{language}_range_{start}_{end}.jsonl"
    )
    with jsonlines.open(output_path, "w") as fout:
        for sample in sample_buffer:
            fout.write(sample)

if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--input_path", type=str, required=True, help="Path containing the input data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the batches.")
    args = parser.parse_args()
    create_batch(**vars(args))