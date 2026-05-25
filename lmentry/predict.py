import json
import logging
import os
import time
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import List
from tqdm import tqdm
import argparse

import openai
import torch

from transformers import AutoModelForSeq2SeqLM, PreTrainedModel, AutoTokenizer, AutoModelForCausalLM, pipeline

from datasets import load_dataset
from vllm import LLM, SamplingParams

from lmentry.system_prompts import SYSTEM_PROMPT
from lmentry.models import get_predictor_model_name
from lmentry.constants import initialize_variables, lmentry_random_seed, HF_PATH

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn" # NEEDED if VLLM raise some fork process errors

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)

LANG = None

def _batcher(sequence, batch_size):
    # return a list of strings,
    # each element of the batch is an input where the model is asked to continue properly.

    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i + batch_size]


def _chatter(strings, tokenizer):
    strings_chatter = []

    system_prompt = SYSTEM_PROMPT[LANG]

    print(f"Selected system prompt: '{system_prompt}'")

    for text in strings:
        
        # apply messages structure
        messages = [
            {"role": "system", "content": system_prompt}, # Original: 'You are a helpful assistant. Please give a long and detailed answer.'
            {"role": "user", "content": text},
        ]

        # chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_dict=False)
        strings_chatter.append(messages)

    return strings_chatter


def _ms_since_epoch():
    return time.perf_counter_ns() // 1000000


def generate_task_hf_predictions(task_name, model,
                                model_name, max_new_tokens, batch_size, use_vllm):
    from lmentry.tasks.lmentry_tasks import all_tasks

    task = all_tasks[task_name]()

    if not model_name and not model:
        raise ValueError("must provide either `model_name` or `model`")

    hf_model_name = get_predictor_model_name(model_name)

    logging.info(f"generating predictions for task \"{task_name}\" with model \"{hf_model_name}\"")

    # initialize tokenizer and model
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token}) # added <- DO WE NEED THIS OUTSIDE OF GPT2?

    # LOAD THE DATASET
    ds = load_dataset(
        HF_PATH,
        LANG,
        data_files=f"{LANG}/{task.name}.jsonl"
    )["train"]

    string_inputs = [example for example in ds["input"]]

    # for each string in the batch apply chat template, do it sequentially.
    samples_chat = _chatter(string_inputs, tokenizer=tokenizer)

    print(f"String inputs: {len(samples_chat)}")

    predictions: list[str] = []

    if use_vllm:
        sampling_params = SamplingParams(
            temperature=0, 
            max_tokens=max_new_tokens,
            top_p=1.0,
            top_k=-1,
            min_p=0,
            seed=lmentry_random_seed
        )

        outputs = model.chat(samples_chat, sampling_params)

        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            predictions.append(generated_text.strip())

    else: # generate predictions with HF, batch wise
        print(f"Batch size: {batch_size}")
        
        results = model(
            samples_chat, 
            do_sample=False, 
            max_new_tokens=max_new_tokens, 
            batch_size=batch_size, 
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            min_p=0,
            return_full_text=False,
            num_beams=1)

        for res in results:
            generated_text = res[0]['generated_text']
            predictions.append(generated_text.strip())
        
    # save the predictions
    predictions_data = dict()
    for id_, input_, prediction in zip(ds["id"], samples_chat, predictions):
        predictions_data[id_] = {"input": input_[1]["content"], "prediction": prediction}

    output_path = task.predictions_dir.joinpath(model_name).with_suffix(".json")
    with open(output_path, "w", encoding="utf-8") as f_predictions:
        json.dump(predictions_data, f_predictions, indent=2, ensure_ascii=False)


def generate_all_hf_predictions(task_names, model, model_name, max_new_tokens, batch_size, use_vllm):
    for task_name in task_names:
        generate_task_hf_predictions(task_name=task_name, model=model, model_name=model_name, max_new_tokens=max_new_tokens, batch_size=batch_size, use_vllm=use_vllm)


def main(args):
    global LANG
    lang = args.language
    model_name = args.model_name
    max_new_tokens = args.max_new_tokens
    batch_size = args.batch_size
    use_vllm = args.use_vllm

    ## initialize the global variables
    initialize_variables(lang)
    LANG = lang
    ## ----

    print("#"*30)
    print("Testing the following model:\n", model_name)
    print("Testing the Language:", lang)
    print("#"*30)

    hf_model_name = get_predictor_model_name(model_name)
    logging.info(f"loading model {hf_model_name}")
    
    if use_vllm:
        model = LLM(model=hf_model_name, dtype=torch.bfloat16, tensor_parallel_size=torch.cuda.device_count()) # initiaize with all available gpus.
    else:
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

        model_hf = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype="bfloat16",
            device_map="auto",
            attn_implementation="flash_attention_2"  # 👈 Force FlashAttention
        )

        model = pipeline("text-generation", model=model_hf, tokenizer=tokenizer)
        if model.tokenizer.pad_token_id == None: # if not defined set it as eos, for handling padding in batch generation.
            model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    logging.info(f"finished loading model {hf_model_name}")

    if lang == "eu":
        #for eu
        generate_all_hf_predictions(model=model, task_names=["sentence_containing","sentence_not_containing","word_containing","word_not_containing","most_associated_word","least_associated_word","any_words_from_category","all_words_from_category","first_alphabetically","more_letters","less_letters","bigger_number"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["smaller_number","rhyming_word","word_after","word_before","starts_with_word","ends_with_word","starts_with_letter","ends_with_letter","first_word","last_word","first_letter","last_letter"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["first_alphabetically_far_first_letter","first_alphabetically_different_first_letter","first_alphabetically_consecutive_first_letter","first_alphabetically_same_first_letter","any_words_from_category_5_distractors","any_words_from_category_4_distractors","any_words_from_category_3_distractors","all_words_from_category_0_distractors"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["all_words_from_category_1_distractors","all_words_from_category_2_distractors","rhyming_word_orthographically_similar","more_letters_length_diff_3plus","more_letters_length_diff_1","less_letters_length_diff_3plus","less_letters_length_diff_1"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
    elif lang == "en":
        # for en
        generate_all_hf_predictions(model=model, task_names=["sentence_containing","sentence_not_containing","word_containing","word_not_containing","most_associated_word","least_associated_word","any_words_from_category","all_words_from_category","first_alphabetically","more_letters","less_letters","bigger_number"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["smaller_number","rhyming_word","homophones","word_after","word_before","starts_with_word","ends_with_word","starts_with_letter","ends_with_letter","first_word","last_word","first_letter","last_letter"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["first_alphabetically_far_first_letter","first_alphabetically_different_first_letter","first_alphabetically_consecutive_first_letter","first_alphabetically_same_first_letter","any_words_from_category_5_distractors","any_words_from_category_4_distractors","any_words_from_category_3_distractors","all_words_from_category_0_distractors"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["all_words_from_category_1_distractors","all_words_from_category_2_distractors","rhyming_word_orthographically_similar","more_letters_length_diff_3plus","more_letters_length_diff_1","less_letters_length_diff_3plus","less_letters_length_diff_1"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["rhyming_word_orthographically_different"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["all_words_from_category_1_distractors","all_words_from_category_2_distractors","rhyming_word_orthographically_similar","more_letters_length_diff_3plus","more_letters_length_diff_1","less_letters_length_diff_3plus","less_letters_length_diff_1"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
    elif lang == "ko":
        # for ko
        generate_all_hf_predictions(model=model, task_names=["sentence_containing","sentence_not_containing","word_containing","word_not_containing","most_associated_word","least_associated_word","any_words_from_category","all_words_from_category","first_alphabetically","more_letters","less_letters","bigger_number"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["smaller_number","rhyming_word","homophones","word_after","word_before","starts_with_word","ends_with_word","starts_with_letter","ends_with_letter","first_word","last_word","first_letter","last_letter"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["any_words_from_category_5_distractors","any_words_from_category_4_distractors","any_words_from_category_3_distractors","all_words_from_category_0_distractors"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["all_words_from_category_1_distractors","all_words_from_category_2_distractors","rhyming_word_orthographically_similar","more_letters_length_diff_3plus","more_letters_length_diff_1","less_letters_length_diff_3plus","less_letters_length_diff_1"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)

    else:
        generate_all_hf_predictions(model=model, task_names=["sentence_containing","sentence_not_containing","word_containing","word_not_containing","most_associated_word","least_associated_word","any_words_from_category","all_words_from_category","first_alphabetically","more_letters","less_letters","bigger_number"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["smaller_number","rhyming_word","homophones","word_after","word_before","starts_with_word","ends_with_word","starts_with_letter","ends_with_letter","first_word","last_word","first_letter","last_letter"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["first_alphabetically_far_first_letter","first_alphabetically_different_first_letter","first_alphabetically_consecutive_first_letter","first_alphabetically_same_first_letter","any_words_from_category_5_distractors","any_words_from_category_4_distractors","any_words_from_category_3_distractors","all_words_from_category_0_distractors"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)
        generate_all_hf_predictions(model=model, task_names=["all_words_from_category_1_distractors","all_words_from_category_2_distractors","rhyming_word_orthographically_similar","more_letters_length_diff_3plus","more_letters_length_diff_1","less_letters_length_diff_3plus","less_letters_length_diff_1"], max_new_tokens=max_new_tokens, model_name=model_name, batch_size=batch_size, use_vllm=use_vllm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Execute Prediction'
                    )
    parser.add_argument('-l', '--language', type=str, required=True)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-v', '--use_vllm', action="store_true")
    parser.add_argument('-t', '--max_new_tokens', type=int, default=1024, required=False)
    parser.add_argument('-b', '--batch_size', type=int, default=8, required=False) # NOTE: this must be setted for HF
    args = parser.parse_args()
    main(args)