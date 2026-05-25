import os
import re
import yaml
import jsonlines
from tqdm import tqdm
from argparse import ArgumentParser
from typing import Optional, List
from transformers import AutoTokenizer
import json
import random
from collections import defaultdict
import re
import statistics
from sklearn.metrics import cohen_kappa_score, accuracy_score
import tqdm


pattern = r"\\boxed(?:\\)?\{(\d+)\}"


def check_sample(all_samples, annotated_sample): # O(1) searching complexity
    # for sample in all_samples:
    #     if sample["id"] == annotated_sample["id"]:
    #         return (sample, int(sample["score"]), int(annotated_sample["response"]))
    
    id = annotated_sample["id"]
    if id in all_samples:
        sample = all_samples[id]
        return (sample, int(sample["score"]), int(annotated_sample["response"]))
    else:
        print(f"error-no match: {annotated_sample}")
        exit()


def main(args):
    input_path = args.input_path
    annotated_batch = args.batch
    output_path = args.output_path

    language = input_path.split("/")[-1]

    ## COLLECT ALL SAMPLES

    all_samples = {}
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if ".json" in file:
                model_name = file.split(".json")[0]
                if "ALIA-40b" in model_name: # skip ALIA model
                    continue
                if "DeepSeek-R1-Distill-Llama-8B" in model_name: # skip ALIA model
                    continue
                task = root.split("/")[-1]
                with open(os.path.join(root,file), "r") as f:
                    data = json.load(f)
                    for id, sample in data.items():
                        if not "score" in sample:
                            print(sample)
                            print(file)
                            exit()

                        id = f"request--{language}--{task}--{model_name}--{id}"

                        all_samples[id] = {
                            "id": id,
                            "score": sample["score"],
                            "input": sample["input"],
                            "prediction": sample["prediction"],
                            "task": task
                        }
    
    # TO COMPUTE THE AGREEMENT
    original_results = []
    annotated_results = []
    ## TASK WISE
    tasks_original_results = defaultdict(list)
    tasks_annoated_results = defaultdict(list)
    
    # TO ASSESS THE JUDGE CAPABILITIES 
    miss_annotated_samples = []
    disagreement_samples = []

    with open(annotated_batch, "r") as f:
        for line in tqdm.tqdm(f):
            data = json.loads(line)

            task_name = data["custom_id"].split("--")[2]

            if "DeepSeek-R1-Distill-Llama-8B" in data["custom_id"]:
                    continue

            # EXTRACT THE RESPONSE
            raw_response = data["response"]["body"]["choices"][0]["message"]["content"]
            raw_response = raw_response.replace("\\{", "{").replace("\\}", "}")
            match = re.search(pattern, raw_response)

            if match == None:
                print("ERROR:", data["response"]["body"]["choices"][0]["message"]["content"])
                response = 0 # hardcoded
            else:    
                response = match.group(1)

            ## ---
            
            annotated_sample = {
                "id": data["custom_id"],
                "response": response 
            }

            original_sample, original_result, annotated_result = check_sample(all_samples, annotated_sample)

            if response != "0" and response != "1": # save the samples where the judge cannot answer in a proper format
                miss_annotated_samples.append(original_sample)

            if original_result != annotated_result: # save the samples where the judge and the regex disagree
                disagreement_samples.append(original_sample)

            # collect the predictions over all the annotations
            original_results.append(original_result)
            annotated_results.append(annotated_result)

            # collect the predictions task wise
            tasks_original_results[task_name].append(original_result)
            tasks_annoated_results[task_name].append(annotated_result) 


    global_agreement = cohen_kappa_score(original_results, annotated_results)
    global_accuracy = accuracy_score(original_results, annotated_results)

    # PRINT Cohen's kappa over all the annotations
    print(f"GLOBAL AGREEMENT FOR LANGUAGE {language} -> {global_agreement}")

    # PRINT Accuracy over all the annotations
    print(f"GLOBAL ACCURACY FOR LANGUAGE {language} -> {global_accuracy}")

    tasks_agreements = {
        "global_agreement": global_agreement
    }

    tasks_accuracies = {
        "global_accuracy": global_accuracy
    }

    print(f"Saving the agreements and accuracies task wise")
    for task in tasks_original_results.keys():
        task_original_results = tasks_original_results[task]
        task_annotated_results = tasks_annoated_results[task]

        # compute the agreement per task
        tasks_agreements[task] = cohen_kappa_score(task_original_results, task_annotated_results)

        # compute the accuaracy per task
        tasks_accuracies[task] = accuracy_score(task_original_results, task_annotated_results)

    with open(os.path.join(output_path, "agreement.json"), "w") as f:
        json.dump(tasks_agreements, f, indent=4)

    with open(os.path.join(output_path, "accuracy.json"), "w") as f:
        json.dump(tasks_accuracies, f, indent=4)

    with open(os.path.join(output_path, "disagreement.json"), "w") as f:
        json.dump({
            "disagreement_samples": disagreement_samples,
            "miss_annotated_samples": miss_annotated_samples
        }, f, indent=4)



if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the GPT config.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the GPT config.")
    parser.add_argument("--batch", type=str, required=True, help="Path containing the input data.")
    args = parser.parse_args()
    main(args)