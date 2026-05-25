from argparse import ArgumentParser
import json
import os
import pandas as pd

def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    lang = input_dir.split("/")[-1] # last part of the path is the language

    language_statistics = dict()

    for (root,dirs,files) in os.walk(input_dir,topdown=True):
        
        for file in files:
            task_name = file.split(".json")[0]
            with open(os.path.join(root, file), "r") as f:
                data = json.load(f)
                print(file)
                num_examples_per_template = int(data["settings"]["num_examples_per_template"])
                num_templates = len(data["settings"]["input_templates"])
                total_examples = num_examples_per_template * num_templates

                language_statistics[task_name] = {"num_examples_per_template": num_examples_per_template, "num_templates": num_templates, "total_examples": total_examples}


    df = pd.DataFrame(language_statistics)
    df = df.T
    df.loc["sum"] = df.sum()

    df.to_csv(os.path.join(output_dir, f"statistics_{lang}.csv"))




if __name__ == "__main__":
    parser = ArgumentParser(description="Create batches for GPT APIs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path containing the input data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the statistics.")
    args = parser.parse_args()
    main(args)