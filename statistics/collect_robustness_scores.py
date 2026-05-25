import csv
import os
import pandas as pd

from collections import defaultdict

INPUT = "/home/luca/__Projects/multi_lmentry/results"
OUTPUT = "/home/luca/__Projects/multi_lmentry/statistics/robustness/robustness.csv"

COLUMNS = ["en_robustness", "es_robustness", "de_robustness", "it_robustness", "ko_robustness", "ca_robustness", "gl_robustness", "eu_robustness", "pt_br_robustness"]

def main():
    robustness_scores = defaultdict(dict)

    for (root,dirs,files) in os.walk(INPUT,topdown=True):
        for file in files:
            if "lmentry" in file:
                lang = root.split("/")[-1]
                print(lang)
                with open(os.path.join(root,file), "r") as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        model = row[0]
                        robustness = row[3]
                        

                        robustness_scores[model] = robustness_scores[model] | {f"{lang}_robustness": float(robustness)}

    for k, v in robustness_scores.items():
        print(k)
        robustness_scores[k] = {f"{k_k}": robustness_scores[k][k_k] for k_k in COLUMNS}

    print(robustness_scores)
    df = pd.DataFrame(robustness_scores)
    df = df.T
    # df = df.set_index(df.columns[0])
    print()
    average_row = df.mean()

    # Append the average row to the DataFrame
    df.loc['Average'] = average_row


    print(df)

    df.round(1).to_csv(OUTPUT, index=True)
    

main()