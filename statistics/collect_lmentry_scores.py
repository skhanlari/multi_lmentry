import csv
import os
import pandas as pd

from collections import defaultdict

INPUT = "/home/luca/__Projects/multi_lmentry/results"
OUTPUT = "/home/luca/__Projects/multi_lmentry/statistics/lmentry_scores/lmentry_scores.csv"

COLUMNS = ["en_lmentry", "en_accuracy", "es_lmentry", "es_accuracy", "de_lmentry", "de_accuracy", "it_lmentry", "it_accuracy", "ko_lmentry", "ko_accuracy", "ca_lmentry", "ca_accuracy", "gl_lmentry", "gl_accuracy", "eu_lmentry", "eu_accuracy", "pt_br_lmentry", "pt_br_accuracy"]

def main():
    lmentry_scores = defaultdict(dict)

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
                        lmentry = row[1]
                        accuracy = row[2]

                        lmentry_scores[model] = lmentry_scores[model] | {f"{lang}_lmentry": float(lmentry), f"{lang}_accuracy": float(accuracy)}

    for k, v in lmentry_scores.items():
        print(k)
        lmentry_scores[k] = {f"{k_k}": lmentry_scores[k][k_k] for k_k in COLUMNS}

    print(lmentry_scores)
    df = pd.DataFrame(lmentry_scores)
    df = df.T
    # df = df.set_index(df.columns[0])
    print()
    average_row = df.mean()

    # Append the average row to the DataFrame
    df.loc['Average'] = average_row


    print(df)

    df.round(1).to_csv(OUTPUT, index=True)

                        

    

main()