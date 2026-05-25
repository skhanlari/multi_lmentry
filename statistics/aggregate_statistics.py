import pandas as pd

def preprocess_dataset(df, lang):
    df = df.drop("num_examples_per_template", axis=1)
    df = df.drop("num_templates", axis=1)
    df = df.rename({"total_examples": lang}, axis=1)
    return df

df_en = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_en.csv", index_col=0), "en")
df_it = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_it.csv", index_col=0), "it")
df_es = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_es.csv", index_col=0), "es")
df_de = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_de.csv", index_col=0), "de")
df_ca = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_ca.csv", index_col=0), "ca")
df_gl = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_gl.csv", index_col=0), "gl")
df_eu = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_eu.csv", index_col=0), "eu")
df_ko = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_ko.csv", index_col=0), "ko")
df_pt_br = preprocess_dataset(pd.read_csv("statistics/dataset_statistics/statistics_pt_br.csv", index_col=0), "pt_br")

dt_sum = pd.concat([
    df_en,
    df_it,
    df_es,
    df_de,
    df_ca,
    df_gl,
    df_eu,
    df_ko,
    df_pt_br
],axis=1)

dt_sum.sort_index(key=lambda x: x.str.lower()).to_csv("statistics/dataset_statistics/statistics_aggregated.csv")