import json
import pandas as pd
import random
import os

# import nltk
# nltk.download("wordnet")
# nltk.download("omw-1.4")

from nltk.corpus import wordnet as wn

def get_random_word():
    with open("../pt_br/nouns_by_category.json") as f:
        nouns_by_category = json.load(f)

    random_cat = random.choice(list(nouns_by_category.keys()))
    random_word = random.choice(nouns_by_category[random_cat])

    return random_word

def _get_synonyms(word1, word2):
    def find_first_synonym(word):
        synsets = wn.synsets(word, lang="por")
        synonyms = set()

        for synset in synsets:
            lemmas = synset.lemmas("por")
            synonyms.update(lemma.name() for lemma in lemmas
                            if lemma.name() != word
                            and "_" not in lemma.name())

        try:
            first_synonym = random.choice(list(synonyms))
        except:
            first_synonym = ""

        print(f"get_synonyms({word1}, {word2}): {first_synonym}")
        return first_synonym

    first_synonym_word1 = find_first_synonym(word1)

    if not first_synonym_word1:
        first_synonym_word2 = find_first_synonym(word2)
        return first_synonym_word2

    return first_synonym_word1

def get_synonyms(word) -> list:
    synsets = wn.synsets(word, lang="por")

    synonyms = set()

    for synset in synsets:
        lemmas = synset.lemmas("por")
        lemmas = map(lambda lemma: lemma.name().lower())
        lemmas = filter(lambda lemma: lemma != word and "_" not in lemma)
        synonyms.update(lemmas)

    return list(synonyms)

def fill_homophones():
    df = pd.read_csv("pt_br_homophones.csv")
    df_filled = pd.DataFrame()

    for _, row in df.iterrows():
        word, homophone = row["word"], row["homophone"]

        for attr, term in zip(["semantic1", "semantic2"], [word, homophone]):
            try:
                row[attr] = get_synonyms(term)[0]
            except:
                # no synonyms different from the word itself; use random word
                row[attr] = get_random_word()

        for attr in ["easy1", "easy2"]:
            row[attr] = get_random_word()

            assert row["semantic1"] != row["semantic2"] != row["easy1"] != row["easy2"]

        df_filled = df_filled._append(row, ignore_index=True)

    with open("pt_br_homophones.csv", "w+") as out_file:
        df_filled.to_csv(out_file, index=False)

if __name__ == "__main__":
    random.seed(1)
    fill_homophones()
