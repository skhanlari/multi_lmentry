import collections
import random

import nltk
import pandas as pd
from nltk.corpus import stopwords

from lmentry.constants import RESOURCES_DIR
from lmentry.scorers.it.word_before_scorer import WordBeforeScorer
from lmentry.tasks.task import LMentryTask


class WordBefore(LMentryTask):

    scorer_cls = WordBeforeScorer

    def __init__(self, name="word_before"):
        super().__init__(name)

        self.canonical_template = 'در جمله "{sentence}"، کدام کلمه درست پیش از کلمه "{word}" آمده است؟'
        self.second_template = 'در جمله "{sentence}"، کدام کلمه بلافاصله قبل از کلمه "{word}" آمده است؟'
        self.third_template = 'در جمله "{sentence}" کدام کلمه درست قبل از "{word}" آمده است؟'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["sentence", "query", "answer"]  # sentence is first because it includes the others

    def _create_input(self, template, sentence, word):
        input_ = template.format(sentence=sentence, word=word)
        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        with open(RESOURCES_DIR.joinpath("simple_sentences.csv")) as f:
            source_data = pd.read_csv(f)

        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):

            all_possible_positions = []

            # set random seed
            seed = seed or self.default_seed
            random.seed(seed)

            for _, row in source_data.iterrows():
                sentence = row["sentence"]

                # We remove punctuation.
                sentence = sentence.replace(".", "").replace("\"", "").replace(",", "").replace(":", "")

                # our sentences do not contain punctuation, so we can use space tokenization to get the last word
                sentence_tokenized = sentence.replace('.', '').replace("\"", "").replace(",", "").replace(":", "").split()
                sentence_lower_tokenized = sentence.lower().replace('.', '').replace("\"", "").replace(",", "").replace(":", "").split()
                word_counter = collections.Counter(sentence_lower_tokenized)

                for i, word in enumerate(sentence_tokenized[1:], start=1):
                    # when asking about the word before, we do not ask about the first word of the sentence.
                    if word_counter[word.lower()] > 1:
                        # we avoid asking about words that appear more than once in the sentence.
                        continue

                    # create the input
                    input_ = self._create_input(template, sentence, word)

                    # create the metadata
                    metadata = dict()
                    metadata["sentence"] = sentence
                    metadata["query"] = word
                    metadata["query_position_in_sentence"] = i
                    metadata["answer"] = sentence_tokenized[i-1]
                    metadata["template_id"] = template_id

                    # create the example
                    example = dict()
                    example["input"] = input_
                    example["metadata"] = metadata
                    all_possible_positions.append(example)

            # sample final examples
            sampled_examples = random.sample(all_possible_positions, num_examples)
            for example in sampled_examples:
                examples[str(len(examples) + 1)] = example

        # create the shared settings
        settings = dict()
        settings["canary"] = self.canary_string
        settings["name"] = self.name
        settings["seed"] = seed
        settings["num_examples_per_template"] = num_examples
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data)


if __name__ == '__main__':
    task = WordBefore()
