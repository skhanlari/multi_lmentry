import json
import random
import string
from collections import defaultdict

from lmentry.constants import LMENTRY_WORDS_PATH
from lmentry.scorers.it.first_alphabetically_scorer import FirstAlphabeticallyScorer
from lmentry.tasks.task import LMentryTask


class FirstAlphabeticallyDifferentFirstLetter(LMentryTask):

    scorer_cls = FirstAlphabeticallyScorer

    def __init__(self, name="first_alphabetically_different_first_letter", words_path=None):
        super().__init__(name)
        self.canonical_template = "به ترتیب حروف الفبا، کدام یک از کلمات \"{word1}\" و \"{word2}\" اول می‌آید؟"
        self.second_template = "به ترتیب حروف الفبا، کدام کلمه اول می‌آید، \"{word1}\" یا \"{word2}\"؟"
        self.third_template = "از کلمات \"{word1}\" و \"{word2}\"، کدام کلمه به ترتیب حروف الفبا اول می‌آید؟"
        self.parameter_names = ["answer", "distractor"]
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.words_path = words_path or LMENTRY_WORDS_PATH

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        # get a dict of (<first letter>: <list of words word>) for all words
        with open(self.words_path) as f_words:
            words_data = json.load(f_words)
        es_words = []
        for word, _ in words_data.items():
            es_words.append(word)
        first_letter_to_words = defaultdict(list)
        for word in es_words:
            first_letter_to_words[word[0]].append(word)
        # remove entries where a letter has no words starting with it
        first_letter_to_words = {letter: words for letter, words in first_letter_to_words.items()}
        persian_letters = [
            "ا","ب","پ","ت","ث","ج","چ","ح","خ",
            "د","ذ","ر","ز","ژ","س","ش","ص","ض",
            "ط","ظ","ع","غ","ف","ق","ک","گ","ل",
            "م","ن","و","ه","ی","آ"
        ]

        all_letters = persian_letters
        first_letter_options = [letter for letter in persian_letters if letter in first_letter_to_words]

        # create examples
        examples = dict()
        for template_id, template in enumerate(self.all_templates):

            seed = seed or self.default_seed
            random.seed(seed)
            num_template_examples_created = 0
            already_seen_arguments: set[tuple[str, str, int]] = set()  # word1, word2, answer_index

            while num_template_examples_created < num_examples:
                # sample the two words
                first_letter_word1, first_letter_word2 = random.sample(first_letter_options, 2)
                word1 = random.choice(first_letter_to_words[first_letter_word1])
                word2 = random.choice(first_letter_to_words[first_letter_word2])

                answer_index = 0 if word1 < word2 else 1

                # we can already check for duplicates, but I'm determining the answer and distractor
                # to keep it consistent with other "first alphabetically" tasks
                answer, distractor = (word1, word2) if word1 < word2 else (word2, word1)
                if (answer, distractor, answer_index) in already_seen_arguments:
                    continue
                already_seen_arguments.add((answer, distractor, answer_index))

                input_ = template.format(word1=word1, word2=word2)

                # create the metadata
                metadata = dict()
                metadata["answer"] = answer
                metadata["distractor"] = distractor
                metadata["answer_index"] = answer_index
                metadata["word1"] = word1
                metadata["word2"] = word2
                metadata["template_id"] = template_id

                # create the example
                example = dict()
                example["input"] = input_
                example["metadata"] = metadata
                examples[str(len(examples) + 1)] = example

                num_template_examples_created += 1

        # create the shared settings
        settings = dict()
        settings["canary"] = self.canary_string
        settings["name"] = self.name
        settings["num_examples_per_template"] = num_examples
        settings["seed"] = seed
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data)


if __name__ == '__main__':
    task = FirstAlphabeticallyDifferentFirstLetter()
