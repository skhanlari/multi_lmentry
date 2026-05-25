import json
import random

from lmentry.constants import LMENTRY_WORDS_PATH
from lmentry.scorers.it.first_letter_scorer import FirstLetterScorer
from lmentry.tasks.task import LMentryTask


class FirstLetter(LMentryTask):

    scorer_cls = FirstLetterScorer

    def __init__(self, name="first_letter"):
        super().__init__(name)
        self.canonical_template = 'حرف اول کلمه "{word}" چیست؟'
        self.second_template = 'کلمه "{word}" با کدام حرف شروع می‌شود؟'
        self.third_template = 'حرف اول کلمه "{word}" را بنویسید'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["word", "answer"]

    @staticmethod
    def get_task_words():
        with open(LMENTRY_WORDS_PATH) as f:
            task_words = json.load(f)
        return task_words

    def _create_input(self, template, word):
        input_ = template.format(word=word)
        return input_

    def create_data(self, num_examples=1000, seed=None, task_data_path=None):

        # sample num_examples words from the data
        task_words = self.get_task_words()
        seed = seed or self.default_seed
        random.seed(seed)
        sampled_words = random.sample(list(task_words), min(len(list(task_words)), num_examples))
        task_words = {word: task_words[word] for word in sampled_words}

        # create examples
        examples = {}

        for template_id, template in enumerate(self.all_templates):
            for word, word_info in task_words.items():
                # create the input
                input_ = self._create_input(template, word)

                # create the metadata
                metadata = dict()
                metadata["word"] = word
                metadata["word_pos_tags"] = word_info["pos"]
                metadata["answer"] = word[0]
                metadata["template_id"] = template_id

                # create the example
                example = dict()
                example["input"] = input_
                example["metadata"] = metadata
                examples[f"{len(examples) + 1}"] = example

        # create the shared settings
        settings = dict()
        settings["canary"] = self.canary_string
        settings["name"] = self.name
        settings["num_examples_per_template"] = len(task_words)
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data)


if __name__ == '__main__':
    task = FirstLetter()
