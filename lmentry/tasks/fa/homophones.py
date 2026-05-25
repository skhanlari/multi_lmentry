from pathlib import Path

import pandas as pd

from lmentry.constants import HOMOPHONES_PATH
from lmentry.scorers.it.homophone_scorer import HomophoneScorer
from lmentry.tasks.task import LMentryTask


class Homophones(LMentryTask):

    scorer_cls = HomophoneScorer

    
    

    def __init__(self, name="homophones"):
        super().__init__(name)
        self.canonical_template = "کدام کلمه صدایش شبیه کلمه \"{word1}\"، \"{query}\" یا \"{word2}\" است؟"
        self.second_template = "از بین دو کلمه \"{word1}\" و \"{word2}\"، کدام یک صدایش بیشتر شبیه \"{query}\" است؟"
        self.third_template = "کدام کلمه هم آوای \"{word1}\"، \"{query}\" یا \"{word2}\" است؟"
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["query", "answer", "distractor"]

    def create_data(self, homophones_path: Path = None, task_data_path: Path = None):

        homophones_path = homophones_path or HOMOPHONES_PATH
        homophones_data = pd.read_csv(homophones_path)

        examples = {}
        for template_id, template in enumerate(self.all_templates):
            for _, row in homophones_data.iterrows():
                query = row["word"]
                homophone = row["homophone"]
                for distractor_name in ("easy1", "easy2", "semantic1", "semantic2"):
                    distractor = row[distractor_name]
                    for answer_index in (0, 1):
                        word1 = homophone if answer_index == 0 else distractor
                        word2 = distractor if answer_index == 0 else homophone

                        input_ = template.format(query=query, word1=word1, word2=word2)

                        # create the metadata
                        metadata = dict()
                        metadata["query"] = query
                        metadata["answer"] = homophone
                        metadata["distractor"] = distractor
                        metadata["distractor_type"] = distractor_name[:-1]
                        metadata["answer_index"] = answer_index
                        metadata["template_id"] = template_id

                        # create the example
                        example = dict()
                        example["input"] = input_
                        example["metadata"] = metadata
                        examples[str(len(examples) + 1)] = example

        # create the shared settings
        settings = dict()
        settings["canary"] = self.canary_string
        settings["name"] = self.name
        settings["num_examples_per_template"] = int(len(examples) / len(self.all_templates))
        settings["input_templates"] = self.all_templates

        # build the task_data
        task_data = dict()
        task_data["settings"] = settings
        task_data["examples"] = examples

        # save the data
        self.save_task_data(task_data)


if __name__ == '__main__':
    task = Homophones()
