import string

from lmentry.scorers.it.ends_with_letter_scorer import EndsWithLetterScorer
from lmentry.tasks.task import LMentryTask


class EndsWithLetter(LMentryTask):

    scorer_cls = EndsWithLetterScorer

    def __init__(self, name="ends_with_letter"):
        super().__init__(name)
        
        self.canonical_template = 'کلمه ای بنویسید که به حرف "{letter}" ختم شود'
        self.second_template = 'کلمه ای بنویسید که آخرین حرف آن "{letter}" باشد'
        self.third_template = 'کلمه ای بنویسید که به "{letter}" ختم شود'
        self.all_templates = [self.canonical_template, self.second_template, self.third_template]
        self.parameter_names = ["letter"]

    def _create_input(self, template, letter):
        input_ = template.format(letter=letter)
        return input_

    def create_data(self, task_data_path=None):

        # create examples
        examples = {}
        letters = string.ascii_lowercase
        persian_letters = [
                    "ا","ب","پ","ت","ث","ج","چ","ح","خ",
                    "د","ذ","ر","ز","ژ","س","ش","ص","ض",
                    "ط","ظ","ع","غ","ف","ق","ک","گ","ل",
                    "م","ن","و","ه","ی","آ"
                ]

        # it_letters = set(letters) | {"à", "è", "é", "ì", "ò", "ù", "'"}
        # allowed_letters = set(it_letters) - {"b", "c", "d", "f", "g", "h", "j", "k", "m", "q", "v", "w", "x", "y", "z"}
        allowed_letters = persian_letters
        allowed_letters = sorted(allowed_letters)

        for template_id, template in enumerate(self.all_templates):
            for letter in allowed_letters:
                # create the input
                input_ = self._create_input(template, letter)

                # create the metadata
                metadata = dict()
                metadata["letter"] = letter
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
    task = EndsWithLetter()
