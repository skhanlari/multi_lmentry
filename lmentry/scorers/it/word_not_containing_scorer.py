import re
import string

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, the_letter_regex


class WordNotContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        contains = r"(contiene|ha|include|ha al suo interno)"

        base_patterns = [
            rf"{word} (è una parola che )?non {contains} (la lettera )?{letter}",
            rf"Una parola che non {contains} {letter} è {word}",
            rf"{word} è una parola senza (la lettera )?{letter}",
        ]

        return base_patterns

    def negative_scorer(self, prediction, input_, letter):
        score, certainty = None, None

        # the prediction just repeats the input (maybe asking for a different letter now)
        input_ = self.normalize_string(input_)
        input_ = input_.replace(f'"{letter}"', r'"\w"')

        if re.match(input_ + "$", prediction):
            score = 0
            certainty = 1

        # the prediction is a single word that contains the letter
        if not re.search(r"\W", prediction) and letter in prediction:
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        input_ = example["input"]
        score, certainty = self.negative_scorer(prediction, input_, letter)
        if score is not None:
            return score, certainty

        allowed_letters = string.ascii_lowercase.replace(letter, "")
        allowed_word = rf"(\b[{allowed_letters}]{{2,}}\b)"
        if letter != "a":
            allowed_word += r"|(a)"
        if letter != "i":
            allowed_word += r"|(i)"
        if letter != "e":
            allowed_word += r"|(e)"
        if letter != "o":
            allowed_word += r"|(o)"
        if letter != "a" or letter != "i" or letter != "e" or letter != "o":
            allowed_word = "(" + allowed_word + ")"

        allowed_word = the_word_regex(allowed_word)

        # allowing the answer pattern to repeat is too lax for this task
        # e.g. "A giraffe is bending down to eat some grass." shouldn't be considered accurate for
        # the input "Write a word that doesn't contain the letter "c":"
        score, certainty = self._simple_scorer(
            prediction, allowed_word, allow_answer_pattern_repetitions=False
        )
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, allowed_word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
