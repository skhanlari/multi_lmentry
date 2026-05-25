import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, the_letter_regex


class WordContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        that = r"(که)"
        contains = r"(شامل|دارای|دارد)"

        base_patterns = [
            rf"است{letter} {contains} ? ({that} یک کلمه) {word}",
            rf"است {letter} شامل حرف {word}",
            rf"است{letter} دارای حرف {word}",
        ]

        return base_patterns + self.get_shared_patterns(target=word)

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        word = (
            rf"[آ-ی]*{letter}[آ-ی]*"
            if letter in {"و"}
            else rf"(([آ-ی]+{letter}[آ-ی]*)|([آ-ی]*{letter}[آ-ی]+))"
        )

        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)
        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty