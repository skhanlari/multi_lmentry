import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, the_letter_regex


class WordNotContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        base_patterns = [
            rf"ندارد {letter} حرف {word}",
            rf"نیست {letter} شامل حرف {word} ",
            rf"نیست {letter} دارای حرف {word} ",
            rf"نیست {letter} کلمه‌ای که شامل حرف {word}",
            rf"است{letter} بدون حرف {word}",
        ]

        return base_patterns

    def negative_scorer(self, prediction, input_, letter):
        score, certainty = None, None

        input_ = self.normalize_string(input_)

        # replace letter in input to avoid exact copy trick
        input_ = input_.replace(f"\"{letter}\"", r'"[آ-ی]"')

        if re.match(input_ + "$", prediction):
            score = 0
            certainty = 1

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

        # allowed Persian word (at least 2 letters or "و")
        allowed_word = rf"([آ-ی\u200c]{{2,}})"

        if letter != "و":
            allowed_word += r"|(و)"

        allowed_word = "(" + allowed_word + ")"
        allowed_word = the_word_regex(allowed_word)

        score, certainty = self._simple_scorer(
            prediction,
            allowed_word,
            allow_answer_pattern_repetitions=False
        )
        if score:
            return score, certainty

        letter = the_letter_regex(letter)
        base_patterns = self.get_base_patterns(letter, allowed_word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty