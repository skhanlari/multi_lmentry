import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, the_letter_regex


class EndsWithLetterScorer(LMentryScorer):
    """This class was created by simply mirroring `StartsWithLetterScorer`"""

    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        word_ = r"(کلمه|پاسخ)"
        possible = r"(ممکن)"
        a = r"(یک)"

        base_patterns = [
            rf"است {possible} {word_} {a} {word}",
            rf"است{possible} {word_} {word}",
            rf"تمام می‌شود {letter}با {word}",
            rf"تمام می‌شود {letter} با حرف {word}",
            rf"تمام می‌شود {letter} کلمه ای است که با {a} {word}",
            rf"تمام می‌شود {letter} کلمه ای است که با حرف {a} {word}",
            rf"است {letter} کلمه‌ای است که آخرین حرف آن {a} {word}",
            rf"است {letter} کلمه‌ای است که آخرین حرف آن حرف {a} {word}",
        ]

        return base_patterns + self.get_shared_patterns(target=word)

    def negative_scorer(self, prediction, letter):
        score, certainty = None, None

        # Persian-compatible word extraction
        prediction_words = re.findall(r"[آ-ی]+", prediction)

        if all([word[-1] != letter for word in prediction_words]):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        score, certainty = self.negative_scorer(prediction, letter)
        if score is not None:
            return score, certainty

        before_the_letter = r"\w*"

        word = rf"{before_the_letter}{letter}"
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty