import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, the_letter_regex


class StartsWithLetterScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        word_ = r"(کلمه|پاسخ)"
        possible = r"ممکن"
        starts = r"(شروع می‌شود|آغاز می‌شود)"
        a = r"(یک)"


        base_patterns = [
            rf"است {possible} {word_} {a} {word} ",
            rf"است {possible} {word_} {word}",
            rf"{starts} {letter} با {word}",
            rf"{starts} {letter} کلمه‌ای است که با {word}",
            rf"است {letter}کلمه‌ای است که حرف اول آن {word}",
            rf"{starts} {letter} یک کلمه ای است که با {word}",
            rf"است{word}حرف اول {letter}",
        ]

        return base_patterns + self.get_shared_patterns(target=word)

    def negative_scorer(self, prediction, letter):
        score, certainty = None, None

        prediction_words = re.findall(r"[آ-ی]+", prediction)

        if all([word[0] != letter for word in prediction_words]):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        letter = example["metadata"]["letter"]

        score, certainty = self.negative_scorer(prediction, letter)
        if score is not None:
            return score, certainty

        after_the_letter = r"[آ-ی]*"  # allow single letter words

        word = rf"{letter}{after_the_letter}"
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty