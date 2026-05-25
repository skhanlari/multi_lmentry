import re

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, the_letter_regex


class StartsWithLetterScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, letter, word):

        word_ = r"(parola|risposta)"
        possible = r"possibile"
        starts = r"(inizio|inizia)"
        a = r"una"
        with_ = r"(per|con)"

        base_patterns = [
            rf"{a} {possible} {word_} è {word}",
            rf"{word} è {a} {possible} {word_}",
            rf"{word} {starts} {with_} {letter}",
            rf"{word} è {a} parola che {starts} {with_} {letter}",
            rf"{word} è {a} parola {with_} {letter}",
            rf"Una parola che inizia {with_} {letter} è {word}",
            rf"{word} è {a} parola la cui prima lettera è {letter}",
            rf"{word} è {a} parola la cui prima lettera è la lettera {letter}",
        ]
        return base_patterns + self.get_shared_patterns(target=word)

    def negative_scorer(self, prediction, letter):
        score, certainty = None, None

        prediction_words = re.findall(r"\b[a-z]+\b", prediction)

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

        after_the_letter = (
            r"\w*" if letter in {"a", "i", "o", "e"} else r"\w+"
        )  # Allow single letter words

        word = rf"{letter}-?{after_the_letter}"  # the "-" is for cases like e-mail and x-ray
        word = the_word_regex(word)

        score, certainty = self._simple_scorer(prediction, word)
        if score:
            return score, certainty

        letter = the_letter_regex(letter)

        base_patterns = self.get_base_patterns(letter, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
