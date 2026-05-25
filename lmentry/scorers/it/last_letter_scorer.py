import re

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, the_letter_regex


class LastLetterScorer(LMentryScorer):
    """This class was created by simply mirroring `FirstLetterScorer`"""

    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, word):

        of = "(di|in)"
        ends = "(finisce|termina)"

        base_patterns = [
            rf"L'ultima lettera è {answer}",
            rf"L'ultima lettera {of} {word} è {answer}",
            rf"{answer} è l'ultima lettera {of} {word}",
            rf"{word} {ends} con {answer}",
            rf"La lettera con cui {word} {ends} è {answer}",
            rf"La lettera con la quale {word} {ends} è {answer}",
            rf"{answer} è l'ultima lettera {of} {word}",
            rf"{word}: {answer}",
            rf"Ultima lettera: {answer}",
            rf"La lettera finale è {answer}",
            rf"La lettera finale {of} {word} è {answer}",
            rf"{answer} è la lettera finale",
            rf"{answer} è l'ultima lettera",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        if not re.search(rf"\b{answer}\b", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]
        word = metadata["word"]

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        answer = the_letter_regex(answer)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        word = the_word_regex(word)
        base_patterns = self.get_base_patterns(answer, word)

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
