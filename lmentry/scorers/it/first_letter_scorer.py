import re

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, the_letter_regex


class FirstLetterScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, word):

        of = "(di|in)"
        starts = "(inizio|inizia)"

        base_patterns = [
            rf"La prima lettera è {answer}",
            rf"La prima lettera {of} {word} è {answer}",
            rf"{answer} è la prima lettera {of} {word}",
            rf"{word} {starts} con {answer}",
            rf"La lettera con cui {word} {starts} è {answer}",
            rf"La lettera con la quale {word} {starts} è {answer}",
            rf"{answer} è la lettera iniziale {of} {word}",
            rf"{word}: {answer}",
            rf"Prima lettera: {answer}",
            rf"La lettera iniziale è {answer}",
            rf"La lettera iniziale {of} {word} è {answer}",
            rf"{answer} è la lettera iniziale",
            rf"{answer} è la prima lettera",
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
