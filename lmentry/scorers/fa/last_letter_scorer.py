import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, the_letter_regex


class LastLetterScorer(LMentryScorer):
    """This class was created by simply mirroring `FirstLetterScorer`"""

    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, word):

        of = r"(از|در)"
        ends = r"(تمام می‌شود|پایان می‌یابد)"

        base_patterns = [
            rf"آخرین حرف است{answer}",
            rf"است{word} {of} آخرین حرف {answer}",
            rf"است {answer} {word} {of} آخرین حرف ",
            rf"{ends} {answer} با {word}",
            rf"{ends} با آن {word} حرفی است که {answer}",
            rf"است{word} {of} حرف آخر {answer}",
            rf"{answer} :{word}",
            rf"{answer} :حرف آخر",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        # Persian-safe boundary 
        if not re.search(rf"(?<![آ-ی]){answer}(?![آ-ی])", prediction):
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