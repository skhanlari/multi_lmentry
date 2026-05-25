import re

from lmentry.scorers.gl.scorer import LMentryScorer


class AnyWordsFromCategoryScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer):

        base_patterns = [
            rf"^{answer}\b",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    @staticmethod
    def negative_scorer(prediction, answer):
        score = None
        certainty = None

        opposite_answer = "non" if answer == "si" else "si"
        if re.match(rf"{opposite_answer}\.?$", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        num_words = metadata["num_words"]
        num_distractors = metadata["num_distractors"]

        answer = "si" if num_words > num_distractors else "non"

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer)
        score, certainty = self.certainty_scorer(prediction, base_patterns)

        return score, certainty
