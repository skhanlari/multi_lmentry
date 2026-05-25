import re

from lmentry.scorers.fa.scorer import LMentryScorer


class AnyWordsFromCategoryScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer):

        base_patterns = [
            rf"^({answer}\b",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    @staticmethod
    def negative_scorer(prediction, answer):
        score = None
        certainty = None

        if answer == "بله":
            opposite_pattern = r"(نه|خیر)"
        else:
            opposite_pattern = r"(بله|آره|اره)"

        if re.match(rf"{opposite_pattern}\.?$", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        num_words = metadata["num_words"]
        num_distractors = metadata["num_distractors"]

        answer = "بله" if num_words > num_distractors else "نه"

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty
        
        if answer == "بله":
            answer_pattern = r"(بله|آره|اره)"
        else:
            answer_pattern = r"(نه|خیر)"

        score, certainty = self._simple_scorer(prediction, answer_pattern)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer)
        score, certainty = self.certainty_scorer(prediction, base_patterns)

        return score, certainty