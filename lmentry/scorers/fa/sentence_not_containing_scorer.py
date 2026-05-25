import re

from lmentry.scorers.fa.scorer import LMentryScorer


class SentenceNotContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()

        
        if re.search(rf"(?<![آ-ی]){word}(?![آ-ی])", prediction):
            score = 0
            certainty = 1
        else:
            prediction_words = re.findall(r"[آ-ی]+", prediction)

            if len(prediction_words) == 0:
                score = 0
                certainty = 1

            elif len(prediction_words) == 1:
                # one word → not really a sentence
                score = 0
                certainty = 0

            else:
                # sentence with multiple words
                if word in prediction:
                    # substring case (e.g., کتاب inside کتابخانه)
                    score = 1
                    certainty = 0
                else:
                    # clean case → word not present at all
                    score = 1
                    certainty = 1

        return score, certainty