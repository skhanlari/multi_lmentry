import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_sentence_regex


class SentenceContainingScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()

        # ✅ Persian-safe word boundary
        allowed_sentence = rf"((.*(?<![آ-ی]){word}(?![آ-ی]).+)|(.+(?<![آ-ی]){word}(?![آ-ی]).*))"
        allowed_sentence = the_sentence_regex(allowed_sentence)

        if re.search(allowed_sentence, prediction):
            score = 1
            certainty = 1
        else:
            score = 0
            certainty = 1

        return score, certainty