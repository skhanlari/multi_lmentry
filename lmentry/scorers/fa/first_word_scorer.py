import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_sentence_regex, the_word_regex


class FirstWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "{sentence} در جمله ",
        ])

        self.suffixes.extend([
            "{sentence} (از|در)",
        ])

    def get_base_patterns(self, answer, sentence):

        base_patterns = [
            rf"است {answer} اولین کلمه ",
            rf"است {sentence} اولین کلمه (از|در) {answer} ",
            rf"اولین کلمه است {answer}",
            rf"شروع می‌شود {answer} با {sentence}",
            rf"{answer} :اولین کلمه",
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
        sentence = metadata["sentence"]

        answer = answer.lower()
        sentence = sentence.lower()

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        answer = the_word_regex(answer)
        sentence = the_sentence_regex(sentence)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, sentence)
        self.prefix_kwargs = {"sentence": sentence}
        self.suffix_kwargs = {"sentence": sentence}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty