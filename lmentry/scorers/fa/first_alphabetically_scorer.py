import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex


class FirstAlphabeticallyScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "،به ترتیب الفبایی",
            "،در ترتیب الفبایی",
            "،{word2} و {word1} از بین کلمات ",
            "،{word2} و {word1} بین کلمات ",
        ])

        self.suffixes.extend([
            " به ترتیب الفبایی",
            " در ترتیب الفبایی",
            " در فرهنگ لغت",
        ])

    def get_base_patterns(self, answer, distractor):

        comes = r"(می‌آید|قرار می‌گیرد|است)"
        base_patterns = [
            rf"{comes} {answer} اول",
            rf"{comes} {distractor} دوم",
            rf"{comes} {distractor} بعد",
            rf"{comes} {distractor} قبل از {answer} ",
            rf"{comes} {distractor} بعد از {answer}",
        ]
        return base_patterns + self.get_shared_patterns(target=answer)

    def get_exact_patterns(self, answer):
        return [
            rf"{answer}",
        ]

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        # Persian-compatible word extraction
        predictions_words = re.findall(r"[آ-ی]+", prediction)

        if len(predictions_words) == 1 and predictions_words[0] != answer:
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        distractor = metadata["distractor"]
        word1 = metadata["word1"]
        word2 = metadata["word2"]

        answer = the_word_regex(answer)
        distractor = the_word_regex(distractor)
        word1 = the_word_regex(word1)
        word2 = the_word_regex(word2)

        exact_patterns = self.get_exact_patterns(answer)
        for exact_pattern in exact_patterns:
            score, certainty = self._simple_scorer(prediction, answer=exact_pattern)
            if score:
                return score, certainty

        base_patterns = self.get_base_patterns(answer, distractor)
        self.prefix_kwargs = {"word1": word1, "word2": word2}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty