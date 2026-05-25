import re

from lmentry.scorers.fa.scorer import LMentryScorer, swap_substrings, the_number_regex


class SmallerNumberScorer(LMentryScorer):
    """This class was created by simply mirroring `BiggerNumberScorer`"""

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
             
            "،{n2} و {n1} از بین اعداد",
            "،{n2} و {n1} بین اعداد ",
            "،{n2} و {n1} در میان اعداد ",
        ])


    def get_base_patterns(self, answer, distractor):

        smaller = r"(کوچک‌تر|کمتر)"
        bigger = r"(بزرگ‌تر|بیشتر)"

        base_patterns = [
            rf"است {smaller} {answer}",
            rf"است {smaller} {distractor} از {answer}",
            rf"است {smaller} عدد {answer}",
            rf"است {smaller} {answer} عدد  ",
        ]

        # replace smaller with bigger and swap
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace(smaller, bigger)
            for s in base_patterns
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        alphanumeric_pattern = r"[آ-ی\d]+"
        all_alphanumeric_words = re.findall(alphanumeric_pattern, prediction)

        if len(set(all_alphanumeric_words)) == 1:
            answer = str(answer)
            if all_alphanumeric_words[0] != answer:
                score = 0
                certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        n1 = metadata["n1"]
        n2 = metadata["n2"]
        answer = metadata["answer"]

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        distractor = metadata["distractor"]

        answer = the_number_regex(answer)
        distractor = the_number_regex(distractor)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, distractor)
        self.prefix_kwargs = {"n1": n1, "n2": n2}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty