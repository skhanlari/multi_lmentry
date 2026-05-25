import re

from lmentry.scorers.gl.scorer import LMentryScorer, swap_substrings, the_number_regex


class SmallerNumberScorer(LMentryScorer):
    """This class was created by simply mirroring `BiggerNumberScorer`
    """

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "Dos números {n1} e {n2}, ",
            "Entre os números {n1} e {n2}, ",
            "De entre os números {n1} e {n2}, ",
        ])

    def get_base_patterns(self, answer, distractor):

        bigger = r"(máis grande|maior)"
        smaller = r"(máis pequeno|menor)"

        base_patterns = [
            rf"{answer} é {smaller}",
            rf"{answer} é {smaller} que {distractor}",
            rf"{answer} é o número {smaller}",
            rf"O número {smaller} é {answer}",
        ]
        # replace smaller with bigger and swap answer and distractor
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace(smaller, bigger)
            for s in base_patterns
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        alphanumeric_pattern = r"\b[a-z\d]+\b"
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
