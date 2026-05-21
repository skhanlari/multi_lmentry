import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, swap_substrings


class HomophoneScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        _but = r"(\.,)? (اما|ولی)"
        like = r"(مثل|شبیه|مشابه)"

        exact_patterns = [
            rf"{query} مثل {answer} تلفظ می‌شود",
            rf"{query} {like} {answer} تلفظ می‌شود{_but} نه {like} {distractor}",
            rf"{query} {like} {answer} تلفظ می‌شود{_but} {distractor} نه",
            rf"{query} و {answer} مثل هم تلفظ می‌شوند",
            rf"{query} بیشتر {like} {answer} تلفظ می‌شود",
            rf"{query} بیشتر {like} {answer} تلفظ می‌شود تا {distractor}",
            rf"{query} هم‌آوا با {answer} است",
            rf"{query} یک کلمهٔ هم‌آوا با {answer} است",
            rf"{query} و {answer} هم‌آوا هستند",
            rf"{query} و {answer} کلمات هم‌آوا هستند",
            rf"هم‌آوای {query} {answer} است",
            rf"کلمهٔ هم‌آوای {query} {answer} است",
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        return exact_patterns

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

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        answer = the_word_regex(answer)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        query = metadata["query"]
        distractor = metadata["distractor"]

        query = the_word_regex(query)
        distractor = the_word_regex(distractor)

        exact_patterns = self.get_exact_patterns(answer, query, distractor)

        for exact_pattern in exact_patterns:
            score, certainty = self._simple_scorer(prediction, exact_pattern)
            if score:
                return score, certainty
        else:
            score = 0
            certainty = 0

        return score, certainty