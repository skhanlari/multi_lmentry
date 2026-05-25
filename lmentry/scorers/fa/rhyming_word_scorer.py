import re

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex, swap_substrings


class RhymingWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        a = r"(یک|این)"
        while_ = r"(در حالی که|اما|و)"
        doesnt = r"(نمی‌کند|نمی‌کند|نیست)"

        exact_patterns = [
            rf"{query} با {answer} هم‌قافیه است( و نه با {distractor})?",
            rf"{query} {a} کلمه هم‌قافیه با {answer} است( و نه با {distractor})?",
            rf"{query} و {answer} هم‌قافیه هستند",
            rf"{query} و {answer} کلماتی هستند که هم‌قافیه‌اند",
            rf"{a} کلمه هم‌قافیه با {query}، {answer} است( و نه {distractor})?",
            rf"{a} کلمه‌ای که با {query} هم‌قافیه است {answer} است",
            rf"{query} و {answer} هم‌قافیه هستند، {while_} {distractor} هم‌قافیه نیست",
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        more_exact_patterns_2 = [
            rf"{distractor} با {query} هم‌قافیه نیست(\s+{answer} با {query} هم‌قافیه است)?",
            rf"{query} با {distractor} هم‌قافیه نیست(\s+{query} با {answer} هم‌قافیه است)?",
            rf"{answer} با {query} هم‌قافیه است\s+{distractor} با {query} هم‌قافیه نیست",
        ]
        exact_patterns.extend(more_exact_patterns_2)

        return exact_patterns

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        prediction_words = re.findall(r"[آ-ی]+", prediction)

        if len(prediction_words) == 1 and prediction_words[0] != answer:
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