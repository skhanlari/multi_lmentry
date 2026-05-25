import re

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, swap_substrings


class RhymingWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        a = r"(una|la)"
        while_ = "(Mentre|ma|e)"
        exact_patterns = [
            rf"{query} fa rima con {answer}( e non con {distractor})?",
            rf"{query} è {a} rima di {answer}( e non di {distractor})?",
            rf"{query} e {answer} sono in rima",
            rf"{query} e {answer} son parole che rimano",
            rf"{a} rima di {query} è {answer}( e non {distractor})?",
            rf"{a} parola che rima con {query} è {answer}",
            rf"{query} e {answer} fanno rima, {while_} {distractor} non fa rima con nessuna",
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        more_exact_patterns_2 = [
            rf"{distractor} non fa rima con {query}(\s+{answer} fa rima con {query})?",
            rf"{query} non fa rima con {distractor}(\s+{query} fa rima con {answer})?",
            rf"{answer} fa rima con {query}\s+{distractor} non fa rima con {query}",
        ]
        exact_patterns.extend(more_exact_patterns_2)

        return exact_patterns

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        predictions_words = re.findall(r"\b[a-z]+\b", prediction)
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
