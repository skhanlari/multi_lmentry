import re

from lmentry.scorers.gl.scorer import LMentryScorer, the_word_regex, swap_substrings


class RhymingWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        a = r"(unha|a)"
        while_ = "(mentres que|pero|mais|e)"
        exact_patterns = [
            rf"{query} rima con {answer}( e non con {distractor})?",
            rf"{query} é {a} rima de {answer}( e non de {distractor})?",
            rf"{query} e {answer} riman",
            rf"{query} e {answer} son palabras que riman",
            rf"{a} rima de {query} é {answer}( e non {distractor})?",
            rf"{a} palabra que rima con {query} é {answer}",
            rf"{query} e {answer} riman, {while_} {distractor} non rima con ningunha"
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        more_exact_patterns_2 = [
            rf"{distractor} non rima con {query}(\s+{answer} rima con {query})?",
            rf"{query} non rima con {distractor}(\s+{query} rima con {answer})?",
            rf"{answer} rima con {query}\s+{distractor} non rima con {query}",
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
