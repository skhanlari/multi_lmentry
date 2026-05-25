import re

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex, swap_substrings


class HomophoneScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_exact_patterns(answer, query, distractor):

        _but = r"(\.,)? (ma|e)"
        like = "(come|simile a|in modo simile a|in maniera simile a|allo stesso modo di|alla stessa maniera di|nello stesso modo di|nello stesso modo di)"

        exact_patterns = [
            rf"{query} si pronuncia {like} {answer}",
            rf"{query} si pronuncia {like} {answer}{_but} non {like} {distractor}",
            rf"{query} si pronuncia {like} {answer}{_but} {distractor} no",
            rf"{query} e {answer} si pronunciano (uguali|nello stesso modo|nella stessa maniera|allo stesso modo|alla stessa maniera)",
            rf"{query} si pronuncia di più {like} {answer}",
            rf"{query} si pronuncia di più {like} {answer} che {distractor}",
            rf"{query} è un omofono di {answer}",
            rf"{query} è una parola omofona a {answer}",
            rf"{query} e {answer} sono omofoni",
            rf"{query} e {answer} sono parole omofone",
            rf"L'omofono di {query} è {answer}",
            rf"La parola omofona di {query} è {answer}",
        ]

        # swap answer and query
        more_exact_patterns = [
            swap_substrings(s, subs1=answer, subs2=query) for s in exact_patterns
        ]
        exact_patterns.extend(more_exact_patterns)

        return exact_patterns

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        if not re.search(rf"\b{answer}\b", prediction):
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
