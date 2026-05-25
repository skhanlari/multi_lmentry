import re

from lmentry.scorers.it.scorer import (
    LMentryScorer,
    swap_substrings,
    the_word_regex,
    standardized_number_regex,
)


class MoreLettersScorer(LMentryScorer):

    def __init__(self):
        super().__init__()

        self.prefixes.extend(
            [
                "Delle parole {word1} e {word2}, ",
                "(Tra|Fra) le parole {word1} e {word2}, ",
                "(Tra|Fra) {word1} e {word2}, ",
            ]
        )

    def get_base_patterns(self, answer, distractor, diff):

        base_patterns = [
            rf"{answer} ha più lettere",
            rf"{answer} ha più lettere di {distractor}",
            rf"{answer} ha {diff} lettera(e) in più",
            rf"{answer} ha {diff} lettera(e) in più di {distractor}",
            rf"La parola che ha più lettere è {answer}",
            rf"{answer} è la parola con più lettere",
            rf"{answer} è più lunga",
            rf"{answer} è pià lunga di {distractor}",
            rf"{answer} è pià lunga di {distractor} per {diff} lettera(e)",
            rf"La parola più lunga è {answer}",
            rf"{answer} è la parola più lunga",
        ]
        # replace more with less or fewer and swap answer and distractor
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("più", "meno")
            for s in base_patterns
            if "più" in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace longer with shorter and swap answer and distractor
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("lunga", "corta")
            for s in base_patterns
            if "lunga" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None

        negative_patterns = [
            distractor,
            rf"{distractor} ha più lettere",
            rf"{distractor} ha più lettere di {answer}",
            rf"{distractor} ha {diff} lettera(e) in più",
            rf"{distractor} ha {diff} lettera(e) in più di {answer}",
            rf"La parola con più lettere è {distractor}",
            rf"{distractor} è la parola che ha più lettere",
            rf"{distractor} è più lunga",
            rf"{distractor} è più lunga per {diff} lettera(e)",
            rf"{distractor} è più lunga di {answer}",
            rf"{distractor} è più lunga di {answer} per {diff} lettera(e)",
            rf"La parola più lunga è {distractor}",
            rf"{distractor} è la parola più lunga",
        ]
        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        # the model correctly picks the longer word, but wrongly states the difference in letters:
        number_regex = r"(?P<number>\d+|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|zero|solo una|una sola|una singola|un'unica)"
        wrong_diff_patterns = [
            rf"{answer} ha {number_regex} lettera(e) in più",
            rf"{answer} ha {number_regex} lettera(e) in più di {distractor}",
            rf"{answer} è più lunga per {number_regex} lettera(e)",
            rf"{answer} è più lunga di {distractor} per {number_regex} lettera(e)",
        ]
        for wrong_diff_pattern in wrong_diff_patterns:
            if res := re.match(wrong_diff_pattern, prediction):
                predicted_diff = res.group("number")
                if not re.match(diff, predicted_diff):
                    score = 0
                    certainty = 1
                    break

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]

        distractor = metadata["distractor"]
        diff = len(answer) - len(distractor)
        word1 = metadata["word1"]
        word2 = metadata["word2"]

        answer = the_word_regex(answer)
        distractor = the_word_regex(distractor)
        diff = standardized_number_regex(diff)
        word1 = the_word_regex(word1)
        word2 = the_word_regex(word2)

        score, certainty = self.negative_scorer(prediction, answer, distractor, diff)
        if score is not None:
            return score, certainty

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, distractor, diff)
        self.prefix_kwargs = {"word1": word1, "word2": word2}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
