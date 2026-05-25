import re

from lmentry.scorers.it.scorer import (
    LMentryScorer,
    swap_substrings,
    the_word_regex,
    standardized_number_regex,
)


class LessLettersScorer(LMentryScorer):
    """This class was created by simply mirroring `MoreLettersScorer`"""

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
            rf"{answer} ha meno lettere",
            rf"{answer} ha meno lettere di {distractor}",
            rf"{answer} ha {diff} lettere in meno",
            rf"{answer} ha {diff} lettere in meno di {distractor}",
            rf"La parola con meno lettere è {answer}",
            rf"{answer} è la parola con meno lettere",
            rf"{answer} è più corta",
            rf"{answer} è più corta di {distractor}",
            rf"{answer} è più corta di {distractor} per {diff} lettera(e)",
            rf"La parola più corta è {answer}",
            rf"{answer} è la parola più corta",
            rf"{answer} è più breve",
            rf"{answer} è più breve di {distractor}",
            rf"{answer} è più breve di {distractor} per {diff} lettera(e)",
            rf"La parola più breve è {answer}",
            rf"{answer} è la parola più breve",
        ]
        # replace less with more and swap answer and distractor
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("meno", "più")
            for s in base_patterns
            if "meno" in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace shorter with longer and swap answer and distractor
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("corta", "lunga")
            for s in base_patterns
            if "corta" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None

        negative_patterns = [
            distractor,
            rf"{distractor} ha meno lettere",
            rf"{distractor} ha meno lettere di {answer}",
            rf"{distractor} ha {diff} lettera(e) in meno",
            rf"{distractor} in {diff} lettera(e) in menos di {answer}",
            rf"La parola con meno lettere è {distractor}",
            rf"{distractor} è la parola che ha meno lettere",
            rf"{distractor} è più corta",
            rf"{distractor} è più corta di {diff} lettera(e)",
            rf"{distractor} è più corta di {answer}",
            rf"{distractor} è più corta di {answer} per {diff} lettera(e)",
            rf"La parola più corta è {distractor}",
            rf"{distractor} è la parola più corta",
            rf"{distractor} è più breve",
            rf"{distractor} è più breve di {answer}",
            rf"{distractor} è più breve di {answer} per {diff} lettera(e)",
            rf"La parola più breve è {distractor}",
            rf"{distractor} è la parola più breve",
        ]
        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        # the model correctly picks the shorter word, but wrongly states the difference in letters:
        number_regex = r"(?P<number>\d+|una|due|tre|quattro|cinque|sei|sette|otto|nove|dieci|zero|solo una|una sola|una singola|un'unica)"
        wrong_diff_patterns = [
            rf"{answer} ha {number_regex} lettera(e) in meno",
            rf"{answer} ha {number_regex} lettera(e) in meno di {distractor}",
            rf"{answer} è più corta per {number_regex} lettera(e)",
            rf"{answer} è più corta di {distractor} per {number_regex} lettera(e)",
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
        diff = len(distractor) - len(answer)
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
