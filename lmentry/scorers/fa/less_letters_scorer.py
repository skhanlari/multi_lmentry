import re

from lmentry.scorers.fa.scorer import (
    LMentryScorer,
    swap_substrings,
    the_word_regex,
    standardized_number_regex,
)


class LessLettersScorer(LMentryScorer):
    """This class was created by simply mirroring `MoreLettersScorer`"""

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
          " ،{word2} و {word1} از بین کلمات",
          " ،{word2} و {word1} بین کلمات",
        ])

    def get_base_patterns(self, answer, distractor, diff):

        base_patterns = [
            rf"حروف کمتری دارد {answer}",
            rf"حروف کمتری دارد {distractor} نسبت به {answer}",
            rf"حرف کمتر دارد {diff} {answer}  ",
            rf"دارد{distractor} حرف کمتر از {diff} {answer}",
            rf"است {answer} کلمه‌ای که حروف کمتری دارد  ",
            rf"کلمه‌ای است که حروف کمتری دارد {answer}",
            rf"کوتاه‌تر است {answer} ",
            rf"کوتاه‌تر است {distractor}از {answer}",
            rf"است{answer} حرف کوتاه‌تر از {diff} {distractor}",
            rf"کلمه کوتاه‌تر است {answer}",
            rf"کلمه‌ای که کوتاه‌تر است {answer}",
            rf"کوتاه‌تر است {answer} کلمه ",
        ]

        # replace less with more and swap answer and distractor
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("کمتر", "بیشتر")
            for s in base_patterns
            if "کمتر" in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace shorter with longer and swap answer and distractor
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("کوتاه‌تر", "بلندتر")
            for s in base_patterns
            if "کوتاه‌تر" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None

        negative_patterns = [
            distractor,
            rf" حروف کمتری دارد {distractor}",
            rf"حروف کمتری دارد {distractor} نسبت به {answer}",
            rf"حرف کمتر دارد {diff} {distractor}",
            rf"دارد {distractor} حرف کمتر از {diff} {answer}",
            rf"است {distractor} کلمه‌ای که حروف کمتری دارد",
            rf"کلمه‌ای است که حروف کمتری دارد {distractor}",
            rf" کوتاه‌تر است {distractor}",
            rf"کوتاه‌تر است {distractor} از {answer} ",
            rf"است{distractor} حرف کوتاه‌تر از {diff} {answer}",
            rf"کلمه کوتاه‌تر است {distractor}",
            rf"کلمه‌ای که کوتاه‌تر است {distractor}",
            rf"کوتاه‌تر است {distractor} کلمه ",
        ]

        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        # wrong difference handling
        number_regex = r"(?P<number>\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|صفر|یک عدد|یکتا|تنها یک)"
        wrong_diff_patterns = [
            rf" حرف کمتر دارد{number_regex} {answer}",
            rf"دارد {answer} حرف کمتر از {number_regex} {distractor}",
            rf"حرف کوتاه‌تر است {number_regex} {answer}",
            rf"است {answer} حرف کوتاه‌تر از {number_regex} {distractor}",
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