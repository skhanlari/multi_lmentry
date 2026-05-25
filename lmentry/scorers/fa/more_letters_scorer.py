import re

from lmentry.scorers.fa.scorer import (
    LMentryScorer,
    swap_substrings,
    the_word_regex,
    standardized_number_regex,
)


class MoreLettersScorer(LMentryScorer):

    def __init__(self):
        super().__init__()

        self.prefixes.extend([
            "از بین کلمات {word1} و {word2}، ",
            "بین کلمات {word1} و {word2}، ",
        ])

    def get_base_patterns(self, answer, distractor, diff):

        base_patterns = [
            rf"حروف بیشتری دارد {answer}",
            rf"حروف بیشتری دارد {distractor} نسبت به {answer}",
            rf"حرف بیشتر دارد {diff} {answer}  ",
            rf"دارد{distractor} حرف بیشتر از {diff} {answer}",
            rf"است {answer} کلمه‌ای که حروف بیشتری دارد  ",
            rf"کلمه‌ای است که حروف بیشتری دارد {answer}",
            rf"بلندتر است {answer} ",
            rf"بلندتر است {distractor}از {answer}",
            rf"است{answer} حرف بلندتر از {diff} {distractor}",
            rf"کلمه بلندتر است {answer}",
            rf"کلمه‌ای که بلندتر است {answer}",
            rf"بلندتر است {answer} کلمه ",
        ]

        # replace more with less and swap answer and distractor
        more_base_patterns1 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("بیشتر", "کمتر")
            for s in base_patterns
            if "بیشتر" in s
        ]
        base_patterns.extend(more_base_patterns1)

        # replace longer with shorter and swap
        more_base_patterns2 = [
            swap_substrings(s, subs1=answer, subs2=distractor).replace("بلندتر", "کوتاه‌تر")
            for s in base_patterns
            if "بلندتر" in s
        ]
        base_patterns.extend(more_base_patterns2)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, distractor, diff):
        score, certainty = None, None
        
        negative_patterns = [
            distractor,
            rf" حروف بیشتری دارد {distractor}",
            rf"حروف بیشتری دارد {distractor} نسبت به {answer}",
            rf"حرف بیشتر دارد {diff} {distractor}",
            rf"دارد {distractor} حرف بیشتر از {diff} {answer}",
            rf"است {distractor} کلمه‌ای که حروف بیشتری دارد",
            rf"کلمه‌ای است که حروف بیشتری دارد {distractor}",
            rf" بلندتر است {distractor}",
            rf"بلندتر است {distractor} از {answer} ",
            rf"است{distractor} حرف بلندتر از {diff} {answer}",
            rf"کلمه بلندتر است {distractor}",
            rf"کلمه‌ای که بلندتر است {distractor}",
            rf"بلندتر است {distractor} کلمه ",
        ]

        for negative_pattern in negative_patterns:
            negative_pattern = self.normalize_string(negative_pattern)
            if re.match(rf"{negative_pattern}\.?$", prediction):
                score = 0
                certainty = 1
                break

        number_regex = r"(?P<number>\d+|یک|دو|سه|چهار|پنج|شش|هفت|هشت|نه|ده|صفر|یک عدد|تنها یک)"
        
        wrong_diff_patterns = [
            rf" حرف بیشتر دارد{number_regex} {answer}",
            rf"دارد {answer} حرف بیشتر از {number_regex} {distractor}",
            rf"حرف بلندتر است {number_regex} {answer}",
            rf"است {answer} حرف بلندتر از {number_regex} {distractor}",
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