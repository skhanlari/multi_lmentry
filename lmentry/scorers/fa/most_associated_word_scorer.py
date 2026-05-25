import re

from lmentry.scorers.fa.scorer import (
    LMentryScorer,
    the_word_regex,
    the_words_regex,
    swap_substrings,
)


class MostAssociatedWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        self.prefixes.extend([r"{words}[^آ-ی0-9_]*"])

        self.suffixes.extend([r"[^آ-ی0-9_]*{words}"])

    def get_base_patterns(self, answer, category, words):

        most_associated = r"(بیشترین ارتباط را با|بیشتر مرتبط با|بیشتر مربوط به)"

        base_patterns = [
            rf"کلمه‌ای که {most_associated} {category} است {answer} است",
            rf"کلمه‌ای که {most_associated} {category} در میان {words} است {answer} است",
            rf"{answer} کلمه‌ای است که {most_associated} {category} است",
            rf"{answer} بیشترین ارتباط را با {category} دارد",
            rf"{answer} در میان {words} بیشترین ارتباط را با {category} دارد",
            rf"کلمه‌ای که بیشترین ارتباط را با {category} دارد {answer} است",
        ]

        # swap answer and category
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=category)
            for s in base_patterns
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def get_exact_patterns(self, answer, category):
        return [
            rf"{answer}",
            rf"{category} - {answer}"
        ]

    def negative_scorer(self, prediction, answer):
        score, certainty = None, None

        if not re.search(rf"{answer}", prediction):
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

        category = metadata["category"]
        distractors = metadata["distractors"]
        answer_index = metadata["answer_index"]
        words = distractors[:answer_index] + [answer] + distractors[answer_index:]

        answer = the_word_regex(answer)
        category = the_word_regex(category)

        exact_patterns = self.get_exact_patterns(answer, category)
        for exact_pattern in exact_patterns:
            score, certainty = self._simple_scorer(prediction, answer=exact_pattern)
            if score:
                return score, certainty

        words = the_words_regex(words)

        of = r"(از|در میان|بین)"
        options = r"(کلمات|گزینه‌ها)"
        given = r"(داده‌شده|ارائه‌شده|فهرست‌ شده)"
        words = rf"{of} ({words}|{options} {given}|{given} {options})"

        base_patterns = self.get_base_patterns(answer, category, words)

        self.prefix_kwargs = {"words": words}
        self.suffix_kwargs = {"words": words}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty