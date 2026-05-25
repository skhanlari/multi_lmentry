import re
from typing import List

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex


class EndsWithWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        a_ = r"(یک)"
        last_ = r"(آخرین)"

        # we don't want an empty prefix here
        self.prefixes = [
            r"(جمله)( است)?(:)?",
            r"تمام می‌شود {word} جمله‌ ای که با {a_}",
            r"پایان می‌یابد {word} جمله‌ ای که با {a_}",
            r"است{word} آن {last_} جمله‌ ای که کلمه {a_}",
            r"است{word} جمله‌ای که کلمه آخر آن",
        ]

        self.suffixes = [r"$"]

        def _replace_arguments(xfixes: List[str]):
            for i in range(len(xfixes)):
                xfixes[i] = xfixes[i].format(
                    a_=a_, last_=last_, word="{word}"
                )

        _replace_arguments(self.prefixes)
        _replace_arguments(self.suffixes)

    def get_base_patterns(self, word):

        # minimal adaptation (no vowel logic like English)
        valid_word = r"(و|[آ-ی]{2,})"
        
        # Persian-compatible separator instead of [^a-zA-Z0-9_]
        allowed_sentence = rf".*{valid_word}[^آ-ی0-9_]+{word}[^آ-ی0-9_]*"

        base_patterns = [allowed_sentence]
        return base_patterns + self.get_shared_patterns(target=allowed_sentence)

    def negative_scorer(self, prediction, word):
        score, certainty = None, None

        # Persian-safe word match (no \b)
        if not re.search(rf"{word}", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        word = example["metadata"]["word"].lower()

        score, certainty = self.negative_scorer(prediction, word)
        if score is not None:
            return score, certainty

        word = the_word_regex(word)

        base_patterns = self.get_base_patterns(word)

        score, certainty = self._simple_scorer(prediction, base_patterns[0])
        if score:
            return score, certainty

        self.prefix_kwargs = {"word": word}
        self.suffix_kwargs = {"word": word}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty