import re
from typing import List

from lmentry.scorers.fa.scorer import LMentryScorer, the_word_regex


class StartsWithWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        a_ = r"یک"

        # prefixes
        self.prefixes = [
            r"(جمله)( است)?(:)?",
            r"شروع می‌شود {word} جمله‌ که با {a_}",
            r"آغاز می‌شود {word} جمله‌ که با {a_}",
            r"است{word} جمله‌ که کلمه اول آن {a_} ",
        ]

        # suffixes
        self.suffixes = [
            r"شروع می‌شود {word} با",
            r"آغاز می‌شود {word} با",
            r"شروع می‌شود {word} یک جمله‌ که با ",
            r"آغاز می‌شود {word} یک جمله‌ که با ",
            r"است{word} جمله‌ که کلمه اول آن",
        ]

        def _replace_arguments(xfixes: List[str]):
            for i in range(len(xfixes)):
                xfixes[i] = xfixes[i].format(
                    a_=a_,
                    word="{word}"
                )

        _replace_arguments(self.prefixes)
        _replace_arguments(self.suffixes)

    def get_base_patterns(self, word):

        # Persian valid words: allow 1-letter "و" OR 2+ letters
        valid_word = r"(و|[آ-ی]{2,})"

        W = r"[^آ-ی0-9_]"

        allowed_sentence = rf"{W}*{word}{W}+{valid_word}.*"

        base_patterns = [
            allowed_sentence
        ]
        return base_patterns + self.get_shared_patterns(target=allowed_sentence)

    def negative_scorer(self, prediction, word):
        score, certainty = None, None

        # Persian word boundary fix (no \b)
        if not re.search(rf"(^|[^آ-ی]){word}([^آ-ی]|$)", prediction):
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