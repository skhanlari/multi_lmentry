import re
from typing import List

from lmentry.scorers.it.scorer import LMentryScorer, the_word_regex


class StartsWithWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        a_ = r"una"

        # we don't want an empty prefix here
        self.prefixes = [
            r"(La )?frase ( è)?(:)?",
            r"{a_} frase che inizia con ( la parola)? {word}( è)?",
            r"{a_} frase che comincia con ( la parola)? {word}( è)?",
            r"{a_} frase la cui prima parola è {word}( è)?",
        ]
        # we don't want an empty suffix here
        self.suffixes = [
            r"inizia con {word}",
            r"comincia con {word}",
            r"è una frase che inizia con {word}",
            r"è una frase che comincia con {word}",
            r"è una frase la cui prima parola è {word}",
        ]

        def _replace_arguments(xfixes: List[str]):
            for i in range(len(xfixes)):
                xfixes[i] = xfixes[i].format(
                    a_=a_, word="{word}"
                )  # `word` will be replaced later

        _replace_arguments(self.prefixes)
        _replace_arguments(self.suffixes)

    def get_base_patterns(self, word):

        # we want a word to contain at least two letters, with the exception of "a" and "i"
        valid_word = r"(a|i|o|e|\w\w+)"
        # we use `[^a-zA-Z0-9_]` instead of `\W` as we always lowercase the pattern in
        # `certainty_scorer`.
        # we can't tell if the sentence words are actual English words, but as a sanity we demand at
        # least one additional "word" after the starting word
        allowed_sentence = rf"[^a-zA-Z0-9_]*{word}[^a-zA-Z0-9_]+{valid_word}.*"

        base_patterns = [allowed_sentence]
        return base_patterns + self.get_shared_patterns(target=allowed_sentence)

    def negative_scorer(self, prediction, word):
        score, certainty = None, None

        if not re.search(rf"\b{word}\b", prediction):
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

        # note that the second case of `_simple_scorer` is not relevant
        score, certainty = self._simple_scorer(prediction, base_patterns[0])
        if score:
            return score, certainty

        self.prefix_kwargs = {"word": word}
        self.suffix_kwargs = {"word": word}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
