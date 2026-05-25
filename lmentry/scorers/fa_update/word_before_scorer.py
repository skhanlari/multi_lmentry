import re

from lmentry.scorers.fa.scorer import (
    LMentryScorer,
    the_sentence_regex,
    the_word_regex,
    swap_substrings,
)


class WordBeforeScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        self.prefixes.extend(
            [
                "در جمله {sentence}،? ",
                "در {sentence}،? ",
            ]
        )

        self.suffixes.extend(
            [
                " در {sentence}",
                " در جمله {sentence}",
            ]
        )

    def get_base_patterns(self, answer, query, sentence):

        before = r"(بلافاصله |مستقیماً )?(قبل از|پیش از)"
        comes = r"(می‌آید|قرار می‌گیرد|هست)"

        base_patterns = [
            rf"{answer} {before} {query} {comes}",
            rf"{answer} کلمه‌ای است که {before} {query} {comes}",
            rf"{answer} کلمه قبل از {query} است",
            rf"{answer} کلمه پیش از {query} است",
            rf"{answer} کلمه‌ای است که قبل از {query} می‌آید",
            rf"{answer} کلمه‌ای است که پیش از {query} می‌آید",
            rf"کلمه‌ای که {before} {query} {comes} است {answer}",
            rf"در جمله {sentence}، کلمه‌ای که {before} {query} {comes} {answer} است",
            rf"کلمه قبل از {query} {answer} است",
            rf"کلمه پیش از {query} {answer} است",
        ]

        after = r"(بلافاصله |مستقیماً )?(بعد از|پس از)"
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=query).replace(before, after)
            for s in base_patterns
            if before in s
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def negative_scorer(self, prediction, answer, sentence):
        score, certainty = None, None

        if not re.search(rf"(^|[^آ-ی]){answer}([^آ-ی]|$)", prediction):
            score = 0
            certainty = 1

        if re.sub(r"[^\w ]+", "", prediction) == sentence:
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = metadata["answer"]
        query = metadata["query"]
        sentence = metadata["sentence"]

        answer = answer.lower()
        query = query.lower()
        sentence = sentence.lower()

        score, certainty = self.negative_scorer(prediction, answer, sentence)
        if score is not None:
            return score, certainty

        answer = the_word_regex(answer)
        query = the_word_regex(query)
        sentence = the_sentence_regex(sentence)

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        base_patterns = self.get_base_patterns(answer, query, sentence)
        self.prefix_kwargs = {"sentence": sentence}
        self.suffix_kwargs = {"sentence": sentence}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty