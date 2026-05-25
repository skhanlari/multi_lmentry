import re

from lmentry.scorers.it.scorer import (
    LMentryScorer,
    the_word_regex,
    the_words_regex,
    swap_substrings,
)


class MostAssociatedWordScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

        # we use `[^a-zA-Z0-9_]` instead of `\W` as we always lowercase the pattern in
        # `certainty_scorer`. See the to-do suggestion in normalize_string

        self.prefixes.extend([r"{words}[^a-zA-Z0-9_]*"])

        self.suffixes.extend([r"[^a-zA-Z0-9_]*{words}"])

    def get_base_patterns(self, answer, category, words):

        most_associated = (
            r"più (associata a|associata con|in relazione con|correlata con)"
        )

        base_patterns = [
            rf"La parola {most_associated} {category} è {answer}",
            rf"La parola {most_associated} {category} fra {words} è {answer}",
            rf"La parola {most_associated} {category} tra {words} è {answer}",
            rf"La categoria {category} è la {most_associated} {answer}",
            rf"{answer} è la parola {most_associated} {category}",
            rf"{answer} è la parola {most_associated} {category} fra {words}",
            rf"{answer} è la parola {most_associated} {category} tra {words}",
            rf"{answer} è la {most_associated} {category}",
            rf"{answer} è la {most_associated} {category} fra {words}",
            rf"{answer} è la {most_associated} {category} tra {words}",
            rf"{answer} è {most_associated} {category}",
            rf"{answer} è {most_associated} {category} fra {words}",
            rf"{answer} è {most_associated} {category} tra {words}",
        ]
        # swap answer and category
        more_base_patterns = [
            swap_substrings(s, subs1=answer, subs2=category) for s in base_patterns
        ]
        base_patterns.extend(more_base_patterns)

        return base_patterns + self.get_shared_patterns(target=answer)

    def get_exact_patterns(self, answer, category):
        return [rf"{answer}", rf"{category} - {answer}"]

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
        of = r"(di|delle|tra|fra|fra le|delle seguenti|fra le seguenti|tra le seguenti)"
        options = r"(parole|opzioni)"
        given = r"(indicate|elencate|fornite|menzionate|date)"
        words = rf"{of} ({options} ({given})?|{words})"

        base_patterns = self.get_base_patterns(answer, category, words)

        self.prefix_kwargs = {"words": words}
        self.suffix_kwargs = {"words": words}

        score, certainty = self.certainty_scorer(prediction, base_patterns)
        return score, certainty
