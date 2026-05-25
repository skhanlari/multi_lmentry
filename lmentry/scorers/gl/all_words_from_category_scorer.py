import re

from lmentry.scorers.gl.scorer import LMentryScorer


class AllWordsFromCategoryScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, category):

        all_words = r"(todas estas palabras|todas esas palabras|todas as palabras|todas elas|todas)"
        listed = r"(listadas|na lista|da lista)"
        belong = r"(representan|son tipos de|fan referencia a|refírense a|son exemplos de|contén exclusivamente)"

        base_patterns = [
            rf"{answer}, a lista inclúe exclusivamente {category}",
            rf"{answer}, a lista só inclúe {category}",
            rf"{answer}, as palabras {listed} {belong} {category}"
            rf"{answer}, {all_words}( {listed})? {belong} {category}",
            rf"^{answer}\b",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    @staticmethod
    def category_regex(category):
        if category == "items of clothing":
            category = rf"({category}|clothes|clothing)"
        elif category == "furniture":
            category = rf"({category}|pieces of furniture)"
        elif category == "fruit":
            category = rf"({category}|fruits)"

        return category

    @staticmethod
    def negative_scorer(prediction, answer):
        score = None
        certainty = None

        opposite_answer = "non" if answer == "si" else "si"
        if re.match(rf"{opposite_answer}\.?$", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = "si" if metadata["num_distractors"] == 0 else "non"

        score, certainty = self.negative_scorer(prediction, answer)
        if score is not None:
            return score, certainty

        score, certainty = self._simple_scorer(prediction, answer)
        if score:
            return score, certainty

        category = metadata["category"]
        category = AllWordsFromCategoryScorer.category_regex(category)

        base_patterns = self.get_base_patterns(answer, category)
        score, certainty = self.certainty_scorer(prediction, base_patterns)

        return score, certainty
