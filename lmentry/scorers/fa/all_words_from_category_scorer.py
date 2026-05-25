import re

from lmentry.scorers.fa.scorer import LMentryScorer


class AllWordsFromCategoryScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, category):

        all_words = r"(همه این کلمات|تمام این کلمات|این کلمات همگی)"
        listed = r"(ذکرشده|لیست‌ شده)"
        belong = r"(هستند|متعلق هستند|در دسته)"

        base_patterns = [
            rf"{answer}، قرار دارد {category} این لیست در دسته ",
            rf"{category} {belong} ? ({listed}) {all_words}،{answer}",
            rf"^{answer}\b",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    @staticmethod
    def category_regex(category): # TODO: modify this!
        if category == "items of clothing":
            category = rf"({category}|لباس|پوشاک)"
        elif category == "furniture":
            category = rf"({category}|مبلمان|وسایل خانه)"
        elif category == "fruit":
            category = rf"({category}|میوه|میوه‌ها)"

        return category
    
    @staticmethod
    def negative_scorer(prediction, answer):
        score = None
        certainty = None

        if "بله" in answer:
            opposite_answer = r"(نه|خیر)"
        else:
            opposite_answer = r"(بله|آره|اره)"

        if re.match(rf"{opposite_answer}\.?$", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = r"(بله|آره|اره)" if metadata["num_distractors"] == 0 else r"(نه|خیر)"

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