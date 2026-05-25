import re

from lmentry.scorers.it.scorer import LMentryScorer


class AllWordsFromCategoryScorer(LMentryScorer):
    def __init__(self):
        super().__init__()

    def get_base_patterns(self, answer, category):

        all_words = r"(ogni parola|tutte queste parole|tutte quelle parole|tutte le parole|tutte loro|tutte)"
        listed = r"(elencata|elencate|nell'elenco|dell'elenco|sull'elenco|in elenco|in lista|nella lista|della lista|sulla lista)"
        belong = r"(rappresenta|rappresentano|sono tipi di|si riferiscono a|sono esempi di|sono esclusivamente|sono solo|sono appartenenti a|sono parte di)"

        base_patterns = [
            rf"{answer}, l'elenco include esclusivamente {category}",
            rf"{answer}, l'elenco include solo {category}",
            rf"{answer}, la lista include esclusivamente {category}",
            rf"{answer}, la lista include solo {category}",
            rf"{answer}, le parole {listed} {belong} {category}"
            rf"{answer}, {all_words}( {listed})? {belong} {category}",
            rf"^{answer}\b",
        ]

        return base_patterns + self.get_shared_patterns(target=answer)

    @staticmethod
    def category_regex(category):
        if category == "vestiti":
            category = rf"({category}|vestiario|abbigliamento|abiti)"
        elif category == "verdure":
            category = rf"({category}|verdura|ortaggi)"
        elif category == "professioni":
            category = rf"({category}|lavoro|mestieri|impieghi)"
        elif category == "misure":
            category = rf"({category}|grandezze)"
        elif category == "posti di lavoro":
            category = rf"({category}|luoghi di lavoro)"
        elif category == "arredamento o parti di casa":
            category = rf"({category}|mobili|parti di casa|arredamento|stanze)"
        elif category == "utensili":
            category = rf"({category}|attrezzi)"
        elif category == "genere di film":
            category = rf"({category}|categoria di film)"
        elif category == "parenti":
            category = rf"({category}|familiari|famiglia)"
        elif category == "sport":
            category = rf"({category}|attività sportive)"
        elif category == "catastrofe naturale":
            category = rf"({category}|calamità naturali|cataclismi)"
        elif category == "termini socio-politici":
            category = (
                rf"({category}|termini sociopolitici|termini politici|termini sociali)"
            )
        elif category == "ambienti naturali":
            category = (
                rf"({category}|luoghi naturali|paesaggi naturali|ecosistemi|ecosistema)"
            )

        return category

    @staticmethod
    def negative_scorer(prediction, answer):
        score = None
        certainty = None

        opposite_answer = "no" if answer == "sì" else "sì"
        if re.match(rf"{opposite_answer}\.?$", prediction):
            score = 0
            certainty = 1

        return score, certainty

    def score_prediction(self, prediction, example, truncate_prediction: bool = False):
        prediction = self.normalize_prediction(prediction, truncate_prediction)

        metadata = example["metadata"]
        answer = "sì" if metadata["num_distractors"] == 0 else "no"

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
