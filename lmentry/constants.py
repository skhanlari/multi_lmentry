from pathlib import Path

lmentry_random_seed = 1407

ROOT_DIR = Path(__file__).parent.parent

simple_answer_index_task_names = [
    "more_letters", "less_letters", "first_alphabetically",
    "rhyming_word", "homophones", "bigger_number", "smaller_number"
]

HF_PATH = "BSC-LT/multi_lmentry"

#### GLOBAL VARIABLES

LANG = None # Choose between ca, de, en, es, eu, gl, ko, it OR pt_br.
TASKS_DATA_DIR = None # This variable is used to create data starting from resource
PREDICTIONS_ROOT_DIR = None
RESULTS_DIR = None
VISUALIZATIONS_DIR = None
SCORER_EVALUATION_DIR = None
RESOURCES_DIR = None
LMENTRY_WORDS_PATH = None
RHYME_GROUPS_PATH = None
HOMOPHONES_PATH = None
PHONETICALLY_UNAMBIGUOUS_WORDS_PATH = None


def initialize_variables(lang):

    global LANG
    global TASKS_DATA_DIR
    global PREDICTIONS_ROOT_DIR
    global RESULTS_DIR
    global VISUALIZATIONS_DIR
    global SCORER_EVALUATION_DIR
    global RESOURCES_DIR
    global LMENTRY_WORDS_PATH
    global RHYME_GROUPS_PATH
    global HOMOPHONES_PATH
    global PHONETICALLY_UNAMBIGUOUS_WORDS_PATH

    print("LANG:", lang)

    LANG = lang # Choose between ca, de, en, es, eu, gl, ko, it OR pt_br.

    TASKS_DATA_DIR = ROOT_DIR.joinpath(f"data/{LANG}")
    PREDICTIONS_ROOT_DIR = ROOT_DIR.joinpath(f"predictions/{LANG}")
    RESULTS_DIR = ROOT_DIR.joinpath(f"results/{LANG}")
    VISUALIZATIONS_DIR = RESULTS_DIR.joinpath("visualizations")
    SCORER_EVALUATION_DIR = RESULTS_DIR.joinpath("scorer_evaluation")
    RESOURCES_DIR = ROOT_DIR.joinpath(f"resources/{LANG}")
    LMENTRY_WORDS_PATH = RESOURCES_DIR.joinpath("lmentry_words.json")
    RHYME_GROUPS_PATH = RESOURCES_DIR.joinpath("rhyme_groups.json")
    HOMOPHONES_PATH = RESOURCES_DIR.joinpath("homophones.csv")
    PHONETICALLY_UNAMBIGUOUS_WORDS_PATH = RESOURCES_DIR.joinpath("phonetically_unambiguous_words.json")

