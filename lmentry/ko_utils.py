from jamo import h2j, j2hcj

# groups of letters in the alphabet
BASIC_CONSONANTS = {"ㄱ", "ㄴ", "ㄷ", "ㄹ", "ㅁ", "ㅂ", "ㅅ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"}
TENSE_CONSONANTS = {"ㄲ", "ㄸ", "ㅃ", "ㅉ", "ㅆ"}
BASIC_VOWELS = {"ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ"}
COMPLEX_VOWELS = {"ㅢ", "ㅚ", "ㅐ", "ㅟ", "ㅔ", "ㅒ", "ㅖ", "ㅘ", "ㅝ", "ㅙ", "ㅞ"}

# full alphabet in alphabetical order
FULL_ALPHABET = ["ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ", "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"]

# Auxiliary function to decompose a word in Hangul into letters
# e.g. "한글" => "ㅎㅏㄴㄱㅡㄹ"
hangul_word_to_letters = lambda word: j2hcj(h2j(word))

# Regular expressions to match Hangul
HANGUL_CHAR_REGEX = r"\uac00-\ud7af\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\ud7b0-\ud7ff" # matches a single character, e.g. "나"
HANGUL_WORD_REGEX = rf"[{HANGUL_CHAR_REGEX}]+" # matches a whole word of Hangul characters and nothing else, e.g. "끝나"
HANGUL_ALPHANUMERIC_REGEX = rf"[{HANGUL_CHAR_REGEX}\d]+" # matches Hangul characters and numbers, e.g. "27살입니다"
