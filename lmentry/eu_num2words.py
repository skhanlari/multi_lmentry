LOW_NUMWORDS = {
    0: 'zero', 1: 'bat', 2: 'bi', 3: 'hiru', 4: 'lau', 5: 'bost', 6: 'sei', 7: 'zazpi', 8: 'zortzi', 9: 'bederatzi',
    10: 'hamar', 11: 'hamaika', 12: 'hamabi', 13: 'hamahiru', 14: 'hamalau', 15: 'hamabost', 16: 'hamasei', 17: 'hamazazpi', 18: 'hamazortzi', 19: 'hemeretzi'
}

MID_NUMWORDS = {
    20: 'hogei', 40: 'berrogei', 60: 'hirurogei', 80: 'laurogei',
    100: 'ehun', 200: 'berrehun', 300: 'hirurehun', 400: 'laurehun', 500: 'bostehun', 600: 'seiehun', 700: 'zazpiehun', 800: 'zortziehun', 900: 'bederatziehun',
    1000: 'mila'
}


def num2words(num: int) -> str:
    """Convert a number from 0 to 1000 into its equivalent word representation in Basque.

    :param num: The number to be converted, must be in the range [0, 1000].
    :return: The word representation of the input number in Basque.
    """

    def aux(val: int) -> int:
        try:
            divisor = next(divisor_iter)
            if divisor > val:
                return aux(val)
            div, mod = divmod(val, divisor)
            words.append(MID_NUMWORDS[divisor])
            if mod > 0:
                return aux(mod)
        except StopIteration:
            return val

    words = []
    divisor_iter = iter(sorted(MID_NUMWORDS.keys(), reverse=True))
    dividend = aux(num)
    if dividend:
        words.append(LOW_NUMWORDS[dividend])

    num_words = ' eta '.join(words)
    num_words = num_words.replace('gei eta ', 'geita ')
    return num_words
