import re
import random
import pandas as pd

random.seed(42)

# Leet-speak substitution map
LEET_MAP = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}

SYMBOLS = ['!', '#', '*', '^', '~']


def leet_substitution(text: str, rate: float = 0.4) -> str:
    """Replace vowels/common chars with leet equivalents at given rate."""
    return ''.join(
        LEET_MAP.get(c, c) if c in LEET_MAP and random.random() < rate else c
        for c in text
    )


def symbol_insertion(text: str, rate: float = 0.2) -> str:
    """Insert random symbols after words longer than 3 chars."""
    def maybe_insert(match):
        if random.random() < rate:
            return match.group() + random.choice(SYMBOLS)
        return match.group()
    return re.sub(r'\b\w{4,}\b', maybe_insert, text)


def whitespace_injection(text: str, rate: float = 0.15) -> str:
    """Inject spaces inside words to break pattern matching."""
    def split_word(match):
        word = match.group()
        if len(word) > 4 and random.random() < rate:
            mid = len(word) // 2
            return word[:mid] + ' ' + word[mid:]
        return word
    return re.sub(r'\b\w{5,}\b', split_word, text)


def apply_all(text: str) -> str:
    """Apply all adversarial modifications."""
    text = leet_substitution(text)
    text = symbol_insertion(text)
    text = whitespace_injection(text)
    return text


def generate_adversarial_set(df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
    """
    Create an adversarial version of a DataFrame by modifying spam emails only.
    Ham emails are left unchanged.

    Returns a copy of df with modified text in text_col.
    """
    adv_df = df.copy()
    spam_mask = adv_df['label'] == 1
    adv_df.loc[spam_mask, text_col] = adv_df.loc[spam_mask, text_col].apply(apply_all)
    return adv_df
