import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({'subject', 'email', 'mail', 'com', 'http', 'www', 'nbsp'})

# Decode leet-speak — inverse of adversarial.py LEET_MAP (must stay in sync)
LEET_MAP = str.maketrans({
    '@': 'a', '3': 'e', '1': 'i', '0': 'o',
    '$': 's', '7': 't', '!': 'i', '4': 'a'
})


def _base_clean(text: str) -> str:
    """Shared step: leet-decode → lower → strip HTML/URLs/emails → remove non-alpha."""
    if not isinstance(text, str):
        return ""
    text = text.translate(LEET_MAP)
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess(text: str) -> str:
    """Classical pipeline: _base_clean + stopwords + PorterStemmer."""
    tokens = _base_clean(text).split()
    tokens = [stemmer.stem(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def preprocess_lstm(text: str) -> str:
    """LSTM pipeline: _base_clean + stopwords, no stemming (preserves word order)."""
    tokens = _base_clean(text).split()
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)
