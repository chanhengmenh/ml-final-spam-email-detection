import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_structural(text: str) -> dict:
    """Extract non-textual structural features from a raw email string."""
    if not isinstance(text, str):
        text = ""

    length = max(len(text), 1)

    return {
        'url_count':          len(re.findall(r'http\S+|www\S+', text)),
        'email_addr_count':   len(re.findall(r'\S+@\S+', text)),
        'special_char_ratio': len(re.findall(r'[!$%@#*]', text)) / length,
        'uppercase_ratio':    sum(1 for c in text if c.isupper()) / length,
        'html_tag_count':     len(re.findall(r'<[^>]+>', text)),
        'email_length':       len(text),
        'digit_ratio':        sum(1 for c in text if c.isdigit()) / length,
        'exclamation_count':  text.count('!'),
        'dollar_count':       text.count('$'),
    }


def build_structural_matrix(raw_texts: pd.Series) -> np.ndarray:
    """Convert a series of raw email texts into a structural feature matrix."""
    records = [extract_structural(t) for t in raw_texts]
    return pd.DataFrame(records).values.astype(np.float32)


def build_hybrid_features(
    processed_texts: pd.Series,
    raw_texts: pd.Series,
    tfidf: TfidfVectorizer = None,
    fit: bool = True,
    max_features: int = 10000,
):
    """
    Combine TF-IDF text features with structural features.

    Args:
        processed_texts: Preprocessed (cleaned) email text.
        raw_texts:        Original raw email text (for structural features).
        tfidf:            Existing TfidfVectorizer (pass when transforming test data).
        fit:              Whether to fit the vectorizer (True for train, False for test).
        max_features:     TF-IDF vocabulary size cap.

    Returns:
        X_hybrid (sparse matrix), tfidf (fitted vectorizer)
    """
    if tfidf is None:
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True,
        )

    if fit:
        X_tfidf = tfidf.fit_transform(processed_texts)
    else:
        X_tfidf = tfidf.transform(processed_texts)

    X_struct = build_structural_matrix(raw_texts)
    X_hybrid = hstack([X_tfidf, csr_matrix(X_struct)])

    return X_hybrid, tfidf
