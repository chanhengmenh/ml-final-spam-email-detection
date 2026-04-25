# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the training notebook (main entry point)
jupyter notebook notebooks/01_train_and_evaluate.ipynb

# Run the web app (requires trained models in models/)
streamlit run app/app.py
```

## Architecture

This is a supervised ML pipeline for spam email classification. Data flows in one direction through four stages:

```
raw CSV  →  preprocess.py  →  features.py  →  models.py  →  models/*.pkl
                                                               ↓
                                                           app/app.py
```

**`src/preprocess.py`** — Takes raw email text, applies leet normalization (defense against adversarial inputs), lowercasing, URL/email tokenization, stopword removal, and PorterStemmer. The `LEET_MAP` in this file is the inverse of the one in `adversarial.py` — they must stay in sync.

**`src/features.py`** — Builds a hybrid sparse feature matrix by horizontally stacking TF-IDF (unigram+bigram, `sublinear_tf=True`) with 9 structural features (URL count, uppercase ratio, HTML tag count, etc.). The fitted `TfidfVectorizer` is saved separately as `models/tfidf_vectorizer.pkl` and must be passed to `build_hybrid_features(..., fit=False)` at inference time.

**`src/models.py`** — Trains 6 classifiers (Logistic Regression, ComplementNB, LinearSVC, SGD, Random Forest, Gradient Boosting) and saves each as `models/<name>.pkl`. `ComplementNB` is used instead of `MultinomialNB` because the hybrid feature matrix contains negative-compatible values. All models use `class_weight='balanced'` where supported.

**`src/adversarial.py`** — Modifies spam emails in the test set to simulate evasion attacks (leet substitution, symbol insertion, whitespace splitting). Only applied to spam rows (`label == 1`). Used in the notebook for a before/after F1 comparison.

**`app/app.py`** — Streamlit app that loads any saved `.pkl` model from `models/` and the `tfidf_vectorizer.pkl`, then runs the same `preprocess → build_hybrid_features(..., fit=False)` pipeline on user input.

## Key Constraints

- **`build_hybrid_features` must always receive both `processed_texts` and `raw_texts`** — structural features are computed from raw text, TF-IDF from preprocessed text. Never pass the same series to both arguments.
- **The TF-IDF vectorizer must be fitted only on training data** (`fit=True`) and reused for test/adversarial/cross-domain sets (`fit=False, tfidf=<fitted>`). Fitting on test data causes leakage.
- **Labels are always `spam=1, ham=0`** as integers throughout the pipeline.
- The notebook's `DATASETS` config cell maps raw CSV column names to `text` and `label` — update this when adding new datasets.

## Data

Place raw CSV files in `data/raw/`. The notebook expects at minimum a text column and a label column (configurable in the `DATASETS` list). The cross-domain test uses a separate file (`sms_spam.csv` by default) that is never included in training.
