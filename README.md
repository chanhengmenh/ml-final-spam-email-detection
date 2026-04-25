# Email Spam Detection System

**ITM-390 Machine Learning — Group 5 | American University of Phnom Penh**

| Role                             | Name              | Student ID |
| -------------------------------- | ----------------- | ---------- |
| Team Leader / AI Engineer        | Menh Chanheng     | 2023445    |
| Data Collector / Model Evaluator | Heng Shito        | 2023210    |
| Model Trainer / Progress Tracker | Sorith Pichetroth | 2024089    |

**Advisor:** Prof. PIN Kuntha | **Department:** Digital Technology

---

## Project Overview

A supervised ML pipeline that classifies emails as **spam (1)** or **ham (0)**. The final deployed model is a **BiLSTM** (F1 = 0.9746, ROC-AUC = 0.9957). Classical baselines (Logistic Regression, ComplementNB, LinearSVC, SGD, Random Forest, HistGradientBoosting) are also trained using a hybrid TF-IDF + structural feature matrix.

- **Dataset:** [chanhengmenh/spam_email_detection](https://huggingface.co/datasets/chanhengmenh/spam_email_detection) on HuggingFace (80,000 rows, 52.9% ham / 47.1% spam)
- **Model:** [chanhengmenh/spam_email_detection](https://huggingface.co/chanhengmenh/spam_email_detection) on HuggingFace
- **Kaggle: [https://www.kaggle.com/code/chanhengmenh/notebook3011a4bfac/notebook](https://www.kaggle.com/code/chanhengmenh/notebook3011a4bfac/notebook)**

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download and inspect the dataset (auto-fetches from HuggingFace if not present)
jupyter notebook notebooks/inspector.ipynb

# 3. Run the full training pipeline
jupyter notebook notebooks/spam_email_detection.ipynb

# 4. Launch the web app (auto-downloads models from HuggingFace on first run)
streamlit run app/app.py
```

> **Fresh clone:** run `inspector.ipynb` first — it downloads `data/processed/spam_email.csv`
> from HuggingFace automatically. The Streamlit app downloads model weights on first launch.

---

## Project Structure

```
final-project/
├── notebooks/
│   ├── inspector.ipynb            ← EDA + auto-downloads dataset from HuggingFace
│   ├── spam_email_detection.ipynb ← MAIN NOTEBOOK: full training & evaluation pipeline
│   └── reduce_df.ipynb            ← utility: subsample / merge raw CSVs
├── src/
│   ├── preprocess.py              ← leet normalisation, PorterStemmer, stopword removal
│   ├── features.py                ← TF-IDF + 9 structural features (hybrid matrix)
│   ├── models.py                  ← trains 6 classical classifiers, saves .pkl files
│   └── adversarial.py             ← leet substitution, symbol insertion, whitespace split
├── app/
│   └── app.py                     ← Streamlit app (auto-downloads models from HuggingFace)
├── data/
│   ├── raw/                       ← original CSV files (not tracked in git)
│   └── processed/
│       └── spam_email.csv         ← unified 80k-row dataset (auto-downloaded if missing)
├── models/                        ← trained model files (auto-downloaded if missing)
│   ├── best_lstm.keras            ← BiLSTM (best model, F1=0.9746)
│   ├── lstm_tokenizer.pkl         ← Keras tokenizer for the LSTM
│   └── model_card.txt             ← metrics summary
├── requirements.txt
├── CLAUDE.md                      ← architecture notes for Claude Code sessions
└── README.md
```

---

## Dataset

The unified dataset is hosted on HuggingFace and downloaded automatically by `inspector.ipynb`.

| Property        | Value                                 |
| --------------- | ------------------------------------- |
| Source          | `chanhengmenh/spam_email_detection` |
| Total rows      | 80,000 (after deduplication)          |
| Ham (0)         | 42,291 (52.9%)                        |
| Spam (1)        | 37,709 (47.1%)                        |
| Imbalance ratio | 1.12 : 1 (ham : spam)                 |
| Median words    | Ham = 149 words, Spam = 106 words     |

The raw dataset was assembled from multiple Kaggle / public sources (SpamAssassin, Enron, Ling-Spam, SMS Spam) and processed via `notebooks/reduce_df.ipynb`.

---

## Pipeline Architecture

```
HuggingFace dataset
        ↓
notebooks/inspector.ipynb
  → downloads spam_email.csv if missing
  → EDA: class balance, text length, top tokens, structural features
        ↓
notebooks/spam_email_detection.ipynb
  → Section 1: load data, 80/20 stratified split
  → Section 2: preprocess (leet norm → lowercase → URL/email tokens
                           → stopwords → PorterStemmer)
  → Section 3: hybrid features
               TF-IDF (unigram+bigram, sublinear_tf=True)  ← fit on TRAIN only
               + 9 structural features (url_count, uppercase_ratio,
                 html_tag_count, special_char_ratio, email_length, …)
               → hstack([tfidf_matrix, struct_matrix])
  → Section 4: classical models (6 classifiers, class_weight='balanced')
               ComplementNB | Logistic Regression | LinearSVC
               SGD | Random Forest | HistGradientBoosting
  → Section 5: BiLSTM (best model — saved to models/)
  → Section 6: evaluation
               metrics heatmap, ROC curves, confusion matrices,
               threshold tuning, adversarial robustness, cross-domain test
  → Section 7: 5-fold cross-validation (classical pipelines)
  → Section 8: leaderboard + deployment trade-off plot
        ↓
models/best_lstm.keras + lstm_tokenizer.pkl
        ↓
app/app.py  (Streamlit — auto-downloads from HuggingFace on first run)
```

---

## Best Model

| Metric    | Value  |
| --------- | ------ |
| Model     | BiLSTM |
| F1-Score  | 0.9746 |
| ROC-AUC   | 0.9957 |
| Accuracy  | 0.9758 |
| Precision | 0.9656 |
| Recall    | 0.9838 |

Hyperparameters: Vocab=20k, MaxLen=200, EmbedDim=64, LSTMUnits=64, BatchSize=256, Epochs=5

---

## Key Technical Decisions

| Decision                                 | Reason                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `ComplementNB` over `MultinomialNB`  | Hybrid matrix has float values; CNB also handles imbalance better                    |
| `HistGradientBoostingClassifier`       | 10–50× faster than `GradientBoostingClassifier` on 80k+ rows                     |
| `SGDClassifier(loss='modified_huber')` | Gives `predict_proba` unlike `loss='hinge'`; needed for ROC-AUC                  |
| CV uses TF-IDF only                      | Structural features need raw text; can't be chained in a single sklearn `Pipeline` |
| TF-IDF fit on train only                 | Fitting on test data = leakage;`tfidf.transform()` used for all non-train sets     |
| `class_weight='balanced'`              | Dataset is ~47%/53% imbalanced; prevents majority-class bias                         |
| Adversarial applied to spam only         | Ham emails are not adversarially modified in real attacks                            |
| BiLSTM as final model                    | Outperforms all classical baselines; hosted on HuggingFace for zero-setup deployment |
| HuggingFace for data & model hosting     | Avoids committing large files to git; auto-download on first run                     |

---

## Proposal Requirements Checklist

| Requirement                                 | Status                                           |
| ------------------------------------------- | ------------------------------------------------ |
| Multiple datasets merged                    | ✅                                               |
| Label standardisation (spam=1, ham=0)       | ✅                                               |
| 80/20 train/test split (stratified)         | ✅                                               |
| Tokenisation, lowercasing, stopword removal | ✅                                               |
| Stemming (PorterStemmer)                    | ✅                                               |
| TF-IDF unigram + bigram                     | ✅                                               |
| Hybrid features (TF-IDF + structural)       | ✅ 9 structural features                         |
| Logistic Regression                         | ✅                                               |
| Naïve Bayes (ComplementNB)                 | ✅                                               |
| Linear SVM                                  | ✅                                               |
| SGD                                         | ✅                                               |
| Random Forest                               | ✅                                               |
| Gradient Boosting (Hist variant)            | ✅                                               |
| Deep learning model (LSTM)                  | ✅ BiLSTM, best model                            |
| Accuracy, Precision, Recall, F1             | ✅                                               |
| Confusion Matrix                            | ✅                                               |
| ROC-AUC                                     | ✅                                               |
| Training time + inference time              | ✅                                               |
| Class imbalance handling                    | ✅`class_weight='balanced'` + threshold tuning |
| 5-fold cross-validation                     | ✅                                               |
| Adversarial robustness test                 | ✅ leet / symbol / whitespace                    |
| Cross-domain generalization test            | ✅                                               |
| Web interface                               | ✅ Streamlit app, auto-downloads model           |
| Model saving / deployment                   | ✅ HuggingFace Hub                               |

---

## Session Log

### Session 3 — 2026-04-25

- Moved dataset to HuggingFace (`chanhengmenh/spam_email_detection`)
- Moved trained models to HuggingFace (`chanhengmenh/spam_email_detection`)
- `notebooks/inspector.ipynb` — added auto-download cell (fetches from HuggingFace if `spam_email.csv` missing)
- `app/app.py` — added `_ensure_model_files()` using `snapshot_download`; models auto-fetched on first app launch
- `requirements.txt` — added `huggingface_hub`, `datasets`

### Session 2 — 2026-03-24

- Full pipeline review against proposal (`Group-5-Spam-Email-Detection-System.pdf`)
- Fixed 14 issues: added SGD, hybrid features, adversarial robustness, cross-domain test, `class_weight='balanced'`, switched to `HistGradientBoostingClassifier` and `ComplementNB`, switched lemmatizer → PorterStemmer, added emoji normalisation, inference time & model size tracking, threshold tuning, consistent CV pipelines
- BiLSTM trained and saved as best model (F1=0.9746)

### Session 1 — (Initial State)

- Basic notebook with 5 models, pure TF-IDF, no adversarial/cross-domain testing
