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

A supervised ML pipeline that classifies emails as **spam (1)** or **ham (0)** using a hybrid feature model combining TF-IDF text features with email structural features. The system is evaluated across six classifiers, tested for adversarial robustness, and validated on a cross-domain dataset.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main notebook
jupyter notebook notebooks/spam_detection.ipynb

# Run the web app (requires trained models in models/)
streamlit run app/app.py
```

---

## Project Structure

```
final-project/
├── notebooks/
│   └── spam_detection.ipynb       ← MAIN NOTEBOOK (all-in-one pipeline)
├── src/
│   ├── preprocess.py              ← PorterStemmer, leet normalisation, stopword removal
│   ├── features.py                ← TF-IDF + 5 structural features (hybrid matrix)
│   ├── models.py                  ← trains 6 classifiers, saves .pkl files
│   └── adversarial.py             ← leet substitution, symbol insertion, whitespace split
├── app/
│   └── app.py                     ← Streamlit web interface (load model → predict)
├── data/
│   └── raw/                       ← place all CSV datasets here
├── models/                        ← saved .pkl models (auto-generated after training)
├── requirements.txt
├── CLAUDE.md                      ← architecture notes for Claude Code sessions
└── README.md                      ← this file
```

---

## Datasets

Place all CSV files in `data/raw/` (notebook uses `datasets/` directory — update path if needed).

| File                                 | Source       | Rows  | Notes                                                   |
| ------------------------------------ | ------------ | ----- | ------------------------------------------------------- |
| `combined_data.csv`                | Kaggle       | ~83k  | General email spam                                      |
| `completeSpamAssassin.csv`         | SpamAssassin | ~6k   | Column:`body`, `label`                              |
| `enronSpamSubset.csv`              | Enron        | ~10k  | Column:`body`, `label`                              |
| `lingSpam.csv`                     | Ling-Spam    | ~2.6k | Column:`body`, `label`                              |
| `email_1.csv`                      | Kaggle       | ~5.6k | Column:`category`, `message`                        |
| `email_classification_dataset.csv` | Kaggle       | ~10k  | Column:`email`, `label`                             |
| `emails.csv`                       | Kaggle       | ~5.7k | Column:`text`, `spam`                               |
| `email_spam_1.csv`                 | Kaggle       | 26    | **All spam, no ham** — very small                |
| `email_spam.csv`                   | Kaggle       | ~5.6k | Appears identical to `email_1.csv` — deduplicated    |
| `email_spam_dataset_2.csv`         | Kaggle       | ~320  | Column:`email_text`, `label`                        |
| `spam_Emails_data.csv`             | Kaggle       | ~194k | Auto-detected columns                                   |
| **`sms_spam.csv`**           | UCI SMS      | ~5.6k | **Cross-domain test only** — never train on this |

> After deduplication: ~224,795 rows | spam: 45.1% | ham: 54.9%

---

## Pipeline Architecture

```
Raw CSV files
     ↓
Section 1 — Data Loading & Merging
  load_dataset() normalises all CSVs → unified (text, label) DataFrame
  label: spam=1, ham=0
  deduplication on 'text' column
     ↓
Section 2 — EDA
  class distribution, text length histograms, word clouds
     ↓
Section 3 — Preprocessing  [preprocess()]
  emoji removal → lowercase → strip HTML/URLs/emails
  → remove non-alpha → stopword removal → PorterStemmer
     ↓
Section 4 — Hybrid Feature Extraction
  TF-IDF (1,2)-gram, 50k features, sublinear_tf=True   ← fit on TRAIN only
  + structural features: url_count, special_chars,
    html_tag_count, uppercase_ratio, email_length
  → hstack([tfidf_matrix, struct_matrix])
     ↓
Section 5 — Model Training (6 classifiers)
  ComplementNB | Logistic Regression | Linear SVM
  SGD | Random Forest | Hist Gradient Boosting
  Metrics: Accuracy, Precision, Recall, F1, ROC-AUC,
           Train Time, Inference Time, Model Size
     ↓
Section 6 — Evaluation
  6.1 Metrics Heatmap       6.2 Bar Chart
  6.3 ROC Curves            6.4 Confusion Matrices
  6.5 Classification Report 6.6 Threshold Tuning
  6.7 Adversarial Robustness Test
  6.8 Cross-Domain Generalization Test
     ↓
Section 7 — 5-Fold Cross-Validation
  (TF-IDF only pipelines — structural features not pipeable)
     ↓
Section 8 — Leaderboard + Deployment Trade-off Plot
     ↓
Section 9 — Live Prediction Demo
```

---

## Progress Log

### Session 2 — 2026-03-24

**Reviewed by:** Claude Code (Sonnet 4.6)
**Reference:** `Group-5-Spam-Email-Detection-System.pdf`

#### Issues Found & Fixed in `spam_detection.ipynb`

| #  | Issue                                                                                                | Status                                                   |
| -- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 1  | **SGD model missing** — proposal lists 6 models, notebook had 5                               | ✅ Fixed                                                 |
| 2  | **No hybrid features** — notebook used pure TF-IDF only                                       | ✅ Fixed                                                 |
| 3  | **No adversarial robustness testing** — entire Section III.6 of proposal missing              | ✅ Fixed                                                 |
| 4  | **No cross-domain generalization test** — proposal requires separate held-out dataset         | ✅ Fixed                                                 |
| 5  | **No class imbalance handling** — no `class_weight='balanced'` on any model                 | ✅ Fixed                                                 |
| 6  | **`GradientBoostingClassifier` too slow** — 200 estimators on 224k rows, never finished     | ✅ Fixed →`HistGradientBoostingClassifier`            |
| 7  | **`MultinomialNB` incompatible with hybrid features** — requires non-negative integers only | ✅ Fixed →`ComplementNB`                              |
| 8  | **Lemmatization used, not stemming** — proposal specifies PorterStemmer                       | ✅ Fixed                                                 |
| 9  | **No emoji normalization** — listed as preprocessing step in proposal                         | ✅ Fixed                                                 |
| 10 | **CV pipelines inconsistent** — different hyperparameters from main training, missing SGD     | ✅ Fixed                                                 |
| 11 | **No inference time or model size tracking** — proposal requires lightweight model selection  | ✅ Fixed                                                 |
| 12 | **No threshold tuning** — proposal mentions this for class imbalance                          | ✅ Fixed                                                 |
| 13 | **`predict_email()` used TF-IDF only** — inference didn't use structural features           | ✅ Fixed                                                 |
| 14 | **Leaderboard missing deployment metrics**                                                     | ✅ Fixed → added Infer Time, Model Size, trade-off plot |

#### Cells Changed

| Cell ID      | Section             | What Changed                                                                                                                                                        |
| ------------ | ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `2ccc58f8` | Imports             | Added `SGDClassifier`, `ComplementNB`, `HistGradientBoostingClassifier`, `PorterStemmer`, `scipy.sparse`, `pickle`, `sys`, `precision_recall_curve` |
| `555d3c15` | Preprocessing       | Added `remove_emojis()`, switched lemmatizer → `PorterStemmer`                                                                                                 |
| `2cb0514b` | Section 4 markdown  | Updated to describe hybrid feature model                                                                                                                            |
| `83439d14` | Feature extraction  | Added `extract_structural_features()`, hybrid `hstack`, dual train_test_split for raw+clean text                                                                |
| `dcb5bc63` | Section 5 markdown  | Updated model table to include SGD and HistGB                                                                                                                       |
| `83690670` | Model training      | Added SGD, HistGB, ComplementNB,`class_weight='balanced'`, inference time, model size                                                                             |
| `7d7a44ac` | Evaluation 6.4–6.6 | Added threshold tuning section                                                                                                                                      |
| *(new)*    | Section 6.7         | **Adversarial Robustness Testing** — leet, symbol, whitespace attacks                                                                                        |
| *(new)*    | Section 6.8         | **Cross-Domain Generalization Test** — sms_spam.csv held-out evaluation                                                                                      |
| `f3f4a6b7` | Cross-validation    | Added SGD pipeline, replaced GradientBoosting → HistGB, fixed hyperparameter mismatch                                                                              |
| `5ebed415` | Leaderboard         | Added deployment columns, deployment trade-off scatter plot (Figure 12)                                                                                             |
| `1c238496` | Live prediction     | Updated `predict_email()` to use full hybrid pipeline                                                                                                             |
| `9a389ca8` | Conclusion          | Updated to reflect all implemented features                                                                                                                         |

---

### Session 1 — (Initial State)

- Basic notebook created with 5 models (no SGD), pure TF-IDF features
- Data loading, EDA, preprocessing, and evaluation cells working
- `GradientBoostingClassifier` included but too slow to complete on 80k rows
- Cross-validation section present but with inconsistent hyperparameters

---

## What Still Needs to Be Done

### High Priority

- [ ] **Run notebook end-to-end** and verify all cells execute without error after corrections
- [ ] **Add `sms_spam.csv`** to the dataset directory for the cross-domain test to run
- [ ] **`src/` module alignment** — `src/preprocess.py`, `src/features.py`, `src/models.py`, `src/adversarial.py` are separate files that may not reflect the notebook's corrected logic (e.g., `preprocess.py` uses PorterStemmer, `adversarial.py` has a matching `LEET_MAP`)
- [ ] **`app/app.py`** — Streamlit app loads models from `models/` but no models have been saved yet; notebook needs a cell to save trained models and the fitted `tfidf` vectorizer as `.pkl` files

### Medium Priority

- [ ] **Save models cell** — add a cell at the end of Section 5 to save all trained models:
  ```python
  import joblib, os
  os.makedirs('models', exist_ok=True)
  joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
  for name, clf in trained.items():
      joblib.dump(clf, f"models/{name.replace(' ','_')}.pkl")
  ```
- [ ] **`email_spam_1.csv`** — only 26 rows, all spam (no ham). Consider removing to avoid introducing a tiny all-spam bias
- [ ] **`email_1.csv` vs `email_spam.csv`** — both have 5,572 rows with identical spam/ham counts; likely duplicates. Deduplication handles this, but verify and remove one

### Low Priority

- [ ] Add `scipy` to `requirements.txt` (used for `hstack`/`csr_matrix` but not listed)
- [ ] Final report writing (Week 9 per timeline)
- [ ] Streamlit app demo preparation (Week 10)

---

## Key Technical Decisions

| Decision                                 | Reason                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `ComplementNB` over `MultinomialNB`  | Hybrid matrix has float values; CNB also handles imbalance better                    |
| `HistGradientBoostingClassifier`       | 10–50× faster than `GradientBoostingClassifier` on 80k+ rows                     |
| `SGDClassifier(loss='modified_huber')` | Gives `predict_proba` unlike `loss='hinge'`; needed for ROC-AUC                  |
| CV uses TF-IDF only                      | Structural features need raw text; can't be chained in a single sklearn `Pipeline` |
| TF-IDF fit on train only                 | Fitting on test data = leakage;`tfidf.transform()` used for all non-train sets     |
| `class_weight='balanced'`              | Dataset is 45%/55% imbalanced; prevents majority-class bias                          |
| Adversarial applied to spam only         | Ham emails are not adversarially modified in real attacks                            |

---

## Proposal Requirements Checklist

| Requirement                                 | Status                                                    |
| ------------------------------------------- | --------------------------------------------------------- |
| Multiple datasets merged                    | ✅ 11 CSV files, ~224k rows after dedup                   |
| Label standardisation (spam=1, ham=0)       | ✅                                                        |
| 80/20 train/test split                      | ✅ stratified                                             |
| Tokenisation, lowercasing, stopword removal | ✅                                                        |
| Emoji normalisation                         | ✅                                                        |
| Stemming (PorterStemmer)                    | ✅                                                        |
| TF-IDF unigram + bigram                     | ✅ (1,2)-gram, 50k features                               |
| Hybrid features (TF-IDF + structural)       | ✅ 5 structural features                                  |
| Logistic Regression                         | ✅                                                        |
| Naïve Bayes (ComplementNB)                 | ✅                                                        |
| Linear SVM                                  | ✅                                                        |
| SGD                                         | ✅                                                        |
| Random Forest                               | ✅                                                        |
| Gradient Boosting                           | ✅ (Hist variant)                                         |
| Accuracy, Precision, Recall, F1             | ✅                                                        |
| Confusion Matrix                            | ✅                                                        |
| ROC-AUC                                     | ✅                                                        |
| Training time                               | ✅                                                        |
| Inference time + model size                 | ✅                                                        |
| Class imbalance handling                    | ✅`class_weight='balanced'` + threshold tuning          |
| 5-fold cross-validation                     | ✅                                                        |
| Adversarial robustness test                 | ✅ leet / symbol / whitespace                             |
| Cross-domain generalization test            | ✅ (requires `sms_spam.csv`)                            |
| Web interface                               | ⬜`app/app.py` exists but needs trained `.pkl` models |
| Model saving / deployment                   | ⬜ save cell not yet in notebook                          |
