# Project Plan — Spam Email Detection System

**ITM-390 Machine Learning — Group 5 | AUPP**

| Role                             | Name              | Student ID |
| -------------------------------- | ----------------- | ---------- |
| Team Leader / AI Engineer        | Menh Chanheng     | 2023445    |
| Data Collector / Model Evaluator | Heng Shito        | 2023210    |
| Model Trainer / Progress Tracker | Sorith Pichetroth | 2024089    |

---

## Goal

Build a spam email classifier that compares 4 models, selects the best one, and deploys it as a Streamlit web app.

---

## Architecture

```
data/raw/spam_Emails_data.csv
        │
        ▼
notebooks/inspector.ipynb          ← EDA only (DONE)
        │
        ▼
notebooks/spam_detection.ipynb     ← Full pipeline (train → evaluate → save)
        │
        ├── Classical path ──────── TF-IDF (1,2) + 9 structural features
        │       ├── Logistic Regression
        │       ├── Linear SVM
        │       └── HistGradientBoosting
        │
        └── Deep learning path ──── Keras Tokenizer → padded sequences
                └── Bidirectional LSTM
        │
        ▼
models/
  ├── best_model.pkl   OR  best_lstm.keras   ← winner only
  ├── tfidf_vectorizer.pkl   OR  lstm_tokenizer.pkl
  └── model_card.txt
        │
        ▼
app/app.py                         ← Streamlit web interface
```

---

## Models

| # | Model                        | Feature Input            | Notes                                       |
| - | ---------------------------- | ------------------------ | ------------------------------------------- |
| 1 | Logistic Regression          | TF-IDF + structural      | Linear baseline,`class_weight='balanced'` |
| 2 | Linear SVM                   | TF-IDF + structural      | High-dim text,`class_weight='balanced'`   |
| 3 | HistGradientBoosting         | TF-IDF + structural      | Fast tree ensemble                          |
| 4 | **Bidirectional LSTM** | Embedded token sequences | Deep learning, learns word order            |

All 4 share the **same 80/20 stratified split** so results are directly comparable.

---

## Preprocessing

Two preprocessing variants are derived from the same raw text:

| Variant        | Used by                         | Steps                                                                                                              |
| -------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `clean_text` | Classical models (TF-IDF input) | leet normalise → lowercase → strip HTML/URLs → remove non-alpha → stopwords → PorterStemmer                   |
| `lstm_text`  | LSTM (tokenizer input)          | leet normalise → lowercase → strip HTML/URLs → remove non-alpha*(no stemming — LSTM learns from word order)* |

---

## Saved Artifacts (Production)

Only the **best model** is saved. Exactly 2 files always:

| If best model is… | File 1                     | File 2                          |
| ------------------ | -------------------------- | ------------------------------- |
| LR / SVM / GB      | `models/best_<name>.pkl` | `models/tfidf_vectorizer.pkl` |
| LSTM               | `models/best_lstm.keras` | `models/lstm_tokenizer.pkl`   |

A `models/model_card.txt` is also written with key metrics and hyperparameters.

---

## Streamlit App (`app/app.py`)

**Pages / features planned:**

1. **Classify** — paste email text → model runs full pipeline → SPAM / HAM verdict with confidence score
2. **Model Info** — show model_card.txt (which model is loaded, F1, AUC, train time)
3. **Feature Breakdown** — expandable panel showing the 9 structural feature values for the input email

**Inference logic inside the app:**

```
if best model is LSTM:
    load lstm_tokenizer.pkl  → texts_to_sequences → pad_sequences(MAX_LEN)
    load best_lstm.keras     → model.predict()

if best model is classical:
    load tfidf_vectorizer.pkl → tfidf.transform(clean_text)
    extract_structural(raw_text)
    hstack([tfidf_features, structural_features])
    load best_<name>.pkl      → model.predict()
```

The app must **never re-fit** the tokenizer or vectorizer — only call `.transform()` / `texts_to_sequences()`.

---

## Task Checklist

### Notebook (`spam_detection.ipynb`)

- [ ] **Redesign notebook** with 4-model structure (LSTM + 3 classical)
  - [ ] Dual preprocessing cell (`clean_text` for classical, `lstm_text` for LSTM)
  - [ ] Single train/test split shared by all models
  - [ ] Classical features: TF-IDF + structural → hybrid matrix
  - [ ] LSTM features: Keras Tokenizer → pad_sequences
  - [ ] Train 3 classical models
  - [ ] Build & train BiLSTM (EarlyStopping, restore_best_weights=True)
  - [ ] Unified evaluation table (F1, AUC, train time, infer time)
  - [ ] Comparison plots: heatmap, bar chart, ROC curves, confusion matrices
  - [ ] Save best model + paired artifact to `models/`
  - [ ] Live prediction demo cell

### App (`app/app.py`)

- [ ] **Update app** to support both LSTM and classical model files
  - [ ] Auto-detect whether best model is `.keras` or `.pkl`
  - [ ] Load correct tokenizer / vectorizer based on model type
  - [ ] Run correct inference pipeline per model type
  - [ ] Display confidence score (probability / sigmoid) alongside verdict
  - [ ] Model Info tab showing `model_card.txt`
  - [ ] Feature breakdown expander (structural features)

### Source (`src/`)

- [ ] Verify `src/preprocess.py` matches notebook preprocessing (PorterStemmer, leet map)
- [ ] Verify `src/features.py` `extract_structural` matches the 9 features used in notebook
- [ ] Add `scipy` and `tensorflow` to `requirements.txt`

### Data

- [ ] Confirm `data/raw/spam_Emails_data.csv` is the only dataset needed
  - Current: 80000 rows | spam ~47% | ham ~53%
- [ ] (Optional) Add `sms_spam.csv` for a cross-domain generalization test

---

## Evaluation Metrics

Every model is evaluated on the same held-out test set (20%):

| Metric             | Why                                                     |
| ------------------ | ------------------------------------------------------- |
| **F1-Score** | Primary ranking metric — balances precision and recall |
| Accuracy           | Baseline check                                          |
| Precision          | Cost of false positives (legitimate email marked spam)  |
| Recall             | Cost of false negatives (spam reaching inbox)           |
| ROC-AUC            | Threshold-independent ranking ability                   |
| Train Time         | Practical training cost                                 |
| Inference Time     | Latency requirement for production                      |

---

## Deployment Notes

```bash
# Install all dependencies
pip install -r requirements.txt

# Step 1 — inspect dataset (optional)
jupyter notebook notebooks/inspector.ipynb

# Step 2 — run full pipeline (produces models/ artifacts)
jupyter notebook notebooks/spam_detection.ipynb

# Step 3 — launch web app
streamlit run app/app.py
```

The app reads whatever is in `models/` — no changes needed to `app.py` after retraining.

---

## Key Decisions

| Decision                                 | Reason                                                                      |
| ---------------------------------------- | --------------------------------------------------------------------------- |
| 2 notebooks only                         | inspector for EDA, spam_detection for full pipeline — no need for more     |
| Save only best model                     | Reduces deployment complexity; only 2 files needed at runtime               |
| Dual preprocessing                       | LSTM should not receive stemmed text — stemming destroys word-order signal |
| Single train/test split                  | Fair comparison — all 4 models evaluated on identical data                 |
| `class_weight='balanced'` on classical | Dataset ~47/53 split; prevents majority-class bias                          |
| BiLSTM over plain LSTM                   | Reads sequence in both directions — better context capture                 |
| EarlyStopping + restore_best_weights     | Prevents overfitting; val_loss as monitor                                   |
| Vocab size 20k, maxlen 200               | Covers ~90% of email lengths; keeps model size manageable                   |

---

## Timeline

| Week | Milestone                                                    |
| ---- | ------------------------------------------------------------ |
| Now  | Redesign `spam_detection.ipynb` (4 models)                 |
| +1   | Update `app/app.py` to support LSTM + classical dispatch   |
| +2   | Run end-to-end, verify all cells pass, check saved artifacts |
| +3   | Final demo prep, report writing                              |
