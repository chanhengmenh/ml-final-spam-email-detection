# Project Progress Log

## 2026-04-25 — Session 1

### Completed

#### Notebook (`notebooks/spam_detection.ipynb`) — Full Redesign

- [X] Replaced old 6-model, multi-CSV notebook with a clean 4-model pipeline
- [X] Single data source: `data/raw/spam_Emails_data.csv`
- [X] Dual preprocessing cells: `clean_text` (classical) and `lstm_text` (BiLSTM)
- [X] Single 80/20 stratified split shared by all 4 models
- [X] Classical features: TF-IDF (unigram+bigram, 50k) + 9 structural features → hybrid matrix
- [X] LSTM features: Keras Tokenizer → `pad_sequences(MAX_LEN=200)`
- [X] Train 3 classical models: Logistic Regression, Linear SVM, Hist Gradient Boosting
- [X] Build & train Bidirectional LSTM (EarlyStopping on val_loss, restore_best_weights=True)
- [X] Unified evaluation table (F1, AUC, Accuracy, Precision, Recall, Train Time, Infer Time)
- [X] Comparison plots: metrics heatmap, F1 bar chart, ROC curves, confusion matrices, speed vs F1 scatter
- [X] Save best model + paired artifact (`.keras`+tokenizer or `.pkl`+tfidf) to `models/`
- [X] Write `models/model_card.txt` with metrics and hyperparameters
- [X] Live prediction demo cell (dispatches to LSTM or classical pipeline)

#### Source Files (`src/`)

- [X] `src/preprocess.py` — synced with notebook preprocessing
  - Replaced `urltoken`/`emailtoken` replacement with stripping (matches notebook)
  - Changed token min-length from `> 1` to `> 2`
  - Added extra stopwords (`subject`, `email`, `mail`, etc.)
  - Added `preprocess_lstm()` function (no stemming variant)
  - Tokenization now uses `.split()` consistent with training
- [X] `src/models.py` — updated to match plan's 3 classical models
  - Removed: ComplementNB, SGD, RandomForest, GradientBoosting
  - Kept: Logistic Regression, Linear SVM, Hist Gradient Boosting
  - Added `roc_auc` to evaluation metrics
- [X] `src/features.py` — verified, no changes needed (9 features already match notebook)
- [X] `src/adversarial.py` — verified, LEET_MAP is inverse of `src/preprocess.py` ✓

#### App (`app/app.py`) — Full Update

- [X] Auto-detects model type (`best_lstm.keras` vs `best_*.pkl`)
- [X] Loads correct paired artifact (LSTM tokenizer or TF-IDF vectorizer)
- [X] LSTM inference: `preprocess_lstm` → `texts_to_sequences` → `pad_sequences` → predict
- [X] Classical inference: `preprocess` → `build_hybrid_features(fit=False)` → predict
- [X] Confidence score displayed alongside SPAM/HAM verdict
- [X] Two tabs: **Classify** and **Model Info**
- [X] Model Info tab reads and displays `models/model_card.txt`
- [X] Feature Breakdown expander (9 structural features, classical mode only)

#### Dependencies (`requirements.txt`)

- [X] Added `scipy` and `tensorflow`

#### Report

- [X] `REPORT.md` — full presentation report written (replaces slides)

---

### Not Yet Done

#### Notebook

- [ ] Run end-to-end and verify all cells pass
- [ ] Check that saved artifacts are produced correctly
- [X] Adversarial robustness test cell — added (Section 13, uses `src/adversarial.py`)
- [ ] Cross-domain generalization test cell

#### App

- [ ] Test Streamlit app with trained models (after notebook run)
- [ ] Verify LSTM inference path works end-to-end in the app

#### General

- [ ] `pip install -r requirements.txt` to install `scipy` and `tensorflow`
- [ ] Final demo prep
- [ ] Report writing

---

### Notes

- Notebook uses inline preprocessing functions (not `src/preprocess.py` imports) to keep it self-contained and avoid path issues from `notebooks/` directory
- `src/preprocess.py` functions now produce identical output to the notebook's inline functions — critical for app inference to match training
- Only the best model by F1-Score is saved; `app/app.py` auto-detects which type it is
- `MAX_LEN = 200` is hardcoded in `app/app.py` — must match notebook's `MAX_LEN` config cell

---

### Reminder — Update REPORT.md After Training

`REPORT.md` currently uses assumed/placeholder numbers. After running the notebook end-to-end, update these 4 spots with real values:

1. **Section 2 — Dataset** — actual row count after dedup, spam/ham counts and percentages
2. **Section 4.1 — Model Leaderboard** — copy the full results table from notebook Section 11 output
3. **Section 4.2 — Classification Report + Confusion Matrix** — copy from notebook best model `classification_report()` output
4. **Section 5.2 — Adversarial Robustness table** — copy from notebook Section 13 output

If the best model turns out to be **BiLSTM** (not Linear SVM), also update:
- Abstract — winner name and metrics
- Section 4.1 — bold the correct row
- Section 6 — inference pipeline diagram (swap classical path for LSTM path)
- Section 7 — conclusion table
- Appendix B — model card contents

Everything else (methodology, architecture, reasoning) stays as-is.
