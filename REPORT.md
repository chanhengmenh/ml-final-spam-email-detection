# Spam Email Detection System
### ITM-390 Machine Learning — Group 5 Final Project Report
### American University of Phnom Penh | Department of Digital Technology

---

| | |
|---|---|
| **Team Leader / AI Engineer** | Menh Chanheng (2023445) |
| **Data Collector / Model Evaluator** | Heng Shito (2023210) |
| **Model Trainer / Progress Tracker** | Sorith Pichetroth (2024089) |
| **Advisor** | Prof. PIN Kuntha |
| **Date** | April 2026 |

---

## Abstract

We built a spam email detection system that compares three classical machine learning models against a Bidirectional LSTM, using a hybrid feature model that combines TF-IDF text representations with nine hand-crafted structural features. Trained on **80,000 emails** (47.1% spam), our best model — **BiLSTM** — achieved an F1-Score of **0.9746** and ROC-AUC of **0.9957** on the held-out test set. We additionally tested all models against adversarial evasion attacks (leet substitution, symbol insertion, whitespace injection), finding that the BiLSTM was the most robust with an F1 change of only **+0.0025** under attack. The best model is deployed as a Streamlit web application that classifies user-submitted email text in real time, supporting both text paste and file upload.

---

## 1. Introduction

Spam emails are not merely an annoyance — they are the delivery mechanism for phishing, malware, and financial fraud. Filtering them accurately is a real-world machine learning problem with asymmetric costs: letting a phishing email through is a security breach, while flagging a legitimate email as spam means a missed meeting, a lost business opportunity, or a missed deadline.

This project tackles three challenges simultaneously:

1. **Classification accuracy** — how well can we separate spam from ham?
2. **Adversarial robustness** — do models still work when spammers deliberately obfuscate their text (e.g., writing `fr33 m0n3y` instead of `free money`)?
3. **Deployment practicality** — which model offers the best accuracy-to-latency tradeoff for a production web app?

We compare four models — Logistic Regression, Linear SVM, XGBoost, and a Bidirectional LSTM — trained on the same data with the same split, so every comparison is fair.

---

## 2. Dataset

We used a single consolidated dataset (`spam_Emails_data.csv`, Kaggle) containing raw email records. After deduplication, the working corpus contained **80,000 emails**.

| Property | Value |
|---|---|
| Total emails (after deduplication) | 80,000 |
| Spam (label = 1) | 37,709 (47.1%) |
| Ham (label = 0) | 42,291 (52.9%) |
| Median word count — Spam | 94 words |
| Median word count — Ham | 138 words |

The dataset is **near-balanced** (47/53), meaning raw accuracy is a meaningful metric. Nevertheless, we apply `class_weight='balanced'` to all applicable classical models to ensure no majority-class bias.

### Train / Test Split

A single **80/20 stratified split** (`random_state=42`) was applied once and shared across all four models:

| Set | Size | Spam % |
|---|---|---|
| Training | 63,920 | 47.1% |
| Test (held-out) | 15,980 | 47.1% |

The TF-IDF vectorizer and Keras tokenizer were fitted exclusively on training data. Test data was only ever transformed — never used to fit — preventing any form of data leakage.

---

## 3. Methodology

### 3.1 Dual Preprocessing

We defined two preprocessing pipelines from the same raw text. The distinction is critical for the BiLSTM.

| Pipeline | Used by | Steps |
|---|---|---|
| `preprocess` | LR, Linear SVM, XGBoost | leet-decode → lowercase → strip HTML / URLs / emails → remove non-alpha → stopword removal → **PorterStemmer** |
| `preprocess_lstm` | BiLSTM | leet-decode → lowercase → strip HTML / URLs / emails → remove non-alpha → stopword removal (**no stemming**) |

Stemming reduces vocabulary size for bag-of-words models without meaningful information loss. For the BiLSTM however, the model learns from **word sequences and word order** — stemming collapses `running`, `run`, and `runner` into the same token, destroying the morphological signal the LSTM would otherwise learn.

#### Leet-Decode Defense

The first preprocessing step in both pipelines is a character translation that decodes leet-speak **before** any other cleaning occurs:

```
@ → a    3 → e    1 → i    0 → o
$ → s    7 → t    4 → a    ! → i
```

This map is the exact inverse of the substitution table used in our adversarial attack module, providing a direct, rule-based defense against the most common evasion technique. For example:

```
Adversarial input : "C0ngr@t$ y0u'v3 W0N @ FR33 1Ph0n3!!"
After leet-decode : "Congratss you've WON a FREE iPhone!!"
After full clean  : "congratul won free iphon"     ← clearly spam
```

### 3.2 Feature Engineering

#### Classical Models — Hybrid Feature Matrix

```
X_hybrid  =  hstack([ X_tfidf   ,   X_structural ])
                      50,000 cols       9 cols
              shape: (63,920 × 50,009)
```

**TF-IDF** (fitted on `preprocess` training split only):

| Parameter | Value |
|---|---|
| `ngram_range` | (1, 2) — unigrams + bigrams |
| `max_features` | 50,000 |
| `sublinear_tf` | True — log-scale TF to reduce dominance of high-frequency terms |
| `min_df` | 2 — remove hapax legomena |

**9 Structural Features** (computed from raw email text):

| Feature | What it captures |
|---|---|
| `url_count` | Spam typically contains many embedded links |
| `email_addr_count` | Phishing emails often include fake contact addresses |
| `special_char_ratio` | `!!!`, `$$$`, `###` are hallmarks of spam |
| `uppercase_ratio` | `"CLICK NOW FREE OFFER"` — urgency through capitalization |
| `html_tag_count` | HTML-formatted spam tries to mimic professional newsletters |
| `email_length` | Spam can be very short (one-liners) or very long (wall-of-text) |
| `digit_ratio` | Phone numbers, prices, account numbers |
| `exclamation_count` | Urgency markers |
| `dollar_count` | Financial spam signals |

These structural features capture signals that TF-IDF cannot — a spam email full of URLs and exclamation marks looks the same to TF-IDF as any other text with those words, but the structural features explicitly flag it.

#### BiLSTM — Tokenized Sequences

```
Tokenizer:   vocabulary = 20,000 most frequent tokens, OOV = <OOV>
Sequences:   pad_sequences(maxlen=200, padding='post', truncating='post')
Input shape: (63,920 × 200)
```

Max length of 200 covers approximately **91% of emails** without truncation, keeping the model size manageable.

### 3.3 Models

#### Logistic Regression
A strong linear baseline for high-dimensional sparse text. Produces calibrated probabilities via `predict_proba`. (`C=1.0`, `solver='lbfgs'`, `max_iter=1000`, `class_weight='balanced'`)

#### Linear SVM (`LinearSVC`)
State-of-the-art for text classification. Finds the maximum-margin hyperplane in the 50,009-dimensional hybrid feature space. Fast at both training and inference. (`C=1.0`, `max_iter=2000`, `class_weight='balanced'`)

#### XGBoost
A gradient boosting implementation that captures non-linear interactions (e.g., high URL count combined with high exclamation count) that linear models miss. (`n_estimators=200`, `max_depth=4`, `learning_rate=0.1`, `random_state=42`)

#### Bidirectional LSTM (BiLSTM)
Processes the email token sequence in **both forward and backward directions** simultaneously. This allows the model to use future context when encoding each token — for example, the word "offer" is more suspicious after "limited time" than after "job".

```
Architecture
────────────────────────────────────────
Embedding     (20,000 vocab → 64 dim, input_len=200)
Bidirectional LSTM  (64 units, dropout=0.2, recurrent_dropout=0.2)
Dense         (32, relu)
Dropout       (0.3)
Dense         (1, sigmoid)
────────────────────────────────────────
Optimizer     :  Adam
Loss          :  Binary Crossentropy
Batch Size    :  256
EarlyStopping :  monitor=val_loss, patience=3, restore_best_weights=True
Val split     :  10% of training data
```

Training ran for **5 epochs** (EarlyStopping triggered; best weights from epoch 2 restored). Total training time: **1,038.95 seconds (~17 minutes)**.

---

## 4. Results

### 4.1 Model Leaderboard

All models evaluated on the same 15,980-email held-out test set:

| Model | Accuracy | Precision | Recall | **F1-Score** | ROC-AUC | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|
| **BiLSTM** | **0.9758** | **0.9656** | **0.9838** | **0.9746** | **0.9957** | **1,038.95** | **82,177.57** |
| Logistic Regression | 0.9752 | 0.9632 | 0.9850 | 0.9740 | 0.9966 | 112.61 | 12.07 |
| Linear SVM | 0.9583 | 0.9370 | 0.9770 | 0.9566 | 0.9925 | 7.00 | 5.52 |
| XGBoost | 0.9481 | 0.9222 | 0.9720 | 0.9464 | 0.9895 | 162.29 | 166.99 |

**Winner: BiLSTM** — highest F1 and highest Accuracy, at the cost of significantly longer training and inference time.

### 4.2 Best Model — Detailed Report

```
Classification Report — BiLSTM
─────────────────────────────────────────────────
              precision    recall  f1-score   support

         Ham     0.99      0.97      0.98      8,457
        Spam     0.97      0.98      0.97      7,523

    accuracy                         0.9758    15,980
   macro avg     0.98      0.975     0.975     15,980
weighted avg     0.976     0.976     0.975     15,980
─────────────────────────────────────────────────
```

### 4.3 Key Findings

**BiLSTM edges out Logistic Regression by 0.0006 F1 points.** The margin is narrow, but the BiLSTM's sequential modeling of token order provides a consistent, if small, advantage over bag-of-words approaches.

**Logistic Regression is a strong second** — F1 of 0.9740, nearly identical to BiLSTM, but trains in 112 seconds (vs 1,039 seconds) and infers in 12ms (vs 82,177ms). For most production use cases, Logistic Regression is the more practical choice.

**Linear SVM is the fastest classical model** — trains in only 7 seconds with F1 of 0.9566. It sacrifices about 2 F1 points relative to LR, but its 5.52ms inference makes it ideal for high-throughput real-time filtering.

**XGBoost trails the linear models** on this dataset. Despite capturing non-linear interactions, it underperforms because the 50,000 TF-IDF features are sparse and high-dimensional — a regime where gradient boosted trees are less efficient than linear models.

**BiLSTM inference is extremely slow** — 82 seconds per batch due to TensorFlow/Keras overhead on CPU. In a production deployment, GPU inference would reduce this to under 1 second.

**All models exceed 94.6% F1**, which is a strong result. The hybrid feature model — specifically the 9 structural features — is a meaningful contributor: ablation tests (TF-IDF only) showed an average F1 drop of ~0.8 percentage points across all classical models.

---

## 5. Adversarial Robustness

### 5.1 Attack Setup

Using `src/adversarial.py`, three evasion attacks were applied to all **7,523 spam emails** in the test set. Ham emails were left unchanged.

| Attack | Example | Rate |
|---|---|---|
| **Leet substitution** | `free → fr33`, `spam → $p@m` | 40% of eligible characters |
| **Symbol insertion** | `winner → winner!` | 20% of words >3 chars |
| **Whitespace injection** | `winner → win ner` | 15% of words >4 chars |

### 5.2 Results

| Model | Original F1 | Adversarial F1 | F1 Change |
|---|---|---|---|
| **BiLSTM** | 0.9746 | **0.9771** | **+0.0025** ← Most stable |
| Logistic Regression | 0.9740 | 0.9724 | −0.0016 |
| Linear SVM | 0.9566 | 0.9549 | −0.0017 |
| XGBoost | 0.9464 | 0.9440 | −0.0024 |

**Most robust: BiLSTM** — F1 is essentially unchanged (marginally higher) under all three attacks combined.

### 5.3 Analysis

All four models remained above **94.4% F1** even under adversarial attack. This is primarily due to the **leet-decode preprocessing step** — by decoding `fr33 → free` before any model sees the text, the most impactful attack (leet substitution) is neutralized at the pipeline level.

The BiLSTM shows a marginal F1 *increase* under adversarial conditions. This is because the attacks only modify spam emails, and the BiLSTM's sequential processing partially re-encounters obfuscated patterns that overlap with legitimate-text structure, slightly adjusting the decision boundary in ways that happen to improve overall metrics. The effect is within noise.

Classical models see small F1 drops because **whitespace injection** (`win ner`) creates unknown bigrams that the TF-IDF vectorizer maps to zero vectors, losing that token's signal entirely. The LSTM processes both tokens in sequence and partially recovers the pattern from surrounding context.

The **structural features** also provide robustness: an email with 10 exclamation marks is still caught by `exclamation_count` even if its text has been obfuscated.

---

## 6. Web Application

The best model (BiLSTM) and classical models are deployed as a Streamlit web app (`app/app.py`), with auto-download from HuggingFace Hub on first launch.

### Features

**Classify Tab**
- **File Upload**: Upload a `.txt` or `.eml` email file — its content automatically populates the text area
- **Text Area**: Paste or edit email content directly
- One click → SPAM / HAM verdict with a **confidence score** (e.g., "SPAM — Confidence: 97.2%")
- **Feature Breakdown** expander (classical models only) shows all 9 structural feature values computed for the submitted email, giving the user insight into *why* the model classified it that way

**Model Info Tab**
- Displays `models/model_card.txt` — best model name, all evaluation metrics, hyperparameters, and dataset statistics

### Model Auto-Detection

The app checks for model files in this order:
1. `best_lstm.keras` + `lstm_tokenizer.pkl` → loads BiLSTM pipeline
2. `best_*.pkl` + `tfidf_vectorizer.pkl` → loads classical model pipeline

Missing files are auto-downloaded from `chanhengmenh/spam_email_detection` on HuggingFace Hub.

### Inference Pipeline

**BiLSTM path:**
```
User input (raw email text)
        ↓
preprocess_lstm()     ← leet-decode, clean, no stemming
        ↓
tokenizer.texts_to_sequences()
        ↓
pad_sequences(maxlen=200)
        ↓
BiLSTM.predict()      ← sigmoid probability
        ↓
Verdict + Confidence Score
```

**Classical path:**
```
User input (raw email text)
        ↓
preprocess()          ← leet-decode, clean, stem
        ↓
tfidf.transform()     ← fitted vectorizer, no refitting
        ↓
extract_structural()  ← 9 structural features
        ↓
hstack([tfidf, struct])
        ↓
model.predict()       ← predict_proba or decision_function
        ↓
Verdict + Confidence Score
```

### Example Outputs

| Input | Verdict | Confidence |
|---|---|---|
| `"Congratulations! You've WON a FREE iPhone 15! Click NOW!!"` | **SPAM** | 97.2% |
| `"C0ngr@ts! Y0u'v3 W0N @ FR33 1Ph0n3!"` *(adversarial)* | **SPAM** | 97.1% |
| `"Buy cheap Viagra online — 80% off, no prescription needed"` | **SPAM** | 99.2% |
| `"Hi John, reminder about our team meeting tomorrow at 10am"` | **HAM** | 99.4% |
| `"Dear Professor, writing to ask about the ML assignment deadline"` | **HAM** | 98.3% |

The adversarial input `C0ngr@ts! Y0u'v3 W0N...` is still correctly identified as spam — leet-decode strips the obfuscation before the model sees it.

---

## 7. Conclusion

### What We Built

A complete spam detection pipeline that trains and compares four machine learning models on a dataset of 80,000 emails, tests adversarial robustness, and deploys the winner as a web application with both text-paste and file-upload input.

### Key Results

| Metric | Best Result | Best Model |
|---|---|---|
| F1-Score | **0.9746** | BiLSTM |
| ROC-AUC | **0.9966** | Logistic Regression |
| Fastest Inference | **5.52 ms** | Linear SVM |
| Adversarial Robustness | **+0.0025 F1 change** | BiLSTM |

### What We Learned

**The BiLSTM wins, but barely.** It edges out Logistic Regression by 0.0006 F1 points — essentially a tie — while costing 9× longer training time and 6,800× slower inference. The result confirms that for formal email text with clear lexical spam signals, deep learning's advantage over classical methods is marginal.

**Structural features matter.** Removing the 9 structural features dropped average F1 by ~0.8 points across all classical models. URL counts, uppercase ratios, and exclamation counts capture spam signals that pure text analysis misses — particularly for HTML-formatted and financially-themed spam.

**Preprocessing is the best adversarial defense.** The leet-decode step neutralized leet substitution at the pipeline level before any model was involved, keeping F1 above 94.4% under three simultaneous attacks. Model-level robustness (BiLSTM's sequential processing) provided additional marginal stability.

**Deep learning is not always the answer.** The BiLSTM trained for ~17 minutes, costs 82 seconds per inference batch on CPU, and only marginally outperforms Logistic Regression (F1 0.9746 vs 0.9740). For high-throughput production filtering, Logistic Regression or Linear SVM are far more practical.

### Limitations

- The dataset is English-only; preprocessing (stopwords, stemming) is language-specific
- The model is static — spam tactics evolve, and the classifier will degrade over time without periodic retraining
- No sender or header features (domain reputation, SPF/DKIM) which are used by real-world spam filters
- BiLSTM inference is slow on CPU (~82 seconds per batch); GPU deployment would be required for real-time use

### Future Work

- **Fine-tune a transformer** (DistilBERT) — contextual embeddings would likely close the gap between deep learning and classical models on email text, with better inference speed than LSTM
- **Online learning** — periodically retrain with newly-flagged spam to adapt to evolving attack patterns
- **Header features** — add domain age, SPF record validity, and DKIM signature as additional structural features
- **Explainability** — integrate SHAP to show users which specific words and structural signals drove each prediction

---

## Appendix

### A. Project File Structure

```
final-project/
├── notebooks/
│   └── spam_email_detection.ipynb   ← full pipeline, 16 sections
├── src/
│   ├── preprocess.py                ← preprocess(), preprocess_lstm(), LEET_MAP
│   ├── features.py                  ← extract_structural(), build_hybrid_features()
│   ├── models.py                    ← get_all_models(), train_and_compare()
│   └── adversarial.py               ← generate_adversarial_set(), apply_all()
├── app/
│   └── app.py                       ← Streamlit app (file upload + text paste)
├── data/
│   └── raw/
│       └── spam_Emails_data.csv
├── models/                          ← auto-downloaded from HuggingFace on first run
│   ├── best_lstm.keras
│   ├── lstm_tokenizer.pkl
│   ├── best_logistic_regression.pkl
│   ├── best_linear_svc.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_card.txt
└── requirements.txt
```

### B. Model Card (BiLSTM)

```
Model Card — Spam Email Detection
════════════════════════════════════════
Best Model       : BiLSTM
Model Type       : Deep Learning
F1-Score         : 0.9746
ROC-AUC          : 0.9957
Accuracy         : 0.9758
Precision        : 0.9656
Recall           : 0.9838
Train Time (s)   : 1,038.95
Infer Time (ms)  : 82,177.57

Hyperparameters:
  Vocab Size     : 20,000
  Max Length     : 200
  Embedding Dim  : 64
  LSTM Units     : 64
  Batch Size     : 256
  Epochs Trained : 5

Dataset:
  File           : spam_Emails_data.csv
  Train size     : 63,920
  Test size      : 15,980
  Spam ratio     : 47.1%
```

### C. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models (run all notebook cells top to bottom)
jupyter notebook notebooks/spam_email_detection.ipynb

# Launch the web app (models auto-download from HuggingFace on first run)
streamlit run app/app.py
```
