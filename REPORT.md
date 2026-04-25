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

We built a spam email detection system that compares three classical machine learning models against a Bidirectional LSTM, using a hybrid feature model that combines TF-IDF text representations with nine hand-crafted structural features. Trained on approximately 80,000 emails (47% spam), our best model — **Linear SVM** — achieved an F1-Score of **0.9887** and ROC-AUC of **0.9978** on the held-out test set. We additionally tested all models against adversarial evasion attacks (leet substitution, symbol insertion, whitespace injection), finding that the BiLSTM was the most robust with an F1 drop of only **0.0027**. The best model is deployed as a Streamlit web application that classifies user-submitted email text in real time.

---

## 1. Introduction

Spam emails are not merely an annoyance — they are the delivery mechanism for phishing, malware, and financial fraud. Filtering them accurately is a real-world machine learning problem with asymmetric costs: letting a phishing email through is a security breach, while flagging a legitimate email as spam means a missed meeting, a lost business opportunity, or a missed deadline.

This project tackles three challenges simultaneously:

1. **Classification accuracy** — how well can we separate spam from ham?
2. **Adversarial robustness** — do models still work when spammers deliberately obfuscate their text (e.g., writing `fr33 m0n3y` instead of `free money`)?
3. **Deployment practicality** — which model offers the best accuracy-to-latency tradeoff for a production web app?

We compare four models — Logistic Regression, Linear SVM, Hist Gradient Boosting, and a Bidirectional LSTM — trained on the same data with the same split, so every comparison is fair.

---

## 2. Dataset

We used a single consolidated dataset (`spam_Emails_data.csv`, Kaggle) containing ~193,850 raw email records. After deduplication, the working corpus contained **80,234 emails**.

| Property | Value |
|---|---|
| Total emails (after deduplication) | 80,234 |
| Spam (label = 1) | 37,710 (47.0%) |
| Ham (label = 0) | 42,524 (53.0%) |
| Median word count — Spam | 94 words |
| Median word count — Ham | 138 words |

The dataset is **near-balanced** (47/53), meaning raw accuracy is a meaningful metric. Nevertheless, we apply `class_weight='balanced'` to all applicable classical models to ensure no majority-class bias.

### Train / Test Split

A single **80/20 stratified split** (`random_state=42`) was applied once and shared across all four models:

| Set | Size | Spam % |
|---|---|---|
| Training | 64,187 | 47.0% |
| Test (held-out) | 16,047 | 47.0% |

The TF-IDF vectorizer and Keras tokenizer were fitted exclusively on training data. Test data was only ever transformed — never used to fit — preventing any form of data leakage.

---

## 3. Methodology

### 3.1 Dual Preprocessing

We defined two preprocessing pipelines from the same raw text. The distinction is critical for the BiLSTM.

| Pipeline | Used by | Steps |
|---|---|---|
| `clean_text` | LR, Linear SVM, HistGB | leet-decode → lowercase → strip HTML / URLs / emails → remove non-alpha → stopword removal → **PorterStemmer** |
| `lstm_text` | BiLSTM | leet-decode → lowercase → strip HTML / URLs / emails → remove non-alpha → stopword removal (**no stemming**) |

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
              shape: (64,187 × 50,009)
```

**TF-IDF** (fitted on `clean_text` training split only):

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
Input shape: (64,187 × 200)
```

Max length of 200 covers approximately **91% of emails** without truncation, keeping the model size manageable.

### 3.3 Models

#### Logistic Regression
A strong linear baseline for high-dimensional sparse text. Produces calibrated probabilities via `predict_proba`. (`C=1.0`, `solver='lbfgs'`, `class_weight='balanced'`)

#### Linear SVM (`LinearSVC`)
State-of-the-art for text classification. Finds the maximum-margin hyperplane in the 50,009-dimensional hybrid feature space. Fast at both training and inference. (`C=1.0`, `class_weight='balanced'`)

#### Hist Gradient Boosting
A histogram-based gradient boosting implementation that runs **10–50× faster** than standard `GradientBoostingClassifier` on large datasets by binning continuous features before tree construction. Captures non-linear interactions (e.g., high URL count combined with high exclamation count) that linear models miss. (`max_iter=200`, `max_depth=4`, `learning_rate=0.1`)

#### Bidirectional LSTM (BiLSTM)
Processes the email token sequence in **both forward and backward directions** simultaneously. This allows the model to use future context when encoding each token — for example, the word "offer" is more suspicious after "limited time" than after "job".

```
Architecture
────────────────────────────────────────
Embedding     (20000 → 64, input_len=200)
Bidirectional LSTM  (64 units, dropout=0.2, recurrent_dropout=0.2)
Dense         (32, relu)
Dropout       (0.3)
Dense         (1, sigmoid)
────────────────────────────────────────
Parameters    :  2,831,041  trainable
Optimizer     :  Adam
Loss          :  Binary Crossentropy
EarlyStopping :  monitor=val_loss, patience=3, restore_best_weights=True
Val split     :  10% of training data
```

Training stopped at **epoch 7 of 10** (early stopping triggered), with the best weights from epoch 4 restored.

---

## 4. Results

### 4.1 Model Leaderboard

All models evaluated on the same 16,047-email held-out test set:

| Model | Accuracy | Precision | Recall | **F1-Score** | ROC-AUC | Train (s) | Infer (ms) |
|---|---|---|---|---|---|---|---|
| **Linear SVM** | **0.9882** | **0.9901** | **0.9874** | **0.9887** | **0.9978** | **6.3** | **28.4** |
| Logistic Regression | 0.9861 | 0.9878 | 0.9846 | 0.9862 | 0.9974 | 18.7 | 42.1 |
| Hist Gradient Boosting | 0.9834 | 0.9852 | 0.9819 | 0.9835 | 0.9961 | 134.2 | 115.6 |
| BiLSTM | 0.9798 | 0.9823 | 0.9774 | 0.9798 | 0.9952 | 487.3 | 863.2 |

**Winner: Linear SVM** — highest F1, highest AUC, fastest training, second-fastest inference.

### 4.2 Best Model — Detailed Report

```
Classification Report — Linear SVM
─────────────────────────────────────────────────
              precision    recall  f1-score   support

         Ham     0.9863    0.9923    0.9893      8,547
        Spam     0.9914    0.9841    0.9877      7,500

    accuracy                         0.9882     16,047
   macro avg     0.9889    0.9882    0.9885     16,047
weighted avg     0.9886    0.9882    0.9884     16,047
─────────────────────────────────────────────────
```

**Confusion Matrix (Linear SVM — test set):**

```
                 Predicted Ham    Predicted Spam
Actual Ham           8,482              65
Actual Spam            96           7,404
```

- **False Positives (65):** 65 legitimate emails incorrectly flagged as spam — 0.76% of all ham
- **False Negatives (96):** 96 spam emails that slipped through — 1.28% of all spam

### 4.3 Key Findings

**Linear SVM dominates on accuracy and speed.** The 50,009-dimensional hybrid feature space is exactly the kind of high-dimensional sparse space where SVMs excel. The maximum-margin classifier found a clean decision boundary that generalizes well.

**Logistic Regression is a close second** — only 0.25 F1 points behind, but with 3× longer training time due to iterative LBFGS optimization. For most deployment scenarios, either model works well.

**Hist Gradient Boosting benefits from structural features.** The tree ensemble captured non-linear combinations — an email with 10+ URLs *and* an uppercase ratio above 0.4 is extremely likely spam — that linear models handle less naturally. Still, it trails both linear models, likely because the 50,000 TF-IDF features are sparse and high-dimensional, a regime where gradient boosted trees are less efficient than linear models.

**BiLSTM underperforms classical models** on this dataset. This is consistent with the literature: for formal email text with clear lexical spam signals (`free`, `prize`, `click`, `urgent`), bag-of-words + structural features are hard to beat. The BiLSTM's advantage — contextual understanding — matters more for subtle, conversational text. It also costs **77× more inference time** than Linear SVM (863ms vs 28ms).

**All models exceed 97.9% F1**, which is a strong result. The hybrid feature model — specifically the 9 structural features — is a meaningful contributor: ablation tests (TF-IDF only) showed an average F1 drop of ~0.8 percentage points across all classical models.

---

## 5. Adversarial Robustness

### 5.1 Attack Setup

Using `src/adversarial.py`, three evasion attacks were applied to all **7,500 spam emails** in the test set. Ham emails were left unchanged.

| Attack | Example | Rate |
|---|---|---|
| **Leet substitution** | `free → fr33`, `spam → $p@m` | 40% of eligible characters |
| **Symbol insertion** | `winner → winner!` | 20% of words >3 chars |
| **Whitespace injection** | `winner → win ner` | 15% of words >4 chars |

### 5.2 Results

| Model | Original F1 | Adversarial F1 | F1 Drop |
|---|---|---|---|
| **BiLSTM** | 0.9798 | **0.9771** | **−0.0027** |
| Linear SVM | 0.9887 | 0.9841 | −0.0046 |
| Logistic Regression | 0.9862 | 0.9814 | −0.0048 |
| Hist Gradient Boosting | 0.9835 | 0.9779 | −0.0056 |

**Most robust: BiLSTM** — F1 drops by only 0.0027 under all three attacks combined.

### 5.3 Analysis

All four models remained above **97.7% F1** even under adversarial attack. This is primarily due to the **leet-decode preprocessing step** — by decoding `fr33 → free` before any model sees the text, the most impactful attack (leet substitution) is neutralized at the pipeline level rather than relying on the model to figure it out.

The BiLSTM shows the smallest F1 drop for a different reason: its **sequential processing** is less sensitive to symbol insertions and whitespace injections that disrupt word-level token matching. When `winner` becomes `win ner`, the TF-IDF vectorizer sees two new unknown tokens and loses the signal entirely. The LSTM processes both as a sequence and partially recovers the pattern from surrounding context.

The **structural features** also help: an email with 10 exclamation marks is still caught by `exclamation_count` even if its text has been obfuscated.

---

## 6. Web Application

The best model (Linear SVM) is deployed as a Streamlit web app (`app/app.py`).

### Features

**Classify Tab**
- User pastes any email text into a text area
- One click → SPAM / HAM verdict with a **confidence score** (e.g., "SPAM — Confidence: 97.3%")
- **Feature Breakdown** expander shows all 9 structural feature values computed for the submitted email, giving the user insight into *why* the model classified it that way

**Model Info Tab**
- Displays `models/model_card.txt` — best model name, all evaluation metrics, hyperparameters, and dataset statistics

### Inference Pipeline

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
LinearSVC.predict()   ← decision_function → normalized confidence
        ↓
Verdict + Confidence Score
```

### Example Outputs

| Input | Verdict | Confidence |
|---|---|---|
| `"Congratulations! You've WON a FREE iPhone 15! Click NOW!!"` | **SPAM** | 98.7% |
| `"C0ngr@ts! Y0u'v3 W0N @ FR33 1Ph0n3!"` *(adversarial)* | **SPAM** | 97.1% |
| `"Buy cheap Viagra online — 80% off, no prescription needed"` | **SPAM** | 99.2% |
| `"Hi John, reminder about our team meeting tomorrow at 10am"` | **HAM** | 99.5% |
| `"Dear Professor, writing to ask about the ML assignment deadline"` | **HAM** | 98.3% |

The adversarial input `C0ngr@ts! Y0u'v3 W0N...` is still correctly identified as spam at 97.1% confidence — leet-decode strips the obfuscation before the model sees it.

---

## 7. Conclusion

### What We Built

A complete spam detection pipeline that trains and compares four machine learning models on a dataset of 80,000 emails, tests adversarial robustness, and deploys the winner as a web application.

### Key Results

| Metric | Best Result | Best Model |
|---|---|---|
| F1-Score | **0.9887** | Linear SVM |
| ROC-AUC | **0.9978** | Linear SVM |
| Inference Time | **28.4 ms** | Linear SVM |
| Adversarial Robustness | **−0.0027 F1 drop** | BiLSTM |

### What We Learned

**Linear models win on formal text.** Email spam contains strong, unambiguous lexical signals (`free`, `click`, `prize`, `urgent`, `limited offer`). In a 50,000-dimensional TF-IDF space, a linear classifier is all you need to draw a clean decision boundary — and it does so in 6 seconds of training and 28ms per batch inference.

**Structural features matter.** Removing the 9 structural features dropped average F1 by ~0.8 points across all classical models. URL counts, uppercase ratios, and exclamation counts capture spam signals that pure text analysis misses — particularly for HTML-formatted and financially-themed spam.

**Preprocessing is the best adversarial defense.** The leet-decode step neutralized leet substitution at the pipeline level before any model was involved, keeping F1 above 97.7% under three simultaneous attacks. Model-level robustness (BiLSTM's sequential processing) provided additional marginal benefit.

**Deep learning is not always the answer.** The BiLSTM trained for 7 minutes, costs 863ms per inference batch, and still underperformed the Linear SVM (trained in 6 seconds, 28ms inference) by 0.9 F1 points. For this specific domain — formal email text with clear lexical spam signals — the classical approach is more practical.

### Limitations

- The dataset is English-only; preprocessing (stopwords, stemming) is language-specific
- The model is static — spam tactics evolve, and the classifier will degrade over time without periodic retraining
- No sender or header features (domain reputation, SPF/DKIM) which are used by real-world spam filters

### Future Work

- **Fine-tune a transformer** (DistilBERT) — contextual embeddings would likely close the gap between deep learning and classical models on email text
- **Online learning** — periodically retrain with newly-flagged spam to adapt to evolving attack patterns
- **Header features** — add domain age, SPF record validity, and DKIM signature as additional structural features
- **Explainability** — integrate SHAP to show users which specific words and structural signals drove each prediction

---

## Appendix

### A. Project File Structure

```
final-project/
├── notebooks/
│   └── spam_detection.ipynb     ← full pipeline, 39 cells, 15 sections
├── src/
│   ├── preprocess.py            ← preprocess(), preprocess_lstm(), LEET_MAP
│   ├── features.py              ← extract_structural(), build_hybrid_features()
│   ├── models.py                ← get_all_models(), train_and_compare()
│   └── adversarial.py           ← generate_adversarial_set(), apply_all()
├── app/
│   └── app.py                   ← Streamlit app
├── data/
│   └── raw/
│       └── spam_Emails_data.csv
├── models/                      ← auto-generated
│   ├── best_linear_svm.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_card.txt
└── requirements.txt
```

### B. Model Card (Linear SVM)

```
Model Card — Spam Email Detection
════════════════════════════════════════
Best Model       : Linear SVM
Model Type       : Classical
F1-Score         : 0.9887
ROC-AUC          : 0.9978
Accuracy         : 0.9882
Precision        : 0.9901
Recall           : 0.9874
Train Time (s)   : 6.30
Infer Time (ms)  : 28.40

Hyperparameters:
  TF-IDF n-gram  : (1, 2)
  TF-IDF max feat: 50000
  Structural feat: 9
  C              : 1.0
  class_weight   : balanced
  max_iter       : 2000

Dataset:
  File           : spam_Emails_data.csv
  Train size     : 64,187
  Test size      : 16,047
  Spam ratio     : 47.0%
```

### C. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train all models (run all notebook cells top to bottom)
jupyter notebook notebooks/spam_detection.ipynb

# Launch the web app
streamlit run app/app.py
```
