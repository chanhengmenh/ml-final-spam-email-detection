import sys
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocess import preprocess, preprocess_lstm
from src.features import build_hybrid_features, extract_structural

MODELS_DIR    = Path(__file__).parent.parent / "models"
MAX_LEN       = 200   # must match notebook training config
HF_MODEL_REPO = "chanhengmenh/spam_email_detection"


def _ensure_model_files():
    """Download all model files from HuggingFace if the directory is empty."""
    keras_path = MODELS_DIR / "best_lstm.keras"
    tok_path   = MODELS_DIR / "lstm_tokenizer.pkl"
    if not keras_path.exists() or not tok_path.exists():
        with st.spinner("Downloading models from HuggingFace Hub…"):
            from huggingface_hub import snapshot_download
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=HF_MODEL_REPO,
                local_dir=str(MODELS_DIR),
                ignore_patterns=["*.md", ".gitattributes"],
            )

st.set_page_config(page_title="Spam Email Detector", page_icon="📩", layout="centered")
st.title("Email Spam Detection System")
st.caption("ITM-390 Machine Learning — Group 5 | AUPP")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_best_model():
    """
    Auto-detect model type from models/:
    - If best_lstm.keras exists → load LSTM + lstm_tokenizer.pkl
    - Otherwise → load best_*.pkl classical model + tfidf_vectorizer.pkl
    Falls back to HuggingFace Hub download if files are missing.
    """
    _ensure_model_files()
    keras_path = MODELS_DIR / "best_lstm.keras"
    if keras_path.exists():
        tok_path = MODELS_DIR / "lstm_tokenizer.pkl"
        if not tok_path.exists():
            raise FileNotFoundError("lstm_tokenizer.pkl not found alongside best_lstm.keras")
        from tensorflow.keras.models import load_model as keras_load
        model     = keras_load(str(keras_path))
        with open(tok_path, 'rb') as f:
            artifact = pickle.load(f)
        return model, artifact, 'lstm'

    pkl_files = sorted(MODELS_DIR.glob("best_*.pkl"))
    if not pkl_files:
        return None, None, None

    import joblib
    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    if not tfidf_path.exists():
        raise FileNotFoundError("tfidf_vectorizer.pkl not found alongside best model .pkl")
    model    = joblib.load(pkl_files[0])
    artifact = joblib.load(tfidf_path)
    return model, artifact, 'classical'


def load_model_card() -> str | None:
    path = MODELS_DIR / "model_card.txt"
    return path.read_text(encoding='utf-8') if path.exists() else None


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(raw_text: str, model, artifact, model_type: str) -> tuple[str, float]:
    """Return (verdict, confidence) for raw_text."""
    if model_type == 'lstm':
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        cleaned = preprocess_lstm(raw_text)
        seq     = artifact.texts_to_sequences([cleaned])
        padded  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        proba   = float(model.predict(padded, verbose=0)[0][0])
        pred    = int(proba >= 0.5)
    else:
        raw_s  = pd.Series([raw_text])
        proc_s = raw_s.apply(preprocess)
        X, _   = build_hybrid_features(proc_s, raw_s, tfidf=artifact, fit=False)
        pred   = int(model.predict(X)[0])
        if hasattr(model, 'predict_proba'):
            proba = float(model.predict_proba(X)[0][1])
        elif hasattr(model, 'decision_function'):
            ds    = float(model.decision_function(X)[0])
            proba = float(1 / (1 + np.exp(-ds)))
        else:
            proba = float(pred)

    conf    = proba if pred == 1 else 1 - proba
    verdict = 'SPAM' if pred == 1 else 'HAM'
    return verdict, conf


# ── App ───────────────────────────────────────────────────────────────────────
try:
    model, artifact, model_type = load_best_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

if model is None:
    st.warning("No trained models found in `models/`. Run the training notebook first.")
    st.stop()

tab_classify, tab_info = st.tabs(["Classify", "Model Info"])

# ── Classify tab ──────────────────────────────────────────────────────────────
with tab_classify:
    email_text = st.text_area(
        "Paste email content below:",
        height=250,
        placeholder="Subject: You've WON a FREE prize! Click here now!!!",
    )

    if st.button("Classify", use_container_width=True):
        if not email_text.strip():
            st.warning("Please enter some email text.")
        else:
            verdict, conf = predict(email_text, model, artifact, model_type)

            if verdict == 'SPAM':
                st.error(f"**SPAM** detected — Confidence: {conf*100:.1f}%")
            else:
                st.success(f"**HAM** (legitimate) — Confidence: {conf*100:.1f}%")

            if model_type == 'classical':
                with st.expander("Feature Breakdown (9 structural features)"):
                    st.json(extract_structural(email_text))

# ── Model Info tab ────────────────────────────────────────────────────────────
with tab_info:
    card = load_model_card()
    if card:
        st.subheader("Model Card")
        st.code(card, language=None)
    else:
        st.info("No `model_card.txt` found. Run the training notebook to generate one.")
