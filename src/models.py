import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

MODELS_DIR   = Path(__file__).parent.parent / "models"
RANDOM_STATE = 42


def get_all_models() -> dict:
    """Return the 3 classical models used in the pipeline."""
    return {
        "Logistic Regression"   : LogisticRegression(
                                      C=1.0, max_iter=1000, solver='lbfgs',
                                      class_weight='balanced',
                                      random_state=RANDOM_STATE),
        "Linear SVM"            : LinearSVC(
                                      C=1.0, max_iter=2000,
                                      class_weight='balanced',
                                      random_state=RANDOM_STATE),
        "XGBoost"               : XGBClassifier(
                                      n_estimators=200, max_depth=4, learning_rate=0.1,
                                      random_state=RANDOM_STATE,
                                      eval_metric='logloss', verbosity=0),
    }


def evaluate(model, X_test, y_test) -> dict:
    """Return evaluation metrics for a fitted model."""
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        ds = model.decision_function(X_test)
        y_score = (ds - ds.min()) / (ds.max() - ds.min() + 1e-9)
    else:
        y_score = y_pred.astype(float)
    return {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, zero_division=0),
        "recall"    : recall_score(y_test, y_pred, zero_division=0),
        "f1"        : f1_score(y_test, y_pred, zero_division=0),
        "roc_auc"   : roc_auc_score(y_test, y_score),
    }


def train_and_compare(X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Train all classical models, evaluate, save each, and return a comparison DataFrame."""
    MODELS_DIR.mkdir(exist_ok=True)
    results = []

    for name, model in get_all_models().items():
        print(f"  Training {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        metrics = evaluate(model, X_test, y_test)
        metrics["model"]        = name
        metrics["train_time_s"] = round(train_time, 3)
        results.append(metrics)

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODELS_DIR / f"{safe_name}.pkl")

    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"]
    df = pd.DataFrame(results)[cols]
    return df.sort_values("f1", ascending=False).reset_index(drop=True)


def load_model(name: str):
    """Load a saved model by filename stem (e.g. 'logistic_regression')."""
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    return joblib.load(path)
