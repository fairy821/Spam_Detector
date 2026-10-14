"""
Train a spam classifier on the SMS Spam Collection dataset (data/spam.csv).

Usage:
    python train_sms.py                       # uses data/spam.csv
    python train_sms.py --data path/to.csv

Outputs (in models/):
    sms_spam_model.joblib      TF-IDF + Logistic Regression for SMS
    sms_metrics.json           test-set metrics
    sms_confusion_matrix.png   confusion matrix
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spam_utils import clean_text

RANDOM_STATE = 42
MODEL_DIR = Path("models")


def load_sms_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")
    df["spam"] = (df["label"].str.strip().str.lower() == "spam").astype(int)
    df["clean"] = df["text"].apply(clean_text)
    df = df[df["clean"].str.len() > 0]
    print(f"Loaded {len(df)} SMS messages")
    print(df["spam"].value_counts().rename({0: "ham", 1: "spam"}).to_string(), "\n")
    return df


def build_models() -> dict:
    def tfidf():
        return TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            min_df=1,
            sublinear_tf=True,
        )

    return {
        "Logistic Regression": Pipeline(
            [
                ("tfidf", tfidf()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Linear SVM": Pipeline(
            [
                ("tfidf", tfidf()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(class_weight="balanced"), cv=3
                    ),
                ),
            ]
        ),
    }


def main(data_path: str) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_sms_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"],
        df["spam"],
        test_size=0.2,
        stratify=df["spam"],
        random_state=RANDOM_STATE,
    )

    models = build_models()
    metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in models.items():
        print(f"=== {name} ===")
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "cv_f1_mean": float(cv_f1.mean()),
            "cv_f1_std": float(cv_f1.std()),
        }
        print(f"5-fold CV F1: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
        print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))

    deployed = models["Logistic Regression"]
    joblib.dump(deployed, MODEL_DIR / "sms_spam_model.joblib")

    metrics["_meta"] = {
        "deployed_model": "Logistic Regression",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "dataset": "SMS Spam Collection",
    }
    (MODEL_DIR / "sms_metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        deployed, X_test, y_test, display_labels=["Not spam", "Spam"], cmap="Purples", ax=ax
    )
    ax.set_title("SMS Logistic Regression - test set")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "sms_confusion_matrix.png", dpi=150)

    print("Saved sms_spam_model.joblib, sms_metrics.json and sms_confusion_matrix.png to models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/spam.csv")
    main(parser.parse_args().data)
