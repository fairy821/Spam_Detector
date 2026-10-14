"""Helpers shared by train.py and app.py (and later the Gmail connector)."""
import re

_SUBJECT_RE = re.compile(r"^\s*subject\s*:\s*", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise an email so training and prediction see identical input."""
    text = str(text)
    text = _SUBJECT_RE.sub("", text, count=1)  # drop leading "Subject:"
    text = _HTML_RE.sub(" ", text)
    text = _URL_RE.sub(" urltoken ", text)
    text = _EMAIL_RE.sub(" emailtoken ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip().lower()


def predict_email(model, subject: str, body: str) -> float:
    """Return the probability (0-1) that an email is spam."""
    raw = f"Subject: {subject}\n{body}"
    return float(model.predict_proba([clean_text(raw)])[0, 1])


def top_contributing_words(model, subject: str, body: str, n: int = 8):
    """
    Explain a prediction: which words pushed the score toward spam / not spam.
    Works because the deployed model is a linear model (word weight x tf-idf value).
    Returns (spam_words, ham_words) as lists of (word, score).
    """
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    vec = tfidf.transform([clean_text(f"Subject: {subject}\n{body}")])
    contrib = vec.multiply(clf.coef_[0]).tocoo()
    if contrib.nnz == 0:
        return [], []

    names = tfidf.get_feature_names_out()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: p[1])

    spam_words = [(names[i], float(v)) for i, v in reversed(pairs) if v > 0][:n]
    ham_words = [(names[i], float(v)) for i, v in pairs if v < 0][:n]
    return spam_words, ham_words


def predict_message(model, message: str) -> float:
    """Return the probability (0-1) that an SMS/message is spam."""
    return float(model.predict_proba([clean_text(message)])[0, 1])


def top_contributing_words_message(model, message: str, n: int = 8):
    """
    Explain an SMS prediction: which words pushed the score toward spam / not spam.
    Returns (spam_words, ham_words) as lists of (word, score).
    """
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]

    vec = tfidf.transform([clean_text(message)])
    contrib = vec.multiply(clf.coef_[0]).tocoo()
    if contrib.nnz == 0:
        return [], []

    names = tfidf.get_feature_names_out()
    pairs = sorted(zip(contrib.col, contrib.data), key=lambda p: p[1])

    spam_words = [(names[i], float(v)) for i, v in reversed(pairs) if v > 0][:n]
    ham_words = [(names[i], float(v)) for i, v in pairs if v < 0][:n]
    return spam_words, ham_words


def verdict(prob: float) -> str:
    if prob >= 0.85:
        return "🚨 Spam"
    if prob >= 0.5:
        return "⚠️ Likely spam"
    if prob >= 0.25:
        return "🔍 Suspicious"
    return "✅ Looks safe"

