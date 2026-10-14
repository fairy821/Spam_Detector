"""
Streamlit web app for the spam detector (email + SMS).

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from spam_utils import (
    clean_text,
    predict_email,
    predict_message,
    top_contributing_words,
    top_contributing_words_message,
    verdict,
)

MODEL_PATH = Path("models/spam_model.joblib")
METRICS_PATH = Path("models/metrics.json")
CM_PATH = Path("models/confusion_matrix.png")

SMS_MODEL_PATH = Path("models/sms_spam_model.joblib")
SMS_METRICS_PATH = Path("models/sms_metrics.json")
SMS_CM_PATH = Path("models/sms_confusion_matrix.png")

st.set_page_config(page_title="Spam Detector", page_icon="🛡️", layout="wide")


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_sms_model():
    return joblib.load(SMS_MODEL_PATH)


# ── guard: email model is required ───────────────────────────────────────────
if not MODEL_PATH.exists():
    st.error("Email model not found. Run `py train.py` first.")
    st.stop()

model = load_model()

sms_model_available = SMS_MODEL_PATH.exists()
if sms_model_available:
    sms_model = load_sms_model()

# ── page header ──────────────────────────────────────────────────────────────
st.title("🛡️ Spam Detector")
st.caption("TF-IDF + Logistic Regression • Emails & SMS Messages")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab_email, tab_sms, tab_perf = st.tabs(["📧 Check an email", "💬 Check a message", "📊 Model performance"])


# ═══════════════════════════════════════════════════════════════ EMAIL TAB ═══
with tab_email:
    st.markdown(
        "Paste any **email** to check whether it looks like spam. "
        "The model was trained on the [Spam Email Dataset](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) "
        "(~5,700 labelled emails)."
    )

    SAMPLES = {
        "-- choose a sample --": ("", ""),
        "Spam-like": (
            "URGENT: You have WON $1,000,000!!!",
            "Congratulations! Click here http://claim-prize.example to claim your "
            "free money now. Limited time offer, act now and send your bank details.",
        ),
        "Normal work email": (
            "Meeting notes from Tuesday",
            "Hi team, attached are the notes from Tuesday's meeting. Please review "
            "the action items and let me know if anything is missing before Friday.",
        ),
    }

    def _on_email_sample_change():
        subj, body_val = SAMPLES[st.session_state["email_sample"]]
        st.session_state["email_subject"] = subj
        st.session_state["email_body"] = body_val

    st.selectbox(
        "Try a sample email",
        list(SAMPLES),
        key="email_sample",
        on_change=_on_email_sample_change,
    )

    left, right = st.columns([3, 2])
    with left:
        subject = st.text_input("Subject", key="email_subject")
        body = st.text_area("Email body", height=260, key="email_body")
        run = st.button("Analyse email", type="primary", key="email_run")

    with right:
        if run:
            if not (subject.strip() or body.strip()):
                st.warning("Enter a subject or body first.")
            else:
                prob = predict_email(model, subject, body)
                label = verdict(prob)

                st.subheader(label)
                st.progress(min(max(prob, 0.0), 1.0), text=f"Spam probability: {prob:.1%}")

                spam_words, ham_words = top_contributing_words(model, subject, body)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Pushed toward spam**")
                    if spam_words:
                        st.dataframe(
                            pd.DataFrame(spam_words, columns=["word", "weight"]),
                            hide_index=True,
                        )
                    else:
                        st.write("-")
                with c2:
                    st.markdown("**Pushed toward safe**")
                    if ham_words:
                        st.dataframe(
                            pd.DataFrame(ham_words, columns=["word", "weight"]),
                            hide_index=True,
                        )
                    else:
                        st.write("-")
        else:
            st.info("Paste an email and click **Analyse email**.")

    st.divider()
    st.subheader("Batch check")
    st.caption("Upload a CSV with a `text` column to score many emails at once.")
    up = st.file_uploader("CSV file", type="csv", key="email_batch")
    if up is not None:
        batch = pd.read_csv(up)
        if "text" not in batch.columns:
            st.error("The CSV needs a column named `text`.")
        else:
            probs = model.predict_proba(batch["text"].astype(str).map(clean_text))[:, 1]
            batch["spam_probability"] = probs.round(4)
            batch["verdict"] = [verdict(p) for p in probs]
            st.dataframe(batch, use_container_width=True)
            st.download_button(
                "Download results",
                batch.to_csv(index=False).encode("utf-8"),
                "spam_predictions.csv",
                "text/csv",
                key="email_download",
            )


# ═══════════════════════════════════════════════════════════════ SMS TAB ════
with tab_sms:
    if not sms_model_available:
        st.warning(
            "SMS model not found. Run `py train_sms.py` first to train the SMS classifier."
        )
        st.stop()

    st.markdown(
        "Paste any **SMS or short message** to check whether it looks like spam. "
        "The model was trained on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) "
        "(5,574 labelled messages)."
    )

    SMS_SAMPLES = {
        "-- choose a sample --": "",
        "Spam: Prize winner": (
            "Congratulations! You've been selected as a winner. "
            "Call 08712300150 NOW to claim your £1000 prize. T&C apply."
        ),
        "Spam: Free offer": (
            "FREE MSG: Buy any of our services and get 50% off! "
            "Reply YES to claim or STOP to opt out. Offer ends midnight."
        ),
        "Ham: Friend message": "Hey, are you coming to the party tonight? Let me know!",
        "Ham: Reminder": "Don't forget your dentist appointment tomorrow at 10am.",
    }

    def _on_sms_sample_change():
        st.session_state["sms_input"] = SMS_SAMPLES[st.session_state["sms_sample"]]

    st.selectbox(
        "Try a sample message",
        list(SMS_SAMPLES),
        key="sms_sample",
        on_change=_on_sms_sample_change,
    )

    sms_left, sms_right = st.columns([3, 2])

    with sms_left:
        sms_text = st.text_area(
            "Message text",
            height=180,
            placeholder="Type or paste an SMS / short message here…",
            key="sms_input",
        )
        char_count = len(sms_text)
        st.caption(f"Characters: {char_count}/160" + (" *(multi-part)*" if char_count > 160 else ""))
        sms_run = st.button("Analyse message", type="primary", key="sms_run")

    with sms_right:
        if sms_run:
            if not sms_text.strip():
                st.warning("Enter a message first.")
            else:
                prob = predict_message(sms_model, sms_text)
                label = verdict(prob)

                st.subheader(label)
                st.progress(min(max(prob, 0.0), 1.0), text=f"Spam probability: {prob:.1%}")

                spam_words, ham_words = top_contributing_words_message(sms_model, sms_text)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Pushed toward spam**")
                    if spam_words:
                        st.dataframe(
                            pd.DataFrame(spam_words, columns=["word", "weight"]),
                            hide_index=True,
                        )
                    else:
                        st.write("-")
                with c2:
                    st.markdown("**Pushed toward safe**")
                    if ham_words:
                        st.dataframe(
                            pd.DataFrame(ham_words, columns=["word", "weight"]),
                            hide_index=True,
                        )
                    else:
                        st.write("-")
        else:
            st.info("Type or paste a message and click **Analyse message**.")

    st.divider()
    st.subheader("Batch SMS check")
    st.caption("Upload a CSV with a `text` column to score many messages at once.")
    sms_up = st.file_uploader("CSV file", type="csv", key="sms_batch")
    if sms_up is not None:
        sms_batch = pd.read_csv(sms_up)
        if "text" not in sms_batch.columns:
            st.error("The CSV needs a column named `text`.")
        else:
            sms_probs = sms_model.predict_proba(
                sms_batch["text"].astype(str).map(clean_text)
            )[:, 1]
            sms_batch["spam_probability"] = sms_probs.round(4)
            sms_batch["verdict"] = [verdict(p) for p in sms_probs]
            st.dataframe(sms_batch, use_container_width=True)
            st.download_button(
                "Download results",
                sms_batch.to_csv(index=False).encode("utf-8"),
                "sms_spam_predictions.csv",
                "text/csv",
                key="sms_download",
            )


# ═══════════════════════════════════════════════════════════ PERFORMANCE TAB ═
with tab_perf:
    perf_email, perf_sms = st.tabs(["Email model", "SMS model"])

    with perf_email:
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text())
            meta = metrics.pop("_meta", {})
            table = pd.DataFrame(metrics).T[
                ["accuracy", "precision", "recall", "f1", "cv_f1_mean"]
            ].round(4)
            table.columns = ["Accuracy", "Precision", "Recall", "F1", "5-fold CV F1"]
            st.dataframe(table, use_container_width=True)
            st.caption(
                f"Deployed: {meta.get('deployed_model', '?')} | "
                f"Test set: {meta.get('n_test', '?')} emails."
            )
        if CM_PATH.exists():
            st.image(str(CM_PATH), width=420)
        st.info(
            "Scores come from an older benchmark dataset. Real-world accuracy on "
            "modern phishing and promotional email will be lower."
        )

    with perf_sms:
        if SMS_METRICS_PATH.exists():
            sms_metrics = json.loads(SMS_METRICS_PATH.read_text())
            sms_meta = sms_metrics.pop("_meta", {})
            sms_table = pd.DataFrame(sms_metrics).T[
                ["accuracy", "precision", "recall", "f1", "cv_f1_mean"]
            ].round(4)
            sms_table.columns = ["Accuracy", "Precision", "Recall", "F1", "5-fold CV F1"]
            st.dataframe(sms_table, use_container_width=True)
            st.caption(
                f"Deployed: {sms_meta.get('deployed_model', '?')} | "
                f"Dataset: {sms_meta.get('dataset', 'SMS Spam Collection')} | "
                f"Test set: {sms_meta.get('n_test', '?')} messages."
            )
        if SMS_CM_PATH.exists():
            st.image(str(SMS_CM_PATH), width=420)
        else:
            st.info("Run `py train_sms.py` to generate the SMS model and its metrics.")
        st.info(
            "SMS model trained on the UCI SMS Spam Collection dataset (5,574 messages). "
            "Accuracy may vary on modern messaging platforms."
        )
