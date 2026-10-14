# 🛡️ Spam Detector — Email & SMS

> A machine-learning spam classifier with an interactive Streamlit web app.  
> Detects spam in both **emails** and **SMS messages** using TF-IDF + Logistic Regression, with word-level explainability.

---

## ✨ Features

- 📧 **Email spam detection** — single email (subject + body) or batch CSV upload
- 💬 **SMS spam detection** — single message or batch CSV upload
- 🔍 **Explainability** — top words pushing a prediction toward spam vs. safe
- 📊 **Model performance dashboard** — accuracy, precision, recall, F1, confusion matrix
- ⚡ **Live character counter** for SMS (flags multi-part messages > 160 chars)
- 📥 **Batch download** — export scored results as CSV

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models

**Email model** (requires `data/emails.csv` with `text` and `spam` columns):
```bash
py train.py
```

**SMS model** (uses the included `data/spam.csv`):
```bash
py train_sms.py
```

### 3. Launch the app
```bash
streamlit run app.py
```

---

## 📊 Model Results

### Email Classifier — held-out test set (1,139 emails)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** ✅ deployed | 0.990 | 0.971 | 0.989 | 0.980 |
| Naive Bayes | 0.994 | 0.989 | 0.985 | 0.987 |
| Linear SVM | 0.994 | 0.985 | 0.989 | 0.987 |

### SMS Classifier — held-out test set (1,034 messages)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** ✅ deployed | 0.977 | 0.921 | 0.893 | 0.907 |
| Naive Bayes | 0.983 | 0.975 | 0.885 | 0.928 |
| Linear SVM | 0.980 | 0.930 | 0.908 | 0.919 |

> Logistic Regression is deployed for both because its word weights let the app explain *why* a message was flagged.

---

## 🗂️ Project Structure

```
spam-detector/
├── data/
│   ├── emails.csv          # Email dataset (text, spam columns)
│   └── spam.csv            # SMS Spam Collection dataset (v1, v2 columns)
├── models/
│   ├── spam_model.joblib         # Trained email classifier
│   ├── sms_spam_model.joblib     # Trained SMS classifier
│   ├── metrics.json              # Email model metrics
│   ├── sms_metrics.json          # SMS model metrics
│   ├── confusion_matrix.png      # Email confusion matrix
│   └── sms_confusion_matrix.png  # SMS confusion matrix
├── app.py                  # Streamlit web app (3 tabs)
├── spam_utils.py           # Text cleaning, prediction, explainability helpers
├── train.py                # Train & evaluate email classifier
├── train_sms.py            # Train & evaluate SMS classifier
└── requirements.txt
```

---

## 🧠 How It Works

1. **Text cleaning** (`spam_utils.py`) — strips HTML, replaces URLs with `urltoken` and emails with `emailtoken`, normalises whitespace
2. **TF-IDF vectorisation** — unigrams + bigrams, up to 30k features, sublinear TF scaling
3. **Logistic Regression** — trained with balanced class weights to handle the spam/ham imbalance
4. **Explainability** — contribution of each word = TF-IDF value × classifier coefficient; top positive contributors push toward spam, top negative push toward safe

---

## 📦 Datasets

| Dataset | Source | Size |
|---|---|---|
| Email | [Spam Email Dataset — Kaggle](https://www.kaggle.com/datasets/jackksoncsie/spam-email-dataset) | ~5,700 labelled emails |
| SMS | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | 5,574 labelled messages |

---

## ⚠️ Limitations

- Email model trained on an older benchmark; accuracy on modern phishing and promotional mail will be lower.
- SMS model performs well on the UCI dataset but may struggle with newer slang or platform-specific spam patterns.
- Both models are English-language only.

---

## 🗺️ Roadmap

- [x] Phase 1 — Data cleaning, model comparison, evaluation
- [x] Phase 2 — Web app (email + SMS, batch CSV, explainability)
- [ ] Phase 3 — Gmail API integration
