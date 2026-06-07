# 📧 Email Spam Detection System using Naive Bayes

## 📌 Project Overview

This project is a Machine Learning based Email Spam Detection System that classifies emails as **Spam** or **Not Spam (Ham)** using the SpamBase dataset and the Naive Bayes algorithm.

The system analyzes email content, extracts spam-related features, and predicts whether the email is legitimate or spam. A Streamlit web application is provided for interactive testing and demonstration.

---

## 🎯 Objective

The primary objective of this project is to automatically identify unwanted spam emails using Machine Learning techniques and provide a simple user-friendly interface for prediction.

---

## 🚀 Live Demo

### Streamlit Application

```text
[Spam_Email_Detector](https://p8spamemaildetector.streamlit.app/)
```

### YouTube Demonstration

```text
https://youtube.com/your-demo-video-link
```

---

## 📂 Project Structure

```text
spam_email_detector/
│
├── data/
│   ├── spambase.data
│   ├── spambase.names
│   └── spambase.csv
│
├── models/
│   ├── spam_model.pkl
│   └── model_info.pkl
│
├── create_dataset.py
├── train_model.py
├── analyze_features.py
├── feature_extractor.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Information

Dataset Used: SpamBase Dataset

* Total Samples: 4601 Emails
* Features: 57
* Target Classes:

  * 1 → Spam
  * 0 → Not Spam

The dataset contains word frequency features, character frequency features, and capital letter statistics extracted from email messages.

Examples of features:

* word_freq_free
* word_freq_money
* word_freq_credit
* char_freq_dollar
* char_freq_exclamation
* capital_run_length_total

---

## 🤖 Machine Learning Algorithms Evaluated

Two variants of Naive Bayes were evaluated:

### Gaussian Naive Bayes

* Accuracy: 83.39%
* Precision: 71.78%
* Recall: 95.32%
* F1 Score: 81.89%

### Bernoulli Naive Bayes

* Accuracy: 87.62%
* Precision: 87.16%
* Recall: 80.44%
* F1 Score: 83.67%

### Best Model Selected

**Bernoulli Naive Bayes**

Final Accuracy:

```text
87.62%
```

The Bernoulli Naive Bayes model achieved better overall performance and was selected as the final model for deployment.

---

## 🔍 Feature Analysis

Feature analysis revealed that spam emails frequently contain:

* FREE
* MONEY
* CREDIT
* REMOVE
* BUSINESS
* EMAIL

Special characters such as:

* !
* $

and excessive use of uppercase letters were also strong spam indicators.

Top spam-related features identified during analysis:

* capital_run_length_total
* capital_run_length_longest
* capital_run_length_average
* word_freq_you
* word_freq_your
* word_freq_free
* word_freq_remove
* word_freq_money
* word_freq_credit
* char_freq_dollar

---

## ⚙️ Project Workflow

```text
SpamBase Dataset
        │
        ▼
Dataset Preparation
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Feature Analysis
        │
        ▼
Train/Test Split
        │
        ▼
Model Comparison
(GaussianNB vs BernoulliNB)
        │
        ▼
Best Model Selection
        │
        ▼
Model Serialization
(.pkl)
        │
        ▼
Feature Extraction
        │
        ▼
Streamlit Deployment
        │
        ▼
Spam Prediction
```

---

## 🧠 How Prediction Works

1. User enters email content into the Streamlit application.
2. The feature extraction module analyzes the text.
3. SpamBase-style features are generated.
4. The trained Bernoulli Naive Bayes model receives these features.
5. The model predicts:

   * Spam
   * Not Spam
6. Prediction probabilities are displayed to the user.

---

## 🛠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Streamlit

---

## 📈 Model Performance

| Metric    | Bernoulli Naive Bayes |
| --------- | --------------------- |
| Accuracy  | 87.62%                |
| Precision | 87.16%                |
| Recall    | 80.44%                |
| F1 Score  | 83.67%                |

Confusion Matrix:

```text
[[515  43]
 [ 71 292]]
```

---


## 💡 Key Learning Outcomes

This project demonstrates:

* Classification using Naive Bayes
* Model comparison and evaluation
* Feature engineering
* Exploratory Data Analysis (EDA)
* Model serialization using Joblib
* Streamlit deployment
* End-to-end Machine Learning workflow

---

## ⚠️ Limitations

Although the Bernoulli Naive Bayes model achieved 87.62% accuracy on the SpamBase dataset, there are several limitations:

### 1. Approximate Feature Extraction

The original SpamBase dataset contains pre-engineered features generated using specific email-processing methods.

During deployment, raw email text is converted into features using a custom feature extraction module.

Because this process only approximates the original SpamBase feature generation technique, real-world prediction accuracy may differ from training accuracy.

---

### 2. Limited Vocabulary

The system primarily focuses on words and patterns present in the SpamBase dataset.

New spam tactics, slang, or previously unseen promotional content may not be detected effectively.

---

### 3. Dataset Age

The SpamBase dataset is relatively old compared to modern email communication.

Current spam messages often contain sophisticated phishing techniques and language patterns not represented in the dataset.

---

### 4. Naive Bayes Assumption

Naive Bayes assumes that all features are independent of one another.

In real-world emails, many features are related, which can limit prediction performance.

---

### 5. Not Production Ready

This project is intended for educational purposes, learning machine learning concepts, and demonstrating deployment workflows.

Production-grade spam filtering systems typically use advanced NLP techniques, larger datasets, ensemble methods, and deep learning models.

---

## 👨‍💻 Author

Manan Kohli

---

## ⭐ Future Improvements

* TF-IDF based text processing
* NLP preprocessing pipeline
* Advanced feature engineering
* Random Forest comparison
* XGBoost comparison
* Deep Learning models
* Real-time email integration
* Improved feature extraction pipeline
* Modern dashboard analytics
