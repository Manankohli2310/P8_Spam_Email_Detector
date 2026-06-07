# app.py

import streamlit as st
import joblib

from feature_extractor import extract_features

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("models/spam_model.pkl")

model = load_model()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("📊 Model Information")

    st.success("Best Model Selected")

    st.write("**Algorithm:** Bernoulli Naive Bayes")

    st.write("**Accuracy:** 87.62%")

    st.write("**Dataset:** SpamBase")

    st.write("**Features:** 57")

    st.markdown("---")

    st.subheader("📌 Project Workflow")

    st.write("""
    1. User enters email text
    2. Features are extracted
    3. BernoulliNB model predicts
    4. Spam probability displayed
    """)

    st.markdown("---")

    st.subheader("⚠ Common Spam Indicators")

    st.write("""
    - FREE
    - MONEY
    - CREDIT
    - REMOVE
    - !!!
    - $$$
    - Excessive CAPITAL LETTERS
    """)

# ==========================================
# MAIN TITLE
# ==========================================

st.title("📧 Spam Email Detection System")

st.markdown("""
Detect whether an email is **Spam** or **Not Spam**
using a machine learning model trained on the
SpamBase dataset.
""")

st.markdown("---")

# ==========================================
# EMAIL INPUT
# ==========================================

email_text = st.text_area(
    "Paste Email Content Below",
    height=250,
    placeholder="""
Example:

CONGRATULATIONS!!!

You have won FREE MONEY.

Claim your reward now.
"""
)

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔍 Analyze Email", use_container_width=True):

    if not email_text.strip():

        st.warning("Please enter email content.")

    else:

        # ==============================
        # FEATURE EXTRACTION
        # ==============================

        features = extract_features(email_text)

        # ==============================
        # PREDICTION
        # ==============================

        prediction = model.predict(features)[0]

        probability = model.predict_proba(features)[0]

        not_spam_prob = probability[0] * 100
        spam_prob = probability[1] * 100

        st.markdown("---")

        # ==============================
        # RESULT
        # ==============================

        if prediction == 1:

            st.error("🚨 Spam Email Detected")

        else:

            st.success("✅ Legitimate Email")

        # ==============================
        # PROBABILITIES
        # ==============================

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Spam Probability",
                f"{spam_prob:.2f}%"
            )

        with col2:

            st.metric(
                "Not Spam Probability",
                f"{not_spam_prob:.2f}%"
            )

        # ==============================
        # CONFIDENCE BAR
        # ==============================

        st.subheader("Prediction Confidence")

        st.progress(float(spam_prob / 100))

        # ==============================
        # EXTRACTED FEATURES
        # ==============================

        st.markdown("---")

        st.subheader("Top Extracted Feature Values")

        important_features = [
            "word_freq_free",
            "word_freq_money",
            "word_freq_credit",
            "word_freq_you",
            "word_freq_your",
            "char_freq_exclamation",
            "char_freq_dollar",
            "capital_run_length_total"
        ]

        display_df = features[
            important_features
        ].T

        display_df.columns = ["Value"]

        st.dataframe(
            display_df,
            use_container_width=True
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "Built using Streamlit, Scikit-Learn, Bernoulli Naive Bayes, and the SpamBase Dataset."
)