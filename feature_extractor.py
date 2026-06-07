# feature_extractor.py

import pandas as pd
import re

# ==========================================
# SPAMBASE FEATURES
# ==========================================

FEATURE_COLUMNS = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",

    "char_freq_semicolon",
    "char_freq_left_paren",
    "char_freq_left_bracket",
    "char_freq_exclamation",
    "char_freq_dollar",
    "char_freq_hash",

    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total"
]


# ==========================================
# FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_features(email_text):

    email_lower = email_text.lower()

    words = re.findall(r"\b\w+\b", email_lower)

    total_words = max(len(words), 1)

    features = {}

    # ======================================
    # WORD FREQUENCIES
    # ======================================

    spam_words = [
        "make", "address", "all", "3d", "our",
        "over", "remove", "internet", "order",
        "mail", "receive", "will", "people",
        "report", "addresses", "free",
        "business", "email", "you", "credit",
        "your", "font", "000", "money",
        "hp", "hpl", "george", "650",
        "lab", "labs", "telnet", "857",
        "data", "415", "85", "technology",
        "1999", "parts", "pm", "direct",
        "cs", "meeting", "original",
        "project", "re", "edu",
        "table", "conference"
    ]

    for word in spam_words:

        count = words.count(word)

        percentage = (count / total_words) * 100

        features[f"word_freq_{word}"] = percentage

    # ======================================
    # CHARACTER FREQUENCIES
    # ======================================

    total_chars = max(len(email_text), 1)

    features["char_freq_semicolon"] = (
        email_text.count(";") / total_chars
    ) * 100

    features["char_freq_left_paren"] = (
        email_text.count("(") / total_chars
    ) * 100

    features["char_freq_left_bracket"] = (
        email_text.count("[") / total_chars
    ) * 100

    features["char_freq_exclamation"] = (
        email_text.count("!") / total_chars
    ) * 100

    features["char_freq_dollar"] = (
        email_text.count("$") / total_chars
    ) * 100

    features["char_freq_hash"] = (
        email_text.count("#") / total_chars
    ) * 100

    # ======================================
    # CAPITAL LETTER STATISTICS
    # ======================================

    capital_runs = re.findall(
        r"[A-Z]+",
        email_text
    )

    if capital_runs:

        lengths = [len(run) for run in capital_runs]

        features["capital_run_length_average"] = (
            sum(lengths) / len(lengths)
        )

        features["capital_run_length_longest"] = max(lengths)

        features["capital_run_length_total"] = sum(lengths)

    else:

        features["capital_run_length_average"] = 0
        features["capital_run_length_longest"] = 0
        features["capital_run_length_total"] = 0

    # ======================================
    # ENSURE ALL FEATURES EXIST
    # ======================================

    for col in FEATURE_COLUMNS:

        if col not in features:

            features[col] = 0

    # Preserve training column order
    df = pd.DataFrame(
        [features]
    )

    df = df[FEATURE_COLUMNS]

    return df