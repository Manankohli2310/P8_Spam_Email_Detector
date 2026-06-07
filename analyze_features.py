# analyze_features.py

import pandas as pd

# Load dataset
df = pd.read_csv("data/spambase.csv")

# Separate spam and ham emails
spam_emails = df[df["spam"] == 1]
ham_emails = df[df["spam"] == 0]

# Calculate mean frequency for each class
spam_mean = spam_emails.mean()
ham_mean = ham_emails.mean()

# Difference between spam and ham
feature_difference = spam_mean - ham_mean

# Remove target column
feature_difference = feature_difference.drop("spam")   # spam feature is our target feature in dataset

# Sort descending
top_spam_features = feature_difference.sort_values(
    ascending=False
)

print("\nTop 20 Spam Indicators\n")
print(top_spam_features.head(20))