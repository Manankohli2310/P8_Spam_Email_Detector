# train_model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import (
    GaussianNB,
    BernoulliNB
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# =====================================
# LOAD DATASET
# =====================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("data/spambase.csv")

print("Dataset Loaded Successfully!")
print(f"Dataset Shape: {df.shape}")

# =====================================
# CHECK MISSING VALUES
# =====================================

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

missing_values = df.isnull().sum().sum()

print(f"Total Missing Values: {missing_values}")

# =====================================
# CLASS DISTRIBUTION
# =====================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["spam"].value_counts()

print(class_counts)

print("\nPercentage Distribution:")

print(
    round(
        class_counts / len(df) * 100,
        2
    )
)

# =====================================
# FEATURES & TARGET
# =====================================

X = df.drop("spam", axis=1)

y = df["spam"]

# =====================================
# TRAIN TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================
# MODELS TO COMPARE
# =====================================

models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB()
}

best_model = None
best_model_name = ""
best_accuracy = 0

# =====================================
# TRAIN & EVALUATE EACH MODEL
# =====================================

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(f"TRAINING : {model_name}")
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    print(f"\nAccuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Track Best Model
    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_model_name = model_name

# =====================================
# BEST MODEL SUMMARY
# =====================================

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Model Name : {best_model_name}")
print(f"Accuracy   : {best_accuracy * 100:.2f}%")

# =====================================
# SAVE BEST MODEL
# =====================================

joblib.dump(
    best_model,
    "models/spam_model.pkl"
)

print("\nBest Model Saved Successfully!")
print("Location : models/spam_model.pkl")

# =====================================
# SAVE MODEL INFO
# =====================================

model_info = {
    "best_model": best_model_name,
    "accuracy": round(best_accuracy * 100, 2)
}

joblib.dump(
    model_info,
    "models/model_info.pkl"
)

print("Model Info Saved Successfully!")