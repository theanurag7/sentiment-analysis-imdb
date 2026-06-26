"""
save_model.py
--------------
Run this ONCE to train the Voting Classifier on imdb_dataset.csv
and save the trained model + TF-IDF vectorizer to disk.

After running this, predict_only.py can load the saved files
instantly without retraining.

Usage:
    python save_model.py
"""

import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV


def clean_text(text):
    """Remove HTML tags, special characters, lowercase. (Same as training script)"""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def main():
    print("[1] Loading dataset...")
    df = pd.read_csv("imdb_dataset.csv")

    print("[2] Cleaning text...")
    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    X = df["clean_review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[3] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("[4] Training Voting Classifier (NB + LogReg + Calibrated SVM)...")
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svm_calibrated = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))

    voting_clf = VotingClassifier(
        estimators=[
            ("naive_bayes", nb),
            ("logistic_regression", lr),
            ("svm_calibrated", svm_calibrated),
        ],
        voting="soft"
    )

    voting_clf.fit(X_train_tfidf, y_train)

    print("[5] Saving model and vectorizer to disk...")
    joblib.dump(voting_clf, "sentiment_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    print("\nDone! Created:")
    print("  - sentiment_model.pkl")
    print("  - vectorizer.pkl")
    print("\nThese will be loaded by predict_only.py")


if __name__ == "__main__":
    main()