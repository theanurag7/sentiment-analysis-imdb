"""
Sentiment Analysis Using Machine Learning
Group Project- MACHINE LEARNING CONCEPT-2 (CSE 3968)
SIKSHA ’O’ ANUSANDHAN (Deemed University)

Models included:
1. Naive Bayes (MultinomialNB)
2. Logistic Regression
3. SVM (LinearSVC)
4. Random Forest
5. Voting Classifier (Ensemble)
"""

# 1. IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# 2. LOAD DATASET
def load_data(filepath="imdb_dataset.csv"):
    """Load IMDB dataset from a CSV file."""
    print("[1] Loading dataset...")
    df = pd.read_csv(filepath)

    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nLabel distribution:\n{df['sentiment'].value_counts()}\n")

    return df


# 3. TEXT PREPROCESSING
def clean_text(text):
    """Remove HTML tags, special characters, lowercase."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def preprocess(df):
    """Apply text cleaning and encode labels."""
    print("[2] Preprocessing text...")

    df["clean_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})

    print("\nSample cleaned review:")
    print(df["clean_review"].iloc[0][:100], "\n")

    return df


# 4. FEATURE EXTRACTION (TF-IDF)
def extract_features(df, max_features=10000):
    """Convert text to TF-IDF feature vectors."""
    print("[3] Extracting TF-IDF features...")

    X = df["clean_review"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2)
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"\nTraining samples : {X_train_tfidf.shape[0]}")
    print(f"Test samples     : {X_test_tfidf.shape[0]}")
    print(f"Features         : {X_train_tfidf.shape[1]}\n")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# 5. MODEL TRAINING AND EVALUATION
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train 5 models and compare results."""
    print("[4] Training models...\n")

    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=1000, random_state=42)
    svm = LinearSVC(max_iter=2000, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    svm_calibrated = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=42)
    )

    voting_clf = VotingClassifier(
        estimators=[
            ("naive_bayes", nb),
            ("logistic_regression", lr),
            ("svm_calibrated", svm_calibrated),
        ],
        voting="soft"
    )

    models = {
        "Naive Bayes": nb,
        "Logistic Regression": lr,
        "SVM (LinearSVC)": svm,
        "Random Forest": rf,
        "Voting Classifier": voting_clf,
    }

    results = {}

    for name, model in models.items():
        print(f" Training: {name} ...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results[name] = {
            "model": model,
            "preds": preds,
            "accuracy": acc
        }

        print(f" Accuracy: {acc:.4f}")
        print(classification_report(
            y_test, preds,
            target_names=["Negative", "Positive"]
        ))

    return results


# 6. VISUALISATIONS
def plot_results(results, y_test):
    print("[5] Generating plots...\n")

    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]

    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

    plt.figure(figsize=(11, 4))
    bars = plt.bar(names, accs, color=colors)
    plt.ylim(0.78, 1.0)
    plt.title("Model Accuracy Comparison- IMDB Dataset", fontsize=13)
    plt.ylabel("Accuracy")
    plt.xticks(fontsize=9)

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150)
    plt.show()

    print("\nSaved: accuracy_comparison.png")

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, results[name]["preds"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Neg", "Pos"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name, fontsize=9)

    plt.suptitle("Confusion Matrices- All Models", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()

    print("\nSaved: confusion_matrices.png\n")


# 7. PREDICT NEW REVIEW
def predict_sentiment(text, model, vectorizer):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    label = "Positive" if prediction == 1 else "Negative"

    print(f" Input : {text}")
    print(f" Result : {label}\n")

    return label


# 8. MAIN PIPELINE
if __name__ == "__main__":
    df = load_data("imdb_dataset.csv")
    df = preprocess(df)

    X_train, X_test, y_train, y_test, vectorizer = extract_features(df)

    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    plot_results(results, y_test)

    print("=" * 50)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("=" * 50)

    for name in results:
        print(f"{name:<25} {results[name]['accuracy']:>10.4f}")

    print("=" * 50)

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = results[best_name]["model"]

    print(f"\n[6] Best model: {best_name} "
          f"(Accuracy: {results[best_name]['accuracy']:.4f})\n")

    print("[7] Sample predictions using best model:")

    samples = [
        "This movie was absolutely fantastic! I loved every moment.",
        "Terrible film, complete waste of time. Boring and predictable.",
        "It was okay, not great but not bad either.",
    ]

    for s in samples:
        predict_sentiment(s, best_model, vectorizer)