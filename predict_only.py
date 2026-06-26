"""
predict_only.py
-----------------
Lean prediction script called by SentimentController.cs via subprocess.

Usage (this is exactly how C# calls it):
    python predict_only.py "This movie was great!"

IMPORTANT:
- Prints ONLY "Positive" or "Negative" to stdout (nothing else).
  The C# controller reads the entire stdout and trims it, so any
  extra print() statements here would break the API response.
- Requires sentiment_model.pkl and vectorizer.pkl to already exist
  in the same folder (created by running save_model.py once).
"""

import sys
import re
import joblib
import os

# Suppress sklearn warnings so they don't leak into stdout/stderr noise
import warnings
warnings.filterwarnings("ignore")


def clean_text(text):
    """Same cleaning logic used during training."""
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower().strip()
    return text


def main():
    if len(sys.argv) < 2:
        print("Error")
        sys.exit(1)

    review_text = sys.argv[1]

    # Load model + vectorizer (saved by save_model.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "sentiment_model.pkl")
    vectorizer_path = os.path.join(script_dir, "vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    cleaned = clean_text(review_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]

    label = "Positive" if prediction == 1 else "Negative"

    print(label)


if __name__ == "__main__":
    main()
