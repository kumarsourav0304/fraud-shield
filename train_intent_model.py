"""
train_intent_model.py
Trains a lightweight TF-IDF + Calibrated Linear SVM intent classifier.

Reads the annotated corpus from data/intent_corpus.csv, normalizes each
sample through text_normalizer, trains a TfidfVectorizer + CalibratedClassifierCV
pipeline, prints a classification report + confusion matrix, and persists the
pipeline to models/intent_pipeline.joblib.

This script is run ONCE (or when the corpus is updated). The runtime
application loads the persisted artifact — it never retrains.
"""

import os
import csv
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from text_normalizer import normalize

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
CORPUS_PATH = os.path.join("data", "intent_corpus.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_pipeline.joblib")

INTENT_CLASSES = [
    "BENIGN", "AUTHORITY", "CREDENTIAL", "THREAT", "URGENCY",
    "PAYMENT_COERCION", "SECRECY", "REMOTE_ACCESS", "DIGITAL_ARREST",
]


def load_corpus(path):
    """Load the annotated corpus CSV and normalize all text."""
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_text = row["text"].strip()
            label = row["label"].strip()
            if raw_text and label:
                texts.append(normalize(raw_text))
                labels.append(label)
    return texts, labels


def main():
    print("=" * 60)
    print("FraudShield NLP Intent Model Training")
    print("=" * 60)

    # --- Load and validate ---
    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    texts, labels = load_corpus(CORPUS_PATH)
    print(f"\nLoaded {len(texts)} samples from {CORPUS_PATH}")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    print("\nClass distribution:")
    for cls in INTENT_CLASSES:
        count = dist.get(cls, 0)
        print(f"  {cls:20s}: {count:3d} samples")

    unknown = set(labels) - set(INTENT_CLASSES)
    if unknown:
        print(f"\nWARNING: Unknown labels found: {unknown}")

    # --- Build pipeline ---
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),          # unigrams + bigrams + trigrams
            max_features=8000,
            sublinear_tf=True,           # better for text classification
            min_df=1,
            max_df=0.95,
        )),
        ("svm", CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",   # handle class imbalance
            ),
            cv=3,                         # calibration with 3-fold CV
            method="sigmoid",
        )),
    ])

    # --- Cross-validated evaluation ---
    print("\n--- Cross-validated evaluation (5-fold stratified) ---")
    labels_arr = np.array(labels)

    # Only do cross-validation if we have enough samples per class
    min_class_size = min(dist.values())
    n_folds = min(5, min_class_size)

    if n_folds >= 2:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)
        print(classification_report(labels, y_pred, labels=INTENT_CLASSES,
                                    zero_division=0))

        print("Confusion Matrix:")
        cm = confusion_matrix(labels, y_pred, labels=INTENT_CLASSES)
        # Header
        print(f"{'':20s}", end="")
        for cls in INTENT_CLASSES:
            print(f"{cls[:6]:>7s}", end="")
        print()
        for i, cls in enumerate(INTENT_CLASSES):
            print(f"{cls:20s}", end="")
            for j in range(len(INTENT_CLASSES)):
                print(f"{cm[i][j]:7d}", end="")
            print()
    else:
        print(f"Skipping cross-validation: smallest class has only {min_class_size} samples")

    # --- Train final model on all data ---
    print("\nTraining final model on all data...")
    pipeline.fit(texts, labels)

    # --- Persist ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH} ({model_size / 1024:.1f} KB)")

    # --- Quick smoke test ---
    print("\n--- Smoke test ---")
    test_cases = [
        "I am calling from your bank your account is frozen",
        "Please share the OTP that was sent to your phone",
        "You are under digital arrest transfer money now",
        "Hey let's go for dinner tonight",
        "Install AnyDesk and share your screen",
        "Digital arrest scams are increasing in India",
        "Abhi payment karo warna account block ho jayega",
    ]
    for text in test_cases:
        normalized = normalize(text)
        probs = pipeline.predict_proba([normalized])[0]
        top_idx = probs.argmax()
        intent = pipeline.classes_[top_idx]
        conf = probs[top_idx]
        print(f"  {text[:60]:60s} -> {intent:20s} ({conf:.0%})")

    print("\n✓ Training complete.")


if __name__ == "__main__":
    main()
