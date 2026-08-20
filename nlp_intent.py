"""
nlp_intent.py
Runtime NLP intent prediction for the Fraud Shield.

Loads a pre-trained TF-IDF + Calibrated Linear SVM pipeline from
models/intent_pipeline.joblib and predicts the intent class of a
normalized transcript.

If the model artifact is missing or fails to load, the module
degrades gracefully — it never crashes the application.
"""

import os
import threading

import text_normalizer

# ---------------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "intent_pipeline.joblib")

_pipeline = None
_load_error = None
_lock = threading.Lock()


def _load_model():
    """Lazy-load the model once, on first use."""
    global _pipeline, _load_error

    if _pipeline is not None or _load_error is not None:
        return _pipeline

    with _lock:
        if _pipeline is not None or _load_error is not None:
            return _pipeline
        try:
            import joblib
            if not os.path.exists(MODEL_PATH):
                _load_error = f"Model file not found: {MODEL_PATH}"
                return None
            _pipeline = joblib.load(MODEL_PATH)
        except Exception as exc:
            _load_error = str(exc)
            _pipeline = None
    return _pipeline


def is_available():
    """True if the NLP model can produce predictions right now."""
    return _load_model() is not None


def get_error():
    """If the model failed to load, return the reason."""
    _load_model()
    return _load_error


def predict_intent(text: str) -> dict:
    """
    Predict the intent class of a transcript.

    Parameters:
        text: The transcript text (ideally already normalized, but
              normalization is applied here as a safety net).

    Returns:
        {
            "intent": str,        # e.g. "DIGITAL_ARREST" or "BENIGN"
            "confidence": float,  # calibrated probability 0-1
            "all_probs": dict,    # {class: probability} for all classes
        }

    If the model is unavailable, returns:
        {"intent": "UNAVAILABLE", "confidence": 0.0, "all_probs": {}}
    """
    pipeline = _load_model()

    if pipeline is None:
        return {
            "intent": "UNAVAILABLE",
            "confidence": 0.0,
            "all_probs": {},
        }

    try:
        # Normalize as safety net (caller should already normalize)
        normalized = text_normalizer.normalize(text) if text else ""
        if not normalized:
            return {
                "intent": "BENIGN",
                "confidence": 1.0,
                "all_probs": {"BENIGN": 1.0},
            }

        probs = pipeline.predict_proba([normalized])[0]
        classes = pipeline.classes_
        all_probs = {cls: round(float(p), 4) for cls, p in zip(classes, probs)}

        top_idx = probs.argmax()
        intent = classes[top_idx]
        confidence = float(probs[top_idx])

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_probs": all_probs,
        }

    except Exception:
        return {
            "intent": "UNAVAILABLE",
            "confidence": 0.0,
            "all_probs": {},
        }


# Quick self-test
if __name__ == "__main__":
    print(f"Model available: {is_available()}")
    if not is_available():
        print(f"Error: {get_error()}")
        print("Run: python train_intent_model.py")
    else:
        tests = [
            "I am calling from your bank",
            "Share the OTP now",
            "You are under digital arrest",
            "Hey let's go for lunch",
            "Install AnyDesk on your phone",
            "Digital arrest scams are increasing in India",
        ]
        for t in tests:
            r = predict_intent(t)
            print(f"  {t:50s} -> {r['intent']:20s} ({r['confidence']:.0%})")
