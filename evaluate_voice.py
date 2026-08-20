"""
evaluate_voice.py
Benchmark the voice phishing detection system against the adversarial corpus.

Reports:
  - Precision, Recall, F1, FPR, FNR
  - Average latency per sample
  - Breakdown by detection mode: baseline (rules only) vs. hybrid (rules + NLP)
  - Per-class detection rates

All metrics are computed from the actual corpus and current implementation.
Nothing is hardcoded.
"""

import csv
import os
import time

from text_normalizer import normalize
from voice_phishing import analyse_transcript, _match_rules

# Try NLP availability
try:
    from nlp_intent import is_available as nlp_is_available, predict_intent
except ImportError:
    nlp_is_available = lambda: False
    predict_intent = None

CORPUS_PATH = os.path.join("data", "adversarial_corpus.csv")
BENIGN_LABEL = "BENIGN"


def load_corpus():
    """Load the adversarial test corpus."""
    samples = []
    with open(CORPUS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            label = row["label"].strip()
            if text and label:
                samples.append({"text": text, "label": label})
    return samples


def evaluate_baseline(samples):
    """Evaluate using deterministic rules only (no NLP)."""
    tp = fp = tn = fn = 0
    latencies = []

    for s in samples:
        is_scam = s["label"] != BENIGN_LABEL
        started = time.perf_counter()

        normalized = normalize(s["text"])
        fired, score = _match_rules(normalized)

        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)

        flagged = score >= 25  # using WARN_AT threshold

        if flagged and is_scam:     tp += 1
        elif flagged and not is_scam: fp += 1
        elif not flagged and is_scam: fn += 1
        else:                         tn += 1

    return tp, fp, tn, fn, latencies


def evaluate_hybrid(samples):
    """Evaluate using the full hybrid system (rules + NLP + normalization)."""
    tp = fp = tn = fn = 0
    latencies = []
    per_class_results = {}

    for s in samples:
        is_scam = s["label"] != BENIGN_LABEL
        started = time.perf_counter()

        result = analyse_transcript(s["text"])

        elapsed = (time.perf_counter() - started) * 1000
        latencies.append(elapsed)

        # A sample is "flagged" if voice_level is SUSPICIOUS or HIGH_RISK
        flagged = result["voice_level"] in ("SUSPICIOUS", "HIGH_RISK")

        # Apply BENIGN suppression
        if result.get("is_benign"):
            flagged = False

        if flagged and is_scam:       tp += 1
        elif flagged and not is_scam: fp += 1
        elif not flagged and is_scam: fn += 1
        else:                         tn += 1

        # Per-class tracking
        label = s["label"]
        if label not in per_class_results:
            per_class_results[label] = {"correct": 0, "total": 0}
        per_class_results[label]["total"] += 1

        if is_scam and flagged:
            per_class_results[label]["correct"] += 1
        elif not is_scam and not flagged:
            per_class_results[label]["correct"] += 1

    return tp, fp, tn, fn, latencies, per_class_results


def compute_metrics(tp, fp, tn, fn):
    """Compute precision, recall, F1, FPR, FNR."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    return precision, recall, f1, fpr, fnr


def print_metrics(label, tp, fp, tn, fn, latencies):
    """Print formatted metrics."""
    precision, recall, f1, fpr, fnr = compute_metrics(tp, fp, tn, fn)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"{'=' * 50}")
    print(f"  True Positives:   {tp}")
    print(f"  False Positives:  {fp}")
    print(f"  True Negatives:   {tn}")
    print(f"  False Negatives:  {fn}")
    print(f"")
    print(f"  Precision:        {precision:.2%}")
    print(f"  Recall:           {recall:.2%}")
    print(f"  F1 Score:         {f1:.2%}")
    print(f"  FPR:              {fpr:.2%}")
    print(f"  FNR:              {fnr:.2%}")
    print(f"  Avg Latency:      {avg_latency:.2f} ms")


def main():
    print("=" * 60)
    print("  FraudShield Voice Detection Benchmark")
    print("  Adversarial corpus evaluation")
    print("=" * 60)

    if not os.path.exists(CORPUS_PATH):
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        return

    samples = load_corpus()
    total = len(samples)
    scam_count = sum(1 for s in samples if s["label"] != BENIGN_LABEL)
    benign_count = total - scam_count

    print(f"\nCorpus: {total} samples ({scam_count} scam, {benign_count} benign)")

    # --- Baseline: rules only ---
    tp, fp, tn, fn, lat = evaluate_baseline(samples)
    print_metrics("BASELINE (Deterministic Rules Only)", tp, fp, tn, fn, lat)

    # --- Hybrid: rules + NLP ---
    tp, fp, tn, fn, lat, per_class = evaluate_hybrid(samples)
    print_metrics("HYBRID (Rules + NLP + Normalization)", tp, fp, tn, fn, lat)

    nlp_status = "available" if nlp_is_available() else "NOT available (fallback to rules)"
    print(f"\n  NLP model: {nlp_status}")

    # Per-class breakdown
    print(f"\n{'=' * 50}")
    print("  Per-class accuracy")
    print(f"{'=' * 50}")
    for cls in sorted(per_class.keys()):
        info = per_class[cls]
        acc = info["correct"] / info["total"] if info["total"] > 0 else 0
        print(f"  {cls:20s}: {info['correct']:2d}/{info['total']:2d} ({acc:.0%})")

    # Historical reference (from PDF — NOT forced)
    print(f"\n{'=' * 50}")
    print("  Historical reference values (PDF, for comparison)")
    print(f"{'=' * 50}")
    print("  Precision: 92.50%")
    print("  Recall:    74.00%")
    print("  F1:        82.22%")
    print("  FPR:        6.00%")
    print("  FNR:       26.00%")
    print("  Avg latency: 20.50 ms")
    print("\n  Note: These are reference values from an earlier evaluation.")
    print("  The actual results above are from the CURRENT implementation")
    print("  against the CURRENT adversarial corpus.")

    print("\n✓ Benchmark complete.")


if __name__ == "__main__":
    main()
