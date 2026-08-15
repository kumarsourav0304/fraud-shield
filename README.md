# 🛡️ Explainable Real-Time Fraud Shield

**SOA IDEATHON 2026 — Problem Statement S40 (Fintech)**
Real-time UPI fraud detection for payments, voice phishing, and social engineering — with a plain-language reason for every decision.

🔗 **Live demo:** https://fraud-shield-b1h3.onrender.com

*(Free hosting — the first load after a while may take ~50 seconds to wake up, then it's fast.)*

---

## The problem

UPI scams cost India over ₹1,000 crore a year. The dangerous ones happen on a phone call — a fake "bank official" pressures the victim to share an OTP or make an "urgent" payment. The victim then approves a transaction that looks completely normal to any payment-only fraud check.

Most fraud systems are also a black box: the user just sees "declined" with no reason, so real scams and genuine urgent payments get treated the same.

## What this does

The Fraud Shield combines **two explainable detection layers** into one decision:

1. **Payment-behaviour engine** — learns each user's normal (usual devices, payees, cities, amounts) and flags deviations with a weighted risk score.
2. **Voice-phishing engine** — reads the call transcript for coercion tactics (authority, urgency, secrecy, OTP requests, threats) and quotes the exact words heard.

Fused, a normal-looking payment made right after a coercive call escalates to **BLOCK** instead of slipping through — while the human always confirms or overrides, and every decision is logged.

## Key result

On a synthetic dataset of 600 transactions: **94% precision, 94% recall, 0.7% false-positive rate** — computed live in the app, not hard-coded. The data is deliberately varied so some scams look almost normal; a perfect score would be a red flag.

## The three screens

| Screen | URL | Purpose |
|--------|-----|---------|
| Payment shield | `/` | User-facing — verdict + reasons + confirm/override |
| Live performance | `/stats.html` | Precision/recall computed live |
| Bank review console | `/review.html` | Analysts review flagged payments & overrides |

## Tech stack

Python · FastAPI · rule-based risk scoring · transcript analysis · plain HTML/CSS/JS · CSV audit trail. Runs fully offline on a laptop.

## Run it locally

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python generate_data.py           # first time only, creates the dataset
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000**

## Try the key demo

1. Payment **₹2,000 to a new payee** + **No call** → APPROVE.
2. Same payment + **Scam call** → BLOCK.

The payment didn't change — the coercive call flips the decision. That's the point: it catches the scam *conversation*, not just the transaction.

## Responsible design

Runs on-device (no raw call audio or personal data leaves the machine) · uses only behavioural signals · human always in control · every decision auditable.

## Roadmap

Real transaction feeds and on-device speech-to-text · adaptive per-bank thresholds · a pilot integration with a partner bank.