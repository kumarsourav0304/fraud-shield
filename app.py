"""
app.py
The Fraud Shield backend server.
It exposes these actions to the web pages:
  /assess   -> score a transaction (payment + optional call transcript), return decision + reasons
  /decision -> record what the user chose (confirm / cancel) into the audit log
  /audit    -> return the audit log + summary for the bank review console
  /stats    -> compute live precision/recall over the whole dataset
The audit log is a running record for banks to review later.
"""

import csv
import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from risk_engine import build_user_profile, assess_transaction
from voice_phishing import analyse_transcript

app = FastAPI(title="Explainable Fraud Shield")

AUDIT_FILE = "audit_log.csv"


# ---- Load transaction history once, so we can build user profiles ----
def load_profiles():
    """Build a 'normal behaviour' profile for every user from transactions.csv."""
    history_by_user = {}
    with open("transactions.csv", newline="") as f:
        for tx in csv.DictReader(f):
            # only learn from genuine (non-fraud) history for clean profiles
            if int(tx["is_fraud"]) == 0:
                history_by_user.setdefault(tx["user_id"], []).append(tx)
    return {uid: build_user_profile(hist) for uid, hist in history_by_user.items()}


PROFILES = load_profiles()


# ---- Shape of the data the web page will send us ----
class Transaction(BaseModel):
    user_id: str
    amount: float
    payee: str
    city: str
    device: str
    timestamp: str
    transcript: str = ""   # optional call transcript, if a call happened before payment


class Decision(BaseModel):
    user_id: str
    amount: float
    payee: str
    decision: str        # what the shield said: APPROVE / WARN / BLOCK
    score: int
    user_action: str     # what the human chose: CONFIRMED / CANCELLED


# ---- Action 1: assess a transaction (payment behaviour + voice phishing) ----
@app.post("/assess")
def assess(tx: Transaction):
    profile = PROFILES.get(tx.user_id)

    # --- 1. Payment-behaviour risk ---
    if profile is None:
        payment = {"score": 0, "decision": "APPROVE",
                   "reasons": ["No history for this user yet"], "fired_rules": []}
    else:
        payment = assess_transaction(tx.model_dump(), profile)

    # --- 2. Voice-phishing risk from the call transcript (if any) ---
    voice = analyse_transcript(tx.transcript)

    # --- 3. Fuse the two into one final decision ---
    # A coercive call is a strong signal on its own, so we add a share of its
    # score to the payment score. The scam pattern (normal-looking payment right
    # after a coercive call) now escalates instead of slipping through.
    combined = payment["score"] + int(voice["voice_score"] * 0.6)
    combined = min(combined, 100)

    reasons = list(payment["reasons"]) if payment["reasons"] else []
    for sig in voice["voice_signals"]:
        reasons.append(f'Call red flag: {sig["reason"]} (heard: "{sig["evidence"]}")')

    if combined >= 60:
        decision = "BLOCK"
    elif combined >= 30:
        decision = "WARN"
    else:
        decision = "APPROVE"

    # If nothing at all fired, say so plainly
    if not reasons:
        reasons = ["No suspicious signals detected"]

    return {
        "score": combined,
        "decision": decision,
        "reasons": reasons,
        "payment_score": payment["score"],
        "voice_score": voice["voice_score"],
        "voice_level": voice["voice_level"],
    }


# ---- Action 2: record the human's decision into the audit log ----
@app.post("/decision")
def record_decision(d: Decision):
    file_exists = os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["logged_at", "user_id", "amount", "payee",
                             "shield_decision", "score", "user_action"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                         d.user_id, d.amount, d.payee, d.decision, d.score, d.user_action])
    return {"status": "logged"}


# ---- Action 3: return the audit log for the bank review dashboard ----
@app.get("/audit")
def get_audit():
    rows = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
    # newest first
    rows.reverse()

    total = len(rows)
    blocked = sum(1 for r in rows if r["shield_decision"] == "BLOCK")
    warned = sum(1 for r in rows if r["shield_decision"] == "WARN")
    # "overrides": the shield warned/blocked but the user confirmed anyway
    overrides = sum(1 for r in rows
                    if r["shield_decision"] in ("WARN", "BLOCK") and r["user_action"] == "CONFIRMED")

    return {
        "summary": {
            "total": total,
            "blocked": blocked,
            "warned": warned,
            "overrides": overrides,
        },
        "rows": rows,
    }


# ---- Action 4: compute live detection metrics over the whole dataset ----
@app.get("/stats")
def get_stats():
    """
    Run the payment engine over every transaction in transactions.csv and
    report precision / recall against the known fraud labels — computed live,
    so the numbers shown in the UI are real, not hard-coded.
    """
    with open("transactions.csv", newline="") as f:
        all_tx = list(csv.DictReader(f))

    history_by_user = {}
    tp = fp = tn = fn = 0

    for tx in all_tx:
        uid = tx["user_id"]
        past = history_by_user.get(uid, [])

        if len(past) < 3:
            decision = "APPROVE"
        else:
            profile = build_user_profile(past)
            decision = assess_transaction(tx, profile)["decision"]

        flagged = decision in ("WARN", "BLOCK")
        fraud = int(tx["is_fraud"]) == 1
        if flagged and fraud:       tp += 1
        elif flagged and not fraud: fp += 1
        elif not flagged and fraud: fn += 1
        else:                       tn += 1

        if decision == "APPROVE":
            history_by_user.setdefault(uid, []).append(tx)

    total = tp + fp + tn + fn
    precision = round(tp / (tp + fp) * 100) if (tp + fp) else 0
    recall = round(tp / (tp + fn) * 100) if (tp + fn) else 0
    fp_rate = round(fp / (fp + tn) * 100, 1) if (fp + tn) else 0
    fraud_total = sum(int(t["is_fraud"]) for t in all_tx)

    return {
        "total": total,
        "fraud_total": fraud_total,
        "normal_total": total - fraud_total,
        "caught": tp,
        "missed": fn,
        "false_alarms": fp,
        "cleared": tn,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": fp_rate,
    }


# ---- Serve the web pages ----
app.mount("/", StaticFiles(directory="static", html=True), name="static")