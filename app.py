"""
app.py
The Fraud Shield backend server.

Endpoints:
  /assess       -> score a transaction (payment + optional call transcript)
  /transcribe   -> turn real call audio into text via Whisper
  /capabilities -> tell the UI whether speech-to-text is usable right now
  /decision     -> record what the user chose (confirm / cancel)
  /outcome      -> record the real-world result of a flagged payment
  /audit        -> return the audit log + summary for the bank review console
  /stats        -> compute live precision/recall over the whole dataset
"""

import csv
import os
import time
from datetime import datetime

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import transcription
from risk_engine import (
    build_user_profile,
    assess_transaction,
    merge_external_signals,
    privacy_manifest,
    decision_display,
    WARN_AT,
    BLOCK_AT,
)
from voice_phishing import analyse_transcript

app = FastAPI(title="Explainable Fraud Shield")

AUDIT_FILE = "audit_log.csv"

AUDIT_HEADER = [
    "logged_at", "user_id", "amount", "payee",
    "shield_decision", "score", "user_action",
    "confidence", "signal_count", "outcome",
]

VOICE_WEIGHT = 0.6

VALID_OUTCOMES = {"PENDING", "CONFIRMED_FRAUD", "CONFIRMED_LEGITIMATE", "UNKNOWN"}

# Uploaded audio larger than this is refused before it reaches the model.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# ---------------------------------------------------------------------------
# CSV SAFETY
#
# A spreadsheet treats a cell beginning with = + - or @ as a FORMULA. Since
# the audit log is meant to be opened by a bank reviewer in Excel, a payee
# name could otherwise execute on open. Prefixing a quote forces plain text.
# ---------------------------------------------------------------------------
def csv_safe(value):
    """Neutralise spreadsheet formula injection in a value bound for CSV."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    if text[:1] in ("=", "+", "-", "@"):
        text = "'" + text
    return text


# ---- Load transaction history once, so we can build user profiles ----
def load_profiles():
    """Build a 'normal behaviour' profile for every user from transactions.csv."""
    history_by_user = {}
    with open("transactions.csv", newline="") as f:
        for tx in csv.DictReader(f):
            if int(tx["is_fraud"]) == 0:
                history_by_user.setdefault(tx["user_id"], []).append(tx)
    return {uid: build_user_profile(hist) for uid, hist in history_by_user.items()}


PROFILES = load_profiles()


# ---------------------------------------------------------------------------
# REQUEST SHAPES
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    user_id: str = Field(..., max_length=50)
    amount: float = Field(..., ge=0, le=10_000_000)
    payee: str = Field(..., max_length=200)
    city: str = Field(..., max_length=100)
    device: str = Field(..., max_length=100)
    timestamp: str = Field(..., max_length=40)
    transcript: str = Field("", max_length=5000)
    accessibility_service_active: bool = False


class Decision(BaseModel):
    user_id: str = Field(..., max_length=50)
    amount: float = Field(..., ge=0, le=10_000_000)
    payee: str = Field(..., max_length=200)
    decision: str = Field(..., max_length=20)
    score: int = Field(..., ge=0, le=100)
    user_action: str = Field(..., max_length=20)
    confidence: str = Field("", max_length=20)
    signal_count: int = Field(0, ge=0, le=100)


class Outcome(BaseModel):
    row_index: int = Field(..., ge=0)
    outcome: str = Field(..., max_length=40)


# ---- Turn voice signals into the same decomposed shape as payment rules ----
def voice_signals_as_attribution(voice):
    """
    Split the weighted voice contribution across the individual call red flags,
    so each carries its own point value instead of hiding inside a fused score.
    The TOTAL stays identical to int(voice_score * VOICE_WEIGHT).
    """
    signals = voice.get("voice_signals", []) or []
    total_points = int(voice.get("voice_score", 0) * VOICE_WEIGHT)
    if not signals or total_points <= 0:
        return []

    weights = [float(s.get("points", s.get("weight", 1)) or 1) for s in signals]
    weight_sum = sum(weights) or len(signals)

    out = []
    allocated = 0
    for i, sig in enumerate(signals):
        if i == len(signals) - 1:
            pts = total_points - allocated
        else:
            pts = int(round(total_points * weights[i] / weight_sum))
            allocated += pts
        evidence = sig.get("evidence", "")
        out.append({
            "code": "voice_" + str(sig.get("code", i)),
            "label": "Call red flag — " + sig.get("reason", "suspicious call pattern"),
            "points": max(pts, 0),
            "reason": sig.get("reason", "Coercive pattern detected in the call"),
            "evidence": f'Heard: "{evidence}"' if evidence else "",
        })
    return out


# ---------------------------------------------------------------------------
# Action 0: what can this server actually do?
#
# The UI asks this on load so it can offer audio input only when Whisper is
# genuinely usable, and fall back to preset transcripts when it is not.
# ---------------------------------------------------------------------------
@app.get("/capabilities")
def capabilities():
    return {"transcription": transcription.status()}


# ---------------------------------------------------------------------------
# Action 1: real call audio -> text
#
# This is the only machine-learning component in the system: Whisper, a
# transformer trained on a large corpus of speech. It does NOT make any fraud
# decision - it only produces the transcript that the existing rule-based
# voice engine then scans. The scoring logic is untouched.
#
# The audio itself is never stored: it is written to a temp file, read once,
# and deleted. Only the derived text continues through the pipeline.
# ---------------------------------------------------------------------------
@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...), language: str = Form("")):
    data = await audio.read()

    if len(data) > MAX_UPLOAD_BYTES:
        return {"ok": False,
                "error": f"Audio is too large (limit {MAX_UPLOAD_BYTES // (1024*1024)} MB)."}

    result = transcription.transcribe_audio(
        data,
        filename=audio.filename or "audio.wav",
        language=(language or None),
    )

    # If audio was understood but contained no speech, say so plainly rather
    # than handing an empty string to the fraud engine.
    if result.get("ok") and not result.get("transcript"):
        result["ok"] = False
        result["error"] = "No speech detected in that audio."

    return result


# ---- Action 2: assess a transaction (payment behaviour + voice phishing) ----
@app.post("/assess")
def assess(tx: Transaction):
    started = time.perf_counter()
    profile = PROFILES.get(tx.user_id)
    tx_data = tx.model_dump()

    if profile is None:
        payment = {
            "score": 0, "decision": "APPROVE",
            "reasons": [], "fired_rules": [], "signals": [],
            "raw_score": 0, "capped": False,
            "confidence": {"level": "Low",
                           "note": "No history for this user yet",
                           "borderline": False},
            "display": decision_display("APPROVE"),
            "thresholds": {"warn_at": WARN_AT, "block_at": BLOCK_AT},
            "privacy": privacy_manifest(tx_data),
            "latency_ms": 0.0,
        }
    else:
        payment = assess_transaction(tx_data, profile)

    voice = analyse_transcript(tx.transcript)
    result = merge_external_signals(payment, voice_signals_as_attribution(voice))

    total_latency = round((time.perf_counter() - started) * 1000, 2)
    reasons = result["reasons"] if result["reasons"] else ["No suspicious signals detected"]

    return {
        "score": result["score"],
        "decision": result["decision"],
        "reasons": reasons,
        "payment_score": payment["score"],
        "voice_score": voice["voice_score"],
        "voice_level": voice["voice_level"],
        "signals": result["signals"],
        "raw_score": result["raw_score"],
        "capped": result["capped"],
        "confidence": result["confidence"],
        "display": result.get("display", {}),
        "thresholds": {"warn_at": WARN_AT, "block_at": BLOCK_AT},
        "privacy": payment["privacy"],
        "latency_ms": total_latency,
    }


# ---- Action 3: record the human's decision into the audit log ----
@app.post("/decision")
def record_decision(d: Decision):
    file_exists = os.path.exists(AUDIT_FILE)
    with open(AUDIT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(AUDIT_HEADER)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            csv_safe(d.user_id),
            d.amount,
            csv_safe(d.payee),
            csv_safe(d.decision),
            d.score,
            csv_safe(d.user_action),
            csv_safe(d.confidence),
            d.signal_count,
            "PENDING",
        ])
    return {"status": "logged"}


# ---- Action 4: bank reviewer records what actually happened ----
@app.post("/outcome")
def record_outcome(o: Outcome):
    """
    Closes the feedback loop the problem statement asks for: when a user
    overrides a warning, the bank can later mark whether it really was fraud.
    """
    if o.outcome not in VALID_OUTCOMES:
        return {"status": "invalid outcome"}

    if not os.path.exists(AUDIT_FILE):
        return {"status": "no log"}

    with open(AUDIT_FILE, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return {"status": "empty"}

    header, body = rows[0], rows[1:]
    target = len(body) - 1 - o.row_index
    if target < 0 or target >= len(body):
        return {"status": "bad index"}

    while len(body[target]) < len(AUDIT_HEADER):
        body[target].append("")
    body[target][AUDIT_HEADER.index("outcome")] = o.outcome

    with open(AUDIT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(AUDIT_HEADER)
        writer.writerows(body)
    return {"status": "updated", "outcome": o.outcome}


# ---- Action 5: return the audit log for the bank review dashboard ----
@app.get("/audit")
def get_audit():
    rows = []
    if os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, newline="") as f:
            rows = list(csv.DictReader(f))
    rows.reverse()

    for r in rows:
        r.setdefault("confidence", "")
        r.setdefault("signal_count", "")
        r.setdefault("outcome", "PENDING")
        if not r.get("outcome"):
            r["outcome"] = "PENDING"

    total = len(rows)
    blocked = sum(1 for r in rows if r["shield_decision"] == "BLOCK")
    warned = sum(1 for r in rows if r["shield_decision"] == "WARN")
    overrides = sum(1 for r in rows
                    if r["shield_decision"] in ("WARN", "BLOCK") and r["user_action"] == "CONFIRMED")
    confirmed_fraud = sum(1 for r in rows if r["outcome"] == "CONFIRMED_FRAUD")
    confirmed_legit = sum(1 for r in rows if r["outcome"] == "CONFIRMED_LEGITIMATE")
    pending = sum(1 for r in rows if r["outcome"] == "PENDING")

    return {
        "summary": {
            "total": total,
            "blocked": blocked,
            "warned": warned,
            "overrides": overrides,
            "confirmed_fraud": confirmed_fraud,
            "confirmed_legitimate": confirmed_legit,
            "pending_review": pending,
        },
        "rows": rows,
    }


# ---- Action 6: compute live detection metrics over the whole dataset ----
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
        "thresholds": {"warn_at": WARN_AT, "block_at": BLOCK_AT},
    }


# ---- Serve the web pages ----
app.mount("/", StaticFiles(directory="static", html=True), name="static")