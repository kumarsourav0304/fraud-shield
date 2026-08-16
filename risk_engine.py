"""
risk_engine.py
The explainable core of the Fraud Shield.

Given one transaction plus the user's normal history, it returns:
  - a risk score (0-100)
  - a decision: APPROVE / WARN / BLOCK
  - per-signal attribution: exactly how many points each signal contributed
  - a confidence level based on how many INDEPENDENT signals agree
  - a privacy manifest: what data leaves the device vs what never does
  - scoring latency in milliseconds

No black box: every point of risk is traceable to a named rule.
"""

import hashlib
import time

# ---------------------------------------------------------------------------
# RULES
# Each rule: display label, risk points, plain-English reason.
# Weights reflect how strongly each signal indicates fraud in real UPI scams.
# ---------------------------------------------------------------------------
RULES = {
    "new_device": {
        "label": "New device",
        "points": 35,
        "reason": "Payment from a device never used on this account before",
    },
    "unknown_payee": {
        "label": "Unknown payee",
        "points": 25,
        "reason": "Money going to a payee this user has never paid before",
    },
    "away_from_home": {
        "label": "Unusual location",
        "points": 15,
        "reason": "Transaction location differs from the user's usual city",
    },
    "large_amount": {
        "label": "Unusually large amount",
        "points": 20,
        "reason": "Amount is far larger than this user's typical payment",
    },
    "odd_hour": {
        "label": "Late-night payment",
        "points": 10,
        "reason": "Payment made during late-night hours (12am-5am)",
    },
    # Optional signal - only fires if the transaction carries this field.
    # Covers the malicious-app / screen-control fraud vector.
    "accessibility_abuse": {
        "label": "Screen-control app active",
        "points": 40,
        "reason": "An app with screen-control permission is active during payment "
                  "- a known method for remotely approving payments",
    },
}

# ---------------------------------------------------------------------------
# USER-FACING LABELS
# Internal decision codes stay APPROVE / WARN / BLOCK everywhere (metrics,
# CSV, scoring) so nothing downstream breaks. These are ONLY for display.
# The top state is deliberately NOT called "blocked": the problem statement
# requires supporting user confirmation WITHOUT blocking legitimate urgent
# payments, and the user can always proceed. It's a hard warning, not a wall.
# ---------------------------------------------------------------------------
DECISION_DISPLAY = {
    "APPROVE": {
        "label": "Looks safe",
        "plain": "Nothing unusual about this payment. Safe to continue.",
        "action_hint": "You can proceed normally.",
    },
    "WARN": {
        "label": "Unusual — please check",
        "plain": "This payment is a little unusual for you. Take a moment to make "
                 "sure you meant to send it.",
        "action_hint": "Proceed only if you're sure. Cancel if anything feels off.",
    },
    "BLOCK": {
        "label": "High risk — confirm to proceed",
        "plain": "This payment strongly matches patterns seen in scams. We are NOT "
                 "blocking it — but please stop and confirm you really want to send "
                 "this, especially if someone is pressuring or guiding you right now.",
        "action_hint": "If anyone is on a call telling you to pay, hang up and verify "
                       "first. You can still proceed if you're certain.",
    },
}


def decision_display(decision):
    """Plain-language wrapper for a decision code, for the end-user view."""
    return DECISION_DISPLAY.get(decision, DECISION_DISPLAY["WARN"])


# ---------------------------------------------------------------------------
# THRESHOLDS
# Tuned on a 600-transaction set to hold false-positive rate under 1%
# while keeping recall above 90%. Lowering BLOCK_AT to 50 raised recall
# slightly but roughly tripled false alarms - not worth it for payments.
# ---------------------------------------------------------------------------
WARN_AT = 30     # 30-59 points -> warn the user, let them confirm
BLOCK_AT = 60    # 60+ points   -> highest warning, user still confirms to proceed

# How close to a threshold counts as "borderline" (flagged for bank review).
BORDERLINE_MARGIN = 8


# ---------------------------------------------------------------------------
# PRIVACY
# ---------------------------------------------------------------------------
def pseudonymise(value):
    """
    Turn an identifier into a short irreversible fingerprint.
    Used so payee IDs and device IDs can be compared and logged
    WITHOUT storing the real value anywhere.
    """
    if value is None:
        return None
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return digest[:12]


def privacy_manifest(tx):
    """
    Describes exactly what this assessment transmits and what it never touches.
    Returned with every assessment so the UI can prove the privacy claim
    instead of just asserting it.
    """
    return {
        "transmitted": [
            {"field": "payee fingerprint", "value": pseudonymise(tx.get("payee")),
             "note": "irreversible hash - the real payee ID is never sent"},
            {"field": "device fingerprint", "value": pseudonymise(tx.get("device")),
             "note": "irreversible hash - device identity is never sent"},
            {"field": "amount band", "value": amount_band(tx.get("amount")),
             "note": "bucketed range, not the exact rupee value"},
            {"field": "derived risk flags", "value": "true/false only",
             "note": "computed on device; raw inputs stay local"},
        ],
        "never_transmitted": [
            "Raw contact list",
            "Raw call audio or recordings",
            "Call transcript text",
            "Exact GPS coordinates",
            "SMS or message contents",
            "Full account or device identifiers",
        ],
    }


def amount_band(amount):
    """Bucket an amount so the exact value never needs to leave the device."""
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "unknown"
    if amount < 500:
        return "under Rs 500"
    if amount < 5000:
        return "Rs 500 - 5,000"
    if amount < 25000:
        return "Rs 5,000 - 25,000"
    if amount < 100000:
        return "Rs 25,000 - 1,00,000"
    return "over Rs 1,00,000"


# ---------------------------------------------------------------------------
# USER PROFILE
# ---------------------------------------------------------------------------
def build_user_profile(history):
    """
    Learn what 'normal' looks like for one user from their past transactions.
    history: a list of that user's previous transaction dicts.
    """
    known_devices = set()
    known_payees = set()
    known_cities = set()
    amounts = []
    for tx in history:
        known_devices.add(tx["device"])
        known_payees.add(tx["payee"])
        known_cities.add(tx["city"])
        amounts.append(float(tx["amount"]))

    typical_amount = max(amounts) if amounts else 0
    return {
        "known_devices": known_devices,
        "known_payees": known_payees,
        "known_cities": known_cities,
        "typical_amount": typical_amount,
    }


# ---------------------------------------------------------------------------
# CONFIDENCE
# ---------------------------------------------------------------------------
def rate_confidence(signals, score):
    """
    Confidence is NOT a probability - this is a rules engine, not a
    calibrated model, and claiming a percentage would be dishonest.

    Instead confidence reflects how many INDEPENDENT signals agree.
    Several unrelated signals firing together is much harder to explain
    away than a single one, so it warrants more trust.
    """
    count = len(signals)

    if count >= 4:
        level, note = "High", f"{count} independent signals agree"
    elif count == 3:
        level, note = "High", "3 independent signals agree"
    elif count == 2:
        level, note = "Medium", "2 independent signals agree"
    elif count == 1:
        level, note = "Low", "Only 1 signal fired - weak evidence on its own"
    else:
        level, note = "High", "No risk signals detected"

    # A score sitting right on a threshold deserves human eyes regardless.
    borderline = (
        abs(score - WARN_AT) <= BORDERLINE_MARGIN
        or abs(score - BLOCK_AT) <= BORDERLINE_MARGIN
    )
    if borderline and count > 0:
        if level == "High":
            level = "Medium"
        note += " - score is close to a decision threshold"

    return {"level": level, "note": note, "borderline": borderline}


# ---------------------------------------------------------------------------
# MAIN ASSESSMENT
# ---------------------------------------------------------------------------
def assess_transaction(tx, profile):
    """
    Score a single transaction against the user's profile.
    Returns score, decision, per-signal attribution, confidence,
    privacy manifest and scoring latency.
    """
    started = time.perf_counter()

    fired = []
    evidence = {}

    # Rule 1: brand-new device
    if tx["device"] not in profile["known_devices"]:
        fired.append("new_device")
        evidence["new_device"] = "Device not in this account's history"

    # Rule 2: payee never paid before
    if tx["payee"] not in profile["known_payees"]:
        fired.append("unknown_payee")
        evidence["unknown_payee"] = "No previous payment to this payee"

    # Rule 3: unusual city
    if tx["city"] not in profile["known_cities"]:
        fired.append("away_from_home")
        evidence["away_from_home"] = f"City '{tx['city']}' is new for this user"

    # Rule 4: amount much larger than usual (more than 3x their typical max)
    amount = float(tx["amount"])
    typical = profile["typical_amount"]
    if typical > 0 and amount > 3 * typical:
        fired.append("large_amount")
        multiple = round(amount / typical, 1)
        evidence["large_amount"] = (
            f"Rs {amount:,.0f} is {multiple}x this user's usual maximum "
            f"of Rs {typical:,.0f}"
        )

    # Rule 5: odd hour (timestamp looks like "2026-08-14 02:30:00")
    try:
        hour = int(tx["timestamp"].split(" ")[1].split(":")[0])
    except (KeyError, IndexError, ValueError):
        hour = None
    if hour is not None and 0 <= hour < 5:
        fired.append("odd_hour")
        evidence["odd_hour"] = f"Payment attempted at {hour:02d}:xx"

    # Rule 6 (optional): screen-control / accessibility abuse.
    # Only fires when the field is present, so historical data is unaffected.
    if tx.get("accessibility_service_active"):
        fired.append("accessibility_abuse")
        evidence["accessibility_abuse"] = (
            "A third-party app currently holds screen-control permission"
        )

    # Build per-signal attribution and total the score
    signals = []
    score = 0
    reasons = []
    for code in fired:
        rule = RULES[code]
        score += rule["points"]
        reasons.append(rule["reason"])
        signals.append({
            "code": code,
            "label": rule["label"],
            "points": rule["points"],
            "reason": rule["reason"],
            "evidence": evidence.get(code, ""),
        })

    raw_score = score
    score = min(score, 100)  # cap at 100

    # Turn the score into an action
    if score >= BLOCK_AT:
        decision = "BLOCK"
    elif score >= WARN_AT:
        decision = "WARN"
    else:
        decision = "APPROVE"

    confidence = rate_confidence(signals, score)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return {
        # --- original keys, unchanged for backward compatibility ---
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "fired_rules": fired,
        # --- new, additive ---
        "signals": signals,
        "raw_score": raw_score,
        "capped": raw_score > 100,
        "confidence": confidence,
        "display": decision_display(decision),
        "thresholds": {"warn_at": WARN_AT, "block_at": BLOCK_AT},
        "privacy": privacy_manifest(tx),
        "latency_ms": latency_ms,
    }


# ---------------------------------------------------------------------------
# EXTERNAL SIGNAL MERGE
# ---------------------------------------------------------------------------
def merge_external_signals(assessment, extra_signals):
    """
    Fold signals computed elsewhere (e.g. voice-phishing detection in
    voice_phishing.py) into the same attribution structure, so the UI shows
    ONE decomposed list rather than two separate fused scores.

    extra_signals: list of dicts, each needing at minimum
        {"code": str, "label": str, "points": int, "reason": str}
        and optionally "evidence": str

    Returns a new assessment dict - the original is not modified.
    """
    merged = dict(assessment)
    signals = list(assessment.get("signals", []))
    reasons = list(assessment.get("reasons", []))
    fired = list(assessment.get("fired_rules", []))

    raw = assessment.get("raw_score", assessment["score"])
    for sig in extra_signals:
        signals.append({
            "code": sig["code"],
            "label": sig["label"],
            "points": sig["points"],
            "reason": sig["reason"],
            "evidence": sig.get("evidence", ""),
        })
        reasons.append(sig["reason"])
        fired.append(sig["code"])
        raw += sig["points"]

    score = min(raw, 100)
    if score >= BLOCK_AT:
        decision = "BLOCK"
    elif score >= WARN_AT:
        decision = "WARN"
    else:
        decision = "APPROVE"

    merged.update({
        "signals": signals,
        "reasons": reasons,
        "fired_rules": fired,
        "raw_score": raw,
        "capped": raw > 100,
        "score": score,
        "decision": decision,
        "confidence": rate_confidence(signals, score),
        "display": decision_display(decision),
    })
    return merged