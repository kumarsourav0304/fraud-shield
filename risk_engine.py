"""
risk_engine.py
The explainable core of the Fraud Shield.
Given one transaction plus the user's normal history, it returns:
  - a risk score (0-100)
  - a decision: APPROVE / WARN / BLOCK
  - a plain-language reason for every signal that fired
No black box: every point of risk is traceable to a named rule.
"""

# Each rule has: a name, how many risk points it adds, and a plain-English reason.
# Weights reflect how strongly each signal indicates fraud in real UPI scams.
RULES = {
    "new_device":      {"points": 35, "reason": "Payment from a device never used on this account before"},
    "unknown_payee":   {"points": 25, "reason": "Money going to a payee this user has never paid before"},
    "away_from_home":  {"points": 15, "reason": "Transaction location differs from the user's usual city"},
    "large_amount":    {"points": 20, "reason": "Amount is far larger than this user's typical payment"},
    "odd_hour":        {"points": 10, "reason": "Payment made during late-night hours (12am-5am)"},
}

# Score thresholds that turn a number into an action.
WARN_AT = 30    # 30-59 points -> warn the user, let them confirm
BLOCK_AT = 60   # 60+ points   -> block and ask for review


def build_user_profile(history):
    """
    Learn what 'normal' looks like for one user from their past transactions.
    history: a list of that user's previous transaction dicts.
    Returns a profile the rules can check against.
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


def assess_transaction(tx, profile):
    """
    Score a single transaction against the user's profile.
    Returns a dict with score, decision, and the list of reasons that fired.
    """
    fired = []          # which rules triggered
    score = 0

    # Rule 1: brand-new device
    if tx["device"] not in profile["known_devices"]:
        fired.append("new_device")

    # Rule 2: payee never paid before
    if tx["payee"] not in profile["known_payees"]:
        fired.append("unknown_payee")

    # Rule 3: unusual city
    if tx["city"] not in profile["known_cities"]:
        fired.append("away_from_home")

    # Rule 4: amount much larger than usual (more than 3x their typical max)
    if profile["typical_amount"] > 0 and float(tx["amount"]) > 3 * profile["typical_amount"]:
        fired.append("large_amount")

    # Rule 5: odd hour (timestamp looks like "2026-08-14 02:30:00")
    hour = int(tx["timestamp"].split(" ")[1].split(":")[0])
    if 0 <= hour < 5:
        fired.append("odd_hour")

    # Add up points and collect reasons
    reasons = []
    for rule_name in fired:
        score += RULES[rule_name]["points"]
        reasons.append(RULES[rule_name]["reason"])

    score = min(score, 100)  # cap at 100

    # Turn the score into an action
    if score >= BLOCK_AT:
        decision = "BLOCK"
    elif score >= WARN_AT:
        decision = "WARN"
    else:
        decision = "APPROVE"

    return {
        "score": score,
        "decision": decision,
        "reasons": reasons,
        "fired_rules": fired,
    }