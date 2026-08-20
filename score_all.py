"""
score_all.py
Runs the risk engine across every transaction in transactions.csv,
prints a few example decisions, and reports how accurate it was
against the known fraud labels.
"""

import csv
from risk_engine import build_user_profile, assess_transaction

# Load all transactions from the CSV
with open("transactions.csv", newline="") as f:
    all_tx = list(csv.DictReader(f))

# Group each user's transactions in time order so we can learn their "normal".
# We build a user's profile from what we've seen BEFORE the current transaction.
history_by_user = {}

results = []
for tx in all_tx:
    uid = tx["user_id"]
    past = history_by_user.get(uid, [])

    # Need some history to judge against; give the first few per user a pass.
    if len(past) < 3:
        assessment = {"score": 0, "decision": "APPROVE", "reasons": ["Not enough history yet"], "fired_rules": []}
    else:
        profile = build_user_profile(past)
        assessment = assess_transaction(tx, profile)

    results.append({
        "timestamp": tx["timestamp"],
        "user_id": uid,
        "amount": tx["amount"],
        "payee": tx["payee"],
        "decision": assessment["decision"],
        "score": assessment["score"],
        "reasons": assessment["reasons"],
        "is_fraud": int(tx["is_fraud"]),
    })

    # Only learn from transactions we APPROVED (a real system wouldn't
    # add a blocked scam payee to "normal"). This keeps profiles clean.
    if assessment["decision"] in ("APPROVE", "VERIFY"):
        history_by_user.setdefault(uid, []).append(tx)

# ---- Print a few flagged examples so you can see it working ----
print("\n=== Sample flagged transactions ===")
shown = 0
for r in results:
    if r["decision"] != "APPROVE" and shown < 8:
        print(f"\n[{r['decision']}] score {r['score']}  {r['user_id']}  Rs.{r['amount']} -> {r['payee']}")
        for reason in r["reasons"]:
            print(f"    - {reason}")
        shown += 1

# ---- Accuracy report against the known labels ----
# We treat WARN or BLOCK as "flagged as fraud".
tp = fp = tn = fn = 0
for r in results:
    flagged = r["decision"] in ("WARN", "BLOCK")
    fraud = r["is_fraud"] == 1
    if flagged and fraud:       tp += 1   # correctly caught
    elif flagged and not fraud: fp += 1   # false alarm
    elif not flagged and fraud: fn += 1   # missed
    else:                       tn += 1   # correctly left alone

precision = tp / (tp + fp) if (tp + fp) else 0
recall    = tp / (tp + fn) if (tp + fn) else 0
fp_rate   = fp / (fp + tn) if (fp + tn) else 0

print("\n=== Detection performance ===")
print(f"Caught (true positives):   {tp}")
print(f"Missed (false negatives):  {fn}")
print(f"False alarms (false pos):  {fp}")
print(f"Correctly cleared:         {tn}")
print(f"\nPrecision: {precision:.0%}  (of what we flagged, how much was really fraud)")
print(f"Recall:    {recall:.0%}  (of all real fraud, how much we caught)")
print(f"False-positive rate: {fp_rate:.1%}  (legit payments we wrongly flagged)")