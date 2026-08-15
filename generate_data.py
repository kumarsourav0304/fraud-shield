"""
generate_data.py
Creates a synthetic UPI transaction dataset for the Fraud Shield.

This version makes fraud REALISTIC and VARIED: instead of every scam tripping
all signals at once, each fraudulent transaction trips a random SUBSET of
signals. This mirrors real fraud (some scams only look mildly unusual) and
means the engine will have honest misses and edge cases to discuss — not a
suspiciously perfect score.
"""

import random
import csv
from datetime import datetime, timedelta

random.seed(42)  # reproducible for demos

USERS = [
    {"user_id": "U001", "home_city": "Ranchi",    "device": "DEV-A1"},
    {"user_id": "U002", "home_city": "Kolkata",   "device": "DEV-B2"},
    {"user_id": "U003", "home_city": "Delhi",     "device": "DEV-C3"},
    {"user_id": "U004", "home_city": "Mumbai",    "device": "DEV-D4"},
    {"user_id": "U005", "home_city": "Bengaluru", "device": "DEV-E5"},
]

KNOWN_PAYEES = ["merchant_grocery", "friend_amit", "electricity_board",
                "mobile_recharge", "landlord_rent", "medical_store", "fuel_station"]

SCAM_PAYEES = ["unknown_9284", "quickcash_pay", "verify_kyc_now",
               "lucky_winner_44", "support_refund", "kyc_update_88", "prize_claim_01"]

OTHER_CITIES = ["Chennai", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Surat"]

# Normal amounts a real user makes
NORMAL_AMOUNTS = [50, 120, 250, 480, 800, 1200, 1500, 2200, 3000]
# Larger fraud amounts, but with some overlap into "normal-ish" territory
FRAUD_AMOUNTS_BIG   = [15000, 24999, 40000, 49999]
FRAUD_AMOUNTS_SMALL = [2500, 3500, 5000, 8000]   # harder-to-catch smaller scams

rows = []
start_time = datetime(2026, 8, 10, 8, 0, 0)


def make_normal(user, ts):
    return {
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user["user_id"],
        "amount": random.choice(NORMAL_AMOUNTS),
        "payee": random.choice(KNOWN_PAYEES),
        "city": user["home_city"],
        "device": user["device"],
        "is_fraud": 0,
    }


def make_fraud(user, ts):
    """
    Build a fraud transaction that trips a RANDOM SUBSET of signals.
    Every fraud has an unknown payee (that's the one constant in a scam),
    but the other signals (new device, other city, big amount, odd hour)
    appear only sometimes. This produces easy, medium and hard cases.
    """
    # Start from the user's normal footprint...
    tx = {
        "user_id": user["user_id"],
        "amount": random.choice(FRAUD_AMOUNTS_SMALL),  # default: smaller amount
        "payee": random.choice(SCAM_PAYEES),           # always a scam payee
        "city": user["home_city"],                     # default: same city
        "device": user["device"],                      # default: same device
    }

    # ...then randomly switch on extra fraud signals
    if random.random() < 0.6:                          # 60% use a new device
        tx["device"] = "DEV-NEW-" + str(random.randint(10, 99))
    if random.random() < 0.5:                          # 50% from another city
        tx["city"] = random.choice(OTHER_CITIES)
    if random.random() < 0.55:                         # 55% large amount
        tx["amount"] = random.choice(FRAUD_AMOUNTS_BIG)

    # odd-hour timing on some
    if random.random() < 0.4:
        ts = ts.replace(hour=random.randint(0, 4))

    tx["timestamp"] = ts.strftime("%Y-%m-%d %H:%M:%S")
    tx["is_fraud"] = 1
    return tx


# Build 600 transactions, ~12% fraud
ts = start_time
for i in range(600):
    user = random.choice(USERS)
    ts = ts + timedelta(minutes=random.randint(1, 35))
    if random.random() < 0.12:
        rows.append(make_fraud(user, ts))
    else:
        rows.append(make_normal(user, ts))

fieldnames = ["timestamp", "user_id", "amount", "payee", "city", "device", "is_fraud"]
with open("transactions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

fraud_count = sum(r["is_fraud"] for r in rows)
print(f"Done. Wrote {len(rows)} transactions to transactions.csv")
print(f"Of those, {fraud_count} are fraud and {len(rows) - fraud_count} are normal.")
print("Fraud now trips a VARIED subset of signals (easy, medium, hard cases).")