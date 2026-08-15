"""
voice_phishing.py
Detects coercion and voice-phishing (vishing) patterns in a call transcript.
Real UPI scams happen over a phone call that pressures the victim just before
they pay. This reads that transcript and scores how scam-like the conversation is.
Every hit is explainable: we return the exact phrase and category that fired.
"""

import re

# Coercion / vishing categories. Each is a real tactic used in Indian UPI scams.
# Patterns are lowercase substrings or simple regex fragments.
VISHING_PATTERNS = {
    "authority_impersonation": {
        "points": 20,
        "reason": "Caller claims to be from a bank / police / government body",
        "phrases": ["from the bank", "bank official", "from your bank", "police",
                    "cyber cell", "income tax", "rbi", "customs", "arrest",
                    "legal action", "court", "cbi", "ed officer"],
    },
    "urgency_pressure": {
        "points": 18,
        "reason": "Creates false urgency to stop the victim from thinking",
        "phrases": ["act now", "immediately", "within 10 minutes", "right now",
                    "urgent", "expire", "expires today", "last chance",
                    "account will be blocked", "account will be closed",
                    "before it's too late", "hurry"],
    },
    "secrecy_isolation": {
        "points": 22,
        "reason": "Tells the victim to keep it secret or not consult anyone",
        "phrases": ["don't tell anyone", "do not tell anyone", "keep this confidential",
                    "don't inform", "do not inform", "between us",
                    "don't discuss", "stay on the call", "do not cut the call",
                    "don't hang up", "don't talk to anyone"],
    },
    "credential_request": {
        "points": 25,
        "reason": "Asks for OTP / PIN / password — banks never do this",
        "phrases": ["share the otp", "tell me the otp", "otp", "your pin",
                    "upi pin", "share your pin", "cvv", "password",
                    "verification code", "read the code"],
    },
    "payment_coercion": {
        "points": 20,
        "reason": "Pushes the victim to pay, transfer, or 'verify' by paying",
        "phrases": ["make a payment", "transfer the amount", "pay now",
                    "small verification fee", "refundable", "just pay",
                    "send the money", "scan this qr", "processing fee",
                    "security deposit"],
    },
    "fear_threat": {
        "points": 15,
        "reason": "Uses fear or threats to force compliance",
        "phrases": ["your account is compromised", "suspicious activity",
                    "your account is frozen", "kyc expired", "kyc is pending",
                    "your sim will be blocked", "penalty", "case against you",
                    "money laundering"],
    },
}

WARN_AT = 25
BLOCK_AT = 45


def analyse_transcript(transcript: str):
    """
    Score a call transcript for coercion / vishing signals.
    Returns score, level (CLEAN / SUSPICIOUS / HIGH_RISK), and the reasons + phrases found.
    """
    text = transcript.lower()
    fired = []
    score = 0

    for category, cfg in VISHING_PATTERNS.items():
        matched_phrase = None
        for phrase in cfg["phrases"]:
            # word-ish boundary match so "otp" doesn't match inside another word
            if re.search(r"\b" + re.escape(phrase) + r"\b", text):
                matched_phrase = phrase
                break
        if matched_phrase:
            score += cfg["points"]
            fired.append({
                "category": category,
                "reason": cfg["reason"],
                "evidence": matched_phrase,
            })

    score = min(score, 100)

    if score >= BLOCK_AT:
        level = "HIGH_RISK"
    elif score >= WARN_AT:
        level = "SUSPICIOUS"
    else:
        level = "CLEAN"

    return {
        "voice_score": score,
        "voice_level": level,
        "voice_signals": fired,
    }


# Quick self-test when run directly: python voice_phishing.py
if __name__ == "__main__":
    scam_call = (
        "Hello, I am calling from your bank. Your account is frozen due to "
        "suspicious activity. This is urgent, you must act now. Do not tell "
        "anyone about this call. Please share the OTP to verify your identity "
        "and make a small refundable payment to unblock your account."
    )
    normal_call = (
        "Hey, are we still meeting for lunch tomorrow? I'll book the table "
        "for one o'clock. Let me know if that works for you."
    )

    print("=== SCAM CALL ===")
    result = analyse_transcript(scam_call)
    print("Score:", result["voice_score"], "Level:", result["voice_level"])
    for sig in result["voice_signals"]:
        print(f"  - [{sig['category']}] {sig['reason']}  (heard: \"{sig['evidence']}\")")

    print("\n=== NORMAL CALL ===")
    result = analyse_transcript(normal_call)
    print("Score:", result["voice_score"], "Level:", result["voice_level"])
    print("  signals:", result["voice_signals"])