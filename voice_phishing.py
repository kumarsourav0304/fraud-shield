"""
voice_phishing.py
Detects coercion and voice-phishing (vishing) patterns in a call transcript.

Real UPI scams happen over a phone call that pressures the victim just before
they pay. This reads that transcript and scores how scam-like the conversation is.
Every hit is explainable: we return the exact phrase and category that fired.

v2: Nine canonical intent classes, Hinglish normalization, hybrid NLP+rules
    integration, and BENIGN contextual awareness.
"""

import re
import time

from text_normalizer import normalize as normalize_text

# Try to import NLP — graceful fallback if unavailable
try:
    from nlp_intent import predict_intent, is_available as nlp_is_available
except ImportError:
    predict_intent = None
    nlp_is_available = lambda: False


# ======================================================================
# NINE CANONICAL INTENT CLASSES
#
# Mapped from the original six categories plus three new ones.
# Original mapping:
#   authority_impersonation  ->  AUTHORITY
#   credential_request       ->  CREDENTIAL
#   fear_threat              ->  THREAT
#   urgency_pressure         ->  URGENCY
#   payment_coercion         ->  PAYMENT_COERCION
#   secrecy_isolation        ->  SECRECY
# New:
#   REMOTE_ACCESS
#   DIGITAL_ARREST
#   BENIGN (NLP-only — no deterministic rules fire for benign)
# ======================================================================

VISHING_PATTERNS = {
    "AUTHORITY": {
        "points": 20,
        "reason": "Caller claims to be from a bank / police / government body",
        "phrases": [
            "from the bank", "bank official", "from your bank", "police",
            "cyber cell", "income tax", "rbi", "customs",
            "legal action", "court", "cbi", "ed officer",
            "calling from the bank", "police is calling",
            "from the police", "i am a cbi officer",
            "calling from cbi", "from the cyber cell",
            "from cyber crime", "income tax department",
            "telecom authority", "from trai", "from rbi",
            "i am an officer", "government",
        ],
    },
    "CREDENTIAL": {
        "points": 25,
        "reason": "Asks for OTP / PIN / password — banks never do this",
        "phrases": [
            "share the otp", "tell me the otp", "otp", "your pin",
            "upi pin", "share your pin", "cvv", "password",
            "verification code", "read the code",
            "send the otp", "share your password",
            "share your cvv", "cvv number",
        ],
    },
    "THREAT": {
        "points": 15,
        "reason": "Uses fear or threats to force compliance",
        "phrases": [
            "your account is compromised", "suspicious activity",
            "your account is frozen", "kyc expired", "kyc is pending",
            "your sim will be blocked", "penalty", "case against you",
            "money laundering", "account will be blocked",
            "account will be closed", "account will be frozen",
            "account suspended", "money will be deducted from account",
            "sim blocked", "you will go to jail",
            "we will send you to jail", "warrant issued",
            "fir filed",
        ],
    },
    "URGENCY": {
        "points": 18,
        "reason": "Creates false urgency to stop the victim from thinking",
        "phrases": [
            "act now", "immediately", "within 10 minutes", "right now",
            "urgent", "expire", "expires today", "last chance",
            "before it's too late", "hurry",
            "right now immediately", "do it now", "hurry up",
            "do not delay", "time is running out", "time is up",
            "this is your last chance", "must do it today",
            "if not now then",
        ],
    },
    "PAYMENT_COERCION": {
        "points": 20,
        "reason": "Pushes the victim to pay, transfer, or 'verify' by paying",
        "phrases": [
            "make a payment", "transfer the amount", "pay now",
            "small verification fee", "refundable", "just pay",
            "send the money", "scan this qr", "processing fee",
            "security deposit", "transfer money", "send money",
            "send money for verification", "verification fee",
            "send processing fee", "you will get a refund",
            "for a refund", "scan the qr code",
        ],
    },
    "SECRECY": {
        "points": 22,
        "reason": "Tells the victim to keep it secret or not consult anyone",
        "phrases": [
            "don't tell anyone", "do not tell anyone",
            "keep this confidential", "don't inform", "do not inform",
            "between us", "don't discuss", "stay on the call",
            "do not cut the call", "don't hang up",
            "don't talk to anyone", "don't cut the call",
            "stay on the line", "don't tell your family",
            "this is confidential", "keep it secret",
        ],
    },
    "REMOTE_ACCESS": {
        "points": 22,
        "reason": "Asks the victim to install remote-access software or share screen",
        "phrases": [
            "anydesk", "teamviewer", "screen share", "install this app",
            "remote access", "download this", "install anydesk",
            "install teamviewer", "download this app",
            "share your screen", "give remote access",
            "quick support", "ultraviewer",
        ],
    },
    "DIGITAL_ARREST": {
        "points": 25,
        "reason": "Claims the victim is 'digitally arrested' — a scam unique to India",
        "phrases": [
            "digital arrest", "under arrest", "video verification",
            "stay on video call", "supreme court order",
            "under digital arrest", "you have been arrested",
            "court order", "court order issued", "high court notice",
            "arrested", "arrest",
        ],
    },
}

WARN_AT = 25
BLOCK_AT = 45

# Weight applied to NLP-only signals that lack rule corroboration
NLP_ONLY_WEIGHT = 0.6


def _match_rules(text_normalized):
    """
    Run deterministic pattern matching on normalized text.
    Returns list of fired signals with category, evidence, etc.
    """
    fired = []
    score = 0

    for category, cfg in VISHING_PATTERNS.items():
        matched_phrase = None
        for phrase in cfg["phrases"]:
            if re.search(r"\b" + re.escape(phrase) + r"\b", text_normalized):
                matched_phrase = phrase
                break
        if matched_phrase:
            score += cfg["points"]
            fired.append({
                "category": category,
                "code": category,
                "reason": cfg["reason"],
                "evidence": matched_phrase,
                "points": cfg["points"],
                "source": "rule",
            })

    return fired, min(score, 100)


def analyse_transcript(transcript: str):
    """
    Score a call transcript for coercion / vishing signals.

    This is the hybrid analysis path: it normalizes the text, runs
    deterministic rules, runs NLP classification (if available), merges
    the results with corroboration-aware deduplication, and applies
    BENIGN contextual suppression.

    Backward-compatible: returns the same shape as the original function
    with additional fields.
    """
    started = time.perf_counter()

    if not transcript or not transcript.strip():
        return {
            "voice_score": 0,
            "voice_level": "CLEAN",
            "voice_signals": [],
            "nlp_intent": None,
            "nlp_confidence": 0.0,
            "is_benign": False,
            "latency_ms": 0.0,
        }

    # --- Normalize ---
    text_normalized = normalize_text(transcript)

    # --- Deterministic rules on normalized text ---
    rule_signals, rule_score = _match_rules(text_normalized)
    rule_categories = {s["category"] for s in rule_signals}

    # --- NLP classification (if available) ---
    nlp_result = None
    nlp_intent = None
    nlp_confidence = 0.0
    is_benign = False

    if predict_intent is not None and nlp_is_available():
        try:
            nlp_result = predict_intent(text_normalized)
            nlp_intent = nlp_result.get("intent")
            nlp_confidence = nlp_result.get("confidence", 0.0)
        except Exception:
            nlp_result = None

    # --- BENIGN contextual suppression ---
    # If NLP classifies transcript as BENIGN (e.g. educational, news, awareness),
    # suppress escalation so benign discussions of fraud don't trigger false alarms.
    if nlp_intent == "BENIGN" and nlp_confidence >= 0.50:
        is_benign = True

    # --- Merge NLP + rules with corroboration awareness ---
    merged_signals = list(rule_signals)  # start with rule signals
    merged_score = rule_score

    if nlp_result and nlp_intent and nlp_intent not in ("BENIGN", "UNAVAILABLE"):
        if nlp_intent in rule_categories:
            # NLP corroborates a rule finding — mark it but don't double-count
            for sig in merged_signals:
                if sig["category"] == nlp_intent:
                    sig["nlp_corroborated"] = True
                    sig["nlp_confidence"] = nlp_confidence
                    break
        elif nlp_confidence >= 0.5:
            # NLP found something the rules missed — add at reduced weight
            cat_cfg = VISHING_PATTERNS.get(nlp_intent, {})
            base_points = cat_cfg.get("points", 15)
            nlp_points = int(base_points * NLP_ONLY_WEIGHT)
            merged_signals.append({
                "category": nlp_intent,
                "code": nlp_intent,
                "reason": cat_cfg.get("reason", f"NLP detected {nlp_intent} intent"),
                "evidence": f"NLP classification: {nlp_intent} ({nlp_confidence:.0%} confident)",
                "points": nlp_points,
                "source": "nlp",
                "nlp_confidence": nlp_confidence,
            })
            merged_score += nlp_points

    # Apply BENIGN suppression
    if is_benign:
        merged_score = min(merged_score, 15)  # cap at low level

    merged_score = min(merged_score, 100)

    # --- Determine level ---
    if merged_score >= BLOCK_AT:
        level = "HIGH_RISK"
    elif merged_score >= WARN_AT:
        level = "SUSPICIOUS"
    else:
        level = "CLEAN"

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    return {
        "voice_score": merged_score,
        "voice_level": level,
        "voice_signals": merged_signals,
        "nlp_intent": nlp_intent,
        "nlp_confidence": round(nlp_confidence, 3),
        "is_benign": is_benign,
        "latency_ms": latency_ms,
    }


# Quick self-test when run directly: python voice_phishing.py
if __name__ == "__main__":
    tests = [
        ("SCAM CALL (English)", (
            "Hello, I am calling from your bank. Your account is frozen due to "
            "suspicious activity. This is urgent, you must act now. Do not tell "
            "anyone about this call. Please share the OTP to verify your identity "
            "and make a small refundable payment to unblock your account."
        )),
        ("SCAM CALL (Hinglish)", (
            "Sir main bank se bol raha hoon. Aapka account block ho jayega "
            "agar abhi payment nahi karo. OTP batao aur kisi ko mat batao. "
            "Turant paise bhejo."
        )),
        ("DIGITAL ARREST", (
            "You are under digital arrest. Supreme court ka order hai. "
            "Stay on video call. Transfer the amount immediately or "
            "warrant will be issued."
        )),
        ("REMOTE ACCESS", (
            "Please install AnyDesk on your phone and share your screen "
            "so we can fix the issue. Don't hang up."
        )),
        ("NORMAL CALL", (
            "Hey, are we still meeting for lunch tomorrow? I'll book the table "
            "for one o'clock. Let me know if that works for you."
        )),
        ("BENIGN EDUCATIONAL", (
            "Digital arrest scams are increasing in India. Yesterday I read "
            "about how scammers impersonate police and ask people to share OTP."
        )),
    ]

    for label, transcript in tests:
        print(f"=== {label} ===")
        result = analyse_transcript(transcript)
        print(f"Score: {result['voice_score']}  Level: {result['voice_level']}")
        if result['nlp_intent']:
            print(f"NLP: {result['nlp_intent']} ({result['nlp_confidence']:.0%})")
        if result['is_benign']:
            print("→ BENIGN context detected — score suppressed")
        for sig in result["voice_signals"]:
            src = f"[{sig['source']}]" if 'source' in sig else ""
            print(f"  - [{sig['category']}] {sig['reason']}  "
                  f"(heard: \"{sig['evidence']}\") {src}")
        print()