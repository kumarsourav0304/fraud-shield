"""
risk_policy.py
The policy layer that separates DETECTION from INTERVENTION.

Detection modules (transaction rules, voice rules, NLP classifier) produce
signals. This module evaluates the combined context and determines the
appropriate intervention level:

    APPROVE  ->  no meaningful suspicious evidence
    VERIFY   ->  suspicious but insufficiently corroborated
    WARN     ->  strong social-engineering evidence
    BLOCK    ->  strong corroborating multimodal evidence

The key principle: detecting suspicious language does NOT automatically
mean BLOCK. The final intervention depends on corroborating context
from independent signal sources.
"""


# Thresholds for the four intervention levels
# Kept consistent with the existing WARN_AT=30, BLOCK_AT=60 in risk_engine.py.
# VERIFY slots between APPROVE and WARN as a new intermediate level.
APPROVE_MAX = 14        # 0-14: approve
VERIFY_MAX = 29         # 15-29: verify (new level)
WARN_MAX = 59           # 30-59: warn  (matches existing WARN_AT=30)
# 60+: block            (matches existing BLOCK_AT=60)


def evaluate_risk_context(tx_signals, voice_result, score):
    """
    Evaluate the combined risk context from all detection sources.

    Returns a context dict with:
        - signal_sources: set of independent source types that fired
        - is_multimodal: True if both transaction and voice signals fired
        - is_corroborated: True if multiple independent sources agree
        - is_benign: True if voice context is benign/educational
        - independent_signal_count: number of truly independent signals
        - policy_reasons: list of human-readable policy explanations
    """
    # Identify which independent sources have fired
    signal_sources = set()
    policy_reasons = []

    # Transaction signals
    tx_signal_count = 0
    if tx_signals:
        for sig in tx_signals:
            code = sig.get("code", "")
            if not code.startswith("voice_"):
                tx_signal_count += 1
                signal_sources.add("transaction")

    # Voice / NLP signals
    voice_signals = voice_result.get("voice_signals", []) if voice_result else []
    voice_score = voice_result.get("voice_score", 0) if voice_result else 0
    nlp_intent = voice_result.get("nlp_intent") if voice_result else None
    nlp_confidence = voice_result.get("nlp_confidence", 0) if voice_result else 0
    is_benign = voice_result.get("is_benign", False) if voice_result else False

    rule_signals = [s for s in voice_signals if s.get("source") == "rule"]
    nlp_signals = [s for s in voice_signals if s.get("source") == "nlp"]

    if rule_signals:
        signal_sources.add("voice_rule")
    if nlp_signals:
        signal_sources.add("voice_nlp")
    if nlp_intent and nlp_intent not in ("BENIGN", "UNAVAILABLE", None):
        signal_sources.add("voice_nlp")

    is_multimodal = "transaction" in signal_sources and (
        "voice_rule" in signal_sources or "voice_nlp" in signal_sources
    )

    # Check for NLP+rule corroboration on same category
    rule_categories = {s["category"] for s in rule_signals}
    nlp_corroborated = any(s.get("nlp_corroborated") for s in voice_signals)

    is_corroborated = (
        is_multimodal
        or nlp_corroborated
        or len(signal_sources) >= 2
    )

    # Count independent signals
    independent_count = tx_signal_count + len(voice_signals)

    # Build policy reasons
    if is_benign:
        policy_reasons.append(
            "Voice context appears educational/informational, not a live scam"
        )

    if is_multimodal:
        policy_reasons.append(
            "Both transaction behaviour and voice signals indicate risk"
        )

    if nlp_corroborated:
        policy_reasons.append(
            "NLP classification corroborates deterministic rule evidence"
        )

    if nlp_intent and nlp_intent not in ("BENIGN", "UNAVAILABLE", None):
        if nlp_confidence >= 0.8:
            policy_reasons.append(
                f"High-confidence NLP intent: {nlp_intent} ({nlp_confidence:.0%})"
            )
        elif nlp_confidence >= 0.5:
            policy_reasons.append(
                f"NLP detected {nlp_intent} intent ({nlp_confidence:.0%} confident)"
            )

    if tx_signal_count >= 3:
        policy_reasons.append(
            f"{tx_signal_count} independent transaction anomalies detected"
        )

    # Check for high-severity categories
    high_severity_cats = {"DIGITAL_ARREST", "CREDENTIAL", "REMOTE_ACCESS"}
    detected_high = rule_categories & high_severity_cats
    if detected_high:
        policy_reasons.append(
            f"High-severity pattern detected: {', '.join(sorted(detected_high))}"
        )

    return {
        "signal_sources": signal_sources,
        "is_multimodal": is_multimodal,
        "is_corroborated": is_corroborated,
        "is_benign": is_benign,
        "independent_signal_count": independent_count,
        "policy_reasons": policy_reasons,
        "nlp_intent": nlp_intent,
        "nlp_confidence": nlp_confidence,
    }


def determine_intervention(score, context):
    """
    Determine the appropriate intervention level.

    Uses both the numeric risk score AND the qualitative context to decide.
    Context can upgrade or downgrade the intervention relative to what the
    score alone would suggest.

    Returns:
        (intervention, policy_reasons)
        intervention: "APPROVE" | "VERIFY" | "WARN" | "BLOCK"
        policy_reasons: list of str explaining the decision
    """
    reasons = list(context.get("policy_reasons", []))

    # --- BENIGN suppression ---
    if context.get("is_benign"):
        if score <= WARN_MAX:
            reasons.append("Score suppressed due to benign/educational context")
            return "APPROVE", reasons

    # --- Score-based baseline ---
    if score >= WARN_MAX + 1:  # 60+
        intervention = "BLOCK"
    elif score >= VERIFY_MAX + 1:  # 40-59
        intervention = "WARN"
    elif score >= APPROVE_MAX + 1:  # 20-39
        intervention = "VERIFY"
    else:  # 0-19
        intervention = "APPROVE"

    # --- Context-based adjustments ---

    # Upgrade: multimodal corroboration can escalate VERIFY -> WARN
    if intervention == "VERIFY" and context.get("is_multimodal"):
        intervention = "WARN"
        reasons.append(
            "Upgraded to WARN: multimodal evidence from independent sources"
        )

    # Upgrade: high-severity category with corroboration -> at least WARN
    if intervention in ("VERIFY", "APPROVE"):
        nlp_intent = context.get("nlp_intent")
        nlp_conf = context.get("nlp_confidence", 0)
        if nlp_intent in ("DIGITAL_ARREST", "CREDENTIAL") and nlp_conf >= 0.7:
            if context.get("is_corroborated"):
                intervention = "WARN"
                reasons.append(
                    f"Upgraded to WARN: high-confidence {nlp_intent} with corroboration"
                )

    # Upgrade: very strong multimodal + high-severity -> BLOCK
    if intervention == "WARN" and context.get("is_multimodal"):
        if context.get("independent_signal_count", 0) >= 4:
            intervention = "BLOCK"
            reasons.append(
                "Upgraded to BLOCK: strong multimodal evidence with 4+ independent signals"
            )

    # Downgrade: isolated weak signal should not WARN
    if intervention == "WARN":
        if context.get("independent_signal_count", 0) <= 1 and not context.get("is_corroborated"):
            intervention = "VERIFY"
            reasons.append(
                "Downgraded to VERIFY: only 1 signal, insufficient corroboration"
            )

    if not reasons:
        if intervention == "APPROVE":
            reasons.append("No meaningful suspicious evidence detected")
        elif intervention == "VERIFY":
            reasons.append("Unusual signal detected — recommend secondary verification")
        elif intervention == "WARN":
            reasons.append("Strong social-engineering evidence detected")
        elif intervention == "BLOCK":
            reasons.append("Strong corroborating multimodal evidence")

    return intervention, reasons
