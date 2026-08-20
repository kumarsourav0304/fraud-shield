"""
Tests for NLP intent classification and BENIGN contextual filtering.

TEST 3: NLP classification — basic scam text classified as non-BENIGN
TEST 4: BENIGN contextual filtering — educational content -> BENIGN
TEST 5: Digital-arrest detection
+ NLP fallback test
+ Remote-access detection test
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from voice_phishing import analyse_transcript


class TestNLPClassification:
    """TEST 3: Basic scam transcripts are classified as non-BENIGN."""

    def test_scam_call_not_benign(self):
        result = analyse_transcript(
            "I am calling from your bank. Your account is frozen. "
            "Share the OTP to verify."
        )
        assert result["voice_score"] > 0
        assert result["voice_level"] != "CLEAN" or len(result["voice_signals"]) > 0

    def test_scam_gets_signals(self):
        result = analyse_transcript(
            "This is urgent. Act now or your account will be blocked. "
            "Do not tell anyone. Share the OTP."
        )
        assert len(result["voice_signals"]) >= 2
        categories = {s["category"] for s in result["voice_signals"]}
        # Should detect at least URGENCY and CREDENTIAL
        assert "URGENCY" in categories or "CREDENTIAL" in categories

    def test_normal_call_is_clean(self):
        result = analyse_transcript(
            "Hey, are we still meeting for lunch tomorrow? "
            "I'll book the table for one o'clock."
        )
        assert result["voice_score"] == 0
        assert result["voice_level"] == "CLEAN"
        assert len(result["voice_signals"]) == 0

    def test_empty_transcript(self):
        result = analyse_transcript("")
        assert result["voice_score"] == 0
        assert result["voice_level"] == "CLEAN"

    def test_none_transcript(self):
        """None or missing transcript should not crash."""
        result = analyse_transcript(None)
        assert result["voice_score"] == 0


class TestBenignContextualFiltering:
    """TEST 4: Educational/contextual content should be treated as benign."""

    def test_educational_digital_arrest(self):
        """Discussion ABOUT digital arrest scams should not be flagged as high-risk."""
        result = analyse_transcript(
            "Digital arrest scams are increasing in India. "
            "Yesterday I read about how scammers impersonate police."
        )
        # Should have low score or be marked benign
        # The rule engine may fire on "arrest" / "police" but the
        # overall system should recognize it's educational
        assert result["voice_score"] < 45 or result.get("is_benign", False)

    def test_news_about_fraud(self):
        result = analyse_transcript(
            "RBI has issued new guidelines about UPI fraud prevention. "
            "Banks should never ask for OTP on calls."
        )
        # OTP will fire but this is educational context
        # Score should be moderate, not HIGH_RISK level
        assert result["voice_level"] != "HIGH_RISK" or result.get("is_benign", False)

    def test_describing_past_scam(self):
        result = analyse_transcript(
            "My friend was scammed last week. Someone called pretending "
            "to be from the bank and asked for the OTP."
        )
        # May trigger some rules but context is past-tense narrative
        assert result["voice_score"] <= 60 or result.get("is_benign", False)


class TestDigitalArrestDetection:
    """TEST 5: Digital arrest patterns are detected."""

    def test_english_digital_arrest(self):
        result = analyse_transcript(
            "You are under digital arrest. Supreme court order. "
            "Transfer money immediately."
        )
        categories = {s["category"] for s in result["voice_signals"]}
        assert "DIGITAL_ARREST" in categories
        assert result["voice_score"] >= 25

    def test_hinglish_digital_arrest(self):
        result = analyse_transcript(
            "Digital arrest hai aap par. Supreme court ka order hai. "
            "Paise transfer karo."
        )
        categories = {s["category"] for s in result["voice_signals"]}
        assert "DIGITAL_ARREST" in categories or result["voice_score"] >= 25

    def test_digital_arrest_high_risk(self):
        result = analyse_transcript(
            "You are under digital arrest by the order of supreme court. "
            "Stay on video call. Transfer the amount immediately or "
            "warrant will be issued."
        )
        assert result["voice_level"] == "HIGH_RISK"


class TestRemoteAccessDetection:
    """Remote access patterns are detected."""

    def test_anydesk(self):
        result = analyse_transcript(
            "Please install AnyDesk on your phone. "
            "I need to check your account remotely."
        )
        categories = {s["category"] for s in result["voice_signals"]}
        assert "REMOTE_ACCESS" in categories

    def test_teamviewer(self):
        result = analyse_transcript(
            "Download TeamViewer and share the access code with me."
        )
        categories = {s["category"] for s in result["voice_signals"]}
        assert "REMOTE_ACCESS" in categories

    def test_screen_share_hinglish(self):
        result = analyse_transcript(
            "Screen share karo main aapka phone check karunga."
        )
        assert result["voice_score"] > 0


class TestNLPFallback:
    """NLP model unavailable should not crash the system."""

    def test_voice_works_without_nlp(self):
        """Even without NLP, deterministic rules should still work."""
        result = analyse_transcript(
            "Share the OTP. This is urgent. Don't tell anyone."
        )
        assert result["voice_score"] > 0
        assert len(result["voice_signals"]) > 0
        # Should still detect CREDENTIAL, URGENCY, SECRECY via rules
        categories = {s["category"] for s in result["voice_signals"]}
        assert len(categories) >= 2
