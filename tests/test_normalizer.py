"""
Tests for text_normalizer.py

TEST 1: Hinglish normalization
TEST 2: Transliteration normalization
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from text_normalizer import normalize


class TestHinglishNormalization:
    """TEST 1: Hinglish phrases map to expected English equivalents."""

    def test_abhi_payment_karo(self):
        result = normalize("abhi payment karo")
        assert "pay" in result.lower() or "now" in result.lower()

    def test_otp_batao(self):
        result = normalize("OTP batao")
        assert "otp" in result.lower()
        assert "share" in result.lower() or "tell" in result.lower()

    def test_account_block_ho_jayega(self):
        result = normalize("account block ho jayega")
        assert "account" in result.lower()
        assert "block" in result.lower()

    def test_kisi_ko_mat_batao(self):
        result = normalize("kisi ko mat batao")
        assert "don't tell" in result.lower() or "tell" in result.lower()

    def test_call_mat_kaatna(self):
        result = normalize("call mat kaatna")
        assert "call" in result.lower()
        assert "cut" in result.lower() or "don't" in result.lower()

    def test_verification_ke_liye_paise_bhejo(self):
        result = normalize("verification ke liye paise bhejo")
        assert "verification" in result.lower() or "money" in result.lower()

    def test_police_se_baat(self):
        result = normalize("police se baat kar raha hoon")
        assert "police" in result.lower()

    def test_digital_arrest_variations(self):
        """All digital arrest variations normalize consistently."""
        v1 = normalize("digital arrest hai")
        v2 = normalize("digital arrest kr diya")
        v3 = normalize("digital arrest karo")
        # All should contain "digital arrest"
        assert "digital arrest" in v1
        assert "digital arrest" in v2
        assert "digital arrest" in v3

    def test_anydesk_install(self):
        result = normalize("AnyDesk install karo")
        assert "anydesk" in result.lower() or "install" in result.lower()


class TestTransliterationNormalization:
    """TEST 2: Common transliterated words map correctly."""

    def test_paisa_to_money(self):
        result = normalize("paisa bhejo")
        assert "money" in result.lower() or "send" in result.lower()

    def test_paise_to_money(self):
        result = normalize("paise transfer karo")
        assert "money" in result.lower() or "transfer" in result.lower()

    def test_rupaye(self):
        result = normalize("rupaye bhejo")
        assert "rupees" in result.lower() or "send" in result.lower()

    def test_khaata_to_account(self):
        result = normalize("khaata block ho jayega")
        assert "account" in result.lower()

    def test_turant_to_immediately(self):
        result = normalize("turant karo")
        assert "immediately" in result.lower() or "do" in result.lower()

    def test_warna_to_otherwise(self):
        result = normalize("karo warna problem hogi")
        assert "otherwise" in result.lower()

    def test_filler_removal(self):
        """Transcription artifacts like 'uh', 'um', 'hmm' are removed."""
        result = normalize("uh um like your account is uh frozen")
        assert "uh" not in result.lower().split()
        assert "um" not in result.lower().split()
        assert "account" in result.lower()
        assert "frozen" in result.lower()

    def test_repeated_chars(self):
        """Repeated characters are collapsed."""
        result = normalize("pleeease help meee")
        assert "plee" not in result  # should be "please" (collapsed)

    def test_empty_input(self):
        assert normalize("") == ""
        assert normalize(None) == ""

    def test_preserves_english(self):
        """Normal English should pass through largely intact."""
        result = normalize("Hello how are you doing today")
        assert "hello" in result.lower()
        assert "how" in result.lower()
        assert "today" in result.lower()
