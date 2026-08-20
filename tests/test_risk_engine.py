"""
Tests for the risk engine — velocity, existing rules, and recent_burst.

TEST 6: Transaction velocity / recent_burst
+ Existing transaction rules regression
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_engine import (
    assess_transaction,
    build_user_profile,
    merge_external_signals,
    RULES,
)


def _make_profile():
    """Build a test profile from known history."""
    history = [
        {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
         "amount": "250", "timestamp": "2026-08-10 14:00:00"},
        {"device": "DEV-A1", "payee": "friend_amit", "city": "Ranchi",
         "amount": "1200", "timestamp": "2026-08-10 15:00:00"},
        {"device": "DEV-A1", "payee": "electricity_board", "city": "Ranchi",
         "amount": "3000", "timestamp": "2026-08-11 10:00:00"},
    ]
    return build_user_profile(history)


class TestVelocityBurst:
    """TEST 6: Transaction velocity / recent_burst signals."""

    def test_burst_detected_fires_rule(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "500", "timestamp": "2026-08-12 14:00:00"}
        burst = {
            "transaction_count": 6,
            "cumulative_amount": 18000,
            "time_window_minutes": 10,
            "average_amount": 3000,
            "max_amount": 5000,
            "burst_detected": True,
        }
        result = assess_transaction(tx, profile, recent_burst=burst)
        fired = result["fired_rules"]
        assert "recent_transaction_burst" in fired
        # Evidence should mention the count and total
        evidence = {s["code"]: s["evidence"] for s in result["signals"]}
        assert "6 transactions" in evidence.get("recent_transaction_burst", "")

    def test_amount_burst_fires(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "500", "timestamp": "2026-08-12 14:00:00"}
        burst = {
            "transaction_count": 5,
            "cumulative_amount": 15000,  # 5x typical max of 3000
            "time_window_minutes": 10,
            "average_amount": 3000,
            "max_amount": 5000,
            "burst_detected": True,
        }
        result = assess_transaction(tx, profile, recent_burst=burst)
        fired = result["fired_rules"]
        assert "amount_burst" in fired

    def test_no_burst_no_velocity_rules(self):
        """Without recent_burst, velocity rules should NOT fire."""
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "500", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        fired = result["fired_rules"]
        assert "recent_transaction_burst" not in fired
        assert "amount_burst" not in fired

    def test_burst_not_detected_no_rules(self):
        """If burst_detected is False, velocity rules don't fire."""
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "500", "timestamp": "2026-08-12 14:00:00"}
        burst = {
            "transaction_count": 2,
            "cumulative_amount": 1000,
            "time_window_minutes": 30,
            "burst_detected": False,
        }
        result = assess_transaction(tx, profile, recent_burst=burst)
        fired = result["fired_rules"]
        assert "recent_transaction_burst" not in fired


class TestExistingRulesRegression:
    """Existing transaction rules must continue working after upgrade."""

    def test_new_device(self):
        profile = _make_profile()
        tx = {"device": "DEV-NEW-99", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert "new_device" in result["fired_rules"]

    def test_unknown_payee(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "scam_payee_123", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert "unknown_payee" in result["fired_rules"]

    def test_away_from_home(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Chennai",
              "amount": "250", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert "away_from_home" in result["fired_rules"]

    def test_large_amount(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "49999", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert "large_amount" in result["fired_rules"]

    def test_odd_hour(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 02:30:00"}
        result = assess_transaction(tx, profile)
        assert "odd_hour" in result["fired_rules"]

    def test_accessibility_abuse(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 14:00:00",
              "accessibility_service_active": True}
        result = assess_transaction(tx, profile)
        assert "accessibility_abuse" in result["fired_rules"]

    def test_normal_transaction_approves(self):
        profile = _make_profile()
        tx = {"device": "DEV-A1", "payee": "merchant_grocery", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert result["decision"] == "APPROVE"
        assert result["score"] == 0

    def test_verify_level_exists(self):
        """VERIFY decision should be possible for medium scores."""
        profile = _make_profile()
        # Unknown payee = 25 points -> should be VERIFY (15-29)
        tx = {"device": "DEV-A1", "payee": "new_payee", "city": "Ranchi",
              "amount": "250", "timestamp": "2026-08-12 14:00:00"}
        result = assess_transaction(tx, profile)
        assert result["decision"] == "VERIFY"

    def test_score_capped_at_100(self):
        profile = _make_profile()
        tx = {"device": "DEV-NEW-99", "payee": "scam_payee", "city": "Chennai",
              "amount": "99999", "timestamp": "2026-08-12 02:30:00",
              "accessibility_service_active": True}
        result = assess_transaction(tx, profile)
        assert result["score"] <= 100
        assert result["capped"] is True

    def test_rules_dict_unchanged(self):
        """Original rule codes must still exist."""
        original_codes = {"new_device", "unknown_payee", "away_from_home",
                          "large_amount", "odd_hour", "accessibility_abuse"}
        assert original_codes.issubset(set(RULES.keys()))
