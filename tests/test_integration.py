"""
Integration tests for the FastAPI application.

TEST 7: Multimodal fusion and intervention policy
TEST 8: /assess endpoint preserves recent_burst
+ API validation, privacy behavior
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from app import app


client = TestClient(app)


class TestMultimodalFusionPolicy:
    """TEST 7: Multimodal fusion and intervention policy."""

    def test_normal_payment_approves(self):
        """Normal payment with no call -> APPROVE."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "APPROVE"
        assert data["score"] == 0

    def test_scam_call_escalates(self):
        """Normal-ish payment + scam call -> should escalate."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 2000, "payee": "new_payee_xyz",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00",
            "transcript": (
                "Hello, I am calling from your bank. Your account is frozen. "
                "This is urgent, you must act now. Do not tell anyone. "
                "Share the OTP and make a payment to unblock."
            )
        })
        assert resp.status_code == 200
        data = resp.json()
        # Score should be elevated due to voice signals
        assert data["score"] > 0
        assert data["voice_score"] > 0
        # Decision should be at least WARN or BLOCK
        assert data["decision"] in ("WARN", "BLOCK")

    def test_normal_call_no_escalation(self):
        """Normal payment + normal call -> should stay APPROVE or low."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00",
            "transcript": "Hey, are we still meeting for lunch tomorrow?"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["decision"] == "APPROVE"
        assert data["voice_score"] == 0

    def test_four_intervention_levels(self):
        """All four intervention levels should be possible."""
        # APPROVE: normal payment
        r1 = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        assert r1.json()["decision"] == "APPROVE"

        # BLOCK: extreme transaction + scam call
        r2 = client.post("/assess", json={
            "user_id": "U001", "amount": 49999, "payee": "unknown_9284",
            "city": "Chennai", "device": "DEV-NEW-77",
            "timestamp": "2026-08-15 02:30:00",
            "transcript": (
                "I am from your bank. Share the OTP immediately. "
                "Don't tell anyone. Transfer the amount now."
            )
        })
        assert r2.json()["decision"] == "BLOCK"

    def test_policy_field_present(self):
        """Response should include the new policy field."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        data = resp.json()
        assert "policy" in data
        assert "decision" in data["policy"]
        assert "reasons" in data["policy"]
        assert "corroborated" in data["policy"]

    def test_nlp_field_present(self):
        """Response should include the NLP intent field."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00",
            "transcript": "Share the OTP now."
        })
        data = resp.json()
        assert "nlp" in data
        assert "intent" in data["nlp"]
        assert "confidence" in data["nlp"]


class TestRecentBurstAPI:
    """TEST 8: /assess endpoint preserves recent_burst through full pipeline."""

    def test_recent_burst_accepted(self):
        """recent_burst field should be accepted by the API."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 500, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": "",
            "recent_burst": {
                "transaction_count": 6,
                "cumulative_amount": 18000,
                "time_window_minutes": 10,
                "average_amount": 3000,
                "max_amount": 5000,
                "burst_detected": True
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        # Velocity signals should appear
        signal_codes = [s["code"] for s in data.get("signals", [])]
        assert "recent_transaction_burst" in signal_codes

    def test_without_recent_burst(self):
        """Requests without recent_burst should still work (backward compat)."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        assert resp.status_code == 200
        data = resp.json()
        signal_codes = [s["code"] for s in data.get("signals", [])]
        assert "recent_transaction_burst" not in signal_codes

    def test_burst_not_detected_no_signal(self):
        """burst_detected=False should not fire velocity rules."""
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 500, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": "",
            "recent_burst": {
                "transaction_count": 2,
                "cumulative_amount": 500,
                "time_window_minutes": 30,
                "burst_detected": False
            }
        })
        assert resp.status_code == 200
        data = resp.json()
        signal_codes = [s["code"] for s in data.get("signals", [])]
        assert "recent_transaction_burst" not in signal_codes


class TestAPICompatibility:
    """Backward-compatible API response fields."""

    def test_all_required_fields_present(self):
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        data = resp.json()
        required_fields = [
            "score", "decision", "reasons", "payment_score",
            "voice_score", "voice_level", "signals", "raw_score",
            "capped", "confidence", "display", "thresholds",
            "privacy", "latency_ms"
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_privacy_manifest_present(self):
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": 250, "payee": "merchant_grocery",
            "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00", "transcript": ""
        })
        privacy = resp.json()["privacy"]
        assert "transmitted" in privacy
        assert "never_transmitted" in privacy
        # HMAC fingerprints should be present
        transmitted = {t["field"]: t for t in privacy["transmitted"]}
        assert "payee fingerprint" in transmitted
        assert "device fingerprint" in transmitted

    def test_capabilities_endpoint(self):
        resp = client.get("/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "transcription" in data
        assert "nlp" in data

    def test_stats_endpoint(self):
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "precision" in data
        assert "recall" in data
        assert "total" in data

    def test_validation_rejects_bad_input(self):
        resp = client.post("/assess", json={
            "user_id": "U001", "amount": -100,
            "payee": "test", "city": "Ranchi", "device": "DEV-A1",
            "timestamp": "2026-08-15 14:30:00"
        })
        assert resp.status_code == 422
