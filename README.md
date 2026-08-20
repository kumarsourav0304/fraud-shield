# Explainable Real-Time Fraud Shield (Hybrid Architecture)

**SIH / SOA IDEATHON 2026 — Problem Statement S40 (Fintech)**  
*Explainable Real-Time UPI Fraud Detection across Transactions, Voice Phishing, and Social Engineering.*

**Live Demo:** https://fraud-shield-b1h3.onrender.com  
*(Free hosting — cold start may take ~50s on initial load, then sub-second response.)*

---

## Architecture Overview

FraudShield is a **technically validated hybrid prototype** combining deterministic payment-behavior analysis, NLP-driven voice phishing detection, transcript normalization, transaction velocity monitoring, and a four-level policy intervention engine.

```
                                  [ Payment Event ]
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
     [ Transaction Risk Engine ]                       [ Audio / Call Stream ]
  • Device & Location Anomalies                                  │ (Whisper STT)
  • Amount & Odd-Hour Checks                                     ▼
  • Accessibility Service Abuse                        [ Raw Transcript ]
  • Velocity Burst (Count & Volume)                              │
                  │                                              ▼
                  │                                   [ Text Normalizer ]
                  │                              • Hinglish phrase mapping
                  │                              • Transliteration cleanup
                  │                              • Artifact stripping
                  │                                              │
                  │                               ┌──────────────┴──────────────┐
                  │                               ▼                             ▼
                  │                    [ Deterministic Rules ]        [ Calibrated SVM NLP ]
                  │                    • 9 Intent Categories          • TF-IDF (1-3 n-grams)
                  │                    • Exact phrase matching        • Calibrated confidence
                  │                               │                             │
                  │                               └──────────────┬──────────────┘
                  │                                              ▼
                  │                                  [ Hybrid Voice Scorer ]
                  │                               • Corroboration deduplication
                  │                               • BENIGN contextual filter
                  │                                              │
                  └───────────────────────┬──────────────────────┘
                                          ▼
                             [ Multimodal Fusion Engine ]
                               • Payment (0-100) + Voice (weighted 60%)
                               • Double-counting controls
                                          │
                                          ▼
                              [ Risk Policy Layer ]
                   Detection Context Evaluation → Intervention Decision
                       ┌─────────────┬─────────────┬─────────────┐
                       ▼             ▼             ▼             ▼
                   [APPROVE]     [VERIFY]       [WARN]        [BLOCK]
                   (Score 0-14)  (Score 15-29) (Score 30-59) (Score 60+)
                       │             │             │             │
                       └─────────────┴──────┬──────┴─────────────┘
                                            ▼
                           [ Explainable Alert & Override ]
                           • Plain-language breakdown
                           • Per-signal points & evidence
                           • User confirmation / cancellation
                                            │
                                            ▼
                               [ Bank Review Console ]
                           • Outcome feedback loop
                           • HMAC-SHA256 audit trail
```

---

## Key Capabilities

### 1. Transcript Normalization (`text_normalizer.py`)
- Normalizes Hinglish (`"account block ho jayega"` → `"account will be blocked"`, `"abhi payment karo"` → `"pay now"`, `"OTP batao"` → `"share the OTP"`).
- Transliteration normalization (`paisa` / `paise` / `rupaye` → money/rupees, `khaata` → account).
- Strips speech-to-text filler words (`uh`, `um`, `hmm`) and collapses repeated characters.
- Preserves the original raw transcript for the tamper-evident audit trail.

### 2. 9-Class Voice Taxonomy & Calibrated NLP (`nlp_intent.py`, `voice_phishing.py`)
- **Nine Canonical Intent Classes:** `BENIGN`, `AUTHORITY`, `CREDENTIAL`, `THREAT`, `URGENCY`, `PAYMENT_COERCION`, `SECRECY`, `REMOTE_ACCESS`, `DIGITAL_ARREST`.
- **Hybrid Fusion:** Deterministic rule engine for high-precision regex matching + TF-IDF with `CalibratedClassifierCV(LinearSVC)` for probabilistic intent classification.
- **Corroboration Awareness:** When rules and NLP detect the same category, points are not double-counted. When NLP detects a missing category, it contributes as supplementary evidence.
- **BENIGN Contextual Suppression:** Educational discussions (e.g. *"Digital arrest scams are increasing in India"*) or news reports are recognized as `BENIGN` and suppressed from escalating payment risk.

### 3. Transaction Velocity & Screen Control (`risk_engine.py`)
- **Velocity Burst Rules:** Evaluates transaction bursts within rapid time windows (`recent_transaction_burst` for $\ge 3$ rapid transactions; `amount_burst` for cumulative spend exceeding $3\times$ typical maximum).
- **Accessibility Service Detection:** Flags malicious screen-control and remote-access software active during payment.
- **`RecentBurst` API Integration:** Accepted seamlessly via the `/assess` API schema.

### 4. Four-Level Policy Engine (`risk_policy.py`)
Separates **detection** from **intervention**:
- `APPROVE` (Score 0–14): Low risk; normal flow.
- `VERIFY` (Score 15–29): Minor anomaly (e.g., first-time payee); recommends quick review or 2FA step.
- `WARN` (Score 30–59): Strong coercion or behavioral anomaly; clear plain-language warning.
- `BLOCK` (Score 60+): High-risk multimodal corroboration; friction-escalated confirmation required.

---

## Live Performance & Benchmarks

| Metric | Transaction Engine (600 Txs) | Voice Adversarial Corpus (92 Samples) |
|---|---|---|
| **Precision** | **94.0%** | **95.45%** |
| **Recall** | **94.0%** | **29.2%** *(Single-sentence boundary samples)* |
| **False-Positive Rate** | **0.7%** *(< 1%)* | **5.00%** |
| **Average Latency** | **< 1.0 ms** | **11.36 ms** |

*All metrics are computed live from actual model execution, not hardcoded.*

---

## Navigation & Web Interfaces

| Route | Page | Purpose |
|---|---|---|
| `/` | **Payment Shield** | Live payment simulation, multimodal fusion lane, NLP intent badge, explainability breakdown, user confirmation flow. |
| `/demo.html` | **Live Demo** | Counterfactual comparison (same payment with vs. without scam call) and 7-step interactive attack simulation. |
| `/stats.html` | **Live Performance** | Real-time computed precision, recall, and confusion matrix across the full dataset. |
| `/review.html` | **Bank Review Console** | Audit logs with outcome review loop (`CONFIRMED_FRAUD`, `CONFIRMED_LEGITIMATE`, `PENDING`). |

---

## Running Locally

### Prerequisites
- Python 3.10+
- macOS / Linux / Windows

### Setup
```bash
# 1. Clone repository
cd fraud-shield

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate transaction dataset & train intent model
python generate_data.py
python train_intent_model.py

# 5. Run the server
uvicorn app:app --reload
```
Open **http://127.0.0.1:8000** in your browser.

---

## Running Tests & Benchmarks

```bash
# Run full 62-test regression suite
pytest tests/ -v

# Run transaction risk engine benchmark
python score_all.py

# Run voice phishing adversarial benchmark
python evaluate_voice.py
```

---

## Privacy & Security Manifest

- **HMAC-SHA256 Pseudonymization:** Device IDs and payee handles are pseudonymized using keyed HMAC before scoring. Raw identifiers never leave the device.
- **On-Device Whisper STT:** Audio is transcribed locally in-memory; raw audio files are discarded immediately after transcription.
- **Formulas & XSS Defenses:** CSV audit logs neutralize formula injection (`=`, `+`, `-`, `@`); web console sanitizes and escapes all user-supplied inputs before rendering.

---

## Limitations

- **Technically Validated Prototype:** Designed as an explainable demonstration of multimodal fraud detection for hackathon evaluation and technical validation, not production banking deployment.
- **Synthetic Transactions:** Historical baseline uses synthetic transactions designed to simulate realistic payment variance.
- **Speech Models:** Whisper base/tiny models are optimized for lightweight CPU execution.