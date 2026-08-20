"""
text_normalizer.py
Transcript normalization for the Fraud Shield.

Converts Hinglish, transliterated Hindi, informal abbreviations, and
common speech-to-text artifacts into normalized English so the downstream
detectors (both deterministic rules and the NLP classifier) see a
consistent surface form.

The original transcript is NEVER replaced in the evidence trail — this
module only produces a cleaned copy for internal scoring.
"""

import re

# ---------------------------------------------------------------------------
# HINGLISH / TRANSLITERATED PHRASE MAP
#
# Each key is a lowercase pattern (may be regex); its value is the English
# normalised form.  Ordered longest-first so longer phrases match before
# their shorter substrings.
# ---------------------------------------------------------------------------
_PHRASE_MAP = [
    # --- Digital arrest / authority ---
    (r"\bdigital arrest kr diya\b", "digital arrest"),
    (r"\bdigital arrest karo\b", "digital arrest"),
    (r"\bdigital arrest hai\b", "digital arrest"),
    (r"\bdigital arrest mein\b", "under digital arrest"),
    (r"\bgirftar kar liya\b", "you have been arrested"),
    (r"\bgirftaar\b", "arrested"),
    (r"\bfir darj\b", "fir filed"),
    (r"\bfir likhi hai\b", "fir filed"),
    (r"\bcourt ka order hai\b", "court order"),
    (r"\bcourt se order aaya hai\b", "court order issued"),
    (r"\bsupreme court ka order\b", "supreme court order"),
    (r"\bhigh court ka notice\b", "high court notice"),
    (r"\bwarrant nikla hai\b", "warrant issued"),
    (r"\bjail ho jaoge\b", "you will go to jail"),
    (r"\bjail bhej denge\b", "we will send you to jail"),

    # --- Authority impersonation ---
    (r"\bbank se bol raha hoon\b", "calling from the bank"),
    (r"\bbank se baat kar raha hoon\b", "calling from the bank"),
    (r"\bbank ki taraf se\b", "from the bank"),
    (r"\bpolice se baat kar raha hoon\b", "police is calling"),
    (r"\bpolice ki taraf se\b", "from the police"),
    (r"\bcbi officer hoon\b", "i am a cbi officer"),
    (r"\bcbi se bol raha\b", "calling from cbi"),
    (r"\bed officer\b", "ed officer"),
    (r"\bcyber cell se\b", "from the cyber cell"),
    (r"\bcyber crime se\b", "from cyber crime"),
    (r"\bincome tax department\b", "income tax department"),
    (r"\btelecom authority\b", "telecom authority"),
    (r"\btrai se\b", "from trai"),
    (r"\brbi se\b", "from rbi"),
    (r"\badhikari hoon\b", "i am an officer"),
    (r"\bsarkari\b", "government"),

    # --- Urgency / pressure ---
    (r"\babhi turant\b", "right now immediately"),
    (r"\babhi karo\b", "do it now"),
    (r"\babhi payment karo\b", "pay now"),
    (r"\bjaldi karo\b", "hurry up"),
    (r"\bjaldi se karo\b", "hurry up"),
    (r"\bder mat karo\b", "do not delay"),
    (r"\bsamay khatam ho raha hai\b", "time is running out"),
    (r"\btime khatam\b", "time is up"),
    (r"\blast chance hai\b", "this is your last chance"),
    (r"\baakhri mauka\b", "last chance"),
    (r"\baaj hi karna hoga\b", "must do it today"),
    (r"\babhi nahi toh\b", "if not now then"),
    (r"\bturant\b", "immediately"),

    # --- Account / blocking threats ---
    (r"\baccount block ho jayega\b", "account will be blocked"),
    (r"\baccount band ho jayega\b", "account will be closed"),
    (r"\baccount freeze ho jayega\b", "account will be frozen"),
    (r"\baccount suspend\b", "account suspended"),
    (r"\baccount se paisa kat jayega\b", "money will be deducted from account"),
    (r"\bsim block ho jayega\b", "your sim will be blocked"),
    (r"\bsim band\b", "sim blocked"),
    (r"\bkyc expire ho gaya\b", "kyc expired"),
    (r"\bkyc update karo\b", "update kyc"),
    (r"\bkyc pending hai\b", "kyc is pending"),
    (r"\bkyc verify karo\b", "verify kyc"),

    # --- Credential requests ---
    (r"\botp batao\b", "share the otp"),
    (r"\botp bata do\b", "share the otp"),
    (r"\botp bhejo\b", "send the otp"),
    (r"\botp share karo\b", "share the otp"),
    (r"\botp dijiye\b", "please share the otp"),
    (r"\bpin batao\b", "share your pin"),
    (r"\bpin bata do\b", "share your pin"),
    (r"\bpassword batao\b", "share your password"),
    (r"\bpassword dijiye\b", "please share your password"),
    (r"\bcvv batao\b", "share your cvv"),
    (r"\bcvv number\b", "cvv number"),

    # --- Payment coercion ---
    (r"\bpayment karo\b", "make a payment"),
    (r"\bpaise bhejo\b", "send money"),
    (r"\bpaisa bhejo\b", "send money"),
    (r"\bpaisay bhejo\b", "send money"),
    (r"\bpaise transfer karo\b", "transfer money"),
    (r"\bpaisa transfer karo\b", "transfer money"),
    (r"\bramount bhejo\b", "send the amount"),
    (r"\bramount transfer karo\b", "transfer the amount"),
    (r"\bverification ke liye paise bhejo\b", "send money for verification"),
    (r"\bverification fee\b", "verification fee"),
    (r"\bprocessing fee bhejo\b", "send processing fee"),
    (r"\brefund milega\b", "you will get a refund"),
    (r"\brefund ke liye\b", "for a refund"),
    (r"\bqr code scan karo\b", "scan the qr code"),
    (r"\bqr scan karo\b", "scan the qr code"),

    # --- Secrecy / isolation ---
    (r"\bkisi ko mat batao\b", "don't tell anyone"),
    (r"\bkisi ko mat batana\b", "don't tell anyone"),
    (r"\bkisi se mat bolo\b", "don't tell anyone"),
    (r"\bkisi ko bhi nahi\b", "don't tell anyone"),
    (r"\bcall mat kaatna\b", "don't cut the call"),
    (r"\bcall mat kaato\b", "don't cut the call"),
    (r"\bphone mat rakhna\b", "don't hang up"),
    (r"\bphone mat rakho\b", "don't hang up"),
    (r"\bcall pe raho\b", "stay on the call"),
    (r"\bline pe raho\b", "stay on the line"),
    (r"\bghar walo ko mat batao\b", "don't tell your family"),
    (r"\bfamily ko mat batao\b", "don't tell your family"),
    (r"\byeh confidential hai\b", "this is confidential"),
    (r"\bsecret rakhna\b", "keep it secret"),

    # --- Remote access ---
    (r"\banydesk install karo\b", "install anydesk"),
    (r"\bteamviewer install karo\b", "install teamviewer"),
    (r"\bapp download karo\b", "download this app"),
    (r"\bscreen share karo\b", "share your screen"),
    (r"\bremote access do\b", "give remote access"),
    (r"\bapna phone de do\b", "give your phone"),

    # --- Common word-level transliterations ---
    (r"\bpaisa\b", "money"),
    (r"\bpaise\b", "money"),
    (r"\bpaisay\b", "money"),
    (r"\brupaye\b", "rupees"),
    (r"\brupaya\b", "rupees"),
    (r"\bkhaata\b", "account"),
    (r"\bkhata\b", "account"),
    (r"\bbhejo\b", "send"),
    (r"\bbhejiye\b", "please send"),
    (r"\bbhej do\b", "send"),
    (r"\bkaro\b", "do"),
    (r"\bkariye\b", "please do"),
    (r"\bkar do\b", "do"),
    (r"\bkar dijiye\b", "please do"),
    (r"\bbatao\b", "tell"),
    (r"\bbata do\b", "tell"),
    (r"\bbataye\b", "tell"),
    (r"\bdijiye\b", "please give"),
    (r"\bde do\b", "give"),
    (r"\bnahi toh\b", "otherwise"),
    (r"\bwarna\b", "otherwise"),
    (r"\bnahi to\b", "otherwise"),
    (r"\babhi\b", "now"),
    (r"\bforan\b", "immediately"),
]

# Pre-compile all patterns for speed
_COMPILED_PHRASES = [(re.compile(pat, re.IGNORECASE), repl)
                     for pat, repl in _PHRASE_MAP]


# ---------------------------------------------------------------------------
# BASIC TEXT CLEANUP
# ---------------------------------------------------------------------------
_REPEATED_CHAR = re.compile(r"(.)\1{2,}")          # aaaa -> aa
_MULTI_SPACE = re.compile(r"\s+")                   # collapse whitespace
_PUNCTUATION_JUNK = re.compile(r"[^\w\s₹@.,!?;:'\"-]")  # remove odd unicode
_TRANSCRIPTION_ARTIFACTS = re.compile(
    r"\b(uh+|um+|hmm+|ah+|er+|like|you know)\b", re.IGNORECASE
)


def normalize(text: str) -> str:
    """
    Normalize a transcript for downstream fraud detection.

    Steps:
      1. Lowercase
      2. Strip transcription filler words (uh, um, hmm, etc.)
      3. Collapse repeated characters (pleeease -> please)
      4. Normalize whitespace
      5. Apply Hinglish / transliteration phrase map
      6. Clean up residual punctuation artifacts

    The original transcript should be preserved separately for evidence.
    """
    if not text:
        return ""

    t = text.lower().strip()

    # Remove common transcription artifacts / filler words
    t = _TRANSCRIPTION_ARTIFACTS.sub("", t)

    # Collapse repeated characters (e.g. pleeease -> please)
    t = _REPEATED_CHAR.sub(r"\1", t)

    # Normalize whitespace
    t = _MULTI_SPACE.sub(" ", t).strip()

    # Remove unusual punctuation (keep standard set)
    t = _PUNCTUATION_JUNK.sub("", t)

    # Apply Hinglish / transliteration mappings
    for pattern, replacement in _COMPILED_PHRASES:
        t = pattern.sub(replacement, t)

    # Final whitespace cleanup
    t = _MULTI_SPACE.sub(" ", t).strip()

    return t


# Quick self-test
if __name__ == "__main__":
    tests = [
        "Abhi payment karo, OTP batao, call mat kaatna",
        "Aapka account block ho jayega agar turant paise nahi bheje",
        "Police se baat kar raha hoon, digital arrest kr diya hai",
        "Yesterday I read about digital arrest scams in India",
        "Sir aap abhi payment kar do warna account block ho jayega",
        "Ummm like your KYC is uh pending and you need to update it",
        "Verification ke liye paise bhejo, refund milega",
        "Install AnyDesk karo and screen share karo",
    ]
    for t in tests:
        print(f"IN:  {t}")
        print(f"OUT: {normalize(t)}")
        print()
