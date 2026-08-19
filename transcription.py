"""
transcription.py
Speech-to-text for the Fraud Shield.

Turns real call audio into text so the existing voice_phishing.py engine
can scan it for coercion patterns. Nothing about the fraud logic changes -
this only replaces "user pastes a transcript" with "user supplies audio".

Uses faster-whisper (CTranslate2 build of OpenAI's Whisper). Chosen over the
reference implementation because it needs no PyTorch, which keeps the whole
stack small enough to deploy on a free-tier host.

The model is loaded lazily and failure is non-fatal: if it cannot load, the
app keeps running and the UI falls back to the preset transcripts. A demo
that degrades is better than a demo that crashes.
"""

import os
import tempfile
import threading
import time

# Model size. "tiny" is ~75MB and fast enough for short call clips.
# "base" (~145MB) is more accurate but heavier on a small host.
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "tiny")

# Audio longer than this is rejected rather than tying up the server.
MAX_AUDIO_BYTES = 25 * 1024 * 1024      # 25 MB

_model = None
_load_error = None
_lock = threading.Lock()


def _load_model():
    """
    Load the Whisper model once, on first use.

    Loading lazily rather than at import means a slow or failed download
    never prevents the server from starting - the rest of the shield stays
    available either way.
    """
    global _model, _load_error

    if _model is not None or _load_error is not None:
        return _model

    with _lock:
        if _model is not None or _load_error is not None:
            return _model
        try:
            from faster_whisper import WhisperModel
            _model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",      # 4x smaller than float32, ample for speech
                cpu_threads=2,
            )
        except Exception as exc:          # noqa: BLE001 - any failure must be survivable
            _load_error = str(exc)
            _model = None
    return _model


def is_available():
    """True if speech-to-text can actually run right now."""
    return _load_model() is not None


def status():
    """Describes the transcription capability, for the UI to adapt to."""
    available = is_available()
    return {
        "available": available,
        "model": MODEL_SIZE if available else None,
        "engine": "faster-whisper (CTranslate2)" if available else None,
        "error": _load_error if not available else None,
    }


def transcribe_audio(audio_bytes, filename="audio", language=None):
    """
    Convert audio bytes to text.

    audio_bytes: raw contents of an uploaded or recorded file
    filename:    original name, used only to keep the file extension
    language:    ISO code such as "en" or "hi". None lets Whisper detect it.

    Returns a dict with the transcript plus metadata, or an error description.
    Any container ffmpeg understands works - wav, mp3, m4a, webm, ogg.
    """
    if not audio_bytes:
        return {"ok": False, "error": "No audio received."}

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return {"ok": False,
                "error": f"Audio is too large (limit {MAX_AUDIO_BYTES // (1024*1024)} MB)."}

    model = _load_model()
    if model is None:
        return {"ok": False,
                "error": "Speech-to-text is unavailable on this server.",
                "unavailable": True}

    suffix = os.path.splitext(filename)[1] or ".wav"
    tmp_path = None
    started = time.perf_counter()

    try:
        # faster-whisper reads from a path, so the upload is staged on disk
        # and removed immediately afterwards - call audio is never retained.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        segments, info = model.transcribe(
            tmp_path,
            language=language,        # None = auto-detect
            beam_size=1,              # greedy: faster, ample for clear speech
            vad_filter=True,          # skip silence, which speeds things up
            vad_parameters={"min_silence_duration_ms": 500},
        )

        parts = []
        timeline = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            parts.append(text)
            timeline.append({
                "start": round(seg.start, 1),
                "end": round(seg.end, 1),
                "text": text,
            })

        transcript = " ".join(parts).strip()
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        return {
            "ok": True,
            "transcript": transcript,
            "segments": timeline,
            "language": getattr(info, "language", None),
            "language_confidence": round(getattr(info, "language_probability", 0) or 0, 2),
            "audio_seconds": round(getattr(info, "duration", 0) or 0, 1),
            "transcribe_ms": elapsed_ms,
            "model": MODEL_SIZE,
        }

    except Exception as exc:              # noqa: BLE001
        return {"ok": False, "error": f"Could not read that audio file: {exc}"}

    finally:
        # The audio itself is never kept - only the derived text is used.
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass