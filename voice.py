"""
Voice I/O for Hushdoc.

Speech-in:  openai/whisper-base.en   (English-only, CPU)
Speech-out: hexgrad/Kokoro-82M       (English-only, CPU)

Both models are loaded lazily on first use so that users who never enable
voice mode pay no startup cost. They run on CPU on purpose - the GPU is
reserved for the LLM, and these models are small enough that CPU latency
is acceptable (sub-second for short utterances).
"""
from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("voice")

VOICE_LANG_NOTICE = (
    "🌐 Voice features are currently English-only — Whisper-base.en for "
    "speech-in, Kokoro-82M for speech-out. Other languages still work "
    "via the text input."
)

# Lazy singletons. None until first use.
_asr = None   # transformers ASR pipeline
_tts = None   # kokoro KPipeline


def _load_asr():
    """Lazy-load Whisper-base.en on CPU."""
    global _asr
    if _asr is not None:
        return _asr
    from transformers import pipeline
    logger.info("Loading openai/whisper-base.en on CPU (first use)...")
    _asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base.en",
        device="cpu",
        chunk_length_s=30,
        return_timestamps=False,
    )
    logger.info("Whisper-base.en ready.")
    return _asr


def _load_tts():
    """Lazy-load Kokoro-82M on CPU. lang_code='a' = American English."""
    global _tts
    if _tts is not None:
        return _tts
    from kokoro import KPipeline
    logger.info("Loading hexgrad/Kokoro-82M on CPU (first use)...")
    # Explicit device='cpu' so we don't accidentally grab the GPU that
    # llama-server is already holding.
    try:
        _tts = KPipeline(lang_code="a", device="cpu")
    except TypeError:
        # Older kokoro versions don't accept the device kwarg.
        _tts = KPipeline(lang_code="a")
    logger.info("Kokoro-82M ready.")
    return _tts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def transcribe(audio_bytes: bytes) -> str:
    """Convert recorded WAV bytes (e.g. from ``st.audio_input``) to text.

    Whisper-base.en is English-only; non-English speech will be force-
    decoded as nonsense English. The UI surfaces this constraint via
    ``VOICE_LANG_NOTICE``. Empty input yields an empty string.

    We decode the WAV bytes via ``soundfile`` (no ffmpeg dependency) and
    feed the resulting numpy array directly to the ASR pipeline as
    ``{"raw": data, "sampling_rate": 16000}``. Non-16kHz inputs are
    resampled in-process with ``scipy.signal.resample_poly``.
    """
    if not audio_bytes:
        return ""

    import numpy as np
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        logger.exception("soundfile failed to decode the recorded audio.")
        raise

    # Mono-fold any multichannel input.
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Whisper wants 16 kHz mono.
    if sr != 16000:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(int(sr), 16000)
            data = resample_poly(data, 16000 // g, int(sr) // g)
            data = data.astype(np.float32, copy=False)
        except ImportError:
            # Fallback: naive linear interpolation. Good enough for ASR.
            ratio = 16000 / sr
            new_len = int(round(len(data) * ratio))
            old_idx = np.linspace(0, len(data) - 1, num=new_len)
            data = np.interp(old_idx, np.arange(len(data)), data).astype(np.float32)

    asr = _load_asr()
    try:
        result = asr({"raw": data, "sampling_rate": 16000})
        text = (result.get("text") if isinstance(result, dict) else "") or ""
        return text.strip()
    except Exception:
        logger.exception("Whisper transcription failed.")
        raise


def synthesize(text: str, voice: str = "af_heart") -> bytes:
    """Generate WAV (24 kHz mono) bytes for the given English text.

    Returns ``b""`` if generation fails or the input is empty. The default
    voice ``af_heart`` is one of Kokoro's higher-quality American-English
    voices; other voices in Kokoro's pack ('af_bella', 'am_adam', ...) can
    be selected by passing ``voice=``.
    """
    text = (text or "").strip()
    if not text:
        return b""
    try:
        import numpy as np
        import soundfile as sf
        tts = _load_tts()
        chunks = []
        # Kokoro >= 0.9 yields a `Result` dataclass per processed segment
        # with an `.audio` torch.Tensor attribute. Older versions yielded a
        # 3-tuple (graphemes, phonemes, audio). Support both.
        for item in tts(text, voice=voice):
            audio = getattr(item, "audio", None)
            if audio is None and isinstance(item, tuple) and len(item) >= 3:
                audio = item[2]
            if audio is None:
                continue
            try:
                arr = audio.detach().cpu().numpy()
            except AttributeError:
                arr = np.asarray(audio)
            chunks.append(arr.flatten())
        if not chunks:
            return b""
        full = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, full, 24000, format="WAV")
        return buf.getvalue()
    except Exception:
        logger.exception("Kokoro TTS synthesis failed.")
        return b""


def warmup() -> None:
    """Eagerly load both pipelines. Useful to absorb the cold-start cost
    when the user toggles voice mode on, instead of paying it on first
    speak / first answer."""
    try:
        _load_asr()
    except Exception:
        logger.exception("Whisper warmup failed.")
    try:
        _load_tts()
    except Exception:
        logger.exception("Kokoro warmup failed.")
