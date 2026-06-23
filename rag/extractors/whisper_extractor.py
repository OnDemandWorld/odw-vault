"""Whisper extractor — transcribes audio/video via pywhispercpp.

Uses the whisper.cpp C++ backend for fast local transcription.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

# Lazy import — pywhispercpp is optional
_WHISPER = None


def _get_whisper_model():
    global _WHISPER
    if _WHISPER is None:
        from pywhispercpp.model import Model

        _WHISPER = Model
    return _WHISPER


def extract_whisper(
    filepath: str,
    *,
    model_name: str = "large-v3",
    language: str | None = None,
    threads: int = 8,
) -> tuple[str | None, dict, bool, str | None]:
    """Transcribe audio/video via whisper.cpp.

    Converts input to WAV 16kHz mono via ffmpeg, then runs whisper.cpp.
    Returns (text, metadata, succeeded, error_message).
    """
    wav_path = None
    try:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    filepath,
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=True,
            )
        except subprocess.TimeoutExpired:
            return None, {"tool": "whisper"}, False, "ffmpeg conversion timeout"

        Model = _get_whisper_model()
        model = Model(
            model_name, models_dir=Path(__file__).parent.parent.parent / ".rag-cache" / "models"
        )
        segments = model.transcribe(wav_path, threads=threads, language=language)

        text_parts = []
        duration = 0.0
        for seg in segments:
            text_parts.append(seg.text)
            duration = max(duration, seg.t1)

        full_text = " ".join(t.strip() for t in text_parts if t.strip())
        if wav_path:
            os.unlink(wav_path)

        if not full_text.strip():
            return None, {"tool": "whisper", "model": model_name}, False, "no speech detected"

        return (
            full_text,
            {
                "tool": "whisper.cpp",
                "model": model_name,
                "char_count": len(full_text),
                "duration_seconds": round(duration, 1),
            },
            True,
            None,
        )

    except Exception as e:
        if wav_path:
            os.unlink(wav_path)
        return None, {"tool": "whisper", "model": model_name}, False, str(e)
