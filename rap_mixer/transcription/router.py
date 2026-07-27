from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from rap_mixer.providers.openai_audio import transcribe_numpy


def transcribe_selected(provider: str, audio: tuple[int, np.ndarray], openai_client=None,
                        openai_model: str = "gpt-4o-mini-transcribe") -> tuple[str, str]:
    if provider == "OpenAI transcription API":
        if openai_client is None:
            raise ValueError("OpenAI transcription requires a configured credential and consent.")
        return transcribe_numpy(openai_client, audio, openai_model), f"OpenAI · {openai_model}"
    sr, data = audio
    path = tempfile.NamedTemporaryFile(prefix="rap-local-asr-", suffix=".wav", delete=False).name
    try:
        sf.write(path, data, sr, subtype="PCM_16")
        if provider == "faster-whisper":
            from rap_mixer.transcription.faster_whisper import FasterWhisperTranscriber
            model = os.getenv("WHISPER_MODEL", "small.en")
            return FasterWhisperTranscriber(model).transcribe_file(path), f"faster-whisper · {model}"
        if provider == "transformers-whisper":
            from rap_mixer.transcription.transformers_whisper import TransformersWhisperTranscriber
            model = os.getenv("TRANSFORMERS_WHISPER_MODEL", "openai/whisper-small.en")
            return TransformersWhisperTranscriber(model).transcribe_file(path), f"Transformers · {model}"
        raise ValueError(f"Unsupported transcription provider: {provider}")
    finally:
        Path(path).unlink(missing_ok=True)
