from __future__ import annotations

import tempfile

import numpy as np
import soundfile as sf


def mix_voice_and_backing(voice_path: str, backing, vocal_gain: float = 0.82,
                          backing_gain: float = 0.62) -> str:
    voice, sr = sf.read(voice_path, dtype="float32")
    if voice.ndim > 1:
        voice = voice.mean(axis=1)
    backing_sr, music = backing
    music = np.asarray(music, dtype=np.float32)
    if music.ndim > 1:
        music = music.mean(axis=1)
    if backing_sr != sr and music.size:
        source = np.arange(len(music)) / backing_sr
        target = np.arange(round(len(music) * sr / backing_sr)) / sr
        music = np.interp(target, source, music).astype("float32")
    # Match perceived loudness before applying the user-facing balance. Raw provider output
    # levels vary widely, which previously made generated/MRT2 beds nearly inaudible.
    vocal_active = voice[np.abs(voice) > 1e-4]
    vocal_rms = float(np.sqrt(np.mean(vocal_active ** 2))) if vocal_active.size else .1
    music_rms = float(np.sqrt(np.mean(music ** 2))) if music.size else 0
    if music_rms > 1e-6:
        music = music * min(8.0, vocal_rms / music_rms)
    length = max(len(voice), len(music))
    vocal = np.pad(voice, (0, length - len(voice)))
    bed = np.resize(music, length) if music.size else np.zeros(length, dtype="float32")
    mixed = vocal * vocal_gain + bed * backing_gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0
    if peak > 0.98:
        mixed *= 0.98 / peak
    target_path = tempfile.NamedTemporaryFile(prefix="rap-ai-mix-", suffix=".wav", delete=False).name
    sf.write(target_path, mixed, sr, subtype="PCM_16")
    return target_path
