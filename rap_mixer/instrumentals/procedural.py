from __future__ import annotations

import hashlib

import numpy as np


def generated_beat(bpm: float, bars: int, sample_rate: int = 48000, *,
                   profile: dict | None = None, seed_text: str = ""):
    """Create a musical, royalty-free backing bed locally from synthesis primitives."""
    bpm = max(30, min(300, float(bpm)))
    duration = bars * 4 * 60 / bpm
    size = int(duration * sample_rate)
    y = np.zeros(size, dtype=np.float32)
    beat_samples = sample_rate * 60 / bpm
    profile = profile or {}
    energy = float(profile.get("energy", 0.55))
    density = float(profile.get("drum_density", 0.55))
    groove = str(profile.get("groove", "straight")).lower()
    seed_material = f"{seed_text}|{profile}|{bpm:.2f}|{bars}"
    seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    root_midi = 36 + seed % 12
    progression = rng.choice([0, 3, 5, 7, 8, 10], size=4, replace=True)
    bass_notes = [440 * 2 ** ((root_midi + int(step) - 69) / 12) for step in progression]
    swing = 0.08 if any(x in groove for x in ("swing", "shuffle", "boom")) else 0.0

    def add(start: int, signal: np.ndarray):
        end = min(size, start + signal.size)
        if end > start:
            y[start:end] += signal[:end - start]

    for beat in range(bars * 4):
        start = int(beat * beat_samples)
        kick_n = min(int(sample_rate * 0.24), size - start)
        kick_pattern = beat % 4 in ({0, 2, 3} if density > 0.68 else {0, 2})
        if kick_n > 0 and kick_pattern:
            t = np.arange(kick_n) / sample_rate
            phase = 2 * np.pi * (75 * t - 28 * t**2)
            add(start, (0.38 + 0.28 * energy) * np.sin(phase) * np.exp(-t * 18))
        if beat % 4 in {1, 3}:
            snare_n = min(int(sample_rate * 0.16), size - start)
            if snare_n > 0:
                noise_rng = np.random.default_rng(seed + beat + bars * 101)
                t = np.arange(snare_n) / sample_rate
                noise = noise_rng.standard_normal(snare_n).astype(np.float32)
                add(start, (0.1 + 0.12 * energy) * noise * np.exp(-t * 28))
        bass_n = min(int(beat_samples * 0.8), size - start)
        if bass_n > 0:
            t = np.arange(bass_n) / sample_rate
            frequency = bass_notes[(beat // 4) % len(bass_notes)]
            add(start, 0.2 * np.sin(2 * np.pi * frequency * t) * np.exp(-t * 2.2))
        hat_steps = (0, 0.25, 0.5, 0.75) if density > 0.72 else (0, 0.5)
        for half in hat_steps:
            shifted = half + (swing if half in {0.25, 0.75} else 0)
            hat_start = start + int(shifted * beat_samples)
            hat_n = min(int(sample_rate * 0.045), max(0, size - hat_start))
            if hat_n > 0:
                hat_rng = np.random.default_rng(seed + beat * 7 + int(half * 100) + 3)
                t = np.arange(hat_n) / sample_rate
                add(hat_start, (0.035 + 0.04 * energy)
                    * hat_rng.standard_normal(hat_n) * np.exp(-t * 85))
    peak = float(np.max(np.abs(y))) if y.size else 0
    if peak:
        y *= 0.8 / peak
    return sample_rate, y
