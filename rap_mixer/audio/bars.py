from __future__ import annotations

import re

import numpy as np

from rap_mixer.analysis.lyrics import syllables
from rap_mixer.analysis.schemas import BarFeatures


def segment_text(text: str, duration: float, bpm: float = 90, beats_per_bar: int = 4) -> list[BarFeatures]:
    phrases = [x.strip() for x in re.split(r"\n+|(?<=[.!?])\s+", text) if x.strip()]
    bar_seconds = 60 / max(1, bpm) * beats_per_bar
    count = max(1, round(duration / bar_seconds)) if duration else max(1, len(phrases))
    words = text.split()
    per = max(1, (len(words) + count - 1) // count)
    bars = []
    for i in range(count):
        part = " ".join(words[i * per:(i + 1) * per])
        if not part and i >= len(phrases):
            continue
        start, end = i * bar_seconds, min(duration or (i + 1) * bar_seconds, (i + 1) * bar_seconds)
        sec = max(0.1, end - start)
        wc = len(part.split())
        sc = sum(syllables(x) for x in part.split())
        bars.append(BarFeatures(number=i + 1, start=start, end=end, transcript=part,
            word_count=wc, syllable_count=sc, words_per_second=wc / sec, syllables_per_second=sc / sec,
            rhyme_endings=(part.split()[-1][-3:] if part else ""), semantic_role=_role(i, count), active=i >= count - 4))
    return bars


def _role(i: int, count: int) -> str:
    if i == count - 1:
        return "Punch"
    return ["Setup", "Claim", "Evidence", "Reframe"][i % 4]


def latest_four_completed(bars: list[BarFeatures]) -> list[BarFeatures]:
    return bars[-4:]


def enrich_bars_audio(bars: list[BarFeatures], sample_rate: int, audio) -> list[BarFeatures]:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    for bar in bars:
        start = max(0, int(bar.start * sample_rate))
        end = min(len(y), int(bar.end * sample_rate))
        chunk = y[start:end]
        if not chunk.size:
            continue
        frame_size = max(64, min(1024, len(chunk)))
        frames = [chunk[i:i + frame_size] for i in range(0, len(chunk), frame_size)]
        rms_values = np.array([np.sqrt(np.mean(frame**2)) for frame in frames if frame.size])
        bar.rms_energy = float(np.sqrt(np.mean(chunk**2)))
        bar.dynamic_range = float(20 * np.log10(
            (np.percentile(rms_values, 90) + 1e-8) / (np.percentile(rms_values, 10) + 1e-8)
        ))
        onsets = np.abs(np.diff(chunk, prepend=chunk[0]))
        threshold = np.mean(onsets) + 1.5 * np.std(onsets)
        bar.vocal_onset = float(np.sum(onsets > threshold) / max(0.1, bar.end - bar.start))
        bar.beat_alignment = max(0, min(100, 100 - abs(bar.vocal_onset - 4) * 10))
    return bars
