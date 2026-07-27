from __future__ import annotations

import numpy as np


class AudioRingBuffer:
    def __init__(self, max_seconds: float = 120):
        self.max_seconds = max_seconds
        self.sample_rate: int | None = None
        self.data = np.zeros(0, dtype=np.float32)

    def append(self, chunk: tuple[int, np.ndarray]) -> np.ndarray:
        sr, audio = chunk
        samples = np.asarray(audio)
        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            scale = float(max(abs(info.min), info.max))
            audio = samples.astype(np.float32) / scale
        else:
            audio = samples.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if self.sample_rate not in (None, sr):
            self.data = np.zeros(0, dtype=np.float32)
        self.sample_rate = sr
        self.data = np.concatenate((self.data, audio))[-int(sr * self.max_seconds):]
        return self.data


def deduplicate_text(stable: str, new: str, max_overlap_words: int = 20) -> str:
    old, incoming = stable.split(), new.split()
    overlap = 0
    for n in range(1, min(len(old), len(incoming), max_overlap_words) + 1):
        if [x.lower() for x in old[-n:]] == [x.lower() for x in incoming[:n]]:
            overlap = n
    return " ".join(old + incoming[overlap:]).strip()
