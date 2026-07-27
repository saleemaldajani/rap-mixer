import numpy as np


def metronome(bpm: float, bars: int, sample_rate: int = 22050):
    duration = bars * 4 * 60 / bpm
    y = np.zeros(int(duration * sample_rate), dtype=np.float32)
    for beat in range(bars * 4):
        start = int(beat * 60 / bpm * sample_rate)
        n = min(700, len(y) - start)
        y[start:start + n] = 0.2 * np.sin(2 * np.pi * 1000 * np.arange(n) / sample_rate) * np.linspace(1, 0, n)
    return sample_rate, y
