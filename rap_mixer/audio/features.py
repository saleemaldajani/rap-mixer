from __future__ import annotations

import numpy as np


class AudioFeatureExtractor:
    def analyze_audio(self, sample_rate: int, audio: np.ndarray) -> dict[str, float | bool | str]:
        samples = np.asarray(audio)
        if np.issubdtype(samples.dtype, np.integer):
            info = np.iinfo(samples.dtype)
            y = samples.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            y = samples.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.size == 0:
            raise ValueError("The recording is empty.")
        peak = float(np.max(np.abs(y)))
        rms = float(np.sqrt(np.mean(y**2)))
        if rms < 1e-5:
            raise ValueError("The recording appears silent.")
        frames = [y[i:i + 2048] for i in range(0, len(y), 2048) if len(y[i:i + 2048])]
        frame_rms = np.array([np.sqrt(np.mean(x**2)) for x in frames])
        zcr = float(np.mean(np.abs(np.diff(np.signbit(y))))) * sample_rate / 2
        spectrum = np.abs(np.fft.rfft(y[:min(y.size, sample_rate * 20)]))
        frequencies = np.fft.rfftfreq(min(y.size, sample_rate * 20), 1 / sample_rate)
        spectral_centroid = float(
            np.sum(spectrum * frequencies) / max(np.sum(spectrum), 1e-8)
        )
        clipping = float(np.mean(np.abs(y) >= 0.99))
        noise_ratio = float(np.percentile(frame_rms, 20) / max(np.percentile(frame_rms, 80), 1e-8))
        warning = ""
        if clipping > 0.005:
            warning = "Clipping detected; voice measurements have higher uncertainty."
        elif noise_ratio > 0.65:
            warning = "High background noise or low dynamic contrast detected."
        return {"duration": len(y) / sample_rate, "peak": peak, "rms": rms,
                "dynamic_range": float(20 * np.log10((frame_rms.max() + 1e-8) / (frame_rms.min() + 1e-8))),
                "onset_rate": min(10.0, zcr / 1000), "clipping": clipping,
                "noise_ratio": noise_ratio, "spectral_centroid": spectral_centroid,
                "crest_factor": peak / max(rms, 1e-8), "warning": warning}
