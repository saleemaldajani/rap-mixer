import numpy as np
import pytest

from rap_mixer.audio.features import AudioFeatureExtractor


def test_empty_and_silent_fail_gracefully():
    extractor = AudioFeatureExtractor()
    with pytest.raises(ValueError, match="empty"):
        extractor.analyze_audio(16000, np.array([]))
    with pytest.raises(ValueError, match="silent"):
        extractor.analyze_audio(16000, np.zeros(16000))


def test_clipping_and_noise_warnings():
    extractor = AudioFeatureExtractor()
    clipped = np.tile(np.array([1.0, -1.0, 0.2], dtype=np.float32), 6000)
    assert "Clipping" in extractor.analyze_audio(16000, clipped)["warning"]
    rng = np.random.default_rng(3)
    noisy = rng.normal(0, 0.1, 16000)
    assert extractor.analyze_audio(16000, noisy)["noise_ratio"] > 0

