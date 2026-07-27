import numpy as np

from rap_mixer.instrumentals.magenta_rt import MagentaRealtimeProvider
from rap_mixer.instrumentals.procedural import generated_beat
from rap_mixer.instrumentals.uploaded import UploadedInstrumentalProvider


def test_uploaded_works_and_magenta_falls_back():
    sr = 8000
    y = np.sin(2 * np.pi * 100 * np.arange(sr) / sr).astype("float32") * 0.1
    assert UploadedInstrumentalProvider().analyze((sr, y))["available"]
    capability = MagentaRealtimeProvider().capabilities()
    assert capability["available"] or capability["fallback"]


def test_generated_beat_is_musical_audio_not_silence():
    sr, audio = generated_beat(90, 2)
    assert sr == 48000
    assert len(audio) == int(2 * 4 * 60 / 90 * sr)
    assert float(np.sqrt(np.mean(audio**2))) > 0.01


def test_generated_beat_changes_with_rap_and_analyzed_profile():
    _, first = generated_beat(
        92, 2, profile={"energy": 0.3, "drum_density": 0.3, "groove": "straight"},
        seed_text="quiet reflective bars",
    )
    _, second = generated_beat(
        92, 2, profile={"energy": 0.9, "drum_density": 0.9, "groove": "boom-bap"},
        seed_text="direct aggressive counter bars",
    )
    assert not np.allclose(first, second)
