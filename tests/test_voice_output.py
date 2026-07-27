from pathlib import Path

import numpy as np
import soundfile as sf

from rap_mixer.performance.mixing import mix_voice_and_backing
from rap_mixer.performance.voice import LocalSyntheticVoiceProvider, beat_phrases, speech_safe_text


def test_local_synthetic_voice_is_browser_safe_wav():
    provider = LocalSyntheticVoiceProvider()
    if not provider.capabilities()["audio"]:
        return
    path = provider.synthesize_verse("This is a clearly synthetic test response.")
    try:
        assert path.endswith(".wav")
        audio, sample_rate = sf.read(path)
        assert sample_rate > 0 and len(audio) > 0
    finally:
        Path(path).unlink(missing_ok=True)


def test_synthetic_voice_can_be_mixed_with_backing():
    provider = LocalSyntheticVoiceProvider()
    if not provider.capabilities()["audio"]:
        return
    voice = provider.synthesize_verse("Synthetic vocal over a backing track.")
    backing = (22050, np.sin(2 * np.pi * 110 * np.arange(22050) / 22050).astype("float32") * 0.1)
    mixed = mix_voice_and_backing(voice, backing)
    try:
        audio, sample_rate = sf.read(mixed)
        assert sample_rate > 0 and len(audio) > 0
        assert np.max(np.abs(audio)) <= 1
    finally:
        Path(voice).unlink(missing_ok=True)
        Path(mixed).unlink(missing_ok=True)


def test_internal_human_bar_ids_are_not_spoken():
    cleaned = speech_safe_text("H1: I answer H2, then I reverse human bar 3.")
    assert "H1" not in cleaned and "H2" not in cleaned
    assert "bar 3" not in cleaned
    assert "," not in cleaned


def test_rhythmic_voice_varies_delivery_and_fits_bars(monkeypatch):
    provider = LocalSyntheticVoiceProvider()
    calls = []

    monkeypatch.setattr(provider, "capabilities", lambda: {"audio": True})

    def fake_say(line, rate, pitch):
        calls.append((line, rate, pitch))
        return np.ones(1200, dtype="float32") * .1, 1000

    monkeypatch.setattr(provider, "_say_line", fake_say)
    path = provider.synthesize_performance(
        "Short first bar\nThis second bar carries several more syllables into the landing",
        bpm=120, energy=80, aggression=70,
    )
    try:
        audio, sample_rate = sf.read(path)
        assert len(audio) >= 2 * 4 * 60 / 120 * sample_rate
        assert len(calls) == 7  # the three-word line leaves one intentional breath beat
        assert len({call[1:] for call in calls}) > 1
        assert np.max(audio) < .12  # phrases did not overlap and sum on top of one another
    finally:
        Path(path).unlink(missing_ok=True)


def test_dense_voice_is_not_time_compressed(monkeypatch):
    provider = LocalSyntheticVoiceProvider()
    original = np.linspace(-.2, .2, 3500, dtype="float32")
    monkeypatch.setattr(provider, "capabilities", lambda: {"audio": True})
    monkeypatch.setattr(provider, "_say_line", lambda *_: (original, 1000))
    path = provider.synthesize_performance("A deliberately dense synthetic line", bpm=120)
    try:
        audio, sample_rate = sf.read(path)
        assert sample_rate == 1000
        assert len(audio) >= len(original)
        assert np.count_nonzero(audio) >= len(original) - 2
    finally:
        Path(path).unlink(missing_ok=True)


def test_line_is_split_into_one_ordered_phrase_per_beat():
    phrases = beat_phrases("one two three four five six seven eight")
    assert phrases == ["one two", "three four", "five six", "seven eight"]
