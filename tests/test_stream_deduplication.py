import numpy as np

from rap_mixer.audio.buffering import AudioRingBuffer, deduplicate_text
from rap_mixer.state import LiveSession


def test_overlap_deduplication_and_stale_guard():
    assert deduplicate_text("we own the night", "the night forever") == "we own the night forever"
    state = LiveSession()
    state.generation = 2
    assert not state.accept_result(1, "stale")
    assert state.accept_result(2, "fresh")


def test_live_integer_microphone_audio_is_normalized():
    buffer = AudioRingBuffer()
    data = buffer.append((48000, np.array([0, 16384, -16384], dtype=np.int16)))
    assert np.max(np.abs(data)) == 0.5
