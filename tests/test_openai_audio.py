from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from rap_mixer.providers.openai_audio import generate_structured_battle, transcribe_numpy
from rap_mixer.transcription.router import transcribe_selected


class FakeTranscriptions:
    def create(self, **kwargs):
        assert kwargs["model"] == "gpt-4o-mini-transcribe"
        assert kwargs["file"].read(4) == b"RIFF"
        return SimpleNamespace(text="the room hears my claim")


class FakeResponses:
    def create(self, **kwargs):
        assert kwargs["model"] == "custom-model-id"
        return SimpleNamespace(
            output_text='{"bars":[{"text":"I answer your claim in the room",'
            '"addressed_human_bar_ids":["H1"],"function":"answer",'
            '"delivery_note":"pause before room"}]}'
        )


class EmptyTranscriptions:
    def create(self, **kwargs):
        return SimpleNamespace(text="   ")


class InspectIntegerTranscriptions:
    def create(self, **kwargs):
        kwargs["file"].seek(0)
        samples, _ = sf.read(kwargs["file"])
        assert np.max(np.abs(samples)) < 0.51
        assert np.max(np.abs(samples)) > 0.49
        return SimpleNamespace(text="normalized speech")


def test_transcription_model_is_selectable_and_temp_audio_removed():
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    audio = (8000, np.ones(8000, dtype="float32") * 0.1)
    assert transcribe_numpy(client, audio, "gpt-4o-mini-transcribe") == "the room hears my claim"


def test_integer_audio_is_normalized_before_upload():
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=InspectIntegerTranscriptions()))
    audio = (8000, np.ones(8000, dtype=np.int16) * 16384)
    assert transcribe_numpy(client, audio, "gpt-4o-mini-transcribe") == "normalized speech"


def test_empty_transcription_is_reported_as_failure():
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=EmptyTranscriptions()))
    audio = (8000, np.ones(8000, dtype="float32") * 0.1)
    with pytest.raises(ValueError, match="No speech was detected"):
        transcribe_numpy(client, audio, "gpt-4o-mini-transcribe")


def test_generation_model_is_selectable_and_cites_human_bar():
    client = SimpleNamespace(responses=FakeResponses())
    bars = generate_structured_battle(
        client, "custom-model-id", "H1: your claim", {"primary_angle": "reverse"}, 1, "family"
    )
    assert bars[0]["addressed_human_bar_ids"] == ["H1"]


def test_transcription_router_honors_openai_selection():
    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=FakeTranscriptions()))
    audio = (8000, np.ones(8000, dtype="float32") * 0.1)
    text, provenance = transcribe_selected(
        "OpenAI transcription API", audio, client, "gpt-4o-mini-transcribe"
    )
    assert text == "the room hears my claim"
    assert provenance.startswith("OpenAI")
