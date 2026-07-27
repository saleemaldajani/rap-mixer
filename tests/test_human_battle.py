from types import SimpleNamespace

import numpy as np

from rap_mixer.scoring.forward import DeterministicScoringEngine
from rap_mixer.ui import human_battle
from rap_mixer.ui.app import CONTEXTS

SCORER = DeterministicScoringEngine()
REQUEST = SimpleNamespace(session_hash="test-human-battle")
AUDIO = (16000, np.zeros(16000, dtype="float32"))


def configure_automatic_analysis(monkeypatch):
    transcripts = iter(["person one responsive punchline", "person two stronger rebuttal"])
    monkeypatch.setattr(
        human_battle, "transcribe_selected",
        lambda *_: (next(transcripts), "test AI transcription"),
    )
    estimates = iter([
        ({"person": 50, "words": 55, "voice": 52, "music": 50,
          "interaction": 55, "culture": 50}, .8, ""),
        ({"person": 70, "words": 72, "voice": 68, "music": 60,
          "interaction": 75, "culture": 65}, .85, ""),
    ])
    monkeypatch.setattr(human_battle, "estimate_features", lambda *_: next(estimates))


def submit(state):
    return human_battle.process_human_turn(
        state, AUDIO, "Alpha", "Beta", "faster-whisper",
        "Local/open-source—no API key", False, "unused", SCORER, CONTEXTS, REQUEST,
    )


def test_two_recordings_are_automatically_transcribed_analyzed_and_judged(monkeypatch):
    configure_automatic_analysis(monkeypatch)
    first = submit(human_battle._empty_match())
    state = first[0]
    assert state["performer"] == 2
    assert first[4] == "person one responsive punchline"
    assert "Record Person 2" in first[2]
    second = submit(state)
    assert second[0]["phase"] == "results"
    assert second[5] == "person two stronger rebuttal"
    assert len(second[7]) == 2
    assert "Round winner: Beta" in second[8]
    assert len(second[9]) == 2


def test_result_button_starts_next_round_with_history(monkeypatch):
    configure_automatic_analysis(monkeypatch)
    state = submit(human_battle._empty_match())[0]
    result = submit(state)
    restarted = submit(result[0])
    assert restarted[0]["round"] == 2
    assert restarted[0]["performer"] == 1
    assert len(restarted[0]["history"]) == 2


def test_openai_requires_visible_per_recording_consent():
    response = human_battle.process_human_turn(
        human_battle._empty_match(), AUDIO, "Alpha", "Beta", "OpenAI transcription API",
        "Use my own API key", False, "gpt-4o-mini-transcribe", SCORER, CONTEXTS, REQUEST,
    )
    assert "consent" in response[2].lower()
    assert response[0]["performer"] == 1
