from types import SimpleNamespace

import numpy as np

from rap_mixer.scoring.forward import DeterministicScoringEngine
from rap_mixer.ui import human_battle
from rap_mixer.ui.app import CONTEXTS

SCORER = DeterministicScoringEngine()
REQUEST = SimpleNamespace(session_hash="test-human-battle")
AUDIO = (16000, np.zeros(16000, dtype="float32"))


CLOSE = ({"person": 50, "words": 55, "voice": 52, "music": 50, "interaction": 55, "culture": 50},
         {"person": 56, "words": 62, "voice": 57, "music": 55, "interaction": 63, "culture": 55})
APART = ({"person": 20, "words": 25, "voice": 30, "music": 30, "interaction": 15, "culture": 25},
         {"person": 90, "words": 90, "voice": 88, "music": 85, "interaction": 95, "culture": 85})


def configure_automatic_analysis(monkeypatch, profiles=APART):
    transcripts = iter(["person one responsive punchline", "person two stronger rebuttal"])
    monkeypatch.setattr(
        human_battle, "transcribe_selected",
        lambda *_: (next(transcripts), "test AI transcription"),
    )
    estimates = iter([(profiles[0], .8, ""), (profiles[1], .85, "")])
    monkeypatch.setattr(human_battle, "estimate_features", lambda *_: next(estimates))


def submit(state, contexts=("Battle",), outcomes=None):
    return human_battle.process_human_turn(
        state, AUDIO, "Alpha", "Beta", "faster-whisper",
        "Local/open-source—no API key", False, "unused", SCORER, CONTEXTS, REQUEST,
        selected_contexts=list(contexts), selected_outcomes=outcomes,
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
    assert list(second[7]["Outcome"]) == [*human_battle.BATTLE_OUTCOMES, "Round score (mean)"]
    assert "Round winner: Beta" in second[8]
    assert len(second[9]) == 2


def test_no_winner_is_named_when_the_margin_is_inside_the_bands(monkeypatch):
    configure_automatic_analysis(monkeypatch, CLOSE)
    result = submit(submit(human_battle._empty_match())[0])
    table = result[7].set_index("Outcome")
    alpha, beta = (float(table.loc["Round score (mean)", name].split(" ± ")[0]) for name in ("Alpha", "Beta"))
    bands = [float(table.loc["Round score (mean)", name].split(" ± ")[1]) for name in ("Alpha", "Beta")]
    assert 1 < abs(alpha - beta) <= sum(bands)  # the earlier rule would have named Beta the winner
    assert "No decision" in result[8] and "Round winner" not in result[8]
    assert table.loc["Round score (mean)", "Reading"] == "inside the bands"
    assert {row["Result"] for row in result[9].to_dict("records")} == {"no decision"}


def test_the_round_needs_a_context_chosen_by_the_performers(monkeypatch):
    configure_automatic_analysis(monkeypatch)
    blocked = submit(human_battle._empty_match(), contexts=())
    assert blocked[2] == human_battle.NO_CONTEXT and blocked[0]["performer"] == 1
    first = submit(human_battle._empty_match(), contexts=("Cypher", "Club"), outcomes=["Musicality"])
    assert first[0]["context_names"] == ["Cypher", "Club"]
    second = submit(first[0], contexts=("Battle",))  # Person 2 cannot change the agreed context
    assert second[0]["context_names"] == ["Cypher", "Club"]
    assert list(second[7]["Outcome"]) == ["Musicality", "Round score (mean)"]
    assert "Cypher + Club" in second[8]


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
        selected_contexts=["Battle"],
    )
    assert "consent" in response[2].lower()
    assert response[0]["performer"] == 1
