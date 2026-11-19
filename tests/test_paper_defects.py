"""Regression tests for five defects found while checking the paper against the code."""
import numpy as np

from rap_mixer.analysis.feature_estimation import estimate_features, named_or_numeric_count
from rap_mixer.analysis.interactions import calculate
from rap_mixer.scoring.forward import TRACE_FORMULA, DeterministicScoringEngine
from rap_mixer.state import LiveSession
from rap_mixer.tuner.view_model import live_analysis_to_tuner_view
from rap_mixer.ui import app as ui_app
from tests.test_forward import A, B
from tests.test_performance_tuner import populated_state

LAYERS = ("person", "words", "voice", "music", "interaction", "culture")


def test_trace_formula_states_the_additive_context_shift():
    bundle = DeterministicScoringEngine().score(A, {**B, "groove": 1.5, "familiarity": 0.5})
    musicality = next(x for x in bundle.outputs if x.name == "Musicality")
    assert musicality.trace["formula"] == TRACE_FORMULA
    assert "context focus" in TRACE_FORMULA and "context ×" not in TRACE_FORMULA
    shift = 0.35 * (musicality.trace["context_factor"] - 1)
    context_terms = [x.amount for x in musicality.positive + musicality.negative if x.kind == "context"]
    assert context_terms and abs(context_terms[0] - shift) < 1e-9


def test_context_interaction_terms_follow_the_context_weights():
    neutral = calculate(A, B)
    club = calculate(A, {**B, "groove": 1.5})
    battle = calculate(A, {**B, "response": 1.4})
    assert abs(neutral["music_context"] - (A["music"] / 100 - 0.5)) < 1e-9
    assert club["music_context"] - neutral["music_context"] > 0.25
    assert battle["interaction_context"] - neutral["interaction_context"] > 0.2


def test_live_cue_follows_the_direction_of_the_correction():
    targets = {name: 65 for name in LAYERS}

    def cue(person, culture, selected):
        state = populated_state(65)
        state.latest_analysis["features"].update(person=person, culture=culture)
        return live_analysis_to_tuner_view(state, "Custom", selected, targets, 50, 95).primary_cue

    assert cue(95, 65, ["person"]) == "STAY ON THE CLAIM"
    assert cue(20, 65, ["person"]) == "INCREASE SPECIFICITY"
    assert cue(65, 95, ["culture"]) == "DROP A REFERENCE"
    assert cue(65, 20, ["culture"]) == "CLARIFY THE REFERENCE"


def test_line_initial_capitals_and_the_pronoun_i_are_not_named_references():
    verse = "I came up counting quarters\nNow I count the bars\nYou said the city sleeps. Writing down the names"
    assert named_or_numeric_count(verse) == 0
    assert named_or_numeric_count("Back in Chicago with 3 notebooks\nand Nina on the hook") == 3
    plain, _, _ = estimate_features((55,) * 6, verse, None, "Auto-estimate from performance")
    named, _, _ = estimate_features(
        (55,) * 6, verse + "\nBack in Chicago with Nina", None, "Auto-estimate from performance"
    )
    assert named["culture"] > plain["culture"] and named["person"] > plain["person"]


def test_silent_live_window_reports_a_status_instead_of_raising():
    silent = (16000, np.zeros(16000, dtype=np.int16))
    tuner = ("Clear and direct", list(LAYERS), "Focus", False, False, True, 40, 68, 78, False,
             65, 65, 65, 65, 65, 65)
    result = ui_app.live_update(
        silent, LiveSession(), 90, "typed verse", ["Cypher"], 50, ["Intelligibility"], [], [], 5, 12,
        "Use supplied lyrics", "faster-whisper", "Deterministic local baseline",
        "Local/open-source—no API key", False, "", 55, 55, 55, 55, 55, 55, *tuner, request=None,
    )
    assert "silent" in result[2].lower() and "unchanged" in result[2].lower()


def test_no_context_is_preselected_and_scoring_asks_for_one():
    status = ui_app.analyze(None, "typed verse about the block", 90, [], 50, "Auto-estimate, then edit",
                            55, 55, 55, 55, 55, 55)[0]
    assert status == ui_app.NO_CONTEXT
    audio = (16000, (np.sin(np.arange(16000) / 12) * 9000).astype(np.int16))
    tuner = ("Clear and direct", list(LAYERS), "Focus", False, False, True, 40, 68, 78, False,
             65, 65, 65, 65, 65, 65)
    result = ui_app.live_update(
        audio, LiveSession(), 90, "typed verse", None, 50, ["Intelligibility"], [], [], 5, 12,
        "Use supplied lyrics", "faster-whisper", "Deterministic local baseline",
        "Local/open-source—no API key", False, "", 55, 55, 55, 55, 55, 55, *tuner, request=None,
    )
    assert result[2].startswith(ui_app.NO_CONTEXT) and "unchanged" in result[2]
