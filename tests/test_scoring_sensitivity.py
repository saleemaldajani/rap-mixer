import numpy as np

from rap_mixer.audio.bars import enrich_bars_audio, segment_text
from rap_mixer.scoring.forward import DeterministicScoringEngine, output_map
from rap_mixer.scoring.inverse import RecommendationEngine
from rap_mixer.ui.app import _features
from rap_mixer.ui.battle import freestyle, freestyle_with_youtube
from tests.test_forward import A


def test_context_does_not_shift_every_output_identically():
    scorer = DeterministicScoringEngine()
    battle = {"flow": 1, "clarity": 1.15, "groove": 0.8, "response": 1.4,
              "replay": 0.75, "lineage": 1, "familiarity": 0.5}
    club = {"flow": 0.9, "clarity": 1.1, "groove": 1.5, "response": 1.2,
            "replay": 1, "lineage": 0.65, "familiarity": 0.5}
    first = output_map(scorer.score(A, battle))
    second = output_map(scorer.score(A, club))
    deltas = {name: round(second[name] - first[name], 1) for name in first}
    assert len(set(deltas.values())) >= 5
    assert deltas["Musicality"] > 0
    assert deltas["Cultural resonance"] < 0


def test_outputs_respond_differently_to_a_dimensions():
    scorer = DeterministicScoringEngine()
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    words = {**A, "words": 90, "voice": 20}
    voice = {**A, "words": 20, "voice": 90}
    word_scores = output_map(scorer.score(words, context))
    voice_scores = output_map(scorer.score(voice, context))
    assert word_scores["Replay depth"] > voice_scores["Replay depth"]
    assert voice_scores["Musicality"] > word_scores["Musicality"]


def test_bar_audio_metrics_are_not_global_copies():
    sr = 1000
    quiet = np.sin(2 * np.pi * 20 * np.arange(sr * 2) / sr) * 0.02
    loud = np.sin(2 * np.pi * 20 * np.arange(sr * 2) / sr) * 0.3
    bars = segment_text("quiet bar. loud bar.", 4, 120)
    enrich_bars_audio(bars, sr, np.concatenate([quiet, loud]))
    assert bars[1].rms_energy > bars[0].rms_energy * 5


def test_manual_mode_does_not_silently_override_a():
    values = [11, 22, 33, 44, 55, 66]
    result, confidence, _ = _features(values, "dense transcript", None, "Manually configure")
    assert list(result.values()) == values
    assert confidence == 1


def test_freestyle_controls_change_generated_text():
    base = [None, "Metronome / click track", False, "pressure", "move forward",
            "clock", "MIT", "family", 90, 4]
    tight = freestyle(*base, 10, 90, 80, *([55] * 6))[2]
    loose = freestyle(*base, 90, 20, 20, *([55] * 6))[2]
    assert tight != loose


def test_repeated_freestyle_runs_produce_new_aligned_bars():
    base = [None, "Metronome / click track", False, "pressure", "move forward",
            "clock", "MIT", "family", 90, 4, 40, 80, 80, *([55] * 6)]
    first = freestyle(*base)[2]
    second = freestyle(*base)[2]
    assert first != second


def test_youtube_instrumental_analysis_is_carried_into_freestyle():
    result = freestyle_with_youtube(
        None, "Uploaded / recorded instrumental", False, "pressure", "move forward", "clock",
        "MIT", "family", 92, 4, 40, 80, 80,
        {"confidence": 0.8, "summary": "Boom-bap groove with a four-bar loop."},
        *([55] * 6),
    )
    assert "YouTube instrumental analysis used" in result[0]
    assert "Generated a local drum, bass, and hat backing track" in result[0]


def test_recommendation_wording_changes_by_workflow_and_target():
    engine = RecommendationEngine()
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    live = engine.recommend(A, context, ["Intelligibility"], workflow="live")
    recorded = engine.recommend(A, context, ["Musicality"], workflow="prerecorded")
    assert live.recommendations[0].parameter != recorded.recommendations[0].parameter
    assert live.recommendations[0].action != recorded.recommendations[0].action


def test_distinct_transcripts_change_multiple_a_layers_and_scores():
    controls = [55] * 6
    generic, _, _ = _features(
        controls, "Ideas move through concepts and meanings", None, "Auto-estimate, then edit"
    )
    specific, _, _ = _features(
        controls,
        "I held the mic at Central Square in 2026; you missed the snare and the crowd saw it",
        None,
        "Auto-estimate, then edit",
    )
    changed = [key for key in generic if abs(generic[key] - specific[key]) > 1]
    assert {"person", "interaction", "culture"}.issubset(changed)
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    scorer = DeterministicScoringEngine()
    first = output_map(scorer.score(generic, context))
    second = output_map(scorer.score(specific, context))
    assert sum(first[name] != second[name] for name in first) >= 6


def test_evidence_changes_rehearsal_action():
    engine = RecommendationEngine()
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    dense = engine.recommend(
        A, context, ["Intelligibility"], evidence="bar 2 (7.4 syllables/s): ideas move",
        workflow="live",
    )
    sparse = engine.recommend(
        A, context, ["Intelligibility"], evidence="bar 2 (2.1 syllables/s): room light",
        workflow="live",
    )
    assert [x.action for x in dense.recommendations] != [x.action for x in sparse.recommendations]


def test_different_integer_recordings_change_audio_features_and_scores():
    sr = 16000
    t = np.arange(sr * 2) / sr
    smooth = (np.sin(2 * np.pi * 120 * t) * 5000).astype(np.int16)
    percussive = (np.sin(2 * np.pi * 1800 * t) * np.linspace(500, 25000, t.size)).astype(np.int16)
    controls = [55] * 6
    first, _, _ = _features(controls, "same words", (sr, smooth), "Auto-estimate from performance")
    second, _, _ = _features(
        controls, "same words", (sr, percussive), "Auto-estimate from performance"
    )
    assert abs(first["voice"] - second["voice"]) > 5
    assert abs(first["music"] - second["music"]) > 5
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    scorer = DeterministicScoringEngine()
    assert output_map(scorer.score(first, context)) != output_map(scorer.score(second, context))


def test_recommendations_depend_on_weak_dimensions_and_improve_targets():
    engine = RecommendationEngine()
    context = {"flow": 1, "clarity": 1, "groove": 1, "response": 1,
               "replay": 1, "lineage": 1, "familiarity": 1}
    targets = list(engine.scorer.config["outputs"])
    weak_words = {**A, "words": 10, "voice": 85, "music": 85}
    weak_voice = {**A, "words": 85, "voice": 10, "music": 85}
    first = engine.recommend(weak_words, context, targets)
    second = engine.recommend(weak_voice, context, targets)
    assert first.recommendations[0].parameter != second.recommendations[0].parameter
    assert sum(first.after.values()) > sum(first.before.values())
    assert sum(second.after.values()) > sum(second.before.values())
