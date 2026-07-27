import time
from pathlib import Path
from types import SimpleNamespace

from rap_mixer.state import LiveSession
from rap_mixer.tuner.renderer import render_tuner
from rap_mixer.tuner.view_model import live_analysis_to_tuner_view
from rap_mixer.ui.app import _latest_two_transcript


def populated_state(value=65.0):
    state = LiveSession()
    state.latest_analysis = {
        "features": {name: value for name in ("person", "words", "voice", "music", "interaction", "culture")},
        "confidence": .85,
        "context_name": "Cypher",
        "completed_bar_count": 4,
        "active_window_id": "5-6-7-8",
        "active_bars": [
            {"number": number, "syllables_per_second": 4.5, "transcript": f"bar {number}"}
            for number in range(5, 9)
        ],
        "outputs": {"Intelligibility": 73},
        "interactions": {"words_voice": .3},
        "updated_at": time.time(),
    }
    return state


def make_view(state, selected=None, frozen=False):
    return live_analysis_to_tuner_view(
        state, "Custom", selected or ["person", "words", "voice", "music", "interaction", "culture"],
        {name: 65 for name in ("person", "words", "voice", "music", "interaction", "culture")},
        sensitivity=50, smoothing=50, frozen=frozen,
    )


def test_tuner_uses_shared_window_without_new_pipeline():
    state = populated_state()
    view = make_view(state)
    assert view.active_window_id == "5-6-7-8"
    assert [bar.transcript for bar in view.bar_states] == ["bar 5", "bar 6", "bar 7", "bar 8"]
    assert {metric.group for metric in view.metrics} >= {"Bank A", "Outputs", "Interactions"}


def test_direction_lock_hysteresis_and_selected_master_metrics():
    state = populated_state(20)
    first = make_view(state, ["voice"])
    assert next(metric for metric in first.metrics if metric.id == "voice").direction == "increase"
    state.latest_analysis["features"]["voice"] = 65
    for _ in range(10):
        locked = make_view(state, ["voice"])
    assert next(metric for metric in locked.metrics if metric.id == "voice").direction == "hold"
    assert locked.master_proximity is not None


def test_missing_evidence_and_reduced_motion_renderer():
    view = make_view(LiveSession())
    assert view.master_state == "INSUFFICIENT EVIDENCE"
    rendered = render_tuner(
        view, reduced_motion=True, high_contrast=True, show_numbers=True,
        paused=False, motion_speed=50, full=True,
    )
    assert "reduced" in rendered[0]
    assert "NOT ENOUGH SIGNAL" in rendered[3]
    assert "WAIT FOR EVIDENCE" in rendered[0]


def test_off_target_renderer_has_directional_non_color_guidance():
    view = make_view(populated_state(20))
    rendered = render_tuner(
        view, reduced_motion=False, high_contrast=False, show_numbers=True,
        paused=False, motion_speed=50, full=True,
    )
    assert "dir-increase" in rendered[1]
    assert "BELOW TARGET · INCREASE" in rendered[1]
    assert "INCREASE" in rendered[0]
    assert "DO THIS:" in rendered[1]
    assert "concrete image" in rendered[1]


def test_high_responsiveness_moves_words_meter_more():
    slow_state = populated_state(20)
    fast_state = populated_state(20)
    make_view(slow_state)
    make_view(fast_state)
    slow_state.latest_analysis["features"]["words"] = 90
    fast_state.latest_analysis["features"]["words"] = 90
    manual = {name: 65 for name in ("person", "words", "voice", "music", "interaction", "culture")}
    slow = live_analysis_to_tuner_view(slow_state, "Custom", ["words"], manual, 78, 20)
    fast = live_analysis_to_tuner_view(fast_state, "Custom", ["words"], manual, 78, 80)
    slow_words = next(metric for metric in slow.metrics if metric.id == "words")
    fast_words = next(metric for metric in fast.metrics if metric.id == "words")
    assert fast_words.current_value > slow_words.current_value


def test_single_app_single_live_stream_integration():
    root = Path(__file__).parents[1]
    app_source = (root / "rap_mixer/ui/app.py").read_text()
    entry_source = (root / "app.py").read_text()
    tuner_source = "".join(path.read_text() for path in (root / "rap_mixer/tuner").glob("*.py"))
    assert app_source.count("gr.Blocks(") == 1
    assert app_source.count("live_audio.stream(") == 1
    assert "## Performance Tuner" in app_source
    assert 'with gr.Tab("Performance Tuner")' not in app_source
    assert entry_source.count(".launch(") == 1
    assert ".launch(" not in tuner_source
    assert "gr.Audio" not in tuner_source


def test_latest_two_transcripts_are_shown_in_order():
    bars = [SimpleNamespace(number=index, transcript=f"lyrics {index}") for index in range(1, 5)]
    rendered = _latest_two_transcript(bars)
    assert "BAR 3" in rendered and "lyrics 3" in rendered
    assert "BAR 4" in rendered and "lyrics 4" in rendered
    assert "BAR 2" not in rendered
