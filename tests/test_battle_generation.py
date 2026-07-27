from rap_mixer.battle.argument_graph import build_argument_graph
from rap_mixer.battle.strategy import plan_strategy
from rap_mixer.battle.verse_plan import generate_bars, repeated_angles
from rap_mixer.ui.battle import _performer_lyrics


def test_strategy_cites_actual_bars_and_count():
    nodes, _ = build_argument_graph(["You say my cadence drifts", "Your proof is just volume"])
    strategy = plan_strategy(nodes, 4, "friendly", "timing", "family", 90)
    bars = generate_bars(strategy, nodes, 90, "family")
    assert strategy.human_lines_addressed == ["H1", "H2"]
    assert len(bars) == 4
    assert all(x.addressed_human_bar_ids[0] in {"H1", "H2"} for x in bars)
    assert "cadence" in bars[0].text.lower()
    assert not repeated_angles(bars)


def test_one_bar_can_be_regenerated_in_isolation():
    nodes, _ = build_argument_graph(["You claim the room", "I question your proof"])
    strategy = plan_strategy(nodes, 4, "comedy", "", "", 100)
    original = generate_bars(strategy, nodes, 100)
    replacement_strategy = strategy.model_copy(update={"target_bar_count": 1})
    replacement = generate_bars(replacement_strategy, [nodes[1]], 100)
    revised = original[:3] + replacement
    assert revised[:3] == original[:3] and revised[3] != original[3]


def test_performer_lyrics_are_numbered_for_reading_over_beat():
    nodes, _ = build_argument_graph(["You claim the room"])
    bars = generate_bars(plan_strategy(nodes, 2, "friendly", "", "", 90), nodes, 90)
    sheet = _performer_lyrics(bars)
    assert "Your lyrics" in sheet and "**1.**" in sheet and bars[0].text in sheet
