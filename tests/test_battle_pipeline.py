from rap_mixer.battle.engine import run_battle_round
from rap_mixer.battle.state_machine import BattleSession
from rap_mixer.schemas.battle import BattlePhase
from rap_mixer.scoring.forward import DeterministicScoringEngine
from tests.test_forward import A, B


def test_text_fallback_full_turn_and_explained_scores():
    session, bars, graph, strategy, generated, human, ai, audio, latency = run_battle_round(
        BattleSession(), "You say your flow is flawless. But your timing needs proof.", None,
        90, 4, 4, "friendly", "timing", "family", "Text only", A, B,
        DeterministicScoringEngine(),
    )
    assert session.phase == BattlePhase.WAITING
    assert audio is None and len(generated) == 4
    assert strategy.human_lines_addressed
    assert all(x.uncertainty > 0 and x.trace for x in ai.outputs)
    assert latency["total"] >= 0 and len(session.turns) == 2


def test_successive_rounds_vary_local_punchlines():
    session = BattleSession()
    first = run_battle_round(
        session, "You claim the crown but never show the work.", None,
        90, 4, 4, "friendly", "", "", "Text only", A, B,
        DeterministicScoringEngine(),
    )[4]
    second = run_battle_round(
        session, "You claim the crown but never show the work.", None,
        90, 4, 4, "friendly", "", "", "Text only", A, B,
        DeterministicScoringEngine(),
    )[4]
    assert [bar.text for bar in first] != [bar.text for bar in second]
