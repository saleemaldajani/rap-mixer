from rap_mixer.battle.state_machine import BattleSession
from rap_mixer.schemas.battle import BattlePhase


def test_human_turn_and_stale_state():
    state = BattleSession()
    generation = state.start()
    assert state.phase == BattlePhase.LISTENING
    assert state.advance(BattlePhase.FINALIZING, generation)
    assert not state.advance(BattlePhase.WRITING, generation - 1)


def test_stop_invalidates_generation():
    state = BattleSession()
    generation = state.start()
    state.stop()
    assert state.phase == BattlePhase.IDLE
    assert not state.advance(BattlePhase.PERFORMING, generation)

