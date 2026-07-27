from __future__ import annotations

from dataclasses import dataclass, field

from rap_mixer.schemas.battle import BattlePhase, BattleTurn

ORDER = list(BattlePhase)


@dataclass
class BattleSession:
    phase: BattlePhase = BattlePhase.IDLE
    generation: int = 0
    turns: list[BattleTurn] = field(default_factory=list)
    stopped: bool = False

    def start(self) -> int:
        self.generation += 1
        self.phase = BattlePhase.LISTENING
        self.stopped = False
        return self.generation

    def advance(self, phase: BattlePhase, generation: int) -> bool:
        if generation != self.generation or self.stopped:
            return False
        self.phase = phase
        return True

    def stop(self) -> None:
        self.generation += 1
        self.stopped = True
        self.phase = BattlePhase.IDLE

