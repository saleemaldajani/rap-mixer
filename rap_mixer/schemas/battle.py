from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rap_mixer.compat import StrEnum


class BattlePhase(StrEnum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    FINALIZING = "FINALIZING_HUMAN_TURN"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    WRITING = "WRITING"
    ALIGNING = "ALIGNING"
    RENDERING = "RENDERING"
    PERFORMING = "PERFORMING"
    SCORING = "SCORING"
    WAITING = "WAITING_FOR_NEXT_ROUND"


class ArgumentNode(BaseModel):
    id: str
    bar_id: str
    kind: Literal["Premise", "Evidence", "Claim", "Boast", "Attack", "Rebuttal", "Joke", "Cultural reference"]
    text: str
    rebuttable: bool = True


class ArgumentEdge(BaseModel):
    source: str
    target: str
    relation: Literal["Supports", "Implies", "Contradicts", "Undercuts", "Reframes", "Answers", "Reverses", "Escalates", "Calls back to"]


class BattleStrategy(BaseModel):
    primary_angle: str
    secondary_angle: str | None = None
    response_moves: list[str]
    human_lines_addressed: list[str]
    facts_allowed: list[str]
    facts_disallowed: list[str]
    desired_effect: str
    audience_model: str
    tone: str
    round_arc: list[str]
    target_bar_count: int = Field(ge=1, le=32)
    rhyme_constraints: dict[str, str | float]
    cadence_constraints: dict[str, str | float]
    safety_constraints: list[str]


class GeneratedBar(BaseModel):
    bar_number: int
    text: str
    function: str
    addressed_human_bar_ids: list[str]
    rhyme_family: str | None
    target_syllables: int
    stress_pattern: list[int] | None
    intended_start_beat: float
    intended_end_beat: float
    delivery_note: str
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = []


class BattleTurn(BaseModel):
    turn_id: int
    speaker: Literal["Human", "AI"]
    transcript: str
    bar_texts: list[str]
    claims: list[str] = []
    strategy: BattleStrategy | None = None
    generated_bars: list[GeneratedBar] = []
    score_bundle: dict = {}
    provider: str = "deterministic-local"
    safety_result: str = "passed"
    latency_ms: dict[str, float] = {}

