from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr

from rap_mixer.compat import StrEnum


class Provenance(StrEnum):
    AUDIO = "Directly measured from audio"
    TRANSCRIPT = "Calculated from transcript"
    LOCAL = "Inferred by a local model"
    CLOUD = "Inferred by a cloud model"
    MANUAL = "Manually supplied"
    UNAVAILABLE = "Unavailable"


class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = Field(ge=0, le=1)


class TranscriptSegment(BaseModel):
    text: str
    start: float
    end: float
    confidence: float = Field(default=0.5, ge=0, le=1)
    words: list[WordTimestamp] = []


class BarBoundary(BaseModel):
    number: int
    start: float
    end: float
    locked: bool = False
    estimated: bool = True


class FeatureValue(BaseModel):
    value: float = Field(ge=0, le=100)
    provenance: Provenance
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = []


class BarFeatures(BaseModel):
    number: int
    start: float
    end: float
    transcript: str
    word_count: int = 0
    syllable_count: int = 0
    words_per_second: float = 0
    syllables_per_second: float = 0
    rhyme_endings: str = ""
    internal_rhyme_count: int = 0
    pause_duration: float = 0
    mean_pitch: float | None = None
    pitch_range: float | None = None
    rms_energy: float = 0
    dynamic_range: float = 0
    beat_alignment: float = 0
    vocal_onset: float = 0
    asr_confidence: float = 0
    semantic_role: str = "Transition"
    active: bool = False


class ScoreContribution(BaseModel):
    feature: str
    amount: float
    kind: Literal["feature", "interaction", "context"] = "feature"


class OutputScore(BaseModel):
    name: str
    score: float = Field(ge=0, le=100)
    uncertainty: float = Field(ge=0)
    positive: list[ScoreContribution]
    negative: list[ScoreContribution]
    missing_evidence: list[str]
    trace: dict[str, Any]


class ScoreBundle(BaseModel):
    outputs: list[OutputScore]
    context: dict[str, float]
    interactions: dict[str, float]
    provider: str = "deterministic-local"
    model: str = "transparent-logistic-v1"
    config_version: str
    warning: str | None = None


class Recommendation(BaseModel):
    parameter: str
    direction: Literal["increase", "decrease"]
    magnitude: float
    rationale: str
    evidence: str
    action: str
    tradeoffs: str


class RecommendationBundle(BaseModel):
    recommendations: list[Recommendation]
    before: dict[str, float]
    after: dict[str, float]
    uncertainty: float
    feasible: bool


class ProviderStatus(BaseModel):
    status: Literal["Connected", "Authentication failed", "Provider unavailable", "Rate limited", "Model unavailable"]


class SessionCredentials(BaseModel):
    keys: dict[str, SecretStr] = {}

    model_config = {"arbitrary_types_allowed": True}
