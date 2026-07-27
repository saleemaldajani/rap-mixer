from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TunerMetric(BaseModel):
    id: str
    label: str
    group: str
    current_value: float | None = None
    target_value: float | None = None
    target_min: float | None = None
    target_max: float | None = None
    normalized_error: float | None = None
    target_proximity: float | None = None
    direction: Literal["increase", "decrease", "hold", "unknown"] = "unknown"
    stability: float | None = None
    trend: Literal["improving", "worsening", "stable", "unknown"] = "unknown"
    confidence: float = Field(ge=0, le=1)
    provenance: str
    measured: bool = False
    inferred: bool = True
    provisional: bool = False
    stale: bool = False
    controllable: bool = True
    recommendation: str | None = None


class BarTunerState(BaseModel):
    bar_id: str
    label: str
    state: Literal["Locked", "Improving", "Drifting", "Off-target", "Provisional"]
    transcript: str = ""


class PerformanceTunerView(BaseModel):
    session_id: str
    active_window_id: str | None = None
    completed_bar_count: int = 0
    master_state: str
    master_proximity: float | None = None
    primary_cue: str
    context_name: str
    target_profile_name: str
    latency_ms: float | None = None
    metrics: list[TunerMetric] = Field(default_factory=list)
    bar_states: list[BarTunerState] = Field(default_factory=list)
