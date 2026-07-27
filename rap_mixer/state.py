from __future__ import annotations

from dataclasses import dataclass, field

from rap_mixer.audio.buffering import AudioRingBuffer


@dataclass
class LiveSession:
    buffer: AudioRingBuffer = field(default_factory=AudioRingBuffer)
    stable_transcript: str = ""
    provisional_transcript: str = ""
    bar_history: list = field(default_factory=list)
    analyses: list = field(default_factory=list)
    generation: int = 0
    latency_ms: float = 0
    consent: bool = False
    last_transcription_duration: float = 0
    semantic_text: str = ""
    semantic_features: dict[str, float] = field(default_factory=dict)
    semantic_provider: str = "deterministic local"
    latest_analysis: dict = field(default_factory=dict)
    tuner_smoothed: dict[str, float] = field(default_factory=dict)
    tuner_locked: dict[str, bool] = field(default_factory=dict)
    tuner_lock_counts: dict[str, int] = field(default_factory=dict)
    tuner_proximity: dict[str, float] = field(default_factory=dict)
    tuner_paused: bool = False
    frozen_tuner_view: dict | None = None
    saved_tuner_target: dict[str, float] = field(default_factory=dict)
    previous_tuner_view: dict | None = None

    def accept_result(self, generation: int, result) -> bool:
        if generation != self.generation:
            return False
        self.analyses.append(result)
        return True
