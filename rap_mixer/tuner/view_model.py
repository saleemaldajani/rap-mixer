from __future__ import annotations

import time

from rap_mixer.tuner.schemas import BarTunerState, PerformanceTunerView, TunerMetric
from rap_mixer.tuner.targets import CONTEXT_PRIORITY, TOLERANCES, target_values

LAYERS = ["person", "words", "voice", "music", "interaction", "culture"]
OUTPUT_LABELS = {
    "rap_membership": "Rap membership", "intelligibility": "Intelligibility",
    "musicality": "Musicality", "potency": "Potency", "replay_depth": "Replay depth",
    "cultural_resonance": "Cultural resonance", "battle_effectiveness": "Battle effectiveness",
    "emotional_impact": "Emotional impact", "innovation": "Innovation",
    "artistic_distinctiveness": "Artistic distinctiveness", "rebuttal_relevance": "Rebuttal relevance",
    "beat_fit": "Beat fit", "cadence_fit": "Cadence fit", "crowd_accessibility": "Crowd accessibility",
}
INTERACTION_LABELS = {
    "words_voice": "Words × Voice", "voice_music": "Voice × Music",
    "person_culture": "Person × Culture", "words_culture": "Words × Culture",
    "music_context": "Music × Context", "interaction_context": "Interaction × Context",
}


def _cue(metric: TunerMetric) -> str:
    if metric.direction == "hold":
        return "HOLD"
    cues = {
        "person": "INCREASE SPECIFICITY",
        "words": "SIMPLIFY THE CLAIM" if metric.direction == "decrease" else "CLARIFY THE REFERENCE",
        "voice": "SLOW CADENCE" if metric.direction == "decrease" else "INCREASE ARTICULATION",
        "music": "ADD SPACE" if metric.direction == "decrease" else "STAY IN THE POCKET",
        "interaction": "MORE ENERGY" if metric.direction == "increase" else "REDUCE ENERGY",
        "culture": "CLARIFY THE REFERENCE",
    }
    return cues.get(metric.id, "KEEP THIS SETTING")


def live_analysis_to_tuner_view(state, profile: str, selected_metrics: list[str],
                                manual_targets: dict[str, float], sensitivity: float,
                                smoothing: float, frozen: bool = False) -> PerformanceTunerView:
    snapshot = getattr(state, "latest_analysis", None) or {}
    if frozen and getattr(state, "frozen_tuner_view", None):
        return PerformanceTunerView.model_validate(state.frozen_tuner_view)
    features = snapshot.get("features", {})
    confidence = float(snapshot.get("confidence", 0))
    context = snapshot.get("context_name", "Cypher")
    targets = target_values(profile, manual_targets)
    metrics = []
    stale = bool(snapshot and time.time() - snapshot.get("updated_at", 0) > 4)
    alpha_base = max(0.08, min(0.95, smoothing / 100))
    priority = CONTEXT_PRIORITY.get(context, LAYERS)
    recommendation = snapshot.get("recommendation", {})
    for metric_id in LAYERS:
        current = features.get(metric_id)
        target = targets[metric_id]
        tolerance = TOLERANCES[metric_id] * max(0.55, 1.5 - sensitivity / 100)
        if current is None:
            metrics.append(TunerMetric(id=metric_id, label=metric_id.title(), group="Bank A",
                confidence=0, provenance="missing", measured=False, inferred=False,
                provisional=True, stale=stale, controllable=True))
            continue
        old = state.tuner_smoothed.get(metric_id, current)
        metric_alpha = min(
            0.95,
            alpha_base * (
                1.15 if metric_id in {"voice", "music"}
                else 0.75 if metric_id in {"person", "culture"}
                else 1
            ),
        )
        metric_alpha *= 0.35 + 0.65 * confidence
        # Bound one-frame jumps while preserving fast response for audio-led dimensions.
        bounded = max(old - tolerance * 2.5, min(old + tolerance * 2.5, float(current)))
        smoothed = metric_alpha * bounded + (1 - metric_alpha) * old
        state.tuner_smoothed[metric_id] = smoothed
        error = smoothed - target
        normalized = max(-1, min(1, error / tolerance))
        proximity = 1 - min(1, abs(normalized))
        was_locked = state.tuner_locked.get(metric_id, False)
        in_zone = abs(error) <= tolerance * (1.15 if was_locked else 0.82)
        count = state.tuner_lock_counts.get(metric_id, 0) + 1 if in_zone else 0
        state.tuner_lock_counts[metric_id] = count
        locked = in_zone and (was_locked or count >= 2)
        state.tuner_locked[metric_id] = locked
        direction = "hold" if locked else "increase" if error < 0 else "decrease"
        previous = state.tuner_proximity.get(metric_id, proximity)
        trend = "stable" if abs(proximity - previous) < 0.025 else (
            "improving" if proximity > previous else "worsening")
        state.tuner_proximity[metric_id] = proximity
        recommendation_text = None
        if recommendation.get("parameter") == metric_id:
            recommendation_text = recommendation.get("action")
        metrics.append(TunerMetric(
            id=metric_id, label=metric_id.title(), group="Bank A", current_value=smoothed,
            target_value=target, target_min=target - tolerance, target_max=target + tolerance,
            normalized_error=normalized, target_proximity=proximity, direction=direction,
            stability=max(0, 1 - abs(smoothed - old) / max(tolerance, 1)), trend=trend,
            confidence=confidence, provenance=snapshot.get("provenance", "shared live analysis"),
            measured=metric_id in {"voice", "music"}, inferred=metric_id not in {"voice", "music"},
            provisional=snapshot.get("completed_bar_count", 0) < 4, stale=stale,
            controllable=True, recommendation=recommendation_text,
        ))
    # These are presentation-only readings from the already-computed scoring bundle.
    for metric_id, current in snapshot.get("outputs", {}).items():
        label = OUTPUT_LABELS.get(metric_id, metric_id.replace("_", " ").title())
        target, tolerance = 70.0, 14.0
        error = float(current) - target
        normalized = max(-1.0, min(1.0, error / tolerance))
        metrics.append(TunerMetric(
            id=f"output:{metric_id}", label=label, group="Outputs", current_value=float(current),
            target_value=target, target_min=target - tolerance, target_max=target + tolerance,
            normalized_error=normalized, target_proximity=1 - abs(normalized),
            direction="hold" if abs(error) <= tolerance * .82 else "increase" if error < 0 else "decrease",
            stability=None, trend="unknown", confidence=confidence,
            provenance="shared forward scoring", inferred=True,
            provisional=snapshot.get("completed_bar_count", 0) < 4, stale=stale, controllable=False,
        ))
    for metric_id, raw in snapshot.get("interactions", {}).items():
        if metric_id not in INTERACTION_LABELS:
            continue
        current = max(0.0, min(100.0, (float(raw) + 1.0) * 50.0))
        target, tolerance = 65.0, 18.0
        error = current - target
        normalized = max(-1.0, min(1.0, error / tolerance))
        metrics.append(TunerMetric(
            id=f"interaction:{metric_id}", label=INTERACTION_LABELS[metric_id], group="Interactions",
            current_value=current, target_value=target, target_min=target - tolerance,
            target_max=target + tolerance, normalized_error=normalized,
            target_proximity=1 - abs(normalized), direction="hold" if abs(error) <= tolerance * .82
            else "increase" if error < 0 else "decrease", confidence=confidence,
            provenance="shared interaction calculation", inferred=True,
            provisional=snapshot.get("completed_bar_count", 0) < 4, stale=stale, controllable=False,
        ))
    selected = [m for m in metrics if m.id in (selected_metrics or LAYERS) and m.current_value is not None]
    weights = {name: len(priority) - priority.index(name) if name in priority else 1 for name in LAYERS}
    denom = sum(weights[m.id] for m in selected)
    proximity = sum((m.target_proximity or 0) * weights[m.id] for m in selected) / denom if denom else None
    completed = int(snapshot.get("completed_bar_count", 0))
    if not snapshot:
        master = "INSUFFICIENT EVIDENCE"
    elif state.tuner_paused:
        master = "PAUSED"
    elif stale:
        master = "STALE"
    elif completed < 4:
        master = "BUILDING WINDOW"
    elif proximity is not None and proximity >= 0.82:
        master = "LOCKED"
    elif proximity is not None and proximity >= 0.62:
        master = "CLOSE"
    elif proximity is not None and proximity >= 0.35:
        master = "DRIFTING"
    else:
        master = "OFF TARGET"
    worst = min(selected, key=lambda m: m.target_proximity or 0) if selected else None
    cue = "NOT ENOUGH SIGNAL" if not worst or confidence < 0.35 else _cue(worst)
    bars = []
    for item in snapshot.get("active_bars", []):
        rate = float(item.get("syllables_per_second", 0))
        distance = abs(rate - 4.5)
        bar_state = "Locked" if distance < 0.8 else "Improving" if distance < 1.6 else (
            "Drifting" if distance < 2.5 else "Off-target")
        bars.append(BarTunerState(bar_id=f"B{item['number']}", label=f"BAR {item['number']}",
                                  state=bar_state, transcript=item.get("transcript", "")))
    return PerformanceTunerView(
        session_id=str(id(state)), active_window_id=snapshot.get("active_window_id"),
        completed_bar_count=completed, master_state=master, master_proximity=proximity,
        primary_cue=cue, context_name=context, target_profile_name=profile,
        latency_ms=snapshot.get("latency_ms"), metrics=metrics, bar_states=bars,
    )
