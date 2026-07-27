from __future__ import annotations

from html import escape

from rap_mixer.tuner.schemas import PerformanceTunerView


def _how_to(metric_id: str, direction: str) -> str:
    metric_id = metric_id.split(":", 1)[-1]
    advice = {
        "person": {
            "increase": "Name the stake, point of view, or one detail only you would say.",
            "decrease": "Remove autobiography that distracts from the immediate claim.",
        },
        "words": {
            "increase": "Use a concrete image, sharper verb, or one clear internal rhyme.",
            "decrease": "Cut filler and clauses; land one claim in fewer words.",
        },
        "voice": {
            "increase": "Project consonants, vary emphasis, and support the line with breath.",
            "decrease": "Lower intensity, slow the dense phrase, and leave a short breath gap.",
        },
        "music": {
            "increase": "Enter on the pulse and land the final stressed word on the beat.",
            "decrease": "Use less melodic motion and leave more space around the downbeat.",
        },
        "interaction": {
            "increase": "Address the opponent or crowd directly and add a specific callback.",
            "decrease": "Stop chasing reactions; complete the thought before the next address.",
        },
        "culture": {
            "increase": "Add one context-legible reference and make its meaning clear.",
            "decrease": "Remove unexplained references that obscure the central line.",
        },
    }
    if direction == "hold":
        return "Repeat this delivery shape; change another dimension instead."
    return advice.get(metric_id, {}).get(
        direction, "Move this dimension toward the highlighted cyan target corridor."
    )


def _cards(view, groups, show_numbers):
    cards = []
    for metric in (m for m in view.metrics if m.group in groups):
        pos = 50 if metric.normalized_error is None else 50 + metric.normalized_error * 42
        numeric = "" if not show_numbers or metric.current_value is None else (
            f"<strong>{metric.current_value:.0f}</strong> / target {metric.target_value:.0f}")
        uncertain = " LOW CONFIDENCE" if metric.confidence < 0.5 else ""
        recommendation = " ★" if metric.recommendation else ""
        error = abs(metric.normalized_error or 0)
        metric_speed = max(.8, 5.5 - error * 4.4)
        guidance = {
            "increase": "BELOW TARGET · INCREASE ↑",
            "decrease": "ABOVE TARGET · REDUCE ↓",
            "hold": "✓ CENTERED · HOLD",
            "unknown": "? WAITING FOR EVIDENCE",
        }[metric.direction]
        how_to = _how_to(metric.id, metric.direction)
        cards.append(f"""<article class='pt-card dir-{metric.direction}' style='--metric-speed:{metric_speed:.2f}s;--proximity:{(metric.target_proximity or 0):.2f}' aria-label='{escape(metric.label)}: {metric.direction}'>
<header>{escape(metric.label)}{recommendation}</header><div class='pt-meter'><span class='pt-zone'></span>
<span class='pt-marker' style='left:{pos:.1f}%'></span><span class='pt-stripes'></span></div>
<div class='pt-guidance'>{guidance}</div><div class='pt-how'><b>DO THIS:</b> {escape(how_to)}</div>
<b>{metric.direction.upper()}</b> · {metric.trend.upper()}{uncertain}<br>{numeric}</article>""")
    return "".join(cards)


def render_tuner(view: PerformanceTunerView, *, reduced_motion: bool, high_contrast: bool,
                 show_numbers: bool, paused: bool, motion_speed: float, full: bool):
    classes = "pt-root" + (" reduced" if reduced_motion or paused else "")
    if high_contrast:
        classes += " contrast"
    speed = max(0.8, 8 - motion_speed / 15)
    proximity = "—" if view.master_proximity is None else f"{view.master_proximity:.0%}"
    state_class = view.master_state.lower().replace(" ", "-")
    actionable = [m for m in view.metrics if m.group == "Bank A" and m.direction != "unknown"]
    worst = min(actionable, key=lambda metric: metric.target_proximity or 0) if actionable else None
    master_direction = "WAIT FOR EVIDENCE" if worst is None else (
        f"{worst.direction.upper()} {worst.label.upper()}" if worst.direction != "hold" else "HOLD CURRENT DELIVERY"
    )
    master_action = "Keep performing to build evidence." if worst is None else _how_to(worst.id, worst.direction)
    master = f"""<div class='{classes} state-{state_class}' style='--drift-speed:{speed:.1f}s'>
<section class='pt-master' aria-label='Master performance lock: {escape(view.master_state)}'>
<div class='pt-phase'></div><h2>{escape(view.master_state)}</h2><div class='pt-big'>{proximity}</div>
<div class='pt-master-direction'>{escape(master_direction)}</div>
<p class='pt-master-action'><b>HOW TO GET BACK:</b> {escape(master_action)}</p>
<p>{view.completed_bar_count} OF 4 BARS · {escape(view.context_name)} · {escape(view.target_profile_name)}</p>
<p>Confidence-aware shared analysis · latency {view.latency_ms or 0:.0f} ms</p></section></div>"""
    bank = f"<div class='{classes} pt-grid'>{_cards(view, {'Bank A'}, show_numbers)}</div>"
    cue = f"<div class='pt-cue' role='status'><small>ONE LIVE CUE</small><h2>{escape(view.primary_cue)}</h2></div>"
    timeline_items = [f"<div><b>{escape(bar.label)}</b><br>{escape(bar.state)}</div>" for bar in view.bar_states]
    for index in range(len(timeline_items), 4):
        timeline_items.append(f"<div><b>BAR {index + 1}</b><br>Provisional</div>")
    timeline = f"<div class='pt-timeline'>{''.join(timeline_items)}</div>"
    detail = "\n\n".join(f"**{m.label}:** {m.direction} · {m.trend} · confidence {m.confidence:.0%}"
                               for m in view.metrics)
    outputs = "" if not full else (
        f"<div class='{classes} pt-grid compact'>{_cards(view, {'Outputs', 'Interactions'}, show_numbers)}</div>"
    )
    return master, bank, outputs, cue, timeline, detail
