from __future__ import annotations

import re

import pandas as pd
import plotly.graph_objects as go


def commercial_indicators(features: dict[str, float], text: str, confidence: float):
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique = len(set(words)) / max(1, len(words))
    repetition = 1 - unique
    avg_line = len(words) / max(1, len(lines))
    hook_shape = max(0, 100 - abs(repetition - 0.22) * 230 - abs(avg_line - 8) * 3)
    indicators = {
        "Hook clarity": 0.55 * features["words"] + 0.45 * hook_shape,
        "Immediate accessibility": 0.45 * features["words"] + 0.3 * features["voice"]
        + 0.25 * features["music"],
        "Replay potential": 0.35 * features["words"] + 0.25 * features["music"]
        + 0.2 * features["culture"] + 0.2 * hook_shape,
        "Audience connection": 0.45 * features["interaction"] + 0.3 * features["culture"]
        + 0.25 * features["person"],
        "Artist distinctiveness": 0.45 * features["person"] + 0.3 * features["culture"]
        + 0.25 * features["words"],
        "Production readiness": 0.55 * features["music"] + 0.45 * features["voice"],
        "Short-form readiness": 0.4 * hook_shape + 0.35 * features["interaction"]
        + 0.25 * features["words"],
    }
    uncertainty = round(3 + 16 * (1 - confidence), 1)
    rows = [{"indicator": name, "score": round(max(0, min(100, score)), 1),
             "uncertainty_±": uncertainty} for name, score in indicators.items()]
    return rows


def commercial_advice(rows, features: dict[str, float], text: str):
    actions = {
        "Hook clarity": "Write one 6–10 word hook that states the central payoff; repeat it twice with one changed word.",
        "Immediate accessibility": "A/B a take with clearer consonants and one fewer competing element under the lead phrase.",
        "Replay potential": "Add a second-listen callback or internal rhyme that changes the meaning of an earlier line.",
        "Audience connection": "Add one direct audience-facing line and one context the intended listener recognizes immediately.",
        "Artist distinctiveness": "Replace a transferable phrase with a specific place, stake, image, or consequence unique to this artist.",
        "Production readiness": "Balance vocal-to-backing level, remove clipping/noise, and preserve one beat of space around the hook.",
        "Short-form readiness": "Move the strongest self-contained 10–20 second moment earlier and make its first line understandable without setup.",
    }
    weakest = sorted(rows, key=lambda row: row["score"])[:4]
    concrete = bool(re.search(r"\b(street|room|door|hand|face|night|city|stage|mic|crowd)\b",
                              text.lower()))
    result = []
    for rank, row in enumerate(weakest, 1):
        action = actions[row["indicator"]]
        if row["indicator"] == "Artist distinctiveness" and not concrete:
            action = "Replace one abstract claim with a named place, physical object, and personal consequence."
        modeled_gain = round(min(12, max(3, (75 - row["score"]) * 0.22)), 1)
        result.append({"priority": rank, "indicator": row["indicator"],
                       "current": row["score"],
                       "modeled_after": round(min(100, row["score"] + modeled_gain), 1),
                       "specific_revision": action,
                       "why": f"This is among the weakest observed indicators; relevant A estimates are "
                              f"words {features['words']:.0f}, voice {features['voice']:.0f}, "
                              f"music {features['music']:.0f}, interaction {features['interaction']:.0f}."})
    return result


def commercial_plot(rows, advice):
    proposed = {row["indicator"]: row["modeled_after"] for row in advice}
    names = [row["indicator"] for row in rows]
    fig = go.Figure()
    fig.add_bar(name="Current", x=[row["score"] for row in rows], y=names,
                orientation="h", marker_color="#7C3AED")
    fig.add_bar(name="After prioritized revision",
                x=[proposed.get(row["indicator"], row["score"]) for row in rows], y=names,
                orientation="h", marker_color="#10B981")
    fig.update_layout(template="plotly_dark", barmode="group", xaxis_range=[0, 100], height=480,
                      title="Directional commercial-readiness indicators")
    return fig


def frames(features, text, confidence):
    rows = commercial_indicators(features, text, confidence)
    advice = commercial_advice(rows, features, text)
    return pd.DataFrame(rows), pd.DataFrame(advice), commercial_plot(rows, advice)
