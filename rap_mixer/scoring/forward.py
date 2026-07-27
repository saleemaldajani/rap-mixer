from __future__ import annotations

import math
from pathlib import Path

import yaml

from rap_mixer.analysis.interactions import calculate
from rap_mixer.analysis.schemas import OutputScore, ScoreBundle, ScoreContribution

ROOT = Path(__file__).resolve().parents[2]

CONTEXT_FOCUS = {
    "Rap membership": {"flow": 0.5, "lineage": 0.5},
    "Intelligibility": {"clarity": 0.8, "familiarity": 0.2},
    "Musicality": {"groove": 0.8, "flow": 0.2},
    "Potency": {"response": 0.7, "clarity": 0.3},
    "Replay depth": {"replay": 0.8, "lineage": 0.2},
    "Cultural resonance": {"lineage": 0.7, "familiarity": 0.3},
    "Battle effectiveness": {"response": 0.8, "clarity": 0.2},
    "Emotional impact": {"response": 0.5, "replay": 0.5},
    "Innovation": {"lineage": 0.5, "replay": 0.5},
    "Commercial accessibility": {"clarity": 0.5, "groove": 0.5},
    "Artistic distinctiveness": {"lineage": 0.6, "replay": 0.4},
    "Audience fit": {"familiarity": 0.5, "response": 0.5},
}


class DeterministicScoringEngine:
    def __init__(self, path: Path | None = None):
        self.config = yaml.safe_load((path or ROOT / "config/scoring.yaml").read_text())

    def score(self, features: dict[str, float], context: dict[str, float], confidence: float = 0.75) -> ScoreBundle:
        interactions = calculate(features, context)
        outputs = []
        for name, spec in self.config["outputs"].items():
            terms: list[ScoreContribution] = []
            raw = float(spec["bias"])
            focus = CONTEXT_FOCUS.get(name, {"clarity": 1.0})
            context_factor = sum(context.get(k, 1.0) * weight for k, weight in focus.items())
            context_amount = (context_factor - 1.0) * 0.35
            raw += context_amount
            terms.append(ScoreContribution(
                feature="active context profile", amount=context_amount, kind="context"
            ))
            for feature, coefficient in spec.get("features", {}).items():
                amount = coefficient * features.get(feature, 50) / 100
                raw += amount
                terms.append(ScoreContribution(feature=feature, amount=amount, kind="feature"))
            for key, coefficient in spec.get("interactions", {}).items():
                amount = coefficient * interactions.get(key, 0)
                raw += amount
                terms.append(ScoreContribution(feature=key, amount=amount, kind="interaction"))
            value = 100 / (1 + math.exp(-raw))
            missing = [key for key in spec.get("features", {}) if key not in features]
            uncertainty = min(25, 3 + 14 * (1 - confidence) + 2 * len(missing))
            outputs.append(OutputScore(
                name=name, score=round(value, 1), uncertainty=round(uncertainty, 1),
                positive=sorted([x for x in terms if x.amount >= 0], key=lambda x: -x.amount)[:4],
                negative=sorted([x for x in terms if x.amount < 0], key=lambda x: x.amount)[:4],
                missing_evidence=missing,
                trace={"formula": "100 × sigmoid(bias + Σ(context × coefficient × q(A)) + Σγz)",
                       "bias": spec["bias"], "normalized_A": {k: round(v / 100, 3) for k, v in features.items()},
                       "active_context_focus": focus, "context_factor": round(context_factor, 3),
                       "raw": round(raw, 4), "uncertainty_method": "heuristic evidence confidence"},
            ))
        return ScoreBundle(outputs=outputs, context=context, interactions=interactions, config_version=self.config["version"])


def output_map(bundle: ScoreBundle) -> dict[str, float]:
    return {x.name: x.score for x in bundle.outputs}
