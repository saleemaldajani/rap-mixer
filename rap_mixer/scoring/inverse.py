from __future__ import annotations

import re

from rap_mixer.analysis.schemas import Recommendation, RecommendationBundle
from rap_mixer.scoring.forward import DeterministicScoringEngine, output_map


class RecommendationEngine:
    def __init__(self, scorer: DeterministicScoringEngine | None = None):
        self.scorer = scorer or DeterministicScoringEngine()

    def recommend(self, features: dict[str, float], context: dict[str, float], targets: list[str],
                  desired: float = 5, protected_features: set[str] | None = None,
                  protected_outputs: set[str] | None = None, max_change: float = 12,
                  tolerance: float = 2, evidence: str = "",
                  workflow: str = "general") -> RecommendationBundle:
        protected_features = protected_features or set()
        protected_outputs = protected_outputs or set()
        before_bundle = self.scorer.score(features, context)
        before = output_map(before_bundle)
        current = dict(features)
        recommendations = []
        candidates = [x for x in features if x not in protected_features]
        for _ in range(min(3, len(candidates))):
            best = None
            for feature in candidates:
                if any(r.parameter == feature for r in recommendations):
                    continue
                trial = dict(current)
                probe = min(max_change, 4.0)
                trial[feature] = min(100, trial[feature] + probe)
                probe_after = output_map(self.scorer.score(trial, context))
                leverage = sum(probe_after.get(t, 0) - before.get(t, 0) for t in targets) / probe
                delta = min(max_change, max(3.0, desired / max(leverage, 0.15)))
                trial[feature] = min(100, current[feature] + delta)
                after = output_map(self.scorer.score(trial, context))
                gain = sum(after.get(t, 0) - before.get(t, 0) for t in targets)
                opportunity = max(0.05, (100 - current[feature]) / 100)
                input_specific_gain = gain * opportunity
                valid = all(after.get(p, 0) >= before.get(p, 0) - tolerance for p in protected_outputs)
                if valid and (best is None or input_specific_gain > best[0]):
                    best = (input_specific_gain, feature, delta, trial, after)
            if not best or best[0] <= 0:
                break
            _, feature, delta, current, after = best
            base_action = {
                "words": "Replace one abstract phrase with a concrete image and repeat the central proposition once.",
                "voice": "Keep the cadence; rehearse consonants at 90% tempo, then add a 150–250 ms pause before the key claim.",
                "music": "Reduce competing midrange around the key phrase and preserve one beat of space.",
                "person": "Rewrite one line with a specific stake, place, or consequence only this artist could claim.",
                "interaction": "Add a direct call-and-response or audience-facing turn in one bar.",
                "culture": "Add one audience-legible reference without removing the artist's regional language.",
            }.get(feature, f"Rehearse one observable change to {feature} and compare takes.")
            target_action = {
                "Intelligibility": "Prioritize separation around the central claim.",
                "Musicality": "Test the revision against the beat and preserve the pocket.",
                "Potency": "Place the strongest consequence at the end of the bar.",
                "Replay depth": "Add a detail or callback that rewards a second listen.",
                "Cultural resonance": "Check that the reference is legible to the selected context.",
                "Battle effectiveness": "Directly answer one surviving premise from the opponent.",
                "Audience fit": "Use one cue the selected audience can recognize immediately.",
            }.get(targets[0], f"Optimize specifically for {targets[0]}.")
            workflow_action = {
                "prerecorded": (
                    "Duplicate the take, change only this parameter, and A/B the same bars."
                ),
                "live": (
                    "Apply this on the next completed bar while preserving the current four-bar arc."
                ),
                "battle": (
                    "Revise only the weakest counter-bar and keep its addressed human bar ID."
                ),
                "freestyle": (
                    "Regenerate one weak bar over the same instrumental section; keep the other bars."
                ),
            }.get(workflow, "Test one controlled revision before changing another parameter.")
            observed_action = self._evidence_action(feature, evidence)
            action = f"{target_action} {observed_action or base_action} {workflow_action}"
            evidence_text = evidence.strip()[:220] if evidence.strip() else (
                f"Current {feature} estimate is {features[feature]:.0f}/100."
            )
            recommendations.append(Recommendation(parameter=feature, direction="increase", magnitude=delta,
                rationale=f"The configured model gives {feature} positive leverage on {', '.join(targets)}.",
                evidence=f"{evidence_text} Current {feature}: {features[feature]:.0f}/100; modeled change: {delta:.1f}.",
                action=action,
                tradeoffs=f"Optimizes {', '.join(targets)} with context fixed; other meters are shown in the before/after chart."))
            if all(after.get(t, 0) >= before.get(t, 0) + desired for t in targets):
                break
        final = output_map(self.scorer.score(current, context))
        feasible = all(final.get(t, 0) >= before.get(t, 0) + desired - 1 for t in targets)
        return RecommendationBundle(recommendations=recommendations, before=before, after=final,
                                    uncertainty=7, feasible=feasible)

    @staticmethod
    def _evidence_action(feature: str, evidence: str) -> str:
        text = evidence.lower()
        rates = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?) syllables/s", text)]
        high_density = bool(rates and max(rates) > 6.5)
        low_density = bool(rates and max(rates) < 3.0)
        direct_address = bool(re.search(r"\b(you|your|you're)\b", text))
        concrete = bool(re.search(r"\b(room|street|door|hand|face|night|light|city|stage|mic|snare|kick|crowd)\b", text))
        if feature == "voice" and high_density:
            return "Reduce syllable density 8–12% in the densest active bar and add a 180 ms pre-claim pause."
        if feature == "voice" and low_density:
            return "Keep the available space but move one stressed word onto beat 2 or 4; do not add more syllables yet."
        if feature == "words" and not concrete:
            return "Replace one abstract phrase in the displayed bars with a concrete place, object, or physical action."
        if feature == "interaction" and not direct_address:
            return "Turn one displayed proposition into a direct answer using the opponent or audience as grammatical subject."
        if feature == "interaction" and direct_address:
            return "Quote or paraphrase one exact opposing claim, then reverse its premise in the following half-bar."
        if feature == "culture" and not concrete:
            return "Add one context-legible reference and verify that the selected audience can decode it without explanation."
        if feature == "music" and high_density:
            return "Open one beat of production space under the densest phrase and keep the vocal entry unchanged."
        if feature == "person" and not re.search(r"\b(i|my|me|mine)\b", text):
            return "Add one first-person stake or consequence that could not be transferred unchanged to another artist."
        return ""
