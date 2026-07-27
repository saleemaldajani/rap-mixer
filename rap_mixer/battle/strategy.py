from __future__ import annotations

from rap_mixer.schemas.battle import ArgumentNode, BattleStrategy


def plan_strategy(nodes: list[ArgumentNode], bar_count: int, tone: str, allowed: str,
                  disallowed: str, bpm: float) -> BattleStrategy:
    addressed = [x.bar_id for x in nodes if x.rebuttable] or ["H1"]
    primary = nodes[0].text if nodes else "the opponent's performance claim"
    return BattleStrategy(
        primary_angle=f"Reverse the framing of: {primary}",
        secondary_angle="Contrast specificity and technique",
        response_moves=["Direct rebuttal", "Reverse the attack", "Callback"],
        human_lines_addressed=addressed,
        facts_allowed=[x.strip() for x in allowed.split(",") if x.strip()],
        facts_disallowed=[x.strip() for x in disallowed.split(",") if x.strip()],
        desired_effect="clear responsive counter-round",
        audience_model="selected battle context",
        tone=tone,
        round_arc=["answer", "reverse", "demonstrate", "close"],
        target_bar_count=bar_count,
        rhyme_constraints={"density": "medium", "landing": "bar end"},
        cadence_constraints={"bpm": bpm, "max_syllables_per_second": 7.5},
        safety_constraints=["attack claims and craft, not protected identity", "no credible threats"],
    )

