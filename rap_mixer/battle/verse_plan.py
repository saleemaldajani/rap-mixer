from __future__ import annotations

import re
from hashlib import sha256

from rap_mixer.battle.safety import safe_alternative, validate_boundaries
from rap_mixer.performance.cadence import cadence_plan
from rap_mixer.schemas.battle import ArgumentNode, BattleStrategy, GeneratedBar


def generate_bars(strategy: BattleStrategy, nodes: list[ArgumentNode], bpm: float,
                  disallowed: str = "", variation: int = 0) -> list[GeneratedBar]:
    source = nodes or [ArgumentNode(id="N1", bar_id="H1", kind="Claim", text="your claim")]
    endings = ["frame", "claim", "proof", "room", "stage", "name", "scheme", "sound"]
    templates = [
        "You called it {quote} — now your own {end} cracks",
        "That {quote} had no proof — this punch sends it back",
        "You sold {quote} — the receipt says it lied",
        "Your {quote} hit the room — my callback made it collide",
        "You reached for {quote} — I made that reach look small",
        "All {quote} no landing — I just answered it all",
        "Your {quote} rang hollow — hear the echo expose it",
        "You framed it as {quote} — one punch and I closed it",
    ]
    fingerprint = int(sha256("|".join(node.text for node in source).encode()).hexdigest()[:8], 16)
    offset = (fingerprint + variation * 3) % len(templates)
    bars = []
    for i in range(strategy.target_bar_count):
        node = source[i % len(source)]
        quote = " ".join(node.text.split()[:6]).rstrip(".,!?")
        text = templates[(offset + i) % len(templates)].format(
            quote=quote, end=endings[(offset + i) % len(endings)], bar=node.bar_id[1:]
        )
        safe, reason = validate_boundaries(text, disallowed)
        if not safe:
            text = safe_alternative()
        cadence = cadence_plan(text, bpm)
        words = re.findall(r"[a-z]+", text.lower())
        bars.append(GeneratedBar(
            bar_number=i + 1, text=text, function=strategy.round_arc[i % 4],
            addressed_human_bar_ids=[node.bar_id], rhyme_family=words[-1][-3:] if words else None,
            target_syllables=cadence["target_syllables"], stress_pattern=cadence["stress_pattern"],
            intended_start_beat=i * 4, intended_end_beat=(i + 1) * 4,
            delivery_note=f"Land rhyme on beat 4; {cadence['pause']}", confidence=0.72,
            warnings=cadence["warnings"] + ([] if safe else [reason]),
        ))
    return bars


def repeated_angles(bars: list[GeneratedBar]) -> bool:
    addressed = [tuple(x.addressed_human_bar_ids) for x in bars]
    return len(addressed) > 2 and len(set(addressed)) == 1
