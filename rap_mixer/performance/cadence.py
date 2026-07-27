from __future__ import annotations

from rap_mixer.analysis.lyrics import syllables


def cadence_plan(text: str, bpm: float, beats: float = 4) -> dict:
    count = sum(syllables(x) for x in text.split())
    duration = beats * 60 / max(1, bpm)
    rate = count / duration
    warnings = []
    if rate > 7.5:
        warnings.append("Too many syllables for clear delivery")
    if count > 0 and len(text.split()) > 12:
        warnings.append("Limited breath space")
    return {"target_syllables": count, "duration": duration, "syllables_per_second": rate,
            "stress_pattern": [1 if i % 4 == 0 else 0 for i in range(count)],
            "pause": "after beat 3", "rhyme_landing": "beat 4", "warnings": warnings}

