from __future__ import annotations

import re

PROTECTED_ATTACKS = re.compile(r"\b(race|religion|disability|ethnicity|sexual orientation)\b", re.I)
THREATS = re.compile(r"\b(kill|shoot|stab|dox|address|self-harm|suicide)\b", re.I)


def validate_boundaries(text: str, disallowed: str = "") -> tuple[bool, str]:
    if THREATS.search(text):
        return False, "credible threats, private data, and self-harm content are outside this battle"
    if PROTECTED_ATTACKS.search(text):
        return False, "protected characteristics cannot be targets"
    forbidden = [x.strip().lower() for x in disallowed.split(",") if x.strip()]
    if any(x in text.lower() for x in forbidden):
        return False, "a configured disallowed topic appeared"
    return True, "passed"


def safe_alternative() -> str:
    return "I won't chase that premise—I'll battle the claim and the craft instead."

