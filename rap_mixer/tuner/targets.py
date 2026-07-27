from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "config/tuner_targets.yaml").read_text())
PROFILES = CONFIG["profiles"]
TOLERANCES = CONFIG["tolerances"]
CONTEXT_PRIORITY = CONFIG["context_priority"]


def target_values(profile: str, manual: dict[str, float] | None = None) -> dict[str, float]:
    if profile == "Custom" and manual:
        return dict(manual)
    return dict(PROFILES.get(profile, PROFILES["Clear and direct"]))
