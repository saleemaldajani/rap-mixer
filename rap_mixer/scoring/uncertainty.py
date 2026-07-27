def evidence_uncertainty(confidence: float, missing: int = 0, noisy: bool = False) -> float:
    return min(30, 3 + 15 * (1 - confidence) + 2 * missing + (6 if noisy else 0))

