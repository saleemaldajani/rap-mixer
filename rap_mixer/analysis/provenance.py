from rap_mixer.analysis.schemas import FeatureValue, Provenance


def manual(value: float, evidence: str = "User control") -> FeatureValue:
    return FeatureValue(value=value, provenance=Provenance.MANUAL, confidence=1, evidence=[evidence])


def measured(value: float, evidence: str, confidence: float = 0.8) -> FeatureValue:
    return FeatureValue(value=value, provenance=Provenance.AUDIO, confidence=confidence, evidence=[evidence])
