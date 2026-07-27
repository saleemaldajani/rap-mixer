from rap_mixer.providers.deterministic import DeterministicSemanticAnalyzer
from rap_mixer.scoring.forward import DeterministicScoringEngine
from tests.test_forward import A, B


def test_deterministic_remains_available_after_provider_failure():
    try:
        raise RuntimeError("cloud down")
    except RuntimeError:
        bundle = DeterministicScoringEngine().score(A, B)
    assert bundle.provider == "deterministic-local"
    assert DeterministicSemanticAnalyzer().analyze_bars("cold night city light")

