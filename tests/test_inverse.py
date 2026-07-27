from rap_mixer.scoring.inverse import RecommendationEngine
from tests.test_forward import A, B


def test_inverse_respects_protection_and_fixed_context():
    context = B.copy()
    result = RecommendationEngine().recommend(A, context, ["Intelligibility"], protected_features={"voice"}, protected_outputs={"Musicality"})
    assert all(x.parameter != "voice" for x in result.recommendations)
    assert result.after["Musicality"] >= result.before["Musicality"] - 2
    assert context == B

