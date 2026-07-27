from rap_mixer.scoring.forward import DeterministicScoringEngine, output_map

A = {x: 60 for x in ["person", "words", "voice", "music", "interaction", "culture"]}
B = {"flow": 1, "clarity": 1, "groove": 1, "response": 1, "replay": 1, "lineage": 1}


def test_forward_is_deterministic_and_bounded():
    scorer = DeterministicScoringEngine()
    one, two = scorer.score(A, B), scorer.score(A, B)
    assert one == two
    assert all(0 <= x.score <= 100 for x in one.outputs)


def test_context_changes_scores_not_a():
    original = A.copy()
    scorer = DeterministicScoringEngine()
    first = output_map(scorer.score(A, B))
    second = output_map(scorer.score(A, {**B, "groove": 1.5}))
    assert first != second
    assert A == original

