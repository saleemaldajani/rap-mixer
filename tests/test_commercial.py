from rap_mixer.analysis.commercial import commercial_advice, commercial_indicators


def test_commercial_indicators_and_advice_change_with_evidence():
    weak = {"person": 20, "words": 25, "voice": 30, "music": 30,
            "interaction": 20, "culture": 20}
    strong = {"person": 80, "words": 82, "voice": 78, "music": 80,
              "interaction": 75, "culture": 72}
    weak_rows = commercial_indicators(weak, "ideas move in a general way", 0.6)
    strong_rows = commercial_indicators(
        strong, "Cambridge stage, my hand on the mic\nCrowd calls back when the city lights", 0.85
    )
    assert sum(row["score"] for row in strong_rows) > sum(row["score"] for row in weak_rows)
    advice = commercial_advice(weak_rows, weak, "ideas move in a general way")
    assert len(advice) == 4
    assert all(row["modeled_after"] > row["current"] for row in advice)
