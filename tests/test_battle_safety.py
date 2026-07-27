from rap_mixer.battle.safety import validate_boundaries


def test_threats_protected_attacks_and_disallowed_topics_rejected():
    assert not validate_boundaries("I will shoot you")[0]
    assert not validate_boundaries("Your race is the target")[0]
    assert not validate_boundaries("Your family angle", "family")[0]
    assert validate_boundaries("Your claim contradicts your last bar")[0]

