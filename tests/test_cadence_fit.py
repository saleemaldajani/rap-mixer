from rap_mixer.performance.cadence import cadence_plan


def test_density_violation_and_timing():
    clear = cadence_plan("short claim lands clean", 120)
    dense = cadence_plan("rapid " * 40, 120)
    assert clear["duration"] == 2
    assert "Too many syllables for clear delivery" in dense["warnings"]

