from rap_mixer.audio.bars import latest_four_completed, segment_text


def test_latest_four_and_history():
    bars = segment_text("one two three four five six seven eight nine ten " * 8, 24, 120)
    assert latest_four_completed(bars) == bars[-4:]
    assert len(bars) > len(latest_four_completed(bars))

