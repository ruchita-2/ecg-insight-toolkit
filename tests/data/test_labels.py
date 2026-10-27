from ecg_workbench.data.labels import group_beat, is_beat_annotation


def test_grouping_distinguishes_beats_from_nonbeats():
    assert group_beat("L") == "N"
    assert group_beat("A") == "S"
    assert group_beat("E") == "V"
    assert group_beat("+") is None
    assert group_beat("/") is None
    assert is_beat_annotation("/") is True
    assert is_beat_annotation("+") is False
