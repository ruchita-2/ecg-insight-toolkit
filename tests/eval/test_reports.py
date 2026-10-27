import json

import pytest

from ecg_workbench.eval.reports import write_report


def test_report_is_valid_json_and_rejects_nan(tmp_path):
    path = tmp_path / "report.json"
    write_report(path, {"count": 2})
    assert json.loads(path.read_text()) == {"count": 2}
    with pytest.raises(ValueError):
        write_report(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"count": 2}
