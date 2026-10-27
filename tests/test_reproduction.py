from ecg_workbench.eval.reports import validate_report


def test_every_report_has_provenance_and_support():
    valid = {"dataset_version": "1.0.0", "split_sha256": "abc", "model_id": "feature-v1", "metrics": [{"support": 2}]}
    assert validate_report(valid) == []
    assert "split_sha256" in validate_report({**valid, "split_sha256": ""})
