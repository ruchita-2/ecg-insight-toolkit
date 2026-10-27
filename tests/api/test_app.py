import json

from fastapi.testclient import TestClient

from ecg_workbench.api.app import RunCatalog, create_app


def _client(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "end-to-end-demo.json").write_text(json.dumps({
        "dataset": "mitdb", "dataset_version": "1.0.0",
        "split_sha256": "abc", "model_id": "forest-v1",
        "detector_id": "pan-v1", "pooled": {},
        "records": {"100": {"rows": [], "metrics": {}}},
    }))
    return TestClient(create_app(RunCatalog(runs, tmp_path / "data")))


def test_window_is_bounded(tmp_path):
    response = _client(tmp_path).get(
        "/api/runs/end-to-end-demo/records/100/window?start=0&duration=11"
    )
    assert response.status_code == 422


def test_unknown_record_is_not_a_file_path(tmp_path):
    response = _client(tmp_path).get("/api/runs/end-to-end-demo/records/..%2Fsecret")
    assert response.status_code in (400, 404)


def test_summary_has_provenance(tmp_path):
    response = _client(tmp_path).get("/api/runs/end-to-end-demo/summary")
    assert response.status_code == 200
    assert response.json()["split_sha256"] == "abc"


def test_newest_run_uses_its_own_classifier_report(tmp_path):
    client = _client(tmp_path)
    runs = tmp_path / "runs"
    original = json.loads((runs / "end-to-end-demo.json").read_text())
    for version in ("v1", "v2"):
        (runs / f"end-to-end-{version}.json").write_text(json.dumps({
            **original, "model_id": f"model-{version}",
        }))
        (runs / f"oracle-feature-{version}.json").write_text(json.dumps({
            "model_id": f"model-{version}",
        }))
        (runs / f"noise-{version}.json").write_text(json.dumps({
            "model_id": f"model-{version}",
        }))
    assert client.get("/api/runs").json()[0]["run_id"] == "end-to-end-v2"
    response = client.get("/api/runs/end-to-end-v2/summary")
    assert response.json()["oracle_feature"]["model_id"] == "model-v2"
    assert response.json()["noise"]["model_id"] == "model-v2"


def test_pipeline_v3_uses_v2_classifier_and_v3_benchmarks(tmp_path):
    client = _client(tmp_path)
    runs = tmp_path / "runs"
    original = json.loads((runs / "end-to-end-demo.json").read_text())
    (runs / "end-to-end-v3.json").write_text(json.dumps({
        **original, "model_id": "classifier-v2",
    }))
    (runs / "oracle-feature-v2.json").write_text(json.dumps({
        "model_id": "classifier-v2",
    }))
    (runs / "detection-v3.json").write_text(json.dumps({"model_id": "detector-v3"}))
    (runs / "noise-v3.json").write_text(json.dumps({"model_id": "classifier-v2"}))
    summary = client.get("/api/runs/end-to-end-v3/summary").json()
    assert summary["oracle_feature"]["model_id"] == "classifier-v2"
    assert summary["detection"]["model_id"] == "detector-v3"
    assert summary["noise"]["model_id"] == "classifier-v2"


def test_pipeline_v4_pairs_v3_classifier_with_v3_detector(tmp_path):
    client = _client(tmp_path)
    runs = tmp_path / "runs"
    original = json.loads((runs / "end-to-end-demo.json").read_text())
    (runs / "end-to-end-v4.json").write_text(json.dumps({
        **original, "model_id": "classifier-v3",
    }))
    (runs / "oracle-feature-v3.json").write_text(json.dumps({
        "model_id": "classifier-v3",
    }))
    (runs / "detection-v3.json").write_text(json.dumps({"model_id": "detector-v3"}))
    (runs / "noise-v4.json").write_text(json.dumps({"model_id": "classifier-v3"}))
    summary = client.get("/api/runs/end-to-end-v4/summary").json()
    assert summary["oracle_feature"]["model_id"] == "classifier-v3"
    assert summary["detection"]["model_id"] == "detector-v3"
    assert summary["noise"]["model_id"] == "classifier-v3"
