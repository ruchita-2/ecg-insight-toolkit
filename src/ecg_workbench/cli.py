"""Command-line entry point for reproducible dataset and benchmark tasks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ecg_workbench.data.labels import group_beat, is_beat_annotation
from ecg_workbench.data.manifest import verify_dataset
from ecg_workbench.data.records import audit_record, load_record
from ecg_workbench.data.splits import (
    SplitManifest,
    create_split,
    split_digest,
    support_by_split,
)
from ecg_workbench.eval.classification import benchmark_oracle
from ecg_workbench.eval.end_to_end import benchmark_end_to_end
from ecg_workbench.eval.events import match_events
from ecg_workbench.eval.external import benchmark_svdb
from ecg_workbench.eval.metrics import detection_metrics, pooled_detection
from ecg_workbench.eval.noise import (
    NSTDB_REQUIRED_FILES,
    assert_noise_isolation,
    load_frozen_pipeline,
    noise_benchmark,
)
from ecg_workbench.eval.reports import write_report
from ecg_workbench.signal.detectors import (
    detect_local,
    detect_pan_tompkins,
    detect_pan_tompkins_gap_rescue,
    detect_pan_tompkins_local,
)
from ecg_workbench.signal.filters import preprocess


def main() -> None:
    parser = argparse.ArgumentParser(prog="ecg-workbench")
    group = parser.add_subparsers(dest="group", required=True)
    data = group.add_parser("data")
    data_commands = data.add_subparsers(dest="action", required=True)
    verify = data_commands.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--nstdb-benchmark-only", action="store_true")
    audit = data_commands.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--record")
    split = data_commands.add_parser("split")
    split.add_argument("--root", type=Path, required=True)
    split.add_argument("--out", type=Path, required=True)
    benchmark = group.add_parser("benchmark")
    benchmark_commands = benchmark.add_subparsers(dest="action", required=True)
    for action in ("detection", "detection-v3"):
        detection = benchmark_commands.add_parser(action)
        detection.add_argument("--split", type=Path, required=True)
        detection.add_argument("--root", type=Path, required=True)
        detection.add_argument("--out", type=Path, required=True)
    for action in ("classify-oracle", "classify-oracle-v2", "classify-oracle-v3"):
        classify = benchmark_commands.add_parser(action)
        classify.add_argument("--split", type=Path, required=True)
        classify.add_argument("--root", type=Path, required=True)
        classify.add_argument("--out", type=Path, required=True)
    waveform = benchmark_commands.add_parser("classify-waveform")
    waveform.add_argument("--split", type=Path, required=True)
    waveform.add_argument("--root", type=Path, required=True)
    waveform.add_argument("--out", type=Path, required=True)
    for action in ("end-to-end", "end-to-end-v2", "end-to-end-v3", "end-to-end-v4"):
        full = benchmark_commands.add_parser(action)
        full.add_argument("--split", type=Path, required=True)
        full.add_argument("--root", type=Path, required=True)
        full.add_argument("--out", type=Path, required=True)
    for action in ("noise", "noise-v2", "noise-v3", "noise-v4"):
        noise = benchmark_commands.add_parser(action)
        noise.add_argument("--split", type=Path, required=True)
        noise.add_argument("--root", type=Path, required=True)
        noise.add_argument("--out", type=Path, required=True)
    external = benchmark_commands.add_parser("external-svdb")
    external.add_argument("--split", type=Path, required=True)
    external.add_argument("--root", type=Path, required=True)
    external.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.group == "data" and args.action == "verify":
        problems = verify_dataset(
            args.root, NSTDB_REQUIRED_FILES if args.nstdb_benchmark_only else None
        )
        for problem in problems:
            print(problem)
        if problems:
            raise SystemExit(1)
        print(f"verified dataset files in {args.root}")
    elif args.group == "data" and args.action == "audit":
        if args.record:
            record_ids = [args.record]
        else:
            index = args.root / "RECORDS"
            if not index.is_file():
                raise FileNotFoundError(f"missing record index: {index}")
            record_ids = [line.strip() for line in index.read_text().splitlines() if line.strip()]
        reports = [audit_record(load_record(args.root, record_id)) for record_id in record_ids]
        print(json.dumps(reports, indent=2, sort_keys=True))
    elif args.group == "data" and args.action == "split":
        index = args.root / "RECORDS"
        if not index.is_file():
            raise FileNotFoundError(f"missing record index: {index}")
        record_ids = [line.strip() for line in index.read_text().splitlines() if line.strip()]
        counts = {}
        for record_id in record_ids:
            record = load_record(args.root, record_id)
            counts[record_id] = Counter(
                group for mark in record.annotations
                if (group := group_beat(mark.symbol)) is not None
            )
        manifest = create_split(counts)
        support = support_by_split(manifest, counts)
        for partition in ("train", "validation", "test"):
            missing = [label for label, count in support[partition].items() if count == 0]
            if missing:
                raise ValueError(f"{partition} has no examples of: {', '.join(missing)}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": "mitdb",
            "dataset_version": "1.0.0",
            "seed": 20261025,
            "partitions": manifest.to_dict(),
            "class_support": support,
            "split_sha256": split_digest(manifest),
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        args.out.write_text(serialized)
        print(serialized, end="")
    elif args.group == "benchmark" and args.action in ("detection", "detection-v3"):
        manifest = json.loads(args.split.read_text())
        record_ids = manifest["partitions"]["test"]
        results = {}
        detectors = [
            ("local", detect_local), ("pan_tompkins_style", detect_pan_tompkins)
        ]
        if args.action == "detection-v3":
            detectors.append(("pan_tompkins_local", detect_pan_tompkins_local))
            detectors.append(("pan_tompkins_gap_rescue", detect_pan_tompkins_gap_rescue))
        for name, detector in detectors:
            rows = []
            for record_id in record_ids:
                record = load_record(args.root, record_id)
                signal = preprocess(record.signals[:, 0], record.fs)
                reference = np.array([
                    mark.sample for mark in record.annotations
                    if is_beat_annotation(mark.symbol)
                ], dtype=int)
                predictions = detector(signal, record.fs)
                matches = match_events(reference, predictions, round(0.15 * record.fs))
                hours = len(signal) / record.fs / 3600
                row = detection_metrics(matches, hours, record.fs)
                row.update({"record_id": record_id, "hours": hours})
                rows.append(row)
            results[name] = {"records": rows, "pooled": pooled_detection(rows)}
        payload = {
            "dataset": "mitdb", "dataset_version": "1.0.0",
            "split_sha256": manifest["split_sha256"],
            "model_id": "detector-comparison-v3" if args.action == "detection-v3" else "detector-comparison-v1",
            "partition": "test", "lead": 0, "tolerance_ms": 150,
            "preprocessing": "offline zero-phase Butterworth 0.5-40 Hz",
            "detectors": results,
        }
        write_report(args.out, payload)
        print(json.dumps({name: result["pooled"] for name, result in results.items()}, indent=2))
    elif args.group == "benchmark" and args.action in (
        "classify-oracle", "classify-oracle-v2", "classify-oracle-v3",
    ):
        manifest = json.loads(args.split.read_text())
        payload = benchmark_oracle(
            args.root, manifest["partitions"], manifest["split_sha256"],
            args.out.parent.parent / "models",
            version=(
                "v3" if args.action.endswith("v3") else
                "v2" if args.action.endswith("v2") else "v1"
            ),
        )
        write_report(args.out, payload)
        print(json.dumps({"selected": payload["model_id"], "test": payload["test"]}, indent=2))
    elif args.group == "benchmark" and args.action == "classify-waveform":
        from ecg_workbench.eval.waveform import benchmark_waveform

        manifest = json.loads(args.split.read_text())
        payload = benchmark_waveform(
            args.root, manifest["partitions"], manifest["split_sha256"],
            args.out.parent.parent / "models",
        )
        write_report(args.out, payload)
        print(json.dumps({"model": payload["model_id"], "test": payload["test"]}, indent=2))
    elif args.group == "benchmark" and args.action in (
        "end-to-end", "end-to-end-v2", "end-to-end-v3", "end-to-end-v4",
    ):
        manifest = json.loads(args.split.read_text())
        version = (
            "v1" if args.action == "end-to-end" else
            "v3" if args.action == "end-to-end-v4" else "v2"
        )
        oracle = json.loads((args.out.parent / f"oracle-feature-{version}.json").read_text())
        model_path = args.out.parent.parent / "models" / f"{oracle['model_id']}.joblib"
        payload = benchmark_end_to_end(
            args.root, manifest["partitions"], manifest["split_sha256"], model_path,
            pipeline_version="v3" if args.action in ("end-to-end-v3", "end-to-end-v4") else "v1",
        )
        write_report(args.out, payload)
        print(json.dumps(payload["pooled"], indent=2))
    elif args.group == "benchmark" and args.action in (
        "noise", "noise-v2", "noise-v3", "noise-v4",
    ):
        problems = verify_dataset(args.root, NSTDB_REQUIRED_FILES)
        if problems:
            raise ValueError(f"NSTDB file verification failed: {problems[:5]}")
        manifest = json.loads(args.split.read_text())
        assert_noise_isolation(SplitManifest(**{
            key: tuple(value) for key, value in manifest["partitions"].items()
        }))
        version = "v1" if args.action == "noise" else args.action.rsplit("-", 1)[-1]
        full_report = json.loads((args.out.parent / f"end-to-end-{version}.json").read_text())
        if full_report["split_sha256"] != manifest["split_sha256"]:
            raise ValueError("end-to-end report and split differ")
        model_path = args.out.parent.parent / "models" / f"{full_report['model_id']}.joblib"
        frozen = load_frozen_pipeline(model_path, full_report)
        payload = noise_benchmark(args.root, frozen, manifest["split_sha256"])
        write_report(args.out, payload)
        print(json.dumps(payload["summary"], indent=2))
    elif args.group == "benchmark" and args.action == "external-svdb":
        manifest = json.loads(args.split.read_text())
        models_root = args.out.parent.parent / "models"
        oracle_v2 = json.loads((args.out.parent / "oracle-feature-v2.json").read_text())
        oracle_v3 = json.loads((args.out.parent / "oracle-feature-v3.json").read_text())
        payload = benchmark_svdb(
            args.root,
            {
                "v2": models_root / f"{oracle_v2['model_id']}.joblib",
                "v3": models_root / f"{oracle_v3['model_id']}.joblib",
            },
            manifest["split_sha256"],
        )
        write_report(args.out, payload)
        print(json.dumps({
            name: result["pooled"] for name, result in payload["models"].items()
        }, indent=2))
