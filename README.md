# ECG Insight Toolkit

An offline research workbench that finds heartbeats in continuous ECG recordings, labels selected beats as **N** (normal or normal-like), **S** (supraventricular), or **V** (ventricular), and shows when noise or low model confidence makes a label unreliable. The web app provides a waveform viewer, beat inspector, error queue, and benchmark dashboard. It is a research and portfolio project, not a clinical tool.

## How it works

1. Load a two-lead ECG and its reference beat annotations.
2. Filter the signal and detect candidate heartbeat locations.
3. Measure waveform shape and timing around each beat; a trained model predicts N, S, or V.
4. Estimate signal quality and withhold uncertain predictions.
5. Compare results with reference annotations and report detection, classification, abstention, and noise performance.

The latest run is **pipeline v4**: classifier v3 with the v3 detector and quality checks. Model selection uses subject-separated MIT-BIH training and validation records. The app does not accept uploads. See the [data protocol](docs/data-protocol.md), [evaluation report](docs/evaluation.md), and [classifier model card](docs/model-card-v3-classifier.md) for definitions and limitations.

## Quick start

Install Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22+, and a way to download [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/). From this directory:

```sh
uv sync --frozen
cd web && npm ci && cd ..
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ data/mitdb/1.0.0/
uv run --frozen ecg-workbench data verify --root data/mitdb/1.0.0
uv run --frozen ecg-workbench data split --root data/mitdb/1.0.0 --out artifacts/splits/v1.json
uv run --frozen ecg-workbench benchmark classify-oracle-v3 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/oracle-feature-v3.json
uv run --frozen ecg-workbench benchmark end-to-end-v4 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/end-to-end-v4.json
```

The download command uses the AWS CLI. You can instead download the files from PhysioNet into `data/mitdb/1.0.0/`. The benchmarks generate local models and reports under `artifacts/`; both directories are ignored by Git.

Start the API and web app in separate terminals:

```sh
uv run --frozen uvicorn ecg_workbench.api.app:app --host 127.0.0.1 --port 8000
```

```sh
cd web && npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The optional noise charts and separate SVDB comparison report need the additional datasets and benchmark commands in the [reproduction guide](docs/reproduce.md).

## Project map

| Path | Purpose |
| --- | --- |
| `src/ecg_workbench/data/` | WFDB loading, labels, checksum verification, and subject-separated split |
| `src/ecg_workbench/signal/` | Filtering, heartbeat detection, beat features, and quality indicators |
| `src/ecg_workbench/models/` | Classifier training and confidence calibration |
| `src/ecg_workbench/eval/` | Reference matching and reproducible benchmark reports |
| `src/ecg_workbench/api/` | Read-only API for saved runs and waveform windows |
| `web/` | React viewer and dashboard |
| `tests/` | Synthetic unit and API tests; no dataset download required |

## Checks

```sh
uv run --frozen pytest -q
uv run --frozen ruff check src tests
cd web && npm test -- --run && npm run build
```

CI runs these checks without downloading ECG data. For a containerized viewer after generating artifacts, run `docker compose up --build` and open [http://localhost:8080](http://localhost:8080).
