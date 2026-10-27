# Reproduce benchmark reports

Run commands from the repository root after `uv sync --frozen --extra ml`. The `ml` extra is needed only for the optional waveform CNN baseline. Install web dependencies with `cd web && npm ci && cd ..` if you want the viewer. Source datasets and generated outputs stay in ignored `data/` and `artifacts/` directories.

## Download and verify data

All datasets are open access from [PhysioNet](https://physionet.org/). The commands below use the AWS CLI; direct downloads from the linked dataset pages work too.

```sh
aws s3 sync --no-sign-request s3://physionet-open/mitdb/1.0.0/ data/mitdb/1.0.0/
aws s3 sync --no-sign-request s3://physionet-open/nstdb/1.0.0/ data/nstdb/1.0.0/
aws s3 sync --no-sign-request s3://physionet-open/svdb/1.0.0/ data/svdb/1.0.0/ --exclude '*' --include '*.dat' --include '*.hea' --include '*.atr' --include 'RECORDS' --include 'SHA256SUMS.txt'
uv run --frozen ecg-workbench data verify --root data/mitdb/1.0.0
uv run --frozen ecg-workbench data verify --root data/nstdb/1.0.0 --nstdb-benchmark-only
```

The SVDB benchmark verifies all 78 records it uses against PhysioNet's checksums. Dataset versions, class mapping, and partition membership are in [data-protocol.md](data-protocol.md).

## Generate reports

Each command writes its report to `artifacts/runs/` and required model packages to `artifacts/models/`. Keep the order below because later commands load earlier artifacts.

```sh
uv run --frozen ecg-workbench data split --root data/mitdb/1.0.0 --out artifacts/splits/v1.json
uv run --frozen ecg-workbench benchmark detection --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/detection-v1.json
uv run --frozen ecg-workbench benchmark classify-oracle --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/oracle-feature-v1.json
uv run --frozen --extra ml ecg-workbench benchmark classify-waveform --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/oracle-waveform-v1.json
uv run --frozen ecg-workbench benchmark end-to-end --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/end-to-end-v1.json
uv run --frozen ecg-workbench benchmark noise --split artifacts/splits/v1.json --root data/nstdb/1.0.0 --out artifacts/runs/noise-v1.json
uv run --frozen ecg-workbench benchmark classify-oracle-v2 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/oracle-feature-v2.json
uv run --frozen ecg-workbench benchmark end-to-end-v2 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/end-to-end-v2.json
uv run --frozen ecg-workbench benchmark noise-v2 --split artifacts/splits/v1.json --root data/nstdb/1.0.0 --out artifacts/runs/noise-v2.json
uv run --frozen ecg-workbench benchmark detection-v3 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/detection-v3.json
uv run --frozen ecg-workbench benchmark end-to-end-v3 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/end-to-end-v3.json
uv run --frozen ecg-workbench benchmark noise-v3 --split artifacts/splits/v1.json --root data/nstdb/1.0.0 --out artifacts/runs/noise-v3.json
uv run --frozen ecg-workbench benchmark classify-oracle-v3 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/oracle-feature-v3.json
uv run --frozen ecg-workbench benchmark end-to-end-v4 --split artifacts/splits/v1.json --root data/mitdb/1.0.0 --out artifacts/runs/end-to-end-v4.json
uv run --frozen ecg-workbench benchmark noise-v4 --split artifacts/splits/v1.json --root data/nstdb/1.0.0 --out artifacts/runs/noise-v4.json
uv run --frozen ecg-workbench benchmark external-svdb --split artifacts/splits/v1.json --root data/svdb/1.0.0 --out artifacts/runs/external-svdb-v3.json
```

The split SHA-256 should be `c5e69ea99d79e895ae9a22d6a304067d1c29011ce7c9712e8534045dcbc6e73f`. Pipeline v4 uses classifier v3 with detector/quality v3. The [evaluation report](evaluation.md) explains each metric and why older test and noise results are exploratory.
