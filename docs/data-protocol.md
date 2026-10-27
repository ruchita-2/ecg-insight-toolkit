# Data and evaluation protocol

## Sources

- [MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/) supplies continuous two-channel ECG recordings and reference annotations.
- [MIT-BIH Noise Stress Test Database v1.0.0](https://physionet.org/content/nstdb/1.0.0/) supplies a robustness stress test reserved from fitting; its outcomes were inspected during later pipeline development.
- [MIT-BIH Supraventricular Arrhythmia Database v1.0.0](https://physionet.org/content/svdb/1.0.0/) supplies a separate S-focused reference-location evaluation corpus. It is never used for fitting or selecting the v3 classifier.

All three are licensed under Open Data Commons Attribution License v1.0. The local MIT-BIH Arrhythmia copy was checked against its official `SHA256SUMS.txt`. The 36 NSTDB waveform, header, and annotation files used for the twelve benchmark recordings were also checked against the official `SHA256SUMS.txt`, with no mismatches. Other NSTDB auxiliary files were not downloaded in this workspace. Raw recordings are excluded from Git.

The 78 SVDB records use ECG1 at 128 Hz for the frozen classifier comparison. Their 234 waveform, header, and annotation files plus `RECORDS` were checked against the published `SHA256SUMS.txt`. One normal-beat annotation in record 860 occurs exactly at the end of the waveform and is excluded because it has no signal sample. The SVDB run scores reference beat locations only; it does not claim detector or signal-quality performance on this corpus.

## Frozen split v1

The subject-separated split was generated with seed `20261025` after auditing every MIT-BIH record. Records 201 and 202, from one subject, are in validation together. Records 118 and 119 are reserved exclusively for the noise benchmark and are excluded from fitting, model selection, probability calibration, and threshold selection.

Canonical partition SHA-256: `c5e69ea99d79e895ae9a22d6a304067d1c29011ce7c9712e8534045dcbc6e73f`

| Partition | Record IDs | N beats | S beats | V beats |
| --- | --- | ---: | ---: | ---: |
| Train | 101, 102, 103, 104, 105, 106, 111, 113, 114, 115, 117, 121, 122, 123, 124, 203, 205, 208, 210, 212, 213, 215, 219, 220, 221, 232, 233 | 51,232 | 1,606 | 4,039 |
| Validation | 100, 107, 116, 200, 201, 202, 207, 214, 222, 231 | 17,368 | 564 | 1,680 |
| Test | 108, 109, 112, 209, 217, 223, 228, 230, 234 | 18,322 | 515 | 1,057 |
| Noise source | 118, 119 | 3,709 | 96 | 460 |

The counts use the first-release mapping in `src/ecg_workbench/data/labels.py`. N includes `N`, `L`, `R`, `e`, `j`; S includes `A`, `a`, `J`, `S`; V includes `V`, `E`. Other beat annotations remain in detection evaluation and are counted separately for classification. Rhythm and quality annotations are never counted as beats.

Because the classes are imbalanced, classification reports include per-class support, recall, precision, F1, and macro F1. No threshold or model choice may use the test or noise-source partitions.

Regenerate the ignored `artifacts/splits/v1.json` with:

```sh
UV_CACHE_DIR=.cache/uv UV_PYTHON_INSTALL_DIR=.cache/python uv run --frozen ecg-workbench data split --root data/mitdb/1.0.0 --out artifacts/splits/v1.json
```

The printed partition hash must equal the value above. A mismatch is a protocol change and requires a documented new split version.
