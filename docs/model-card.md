# Model card: ECG beat analysis v1

## Intended use

Research, learning, and portfolio demonstration on the public MIT-BIH ECG corpus. Outputs are reviewable beat categories and reliability estimates. The system is not validated for diagnosis, triage, monitoring, or patient care.

## Model and data

The primary classifier is a balanced random forest on 17 morphology, spectral, and past-only timing features. Training uses 27 MIT-BIH records; validation uses 10; test uses 9. Records 118 and 119 are reserved for noise robustness. Records 201 and 202 are grouped by subject. The exact split is in [data-protocol.md](data-protocol.md). A scaled logistic regression and compact 1D CNN are comparisons. Detection uses an offline Pan–Tompkins-style algorithm. Categories are N/S/V grouped beat annotations.

## Observed held-out results

| Setting | N | S | V |
| --- | ---: | ---: | ---: |
| Reference-location F1 | 0.933 (n=18,322) | 0.005 (n=515) | 0.799 (n=1,057) |
| Continuous end-to-end recall | 0.749 | 0.000 | 0.656 |

Detection yielded 21,464 one-to-one matches, 250 misses, and 15 extras. The full pipeline abstained on 4,720 of 21,479 predictions. S beats are effectively unsupported by this trained system despite appearing in the benchmark. This failure must remain visible in any presentation of results.

## Reliability and limits

Probabilities are temperature-calibrated on validation. A validation-selected score threshold and signal-quality floor govern abstention. Quality uses waveform proxies, not comprehensive expert usability labels. A score is not proof that an individual prediction is correct. MIT-BIH is a historical, selected Holter corpus with limited subjects and acquisition conditions. NSTDB has only two source people repeated at six noise levels. Neither dataset establishes modern clinical generalization, demographic fairness, or safety across devices.

The zero-phase filter is offline. This release does not classify rhythms, perform diagnosis, learn R-peak detection, or support live devices. External validation, stronger S-beat modeling, and clinical study would be needed before medical use could be considered.

The locked NSTDB run exposed additional limitations: mixed severe noise can suppress even clean-interval detections because the detector uses a record-wide threshold, and the quality proxy flags nearly all high-SNR NSTDB beats despite strong detection. The system therefore cannot claim reliable abstention under domain shift. See the [evaluation protocol](evaluation.md) for the paired noise results.

## Traceability

Every generated report includes dataset version, split SHA-256, model ID, class support, and record-level evidence. Raw waveforms and model artifacts are excluded from Git. See the [README](../README.md) and [metric definitions](evaluation.md) to reproduce and interpret the results.
