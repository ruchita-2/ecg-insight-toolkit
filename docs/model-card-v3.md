# Reliability pipeline card: ECG beat analysis v3

## Intended use

Research and portfolio demonstration on public ECG recordings. The displayed N/S/V beat labels and signal-quality scores are **not validated for diagnosis, triage, monitoring, or patient care**.

## What changed

V3 uses the exact v2 histogram gradient boosting beat classifier, trained only on MIT-BIH Arrhythmia training records. No NSTDB examples, synthetic noisy beats, or SMOTE samples enter classifier training. The v2 model's [training and class definitions](model-card-v2.md) still apply.

V3 changes the surrounding pipeline:

- **Detection:** the original whole-record detector searches again with a 30-second local threshold only where it left a gap longer than one minute.
- **Quality:** constant ECG voltage offsets no longer automatically imply baseline drift. Two ECG leads are compared in 30-second blocks; when the second lead is insufficient, primary-lead QRS shape stability is used. A severe QRS instability check applies even when both leads have enough detections. The morphology limits come from training records; the quality floor comes from validation.
- **Abstention:** predictions below the calibrated class-score threshold or quality floor are withheld. S predictions also need the conservative original quality check. The viewer reports whether the reason was model score, signal quality, or S confirmation.

The quality score is an engineering proxy for analysability, not an expert-labelled ECG quality diagnosis. A high score does not prove that a beat category is correct.

## Observed results

The held-out nine-record MIT-BIH test retained 21,464 correctly located beats, 250 misses, and 15 extra detections. Accepted correct classifications were N 13,070/18,322 (71.3%), S 34/515 (6.6%), and V 794/1,057 (75.1%). The pipeline abstained on 5,443/21,479 detections (25.3%). Relative to v2, S correct counts were unchanged, N fell by 25, and V fell by 34. S matched accepted precision was 34/63 (54.0%), but recall remains too low for dependable recognition.

The separate [NSTDB stress test](https://physionet.org/content/nstdb/1.0.0/) adds controlled noise to source records 118 and 119. At −6 dB, v3 detected 2,406/2,409 beats in the nominally clean intervals, versus none with v2. It withheld all predictions in the noisy intervals at −6 and 0 dB. At +24 dB noisy intervals, it correctly matched 796 of 937 accepted predictions. Source 118 still had only 35 correct matched classifications among 176 accepted noisy-interval predictions, showing that its classifier errors can pass the quality gate.

## Limits of this evidence

The source records and MIT-BIH test outcomes were repeatedly examined during development. The v3 comparison is exploratory and has no fresh blind test. NSTDB contains only two independent source records for this benchmark. Severe-noise abstention can hide useful beats, and V recall is lower than v2 on the original held-out set. The system does not predict rhythm diagnoses, accept uploaded patient data, or offer clinical interpretation. External expert-labelled quality data and genuinely unseen ECG sources are needed to assess generalization.

The [evaluation report](evaluation.md) defines every metric and gives per-source caveats. Generated reports record the split hash, detector and model IDs, thresholds, and per-record counts; raw data and models are excluded from Git.
