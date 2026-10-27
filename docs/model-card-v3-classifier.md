# Model card: ECG beat classifier v3

## Intended use

Research and portfolio demonstration on public ECG recordings. The classifier assigns N, S, or V to a detected beat. Its predictions are not validated for diagnosis, triage, monitoring, or patient care.

## Training and inference

The selected model is a 150-iteration histogram gradient boosting classifier with 24 hand-built features. It uses waveform amplitude and shape, spectral power, prior and following beat timing, and QRS similarity to recent beats. The following interval makes inference offline. Training uses only the 27 MIT-BIH Arrhythmia training records; 10 subject-separated validation records select among four bounded models by macro F1. S class weights are balanced and any one training record is capped at the equivalent of 100 S beats before restoring total S weight. No SMOTE or external data are used in training.

Reference-location fitting uses all annotated beat positions for timing context, then keeps only N/S/V labels for fitting. Continuous inference uses detected beat positions. Full pipeline v4 combines this classifier with the v3 detector and quality rules described in the [reliability card](model-card-v3.md). Model score and signal-quality checks may withhold a category.

## Results

| Evaluation | V2 classifier | V3 classifier |
| --- | ---: | ---: |
| MIT-BIH validation macro F1 | 0.631 | 0.742 |
| MIT-BIH validation S F1 | 0.293 | 0.351 |
| MIT-BIH previously viewed test S F1 | 0.171 | 0.173 |
| SVDB external S precision | 25.3% | 28.9% |
| SVDB external S recall | 41.0% | 40.7% |
| SVDB external S F1 | 0.313 | 0.338 |

The external result covers all 78 MIT-BIH Supraventricular Arrhythmia Database recordings and 12,198 S beats at known reference locations. External macro F1 rose from 0.496 to 0.581, partly because N and V classification improved. On the original MIT-BIH test, the full v4 pipeline correctly accepted 37/515 S beats (7.2%) and 762/1,057 V beats (72.1%). The preceding v3 pipeline correctly accepted 34/515 S (6.6%) and 794/1,057 V (75.1%). V4 accepted matched S precision was 37/47 (78.7%), with more abstentions overall. See [evaluation.md](evaluation.md) for metric definitions and exact counts.

## Limits

The original MIT-BIH test and NSTDB were viewed during earlier iterations, so their v3/v4 comparisons are exploratory. The SVDB check is a separate corpus, but its different 128 Hz sampling rate and recording selection create distribution shift, and subject overlap with MIT-BIH cannot be ruled out from the supplied metadata. Its result scores classifier decisions at annotation locations only. S recall remains weak and does not improve on SVDB; the full pipeline withholds most S beats. V4's higher S accepted precision comes with reduced V recall. The model is unsuitable for clinical decisions.

The generated model package and reports record feature version, class mapping, split hash, model ID, seed, per-record outcomes, thresholds, and verified external files. Raw ECG files and model artifacts are excluded from Git.
