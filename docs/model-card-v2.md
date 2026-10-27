# Model card: ECG beat classifier v2

## Intended use

Research and portfolio demonstration on public MIT-BIH recordings. This is not validated for diagnosis, triage, monitoring, or patient care.

## Model and training

The classifier is a 150-iteration histogram gradient boosting model using 20 hand-built waveform, spectral, timing, and within-record context features from channel 0. Training uses 27 subject-separated MIT-BIH records; validation uses 10; test uses 9. The selected model was chosen by validation macro F1 from four bounded tree-model candidates. Class-balanced sample weights prevent N from dominating; S examples from any one record have a capped contribution of 100 before the S weight total is restored. This reduces reliance on record 232, which contains 1,382 of the 1,606 training S beats. No SMOTE or NSTDB noise augmentation was used.

V2 computes timing context from every reference beat during training, then selects the N/S/V labelled rows. Inference computes the same features from all detected beats. The detector and heuristic signal-quality logic remain v1.

## Observed results

| Held-out metric | N | S | V |
| --- | ---: | ---: | ---: |
| Reference-location precision | 96.6% | 11.5% | 66.5% |
| Reference-location recall | 90.3% | 32.6% | 94.2% |
| Reference-location F1 | 0.938 | 0.171 | 0.780 |
| Continuous end-to-end recall after abstention | 71.5% | 6.6% | 78.3% |

Reference-location macro F1 was 0.630, versus 0.579 for v1. The full pipeline abstained on 5,370/21,479 detections (25.0%). Its accepted, matched S predictions were correct 34/64 times (53.1%), but it correctly delivered only 34/515 true S beats. S recognition therefore remains insufficient for dependable use. V recall improved, but V reference-location precision decreased. See [evaluation.md](evaluation.md) for exact counts and definitions.

## Limitations and provenance

The same nine-record test was examined during v1 and an earlier v2 experiment, so this v2 result is an exploratory comparison, not an untouched blind test. NSTDB was reserved from fitting and was run only as an already-seen comparative stress test. V2 still misses every beat in the nominally clean intervals at −6 dB and abstains on every 24 dB noisy-interval prediction despite finding nearly all beats there. At −6 dB noisy intervals it allows 326 predictions through, only 41 of which are correct matched N/S/V classifications. Signal quality remains a heuristic based on waveform proxies, not comprehensive expert usability labels. The source corpus is historical and small. External validation on genuinely unseen recordings is required before claiming robust generalization.

The generated oracle and end-to-end reports record model ID, dataset version, split hash, class support, per-record outcomes, and confidence settings. Raw data and model artifacts are excluded from Git.
