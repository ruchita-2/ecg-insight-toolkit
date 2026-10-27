# Evaluation and result interpretation

## Frozen experiment

MIT-BIH Arrhythmia Database v1.0.0 is split by subject into train, validation, and test. Records 201 and 202 stay together; records 118 and 119 are reserved for NSTDB. The exact IDs and SHA-256 are in [data-protocol.md](data-protocol.md). Channel 0 is filtered with a zero-phase 0.5–40 Hz Butterworth filter. Channel 1 is displayed for comparison. This offline filter uses future samples and is not a streaming claim.

The classifier predicts N (`N,L,R,e,j`), S (`A,a,J,S`), and V (`V,E`). Fusion, paced, and unclassifiable beats are excluded from three-class model fitting and scoring, but remain in detection ground truth. Non-beat marks are not heartbeats. Reports include excluded counts.

## Detection

Locations are paired one-to-one within ±150 ms, choosing smallest absolute offset without reusing an event. A matched pair is TP; an unmatched annotation is FN; an unmatched prediction is FP. Sensitivity = TP/(TP+FN), positive predictivity = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN). False detections per hour use record duration. Aggregate rates use summed counts. Timing error is reported in milliseconds.

On nine held-out records, the offline Pan–Tompkins-style detector gave 21,464 matches, 250 misses, and 15 extras: 98.85% sensitivity and 99.93% positive predictivity. The local maximum baseline gave 21,311 matches, 403 misses, and 15,694 extras: 98.14% sensitivity and 57.59% positive predictivity. The Pan–Tompkins-style path is an adaptation, not a bit-exact implementation of the original algorithm.

## Reference-location classification

Using annotation locations isolates classification from detection mistakes. Models fit training records only; validation macro F1 chooses between balanced logistic regression and random forest. The compact CNN uses training-only normalization and validation early stopping. Reports give per-class precision, recall, F1, support, confusion matrix, macro F1, and record-level results.

The selected random forest scored macro F1 0.579 across 19,894 in-scope test beats: N F1 0.933 (n=18,322), S F1 0.005 (n=515), V F1 0.799 (n=1,057). The compact CNN scored macro F1 0.370. S is the principal failure. Overall accuracy would hide it without class support and macro F1.

## Continuous end-to-end analysis

The selected detector produces candidate locations. Each candidate receives a feature-based category, a validation-fitted temperature-calibrated class score, and a separate signal-quality estimate. The abstention threshold is selected on validation with at least 80% coverage. Quality indicators cover baseline drift, high-frequency residue, flatline, clipping, and detector disagreement; scaling fits training records and the floor uses validation. MIT-BIH does not provide complete clinical usability labels for every segment.

The recording viewer derives heart rate from detected beat intervals only. For each detection it uses the median of up to five previous valid R–R intervals (0.25–3.0 seconds), then computes `60 / median_interval_seconds`. The first detection has no rate. Missed or extra detections can distort this trend; it is not calculated from reference annotations.

Correct end-to-end classification requires a matched location, an in-scope reference class, the correct class, and no abstention. Per-class recall divides by all in-scope reference beats, including misses. Matched category precision divides correct predictions by non-abstained matched predictions of that category. Misses, extras, out-of-scope references, and abstentions are separate counts.

The held-out run made 21,479 predictions, missed 250 locations, produced 15 extras, and abstained on 4,720 predictions (21.97%). End-to-end recall was N 74.86% (13,715/18,322), S 0% (0/515), and V 65.56% (693/1,057). These are observed research results, not clinical performance claims.

## Noise stress test

NSTDB repeats source records 118 and 119 at −6, 0, 6, 12, 18, and 24 dB. The first five minutes are clean; two-minute noisy and clean intervals alternate thereafter. The frozen pipeline is scored separately for each interval, source, and SNR. Reports include detection, N/S/V category recall and matched precision, quality flags, and abstentions. Twelve noisy recordings represent **two source people**, so no twelve-patient or population uncertainty claim is justified. Known noisy intervals are only a proxy for signal usability.

The locked run exposed two important failures. At −6 dB, the detector found no beats in the nominally clean intervals of either source record because its threshold is set from the entire recording, including severe noise. At 24 dB, noisy-interval detection sensitivity recovered to 99.90% on source 118 and 99.88% on source 119, yet the quality proxy flagged every predicted beat and the pipeline abstained on every prediction. These results show that the quality estimate does not transfer reliably to NSTDB, even when detection does. The full per-source/SNR table is in the generated `noise-v1.json` report. These issues are reported without retuning on the locked noise test.

Thresholds and model settings are not revised after examining held-out or NSTDB results. Further development requires a new protocol version.

## Beat classifier v2 experiment

V2 keeps the same subject-separated split and detector, while calculating timing over all annotated beats before excluding labels outside N/S/V. It adds three past-only context features: current RR interval relative to recent and longer within-record intervals, and R amplitude relative to recent beats. Four classifier candidates were compared on validation; the selected histogram gradient boosting model limits any one training record to the equivalent of 100 S examples before restoring overall S class weight. It does not use SMOTE or NSTDB data for training.

Validation macro F1 increased from 0.533 for the selected v1 model to 0.631 for the selected v2 model. On the nine-record test, reference-location macro F1 was 0.630 versus 0.579 in v1. S true positives rose from 5/515 to 168/515, but S precision was still only 11.5% (168 correct among 1,455 S predictions). V recall rose from 86.4% to 94.2% while V precision fell from 74.3% to 66.5%. These are trade-offs, not universal improvement.

End to end, v2 produced 21,479 detections and abstained on 5,370 (25.0%). Correct accepted results were N 13,095/18,322 (71.5%), S 34/515 (6.6%), and V 828/1,057 (78.3%). Matched accepted S precision was 53.1% (34/64), which excludes extra detections and out-of-scope references. The confidence threshold selected on validation rose from 0.7 to 0.8 and rejected many otherwise-correct S classifications.

The frozen v2 pipeline was also run on NSTDB without fitting to it. The detector and quality rules are unchanged, so at −6 dB it still found 0/2,409 reference beats in the nominally clean intervals. In the severely noisy intervals it issued 326 non-abstained predictions across the two sources, but only 41 were correct N/S/V classifications matched to an annotation. At 24 dB it detected 1,854/1,856 beats in noisy intervals yet abstained on every prediction because quality flagged every beat. V2 improves classification on original MIT-BIH records but does not solve noise reliability; its higher confidence can allow more predictions through at extreme noise.

The v2 test comparison is exploratory rather than fresh blind confirmation: the v1 test, the v1 NSTDB result, and an earlier v2 candidate's test outcomes were inspected before finalizing v2 context features. Fresh external records and a new untouched holdout are needed for a final generalization claim. The v1 reports remain unchanged.

## Reliability pipeline v3 experiment

V3 reuses the frozen v2 beat classifier, feature set, temperature calibration method, and 0.8 class-score threshold. It changes beat detection and signal-quality assessment. The detector retains the record-wide threshold, then applies a 30-second local threshold only inside gaps longer than one minute. Quality calculation corrects extreme constant ECG offsets, checks agreement between both ECG leads in 30-second blocks, and checks block-level QRS shape stability. The morphology limits are the 97th and 99th percentiles of training-record blocks; the quality floor is selected on validation. A predicted S beat still needs the conservative original quality check. These gates can abstain from a label; they do not retrain the classifier.

On the nine held-out MIT-BIH test records, beat detection remains **21,464 matched, 250 missed, 15 extra**. Compared with v2, v3 end-to-end accepted correct results changed as follows:

| Class | V2 correct / reference | V3 correct / reference | V3 matched accepted precision |
| --- | ---: | ---: | ---: |
| N | 13,095 / 18,322 (71.5%) | 13,070 / 18,322 (71.3%) | 98.6% |
| S | 34 / 515 (6.6%) | 34 / 515 (6.6%) | 54.0% (34/63) |
| V | 828 / 1,057 (78.3%) | 794 / 1,057 (75.1%) | 74.4% |

V3 abstained on 5,443/21,479 detections (25.3%), versus 5,370 (25.0%) for v2. The V recall loss is a material cost of the stronger quality gate. S recall remains very low; v3 has not solved S classification.

On NSTDB's −6 dB files, v3 found 2,406/2,409 beats in nominally clean intervals, versus zero for v2. In the noisy intervals at −6 and 0 dB, v3 made **zero accepted predictions** across both source records. V2 accepted 326 predictions at −6 dB, only 41 of which were correct matched N/S/V classifications, and 466 at 0 dB, only 75 correct. At +24 dB, v3 detected 1,854/1,856 noisy-interval beats and accepted 937 predictions, of which 796 were correct matched classifications. V2 accepted none there because its offset-sensitive quality proxy marked every beat unusable. The accepted predictions that are not counted correct include wrong classes, unmatched detections, and predictions outside the scored N/S/V reference set; the report separates these cases.

Source record 118 remains a conspicuous failure: at +24 dB, just 35 of 176 accepted noisy-interval predictions were correct matched classifications; its nominally clean interval had 22 correct among 110 accepted. The corresponding figures for source 119 were 761/761 noisy and 975/975 clean. Thus the quality gate does not detect every case of classifier domain shift. The NSTDB comparison has only two source records and was repeatedly inspected while developing v3, so these numbers are exploratory engineering results, not independent validation of noise robustness. The full per-source and per-SNR counts are in generated `noise-v3.json`; v1 and v2 reports are preserved.

## S-beat classifier v3 / full pipeline v4

The v3 classifier adds four offline features to v2: the next RR interval, its ratio to the previous and recent intervals, and normalized QRS shape mismatch with the previous eight beats. These features are computed from **all annotated beats** for reference-location training and from **all detected beats** during continuous inference. The following RR needs the next beat, so this remains an offline system. Four bounded models were fitted on the original MIT-BIH training records. Validation macro F1 selected the 100-example-per-record S weight cap, the same cap size used in v2. No SVDB, NSTDB, or test examples were used in fitting or selection.

On validation, macro F1 rose from 0.631 (v2) to 0.742 (v3 classifier). S F1 rose from 0.293 to 0.351; S precision rose from 20.2% to 24.9%, while S recall rose from 53.2% to 59.2%. The selected classifier was then evaluated on the already-viewed nine-record MIT-BIH test: macro F1 was 0.637 versus 0.630 for v2; S F1 was 0.173 versus 0.171, with 167/515 S beats correctly classified versus 168/515 for v2. This is a negligible S gain on that test and cannot establish robust improvement.

With the same v3 detector and quality method, the new classifier is called **full pipeline v4**. It selected a 0.9 probability-abstention threshold on validation, compared with 0.8 for full pipeline v3. On the MIT-BIH test it delivered 37/515 correct accepted S beats (7.2%) versus 34/515 (6.6%) for v3, with accepted matched S precision 37/47 (78.7%) versus 34/63 (54.0%). It also accepted fewer predictions overall (5,808/21,479 abstentions versus 5,443) and reduced correct accepted V beats from 794/1,057 (75.1%) to 762/1,057 (72.1%). The apparent S precision improvement therefore comes with lost coverage and a V-recall cost.

For a separate corpus check, the frozen v2 and v3 classifiers were scored at reference locations on all 78 [MIT-BIH Supraventricular Arrhythmia Database](https://physionet.org/content/svdb/1.0.0/) recordings, using ECG1 at 128 Hz. V3 external macro F1 was 0.581 versus 0.496 for v2. External S precision was 28.9% versus 25.3%, recall 40.7% versus 41.0%, and F1 0.338 versus 0.313, across 12,198 S beats. N F1 rose from 0.833 to 0.906 and V F1 from 0.341 to 0.498. Thus the external improvement is broader than S alone; S recall did not improve. SVDB was never used to train, calibrate, or select either model. This check isolates classification at annotated beat positions; it does not evaluate detector or quality behavior on SVDB. Separate corpus does not imply a representative clinical population or proven patient-level independence from MIT-BIH.

In the NSTDB comparison, full pipeline v4 still withheld every prediction in −6 and 0 dB noisy intervals. At +24 dB noisy intervals it accepted 912 predictions and made 797 correct matched N/S/V classifications, versus 937 accepted and 796 correct for pipeline v3. The noise source records remain previously inspected engineering cases, not fresh external confirmation. The full counts are in ignored generated reports `oracle-feature-v3.json`, `end-to-end-v4.json`, `noise-v4.json`, and `external-svdb-v3.json`.
