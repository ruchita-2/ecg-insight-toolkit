export type BeatRow = {
  sample: number;
  seconds: number;
  kind: 'correct' | 'wrong_class' | 'missed' | 'extra' | 'abstained' | 'out_of_scope';
  group: string | null;
  reference_group: string | null;
  reference_symbol: string | null;
  score: number | null;
  signal_quality: number | null;
  abstained: boolean;
  abstention_reason?: 'low_classifier_score' | 'low_signal_quality' | 's_confirmation' | null;
  timing_error_ms: number | null;
};

export type RecordResult = {
  run_id: string;
  record_id: string;
  dataset_version: string;
  split_sha256: string;
  model_id: string;
  metrics: {
    classes: Record<string, { support: number; correct: number; recall: number | null; matched_precision: number | null }>;
    missed_beats: number;
    extra_detections: number;
    abstentions: number;
    predictions: number;
  };
  rows: BeatRow[];
  heart_rate: Array<{ sample: number; seconds: number; bpm: number | null }>;
  lead_names: string[];
  sample_rate_hz: number;
};

export type WindowResult = {
  record_id: string;
  sample_rate_hz: number;
  lead_names: string[];
  start_sample: number;
  end_sample: number;
  seconds: number[];
  raw: number[][];
  filtered_primary: number[];
};

export type Summary = {
  run_id: string;
  dataset: string;
  dataset_version: string;
  split_sha256: string;
  model_id: string;
  detector_id: string;
  records: string[];
  pooled: RecordResult['metrics'];
  reliability: { validation_reference_locations: { support: number; calibration_bins: Array<{ mean_score: number | null; accuracy: number | null; support: number }> ; accuracy_coverage: Array<{ threshold: number; coverage: number | null; accuracy: number | null; support: number }> }; test_matched_detections: { support: number; calibration_bins: Array<{ mean_score: number | null; accuracy: number | null; support: number }> ; accuracy_coverage: Array<{ threshold: number; coverage: number | null; accuracy: number | null; support: number }> } } | null;
  detection: { detectors: Record<string, { pooled: { sensitivity: number; positive_predictivity: number; f1: number; tp: number; fp: number; fn: number }; records: Array<{ record_id: string; tp: number; fp: number; fn: number; median_absolute_error_ms: number | null; p95_absolute_error_ms: number | null }> }> } | null;
  oracle_feature: { model_id: string; test: { classes: Record<string, { support: number; recall: number; 'f1-score': number }>; macro_f1: number; confusion_matrix: number[][] } } | null;
  oracle_waveform: { model_id: string; test: { classes: Record<string, { support: number; recall: number; 'f1-score': number }>; macro_f1: number } } | null;
  noise: { summary: Array<{ source: string; snr_db: number; interval: string; predictions: number; detection: { sensitivity: number; positive_predictivity: number }; classes: Record<string, { support: number; correct: number; recall: number | null; matched_precision: number | null }>; abstention_rate: number | null; quality_flag_rate: number | null; beat_support: number }> } | null;
};

async function read<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load results (${response.status}). Check that the API is running and the benchmark files exist.`);
  return response.json() as Promise<T>;
}

export const api = {
  runs: () => read<Array<{ run_id: string; model_id: string }>>('/api/runs'),
  summary: (runId: string) => read<Summary>(`/api/runs/${encodeURIComponent(runId)}/summary`),
  record: (runId: string, recordId: string) => read<RecordResult>(`/api/runs/${encodeURIComponent(runId)}/records/${encodeURIComponent(recordId)}`),
  window: (runId: string, recordId: string, start: number, duration = 10) => read<WindowResult>(`/api/runs/${encodeURIComponent(runId)}/records/${encodeURIComponent(recordId)}/window?start=${start}&duration=${duration}`),
};
