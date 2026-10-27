import { render, screen } from '@testing-library/react';
import { BenchmarkPage } from './BenchmarkPage';
import type { Summary } from '../api/client';

it('shows the S-beat limitation with support', () => {
  const classes = { N: { support: 10, correct: 9, recall: 0.9, matched_precision: 0.9 }, S: { support: 2, correct: 0, recall: 0, matched_precision: 0 }, V: { support: 3, correct: 2, recall: 2 / 3, matched_precision: 0.8 } };
  const summary = { run_id: 'demo', dataset: 'mitdb', dataset_version: '1.0.0', split_sha256: 'abc', model_id: 'forest-v1', detector_id: 'pan-v1', records: ['100'], pooled: { classes, missed_beats: 1, extra_detections: 0, abstentions: 2, predictions: 15 }, detection: null, oracle_feature: null, oracle_waveform: null, reliability: null, noise: null } as Summary;
  render(<BenchmarkPage summary={summary} />);
  expect(screen.getByText('S beats are rare across subjects and remain a major failure mode. The waveform model is a comparison, not a replacement chosen on test data.')).toBeInTheDocument();
  expect(screen.getByText('0 / 2 reference beats')).toBeInTheDocument();
});

it('shows accepted predictions beside correct matched predictions under noise', () => {
  const classes = { N: { support: 10, correct: 2, recall: 0.2, matched_precision: 0.5 }, S: { support: 1, correct: 0, recall: 0, matched_precision: null }, V: { support: 1, correct: 1, recall: 1, matched_precision: 1 } };
  const row = { source: '118', snr_db: 24, interval: 'noisy', predictions: 20, detection: { sensitivity: 1, positive_predictivity: 1 }, classes, abstention_rate: 0.5, quality_flag_rate: 0, beat_support: 12 };
  const summary = { run_id: 'demo', dataset: 'mitdb', dataset_version: '1.0.0', split_sha256: 'abc', model_id: 'forest-v1', detector_id: 'pan-v1', records: ['100'], pooled: { classes, missed_beats: 0, extra_detections: 0, abstentions: 0, predictions: 12 }, detection: null, oracle_feature: null, oracle_waveform: null, reliability: null, noise: { summary: [row] } } as Summary;
  render(<BenchmarkPage summary={summary} />);
  expect(screen.getByText('Accepted')).toBeInTheDocument();
  expect(screen.getByText('Correct matched')).toBeInTheDocument();
  expect(screen.getByText('10')).toBeInTheDocument();
  expect(screen.getByText('3')).toBeInTheDocument();
});
