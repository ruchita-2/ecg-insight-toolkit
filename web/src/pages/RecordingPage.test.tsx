import { render, screen, waitFor } from '@testing-library/react';
import { vi } from 'vitest';
import { RecordingPage } from './RecordingPage';
import type { Summary } from '../api/client';

vi.mock('../components/WaveformViewer', () => ({ WaveformViewer: () => <div>Waveform fixture</div> }));
vi.mock('../api/client', () => ({ api: {
  record: async () => ({ record_id: '100', lead_names: ['MLII', 'V1'], rows: [], heart_rate: [], metrics: { classes: { N: { support: 2 } }, missed_beats: 0, extra_detections: 0, abstentions: 0 } }),
  window: async () => ({ record_id: '100', sample_rate_hz: 360, lead_names: ['MLII', 'V1'], start_sample: 0, end_sample: 3600, seconds: [], raw: [[], []], filtered_primary: [] }),
} }));

it('loads a held-out recording and its bounded signal strip', async () => {
  const summary = { records: ['100'], dataset_version: '1.0.0', split_sha256: 'abc', model_id: 'forest-v1' } as Summary;
  render(<RecordingPage runId="demo" summary={summary} />);
  await waitFor(() => expect(screen.getByText('Waveform fixture')).toBeInTheDocument());
  expect(screen.getByText('Missed detections')).toBeInTheDocument();
  expect(screen.getByLabelText('Recording')).toHaveValue('100');
});
