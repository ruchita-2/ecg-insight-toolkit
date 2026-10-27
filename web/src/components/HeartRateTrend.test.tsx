import { render, screen } from '@testing-library/react';
import { HeartRateTrend } from './HeartRateTrend';
import type { RecordResult, WindowResult } from '../api/client';

it('shows a detected heart rate for the selected waveform window', () => {
  const window = { start_sample: 0, end_sample: 3600, sample_rate_hz: 360 } as WindowResult;
  const series = [{ sample: 360, seconds: 1, bpm: null }, { sample: 720, seconds: 2, bpm: 60 }, { sample: 1080, seconds: 3, bpm: 60 }] as RecordResult['heart_rate'];
  render(<HeartRateTrend window={window} series={series} />);
  expect(screen.getByText('60')).toBeInTheDocument();
  expect(screen.getByRole('img', { name: /heart-rate trend/i })).toBeInTheDocument();
});
