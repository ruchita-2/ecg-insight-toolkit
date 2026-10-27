import type { RecordResult, WindowResult } from '../api/client';

export function HeartRateTrend({ series, window }: { series: RecordResult['heart_rate']; window: WindowResult | null }) {
  if (!window) return null;
  const start = window.start_sample / window.sample_rate_hz;
  const end = window.end_sample / window.sample_rate_hz;
  const visible = series.filter(row => row.bpm !== null && row.seconds >= start && row.seconds <= end);
  const values = visible.map(row => row.bpm!);
  const median = values.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : null;
  const points = visible.map(row => `${((row.seconds - start) / Math.max(end - start, .001)) * 500},${60 - Math.min(200, row.bpm!) / 200 * 60}`).join(' ');
  return <div className="heart-rate-trend"><div><span className="eyebrow">Detected rate</span><strong>{median === null ? '—' : Math.round(median)} <small>bpm median</small></strong></div><svg viewBox="0 0 500 65" preserveAspectRatio="none" role="img" aria-label="Heart-rate trend from detected beats in this window"><line x1="0" x2="500" y1="60" y2="60" stroke="#d9e6e8" />{visible.length > 1 && <polyline points={points} fill="none" stroke="#218d87" strokeWidth="2" />}{visible.map(row => <circle key={row.sample} cx={((row.seconds - start) / Math.max(end - start, .001)) * 500} cy={60 - Math.min(200, row.bpm!) / 200 * 60} r="3" fill="#218d87" />)}</svg><span className="trend-axis">0–200 bpm</span></div>;
}
