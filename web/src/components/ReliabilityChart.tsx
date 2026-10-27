import { useState } from 'react';
import type { Summary } from '../api/client';

type Evidence = NonNullable<Summary['reliability']>;

export function ReliabilityChart({ evidence }: { evidence: Evidence }) {
  const [mode, setMode] = useState<'calibration' | 'coverage'>('calibration');
  const series = [
    { name: 'Validation · known locations', data: evidence.validation_reference_locations },
    { name: 'Test · matched detections', data: evidence.test_matched_detections },
  ];
  const x = (value: number) => 50 + value * 510;
  const y = (value: number) => 190 - value * 160;
  const points = (data: Evidence['validation_reference_locations']) => mode === 'calibration'
    ? data.calibration_bins.filter(row => row.support > 0 && row.mean_score !== null && row.accuracy !== null).map(row => [row.mean_score!, row.accuracy!] as const)
    : data.accuracy_coverage.filter(row => row.support > 0 && row.coverage !== null && row.accuracy !== null).map(row => [row.coverage!, row.accuracy!] as const).reverse();
  return <div className="reliability-chart"><div className="chart-controls" role="group" aria-label="Reliability chart measure"><button className={mode === 'calibration' ? 'active' : ''} onClick={() => setMode('calibration')}>Calibration</button><button className={mode === 'coverage' ? 'active' : ''} onClick={() => setMode('coverage')}>Accuracy vs coverage</button></div><svg viewBox="0 0 620 230" role="img" aria-label={mode === 'calibration' ? 'Class score versus observed accuracy' : 'Coverage versus observed accuracy'}><g className="chart-grid">{[0, .5, 1].map(tick => <g key={tick}><line x1="50" x2="570" y1={y(tick)} y2={y(tick)} /><text x="7" y={y(tick) + 4}>{Math.round(tick * 100)}%</text><text className="chart-label" x={x(tick)} y="218" textAnchor="middle">{Math.round(tick * 100)}%</text></g>)}</g>{mode === 'calibration' && <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#a8bac0" strokeDasharray="5 5" />}{series.map(({ name, data }, index) => <g key={name} className={index === 0 ? 'series-a' : 'series-b'}><polyline points={points(data).map(([left, top]) => `${x(left)},${y(top)}`).join(' ')} />{points(data).map(([left, top], pointIndex) => <circle key={`${left}-${pointIndex}`} cx={x(left)} cy={y(top)} r="3" />)}</g>)}</svg><div className="chart-footer"><span>{mode === 'calibration' ? 'Mean score → observed accuracy' : 'Accepted coverage → accuracy'}</span><span><i className="line-a" /> Validation n={series[0].data.support.toLocaleString()} &nbsp; <i className="line-b" /> Test n={series[1].data.support.toLocaleString()}</span></div></div>;
}
