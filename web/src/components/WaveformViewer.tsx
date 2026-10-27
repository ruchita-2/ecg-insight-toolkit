import { useEffect, useRef } from 'react';
import type { BeatRow, WindowResult } from '../api/client';

const markerColor: Record<string, string> = {
  correct: '#187b76', wrong_class: '#c36041', missed: '#c24e54',
  extra: '#b37531', abstained: '#7d69a4', out_of_scope: '#87949a',
};

export function WaveformViewer({ window, rows, channel, showRaw, showFiltered, onSelect }: {
  window: WindowResult | null; rows: BeatRow[]; channel: number;
  showRaw: boolean; showFiltered: boolean; onSelect: (row: BeatRow) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element || !window) return;
    let active = true;
    const start = window.start_sample / window.sample_rate_hz;
    const end = window.end_sample / window.sample_rate_hz;
    const visible = rows.filter(row => row.seconds >= start && row.seconds <= end);
    const traces: unknown[] = [];
    if (showRaw) traces.push({ type: 'scattergl', mode: 'lines', name: `Raw · ${window.lead_names[channel]}`, x: window.seconds, y: window.raw[channel], line: { color: '#244957', width: 1.4 } });
    if (showFiltered && channel === 0) traces.push({ type: 'scattergl', mode: 'lines', name: 'Filtered · primary', x: window.seconds, y: window.filtered_primary, line: { color: '#26a19a', width: 1.4 } });
    const lower = Math.min(...window.raw[channel]);
    const upper = Math.max(...window.raw[channel]);
    traces.push({ type: 'scattergl', mode: 'markers', name: 'Beat markers', x: visible.map(row => row.seconds), y: visible.map(row => lower + (upper - lower) * 0.9), customdata: visible.map((_, index) => index), marker: { size: 10, color: visible.map(row => markerColor[row.kind]), symbol: visible.map(row => row.kind === 'missed' ? 'x' : 'circle'), line: { color: '#fff', width: 1 } } });
    const layout = {
      margin: { l: 40, r: 12, t: 10, b: 35 }, paper_bgcolor: '#ffffff', plot_bgcolor: '#ffffff',
      font: { family: 'IBM Plex Mono, monospace', size: 11, color: '#547078' },
      xaxis: { title: 'Time (seconds)', showgrid: true, gridcolor: '#e7eef0', zeroline: false },
      yaxis: { title: 'mV', showgrid: true, gridcolor: '#e7eef0', zeroline: false },
      showlegend: true, legend: { orientation: 'h', y: 1.16 },
      dragmode: 'pan', hovermode: 'closest',
    };
    const plot = element as HTMLElement & { on?: (event: string, callback: (event: { points: Array<{ curveNumber: number; pointIndex: number }> }) => void) => void; removeAllListeners?: (event: string) => void };
    void import('plotly.js-dist-min').then(({ default: Plotly }) => {
      if (!active) return;
      void Plotly.react(element, traces, layout, { displaylogo: false, responsive: true, scrollZoom: true }).then(() => {
        if (!active) return;
        plot.removeAllListeners?.('plotly_click');
        plot.on?.('plotly_click', event => {
          const point = event.points[0];
          if (point.curveNumber === traces.length - 1 && visible[point.pointIndex]) onSelect(visible[point.pointIndex]);
        });
      });
    });
    return () => { active = false; plot.removeAllListeners?.('plotly_click'); void import('plotly.js-dist-min').then(({ default: Plotly }) => Plotly.purge(element)); };
  }, [window, rows, channel, showRaw, showFiltered, onSelect]);
  return <div className="waveform" ref={ref} role="img" aria-label="ECG waveform with beat markers">{!window && <div className="waveform-empty">Select a recording to view its ECG signal.</div>}</div>;
}
