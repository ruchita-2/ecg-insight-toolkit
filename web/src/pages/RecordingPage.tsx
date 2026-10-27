import { useCallback, useEffect, useRef, useState } from 'react';
import { api, type BeatRow, type RecordResult, type Summary, type WindowResult } from '../api/client';
import { BeatInspector } from '../components/BeatInspector';
import { ErrorList } from '../components/ErrorList';
import { HeartRateTrend } from '../components/HeartRateTrend';
import { WaveformViewer } from '../components/WaveformViewer';

export function RecordingPage({ runId, summary }: { runId: string; summary: Summary }) {
  const [recordId, setRecordId] = useState(summary.records[0] ?? '');
  const [record, setRecord] = useState<RecordResult | null>(null);
  const [windowData, setWindowData] = useState<WindowResult | null>(null);
  const [start, setStart] = useState(0);
  const [channel, setChannel] = useState(0);
  const [showRaw, setShowRaw] = useState(true);
  const [showFiltered, setShowFiltered] = useState(true);
  const [selected, setSelected] = useState<BeatRow | null>(null);
  const [error, setError] = useState('');
  const viewerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!recordId) return;
    let live = true;
    setRecord(null); setSelected(null); setStart(0);
    api.record(runId, recordId).then(result => { if (live) setRecord(result); }).catch(reason => { if (live) setError(String(reason)); });
    return () => { live = false; };
  }, [runId, recordId]);
  useEffect(() => {
    if (!recordId) return;
    let live = true;
    setWindowData(null);
    api.window(runId, recordId, start).then(result => { if (live) { setWindowData(result); setError(''); } }).catch(reason => { if (live) setError(String(reason)); });
    return () => { live = false; };
  }, [runId, recordId, start]);
  const selectBeat = useCallback((beat: BeatRow) => {
    setSelected(beat);
    const target = Math.max(0, Math.floor(beat.seconds - 4));
    if (target !== start) setStart(target);
    const reducedMotion = typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    viewerRef.current?.scrollIntoView?.({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' });
  }, [start]);
  return <main className="content">
    <div className="page-intro"><div><span className="eyebrow">Continuous ECG · held-out test records</span><h1>See every beat.<br/><em>Question every prediction.</em></h1></div><p>Explore two leads, compare detector output with MIT-BIH annotations, and inspect errors in their waveform context.</p></div>
    <section className="record-toolbar" aria-label="Recording controls">
      <label>Recording<select value={recordId} onChange={event => setRecordId(event.target.value)}>{summary.records.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
      <label>Lead<select value={channel} onChange={event => setChannel(Number(event.target.value))}>{(record?.lead_names ?? ['MLII', 'V1']).map((name, index) => <option key={name} value={index}>{name}</option>)}</select></label>
      <label>Start (seconds)<input type="number" min="0" step="1" value={start} onChange={event => setStart(Math.max(0, Number(event.target.value) || 0))} /></label>
      <div className="step-control"><button type="button" onClick={() => setStart(Math.max(0, start - 10))}>← 10 s</button><button type="button" onClick={() => setStart(start + 10)}>10 s →</button></div>
    </section>
    {error && <div className="error-banner" role="alert">{error}</div>}
    <div className="viewer-grid" ref={viewerRef}><section className="signal-panel"><div className="section-heading"><div><span className="eyebrow">Signal strip</span><h2>{recordId} <span className="light">/ {start}–{start + 10} s</span></h2></div><span className="sampling">360 samples / second</span></div><div className="trace-toggles"><label><input type="checkbox" checked={showRaw} onChange={event => setShowRaw(event.target.checked)} /> Raw signal</label><label><input type="checkbox" checked={showFiltered} onChange={event => setShowFiltered(event.target.checked)} disabled={channel !== 0} /> Filtered primary</label><span className="trace-key"><i className="key-dot reference" /> reference matched &nbsp; <i className="key-dot miss" /> missed</span></div><WaveformViewer window={windowData} rows={record?.rows ?? []} channel={channel} showRaw={showRaw} showFiltered={showFiltered} onSelect={selectBeat} /><HeartRateTrend series={record?.heart_rate ?? []} window={windowData} /></section><BeatInspector beat={selected} /></div>
    {record && <div className="record-facts"><div><span className="field-label">Beats evaluated</span><strong>{Object.values(record.metrics.classes).reduce((total, row) => total + row.support, 0).toLocaleString()}</strong></div><div><span className="field-label">Missed detections</span><strong>{record.metrics.missed_beats}</strong></div><div><span className="field-label">Extra detections</span><strong>{record.metrics.extra_detections}</strong></div><div><span className="field-label">Abstentions</span><strong>{record.metrics.abstentions}</strong></div></div>}
    <ErrorList rows={record?.rows ?? []} onSelect={selectBeat} />
    <p className="provenance">MIT-BIH Arrhythmia Database v{summary.dataset_version} · Split {summary.split_sha256.slice(0, 12)} · {summary.model_id} · Research use only</p>
  </main>;
}
