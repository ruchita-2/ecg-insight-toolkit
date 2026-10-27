import { useEffect, useState } from 'react';
import { api, type Summary } from './api/client';
import { BenchmarkPage } from './pages/BenchmarkPage';
import { RecordingPage } from './pages/RecordingPage';

export default function App() {
  const [view, setView] = useState<'recordings' | 'benchmarks'>('recordings');
  const [summary, setSummary] = useState<Summary | null>(null);
  const [error, setError] = useState('');
  useEffect(() => {
    api.runs().then(runs => {
      if (!runs.length) throw new Error('No benchmark runs found. Generate the end-to-end report first.');
      return api.summary(runs[0].run_id);
    }).then(setSummary).catch(reason => setError(String(reason)));
  }, []);
  return <div className="app-shell"><header className="topbar"><div className="brand"><span className="brand-mark"><i /><i /></span><div><strong>ECG Workbench</strong><small>Beat analysis / reliability</small></div></div><nav aria-label="Main navigation"><button type="button" className={view === 'recordings' ? 'active' : ''} onClick={() => setView('recordings')}>Recording explorer</button><button type="button" className={view === 'benchmarks' ? 'active' : ''} onClick={() => setView('benchmarks')}>Benchmark evidence</button></nav><div className="topbar-tag">MIT-BIH · research mode</div></header>{error ? <main className="content"><div className="error-banner" role="alert">{error}</div><p>Start the API and generate the benchmark reports using the commands in the README.</p></main> : !summary ? <main className="content"><p className="loading">Loading run evidence…</p></main> : view === 'recordings' ? <RecordingPage runId={summary.run_id} summary={summary} /> : <BenchmarkPage summary={summary} />}</div>;
}
