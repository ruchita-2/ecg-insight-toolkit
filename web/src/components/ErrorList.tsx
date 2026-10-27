import { useMemo, useState } from 'react';
import type { BeatRow } from '../api/client';

const filters = ['all', 'missed', 'extra', 'wrong_class', 'abstained', 'correct'] as const;

export function ErrorList({ rows, onSelect }: { rows: BeatRow[]; onSelect: (row: BeatRow) => void }) {
  const [filter, setFilter] = useState<string>('all');
  const [page, setPage] = useState(0);
  const visible = useMemo(() => rows.filter(row => filter === 'all' ? row.kind !== 'correct' : row.kind === filter), [rows, filter]);
  const pageSize = 100;
  const pageCount = Math.max(1, Math.ceil(visible.length / pageSize));
  const currentPage = Math.min(page, pageCount - 1);
  return <section className="error-panel" aria-label="Beat review">
    <div className="section-heading"><div><span className="eyebrow">Review queue</span><h2>Find the failures</h2></div><span className="support-count">{visible.length.toLocaleString()} rows</span></div>
    <div className="filter-row" role="group" aria-label="Filter beat results">{filters.map(item => <button key={item} type="button" className={filter === item ? 'filter active' : 'filter'} onClick={() => { setFilter(item); setPage(0); }}>{item.replace('_', ' ')}</button>)}</div>
    <div className="error-list">{visible.slice(currentPage * pageSize, (currentPage + 1) * pageSize).map((row, index) => <button type="button" className="error-row" key={`${row.sample}-${row.kind}-${index}`} onClick={() => onSelect(row)}>
      <span className="mono">{row.seconds.toFixed(2)} s</span><span className={`status status-${row.kind}`}>{row.kind.replace('_', ' ')}</span><span>{row.reference_group ?? '—'} → {row.abstained ? 'abstain' : (row.group ?? '—')}</span><span className="row-arrow">↗</span>
    </button>)}{visible.length === 0 && <p className="empty-message">No beats match this filter.</p>}</div>
    {pageCount > 1 && <div className="pagination"><button type="button" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}>← Previous</button><span>Page {currentPage + 1} of {pageCount}</span><button type="button" disabled={currentPage >= pageCount - 1} onClick={() => setPage(currentPage + 1)}>Next →</button></div>}
  </section>;
}
