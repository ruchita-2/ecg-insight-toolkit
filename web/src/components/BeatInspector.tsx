import type { BeatRow } from '../api/client';

const groupName: Record<string, string> = { N: 'Normal family', S: 'Supraventricular', V: 'Ventricular' };

export function BeatInspector({ beat }: { beat: BeatRow | null }) {
  if (!beat) return <aside className="inspector empty" aria-label="Beat details"><span className="eyebrow">Beat details</span><p>Select a marker or an error row to inspect one beat.</p></aside>;
  return <aside className="inspector" aria-label="Beat details">
    <div className="inspector-header"><span className="eyebrow">Beat details</span><span className={`status status-${beat.kind}`}>{beat.kind.replace('_', ' ')}</span></div>
    <div className="beat-time">{beat.seconds.toFixed(2)}<small> seconds</small></div>
    <div className="inspection-grid">
      <div><span className="field-label">Predicted</span><strong>{beat.abstained ? 'Abstained' : (beat.group ?? 'No detection')}</strong><small>{beat.group ? groupName[beat.group] : 'No category'}</small></div>
      <div><span className="field-label">Reference</span><strong>{beat.reference_group ?? 'Out of scope / none'}</strong><small>{beat.reference_symbol ? `MIT-BIH code ${beat.reference_symbol}` : 'No matched annotation'}</small></div>
    </div>
    <div className="measurement"><span>Classifier score</span><strong>{beat.score === null ? '—' : beat.score.toFixed(2)}</strong></div>
    <div className="measurement"><span>Signal quality estimate</span><strong>{beat.signal_quality === null ? '—' : beat.signal_quality.toFixed(2)}</strong></div>
    {beat.abstained && beat.abstention_reason && <div className="measurement"><span>Why abstained</span><strong>{beat.abstention_reason === 'low_classifier_score' ? 'Low model confidence' : beat.abstention_reason === 'low_signal_quality' ? 'Signal needs review' : 'S prediction needs review'}</strong></div>}
    <div className="measurement"><span>Timing offset</span><strong>{beat.timing_error_ms === null ? '—' : `${beat.timing_error_ms.toFixed(1)} ms`}</strong></div>
    <p className="microcopy">Scores and signal quality are separate estimates. They are not clinical confidence.</p>
  </aside>;
}
