import { render, screen } from '@testing-library/react';
import { BeatInspector } from './BeatInspector';

it('shows prediction and reference side by side', () => {
  render(<BeatInspector beat={{ sample: 3672, seconds: 10.2, kind: 'wrong_class', group: 'V', reference_group: 'N', reference_symbol: 'N', score: 0.82, signal_quality: 0.76, abstained: false, timing_error_ms: 3 }} />);
  expect(screen.getByText('V')).toBeInTheDocument();
  expect(screen.getByText('N')).toBeInTheDocument();
  expect(screen.getByText('0.82')).toBeInTheDocument();
});
