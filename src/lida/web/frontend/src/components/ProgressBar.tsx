import React from 'react';

interface ProgressBarProps {
  processed: number;
  total: number;
  label: string;
  unit?: 'bytes' | 'count';
  variant?: 'primary' | 'accent' | 'success';
  indeterminate?: boolean;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  processed,
  total,
  label,
  unit = 'bytes',
  variant = 'primary',
  indeterminate = false,
}) => {
  const ratio = total > 0 ? Math.min(1, processed / total) : 0;
  const percent = (ratio * 100).toFixed(0);
  const formatVal = (n: number) => (unit === 'bytes' ? formatBytes(n) : n.toString());
  const detail = total > 0 ? `${formatVal(processed)} / ${formatVal(total)}` : '—';

  return (
    <div className={`pipeline-bar pipeline-bar--${variant}`}>
      <div className="pipeline-bar__row">
        <span className="pipeline-bar__label">{label}</span>
        <span className="pipeline-bar__detail">{indeterminate ? '…' : `${percent}% · ${detail}`}</span>
      </div>
      <div className="pipeline-bar__track">
        {indeterminate ? (
          <div className="pipeline-bar__indeterminate" />
        ) : (
          <div className="pipeline-bar__fill" style={{ width: `${ratio * 100}%` }} />
        )}
      </div>
    </div>
  );
};
