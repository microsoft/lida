import React from 'react';
import type {
  ChartEntry,
  GoalWithSource,
  PlotStatus,
} from '../hooks/useUploadPipeline';

interface GoalsBoardProps {
  goals: GoalWithSource[];
  charts: Record<number, ChartEntry>;
  plotStatuses: Record<number, PlotStatus>;
  plotErrors: Record<number, string>;
  totalExpected: number;
  onGoalSelect: (item: GoalWithSource) => void;
  selectedIndex: number | null;
  showDataSource: boolean;
}

const STATUS_COPY: Record<PlotStatus, string> = {
  pending: 'queued',
  generating: 'generating code',
  code_ready: 'rendering chart',
  rendered: 'done',
  failed: 'failed',
};

const ChartSlot: React.FC<{ chart?: ChartEntry; status: PlotStatus; error?: string }> = ({
  chart,
  status,
  error,
}) => {
  if (chart?.raster) {
    return (
      <div className="chart-slot chart-slot--ready">
        <img src={`data:image/png;base64,${chart.raster}`} alt="generated chart" />
      </div>
    );
  }
  if (status === 'failed' || (chart && !chart.raster)) {
    const detail = error || (chart?.error?.message as string | undefined) || 'Chart failed';
    return (
      <div className="chart-slot chart-slot--failed">
        <span className="chart-slot__icon" aria-hidden>✕</span>
        <span>{detail}</span>
      </div>
    );
  }
  return (
    <div className="chart-slot chart-slot--busy">
      <span className="chart-slot__spinner" aria-hidden />
      <span>{STATUS_COPY[status]}</span>
    </div>
  );
};

export const GoalsBoard: React.FC<GoalsBoardProps> = ({
  goals,
  charts,
  plotStatuses,
  plotErrors,
  totalExpected,
  onGoalSelect,
  selectedIndex,
  showDataSource,
}) => {
  const placeholders = Math.max(0, totalExpected - goals.length);

  return (
    <div className="goals-board">
      {goals.map((item) => {
        const goal = item.goal;
        const status = plotStatuses[goal.index] ?? 'pending';
        const chart = charts[goal.index];
        const isSelected = selectedIndex === goal.index;
        return (
          <article
            key={goal.index}
            className={`goal-row ${isSelected ? 'goal-row--selected' : ''}`}
            onClick={() => onGoalSelect(item)}
          >
            <header className="goal-row__header">
              <span className="goal-row__index">{goal.index + 1}</span>
              <h4 className="goal-row__question">{goal.question}</h4>
              <span className={`goal-row__status goal-row__status--${status}`}>
                {STATUS_COPY[status]}
              </span>
            </header>
            {showDataSource && item.dataSource && (
              <div className="goal-row__source-pill">{item.dataSource}</div>
            )}
            <p className="goal-row__rationale">{goal.rationale}</p>
            <ChartSlot chart={chart} status={status} error={plotErrors[goal.index]} />
          </article>
        );
      })}
      {Array.from({ length: placeholders }).map((_, i) => (
        <article key={`placeholder-${i}`} className="goal-row goal-row--placeholder">
          <header className="goal-row__header">
            <span className="goal-row__index">{goals.length + i + 1}</span>
            <span className="goal-row__placeholder-label">awaiting goal…</span>
          </header>
        </article>
      ))}
    </div>
  );
};
