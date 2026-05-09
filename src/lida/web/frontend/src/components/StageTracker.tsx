import React from 'react';
import type { StageState } from '../hooks/useUploadPipeline';

interface StageTrackerProps {
  stages: StageState[];
}

const STATUS_GLYPH: Record<StageState['status'], string> = {
  pending: '○',
  active: '◔',
  done: '●',
  error: '✕',
};

export const StageTracker: React.FC<StageTrackerProps> = ({ stages }) => {
  return (
    <ol className="stage-tracker" role="list">
      {stages.map((stage, i) => (
        <li key={stage.id} className={`stage-tracker__item stage-tracker__item--${stage.status}`}>
          <div className="stage-tracker__node">
            <span className="stage-tracker__glyph" aria-hidden>
              {stage.status === 'active' ? <span className="stage-tracker__spinner" /> : STATUS_GLYPH[stage.status]}
            </span>
            <span className="stage-tracker__index">{i + 1}</span>
          </div>
          <div className="stage-tracker__body">
            <div className="stage-tracker__label">{stage.label}</div>
            {stage.detail && <div className="stage-tracker__detail">{stage.detail}</div>}
          </div>
          {i < stages.length - 1 && <span className="stage-tracker__connector" aria-hidden />}
        </li>
      ))}
    </ol>
  );
};
