import React from 'react';
import type { PipelineState } from '../hooks/useUploadPipeline';
import { StageTracker } from './StageTracker';
import { ProgressBar } from './ProgressBar';
import { TokenCounter } from './TokenCounter';
import { TableProgressList } from './TableProgressList';

interface PipelineProgressProps {
  state: PipelineState;
}

const stageActive = (state: PipelineState, id: string) =>
  state.stages.find((s) => s.id === id)?.status === 'active';

export const PipelineProgress: React.FC<PipelineProgressProps> = ({ state }) => {
  const compressActive = stageActive(state, 'compress');
  const uploadActive = stageActive(state, 'upload');
  const analyzeActive = stageActive(state, 'analyze');

  const compressDone = state.stages.find((s) => s.id === 'compress')?.status === 'done';
  const uploadDone = state.stages.find((s) => s.id === 'upload')?.status === 'done';

  return (
    <div className="pipeline-progress">
      <StageTracker stages={state.stages} />

      <div className="pipeline-progress__bars">
        {(compressActive || compressDone) && (
          <ProgressBar
            label="Client-side compression"
            processed={state.compress.processed}
            total={state.compress.total}
            unit="bytes"
            variant="primary"
          />
        )}
        {(uploadActive || uploadDone) && (
          <ProgressBar
            label="Upload to server"
            processed={state.upload.processed}
            total={state.upload.total || state.compress.total}
            unit="bytes"
            variant="accent"
          />
        )}
      </div>

      {(analyzeActive || state.llm.tokens > 0) && (
        <TokenCounter llm={state.llm} active={analyzeActive} />
      )}

      <TableProgressList tables={state.tables} dispatchKind={state.dispatchKind} />

      {state.error && <div className="pipeline-progress__error">{state.error}</div>}
    </div>
  );
};
