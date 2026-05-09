import React from 'react';
import type { LlmActivity } from '../hooks/useUploadPipeline';

interface TokenCounterProps {
  llm: LlmActivity;
  active: boolean;
}

export const TokenCounter: React.FC<TokenCounterProps> = ({ llm, active }) => {
  const tokensPerSec = llm.elapsedMs > 0 ? (llm.tokens / (llm.elapsedMs / 1000)) : 0;
  const showTail = active && llm.tail.length > 0;

  return (
    <div className={`token-counter ${active ? 'token-counter--active' : ''}`}>
      <div className="token-counter__header">
        <span className="token-counter__pulse" aria-hidden />
        <span className="token-counter__title">LLM activity</span>
        {active && <span className="token-counter__hint">streaming…</span>}
      </div>
      <div className="token-counter__metrics">
        <div className="token-counter__metric">
          <div className="token-counter__metric-value">{llm.tokens.toLocaleString()}</div>
          <div className="token-counter__metric-label">tokens</div>
        </div>
        <div className="token-counter__metric">
          <div className="token-counter__metric-value">{tokensPerSec.toFixed(1)}</div>
          <div className="token-counter__metric-label">tok/s</div>
        </div>
        <div className="token-counter__metric">
          <div className="token-counter__metric-value">{(llm.elapsedMs / 1000).toFixed(1)}s</div>
          <div className="token-counter__metric-label">elapsed</div>
        </div>
      </div>
      {showTail && (
        <div className="token-counter__tail" aria-live="polite">
          <span className="token-counter__tail-text">{llm.tail}</span>
          <span className="token-counter__caret" aria-hidden />
        </div>
      )}
    </div>
  );
};
