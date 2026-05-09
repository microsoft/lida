import React, { useCallback, useState } from 'react';
import type { PipelineState } from '../hooks/useUploadPipeline';
import { PipelineProgress } from './PipelineProgress';

interface FileUploaderProps {
  nGoals: number;
  onNGoalsChange: (n: number) => void;
  onStart: (file: File) => void;
  onUrlSubmit: (url: string) => Promise<void>;
  onReset: () => void;
  pipeline: PipelineState;
}

export const FileUploader: React.FC<FileUploaderProps> = ({
  nGoals,
  onNGoalsChange,
  onStart,
  onUrlSubmit,
  onReset,
  pipeline,
}) => {
  const [urlBusy, setUrlBusy] = useState(false);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!e.target.files?.length) return;
      const file = e.target.files[0];
      e.target.value = '';
      onStart(file);
    },
    [onStart],
  );

  const handleUrlSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const url = (e.currentTarget.elements.namedItem('url') as HTMLInputElement).value;
    if (!url) return;
    setUrlBusy(true);
    try {
      await onUrlSubmit(url);
    } finally {
      setUrlBusy(false);
    }
  };

  const showPipeline = pipeline.active || pipeline.finished;

  return (
    <div className="file-uploader">
      <h3>Data Upload</h3>
      {!showPipeline && (
        <>
          <div className="upload-config">
            <label className="goal-slider-label">
              <span className="goal-slider-label__text">Goals to generate</span>
              <span className="goal-slider-value">{nGoals}</span>
            </label>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={nGoals}
              onChange={(e) => onNGoalsChange(parseInt(e.target.value, 10))}
              className="goal-slider"
              aria-label="Number of goals"
            />
            <div className="goal-slider-ticks" aria-hidden>
              <span>1</span>
              <span>2</span>
              <span>3</span>
              <span>4</span>
              <span>5</span>
              <span>6</span>
              <span>7</span>
              <span>8</span>
              <span>9</span>
              <span>10</span>
            </div>
          </div>
          <div className="upload-methods">
            <div className="upload-method">
              <label className="button-like">
                Upload File (CSV, JSON, XLSX, SQLite, .tar, .gz, .tar.gz)
                <input
                  type="file"
                  onChange={handleFileChange}
                  accept=".csv,.json,.xlsx,.db,.sqlite,.sqlite3,.tar,.tgz,.gz"
                  hidden
                />
              </label>
            </div>
            <div className="divider">OR</div>
            <form onSubmit={handleUrlSubmit} className="upload-method url-form">
              <input type="url" name="url" placeholder="Enter CSV URL..." required />
              <button type="submit" disabled={urlBusy}>
                {urlBusy ? 'Processing...' : 'Load from URL'}
              </button>
            </form>
          </div>
        </>
      )}

      {showPipeline && (
        <>
          <PipelineProgress state={pipeline} />
          {pipeline.finished && (
            <div className="pipeline-progress__actions">
              <button type="button" className="secondary" onClick={onReset}>
                Upload another file
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
};
