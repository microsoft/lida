import { useCallback, useMemo, useState } from 'react';
import './App.css';
import { ModelSelector } from './components/ModelSelector';
import { FileUploader } from './components/FileUploader';
import { GoalsBoard } from './components/GoalsBoard';
import { Visualizer } from './components/Visualizer';
import { useUploadPipeline } from './hooks/useUploadPipeline';
import type { GoalEntry, TableProgress } from './hooks/useUploadPipeline';

interface SelectedGoal {
  goal: GoalEntry;
  dataSource: string;
}

function App() {
  const [activeModel, setActiveModel] = useState<{ provider: string; model: string }>({ provider: '', model: '' });
  const [nGoals, setNGoals] = useState(5);
  const [activeTableName, setActiveTableName] = useState<string | null>(null);
  const [selectedGoal, setSelectedGoal] = useState<SelectedGoal | null>(null);
  const [errorBanner, setErrorBanner] = useState<string>('');
  const [urlResult, setUrlResult] = useState<{
    summary: any;
    goals: any[];
    charts: any[];
    data_filename?: string;
  } | null>(null);

  const pipeline = useUploadPipeline({
    onComplete: () => {
      setErrorBanner('');
    },
    onError: (msg) => setErrorBanner(msg),
  });

  const handleModelChange = useCallback((provider: string, model: string) => {
    setActiveModel({ provider, model });
  }, []);

  const handleStart = useCallback(
    (file: File) => {
      setErrorBanner('');
      setSelectedGoal(null);
      setActiveTableName(null);
      setUrlResult(null);
      pipeline.start(file, { nGoals });
    },
    [pipeline, nGoals],
  );

  const handleReset = useCallback(() => {
    setSelectedGoal(null);
    setActiveTableName(null);
    setErrorBanner('');
    setUrlResult(null);
    pipeline.reset();
  }, [pipeline]);

  const handleUrlSubmit = useCallback(
    async (url: string) => {
      setErrorBanner('');
      setSelectedGoal(null);
      setActiveTableName(null);
      try {
        const res = await fetch('/api/v1/summarize/url', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url }),
        });
        const data = await res.json();
        if (!data.status) {
          setErrorBanner(data.message || 'URL processing failed');
          return;
        }
        setUrlResult({
          summary: data.summary,
          goals: data.goals || [],
          charts: data.charts || [],
          data_filename: data.data_filename,
        });
      } catch (err: any) {
        setErrorBanner(err?.message || 'Network error');
      }
    },
    [],
  );

  const tables = pipeline.state.tables;
  const isMultiTable = tables.length > 1 || pipeline.state.dispatchKind === 'sqlite' || pipeline.state.dispatchKind === 'tar';

  const activeTable: TableProgress | null = useMemo(() => {
    if (tables.length === 0) return null;
    if (activeTableName) {
      const found = tables.find((t) => t.name === activeTableName);
      if (found) return found;
    }
    return tables[0];
  }, [tables, activeTableName]);

  // Goals + charts live at the top of the pipeline state (one cross-dataset
  // call produces them all). For multi-table uploads the goals are shown
  // together; the data_source pill on each card identifies the table.
  const goals = pipeline.state.goals;
  const goalsTotal = pipeline.state.goalsTotal || nGoals;
  const charts = pipeline.state.charts;
  const plotStatuses = pipeline.state.plotStatuses;
  const plotErrors = pipeline.state.plotErrors;

  const hasProgressiveSummary = !!activeTable?.summary;
  const showResults = hasProgressiveSummary || urlResult !== null;

  // URL upload (non-streaming) — wrap into the same shape so the rest of the
  // UI doesn't have to branch on it.
  const urlGoals = useMemo(() => {
    if (!urlResult) return [];
    return (urlResult.goals || []).map((g: any, i: number) => ({
      goal: {
        index: typeof g.index === 'number' ? g.index : i,
        question: g.question || '',
        visualization: g.visualization || '',
        rationale: g.rationale || '',
      },
      dataSource: urlResult.data_filename || '',
    }));
  }, [urlResult]);

  const urlCharts = useMemo(() => {
    if (!urlResult) return {};
    const out: Record<number, any> = {};
    (urlResult.charts || []).forEach((c: any, i: number) => { out[i] = c; });
    return out;
  }, [urlResult]);

  const urlPlotStatuses = useMemo(() => {
    if (!urlResult) return {};
    const out: Record<number, any> = {};
    (urlResult.charts || []).forEach((_: any, i: number) => { out[i] = 'rendered'; });
    return out;
  }, [urlResult]);

  const displayedSummary = activeTable?.summary ?? urlResult?.summary ?? null;
  const displayedGoals = urlResult ? urlGoals : goals;
  const displayedCharts = urlResult ? urlCharts : charts;
  const displayedPlotStatuses = urlResult ? urlPlotStatuses : plotStatuses;
  const displayedPlotErrors = urlResult ? {} : plotErrors;
  const displayedTotal = urlResult ? urlGoals.length : goalsTotal;
  const displayedTableName = activeTable?.name ?? urlResult?.data_filename ?? '';

  return (
    <div className="container">
      <header className="header">
        <h1>LIDA</h1>
        <p>Automatic Generation of Visualizations and Infographics using Large Language Models</p>
      </header>

      <main style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem', alignItems: 'center' }}>
        <section className="controls" style={{ maxWidth: '800px' }}>
          <ModelSelector onModelChange={handleModelChange} />
          <div className="status" style={{ marginTop: '1rem', color: 'var(--color-text-muted)' }}>
            Active Model: <span className="highlight">{activeModel.model || 'None'}</span> <small>({activeModel.provider})</small>
          </div>
        </section>

        {errorBanner && (
          <div
            className="error-banner"
            style={{
              backgroundColor: '#ef444422',
              border: '1px solid #ef4444',
              color: '#fca5a5',
              padding: '1rem',
              borderRadius: '8px',
              width: '100%',
              maxWidth: '800px',
            }}
          >
            {errorBanner}
          </div>
        )}

        <section className="upload-area">
          <FileUploader
            nGoals={nGoals}
            onNGoalsChange={setNGoals}
            onStart={handleStart}
            onUrlSubmit={handleUrlSubmit}
            onReset={handleReset}
            pipeline={pipeline.state}
          />
        </section>

        {showResults && displayedSummary && (
          <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
            {isMultiTable && tables.length > 0 && (
              <section className="tables-area">
                <div className="summary-card">
                  <h3>Tables</h3>
                  <p style={{ color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                    {tables.length} table{tables.length === 1 ? '' : 's'} detected. Select one to view its summary.
                  </p>
                  <div style={{ marginTop: '1rem', display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                    {tables.map((t) => {
                      const goalsForTable = goals.filter((g) => g.dataSource === t.name).length;
                      return (
                        <button
                          key={t.name}
                          onClick={() => setActiveTableName(t.name)}
                          className={displayedTableName === t.name ? 'primary' : 'secondary'}
                          style={{ fontSize: '0.9rem' }}
                        >
                          {t.name}
                          <small style={{ marginLeft: '0.4rem', opacity: 0.75 }}>
                            ({goalsForTable} goal{goalsForTable === 1 ? '' : 's'})
                          </small>
                        </button>
                      );
                    })}
                  </div>
                </div>
              </section>
            )}

            <section className="summary-area">
              <div className="summary-card">
                <h3>Dataset Summary{displayedTableName ? ` — ${displayedTableName}` : ''}</h3>
                <div style={{ marginTop: '1rem', display: 'grid', gap: '0.5rem' }}>
                  <div><strong>Name:</strong> {displayedSummary.name}</div>
                  <div><strong>Description:</strong> {displayedSummary.dataset_description}</div>
                </div>
              </div>
            </section>

            <section className="goals-area">
              <h3>Goals & Visualizations</h3>
              <GoalsBoard
                goals={displayedGoals}
                charts={displayedCharts}
                plotStatuses={displayedPlotStatuses}
                plotErrors={displayedPlotErrors}
                totalExpected={displayedTotal}
                onGoalSelect={(item) => setSelectedGoal(item)}
                selectedIndex={selectedGoal ? selectedGoal.goal.index : null}
                showDataSource={isMultiTable}
              />
            </section>

            {selectedGoal && (
              <section className="visualization-area">
                <h3>Visualization</h3>
                <p style={{ marginBottom: '1rem' }}>
                  <strong style={{ color: 'var(--color-accent)' }}>Goal:</strong> {selectedGoal.goal.question}
                  {isMultiTable && (
                    <span className="goal-row__source-pill" style={{ marginLeft: '0.5rem' }}>
                      {selectedGoal.dataSource}
                    </span>
                  )}
                </p>
                <Visualizer
                  summary={
                    tables.find((t) => t.name === selectedGoal.dataSource)?.summary ?? displayedSummary
                  }
                  goal={selectedGoal.goal}
                  library="seaborn"
                />
              </section>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
