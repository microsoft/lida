import { useCallback, useReducer, useRef } from 'react';
import pako from 'pako';

export type StageId = 'compress' | 'upload' | 'analyze' | 'goals' | 'visualize';
export type StageStatus = 'pending' | 'active' | 'done' | 'error';

export interface StageState {
  id: StageId;
  label: string;
  status: StageStatus;
  detail?: string;
}

export interface ByteProgress {
  processed: number;
  total: number;
}

export interface LlmActivity {
  tokens: number;
  elapsedMs: number;
  // Most recent ~120 chars of streamed output, used as a "the model is talking"
  // ticker. Kept short so React reconciliation is cheap.
  tail: string;
}

export type PlotStatus = 'pending' | 'generating' | 'code_ready' | 'rendered' | 'failed';

export interface GoalEntry {
  index: number;
  question: string;
  visualization: string;
  rationale: string;
}

export interface ChartEntry {
  spec?: any;
  status: boolean;
  raster?: string | null;
  code?: string;
  library?: string;
  error?: any;
}

export interface TableProgress {
  name: string;
  status: 'pending' | 'analyzing' | 'done' | 'error';
  error?: string;
  summary: any | null;
}

export interface GoalWithSource {
  goal: GoalEntry;
  dataSource: string;
}

export interface PipelineState {
  active: boolean;
  finished: boolean;
  stages: StageState[];
  compress: ByteProgress;
  upload: ByteProgress;
  llm: LlmActivity;
  tables: TableProgress[];
  // Goals + charts are top-level: one cross-dataset goal call produces all
  // of them, and each plot is 1:1 with a goal. ``dataSource`` on the goal
  // identifies which table summary the chart is about.
  goals: GoalWithSource[];
  goalsTotal: number;
  charts: Record<number, ChartEntry>;
  plotStatuses: Record<number, PlotStatus>;
  plotErrors: Record<number, string>;
  dispatchKind: 'single' | 'sqlite' | 'tar' | null;
  result: any | null;
  error: string | null;
}

const newTable = (name: string, status: TableProgress['status'] = 'pending'): TableProgress => ({
  name,
  status,
  summary: null,
});

const STAGE_LABELS: Record<StageId, string> = {
  compress: 'Compressing',
  upload: 'Uploading',
  analyze: 'Analyzing',
  goals: 'Generating goals',
  visualize: 'Rendering charts',
};

const STAGE_ORDER: StageId[] = ['compress', 'upload', 'analyze', 'goals', 'visualize'];

const initialState = (): PipelineState => ({
  active: false,
  finished: false,
  stages: STAGE_ORDER.map((id) => ({ id, label: STAGE_LABELS[id], status: 'pending' })),
  compress: { processed: 0, total: 0 },
  upload: { processed: 0, total: 0 },
  llm: { tokens: 0, elapsedMs: 0, tail: '' },
  tables: [],
  goals: [],
  goalsTotal: 0,
  charts: {},
  plotStatuses: {},
  plotErrors: {},
  dispatchKind: null,
  result: null,
  error: null,
});

type Action =
  | { type: 'reset' }
  | { type: 'begin'; total: number }
  | { type: 'compress.progress'; processed: number; total: number }
  | { type: 'compress.done' }
  | { type: 'upload.progress'; processed: number; total: number }
  | { type: 'upload.done' }
  | { type: 'stage'; stage: StageId; status: StageStatus; detail?: string }
  | { type: 'llm.token'; delta: string; tokens: number; elapsedMs: number }
  | { type: 'dispatch'; kind: 'single' | 'sqlite' | 'tar' }
  | { type: 'tables.detected'; names: string[] }
  | { type: 'table.status'; name: string; status: TableProgress['status']; error?: string }
  | { type: 'table.summary'; name: string; summary: any }
  | { type: 'goals.total'; total: number }
  | { type: 'goal.ready'; goal: GoalEntry; dataSource: string }
  | { type: 'plot.status'; index: number; status: PlotStatus; error?: string }
  | { type: 'chart.rendered'; index: number; chart: ChartEntry }
  | { type: 'complete'; result: any }
  | { type: 'error'; message: string };

function updateTable(state: PipelineState, name: string, mut: (t: TableProgress) => TableProgress): PipelineState {
  // If the table doesn't exist yet (e.g., events arrive before dispatch had
  // a chance to populate the list), create it on demand.
  if (!state.tables.some((t) => t.name === name)) {
    state = { ...state, tables: [...state.tables, newTable(name)] };
  }
  return {
    ...state,
    tables: state.tables.map((t) => (t.name === name ? mut(t) : t)),
  };
}

function reducer(state: PipelineState, action: Action): PipelineState {
  switch (action.type) {
    case 'reset':
      return initialState();
    case 'begin':
      return {
        ...initialState(),
        active: true,
        compress: { processed: 0, total: action.total },
        upload: { processed: 0, total: 0 },
        stages: STAGE_ORDER.map((id) => ({
          id,
          label: STAGE_LABELS[id],
          status: id === 'compress' ? 'active' : 'pending',
        })),
      };
    case 'compress.progress':
      return { ...state, compress: { processed: action.processed, total: action.total } };
    case 'compress.done':
      return {
        ...state,
        compress: { processed: state.compress.total, total: state.compress.total },
        stages: state.stages.map((s) =>
          s.id === 'compress' ? { ...s, status: 'done' } : s.id === 'upload' ? { ...s, status: 'active' } : s,
        ),
      };
    case 'upload.progress':
      return { ...state, upload: { processed: action.processed, total: action.total } };
    case 'upload.done':
      return {
        ...state,
        upload: { processed: state.upload.total || state.upload.processed, total: state.upload.total || state.upload.processed },
        stages: state.stages.map((s) =>
          s.id === 'upload' ? { ...s, status: 'done' } : s.id === 'analyze' ? { ...s, status: 'active' } : s,
        ),
      };
    case 'stage':
      return {
        ...state,
        stages: state.stages.map((s) => {
          if (s.id !== action.stage) return s;
          return { ...s, status: action.status, detail: action.detail };
        }),
      };
    case 'llm.token': {
      const tail = (state.llm.tail + action.delta).slice(-120);
      return {
        ...state,
        llm: { tokens: action.tokens, elapsedMs: action.elapsedMs, tail },
      };
    }
    case 'dispatch':
      return { ...state, dispatchKind: action.kind };
    case 'tables.detected':
      return {
        ...state,
        tables: action.names.map((name) => newTable(name)),
      };
    case 'table.status':
      return updateTable(state, action.name, (t) => ({
        ...t,
        status: action.status,
        error: action.error,
      }));
    case 'table.summary':
      return updateTable(state, action.name, (t) => ({ ...t, summary: action.summary, status: 'done' }));
    case 'goals.total':
      return { ...state, goalsTotal: Math.max(state.goalsTotal, action.total) };
    case 'goal.ready': {
      if (state.goals.some((g) => g.goal.index === action.goal.index)) return state;
      const goals = [...state.goals, { goal: action.goal, dataSource: action.dataSource }]
        .sort((a, b) => a.goal.index - b.goal.index);
      return { ...state, goals, goalsTotal: Math.max(state.goalsTotal, goals.length) };
    }
    case 'plot.status': {
      const plotStatuses = { ...state.plotStatuses, [action.index]: action.status };
      const plotErrors = action.error
        ? { ...state.plotErrors, [action.index]: action.error }
        : state.plotErrors;
      return { ...state, plotStatuses, plotErrors };
    }
    case 'chart.rendered': {
      const charts = { ...state.charts, [action.index]: action.chart };
      const plotStatuses = { ...state.plotStatuses, [action.index]: 'rendered' as PlotStatus };
      return { ...state, charts, plotStatuses };
    }
    case 'complete':
      return {
        ...state,
        active: false,
        finished: true,
        result: action.result,
        stages: state.stages.map((s) => (s.status === 'pending' || s.status === 'active' ? { ...s, status: 'done' } : s)),
      };
    case 'error':
      return {
        ...state,
        active: false,
        finished: true,
        error: action.message,
        stages: state.stages.map((s) => (s.status === 'active' ? { ...s, status: 'error', detail: action.message } : s)),
      };
    default:
      return state;
  }
}

const GZIP_MAGIC = [0x1f, 0x8b];

function isGzipBytes(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === GZIP_MAGIC[0] && bytes[1] === GZIP_MAGIC[1];
}

async function chunkedGzip(file: File, onProgress: (processed: number, total: number) => void): Promise<Uint8Array> {
  const total = file.size;
  const deflater = new pako.Deflate({ level: 9, gzip: true });
  const pieces: Uint8Array[] = [];

  return new Promise<Uint8Array>((resolve, reject) => {
    deflater.onData = (chunk: Uint8Array) => {
      pieces.push(chunk);
    };
    deflater.onEnd = (status: number) => {
      if (status !== 0) {
        reject(new Error(`gzip failed (pako status ${status})`));
        return;
      }
      const totalLen = pieces.reduce((n, p) => n + p.length, 0);
      const out = new Uint8Array(totalLen);
      let offset = 0;
      for (const p of pieces) {
        out.set(p, offset);
        offset += p.length;
      }
      resolve(out);
    };

    (async () => {
      try {
        const reader = file.stream().getReader();
        let processed = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) {
            deflater.push(new Uint8Array(0), true);
            break;
          }
          deflater.push(value, false);
          processed += value.length;
          onProgress(processed, total);
        }
      } catch (err) {
        reject(err);
      }
    })();
  });
}

interface ParsedFrame {
  event: string;
  ts?: number;
  [k: string]: any;
}

export interface UploadPipelineOptions {
  onComplete?: (result: any) => void;
  onError?: (message: string) => void;
}

export function useUploadPipeline(options: UploadPipelineOptions = {}) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  const reset = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
    dispatch({ type: 'reset' });
  }, []);

  const handleEvent = useCallback(
    (frame: ParsedFrame) => {
      switch (frame.event) {
        case 'stage': {
          const stageId = frame.stage as StageId;
          if (stageId) {
            dispatch({
              type: 'stage',
              stage: stageId,
              status: frame.status === 'start' ? 'active' : frame.status === 'done' ? 'done' : frame.status === 'error' ? 'error' : 'pending',
              detail: frame.label || frame.message,
            });
          }
          break;
        }
        case 'llm.token':
          dispatch({
            type: 'llm.token',
            delta: frame.delta || '',
            tokens: frame.tokens || 0,
            elapsedMs: frame.elapsed_ms || 0,
          });
          break;
        case 'dispatch':
          dispatch({ type: 'dispatch', kind: frame.kind });
          if (frame.kind === 'single' && frame.file_name) {
            dispatch({ type: 'tables.detected', names: [frame.file_name] });
            dispatch({ type: 'table.status', name: frame.file_name, status: 'pending' });
          }
          break;
        case 'tables.detected':
          dispatch({ type: 'tables.detected', names: frame.names || [] });
          break;
        case 'table.started':
          dispatch({ type: 'table.status', name: frame.name, status: 'analyzing' });
          break;
        case 'table.done':
          dispatch({ type: 'table.status', name: frame.name, status: 'done' });
          break;
        case 'table.error':
          dispatch({ type: 'table.status', name: frame.name || frame.file_name || 'unknown', status: 'error', error: frame.error });
          break;
        case 'summary.started':
          dispatch({ type: 'table.status', name: frame.file_name, status: 'analyzing' });
          break;
        case 'summary.ready':
          if (frame.summary) {
            dispatch({ type: 'table.summary', name: frame.file_name, summary: frame.summary });
          }
          break;
        case 'goals.started':
          if (typeof frame.n_requested === 'number') {
            dispatch({ type: 'goals.total', total: frame.n_requested });
          }
          break;
        case 'goal.ready':
          if (frame.goal) {
            dispatch({
              type: 'goal.ready',
              dataSource: frame.data_source || '',
              goal: {
                index: frame.index,
                question: frame.goal.question || '',
                visualization: frame.goal.visualization || '',
                rationale: frame.goal.rationale || '',
              },
            });
            dispatch({ type: 'plot.status', index: frame.index, status: 'pending' });
          }
          break;
        case 'goals.ready':
          if (typeof frame.count === 'number') {
            dispatch({ type: 'goals.total', total: frame.count });
          }
          break;
        case 'plots.started':
          if (typeof frame.total === 'number') {
            dispatch({ type: 'goals.total', total: frame.total });
          }
          break;
        case 'plot.started':
          dispatch({ type: 'plot.status', index: frame.goal_index, status: 'generating' });
          break;
        case 'plot.code.ready':
          dispatch({ type: 'plot.status', index: frame.goal_index, status: 'code_ready' });
          break;
        case 'plot.failed':
          dispatch({
            type: 'plot.status',
            index: frame.goal_index,
            status: 'failed',
            error: frame.error,
          });
          break;
        case 'chart.rendered':
          if (typeof frame.goal_index === 'number' && frame.chart) {
            dispatch({
              type: 'chart.rendered',
              index: frame.goal_index,
              chart: frame.chart,
            });
          }
          break;
        case 'complete':
          dispatch({ type: 'complete', result: frame.data });
          options.onComplete?.(frame.data);
          break;
        case 'error':
          dispatch({ type: 'error', message: frame.message || 'Unknown error' });
          options.onError?.(frame.message || 'Unknown error');
          break;
        case 'done':
          break;
      }
    },
    [options],
  );

  const start = useCallback(
    async (file: File, opts: { nGoals?: number } = {}) => {
      const nGoals = Math.max(1, Math.min(10, Math.round(opts.nGoals ?? 5)));
      dispatch({ type: 'begin', total: file.size });
      let payload: Uint8Array;
      let outName: string;
      try {
        const headerBuf = await file.slice(0, 2).arrayBuffer();
        const isGz = isGzipBytes(new Uint8Array(headerBuf));
        if (isGz) {
          // Already gzipped — skip the compress phase but still report it.
          payload = new Uint8Array(await file.arrayBuffer());
          outName = /\.gz$/i.test(file.name) ? file.name : `${file.name}.gz`;
          dispatch({
            type: 'compress.progress',
            processed: file.size,
            total: file.size,
          });
        } else {
          payload = await chunkedGzip(file, (processed, total) => {
            dispatch({ type: 'compress.progress', processed, total });
          });
          outName = `${file.name}.gz`;
        }
        dispatch({ type: 'compress.done' });
      } catch (err: any) {
        dispatch({ type: 'error', message: err?.message || 'Compression failed' });
        options.onError?.(err?.message || 'Compression failed');
        return;
      }

      const form = new FormData();
      form.append('file', new Blob([payload as BlobPart], { type: 'application/gzip' }), outName);

      await new Promise<void>((resolve) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;
        xhr.open('POST', `/api/v1/summarize/stream?n_goals=${nGoals}&visualize=true`);

        // responseText grows monotonically. We track how much we've already
        // parsed (`consumedTo`) and on each tick drain any complete `data: ...
        // \n\n` frames from the new tail. Partial trailing frames are left for
        // the next pass.
        let consumedTo = 0;

        const drain = (final: boolean) => {
          if (!xhr.responseText) return;
          const tail = xhr.responseText.slice(consumedTo);
          if (!tail) return;
          let remaining = tail;
          let advanced = 0;
          for (;;) {
            const idx = remaining.indexOf('\n\n');
            if (idx === -1) break;
            const raw = remaining.slice(0, idx);
            remaining = remaining.slice(idx + 2);
            advanced += idx + 2;
            if (!raw.startsWith('data:')) continue;
            const body = raw.replace(/^data:\s*/, '');
            try {
              const frame = JSON.parse(body);
              handleEvent(frame);
            } catch (err) {
              console.warn('Failed to parse SSE frame', err, body);
            }
          }
          consumedTo += advanced;
          if (final && remaining.startsWith('data:')) {
            const body = remaining.replace(/^data:\s*/, '');
            try {
              handleEvent(JSON.parse(body));
            } catch {
              // tail without trailing \n\n on a clean close — ignore.
            }
          }
        };

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            dispatch({ type: 'upload.progress', processed: e.loaded, total: e.total });
          }
        };
        xhr.upload.onload = () => {
          dispatch({ type: 'upload.done' });
        };
        xhr.onprogress = () => drain(false);
        xhr.onload = () => {
          drain(true);
          if (xhr.status >= 400) {
            dispatch({ type: 'error', message: `HTTP ${xhr.status}` });
            options.onError?.(`HTTP ${xhr.status}`);
          }
          xhrRef.current = null;
          resolve();
        };
        xhr.onerror = () => {
          dispatch({ type: 'error', message: 'Network error' });
          options.onError?.('Network error');
          xhrRef.current = null;
          resolve();
        };
        xhr.onabort = () => {
          xhrRef.current = null;
          resolve();
        };
        xhr.send(form);
      });
    },
    [handleEvent, options],
  );

  return { state, start, reset } as const;
}
