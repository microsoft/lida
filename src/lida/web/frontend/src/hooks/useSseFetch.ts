import { useCallback, useRef } from 'react';

export interface SseFrame {
  event: string;
  ts?: number;
  [k: string]: any;
}

export interface SseFetchHandlers {
  onFrame: (frame: SseFrame) => void;
  onError?: (message: string) => void;
}

/**
 * Minimal SSE-over-XHR consumer. POSTs a JSON body to ``url`` and parses the
 * text/event-stream response incrementally, calling ``onFrame`` for each
 * complete ``data: {...}\n\n`` block. Use this for endpoints that need to
 * stream progress alongside their final result (goal generation, viz repair,
 * etc.). For multipart uploads with byte-progress, use useUploadPipeline.
 */
export function useSseFetch() {
  const xhrRef = useRef<XMLHttpRequest | null>(null);

  const cancel = useCallback(() => {
    if (xhrRef.current) {
      xhrRef.current.abort();
      xhrRef.current = null;
    }
  }, []);

  const start = useCallback(
    (url: string, body: any, handlers: SseFetchHandlers) => {
      return new Promise<void>((resolve) => {
        const xhr = new XMLHttpRequest();
        xhrRef.current = xhr;
        xhr.open('POST', url);
        xhr.setRequestHeader('Content-Type', 'application/json');

        let consumedTo = 0;
        const drain = () => {
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
            // Skip SSE comments (": keepalive" etc.)
            if (!raw.startsWith('data:')) continue;
            const jsonStr = raw.replace(/^data:\s*/, '');
            try {
              handlers.onFrame(JSON.parse(jsonStr));
            } catch (err) {
              console.warn('SSE parse failed', err, jsonStr);
            }
          }
          consumedTo += advanced;
        };

        xhr.onprogress = () => drain();
        xhr.onload = () => {
          drain();
          if (xhr.status >= 400) {
            handlers.onError?.(`HTTP ${xhr.status}`);
          }
          xhrRef.current = null;
          resolve();
        };
        xhr.onerror = () => {
          handlers.onError?.('Network error');
          xhrRef.current = null;
          resolve();
        };
        xhr.onabort = () => {
          xhrRef.current = null;
          resolve();
        };

        xhr.send(JSON.stringify(body));
      });
    },
    [],
  );

  return { start, cancel } as const;
}
