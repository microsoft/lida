import React, { useEffect, useRef, useState } from 'react';
import { useSseFetch } from '../hooks/useSseFetch';
import { TokenCounter } from './TokenCounter';
import type { LlmActivity } from '../hooks/useUploadPipeline';

interface Goal {
    question: string;
    visualization: string;
    rationale: string;
    index: number;
}

interface Summary {
    name: string;
    file_name: string;
    dataset_description: string;
    field_names: string[];
}

interface GoalGeneratorProps {
    summary: Summary;
    onGoalSelect: (goal: Goal) => void;
    initialGoals?: Goal[];
}

type StageStatus = 'idle' | 'active' | 'done' | 'error';

const emptyLlm = (): LlmActivity => ({ tokens: 0, elapsedMs: 0, tail: '' });

export const GoalGenerator: React.FC<GoalGeneratorProps> = ({ summary, onGoalSelect, initialGoals = [] }) => {
    const [goals, setGoals] = useState<Goal[]>(initialGoals);
    const [n, setN] = useState(2);
    const [stage, setStage] = useState<StageStatus>('idle');
    const [errorMsg, setErrorMsg] = useState<string | null>(null);
    const [llm, setLlm] = useState<LlmActivity>(emptyLlm);
    const tailRef = useRef('');
    const { start, cancel } = useSseFetch();

    useEffect(() => {
        if (initialGoals && initialGoals.length > 0) {
            setGoals(initialGoals);
        }
    }, [initialGoals]);

    useEffect(() => () => cancel(), [cancel]);

    const generateGoals = async () => {
        setStage('active');
        setErrorMsg(null);
        setLlm(emptyLlm());
        tailRef.current = '';
        // Clear so streaming goals appear progressively rather than after
        // a previous run's list.
        setGoals([]);

        await start(
            '/api/v1/goal/stream',
            { summary, n, textgen_config: { n: 1, temperature: 0 } },
            {
                onFrame: (frame) => {
                    switch (frame.event) {
                        case 'stage':
                            if (frame.status === 'done') setStage('done');
                            else if (frame.status === 'error') {
                                setStage('error');
                                setErrorMsg(frame.message || 'Goal generation failed');
                            }
                            break;
                        case 'llm.token': {
                            tailRef.current = (tailRef.current + (frame.delta || '')).slice(-120);
                            setLlm({
                                tokens: frame.tokens || 0,
                                elapsedMs: frame.elapsed_ms || 0,
                                tail: tailRef.current,
                            });
                            break;
                        }
                        case 'goal.ready': {
                            // Append as the goal's closing brace arrives.
                            const incoming = frame.goal as Goal | undefined;
                            if (!incoming) break;
                            setGoals((prev) => {
                                // Avoid double-adding if the final 'goals'
                                // frame has already replaced the list.
                                if (prev.some((g) => g.index === incoming.index)) return prev;
                                return [...prev, incoming];
                            });
                            break;
                        }
                        case 'goals':
                            // Final authoritative list — replaces incremental
                            // additions in case the streaming parser missed any.
                            if (Array.isArray(frame.data)) setGoals(frame.data);
                            break;
                        case 'error':
                            setStage('error');
                            setErrorMsg(frame.message || 'Goal generation failed');
                            break;
                        case 'complete':
                            break;
                        case 'done':
                            break;
                    }
                },
                onError: (msg) => {
                    setStage('error');
                    setErrorMsg(msg);
                },
            },
        );
        setStage((s) => (s === 'active' ? 'done' : s));
    };

    const busy = stage === 'active';

    return (
        <div className="goal-generator">
            <div className="goal-controls">
                <label>
                    Number of Goals:
                    <input
                        type="number"
                        min="1"
                        max="10"
                        value={n}
                        onChange={(e) => setN(parseInt(e.target.value))}
                        disabled={busy}
                    />
                </label>
                <button onClick={generateGoals} disabled={busy}>
                    {busy ? 'Generating…' : 'Generate Goals'}
                </button>
            </div>

            {(busy || llm.tokens > 0) && (
                <div className="goal-stream">
                    <TokenCounter llm={llm} active={busy} />
                </div>
            )}

            {errorMsg && (
                <div className="goal-stream__error">{errorMsg}</div>
            )}

            <div className="goal-list">
                {goals.map((goal, idx) => (
                    <div key={idx} className="goal-card" onClick={() => onGoalSelect(goal)}>
                        <h4>{goal.question}</h4>
                        <p>{goal.rationale}</p>
                        <div className="viz-code-preview">
                            <small>{goal.visualization.slice(0, 50)}...</small>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
