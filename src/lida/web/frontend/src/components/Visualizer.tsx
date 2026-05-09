import React, { useEffect, useState } from 'react';

interface VisualizerProps {
    summary?: any;
    goal?: any;
    library?: string;
    precomputedCharts?: any[];
}

export const Visualizer: React.FC<VisualizerProps> = ({ summary, goal, library, precomputedCharts }) => {
    const [charts, setCharts] = useState<any[]>([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        if (precomputedCharts && precomputedCharts.length > 0) {
            setCharts(precomputedCharts);
            return;
        }

        if (!summary || !goal) return;

        const fetchViz = async () => {
            setLoading(true);
            setError('');
            try {
                const res = await fetch('/api/v1/visualize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        summary,
                        goal,
                        library,
                        textgen_config: { n: 1, temperature: 0 }
                    })
                });
                const data = await res.json();
                if (data.status) {
                    setCharts(data.charts);
                } else {
                    setError(data.message);
                }
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchViz();
    }, [summary, goal, library, precomputedCharts]);

    if (loading) return <div className="loader">Generating Visualization...</div>;
    if (error) return <div className="error-banner">{error}</div>;
    if (!charts.length) return <div>No charts generated.</div>;

    return (
        <div className="visualizer">
            {charts.map((chart, i) => (
                <div key={i} className="chart-container">
                    {chart.raster ? (
                         <img src={`data:image/png;base64,${chart.raster}`} alt="Visualization" />
                    ) : (
                        <pre>{chart.code}</pre>
                    )}
                    <div className="chart-code">
                        <details>
                            <summary>Show Code</summary>
                            <pre><code>{chart.code}</code></pre>
                        </details>
                    </div>
                </div>
            ))}
        </div>
    );
};
