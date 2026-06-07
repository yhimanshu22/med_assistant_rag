import { useCallback, useEffect, useState } from 'react';
import { Activity, AlertCircle, ChevronDown, ChevronUp, Loader2, RefreshCw } from 'lucide-react';
import { getMetrics } from '../api';
import type { MetricsSnapshot } from '../types';

const formatMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${Math.round(ms)}ms`);

const formatPct = (rate: number) => `${Math.round(rate * 100)}%`;

type ModelMetricsDrawerProps = {
  refreshKey?: number;
};

export default function ModelMetricsDrawer({ refreshKey = 0 }: ModelMetricsDrawerProps) {
  const [open, setOpen] = useState(false);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await getMetrics();
      setMetrics(data);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load metrics';
      setError(
        msg.includes('404')
          ? 'Metrics endpoint not found — restart the backend (uvicorn) to load the latest code.'
          : msg,
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 60_000);
    return () => clearInterval(interval);
  }, [open, fetchMetrics, refreshKey]);

  return (
    <div className={`sidebar-metrics-drawer ${open ? 'open' : ''}`}>
      <button
        type="button"
        className="sidebar-metrics-toggle"
        onClick={() => setOpen((prev) => !prev)}
        aria-expanded={open}
      >
        <span className="sidebar-metrics-toggle-label">
          <Activity size={16} />
          Model Metrics
        </span>
        {open ? <ChevronDown size={16} /> : <ChevronUp size={16} />}
      </button>

      {open && (
        <div className="sidebar-metrics-panel">
          <div className="sidebar-metrics-toolbar">
            <span className="sidebar-metrics-live">Live · 1 min refresh</span>
            <button
              type="button"
              className="sidebar-metrics-refresh"
              onClick={fetchMetrics}
              disabled={loading}
              title="Refresh now"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            </button>
          </div>

          {error && (
            <div className="sidebar-metrics-error">
              <AlertCircle size={14} />
              <span>{error}</span>
            </div>
          )}

          {metrics && !error && (
            <div className="sidebar-metrics-content">
              <div className="metrics-grid">
                <div className="metric-card">
                  <span className="metric-label">Queries</span>
                  <span className="metric-value">{metrics.queries_total}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Errors</span>
                  <span className={`metric-value ${metrics.errors_total > 0 ? 'warn' : ''}`}>
                    {metrics.errors_total}
                  </span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Retrieval hit</span>
                  <span className="metric-value">{formatPct(metrics.retrieval_hit_rate)}</span>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Cache hits</span>
                  <span className="metric-value">{metrics.cache_hits}</span>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Query latency</h3>
                <div className="metrics-row">
                  <span>Avg</span>
                  <strong>{formatMs(metrics.latency_ms.avg_ms)}</strong>
                </div>
                <div className="metrics-row">
                  <span>P50</span>
                  <strong>{formatMs(metrics.latency_ms.p50_ms)}</strong>
                </div>
                <div className="metrics-row">
                  <span>P95</span>
                  <strong>{formatMs(metrics.latency_ms.p95_ms)}</strong>
                </div>
              </div>

              <div className="metrics-section">
                <h3>Stage latency (avg)</h3>
                {(['rewrite', 'retrieve', 'llm', 'eval'] as const).map((stage) => (
                  <div key={stage} className="metrics-row">
                    <span>{stage}</span>
                    <strong>{formatMs(metrics.stages_ms[stage].avg_ms)}</strong>
                  </div>
                ))}
              </div>

              <div className="metrics-section">
                <h3>Evaluation scores (avg)</h3>
                <div className="metrics-row">
                  <span>Faithfulness</span>
                  <strong>{formatPct(metrics.evaluation_scores.faithfulness.avg)}</strong>
                </div>
                <div className="metrics-row">
                  <span>Relevance</span>
                  <strong>{formatPct(metrics.evaluation_scores.relevance.avg)}</strong>
                </div>
                <div className="metrics-row">
                  <span>Confidence</span>
                  <strong>{formatPct(metrics.evaluation_scores.confidence.avg)}</strong>
                </div>
              </div>

              {metrics.recent_errors.length > 0 && (
                <div className="metrics-section">
                  <h3>Recent errors</h3>
                  <ul className="metrics-error-list">
                    {metrics.recent_errors.slice().reverse().map((entry, idx) => (
                      <li key={`${entry.at}-${idx}`}>
                        <span className="metrics-error-event">{entry.event}</span>
                        <span className="metrics-error-msg">{entry.error}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
