import React from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, errorMessage } from "../api/client.js";
import { Button, Card, DECISION_COLORS, DecisionBadge, Empty, ErrorBox, Notice, Spinner, Stat, Table, pct, useLoader } from "../components/ui.jsx";

function FeedbackCell({ row, classes, pending, setPending }) {
  if (row.actual !== null && row.actual !== undefined) {
    const ok = row.actual === row.prediction;
    return <span className={`text-sm font-medium ${ok ? "text-emerald-700" : "text-rose-700"}`}>{row.actual} {ok ? "✓" : "✗"}</span>;
  }
  return (
    <select
      value={pending[row.prediction_id] ?? ""}
      onChange={(e) => setPending((p) => ({ ...p, [row.prediction_id]: e.target.value }))}
      className="rounded-md border border-slate-300 bg-white px-2 py-1 text-xs"
    >
      <option value="">— outcome —</option>
      {classes.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  );
}

export default function MonitoringPage() {
  const { id } = useParams();
  const [awaiting, setAwaiting] = React.useState(false);
  const mon = useLoader(() => api.monitoring(id, { awaiting_only: awaiting, recent: 100 }), [id, awaiting]);
  const schema = useLoader(() => api.schema(id), [id]);
  const [pending, setPending] = React.useState({});
  const [saving, setSaving] = React.useState(false);
  const [msg, setMsg] = React.useState(null);

  if (mon.loading && !mon.data) return <Spinner label="Loading monitoring…" />;
  if (mon.error)
    return (
      <div className="space-y-3">
        <ErrorBox message={errorMessage(mon.error)} onRetry={mon.reload} />
        <Link className="text-indigo-600 underline" to={`/s/${id}/train`}>See training status</Link>
      </div>
    );

  const m = mon.data;
  const classes = schema.data?.classes || [];
  const ove = m.observed_vs_estimated;
  const pendingCount = Object.values(pending).filter(Boolean).length;

  const submitFeedback = async () => {
    setSaving(true);
    setMsg(null);
    try {
      const items = Object.entries(pending).filter(([, v]) => v).map(([prediction_id, actual]) => ({ prediction_id, actual }));
      const res = await api.feedback(id, items);
      setMsg({ tone: res.rejected.length ? "warn" : "good", text: `${res.accepted} outcome(s) saved${res.rejected.length ? `, ${res.rejected.length} rejected` : ""}.` });
      setPending({});
      mon.reload();
    } catch (e) {
      setMsg({ tone: "bad", text: errorMessage(e) });
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Monitoring</h1>
          <p className="mt-1 text-sm text-slate-600">
            When real outcomes arrive, compare ModelDoctor's estimates with what actually happened.
          </p>
        </div>
        <Button variant="secondary" onClick={mon.reload}>{mon.loading ? "Refreshing…" : "Refresh"}</Button>
      </div>

      {m.total_predictions === 0 ? (
        <Card>
          <Empty title="No predictions served yet">
            Make predictions on the <Link className="text-indigo-600 underline" to={`/s/${id}/predict`}>Predict page</Link>. Tip: upload a batch CSV that
            includes the target column to record outcomes automatically.
          </Empty>
        </Card>
      ) : (
        <>
          {m.alerts.length > 0 ? (
            <div className="space-y-2">
              {m.alerts.map((a) => (
                <Notice key={a.code} tone={a.level === "critical" ? "bad" : "warn"}>
                  <b>{a.code.replace(/_/g, " ")}:</b> {a.message}
                </Notice>
              ))}
              {m.retraining_recommended && (
                <Notice tone="bad">Retraining is recommended. Upload fresh labeled data and train a new session.</Notice>
              )}
            </div>
          ) : (
            <Notice tone="good">
              No degradation or drift detected{m.labeled_predictions < 20 ? " (alerts need at least 20 labeled predictions)" : ""}.
            </Notice>
          )}

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="Predictions served" value={m.total_predictions.toLocaleString()} hint={`${m.labeled_predictions} with real outcomes`} />
            <Stat label="Coverage (TRUST)" value={pct(m.coverage)} hint={`test split: ${pct(m.baseline?.test_coverage)}`} tone="accent" />
            <Stat label="Estimated reliability" value={pct(ove.mean_estimated_trustworthiness)} hint="mean trustworthiness, labeled rows" />
            <Stat label="Observed accuracy" value={pct(ove.observed_accuracy)} hint={`model claimed ${pct(ove.mean_model_confidence)}`}
              tone={ove.observed_accuracy !== null && ove.mean_estimated_trustworthiness !== null && Math.abs(ove.observed_accuracy - ove.mean_estimated_trustworthiness) > 0.1 ? "warn" : "good"} />
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Decision distribution" subtitle="Bars: number of predictions · table: observed accuracy where outcomes are known">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={m.decision_distribution} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="decision" fontSize={12} />
                  <YAxis allowDecimals={false} fontSize={12} width={40} />
                  <Tooltip />
                  <Bar dataKey="count" name="Predictions" radius={[4, 4, 0, 0]}>
                    {m.decision_distribution.map((d) => <Cell key={d.decision} fill={DECISION_COLORS[d.decision]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
              <Table
                rows={m.decision_distribution}
                rowKey={(d) => d.decision}
                columns={[
                  { key: "decision", label: "Decision", render: (d) => <DecisionBadge decision={d.decision} /> },
                  { key: "count", label: "Count", align: "right" },
                  { key: "labeled", label: "Labeled", align: "right" },
                  { key: "mean_trustworthiness", label: "Est. trust", align: "right", render: (d) => pct(d.mean_trustworthiness) },
                  { key: "observed_accuracy", label: "Observed acc.", align: "right", render: (d) => pct(d.observed_accuracy) },
                ]}
              />
            </Card>
            <Card title="Over time" subtitle="Sequential windows of served predictions">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={m.timeline} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="window" fontSize={12} tickFormatter={(w) => `#${w}`} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
                  <Tooltip formatter={(v) => pct(v)} labelFormatter={(w) => `Window ${w}`} />
                  <Legend verticalAlign="top" height={28} />
                  <Line dataKey="coverage" name="Coverage" stroke="#059669" strokeWidth={2} dot={false} />
                  <Line dataKey="mean_trustworthiness" name="Est. trust" stroke="#4f46e5" strokeWidth={2} dot={false} />
                  <Line dataKey="observed_accuracy" name="Observed acc." stroke="#0f172a" strokeWidth={2} connectNulls dot={{ r: 3 }} />
                  <Line dataKey="anomaly_rate" name="Input anomalies" stroke="#d97706" strokeDasharray="4 4" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {ove.bins.length > 0 && (
            <Card title="Observed vs estimated reliability" subtitle="Labeled predictions grouped by estimated trustworthiness">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={ove.bins} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="bin" fontSize={12} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
                  <Tooltip formatter={(v) => pct(v)} />
                  <Legend verticalAlign="top" height={28} />
                  <Bar dataKey="mean_estimated" name="Estimated" fill="#a5b4fc" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="observed_accuracy" name="Observed" fill="#4f46e5" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          <Card
            title="Recent predictions & feedback"
            subtitle="Select the real outcome for a prediction, then save"
            actions={
              <div className="flex items-center gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-600">
                  <input type="checkbox" checked={awaiting} onChange={(e) => setAwaiting(e.target.checked)} />
                  Awaiting outcome only
                </label>
                <Button onClick={submitFeedback} disabled={!pendingCount || saving}>
                  {saving ? "Saving…" : `Save ${pendingCount || ""} outcome${pendingCount === 1 ? "" : "s"}`}
                </Button>
              </div>
            }
          >
            {msg && <div className="mb-3"><Notice tone={msg.tone}>{msg.text}</Notice></div>}
            <Table
              empty={awaiting ? "Every prediction has an outcome" : "No predictions"}
              rows={m.recent_predictions}
              rowKey={(r) => r.prediction_id}
              columns={[
                { key: "timestamp", label: "Time", render: (r) => new Date(r.timestamp).toLocaleTimeString() },
                { key: "source", label: "Source" },
                { key: "prediction", label: "Prediction" },
                { key: "confidence", label: "Confidence", align: "right", render: (r) => pct(r.confidence) },
                { key: "trustworthiness", label: "Trust", align: "right", render: (r) => pct(r.trustworthiness) },
                { key: "decision", label: "Decision", render: (r) => <DecisionBadge decision={r.decision} /> },
                { key: "actual", label: "Real outcome", render: (r) => <FeedbackCell row={r} classes={classes} pending={pending} setPending={setPending} /> },
              ]}
            />
          </Card>
        </>
      )}
    </div>
  );
}
