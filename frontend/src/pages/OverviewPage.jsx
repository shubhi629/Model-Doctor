import React from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, errorMessage } from "../api/client.js";
import { DeployControl } from "../components/status.jsx";
import { TrafficLight } from "../components/decision.jsx";
import { Button, Card, DECISION_COLORS, Empty, ErrorBox, Spinner, Stat, pct, useLoader } from "../components/ui.jsx";

const DECISIONS = ["TRUST", "REVIEW", "REJECT"];
const SOURCE_LABELS = { single: "Dashboard form", batch: "Dashboard CSV", api: "API /v1/predict", api_batch: "API /v1/predict/batch" };

function DecisionStat({ decision, row, total }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">{decision}</span>
        <TrafficLight decision={decision} size="sm" />
      </div>
      <div className="tabular mt-1 text-2xl font-semibold" style={{ color: DECISION_COLORS[decision] }}>
        {(row?.count ?? 0).toLocaleString()}
      </div>
      <div className="mt-1 text-xs text-slate-500">{total ? pct((row?.count ?? 0) / total) : "—"} of predictions</div>
    </div>
  );
}

function Histogram({ data, label }) {
  return (
    <ResponsiveContainer width="100%" height={240}>
      <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 18 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="bin" fontSize={11} interval={0} angle={-30} textAnchor="end" height={40}
          label={{ value: label, position: "insideBottom", offset: -14, fontSize: 12 }} />
        <YAxis allowDecimals={false} fontSize={12} width={40} />
        <Tooltip />
        <Legend verticalAlign="top" height={28} />
        {DECISIONS.map((d) => <Bar key={d} dataKey={d} stackId="d" fill={DECISION_COLORS[d]} name={d} />)}
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function OverviewPage() {
  const { id } = useParams();
  const mon = useLoader(() => api.monitoring(id, { recent: 0 }), [id]);

  if (mon.loading && !mon.data) return <Spinner label="Loading overview…" />;
  if (mon.error)
    return (
      <div className="space-y-3">
        <ErrorBox message={errorMessage(mon.error)} onRetry={mon.reload} />
        <Link className="text-indigo-600 underline" to={`/s/${id}/train`}>See training status</Link>
      </div>
    );

  const m = mon.data;
  const n = m.total_predictions;
  const byDecision = Object.fromEntries(m.decision_distribution.map((d) => [d.decision, d]));
  const pie = m.decision_distribution.filter((d) => d.count > 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Overview</h1>
          <p className="mt-1 text-sm text-slate-600">
            Every prediction this model has served, from the dashboard and from the API.
            {n > 0 && ` ${new Date(m.first_prediction_at).toLocaleString()} – ${new Date(m.last_prediction_at).toLocaleString()}.`}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <DeployControl sessionId={id} />
          <Button variant="secondary" onClick={mon.reload}>{mon.loading ? "Refreshing…" : "Refresh"}</Button>
        </div>
      </div>

      {n === 0 ? (
        <Card>
          <Empty title="No predictions served yet">
            Score records on the <Link className="text-indigo-600 underline" to={`/s/${id}/predict`}>Predict page</Link> or
            through <code>POST /api/v1/predict</code>; the charts here are built from those predictions only.
          </Empty>
        </Card>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Stat label="Total predictions" value={n.toLocaleString()} hint={`${m.labeled_predictions} with real outcomes`} />
            {DECISIONS.map((d) => <DecisionStat key={d} decision={d} row={byDecision[d]} total={n} />)}
            <Stat label="Avg confidence" value={pct(m.mean_confidence)} hint="Model 1's claim" />
            <Stat label="Avg trustworthiness" value={pct(m.mean_trustworthiness)} hint="ModelDoctor's estimate" tone="accent" />
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Card title="Decision distribution" subtitle="TRUST / REVIEW / REJECT">
              <ResponsiveContainer width="100%" height={240}>
                <PieChart>
                  <Pie data={pie} dataKey="count" nameKey="decision" innerRadius={50} outerRadius={80} paddingAngle={2}>
                    {pie.map((d) => <Cell key={d.decision} fill={DECISION_COLORS[d.decision]} />)}
                  </Pie>
                  <Tooltip formatter={(v, name) => [`${v} predictions`, name]} />
                  <Legend verticalAlign="bottom" height={28}
                    formatter={(name) => `${name} ${pct(byDecision[name]?.share, 0)}`} />
                </PieChart>
              </ResponsiveContainer>
            </Card>
            <Card className="lg:col-span-2" title="Prediction statistics" subtitle="What was predicted, and where requests came from">
              <div className="grid gap-6 sm:grid-cols-2">
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={m.predicted_class_distribution} layout="vertical" margin={{ top: 0, right: 16, left: 8, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                    <XAxis type="number" allowDecimals={false} fontSize={12} />
                    <YAxis type="category" dataKey="class" fontSize={12} width={70} />
                    <Tooltip />
                    <Bar dataKey="count" name="Predictions" fill="#4f46e5" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
                <dl className="space-y-2 text-sm">
                  {Object.entries(m.source_distribution).map(([src, count]) => (
                    <div key={src} className="flex justify-between gap-2 border-b border-slate-100 pb-1">
                      <dt className="text-slate-600">{SOURCE_LABELS[src] || src}</dt>
                      <dd className="tabular font-medium">{count.toLocaleString()}</dd>
                    </div>
                  ))}
                  <div className="flex justify-between gap-2 border-b border-slate-100 pb-1">
                    <dt className="text-slate-600">Inputs with anomaly warnings</dt>
                    <dd className="tabular font-medium">{pct(m.anomaly_rate)}</dd>
                  </div>
                  <div className="flex justify-between gap-2 border-b border-slate-100 pb-1">
                    <dt className="text-slate-600">TRUST coverage (held-out test split)</dt>
                    <dd className="tabular font-medium">{pct(m.baseline?.test_coverage)}</dd>
                  </div>
                </dl>
              </div>
            </Card>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Confidence distribution" subtitle="Model 1's top-class probability, coloured by final decision">
              <Histogram data={m.confidence_histogram} label="Model confidence" />
            </Card>
            <Card title="Trustworthiness distribution" subtitle="ModelDoctor's estimated P(correct), coloured by final decision">
              <Histogram data={m.trustworthiness_histogram} label="Trustworthiness" />
            </Card>
          </div>

          {m.timeline.length > 1 && (
            <Card title="Trend" subtitle={`Sequential windows of ${m.timeline[0].count} predictions`}>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={m.timeline} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="window" fontSize={12} tickFormatter={(w) => `#${w}`} />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
                  <Tooltip formatter={(v) => pct(v)} labelFormatter={(w) => `Window ${w}`} />
                  <Legend verticalAlign="top" height={28} />
                  <Line dataKey="coverage" name="TRUST share" stroke={DECISION_COLORS.TRUST} strokeWidth={2} dot={false} />
                  <Line dataKey="mean_trustworthiness" name="Avg trustworthiness" stroke="#4f46e5" strokeWidth={2} dot={false} />
                  <Line dataKey="mean_confidence" name="Avg confidence" stroke="#64748b" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
