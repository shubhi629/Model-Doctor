import React from "react";
import { Link, useParams } from "react-router-dom";
import {
  Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { api, errorMessage } from "../api/client.js";
import { DeployControl } from "../components/status.jsx";
import { Card, DECISION_COLORS, DecisionBadge, ErrorBox, Notice, Spinner, Stat, Table, num, pct, useLoader } from "../components/ui.jsx";

const MD = "#4f46e5";
const BASE = "#94a3b8";

function RiskCoverageChart({ curves, operating }) {
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="coverage" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)}
          label={{ value: "Coverage (share of predictions accepted)", position: "insideBottom", offset: -10, fontSize: 12 }} fontSize={12} />
        <YAxis tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
        <Tooltip formatter={(v) => pct(v)} labelFormatter={(v) => `Coverage ${pct(v)}`} />
        <Legend verticalAlign="top" height={28} />
        <Line data={curves.baseline} dataKey="selective_risk" name="Confidence baseline" stroke={BASE} strokeWidth={2} dot={false} />
        <Line data={curves.modeldoctor} dataKey="selective_risk" name="ModelDoctor" stroke={MD} strokeWidth={2.5} dot={false} />
        {operating > 0 && <ReferenceLine x={operating} stroke="#059669" strokeDasharray="4 4" label={{ value: "TRUST cut-off", fontSize: 11, fill: "#059669", position: "insideTopRight" }} />}
      </LineChart>
    </ResponsiveContainer>
  );
}

function CalibrationChart({ curves }) {
  const diag = [{ mean_predicted: 0, observed_accuracy: 0 }, { mean_predicted: 1, observed_accuracy: 1 }];
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart margin={{ top: 10, right: 16, bottom: 20, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis type="number" dataKey="mean_predicted" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12}
          label={{ value: "Estimated probability of being correct", position: "insideBottom", offset: -10, fontSize: 12 }} />
        <YAxis type="number" domain={[0, 1]} tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
        <Legend verticalAlign="top" height={28} />
        <Line data={diag} dataKey="observed_accuracy" name="Perfect calibration" stroke="#cbd5e1" strokeDasharray="5 5" dot={false} legendType="plainline" />
        <Line data={curves.baseline} dataKey="observed_accuracy" name="Model 1 confidence" stroke={BASE} strokeWidth={2} dot={{ r: 3 }} />
        <Line data={curves.modeldoctor} dataKey="observed_accuracy" name="ModelDoctor trustworthiness" stroke={MD} strokeWidth={2.5} dot={{ r: 3.5 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

function DecisionChart({ rows }) {
  const data = rows.map((r) => ({ ...r, share_pct: r.share }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 10, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
        <XAxis dataKey="decision" fontSize={12} />
        <YAxis tickFormatter={(v) => pct(v, 0)} fontSize={12} width={48} />
        <Tooltip formatter={(v, n) => (n === "share_pct" ? [pct(v), "Share of test predictions"] : v)} />
        <Bar dataKey="share_pct" radius={[4, 4, 0, 0]}>
          {data.map((r) => <Cell key={r.decision} fill={DECISION_COLORS[r.decision]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function ReportPage() {
  const { id } = useParams();
  const { loading, error, data: r, reload } = useLoader(() => api.report(id), [id]);

  if (loading) return <Spinner label="Loading report…" />;
  if (error)
    return (
      <div className="space-y-3">
        <ErrorBox message={errorMessage(error)} onRetry={reload} />
        {error?.response?.status === 409 && <Link className="text-indigo-600 underline" to={`/s/${id}/train`}>See training status</Link>}
      </div>
    );

  const cmp = Object.fromEntries(r.comparison.map((c) => [c.method, c]));
  const md = cmp["ModelDoctor"], base = cmp["Confidence baseline"];
  const better = r.selective_risk_improvement > 0.005 && r.reliability_mode === "model2";
  const worse = r.selective_risk_improvement < -0.005;
  const t = r.thresholds;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">System report</h1>
          <p className="mt-1 text-sm text-slate-600">
            Target <b>{r.target_column}</b> · {r.task_detail?.replace("_", " ")} · classes {r.classes.join(", ")} · Model 1:{" "}
            <b>{r.selected_model}</b> ({r.calibration_method} calibrated)
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <DeployControl sessionId={id} />
          <Link to={`/s/${id}/predict`} className="rounded-lg bg-indigo-600 px-3.5 py-2 text-sm font-medium text-white hover:bg-indigo-500">
            Make predictions →
          </Link>
        </div>
      </div>

      <Notice tone={better ? "good" : worse ? "bad" : "info"}>
        <span className="font-medium">Verdict (held-out test split, {r.n_test} rows): </span>
        {r.verdict}
      </Notice>
      {r.reliability_mode === "confidence_fallback" && (
        <Notice tone="warn">
          <b>Insufficient error signal.</b> {r.model2.message} Trustworthiness is Model 1's calibrated confidence and
          thresholds were tuned on the calibration split.
        </Notice>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Stat label="Error, no filtering" value={pct(r.overall_error_rate)} hint="Model 1 on all test predictions" />
        <Stat label="ModelDoctor error (TRUST)" value={pct(md.selective_risk)} hint={`at ${pct(md.coverage)} coverage`} tone="accent" />
        <Stat label="Baseline error, same coverage" value={pct(base.selective_risk)} hint="top-confidence predictions" />
        <Stat
          label="Error-detection AUROC"
          value={num(r.auroc.modeldoctor, 3)}
          hint={`baseline ${num(r.auroc.confidence_baseline, 3)}`}
          tone={r.auroc.modeldoctor > r.auroc.confidence_baseline ? "good" : "default"}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Risk–coverage curve" subtitle="Error among accepted predictions as more are accepted. Lower is better.">
          <RiskCoverageChart curves={r.risk_coverage_curve} operating={md.coverage} />
          <div className="mt-2 text-xs text-slate-500">
            AURC: ModelDoctor {num(r.aurc.modeldoctor, 4)} · baseline {num(r.aurc.confidence_baseline, 4)}
          </div>
        </Card>
        <Card title="Calibration (reliability diagram)" subtitle="Do estimated probabilities match how often predictions are actually correct?">
          <CalibrationChart curves={r.calibration_curve} />
          <div className="mt-2 text-xs text-slate-500">
            ECE: ModelDoctor {num(r.calibration.modeldoctor.ece, 4)} · confidence {num(r.calibration.confidence_baseline.ece, 4)} ·
            Brier: {num(r.calibration.modeldoctor.brier, 4)} vs {num(r.calibration.confidence_baseline.brier, 4)}
          </div>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="ModelDoctor vs baseline" subtitle="Compared at the same coverage (fair comparison)">
          <Table
            rows={r.comparison}
            rowKey={(x) => x.method}
            columns={[
              { key: "method", label: "Method", render: (x) => <b className={x.method === "ModelDoctor" ? "text-indigo-700" : ""}>{x.method}</b> },
              { key: "coverage", label: "Coverage", align: "right", render: (x) => pct(x.coverage) },
              { key: "selective_risk", label: "Error among accepted", align: "right", render: (x) => pct(x.selective_risk) },
              { key: "accepted", label: "Accepted", align: "right", render: (x) => `${x.accepted} / ${x.total}` },
            ]}
          />
          <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
            {["modeldoctor", "confidence_baseline"].map((k) => (
              <div key={k} className="rounded-lg bg-slate-50 p-3">
                <div className="text-xs font-medium uppercase text-slate-500">{k === "modeldoctor" ? "ModelDoctor" : "Baseline"} error detection</div>
                <div className="tabular mt-1">Precision {pct(r.error_detection[k].precision)} · Recall {pct(r.error_detection[k].recall)}</div>
                <div className="tabular text-xs text-slate-500">
                  caught {r.error_detection[k].errors_caught} of {r.error_detection[k].errors_total} errors · AUPRC {num(r.auprc[k], 3)}
                </div>
              </div>
            ))}
          </div>
        </Card>
        <Card title="Decisions on the test split" subtitle="Share of predictions and observed error rate per decision">
          <DecisionChart rows={r.decision_distribution} />
          <Table
            rows={r.decision_distribution}
            rowKey={(x) => x.decision}
            columns={[
              { key: "decision", label: "Decision", render: (x) => <DecisionBadge decision={x.decision} /> },
              { key: "count", label: "Count", align: "right" },
              { key: "observed_error_rate", label: "Observed error", align: "right", render: (x) => pct(x.observed_error_rate) },
              { key: "mean_trustworthiness", label: "Mean trust", align: "right", render: (x) => pct(x.mean_trustworthiness) },
            ]}
          />
        </Card>
      </div>

      <Card title="Thresholds" subtitle="Tuned on out-of-fold predictions on the calibration split — never on test data">
        <div className="grid gap-4 md:grid-cols-3">
          <Stat label="TRUST if trust ≥" value={pct(t.trust_threshold)} tone="good" />
          <Stat label="REJECT if trust <" value={t.reject_threshold === null ? "disabled" : pct(t.reject_threshold)} tone="bad" />
          <Stat label="Target risk / min coverage" value={`${pct(t.target_risk, 0)} / ${pct(t.min_coverage, 0)}`}
            hint={t.target_risk_met ? "target met on calibration" : "target not achievable; best risk chosen"} tone={t.target_risk_met ? "default" : "warn"} />
        </div>
        <p className="mt-3 text-sm text-slate-600">{t.message}</p>
        <div className="mt-3">
          <Table
            rows={t.band_metadata}
            rowKey={(b) => b.name}
            columns={[
              { key: "name", label: "Band", render: (b) => <DecisionBadge decision={b.name} /> },
              { key: "threshold_range", label: "Range" },
              { key: "count", label: "Calibration rows", align: "right" },
              { key: "share", label: "Share", align: "right", render: (b) => pct(b.share) },
              { key: "observed_error_rate", label: "Observed error", align: "right", render: (b) => pct(b.observed_error_rate) },
            ]}
          />
          <p className="mt-2 text-xs text-slate-500">{t.band_metadata?.[0]?.note}</p>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Model 1 leaderboard" subtitle={`5-fold stratified CV on the train split · ranked by ${r.primary_metric.replace("_mean", "")}`}>
          <Table
            rows={r.leaderboard}
            rowKey={(x) => x.model}
            columns={[
              { key: "model", label: "Model", render: (x) => <span className={x.model === r.selected_model ? "font-semibold text-indigo-700" : ""}>{x.model}{x.model === r.selected_model ? " ★" : ""}</span> },
              { key: "score", label: "Score", align: "right", render: (x) => (x.status === "failed" ? <span title={x.error_message} className="text-rose-600">failed</span> : num(x.score, 4)) },
              { key: "accuracy_mean", label: "Accuracy", align: "right", render: (x) => num(x.accuracy_mean, 3) },
              { key: "weighted_f1_mean", label: "Weighted F1", align: "right", render: (x) => num(x.weighted_f1_mean, 3) },
              { key: "train_seconds", label: "Seconds", align: "right", render: (x) => num(x.train_seconds, 1) },
            ]}
          />
        </Card>
        <Card title="Model 2 (the doctor)" subtitle="Learns P(correct) from reliability features on the calibration split">
          {r.reliability_mode === "model2" ? (
            <div className="space-y-3 text-sm">
              <div>
                Winner: <b>{r.model2.winner}</b> · calibration OOF ECE {num(r.model2.calibration_ece_oof, 4)} · trained on{" "}
                {r.model2.n_correct_calibration + r.model2.n_incorrect_calibration} rows ({r.model2.n_incorrect_calibration} errors)
              </div>
              <Table
                rows={Object.entries(r.model2.candidates).map(([name, v]) => ({ name, ...v }))}
                rowKey={(x) => x.name}
                columns={[
                  { key: "name", label: "Candidate (OOF on calibration)" },
                  { key: "cv_auroc", label: "Error AUROC", align: "right", render: (x) => num(x.cv_auroc, 4) },
                  { key: "ece", label: "ECE", align: "right", render: (x) => num(x.ece, 4) },
                  { key: "brier", label: "Brier", align: "right", render: (x) => num(x.brier, 4) },
                ]}
              />
              <div className="text-xs text-slate-500">Features: {r.model2.feature_names.join(", ")}</div>
              {r.model2.feature_effects && (
                <div className="text-xs text-slate-500">
                  Strongest effects (standardized coefficient; + raises trust):{" "}
                  {r.model2.feature_effects.slice(0, 5).map((f) => `${f.feature} ${f.coefficient > 0 ? "+" : ""}${f.coefficient}`).join(" · ")}
                </div>
              )}
            </div>
          ) : (
            <Notice tone="warn">{r.model2.message}</Notice>
          )}
        </Card>
      </div>

      <Card title="Dataset profile" subtitle={`${r.profile.row_count.toLocaleString()} rows · ${r.profile.feature_count} features`}>
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <div className="mb-1 text-xs font-medium uppercase text-slate-500">Splits</div>
            {["train", "calibration", "test"].map((s) => (
              <div key={s} className="tabular flex justify-between text-sm">
                <span className="capitalize">{s}</span>
                <span>{r.splits[s].rows} rows ({r.splits[s].percentage}%)</span>
              </div>
            ))}
          </div>
          <div>
            <div className="mb-1 text-xs font-medium uppercase text-slate-500">Class distribution</div>
            {Object.entries(r.profile.class_distribution).map(([k, v]) => (
              <div key={k} className="tabular flex justify-between text-sm">
                <span>{k}</span>
                <span>{v} ({r.profile.class_percentages[k]}%)</span>
              </div>
            ))}
          </div>
          <div>
            <div className="mb-1 text-xs font-medium uppercase text-slate-500">Features used</div>
            <div className="text-sm">{r.features.numeric.length} numeric · {r.features.categorical.length} categorical</div>
            {r.dropped_columns.length > 0 && (
              <div className="mt-1 text-xs text-slate-500">
                Dropped: {r.dropped_columns.map((d) => `${d.column} (${d.reason})`).join(", ")}
              </div>
            )}
          </div>
        </div>
        {r.profile.warnings.length > 0 && (
          <div className="mt-4">
            <Notice tone="warn">
              <ul className="list-disc space-y-0.5 pl-5">{r.profile.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </Notice>
          </div>
        )}
        <div className="mt-4 text-xs text-slate-500">
          Training took {r.total_seconds}s ·{" "}
          {Object.entries(r.timings_seconds).map(([k, v]) => `${k.replace("_", " ")} ${v}s`).join(" · ")}
        </div>
      </Card>
    </div>
  );
}
