import React from "react";
import { Link, useParams } from "react-router-dom";
import { api, errorMessage } from "../api/client.js";
import { Button, Card, DecisionBadge, Empty, ErrorBox, Notice, Spinner, Table, pct, useLoader } from "../components/ui.jsx";
import { DecisionSummary, ExplanationPanel, RiskBadge, TrafficLight } from "../components/decision.jsx";

/** Client-side checks that mirror the API's validation, so obvious mistakes are caught before sending. */
export function validateValues(features, values) {
  const errors = {};
  for (const f of features) {
    const v = values[f.name];
    if (f.type === "numeric" && v !== "" && v !== undefined && !Number.isFinite(Number(v))) errors[f.name] = "must be a finite number";
    if (f.type === "categorical" && typeof v === "string" && v.length > 1000) errors[f.name] = "at most 1000 characters";
  }
  return errors;
}

function typicalValues(features) {
  return Object.fromEntries(
    features.map((f) => {
      if (f.type === "numeric") {
        const v = f.median ?? "";
        return [f.name, v === "" ? "" : String(f.integer ? Math.round(v) : +v.toFixed(4))];
      }
      return [f.name, f.most_frequent ?? ""];
    })
  );
}

function Field({ spec, value, onChange, error }) {
  const id = `f-${spec.name}`;
  const base = "mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500";
  const outOfRange =
    spec.type === "numeric" && value !== "" && !isNaN(Number(value)) && (Number(value) < spec.min || Number(value) > spec.max);
  const unseen = spec.type === "categorical" && value !== "" && spec.categories && !spec.categories.includes(value);
  return (
    <label htmlFor={id} className="block">
      <span className="flex items-baseline justify-between gap-2">
        <span className="truncate text-sm font-medium text-slate-700">{spec.name}</span>
        <span className="shrink-0 text-xs text-slate-400">
          {spec.type === "numeric" ? `${spec.min ?? "?"} – ${spec.max ?? "?"}` : `${spec.n_categories} options`}
        </span>
      </span>
      {spec.type === "numeric" ? (
        <input id={id} type="number" step="any" className={base} value={value} placeholder="blank = missing"
          aria-invalid={Boolean(error)} aria-describedby={error ? `${id}-err` : undefined}
          onChange={(e) => onChange(e.target.value)} />
      ) : (
        <>
          <input id={id} list={`${id}-list`} className={base} value={value} placeholder="blank = missing"
            onChange={(e) => onChange(e.target.value)} />
          <datalist id={`${id}-list`}>
            {spec.categories?.map((c) => <option key={c} value={c} />)}
          </datalist>
        </>
      )}
      {error && <span id={`${id}-err`} className="text-xs font-medium text-rose-700">{error}</span>}
      {outOfRange && <span className="text-xs text-amber-700">outside the training range</span>}
      {unseen && <span className="text-xs text-amber-700">not seen in training</span>}
    </label>
  );
}

function GapBar({ label, value, color, hint }) {
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-sm font-medium text-slate-700">{label}</span>
        <span className="tabular text-2xl font-semibold" style={{ color }}>{pct(value)}</span>
      </div>
      <div className="mt-1 h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full" style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <div className="mt-1 text-xs text-slate-500">{hint}</div>
    </div>
  );
}

function ResultCard({ result, target }) {
  const gap = result.confidence - result.trustworthiness;
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Prediction · {target}</div>
          <div className="mt-1 text-3xl font-semibold">{result.prediction}</div>
          <div className="mt-1"><RiskBadge level={result.risk_level} /></div>
        </div>
        <DecisionSummary result={result} />
      </div>

      <div className="mt-5 grid gap-5 sm:grid-cols-2">
        <GapBar label="Model confidence" value={result.confidence} color="#64748b" hint="What Model 1 claims (its top class probability)" />
        <GapBar label="Trustworthiness" value={result.trustworthiness} color="#4f46e5"
          hint={result.reliability_mode === "model2" ? "ModelDoctor's estimate: P(this prediction is correct)" : "Calibrated confidence (reliability model unavailable)"} />
      </div>
      {Math.abs(gap) >= 0.05 && (
        <p className="mt-3 text-sm text-slate-600">
          {gap > 0
            ? `ModelDoctor is ${pct(gap, 0)} less confident than Model 1: among past predictions that looked like this, fewer were correct than Model 1 claims.`
            : `ModelDoctor is ${pct(-gap, 0)} more confident than Model 1 claims for inputs like this.`}
        </p>
      )}

      <div className="mt-5"><ExplanationPanel result={result} /></div>

      <details className="mt-5 text-sm">
        <summary className="cursor-pointer text-slate-600">Class probabilities & reliability signals</summary>
        <div className="mt-2 grid gap-4 sm:grid-cols-2">
          <div>
            {Object.entries(result.class_probabilities).map(([k, v]) => (
              <div key={k} className="tabular flex justify-between"><span>{k}</span><span>{pct(v)}</span></div>
            ))}
          </div>
          <div className="text-xs text-slate-600">
            {Object.entries(result.reliability_features || {}).map(([k, v]) => (
              <div key={k} className="tabular flex justify-between"><span>{k}</span><span>{typeof v === "number" ? +v.toFixed(4) : String(v)}</span></div>
            ))}
          </div>
        </div>
      </details>
      <div className="mt-3 text-xs text-slate-400">Prediction ID {result.prediction_id} · add the real outcome later on the Monitoring page.</div>
    </Card>
  );
}

function toCsv(rows, cols) {
  const esc = (v) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
}

function BatchPanel({ id }) {
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [res, setRes] = React.useState(null);
  const [filter, setFilter] = React.useState("ALL");
  const [selected, setSelected] = React.useState(null);

  const onFile = async (file) => {
    setBusy(true);
    setError(null);
    setRes(null);
    setSelected(null);
    try {
      setRes(await api.predictBatch(id, file));
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  const download = () => {
    const rows = res.results.map((r) => ({
      prediction_id: r.prediction_id, prediction: r.prediction, confidence: r.confidence,
      trustworthiness: r.trustworthiness, decision: r.decision, risk_level: r.risk_level, actual: r.actual ?? "",
      warnings: r.warnings.map((w) => w.code).join("|"), explanation: r.explanation,
    }));
    const blob = new Blob([toCsv(rows, Object.keys(rows[0]))], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "modeldoctor_predictions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const shown = res ? res.results.filter((r) => filter === "ALL" || r.decision === filter) : [];
  return (
    <Card title="Batch prediction" subtitle="Upload a CSV with the same feature columns. If it also contains the target column, those values are stored as real outcomes for monitoring (never used to predict).">
      <label className="flex cursor-pointer items-center justify-center rounded-lg border-2 border-dashed border-slate-300 px-4 py-6 text-sm text-slate-600 hover:border-indigo-400">
        {busy ? "Scoring…" : "Choose a CSV file"}
        <input type="file" accept=".csv,text/csv" className="hidden" disabled={busy}
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      </label>
      <div className="mt-3"><ErrorBox message={error} /></div>
      {res && (
        <div className="mt-4 space-y-3">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-medium">{res.count} rows:</span>
            {["ALL", "TRUST", "REVIEW", "REJECT"].map((d) => (
              <button key={d} onClick={() => setFilter(d)}
                className={`rounded-md px-2 py-1 text-xs font-medium ${filter === d ? "bg-slate-900 text-white" : "bg-slate-100 text-slate-700 hover:bg-slate-200"}`}>
                {d === "ALL" ? "All" : `${d} ${res.summary[d]}`}
              </button>
            ))}
            {res.summary.labels_recorded > 0 && <span className="text-xs text-emerald-700">{res.summary.labels_recorded} outcomes recorded for monitoring</span>}
            <Button variant="secondary" className="ml-auto" onClick={download}>Download CSV</Button>
          </div>
          {res.ignored_columns.length > 0 && <div className="text-xs text-slate-500">Ignored columns: {res.ignored_columns.join(", ")}</div>}
          <div className="max-h-[28rem] overflow-auto">
            <Table
              rows={shown.slice(0, 500)}
              rowKey={(r) => r.prediction_id}
              columns={[
                { key: "i", label: "#", render: (r) => res.results.indexOf(r) + 1 },
                { key: "prediction", label: "Prediction" },
                { key: "actual", label: "Actual", render: (r) => r.actual ?? "—" },
                { key: "confidence", label: "Confidence", align: "right", render: (r) => pct(r.confidence) },
                { key: "trustworthiness", label: "Trust", align: "right", render: (r) => pct(r.trustworthiness) },
                { key: "decision", label: "Decision", render: (r) => (
                  <span className="inline-flex items-center gap-2"><TrafficLight decision={r.decision} size="sm" /><DecisionBadge decision={r.decision} /></span>
                ) },
                { key: "risk", label: "Risk", render: (r) => <RiskBadge level={r.risk_level} /> },
                { key: "warnings", label: "Warnings", render: (r) => <span className="text-xs text-slate-500">{r.warnings.map((w) => w.code).join(", ") || "—"}</span> },
                { key: "why", label: "", render: (r) => (
                  <button type="button" onClick={() => setSelected(r)} aria-label={`Explain row ${res.results.indexOf(r) + 1}`}
                    className={`text-xs font-medium underline ${selected === r ? "text-slate-900" : "text-indigo-600"}`}>Why?</button>
                ) },
              ]}
            />
            {shown.length > 500 && <div className="p-2 text-xs text-slate-500">Showing first 500 rows; download the CSV for all.</div>}
          </div>
          {selected && (
            <div className="rounded-lg border border-slate-200 p-4">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="text-sm">
                  Row {res.results.indexOf(selected) + 1}: predicted <b>{selected.prediction}</b> · confidence {pct(selected.confidence)}
                  {" "}· <RiskBadge level={selected.risk_level} />
                </div>
                <DecisionSummary result={selected} />
              </div>
              <ExplanationPanel result={selected} />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

export default function PredictPage() {
  const { id } = useParams();
  const { loading, error, data: schema, reload } = useLoader(() => api.schema(id), [id]);
  const [values, setValues] = React.useState({});
  const [result, setResult] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [predError, setPredError] = React.useState(null);
  const [fieldErrors, setFieldErrors] = React.useState({});

  React.useEffect(() => {
    if (schema) setValues(typicalValues(schema.features));
  }, [schema]);

  if (loading) return <Spinner label="Loading input schema…" />;
  if (error)
    return (
      <div className="space-y-3">
        <ErrorBox message={errorMessage(error)} onRetry={reload} />
        <Link className="text-indigo-600 underline" to={`/s/${id}/train`}>See training status</Link>
      </div>
    );

  const submit = async (e) => {
    e.preventDefault();
    const errors = validateValues(schema.features, values);
    setFieldErrors(errors);
    if (Object.keys(errors).length) {
      setPredError("Fix the highlighted fields first.");
      return;
    }
    setBusy(true);
    setPredError(null);
    try {
      const features = Object.fromEntries(
        schema.features.map((f) => {
          const v = values[f.name];
          if (v === "" || v === undefined) return [f.name, null];
          return [f.name, f.type === "numeric" ? Number(v) : v];
        })
      );
      setResult(await api.predict(id, features));
    } catch (err) {
      setPredError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Predict</h1>
        <p className="mt-1 text-sm text-slate-600">
          Enter a new record. The true outcome is unknown here. ModelDoctor reports Model 1's claimed confidence next to its
          own estimate of how likely the prediction is to be correct.
        </p>
      </div>
      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3" title="New record" subtitle={`${schema.features.length} features · ranges and options come from the train split`}
          actions={
            <div className="flex gap-2">
              <Button type="button" variant="secondary" onClick={() => setValues(typicalValues(schema.features))}>Typical values</Button>
              <Button type="button" variant="secondary" onClick={() => setValues(Object.fromEntries(schema.features.map((f) => [f.name, ""])))}>Clear</Button>
            </div>
          }>
          <form onSubmit={submit} className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              {schema.features.map((f) => (
                <Field key={f.name} spec={f} value={values[f.name] ?? ""} error={fieldErrors[f.name]}
                  onChange={(v) => setValues((s) => ({ ...s, [f.name]: v }))} />
              ))}
            </div>
            <ErrorBox message={predError} />
            <Button type="submit" disabled={busy} className="w-full sm:w-auto">{busy ? "Scoring…" : "Predict"}</Button>
          </form>
        </Card>
        <div className="lg:col-span-2">
          {result ? (
            <ResultCard result={result} target={schema.target_column} />
          ) : (
            <Card>
              <Empty title="No prediction yet">Fill the form and press Predict.</Empty>
              <div className="mt-4">
                <Notice tone="info">
                  <b>Confidence vs trustworthiness.</b> Confidence is what the model claims. Trustworthiness is learned from
                  Model 1's past mistakes on data it never trained on, using signals such as model disagreement, distance
                  from the training data and missing or out-of-range inputs.
                </Notice>
              </div>
            </Card>
          )}
        </div>
      </div>
      <BatchPanel id={id} />
    </div>
  );
}
