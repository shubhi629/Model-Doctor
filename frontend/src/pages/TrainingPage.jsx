import React from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, errorMessage } from "../api/client.js";
import { Button, Card, ErrorBox, Notice, Spinner } from "../components/ui.jsx";

const STAGE_INFO = {
  profiling: ["Profiling", "Rows, types, missing values, class balance"],
  splitting: ["Three-way split", "Stratified 60% train / 20% calibration / 20% test"],
  model_selection: ["Model 1 selection", "Cross-validate 5 candidates on the train split, calibrate the winner"],
  reliability_features: ["Reliability features", "Bootstrap copies, kNN distance, Isolation Forest, raw input checks"],
  reliability_training: ["Model 2 & thresholds", "Learn P(correct) on calibration data; tune TRUST / REJECT cut-offs out-of-fold"],
  evaluation: ["Evaluation", "Held-out test split: ModelDoctor vs confidence baseline"],
  saving: ["Saving", "Models, thresholds and report saved under the session"],
};

export default function TrainingPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [status, setStatus] = React.useState(null);
  const [error, setError] = React.useState(null);

  React.useEffect(() => {
    let timer;
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.status(id);
        if (cancelled) return;
        setStatus(s);
        setError(null);
        if (s.status === "pending" || s.status === "running") timer = setTimeout(poll, 2000);
      } catch (e) {
        if (cancelled) return;
        setError(errorMessage(e));
        if (e?.response?.status !== 404) timer = setTimeout(poll, 4000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [id]);

  React.useEffect(() => {
    if (status?.status === "completed") {
      const t = setTimeout(() => navigate(`/s/${id}/report`), 1200);
      return () => clearTimeout(t);
    }
  }, [status, id, navigate]);

  if (!status && !error) return <Spinner label="Checking training status…" />;
  if (!status) return <ErrorBox message={error} />;

  const stages = status.stages || Object.keys(STAGE_INFO);
  const currentIdx = status.status === "completed" ? stages.length : Math.max(0, stages.indexOf(status.stage));
  const progress = status.status === "completed" ? 1 : status.progress ?? 0;

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Training</h1>
        <p className="mt-1 text-sm text-slate-600">Session <code className="text-xs">{id}</code></p>
      </div>
      <ErrorBox message={error} />

      {status.status === "uploaded" && (
        <Notice tone="info">
          Training has not been started for this session. <Link className="underline" to="/">Go to upload</Link> to choose a target.
        </Notice>
      )}

      <Card>
        <div className="mb-5">
          <div className="mb-1 flex justify-between text-sm">
            <span className="font-medium capitalize text-slate-700">{status.status}</span>
            <span className="tabular text-slate-500">{Math.round(progress * 100)}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className={`h-full rounded-full transition-all duration-700 ${status.status === "failed" ? "bg-rose-500" : "bg-indigo-600"}`}
              style={{ width: `${Math.max(progress * 100, status.status === "running" ? 4 : 0)}%` }}
            />
          </div>
        </div>

        <ol className="space-y-3">
          {stages.map((s, i) => {
            const done = i < currentIdx;
            const active = i === currentIdx && (status.status === "running" || status.status === "pending");
            const failed = status.status === "failed" && s === status.stage;
            const [label, desc] = STAGE_INFO[s] || [s, ""];
            return (
              <li key={s} className="flex items-start gap-3">
                <span
                  className={`mt-0.5 flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                    failed ? "bg-rose-100 text-rose-700" : done ? "bg-emerald-100 text-emerald-700" : active ? "bg-indigo-100 text-indigo-700" : "bg-slate-100 text-slate-400"
                  }`}
                >
                  {failed ? "!" : done ? "✓" : active ? <span className="h-3 w-3 animate-spin rounded-full border-2 border-indigo-300 border-t-indigo-700" /> : i + 1}
                </span>
                <div>
                  <div className={`text-sm font-medium ${done || active ? "text-slate-900" : "text-slate-500"}`}>{label}</div>
                  <div className="text-xs text-slate-500">{active && status.message ? status.message : desc}</div>
                </div>
              </li>
            );
          })}
        </ol>
      </Card>

      {status.status === "failed" && (
        <Notice tone="bad">
          <div className="font-medium">Training failed</div>
          <div className="mt-1">{status.error}</div>
          <Link to="/" className="mt-2 inline-block font-medium underline">Back to upload</Link>
        </Notice>
      )}
      {status.status === "completed" && (
        <Notice tone="good">
          Training complete. Opening the report… <Button className="ml-2" onClick={() => navigate(`/s/${id}/report`)}>View report</Button>
        </Notice>
      )}
    </div>
  );
}
