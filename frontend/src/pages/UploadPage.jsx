import React from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, errorMessage } from "../api/client.js";
import { Button, Card, Empty, ErrorBox, Notice, Spinner, Table, pct, useLoader } from "../components/ui.jsx";

const MODELS = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "KNN"];

function Dropzone({ onFile, busy }) {
  const [over, setOver] = React.useState(false);
  const inputRef = React.useRef(null);
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition ${
        over ? "border-indigo-500 bg-indigo-50" : "border-slate-300 bg-white hover:border-indigo-400"
      }`}
    >
      <svg className="h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
      </svg>
      <div className="mt-3 font-medium text-slate-800">{busy ? "Uploading…" : "Drop a CSV here, or click to browse"}</div>
      <div className="mt-1 text-sm text-slate-500">Tabular classification data · at least 500 rows · up to 50 MB</div>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function ProfilePanel({ profile }) {
  if (!profile) return null;
  const classes = Object.entries(profile.class_distribution || {});
  const total = classes.reduce((a, [, v]) => a + v, 0) || 1;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Mini label="Rows" value={profile.row_count.toLocaleString()} />
        <Mini label="Features" value={profile.feature_count} />
        <Mini label="Task" value={profile.detected_task ? profile.detected_task.replace("_classification", "") : "—"} />
        <Mini label="Imbalance" value={profile.imbalance_ratio ? `${profile.imbalance_ratio}×` : "—"} />
      </div>
      {profile.valid ? (
        classes.length > 0 && (
          <div>
            <div className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">Class balance</div>
            <div className="flex h-6 w-full overflow-hidden rounded-md">
              {classes.map(([k, v], i) => (
                <div
                  key={k}
                  title={`${k}: ${v}`}
                  style={{ width: `${(v / total) * 100}%`, background: ["#4f46e5", "#0ea5e9", "#14b8a6", "#f59e0b", "#ec4899", "#64748b"][i % 6] }}
                  className="flex items-center justify-center text-[11px] font-medium text-white"
                >
                  {v / total > 0.08 ? `${k} · ${pct(v / total, 0)}` : ""}
                </div>
              ))}
            </div>
          </div>
        )
      ) : (
        <Notice tone="bad">{profile.validation_error}</Notice>
      )}
      {profile.warnings?.length > 0 && (
        <Notice tone="warn">
          <div className="mb-1 font-medium">Data quality warnings</div>
          <ul className="list-disc space-y-0.5 pl-5">
            {profile.warnings.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
        </Notice>
      )}
    </div>
  );
}

function Mini({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="tabular mt-0.5 font-semibold capitalize text-slate-900">{value}</div>
    </div>
  );
}

function RecentSessions() {
  const { loading, error, data, reload } = useLoader(() => api.listSessions(), []);
  const remove = async (id) => {
    if (!window.confirm("Delete this session and all its saved models?")) return;
    try {
      await api.deleteSession(id);
      reload();
    } catch (e) {
      alert(errorMessage(e));
    }
  };
  return (
    <Card title="Recent sessions" subtitle="Saved under backend/storage/{session_id}">
      {loading ? (
        <Spinner />
      ) : error ? (
        <ErrorBox message={errorMessage(error)} onRetry={reload} />
      ) : !data?.length ? (
        <Empty title="No sessions yet">Upload a dataset to create one.</Empty>
      ) : (
        <Table
          rowKey={(r) => r.session_id}
          rows={data}
          columns={[
            { key: "filename", label: "Dataset" },
            { key: "target_column", label: "Target" },
            { key: "rows", label: "Rows", align: "right", render: (r) => r.rows?.toLocaleString() ?? "—" },
            {
              key: "status",
              label: "Status",
              render: (r) => (
                <span
                  className={`rounded px-2 py-0.5 text-xs font-medium ${
                    r.status === "completed"
                      ? "bg-emerald-50 text-emerald-700"
                      : r.status === "failed"
                      ? "bg-rose-50 text-rose-700"
                      : "bg-slate-100 text-slate-700"
                  }`}
                >
                  {r.status}
                </span>
              ),
            },
            { key: "created_at", label: "Created", render: (r) => (r.created_at ? new Date(r.created_at).toLocaleString() : "—") },
            {
              key: "actions",
              label: "",
              render: (r) => (
                <div className="flex justify-end gap-3 text-sm">
                  {r.status === "completed" ? (
                    <>
                      <Link className="text-indigo-600 hover:underline" to={`/s/${r.session_id}/report`}>Report</Link>
                      <Link className="text-indigo-600 hover:underline" to={`/s/${r.session_id}/predict`}>Predict</Link>
                    </>
                  ) : (
                    <Link className="text-indigo-600 hover:underline" to={`/s/${r.session_id}/train`}>Open</Link>
                  )}
                  <button className="text-rose-600 hover:underline" onClick={() => remove(r.session_id)}>Delete</button>
                </div>
              ),
            },
          ]}
        />
      )}
    </Card>
  );
}

export default function UploadPage() {
  const navigate = useNavigate();
  const [upload, setUpload] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const [target, setTarget] = React.useState("");
  const [profile, setProfile] = React.useState(null);
  const [profileLoading, setProfileLoading] = React.useState(false);
  const [opts, setOpts] = React.useState({ target_risk: 5, min_coverage: 30, force_model: "", fast: false, force_review_on_anomalies: false });
  const [starting, setStarting] = React.useState(false);

  const onFile = async (file) => {
    setBusy(true);
    setError(null);
    setUpload(null);
    setProfile(null);
    try {
      const res = await api.upload(file);
      setUpload(res);
      setTarget(res.suggested_target || res.columns[res.columns.length - 1].name);
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => {
    if (!upload || !target) return;
    let cancelled = false;
    setProfileLoading(true);
    api
      .profile(upload.session_id, target)
      .then((p) => !cancelled && setProfile(p))
      .catch((e) => !cancelled && setProfile({ valid: false, validation_error: errorMessage(e), row_count: upload.rows }))
      .finally(() => !cancelled && setProfileLoading(false));
    return () => {
      cancelled = true;
    };
  }, [upload, target]);

  const start = async () => {
    setStarting(true);
    setError(null);
    try {
      await api.train(upload.session_id, {
        target_column: target,
        target_risk: Number(opts.target_risk) / 100,
        min_coverage: Number(opts.min_coverage) / 100,
        force_model: opts.force_model || null,
        fast: opts.fast,
        force_review_on_anomalies: opts.force_review_on_anomalies,
      });
      navigate(`/s/${upload.session_id}/train`);
    } catch (e) {
      setError(errorMessage(e));
      setStarting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Upload a dataset</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-600">
          ModelDoctor picks a primary model (Model 1) by cross-validation, then trains a second model (the doctor) on
          data Model 1 never saw, to estimate how likely each individual prediction is to be correct.
        </p>
      </div>

      <Dropzone onFile={onFile} busy={busy} />
      <ErrorBox message={error} />

      {upload && (
        <div className="grid gap-6 lg:grid-cols-3">
          <Card
            className="lg:col-span-2"
            title={upload.filename}
            subtitle={`${upload.rows.toLocaleString()} rows · ${upload.columns.length} columns · first 10 rows shown`}
          >
            <div className="max-h-96 overflow-auto">
              <table className="min-w-full text-xs">
                <thead className="sticky top-0 bg-white">
                  <tr className="border-b border-slate-200">
                    {upload.columns.map((c) => (
                      <th key={c.name} className={`whitespace-nowrap px-2 py-2 text-left font-medium ${c.name === target ? "text-indigo-700" : "text-slate-600"}`}>
                        {c.name}
                        <div className="font-normal text-slate-400">{c.dtype}{c.missing_count ? ` · ${c.missing_count} missing` : ""}</div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {upload.preview.map((row, i) => (
                    <tr key={i}>
                      {upload.columns.map((c) => (
                        <td key={c.name} className={`whitespace-nowrap px-2 py-1.5 ${c.name === target ? "bg-indigo-50/60 font-medium" : ""}`}>
                          {row[c.name] === null ? <span className="text-slate-300">null</span> : String(row[c.name])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="Target & training" subtitle="Choose what to predict">
            <div className="space-y-4">
              <label className="block">
                <span className="text-sm font-medium text-slate-700">Target column</span>
                <select
                  value={target}
                  onChange={(e) => setTarget(e.target.value)}
                  className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                >
                  {upload.columns.map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name} {c.is_target_candidate ? "" : "(unlikely target)"}
                    </option>
                  ))}
                </select>
              </label>

              {profileLoading ? <Spinner label="Profiling…" /> : <ProfilePanel profile={profile} />}

              <details className="rounded-lg border border-slate-200 p-3 text-sm">
                <summary className="cursor-pointer font-medium text-slate-700">Advanced options</summary>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <label className="block">
                    <span className="text-xs text-slate-600">Target risk (max error among TRUST, %)</span>
                    <input type="number" min="0.5" max="50" step="0.5" value={opts.target_risk}
                      onChange={(e) => setOpts({ ...opts, target_risk: e.target.value })}
                      className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1" />
                  </label>
                  <label className="block">
                    <span className="text-xs text-slate-600">Minimum coverage (%)</span>
                    <input type="number" min="1" max="100" step="1" value={opts.min_coverage}
                      onChange={(e) => setOpts({ ...opts, min_coverage: e.target.value })}
                      className="mt-1 w-full rounded-md border border-slate-300 px-2 py-1" />
                  </label>
                  <label className="col-span-2 block">
                    <span className="text-xs text-slate-600">Model 1</span>
                    <select value={opts.force_model} onChange={(e) => setOpts({ ...opts, force_model: e.target.value })}
                      className="mt-1 w-full rounded-md border border-slate-300 bg-white px-2 py-1">
                      <option value="">Automatic (best cross-validation score)</option>
                      {MODELS.map((m) => <option key={m} value={m}>Force {m}</option>)}
                    </select>
                  </label>
                  <label className="col-span-2 flex items-center gap-2">
                    <input type="checkbox" checked={opts.fast} onChange={(e) => setOpts({ ...opts, fast: e.target.checked })} />
                    <span className="text-xs text-slate-600">Fast mode (smaller ensembles, for quick demos)</span>
                  </label>
                  <label className="col-span-2 flex items-center gap-2">
                    <input type="checkbox" checked={opts.force_review_on_anomalies}
                      onChange={(e) => setOpts({ ...opts, force_review_on_anomalies: e.target.checked })} />
                    <span className="text-xs text-slate-600">Always REVIEW inputs with out-of-range / unseen values</span>
                  </label>
                </div>
              </details>

              <Button className="w-full" onClick={start} disabled={!profile?.valid || starting || profileLoading}>
                {starting ? "Starting…" : "Start training"}
              </Button>
            </div>
          </Card>
        </div>
      )}

      <RecentSessions />
    </div>
  );
}
