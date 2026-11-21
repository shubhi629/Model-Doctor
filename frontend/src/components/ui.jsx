import React from "react";

export const pct = (v, digits = 1) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`);
export const num = (v, digits = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));

export function Card({ title, subtitle, actions, children, className = "" }) {
  return (
    <section className={`rounded-xl border border-slate-200 bg-white p-5 shadow-sm ${className}`}>
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-2">
          <div>
            {title && <h2 className="text-base font-semibold text-slate-900">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone = "default" }) {
  const tones = {
    default: "text-slate-900",
    good: "text-emerald-700",
    warn: "text-amber-700",
    bad: "text-rose-700",
    accent: "text-indigo-700",
  };
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`tabular mt-1 text-2xl font-semibold ${tones[tone]}`}>{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-500">{hint}</div>}
    </div>
  );
}

const DECISION_STYLES = {
  TRUST: "bg-emerald-100 text-emerald-800 ring-emerald-600/20",
  REVIEW: "bg-amber-100 text-amber-800 ring-amber-600/20",
  REJECT: "bg-rose-100 text-rose-800 ring-rose-600/20",
};

export const DECISION_COLORS = { TRUST: "#059669", REVIEW: "#d97706", REJECT: "#e11d48" };

export function DecisionBadge({ decision, large = false }) {
  return (
    <span
      className={`inline-flex items-center rounded-md font-semibold ring-1 ring-inset ${
        DECISION_STYLES[decision] || "bg-slate-100 text-slate-700 ring-slate-500/20"
      } ${large ? "px-3 py-1 text-lg" : "px-2 py-0.5 text-xs"}`}
    >
      {decision}
    </span>
  );
}

export function Button({ variant = "primary", className = "", ...props }) {
  const variants = {
    primary: "bg-indigo-600 text-white hover:bg-indigo-500 disabled:bg-indigo-300",
    secondary: "bg-white text-slate-700 ring-1 ring-inset ring-slate-300 hover:bg-slate-50 disabled:text-slate-400",
    danger: "bg-white text-rose-700 ring-1 ring-inset ring-rose-200 hover:bg-rose-50",
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:cursor-not-allowed ${variants[variant]} ${className}`}
      {...props}
    />
  );
}

export function ErrorBox({ message, onRetry }) {
  if (!message) return null;
  return (
    <div role="alert" className="flex items-start justify-between gap-3 rounded-lg border border-rose-200 bg-rose-50 p-3 text-sm text-rose-800">
      <span>{message}</span>
      {onRetry && (
        <button className="shrink-0 font-medium underline" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Notice({ tone = "info", children }) {
  const tones = {
    info: "border-indigo-200 bg-indigo-50 text-indigo-900",
    warn: "border-amber-200 bg-amber-50 text-amber-900",
    good: "border-emerald-200 bg-emerald-50 text-emerald-900",
    bad: "border-rose-200 bg-rose-50 text-rose-900",
  };
  return <div className={`rounded-lg border p-3 text-sm ${tones[tone]}`}>{children}</div>;
}

export function Spinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-3 py-10 text-sm text-slate-500" role="status">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      {label}
    </div>
  );
}

export function Empty({ title, children }) {
  return (
    <div className="rounded-lg border border-dashed border-slate-300 p-8 text-center">
      <div className="font-medium text-slate-700">{title}</div>
      {children && <div className="mt-1 text-sm text-slate-500">{children}</div>}
    </div>
  );
}

export function Table({ columns, rows, rowKey, empty = "No rows" }) {
  if (!rows?.length) return <Empty title={empty} />;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
            {columns.map((c) => (
              <th key={c.key} className={`whitespace-nowrap px-3 py-2 font-medium ${c.align === "right" ? "text-right" : ""}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">
          {rows.map((row, i) => (
            <tr key={rowKey ? rowKey(row, i) : i} className="hover:bg-slate-50">
              {columns.map((c) => (
                <td key={c.key} className={`whitespace-nowrap px-3 py-2 ${c.align === "right" ? "tabular text-right" : ""}`}>
                  {c.render ? c.render(row, i) : row[c.key] ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Small hook for "load once, with loading / error / reload" state. */
export function useLoader(fn, deps) {
  const [state, setState] = React.useState({ loading: true, error: null, data: null });
  const load = React.useCallback(async () => {
    setState((s) => ({ ...s, loading: true, error: null }));
    try {
      const data = await fn();
      setState({ loading: false, error: null, data });
    } catch (e) {
      setState({ loading: false, error: e, data: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  React.useEffect(() => {
    load();
  }, [load]);
  return { ...state, reload: load, setData: (d) => setState((s) => ({ ...s, data: d })) };
}
