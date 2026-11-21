import React from "react";
import { pct } from "./ui.jsx";

/*
 * Presentation of backend decisions. Nothing here decides anything: the decision, risk level,
 * explanation, reasons and warning categories all come from the API response as-is.
 */

export const LIGHTS = [
  { decision: "TRUST", color: "green", on: "bg-emerald-500 ring-4 ring-emerald-400/40", label: "Trust", meaning: "Accepted automatically" },
  { decision: "REVIEW", color: "yellow", on: "bg-amber-400 ring-4 ring-amber-300/40", label: "Review", meaning: "Needs human verification" },
  { decision: "REJECT", color: "red", on: "bg-rose-500 ring-4 ring-rose-400/40", label: "Reject", meaning: "ModelDoctor abstains" },
];

export function TrafficLight({ decision, size = "md" }) {
  const lamp = size === "sm" ? "h-3 w-3" : "h-6 w-6";
  const active = LIGHTS.find((l) => l.decision === decision);
  return (
    <div
      role="img"
      aria-label={active ? `Decision ${decision}: ${active.color} light, ${active.meaning.toLowerCase()}` : "No decision"}
      data-decision={decision}
      className={`inline-flex items-center gap-1.5 rounded-full bg-slate-800 ${size === "sm" ? "px-1.5 py-1" : "px-2.5 py-2"}`}
    >
      {LIGHTS.map((l) => (
        <span
          key={l.decision}
          data-lit={l.decision === decision}
          className={`${lamp} rounded-full ${l.decision === decision ? l.on : "bg-slate-600"}`}
        />
      ))}
    </div>
  );
}

const RISK_STYLES = {
  LOW: "bg-emerald-50 text-emerald-800 ring-emerald-600/20",
  MEDIUM: "bg-amber-50 text-amber-800 ring-amber-600/20",
  HIGH: "bg-rose-50 text-rose-800 ring-rose-600/20",
};

export function RiskBadge({ level }) {
  if (!level) return null;
  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${RISK_STYLES[level] || ""}`}>
      {level} risk
    </span>
  );
}

// The four groups the explanation panel always shows, then any other group that has warnings.
const CORE_GROUPS = [
  ["out_of_distribution", "Out-of-distribution", "Values outside the training data, unseen categories, unusual distance"],
  ["anomaly", "Anomalies", "Flagged by the anomaly detector"],
  ["missing_information", "Missing information", "Blank or null inputs"],
  ["model_disagreement", "Model disagreement", "Bootstrap copies of Model 1 disagree"],
];
const EXTRA_GROUPS = {
  uncertainty: "Model uncertainty",
  low_trustworthiness: "Low trustworthiness",
  other: "Other",
};

const HEADINGS = {
  TRUST: "Why this prediction is trusted",
  REVIEW: "Why human review is needed",
  REJECT: "Why ModelDoctor abstains",
};

export function WarningGroups({ warnings }) {
  const byCat = {};
  for (const w of warnings || []) (byCat[w.category || "other"] ||= []).push(w);
  const extras = Object.keys(byCat).filter((c) => !CORE_GROUPS.some(([k]) => k === c));
  const groups = [...CORE_GROUPS, ...extras.map((c) => [c, EXTRA_GROUPS[c] || c, ""])];
  return (
    <dl className="grid gap-2 sm:grid-cols-2">
      {groups.map(([key, label, hint]) => {
        const items = byCat[key] || [];
        return (
          <div key={key} data-category={key}
            className={`rounded-lg border p-2.5 ${items.length ? "border-amber-200 bg-amber-50/60" : "border-slate-200"}`}>
            <dt className="flex items-center justify-between gap-2 text-xs font-semibold uppercase tracking-wide text-slate-600">
              <span title={hint}>{label}</span>
              <span className={items.length ? "text-amber-700" : "text-slate-400"}>{items.length ? items.length : "none"}</span>
            </dt>
            {items.length > 0 && (
              <dd className="mt-1 space-y-1">
                {items.map((w, i) => (
                  <div key={i} className="text-sm text-slate-800">
                    <code className="mr-1 rounded bg-white px-1 py-0.5 text-[10px] font-medium text-amber-800 ring-1 ring-amber-200">{w.code}</code>
                    {w.message}
                  </div>
                ))}
              </dd>
            )}
          </div>
        );
      })}
    </dl>
  );
}

export function ExplanationPanel({ result }) {
  return (
    <div className="space-y-4" data-testid="explanation">
      <div>
        <div className="mb-1 text-sm font-semibold text-slate-800">{HEADINGS[result.decision] || "Explanation"}</div>
        <p className="text-sm text-slate-700">{result.explanation}</p>
        {result.reasons?.length > 0 && (
          <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
            {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        )}
      </div>
      <div>
        <div className="mb-2 text-sm font-semibold text-slate-800">Detected signals</div>
        <WarningGroups warnings={result.warnings} />
      </div>
    </div>
  );
}

export function DecisionSummary({ result }) {
  const light = LIGHTS.find((l) => l.decision === result.decision);
  return (
    <div className="flex items-center gap-3">
      <TrafficLight decision={result.decision} />
      <div>
        <div className="text-lg font-semibold leading-tight">{result.decision}</div>
        <div className="text-xs text-slate-500">
          {light?.meaning} · trustworthiness {pct(result.trustworthiness)}
        </div>
      </div>
    </div>
  );
}
