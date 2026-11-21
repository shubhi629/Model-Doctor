import React from "react";
import { api, errorMessage } from "../api/client.js";
import { Button } from "./ui.jsx";

const HEALTH_POLL_MS = 30000;

/** Shared /health state, refreshed periodically and on demand (e.g. after deploying a model). */
const HealthContext = React.createContext({ health: null, offline: false, refresh: () => {} });

export function HealthProvider({ children }) {
  const [state, setState] = React.useState({ health: null, offline: false });
  const refresh = React.useCallback(async () => {
    try {
      setState({ health: await api.serviceHealth(), offline: false });
    } catch {
      setState({ health: null, offline: true });
    }
  }, []);
  React.useEffect(() => {
    refresh();
    const t = setInterval(refresh, HEALTH_POLL_MS);
    return () => clearInterval(t);
  }, [refresh]);
  return <HealthContext.Provider value={{ ...state, refresh }}>{children}</HealthContext.Provider>;
}

export const useHealth = () => React.useContext(HealthContext);

export function ApiStatus() {
  const { health, offline, refresh } = useHealth();
  const [tone, text] = offline
    ? ["bg-rose-500", "API offline"]
    : !health
      ? ["bg-slate-300", "Checking API…"]
      : health.status === "degraded"
        ? ["bg-amber-500", "Model unavailable"]
        : ["bg-emerald-500", health.model_loaded ? "API online" : "API online · no model deployed"];
  return (
    <button onClick={refresh} title={health?.detail || (offline ? "Cannot reach the backend on port 8000" : "Click to re-check")}
      className="inline-flex items-center gap-2 rounded-md px-2 py-1 text-xs text-slate-600 hover:bg-slate-100">
      <span className={`h-2 w-2 rounded-full ${tone}`} aria-hidden="true" />
      <span role="status">{text}</span>
    </button>
  );
}

export function OfflineBanner() {
  const { offline } = useHealth();
  if (!offline) return null;
  return (
    <div role="alert" className="border-b border-rose-200 bg-rose-50 px-4 py-2 text-center text-sm text-rose-800">
      Cannot reach the ModelDoctor API. Start the backend (uvicorn on port 8000); this page retries automatically.
    </div>
  );
}

/** Shows whether this session is the model served on /api/v1 and lets the user deploy it. */
export function DeployControl({ sessionId }) {
  const { health, refresh } = useHealth();
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState(null);
  const active = health?.active_model === sessionId && health?.model_loaded;

  const deploy = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.activateModel(sessionId);
      await refresh();
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      {active ? (
        <span className="inline-flex items-center gap-1.5 rounded-md bg-emerald-50 px-2.5 py-1.5 font-medium text-emerald-800 ring-1 ring-inset ring-emerald-600/20">
          <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden="true" /> Served on /api/v1
        </span>
      ) : (
        <Button variant="secondary" onClick={deploy} disabled={busy || !health}>
          {busy ? "Deploying…" : "Serve this model on /api/v1"}
        </Button>
      )}
      {error && <span role="alert" className="text-rose-700">{error}</span>}
    </div>
  );
}
