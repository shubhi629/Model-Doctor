import axios from "axios";

const http = axios.create({ baseURL: import.meta.env.VITE_API_URL || "" });

export const UNREACHABLE = "Cannot reach the ModelDoctor API. Is the backend running on port 8000?";

/** Turn any axios error into a readable message (backend returns {detail, code}). */
export function errorMessage(err) {
  const detail = err?.response?.data?.detail;
  if (detail) {
    const rid = err.response.data.request_id;
    return err.response.status >= 500 && rid && rid !== "-" ? `${detail} (request ${rid})` : String(detail);
  }
  // No response at all, or a gateway error without the backend's JSON body (e.g. the Vite proxy
  // answering because the API is down): the backend is unreachable.
  const gatewayError = err?.response && err.response.status >= 500 && !err.response.data?.code;
  if ((err?.request && !err?.response) || gatewayError) return UNREACHABLE;
  if (err?.response) return `Request failed (${err.response.status})`;
  return err?.message || "Unexpected error";
}

const data = (p) => p.then((r) => r.data);

export const api = {
  health: () => data(http.get("/api/health")),
  serviceHealth: () => data(http.get("/health")),
  modelInfo: () => data(http.get("/api/v1/model/info")),
  activateModel: (sessionId) => data(http.put("/api/v1/model/active", { session_id: sessionId })),
  listSessions: () => data(http.get("/api/sessions")),
  deleteSession: (id) => data(http.delete(`/api/sessions/${id}`)),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file);
    return data(http.post("/api/datasets/upload", form));
  },
  profile: (id, target) => data(http.get(`/api/datasets/${id}/profile`, { params: { target_column: target } })),
  train: (id, body) => data(http.post(`/api/train/${id}`, body)),
  status: (id) => data(http.get(`/api/train/${id}/status`)),
  report: (id) => data(http.get(`/api/report/${id}`)),
  schema: (id) => data(http.get(`/api/schema/${id}`)),
  predict: (id, features) => data(http.post(`/api/predict/${id}`, { features })),
  predictBatch: (id, file) => {
    const form = new FormData();
    form.append("file", file);
    return data(http.post(`/api/predict/${id}/batch`, form));
  },
  feedback: (id, items) => data(http.post(`/api/feedback/${id}`, { items })),
  monitoring: (id, params = {}) => data(http.get(`/api/monitoring/${id}`, { params })),
};
