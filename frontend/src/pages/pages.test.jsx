import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";

vi.mock("../api/client.js", async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    api: { schema: vi.fn(), predict: vi.fn(), predictBatch: vi.fn(), monitoring: vi.fn(), activateModel: vi.fn() },
  };
});

import { api } from "../api/client.js";
import PredictPage, { validateValues } from "./PredictPage.jsx";
import OverviewPage from "./OverviewPage.jsx";

const SCHEMA = {
  session_id: "s1",
  target_column: "default",
  classes: ["0", "1"],
  features: [
    { name: "age", type: "numeric", missing_rate: 0, integer: true, min: 18, max: 80, median: 40 },
    { name: "income", type: "numeric", missing_rate: 0, integer: false, min: 0, max: 1e6, median: 50000 },
    { name: "job", type: "categorical", missing_rate: 0, categories: ["a", "b"], most_frequent: "a", n_categories: 2 },
  ],
};

const REJECTED = {
  prediction_id: "p1",
  prediction: "1",
  confidence: 0.62,
  trustworthiness: 0.31,
  decision: "REJECT",
  risk_level: "HIGH",
  explanation: "Rejected: ModelDoctor abstains because this prediction is too likely to be wrong to act on.",
  reasons: ["Trustworthiness 31.0% is below the reject threshold 40.0%."],
  warnings: [{ code: "MODEL_DISAGREEMENT", message: "model copies disagree strongly (40% disagreement)", category: "model_disagreement" }],
  reliability_mode: "model2",
  class_probabilities: { 0: 0.38, 1: 0.62 },
  reliability_features: {},
};

function renderAt(path, element, route) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes><Route path={route} element={element} /></Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => vi.clearAllMocks());

describe("PredictPage", () => {
  it("sends typed values, blank as null, and shows the backend's decision", async () => {
    api.schema.mockResolvedValue(SCHEMA);
    api.predict.mockResolvedValue(REJECTED);
    renderAt("/s/s1/predict", <PredictPage />, "/s/:id/predict");

    const income = await screen.findByLabelText(/income/);
    fireEvent.change(income, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Predict" }));

    await waitFor(() => expect(api.predict).toHaveBeenCalledTimes(1));
    expect(api.predict).toHaveBeenCalledWith("s1", { age: 40, income: null, job: "a" });

    expect(await screen.findByText(REJECTED.explanation)).toBeTruthy();
    expect(screen.getByRole("img", { name: /Decision REJECT: red light/ })).toBeTruthy();
    expect(screen.getByText("HIGH risk")).toBeTruthy();
    expect(screen.getByText("Why ModelDoctor abstains")).toBeTruthy();
    expect(screen.getByText(REJECTED.reasons[0])).toBeTruthy();
  });

  it("shows the API error and keeps the form usable", async () => {
    api.schema.mockResolvedValue(SCHEMA);
    api.predict.mockRejectedValue({ response: { status: 422, data: { detail: "Field 'age' must be numeric", code: "validation_error" } } });
    renderAt("/s/s1/predict", <PredictPage />, "/s/:id/predict");
    fireEvent.click(await screen.findByRole("button", { name: "Predict" }));
    expect(await screen.findByText("Field 'age' must be numeric")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Predict" }).disabled).toBe(false);
  });

  it("validates fields before sending", () => {
    expect(validateValues(SCHEMA.features, { age: "abc", income: "", job: "x" })).toEqual({ age: "must be a finite number" });
    expect(validateValues(SCHEMA.features, { age: "Infinity", income: "1", job: "a".repeat(1001) }))
      .toEqual({ age: "must be a finite number", job: "at most 1000 characters" });
  });
});

describe("OverviewPage", () => {
  const MON = {
    session_id: "s1",
    total_predictions: 12,
    labeled_predictions: 0,
    coverage: 0.5,
    mean_confidence: 0.8123,
    mean_trustworthiness: 0.7456,
    anomaly_rate: 0.25,
    decision_distribution: [
      { decision: "TRUST", count: 6, share: 0.5 },
      { decision: "REVIEW", count: 4, share: 0.3333 },
      { decision: "REJECT", count: 2, share: 0.1667 },
    ],
    confidence_histogram: [], trustworthiness_histogram: [], predicted_class_distribution: [{ class: "0", count: 12 }],
    source_distribution: { single: 2, api: 10 }, timeline: [], baseline: { test_coverage: 0.36 },
    first_prediction_at: "2026-01-01T00:00:00+00:00", last_prediction_at: "2026-01-01T01:00:00+00:00",
    observed_vs_estimated: { bins: [] }, alerts: [], retraining_recommended: false, recent_predictions: [],
  };

  it("renders KPIs straight from the monitoring API", async () => {
    api.monitoring.mockResolvedValue(MON);
    renderAt("/s/s1/overview", <OverviewPage />, "/s/:id/overview");
    expect(await screen.findByText("Total predictions")).toBeTruthy();
    expect(screen.getByText("81.2%")).toBeTruthy();
    expect(screen.getByText("74.6%")).toBeTruthy();
    expect(screen.getByText("6")).toBeTruthy();
    expect(screen.getByText("API /v1/predict")).toBeTruthy();
    expect(api.monitoring).toHaveBeenCalledWith("s1", { recent: 0 });
  });

  it("shows an empty state when nothing has been predicted", async () => {
    api.monitoring.mockResolvedValue({ ...MON, total_predictions: 0 });
    renderAt("/s/s1/overview", <OverviewPage />, "/s/:id/overview");
    expect(await screen.findByText("No predictions served yet")).toBeTruthy();
  });

  it("shows an error with retry when the backend is unreachable", async () => {
    api.monitoring.mockRejectedValue({ request: {} });
    renderAt("/s/s1/overview", <OverviewPage />, "/s/:id/overview");
    expect(await screen.findByText(/Cannot reach the ModelDoctor API/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });
});
