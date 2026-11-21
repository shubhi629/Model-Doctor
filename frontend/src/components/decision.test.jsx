import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { ExplanationPanel, TrafficLight } from "./decision.jsx";

describe("TrafficLight", () => {
  it.each([
    ["TRUST", "green"],
    ["REVIEW", "yellow"],
    ["REJECT", "red"],
  ])("lights only the lamp for the backend decision %s", (decision, color) => {
    render(<TrafficLight decision={decision} />);
    const light = screen.getByRole("img");
    expect(light.getAttribute("data-decision")).toBe(decision);
    expect(light.getAttribute("aria-label")).toContain(`${color} light`);
    const lit = [...light.querySelectorAll("span")].filter((s) => s.getAttribute("data-lit") === "true");
    expect(lit).toHaveLength(1);
  });

  it("lights nothing for an unknown decision", () => {
    render(<TrafficLight decision={undefined} />);
    const light = screen.getByRole("img", { name: "No decision" });
    expect(light.querySelectorAll('[data-lit="true"]')).toHaveLength(0);
  });
});

describe("ExplanationPanel", () => {
  const result = {
    decision: "REVIEW",
    trustworthiness: 0.71,
    explanation: "Needs human review: a person should verify this prediction before it is used.",
    reasons: ["Trustworthiness 71.0% is below the trust threshold 85.0%.", "Field 'age' is missing."],
    warnings: [
      { code: "MISSING_FIELD", message: "field 'age' is missing", category: "missing_information" },
      { code: "OUT_OF_RANGE", message: "'income' is outside the training range", category: "out_of_distribution" },
      { code: "LOW_CONFIDENCE", message: "primary model confidence is low (55.0%)", category: "uncertainty" },
    ],
  };

  it("shows the backend explanation and reasons verbatim", () => {
    render(<ExplanationPanel result={result} />);
    expect(screen.getByText("Why human review is needed")).toBeTruthy();
    expect(screen.getByText(result.explanation)).toBeTruthy();
    for (const r of result.reasons) expect(screen.getByText(r)).toBeTruthy();
  });

  it("groups warnings by their backend category and marks empty groups", () => {
    const { container } = render(<ExplanationPanel result={result} />);
    const group = (cat) => container.querySelector(`[data-category="${cat}"]`);
    expect(within(group("missing_information")).getByText("MISSING_FIELD")).toBeTruthy();
    expect(within(group("out_of_distribution")).getByText("OUT_OF_RANGE")).toBeTruthy();
    expect(within(group("uncertainty")).getByText("LOW_CONFIDENCE")).toBeTruthy();
    expect(within(group("anomaly")).getByText("none")).toBeTruthy();
    expect(within(group("model_disagreement")).getByText("none")).toBeTruthy();
  });

  it("uses the abstention heading for REJECT", () => {
    render(<ExplanationPanel result={{ ...result, decision: "REJECT" }} />);
    expect(screen.getByText("Why ModelDoctor abstains")).toBeTruthy();
  });
});
