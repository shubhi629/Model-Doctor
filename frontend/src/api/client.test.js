import { describe, expect, it } from "vitest";
import { UNREACHABLE, errorMessage } from "./client.js";

describe("errorMessage", () => {
  it("returns the backend detail for client errors", () => {
    const err = { response: { status: 422, data: { detail: "Field 'age' must be numeric", code: "validation_error", request_id: "r1" } } };
    expect(errorMessage(err)).toBe("Field 'age' must be numeric");
  });

  it("adds the request ID to server errors so they can be traced in the logs", () => {
    const err = { response: { status: 503, data: { detail: "The model is unavailable.", code: "model_unavailable", request_id: "abc123" } } };
    expect(errorMessage(err)).toBe("The model is unavailable. (request abc123)");
  });

  it("reports an unreachable backend when there is no response", () => {
    expect(errorMessage({ request: {} })).toBe(UNREACHABLE);
  });

  it("reports an unreachable backend when the dev proxy answers without the API's JSON", () => {
    expect(errorMessage({ request: {}, response: { status: 500, data: "" } })).toBe(UNREACHABLE);
    expect(errorMessage({ request: {}, response: { status: 502, data: {} } })).toBe(UNREACHABLE);
  });

  it("falls back to the status code or message", () => {
    expect(errorMessage({ response: { status: 404, data: {} } })).toBe("Request failed (404)");
    expect(errorMessage(new Error("boom"))).toBe("boom");
  });
});
