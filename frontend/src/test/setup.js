import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

// Recharts' ResponsiveContainer needs ResizeObserver, which jsdom does not provide.
globalThis.ResizeObserver ||= class {
  observe() {}
  unobserve() {}
  disconnect() {}
};

afterEach(() => cleanup());
