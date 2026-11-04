import { describe, expect, it } from "vitest";
import { summarizeMetrics } from "@/lib/experiments";
import { parseFeatureCsv } from "@/pages/EstimatePage";
import type { MetricRow } from "@/types/experiments";

describe("recorded comparison aggregation", () => {
  it("averages training-seed errors without counting seeds as new cases", () => {
    const base = { method: "fedavg", scenario: "iid", split: "test", groups: 4, count: 100 };
    const rows = [{ ...base, seed: 11, mae: .1, rmse: .2 }, { ...base, seed: 22, mae: .3, rmse: .4 }] as MetricRow[];
    const [result] = summarizeMetrics(rows);
    expect(result.mae).toBeCloseTo(.2);
    expect(result.rmse).toBeCloseTo(.3);
    expect(result.groups).toBe(4);
    expect(result.count).toBe(100);
    expect(result.seeds).toBe(2);
  });
});

describe("feature CSV input", () => {
  it("reads signed/scientific numbers in the declared feature order", () => {
    expect(parseFeatureCsv('a,b\n-1.5,2e-3\n', ["a", "b"])).toEqual([[-1.5, .002]]);
  });
  it.each(['b,a\n1,2', 'a,b,wear_target\n1,2,.2', 'a,a\n1,2', 'a,b\n,2', 'a,b\nNaN,2', 'a,b\nInfinity,2'])
    ("rejects mismatched or nonfinite input: %s", (text) => {
      expect(() => parseFeatureCsv(text, ["a", "b"])).toThrow();
    });
});
