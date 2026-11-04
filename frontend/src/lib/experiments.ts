import { useCallback, useEffect, useState } from "react";
import type { ExperimentOption, MetricRow, Route, Study } from "@/types/experiments";

export const methodNames: Record<string, string> = {
  fedavg: "FedAvg", fedprox: "FedProx", centralized_mlp: "Centralized MLP", random_forest: "Random forest",
  mean: "Mean baseline", median: "Median baseline", local_client_0: "Client A only", local_client_1: "Client B only", local_client_2: "Client C only",
};
export const scenarioNames: Record<string, string> = { iid: "Balanced data", condition: "Machining conditions", quantity: "Unequal data volumes", label: "Wear distribution", few_shot: "Limited labels" };
export const routeNames: Record<Route, string> = { overview: "Overview", experiments: "Experiments", clients: "Clients", dataset: "Dataset", estimate: "Wear estimate" };
export const methodName = (name: string) => methodNames[name] ?? name;
export const number = (value: number | null | undefined, digits = 4) => value != null && Number.isFinite(value) ? value.toFixed(digits) : "—";
export const count = (value: number) => new Intl.NumberFormat("en").format(value);
export const experimentName = (id: string) => ({ "nasa-final": "NASA main study", "nasa-smoke": "NASA smoke test", "nasa-efficiency": "NASA efficiency study", "nasa-spectral": "NASA spectral study", "nasa-reproduction": "NASA reproduction" })[id] ?? id;

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const error = await response.json(); if (typeof error.detail === "string") message = error.detail;
      else if (Array.isArray(error.detail)) message = error.detail.map((entry: { msg?: string }) => entry.msg ?? "Invalid input").join("; ");
    } catch { /* HTTP status remains useful. */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export function useStudy() {
  const [options, setOptions] = useState<ExperimentOption[]>([]);
  const [id, setId] = useState("");
  const [study, setStudy] = useState<Study | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const retry = useCallback(() => setRevision((r) => r + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setError("");
    request<{ experiments: ExperimentOption[]; default_id: string | null }>("/api/experiments", { signal: controller.signal })
      .then((result) => { if (controller.signal.aborted) return; setOptions(result.experiments); setId((current) => result.experiments.some((entry) => entry.id === current) ? current : result.default_id ?? ""); if (!result.experiments.length) setLoading(false); })
      .catch((reason: Error) => { if (!controller.signal.aborted) { setError(reason.message); setLoading(false); } });
    return () => controller.abort();
  }, [revision]);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    setLoading(true); setStudy(null); setError("");
    request<Study>(`/api/experiments/${encodeURIComponent(id)}`, { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) { setStudy(result); setLoading(false); } })
      .catch((reason: Error) => { if (!controller.signal.aborted) { setError(reason.message); setLoading(false); } });
    return () => controller.abort();
  }, [id, revision]);
  const matchingStudy = study?.id === id ? study : null;
  return { options, id, setId, study: matchingStudy, error,
    loading: loading || (Boolean(id) && !matchingStudy && !error), retry };
}

export function summarizeMetrics(rows: MetricRow[]) {
  const groups = new Map<string, MetricRow[]>();
  rows.forEach((row) => groups.set(row.method, [...(groups.get(row.method) ?? []), row]));
  return [...groups].map(([method, values]) => ({ method, mae: values.reduce((sum, row) => sum + row.mae, 0) / values.length,
    rmse: values.reduce((sum, row) => sum + row.rmse, 0) / values.length, seeds: new Set(values.map((row) => row.seed)).size,
    count: values[0].count, groups: values[0].groups,
  })).sort((a, b) => a.mae - b.mae);
}
