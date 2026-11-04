import { useEffect, useRef, useState } from "react";
import Papa from "papaparse";
import { ArrowUpRight, FileUp, LoaderCircle, ScanLine, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { count, methodName, number, request } from "@/lib/experiments";
import { InfoNote, SectionHeading, Select } from "@/components/research-controls";
import type { Prediction, Sample, Study } from "@/types/experiments";

export function parseFeatureCsv(text: string, expected: string[]): number[][] {
  const parsed = Papa.parse<string[]>(text, { skipEmptyLines: "greedy" });
  if (parsed.errors.length) throw new Error("This CSV could not be read. Check its separators and quoted values.");
  const [header, ...rows] = parsed.data;
  if (!header || header.length !== expected.length || header.some((name, index) => name.replace(/^\uFEFF/, "") !== expected[index])) {
    throw new Error("The column names and order must match this model’s feature header. Exclude labels and metadata.");
  }
  if (!rows.length || rows.length > 1000) throw new Error("Choose a CSV with 1–1,000 feature rows.");
  return rows.map((row) => {
    if (row.length !== expected.length || row.some((value) => !value.trim() || !Number.isFinite(Number(value)))) throw new Error("Each feature must contain a finite numeric value.");
    return row.map(Number);
  });
}

export function EstimatePage({ study }: { study: Study }) {
  const [model, setModel] = useState(study.models.find((entry) => entry.id === "fedavg")?.id ?? study.models[0]?.id ?? "");
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleId, setSampleId] = useState("");
  const [mode, setMode] = useState<"recorded" | "upload">("recorded");
  const [upload, setUpload] = useState<number[][]>([]);
  const [fileName, setFileName] = useState("");
  const [row, setRow] = useState(0);
  const [explain, setExplain] = useState(true);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [sampleError, setSampleError] = useState("");
  const activeRequest = useRef<AbortController | null>(null);
  const selectedModel = study.models.find((entry) => entry.id === model);
  useEffect(() => {
    const controller = new AbortController();
    request<{ samples: Sample[] }>(`/api/experiments/${encodeURIComponent(study.id)}/samples?limit=200`, { signal: controller.signal })
      .then((data) => { setSamples(data.samples); setSampleId(data.samples[0]?.id ?? ""); })
      .catch((reason: Error) => { if (!controller.signal.aborted) setSampleError(reason.message); });
    return () => controller.abort();
  }, [study.id]);
  useEffect(() => { activeRequest.current?.abort(); setResult(null); setError(""); setPending(false); }, [model, sampleId, mode, upload, row]);
  useEffect(() => () => activeRequest.current?.abort(), []);
  function downloadHeader() {
    const url = URL.createObjectURL(new Blob([selectedModel?.feature_names.join(",") ?? ""], { type: "text/csv" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${model}-feature-header.csv`; anchor.click(); URL.revokeObjectURL(url);
  }
  async function uploadFile(file?: File) {
    if (!file || !selectedModel) return;
    setError(""); setResult(null); setUpload([]);
    try {
      if (file.size > 10 * 1024 * 1024) throw new Error("Choose a feature CSV smaller than 10 MiB.");
      setUpload(parseFeatureCsv(await file.text(), selectedModel.feature_names)); setRow(0); setFileName(file.name);
    } catch (reason) { setError((reason as Error).message); }
  }
  async function predict() {
    if (!model) return;
    const controller = new AbortController(); activeRequest.current?.abort(); activeRequest.current = controller;
    setPending(true); setError(""); setResult(null);
    try {
      const prediction = await request<Prediction>(`/api/experiments/${encodeURIComponent(study.id)}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ model, explain, ...(mode === "recorded" ? { sample_id: sampleId } : { features: upload[row] }) }),
      });
      if (!controller.signal.aborted) setResult(prediction);
    } catch (reason) { if (!controller.signal.aborted) setError((reason as Error).message); }
    finally { if (!controller.signal.aborted) setPending(false); }
  }
  const contributions = result?.explanation?.feature_names.map((name, index) => ({ name, value: result.explanation!.values[index] })).sort((a,b) => Math.abs(b.value)-Math.abs(a.value)).slice(0, 8) ?? [];
  const maxContribution = Math.max(.00001, ...contributions.map((entry) => Math.abs(entry.value)));
  return <><div className="page-heading"><div><h1>Understand one estimate.</h1><p>Use a recorded sample or your own matching features, then inspect the model’s reasoning.</p></div><span className="status-chip"><ScanLine size={14} /> Offline inference</span></div>
    <div className="estimate-grid"><section className="panel estimate-form"><SectionHeading title="Choose your input" description="Every estimate uses the selected model’s saved transformation." />
      <Select label="Saved model" value={model} disabled={pending} onChange={(e) => { setModel(e.target.value); setUpload([]); setFileName(""); }}>{study.models.map((entry) => <option value={entry.id} key={entry.id}>{methodName(entry.id)}</option>)}</Select>
      <p className="field-help">{selectedModel?.feature_names.length ?? 0} ordered features, output in {selectedModel?.wear_unit ?? "source units"}. Saved {study.default_scenario} model, seed {study.default_seed}.</p>
      <div className="segmented input-tabs" aria-label="Input source"><button disabled={pending} aria-pressed={mode === "recorded"} onClick={() => setMode("recorded")}>Recorded sample</button><button disabled={pending} aria-pressed={mode === "upload"} onClick={() => setMode("upload")}>Upload features</button></div>
      {mode === "recorded" ? <><Select label="Held-out sample" disabled={pending || !samples.length} value={sampleId} onChange={(e) => setSampleId(e.target.value)}>{!samples.length && <option value="">Loading recorded samples…</option>}{samples.map((sample) => <option key={sample.id} value={sample.id}>{sample.run_id} / window {sample.window_id} / {sample.split}</option>)}</Select><p className="field-help">Recorded source identities stay attached to the prediction.</p>{sampleError && <p role="alert" className="form-error">{sampleError}</p>}</> : <><div className="upload-area"><FileUp size={28} /><strong>{fileName || "Choose a feature CSV"}</strong><p>Exact model columns, no measured-wear label. Up to 10 MiB.</p><label className="button button-outline">Choose file<input type="file" accept=".csv,text/csv" disabled={pending} aria-label="Upload feature CSV" onChange={(e) => void uploadFile(e.target.files?.[0])} /></label></div><button className="text-link" onClick={downloadHeader}>Download matching header <ArrowUpRight size={13} /></button>{upload.length > 0 && <label className="number-field">Feature row <input type="number" min={1} max={upload.length} value={row+1} disabled={pending} onChange={(e) => setRow(Math.max(0, Math.min(upload.length-1, Math.trunc(Number(e.target.value))-1)))} /><span>of {count(upload.length)}</span></label>}</>}
      <label className="checkbox-field"><input type="checkbox" checked={explain} disabled={pending} onChange={(e) => setExplain(e.target.checked)} /><span><strong>Explain this estimate</strong><small>SHAP contributions from a training-only reference</small></span></label>
      {error && <p role="alert" className="form-error">{error}</p>}
      <Button className="estimate-button" disabled={pending || !model || (mode === "recorded" ? !sampleId : !upload.length)} onClick={() => void predict()}>{pending ? <LoaderCircle className="spin" size={17} /> : <ScanLine size={17} />}{pending ? "Computing estimate…" : "Estimate wear"}</Button>
    </section><section className="panel estimate-output" aria-live="polite" aria-busy={pending}>
      {!result ? <div className="estimate-empty"><div className="estimate-empty-icon"><ScanLine size={40} strokeWidth={1.1} /></div><h2>{pending ? "Reading the selected model…" : "A prediction with context."}</h2><p>{pending ? "Computing wear and the requested feature contributions." : "Choose a sample to see estimated flank wear, measured wear when available, and the features that shaped the result."}</p><div><span>Measured units</span><span>Model provenance</span><span>Feature contributions</span></div></div> : <>
        <SectionHeading title="Estimated flank wear" action={<span className="tag">{methodName(result.model)}</span>} />
        <div className="prediction-number">{number(result.predicted_wear)}<span>{result.wear_unit}</span></div>
        <div className="prediction-reference"><div><span>Measured wear</span><strong>{result.measured_wear == null ? "Not supplied" : `${number(result.measured_wear)} ${result.wear_unit}`}</strong></div><div><span>Absolute error</span><strong>{result.measured_wear == null ? "Unavailable" : `${number(Math.abs(result.predicted_wear-result.measured_wear))} ${result.wear_unit}`}</strong></div></div>
        {result.sample && <p className="prediction-source">{result.sample.run_id}, window {result.sample.window_id}. {result.sample.split === "test" ? "Final test" : "Validation"} sample.</p>}
        <details className="feature-details"><summary>Inspect input features</summary><div className="table-scroll"><table><thead><tr><th>Raw feature</th><th>Recorded value</th></tr></thead><tbody>{result.features.map((feature) => <tr key={feature.name}><th scope="row">{feature.name.replace(/_+/g, " ")}</th><td>{number(feature.value, 5)}</td></tr>)}</tbody></table></div></details>
        {result.explanation && <div className="explanation"><h3><Sparkles size={16} /> What influenced this prediction</h3><p>Signed model contributions relative to the training reference.</p><div className="contribution-list">{contributions.map((entry) => <div key={entry.name}><span>{entry.name.replace(/_+/g, " ")}</span><div className="contribution-track"><i className={entry.value < 0 ? "negative" : "positive"} style={{ width: `${Math.abs(entry.value)/maxContribution*48}%` }} /></div><strong>{entry.value > 0 ? "+" : ""}{number(entry.value)}</strong></div>)}</div><p className="field-help">Reference {number(result.explanation.base_value)} {result.wear_unit}. Reconstruction error {result.explanation.reconstruction_error.toExponential(1)}.</p></div>}
        <InfoNote>Research estimate for human review. Attributions describe the model, not the physical causes of wear. No calibrated uncertainty or inspection threshold is available.</InfoNote>
      </>}
    </section></div>
  </>;
}
