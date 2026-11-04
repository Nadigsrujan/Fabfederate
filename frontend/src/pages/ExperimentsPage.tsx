import { useState } from "react";
import { BarChart3, Download, TrendingDown } from "lucide-react";
import { LinePlot, MethodBars, colors } from "@/components/research-charts";
import { count, methodName, number, scenarioNames, summarizeMetrics } from "@/lib/experiments";
import { InfoNote, SectionHeading, Select } from "@/components/research-controls";
import type { Study } from "@/types/experiments";

export function ExperimentsPage({ study }: { study: Study }) {
  const [scenario, setScenario] = useState(study.default_scenario);
  const [tab, setTab] = useState<"performance" | "convergence">("performance");
  const [seed, setSeed] = useState(study.default_seed);
  const rows = study.metrics.filter((row) => row.scenario === scenario && row.split === "test");
  const summary = summarizeMetrics(rows);
  const rounds = study.rounds.filter((row) => row.scenario === scenario && row.seed === seed);
  const series = ["fedavg", "fedprox"].map((method) => ({ name: methodName(method), color: colors[method as keyof typeof colors], values: rounds.filter((row) => row.method === method).sort((a, b) => a.round-b.round).map((row) => ({ x: row.round, y: row.validation_mae })) }));
  const bytes = rounds.reduce((sum, row) => sum + row.upload_bytes + row.download_bytes, 0);
  function download() {
    const csv = ["method,scenario,mae,rmse,seeds,units", ...summary.map((row) => `${row.method},${scenario},${row.mae},${row.rmse},${row.seeds},${study.dataset.wear_unit}`)].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${study.id}-${scenario}-comparison.csv`; anchor.click(); URL.revokeObjectURL(url);
  }
  return <><div className="page-heading"><div><h1>Compare the evidence.</h1><p>Matched models, the same held-out cases, and every outcome on record.</p></div><button className="button button-outline" onClick={download}><Download size={16} /> Export comparison</button></div>
    <div className="filter-toolbar"><div className="segmented" aria-label="Experiment view"><button aria-pressed={tab === "performance"} onClick={() => setTab("performance")}><BarChart3 size={16} /> Performance</button><button aria-pressed={tab === "convergence"} onClick={() => setTab("convergence")}><TrendingDown size={16} /> Convergence & cost</button></div><div className="toolbar-fields"><Select label="Data scenario" value={scenario} onChange={(e) => setScenario(e.target.value)}>{study.scenarios.map((entry) => <option key={entry} value={entry}>{scenarioNames[entry] ?? entry}</option>)}</Select>{tab === "convergence" && <Select label="Training seed" value={seed} onChange={(e) => setSeed(Number(e.target.value))}>{study.seeds.map((entry) => <option key={entry}>{entry}</option>)}</Select>}</div></div>
    {tab === "performance" ? <><section className="panel"><SectionHeading title="Held-out model performance" description={`Mean absolute error in ${study.dataset.wear_unit}. Each value averages the recorded training seeds.`} /><div className="wide-bars"><MethodBars rows={summary} showAll /></div></section>
      <section className="panel results-table"><SectionHeading title="Results, in detail" description="Windows share run-level measurements; they are not independent tools." /><div className="table-scroll"><table><thead><tr><th>Model</th><th>MAE ({study.dataset.wear_unit})</th><th>RMSE ({study.dataset.wear_unit})</th><th>Training seeds</th><th>Test cases</th><th>Test windows</th></tr></thead><tbody>{summary.map((row) => <tr key={row.method} className={row.method === "fedavg" ? "highlight-row" : ""}><th scope="row">{methodName(row.method)}{row.method === "fedavg" && <span className="tag">Federated</span>}</th><td>{number(row.mae)}</td><td>{number(row.rmse)}</td><td>{row.seeds}</td><td>{row.groups}</td><td>{count(row.count)}</td></tr>)}</tbody></table></div></section></> : <><section className="panel"><SectionHeading title="Learning over communication rounds" description="Validation error is used for model selection. Test labels never choose a checkpoint." /><LinePlot series={series} /></section><div className="cost-summary"><div><span>Recorded training time</span><strong>{number(rounds.reduce((sum, row) => sum + row.duration_seconds, 0), 2)} <small>seconds</small></strong></div><div><span>Total model payload</span><strong>{number(bytes / 2**20, 2)} <small>MiB</small></strong></div><div><span>Participating clients</span><strong>{study.clients.length} <small>logical clients</small></strong></div></div></>}
    <InfoNote>Federated learning can help or harm under different data conditions. These are public-data simulations, not validated cross-factory or campus-machine results.</InfoNote>
  </>;
}
