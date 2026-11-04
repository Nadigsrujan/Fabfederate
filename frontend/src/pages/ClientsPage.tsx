import { Cpu, GitMerge, Network } from "lucide-react";
import { count, methodName, number } from "@/lib/experiments";
import { InfoNote, SectionHeading } from "@/components/research-controls";
import type { Study } from "@/types/experiments";

export function ClientsPage({ study }: { study: Study }) {
  const total = study.clients.reduce((sum, client) => sum + client.training_windows, 0);
  const errors = ["centralized_mlp", "fedavg", "fedprox"].map((method) => {
    const rows = study.metrics.filter((entry) => entry.method === method && entry.scenario === study.default_scenario && entry.split === "test" && entry.worst_client_mae != null);
    return { method, mae: rows.length ? rows.reduce((sum, row) => sum + row.worst_client_mae!, 0) / rows.length : null, seeds: rows.length };
  });
  return <><div className="page-heading"><div><h1>Different data. Shared learning.</h1><p>Inspect the logical data partitions behind the selected saved model.</p></div><span className="status-chip"><Network size={14} /> {study.clients.length} simulated clients</span></div>
    <section className="panel federation-panel"><SectionHeading title="The federated training path" description="Raw training rows stay in each logical partition. The coordinator aggregates model parameters." />
      <div className="federation-diagram"><div className="client-nodes">{study.clients.map((client, index) => <div className="client-node" key={client.id}><span className="client-node-icon"><Cpu size={28} strokeWidth={1.5} /></span><strong>Client {String.fromCharCode(65+index)}</strong><span>{client.training_groups} training cases</span><span className="node-line" aria-hidden="true" /></div>)}</div><div className="coordinator"><GitMerge size={22} /><div><strong>Shared global model</strong><span>Flower coordinator with FedAvg / FedProx</span></div><span className="tag">One computer</span></div></div>
    </section>
    <section className="panel"><SectionHeading title="Training data allocation" description={`The inference bundles use ${study.default_scenario === "iid" ? "balanced data" : study.default_scenario}, seed ${study.default_seed}.`} />
      <div className="allocation-bar" aria-hidden="true">{study.clients.map((client, index) => <span key={client.id} className={`client-color-${index}`} style={{ flex: client.training_windows / Math.max(total, 1) }} />)}</div>
      <div className="table-scroll"><table><thead><tr><th>Logical client</th><th>Training cases</th><th>Recorded runs</th><th>Feature windows</th><th>Aggregation share</th></tr></thead><tbody>{study.clients.map((client, index) => <tr key={client.id}><th scope="row"><span className={`table-dot client-color-${index}`} />Client {String.fromCharCode(65+index)}</th><td>{client.training_groups}</td><td>{client.training_runs}</td><td>{count(client.training_windows)}</td><td>{(client.training_windows/Math.max(total, 1)*100).toFixed(1)}%</td></tr>)}</tbody></table></div>
    </section><section className="panel client-errors"><SectionHeading title="Worst supported client-domain error" description="Mean across recorded seeds for the saved model’s scenario. Only exact condition matches define a held-out domain." /><div className="table-scroll"><table><thead><tr><th>Model</th><th>Worst-client MAE ({study.dataset.wear_unit})</th><th>Recorded seeds</th></tr></thead><tbody>{errors.map((entry) => <tr key={entry.method}><th scope="row">{methodName(entry.method)}</th><td>{entry.mae == null ? "No supported coverage" : number(entry.mae)}</td><td>{entry.seeds}</td></tr>)}</tbody></table></div><p className="field-help">Unmapped conditions are excluded from this client-domain summary; overall test errors remain available in Experiments.</p></section><InfoNote>A client is a logical subset of a public dataset. This demonstrates algorithm behavior on one computer; it does not establish independent data custody or formal privacy. Parameter updates can reveal information.</InfoNote>
  </>;
}
