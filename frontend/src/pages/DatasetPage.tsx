import { ArrowUpRight, Check, Database, FileCheck2, Layers3 } from "lucide-react";
import { count, datasetInfo } from "@/lib/experiments";
import { SplitDiagram } from "@/components/research-charts";
import { InfoNote, SectionHeading } from "@/components/research-controls";
import type { Study } from "@/types/experiments";

const channelLabels: Record<string, string> = { AE_spindle: "Acoustic emission at spindle", AE_table: "Acoustic emission at table", smcAC: "AC spindle motor current", smcDC: "DC spindle motor current", vib_table: "Table vibration", vib_spindle: "Spindle vibration" };
export function DatasetPage({ study }: { study: Study }) {
  const data = study.dataset;
  const source = datasetInfo(data.name);
  return <><div className="page-heading"><div><h1>Start with the source.</h1><p>Know what was recorded, what was measured, and what was excluded.</p></div><a className="button button-outline" href={source.url} target="_blank" rel="noreferrer">Source catalogue <ArrowUpRight size={15} /></a></div>
    <section className="panel source-overview"><div className="source-overview-intro"><span className="source-symbol"><Database size={28} /></span><div><span className="status-chip"><Check size={13} /> Source audited</span><h2>{source.title}</h2><p>{source.description}</p></div></div><div className="source-stats"><div><strong>{data.run_count}</strong><span>{data.name === "uniwear_nuaa" ? "Source experiments" : "Recorded runs"}</span></div><div><strong>{data.labeled_runs}</strong><span>{data.name === "uniwear_nuaa" ? "Labeled experiments" : "Measured labels"}</span></div><div><strong>{data.case_count}</strong><span>Recorded cases</span></div><div><strong>{data.unlabeled_runs}</strong><span>Unlabeled runs excluded</span></div></div></section>
    <div className="dataset-detail-grid"><section className="panel"><SectionHeading title="Signal channel inventory" description="Original channel names and audited sample coverage." /><div className="table-scroll"><table><thead><tr><th>Recorded channel</th><th>Samples</th><th>Nonfinite</th></tr></thead><tbody>{data.channels.map((channel) => <tr key={channel.name}><th scope="row"><strong>{channelLabels[channel.name] ?? channel.name}</strong><small className="channel-id">{channel.name}</small></th><td>{count(channel.samples)}</td><td><span className="valid-value"><Check size={13} />{channel.nonfinite_samples}</span></td></tr>)}</tbody></table></div></section>
    <div className="dataset-detail-aside"><section className="panel"><SectionHeading title="Group-disjoint splits" action={<Layers3 size={18} className="muted-icon" />} /><SplitDiagram splits={data.splits} /><dl className="dataset-facts"><div><dt>Prepared windows</dt><dd>{count(data.window_count)}</dd></div><div><dt>Engineered features</dt><dd>{data.feature_count}</dd></div><div><dt>Wear target units</dt><dd>{data.wear_unit}</dd></div></dl></section><section className="panel provenance-panel"><FileCheck2 size={22} /><h2>Versioned and traceable</h2><p>Each saved estimate links to the source dataset and executed implementation.</p><dl><dt>Source SHA-256</dt><dd>{study.provenance.source_sha256 || "Not recorded"}</dd><dt>Code SHA-256</dt><dd>{study.provenance.code_sha256 || "Not recorded"}</dd></dl></section></div></div>
    <InfoNote>{source.limitation}</InfoNote>
  </>;
}
