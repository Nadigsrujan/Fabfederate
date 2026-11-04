import { ArrowUpRight, Check, Database, FlaskConical, Network, ShieldCheck } from "lucide-react";
import { ResearchBackground } from "@/components/ui/research-background";
import { Button } from "@/components/ui/button";
import { LinePlot, MethodBars, SplitDiagram } from "@/components/research-charts";
import { count, number, summarizeMetrics } from "@/lib/experiments";
import { SectionHeading } from "@/components/research-controls";
import type { Study } from "@/types/experiments";

export function OverviewPage({ study }: { study: Study }) {
  const rows = study.metrics.filter((row) => row.scenario === study.default_scenario && row.split === "test");
  const metrics = summarizeMetrics(rows);
  const rounds = study.rounds.filter((row) => row.method === "fedavg" && row.seed === study.default_seed && row.scenario === study.default_scenario).sort((a, b) => a.round - b.round);
  const test = study.dataset.splits.find((split) => split.split === "test");
  return <>
    <ResearchBackground className="workspace-hero min-h-0" intensity="subtle">
      <div className="hero-content">
        <div className="hero-copy"><span className="hero-status"><span /> Evidence from recorded milling runs</span>
          <h1>Tool wear,<br />with the evidence.</h1>
          <p>Compare how models learn. Understand every estimate. Turn machining signals into a clearer view of tool condition.</p>
          <div className="hero-actions"><Button asChild className="button-light"><a href="#experiments">Explore experiments <ArrowUpRight size={16} /></a></Button><a className="hero-secondary" href="#estimate">Estimate tool wear</a></div>
        </div>
        <div className="hero-signal"><div className="signal-heading"><span><Network size={16} /> FedAvg validation</span><span>{rounds.length} rounds</span></div>
          <LinePlot compact series={[{ name: `Seed ${study.default_seed}`, color: "#84c9ff", values: rounds.map((row) => ({ x: row.round, y: row.validation_mae })) }]} />
          <div className="signal-caption"><span>{study.clients.length} logical clients, seed {study.default_seed}</span><span>Recorded, not live</span></div>
        </div>
      </div>
    </ResearchBackground>
    <div className="overview-intro"><div><h2>A view of the whole experiment.</h2><p>One audited dataset. Matched comparisons. Results you can trace back.</p></div><span className="subtle-status"><Check size={14} /> Study complete</span></div>
    <div className="overview-grid">
      <section className="panel performance-panel"><SectionHeading title="How the models compare" description={`Held-out error averaged across ${study.seeds.length} training seeds.`} action={<a className="text-link" href="#experiments">Full comparison <ArrowUpRight size={14} /></a>} />
        <div className="chart-kicker"><span>Mean absolute error</span><span>Lower is better · {study.dataset.wear_unit}</span></div>
        <MethodBars rows={metrics} />
        <div className="chart-footer"><span>Balanced-data scenario</span><span>{test?.groups ?? "—"} held-out cases</span></div>
      </section>
      <section className="panel dataset-panel"><SectionHeading title="The data behind it" action={<Database size={19} className="muted-icon" />} />
        <div className="source-name"><span className="source-symbol"><Database size={23} /></span><div><h3>NASA Milling Wear</h3><p>Primary public benchmark</p></div></div>
        <dl className="dataset-facts"><div><dt>Recorded machining runs</dt><dd>{count(study.dataset.run_count)}</dd></div><div><dt>Measured wear labels</dt><dd>{count(study.dataset.labeled_runs)}</dd></div><div><dt>Recorded cases</dt><dd>{study.dataset.case_count}</dd></div><div><dt>Sensor channels</dt><dd>{study.dataset.channels.length}</dd></div></dl>
        <SplitDiagram splits={study.dataset.splits} /><a href="#dataset" className="text-link dataset-link">Inspect dataset <ArrowUpRight size={14} /></a>
      </section>
    </div>
    <section className="evidence-strip"><div className="evidence-mark"><ShieldCheck size={24} /></div><div><h2>Every result has a record.</h2><p>Source hashes, case-disjoint splits, and versioned models keep the comparison reproducible.</p></div><a className="text-link" href={`/api/experiments/${encodeURIComponent(study.id)}/report`} download>Read the study <ArrowUpRight size={15} /></a></section>
    <div className="overview-bottom"><div><FlaskConical size={19} /><span><strong>{study.scenarios.length} data scenarios</strong><small>Balanced, skewed, and limited-label conditions</small></span></div><div><Network size={19} /><span><strong>{study.clients.length} simulated clients</strong><small>Logical partitions on one development computer</small></span></div><div><ShieldCheck size={19} /><span><strong>{number(metrics.find((row) => row.method === "fedavg")?.mae)} {study.dataset.wear_unit} FedAvg MAE</strong><small>Recorded error; improvement is not assumed</small></span></div></div>
  </>;
}
