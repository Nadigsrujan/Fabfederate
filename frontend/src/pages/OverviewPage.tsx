import { Activity, ArrowUpRight, Check, Database, FlaskConical, Network, ShieldCheck } from "lucide-react";
import { motion, stagger, useReducedMotion, type Variants } from "motion/react";
import { LinePlot, MethodBars, SplitDiagram } from "@/components/research-charts";
import { SectionHeading } from "@/components/research-controls";
import { Button } from "@/components/ui/button";
import { ResearchBackground } from "@/components/ui/research-background";
import { count, datasetInfo, number, scenarioNames, summarizeMetrics } from "@/lib/experiments";
import type { Study } from "@/types/experiments";

const entrance: Variants = {
  ready: { transition: { delayChildren: stagger(0.12) } },
};
const rise: Variants = {
  waiting: { opacity: 0, y: 12 },
  ready: { opacity: 1, y: 0, transition: { duration: 0.48, ease: [0.22, 1, 0.36, 1] } },
};

export function OverviewPage({ study }: { study: Study }) {
  const reducedMotion = useReducedMotion();
  const { dataset, default_scenario: scenario, default_seed: seed } = study;
  const metrics = summarizeMetrics(study.metrics.filter((row) => row.scenario === scenario && row.split === "test"));
  const rounds = study.rounds
    .filter((row) => row.method === "fedavg" && row.seed === seed && row.scenario === scenario)
    .sort((a, b) => a.round - b.round);
  const lastRound = rounds.at(-1);
  const test = dataset.splits.find((split) => split.split === "test");
  const trace = {
    name: "FedAvg",
    color: "var(--signal)",
    values: rounds.map((row) => ({ x: row.round, y: row.validation_mae })),
  };

  return (
    <motion.div className="overview" variants={entrance} initial={reducedMotion ? false : "waiting"} animate="ready">
      <motion.section variants={rise} aria-labelledby="overview-title">
        <ResearchBackground className="workspace-hero min-h-0" intensity="subtle" animated={false}>
          <div className="hero-content">
            <div className="hero-copy">
              <div className="hero-status"><span aria-hidden="true" /> CNC milling research</div>
              <h1 id="overview-title">Tool wear<br /><span>analysis.</span></h1>
              <p>Compare models on held-out milling runs and see which sensor readings drive each wear estimate.</p>
              <div className="hero-actions">
                <Button asChild><a href="#estimate">Estimate tool wear <ArrowUpRight size={17} /></a></Button>
                <a className="hero-secondary" href="#experiments">Explore experiments <ArrowUpRight size={15} /></a>
              </div>
            </div>
            <figure className="hero-signal">
              <figcaption className="signal-heading">
                <span><Activity size={17} aria-hidden="true" /> FedAvg validation</span>
                <span>Recorded</span>
              </figcaption>
              {lastRound && (
                <div className="signal-readout">
                  <div><strong>{number(lastRound.validation_mae)}</strong> <span>{dataset.wear_unit}</span></div>
                  <span>Final-round MAE</span>
                </div>
              )}
              <LinePlot compact series={[trace]} />
              <div className="signal-caption">
                <span>{study.clients.length} logical clients</span>
                <span>Seed {seed}{lastRound ? ` / Round ${lastRound.round}` : ""}</span>
              </div>
            </figure>
          </div>
        </ResearchBackground>
      </motion.section>

      <motion.div variants={rise}>
        <div className="overview-intro">
          <div><h2>Experiment results</h2><p>Model performance and the data behind it.</p></div>
          <span className="subtle-status"><Check size={14} /> Study complete</span>
        </div>
        <div className="overview-grid">
          <section className="panel performance-panel">
            <SectionHeading
              title="Model comparison"
              description={`Held-out error averaged across ${study.seeds.length} training seeds.`}
              action={<a className="text-link" href="#experiments">Full comparison <ArrowUpRight size={14} /></a>}
            />
            <div className="chart-kicker"><span>Mean absolute error</span><span>Lower is better · {dataset.wear_unit}</span></div>
            <MethodBars rows={metrics} />
            <div className="chart-footer">
              <span>{scenarioNames[scenario] ?? scenario}</span>
              <span>{test?.groups ?? "—"} held-out cases</span>
            </div>
          </section>
          <section className="panel dataset-panel">
            <SectionHeading title="Dataset" action={<Database size={19} className="muted-icon" />} />
            <div className="source-name">
              <span className="source-symbol"><Database size={23} /></span>
              <div><h3>{datasetInfo(dataset.name).title}</h3><p>Public benchmark</p></div>
            </div>
            <dl className="dataset-facts">
              <div><dt>{dataset.name === "uniwear_nuaa" ? "Source experiments" : "Recorded machining runs"}</dt><dd>{count(dataset.run_count)}</dd></div>
              <div><dt>{dataset.name === "uniwear_nuaa" ? "Labeled experiments" : "Measured wear labels"}</dt><dd>{count(dataset.labeled_runs)}</dd></div>
              <div><dt>Recorded cases</dt><dd>{dataset.case_count}</dd></div>
              <div><dt>Sensor channels</dt><dd>{dataset.channels.length}</dd></div>
            </dl>
            <SplitDiagram splits={dataset.splits} />
            <a href="#dataset" className="text-link dataset-link">Inspect dataset <ArrowUpRight size={14} /></a>
          </section>
        </div>
      </motion.div>

      <motion.div variants={rise}>
        <section className="evidence-strip">
          <div className="evidence-mark"><ShieldCheck size={24} /></div>
          <div><h2>Study provenance</h2><p>Source hashes, case-disjoint splits, and versioned models keep the comparison reproducible.</p></div>
          <a className="text-link" href={`/api/experiments/${encodeURIComponent(study.id)}/report`} download>Read the study <ArrowUpRight size={15} /></a>
        </section>
        <div className="overview-bottom">
          <div><FlaskConical size={19} /><span><strong>{study.scenarios.length} data scenarios</strong><small>Balanced, skewed, and limited-label conditions</small></span></div>
          <div><Network size={19} /><span><strong>{study.clients.length} simulated clients</strong><small>Logical partitions on one development computer</small></span></div>
          <div><ShieldCheck size={19} /><span><strong>{number(metrics.find((row) => row.method === "fedavg")?.mae)} {dataset.wear_unit} FedAvg MAE</strong><small>Mean absolute error on held-out data</small></span></div>
        </div>
      </motion.div>
    </motion.div>
  );
}
