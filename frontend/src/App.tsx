import { useEffect, useState } from "react";
import { Activity, ArrowDownToLine, ArrowUpRight, CheckCircle2, ChevronDown, Cpu, Database, FlaskConical, Grid2X2, ScanLine } from "lucide-react";
import { useStudy, experimentName, routeNames } from "@/lib/experiments";
import { OverviewPage } from "@/pages/OverviewPage";
import { ExperimentsPage } from "@/pages/ExperimentsPage";
import { ClientsPage } from "@/pages/ClientsPage";
import { DatasetPage } from "@/pages/DatasetPage";
import { EstimatePage } from "@/pages/EstimatePage";
import { ErrorState } from "@/components/research-controls";
import type { Route } from "@/types/experiments";

const navigation = [
  { route: "overview", label: "Overview", icon: Grid2X2 },
  { route: "experiments", label: "Experiments", icon: FlaskConical },
  { route: "clients", label: "Clients", icon: Cpu },
  { route: "dataset", label: "Dataset", icon: Database },
  { route: "estimate", label: "Wear estimate", icon: ScanLine },
] as const;
function currentRoute(): Route {
  const candidate = window.location.hash.slice(1);
  return navigation.some((entry) => entry.route === candidate) ? candidate as Route : "overview";
}

export default function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  const resource = useStudy();
  useEffect(() => { const navigate = () => { if (window.location.hash === "#main-content") return; setRoute(currentRoute()); window.scrollTo({ top: 0 }); }; window.addEventListener("hashchange", navigate); return () => window.removeEventListener("hashchange", navigate); }, []);
  useEffect(() => { document.title = `${routeNames[route]} — FabFederate`; }, [route]);
  const study = resource.study;
  return <div className="app-shell"><a className="skip-link" href="#main-content">Skip to content</a>
    <aside className="sidebar"><a href="#overview" className="wordmark" aria-label="FabFederate overview"><span className="brand-icon"><svg viewBox="0 0 24 24" width="23" height="23" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M9 3h6v6l2 2v7l-2 3H9l-2-3v-7l2-2V3ZM9 9h6M7 14l10-3M7 18l10-3M10 21l7-3" /></svg></span><span>FabFederate<span className="brand-description">CNC research</span></span></a>
      <div className="workspace-label">Research workspace</div><nav aria-label="Main navigation">{navigation.map(({ route: target, label, icon: Icon }) => <a key={target} href={`#${target}`} aria-current={route === target ? "page" : undefined} className={route === target ? "nav-item active" : "nav-item"}><Icon size={18} strokeWidth={1.6} /><span>{label}</span>{target === "estimate" && <ArrowUpRight className="nav-shortcut" size={14} />}</a>)}</nav>
      <div className="sidebar-bottom"><div className="research-note"><span className="research-note-icon"><FlaskConical size={18} /></span><strong>Research use</strong><p>Public benchmarks.<br />Human-reviewed estimates.</p></div><a className="sidebar-report" href={resource.id ? `/api/experiments/${encodeURIComponent(resource.id)}/report` : "#dataset"}><FileIcon /> Project report <ArrowUpRight size={14} /></a><div className="sidebar-version"><span className="local-dot" /> Local research prototype</div></div>
    </aside>
    <div className="workspace"><header className="topbar"><div className="breadcrumb">Workspace <span>/</span><strong>{routeNames[route]}</strong></div><div className="topbar-actions"><label className="experiment-picker"><Database size={15} aria-hidden="true" /><span className="sr-only">Experiment</span><select aria-label="Experiment" value={resource.id} onChange={(e) => resource.setId(e.target.value)} disabled={!resource.options.length}>{!resource.options.length && <option value="">Loading experiments…</option>}{resource.options.map((option) => <option key={option.id} value={option.id}>{experimentName(option.id)}</option>)}</select><ChevronDown size={14} aria-hidden="true" /></label>{resource.id && <a className="report-button" href={`/api/experiments/${encodeURIComponent(resource.id)}/report`} download><ArrowDownToLine size={16} /><span>Report</span></a>}</div></header>
      <main id="main-content" tabIndex={-1} className="main-content">
        {resource.error ? <ErrorState message={resource.error} retry={resource.retry} /> : resource.loading ? <div className="workspace-loading" aria-busy="true" role="status"><div className="loading-hero" /><div className="loading-lines"><div /><div /><div /></div><span className="sr-only">Loading recorded research artifacts</span></div> : !study ? <section className="empty-state"><Database size={36} /><h1>No completed studies yet.</h1><p>Run a saved-data experiment to populate this research workspace.</p><code>fabfederate study --config configs/default.yaml</code></section> : <div key={`${study.id}-${route}`} className="view-content">{route === "overview" ? <OverviewPage study={study} /> : route === "experiments" ? <ExperimentsPage study={study} /> : route === "clients" ? <ClientsPage study={study} /> : route === "dataset" ? <DatasetPage study={study} /> : <EstimatePage study={study} />}</div>}
        <footer className="workspace-footer"><span><Activity size={13} /> {study ? "Saved experiment artifacts" : "Local research workspace"}</span><span>{study ? <><CheckCircle2 size={13} /> Local API online</> : "CNC milling-tool wear"}</span><span>Campus validation pending</span></footer>
      </main>
    </div>
  </div>;
}
function FileIcon() { return <ArrowDownToLine size={15} />; }
