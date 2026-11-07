import { useId } from "react";
import { number, methodName } from "@/lib/experiments";

export const colors = {
  fedavg: "var(--signal)",
  fedprox: "var(--sage)",
  centralized_mlp: "var(--amber)",
  median: "var(--neutral-trace)",
  random_forest: "var(--orange)",
};
export interface Series { name: string; color: string; values: { x: number; y: number }[] }

export function LinePlot({ series, compact = false }: { series: Series[]; compact?: boolean }) {
  const id = useId().replaceAll(":", "");
  const traces = series
    .map((s) => ({ ...s, values: s.values.filter(({ x, y }) => Number.isFinite(x) && Number.isFinite(y)) }))
    .filter((s) => s.values.length);
  const points = traces.flatMap((s) => s.values);
  if (!points.length) return <p className="empty-inline">No recorded rounds for this selection.</p>;
  const width = compact ? 500 : 720;
  const height = compact ? 175 : 300;
  const left = compact ? 43 : 52;
  const right = width - 18;
  const top = 15;
  const baseline = height - 27;
  const maxX = Math.max(1, ...points.map((p) => p.x));
  const maxY = Math.max(.01, ...points.map((p) => p.y)) * 1.08;
  const x = (value: number) => left + value / maxX * (right - left);
  const y = (value: number) => baseline - value / maxY * (baseline - top);
  const steps = Math.min(maxX, 5);
  const xTicks = Array.from({ length: steps + 1 }, (_, i) => Math.round(maxX * i / steps));

  return <div className={compact ? "line-plot compact-plot" : "line-plot"}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Recorded validation MAE through round ${maxX}`}>
      <defs>
        <linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={traces[0].color} stopOpacity={compact ? ".28" : ".12"} />
          <stop offset="100%" stopColor={traces[0].color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, 1, 2, 3, 4].map((tick) => {
        const value = maxY * tick / 4;
        return <g key={tick}>
          <line className="chart-grid" x1={left} y1={y(value)} x2={right} y2={y(value)} />
          <text x={left - 7} y={y(value) + 4} textAnchor="end">{number(value, maxY < .08 ? 3 : 2)}</text>
        </g>;
      })}
      {xTicks.map((tick) => <g key={tick}>
        {compact && <line className="chart-grid" x1={x(tick)} y1={top} x2={x(tick)} y2={baseline} />}
        <text x={x(tick)} y={height - 7} textAnchor="middle">{tick}</text>
      </g>)}
      {traces.map((s, index) => {
        const first = s.values[0];
        const last = s.values[s.values.length - 1];
        const path = s.values.map((p, i) => `${i ? "L" : "M"}${x(p.x)},${y(p.y)}`).join(" ");
        return <g key={s.name}>
          {index === 0 && <path d={`${path} L${x(last.x)},${baseline} L${x(first.x)},${baseline} Z`} fill={`url(#fill-${id})`} />}
          <path className={compact ? "chart-trace" : undefined} d={path} fill="none" stroke={s.color} strokeWidth={compact ? 2 : 2.4} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
          {compact && <circle className="chart-trace" cx={x(last.x)} cy={y(last.y)} r="4" fill={s.color} />}
        </g>;
      })}
    </svg>
    <div className="chart-axis-labels"><span>Validation MAE (mm)</span><span>Communication rounds</span></div>
    {!compact && <div className="chart-legend">{traces.map((s) => <span key={s.name}><i style={{ backgroundColor: s.color }} />{s.name}</span>)}</div>}
  </div>;
}

export function MethodBars({ rows, showAll = false }: { rows: { method: string; mae: number }[]; showAll?: boolean }) {
  const preferred = ["centralized_mlp", "fedavg", "fedprox", "random_forest", "median"];
  const chosen = showAll ? rows : rows.filter((row) => preferred.includes(row.method));
  const max = Math.max(.001, ...chosen.map((row) => row.mae)) * 1.1;
  return <div className="method-bars" role="list" aria-label="Mean absolute error by method">
    {chosen.map((row) => <div className={`method-bar-row ${row.method === "fedavg" ? "featured-method" : ""}`} key={row.method} role="listitem">
      <div className="method-bar-label"><span>{methodName(row.method)}</span>{row.method === "fedavg" && <small>Federated</small>}</div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${row.mae/max*100}%`, backgroundColor: colors[row.method as keyof typeof colors] ?? "var(--neutral-trace)" }} /></div>
      <span className="bar-number">{number(row.mae)}</span>
    </div>)}
  </div>;
}

export function SplitDiagram({ splits }: { splits: { split: string; windows: number; groups: number }[] }) {
  const total = splits.reduce((sum, split) => sum + split.windows, 0);
  return <div className="split-diagram"><div className="split-track" aria-hidden="true">{splits.map((split) => <div key={split.split} className={`split-${split.split}`} style={{ flex: split.windows / Math.max(total, 1) }} />)}</div>
    <div className="split-legend">{splits.map((split) => <span key={split.split}><i className={`split-${split.split}`} /><span>{split.split === "train" ? "Train" : split.split === "test" ? "Test" : "Validation"}<b>{split.groups} cases</b></span></span>)}</div>
  </div>;
}
