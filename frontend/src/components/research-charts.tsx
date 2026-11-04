import { useId } from "react";
import { number, methodName } from "@/lib/experiments";

export const colors = { fedavg: "#3567e8", fedprox: "#237f84", centralized_mlp: "#9a77c4", median: "#8394a6", random_forest: "#b28b54" };
export interface Series { name: string; color: string; values: { x: number; y: number }[] }

export function LinePlot({ series, compact = false }: { series: Series[]; compact?: boolean }) {
  const id = useId().replaceAll(":", "");
  const points = series.flatMap((s) => s.values).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!points.length) return <p className="empty-inline">No recorded rounds for this selection.</p>;
  const width = 720, height = compact ? 205 : 300, left = compact ? 5 : 52, right = 18, top = 20, bottom = compact ? 14 : 42;
  const maxX = Math.max(1, ...points.map((p) => p.x));
  const minY = compact ? Math.min(...points.map((p) => p.y)) * .85 : 0;
  const maxY = Math.max(minY + .01, ...points.map((p) => p.y)) * 1.08;
  const x = (value: number) => left + value / maxX * (width - left - right);
  const y = (value: number) => top + (maxY - value) / (maxY - minY) * (height - top - bottom);
  const line = (values: Series["values"]) => values.filter((p) => Number.isFinite(p.y)).map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(" ");
  return <div className={compact ? "line-plot compact-plot" : "line-plot"}>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`Recorded validation MAE through ${maxX} rounds`}>
      <defs><linearGradient id={`fill-${id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={series[0]?.color} stopOpacity={compact ? ".22" : ".09"} /><stop offset="100%" stopColor={series[0]?.color} stopOpacity="0" /></linearGradient></defs>
      {[0, 1, 2, 3, 4].map((tick) => { const yy = top + tick / 4 * (height - top - bottom); return <g key={tick}><line className="chart-grid" x1={left} y1={yy} x2={width-right} y2={yy} />{!compact && <text x={left-12} y={yy+4} textAnchor="end">{number(maxY - tick / 4 * (maxY-minY), 2)}</text>}</g>; })}
      {series.map((s, index) => <g key={s.name}>{index === 0 && <path d={`${line(s.values)} L${x(s.values.at(-1)?.x ?? 0)},${height-bottom} L${x(s.values[0]?.x ?? 0)},${height-bottom} Z`} fill={`url(#fill-${id})`} />}<path d={line(s.values)} fill="none" stroke={s.color} strokeWidth={compact ? 2.5 : 2.4} strokeLinecap="round" strokeLinejoin="round" /></g>)}
      {!compact && [0, 1, 2, 3, 4, 5].map((tick) => <text key={tick} x={x(maxX*tick/5)} y={height-16} textAnchor="middle">{Math.round(maxX*tick/5)}</text>)}
    </svg>
    {!compact && <div className="chart-axis-labels"><span>Validation error (mm)</span><span>Communication rounds</span></div>}
    <div className="chart-legend">{series.map((s) => <span key={s.name}><i style={{ backgroundColor: s.color }} />{s.name}</span>)}</div>
  </div>;
}

export function MethodBars({ rows, showAll = false }: { rows: { method: string; mae: number }[]; showAll?: boolean }) {
  const preferred = ["centralized_mlp", "fedavg", "fedprox", "random_forest", "median"];
  const chosen = showAll ? rows : rows.filter((row) => preferred.includes(row.method));
  const max = Math.max(.001, ...chosen.map((row) => row.mae)) * 1.1;
  return <div className="method-bars" role="list" aria-label="Mean absolute error by method">
    {chosen.map((row) => <div className={`method-bar-row ${row.method === "fedavg" ? "featured-method" : ""}`} key={row.method} role="listitem">
      <div className="method-bar-label"><span>{methodName(row.method)}</span>{row.method === "fedavg" && <small>Federated</small>}</div>
      <div className="bar-track"><div className="bar-fill" style={{ width: `${row.mae/max*100}%`, backgroundColor: colors[row.method as keyof typeof colors] ?? "#afbdcc" }} /></div>
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
