"""Generate a compact research report strictly from experiment CSV artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MAX_BARS = 40


def _table(frame: pd.DataFrame) -> str:
    """Small dependency-free Markdown table, with explicit missing results."""
    def display(value):
        if pd.isna(value):
            return 'unavailable'
        if isinstance(value, (float, np.floating)):
            return f'{value:.6g}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    lines += ['| ' + ' | '.join(display(v) for v in row) + ' |'
              for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(lines)


def _overview(frame):
    """Use held-out test and IID when recorded; never mix different split strata."""
    selected = frame.copy()
    scope = []
    if 'split' in selected:
        splits = selected['split'].dropna().unique()
        split = 'test' if 'test' in splits else ('validation' if 'validation' in splits else splits[0])
        selected = selected.loc[selected['split'].eq(split)]
        scope.append(f'split={split}')
    if 'scenario' in selected:
        scenarios = sorted(selected.scenario.dropna().unique())
        scenario = 'iid' if 'iid' in scenarios else scenarios[0]
        selected = selected.loc[selected.scenario.eq(scenario)]
        scope.append(f'scenario={scenario}')
    for column in ['dataset_id', 'wear_unit', 'units']:
        if column in selected and selected[column].nunique() > 1:
            value = sorted(selected[column].dropna().unique())[0]
            selected = selected.loc[selected[column].eq(value)]
            scope.append(f'{column}={value}')
    return selected, ', '.join(scope) or 'all recorded rows'


def generate_report(experiment_dir: Path) -> Path:
    """Write ``report.md`` and bounded overview figures from persisted evidence.

    Full per-seed records stay in CSV. Report tables summarize repeated seeds
    and figures focus the recorded IID test stratum when it is available.
    """
    directory = Path(experiment_dir)
    metrics = pd.read_csv(directory / 'metrics.csv')
    required = {'method', 'split', 'mae', 'rmse'}
    if metrics.empty or not required.issubset(metrics):
        raise ValueError('metrics.csv must contain real method/split/mae/rmse rows')
    if not np.isfinite(metrics[['mae', 'rmse']].to_numpy(dtype=float)).all():
        raise ValueError('metrics.csv contains nonfinite errors')
    figures = directory / 'figures'
    figures.mkdir(exist_ok=True)
    selected_metrics = metrics.loc[metrics['split'].eq('test')] if metrics['split'].eq('test').any() else metrics
    strata = [c for c in ['method', 'scenario', 'split', 'units'] if c in metrics]
    aggregation = {'runs': ('mae', 'size'), 'mean_mae': ('mae', 'mean'),
                   'seed_std_mae': ('mae', 'std'), 'mean_rmse': ('rmse', 'mean')}
    if 'seed' in metrics:
        aggregation['seeds'] = ('seed', 'nunique')
    if 'group_mae' in metrics:
        aggregation['mean_group_mae'] = ('group_mae', 'mean')
    summary = selected_metrics.groupby(strata, dropna=False).agg(**aggregation).reset_index()
    summary.to_csv(directory / 'evaluation_summary.csv', index=False)
    sections = [
        '# FabFederate-CNC experiment report',
        f'Artifact directory: `{directory.name}`. Every value and figure below comes from '
        'saved CSV artifacts. NASA evaluation is case-heldout; independent physical-tool '
        'identity is unverified. Windows are not additional independent machining cases.',
        '## Held-out evaluation across recorded seeds', _table(summary.head(60)),
        '[Full evaluation summary](evaluation_summary.csv) · [All per-run metrics](metrics.csv) '
        '· [All saved predictions](predictions.csv)',
        'Means and standard deviations describe recorded seeds, not independent new cases or '
        'calibrated prediction intervals. Single-seed standard deviations are unavailable. '
        'Report tables show at most 60 rows; CSV files retain every row.',
    ]
    prediction_path = directory / 'predictions.csv'
    if prediction_path.exists():
        predictions = pd.read_csv(prediction_path)
        if not predictions.empty and {'wear_target', 'prediction'}.issubset(predictions):
            sections += _prediction_figures(predictions, directory, figures)
    rounds_path = directory / 'rounds.csv'
    if rounds_path.exists():
        rounds = pd.read_csv(rounds_path)
        if not rounds.empty and {'round', 'validation_mae'}.issubset(rounds):
            sections += _round_figures(rounds, directory, figures)
    sections += [
        '## Interpretation and limitations',
        'Compare methods only within matched datasets, splits, scenarios and budgets. '
        'Recorded outcomes include negative results; no guaranteed federated-learning benefit '
        'is established. Missing client-domain errors remain unavailable.',
        'Public-data clients are logical simulation partitions on one computer. Pooled public '
        'preprocessing and parameter exchange provide no formal privacy guarantee. SHAP '
        'attributions describe the fitted predictor; correlated features and reference choices '
        'can change attribution, and attribution is not causation.',
        'Campus validation is unavailable unless a separately documented approved capture '
        'and measured-label study has been performed. This is an offline research demonstrator '
        'for human review, not a validated inspection threshold or machine-control system.',
    ]
    path = directory / 'report.md'
    path.write_text('\n\n'.join(sections) + '\n')
    return path


def _prediction_figures(predictions, directory, figures):
    plt = _plotting()
    overview, scope = _overview(predictions)
    unit = str(overview.wear_unit.iloc[0]) if 'wear_unit' in overview else 'source units'
    points = _prediction_points(overview)
    points.to_csv(directory / 'overview_predictions.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = points.groupby('method') if 'method' in points else [('all', points)]
    for method, part in groups:
        ax.scatter(part.wear_target, part.prediction, alpha=.55, s=18, label=str(method))
    values = points[['wear_target', 'prediction']].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size:
        low, high = float(finite.min()), float(finite.max())
        ax.plot([low, high], [low, high], '--', color='#607080', linewidth=1)
    ax.set(xlabel=f'Measured wear ({unit})', ylabel=f'Mean prediction across seeds ({unit})',
           title='Held-out measured and predicted wear')
    ax.legend(fontsize=7)
    _save(fig, figures / 'measured_vs_predicted.png')
    sections = ['## Measured versus predicted',
                f'Overview: {scope}. Repeated-seed predictions are averaged for the same '
                'saved window, preserving dataset, source cut/run, group, window and other recorded '
                'provenance. Only the training seed is averaged; this descriptive point estimate '
                'is not a separately fitted ensemble. '
                '[Exact plotted points and seed counts](overview_predictions.csv).',
                '![Measured and predicted wear](figures/measured_vs_predicted.png)']
    if 'group_id' not in predictions:
        return sections
    errors = predictions.assign(absolute_error=(predictions.prediction - predictions.wear_target).abs())
    group_columns = [c for c in ['method', 'seed', 'scenario', 'split', 'group_id'] if c in errors]
    group_errors = errors.groupby(group_columns).absolute_error.mean().reset_index()
    group_errors.to_csv(directory / 'group_errors.csv', index=False)
    selected_errors, _ = _overview(errors)
    sections += ['## Independent-case error',
                 '[All case errors, including every seed](group_errors.csv). '
                 f'Overview figures show at most {MAX_BARS} highest-error method/case combinations; '
                 'each bar averages the recorded seed-level case MAEs. Full CSV coverage is retained.']
    sections += _error_plot(selected_errors, 'group_id', figures)
    failures = selected_errors.sort_values('absolute_error', ascending=False).head(10)
    columns = [c for c in ['method', 'seed', 'scenario', 'split', 'group_id', 'window_id',
                           'wear_target', 'prediction', 'absolute_error'] if c in failures]
    sections += ['### Largest recorded window errors in the overview stratum', _table(failures[columns])]
    if 'client_id' in selected_errors:
        supported = selected_errors.client_id.notna() & selected_errors.client_id.astype(str).str.strip().ne('')
        if supported.any():
            sections += _error_plot(selected_errors.loc[supported], 'client_id', figures)
        else:
            sections += ['Client-domain evaluation is unavailable: held-out rows have no '
                         'supported client assignment.']
    return sections



def _prediction_points(overview):
    """Average training seeds only; source window numbers can restart per cut.

    Every other persisted column is part of sample/method identity. In
    particular, group/window/target alone cannot identify a NASA source cut.
    """
    if 'seed' not in overview:
        return overview.assign(recorded_seeds=1)
    keys = [column for column in overview.columns if column not in {'seed', 'prediction'}]
    if overview.duplicated(keys + ['seed']).any():
        raise ValueError('Duplicate prediction provenance within one training seed')
    return overview.groupby(keys, dropna=False).agg(
        prediction=('prediction', 'mean'), recorded_seeds=('seed', 'nunique')).reset_index()


def _round_figures(rounds, directory, figures):
    plt = _plotting()
    overview, scope = _overview(rounds)
    fig, ax = plt.subplots(figsize=(8, 4))
    groups = overview.groupby('method') if 'method' in overview else [('all', overview)]
    for method, part in groups:
        curve = part.groupby('round').validation_mae.agg(['mean', 'std']).reset_index()
        ax.plot(curve['round'], curve['mean'], label=str(method))
        if curve['std'].notna().any():
            ax.fill_between(curve['round'], curve['mean'] - curve['std'].fillna(0),
                            curve['mean'] + curve['std'].fillna(0), alpha=.12)
    ax.set(xlabel='Communication round', ylabel='Validation MAE (source units)',
           title='Recorded validation convergence')
    ax.legend(fontsize=8)
    _save(fig, figures / 'convergence.png')
    sections = ['## Convergence', f'Overview: {scope}. Lines show seed means; shading shows '
                'one descriptive standard deviation when multiple seeds are present.',
                '![Recorded convergence](figures/convergence.png)']
    costs = [c for c in ['duration_seconds', 'upload_bytes', 'download_bytes'] if c in rounds]
    keys = [c for c in ['method', 'seed', 'scenario'] if c in rounds]
    if costs and keys:
        per_run = rounds.groupby(keys, dropna=False)[costs].sum().reset_index()
        per_run.to_csv(directory / 'cost_by_run.csv', index=False)
        group_keys = [key for key in keys if key != 'seed']
        cost = per_run.groupby(group_keys, dropna=False)[costs].mean().reset_index()
        sections += ['## Measured simulation cost', _table(cost.head(60)),
                     '[Full cost records by run](cost_by_run.csv). Values above are the mean '
                     'total per run across recorded seeds. Payload bytes count serialized model '
                     'updates, not network protocol overhead or production deployment cost.']
    return sections


def _plotting():
    # Lazy import avoids platform font discovery during package/test collection.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    _plotting().close(fig)


def _error_plot(errors, column, figures):
    plt = _plotting()
    keys = [c for c in ['method', 'seed', column] if c in errors]
    per_seed = errors.groupby(keys).absolute_error.mean().reset_index()
    summary_keys = [key for key in keys if key != 'seed']
    data = per_seed.groupby(summary_keys).absolute_error.mean().reset_index()
    data = data.nlargest(MAX_BARS, 'absolute_error').sort_values('absolute_error')
    data['label'] = data[summary_keys].astype(str).agg(' / '.join, axis=1)
    fig, ax = plt.subplots(figsize=(9, max(3, len(data) * .22)))
    ax.barh(data.label, data.absolute_error, color='#0c7775')
    ax.set(xlabel='Mean seed-level MAE (source units)', title=f'Error by {column} (up to {MAX_BARS} entries)')
    filename = f'error_by_{column}.png'
    _save(fig, figures / filename)
    return [f'![Error by {column}](figures/{filename})']
