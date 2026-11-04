"""FabFederate-CNC: local, artifact-backed research dashboard.

Run with ``streamlit run app/streamlit_app.py`` from the repository root.
"""
from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from fabfederate.contracts import METADATA_COLUMNS  # noqa: E402

PAGES = ['Data Explorer', 'Experiment Comparison', 'Client Heterogeneity',
         'Convergence / Cost', 'Wear Estimate']


def _read_csv(directory, name):
    path = directory / name
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _features(directory):
    path = directory / 'features.parquet'
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _empty(message):
    st.info(message)


def _table(frame):
    st.dataframe(frame, width='stretch', hide_index=True)



def _bar_chart(frame, category, measures, *, x_title, y_title, colors=None):
    """Explicit grouped bars without categorical scale bindings or stacking."""
    palette = colors or ['#087f8c', '#193c55', '#b7c9d6']
    values = frame[[category, *measures]].melt(
        id_vars=category, value_vars=measures, var_name='Series', value_name='Value')
    chart = alt.Chart(values).mark_bar().encode(
        x=alt.X(f'{category}:N', title=x_title, sort=None),
        y=alt.Y('Value:Q', title=y_title, stack=None),
        tooltip=[alt.Tooltip(f'{category}:N', title=x_title),
                 alt.Tooltip('Series:N', title='Statistic'),
                 alt.Tooltip('Value:Q', title=y_title, format='.5g')],
    )
    if len(measures) > 1:
        return chart.encode(
            xOffset=alt.XOffset('Series:N', sort=measures),
            color=alt.Color('Series:N', title=None, sort=measures,
                            scale=alt.Scale(domain=measures, range=palette[:len(measures)])),
        )
    return chart.encode(color=alt.value(palette[0]))


def _filters(frame, columns, prefix):
    selected = frame.copy()
    for column in columns:
        if column in selected and selected[column].notna().any():
            options = selected[column].dropna().unique().tolist()
            value = st.selectbox(column.replace('_', ' ').title(), options,
                                 key=f'{prefix}_{column}')
            selected = selected.loc[selected[column] == value]
    return selected


def data_explorer(directory):
    st.header('Data Explorer')
    st.caption('Inspect source coverage and independent cases before interpreting window-level results.')
    frame = _features(directory)
    if frame.empty:
        _empty('No features.parquet in this experiment. Prepare the audited data and save its feature table.')
        return
    columns = st.columns(4)
    columns[0].metric('Feature windows', f'{len(frame):,}')
    columns[1].metric('Independent groups', frame.group_id.nunique() if 'group_id' in frame else 'Unknown')
    feature_names = [name for name in frame if name not in METADATA_COLUMNS]
    columns[2].metric('Predictor features', len(feature_names))
    units = ', '.join(frame.wear_unit.dropna().astype(str).unique()) if 'wear_unit' in frame else 'Unknown'
    columns[3].metric('Wear units', units)
    st.subheader('Frozen split coverage')
    if {'split', 'group_id'}.issubset(frame):
        coverage = frame.groupby('split', dropna=False).agg(
            windows=('group_id', 'size'), independent_groups=('group_id', 'nunique')).reset_index()
        _table(coverage)
        overlap = frame.groupby('group_id')['split'].nunique().gt(1).sum()
        if overlap:
            st.error(f'Group integrity failure: {overlap} groups occur in multiple splits.')
        else:
            st.caption('No group crosses the recorded splits in this feature table.')
    st.subheader('Wear across groups')
    if {'group_id', 'wear_target'}.issubset(frame):
        wear = frame.groupby('group_id').wear_target.agg(['min', 'median', 'max'])
        st.altair_chart(_bar_chart(
            wear.reset_index(), 'group_id', ['min', 'median', 'max'],
            colors=['#b7c9d6', '#087f8c', '#193c55'], x_title='Independent group',
            y_title=f'Measured wear ({units})'), width='stretch')
    st.subheader('Feature records')
    splits = ['All'] + sorted(frame['split'].dropna().unique().tolist()) if 'split' in frame else ['All']
    split = st.selectbox('Browse split', splits)
    display = frame if split == 'All' else frame.loc[frame['split'] == split]
    _table(display.head(500))
    st.caption(f'Showing up to 500 of {len(display):,} records. Metadata and measured wear are never predictor inputs.')
    with st.expander('Source audit and provenance'):
        audit = directory / 'audit.json'
        if audit.exists():
            st.json(json.loads(audit.read_text()))
        else:
            st.info('No audit.json is bundled here. Refer to the dataset card and acquisition manifest.')
        manifest = directory / 'manifest.json'
        if manifest.exists():
            st.json(json.loads(manifest.read_text()))
        st.write('NASA results use case-heldout splits. Cases are not verified independent physical '
                 'tools. Windows from one case do not create additional independent tools. '
                 'Source-to-campus transfer has not been established.')


def experiment_comparison(directory):
    st.header('Experiment Comparison')
    st.caption('Matched saved evaluations in original wear units. No scores are generated by this view.')
    metrics = _read_csv(directory, 'metrics.csv')
    if metrics.empty:
        _empty('No metrics.csv found. Run the evaluation command for this experiment first.')
        return
    metrics = _filters(metrics, ['scenario', 'split', 'units'], 'comparison')
    st.subheader('Error by method')
    methods = metrics.groupby('method', dropna=False).agg(
        mean_mae=('mae', 'mean'), mean_rmse=('rmse', 'mean'), recorded_runs=('mae', 'size')).reset_index()
    st.altair_chart(_bar_chart(
        methods, 'method', ['mean_mae', 'mean_rmse'], x_title='Method',
        y_title='Error (selected wear units)'), width='stretch')
    _table(methods)
    st.caption('Means summarize the selected recorded runs. RMSE within each run is pooled over '
               'predictions; variation across seeds is descriptive, not calibrated uncertainty.')
    st.subheader('Run-level evidence')
    _table(metrics)
    st.download_button('Download selected evaluation rows', metrics.to_csv(index=False),
                       'evaluation.csv', mime='text/csv')
    predictions = _read_csv(directory, 'predictions.csv')
    if not predictions.empty and {'wear_target', 'prediction'}.issubset(predictions):
        for column in ['scenario', 'split']:
            if column in metrics and column in predictions:
                predictions = predictions.loc[predictions[column].isin(metrics[column])]
        st.subheader('Measured versus predicted')
        st.scatter_chart(predictions, x='wear_target', y='prediction',
                         color='method' if 'method' in predictions else '#087f8c',
                         x_label='Measured wear', y_label='Predicted wear')
    st.info('Federated methods may perform better or worse than baselines. Compare matched '
            'splits, budgets and seeds; the display establishes no guaranteed FL benefit.')


def client_heterogeneity(directory):
    st.header('Client Heterogeneity')
    st.caption('Logical client partitions describe the training simulation on this computer.')
    frame = _features(directory)
    if frame.empty or 'client_id' not in frame:
        _empty('Client assignments are unavailable. Save the prepared training client manifest.')
        return
    supported = frame.client_id.notna() & frame.client_id.astype(str).str.strip().ne('')
    train = frame.loc[supported & frame['split'].eq('train')] if 'split' in frame else frame.loc[supported]
    if train.empty:
        _empty('No supported training client assignments exist in the feature table.')
        return
    st.subheader('Training data distribution')
    counts = train.groupby('client_id').agg(windows=('group_id', 'size'),
                                            groups=('group_id', 'nunique'),
                                            wear_min=('wear_target', 'min'),
                                            wear_median=('wear_target', 'median'),
                                            wear_max=('wear_target', 'max')).reset_index()
    st.altair_chart(_bar_chart(counts, 'client_id', ['windows'],
                               x_title='Training client', y_title='Training windows'), width='stretch')
    _table(counts)
    if 'condition' in train:
        st.subheader('Recorded condition coverage')
        _table(train.groupby(['client_id', 'condition'], dropna=False).size().rename('windows').reset_index())
    predictions = _read_csv(directory, 'predictions.csv')
    st.subheader('Held-out client-domain errors')
    if not predictions.empty and 'client_id' in predictions:
        supported = predictions.client_id.notna() & predictions.client_id.astype(str).str.strip().ne('')
        test = predictions.loc[supported]
        if 'split' in test:
            test = test.loc[test['split'].isin(['validation', 'test'])]
        if not test.empty:
            test = _filters(test, ['scenario', 'split', 'seed'], 'client')
            test = test.assign(absolute_error=(test.prediction - test.wear_target).abs())
            keys = [name for name in ['method', 'client_id'] if name in test]
            _table(test.groupby(keys).agg(mae=('absolute_error', 'mean'),
                                         windows=('absolute_error', 'size'),
                                         groups=('group_id', 'nunique')).reset_index())
        else:
            st.info('Unavailable: held-out groups have no supported client-domain assignments. '
                    'Training clients are not assigned fabricated test domains.')
    else:
        st.info('Unavailable: no held-out prediction artifact with supported client IDs.')
    st.caption('Public-data partitions do not establish independent custody or formal privacy. '
               'Parameter updates can disclose information; this demonstrator has no privacy guarantee.')


def convergence_cost(directory):
    st.header('Convergence / Cost')
    st.caption('Recorded validation trajectories, simulation duration and model payload bytes.')
    rounds = _read_csv(directory, 'rounds.csv')
    if rounds.empty:
        _empty('No rounds.csv found. Run a federated experiment to record convergence and cost.')
        return
    rounds = _filters(rounds, ['scenario', 'seed', 'method'], 'rounds').sort_values('round')
    columns = st.columns(3)
    duration = rounds.duration_seconds.sum() if 'duration_seconds' in rounds else None
    upload = rounds.upload_bytes.sum() if 'upload_bytes' in rounds else None
    download = rounds.download_bytes.sum() if 'download_bytes' in rounds else None
    columns[0].metric('Recorded time', f'{duration:.2f} s' if duration is not None else 'Unavailable')
    columns[1].metric('Model uploads', f'{upload / 2**20:.3f} MiB' if upload is not None else 'Unavailable')
    columns[2].metric('Model downloads', f'{download / 2**20:.3f} MiB' if download is not None else 'Unavailable')
    st.subheader('Validation wear error')
    if 'validation_mae' in rounds:
        st.line_chart(rounds, x='round', y='validation_mae', color='#087f8c',
                      x_label='Round', y_label='Validation MAE (source units)')
    if 'train_loss' in rounds:
        st.subheader('Local training objective')
        st.line_chart(rounds, x='round', y='train_loss', color='#193c55',
                      x_label='Round', y_label='Recorded training loss')
    with st.expander('Round records and cost definition'):
        _table(rounds)
        st.write('Bytes describe recorded model payloads. They exclude network protocol overhead, '
                 'raw data acquisition, and cloud/deployment costs. Training and validation objectives '
                 'can use different scales; use saved units and model metadata when interpreting loss.')


def validate_upload(content: bytes, feature_names: list[str]) -> pd.DataFrame:
    """Strict ordered CSV schema; labels and source identifiers are not accepted."""
    if len(content) > 10 * 1024 * 1024:
        raise ValueError('Feature CSV must be at most 10 MiB')
    try:
        header = next(csv.reader(io.StringIO(content.decode('utf-8-sig'))))
    except (UnicodeError, StopIteration, csv.Error) as exc:
        raise ValueError('Upload must be a UTF-8 CSV with a header') from exc
    if header != list(feature_names):
        raise ValueError('CSV columns and order must exactly match the selected model feature schema. '
                         'Do not include labels or metadata.')
    frame = pd.read_csv(io.BytesIO(content))
    if frame.empty or len(frame) > 1000:
        raise ValueError('Upload must contain 1–1000 feature rows')
    try:
        finite = np.isfinite(frame.to_numpy(dtype=float)).all()
    except (TypeError, ValueError) as exc:
        raise ValueError('All uploaded features must be numeric and finite') from exc
    if not finite:
        raise ValueError('All uploaded features must be numeric and finite')
    return frame


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_explanation(path, artifact_version, background, sample):
    from fabfederate.models.bundle import load_bundle
    from fabfederate.explain.shap_analysis import explain_prediction
    return explain_prediction(load_bundle(Path(path)), background, sample)


def wear_estimate(directory):
    st.header('Wear Estimate')
    st.caption('Use the selected saved model and its fitted transform for local feature inference.')
    model_root = directory / 'models'
    models = sorted(p for p in model_root.iterdir() if p.is_dir() and
                    (p / 'bundle.joblib').exists() and p.resolve().is_relative_to(directory.resolve())) if model_root.exists() else []
    if not models:
        _empty('No model bundle found. Train and save a model under models/<method>/, '
               'then return here to estimate wear.')
        return
    from fabfederate.models.bundle import load_bundle
    model_path = st.selectbox('Saved model', models, format_func=lambda path: path.name)
    bundle = load_bundle(model_path)
    st.caption(f'Model: {bundle.model_kind} · Output unit: {bundle.wear_unit} · '
               f'{len(bundle.feature_names)} ordered features · {model_path.relative_to(directory)}')
    with st.expander('Model provenance and input schema'):
        st.json(bundle.metadata)
        st.code(','.join(bundle.feature_names), language=None)
        card = model_path / 'model_card.md'
        if card.exists():
            st.markdown(card.read_text())
    frame = _features(directory)
    source = st.radio('Input source', ['Held-out feature record', 'Upload feature CSV'], horizontal=True)
    measured = None
    sample = None
    if source == 'Upload feature CSV':
        template = pd.DataFrame(columns=bundle.feature_names).to_csv(index=False)
        st.download_button('Download feature header', template, 'feature_schema.csv', mime='text/csv')
        upload = st.file_uploader('Raw named features (CSV)', type=['csv'],
                                  help='Exact ordered features only; no wear labels or source metadata.')
        if upload is not None:
            try:
                uploaded = validate_upload(upload.getvalue(), bundle.feature_names)
                row = st.number_input('Upload row', min_value=0, max_value=len(uploaded) - 1, step=1)
                sample = uploaded.iloc[[row]].reset_index(drop=True)
                if st.checkbox('Add a measured wear value for comparison',
                               help='Optional observed label; never supplied to the predictor.'):
                    measured = st.number_input(f'Measured wear ({bundle.wear_unit})',
                                               min_value=0., value=0., format='%.4f')
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
                return
    elif not frame.empty and 'split' in frame:
        heldout = frame.loc[frame['split'].isin(['validation', 'test'])]
        if len(heldout):
            index = st.selectbox('Held-out record', heldout.index.tolist(),
                                 format_func=lambda value: f'{heldout.loc[value].get("window_id", value)} · '
                                 f'{heldout.loc[value].get("group_id", "group unavailable")} · '
                                 f'{heldout.loc[value]["split"]}')
            sample = heldout.loc[[index], bundle.feature_names].reset_index(drop=True)
            measured = heldout.loc[index].get('wear_target')
        else:
            st.info('There are no validation/test records in features.parquet.')
    if sample is None:
        st.info('Select a held-out record or upload a matching feature CSV to see a wear estimate.')
        return
    prediction = float(bundle.predict(sample)[0])
    columns = st.columns(3)
    columns[0].metric('Estimated wear', f'{prediction:.4f} {bundle.wear_unit}')
    columns[1].metric('Measured wear', f'{float(measured):.4f} {bundle.wear_unit}'
                      if measured is not None and pd.notna(measured) else 'Not supplied')
    columns[2].metric('Absolute error', f'{abs(prediction - float(measured)):.4f} {bundle.wear_unit}'
                      if measured is not None and pd.notna(measured) else 'Unavailable')
    st.info('Research estimate for human review. No calibrated uncertainty or validated '
            'inspection threshold is available; no machine commands are generated.')
    with st.expander('Raw predictor inputs'):
        _table(sample)
    st.subheader('Why this model produced this estimate')
    st.caption('SHAP describes this selected predictor relative to training reference rows. '
               'Attributions are not causes; correlated features can share or shift credit.')
    if st.button('Explain selected estimate', type='primary'):
        if frame.empty or 'split' not in frame:
            st.warning('Training background is unavailable: save features.parquet with frozen split metadata.')
            return
        background = frame.loc[frame['split'].eq('train'), bundle.feature_names].dropna()
        if background.empty:
            st.warning('No finite complete training background is available for this feature schema.')
            return
        version = (model_path / 'bundle.joblib').stat().st_mtime_ns
        with st.spinner('Explaining the selected model with a bounded training background…'):
            result = _cached_explanation(str(model_path), version, background, sample)
        attribution = pd.DataFrame({'Feature': result['feature_names'],
                                    'SHAP contribution': result['values'][0]})
        attribution['magnitude'] = attribution['SHAP contribution'].abs()
        attribution = attribution.nlargest(12, 'magnitude').drop(columns='magnitude')
        st.altair_chart(_bar_chart(
            attribution, 'Feature', ['SHAP contribution'], x_title='Feature',
            y_title=f'Contribution ({bundle.wear_unit})'), width='stretch')
        _table(attribution)
        st.caption(f'Training reference prediction: {result["base_values"][0]:.5f} {bundle.wear_unit} · '
                   f'Reconstruction error: {result["reconstruction_error"][0]:.2g} · '
                   f'Runtime: {result["runtime_seconds"]:.2f} s · '
                   f'Background: {result["settings"]["background_rows"]} training rows')
        st.download_button('Download explanation and settings', json.dumps(result, indent=2),
                           'explanation.json', mime='application/json')


def main():
    st.set_page_config(page_title='FabFederate-CNC', page_icon='⚙', layout='wide')
    st.markdown('''<style>
    :root { --ff-ink:#15354b; --ff-accent:#087f8c; --ff-border:#dce5ea; }
    .stApp { background:#f7f9fb; }
    [data-testid="stSidebar"] { background:#edf3f6; border-right:1px solid var(--ff-border); }
    h1,h2,h3 { color:var(--ff-ink); letter-spacing:-.02em; }
    [data-testid="stMetric"] { border-top:3px solid var(--ff-accent); padding:.75rem 0; }
    [data-testid="stMetricLabel"] { color:#405e70; }
    .block-container { padding-top:2rem; padding-bottom:3rem; max-width:1440px; }
    </style>''', unsafe_allow_html=True)
    st.title('FabFederate-CNC')
    st.caption('OFFLINE RESEARCH WORKSPACE  /  CNC MILLING WEAR  /  PUBLIC-DATA SIMULATION')
    root = Path(os.environ.get('FABFEDERATE_EXPERIMENTS_DIR', str(REPO / 'experiments')))
    with st.sidebar:
        st.subheader('Research workspace')
        page = st.radio('Explore', PAGES, label_visibility='collapsed')
        experiments = sorted(p for p in root.iterdir() if p.is_dir() and
                             p.resolve().is_relative_to(root.resolve())) if root.exists() else []
        if experiments:
            directory = st.selectbox('Experiment', experiments, format_func=lambda path: path.name)
        else:
            directory = None
        st.divider()
        st.caption('Local artifacts · No cloud services')
        st.caption('Campus validation: unavailable')
        st.caption('Case-heldout evaluation; physical-tool independence is unverified.')
    if directory is None:
        st.header(page)
        st.info('No saved experiments yet. Acquire and audit the NASA milling source, prepare '
                'case-disjoint features, then run the offline study to populate this workspace.')
        st.code('fabfederate --help', language='bash')
        st.write('Expected experiment artifacts: features.parquet, audit.json, metrics.csv, '
                 'predictions.csv, rounds.csv, and models/<method>/bundle.joblib.')
    else:
        st.caption(f'ARTIFACT SET  /  {directory.name}')
        render = dict(zip(PAGES, [data_explorer, experiment_comparison, client_heterogeneity,
                                  convergence_cost, wear_estimate], strict=True))
        try:
            render[page](directory)
        except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
            st.error(f'Artifact validation failed: {exc}')
            st.caption('Review the selected experiment files or regenerate them with the documented CLI.')
    st.divider()
    st.caption('FabFederate-CNC · Offline research demonstrator. Logical clients share one computer. '
               'No formal privacy guarantee, causal interpretation, or campus performance claim.')


if __name__ == '__main__':
    main()
