"""Local experiment, prediction and measured-feedback adapter for the React client.

Only trusted, existing project bundles are loaded. No uploads or user-supplied
filesystem paths are accepted; serve this research adapter on loopback only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4
import math
import os
from pathlib import Path
import re
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

from fabfederate.contracts import METADATA_COLUMNS
from fabfederate.explainability import explain_prediction
from fabfederate.models.bundle import load_bundle

SAFE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z')
LIMITATIONS = [
    'Offline public-data research estimate; no automatic CNC control.',
    'Case-heldout evaluation does not establish a physical-tool-heldout deployment result.',
    'No calibrated uncertainty or universal wear threshold is provided.',
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=128)
    sample_id: str | None = Field(default=None, max_length=128)
    features: list[StrictFloat] | None = Field(default=None, min_length=1, max_length=512)
    explain: bool = False

    @model_validator(mode='after')
    def one_source(self):
        if (self.sample_id is None) == (self.features is None):
            raise ValueError('provide exactly one of sample_id or features')
        return self


class MeasurementRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=128)
    sample_id: str = Field(min_length=1, max_length=128)
    measured_wear: StrictFloat = Field(ge=0)
    wear_unit: str = Field(min_length=1, max_length=16)
    note: str = Field(default='', max_length=1000)


def _finite(value):
    """Keep missing and nonfinite saved statistics valid JSON."""
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _contained(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise HTTPException(404, 'Artifact not found')
    return path


def _identifier(value: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise HTTPException(404, 'Artifact not found')
    return value


def _json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        raise HTTPException(400, 'Saved metadata is unreadable') from None
    if not isinstance(value, dict):
        raise HTTPException(400, 'Saved metadata must be an object')
    return value


def _csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    try:
        return pd.read_csv(path).to_dict('records')
    except (OSError, ValueError):
        raise HTTPException(400, 'Saved table is unreadable') from None


def _frame(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(_contained(path, 'features.parquet'))
    except (OSError, ValueError):
        raise HTTPException(400, 'Saved features are unreadable') from None
    if not {'split', 'group_id', 'window_id'}.issubset(frame.columns):
        raise HTTPException(400, 'Saved features lack split or sample metadata')
    labels = frame.index.map(str)
    if not labels.is_unique:
        raise HTTPException(400, 'Saved sample identifiers are ambiguous')
    frame.index = labels
    return frame


def _feature_names(path: Path, frame: pd.DataFrame) -> list[str]:
    schema = _json(_contained(path, 'schema.json'))
    names = schema.get('feature_names', [name for name in frame if name not in METADATA_COLUMNS])
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or not all(isinstance(name, str) and name in frame for name in names)):
        raise HTTPException(400, 'Saved feature schema is invalid')
    return names


def _model_manifests(path: Path):
    models = _contained(path, 'models')
    if not models.is_dir():
        return
    for child in sorted(models.iterdir()):
        if not child.is_dir() or not SAFE_ID.fullmatch(child.name):
            continue
        try:
            directory = _contained(models, child.name)
            payload = _contained(directory, 'bundle.joblib')
            manifest = _json(_contained(directory, 'manifest.json'))
        except HTTPException:
            continue
        if payload.is_file() and manifest:
            yield child.name, manifest


def _sample_metadata(row: pd.Series) -> dict:
    return {name: row.get(name) for name in ('run_id', 'group_id', 'window_id', 'split')}


def create_app(experiments_root: Path | None = None) -> FastAPI:
    root = Path(experiments_root or os.environ.get('FABFEDERATE_EXPERIMENTS_DIR')
                or Path(__file__).resolve().parents[2] / 'artifacts' / 'experiments').resolve()
    application = FastAPI(title='FabFederate local research API')

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, exc):
        # Pydantic's input/context may contain NaN or exception objects. Return
        # only concise locations and messages, never arbitrary submitted values.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': list(error['loc']), 'msg': error['msg']} for error in exc.errors()]})

    def experiment(identifier: str):
        path = _contained(root, _identifier(identifier))
        manifest = _json(_contained(path, 'manifest.json'))
        if (manifest.get('status') != 'complete'
                or not _contained(path, 'features.parquet').is_file()
                or not _contained(path, 'metrics.csv').is_file()):
            raise HTTPException(404, 'Complete experiment not found')
        return path, manifest

    def choices(manifest):
        config = manifest.get('config', {})
        runs = manifest.get('runs', [])
        scenarios = config.get('scenarios') or sorted({r['scenario'] for r in runs})
        seeds = config.get('seeds') or sorted({r['seed'] for r in runs})
        return {'scenarios': scenarios, 'seeds': seeds}

    @application.get('/api/health')
    def health():
        return {'status': 'ok'}

    @application.get('/api/experiments')
    def experiments():
        items = []
        for child in sorted(root.iterdir()) if root.is_dir() else []:
            if not child.is_dir() or not SAFE_ID.fullmatch(child.name):
                continue
            try:
                path, manifest = experiment(child.name)
                frame = _frame(path)
                if frame.empty or not _csv(_contained(path, 'metrics.csv')):
                    continue
                _feature_names(path, frame)
                items.append({'id': child.name, 'status': 'complete', **choices(manifest)})
            except HTTPException:
                continue
        default = next((preferred for preferred in ('nasa-optimized', 'nasa-final')
                        if any(x['id'] == preferred for x in items)),
                       items[0]['id'] if items else None)
        return _finite({'experiments': [{**item, 'default': item['id'] == default}
                                         for item in items], 'default_id': default})

    @application.get('/api/experiments/{identifier}')
    def detail(identifier: str):
        path, manifest = experiment(identifier)
        frame = _frame(path)
        names = _feature_names(path, frame)
        audit = _json(_contained(path, 'audit.json'))
        schema = _json(_contained(path, 'schema.json'))
        clients = _json(_contained(path, 'client_manifest.json')).get('clients', {})
        models = []
        for name, model in _model_manifests(path):
            metadata = model.get('metadata', {})
            models.append({'id': name, 'kind': model.get('model_kind'),
                'feature_names': model.get('feature_names'), 'wear_unit': model.get('wear_unit'),
                'selected_round': metadata.get('selected_round'),
                'validation_mae': metadata.get('validation_mae')})
        default = manifest.get('dashboard_default', {})
        config = manifest.get('config', {})
        choice = choices(manifest)
        result = {
            'id': identifier, 'status': manifest['status'], **choice,
            'default_scenario': default.get('scenario', config.get('scenario',
                choice['scenarios'][0] if choice['scenarios'] else None)),
            'default_seed': default.get('seed', config.get('seed',
                choice['seeds'][0] if choice['seeds'] else None)),
            'dataset': {
                'name': config.get('dataset'), 'run_count': audit.get('run_count'),
                'labeled_runs': audit.get('labeled_run_count'),
                'unlabeled_runs': audit.get('unlabeled_run_count'),
                'case_count': audit.get('independent_group_count'),
                'feature_count': len(names), 'window_count': len(frame),
                'wear_unit': schema.get('wear_unit', config.get('wear_unit')),
                'channels': [{'name': name, 'samples': channel.get('samples'),
                              'nonfinite_samples': channel.get('nonfinite_samples')}
                             for name, channel in audit.get('channels', {}).items()],
                'splits': [{'split': split, 'windows': len(rows),
                            'groups': rows.group_id.nunique()}
                           for split, rows in frame.groupby('split', sort=True)],
            },
            'models': models,
            'metrics': _csv(_contained(path, 'metrics.csv')),
            'rounds': _csv(_contained(path, 'rounds.csv')),
            'clients': [{'id': name, **{key: values.get(key) for key in (
                'training_groups', 'training_runs', 'training_windows')}}
                        for name, values in sorted(clients.items())],
            'provenance': {'source_sha256': manifest.get('source_sha256'),
                'code_sha256': manifest.get('executed_source', {}).get('sha256'),
                'completed_at': manifest.get('completed_at_utc')},
        }
        return _finite(result)

    @application.get('/api/experiments/{identifier}/samples')
    def samples(identifier: str, limit: Annotated[int, Query(ge=1, le=500)] = 100):
        path, _ = experiment(identifier)
        frame = _frame(path)
        heldout = frame[frame.split.isin(['validation', 'test'])].head(limit)
        schema = _json(_contained(path, 'schema.json'))
        return _finite({'samples': [
            {'id': str(index), **_sample_metadata(row), 'wear_target': row.get('wear_target'),
             'wear_unit': row.get('wear_unit', schema.get('wear_unit'))}
            for index, row in heldout.iterrows()], 'feature_names': _feature_names(path, frame)})

    @application.post('/api/experiments/{identifier}/predict')
    def predict(identifier: str, request: PredictionRequest):
        path, _ = experiment(identifier)
        model_id = _identifier(request.model)
        model_directory = _contained(_contained(path, 'models'), model_id)
        if model_id not in {name for name, _ in _model_manifests(path)}:
            raise HTTPException(404, 'Model not found in this experiment')
        try:
            bundle = load_bundle(model_directory)
        except (OSError, ValueError, KeyError):
            raise HTTPException(400, 'Saved model bundle is invalid') from None
        frame = _frame(path)
        names = _feature_names(path, frame)
        if bundle.feature_names != names:
            raise HTTPException(400, 'Selected model and experiment feature schemas differ')
        metadata, measured = None, None
        if request.sample_id is not None:
            if request.sample_id not in frame.index or frame.loc[request.sample_id, 'split'] not in (
                    'validation', 'test'):
                raise HTTPException(404, 'Held-out sample not found')
            row = frame.loc[request.sample_id]
            sample = frame.loc[[request.sample_id], names]
            metadata, measured = _sample_metadata(row), row.get('wear_target')
        else:
            if len(request.features) != len(names):
                raise HTTPException(422, f'Expected {len(names)} ordered feature values')
            sample = pd.DataFrame([request.features], columns=names)
        try:
            values = sample.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('features must be finite')
            predicted = float(bundle.predict(sample)[0])
            explanation = None
            if request.explain:
                background = frame.loc[frame.split.eq('train'), names]
                background = background[np.isfinite(background.to_numpy(dtype=float)).all(axis=1)]
                if background.empty:
                    raise ValueError('training reference unavailable')
                indices = np.linspace(0, len(background) - 1, min(16, len(background)), dtype=int)
                result = explain_prediction(bundle, background.iloc[indices], sample,
                                            max_background=16, max_evals=3 * (2 * len(names) + 1))
                explanation = {
                    'feature_names': result['feature_names'], 'values': result['values'][0],
                    'base_value': result['base_values'][0], 'prediction': result['predictions'][0],
                    'reconstruction_error': result['reconstruction_error'][0],
                    'runtime_seconds': result['runtime_seconds'],
                    'diagnostic': result['diagnostics'][0],
                }
        except (TypeError, ValueError):
            raise HTTPException(400, 'Prediction requires finite features and a valid training reference') from None
        training = frame.loc[frame.split.eq('train'), names]
        outside = (values[0] < training.min().to_numpy()) | (values[0] > training.max().to_numpy())
        outside_count = int(outside.sum())
        advisory = {'status': 'review_required' if outside_count else 'within_reference',
            'out_of_range_features': outside_count,
            'message': (f'{outside_count} features are outside the saved training minimum/maximum. '
                        'Verify acquisition and feature compatibility before using this estimate.'
                        if outside_count else 'Inputs lie within the saved training feature ranges. '
                        'Confirm wear by qualified measurement before an operational decision.'),
            'policy': 'Training-only feature min/max check; not calibrated uncertainty or a wear threshold.'}
        return _finite({'advisory': advisory, 'model': model_id, 'sample': metadata, 'predicted_wear': predicted,
            'measured_wear': measured, 'wear_unit': bundle.wear_unit,
            'features': [{'name': name, 'value': values[0, index]}
                         for index, name in enumerate(names)],
            'explanation': explanation, 'limitations': LIMITATIONS})

    @application.post('/api/experiments/{identifier}/measurements', status_code=201)
    def save_measurement(identifier: str, request: MeasurementRequest):
        path, _ = experiment(identifier)
        model_id = _identifier(request.model)
        manifests = dict(_model_manifests(path))
        if model_id not in manifests:
            raise HTTPException(404, 'Model not found in this experiment')
        if request.wear_unit != manifests[model_id].get('wear_unit'):
            raise HTTPException(422, 'Measurement unit must match the selected model')
        frame = _frame(path)
        if request.sample_id not in frame.index or frame.loc[request.sample_id, 'split'] not in ('test', 'validation'):
            raise HTTPException(404, 'Held-out sample not found')
        record = {'id': uuid4().hex, 'recorded_at': datetime.now(timezone.utc).isoformat(),
            'experiment': identifier, 'model': model_id,
            'bundle_sha256': manifests[model_id].get('sha256'),
            'sample_id': request.sample_id, 'source': _sample_metadata(frame.loc[request.sample_id]),
            'measured_wear': request.measured_wear, 'wear_unit': request.wear_unit,
            'note': request.note, 'status': 'operator_entered_pending_review',
            'used_for_training': False}
        directory = _contained(root, 'feedback')
        directory.mkdir(parents=True, exist_ok=True)
        target = _contained(directory, 'measurements.jsonl')
        descriptor = os.open(target, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            os.write(descriptor, (json.dumps(_finite(record), allow_nan=False) + '\n').encode())
        finally:
            os.close(descriptor)
        return {'id': record['id'], 'status': record['status'],
                'message': 'Measurement saved for review. Existing models and test labels are unchanged.'}

    @application.get('/api/experiments/{identifier}/report')
    def report(identifier: str):
        path, _ = experiment(identifier)
        report_path = _contained(path, 'report.md')
        if not report_path.is_file():
            raise HTTPException(404, 'Report not available')
        return FileResponse(report_path, media_type='text/markdown',
                            filename=f'fabfederate-{identifier}-report.md')

    return application


app = create_app()
