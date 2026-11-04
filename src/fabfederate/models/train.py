"""Baseline training without reading held-out test feature/target values."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fabfederate.contracts import METADATA_COLUMNS, PreparedData
from fabfederate.models.bundle import ModelBundle
from fabfederate.models.mlp import make_mlp, train_epochs

PREPROCESSING_DISCLOSURE = ('Common median imputer and standard scaler fit once on the pooled '
    'allowed public training partition, frozen and shared with central/local/federated models. '
    'This benchmark policy shares pooled training statistics and is not a private deployment protocol.')


def prepare_training(data: PreparedData):
    names = data.feature_names
    if not names or len(set(names)) != len(names) or set(names) & METADATA_COLUMNS:
        raise ValueError('invalid predictor feature allowlist')
    required = set(names) | {'split', 'client_id', 'wear_target', 'wear_unit'}
    if required - set(data.frame.columns):
        raise ValueError(f'missing training columns: {sorted(required - set(data.frame.columns))}')
    train = data.frame.loc[data.frame.split == 'train'].copy()
    validation = data.frame.loc[data.frame.split == 'validation'].copy()
    if train.empty or validation.empty:
        raise ValueError('nonempty train and validation partitions are required')
    if train.client_id.isna().any():
        raise ValueError('each train row requires a client_id')
    units = pd.concat([train.wear_unit, validation.wear_unit]).unique()
    if len(units) != 1 or pd.isna(units[0]) or not str(units[0]).strip():
        raise ValueError('train/validation must have one declared wear unit')
    for frame in (train, validation):
        values = frame[names].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('features must be finite or missing')
        if not np.isfinite(frame.wear_target.to_numpy(dtype=float)).all():
            raise ValueError('wear targets must be finite')
    if train[names].isna().all().any():
        raise ValueError('train feature is entirely missing')
    imputer = SimpleImputer(strategy='median')
    imputed = imputer.fit_transform(train[names].to_numpy(dtype=float))
    scaler = StandardScaler().fit(imputed)
    return train, validation, imputer, scaler, str(units[0])


def training_metadata(data, train, config):
    identity = train[[c for c in ['group_id', 'run_id', 'window_id'] if c in train]]
    return dict(schema_version=1, seed=int(config.get('seed', 42)), config=dict(config),
                preprocessing=PREPROCESSING_DISCLOSURE, train_examples=len(train),
                train_groups=int(train.group_id.nunique()) if 'group_id' in train else None,
                feature_hash=hashlib.sha256(json.dumps(data.feature_names).encode()).hexdigest(),
                split_hash=hashlib.sha256(identity.to_csv(index=False).encode()).hexdigest(),
                data_provenance=dict(data.metadata),
                architecture=dict(hidden_dims=config.get('hidden_dims', [64, 32]),
                                  dropout=config.get('dropout', .1)),
                optimizer='Adam; fresh state per centralized/local fit and each FL round',
                validation_selection='minimum heldout validation MAE; no test access')


def run_baselines(data: PreparedData, output_dir: Path, config: dict) -> dict[str, ModelBundle]:
    train, validation, imputer, scaler, unit = prepare_training(data)
    x = scaler.transform(imputer.transform(train[data.feature_names].to_numpy(dtype=float)))
    vx = scaler.transform(imputer.transform(validation[data.feature_names].to_numpy(dtype=float)))
    y, vy = train.wear_target.to_numpy(dtype=float), validation.wear_target.to_numpy(dtype=float)
    seed = int(config.get('seed', 42))
    estimators = {'mean': DummyRegressor(strategy='mean'),
                  'median': DummyRegressor(strategy='median'),
                  'random_forest': RandomForestRegressor(n_estimators=int(config.get('rf_trees', 100)),
                                                         random_state=seed, n_jobs=1)}
    bundles = {}
    for name, estimator in estimators.items():
        estimator.fit(x, y)
        metadata = training_metadata(data, train, config)
        metadata.update(validation_mae=float(np.mean(np.abs(estimator.predict(vx) - vy))),
                        selected_epoch=None, training_budget='one estimator fit',
                        rf_preprocessing='same frozen train-only imputer/scaler; scaling does not '
                        'alter tree split ordering' if name == 'random_forest' else None)
        bundles[name] = ModelBundle(data.feature_names, unit, name, estimator, imputer, scaler,
                                    metadata)
    indices = [('centralized_mlp', np.ones(len(train), dtype=bool))]
    for client_id in sorted(train.client_id.astype(str).unique()):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', client_id):
            raise ValueError('client_id must contain only letters, digits, underscores or hyphens')
        indices.append((f'local_{client_id}', train.client_id.astype(str).eq(client_id).to_numpy()))
    for name, subset in indices:
        model = make_mlp(len(data.feature_names), config)
        history, selected = train_epochs(model, x[subset], y[subset], config,
                                          config.get('epochs', 30), validation=(vx, vy))
        metadata = training_metadata(data, train.loc[subset], config)
        metadata.update(validation_mae=min(row['validation_mae'] for row in history),
                        selected_epoch=selected, training_budget=dict(epochs=config.get('epochs', 30),
                        batch_size=config.get('batch_size', 32)),
                        validation_scope='common heldout validation partition')
        bundles[name] = ModelBundle(data.feature_names, unit, name, model, imputer, scaler, metadata)
        target = Path(output_dir) / name
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(target / 'epochs.csv', index=False)
    for name, bundle in bundles.items():
        bundle.save(Path(output_dir) / name)
    return bundles
