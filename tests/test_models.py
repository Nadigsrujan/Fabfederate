import json

import numpy as np
import pandas as pd
import pytest

from fabfederate.contracts import PreparedData
from fabfederate.models.bundle import load_bundle
from fabfederate.models.train import run_baselines


def tiny_data():
    rows = []
    for split, count in [('train', 18), ('validation', 6), ('test', 6)]:
        for i in range(count):
            rows.append(dict(x=float(i) if split != 'test' else 100000.,
                             z=float(i % 4), wear_target=float(i) / 10,
                             split=split, client_id=str(i % 3), wear_unit='mm',
                             group_id=f'{split}-{i}', run_id=f'{split}-{i}',
                             window_id=f'{split}-{i}', condition='one'))
    return PreparedData(pd.DataFrame(rows), ['x', 'z'], {'synthetic': True})


def test_baseline_preprocessing_persistence_and_schema(tmp_path):
    data = tiny_data()
    data.frame.loc[0, 'z'] = np.nan
    models = run_baselines(data, tmp_path, dict(epochs=2, hidden_dims=[4], rf_trees=3))
    assert set(models) == {'mean', 'median', 'random_forest', 'centralized_mlp',
                           'local_0', 'local_1', 'local_2'}
    bundle = models['centralized_mlp']
    assert bundle.scaler.mean_[0] == pytest.approx(8.5)
    assert bundle.metadata['train_examples'] == 18
    assert models['local_0'].metadata['train_examples'] == 6
    frame = data.frame[data.frame.split == 'validation']
    restored = load_bundle(tmp_path / 'centralized_mlp')
    np.testing.assert_allclose(restored.predict(frame), bundle.predict(frame), atol=1e-7)
    np.testing.assert_allclose(bundle.predict_array(frame[['x', 'z']].to_numpy()),
                               bundle.predict(frame))
    with pytest.raises(ValueError, match='missing'):
        bundle.predict(frame.drop(columns='x'))
    with pytest.raises(ValueError, match='unknown'):
        bundle.predict(frame.assign(unexpected=1))
    with pytest.raises(ValueError, match='unit'):
        bundle.predict(frame.assign(wear_unit='inches'))
    with pytest.raises(ValueError, match='finite'):
        bundle.predict(frame.assign(x=np.inf))
    manifest = json.loads((tmp_path / 'centralized_mlp' / 'manifest.json').read_text())
    assert 'sha256' in manifest
    payload = tmp_path / 'centralized_mlp' / 'bundle.joblib'
    payload.write_bytes(payload.read_bytes() + b'tamper')
    with pytest.raises(ValueError, match='hash'):
        load_bundle(tmp_path / 'centralized_mlp')


def test_missing_feature_imputation_and_invalid_training(tmp_path):
    data = tiny_data()
    data.frame.loc[data.frame.split == 'train', 'z'] = np.nan
    with pytest.raises(ValueError, match='entirely missing'):
        run_baselines(data, tmp_path, {'epochs': 1})


def test_test_rows_never_affect_fit_or_checkpoint(tmp_path):
    data = tiny_data()
    config = dict(epochs=2, hidden_dims=[4], rf_trees=3, seed=7)
    first = run_baselines(data, tmp_path / 'a', config)
    data.frame.loc[data.frame.split == 'test', ['x', 'z', 'wear_target']] = np.nan
    second = run_baselines(data, tmp_path / 'b', config)
    validation = data.frame[data.frame.split == 'validation']
    for name in first:
        np.testing.assert_array_equal(first[name].predict(validation),
                                      second[name].predict(validation))
