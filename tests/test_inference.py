"""File inference keeps the CLI, model schema and train-only explanation boundary aligned."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fabfederate.inference.pipeline import explain_file, predict_file
from fabfederate.models.bundle import ModelBundle


@pytest.fixture
def files(tmp_path):
    features = np.array([[0., 1.], [1., 2.]])
    imputer = SimpleImputer().fit(features)
    scaler = StandardScaler().fit(imputer.transform(features))
    estimator = DummyRegressor(strategy='constant', constant=.25).fit(features, [.1, .4])
    ModelBundle(['x', 'y'], 'mm', 'mean', estimator, imputer, scaler).save(tmp_path / 'model')
    pd.DataFrame({'x': [0.], 'y': [1.]}).to_csv(tmp_path / 'input.csv', index=False)
    pd.DataFrame({'x': [0., 1., 999.], 'y': [1., 2., 999.],
                  'split': ['train', 'train', 'test']}).to_parquet(tmp_path / 'background.parquet')
    return tmp_path


def test_predict_file_uses_saved_transform_and_writes_units(files):
    output = predict_file(files / 'model', files / 'input.csv', files / 'nested/prediction.csv')
    result = pd.read_csv(output)
    assert result.to_dict('list') == {
        'predicted_wear': [.25], 'wear_unit': ['mm'], 'model': ['mean']}


def test_explain_file_excludes_test_reference_rows(files, monkeypatch):
    def explain(bundle, background, sample):
        assert background.to_numpy().max() == 2.
        assert sample.columns.tolist() == bundle.feature_names
        return {'prediction': .25}
    monkeypatch.setattr('fabfederate.inference.pipeline.explain_prediction', explain)
    output = explain_file(files / 'model', files / 'input.csv',
                          files / 'background.parquet', files / 'explanation.json')
    assert json.loads(output.read_text()) == {'prediction': .25}


def test_explain_file_requires_documented_train_reference(files):
    pd.DataFrame({'x': [0.], 'y': [1.], 'split': ['test']}).to_parquet(
        files / 'background.parquet')
    with pytest.raises(ValueError, match='train split'):
        explain_file(files / 'model', files / 'input.csv',
                     files / 'background.parquet', files / 'explanation.json')
