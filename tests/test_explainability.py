import json

import numpy as np
import pandas as pd
import pytest

from fabfederate.explainability.shap_explainer import (
    explain_prediction, save_explanation, sign_agreement, top_k_agreement,
)


class LinearBundle:
    feature_names = ['force_rms', 'vibration_rms']
    wear_unit = 'mm'
    model_kind = 'fixture_linear'
    metadata = {'version': 'test'}

    def __init__(self, offset=0.):
        self.offset = offset
        self.calls = 0

    def predict(self, frame):
        assert list(frame.columns) == self.feature_names
        self.calls += 1
        return frame.to_numpy() @ np.array([2., -1.]) + self.offset


def test_explanation_reconstructs_exact_selected_predictor_and_is_deterministic(tmp_path):
    background = pd.DataFrame([[0., 0.], [1., 1.]], columns=LinearBundle.feature_names)
    sample = pd.DataFrame([[3., 2.]], columns=LinearBundle.feature_names)
    bundle = LinearBundle(offset=7.)
    result = explain_prediction(bundle, background, sample)
    np.testing.assert_allclose(result['predictions'], [11.])
    np.testing.assert_allclose(np.array(result['base_values']) + np.sum(result['values'], axis=1),
                               result['predictions'], atol=1e-6)
    assert max(result['reconstruction_error']) < 1e-6
    assert bundle.calls > 0
    repeated = explain_prediction(bundle, background, sample)
    np.testing.assert_allclose(repeated['values'], result['values'])
    path = save_explanation(result, tmp_path / 'explanation.json')
    assert json.loads(path.read_text())['settings']['seed'] == 42


def test_background_and_sample_schema_and_budget_validation():
    frame = pd.DataFrame([[0., 0.]], columns=LinearBundle.feature_names)
    with pytest.raises(ValueError, match='schema'):
        explain_prediction(LinearBundle(), frame, frame[['vibration_rms', 'force_rms']])
    with pytest.raises(ValueError, match='max_evals'):
        explain_prediction(LinearBundle(), frame, frame, max_evals=1)
    with pytest.raises(ValueError, match='finite'):
        explain_prediction(LinearBundle(), frame, frame * np.nan)


def test_stability_with_known_attributions():
    assert top_k_agreement([3, 2, 1], [3, 1, 2], k=2) == .5
    assert sign_agreement([1, -1, 0], [2, 1, 0]) == pytest.approx(2/3)
    with pytest.raises(ValueError):
        top_k_agreement([1], [1], k=2)


@pytest.mark.parametrize('kind', ['random_forest', 'mlp'])
def test_saved_actual_estimator_and_transform_are_explained(kind, tmp_path):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    import torch
    from fabfederate.models.bundle import ModelBundle, load_bundle

    raw = np.array([[1., 4.], [3., 2.], [5., 7.], [6., 1.]])
    imputer = SimpleImputer().fit(raw)
    scaler = StandardScaler().fit(raw)
    transformed = scaler.transform(raw)
    if kind == 'random_forest':
        model = RandomForestRegressor(n_estimators=3, random_state=42).fit(transformed, [1, 2, 3, 4])
    else:
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2., -1.]]))
            model.bias.fill_(5.)
    bundle = ModelBundle(['a', 'b'], 'mm', kind, model, imputer, scaler)
    bundle.save(tmp_path / kind)
    restored = load_bundle(tmp_path / kind)
    frame = pd.DataFrame(raw, columns=['a', 'b'])
    result = explain_prediction(restored, frame.iloc[:3], frame.iloc[[3]])
    np.testing.assert_allclose(result['predictions'], restored.predict(frame.iloc[[3]]))
    assert max(result['reconstruction_error']) < 1e-5
    assert result['settings']['model_kind'] == kind


def test_zero_baseline_explanation_is_identified_and_explained():
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from fabfederate.models.bundle import ModelBundle
    frame = pd.DataFrame({'a': [0., 1., 2.], 'b': [1., 3., 2.]})
    x = frame.to_numpy()
    bundle = ModelBundle(list(frame), 'mm', 'mean',
        DummyRegressor().fit(x, [0., .3, .6]), SimpleImputer().fit(x),
        StandardScaler().fit(x))
    result = explain_prediction(bundle, frame, frame.iloc[[0]])
    diagnostic = result['diagnostics'][0]
    assert diagnostic['status'] == 'constant_baseline'
    assert diagnostic['influential_features'] == 0
    assert 'same' in diagnostic['summary'].lower()
    assert 'not a failed explanation' in diagnostic['summary'].lower()


def test_learned_explanation_has_a_grounded_summary():
    frame = pd.DataFrame([[0., 0.], [1., 1.]], columns=LinearBundle.feature_names)
    sample = pd.DataFrame([[3., 2.]], columns=LinearBundle.feature_names)
    result = explain_prediction(LinearBundle(), frame, sample)
    diagnostic = result['diagnostics'][0]
    assert diagnostic['status'] == 'attributed'
    assert diagnostic['influential_features'] == 2
    assert 'force rms' in diagnostic['summary']
