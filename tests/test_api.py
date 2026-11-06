"""API boundary tests use tiny, explicitly synthetic saved experiments."""
import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fabfederate.models.bundle import ModelBundle


def make_experiment(root, name='nasa-final', *, prediction=.15, model='mean'):
    path = root / name
    path.mkdir()
    frame = pd.DataFrame({
        'force_rms': [1., 2., 10., 20.], 'vibration_rms': [2., 4., 6., 8.],
        'run_id': ['cut1', 'cut2', 'cut3', 'cut4'],
        'group_id': ['case1', 'case1', 'case2', 'case3'], 'window_id': [0, 0, 0, 0],
        'split': ['train', 'train', 'validation', 'test'],
        'wear_target': [.1, .2, .3, .4], 'wear_unit': ['mm'] * 4,
        'client_id': ['client_0'] * 4,
    })
    frame.to_parquet(path / 'features.parquet')
    names = ['force_rms', 'vibration_rms']
    raw = frame.loc[:1, names].to_numpy()
    imputer = SimpleImputer().fit(raw)
    scaler = StandardScaler().fit(raw)
    predictor = DummyRegressor(strategy='constant', constant=prediction).fit(raw, [.1, .2])
    ModelBundle(names, 'mm', model, predictor, imputer, scaler,
                {'selected_round': 2, 'validation_mae': .07}).save(path / 'models' / model)
    (path / 'manifest.json').write_text(json.dumps({
        'status': 'complete', 'config': {'dataset': 'synthetic', 'scenarios': ['iid'],
                                       'seeds': [11]},
        'dashboard_default': {'scenario': 'iid', 'seed': 11},
        'source_sha256': 'source-digest', 'executed_source': {'sha256': 'code-digest'},
        'completed_at_utc': '2026-11-04T12:00:00Z'}))
    (path / 'audit.json').write_text(json.dumps({
        'run_count': 5, 'labeled_run_count': 4, 'unlabeled_run_count': 1,
        'independent_group_count': 3,
        'channels': {'force': {'samples': 2400, 'nonfinite_samples': 2}}}))
    (path / 'schema.json').write_text(json.dumps({'feature_names': names, 'wear_unit': 'mm'}))
    (path / 'client_manifest.json').write_text(json.dumps({'clients': {
        'client_0': {'training_groups': 1, 'training_runs': 2, 'training_windows': 2}}}))
    pd.DataFrame([{'method': model, 'scenario': 'iid', 'seed': 11, 'split': 'test',
                   'mae': .2, 'worst_client_mae': np.nan}]).to_csv(path / 'metrics.csv', index=False)
    pd.DataFrame([{'method': model, 'round': 1, 'validation_mae': .1}]).to_csv(
        path / 'rounds.csv', index=False)
    return path


@pytest.fixture
def api(tmp_path):
    from fabfederate.api.app import create_app
    path = make_experiment(tmp_path)
    return TestClient(create_app(tmp_path)), tmp_path, path


def test_lists_only_complete_experiments_with_real_metrics_and_features(api):
    client, root, _ = api
    pending = make_experiment(root, 'pending')
    (pending / 'manifest.json').write_text('{"status":"running"}')
    incomplete = make_experiment(root, 'incomplete')
    (incomplete / 'features.parquet').unlink()
    assert client.get('/api/health').json() == {'status': 'ok'}
    assert client.get('/api/experiments').json() == {
        'default_id': 'nasa-final', 'experiments': [
            {'id': 'nasa-final', 'status': 'complete', 'scenarios': ['iid'],
             'seeds': [11], 'default': True}]}
    assert client.get('/api/experiments/pending').status_code == 404


def test_detail_uses_audited_counts_finite_metrics_and_safe_metadata(api):
    client, root, _ = api
    response = client.get('/api/experiments/nasa-final')
    assert response.status_code == 200
    detail = response.json()
    assert detail['dataset'] == {
        'name': 'synthetic', 'run_count': 5, 'labeled_runs': 4, 'unlabeled_runs': 1,
        'case_count': 3, 'feature_count': 2, 'window_count': 4, 'wear_unit': 'mm',
        'channels': [{'name': 'force', 'samples': 2400, 'nonfinite_samples': 2}],
        'splits': [{'split': 'test', 'windows': 1, 'groups': 1},
                   {'split': 'train', 'windows': 2, 'groups': 1},
                   {'split': 'validation', 'windows': 1, 'groups': 1}]}
    assert detail['metrics'][0]['worst_client_mae'] is None
    assert detail['models'][0] == {
        'id': 'mean', 'kind': 'mean', 'feature_names': ['force_rms', 'vibration_rms'],
        'wear_unit': 'mm', 'selected_round': 2, 'validation_mae': .07}
    assert detail['clients'] == [{'id': 'client_0', 'training_groups': 1,
                                  'training_runs': 2, 'training_windows': 2}]
    assert detail['provenance'] == {'source_sha256': 'source-digest',
        'code_sha256': 'code-digest', 'completed_at': '2026-11-04T12:00:00Z'}
    assert str(root) not in response.text


def test_samples_are_unambiguous_heldout_rows_and_predict_selected_bundle(api):
    client, _, _ = api
    samples = client.get('/api/experiments/nasa-final/samples').json()
    assert samples['feature_names'] == ['force_rms', 'vibration_rms']
    assert [row['split'] for row in samples['samples']] == ['validation', 'test']
    assert len({row['id'] for row in samples['samples']}) == 2
    sample = samples['samples'][1]
    response = client.post('/api/experiments/nasa-final/predict', json={
        'model': 'mean', 'sample_id': sample['id']})
    assert response.status_code == 200
    result = response.json()
    assert result['predicted_wear'] == pytest.approx(.15)
    assert result['measured_wear'] == .4
    assert result['sample'] == {'run_id': 'cut4', 'group_id': 'case3',
                                'window_id': 0, 'split': 'test'}
    assert result['features'] == [{'name': 'force_rms', 'value': 20.},
                                  {'name': 'vibration_rms', 'value': 8.}]
    assert result['explanation'] is None
    assert result['limitations']
    custom = client.post('/api/experiments/nasa-final/predict', json={
        'model': 'mean', 'features': [3., 4.]}).json()
    assert custom['sample'] is None and custom['measured_wear'] is None


@pytest.mark.parametrize('body', [
    {'model': 'mean'}, {'model': 'mean', 'features': [1.]},
    {'model': 'mean', 'features': [1., 2., 3.]},
    {'model': 'mean', 'features': [1., 2.], 'sample_id': '2'},
    {'model': 'mean', 'features': ['bad', 2.]},
    {'model': 'mean', 'features': [True, 2.]},
])
def test_rejects_invalid_prediction_schema(api, body):
    assert api[0].post('/api/experiments/nasa-final/predict', json=body).status_code == 422


def test_nonfinite_input_is_rejected_with_finite_json_error(api):
    response = api[0].post('/api/experiments/nasa-final/predict',
        content='{"model":"mean","features":[NaN,2]}',
        headers={'content-type': 'application/json'})
    assert response.status_code == 422
    assert 'NaN' not in response.text


def test_unknown_models_cannot_cross_experiment_or_traverse_paths(api):
    client, root, _ = api
    make_experiment(root, 'other', prediction=.8, model='other-model')
    for model in ['other-model', '../other/models/other-model', '/tmp/model']:
        assert client.post('/api/experiments/nasa-final/predict', json={
            'model': model, 'features': [1., 2.]}).status_code == 404
    other = client.post('/api/experiments/other/predict', json={
        'model': 'other-model', 'features': [1., 2.]}).json()
    assert other['predicted_wear'] == .8
    for experiment in ['..%2Fother', '%2e%2e', 'missing']:
        assert client.get(f'/api/experiments/{experiment}').status_code == 404
    for sample in ['0', '999', '../other']:
        assert client.post('/api/experiments/nasa-final/predict', json={
            'model': 'mean', 'sample_id': sample}).status_code == 404


def test_sample_limit_bounds_and_report_download(api):
    client, _, path = api
    for limit in [0, 501]:
        assert client.get(f'/api/experiments/nasa-final/samples?limit={limit}').status_code == 422
    assert len(client.get('/api/experiments/nasa-final/samples?limit=1').json()['samples']) == 1
    assert client.get('/api/experiments/nasa-final/report').status_code == 404
    (path / 'report.md').write_text('# Synthetic fixture report\n')
    response = client.get('/api/experiments/nasa-final/report')
    assert response.text == '# Synthetic fixture report\n'
    assert 'fabfederate-nasa-final-report.md' in response.headers['content-disposition']


def test_symlinked_experiment_and_bundle_escape_are_rejected(api, tmp_path_factory):
    client, root, path = api
    outside = tmp_path_factory.mktemp('outside')
    external = make_experiment(outside, 'external')
    (root / 'escaped').symlink_to(external, target_is_directory=True)
    assert client.get('/api/experiments/escaped').status_code == 404
    (path / 'models' / 'escaped').symlink_to(external / 'models' / 'mean', target_is_directory=True)
    assert client.post('/api/experiments/nasa-final/predict', json={
        'model': 'escaped', 'features': [1., 2.]}).status_code == 404


def test_explanation_is_exact_model_and_train_only_with_bounded_background(api):
    client, _, _ = api
    response = client.post('/api/experiments/nasa-final/predict', json={
        'model': 'mean', 'sample_id': '3', 'explain': True})
    assert response.status_code == 200
    explanation = response.json()['explanation']
    assert explanation['feature_names'] == ['force_rms', 'vibration_rms']
    assert explanation['base_value'] == pytest.approx(.15)
    assert explanation['values'] == pytest.approx([0., 0.])
    assert explanation['prediction'] == pytest.approx(.15)
    assert explanation['reconstruction_error'] < 1e-8


def test_empty_or_unreadable_tables_do_not_appear_as_complete_experiments(api):
    client, root, _ = api
    empty = make_experiment(root, 'empty')
    pd.DataFrame(columns=['method', 'mae']).to_csv(empty / 'metrics.csv', index=False)
    corrupt = make_experiment(root, 'corrupt')
    (corrupt / 'features.parquet').write_bytes(b'not a parquet file')
    assert [row['id'] for row in client.get('/api/experiments').json()['experiments']] == ['nasa-final']


def test_ambiguous_sample_indexes_are_rejected(api):
    client, _, path = api
    frame = pd.read_parquet(path / 'features.parquet')
    frame.index = [0, 0, 1, 2]
    frame.to_parquet(path / 'features.parquet')
    assert client.get('/api/experiments/nasa-final/samples').status_code == 400


def test_explanation_reference_is_bounded_and_train_only(api, monkeypatch):
    import fabfederate.api.app as module
    client, _, path = api
    frame = pd.read_parquet(path / 'features.parquet')
    extra_train = pd.concat([frame.iloc[:2]] * 20, ignore_index=True)
    frame = pd.concat([frame, extra_train], ignore_index=True)
    frame.to_parquet(path / 'features.parquet')
    actual_explain = module.explain_prediction
    references = []

    def observed_explain(bundle, background, samples, **kwargs):
        references.append(background.copy())
        return actual_explain(bundle, background, samples, **kwargs)

    monkeypatch.setattr(module, 'explain_prediction', observed_explain)
    result = client.post('/api/experiments/nasa-final/predict', json={
        'model': 'mean', 'sample_id': '3', 'explain': True})
    assert result.status_code == 200
    assert len(references[0]) == 16
    assert set(references[0].force_rms) <= {1., 2.}
    assert result.json()['explanation']['prediction'] == pytest.approx(.15)


def test_operator_measurement_is_saved_for_review_without_changing_labels(api):
    client, root, path = api
    original = pd.read_parquet(path / 'features.parquet').wear_target.tolist()
    response = client.post('/api/experiments/nasa-final/measurements', json={
        'model': 'mean', 'sample_id': '3', 'measured_wear': .19, 'wear_unit': 'mm'})
    assert response.status_code == 201
    record = json.loads((root/'feedback/measurements.jsonl').read_text())
    assert record['used_for_training'] is False
    assert record['status'] == 'operator_entered_pending_review'
    assert pd.read_parquet(path/'features.parquet').wear_target.tolist() == original
    assert client.post('/api/experiments/nasa-final/measurements', json={
        'model': 'mean', 'sample_id': '3', 'measured_wear': -1., 'wear_unit': 'mm'}).status_code == 422
    assert client.post('/api/experiments/nasa-final/measurements', json={
        'model': 'mean', 'sample_id': '3', 'measured_wear': .2, 'wear_unit': 'um'}).status_code == 422
