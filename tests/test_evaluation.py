import numpy as np
import pandas as pd
import pytest

from fabfederate.evaluation.metrics import evaluate_predictions, regression_metrics
from fabfederate.evaluation.report import generate_report


def test_pooled_rmse_uses_all_predictions_not_average_client_rmse():
    frame = pd.DataFrame({'wear_target': [0., 0., 0., 0.],
                          'prediction': [0., 0., 0., 4.],
                          'group_id': ['a', 'a', 'a', 'b'],
                          'client_id': ['one', 'one', 'one', 'two']})
    result = evaluate_predictions(frame)
    assert result['pooled']['rmse'] == 2
    assert result['pooled']['mae'] == 1
    assert result['group_mae'] == 2
    assert result['worst_client_mae'] == 4
    assert result['groups'] == 2
    assert result['windows'] == 4


@pytest.mark.parametrize('actual,predicted', [([], []), ([1], []), ([1], [np.nan]),
                                             ([np.inf], [0]), ([[1]], [[1]])])
def test_metrics_reject_invalid_inputs(actual, predicted):
    with pytest.raises(ValueError):
        regression_metrics(actual, predicted)


def test_r2_undefined_for_constant_or_single_target():
    assert regression_metrics([1], [3])['r2'] is None
    assert regression_metrics([2, 2], [2, 2])['r2'] is None
    metrics = regression_metrics([1, 2, 3], [1, 2, 3])
    assert metrics == {'mae': 0., 'rmse': 0., 'median_absolute_error': 0., 'r2': 1.}


def test_heldout_clients_are_unavailable_and_not_invented():
    frame = pd.DataFrame({'wear_target': [1., 2.], 'prediction': [1., 1.],
                          'group_id': ['a', 'b'], 'client_id': ['', None]})
    result = evaluate_predictions(frame)
    assert result['by_client'] == {}
    assert result['worst_client_mae'] is None
    assert not result['client_summary_available']


def test_mixed_units_and_missing_groups_rejected():
    frame = pd.DataFrame({'wear_target': [1., 2.], 'prediction': [1., 1.],
                          'group_id': ['a', 'b'], 'wear_unit': ['mm', 'um']})
    with pytest.raises(ValueError, match='unit'):
        evaluate_predictions(frame)
    frame = frame.drop(columns='wear_unit')
    frame.loc[0, 'group_id'] = None
    with pytest.raises(ValueError, match='group'):
        evaluate_predictions(frame)


def test_report_reads_artifacts_and_creates_figures(tmp_path):
    pd.DataFrame([{'method': 'fedavg', 'seed': 42, 'scenario': 'iid', 'split': 'test',
                   'mae': .125, 'rmse': .25, 'group_mae': .125, 'worst_client_mae': None,
                   'count': 2, 'groups': 2, 'units': 'mm', 'mu': 0}]).to_csv(
                       tmp_path / 'metrics.csv', index=False)
    pd.DataFrame({'wear_target': [1., 2.], 'prediction': [1., 1.75],
                  'group_id': ['a', 'b'], 'method': ['fedavg'] * 2,
                  'seed': [42] * 2, 'scenario': ['iid'] * 2, 'split': ['test'] * 2}).to_csv(
                      tmp_path / 'predictions.csv', index=False)
    report = generate_report(tmp_path)
    assert report.exists()
    assert '0.125' in report.read_text()
    assert 'Campus' in report.read_text()
    assert (tmp_path / 'figures' / 'measured_vs_predicted.png').exists()


def test_report_requires_real_metrics(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report(tmp_path)


def test_report_aggregates_seeds_and_bounds_visual_overview(tmp_path):
    rows = [{'method': 'fedavg', 'seed': seed, 'scenario': scenario, 'split': 'test',
             'mae': .1 * seed, 'rmse': .2 * seed, 'group_mae': .1 * seed,
             'worst_client_mae': None, 'count': 80, 'groups': 80, 'units': 'mm', 'mu': 0}
            for scenario in ['iid', 'condition_skew'] for seed in [1, 2]]
    pd.DataFrame(rows).to_csv(tmp_path / 'metrics.csv', index=False)
    predictions = [{'method': 'fedavg', 'seed': seed, 'scenario': scenario,
                    'split': 'test', 'wear_target': i / 100, 'prediction': (i + seed) / 100,
                    'window_id': f'w{i}', 'group_id': f'g{i}'}
                   for scenario in ['iid', 'condition_skew'] for seed in [1, 2] for i in range(80)]
    pd.DataFrame(predictions).to_csv(tmp_path / 'predictions.csv', index=False)
    report = generate_report(tmp_path).read_text()
    assert 'seed' in report.lower()
    assert 'iid' in report
    assert '40' in report
    assert len(pd.read_csv(tmp_path / 'group_errors.csv')) == 320
    assert (tmp_path / 'evaluation_summary.csv').exists()


def test_report_preserves_cut_identity_when_averaging_training_seeds(tmp_path):
    pd.DataFrame([{'method': 'fedavg', 'seed': seed, 'scenario': 'iid', 'split': 'test',
                   'mae': .1, 'rmse': .2, 'units': 'mm'} for seed in [1, 2]]).to_csv(
                       tmp_path / 'metrics.csv', index=False)
    # Window numbers restart per source cut, even within the same case/target.
    predictions = [dict(dataset_id='NASA', group_id='case1', run_id=cut,
                        window_id='window0', wear_target=.4, wear_unit='mm',
                        method='fedavg', scenario='iid', split='test', seed=seed,
                        prediction=value)
                   for cut, seed, value in [('cut1', 1, .1), ('cut1', 2, .3),
                                            ('cut2', 1, .6), ('cut2', 2, .8)]]
    pd.DataFrame(predictions).to_csv(tmp_path / 'predictions.csv', index=False)
    generate_report(tmp_path)
    points = pd.read_csv(tmp_path / 'overview_predictions.csv')
    assert len(points) == 2
    assert set(points.run_id) == {'cut1', 'cut2'}
    assert points.recorded_seeds.tolist() == [2, 2]
    np.testing.assert_allclose(points.sort_values('run_id').prediction, [.2, .7])


def test_report_preserves_derived_target_provenance(tmp_path):
    import json
    pd.DataFrame([{'method': 'mean', 'split': 'test', 'mae': .1, 'rmse': .12}]).to_csv(
        tmp_path / 'metrics.csv', index=False)
    (tmp_path / 'schema.json').write_text(json.dumps({
        'target_provenance': 'Publisher-smoothed derived wear targets, not independent measurements.'}))
    report = generate_report(tmp_path)
    assert 'Publisher-smoothed derived wear targets' in report.read_text()
