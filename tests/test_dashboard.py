from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'dashboard' / 'app.py'
SCREENS = ['Data Explorer', 'Experiment Comparison', 'Client Heterogeneity',
           'Convergence / Cost', 'Wear Estimate']


def test_empty_dashboard_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv('FABFEDERATE_EXPERIMENTS_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'FabFederate-CNC'
    assert any('experiment' in item.value.lower() for item in app.info)
    assert app.code


def test_all_screens_load_real_artifact_fixture(tmp_path, monkeypatch):
    monkeypatch.setenv('FABFEDERATE_EXPERIMENTS_DIR', str(tmp_path))
    experiment = tmp_path / 'fixture'
    experiment.mkdir()
    pd.DataFrame({'method': ['fedavg'], 'seed': [42], 'scenario': ['iid'],
                  'split': ['test'], 'mae': [.12], 'rmse': [.15], 'group_mae': [.12],
                  'worst_client_mae': [None], 'count': [2], 'groups': [2],
                  'units': ['mm'], 'mu': [0]}).to_csv(experiment / 'metrics.csv', index=False)
    pd.DataFrame({'group_id': ['a', 'b', 'c'], 'window_id': ['a1', 'b1', 'c1'],
                  'split': ['train', 'validation', 'test'], 'client_id': ['one', '', ''],
                  'wear_target': [.1, .2, .3], 'wear_unit': ['mm'] * 3,
                  'force_rms': [1., 2., 3.]}).to_parquet(experiment / 'features.parquet')
    pd.DataFrame({'method': ['fedavg', 'fedavg'], 'seed': [42, 42], 'scenario': ['iid', 'iid'],
                  'round': [1, 2], 'train_loss': [.4, .3], 'validation_mae': [.2, .12],
                  'duration_seconds': [.1, .2], 'upload_bytes': [200, 200],
                  'download_bytes': [200, 200]}).to_csv(experiment / 'rounds.csv', index=False)
    app = AppTest.from_file(str(APP)).run(timeout=30)
    for screen in SCREENS:
        app.sidebar.radio[0].set_value(screen).run(timeout=30)
        assert not app.exception, f'{screen}: {app.exception}'
        assert any(h.value == screen for h in app.header)


def test_wear_estimate_uses_saved_bundle_and_explains_it(tmp_path, monkeypatch):
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from fabfederate.models.bundle import ModelBundle

    monkeypatch.setenv('FABFEDERATE_EXPERIMENTS_DIR', str(tmp_path))
    experiment = tmp_path / 'valid-model'
    experiment.mkdir()
    frame = pd.DataFrame({'force_rms': [1., 2., 3.], 'wear_target': [.1, .2, .3],
                          'group_id': ['case1', 'case2', 'case3'],
                          'window_id': ['w1', 'w2', 'w3'],
                          'split': ['train', 'train', 'test'], 'wear_unit': ['mm'] * 3})
    frame.to_parquet(experiment / 'features.parquet')
    raw = frame.loc[:1, ['force_rms']].to_numpy()
    imputer, scaler = SimpleImputer().fit(raw), StandardScaler().fit(raw)
    model = DummyRegressor().fit(scaler.transform(raw), [.1, .2])
    ModelBundle(['force_rms'], 'mm', 'mean', model, imputer, scaler,
                {'source': 'unit-test-fixture'}).save(experiment / 'models' / 'mean')
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio[0].set_value('Wear Estimate').run(timeout=30)
    assert not app.exception
    assert app.metric[0].value == '0.1500 mm'
    assert app.metric[1].value == '0.3000 mm'
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert any('Reconstruction error' in item.value for item in app.caption)


def test_upload_validation_rejects_labels_reordering_and_nonfinite():
    import runpy
    import pytest
    validate = runpy.run_path(str(APP))['validate_upload']
    assert validate(b'a,b\n1,2\n', ['a', 'b']).shape == (1, 2)
    for content in [b'b,a\n2,1\n', b'a,b,wear_target\n1,2,.1\n',
                    b'a,b\nNaN,2\n', b'a,a\n1,2\n', b'a,b\n']:
        with pytest.raises(ValueError):
            validate(content, ['a', 'b'])


def test_grouped_bars_show_original_statistics_without_stack_or_scale_binding():
    import runpy
    make_bar = runpy.run_path(str(APP))['_bar_chart']
    source = pd.DataFrame({'group_id': ['case1'], 'min': [.07],
                           'median': [.6], 'max': [1.53]})
    spec = make_bar(source, 'group_id', ['min', 'median', 'max'],
                    x_title='Case', y_title='Measured wear (mm)').to_dict()
    assert spec['encoding']['y']['stack'] is None
    assert spec['encoding']['xOffset']['field'] == 'Series'
    assert 'params' not in spec
    assert 'yOffset' not in spec['encoding']
    values = spec['datasets'][spec['data']['name']]
    assert max(row['Value'] for row in values) == 1.53
    single = make_bar(source, 'group_id', ['max'], x_title='Case',
                      y_title='Measured wear (mm)').to_dict()
    assert 'xOffset' not in single['encoding']
    assert 'params' not in single


def test_line_charts_reject_empty_nonfinite_data_and_preserve_round_values():
    import runpy
    import pytest
    line = runpy.run_path(str(APP))['_line_chart']
    data = pd.DataFrame({'round': [1, 2], 'validation_mae': [.4, .2]})
    spec = line(data, 'validation_mae', y_title='Validation MAE').to_dict()
    assert spec['encoding']['x']['field'] == 'round'
    assert spec['encoding']['y']['field'] == 'validation_mae'
    assert 'params' not in spec
    with pytest.raises(ValueError, match='finite'):
        line(data.iloc[:0], 'validation_mae', y_title='Validation MAE')


def test_record_labels_distinguish_different_cuts_with_the_same_window_number():
    import runpy
    label = runpy.run_path(str(APP))['_record_label']
    frame = pd.DataFrame({'run_id': ['cut01', 'cut02'], 'group_id': ['case1', 'case1'],
                          'window_id': [0, 0], 'split': ['test', 'test']})
    assert label(frame, 0) != label(frame, 1)
    assert 'cut01' in label(frame, 0)


def test_switching_experiments_never_reuses_another_experiments_model(tmp_path, monkeypatch):
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from fabfederate.models.bundle import ModelBundle

    monkeypatch.setenv('FABFEDERATE_EXPERIMENTS_DIR', str(tmp_path))
    paths = []
    for name, target in [('a', .15), ('b', .8)]:
        directory = tmp_path / name
        directory.mkdir()
        paths.append(directory)
        frame = pd.DataFrame({'a': [1., 2., 3.], 'wear_target': [target] * 3,
                              'run_id': ['run1', 'run2', 'run3'],
                              'group_id': ['g1', 'g2', 'g3'], 'window_id': [0, 0, 0],
                              'split': ['train', 'train', 'test'], 'wear_unit': ['mm'] * 3})
        frame.to_parquet(directory / 'features.parquet')
        raw = frame.loc[:1, ['a']].to_numpy()
        imputer, scaler = SimpleImputer().fit(raw), StandardScaler().fit(raw)
        model = DummyRegressor().fit(scaler.transform(raw), [target, target])
        ModelBundle(['a'], 'mm', 'mean', model, imputer, scaler).save(directory / 'models' / 'mean')
    app = AppTest.from_file(str(APP)).run(timeout=30)
    app.sidebar.radio[0].set_value('Wear Estimate').run(timeout=30)
    assert app.metric[0].value == '0.1500 mm'
    app.sidebar.selectbox[0].set_value(paths[1]).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.metric[0].value == '0.8000 mm'
    assert app.metric[1].value == '0.8000 mm'
