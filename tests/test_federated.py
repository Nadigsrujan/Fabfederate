import json

import numpy as np
import pandas as pd
import pytest
import torch

from fabfederate.federated.fedavg import aggregate_updates
from fabfederate.federated.fedprox import proximal_penalty
from fabfederate.federated.server import run_federated
from fabfederate.models.bundle import load_bundle
from test_models import tiny_data


def test_weighted_aggregation():
    updates = [([np.array([1., 3.], dtype=np.float32)], 1),
               ([np.array([5., 7.], dtype=np.float32)], 3)]
    result = aggregate_updates(updates)
    np.testing.assert_array_equal(result[0], np.array([4., 6.], dtype=np.float32))
    assert result[0].dtype == np.float32


@pytest.mark.parametrize('updates', [[], [([np.array([1.])], 0)],
    [([np.array([1.])], -1)], [([np.array([1.])], 1.5)],
    [([np.array([np.inf])], 1)],
    [([np.array([1.])], 1), ([np.array([1., 2.])], 1)],
    [([np.array([1.])], 1), ([np.array([1.], dtype=np.float32)], 1)],
    [([np.array([1.])], True)]])
def test_invalid_updates(updates):
    with pytest.raises(ValueError):
        aggregate_updates(updates)


def test_proximal_penalty():
    model = torch.nn.Linear(2, 1)
    reference = [p.detach().clone() for p in model.parameters()]
    assert proximal_penalty(model, reference, 1).item() == 0
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert proximal_penalty(model, reference, 1).item() > 0
    assert proximal_penalty(model, reference, 0).item() == 0


def test_real_flower_three_clients_and_mu_zero_equivalence(tmp_path):
    data = tiny_data()
    config = dict(rounds=2, local_epochs=1, hidden_dims=[4], seed=3, batch_size=8)
    fedavg = run_federated(data, tmp_path, config)
    fedprox = run_federated(data, tmp_path, config, algorithm='fedprox', mu=0.)
    validation = data.frame[data.frame.split == 'validation']
    np.testing.assert_array_equal(fedavg.predict(validation), fedprox.predict(validation))
    restored = load_bundle(tmp_path / 'fedavg')
    np.testing.assert_array_equal(restored.predict(validation), fedavg.predict(validation))
    rounds = pd.read_csv(tmp_path / 'fedavg' / 'rounds.csv')
    assert list(rounds['round']) == [1, 2]
    assert (rounds.clients == 3).all()
    assert (rounds.upload_bytes > 0).all()
    assert (rounds.download_bytes > 0).all()
    manifest = json.loads((tmp_path / 'fedavg' / 'payload_manifest.json').read_text())
    assert manifest['framework'] == 'Flower'
    assert manifest['total_train_examples'] == 18
    assert manifest['fields'] == ['parameters', 'num_examples', 'scalar_metrics']


def test_positive_prox_and_test_values_do_not_change_flower_fit(tmp_path):
    data = tiny_data()
    config = dict(rounds=2, local_epochs=2, hidden_dims=[4], seed=3, batch_size=8)
    first = run_federated(data, tmp_path / 'a', config, algorithm='fedprox', mu=.2)
    data.frame.loc[data.frame.split == 'test', ['x', 'z', 'wear_target']] = np.nan
    second = run_federated(data, tmp_path / 'b', config, algorithm='fedprox', mu=.2)
    validation = data.frame[data.frame.split == 'validation']
    np.testing.assert_array_equal(first.predict(validation), second.predict(validation))
    assert first.metadata['mu'] == .2
    assert first.metadata['selected_round'] in {1, 2}
