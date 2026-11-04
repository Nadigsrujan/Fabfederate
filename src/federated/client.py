"""Private logical client state behind Flower's public client interfaces."""

import time

from flwr.client import NumPyClient
from flwr.common import DisconnectRes
from flwr.server.client_proxy import ClientProxy
import torch

from fabfederate.models.mlp import make_mlp, model_parameters, set_parameters, train_epochs


class WearClient(NumPyClient):
    def __init__(self, client_id, features, target, config, mu):
        self.client_id = client_id
        self.features = features
        self.target = target
        self.config = dict(config)
        self.mu = mu
        self.model = make_mlp(features.shape[1], config)

    def get_properties(self, config):
        return {'client_id': self.client_id, 'schema_version': 1}

    def get_parameters(self, config):
        return model_parameters(self.model)

    def fit(self, parameters, config):
        start = time.perf_counter()
        set_parameters(self.model, parameters)
        # The round/client seed is independent of client execution order.
        training_config = dict(self.config)
        training_config['seed'] = int(self.config.get('seed', 42)) + int(config['server_round'])
        torch.manual_seed(training_config['seed'])
        reference = [p.detach().clone() for p in self.model.parameters()]
        history, _ = train_epochs(self.model, self.features, self.target, training_config,
                                   self.config.get('local_epochs', 3), reference=reference, mu=self.mu)
        return model_parameters(self.model), len(self.target), {
            'train_loss': float(history[-1]['train_loss']),
            'duration_seconds': time.perf_counter() - start,
            'schema_version': 1,
        }


class InProcessClientProxy(ClientProxy):
    """Transport adapter only: Flower Server still coordinates/serializes real Fit messages."""

    def __init__(self, client_id, client):
        super().__init__(client_id)
        self._client = client.to_client()

    def get_properties(self, ins, timeout, group_id):
        return self._client.get_properties(ins)

    def get_parameters(self, ins, timeout, group_id):
        return self._client.get_parameters(ins)

    def fit(self, ins, timeout, group_id):
        return self._client.fit(ins)

    def evaluate(self, ins, timeout, group_id):
        return self._client.evaluate(ins)

    def reconnect(self, ins, timeout, group_id):
        return DisconnectRes(reason='in-process logical client')
