"""Actual Flower Server/FedAvg orchestration with in-process logical clients.

This avoids Ray and network services on a laptop. It is logical partition isolation,
not process isolation, secure aggregation, or a privacy guarantee.
"""

import copy
import json
import math
from pathlib import Path
import time

import flwr
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import Server
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
import numpy as np
import pandas as pd
import torch

from fabfederate.contracts import PreparedData
from fabfederate.federated.client import InProcessClientProxy, WearClient
from fabfederate.federated.fedavg import aggregate_updates
from fabfederate.models.bundle import ModelBundle
from fabfederate.models.mlp import make_mlp, model_parameters, set_parameters
from fabfederate.models.train import prepare_training, training_metadata


class AuditedFedAvg(FedAvg):
    def __init__(self, *, expected_counts, records, mu, **kwargs):
        super().__init__(**kwargs)
        self.expected_counts = expected_counts
        self.records = records
        self.mu = mu
        self.started = {}
        self.download = {}

    def configure_fit(self, server_round, parameters, client_manager):
        self.started[server_round] = time.perf_counter()
        instructions = super().configure_fit(server_round, parameters, client_manager)
        # Fixed ordering and single-worker execution prevent global Torch RNG races.
        instructions.sort(key=lambda entry: entry[0].cid)
        self.download[server_round] = sum(len(p) for p in parameters.tensors) * len(instructions)
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != len(self.expected_counts):
            raise ValueError(f'incomplete Flower round {server_round}: {failures}')
        results = sorted(results, key=lambda entry: entry[0].cid)
        if len({client.cid for client, _ in results}) != len(self.expected_counts):
            raise ValueError('duplicate client updates')
        for client, result in results:
            if result.num_examples != self.expected_counts.get(client.cid):
                raise ValueError('client update unique-example count mismatch')
            if result.metrics.get('schema_version') != 1:
                raise ValueError('client update version mismatch')
            if not all(isinstance(v, (bool, int, float, str, bytes))
                       for v in result.metrics.values()):
                raise ValueError('client metadata must contain only scalars')
            if not math.isfinite(float(result.metrics['train_loss'])):
                raise ValueError('client training loss must be finite')
        # Strictly validate update dimensions/dtypes/counts before Flower performs FedAvg.
        aggregate_updates([(parameters_to_ndarrays(res.parameters), res.num_examples)
                           for _, res in results])
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        total = sum(res.num_examples for _, res in results)
        self.records.append(dict(round=server_round,
            train_loss=sum(res.num_examples * float(res.metrics['train_loss'])
                           for _, res in results) / total,
            validation_mae=None, duration_seconds=time.perf_counter() - self.started[server_round],
            upload_bytes=sum(sum(len(t) for t in res.parameters.tensors) for _, res in results),
            download_bytes=self.download[server_round], clients=len(results), mu=self.mu))
        return parameters, metrics


def run_federated(data: PreparedData, output_dir: Path, config: dict,
                  algorithm='fedavg', mu=0.0) -> ModelBundle:
    if algorithm not in {'fedavg', 'fedprox'}:
        raise ValueError('algorithm must be fedavg or fedprox')
    if not math.isfinite(mu) or mu < 0 or (algorithm == 'fedavg' and mu != 0):
        raise ValueError('mu must be finite/nonnegative and zero for fedavg')
    rounds = config.get('rounds', 30)
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError('rounds must be a positive integer')
    train, validation, imputer, scaler, unit = prepare_training(data)
    x = scaler.transform(imputer.transform(train[data.feature_names].to_numpy(dtype=float)))
    y = train.wear_target.to_numpy(dtype=float)
    vx = scaler.transform(imputer.transform(validation[data.feature_names].to_numpy(dtype=float)))
    vy = validation.wear_target.to_numpy(dtype=float)
    client_ids = sorted(train.client_id.astype(str).unique())
    if len(client_ids) < 2:
        raise ValueError('federation requires at least two nonempty clients')
    model = make_mlp(len(data.feature_names), config)
    initial = model_parameters(model)
    manager = SimpleClientManager()
    counts = {}
    for client_id in client_ids:
        selected = train.client_id.astype(str).eq(client_id).to_numpy()
        counts[client_id] = int(selected.sum())
        manager.register(InProcessClientProxy(client_id,
            WearClient(client_id, x[selected].copy(), y[selected].copy(), config, mu)))
    records = []
    best = dict(mae=float('inf'), state=None, round=None)

    def evaluate(server_round, parameters, _config):
        set_parameters(model, parameters)
        model.eval()
        with torch.no_grad():
            predicted = model(torch.tensor(vx, dtype=torch.float32)).numpy().ravel()
        score = float(np.mean(np.abs(predicted - vy)))
        # Round zero is measured but not selected: selection is among trained checkpoints.
        if server_round > 0:
            records[-1]['validation_mae'] = score
            records[-1]['duration_seconds'] = time.perf_counter() - strategy.started[server_round]
            if score < best['mae']:
                best.update(mae=score, state=copy.deepcopy(model.state_dict()), round=server_round)
        return float(np.mean((predicted - vy) ** 2)), {'validation_mae': score}

    strategy = AuditedFedAvg(expected_counts=counts, records=records, mu=mu,
        fraction_fit=1., fraction_evaluate=0., min_fit_clients=len(client_ids),
        min_available_clients=len(client_ids), min_evaluate_clients=0,
        initial_parameters=ndarrays_to_parameters(initial), evaluate_fn=evaluate,
        on_fit_config_fn=lambda round_number: {'server_round': round_number, 'schema_version': 1},
        accept_failures=False, inplace=False,
        fit_metrics_aggregation_fn=lambda metrics: {})
    server = Server(client_manager=manager, strategy=strategy)
    server.set_max_workers(1)
    server.fit(num_rounds=rounds, timeout=None)
    if len(records) != rounds or best['state'] is None:
        raise RuntimeError('Flower did not complete all requested rounds')
    model.load_state_dict(best['state'])
    metadata = training_metadata(data, train, config)
    metadata.update(validation_mae=best['mae'], selected_round=best['round'],
        algorithm=algorithm, mu=mu, framework='Flower', framework_version=flwr.__version__,
        transport='in-process ClientProxy; Flower Server.fit and FedAvg strategy',
        training_budget=dict(rounds=rounds, local_epochs=config.get('local_epochs', 3),
                             participation='full', optimizer_state='reset each round'),
        communication_accounting='serialized NumPy tensor bytes, excluding scalar/config/transport '
        'envelope; initial distribution accounted in round 1 download', client_counts=counts)
    bundle = ModelBundle(data.feature_names, unit, algorithm, model, imputer, scaler, metadata)
    target = Path(output_dir) / algorithm
    bundle.save(target)
    pd.DataFrame(records).to_csv(target / 'rounds.csv', index=False)
    (target / 'payload_manifest.json').write_text(json.dumps(dict(framework='Flower',
        framework_version=flwr.__version__, schema_version=1,
        fields=['parameters', 'num_examples', 'scalar_metrics'],
        excluded=['raw_signals', 'features', 'labels', 'rows'],
        total_train_examples=len(train), client_counts=counts,
        isolation='logical clients only; coordinator validation is an explicit separate role'),
        indent=2) + '\n')
    return bundle
