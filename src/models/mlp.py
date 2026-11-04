"""Small deterministic CPU MLP shared by all neural comparisons."""

import copy
import random

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_mlp(n_features: int, config: dict) -> nn.Sequential:
    seed_everything(int(config.get('seed', 42)))
    layers = []
    previous = n_features
    for width in config.get('hidden_dims', [64, 32]):
        if not isinstance(width, int) or width < 1:
            raise ValueError('hidden_dims must contain positive integers')
        layers.extend([nn.Linear(previous, width), nn.ReLU(),
                       nn.Dropout(float(config.get('dropout', .1)))])
        previous = width
    layers.append(nn.Linear(previous, 1))
    return nn.Sequential(*layers)


def model_parameters(model: nn.Module) -> list[np.ndarray]:
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    expected = list(model.parameters())
    if len(parameters) != len(expected):
        raise ValueError('parameter count mismatch')
    with torch.no_grad():
        for parameter, value in zip(expected, parameters):
            if parameter.shape != value.shape or not np.isfinite(value).all():
                raise ValueError('incompatible or nonfinite model parameters')
            parameter.copy_(torch.as_tensor(value, dtype=parameter.dtype))


def train_epochs(model, features, target, config, epochs, *, reference=None, mu=0.,
                 validation=None):
    """Fresh Adam per call/round; preserve best validation checkpoint when supplied."""
    from fabfederate.federated.fedprox import proximal_penalty
    if epochs < 1 or int(epochs) != epochs:
        raise ValueError('epochs must be positive integers')
    loss_name = config.get('loss', 'mse')
    if loss_name not in {'mse', 'huber'}:
        raise ValueError('loss must be mse or huber')
    criterion = nn.MSELoss() if loss_name == 'mse' else nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get('learning_rate', .001)))
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(target, dtype=torch.float32).reshape(-1, 1)
    batch_size = int(config.get('batch_size', 32))
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    generator = torch.Generator().manual_seed(int(config.get('seed', 42)))
    best, best_state, selected = float('inf'), None, 0
    history = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(x), generator=generator).split(batch_size):
            optimizer.zero_grad()
            prediction_loss = criterion(model(x[indices]), y[indices])
            objective = prediction_loss
            if reference is not None:
                objective = objective + proximal_penalty(model, reference, mu)
            if not torch.isfinite(objective):
                raise ValueError('nonfinite training loss')
            objective.backward()
            optimizer.step()
            total += float(prediction_loss.detach()) * len(indices)
        model.eval()
        score = None
        if validation is not None:
            vx, vy = validation
            with torch.no_grad():
                score = float(np.mean(np.abs(model(torch.tensor(vx, dtype=torch.float32))
                                             .numpy().ravel() - vy)))
            if score < best:
                best, best_state, selected = score, copy.deepcopy(model.state_dict()), epoch
        history.append(dict(epoch=epoch, train_loss=total / len(x), validation_mae=score))
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, selected
