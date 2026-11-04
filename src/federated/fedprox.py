"""FedProx drift penalty against a frozen incoming global checkpoint."""

import math
import torch


def proximal_penalty(model, reference_parameters, mu):
    if not math.isfinite(mu) or mu < 0:
        raise ValueError('mu must be finite and nonnegative')
    parameters = list(model.parameters())
    reference = list(reference_parameters)
    if len(parameters) != len(reference):
        raise ValueError('reference parameter count mismatch')
    terms = []
    for parameter, incoming in zip(parameters, reference):
        incoming = torch.as_tensor(incoming, dtype=parameter.dtype, device=parameter.device).detach()
        if incoming.shape != parameter.shape:
            raise ValueError('reference parameter shape mismatch')
        terms.append(torch.sum((parameter - incoming) ** 2))
    return .5 * mu * torch.stack(terms).sum()
