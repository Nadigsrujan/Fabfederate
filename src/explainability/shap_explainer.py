"""Deterministic permutation SHAP over the exact persisted raw-feature predictor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


def _features(frame, names, label):
    if not isinstance(frame, pd.DataFrame) or list(frame.columns) != list(names):
        raise ValueError(f'{label} feature schema/order differs from the selected bundle')
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label} features must be numeric and finite') from exc
    if values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError(f'{label} features must be nonempty and finite')
    return values


def explain_prediction(bundle, background: pd.DataFrame, samples: pd.DataFrame,
                       max_background=32, max_evals=None) -> dict:
    """Explain the selected bundle, including its saved raw-feature transform.

    Caller supplies training-only background. A fixed seed and evenly spaced
    deterministic training subset bound cost. Permutation SHAP explains MLP,
    RF and baselines through the same predictor; no surrogate is substituted.
    """
    names = list(bundle.feature_names)
    if not names:
        raise ValueError('Selected bundle has no features')
    background_values = _features(background, names, 'Background')
    values = _features(samples, names, 'Sample')
    if not isinstance(max_background, int) or max_background < 1:
        raise ValueError('max_background must be a positive integer')
    minimum = 2 * len(names) + 1
    budget = 3 * minimum if max_evals is None else max_evals
    if not isinstance(budget, int) or budget < minimum:
        raise ValueError(f'max_evals must be an integer >= {minimum}')
    indices = np.linspace(0, len(background_values) - 1,
                          min(max_background, len(background_values)), dtype=int)
    reference = background_values[indices]

    def predictor(raw):
        prediction = np.asarray(bundle.predict(pd.DataFrame(raw, columns=names)), dtype=float)
        if prediction.shape != (len(raw),) or not np.isfinite(prediction).all():
            raise ValueError('Selected predictor must return one finite wear estimate per row')
        return prediction

    start = perf_counter()
    import shap
    masker = shap.maskers.Independent(reference, max_samples=len(reference))
    explainer = shap.PermutationExplainer(predictor, masker, feature_names=names, seed=42)
    explanation = explainer(values, max_evals=budget, silent=True)
    predictions = predictor(values)
    attributions = np.asarray(explanation.values, dtype=float)
    base = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    if attributions.shape != values.shape or base.shape != predictions.shape:
        raise ValueError('Unexpected explanation dimensions for selected regression predictor')
    error = np.abs(base + attributions.sum(axis=1) - predictions)
    tolerance = 1e-5 * max(1., float(np.abs(predictions).max()))
    if not np.isfinite(attributions).all() or not np.isfinite(base).all() or np.any(error > tolerance):
        raise ValueError('SHAP reconstruction check failed for the selected predictor')
    reference_predictions = predictor(reference)
    reference_range = float(np.ptp(reference_predictions))
    diagnostics = []
    from sklearn.dummy import DummyRegressor
    constant = isinstance(getattr(bundle, 'model', None), DummyRegressor)
    unit = getattr(bundle, 'wear_unit', 'source units')
    for row, prediction, baseline in zip(attributions, predictions, base):
        active = np.flatnonzero(np.abs(row) > 1e-12)
        if constant:
            status = 'constant_baseline'
            summary = (f'This baseline returns the same {prediction:.4f} {unit} for every input. '
                       'It ignores sensor features, so all feature contributions are zero. '
                       'This is expected, not a failed explanation. Choose a learned model '
                       'such as FedAvg or Random forest to inspect sensor influence.')
        elif not len(active):
            status = 'no_local_effect'
            summary = ('No measurable feature contribution was found for this sample against '
                       'the selected training reference. The model may be locally flat; '
                       'this does not establish that the sensors have no physical effect.')
        else:
            status = 'attributed'
            top = int(np.argmax(np.abs(row)))
            label = names[top].replace('_', ' ')
            direction = 'raises' if row[top] > 0 else 'lowers'
            summary = (f'Compared with the training reference ({baseline:.4f} {unit}), '
                       f'{label} has the largest contribution and {direction} the estimate '
                       f'by {abs(row[top]):.4g} {unit}. '
                       f'{len(active)} features have measurable contributions. '
                       'These contributions explain this model output, not physical causation.')
        diagnostics.append({'status': status, 'summary': summary,
                            'influential_features': len(active),
                            'reference_prediction_range': reference_range})
    return {
        'diagnostics': diagnostics,
        'feature_names': names,
        'values': attributions.tolist(),
        'base_values': base.tolist(),
        'predictions': predictions.tolist(),
        'reconstruction_error': error.tolist(),
        'settings': {
            'explainer': 'PermutationExplainer', 'seed': 42, 'max_evals': budget,
            'background_rows': len(reference), 'background_source': 'caller-supplied training rows',
            'background_indices': indices.tolist(),
            'background_sha256': hashlib.sha256(reference.tobytes()).hexdigest(),
            'model_kind': getattr(bundle, 'model_kind', 'unknown'),
            'wear_unit': getattr(bundle, 'wear_unit', 'unknown'),
            'shap_version': shap.__version__,
            'reconstruction_tolerance': tolerance,
        },
        'runtime_seconds': perf_counter() - start,
    }


def save_explanation(explanation: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(explanation, indent=2, allow_nan=False) + '\n')
    return path


def _paired(a, b):
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not left.size:
        raise ValueError('Attributions must be equally sized nonempty vectors')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Attributions must be finite')
    return left, right


def top_k_agreement(a, b, k=5) -> float:
    """Fraction of absolute-magnitude top-k feature indices shared (stable ties)."""
    left, right = _paired(a, b)
    if not isinstance(k, int) or not 1 <= k <= len(left):
        raise ValueError('k must be an integer between 1 and the feature count')
    first = set(np.argsort(-np.abs(left), kind='stable')[:k])
    second = set(np.argsort(-np.abs(right), kind='stable')[:k])
    return len(first & second) / k


def sign_agreement(a, b) -> float:
    """Fraction of feature attribution signs shared; two zeros agree."""
    left, right = _paired(a, b)
    return float(np.mean(np.sign(left) == np.sign(right)))
