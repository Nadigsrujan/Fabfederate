"""Window-pooled errors and explicitly separate independent-group summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    """Return regression errors; constant/singleton targets have undefined R².

    Inputs must be nonempty, equally sized finite one-dimensional arrays. All
    errors retain the original target unit; RMSE is calculated over all rows.
    """
    actual, predicted = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError('Targets and predictions must be equally sized one-dimensional arrays')
    if actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError('Targets and predictions must be nonempty and finite')
    error = predicted - actual
    absolute = np.abs(error)
    squared_error = float(np.sum(error ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        'mae': float(absolute.mean()),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'median_absolute_error': float(np.median(absolute)),
        'r2': float(1 - squared_error / total) if len(actual) > 1 and total > 0 else None,
    }


def evaluate_predictions(frame: pd.DataFrame) -> dict:
    """Summarize one method/seed/scenario/split, without inventing client domains.

    A caller must select one evaluation stratum before calling. ``group_mae``
    weights each independent group equally; ``pooled`` weights each window
    equally. Client metrics describe only rows carrying a supported client ID.
    """
    required = {'wear_target', 'prediction', 'group_id'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing evaluation columns: {sorted(required - set(frame.columns))}')
    for column in ['method', 'seed', 'scenario', 'split', 'dataset_id']:
        if column in frame and frame[column].nunique(dropna=False) > 1:
            raise ValueError(f'Select a single {column} before evaluating predictions')
    if 'wear_unit' in frame and frame['wear_unit'].nunique(dropna=False) > 1:
        raise ValueError('Mixed wear units cannot be pooled')
    if frame['group_id'].isna().any() or frame['group_id'].astype(str).str.strip().eq('').any():
        raise ValueError('Every prediction requires an independent group_id')
    pooled = regression_metrics(frame['wear_target'], frame['prediction'])

    def summaries(data, column):
        return {str(key): {**regression_metrics(part.wear_target, part.prediction),
                           'count': int(len(part)), 'groups': int(part.group_id.nunique())}
                for key, part in data.groupby(column, sort=True)}

    by_group = summaries(frame, 'group_id')
    if 'client_id' in frame:
        present = frame.client_id.notna() & frame.client_id.astype(str).str.strip().ne('')
        by_client = summaries(frame.loc[present], 'client_id')
        client_count = int(present.sum())
    else:
        by_client, client_count = {}, 0
    return {
        'pooled': pooled,
        'group_mae': float(np.mean([value['mae'] for value in by_group.values()])),
        'worst_client_mae': max((v['mae'] for v in by_client.values()), default=None),
        'by_group': by_group,
        'by_client': by_client,
        'client_summary_available': bool(by_client),
        'client_covered_windows': client_count,
        'count': int(len(frame)),
        'windows': int(len(frame)),
        'groups': int(frame.group_id.nunique()),
    }
