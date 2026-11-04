"""Strict sample-weighted neural parameter aggregation."""

import numpy as np


def aggregate_updates(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    if not updates or not updates[0][0]:
        raise ValueError('at least one nonempty update is required')
    reference = updates[0][0]
    for arrays, count in updates:
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count <= 0:
            raise ValueError('update counts must be positive integer unique example counts')
        if len(arrays) != len(reference):
            raise ValueError('update parameter count mismatch')
        for value, expected in zip(arrays, reference):
            if not isinstance(value, np.ndarray) or not isinstance(expected, np.ndarray):
                raise ValueError('updates must contain numpy arrays')
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('update shape/dtype mismatch')
            if value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError('updates must be finite floating point arrays')
    total = sum(int(count) for _, count in updates)
    return [sum(value[i].astype(np.float64) * (int(count) / total)
                for value, count in updates).astype(reference[i].dtype)
            for i in range(len(reference))]
