"""Freeze source-group boundaries before generating correlated windows."""

import math

import numpy as np


def build_group_splits(runs, seed=42, test_fraction=.2, validation_fraction=.2):
    """Fractions refer to all groups; every requested split must be nonempty."""
    if (not 0 <= test_fraction < 1 or not 0 <= validation_fraction < 1
            or test_fraction + validation_fraction >= 1):
        raise ValueError("Split fractions must be nonnegative and sum to less than one")
    groups = sorted({run.group_id for run in runs})
    if any(not group for group in groups):
        raise ValueError("Independent groups must have nonempty identifiers")
    test_count = math.ceil(len(groups) * test_fraction)
    validation_count = math.ceil(len(groups) * validation_fraction)
    if len(groups) <= test_count + validation_count:
        raise ValueError("Too few independent groups for requested train/validation/test splits")
    shuffled = np.random.default_rng(seed).permutation(groups).tolist()
    return {group: ("test" if i < test_count else "validation"
                    if i < test_count + validation_count else "train")
            for i, group in enumerate(shuffled)}
