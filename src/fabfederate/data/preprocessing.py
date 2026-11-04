"""Leakage-safe run-only windows with explicit labels, schema and exclusions."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from fabfederate.contracts import METADATA_COLUMNS, PreparedData
from fabfederate.data.audit import audit_runs, is_labeled
from fabfederate.data.feature_extraction import extract_window_features
from fabfederate.data.partitioning import assign_clients
from fabfederate.data.splitting import build_group_splits


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_dataset(runs, output_dir: Path, config: dict) -> PreparedData:
    """Extract without fitting a scaler or selecting features from held-out data."""
    if not runs:
        raise ValueError("No source runs")
    window_size = config.get("window_size", 1024)
    stride = config.get("stride", window_size)
    if (not isinstance(window_size, int) or window_size < 4
            or not isinstance(stride, int) or stride < 1):
        raise ValueError("window_size must be an integer >=4; stride must be a positive integer")
    wear_unit = config.get("wear_unit")
    if not wear_unit or any(run.wear_unit != wear_unit for run in runs):
        raise ValueError("wear_unit must explicitly match every source-confirmed target unit")
    ids = [(run.dataset_id, run.run_id) for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate dataset/run identities")
    if len({run.dataset_id for run in runs}) != 1:
        raise ValueError("Prepare distinct sources separately; cross-source semantics require an explicit adapter")
    split_seed = config.get("split_seed", config.get("seed", 42))
    partition_seed = config.get("partition_seed", config.get("seed", 42))
    splits = build_group_splits(runs, seed=split_seed,
                               test_fraction=config.get("test_fraction", .2),
                               validation_fraction=config.get("validation_fraction", .2))
    labeled = [run for run in runs if is_labeled(run)]
    schemas = {tuple(sorted(run.signals)) for run in labeled}
    if len(schemas) != 1:
        raise ValueError("All labeled runs must share the same channel schema")
    rows, exclusions = [], []
    feature_names = None
    for run in sorted(runs, key=lambda value: value.run_id):
        if not is_labeled(run):
            continue
        if run.wear_target < 0:
            raise ValueError(f"Negative measured wear in {run.run_id}")
        arrays = {name: np.asarray(values, dtype=float) for name, values in run.signals.items()}
        lengths = {values.size for values in arrays.values()}
        if (not arrays or any(values.ndim != 1 for values in arrays.values()) or len(lengths) != 1):
            exclusions.append({"run_id": run.run_id, "reason": "invalid_channel_shape_or_lengths"})
            continue
        length = next(iter(lengths))
        if length < window_size:
            exclusions.append({"run_id": run.run_id, "reason": "short_run"})
            continue
        condition = json.dumps(run.settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
        sampling_hz = run.sampling_hz
        override_hz = config.get("sampling_hz")
        if override_hz is not None:
            if sampling_hz is not None and not np.isclose(override_hz, sampling_hz):
                raise ValueError("Configured sampling rate contradicts source-confirmed sampling rate")
            sampling_hz = override_hz
        if config.get("spectral", False) and (sampling_hz is None or sampling_hz <= 0):
            raise ValueError("Spectral extraction requires source-confirmed sampling rate")
        for index, start in enumerate(range(0, length - window_size + 1, stride)):
            try:
                features = extract_window_features(
                    {name: values[start:start + window_size] for name, values in arrays.items()},
                    spectral=config.get("spectral", False), sampling_hz=sampling_hz)
            except ValueError as error:
                exclusions.append({"run_id": run.run_id, "window_id": index, "reason": str(error)})
                continue
            names = list(features)
            if feature_names is None:
                feature_names = names
            if names != feature_names:
                raise ValueError("Inconsistent ordered feature schema")
            rows.append({"dataset_id": run.dataset_id, "run_id": run.run_id,
                         "group_id": run.group_id, "window_id": index,
                         "window_start": start, "window_end": start + window_size,
                         "wear_target": float(run.wear_target), "wear_unit": wear_unit,
                         "split": splits[run.group_id], "condition": condition, **features})
    if not rows:
        raise ValueError("No eligible finite nondegenerate labeled windows remain")
    if set(feature_names) & METADATA_COLUMNS:
        raise ValueError("Metadata cannot enter predictor schema")
    frame = pd.DataFrame(rows)
    for split in set(splits.values()):
        if not (frame.split == split).any():
            raise ValueError(f"No eligible windows in requested {split} split")
    frame, client_manifest = assign_clients(
        frame, scenario=config.get("scenario", "iid"), num_clients=config.get("num_clients", 3),
        seed=partition_seed, few_shot_groups=config.get("few_shot_groups", 1))
    metadata = {
        "schema_version": "1", "feature_names": feature_names,
        "feature_schema_sha256": hashlib.sha256(json.dumps(feature_names).encode()).hexdigest(),
        "wear_unit": wear_unit, "config": config,
        "excluded_unlabeled_runs": len(runs) - len(labeled),
        "feature_rows": len(frame), "independent_groups": int(frame.group_id.nunique()),
        "distinct_labeled_runs": int(frame.run_id.nunique()),
        "exclusions": exclusions,
        "exclusion_reason_counts": dict(Counter(entry["reason"] for entry in exclusions)),
        "definitions": {"variance": "population variance, ddof=0", "kurtosis": "Fisher excess, biased moment estimator",
                        "preprocessing": "none; amplitude preserved; common scaling must fit training rows only",
                        "spectral_power": "sum of one-sided unnormalized FFT squared magnitude / N^2; no doubling",
                        "label_alignment": "each run's original measured wear repeated for its windows; never interpolated"},
        "run_provenance": {run.run_id: run.provenance for run in runs},
        "split_counts": {str(key): int(value) for key, value in frame.split.value_counts().items()},
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "features.parquet", index=False)
    _write_json(output_dir / "schema.json", metadata)
    _write_json(output_dir / "split_manifest.json", {"seed": split_seed,
                                                    "group_splits": splits,
                                                    "policy": "source independent groups before run-only windowing"})
    _write_json(output_dir / "client_manifest.json", client_manifest)
    _write_json(output_dir / "audit.json", {**audit_runs(runs), "feature_preparation": metadata})
    return PreparedData(frame=frame, feature_names=feature_names, metadata=metadata)
