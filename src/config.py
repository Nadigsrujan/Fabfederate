"""Validated flat configuration shared by CLI and research modules."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

DEFAULTS = {
    "dataset": "nasa",
    "phm_edge_reduction": None,
    "data_path": "data/raw/nasa/3. Milling/mill.mat",
    "output_dir": "artifacts/experiments/nasa-study",
    "window_size": 1024,
    "stride": 1024,
    "sampling_hz": 250.0,
    "wear_unit": "mm",
    "seed": 42,
    "split_seed": 42,
    "partition_seed": 42,
    "seeds": [11, 22, 33],
    "test_fraction": 0.2,
    "validation_fraction": 0.2,
    "num_clients": 3,
    "scenario": "iid",
    "scenarios": ["iid", "condition", "quantity", "label", "few_shot"],
    "few_shot_groups": 1,
    "spectral": False,
    "hidden_dims": [64, 32],
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 90,
    "local_epochs": 3,
    "rounds": 30,
    "loss": "mse",
    "rf_trees": 100,
    "mu_grid": [0.0, 0.001, 0.01, 0.1],
    "explanation_samples": 3,
    "max_background": 16,
    "robustness_noise": 0.05,
}


def validate_config(config: dict) -> dict:
    if config["dataset"] not in {"nasa", "phm2010"}:
        raise ValueError("dataset must be nasa or phm2010")
    if config["dataset"] == "phm2010" and config["phm_edge_reduction"] not in {"mean", "max"}:
        raise ValueError("PHM requires an explicit mean/max edge-wear reduction")
    for key in ("dropout", "learning_rate", "test_fraction", "validation_fraction", "robustness_noise"):
        if not isinstance(config[key], (int, float)) or not math.isfinite(config[key]):
            raise ValueError(f"{key} must be finite numeric data")
    for key in ("window_size", "stride", "epochs", "local_epochs", "rounds", "batch_size",
                "rf_trees", "few_shot_groups", "max_background", "explanation_samples"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["num_clients"], int) or config["num_clients"] < 2:
        raise ValueError("num_clients must be at least 2")
    fractions = [config["test_fraction"], config["validation_fraction"]]
    if any(not 0 < f < 1 for f in fractions) or sum(fractions) >= 0.8:
        raise ValueError("train/validation/test require nontrivial, disjoint group budgets")
    if not 0 <= config["dropout"] < 1 or config["learning_rate"] <= 0:
        raise ValueError("invalid neural optimizer/dropout settings")
    if not config["hidden_dims"] or any(not isinstance(n, int) or n < 1 for n in config["hidden_dims"]):
        raise ValueError("hidden_dims must contain positive integers")
    if config["loss"] not in {"mse", "huber"}:
        raise ValueError("loss must be mse or huber")
    if not config["mu_grid"] or any(not isinstance(mu, (int, float)) or not math.isfinite(mu)
                                    or mu < 0 for mu in config["mu_grid"]):
        raise ValueError("mu_grid must contain nonnegative strengths")
    if (not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"])
            or any(not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32
                   for seed in config["seeds"])):
        raise ValueError("seeds must be a nonempty list of unique integers")
    allowed = {"iid", "condition", "quantity", "label", "few_shot"}
    if (not config["scenarios"] or config["scenario"] not in allowed
            or not set(config["scenarios"]).issubset(allowed)):
        raise ValueError("unsupported client scenario")
    if config["spectral"] and not config["sampling_hz"]:
        raise ValueError("Spectral features require a verified sampling rate")
    if not config["wear_unit"]:
        raise ValueError("Wear unit must be explicitly recorded")
    return config


def load_config(path: str | Path | None = None) -> dict:
    config = deepcopy(DEFAULTS)
    if path is not None:
        overrides = yaml.safe_load(Path(path).read_text())
        if not isinstance(overrides, dict):
            raise ValueError("Configuration must be a YAML mapping")
        unknown = set(overrides) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        config.update(overrides)
    return validate_config(config)
