"""Small shared contracts; source identifiers are never predictor inputs."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

METADATA_COLUMNS = {
    "dataset_id", "run_id", "group_id", "tool_id", "window_id", "wear_target",
    "wear_unit", "split", "client_id", "condition", "window_start", "window_end",
}


@dataclass
class SignalRun:
    dataset_id: str
    run_id: str
    group_id: str
    signals: dict[str, Any]
    wear_target: float | None
    wear_unit: str
    settings: dict[str, Any] = field(default_factory=dict)
    sampling_hz: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedData:
    frame: pd.DataFrame
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
