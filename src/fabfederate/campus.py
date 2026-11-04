"""Gated read-only validation of already recorded, approved campus feature exports."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fabfederate.contracts import METADATA_COLUMNS
from fabfederate.evaluation.metrics import regression_metrics
from fabfederate.models.bundle import load_bundle
from fabfederate.utils import file_hash, write_json


def validate_recordings(manifest_path: Path, model_path: Path, output_dir: Path) -> Path:
    path = Path(manifest_path)
    spec = json.loads(path.read_text())
    approval = spec.get("approval_record")
    if not approval or not (path.parent / approval).is_file():
        raise ValueError("A real supervisor approval record must be supplied before campus validation")
    if spec.get("read_only") is not True or spec.get("preprocessing_parity_confirmed") is not True:
        raise ValueError("Read-only acquisition and preprocessing parity must be explicitly confirmed")
    bundle = load_bundle(model_path)
    if spec.get("wear_unit") != bundle.wear_unit:
        raise ValueError("Campus wear unit differs from benchmark model")
    expected_hz = bundle.metadata.get("config", {}).get("sampling_hz")
    if not spec.get("sampling_hz") or (expected_hz and spec["sampling_hz"] != expected_hz):
        raise ValueError("Sampling rate must match the recorded benchmark preprocessing")
    csv_path = path.parent / spec["feature_csv"]
    frame = pd.read_csv(csv_path)
    if frame.empty or not {"run_id", "group_id"}.issubset(frame):
        raise ValueError("Recorded exports require nonempty data and run/group identifiers")
    unknown = set(frame) - set(bundle.feature_names) - METADATA_COLUMNS - {"timestamp", "source_id"}
    if unknown:
        raise ValueError(f"Unexpected campus export columns: {sorted(unknown)}")
    inputs = frame[bundle.feature_names].copy()
    inputs["wear_unit"] = spec["wear_unit"]
    prediction = bundle.predict(inputs)
    transformed = bundle.transform(inputs)
    labeled = (pd.to_numeric(frame.wear_target, errors="coerce").notna()
               if "wear_target" in frame else pd.Series(False, index=frame.index))
    if labeled.any() and not np.isfinite(frame.loc[labeled, "wear_target"].to_numpy(dtype=float)).all():
        raise ValueError("Measured campus wear must be finite")
    accuracy = (regression_metrics(frame.loc[labeled, "wear_target"].to_numpy(dtype=float),
                                   prediction[labeled.to_numpy()]) if labeled.any() else None)
    report = {
        "dataset_id": spec.get("dataset_id", "campus_recordings"),
        "rows": len(frame), "groups": int(frame.group_id.nunique()),
        "labeled_rows": int(labeled.sum()), "accuracy": accuracy, "wear_unit": bundle.wear_unit,
        "source_sha256": file_hash(csv_path), "manifest_sha256": file_hash(path),
        "model_kind": bundle.model_kind,
        "fraction_abs_training_z_gt_5": float(np.mean(np.abs(transformed) > 5)),
        "shift_rule": "Absolute standardized feature exceeds 5 training standard deviations; descriptive only, not calibrated confidence",
        "interpretation": ("Measured-label shadow evaluation; no automatic machine action"
                           if labeled.any() else "unlabeled compatibility and input-shift demonstration; accuracy unavailable"),
        "approval_record_sha256": file_hash(path.parent / approval),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "validation.json", report)
    frame.assign(prediction=prediction).to_csv(output / "predictions.csv", index=False)
    return output
