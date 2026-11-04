import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from fabfederate.campus_cnc.validation import validate_recordings
from fabfederate.models.bundle import ModelBundle


def test_campus_adapter_requires_real_approval_evidence(tmp_path):
    manifest = tmp_path / "campus.json"
    manifest.write_text(json.dumps({"read_only": True}))
    with pytest.raises(ValueError, match="approval"):
        validate_recordings(manifest, tmp_path / "model", tmp_path / "result")


def test_unlabeled_campus_export_does_not_claim_accuracy(tmp_path):
    raw = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    imputer = SimpleImputer().fit(raw)
    scaler = StandardScaler().fit(raw)
    model = DummyRegressor().fit(scaler.transform(raw), [0.1, 0.2, 0.3])
    bundle = ModelBundle(["a", "b"], "mm", "mean", model, imputer, scaler,
                         {"config": {"sampling_hz": 250.0}})
    bundle.save(tmp_path / "model")
    (tmp_path / "approval.txt").write_text("Explicit synthetic test approval fixture")
    pd.DataFrame(raw, columns=["a", "b"]).assign(run_id="r1", group_id="g1").to_csv(
        tmp_path / "features.csv", index=False)
    spec = {"read_only": True, "approval_record": "approval.txt", "feature_csv": "features.csv",
            "wear_unit": "mm", "sampling_hz": 250.0, "preprocessing_parity_confirmed": True,
            "dataset_id": "synthetic_campus_test"}
    (tmp_path / "manifest.json").write_text(json.dumps(spec))
    output = validate_recordings(tmp_path / "manifest.json", tmp_path / "model", tmp_path / "result")
    result = json.loads((output / "validation.json").read_text())
    assert result["accuracy"] is None
    assert result["labeled_rows"] == 0
    assert "unlabeled" in result["interpretation"]
