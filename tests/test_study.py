import json

import numpy as np
import pandas as pd
from scipy.io import savemat

from fabfederate.config import load_config
from fabfederate.pipelines.study import run_study


def test_complete_file_driven_workflow_produces_reloadable_real_computations(tmp_path):
    """Source-shaped fixture verifies wiring; its metrics are not NASA research evidence."""
    records = []
    rng = np.random.default_rng(3)
    for case in range(1, 13):
        for run in range(1, 4):
            record = {"case": case, "run": run, "VB": run * 0.1, "time": run,
                      "DOC": 0.75 if case % 2 else 1.5,
                      "feed": 0.25 if case % 3 else 0.5, "material": case % 2 + 1}
            for channel in ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle"):
                record[channel] = rng.normal(run, 0.2, 64)
            records.append(record)
    source = tmp_path / "fixture.mat"
    savemat(source, {"mill": records})
    config = load_config()
    config.update(data_path=str(source), output_dir=str(tmp_path / "study"),
                  seeds=[7], scenarios=["iid"], window_size=32, stride=32,
                  rounds=1, local_epochs=1, epochs=2, rf_trees=5, mu_grid=[0.0])
    output = run_study(config, explain=False)
    assert (output / "report.md").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    metrics = pd.read_csv(output / "metrics.csv")
    assert {"fedavg", "fedprox", "centralized_mlp"}.issubset(set(metrics.method))
    assert set(metrics.split) == {"validation", "test"}
    assert np.isfinite(metrics.mae).all()
    from fabfederate.models.bundle import load_bundle
    features = pd.read_parquet(output / "features.parquet")
    bundle = load_bundle(output / "models" / "fedavg")
    prediction = bundle.predict(features[features.split == "test"].iloc[:1])
    assert len(prediction) == 1 and np.isfinite(prediction).all()
