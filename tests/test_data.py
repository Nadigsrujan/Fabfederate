"""Tiny MATLAB structs exercise parsing; they are not experimental datasets."""

import numpy as np
import pytest
from scipy.io import savemat

from fabfederate.data.audit import audit_runs
from fabfederate.data.nasa_loader import NASA_CHANNELS, load_nasa


def write_mat(path, **changes):
    rows = []
    for run, target in [(1, .0), (2, np.nan), (3, .2)]:
        row = {"case": 1, "run": run, "VB": target, "time": run * 2,
               "DOC": 1.5, "feed": .5, "material": 1,
               **{channel: np.arange(8.) for channel in NASA_CHANNELS}, **changes}
        rows.append(row)
    savemat(path, {"mill": np.array(rows, dtype=object)})


def test_matlab_loader_keeps_source_group_and_missing_labels(tmp_path):
    path = tmp_path / "mill.mat"
    write_mat(path)
    runs = load_nasa(path)
    assert len(runs) == 3
    assert len({run.group_id for run in runs}) == 1
    assert len({run.run_id for run in runs}) == 3
    assert runs[0].wear_target == 0
    assert runs[1].wear_target is None
    assert runs[2].wear_target == .2
    assert runs[0].wear_unit == "mm"
    assert runs[0].sampling_hz == 250
    assert runs[0].settings["material"] == 1
    assert len(runs[0].provenance["source_sha256"]) == 64
    audit = audit_runs(runs)
    assert audit["run_count"] == 3
    assert audit["independent_group_count"] == 1
    assert audit["unlabeled_run_count"] == 1


def test_loader_rejects_invalid_source_identity(tmp_path):
    path = tmp_path / "mill.mat"
    write_mat(path, case=0)
    with pytest.raises(ValueError, match="case"):
        load_nasa(path)


def test_loader_rejects_missing_channel_and_ambiguous_directories(tmp_path):
    savemat(tmp_path / "mill.mat", {"mill": {"case": 1, "run": 1, "VB": .2}})
    with pytest.raises(ValueError, match="fields"):
        load_nasa(tmp_path)


def phm_manifest(tmp_path, cutter="c1"):
    """Explicit synthetic seven-channel files and measured-edge table mapping."""
    import json
    import pandas as pd
    for cut in (1, 2):
        pd.DataFrame(np.arange(56.).reshape(8, 7) + cut).to_csv(
            tmp_path / f"cut{cut}.csv", index=False, header=False)
    pd.DataFrame({"cut": [1], "edge1": [30.], "edge2": [60.], "edge3": [90.]}).to_csv(
        tmp_path / "wear.csv", index=False)
    manifest = {"schema_version": 1, "source_url": "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/",
                "wear_unit": "10^-3 mm", "signal_header": None,
                "signal_columns": {name: i for i, name in enumerate(
                    ["force_x", "force_y", "force_z", "vibration_x", "vibration_y", "vibration_z", "ae_rms"])},
                "cutters": [{"cutter_id": cutter, "label_file": "wear.csv", "label_header": 0,
                             "label_columns": {"cut_id": "cut", "flute_1": "edge1", "flute_2": "edge2", "flute_3": "edge3"},
                             "records": [{"cut_id": cut, "signal_file": f"cut{cut}.csv"} for cut in (1, 2)]}]}
    path = tmp_path / "phm.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_phm_explicit_mapping_preserves_edges_and_groups_and_missing_measurements(tmp_path):
    from fabfederate.data.phm_loader import load_phm
    path, _ = phm_manifest(tmp_path)
    runs = load_phm(path, edge_reduction="max")
    assert len(runs) == 2
    assert {run.group_id for run in runs} == {"c1"}
    assert runs[0].wear_target == 90
    assert runs[0].wear_unit == "um"
    assert runs[0].sampling_hz == 50000
    assert runs[0].provenance["edge_wear"] == {"flute_1": 30., "flute_2": 60., "flute_3": 90.}
    assert runs[1].wear_target is None
    assert runs[0].provenance["edge_reduction"] == "max"
    assert load_phm(path, edge_reduction="mean")[0].wear_target == 60
    assert audit_runs(runs)["independence_limit"].startswith("source cutter")


def test_phm_requires_explicit_target_policy_and_rejects_unreleased_labels(tmp_path):
    from fabfederate.data.phm_loader import load_phm
    path, _ = phm_manifest(tmp_path, cutter="c3")
    with pytest.raises(ValueError, match="edge_reduction"):
        load_phm(path)
    with pytest.raises(ValueError, match="unreleased|incorrect"):
        load_phm(path, edge_reduction="max")


def test_phm_unlabeled_test_signals_are_auditable_and_unit_override_rejected(tmp_path):
    import json
    from fabfederate.data.phm_loader import load_phm
    path, manifest = phm_manifest(tmp_path, cutter="c2")
    manifest["cutters"][0].pop("label_file")
    path.write_text(json.dumps(manifest))
    assert all(run.wear_target is None for run in load_phm(path, edge_reduction="mean"))
    manifest["wear_unit"] = "mm"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unit"):
        load_phm(path, edge_reduction="max")


def test_phm_duplicate_label_cut_and_manifest_path_escape_rejected(tmp_path):
    import json
    import pandas as pd
    from fabfederate.data.phm_loader import load_phm
    path, manifest = phm_manifest(tmp_path)
    pd.DataFrame({"cut": [1, 1], "edge1": [1, 2], "edge2": [2, 3], "edge3": [3, 4]}).to_csv(
        tmp_path / "wear.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_phm(path, edge_reduction="max")
    manifest["cutters"][0].pop("label_file")
    manifest["cutters"][0]["records"][0]["signal_file"] = "../outside.csv"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="within"):
        load_phm(path, edge_reduction="max")
