"""Synthetic group fixtures verify partition logic, not NASA performance."""

import numpy as np
import pandas as pd
import pytest

from fabfederate.contracts import SignalRun
from fabfederate.data.partitioning import assign_clients
from fabfederate.data.splitting import build_group_splits


def runs(count=15):
    return [SignalRun("fixture", f"r{i}-{j}", f"g{i}", {"v": np.arange(16.)},
                      i / 10, "mm", {"feed": i % 3})
            for i in range(count) for j in range(2)]


def test_splits_are_group_disjoint_order_independent_and_seeded():
    first = build_group_splits(runs(), seed=8)
    assert first == build_group_splits(list(reversed(runs())), seed=8)
    assert set(first.values()) == {"train", "validation", "test"}
    assert len(first) == 15
    assert first != build_group_splits(runs(), seed=9)


def test_infeasible_split_rejected():
    with pytest.raises(ValueError, match="groups"):
        build_group_splits(runs(2))
    with pytest.raises(ValueError, match="fraction"):
        build_group_splits(runs(), test_fraction=.8, validation_fraction=.4)


def feature_frame():
    return pd.DataFrame([
        {"group_id": f"g{i}", "run_id": f"r{i}", "split": split,
         "condition": f"feed={i % 3}", "wear_target": i / 10,
         "window_id": j}
        for i, split in enumerate(["train"] * 12 + ["validation"] * 3 + ["test"] * 3)
        for j in range(i % 3 + 1)
    ])


@pytest.mark.parametrize("scenario", ["iid", "condition", "quantity", "label", "few_shot"])
def test_clients_preserve_groups_and_holdout_mapping_ignores_labels(scenario):
    frame = feature_frame()
    assigned, manifest = assign_clients(frame, scenario=scenario)
    assert assigned.groupby("group_id").client_id.nunique().max() == 1
    assert set(assigned.loc[assigned.split == "train", "client_id"]) == {"client_0", "client_1", "client_2"}
    altered = frame.copy()
    altered.loc[altered.split != "train", "wear_target"] = 100000
    reassigned, _ = assign_clients(altered, scenario=scenario)
    heldout = frame.split != "train"
    assert assigned.loc[heldout, "client_id"].equals(reassigned.loc[heldout, "client_id"])
    assert manifest["scenario"] == scenario
    assert manifest["clients"]


def test_unknown_scenario_and_too_few_groups_rejected():
    with pytest.raises(ValueError, match="scenario"):
        assign_clients(feature_frame(), scenario="fictional")
    with pytest.raises(ValueError, match="groups"):
        assign_clients(feature_frame().query("group_id == 'g0'"))


def test_no_heldout_condition_match_has_no_invented_client():
    frame = feature_frame()
    frame.loc[frame.split != "train", "condition"] = "unknown"
    assigned, manifest = assign_clients(frame)
    assert assigned.loc[assigned.split != "train", "client_id"].isna().all()
    assert manifest["unassigned_heldout_groups"]


def test_ambiguous_iid_domain_has_no_heldout_owner():
    frame = feature_frame()
    frame["condition"] = "feed=constant"
    assigned, _ = assign_clients(frame)
    assert assigned.loc[assigned.split != "train", "client_id"].isna().all()


def test_few_shot_reduces_shared_training_union_and_freezes_domain_evaluation():
    frame = feature_frame()
    baseline, _ = assign_clients(frame, scenario="condition")
    limited, manifest = assign_clients(frame, scenario="few_shot", few_shot_groups=1)
    client = manifest["few_shot_client"]
    assert manifest["e4_status"] == "available"
    assert manifest["e4_test_groups"]
    assert len(manifest["excluded_training_groups"]) == 3
    assert limited[limited.split == "train"].group_id.nunique() == 9
    assert limited[(limited.split == "train") & (limited.client_id == client)].group_id.nunique() == 1
    assert not set(manifest["excluded_training_groups"]) & set(limited.group_id)
    pd.testing.assert_frame_equal(baseline[baseline.split != "train"], limited[limited.split != "train"])
    pd.testing.assert_frame_equal(
        baseline[(baseline.split == "train") & (baseline.client_id != client)],
        limited[(limited.split == "train") & (limited.client_id != client)])


def test_few_shot_reports_missing_supported_test_domain():
    frame = feature_frame()
    frame.loc[frame.split == "test", "condition"] = "unseen"
    _, manifest = assign_clients(frame, scenario="few_shot")
    assert manifest["e4_status"] == "infeasible"
    assert manifest["e4_test_groups"] == []
    assert manifest["e4_reason"]
