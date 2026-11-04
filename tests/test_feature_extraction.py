"""Feature math and exclusion tests use explicitly synthetic source-shaped runs."""

import json

import numpy as np
import pytest

from fabfederate.contracts import METADATA_COLUMNS, SignalRun
from fabfederate.data.preprocessing import prepare_dataset
from fabfederate.data.feature_extraction import extract_window_features


def test_feature_definitions_and_spectral_frequency():
    values = np.array([0., 1., 2., 3.])
    result = extract_window_features({"v": values})
    assert result["v__mean"] == 1.5
    assert result["v__rms"] == pytest.approx(np.sqrt(3.5))
    assert result["v__variance"] == 1.25
    assert result["v__std"] == pytest.approx(np.sqrt(1.25))
    assert result["v__peak_to_peak"] == 3
    assert result["v__kurtosis"] == pytest.approx(-1.36)
    sine = np.sin(2 * np.pi * 8 * np.arange(128) / 128)
    spectral = extract_window_features({"v": sine}, spectral=True, sampling_hz=128)
    assert spectral["v__dominant_hz"] == 8


@pytest.mark.parametrize("signal", [np.ones(8), np.array([1., np.nan, 3., 4.]), np.array([1.])])
def test_invalid_or_degenerate_windows_rejected(signal):
    with pytest.raises(ValueError):
        extract_window_features({"v": signal})


def test_spectral_requires_sampling_rate():
    with pytest.raises(ValueError, match="sampling"):
        extract_window_features({"v": np.arange(8.)}, spectral=True)


def test_preparation_never_crosses_runs_and_excludes_unknown_labels(tmp_path):
    runs = [SignalRun("fixture", f"r{i}", f"g{i}", {"v": np.arange(10.) + i},
                      i / 10 if i != 9 else None, "mm", {"feed": i % 3})
            for i in range(10)]
    data = prepare_dataset(runs, tmp_path, {"window_size": 4, "stride": 3,
                                           "wear_unit": "mm", "num_clients": 3})
    assert len(data.frame) == 27
    assert data.frame.groupby("group_id").split.nunique().max() == 1
    assert data.frame.window_end.max() <= 10
    assert set(data.feature_names).isdisjoint(METADATA_COLUMNS)
    assert data.frame[data.feature_names].to_numpy().min() >= -2
    assert "r9" not in set(data.frame.run_id)
    assert data.metadata["excluded_unlabeled_runs"] == 1
    assert data.frame.groupby("run_id").window_id.nunique().eq(3).all()
    for artifact in ["features.parquet", "schema.json", "audit.json", "split_manifest.json", "client_manifest.json"]:
        assert (tmp_path / artifact).exists()
    assert json.loads((tmp_path / "schema.json").read_text())["feature_names"] == data.feature_names


def test_preparation_rejects_unit_relabeling(tmp_path):
    runs = [SignalRun("fixture", f"r{i}", f"g{i}", {"v": np.arange(8.)}, .2, "mm")
            for i in range(9)]
    with pytest.raises(ValueError, match="unit"):
        prepare_dataset(runs, tmp_path, {"window_size": 4, "wear_unit": "inch"})


def test_partition_seed_does_not_change_frozen_group_splits(tmp_path):
    runs = [SignalRun("fixture", f"r{i}", f"g{i}", {"v": np.arange(8.)}, .2,
                      "mm", {"feed": i % 3}) for i in range(15)]
    config = {"window_size": 4, "wear_unit": "mm", "split_seed": 81, "partition_seed": 1}
    first = prepare_dataset(runs, tmp_path / "a", config)
    second = prepare_dataset(runs, tmp_path / "b", {**config, "partition_seed": 2})
    assert first.frame.split.equals(second.frame.split)
    assert not first.frame.client_id.equals(second.frame.client_id)


def test_spectral_centroid_uses_known_sampling_and_no_label_or_id_features(tmp_path):
    runs = [SignalRun("fixture", f"r{i}", f"g{i}", {"v": np.arange(8.)}, .2,
                      "mm", {"feed": i % 3}, sampling_hz=250) for i in range(10)]
    with pytest.raises(ValueError, match="sampling rate"):
        prepare_dataset(runs, tmp_path, {"window_size": 4, "wear_unit": "mm", "sampling_hz": 100000})
