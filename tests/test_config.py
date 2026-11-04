from pathlib import Path

import pytest

from fabfederate.config import load_config, validate_config
from fabfederate.utils import stable_hash, write_json


def test_default_protocol_has_three_clients_and_no_hardware_dependency():
    config = load_config()
    assert config["num_clients"] == 3
    assert config["rounds"] == 30
    assert config["wear_unit"] == "mm"
    assert config["mu_grid"][0] == 0


@pytest.mark.parametrize("field,value", [("rounds", 0), ("local_epochs", -1),
    ("num_clients", 1), ("test_fraction", 0.9), ("mu_grid", [-0.1]),
    ("learning_rate", float("nan")), ("mu_grid", [float("inf")]),
    ("scenarios", []), ("seeds", ["eleven"])])
def test_invalid_experiment_is_rejected(field, value):
    config = load_config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_unknown_configuration_catches_typos(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("roudns: 3\n")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(config)


def test_json_provenance_is_deterministic_and_finite(tmp_path):
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"metric": float("nan")})
    assert not Path(tmp_path / "bad.json").exists()


def test_source_snapshot_changes_when_executed_code_changes(tmp_path):
    from fabfederate.utils import source_snapshot
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "model.py"
    source.write_text("loss = 'mse'\n")
    initial = source_snapshot(tmp_path)
    assert initial == source_snapshot(tmp_path)
    source.write_text("loss = 'huber'\n")
    assert initial["sha256"] != source_snapshot(tmp_path)["sha256"]
