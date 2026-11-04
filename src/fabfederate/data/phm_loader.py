"""Optional local PHM2010 adapter with explicit cut/edge mapping.

No archive downloading, filename-based label guessing, or challenge-test labels.
Official description confirms seven channels, 50 kHz, and wear in 10^-3 mm.
The wear table's actual headers are not assumed: the local manifest maps them.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fabfederate.contracts import SignalRun

PHM_SOURCE_URL = "https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/"
PHM_CHANNEL_UNITS = {"force_x": "N", "force_y": "N", "force_z": "N",
                     "vibration_x": "g", "vibration_y": "g", "vibration_z": "g", "ae_rms": "V"}
RELEASED_CUTTERS = {"c1", "c4", "c6"}
ALL_CUTTERS = {f"c{i}" for i in range(1, 7)}
EDGES = ("flute_1", "flute_2", "flute_3")


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local(root, filename):
    if not isinstance(filename, str) or not filename:
        raise ValueError("Manifest filenames must be nonempty local paths")
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError("PHM source paths must remain within the declared local root")
    if not path.is_file():
        raise ValueError(f"Missing PHM source file: {path}")
    return path


def _positive_cut(value):
    if isinstance(value, bool):
        raise ValueError("cut_id must be a positive source integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("cut_id must be a positive source integer") from error
    if not np.isfinite(number) or number != int(number) or number < 1:
        raise ValueError("cut_id must be a positive source integer")
    return int(number)


def _column(frame, selector):
    if isinstance(selector, int) and not isinstance(selector, bool):
        if not 0 <= selector < len(frame.columns):
            raise ValueError(f"Label column index outside source table: {selector}")
        return frame.iloc[:, selector]
    if isinstance(selector, str) and selector in frame.columns:
        return frame[selector]
    raise ValueError(f"Mapped label column is missing: {selector!r}")


def _labels(cutter, root):
    filename = cutter.get("label_file")
    if not filename:
        return {}, None, None
    if cutter["cutter_id"] not in RELEASED_CUTTERS:
        raise ValueError("PHM c2/c3/c5 labels are unreleased; official c3 wear file was incorrect")
    mapping = cutter.get("label_columns", {})
    if set(mapping) != {"cut_id", *EDGES} or len(set(mapping.values())) != 4:
        raise ValueError("label_columns must explicitly map cut_id and all three distinct flute columns")
    if "label_header" not in cutter or cutter["label_header"] not in (None, 0):
        raise ValueError("Declare label_header as null (no header) or 0 (header row)")
    path = _local(root, filename)
    frame = pd.read_csv(path, header=cutter["label_header"])
    columns = {key: _column(frame, selector) for key, selector in mapping.items()}
    labels = {}
    for row in range(len(frame)):
        cut = _positive_cut(columns["cut_id"].iloc[row])
        if cut in labels:
            raise ValueError(f"Duplicate PHM measured label cut_id: {cut}")
        edges = {}
        for edge in EDGES:
            value = columns[edge].iloc[row]
            value = None if pd.isna(value) else float(value)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Measured PHM flute wear must be finite and nonnegative or missing")
            edges[edge] = value
        labels[cut] = {"edges": edges, "row": row}
    return labels, path, _digest(path)


def load_phm(manifest_path: str | Path, *, root: str | Path | None = None,
             edge_reduction: str | None = None) -> list[SignalRun]:
    """Read explicitly mapped local CSVs; choose ``max`` or ``mean`` edge wear.

    ``root`` defaults to the manifest directory. All files must be inside it.
    Return source wear in micrometres (um), exactly equivalent to 10^-3 mm;
    all three measured edges and label file/row hashes remain in provenance.
    Missing cuts/edges retain None targets; they are never interpolated.
    """
    if edge_reduction not in {"max", "mean"}:
        raise ValueError("edge_reduction must explicitly be 'max' or 'mean'")
    manifest_path = Path(manifest_path).resolve()
    root = Path(root).resolve() if root is not None else manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("source_url") != PHM_SOURCE_URL:
        raise ValueError("PHM manifest requires schema_version 1 and the verified official source_url")
    if manifest.get("wear_unit") != "10^-3 mm":
        raise ValueError("PHM source wear unit must explicitly be '10^-3 mm'; no implicit rescaling")
    if manifest.get("sampling_hz", 50000) != 50000:
        raise ValueError("PHM sampling_hz contradicts documented 50000 Hz/channel")
    if "signal_header" not in manifest or manifest["signal_header"] not in (None, 0):
        raise ValueError("Declare signal_header as null (no header) or 0 (header row)")
    columns = manifest.get("signal_columns", {})
    if (set(columns) != set(PHM_CHANNEL_UNITS)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in columns.values())
            or set(columns.values()) != set(range(7))):
        raise ValueError("signal_columns must explicitly map the seven documented channels to indices 0–6")
    cutters = manifest.get("cutters")
    if not isinstance(cutters, list) or not cutters:
        raise ValueError("PHM manifest requires a nonempty cutters list")
    runs, identities, cutter_ids, paths = [], set(), set(), set()
    manifest_hash = _digest(manifest_path)
    for cutter in cutters:
        cutter_id = cutter.get("cutter_id")
        if cutter_id not in ALL_CUTTERS or cutter_id in cutter_ids:
            raise ValueError("PHM cutter_id must be a unique original c1–c6 identifier")
        cutter_ids.add(cutter_id)
        labels, label_path, label_hash = _labels(cutter, root)
        if not isinstance(cutter.get("records"), list) or not cutter["records"]:
            raise ValueError("Every cutter requires explicitly mapped cut records")
        for record in cutter["records"]:
            cut = _positive_cut(record.get("cut_id"))
            identity = (cutter_id, cut)
            if identity in identities:
                raise ValueError(f"Duplicate PHM source cut identity: {identity}")
            identities.add(identity)
            path = _local(root, record.get("signal_file"))
            if path in paths:
                raise ValueError("A PHM source acquisition file cannot represent multiple distinct cuts")
            paths.add(path)
            frame = pd.read_csv(path, header=manifest["signal_header"])
            if frame.shape[1] != 7 or frame.empty:
                raise ValueError("PHM acquisition CSV must contain seven nonempty source channels")
            values = frame.to_numpy(dtype=np.float64)
            measured = labels.get(cut)
            edges = measured["edges"] if measured else None
            target = None
            if edges is not None and all(value is not None for value in edges.values()):
                target = float(max(edges.values()) if edge_reduction == "max" else np.mean(list(edges.values())))
            runs.append(SignalRun(
                dataset_id="phm2010_milling", run_id=f"{cutter_id}_cut_{cut:04d}", group_id=cutter_id,
                signals={name: values[:, column].copy() for name, column in columns.items()},
                wear_target=target, wear_unit="um", sampling_hz=50000.,
                settings={"spindle_rpm": 10400, "feed_mm_min": 1555,
                          "radial_depth_mm": .125, "axial_depth_mm": .2},
                provenance={"source_file": str(path), "source_sha256": _digest(path),
                            "source_url": PHM_SOURCE_URL, "source_cut": cut, "source_cutter": cutter_id,
                            "manifest_file": str(manifest_path), "manifest_sha256": manifest_hash,
                            "label_file": str(label_path) if label_path else None,
                            "label_sha256": label_hash, "label_row": measured["row"] if measured else None,
                            "edge_wear": edges, "edge_reduction": edge_reduction,
                            "source_wear_unit": "10^-3 mm", "output_wear_unit": "um", "unit_factor": 1.,
                            "label_timing": "measured after the explicitly matched source cut",
                            "label_status": "released_measured" if target is not None else "unlabeled_or_incomplete",
                            "released_training_cutter": cutter_id in RELEASED_CUTTERS,
                            "channel_units": PHM_CHANNEL_UNITS.copy()},
            ))
    return runs
