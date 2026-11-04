"""NASA PCoE milling archive MATLAB adapter, based on bundled Readme.pdf.

    Readme pp. 1–3: schema, case/run hierarchy and 250 Hz processed data.
    Readme p. 7 Fig. 6: VB in mm, measured microscopically after selected runs.
    Case IDs are experiments; no additional physical cutter identity is invented.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from fabfederate.contracts import SignalRun

NASA_CHANNELS = ("smcAC", "smcDC", "vib_table", "vib_spindle", "AE_table", "AE_spindle")
NASA_FIELDS = {"case", "run", "VB", "time", "DOC", "feed", "material", *NASA_CHANNELS}


def _scalar(value, name):
    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"NASA field {name} must be a numeric scalar")
    number = float(array.reshape(-1)[0])
    if not np.isfinite(number):
        raise ValueError(f"NASA field {name} must be finite")
    return number


def load_nasa(path: str | Path) -> list[SignalRun]:
    """Read the exact ``mill`` schema from a .mat file or containing directory.

    NaN/empty VB means unmeasured; negative/infinite labels and malformed source
    records fail explicitly. Channels with nonfinite values remain auditable and
    are rejected by the downstream quality gate, never silently repaired.
    """
    source = Path(path)
    if source.is_dir():
        candidates = sorted(source.rglob("mill.mat"))
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one mill.mat, found {len(candidates)} in {source}")
        source = candidates[0]
    if source.suffix.lower() != ".mat":
        raise ValueError("NASA loader expects extracted mill.mat or its directory")
    payload = loadmat(source, simplify_cells=True)
    if "mill" not in payload:
        raise ValueError("NASA MATLAB file is missing the mill struct")
    records = payload["mill"]
    if isinstance(records, dict):
        records = [records]
    elif isinstance(records, np.ndarray):
        records = records.reshape(-1).tolist()
    if not isinstance(records, list) or not records:
        raise ValueError("NASA mill must be a nonempty struct array")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    runs, seen = [], set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or not NASA_FIELDS.issubset(record):
            missing = NASA_FIELDS - set(record) if isinstance(record, dict) else NASA_FIELDS
            raise ValueError(f"NASA record {index} missing required fields: {sorted(missing)}")
        case = _scalar(record["case"], "case")
        run = _scalar(record["run"], "run")
        if case != int(case) or not 1 <= case <= 16:
            raise ValueError(f"NASA case outside documented 1–16 range: {case}")
        if run != int(run) or run < 1:
            raise ValueError(f"NASA run must be a positive integer: {run}")
        identity = (int(case), int(run))
        if identity in seen:
            raise ValueError(f"Duplicate NASA case/run identity: {identity}")
        seen.add(identity)
        vb = np.asarray(record["VB"])
        if vb.size == 0 or (vb.size == 1 and np.isnan(float(vb.reshape(-1)[0]))):
            target = None
        else:
            target = _scalar(vb, "VB")
            if target < 0:
                raise ValueError("NASA VB must be nonnegative or missing")
        settings = {key: _scalar(record[key], key) for key in ("DOC", "feed", "material")}
        if settings["material"] not in (1, 2):
            raise ValueError("NASA material must be source code 1 or 2")
        if settings["DOC"] <= 0 or settings["feed"] <= 0:
            raise ValueError("NASA DOC and feed must be positive")
        settings["material"] = int(settings["material"])
        signals = {}
        for channel in NASA_CHANNELS:
            values = np.asarray(record[channel], dtype=np.float64)
            if values.ndim != 1 or not len(values):
                raise ValueError(f"NASA {channel} must be a nonempty 1-D signal")
            signals[channel] = values
        if len({len(values) for values in signals.values()}) != 1:
            raise ValueError(f"NASA case/run {identity} channels have unequal lengths")
        runs.append(SignalRun(
            dataset_id="nasa_milling", run_id=f"case_{int(case):02d}_run_{int(run):03d}",
            group_id=f"case_{int(case):02d}", signals=signals, wear_target=target,
            wear_unit="mm", settings=settings, sampling_hz=250.,
            provenance={"source_file": str(source.resolve()), "source_sha256": digest,
                        "source_record_index": index, "source_case": int(case), "source_run": int(run),
                        "source_time": _scalar(record["time"], "time"),
                        "target_field": "VB", "label_timing": "measured after selected runs; no interpolation",
                        "source_document": "Readme.pdf pp.1–3,7",
                        "sampling_basis": "processed data sampling rate in Readme.pdf p.3",
                        "channel_units": "source acquisition values; physical calibration not established"},
        ))
    return runs
