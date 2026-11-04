"""Finite, serializable inventory of accepted source runs and quality issues."""

from collections import Counter

import numpy as np


def is_labeled(run):
    return run.wear_target is not None and np.isfinite(run.wear_target)


def audit_runs(runs):
    issues = []
    channels = {}
    for run in runs:
        if not is_labeled(run):
            issues.append({"run_id": run.run_id, "reason": "missing_or_nonfinite_wear_target"})
        lengths = set()
        for name, signal in run.signals.items():
            values = np.asarray(signal, dtype=float)
            lengths.add(values.size)
            summary = channels.setdefault(name, {"runs": 0, "samples": 0, "nonfinite_samples": 0,
                                                 "constant_runs": 0, "lengths": set()})
            summary["runs"] += 1
            summary["samples"] += values.size
            summary["lengths"].add(values.size)
            summary["nonfinite_samples"] += int((~np.isfinite(values)).sum())
            if not np.isfinite(values).all():
                issues.append({"run_id": run.run_id, "channel": name, "reason": "nonfinite_signal"})
            elif values.size and np.ptp(values) == 0:
                summary["constant_runs"] += 1
                issues.append({"run_id": run.run_id, "channel": name, "reason": "constant_signal"})
            if values.ndim != 1:
                issues.append({"run_id": run.run_id, "channel": name, "reason": "non_1d_signal"})
        if len(lengths) > 1:
            issues.append({"run_id": run.run_id, "reason": "unequal_channel_lengths"})
    for summary in channels.values():
        summary["lengths"] = sorted(summary["lengths"])
    targets = [float(run.wear_target) for run in runs if is_labeled(run)]
    datasets = {run.dataset_id for run in runs}
    grouping = "caller-declared independent source groups; verify source hierarchy"
    if datasets == {"nasa_milling"}:
        grouping = "case/experiment grouping; no invented physical cutter IDs"
    elif datasets == {"phm2010_milling"}:
        grouping = "source cutter grouping; cuts from the same physical cutter remain together"
    return {
        "run_count": len(runs), "independent_group_count": len({run.group_id for run in runs}),
        "labeled_run_count": len(targets), "unlabeled_run_count": len(runs) - len(targets),
        "group_run_counts": dict(sorted(Counter(run.group_id for run in runs).items())),
        "wear_units": sorted({run.wear_unit for run in runs}),
        "wear_range": [min(targets), max(targets)] if targets else None,
        "sampling_hz": sorted({float(run.sampling_hz) for run in runs
                               if run.sampling_hz is not None and np.isfinite(run.sampling_hz)}),
        "channels": channels, "issues": issues,
        "source_files": sorted({run.provenance["source_file"] for run in runs
                                if "source_file" in run.provenance}),
        "source_sha256": sorted({run.provenance["source_sha256"] for run in runs
                                 if "source_sha256" in run.provenance}),
        "independence_limit": grouping,
    }
