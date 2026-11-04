"""Assign whole training groups; held-out ownership requires exact domain evidence."""

import numpy as np
import pandas as pd

SCENARIOS = {"iid", "condition", "quantity", "label", "few_shot"}


def _few_shot(frame, num_clients, seed, few_shot_groups):
    """Freeze the condition baseline, then remove labels from one client's cases."""
    if not isinstance(few_shot_groups, int) or few_shot_groups < 1:
        raise ValueError("few_shot_groups must be a positive integer")
    if num_clients < 2:
        raise ValueError("Few-shot comparison requires at least two clients")
    baseline, manifest = assign_clients(frame, "condition", num_clients, seed)
    train = baseline[baseline.split == "train"]
    candidates = [client for client in sorted(manifest["clients"])
                  if manifest["clients"][client]["training_groups"] > few_shot_groups]
    if not candidates:
        raise ValueError("Few-shot is infeasible: no client has enough groups for a strict label reduction")
    supported = [client for client in candidates
                 if ((baseline.split == "test") & (baseline.client_id == client)).any()]
    target = (supported or candidates)[0]
    test_rows = baseline[(baseline.split == "test") & (baseline.client_id == target)]
    test_conditions = set(test_rows.condition)
    target_rows = train[train.client_id == target]
    target_groups = sorted(target_rows.group_id.unique())
    # Keep a whole case from a supported evaluation condition when available.
    # Choice uses condition identity only; never inspect held-out wear labels.
    matched = sorted(target_rows[target_rows.condition.isin(test_conditions)].group_id.unique())
    rng = np.random.default_rng(seed)
    preferred = matched[:1]
    remaining = rng.permutation([group for group in target_groups if group not in preferred]).tolist()
    kept = set((preferred + remaining)[:few_shot_groups])
    excluded = sorted(set(target_groups) - kept)
    limited = baseline[~((baseline.split == "train") & baseline.group_id.isin(excluded))].copy()
    manifest.update({
        "scenario": "few_shot", "few_shot_groups": few_shot_groups,
        "few_shot_client": target, "excluded_training_groups": excluded,
        "retained_few_shot_groups": sorted(kept),
        "e4_test_groups": sorted(test_rows.group_id.unique().tolist()),
        "e4_status": "available" if supported else "infeasible",
        "e4_reason": ("Frozen condition-baseline test ownership; same test groups for every method"
                      if supported else "No reducible client has a supported exact-condition test group"),
        "limited_label_policy": "condition baseline frozen; discarded cases excluded from every method's training union",
        "group_assignments": {group: client for group, client in manifest["group_assignments"].items()
                              if group not in excluded},
    })
    target_training = limited[(limited.split == "train") & (limited.client_id == target)]
    manifest["clients"][target] = {
        "training_windows": len(target_training),
        "training_groups": int(target_training.group_id.nunique()),
        "training_runs": int(target_training.run_id.nunique()),
    }
    return limited, manifest


def assign_clients(frame, scenario="iid", num_clients=3, seed=42, few_shot_groups=1):
    """Return a copy and manifest; no label is read for held-out assignment.

    Held-out domains are associated only when all training groups with that exact
    documented condition belong to one client. Other held-out rows retain null
    client_id and remain eligible for pooled and group evaluation.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported client scenario {scenario!r}; choose {sorted(SCENARIOS)}")
    if not isinstance(num_clients, int) or num_clients < 1:
        raise ValueError("num_clients must be a positive integer")
    required = {"group_id", "run_id", "split", "condition", "wear_target"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing partition columns: {sorted(required - set(frame.columns))}")
    if frame.groupby("group_id").split.nunique().gt(1).any():
        raise ValueError("Independent groups cannot cross split boundaries")
    if scenario == "few_shot":
        return _few_shot(frame, num_clients, seed, few_shot_groups)
    result = frame.copy()
    train = result[result.split == "train"]
    groups = sorted(train.group_id.unique())
    if len(groups) < num_clients:
        raise ValueError("Too few independent training groups for requested clients")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups).tolist()
    assignments = {}
    client_names = [f"client_{i}" for i in range(num_clients)]
    if scenario == "condition":
        if train.groupby("group_id").condition.nunique().gt(1).any():
            raise ValueError("Condition scenario requires a single documented condition per group")
        condition_groups = train.groupby("condition").group_id.unique().to_dict()
        if any(pd.isna(key) or key in ("", "unknown", "{}") for key in condition_groups):
            raise ValueError("Condition scenario requires documented conditions")
        if len(condition_groups) < num_clients:
            raise ValueError("Too few independent condition groups for requested clients")
        loads = [0] * num_clients
        for condition in sorted(condition_groups, key=lambda c: (-len(condition_groups[c]), c)):
            client = int(np.argmin(loads))
            for group in condition_groups[condition]:
                assignments[group] = client_names[client]
            loads[client] += len(condition_groups[condition])
    elif scenario == "label":
        # Average distinct run targets, not window-weighted targets.
        means = train.drop_duplicates(["group_id", "run_id"]).groupby("group_id").wear_target.mean()
        ordered = sorted(groups, key=lambda group: (means[group], group))
        for client, chunk in enumerate(np.array_split(ordered, num_clients)):
            assignments.update({group: client_names[client] for group in chunk})
    elif scenario == "quantity":
        if len(groups) == num_clients:
            raise ValueError("Quantity skew needs more training groups than clients")
        weights = np.arange(1, num_clients + 1, dtype=float)
        counts = np.ones(num_clients, dtype=int)
        extra = len(groups) - num_clients
        quota = extra * weights / weights.sum()
        counts += np.floor(quota).astype(int)
        for client in np.argsort(-(quota - np.floor(quota)))[:len(groups) - counts.sum()]:
            counts[client] += 1
        offset = 0
        for client, count in enumerate(counts):
            assignments.update({group: client_names[client] for group in shuffled[offset:offset + count]})
            offset += count
    else:
        assignments = {group: client_names[index % num_clients]
                       for index, group in enumerate(shuffled)}

    result["client_id"] = pd.Series(index=result.index, dtype="object")
    train_mask = result.split == "train"
    result.loc[train_mask, "client_id"] = result.loc[train_mask, "group_id"].map(assignments)
    domain_clients = {}
    for condition, rows in result[train_mask].groupby("condition"):
        owners = sorted(rows.client_id.unique())
        if len(owners) == 1 and condition not in ("", "unknown", "{}"):
            domain_clients[condition] = owners[0]
    heldout = result[~train_mask]
    for group, rows in heldout.groupby("group_id"):
        conditions = rows.condition.drop_duplicates().tolist()
        owners = {domain_clients.get(condition) for condition in conditions}
        if len(owners) == 1 and None not in owners:
            result.loc[result.group_id == group, "client_id"] = next(iter(owners))
    uncovered = result[(result.split != "train") & result.client_id.isna()]
    manifest = {
        "scenario": scenario, "seed": seed, "num_clients": num_clients,
        "few_shot_groups": few_shot_groups,
        "group_assignments": dict(sorted(assignments.items())),
        "heldout_policy": "exact condition match with a unique training-client owner; otherwise unavailable",
        "heldout_domain_clients": domain_clients,
        "unassigned_heldout_groups": sorted(uncovered.group_id.unique().tolist()),
        "uncovered_domains": sorted(uncovered.condition.dropna().unique().tolist()),
        "clients": {client: {"training_windows": int(((result.client_id == client) & train_mask).sum()),
                               "training_groups": int(result.loc[(result.client_id == client) & train_mask, "group_id"].nunique()),
                               "training_runs": int(result.loc[(result.client_id == client) & train_mask, "run_id"].nunique())}
                    for client in client_names},
    }
    return result, manifest
