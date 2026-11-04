"""End-to-end experiment execution with validation-only selection and frozen tests."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import logging
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from fabfederate.config import validate_config
from fabfederate.contracts import PreparedData
from fabfederate.evaluation.metrics import evaluate_predictions, regression_metrics
from fabfederate.data.preprocessing import prepare_dataset
from fabfederate.federated.server import run_federated
from fabfederate.models.train import run_baselines
from fabfederate.data.dataset import load_source
from fabfederate.utils.io import file_hash, read_json, source_snapshot, stable_hash, write_json


def _now():
    return datetime.now(timezone.utc).isoformat()


def _explain_run(bundles, data, directory, config):
    from fabfederate.explainability.shap_explainer import explain_prediction, save_explanation
    training = data.frame[data.frame.split == "train"][data.feature_names]
    samples = (data.frame[data.frame.split == "test"]
               .groupby("group_id", sort=True).head(1).head(config["explanation_samples"]))
    records = []
    for method in ("fedavg", "fedprox"):
        explanation = explain_prediction(
            bundles[method], training, samples[data.feature_names],
            max_background=config["max_background"],
        )
        explanation["model_version"] = bundles[method].metadata
        explanation["sample_ids"] = samples[["run_id", "window_id"]].to_dict("records")
        save_explanation(explanation, directory / "explanations" / f"{method}.json")
        for sample, values in enumerate(explanation["values"]):
            records.append({"method": method, "sample": sample,
                            "run_id": samples.iloc[sample].run_id,
                            "window_id": int(samples.iloc[sample].window_id),
                            "attributions": list(map(float, values))})
    return records


def _robustness(bundles, data, config):
    test = data.frame[data.frame.split == "test"].copy()
    train = data.frame[data.frame.split == "train"]
    rng = np.random.default_rng(config["seed"])
    deviation = train[data.feature_names].std(ddof=0).to_numpy()
    perturbed = test.copy()
    perturbed[data.feature_names] += (
        rng.normal(size=(len(test), len(data.feature_names))) * deviation
        * config["robustness_noise"]
    )
    records = []
    for method in ("centralized_mlp", "fedavg", "fedprox"):
        bundle = bundles[method]
        for label, frame in (("unmodified", test), ("feature_noise", perturbed)):
            records.append({"method": method, "perturbation": label,
                            **regression_metrics(test.wear_target.to_numpy(), bundle.predict(frame)),
                            "noise_std_fraction_of_train_std": config["robustness_noise"]})
        try:
            bundle.predict(test.drop(columns=data.feature_names[0]))
        except ValueError:
            rejected = True
        else:
            raise RuntimeError("Missing predictor was silently accepted")
        records.append({"method": method, "perturbation": "missing_required_feature",
                        "rejected": rejected})
    return records


def run_study(config: dict, explain: bool = True) -> Path:
    """Run all declared seeds/scenarios. Existing studies are never silently overwritten."""
    config = validate_config(deepcopy(config))
    directory = Path(config["output_dir"])
    if (directory / "manifest.json").exists():
        raise FileExistsError(f"Study already exists: {directory}; choose a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "started_at_utc": _now(), "config": config,
                "executed_source": source_snapshot(Path(__file__).resolve().parents[3]),
                "source_sha256": file_hash(config["data_path"]),
                "source_path": str(config["data_path"]), "runs": [], "infeasible": [],
                "methodology": "Fixed case-heldout split and partition seeds; validation-only selection",
                "scope": "offline public-data benchmark; no independently verified physical-tool split",
                "versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "pandas", "scipy", "scikit-learn", "torch", "flwr", "shap")}}
    write_json(directory / "manifest.json", manifest)
    logging.getLogger("flwr").setLevel(logging.WARNING)
    runs = load_source(config)
    metric_rows, prediction_frames, round_frames, robustness_rows, attribution_rows = [], [], [], [], []
    limited_label_rows = []
    default_saved = False
    try:
        for scenario in config["scenarios"]:
            preparation = dict(config, seed=config["split_seed"], scenario=scenario)
            try:
                prepared = prepare_dataset(runs, directory / "prepared" / scenario, preparation)
            except ValueError as error:
                if scenario == "iid":
                    raise
                manifest["infeasible"].append({"scenario": scenario, "reason": str(error)})
                write_json(directory / "manifest.json", manifest)
                continue
            for seed in config["seeds"]:
                started = time.perf_counter()
                run_dir = directory / "runs" / f"{scenario}-seed-{seed}"
                run_config = dict(config, scenario=scenario, seed=seed)
                frame = prepared.frame.copy()
                metadata = dict(prepared.metadata, source_sha256=manifest["source_sha256"],
                                code_sha256=manifest["executed_source"]["sha256"],
                                split_sha256=file_hash(directory / "prepared" / scenario / "split_manifest.json"),
                                partition_sha256=file_hash(directory / "prepared" / scenario / "client_manifest.json"))
                data = PreparedData(frame, prepared.feature_names, metadata)
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "config.json", run_config)
                bundles = run_baselines(data, run_dir / "models", run_config)
                bundles["fedavg"] = run_federated(data, run_dir / "models", run_config)
                candidates = []
                for index, mu in enumerate(config["mu_grid"]):
                    candidate = run_federated(data, run_dir / "candidates" / str(index),
                                              run_config, algorithm="fedprox", mu=mu)
                    candidates.append((float(candidate.metadata["validation_mae"]), index, candidate))
                _, chosen_index, bundles["fedprox"] = min(candidates, key=lambda value: value[:2])
                selected = run_dir / "models" / "fedprox"
                bundles["fedprox"].save(selected)
                shutil.copy2(run_dir / "candidates" / str(chosen_index) / "fedprox" / "rounds.csv",
                             selected / "rounds.csv")
                selection = {"frozen_at_utc": _now(), "selection_split": "validation",
                             "test_used_for_selection": False,
                             "fedprox_candidates": [{"mu": item[2].metadata["mu"],
                                                      "validation_mae": item[0]} for item in candidates],
                             "selected_mu": bundles["fedprox"].metadata["mu"],
                             "models": {name: bundle.metadata for name, bundle in bundles.items()}}
                write_json(run_dir / "selection.json", selection)
                for method, bundle in bundles.items():
                    for split in ("validation", "test"):
                        part = frame[frame.split == split].copy()
                        part["prediction"] = bundle.predict(part)
                        part["method"], part["seed"], part["scenario"] = method, seed, scenario
                        summary = evaluate_predictions(part)
                        prediction_frames.append(part.drop(columns=data.feature_names))
                        metric_rows.append({"method": method, "seed": seed, "scenario": scenario,
                            "split": split, **summary["pooled"], "group_mae": summary["group_mae"],
                            "worst_client_mae": summary["worst_client_mae"],
                            "client_covered_windows": summary["client_covered_windows"],
                            "count": summary["count"], "groups": summary["groups"],
                            "units": bundle.wear_unit, "mu": bundle.metadata.get("mu", 0.0),
                            "dataset_hash": manifest["source_sha256"],
                            "split_hash": metadata["split_sha256"],
                            "partition_hash": metadata["partition_sha256"]})
                        write_json(run_dir / "evaluation" / f"{method}-{split}.json", summary)
                        if scenario == "few_shot" and split == "test":
                            client_manifest = read_json(directory / "prepared" / scenario / "client_manifest.json")
                            client = client_manifest.get("few_shot_client", "client_0")
                            client_rows = part[part.client_id == client]
                            if method in {f"local_{client}", "centralized_mlp", "fedavg", "fedprox"}:
                                limited_label_rows.append({"method": method, "seed": seed, "client": client,
                                    "status": "available" if len(client_rows) else "infeasible_no_test_coverage",
                                    "training_groups": int(frame[(frame.split == "train") & (frame.client_id == client)].group_id.nunique()),
                                    "test_groups": int(client_rows.group_id.nunique()),
                                    **(regression_metrics(client_rows.wear_target.to_numpy(), client_rows.prediction.to_numpy())
                                       if len(client_rows) else {})})
                    log = run_dir / "models" / method / "rounds.csv"
                    if log.exists():
                        rounds = pd.read_csv(log)
                        rounds["method"], rounds["seed"], rounds["scenario"] = method, seed, scenario
                        round_frames.append(rounds)
                robustness_rows.extend({**row, "seed": seed, "scenario": scenario}
                                       for row in _robustness(bundles, data, run_config))
                if explain:
                    attribution_rows.extend({**row, "seed": seed, "scenario": scenario}
                                            for row in _explain_run(bundles, data, run_dir, run_config))
                if not default_saved:
                    frame.to_parquet(directory / "features.parquet", index=False)
                    for name in ("audit.json", "schema.json", "split_manifest.json", "client_manifest.json"):
                        shutil.copy2(directory / "prepared" / scenario / name, directory / name)
                    for method, bundle in bundles.items():
                        bundle.save(directory / "models" / method)
                    manifest["dashboard_default"] = {"scenario": scenario, "seed": seed}
                    default_saved = True
                manifest["runs"].append({"scenario": scenario, "seed": seed,
                                         "directory": str(run_dir), "duration_seconds": time.perf_counter()-started,
                                         "selected_mu": selection["selected_mu"]})
                pd.DataFrame(metric_rows).to_csv(directory / "metrics.csv", index=False)
                pd.concat(prediction_frames, ignore_index=True).to_csv(directory / "predictions.csv", index=False)
                if round_frames:
                    pd.concat(round_frames, ignore_index=True).to_csv(directory / "rounds.csv", index=False)
                pd.DataFrame(robustness_rows).to_csv(directory / "robustness.csv", index=False)
                write_json(directory / "manifest.json", manifest)
        if not metric_rows:
            raise ValueError("No feasible experiment was executed")
        if limited_label_rows:
            pd.DataFrame(limited_label_rows).to_csv(directory / "limited_label_comparison.csv", index=False)
        if attribution_rows:
            from fabfederate.explainability.shap_explainer import top_k_agreement, sign_agreement
            stability = []
            for key, group in pd.DataFrame(attribution_rows).groupby(["method", "scenario", "run_id", "window_id"]):
                rows = list(group.itertuples())
                for index in range(1, len(rows)):
                    stability.append({"method": key[0], "scenario": key[1], "run_id": key[2],
                                      "window_id": key[3], "seed_a": rows[0].seed, "seed_b": rows[index].seed,
                                      "top5_overlap": top_k_agreement(rows[0].attributions, rows[index].attributions),
                                      "sign_agreement": sign_agreement(rows[0].attributions, rows[index].attributions)})
            write_json(directory / "attributions.json", attribution_rows)
            if stability:
                pd.DataFrame(stability).to_csv(directory / "explanation_stability.csv", index=False)
        current_source = source_snapshot(Path(__file__).resolve().parents[3])
        if current_source["sha256"] != manifest["executed_source"]["sha256"]:
            raise RuntimeError("Project source changed during execution; rerun with a stable implementation")
        manifest.update(status="complete", completed_at_utc=_now(), explanations_enabled=explain,
                        config_sha256=stable_hash(config))
        write_json(directory / "manifest.json", manifest)
        from fabfederate.evaluation.report import generate_report
        generate_report(directory)
        return directory
    except Exception as error:
        manifest.update(status="failed", failure=type(error).__name__ + ": " + str(error))
        write_json(directory / "manifest.json", manifest)
        raise


def load_prepared(directory: str | Path) -> PreparedData:
    path = Path(directory)
    schema = read_json(path / "schema.json")
    return PreparedData(pd.read_parquet(path / "features.parquet"), schema["feature_names"], schema)
