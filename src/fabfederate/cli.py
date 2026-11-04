"""Local command-line entry points for auditable file-driven research."""

import argparse
import json
from pathlib import Path

from fabfederate.config import load_config
from fabfederate.utils.io import write_json


def parser():
    root = argparse.ArgumentParser(prog="fabfederate", description="Offline CNC wear research")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Retrieve and hash the NASA-linked archive")
    download.add_argument("--data-root", default="data/raw")
    download.add_argument("--force", action="store_true")
    for command in ("audit", "prepare", "study"):
        entry = commands.add_parser(command)
        entry.add_argument("--config", type=Path)
        entry.add_argument("--source", type=Path)
        entry.add_argument("--output", type=Path)
        if command == "study":
            entry.add_argument("--no-explain", action="store_true", help="Omit SHAP (recorded in manifest)")
            entry.add_argument("--seeds", type=int, nargs="+")
            entry.add_argument("--scenarios", nargs="+", choices=["iid", "condition", "quantity", "label", "few_shot"])
    report = commands.add_parser("report")
    report.add_argument("experiment", type=Path)
    for command in ("predict", "explain"):
        entry = commands.add_parser(command)
        entry.add_argument("--model", type=Path, required=True)
        entry.add_argument("--input", type=Path, required=True)
        entry.add_argument("--output", type=Path, required=True)
        if command == "explain":
            entry.add_argument("--background", type=Path, required=True,
                               help="Prepared Parquet table with explicit train split")
    campus = commands.add_parser("campus-validate", help="Validate approved read-only recordings")
    campus.add_argument("--manifest", type=Path, required=True)
    campus.add_argument("--model", type=Path, required=True)
    campus.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "download":
            from fabfederate.data.acquisition import acquire_nasa
            result = acquire_nasa(args.data_root, args.force)
            print(json.dumps(result, indent=2))
        elif args.command in {"audit", "prepare", "study"}:
            config = load_config(args.config)
            if args.source:
                config["data_path"] = str(args.source)
            if args.output:
                config["output_dir"] = str(args.output)
            if args.command == "study":
                from fabfederate.pipelines.study import run_study
                if args.seeds:
                    config["seeds"] = args.seeds
                if args.scenarios:
                    config["scenarios"] = args.scenarios
                print(run_study(config, explain=not args.no_explain).resolve())
            else:
                from fabfederate.data.dataset import load_source
                runs = load_source(config)
                output = Path(config["output_dir"])
                if args.command == "audit":
                    from fabfederate.data.audit import audit_runs
                    result = audit_runs(runs)
                    write_json(output / "audit.json", result)
                    print(json.dumps(result, indent=2))
                else:
                    from fabfederate.data.preprocessing import prepare_dataset
                    prepared = prepare_dataset(runs, output, config)
                    print(f"{len(prepared.frame)} windows; {len(prepared.feature_names)} features; {output.resolve()}")
        elif args.command == "report":
            from fabfederate.evaluation.report import generate_report
            print(generate_report(args.experiment).resolve())
        elif args.command in {"predict", "explain"}:
            from fabfederate.inference.pipeline import explain_file, predict_file
            if args.command == "predict":
                predict_file(args.model, args.input, args.output)
            else:
                explain_file(args.model, args.input, args.background, args.output)
            print(args.output.resolve())
        elif args.command == "campus-validate":
            from fabfederate.campus_cnc.validation import validate_recordings
            print(validate_recordings(args.manifest, args.model, args.output).resolve())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser().exit(2, f"fabfederate: {error}\n")


if __name__ == "__main__":
    main()
