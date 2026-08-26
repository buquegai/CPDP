from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence

from cpdp.config import ExperimentConfig, load_config
from cpdp.methods.registry import (
    available_methods,
    create_method,
    load_method_plugin,
    normalize_method_name,
)
from cpdp.result_store import ResultStore
from cpdp.runner import ExperimentRunner, validate_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


def _config(value: str) -> ExperimentConfig:
    return load_config(Path(value))


def _load_plugins(paths: Sequence[str]) -> None:
    for path in paths:
        load_method_plugin(Path(path))


def _resolve_methods(value: str) -> Sequence[str]:
    if value.lower() == "all":
        return available_methods()
    requested = tuple(
        dict.fromkeys(normalize_method_name(name.strip()) for name in value.split(",") if name.strip())
    )
    unknown = sorted(set(requested) - set(available_methods()))
    if unknown:
        raise KeyError(f"Unknown method(s): {', '.join(unknown)}")
    return requested


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpdp", description="Unified CPDP experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-methods", help="show available methods")
    _add_config_argument(list_parser)
    list_parser.add_argument("--profile", default="full")
    list_parser.add_argument("--plugin", action="append", default=[])
    list_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser("validate-data", help="validate all datasets")
    _add_config_argument(validate_parser)
    validate_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="run one or more methods")
    _add_config_argument(run_parser)
    run_parser.add_argument("--method", required=True, help="method, comma list, or all")
    run_parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    run_parser.add_argument("--target", action="append", help="family/project; repeatable")
    run_parser.add_argument("--seed", action="append", type=int, help="seed; repeatable")
    run_parser.add_argument("--max-runs", type=int)
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel method-target-seed tasks",
    )
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--plugin", action="append", default=[])

    summarize_parser = subparsers.add_parser("summarize", help="aggregate saved runs")
    _add_config_argument(summarize_parser)
    return parser


def _list_methods(args: argparse.Namespace) -> int:
    config = _config(args.config)
    _load_plugins(args.plugin)
    records = []
    for name in available_methods():
        try:
            params = config.method_params(args.profile, name)
        except KeyError:
            params = {}
        method = create_method(name, config.source_path.parent.parent, params)
        records.append({"name": name, **asdict(method.provenance)})
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        for record in records:
            print(
                f"{record['name']:<13} {record['display_name']:<14} "
                f"{record['fidelity']}"
            )
    return 0


def _validate_data(args: argparse.Namespace) -> int:
    _, records = validate_datasets(_config(args.config))
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        print("family  project       rows features clean defective defect_rate")
        for record in records:
            print(
                f"{record['family']:<7} {record['project']:<12} "
                f"{record['rows']:>5} {record['features']:>8} "
                f"{record['clean']:>5} {record['defective']:>9} "
                f"{record['defect_rate']:>11.4f}"
            )
        print(f"Validated {len(records)} projects.")
    return 0


def _run(args: argparse.Namespace) -> int:
    config = _config(args.config)
    _load_plugins(args.plugin)
    methods = _resolve_methods(args.method)
    report = ExperimentRunner(config).run(
        methods,
        profile=args.profile,
        targets=args.target,
        seeds=args.seed,
        max_runs=args.max_runs,
        workers=args.workers,
        force=args.force,
        fail_fast=args.fail_fast,
        plugin_paths=tuple(Path(path) for path in args.plugin),
    )
    print(
        "Run summary: "
        f"selected={report.selected} attempted={report.attempted} "
        f"success={report.succeeded} failure={report.failed} skipped={report.skipped}"
    )
    return 1 if report.failed else 0


def _summarize(args: argparse.Namespace) -> int:
    config = _config(args.config)
    paths = ResultStore(config.output_dir, config.save_predictions).summarize(
        expected_repetitions=config.repetitions,
        expected_seeds=config.seeds,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "list-methods":
            return _list_methods(args)
        if args.command == "validate-data":
            return _validate_data(args)
        if args.command == "run":
            return _run(args)
        if args.command == "summarize":
            return _summarize(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, AttributeError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")
