from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from cpdp.oversampling_validation.config import load_validation_config
from cpdp.oversampling_validation.runner import OversamplingValidationRunner
from cpdp.oversampling_validation.strategies import available_strategy_kinds


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "oversampling_validation.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oversampling-validation",
        description="Independent quality and utility validation for CPDP oversampling",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="show profiles and variants")
    list_parser.add_argument("--profile", default="smoke")

    run_parser = subparsers.add_parser("run", help="run validation tasks")
    run_parser.add_argument("--profile", default="smoke")
    run_parser.add_argument(
        "--protocol",
        action="append",
        choices=("source_cv", "cpdp"),
        help="repeatable; defaults to protocols enabled by the selected profile",
    )
    run_parser.add_argument("--variant", action="append", help="repeatable or all")
    run_parser.add_argument("--family", action="append", help="repeatable dataset family")
    run_parser.add_argument("--target", action="append", help="repeatable family/project")
    run_parser.add_argument("--seed", action="append", type=int, help="repeatable seed")
    run_parser.add_argument("--fold", action="append", type=int, help="repeatable source_cv fold")
    run_parser.add_argument("--max-tasks", type=int)
    run_parser.add_argument("--force", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")

    summary_parser = subparsers.add_parser("summarize", help="rebuild CSV summaries")
    summary_parser.add_argument("--profile", default="smoke")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_validation_config(Path(args.config))
        runner = OversamplingValidationRunner(config)
        if args.command == "list":
            profile = config.profile(args.profile)
            print(f"profile: {profile.name}")
            print(f"fingerprint: {runner.fingerprint(profile.name)}")
            print(f"variants: {', '.join(profile.variants)}")
            print(f"strategy kinds: {', '.join(available_strategy_kinds())}")
            print(f"seeds: {len(profile.seeds)}")
            return 0
        if args.command == "run":
            profile = config.profile(args.profile)
            default_protocols = tuple(
                protocol
                for protocol, enabled in (
                    ("source_cv", profile.source_cv.enabled),
                    ("cpdp", profile.cpdp.enabled),
                )
                if enabled
            )
            report = runner.run(
                args.profile,
                protocols=tuple(args.protocol or default_protocols),
                variants=args.variant,
                families=args.family,
                targets=args.target,
                seeds=args.seed,
                folds=args.fold,
                max_tasks=args.max_tasks,
                force=args.force,
                fail_fast=args.fail_fast,
            )
            print(
                "Validation summary: "
                f"fingerprint={report.fingerprint} selected={report.selected} "
                f"attempted={report.attempted} success={report.succeeded} "
                f"failure={report.failed} skipped={report.skipped}"
            )
            print(f"output: {report.output_root}")
            return 1 if report.failed else 0
        if args.command == "summarize":
            for name, path in runner.summarize(args.profile).items():
                print(f"{name}: {path}")
            return 0
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    raise AssertionError(f"Unhandled command: {args.command}")
