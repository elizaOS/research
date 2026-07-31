"""Explicit plan/shard/merge CLI for the development-only continual-IA v2 contract.

The historical ``alberta-ia-evidence`` command remains v1-only.  V2 is
available only through this separately named module entrypoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.continual_ia_v2 import (
    load_artifact,
    load_plan,
    load_shard,
    merge_shards_with_validation,
    write_plan,
    write_shard,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Operate the strict, development-only continual-IA v2 lifecycle."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="issue the immutable pre-run plan")
    plan.add_argument("--plan-out", type=Path, required=True)
    plan.add_argument("--shard-dir", type=Path, required=True)
    plan.add_argument("--artifact-out", type=Path, required=True)
    plan.add_argument(
        "--attest-fresh-seeds-60-89",
        action="store_true",
        required=True,
        help="attest before issuance that all exact reserved seeds remain unexecuted",
    )

    shard = commands.add_parser("shard", help="execute one plan-bound seed twice")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--seed", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)

    merge = commands.add_parser("merge", help="merge the exact thirty plan-bound shards")
    merge.add_argument("--plan", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)

    verify_plan = commands.add_parser("verify-plan")
    verify_plan.add_argument("--plan", type=Path, required=True)

    verify_shard = commands.add_parser("verify-shard")
    verify_shard.add_argument("--plan", type=Path, required=True)
    verify_shard.add_argument("--shard", type=Path, required=True)

    verify_artifact = commands.add_parser("verify-artifact")
    verify_artifact.add_argument("--artifact", type=Path, required=True)
    return parser


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    """Operate v2 explicitly; no command is selected by default."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            output = write_plan(
                args.plan_out,
                args.shard_dir,
                args.artifact_out,
                attest_fresh_seeds=args.attest_fresh_seeds_60_89,
            )
            _emit(
                {
                    "artifact": str(output),
                    "internally_accepted": False,
                    "role": "self_issued_development_plan",
                    "valid": True,
                }
            )
            return 0
        if args.command == "shard":
            output = write_shard(args.plan, args.seed, args.output)
            _emit(
                {
                    "artifact": str(output),
                    "internally_accepted": False,
                    "role": "one_seed_shard_with_exact_replay",
                    "seed": args.seed,
                    "valid": True,
                }
            )
            return 0
        if args.command == "merge":
            output, validation = merge_shards_with_validation(args.plan, args.output)
            _emit(
                {
                    "artifact": str(output),
                    "errors": list(validation.errors),
                    "internally_accepted": validation.internally_accepted,
                    "role": "development_only_diagnostic",
                    "valid": validation.valid,
                }
            )
            if not validation.valid:
                return 2
            return 0
        if args.command == "verify-plan":
            load_plan(args.plan)
            _emit(
                {
                    "artifact": str(args.plan),
                    "internally_accepted": False,
                    "role": "self_issued_development_plan",
                    "valid": True,
                }
            )
            return 0
        if args.command == "verify-shard":
            _raw, plan = load_plan(args.plan)
            load_shard(args.shard, plan)
            _emit(
                {
                    "artifact": str(args.shard),
                    "internally_accepted": False,
                    "role": "one_seed_shard",
                    "valid": True,
                }
            )
            return 0
        if args.command == "verify-artifact":
            _raw, _payload, validation = load_artifact(args.artifact)
            _emit(
                {
                    "artifact": str(args.artifact),
                    "errors": list(validation.errors),
                    "internally_accepted": validation.internally_accepted,
                    "role": "development_only_diagnostic",
                    "valid": validation.valid,
                }
            )
            if not validation.valid:
                return 2
            return 0
    except Exception as exc:
        _emit(
            {
                "errors": [str(exc)],
                "internally_accepted": False,
                "valid": False,
            }
        )
        return 2
    raise AssertionError("argparse accepted an unknown v2 command")


if __name__ == "__main__":
    raise SystemExit(main())
