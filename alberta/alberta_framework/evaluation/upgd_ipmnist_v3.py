"""Public fail-closed validators for active UPGD IPMNIST v3 artifacts.

Thin shim over :mod:`alberta_framework.benchmarks.upgd_ipmnist_v3`: it
re-exports the strict plan/partial/artifact validators and adds a CLI entry
point.  The lane replicates the online Input-Permuted MNIST protocol of UPGD
(Utility-based Perturbed Gradient Descent; Elsayed & Mahmood, ICLR 2024,
"Addressing Loss of Plasticity and Catastrophic Forgetting in Continual
Learning").  Sealed v1/v2 payloads predate this contract and are validated
separately by :mod:`alberta_framework.evaluation.upgd_ipmnist_nonpromoting`;
v3 execution is governed by ``UPGD_IPMNIST_V3_RUNBOOK.md``.  Validity never
grants promotion: the v3 schema is permanently nonpromoting because the
execution envelope is self-recorded, with no independent attestation.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.benchmarks.upgd_ipmnist_v3 import (
    UPGDIPMNISTV3Validation,
    validate_artifact,
    validate_partial,
    validate_plan,
)

__all__ = [
    "UPGDIPMNISTV3Validation",
    "validate_artifact",
    "validate_partial",
    "validate_plan",
]

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one namespaced v3 payload; validity never grants promotion."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="kind", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("path", type=Path)
    plan_parser.add_argument("--data-home", type=Path, default=None)
    plan_parser.add_argument("--data-archive", type=Path, default=None)
    partial_parser = subparsers.add_parser("partial")
    partial_parser.add_argument("path", type=Path)
    partial_parser.add_argument("--plan", type=Path, required=True)
    partial_parser.add_argument("--data-home", type=Path, default=None)
    partial_parser.add_argument("--data-archive", type=Path, default=None)
    artifact_parser = subparsers.add_parser("artifact")
    artifact_parser.add_argument("path", type=Path)
    artifact_parser.add_argument("--plan", type=Path, required=True)
    artifact_parser.add_argument("--partials", type=Path, nargs="+", default=None)
    artifact_parser.add_argument("--data-home", type=Path, default=None)
    artifact_parser.add_argument("--data-archive", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)

    if args.kind == "plan":
        report = validate_plan(
            args.path,
            data_home=args.data_home,
            data_archive=args.data_archive,
        )
    elif args.kind == "partial":
        report = validate_partial(
            args.path,
            args.plan,
            data_home=args.data_home,
            data_archive=args.data_archive,
        )
    else:
        report = validate_artifact(
            args.path,
            partial_paths=args.partials,
            plan_path=args.plan,
            data_home=args.data_home,
            data_archive=args.data_archive,
        )
    if report.valid:
        logger.info("valid nonpromoting v3 development artifact")
        return 0
    for error in report.errors:
        logger.error("invalid: %s", error)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
