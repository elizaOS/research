"""CLI for the frozen held-out continual-IA evidence protocol.

Run the promoted 30-seed evaluation with::

    python -m alberta_framework.evaluation.continual_ia_cli \
        --output outputs/continual_ia/evidence.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from alberta_framework.evaluation.continual_ia import (
    PROMOTED_EVIDENCE_SEEDS,
    ContinualIAReport,
    IAAcceptanceThresholds,
    run_continual_ia_benchmark,
)
from alberta_framework.evaluation.continual_ia_artifact import (
    build_ia_evidence_artifact,
    ia_artifact_json,
    load_ia_evidence_artifact,
    validate_ia_evidence_artifact,
)

DEFAULT_OUTPUT = Path("outputs/continual_ia/evidence.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run or verify the frozen hidden-phase causal IA evidence gate.")
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        type=Path,
        help="validate an existing artifact instead of running",
    )
    return parser


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _verify(path: Path) -> int:
    try:
        artifact = load_ia_evidence_artifact(path)
        validation = validate_ia_evidence_artifact(artifact)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(path),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(path),
            "errors": list(validation.errors),
            "valid": validation.valid,
        }
    )
    if not validation.valid:
        return 2
    return 0 if validation.accepted else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    report: ContinualIAReport | None = None,
) -> int:
    """Run/write or verify evidence; report injection avoids test reruns."""

    args = _parser().parse_args(argv)
    if args.verify is not None:
        return _verify(args.verify)
    thresholds = IAAcceptanceThresholds()
    seeds = PROMOTED_EVIDENCE_SEEDS
    if report is None:
        evidence_report = run_continual_ia_benchmark(
            seeds=seeds,
            thresholds=thresholds,
        )
    else:
        if report.aggregate.seeds != seeds:
            _emit(
                {
                    "accepted": False,
                    "errors": ["injected report seeds do not match the requested schedule"],
                    "valid": False,
                }
            )
            return 2
        if report.thresholds != thresholds:
            _emit(
                {
                    "accepted": False,
                    "errors": ["injected report thresholds are not the frozen v1 thresholds"],
                    "valid": False,
                }
            )
            return 2
        evidence_report = report

    artifact = build_ia_evidence_artifact(evidence_report)
    validation = validate_ia_evidence_artifact(artifact)
    if not validation.valid:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": list(validation.errors),
                "valid": False,
            }
        )
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(ia_artifact_json(artifact), encoding="utf-8")
    digest = artifact["content_digest"]
    digest_value = digest.get("sha256") if isinstance(digest, dict) else None
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(args.output),
            "content_sha256": digest_value,
            "errors": list(validation.errors),
            "primary_passed": evidence_report.acceptance.primary_passed,
            "schema_version": artifact["schema_version"],
            "secondary_passed": evidence_report.acceptance.secondary_passed,
            "seed_count": len(evidence_report.aggregate.seeds),
            "valid": validation.valid,
        }
    )
    return 0 if validation.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
