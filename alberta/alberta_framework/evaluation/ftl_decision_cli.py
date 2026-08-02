"""CLI for the frozen sparse-FTL decision-fidelity evidence artifact.

The only promoted run is already consumed: the frozen configuration on
held-out seeds 30--59, pinned immutably at
``outputs/ftl_decision/evidence.v1.json``.  Rerunning the frozen protocol is
reproducibility evidence, never a new promotion, and must write to a NEW
path::

    python -m alberta_framework.evaluation.ftl_decision_cli \
        --output /new/path/ftl-decision-evidence.json

Caveat: unlike ``recurring_feature_cli``, ``--output`` defaults to the pinned
canonical path and generation has no overwrite guard — it clobbers whatever
the target path holds.  Callers must supply a fresh path themselves.

Generation exposes no seed, learner, bootstrap, or threshold tuning flags.
Verification validates an existing artifact without rerunning JAX.

Exit status is 0 for accepted evidence, 1 for a valid scientific rejection,
and 2 for an invalid artifact or execution error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter

from alberta_framework.evaluation.ftl_decision_artifact import (
    load_ftl_decision_artifact,
    validate_ftl_decision_artifact,
    write_ftl_decision_artifact,
)
from alberta_framework.evaluation.ftl_decision_fidelity import (
    DecisionFidelityReport,
    run_ftl_decision_fidelity_evaluation,
)

# The sha-pinned immutable promoted artifact.  Nothing in this CLI refuses to
# overwrite it; pass --output with a new path for any rerun.
DEFAULT_OUTPUT = Path("outputs/ftl_decision/evidence.v1.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the frozen deterministic one-dimensional sparse-FTL "
            "known-reward open-loop decision-fidelity gate."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        type=Path,
        help="strictly validate an existing artifact without rerunning evidence",
    )
    return parser


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _verify(path: Path) -> int:
    try:
        artifact = load_ftl_decision_artifact(path)
        validation = validate_ftl_decision_artifact(artifact)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
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
    if validation.accepted:
        return 0
    return 1 if validation.valid else 2


def main(
    argv: Sequence[str] | None = None,
    *,
    report: DecisionFidelityReport | None = None,
    evaluation_wall_seconds: float | None = None,
    generated_at: datetime | None = None,
) -> int:
    """Generate frozen evidence or verify a file.

    Report and timing injection keep focused CLI tests from rerunning the
    scientific protocol.  They do not expose end-user tuning flags.
    """

    args = _parser().parse_args(argv)
    if args.verify is not None:
        return _verify(args.verify)

    try:
        if report is None:
            started = perf_counter()
            evidence_report = run_ftl_decision_fidelity_evaluation()
            elapsed = perf_counter() - started
        else:
            evidence_report = report
            elapsed = 0.0 if evaluation_wall_seconds is None else evaluation_wall_seconds
        artifact = write_ftl_decision_artifact(
            args.output,
            evidence_report,
            evaluation_wall_seconds=elapsed,
            generated_at=generated_at,
        )
        validation = validate_ftl_decision_artifact(artifact)
    except (OSError, RuntimeError, ValueError) as error:
        _emit(
            {
                "accepted": False,
                "artifact": str(args.output),
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    digest = artifact["scientific_digest"]
    digest_value = digest.get("sha256") if isinstance(digest, dict) else None
    scientific = artifact["scientific_payload"]
    aggregate = scientific.get("aggregate") if isinstance(scientific, dict) else None
    seed_count = aggregate.get("seed_count") if isinstance(aggregate, dict) else None
    _emit(
        {
            "accepted": validation.accepted,
            "artifact": str(args.output),
            "errors": list(validation.errors),
            "schema_version": artifact["schema_version"],
            "scientific_sha256": digest_value,
            "seed_count": seed_count,
            "valid": validation.valid,
        }
    )
    if validation.accepted:
        return 0
    return 1 if validation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
