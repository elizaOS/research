"""CLI for the frozen recurring pair-feature evidence artifact.

Generation requires an explicit new output path; existing files, including the
pinned canonical artifact, are immutable::

    python -m alberta_framework.evaluation.recurring_feature_cli \
        --output /new/path/recurring-feature-evidence.json

Generation intentionally exposes no seed, protocol, or threshold tuning
options.  The historical preregistered 30--59 run is the only promoted run;
executing its now-consumed schedule at a new path is reproducibility evidence,
not a new promotion. Verification never reruns or retunes the experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter

from alberta_framework.evaluation.recurring_feature_artifact import (
    load_recurring_feature_artifact,
    validate_recurring_feature_artifact,
    write_recurring_feature_artifact,
)
from alberta_framework.recurring_feature_gate import (
    RecurringFeatureGateResult,
    run_recurring_feature_gate,
)

DEFAULT_OUTPUT = Path("outputs/recurring_feature/evidence.v1.json")


def _resolved_new_output(path: Path) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve()
    canonical = DEFAULT_OUTPUT.expanduser().resolve()
    if resolved == canonical:
        raise FileExistsError(
            f"refusing to write pinned canonical artifact path: {resolved}"
        )
    if os.path.lexists(expanded) or os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing output path: {resolved}")
    return resolved


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Run or verify the frozen Gaussian/L2 recurring pair-feature gate.")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="new output path; existing files and the pinned default are never overwritten",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help="strictly validate an existing artifact without rerunning the gate",
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
        artifact = load_recurring_feature_artifact(path)
        validation = validate_recurring_feature_artifact(artifact)
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
    return 0 if validation.accepted else (1 if validation.valid else 2)


def main(
    argv: Sequence[str] | None = None,
    *,
    result: RecurringFeatureGateResult | None = None,
    gate_wall_seconds: float | None = None,
    generated_at: datetime | None = None,
) -> int:
    """Generate frozen evidence or verify a file; injection keeps tests cheap."""

    args = _parser().parse_args(argv)
    if args.verify is not None:
        return _verify(args.verify)

    try:
        output_path = _resolved_new_output(args.output)
        if result is None:
            started = perf_counter()
            evidence_result = run_recurring_feature_gate()
            elapsed = perf_counter() - started
        else:
            evidence_result = result
            elapsed = 0.0 if gate_wall_seconds is None else gate_wall_seconds
        artifact = write_recurring_feature_artifact(
            output_path,
            evidence_result,
            gate_wall_seconds=elapsed,
            generated_at=generated_at,
        )
        validation = validate_recurring_feature_artifact(artifact)
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
    return 0 if validation.accepted else (1 if validation.valid else 2)


if __name__ == "__main__":
    raise SystemExit(main())
