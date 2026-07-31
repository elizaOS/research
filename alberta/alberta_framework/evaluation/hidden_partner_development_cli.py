"""Run or verify the nonpromoting hidden-partner development artifact.

The two seed roles are deliberately separate:

* ``tuning-replay`` reconstructs the already consumed development schedule;
* ``robustness`` consumes the named robustness schedule.

Neither role can promote scientific evidence.  A structurally valid artifact
exits with status one when any development falsification check fails.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import jax

from alberta_framework.evaluation.hidden_partner_development import (
    DEFAULT_DEVELOPMENT_SEED_COUNT,
    HIDDEN_PARTNER_CONDITIONS,
    ROBUSTNESS_SEED_NAMESPACE,
    TUNING_SEED_NAMESPACE,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerRunSummary,
    derive_hidden_partner_seed_pairs,
    run_hidden_partner_development_suite,
)
from alberta_framework.evaluation.hidden_partner_development_artifact import (
    build_hidden_partner_development_artifact,
    hidden_partner_development_artifact_json,
    load_hidden_partner_development_artifact,
    source_snapshot,
    validate_hidden_partner_development_artifact,
)

DEFAULT_OUTPUT_DIRECTORY = Path("outputs/hidden_partner_development")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run or verify the development-only integrated hidden-partner "
            "benchmark; no invocation can promote a scientific claim."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--role",
        choices=("tuning-replay", "robustness"),
        help="run the exact named development seed role",
    )
    mode.add_argument(
        "--verify",
        type=Path,
        help="strictly validate an existing artifact without running a life",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_DEVELOPMENT_SEED_COUNT,
        help="paired development lives per condition (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=("new artifact path; defaults to outputs/hidden_partner_development/<role>.v1.json"),
    )
    return parser


def _emit(payload: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _verify(path: Path) -> int:
    try:
        artifact = load_hidden_partner_development_artifact(path)
        validation = validate_hidden_partner_development_artifact(artifact)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        _emit(
            {
                "artifact": str(path),
                "development_checks_passed": False,
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2
    _emit(
        {
            "artifact": str(path),
            "development_checks_passed": validation.development_checks_passed,
            "errors": list(validation.errors),
            "valid": validation.valid,
        }
    )
    if not validation.valid:
        return 2
    return 0 if validation.development_checks_passed else 1


def _role_namespace(role: str) -> str:
    return TUNING_SEED_NAMESPACE if role == "tuning-replay" else ROBUSTNESS_SEED_NAMESPACE


def main(
    argv: Sequence[str] | None = None,
    *,
    summaries: Sequence[HiddenPartnerRunSummary] | None = None,
    wall_seconds: float | None = None,
) -> int:
    """Run one named development schedule or verify an existing record.

    ``summaries`` and ``wall_seconds`` are test seams only.  Normal CLI use
    always executes all fixed v1 conditions on the derived paired seeds.
    """
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(parsed_argv)
    if args.verify is not None:
        return _verify(args.verify)

    role = str(args.role)
    output = DEFAULT_OUTPUT_DIRECTORY / f"{role}.v1.json" if args.output is None else args.output
    if output.exists():
        _emit(
            {
                "artifact": str(output),
                "development_checks_passed": False,
                "errors": ["refusing to overwrite an existing development artifact"],
                "valid": False,
            }
        )
        return 2

    if isinstance(args.count, bool) or args.count < 2:
        _emit(
            {
                "artifact": str(output),
                "development_checks_passed": False,
                "errors": ["count must be an integer of at least two paired lives"],
                "valid": False,
            }
        )
        return 2

    protocol = HiddenPartnerDevelopmentProtocol()
    namespace = _role_namespace(role)
    seed_pairs = derive_hidden_partner_seed_pairs(namespace, args.count)
    try:
        sources_before = source_snapshot()
        if summaries is None:
            started = perf_counter()
            completed_summaries = run_hidden_partner_development_suite(
                seed_pairs,
                protocol=protocol,
                condition_names=tuple(condition.name for condition in HIDDEN_PARTNER_CONDITIONS),
            )
            elapsed = perf_counter() - started
        else:
            completed_summaries = tuple(summaries)
            elapsed = 0.0 if wall_seconds is None else wall_seconds
        sources_after_run = source_snapshot()
        if sources_after_run != sources_before:
            raise RuntimeError(
                "pinned development sources changed during execution; no artifact was written"
            )

        operational_metadata: dict[str, object] = {
            "argv": parsed_argv,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "jax_backend": jax.default_backend(),
            "jax_device_count": jax.device_count(),
            "jax_version": jax.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "wall_seconds": float(elapsed),
        }
        artifact = build_hidden_partner_development_artifact(
            protocol,
            role,
            completed_summaries,
            operational_metadata=operational_metadata,
        )
        validation = validate_hidden_partner_development_artifact(artifact)
        if not validation.valid:
            raise RuntimeError(
                "generated artifact failed strict validation: " + "; ".join(validation.errors)
            )
        if source_snapshot() != sources_before:
            raise RuntimeError(
                "pinned development sources changed before serialization; no artifact was written"
            )
        serialized = hidden_partner_development_artifact_json(artifact)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (FileExistsError, OSError, RuntimeError, ValueError) as error:
        _emit(
            {
                "artifact": str(output),
                "development_checks_passed": False,
                "errors": [str(error)],
                "valid": False,
            }
        )
        return 2

    digest = artifact["scientific_digest"]
    digest_value = digest.get("sha256") if isinstance(digest, dict) else None
    _emit(
        {
            "artifact": str(output),
            "development_checks_passed": validation.development_checks_passed,
            "development_only": True,
            "errors": list(validation.errors),
            "run_role": role,
            "scientific_promotion_allowed": False,
            "scientific_sha256": digest_value,
            "seed_count": len(seed_pairs),
            "valid": validation.valid,
        }
    )
    return 0 if validation.development_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
