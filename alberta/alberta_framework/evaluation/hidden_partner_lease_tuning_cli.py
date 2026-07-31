"""Run or verify the reserved v4 development-only evidence-lease grid."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import jax
import jaxlib.version
import numpy as np

from alberta_framework.evaluation.hidden_partner_lease_tuning_artifact import (
    build_lease_tuning_artifact,
    lease_tuning_artifact_json,
    load_lease_tuning_artifact,
    source_snapshot,
    validate_lease_tuning_artifact,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_v2 import (
    LEASE_TUNING_GRID,
    LEASE_TUNING_SEED_COUNT,
    require_lease_tuning_execution_allowed,
    run_evidence_lease_tuning_grid,
)

DEFAULT_OUTPUT = Path(
    "outputs/hidden_partner_development/lease_tuning_grid_a.v4.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the retired development-only evidence-lease schema. "
            "The reserved execution namespace remains forbidden."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--run",
        action="store_true",
        help="retained compatibility flag; the reserved namespace is forbidden",
    )
    mode.add_argument(
        "--verify",
        type=Path,
        help="strictly validate an existing artifact without running a life",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"retired artifact path (default: {DEFAULT_OUTPUT})",
    )
    return parser


def _emit(payload: Mapping[str, object]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _verification_payload(
    path: Path,
    *,
    valid: bool,
    feasible_cell_selected: bool,
    errors: Sequence[str],
) -> dict[str, object]:
    return {
        "artifact": str(path),
        "development_only": True,
        "errors": list(errors),
        "feasible_cell_selected": feasible_cell_selected,
        "scientific_promotion_allowed": False,
        "valid": valid,
    }


def _verify(path: Path) -> int:
    try:
        artifact = load_lease_tuning_artifact(path)
        validation = validate_lease_tuning_artifact(artifact)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        _emit(
            _verification_payload(
                path,
                valid=False,
                feasible_cell_selected=False,
                errors=(str(error),),
            )
        )
        return 2
    _emit(
        _verification_payload(
            path,
            valid=validation.valid,
            feasible_cell_selected=validation.feasible_cell_selected,
            errors=validation.errors,
        )
    )
    if not validation.valid:
        return 2
    return 0 if validation.feasible_cell_selected else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    record: Mapping[str, object] | None = None,
    wall_seconds: float | None = None,
) -> int:
    """Refuse the retired run path or verify one historical artifact.

    ``record`` and ``wall_seconds`` are legacy test seams only.  The reserved
    namespace is forbidden, so normal CLI use fails before execution or write.
    """
    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(parsed_argv)
    if args.verify is not None:
        return _verify(args.verify)

    output = Path(args.output)
    try:
        require_lease_tuning_execution_allowed()
    except RuntimeError as error:
        _emit(
            _verification_payload(
                output,
                valid=False,
                feasible_cell_selected=False,
                errors=(str(error),),
            )
        )
        return 2
    if output.exists():
        _emit(
            _verification_payload(
                output,
                valid=False,
                feasible_cell_selected=False,
                errors=("refusing to overwrite an existing tuning artifact",),
            )
        )
        return 2

    try:
        sources_before = source_snapshot()
        if record is None:
            started = perf_counter()
            completed_record = run_evidence_lease_tuning_grid()
            elapsed = perf_counter() - started
        else:
            completed_record = dict(record)
            elapsed = 0.0 if wall_seconds is None else float(wall_seconds)
        if source_snapshot() != sources_before:
            raise RuntimeError(
                "pinned tuning sources changed during execution; no artifact was written"
            )

        operational_metadata: dict[str, object] = {
            "argv": parsed_argv,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "jax_backend": jax.default_backend(),
            "jax_device_count": jax.device_count(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jaxlib_version": jaxlib.version.__version__,
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "wall_seconds": float(elapsed),
        }
        artifact = build_lease_tuning_artifact(
            completed_record,
            operational_metadata=operational_metadata,
        )
        validation = validate_lease_tuning_artifact(artifact)
        if not validation.valid:
            raise RuntimeError(
                "generated artifact failed strict validation: "
                + "; ".join(validation.errors)
            )
        if source_snapshot() != sources_before:
            raise RuntimeError(
                "pinned tuning sources changed before serialization; "
                "no artifact was written"
            )
        serialized = lease_tuning_artifact_json(artifact)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(serialized)
    except (
        FileExistsError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        _emit(
            _verification_payload(
                output,
                valid=False,
                feasible_cell_selected=False,
                errors=(str(error),),
            )
        )
        return 2

    digest = artifact["scientific_digest"]
    digest_value = digest.get("sha256") if isinstance(digest, dict) else None
    payload = artifact["scientific_payload"]
    selected = (
        payload.get("selected_cell") if isinstance(payload, Mapping) else None
    )
    _emit(
        {
            **_verification_payload(
                output,
                valid=True,
                feasible_cell_selected=validation.feasible_cell_selected,
                errors=validation.errors,
            ),
            "grid_cell_count": len(LEASE_TUNING_GRID),
            "run_count": (
                len(LEASE_TUNING_GRID) * LEASE_TUNING_SEED_COUNT
            ),
            "scientific_sha256": digest_value,
            "selected_cell": selected,
        }
    )
    return 0 if validation.feasible_cell_selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
