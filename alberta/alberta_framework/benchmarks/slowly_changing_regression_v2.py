"""Strict development protocol for slowly-changing regression.

This module deliberately does **not** claim an exact replication of Dohare
et al. (2024).  It provides a publication-shaped development experiment with
three explicitly different method roles:

* ``publication_bp_relu_sgd`` is a selected ordinary-backprop comparator arm.
  Its learner uses ReLU Kaiming-uniform initialization and the gradient of
  PyTorch's mean-squared-error convention (including the factor of two).
* ``alberta_cbp_relu_local_extension`` and
  ``alberta_upgd_relu_local_extension`` are local Alberta extensions.  They
  are not implementations imported from the publication's source tree.

The v2 protocol is sharded one method/seed at a time.  A self-issued development
plan binds the run specification, relevant source bytes, runtime bytes, and a
derived prescribed-command identity.  Shards bind that plan, their own derived
command identity, and the deterministic environment identity.  Merge exactly replays every
shard before publishing an artifact that binds every shard by size and SHA-256
and contains descriptive summaries only.  Ordinary validation replays again;
structural-only diagnostics can never return ``valid=True``.

The schema is permanently nonpromoting.  Its self-issued timestamps are useful
diagnostics, but they are not independently attested execution chronology.
Structural validity therefore never implies protocol exactness, scientific
promotion, an inferential result, or a SOTA claim.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import secrets
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array, lax
from jaxtyping import Float

from alberta_framework.benchmarks.slowly_changing_regression import (
    SCRLearnerParams,
    SlowlyChangingRegressionConfig,
    build_scr_learner,
    make_scr_env,
    scr_example,
)

logger = logging.getLogger(__name__)

SCR_V2_PLAN_SCHEMA = "alberta.slowly_changing_regression.run_plan.v2"
SCR_V2_SHARD_SCHEMA = "alberta.slowly_changing_regression.shard.v2"
SCR_V2_ARTIFACT_SCHEMA = "alberta.slowly_changing_regression.artifact.v2"
SCR_V2_SOURCE_MANIFEST_SCHEMA = "alberta.slowly_changing_regression.source_manifest.v2"
SCR_V2_RUNTIME_MANIFEST_SCHEMA = "alberta.slowly_changing_regression.runtime_manifest.v2"
SCR_V2_ENVIRONMENT_IDENTITY_SCHEMA = "alberta.slowly_changing_regression.environment_identity.v1"
SCR_V2_COMMAND_PROVENANCE_SCHEMA = (
    "alberta.slowly_changing_regression.command_provenance.v1"
)
SCR_V2_SHARD_RESERVATION_SCHEMA = (
    "alberta.slowly_changing_regression.shard_reservation.v2"
)
SCR_V2_EXECUTION_ENVELOPE_SCHEMA = (
    "alberta.slowly_changing_regression.execution_envelope.self_issued_development.v2"
)

SCR_V2_BENCHMARK = "slowly_changing_regression"
SCR_V2_EVIDENCE_ROLE = "development_publication_shaped_extension"
SCR_V2_REFERENCE_ID = (
    "dohare_et_al_2024_loss_of_plasticity_slowly_changing_regression"
    "@loss-of-plasticity-v1.1-d626b017e403d94335f1c64f9d19f3d6a96af962"
)

PUBLICATION_BP_METHOD = "publication_bp_relu_sgd"
LOCAL_CBP_METHOD = "alberta_cbp_relu_local_extension"
LOCAL_UPGD_METHOD = "alberta_upgd_relu_local_extension"
SCR_V2_METHOD_IDS = (
    PUBLICATION_BP_METHOD,
    LOCAL_CBP_METHOD,
    LOCAL_UPGD_METHOD,
)

_METHOD_ROLES = {
    PUBLICATION_BP_METHOD: "selected_publication_backprop_comparator_arm",
    LOCAL_CBP_METHOD: "local_continual_backprop_extension",
    LOCAL_UPGD_METHOD: "local_upgd_extension",
}

_SOURCE_ROOT_MODULES = (
    "alberta_framework.benchmarks.slowly_changing_regression",
    "alberta_framework.benchmarks.slowly_changing_regression_v2",
)
_SOURCE_AUXILIARY_PATHS = (
    "pyproject.toml",
    "uv.lock",
)

_DATA_CONTRACT: dict[str, Any] = {
    "kind": "deterministic_online_synthetic_stream",
    "external_dataset_required": False,
    "generator": "scr_example",
    "environment_identity_binds": [
        "task_configuration",
        "target_network",
        "slow_bit_schedule",
        "fast_bit_prng_key",
    ],
    "same_seed_environment_shared_across_methods": True,
}

_RUNTIME_ENVIRONMENT_NAMES = (
    "JAX_DEFAULT_MATMUL_PRECISION",
    "JAX_DEFAULT_PRNG_IMPL",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
)

_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_REGULAR_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_INT32_MAX = 0x7FFF_FFFF
_FUTURE_TIMESTAMP_TOLERANCE_SECONDS = 5
_RUNTIME_DISTRIBUTIONS = ("chex", "jax", "jaxlib", "numpy", "jaxtyping")

_PLAN_COMMANDS = {
    "run_shard": (
        "python -m alberta_framework.benchmarks.slowly_changing_regression "
        "run-shard --plan <PLAN> --method <METHOD> --seed-id <SEED>"
    ),
    "merge": (
        "python -m alberta_framework.benchmarks.slowly_changing_regression "
        "merge --plan <PLAN> (--shards-dir <DIR>|--shard <SHARD>...) --output <ARTIFACT>"
    ),
    "validate": (
        "python -m alberta_framework.benchmarks.slowly_changing_regression "
        "validate --artifact <ARTIFACT>  # exact replay is mandatory for valid=True"
    ),
}

_INTERPRETATION: dict[str, Any] = {
    "kind": "descriptive_only",
    "post_hoc_thresholds_used": False,
    "inferential_claim_allowed": False,
    "sota_claim_allowed": False,
    "protocol_exact_replication_claim_allowed": False,
    "notes": [
        "No pass/fail threshold is computed from the observed curves.",
        "Exact replay establishes deterministic reconstruction, not external attestation.",
        "The ordinary-BP result covers one selected ReLU/SGD arm, not the publication sweep.",
        "The Alberta CBP and UPGD methods are local extensions, not publication comparators.",
    ],
}


class SCRV2ValidationError(ValueError):
    """Raised when a v2 plan, shard, or artifact fails closed."""


@dataclass(frozen=True)
class SCRV2ValidationReport:
    """Public validation result; computation and structure are distinguished."""

    valid: bool
    scientific_promotion_allowed: bool
    errors: tuple[str, ...]
    structurally_valid: bool = False
    computational_replay_performed: bool = False


@chex.dataclass(frozen=True)
class PublicationBPState:
    """State for the selected publication-style ordinary-BP comparator."""

    hidden_weights: Float[Array, "hidden input"]
    hidden_bias: Float[Array, " hidden"]
    output_weights: Float[Array, " hidden"]
    output_bias: Float[Array, ""]
    step_count: Array


def _fail(message: str) -> NoReturn:
    raise SCRV2ValidationError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_float(value: object) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _expect_dict(value: object, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} must be an object")
    return cast(dict[str, Any], value)


def _expect_list(value: object, where: str) -> list[Any]:
    _require(isinstance(value, list), f"{where} must be an array")
    return cast(list[Any], value)


def _expect_exact_keys(value: Mapping[str, Any], keys: set[str], where: str) -> None:
    actual = set(value)
    _require(
        actual == keys,
        f"{where} keys differ: missing={sorted(keys - actual)}, extra={sorted(actual - keys)}",
    )


def _reject_json_constant(token: str) -> NoReturn:
    _fail(f"non-finite JSON constant is forbidden: {token}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _check_finite_tree(value: object, where: str = "$") -> None:
    if isinstance(value, float):
        _require(math.isfinite(value), f"{where} contains a non-finite float")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_finite_tree(item, f"{where}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _check_finite_tree(item, f"{where}.{key}")


def _json_exact_equal(actual: object, expected: object) -> bool:
    """Compare JSON trees without Python's boolean/numeric coercions."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict) and isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _json_exact_equal(actual[key], expected[key]) for key in actual
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _json_exact_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _data_contract_dict() -> dict[str, Any]:
    return {
        "kind": _DATA_CONTRACT["kind"],
        "external_dataset_required": _DATA_CONTRACT["external_dataset_required"],
        "generator": _DATA_CONTRACT["generator"],
        "environment_identity_binds": list(_DATA_CONTRACT["environment_identity_binds"]),
        "same_seed_environment_shared_across_methods": _DATA_CONTRACT[
            "same_seed_environment_shared_across_methods"
        ],
    }


def _interpretation_dict() -> dict[str, Any]:
    return {
        "kind": _INTERPRETATION["kind"],
        "post_hoc_thresholds_used": _INTERPRETATION["post_hoc_thresholds_used"],
        "inferential_claim_allowed": _INTERPRETATION["inferential_claim_allowed"],
        "sota_claim_allowed": _INTERPRETATION["sota_claim_allowed"],
        "protocol_exact_replication_claim_allowed": _INTERPRETATION[
            "protocol_exact_replication_claim_allowed"
        ],
        "notes": list(_INTERPRETATION["notes"]),
    }


def strict_scr_json_loads(text: str) -> Any:
    """Decode JSON while rejecting duplicate keys and non-finite numbers."""

    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
        _check_finite_tree(value)
    except SCRV2ValidationError:
        raise
    except (RecursionError, ValueError, UnicodeError) as exc:
        raise SCRV2ValidationError(f"invalid JSON: {exc}") from exc
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SCRV2ValidationError(f"value is not strict JSON: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _compact_json_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise SCRV2ValidationError(f"value is not strict JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_compact_json_bytes(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_path(path: Path) -> str:
    return _lexical_absolute(path).as_posix()


def _canonical_absolute_locator(value: object, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where} must be a nonempty string")
    path = Path(cast(str, value))
    _require(path.is_absolute(), f"{where} must be absolute")
    canonical = _canonical_path(path)
    _require(value == canonical, f"{where} must be lexically canonical")
    return canonical


def _require_not_future_unix(value: object, where: str) -> int:
    _require(
        _is_int(value) and cast(int, value) >= 0,
        f"{where} must be a nonnegative integer",
    )
    timestamp = cast(int, value)
    _require(
        timestamp <= int(time.time()) + _FUTURE_TIMESTAMP_TOLERANCE_SECONDS,
        f"{where} cannot be in the future",
    )
    return timestamp


def _canonical_plan_semantic_argv(
    run_spec_sha256: str,
    source_manifest_sha256: str,
    runtime_manifest_sha256: str,
) -> list[str]:
    return [
        "plan",
        "--run-spec-sha256",
        run_spec_sha256,
        "--source-manifest-sha256",
        source_manifest_sha256,
        "--runtime-manifest-sha256",
        runtime_manifest_sha256,
    ]


def _canonical_shard_semantic_argv(
    plan_sha256: str,
    method_id: str,
    seed_id: int,
    output: Path,
) -> list[str]:
    return [
        "run-shard",
        "--plan-sha256",
        plan_sha256,
        "--method",
        method_id,
        "--seed-id",
        str(seed_id),
        "--output",
        _canonical_path(output),
    ]


def _canonical_merge_semantic_argv(
    plan_sha256: str,
    shard_sha256s: Sequence[str],
    output: Path,
) -> list[str]:
    argv = ["merge", "--plan-sha256", plan_sha256]
    for digest in shard_sha256s:
        argv.extend(("--shard-sha256", digest))
    argv.extend(("--output", _canonical_path(output)))
    return argv


def _build_command_provenance(
    canonical_semantic_argv: Sequence[str],
    *,
    invocation_origin: str,
    process_argv: Sequence[str] | None,
) -> dict[str, Any]:
    canonical = list(canonical_semantic_argv)
    _require(
        bool(canonical) and all(isinstance(item, str) and bool(item) for item in canonical),
        "canonical semantic argv must be a nonempty string array",
    )
    _require(invocation_origin in {"cli", "direct_api"}, "invalid invocation origin")
    if invocation_origin == "direct_api":
        _require(process_argv is None, "direct API provenance cannot claim process argv")
        reported: list[str] | None = None
        reported_sha256: str | None = None
    else:
        _require(process_argv is not None, "CLI provenance requires self-reported process argv")
        reported = list(cast(Sequence[str], process_argv))
        _require(
            bool(reported) and all(isinstance(item, str) for item in reported),
            "self-reported process argv must be a nonempty string array",
        )
        reported_sha256 = _sha256_json(reported)
    return {
        "schema": SCR_V2_COMMAND_PROVENANCE_SCHEMA,
        "invocation_origin": invocation_origin,
        "canonical_semantic_argv": canonical,
        "canonical_semantic_argv_sha256": _sha256_json(canonical),
        "self_reported_process_argv": reported,
        "self_reported_process_argv_sha256": reported_sha256,
    }


def _prevalidate_invocation_provenance(
    *,
    invocation_origin: str,
    process_argv: Sequence[str] | None,
) -> None:
    """Reject impossible provenance before source discovery or numerical work."""

    _build_command_provenance(
        ("preflight-only-prescribed-command",),
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )


def _validate_command_provenance(
    value: object,
    expected_canonical_semantic_argv: Sequence[str],
    where: str,
) -> dict[str, Any]:
    provenance = _expect_dict(value, where)
    _expect_exact_keys(
        provenance,
        {
            "schema",
            "invocation_origin",
            "canonical_semantic_argv",
            "canonical_semantic_argv_sha256",
            "self_reported_process_argv",
            "self_reported_process_argv_sha256",
        },
        where,
    )
    _require(
        provenance["schema"] == SCR_V2_COMMAND_PROVENANCE_SCHEMA,
        f"{where} has wrong schema",
    )
    canonical = _expect_list(
        provenance["canonical_semantic_argv"], f"{where}.canonical_semantic_argv"
    )
    expected = list(expected_canonical_semantic_argv)
    _require(
        _json_exact_equal(canonical, expected),
        f"{where}.canonical_semantic_argv differs from derived command identity",
    )
    _require(
        provenance["canonical_semantic_argv_sha256"] == _sha256_json(canonical),
        f"{where}.canonical_semantic_argv_sha256 mismatch",
    )
    origin = provenance["invocation_origin"]
    _require(origin in {"cli", "direct_api"}, f"{where}.invocation_origin invalid")
    reported = provenance["self_reported_process_argv"]
    reported_sha256 = provenance["self_reported_process_argv_sha256"]
    if origin == "direct_api":
        _require(
            reported is None and reported_sha256 is None,
            f"{where} direct API invocation cannot claim process argv",
        )
    else:
        raw_argv = _expect_list(reported, f"{where}.self_reported_process_argv")
        _require(
            bool(raw_argv) and all(isinstance(item, str) for item in raw_argv),
            f"{where}.self_reported_process_argv invalid",
        )
        _require(
            reported_sha256 == _sha256_json(raw_argv),
            f"{where}.self_reported_process_argv_sha256 mismatch",
        )
    return provenance


def _lexical_absolute(path: Path) -> Path:
    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, TypeError, ValueError) as exc:
        raise SCRV2ValidationError(f"invalid filesystem path {path!r}: {exc}") from exc


def _open_parent_directory(path: Path, *, create: bool) -> tuple[Path, int]:
    """Open a stable parent directory descriptor without following symlinks."""

    destination = _lexical_absolute(path)
    _require(destination != destination.parent, "filesystem path must name a file")
    root = destination.anchor or os.sep
    directory_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in destination.parent.parts[1:]:
            created_component = False
            if create:
                try:
                    os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                    created_component = True
                except FileExistsError:
                    pass
            next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            if created_component:
                # Persist both the new directory inode and its entry in the
                # already-open parent before descending into it.
                try:
                    os.fsync(next_fd)
                    os.fsync(directory_fd)
                except BaseException:
                    os.close(next_fd)
                    raise
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        os.close(directory_fd)
        raise
    return destination, directory_fd


def _assert_parent_locator_stable(destination: Path, directory_fd: int) -> None:
    """Require the held parent descriptor to remain at its requested locator."""

    verified_destination, verification_fd = _open_parent_directory(
        destination, create=False
    )
    try:
        opened = os.fstat(directory_fd)
        current = os.fstat(verification_fd)
        _require(
            (opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino)
            and verified_destination == destination,
            f"ancestor directory changed while accessing lifecycle path: {destination}",
        )
    finally:
        os.close(verification_fd)


def _stable_stat_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_bytes(path: Path, *, require_immutable: bool) -> bytes:
    destination, directory_fd = _open_parent_directory(path, create=False)
    file_fd = -1
    try:
        _assert_parent_locator_stable(destination, directory_fd)
        file_fd = os.open(destination.name, _REGULAR_READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        _require(stat.S_ISREG(before.st_mode), f"{destination} must be a regular file")
        if require_immutable:
            _require(
                stat.S_IMODE(before.st_mode) & 0o222 == 0,
                f"{destination} must have no write permission bits",
            )
            _require(before.st_nlink == 1, f"{destination} must have exactly one hard link")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        locator = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            _stable_stat_signature(before)
            == _stable_stat_signature(after)
            == _stable_stat_signature(locator),
            f"{destination} changed or its locator was replaced while it was being read",
        )
        raw = b"".join(chunks)
        _require(len(raw) == before.st_size, f"{destination} size changed while it was being read")
        _assert_parent_locator_stable(destination, directory_fd)
        return raw
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(directory_fd)


def _atomic_write_new(path: Path, data: bytes) -> Path:
    """Durably publish descriptor-held immutable bytes without replacing paths."""

    destination, directory_fd = _open_parent_directory(path, create=True)
    temporary_name = ""
    file_fd = -1
    target_linked = False
    completed = False
    try:
        _assert_parent_locator_stable(destination, directory_fd)
        try:
            os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
        for _ in range(128):
            candidate = f".{destination.name}.{secrets.token_hex(16)}.tmp"
            try:
                file_fd = os.open(
                    candidate,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        _require(file_fd >= 0, "could not allocate a unique temporary output")
        remaining = memoryview(data)
        while remaining:
            written = os.write(file_fd, remaining)
            _require(written > 0, "short write while publishing immutable output")
            remaining = remaining[written:]
        os.fchmod(file_fd, 0o444)
        os.fsync(file_fd)
        _assert_parent_locator_stable(destination, directory_fd)
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite immutable output: {destination}"
            ) from exc
        target_linked = True
        source = os.fstat(file_fd)
        target = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            (source.st_dev, source.st_ino, source.st_size)
            != (target.st_dev, target.st_ino, target.st_size)
            or not stat.S_ISREG(target.st_mode)
            or stat.S_IMODE(target.st_mode) != 0o444
        ):
            os.unlink(destination.name, dir_fd=directory_fd)
            target_linked = False
            os.fsync(directory_fd)
            _fail("published path does not identify the descriptor-held temporary file")
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        temporary_name = ""
        os.fsync(directory_fd)
        final_target = os.stat(
            destination.name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        _require(
            (final_target.st_dev, final_target.st_ino, final_target.st_size)
            == (source.st_dev, source.st_ino, source.st_size)
            and final_target.st_nlink == 1
            and stat.S_ISREG(final_target.st_mode)
            and stat.S_IMODE(final_target.st_mode) == 0o444,
            "published output identity, mode, or link count changed",
        )
        _assert_parent_locator_stable(destination, directory_fd)
        try:
            published = _read_regular_bytes(destination, require_immutable=True)
            _require(published == data, "published output bytes differ from supplied bytes")
        except BaseException:
            try:
                os.unlink(destination.name, dir_fd=directory_fd)
                target_linked = False
                os.fsync(directory_fd)
            except FileNotFoundError:
                pass
            raise
        _assert_parent_locator_stable(destination, directory_fd)
        completed = True
        return destination
    finally:
        if target_linked and not completed and file_fd >= 0:
            try:
                source = os.fstat(file_fd)
                target = os.stat(
                    destination.name,
                    dir_fd=directory_fd,
                    follow_symlinks=False,
                )
                if (source.st_dev, source.st_ino) == (target.st_dev, target.st_ino):
                    os.unlink(destination.name, dir_fd=directory_fd)
                    os.fsync(directory_fd)
            except FileNotFoundError:
                pass
        if file_fd >= 0:
            os.close(file_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)


def _preflight_new_output(path: Path) -> Path:
    """Reject an occupied output before expensive work; publication rechecks atomically."""

    destination, directory_fd = _open_parent_directory(path, create=True)
    try:
        _assert_parent_locator_stable(destination, directory_fd)
        try:
            os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return destination
        raise FileExistsError(f"refusing to overwrite immutable output: {destination}")
    finally:
        os.close(directory_fd)


def _read_strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = _read_regular_bytes(path, require_immutable=True)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SCRV2ValidationError(f"{path} is not UTF-8 JSON") from exc
    value = strict_scr_json_loads(text)
    return raw, _expect_dict(value, str(path))


def _require_exact_reread(path: Path, expected: bytes, context: str) -> None:
    current = _read_regular_bytes(path, require_immutable=True)
    _require(current == expected, f"{context} bytes changed during the operation")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _module_path(module: str) -> Path | None:
    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    relative = Path(*module.split("."))
    source = _repo_root() / relative.with_suffix(".py")
    if source.is_file():
        return source
    package = _repo_root() / relative / "__init__.py"
    return package if package.is_file() else None


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(_repo_root())
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _parent_packages(module: str) -> set[str]:
    parts = module.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts))}


def _resolve_local_imports(path: Path, raw: bytes) -> set[str]:
    module, is_package = _module_name(path)
    package = module if is_package else module.rpartition(".")[0]
    try:
        tree = ast.parse(raw, filename=str(path))
    except (SyntaxError, UnicodeError) as exc:
        raise SCRV2ValidationError(f"cannot parse source closure member {path}: {exc}") from exc
    discovered: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                if keep < 0:
                    continue
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if base:
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for candidate in candidates:
            parts = candidate.split(".")
            while parts:
                name = ".".join(parts)
                if _module_path(name) is not None:
                    discovered.add(name)
                    discovered.update(_parent_packages(name))
                    break
                parts.pop()
    return discovered


def _source_closure_snapshot() -> dict[str, bytes]:
    pending = set(_SOURCE_ROOT_MODULES)
    for root in _SOURCE_ROOT_MODULES:
        pending.update(_parent_packages(root))
    visited: set[str] = set()
    bytes_by_module: dict[str, bytes] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        path = _module_path(module)
        _require(path is not None, f"source closure module is missing: {module}")
        assert path is not None
        visited.add(module)
        raw = _read_regular_bytes(path, require_immutable=False)
        bytes_by_module[module] = raw
        pending.update(_resolve_local_imports(path, raw) - visited)
    source_root = _repo_root()
    snapshot = {
        cast(Path, _module_path(module)).relative_to(source_root).as_posix(): bytes_by_module[
            module
        ]
        for module in visited
    }
    for relative in _SOURCE_AUXILIARY_PATHS:
        snapshot[relative] = _read_regular_bytes(
            source_root / relative, require_immutable=False
        )
    for relative, expected in snapshot.items():
        _require(
            _read_regular_bytes(source_root / relative, require_immutable=False) == expected,
            f"source file changed while its manifest was being built: {relative}",
        )
    return dict(sorted(snapshot.items()))


def _source_closure_paths() -> tuple[str, ...]:
    return tuple(_source_closure_snapshot())


def _build_source_manifest() -> dict[str, Any]:
    snapshot = _source_closure_snapshot()
    files: list[dict[str, Any]] = []
    for relative, data in snapshot.items():
        files.append(
            {
                "path": relative,
                "byte_size": len(data),
                "sha256": _sha256_bytes(data),
            }
        )
    return {
        "schema": SCR_V2_SOURCE_MANIFEST_SCHEMA,
        "scope": "static_transitive_local_python_imports_plus_lockfiles",
        "root_modules": list(_SOURCE_ROOT_MODULES),
        "files": files,
    }


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _distribution_content_identity(name: str) -> dict[str, Any]:
    """Bind the installed distribution's regular-file bytes, not only its version."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {
            "status": "not_installed",
            "file_count": 0,
            "total_bytes": 0,
            "sha256": _sha256_bytes(b""),
        }
    files = distribution.files
    _require(files is not None, f"installed distribution {name!r} has no file manifest")
    assert files is not None
    digest = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for relative in sorted(files, key=str):
        path = Path(cast(Any, distribution.locate_file(relative))).resolve()
        if not path.is_file():
            continue
        raw = _read_regular_bytes(path, require_immutable=False)
        locator = str(relative).replace(os.sep, "/")
        digest.update(locator.encode("utf-8") + b"\0")
        digest.update(str(len(raw)).encode("ascii") + b"\0")
        digest.update(hashlib.sha256(raw).digest())
        file_count += 1
        total_bytes += len(raw)
    _require(file_count > 0, f"installed distribution {name!r} has no regular files")
    return {
        "status": "content_hashed",
        "file_count": file_count,
        "total_bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _runtime_json_value(value: object) -> str | int | float | bool | None:
    if value is None or type(value) in (str, int, bool):
        return cast(str | int | bool | None, value)
    if isinstance(value, float):
        _require(math.isfinite(value), "JAX runtime configuration contains a non-finite float")
        return value
    return str(value)


def _discover_runtime_manifest() -> dict[str, Any]:
    executable_path = Path(sys.executable).resolve(strict=True)
    executable_raw = _read_regular_bytes(executable_path, require_immutable=False)
    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "platform_version": str(getattr(device.client, "platform_version", "unknown")),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
            "runtime_type": str(getattr(device.client, "runtime_type", "unknown")),
        }
        for device in jax.devices()
    ]
    return {
        "schema": SCR_V2_RUNTIME_MANIFEST_SCHEMA,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": {
            "locator": executable_path.as_posix(),
            "byte_size": len(executable_raw),
            "sha256": _sha256_bytes(executable_raw),
        },
        "jax": jax.__version__,
        "jaxlib": _distribution_version("jaxlib"),
        "numpy": np.__version__,
        "chex": _distribution_version("chex"),
        "jaxtyping": _distribution_version("jaxtyping"),
        "distribution_content": {
            name: _distribution_content_identity(name)
            for name in _RUNTIME_DISTRIBUTIONS
        },
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "jax_backend": jax.default_backend(),
        "jax_devices": devices,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "jax_default_prng_impl": str(jax.config.jax_default_prng_impl),
        "jax_numpy_dtype_promotion": str(jax.config.jax_numpy_dtype_promotion),
        "jax_config": {
            name: _runtime_json_value(value)
            for name, value in sorted(jax.config.values.items())
        },
        "execution_environment": {
            name: os.environ.get(name) for name in _RUNTIME_ENVIRONMENT_NAMES
        },
    }


def _build_runtime_manifest() -> dict[str, Any]:
    """Discover a byte-bound runtime or raise one closed protocol error."""

    try:
        return _discover_runtime_manifest()
    except SCRV2ValidationError:
        raise
    except Exception as exc:
        raise SCRV2ValidationError(f"runtime discovery failed closed: {exc}") from exc


def _validate_method_ids(method_ids: Sequence[str]) -> tuple[str, ...]:
    _require(bool(method_ids), "at least one method must be planned")
    _require(all(isinstance(item, str) for item in method_ids), "method IDs must be strings")
    _require(len(set(method_ids)) == len(method_ids), "method IDs must be unique")
    unknown = set(method_ids) - set(SCR_V2_METHOD_IDS)
    _require(not unknown, f"unknown v2 method IDs: {sorted(unknown)}")
    canonical = tuple(item for item in SCR_V2_METHOD_IDS if item in method_ids)
    _require(tuple(method_ids) == canonical, f"methods must use canonical order {canonical}")
    return canonical


def _validate_seed_ids(seed_ids: Sequence[int]) -> tuple[int, ...]:
    _require(bool(seed_ids), "at least one seed must be planned")
    _require(
        all(_is_int(seed) and 0 <= seed <= 0xFFFF_FFFF for seed in seed_ids),
        "seed IDs must be uint32 integers",
    )
    result = tuple(int(seed) for seed in seed_ids)
    _require(len(set(result)) == len(result), "seed IDs must be unique")
    _require(result == tuple(sorted(result)), "seed IDs must be sorted")
    return result


def _validate_learner_params(params: SCRLearnerParams) -> None:
    _require(
        _is_int(params.hidden_units) and 1 <= params.hidden_units <= _INT32_MAX,
        "hidden_units must fit positive int32",
    )
    _require(
        _is_finite_float(params.step_size) and params.step_size > 0.0,
        "step_size must be a finite float and > 0",
    )
    _require(
        _is_finite_float(params.cbp_replacement_rate)
        and 0.0 <= params.cbp_replacement_rate <= 1.0,
        "cbp_replacement_rate must be in [0, 1]",
    )
    _require(
        _is_int(params.cbp_maturity_threshold)
        and 0 <= params.cbp_maturity_threshold <= _INT32_MAX,
        "cbp_maturity_threshold must fit nonnegative int32",
    )
    for name, value in (
        ("cbp_decay_rate", params.cbp_decay_rate),
        ("upgd_utility_decay", params.upgd_utility_decay),
    ):
        _require(
            _is_finite_float(value) and 0.0 <= value <= 1.0,
            f"{name} must be a finite float in [0, 1]",
        )
    _require(
        _is_finite_float(params.upgd_sigma) and params.upgd_sigma >= 0.0,
        "upgd_sigma must be a finite float and >= 0",
    )
    _require(
        _is_finite_float(params.upgd_beta) and params.upgd_beta >= 0.0,
        "upgd_beta must be a finite float and >= 0",
    )


def _validate_task_config(config: SlowlyChangingRegressionConfig) -> None:
    for name, value in (
        ("num_bits", config.num_bits),
        ("num_flipping_bits", config.num_flipping_bits),
        ("flip_period", config.flip_period),
        ("target_hidden_units", config.target_hidden_units),
        ("num_examples", config.num_examples),
    ):
        _require(
            _is_int(value) and 1 <= value <= _INT32_MAX,
            f"{name} must fit positive int32",
        )
    _require(
        _is_finite_float(config.ltu_beta) and 0.0 <= config.ltu_beta <= 1.0,
        "ltu_beta must be a finite float in [0, 1]",
    )
    try:
        config.validate()
    except ValueError as exc:
        raise SCRV2ValidationError(f"invalid task configuration: {exc}") from exc


def _task_dict(config: SlowlyChangingRegressionConfig) -> dict[str, Any]:
    return {
        "num_bits": config.num_bits,
        "num_flipping_bits": config.num_flipping_bits,
        "flip_period": config.flip_period,
        "target_hidden_units": config.target_hidden_units,
        "ltu_beta": config.ltu_beta,
        "num_examples": config.num_examples,
    }


def _learner_params_dict(params: SCRLearnerParams) -> dict[str, Any]:
    return {
        "hidden_units": params.hidden_units,
        "step_size": params.step_size,
        "cbp_replacement_rate": params.cbp_replacement_rate,
        "cbp_maturity_threshold": params.cbp_maturity_threshold,
        "cbp_decay_rate": params.cbp_decay_rate,
        "upgd_sigma": params.upgd_sigma,
        "upgd_utility_decay": params.upgd_utility_decay,
        "upgd_beta": params.upgd_beta,
    }


def _selected_configuration_match(
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    method_ids: Sequence[str],
    bin_size: int,
) -> dict[str, str]:
    task_matches = (
        config.num_bits == 20
        and config.num_flipping_bits == 15
        and config.flip_period == 10_000
        and config.target_hidden_units == 100
        and config.ltu_beta == 0.7
        and config.num_examples == 3_000_000
        and bin_size == 40_000
    )
    arm_matches = (
        PUBLICATION_BP_METHOD in method_ids
        and params.hidden_units == 5
        and params.step_size == 0.01
    )
    return {
        "scope": "nature_task_shape_and_pinned_source_relu_sgd_0p01_arm_only",
        "task_shape": "match" if task_matches else "mismatch",
        "selected_publication_bp_arm": "match" if arm_matches else "mismatch",
    }


def _closed_deviations() -> list[dict[str, str]]:
    return [
        {
            "id": "article_vs_pinned_source_scale",
            "publication_reference": "nature_methods_3m_examples_and_40000_example_bins",
            "development_execution": (
                "follows_nature_scale_while_pinned_v1p1_relu_config_and_plot_use_1m_and_20000"
            ),
        },
        {
            "id": "implementation_and_rng",
            "publication_reference": "pytorch_reference_source_and_torch_rng",
            "development_execution": "jax_reimplementation_and_explicit_jax_seed_ids",
        },
        {
            "id": "data_materialization_and_reuse",
            "publication_reference": "pregenerated_persisted_sequence_reused_across_learner_arms",
            "development_execution": (
                "deterministic_online_regeneration_with_cross_method_environment_identity"
            ),
        },
        {
            "id": "slow_bit_transition_semantics",
            "publication_reference": "nature_article_one_uniformly_selected_slow_bit_flips",
            "pinned_source_reference": (
                "default_cfg_prob_omits_flip_one_and_resamples_the_full_slow_bit_row"
            ),
            "development_execution": "one_uniformly_selected_slow_bit_flips_each_period",
        },
        {
            "id": "target_affine_bias_encoding",
            "publication_reference": "affine_target_layers_with_sampled_biases",
            "development_execution": "explicit_hidden_bias_bit_and_no_target_output_bias",
        },
        {
            "id": "publication_sweep_scope",
            "publication_reference": "ordinary_backprop_activation_and_step_size_sweep",
            "development_execution": "one_selected_relu_sgd_arm_plus_local_extensions",
        },
        {
            "id": "generated_horizon_boundary",
            "publication_reference": (
                "pinned_source_data_file_allocates_one_extra_period_then_learner_consumes_prefix"
            ),
            "development_execution": "online_exact_num_examples_with_ceiling_segment_schedule",
        },
        {
            "id": "extension_implementations",
            "publication_reference": "publication_source_tree_methods",
            "development_execution": "alberta_cbp_and_upgd_local_implementations",
        },
        {
            "id": "numeric_execution",
            "publication_reference": "pytorch_numeric_and_kernel_semantics",
            "development_execution": "jax_float32_numeric_and_kernel_semantics",
        },
    ]


def build_scr_v2_run_spec(
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    method_ids: Sequence[str],
    seed_ids: Sequence[int],
    bin_size: int,
) -> dict[str, Any]:
    """Build the closed, derived run specification used by every shard."""

    _validate_task_config(config)
    _validate_learner_params(params)
    methods = _validate_method_ids(method_ids)
    seeds = _validate_seed_ids(seed_ids)
    _require(
        _is_int(bin_size) and 1 <= bin_size <= _INT32_MAX,
        "bin_size must fit positive int32",
    )
    _require(
        config.num_examples % bin_size == 0,
        "num_examples must be an exact multiple of bin_size",
    )
    return {
        "reference_id": SCR_V2_REFERENCE_ID,
        "task": _task_dict(config),
        "data_contract": _data_contract_dict(),
        "measurement": {
            "metric": "pre_update_mean_squared_error",
            "bin_size": bin_size,
            "num_bins": config.num_examples // bin_size,
        },
        "learner_parameters": _learner_params_dict(params),
        "methods": [{"method_id": method, "role": _METHOD_ROLES[method]} for method in methods],
        "planned_seed_ids": list(seeds),
        "planned_seed_count": len(seeds),
        "planned_shard_count": len(seeds) * len(methods),
        "selected_configuration_match": _selected_configuration_match(
            config, params, methods, bin_size
        ),
        "deviations": _closed_deviations(),
    }


def build_scr_v2_run_plan(
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    method_ids: Sequence[str],
    seed_ids: Sequence[int],
    bin_size: int,
    *,
    created_unix: int | None = None,
    invocation_origin: str = "direct_api",
    process_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a self-issued, permanently nonpromoting development plan."""

    _prevalidate_invocation_provenance(
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    created = int(time.time()) if created_unix is None else created_unix
    _require_not_future_unix(created, "created_unix")
    run_spec = build_scr_v2_run_spec(config, params, method_ids, seed_ids, bin_size)
    source_manifest = _build_source_manifest()
    runtime_manifest = _build_runtime_manifest()
    run_spec_sha256 = _sha256_json(run_spec)
    source_manifest_sha256 = _sha256_json(source_manifest)
    runtime_manifest_sha256 = _sha256_json(runtime_manifest)
    return {
        "schema": SCR_V2_PLAN_SCHEMA,
        "benchmark": SCR_V2_BENCHMARK,
        "evidence_role": SCR_V2_EVIDENCE_ROLE,
        "scientific_promotion_allowed": False,
        "created_unix": created,
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_manifest": runtime_manifest,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "execution_envelope": {
            "schema": SCR_V2_EXECUTION_ENVELOPE_SCHEMA,
            "kind": "self_issued_development_manifest_without_external_chronology",
            "timestamp_semantics": "self_reported_diagnostic_only_not_external_chronology",
            "external_chronology_attestation_present": False,
            "attestation_identity": "none",
            "scientific_promotion_allowed": False,
        },
        "command_templates": dict(_PLAN_COMMANDS),
        "issuance_command": _build_command_provenance(
            _canonical_plan_semantic_argv(
                run_spec_sha256,
                source_manifest_sha256,
                runtime_manifest_sha256,
            ),
            invocation_origin=invocation_origin,
            process_argv=process_argv,
        ),
    }


def write_scr_v2_run_plan(
    path: Path,
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    method_ids: Sequence[str],
    seed_ids: Sequence[int],
    bin_size: int,
    *,
    created_unix: int | None = None,
    invocation_origin: str = "direct_api",
    process_argv: Sequence[str] | None = None,
) -> Path:
    """Create an immutable pre-run plan at a new path."""

    _prevalidate_invocation_provenance(
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    destination = _preflight_new_output(path)
    plan = build_scr_v2_run_plan(
        config,
        params,
        method_ids,
        seed_ids,
        bin_size,
        created_unix=created_unix,
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    encoded = _canonical_json_bytes(plan)
    _require(
        _json_exact_equal(plan["source_manifest"], _build_source_manifest()),
        "source bytes changed while the run plan was being issued",
    )
    _require(
        _json_exact_equal(plan["runtime_manifest"], _build_runtime_manifest()),
        "runtime changed while the run plan was being issued",
    )
    return _atomic_write_new(destination, encoded)


def _config_from_run_spec(run_spec: Mapping[str, Any]) -> SlowlyChangingRegressionConfig:
    task = _expect_dict(run_spec["task"], "run_spec.task")
    return SlowlyChangingRegressionConfig(  # type: ignore[call-arg]
        num_bits=task["num_bits"],
        num_flipping_bits=task["num_flipping_bits"],
        flip_period=task["flip_period"],
        target_hidden_units=task["target_hidden_units"],
        ltu_beta=task["ltu_beta"],
        num_examples=task["num_examples"],
    )


def _params_from_run_spec(run_spec: Mapping[str, Any]) -> SCRLearnerParams:
    values = _expect_dict(run_spec["learner_parameters"], "run_spec.learner_parameters")
    return SCRLearnerParams(  # type: ignore[call-arg]
        hidden_units=values["hidden_units"],
        step_size=values["step_size"],
        cbp_replacement_rate=values["cbp_replacement_rate"],
        cbp_maturity_threshold=values["cbp_maturity_threshold"],
        cbp_decay_rate=values["cbp_decay_rate"],
        upgd_sigma=values["upgd_sigma"],
        upgd_utility_decay=values["upgd_utility_decay"],
        upgd_beta=values["upgd_beta"],
    )


def _validate_source_manifest(value: object) -> dict[str, Any]:
    manifest = _expect_dict(value, "source_manifest")
    _expect_exact_keys(
        manifest,
        {"schema", "scope", "root_modules", "files"},
        "source_manifest",
    )
    _require(manifest["schema"] == SCR_V2_SOURCE_MANIFEST_SCHEMA, "wrong source manifest schema")
    _require(
        manifest["scope"]
        == "static_transitive_local_python_imports_plus_lockfiles",
        "source manifest scope differs",
    )
    _require(
        _json_exact_equal(manifest["root_modules"], list(_SOURCE_ROOT_MODULES)),
        "source manifest roots differ",
    )
    files = _expect_list(manifest["files"], "source_manifest.files")
    _require(bool(files), "source manifest file list must not be empty")
    paths: list[str] = []
    for index, raw_entry in enumerate(files):
        entry = _expect_dict(raw_entry, f"source_manifest.files[{index}]")
        _expect_exact_keys(
            entry, {"path", "byte_size", "sha256"}, f"source_manifest.files[{index}]"
        )
        _require(isinstance(entry["path"], str), "source path must be a string")
        pure = PurePosixPath(cast(str, entry["path"]))
        _require(
            bool(pure.parts)
            and not pure.is_absolute()
            and ".." not in pure.parts
            and "." not in pure.parts
            and pure.as_posix() == entry["path"],
            "source path must be a canonical repository-relative locator",
        )
        _require(
            _is_int(entry["byte_size"]) and entry["byte_size"] >= 0,
            "source byte_size must be nonnegative",
        )
        _require(_is_sha256(entry["sha256"]), "source sha256 must be lowercase hexadecimal")
        paths.append(entry["path"])
    _require(paths == sorted(set(paths)), "source manifest paths must be unique and sorted")
    _require(
        set(_SOURCE_AUXILIARY_PATHS) <= set(paths),
        "source manifest is missing required lock/configuration files",
    )
    required_modules = set(_SOURCE_ROOT_MODULES)
    for module in _SOURCE_ROOT_MODULES:
        required_modules.update(_parent_packages(module))
    required_paths = {
        cast(Path, _module_path(module)).relative_to(_repo_root()).as_posix()
        for module in required_modules
        if _module_path(module) is not None
    }
    _require(required_paths <= set(paths), "source manifest is missing a root or parent package")
    return manifest


def _validate_runtime_manifest(value: object, where: str) -> dict[str, Any]:
    runtime = _expect_dict(value, where)
    keys = {
        "schema",
        "python",
        "python_implementation",
        "python_executable",
        "jax",
        "jaxlib",
        "numpy",
        "chex",
        "jaxtyping",
        "distribution_content",
        "platform_system",
        "platform_machine",
        "platform_release",
        "jax_backend",
        "jax_devices",
        "jax_enable_x64",
        "jax_default_matmul_precision",
        "jax_default_prng_impl",
        "jax_numpy_dtype_promotion",
        "jax_config",
        "execution_environment",
    }
    _expect_exact_keys(runtime, keys, where)
    _require(runtime["schema"] == SCR_V2_RUNTIME_MANIFEST_SCHEMA, f"{where} has wrong schema")
    for key in keys - {
        "schema",
        "python_executable",
        "distribution_content",
        "jax_devices",
        "jax_enable_x64",
        "jax_config",
        "execution_environment",
    }:
        _require(
            isinstance(runtime[key], str) and bool(runtime[key]),
            f"{where}.{key} must be a nonempty string",
        )
    _require(type(runtime["jax_enable_x64"]) is bool, f"{where}.jax_enable_x64 must be boolean")
    executable = _expect_dict(runtime["python_executable"], f"{where}.python_executable")
    _expect_exact_keys(
        executable,
        {"locator", "byte_size", "sha256"},
        f"{where}.python_executable",
    )
    _canonical_absolute_locator(
        executable["locator"], f"{where}.python_executable.locator"
    )
    _require(
        _is_int(executable["byte_size"]) and executable["byte_size"] > 0,
        f"{where}.python_executable.byte_size invalid",
    )
    _require(
        _is_sha256(executable["sha256"]),
        f"{where}.python_executable.sha256 invalid",
    )
    distribution_content = _expect_dict(
        runtime["distribution_content"], f"{where}.distribution_content"
    )
    _expect_exact_keys(
        distribution_content,
        set(_RUNTIME_DISTRIBUTIONS),
        f"{where}.distribution_content",
    )
    for name in _RUNTIME_DISTRIBUTIONS:
        identity_where = f"{where}.distribution_content.{name}"
        identity = _expect_dict(distribution_content[name], identity_where)
        _expect_exact_keys(
            identity,
            {"status", "file_count", "total_bytes", "sha256"},
            identity_where,
        )
        _require(
            identity["status"] in {"content_hashed", "not_installed"},
            f"{identity_where}.status invalid",
        )
        _require(
            _is_int(identity["file_count"]) and identity["file_count"] >= 0,
            f"{identity_where}.file_count invalid",
        )
        _require(
            _is_int(identity["total_bytes"]) and identity["total_bytes"] >= 0,
            f"{identity_where}.total_bytes invalid",
        )
        _require(_is_sha256(identity["sha256"]), f"{identity_where}.sha256 invalid")
        if identity["status"] == "content_hashed":
            _require(
                identity["file_count"] > 0 and identity["total_bytes"] > 0,
                f"{identity_where} content hash must cover regular bytes",
            )
        else:
            _require(
                identity["file_count"] == 0
                and identity["total_bytes"] == 0
                and identity["sha256"] == _sha256_bytes(b""),
                f"{identity_where} absent-distribution identity differs",
            )
    devices = _expect_list(runtime["jax_devices"], f"{where}.jax_devices")
    _require(bool(devices), f"{where}.jax_devices must not be empty")
    for index, raw_device in enumerate(devices):
        device_where = f"{where}.jax_devices[{index}]"
        device = _expect_dict(raw_device, device_where)
        _expect_exact_keys(
            device,
            {
                "id",
                "platform",
                "platform_version",
                "device_kind",
                "process_index",
                "runtime_type",
            },
            device_where,
        )
        _require(_is_int(device["id"]) and device["id"] >= 0, f"{device_where}.id invalid")
        _require(
            _is_int(device["process_index"]) and device["process_index"] >= 0,
            f"{device_where}.process_index invalid",
        )
        for key in ("platform", "platform_version", "device_kind", "runtime_type"):
            _require(
                isinstance(device[key], str) and bool(device[key]),
                f"{device_where}.{key} invalid",
            )
    environment = _expect_dict(runtime["execution_environment"], f"{where}.execution_environment")
    _expect_exact_keys(
        environment,
        set(_RUNTIME_ENVIRONMENT_NAMES),
        f"{where}.execution_environment",
    )
    _require(
        all(value is None or isinstance(value, str) for value in environment.values()),
        f"{where}.execution_environment values must be strings or null",
    )
    jax_config = _expect_dict(runtime["jax_config"], f"{where}.jax_config")
    _require(bool(jax_config), f"{where}.jax_config must not be empty")
    _require(
        list(jax_config) == sorted(jax_config),
        f"{where}.jax_config keys must be sorted",
    )
    _require(
        all(
            value is None or type(value) in (str, int, float, bool)
            for value in jax_config.values()
        ),
        f"{where}.jax_config values must be JSON scalars",
    )
    _check_finite_tree(jax_config, f"{where}.jax_config")
    return runtime


def _validate_run_spec(
    value: object,
) -> tuple[
    dict[str, Any],
    SlowlyChangingRegressionConfig,
    SCRLearnerParams,
    tuple[str, ...],
    tuple[int, ...],
    int,
]:
    run_spec = _expect_dict(value, "run_spec")
    _expect_exact_keys(
        run_spec,
        {
            "reference_id",
            "task",
            "data_contract",
            "measurement",
            "learner_parameters",
            "methods",
            "planned_seed_ids",
            "planned_seed_count",
            "planned_shard_count",
            "selected_configuration_match",
            "deviations",
        },
        "run_spec",
    )
    task = _expect_dict(run_spec["task"], "run_spec.task")
    _expect_exact_keys(
        task,
        {
            "num_bits",
            "num_flipping_bits",
            "flip_period",
            "target_hidden_units",
            "ltu_beta",
            "num_examples",
        },
        "run_spec.task",
    )
    data_contract = _expect_dict(run_spec["data_contract"], "run_spec.data_contract")
    _require(
        _json_exact_equal(data_contract, _data_contract_dict()),
        "run_spec.data_contract differs from the closed synthetic-data contract",
    )
    measurement = _expect_dict(run_spec["measurement"], "run_spec.measurement")
    _expect_exact_keys(measurement, {"metric", "bin_size", "num_bins"}, "run_spec.measurement")
    params_raw = _expect_dict(run_spec["learner_parameters"], "run_spec.learner_parameters")
    _expect_exact_keys(
        params_raw,
        {
            "hidden_units",
            "step_size",
            "cbp_replacement_rate",
            "cbp_maturity_threshold",
            "cbp_decay_rate",
            "upgd_sigma",
            "upgd_utility_decay",
            "upgd_beta",
        },
        "run_spec.learner_parameters",
    )
    methods_raw = _expect_list(run_spec["methods"], "run_spec.methods")
    methods: list[str] = []
    for index, raw_method in enumerate(methods_raw):
        method = _expect_dict(raw_method, f"run_spec.methods[{index}]")
        _expect_exact_keys(method, {"method_id", "role"}, f"run_spec.methods[{index}]")
        _require(isinstance(method["method_id"], str), "method_id must be a string")
        _require(isinstance(method["role"], str), "method role must be a string")
        methods.append(method["method_id"])
    seeds_raw = _expect_list(run_spec["planned_seed_ids"], "run_spec.planned_seed_ids")
    seeds = _validate_seed_ids(cast(list[int], seeds_raw))
    config = _config_from_run_spec(run_spec)
    params = _params_from_run_spec(run_spec)
    bin_size = measurement["bin_size"]
    expected = build_scr_v2_run_spec(config, params, methods, seeds, bin_size)
    _require(
        _json_exact_equal(run_spec, expected),
        "run_spec differs from its derived closed specification",
    )
    return run_spec, config, params, tuple(methods), seeds, cast(int, bin_size)


def _validate_plan_payload(
    value: object,
    *,
    verify_current_bindings: bool,
) -> dict[str, Any]:
    plan = _expect_dict(value, "run_plan")
    _expect_exact_keys(
        plan,
        {
            "schema",
            "benchmark",
            "evidence_role",
            "scientific_promotion_allowed",
            "created_unix",
            "run_spec",
            "run_spec_sha256",
            "source_manifest",
            "source_manifest_sha256",
            "runtime_manifest",
            "runtime_manifest_sha256",
            "execution_envelope",
            "command_templates",
            "issuance_command",
        },
        "run_plan",
    )
    _require(plan["schema"] == SCR_V2_PLAN_SCHEMA, "wrong run-plan schema")
    _require(plan["benchmark"] == SCR_V2_BENCHMARK, "wrong run-plan benchmark")
    _require(plan["evidence_role"] == SCR_V2_EVIDENCE_ROLE, "wrong evidence role")
    _require(plan["scientific_promotion_allowed"] is False, "run plan must be nonpromoting")
    _require_not_future_unix(plan["created_unix"], "created_unix")
    run_spec, _, _, _, _, _ = _validate_run_spec(plan["run_spec"])
    _require(plan["run_spec_sha256"] == _sha256_json(run_spec), "run_spec_sha256 mismatch")
    source = _validate_source_manifest(plan["source_manifest"])
    _require(
        plan["source_manifest_sha256"] == _sha256_json(source), "source_manifest_sha256 mismatch"
    )
    runtime = _validate_runtime_manifest(plan["runtime_manifest"], "runtime_manifest")
    _require(
        plan["runtime_manifest_sha256"] == _sha256_json(runtime), "runtime_manifest_sha256 mismatch"
    )
    envelope = _expect_dict(plan["execution_envelope"], "execution_envelope")
    _expect_exact_keys(
        envelope,
        {
            "schema",
            "kind",
            "timestamp_semantics",
            "external_chronology_attestation_present",
            "attestation_identity",
            "scientific_promotion_allowed",
        },
        "execution_envelope",
    )
    _require(
        _json_exact_equal(
            envelope,
            {
                "schema": SCR_V2_EXECUTION_ENVELOPE_SCHEMA,
                "kind": "self_issued_development_manifest_without_external_chronology",
                "timestamp_semantics": (
                    "self_reported_diagnostic_only_not_external_chronology"
                ),
                "external_chronology_attestation_present": False,
                "attestation_identity": "none",
                "scientific_promotion_allowed": False,
            },
        ),
        "execution envelope must remain the closed self-issued development nonattestation",
    )
    commands = _expect_dict(plan["command_templates"], "command_templates")
    _require(_json_exact_equal(commands, _PLAN_COMMANDS), "command templates differ")
    _validate_command_provenance(
        plan["issuance_command"],
        _canonical_plan_semantic_argv(
            cast(str, plan["run_spec_sha256"]),
            cast(str, plan["source_manifest_sha256"]),
            cast(str, plan["runtime_manifest_sha256"]),
        ),
        "issuance_command",
    )
    if verify_current_bindings:
        _require_current_plan_bindings(plan, "run-plan validation")
    return plan


def _require_current_plan_bindings(plan: Mapping[str, Any], context: str) -> None:
    _require(
        _json_exact_equal(plan["source_manifest"], _build_source_manifest()),
        f"current source bytes differ from the pre-run plan during {context}",
    )
    _require(
        _json_exact_equal(plan["runtime_manifest"], _build_runtime_manifest()),
        f"current runtime differs from the pre-run plan during {context}",
    )


def _read_validated_plan(
    path: Path,
    *,
    verify_current_bindings: bool,
) -> tuple[bytes, dict[str, Any]]:
    raw, plan = _read_strict_json(path)
    _validate_plan_payload(plan, verify_current_bindings=verify_current_bindings)
    _require(
        raw == _canonical_json_bytes(plan),
        "run plan must use the canonical v2 JSON encoding",
    )
    _require_exact_reread(path, raw, "run plan")
    return raw, plan


def validate_scr_v2_run_plan(
    path: Path,
    *,
    verify_current_bindings: bool = True,
) -> SCRV2ValidationReport:
    """Strictly validate a v2 plan without granting promotion."""

    try:
        _read_validated_plan(path, verify_current_bindings=verify_current_bindings)
    except (OSError, ValueError, OverflowError) as exc:
        return SCRV2ValidationReport(False, False, (str(exc),))
    if not verify_current_bindings:
        return SCRV2ValidationReport(
            False,
            False,
            ("structural checks passed but current source/runtime bindings were not verified",),
            structurally_valid=True,
        )
    return SCRV2ValidationReport(True, False, (), structurally_valid=True)


def init_publication_bp(
    input_dim: int,
    hidden_units: int,
    key: Array,
) -> PublicationBPState:
    """Initialize the selected BP arm with ReLU/linear Kaiming uniform.

    This matches the distribution and zero-bias semantics of the publication's
    PyTorch ``FFNN``.  JAX and PyTorch do not share RNG streams, so equal seed
    integers do not imply byte-identical initial weights.
    """

    _require(_is_int(input_dim) and input_dim >= 1, "input_dim must be >= 1")
    _require(_is_int(hidden_units) and hidden_units >= 1, "hidden_units must be >= 1")
    hidden_key, output_key = jr.split(key)
    hidden_bound = math.sqrt(6.0 / input_dim)
    output_bound = math.sqrt(3.0 / hidden_units)
    hidden_weights = jr.uniform(
        hidden_key,
        (hidden_units, input_dim),
        minval=-hidden_bound,
        maxval=hidden_bound,
        dtype=jnp.float32,
    )
    output_weights = jr.uniform(
        output_key,
        (hidden_units,),
        minval=-output_bound,
        maxval=output_bound,
        dtype=jnp.float32,
    )
    return PublicationBPState(  # type: ignore[call-arg]
        hidden_weights=hidden_weights,
        hidden_bias=jnp.zeros((hidden_units,), dtype=jnp.float32),
        output_weights=output_weights,
        output_bias=jnp.asarray(0.0, dtype=jnp.float32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )


def publication_bp_predict(state: PublicationBPState, observation: Array) -> Array:
    """Return the scalar prediction of the selected ReLU BP arm."""

    hidden = jax.nn.relu(state.hidden_weights @ observation + state.hidden_bias)
    return jnp.dot(state.output_weights, hidden) + state.output_bias


def publication_bp_update(
    state: PublicationBPState,
    observation: Array,
    target: Array,
    step_size: float,
) -> tuple[PublicationBPState, Array]:
    """Apply one true-MSE SGD step and return pre-update squared error.

    For one scalar output, ``mean((prediction-target)**2)`` has derivative
    ``2 * (prediction-target)``.  This is intentionally twice the update made
    by Alberta's LMS convention at the same nominal step size.
    """

    preactivation = state.hidden_weights @ observation + state.hidden_bias
    hidden = jax.nn.relu(preactivation)
    prediction = jnp.dot(state.output_weights, hidden) + state.output_bias
    residual = prediction - jnp.squeeze(target)
    loss_gradient = jnp.float32(2.0) * residual
    active = (preactivation > 0.0).astype(jnp.float32)
    hidden_gradient = loss_gradient * state.output_weights * active
    alpha = jnp.asarray(step_size, dtype=jnp.float32)
    next_state = PublicationBPState(  # type: ignore[call-arg]
        hidden_weights=state.hidden_weights
        - alpha * hidden_gradient[:, None] * observation[None, :],
        hidden_bias=state.hidden_bias - alpha * hidden_gradient,
        output_weights=state.output_weights - alpha * loss_gradient * hidden,
        output_bias=state.output_bias - alpha * loss_gradient,
        step_count=state.step_count + jnp.asarray(1, dtype=jnp.int32),
    )
    return next_state, residual * residual


def _local_update(
    method_id: str,
    learner: Any,
    state: Any,
    observation: Array,
    target: Array,
) -> tuple[Any, Array]:
    if method_id == LOCAL_CBP_METHOD:
        result = learner.update(state, observation, target)
        error = jnp.squeeze(result.error)
        return result.state, error * error
    if method_id == LOCAL_UPGD_METHOD:
        result = learner.update(state, observation, jnp.reshape(target, (1,)))
        error = result.errors[0]
        return result.state, error * error
    _fail(f"method {method_id!r} is not a local extension")


def run_scr_v2_seed(
    method_id: str,
    config: SlowlyChangingRegressionConfig,
    params: SCRLearnerParams,
    seed_id: int,
    bin_size: int,
) -> Float[Array, " num_bins"]:
    """Run one exact planned method/seed shard.

    The explicit constant target-network bias bit is removed before the
    learner, whose dense layers carry their own biases.  This avoids giving
    the learning network two independent bias parameters.  The target-network
    representation difference remains a declared protocol deviation.
    """

    _validate_task_config(config)
    _validate_learner_params(params)
    _validate_method_ids((method_id,))
    _validate_seed_ids((seed_id,))
    _require(
        _is_int(bin_size) and 1 <= bin_size <= _INT32_MAX,
        "bin_size must fit positive int32",
    )
    _require(config.num_examples % bin_size == 0, "num_examples must be divisible by bin_size")
    num_bins = config.num_examples // bin_size

    if method_id == PUBLICATION_BP_METHOD:
        learner: Any = None
    elif method_id == LOCAL_CBP_METHOD:
        learner = build_scr_learner("cbp", params)
    else:
        learner = build_scr_learner("upgd", params)

    def run_one(key: Array) -> Array:
        env_key, init_key = jr.split(key)
        env = make_scr_env(config, env_key)
        if method_id == PUBLICATION_BP_METHOD:
            state0: Any = init_publication_bp(config.num_bits, params.hidden_units, init_key)
        else:
            state0 = learner.init(config.num_bits, init_key)

        def bin_step(state: Any, bin_index: Array) -> tuple[Any, Array]:
            start = bin_index * bin_size

            def example_step(index: Array, carry: tuple[Any, Array]) -> tuple[Any, Array]:
                current, total = carry
                full_observation, target = scr_example(env, config, start + index)
                observation = full_observation[:-1]
                if method_id == PUBLICATION_BP_METHOD:
                    current, squared_error = publication_bp_update(
                        current, observation, target, params.step_size
                    )
                else:
                    current, squared_error = _local_update(
                        method_id, learner, current, observation, target
                    )
                return current, total + squared_error

            next_state, total_error = lax.fori_loop(
                0,
                bin_size,
                example_step,
                (state, jnp.asarray(0.0, dtype=jnp.float32)),
            )
            return next_state, total_error / jnp.asarray(bin_size, dtype=jnp.float32)

        _, curve = lax.scan(
            bin_step,
            state0,
            jnp.arange(num_bins, dtype=jnp.int32),
        )
        return curve

    return cast(Array, jax.jit(run_one)(jr.key(seed_id)))


def _environment_identity(
    config: SlowlyChangingRegressionConfig,
    seed_id: int,
) -> dict[str, Any]:
    env_key, _ = jr.split(jr.key(seed_id))
    env = make_scr_env(config, env_key)
    digest = hashlib.sha256()
    digest.update(b"alberta.scr.environment.v1\0")
    digest.update(_compact_json_bytes(_task_dict(config)))
    arrays = (
        ("input_weights", env.input_weights),
        ("thresholds", env.thresholds),
        ("output_weights", env.output_weights),
        ("slow_bits", env.slow_bits),
        ("data_key", jr.key_data(env.data_key)),
    )
    for name, value in arrays:
        array = np.asarray(jax.device_get(value))
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(_compact_json_bytes(list(array.shape)))
        digest.update(array.tobytes(order="C"))
    return {
        "schema": SCR_V2_ENVIRONMENT_IDENTITY_SCHEMA,
        "seed_id": seed_id,
        "sha256": digest.hexdigest(),
    }


def _default_shard_path(plan_path: Path, method_id: str, seed_id: int) -> Path:
    return _lexical_absolute(plan_path).parent / "shards" / method_id / f"seed-{seed_id:04d}.json"


def _shard_reservation_path(output_path: Path) -> Path:
    destination = _lexical_absolute(output_path)
    return destination.with_name(f"{destination.name}.reservation")


def _build_shard_reservation(
    *,
    plan_path: Path,
    plan_raw: bytes,
    method_id: str,
    seed_id: int,
    output: Path,
    prescribed_command: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SCR_V2_SHARD_RESERVATION_SCHEMA,
        "benchmark": SCR_V2_BENCHMARK,
        "evidence_role": SCR_V2_EVIDENCE_ROLE,
        "scientific_promotion_allowed": False,
        "state": "execution_started_development_seed_irrevocably_consumed",
        "reserved_unix": int(time.time()),
        "timestamp_semantics": "self_reported_diagnostic_only_not_external_chronology",
        "external_chronology_attestation_present": False,
        "plan_locator": _canonical_path(plan_path),
        "plan_binding": {
            "byte_size": len(plan_raw),
            "sha256": _sha256_bytes(plan_raw),
        },
        "method_id": method_id,
        "seed_id": seed_id,
        "target_locator": _canonical_path(output),
        "prescribed_command": dict(prescribed_command),
    }


def run_scr_v2_shard(
    plan_path: Path,
    method_id: str,
    seed_id: int,
    output: Path | None = None,
    *,
    invocation_origin: str = "direct_api",
    process_argv: Sequence[str] | None = None,
) -> Path:
    """Validate a plan, execute one method/seed, and immutably write a shard."""

    _prevalidate_invocation_provenance(
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    _validate_method_ids((method_id,))
    _validate_seed_ids((seed_id,))
    requested_output = output or _default_shard_path(plan_path, method_id, seed_id)
    destination = _preflight_new_output(requested_output)
    reservation_path = _shard_reservation_path(destination)
    _require(
        reservation_path != _lexical_absolute(plan_path),
        "shard reservation and plan locators must be distinct",
    )
    _preflight_new_output(reservation_path)
    plan_raw, plan = _read_validated_plan(plan_path, verify_current_bindings=True)
    run_spec, config, params, methods, seeds, bin_size = _validate_run_spec(plan["run_spec"])
    _require(method_id in methods, f"method {method_id!r} is not planned")
    _require(_is_int(seed_id) and seed_id in seeds, f"seed {seed_id!r} is not planned")
    prescribed_command = _build_command_provenance(
        _canonical_shard_semantic_argv(
            _sha256_bytes(plan_raw), method_id, seed_id, destination
        ),
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    reservation = _build_shard_reservation(
        plan_path=plan_path,
        plan_raw=plan_raw,
        method_id=method_id,
        seed_id=seed_id,
        output=destination,
        prescribed_command=prescribed_command,
    )
    reservation_raw = _canonical_json_bytes(reservation)
    _atomic_write_new(reservation_path, reservation_raw)
    _preflight_new_output(destination)
    started = int(time.time())
    monotonic_start = time.monotonic()
    curve = run_scr_v2_seed(method_id, config, params, seed_id, bin_size)
    curve.block_until_ready()
    duration = time.monotonic() - monotonic_start
    finished = int(time.time())
    runtime = _build_runtime_manifest()
    source_manifest = _build_source_manifest()
    _require(
        _json_exact_equal(runtime, plan["runtime_manifest"]),
        "runtime changed while the shard was executing",
    )
    _require(
        _json_exact_equal(source_manifest, plan["source_manifest"]),
        "source bytes changed while the shard was executing",
    )
    curve_values = [float(value) for value in np.asarray(jax.device_get(curve))]
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in curve_values),
        "run produced invalid errors",
    )
    shard = {
        "schema": SCR_V2_SHARD_SCHEMA,
        "benchmark": SCR_V2_BENCHMARK,
        "evidence_role": SCR_V2_EVIDENCE_ROLE,
        "scientific_promotion_allowed": False,
        "plan_binding": {
            "byte_size": len(plan_raw),
            "sha256": _sha256_bytes(plan_raw),
        },
        "run_spec_sha256": _sha256_json(run_spec),
        "source_manifest_sha256": plan["source_manifest_sha256"],
        "runtime_manifest_sha256": plan["runtime_manifest_sha256"],
        "method_id": method_id,
        "seed_id": seed_id,
        "environment_identity": _environment_identity(config, seed_id),
        "execution": {
            "started_unix": started,
            "finished_unix": finished,
            "duration_seconds": duration,
            "runtime_manifest": runtime,
            "runtime_manifest_sha256": _sha256_json(runtime),
            "source_manifest_sha256": _sha256_json(source_manifest),
            "command": prescribed_command,
        },
        "measurements": {
            "num_bins": len(curve_values),
            "bin_mean_squared_error": curve_values,
        },
    }
    encoded = _canonical_json_bytes(shard)
    _require_current_plan_bindings(plan, "final shard publication")
    _require_exact_reread(plan_path, plan_raw, "external run plan")
    _require_exact_reread(
        reservation_path,
        reservation_raw,
        "persistent shard reservation",
    )
    return _atomic_write_new(destination, encoded)


def _validate_plan_binding(value: object, plan_raw: bytes, where: str) -> None:
    binding = _expect_dict(value, where)
    _expect_exact_keys(binding, {"byte_size", "sha256"}, where)
    _require(
        _is_int(binding["byte_size"]) and binding["byte_size"] >= 0,
        f"{where}.byte_size invalid",
    )
    _require(_is_sha256(binding["sha256"]), f"{where}.sha256 invalid")
    _require(binding["byte_size"] == len(plan_raw), f"{where}.byte_size mismatch")
    _require(binding["sha256"] == _sha256_bytes(plan_raw), f"{where}.sha256 mismatch")


def _validate_environment_identity(
    value: object,
    config: SlowlyChangingRegressionConfig,
    seed_id: int,
) -> dict[str, Any]:
    identity = _expect_dict(value, "environment_identity")
    _expect_exact_keys(identity, {"schema", "seed_id", "sha256"}, "environment_identity")
    _require(
        _json_exact_equal(identity, _environment_identity(config, seed_id)),
        "environment identity mismatch",
    )
    return identity


def _validate_shard_payload(
    value: object,
    plan: dict[str, Any],
    plan_raw: bytes,
    shard_path: Path,
) -> dict[str, Any]:
    shard = _expect_dict(value, "shard")
    _expect_exact_keys(
        shard,
        {
            "schema",
            "benchmark",
            "evidence_role",
            "scientific_promotion_allowed",
            "plan_binding",
            "run_spec_sha256",
            "source_manifest_sha256",
            "runtime_manifest_sha256",
            "method_id",
            "seed_id",
            "environment_identity",
            "execution",
            "measurements",
        },
        "shard",
    )
    _require(shard["schema"] == SCR_V2_SHARD_SCHEMA, "wrong shard schema")
    _require(shard["benchmark"] == SCR_V2_BENCHMARK, "wrong shard benchmark")
    _require(shard["evidence_role"] == SCR_V2_EVIDENCE_ROLE, "wrong shard evidence role")
    _require(shard["scientific_promotion_allowed"] is False, "shard must be nonpromoting")
    _validate_plan_binding(shard["plan_binding"], plan_raw, "plan_binding")
    run_spec, config, _, methods, seeds, _ = _validate_run_spec(plan["run_spec"])
    _require(shard["run_spec_sha256"] == _sha256_json(run_spec), "shard run-spec digest mismatch")
    _require(
        shard["source_manifest_sha256"] == plan["source_manifest_sha256"],
        "shard source digest mismatch",
    )
    _require(
        shard["runtime_manifest_sha256"] == plan["runtime_manifest_sha256"],
        "shard planned runtime digest mismatch",
    )
    _require(
        isinstance(shard["method_id"], str) and shard["method_id"] in methods,
        "shard method is not planned",
    )
    _require(_is_int(shard["seed_id"]) and shard["seed_id"] in seeds, "shard seed is not planned")
    seed_id = cast(int, shard["seed_id"])
    _validate_environment_identity(shard["environment_identity"], config, seed_id)
    execution = _expect_dict(shard["execution"], "execution")
    _expect_exact_keys(
        execution,
        {
            "started_unix",
            "finished_unix",
            "duration_seconds",
            "runtime_manifest",
            "runtime_manifest_sha256",
            "source_manifest_sha256",
            "command",
        },
        "execution",
    )
    started_unix = _require_not_future_unix(execution["started_unix"], "started_unix")
    _require(
        _is_int(execution["finished_unix"])
        and execution["finished_unix"] >= started_unix,
        "finished_unix invalid",
    )
    _require_not_future_unix(execution["finished_unix"], "finished_unix")
    _require(
        isinstance(execution["duration_seconds"], float)
        and math.isfinite(execution["duration_seconds"])
        and execution["duration_seconds"] >= 0.0,
        "duration_seconds invalid",
    )
    runtime = _validate_runtime_manifest(
        execution["runtime_manifest"], "execution.runtime_manifest"
    )
    _require(
        execution["runtime_manifest_sha256"] == _sha256_json(runtime),
        "worker runtime digest mismatch",
    )
    _require(
        execution["runtime_manifest_sha256"] == plan["runtime_manifest_sha256"],
        "worker runtime differs from plan",
    )
    _require(
        execution["source_manifest_sha256"] == plan["source_manifest_sha256"],
        "worker source differs from plan",
    )
    _validate_command_provenance(
        execution["command"],
        _canonical_shard_semantic_argv(
            _sha256_bytes(plan_raw),
            cast(str, shard["method_id"]),
            seed_id,
            shard_path,
        ),
        "execution.command",
    )
    measurements = _expect_dict(shard["measurements"], "measurements")
    _expect_exact_keys(measurements, {"num_bins", "bin_mean_squared_error"}, "measurements")
    expected_bins = run_spec["measurement"]["num_bins"]
    _require(
        _is_int(measurements["num_bins"]) and measurements["num_bins"] == expected_bins,
        "shard num_bins mismatch",
    )
    curve = _expect_list(
        measurements["bin_mean_squared_error"], "measurements.bin_mean_squared_error"
    )
    _require(len(curve) == expected_bins, "shard curve length mismatch")
    _require(
        all(isinstance(value, float) and math.isfinite(value) and value >= 0.0 for value in curve),
        "shard curve must be finite and nonnegative",
    )
    return shard


def _validate_replayed_measurements(shard: dict[str, Any], plan: dict[str, Any]) -> None:
    run_spec, config, params, _, _, bin_size = _validate_run_spec(plan["run_spec"])
    method_id = cast(str, shard["method_id"])
    seed_id = cast(int, shard["seed_id"])
    try:
        replay = run_scr_v2_seed(method_id, config, params, seed_id, bin_size)
        replay.block_until_ready()
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise SCRV2ValidationError(f"deterministic shard replay failed: {exc}") from exc
    replay_values = [float(value) for value in np.asarray(jax.device_get(replay))]
    recorded = _expect_dict(shard["measurements"], "measurements")[
        "bin_mean_squared_error"
    ]
    _require(
        _json_exact_equal(recorded, replay_values),
        "recorded measurements differ from deterministic replay",
    )
    _require(
        len(replay_values) == run_spec["measurement"]["num_bins"],
        "deterministic replay bin count mismatch",
    )


def validate_scr_v2_shard(
    shard_path: Path,
    plan_path: Path,
    *,
    verify_current_bindings: bool = True,
    replay_measurements: bool = True,
) -> SCRV2ValidationReport:
    """Validate one shard; ``valid`` requires an exact computational replay."""

    structurally_valid = False
    replay_performed = False
    try:
        plan_raw, plan = _read_validated_plan(
            plan_path, verify_current_bindings=verify_current_bindings
        )
        raw, shard = _read_strict_json(shard_path)
        _require(raw == _canonical_json_bytes(shard), "shard must use canonical v2 JSON encoding")
        validated_shard = _validate_shard_payload(shard, plan, plan_raw, shard_path)
        structurally_valid = True
        if replay_measurements:
            _validate_replayed_measurements(validated_shard, plan)
            replay_performed = True
        if verify_current_bindings:
            _require_current_plan_bindings(plan, "final shard validation")
        _require_exact_reread(plan_path, plan_raw, "external run plan")
        _require_exact_reread(shard_path, raw, "validated shard")
    except (OSError, ValueError, OverflowError) as exc:
        return SCRV2ValidationReport(
            False,
            False,
            (str(exc),),
            structurally_valid=structurally_valid,
            computational_replay_performed=replay_performed,
        )
    if not replay_measurements:
        return SCRV2ValidationReport(
            False,
            False,
            ("structural checks passed but exact computational replay was not performed",),
            structurally_valid=True,
            computational_replay_performed=False,
        )
    if not verify_current_bindings:
        return SCRV2ValidationReport(
            False,
            False,
            ("exact replay passed but current source/runtime bindings were not verified",),
            structurally_valid=True,
            computational_replay_performed=True,
        )
    return SCRV2ValidationReport(
        True,
        False,
        (),
        structurally_valid=True,
        computational_replay_performed=True,
    )


def _descriptive_results(
    plan: dict[str, Any],
    shards: Mapping[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    run_spec = cast(dict[str, Any], plan["run_spec"])
    seed_ids = cast(list[int], run_spec["planned_seed_ids"])
    output: list[dict[str, Any]] = []
    for method in cast(list[dict[str, str]], run_spec["methods"]):
        method_id = method["method_id"]
        curves = [
            cast(list[float], shards[(method_id, seed)]["measurements"]["bin_mean_squared_error"])
            for seed in seed_ids
        ]
        matrix = np.asarray(curves, dtype=np.float64)
        bin_mean = np.mean(matrix, axis=0)
        bin_population_std = np.std(matrix, axis=0, ddof=0)
        bin_population_std_over_sqrt_n = bin_population_std / math.sqrt(len(seed_ids))
        first_bin_mean = float(bin_mean[0])
        last_bin_mean = float(bin_mean[-1])
        ratio_defined = first_bin_mean != 0.0
        output.append(
            {
                "method_id": method_id,
                "method_role": method["role"],
                "seed_ids": list(seed_ids),
                "per_seed_bin_mean_squared_error": curves,
                "descriptive_summary": {
                    "bin_mean": [float(value) for value in bin_mean],
                    "bin_population_std": [float(value) for value in bin_population_std],
                    "bin_population_std_over_sqrt_seed_count": [
                        float(value) for value in bin_population_std_over_sqrt_n
                    ],
                    "first_bin_mean": first_bin_mean,
                    "last_bin_mean": last_bin_mean,
                    "last_over_first": (last_bin_mean / first_bin_mean if ratio_defined else None),
                    "last_over_first_defined": ratio_defined,
                    "whole_run_mean": float(np.mean(matrix)),
                },
            }
        )
    return output


def _safe_relative_locator(path: Path, parent: Path) -> str:
    resolved_parent = _lexical_absolute(parent)
    resolved = _lexical_absolute(path)
    try:
        relative = resolved.relative_to(resolved_parent)
    except ValueError as exc:
        raise SCRV2ValidationError(
            f"shard {resolved} must be inside artifact directory {resolved_parent}"
        ) from exc
    locator = relative.as_posix()
    pure = PurePosixPath(locator)
    _require(not pure.is_absolute() and ".." not in pure.parts, "unsafe shard locator")
    return locator


def merge_scr_v2_shards(
    plan_path: Path,
    shard_paths: Sequence[Path],
    output: Path,
    *,
    created_unix: int | None = None,
    invocation_origin: str = "direct_api",
    process_argv: Sequence[str] | None = None,
) -> Path:
    """Replay and merge the exact planned Cartesian product into an artifact."""

    _prevalidate_invocation_provenance(
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    created = int(time.time()) if created_unix is None else created_unix
    _require_not_future_unix(created, "created_unix")
    destination = _preflight_new_output(output)
    canonical_plan_path = _lexical_absolute(plan_path)
    canonical_shard_paths = tuple(_lexical_absolute(path) for path in shard_paths)
    _require(
        len(set(canonical_shard_paths)) == len(canonical_shard_paths),
        "duplicate shard locators are forbidden before merge",
    )
    _require(
        destination != canonical_plan_path
        and destination not in canonical_shard_paths
        and canonical_plan_path not in canonical_shard_paths,
        "plan, shard, and artifact locators must be distinct",
    )
    for path in canonical_shard_paths:
        _safe_relative_locator(path, destination.parent)
    plan_raw, plan = _read_validated_plan(plan_path, verify_current_bindings=True)
    run_spec, _, _, methods, seeds, _ = _validate_run_spec(plan["run_spec"])
    expected_pairs = {(method, seed) for method in methods for seed in seeds}
    _require(
        len(canonical_shard_paths) == len(expected_pairs),
        "shard path count differs from planned coverage",
    )
    shards: dict[tuple[str, int], dict[str, Any]] = {}
    raw_by_pair: dict[tuple[str, int], tuple[Path, bytes]] = {}
    environment_by_seed: dict[int, str] = {}
    for path in canonical_shard_paths:
        raw, shard = _read_strict_json(path)
        _validate_shard_payload(shard, plan, plan_raw, path)
        _require(raw == _canonical_json_bytes(shard), "shard must use canonical v2 JSON encoding")
        pair = (cast(str, shard["method_id"]), cast(int, shard["seed_id"]))
        _require(pair not in shards, f"duplicate method/seed shard: {pair}")
        identity = cast(dict[str, Any], shard["environment_identity"])["sha256"]
        previous = environment_by_seed.setdefault(pair[1], identity)
        _require(
            previous == identity, f"methods do not share environment identity for seed {pair[1]}"
        )
        shards[pair] = shard
        raw_by_pair[pair] = (path, raw)
    _require(set(shards) == expected_pairs, "observed method/seed coverage differs from plan")
    for method in methods:
        for seed in seeds:
            _validate_replayed_measurements(shards[(method, seed)], plan)

    manifest: list[dict[str, Any]] = []
    for method in methods:
        for seed in seeds:
            pair = (method, seed)
            path, raw = raw_by_pair[pair]
            identity = cast(dict[str, Any], shards[pair]["environment_identity"])
            manifest.append(
                {
                    "method_id": method,
                    "seed_id": seed,
                    "path": _safe_relative_locator(path, destination.parent),
                    "byte_size": len(raw),
                    "sha256": _sha256_bytes(raw),
                    "environment_sha256": identity["sha256"],
                }
            )

    merge_runtime = _build_runtime_manifest()
    _require(
        _json_exact_equal(merge_runtime, plan["runtime_manifest"]),
        "merge runtime differs from the planned runtime",
    )
    _require(
        _json_exact_equal(_build_source_manifest(), plan["source_manifest"]),
        "source bytes changed while shards were being merged",
    )
    results = _descriptive_results(plan, shards)
    merge_command = _build_command_provenance(
        _canonical_merge_semantic_argv(
            _sha256_bytes(plan_raw),
            [cast(str, entry["sha256"]) for entry in manifest],
            destination,
        ),
        invocation_origin=invocation_origin,
        process_argv=process_argv,
    )
    artifact = {
        "schema": SCR_V2_ARTIFACT_SCHEMA,
        "benchmark": SCR_V2_BENCHMARK,
        "evidence_role": SCR_V2_EVIDENCE_ROLE,
        "scientific_promotion_allowed": False,
        "created_unix": created,
        "run_plan": plan,
        "plan_binding": {
            "byte_size": len(plan_raw),
            "sha256": _sha256_bytes(plan_raw),
        },
        "external_plan": {
            "path": canonical_plan_path.as_posix(),
            "byte_size": len(plan_raw),
            "sha256": _sha256_bytes(plan_raw),
        },
        "shard_manifest": manifest,
        "observed_coverage": {
            "method_ids": list(methods),
            "seed_ids_by_method": {method: list(seeds) for method in methods},
            "paired_seed_ids": list(seeds),
            "shard_count": len(shards),
        },
        "computational_integrity": {
            "kind": "exact_deterministic_replay",
            "merge_exact_replay_performed": True,
            "replay_scope": "all_planned_method_seed_shards",
            "replayed_shard_count": len(shards),
            "trusted_external_receipt": False,
        },
        "results": results,
        "interpretation": _interpretation_dict(),
        "merge_runtime": merge_runtime,
        "merge_runtime_sha256": _sha256_json(merge_runtime),
        "merge_command": merge_command,
    }
    _require(_json_exact_equal(run_spec, plan["run_spec"]), "internal run-spec mismatch")
    encoded = _canonical_json_bytes(artifact)
    _require_current_plan_bindings(plan, "final artifact publication")
    _require_exact_reread(canonical_plan_path, plan_raw, "external run plan")
    for path, raw in raw_by_pair.values():
        _require_exact_reread(path, raw, f"merge input shard {path}")
    return _atomic_write_new(destination, encoded)


def _resolve_artifact_shard(artifact_path: Path, locator: object) -> Path:
    _require(isinstance(locator, str) and bool(locator), "shard path must be a nonempty string")
    pure = PurePosixPath(cast(str, locator))
    _require(
        not pure.is_absolute()
        and ".." not in pure.parts
        and "." not in pure.parts
        and pure.as_posix() == locator,
        "unsafe or noncanonical shard path",
    )
    parent = _lexical_absolute(artifact_path).parent
    candidate = _lexical_absolute(parent / Path(*pure.parts))
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise SCRV2ValidationError("shard path escapes artifact directory") from exc
    return candidate


def _validate_artifact_payload(
    artifact: dict[str, Any],
    artifact_path: Path,
    *,
    verify_current_bindings: bool,
    replay_measurements: bool,
) -> None:
    _expect_exact_keys(
        artifact,
        {
            "schema",
            "benchmark",
            "evidence_role",
            "scientific_promotion_allowed",
            "created_unix",
            "run_plan",
            "plan_binding",
            "external_plan",
            "shard_manifest",
            "observed_coverage",
            "computational_integrity",
            "results",
            "interpretation",
            "merge_runtime",
            "merge_runtime_sha256",
            "merge_command",
        },
        "artifact",
    )
    _require(artifact["schema"] == SCR_V2_ARTIFACT_SCHEMA, "wrong artifact schema")
    _require(artifact["benchmark"] == SCR_V2_BENCHMARK, "wrong artifact benchmark")
    _require(artifact["evidence_role"] == SCR_V2_EVIDENCE_ROLE, "wrong artifact evidence role")
    _require(artifact["scientific_promotion_allowed"] is False, "artifact must be nonpromoting")
    _require_not_future_unix(artifact["created_unix"], "artifact created_unix")
    plan = _validate_plan_payload(
        artifact["run_plan"], verify_current_bindings=verify_current_bindings
    )
    plan_raw = _canonical_json_bytes(plan)
    _validate_plan_binding(artifact["plan_binding"], plan_raw, "artifact.plan_binding")
    external_plan = _expect_dict(artifact["external_plan"], "external_plan")
    _expect_exact_keys(
        external_plan,
        {"path", "byte_size", "sha256"},
        "external_plan",
    )
    external_plan_locator = _canonical_absolute_locator(
        external_plan["path"], "external_plan.path"
    )
    _require(
        _is_int(external_plan["byte_size"]) and external_plan["byte_size"] >= 0,
        "external_plan.byte_size invalid",
    )
    _require(_is_sha256(external_plan["sha256"]), "external_plan.sha256 invalid")
    external_plan_path = Path(external_plan_locator)
    external_plan_raw, external_plan_payload = _read_validated_plan(
        external_plan_path,
        verify_current_bindings=verify_current_bindings,
    )
    _require(
        external_plan["byte_size"] == len(external_plan_raw)
        and external_plan["sha256"] == _sha256_bytes(external_plan_raw),
        "external plan size/hash binding mismatch",
    )
    _require(
        external_plan_raw == plan_raw
        and _json_exact_equal(external_plan_payload, plan),
        "embedded run plan differs from the exact external plan",
    )
    _, _, _, methods, seeds, _ = _validate_run_spec(plan["run_spec"])
    expected_pairs = [(method, seed) for method in methods for seed in seeds]
    manifest = _expect_list(artifact["shard_manifest"], "shard_manifest")
    _require(len(manifest) == len(expected_pairs), "manifest count differs from plan")
    shards: dict[tuple[str, int], dict[str, Any]] = {}
    raw_by_pair: dict[tuple[str, int], tuple[Path, bytes]] = {}
    computed_manifest: list[dict[str, Any]] = []
    environment_by_seed: dict[int, str] = {}
    for index, (raw_entry, expected_pair) in enumerate(zip(manifest, expected_pairs, strict=True)):
        entry = _expect_dict(raw_entry, f"shard_manifest[{index}]")
        _expect_exact_keys(
            entry,
            {"method_id", "seed_id", "path", "byte_size", "sha256", "environment_sha256"},
            f"shard_manifest[{index}]",
        )
        _require(
            (entry["method_id"], entry["seed_id"]) == expected_pair,
            "manifest ordering or coverage differs",
        )
        _require(isinstance(entry["method_id"], str), "manifest method_id must be a string")
        _require(_is_int(entry["seed_id"]), "manifest seed_id must be an integer")
        _require(
            _is_int(entry["byte_size"]) and entry["byte_size"] >= 0,
            "manifest byte_size invalid",
        )
        _require(_is_sha256(entry["sha256"]), "manifest sha256 invalid")
        _require(_is_sha256(entry["environment_sha256"]), "manifest environment hash invalid")
        shard_path = _resolve_artifact_shard(artifact_path, entry["path"])
        raw, shard = _read_strict_json(shard_path)
        _validate_shard_payload(shard, plan, plan_raw, shard_path)
        _require(raw == _canonical_json_bytes(shard), "shard must use canonical v2 JSON encoding")
        pair = (cast(str, shard["method_id"]), cast(int, shard["seed_id"]))
        _require(pair == expected_pair, "manifest identity differs from shard identity")
        identity = cast(dict[str, Any], shard["environment_identity"])["sha256"]
        previous = environment_by_seed.setdefault(pair[1], identity)
        _require(previous == identity, f"shared environment mismatch for seed {pair[1]}")
        computed_entry = {
            "method_id": pair[0],
            "seed_id": pair[1],
            "path": cast(str, entry["path"]),
            "byte_size": len(raw),
            "sha256": _sha256_bytes(raw),
            "environment_sha256": identity,
        }
        computed_manifest.append(computed_entry)
        shards[pair] = shard
        raw_by_pair[pair] = (shard_path, raw)
    _require(
        _json_exact_equal(manifest, computed_manifest),
        "shard manifest size/hash/identity mismatch",
    )
    expected_coverage = {
        "method_ids": list(methods),
        "seed_ids_by_method": {method: list(seeds) for method in methods},
        "paired_seed_ids": list(seeds),
        "shard_count": len(expected_pairs),
    }
    coverage = _expect_dict(artifact["observed_coverage"], "observed_coverage")
    _require(
        _json_exact_equal(coverage, expected_coverage),
        "observed coverage differs from exact plan",
    )
    computational_integrity = _expect_dict(
        artifact["computational_integrity"], "computational_integrity"
    )
    _require(
        _json_exact_equal(
            computational_integrity,
            {
                "kind": "exact_deterministic_replay",
                "merge_exact_replay_performed": True,
                "replay_scope": "all_planned_method_seed_shards",
                "replayed_shard_count": len(expected_pairs),
                "trusted_external_receipt": False,
            },
        ),
        "computational integrity declaration differs",
    )
    results = _expect_list(artifact["results"], "results")
    _require(
        _json_exact_equal(results, _descriptive_results(plan, shards)),
        "descriptive results do not reconstruct from shards",
    )
    interpretation = _expect_dict(artifact["interpretation"], "interpretation")
    _require(
        _json_exact_equal(interpretation, _interpretation_dict()),
        "interpretation policy differs",
    )
    merge_runtime = _validate_runtime_manifest(artifact["merge_runtime"], "merge_runtime")
    _require(
        artifact["merge_runtime_sha256"] == _sha256_json(merge_runtime),
        "merge runtime digest mismatch",
    )
    _require(
        _json_exact_equal(merge_runtime, plan["runtime_manifest"]),
        "merge runtime differs from planned runtime",
    )
    _validate_command_provenance(
        artifact["merge_command"],
        _canonical_merge_semantic_argv(
            _sha256_bytes(plan_raw),
            [cast(str, entry["sha256"]) for entry in computed_manifest],
            artifact_path,
        ),
        "merge_command",
    )
    if replay_measurements:
        for method in methods:
            for seed in seeds:
                _validate_replayed_measurements(shards[(method, seed)], plan)
    if verify_current_bindings:
        _require_current_plan_bindings(plan, "final artifact validation")
    _require_exact_reread(artifact_path, _canonical_json_bytes(artifact), "artifact")
    _require_exact_reread(external_plan_path, external_plan_raw, "external run plan")
    for shard_path, shard_raw in raw_by_pair.values():
        _require_exact_reread(
            shard_path,
            shard_raw,
            f"artifact input shard {shard_path}",
        )


def validate_scr_v2_artifact(
    path: Path,
    *,
    verify_current_bindings: bool = True,
    replay_measurements: bool = True,
) -> SCRV2ValidationReport:
    """Reconstruct v2; ``valid`` requires replay and promotion is always denied."""

    structurally_valid = False
    replay_performed = False
    try:
        raw, artifact = _read_strict_json(path)
        _require(
            raw == _canonical_json_bytes(artifact),
            "artifact must use canonical v2 JSON encoding",
        )
        _validate_artifact_payload(
            artifact,
            path,
            verify_current_bindings=verify_current_bindings,
            replay_measurements=replay_measurements,
        )
        structurally_valid = True
        replay_performed = replay_measurements
    except (OSError, ValueError, OverflowError) as exc:
        return SCRV2ValidationReport(
            False,
            False,
            (str(exc),),
            structurally_valid=structurally_valid,
            computational_replay_performed=replay_performed,
        )
    if not replay_measurements:
        return SCRV2ValidationReport(
            False,
            False,
            ("structural checks passed but exact computational replay was not performed",),
            structurally_valid=True,
            computational_replay_performed=False,
        )
    if not verify_current_bindings:
        return SCRV2ValidationReport(
            False,
            False,
            ("exact replay passed but current source/runtime bindings were not verified",),
            structurally_valid=True,
            computational_replay_performed=True,
        )
    return SCRV2ValidationReport(
        True,
        False,
        (),
        structurally_valid=True,
        computational_replay_performed=True,
    )


def _parse_methods(raw: str) -> tuple[str, ...]:
    aliases = {
        "bp": PUBLICATION_BP_METHOD,
        "cbp": LOCAL_CBP_METHOD,
        "upgd": LOCAL_UPGD_METHOD,
    }
    values = tuple(
        aliases.get(item.strip(), item.strip()) for item in raw.split(",") if item.strip()
    )
    return _validate_method_ids(values)


def _add_protocol_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--examples", type=int, default=3_000_000)
    parser.add_argument("--bin-size", type=int, default=40_000)
    parser.add_argument("--flip-period", type=int, default=10_000)
    parser.add_argument("--num-bits", type=int, default=20)
    parser.add_argument("--num-flipping-bits", type=int, default=15)
    parser.add_argument("--target-hidden-units", type=int, default=100)
    parser.add_argument("--hidden-units", type=int, default=5)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument(
        "--methods",
        default="bp,cbp,upgd",
        help="Comma-separated subset in canonical order: bp,cbp,upgd",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict nonpromoting slowly-changing-regression development runner"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="write an immutable pre-run v2 plan")
    plan.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/slowly_changing_regression/development_v2/run_plan.v2.json"),
    )
    _add_protocol_args(plan)
    shard = commands.add_parser("run-shard", help="run exactly one planned method/seed")
    shard.add_argument("--plan", type=Path, required=True)
    shard.add_argument("--method", choices=SCR_V2_METHOD_IDS, required=True)
    shard.add_argument("--seed-id", type=int, required=True)
    shard.add_argument("--output", type=Path)
    merge = commands.add_parser("merge", help="merge the exact planned shard product")
    merge.add_argument("--plan", type=Path, required=True)
    shard_sources = merge.add_mutually_exclusive_group(required=True)
    shard_sources.add_argument("--shards-dir", type=Path)
    shard_sources.add_argument(
        "--shard",
        dest="shards",
        action="append",
        type=Path,
        help="Explicit shard path; repeat once per planned shard",
    )
    merge.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate", help="strictly validate a merged v2 artifact")
    validate.add_argument("--artifact", type=Path, required=True)
    validate.add_argument(
        "--structural-only",
        action="store_true",
        help="check structure without replay; deliberately returns a nonvalid result",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    """Run one v2 lifecycle command; no command can promote an artifact."""

    args = _parse_args(argv)
    command_argv = tuple(argv) if argv is not None else tuple(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        force=True,
    )
    if args.command == "plan":
        _require(_is_int(args.runs) and args.runs >= 1, "runs must be >= 1")
        _require(args.runs <= 100_000, "runs must be <= 100000")
        _require(_is_int(args.seed_start) and args.seed_start >= 0, "seed-start must be >= 0")
        _require(
            args.seed_start + args.runs - 1 <= 0xFFFF_FFFF,
            "planned seed IDs must fit uint32",
        )
        config = SlowlyChangingRegressionConfig(  # type: ignore[call-arg]
            num_bits=args.num_bits,
            num_flipping_bits=args.num_flipping_bits,
            flip_period=args.flip_period,
            target_hidden_units=args.target_hidden_units,
            num_examples=args.examples,
        )
        params = SCRLearnerParams(  # type: ignore[call-arg]
            hidden_units=args.hidden_units,
            step_size=args.step_size,
        )
        result = write_scr_v2_run_plan(
            args.output,
            config,
            params,
            _parse_methods(args.methods),
            tuple(range(args.seed_start, args.seed_start + args.runs)),
            args.bin_size,
            invocation_origin="cli",
            process_argv=command_argv,
        )
        logger.info("wrote immutable nonpromoting run plan %s", result)
        return result
    if args.command == "run-shard":
        result = run_scr_v2_shard(
            args.plan,
            args.method,
            args.seed_id,
            args.output,
            invocation_origin="cli",
            process_argv=command_argv,
        )
        logger.info("wrote immutable nonpromoting shard %s", result)
        return result
    if args.command == "merge":
        if args.shards_dir is not None:
            shard_paths = tuple(sorted(args.shards_dir.rglob("*.json")))
        else:
            shard_paths = tuple(cast(list[Path], args.shards))
        result = merge_scr_v2_shards(
            args.plan,
            shard_paths,
            args.output,
            invocation_origin="cli",
            process_argv=command_argv,
        )
        logger.info("wrote immutable nonpromoting artifact %s", result)
        return result
    if args.command == "validate":
        report = validate_scr_v2_artifact(
            args.artifact,
            replay_measurements=not bool(args.structural_only),
        )
        if not report.valid:
            raise SCRV2ValidationError("; ".join(report.errors))
        logger.info(
            "artifact passed exact replay; scientific_promotion_allowed=%s",
            report.scientific_promotion_allowed,
        )
        return cast(Path, args.artifact)
    _fail(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
