"""Managed learner-execution boundary for hidden-regime calibration.

The boundary recognizes the exact calibration manifests and the protected
structural manifests by their world content, not by caller-supplied labels.
Ordinary development worlds remain unrestricted.  Exact calibration worlds
require a process-sealed, receipt/genesis/case-bound authorization and consume
their case before the first learner transition.  Protected worlds, including
the registered one-to-fifteen-step final-tail probes, have no issuer and fail
closed.

This is a narrow managed-boundary guarantee.  It cannot prove that a copied or
modified checkout, an external reimplementation, or a process that bypasses
this Python entry point did not execute equivalent worlds.
"""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import fcntl
import functools
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import stat
import sys
import zipfile
import zipimport
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NoReturn, cast

import numpy as np

from alberta_framework.core.slot_signaling_agent import SlotSignalingConfig
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CONSUMED_CALIBRATION_NAMESPACE,
    N_MATCHED_CASES,
    SEED_SNAPSHOT_SHA256,
    build_hidden_regime_factorial_calibration_design,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_CALIBRATION_MANIFESTS,
    HIDDEN_REGIME_STRUCTURAL_MANIFESTS,
    HiddenRegimeWorldConfig,
)

CALIBRATION_EXECUTION_GENESIS_SCHEMA = "alberta.hidden-regime-factorial.execution-genesis.v3"
CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-genesis-receipt-binding.v3"
)
CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-authorization.v3"
)
CALIBRATION_EXECUTION_STARTED_SCHEMA = "alberta.hidden-regime-factorial.execution-started.v3"
CALIBRATION_EXECUTION_COMPLETED_SCHEMA = "alberta.hidden-regime-factorial.execution-completed.v4"
CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-replay-started.v2"
)
CALIBRATION_EXECUTION_FINALIZED_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-shard-finalized.v4"
)
CALIBRATION_EXECUTION_TRACE_AUDIT_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-trace-audit-binding.v1"
)
CALIBRATION_EXECUTION_INVENTORY_SCHEMA = "alberta.hidden-regime-factorial.execution-inventory.v4"
CALIBRATION_ZIP_PROVENANCE_SCHEMA = "alberta.hidden-regime-factorial.execution-zip-provenance.v2"
CALIBRATION_ZIP_PROVENANCE_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-zip-provenance-binding.v1"
)
CALIBRATION_EXECUTION_FINAL_SHARD_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-final-shard.canonical-envelope.v1"
)
CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA = "alberta.hidden-regime-factorial.case-shard.v3"
CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-outcome.component-bundle.v3"
)
CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-summary.canonical-float-hex.v1"
)
CALIBRATION_EXECUTION_CONFIGURATION_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-configuration.canonical-float-hex.v1"
)
CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-resource.canonical-float-hex.v1"
)
CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.trace-digest.v1"
)
CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.final-learner-state.pytree-leaves.v1"
)

READINESS_EXECUTION_GOVERNANCE_FIELD = "execution_governance"
EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT = (
    "authorize exactly one managed nonpromoting hidden-regime calibration case"
)
MANAGED_EXECUTION_BOUNDARY_SCOPE = (
    "enforces exact registered worlds only inside the managed "
    "run_hidden_regime_condition entry point; cannot prove non-execution by an external "
    "clone, modified checkout, reimplementation, debugger, or bypass of that entry point"
)
CRASH_CONSUMPTION_RULE = (
    "atomic started publication consumes the case before learner execution; interruption "
    "permits only an explicitly authorized replay of the identical case binding, never seed, "
    "manifest, configuration, condition, or readiness substitution and never a best-of retry"
)
PROTECTED_EXECUTION_POLICY = (
    "no protected learner-execution authorization issuer exists in this schema"
)
PROCESS_LOCAL_AUTHORIZATION_SCOPE = (
    "authorization and execution-ticket seals are process-local capabilities: the bound "
    "ZIP-only worker must issue and consume them in one process; serialized or cross-process "
    "handoff is unsupported; seals bind the interpreter PID and process-start nonce, and a fork "
    "child is re-keyed with inherited capability registries cleared"
)
ZIP_PROVENANCE_POLICY = (
    "authorization requires an identity-registered process-local capability minted only after "
    "the exact receipt source.zip is hashed from an immutable nonsymlink file, is sys.path[0], "
    "is the sole project source path, the working directory is empty, and every loaded "
    "alberta_framework module is zipimport-loaded from a member of that exact archive; bytecode "
    "writes must be disabled and a command-line pycache prefix must name a fresh empty "
    "nonsymlink directory outside the source, readiness directory, and working directory; "
    "automatic site initialization must be disabled, no site/customization module may be loaded, "
    "and the interpreter prefixes and raw dependency search paths must equal the receipt binding"
)
ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR = "bound-readiness-source-archive"
SHARD_FINALIZATION_POLICY = (
    "a case is complete only after the certified ZIP worker hashes the ephemeral run's exact "
    "configuration, summary, resource report, primitive trace, and final state against immutable "
    "completion both before and after its trace audit, "
    "independently recomputes the full primitive-trace audit inside finalization, validates its "
    "compact shard projection, and publishes one new-only immutable finalization containing the "
    "canonical compact shard envelope and audit-input binding; the raw trace and final state are "
    "ephemeral and therefore cannot be re-audited from the compact ledger alone; case mutation "
    "is serialized among cooperating writers and finalized cases cannot be replayed"
)
REPLAY_ACCOUNTING_POLICY = (
    "the initial immutable started record is attempt zero and every explicitly authorized exact "
    "replay appends one immutable replay-start record before learner execution; each attempt "
    "seals its full request digest and replay-consent bit while the scientific case binding "
    "uses a replay-invariant request projection"
)

type ExecutionSensitivity = Literal["ordinary", "calibration", "protected"]
type ExecutionMode = Literal[
    "first_execution",
    "exact_replay_after_interruption",
    "exact_replay_after_completion",
]

_PROCESS_SEAL_KEY = secrets.token_bytes(32)
_PROCESS_START_NONCE = secrets.token_hex(32)
_SHA256_LENGTH = 64
_MAX_RECORD_BYTES = 12 * 1024 * 1024
_MAX_SOURCE_ARCHIVE_BYTES = 64 * 1024 * 1024
_CASE_DIRECTORY_PREFIX = "case-"
_STARTED_FILE = "started.json"
_COMPLETED_FILE = "completed.json"
_FINALIZED_FILE = "finalized.json"
_REPLAY_FILE_PATTERN = re.compile(r"replay-(?P<attempt_index>[0-9]{6})\.json\Z")
_MAX_ATTEMPTS_PER_CASE = 1_000_000
_AT_EMPTY_PATH = 0x1000
_RENAME_NOREPLACE = 1
_LEDGER_STAGE_SUFFIX = ".execution-ledger-stage-v1"
_ATOMIC_INSTALL_PRECOMMIT_STAGES = (
    "anonymous_stage_opened",
    "stage_bytes_written",
    "stage_mode_sealed",
    "stage_data_synced",
)
_ATOMIC_INSTALL_POSTCOMMIT_STAGES = (
    "final_name_installed",
    "directory_synced",
)


@functools.lru_cache(maxsize=1)
def _frozen_design() -> Any:
    return build_hidden_regime_factorial_calibration_design()


class HiddenRegimeExecutionGovernanceError(RuntimeError):
    """A managed execution, ledger, or authorization contract failed."""


class HiddenRegimeProtectedExecutionError(HiddenRegimeExecutionGovernanceError):
    """A protected structural world reached a learner-execution entry point."""


class HiddenRegimeCaseConsumedError(HiddenRegimeExecutionGovernanceError):
    """A calibration case was already atomically consumed."""


def _fail(message: str) -> NoReturn:
    raise HiddenRegimeExecutionGovernanceError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_value(value: object, *, location: str = "$") -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError(f"{location} contains a non-finite float")
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_value(item, location=f"{location}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{location} contains a non-string key")
            _validate_json_value(item, location=f"{location}.{key}")
        return
    raise TypeError(f"{location} contains unsupported JSON type {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    """Return the governance schema's canonical ASCII JSON bytes."""

    _validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    """Hash one canonical governance payload."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _normalized_json(value: object) -> object:
    """Normalize tuples and dataclass payloads through the canonical JSON codec."""

    return json.loads(canonical_json_bytes(value))


def _strict_json(raw: bytes, label: str) -> dict[str, object]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise HiddenRegimeExecutionGovernanceError(f"{label} is not ASCII") from error

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        _fail(f"{label} contains forbidden JSON constant {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise HiddenRegimeExecutionGovernanceError(f"{label} is invalid JSON") from error
    _require(type(parsed) is dict, f"{label} must contain one plain object")
    result = cast(dict[str, object], parsed)
    _require(canonical_json_bytes(result) == raw, f"{label} is not canonical JSON")
    return result


def _payload_with_digest(body: Mapping[str, object], digest_field: str) -> dict[str, object]:
    normalized = cast(dict[str, object], _normalized_json(dict(body)))
    return {**normalized, digest_field: canonical_sha256(normalized)}


def _validate_payload_digest(
    payload: Mapping[str, object],
    *,
    digest_field: str,
    label: str,
) -> dict[str, object]:
    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    digest = normalized.pop(digest_field, None)
    _require(_is_sha256(digest), f"{label}.{digest_field} is not lowercase SHA-256")
    _require(canonical_sha256(normalized) == digest, f"{label} content digest differs")
    return normalized


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} fields differ from the exact schema")


def _strict_int(value: object, label: str, *, minimum: int = 0, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be a strict integer in [{minimum}, {maximum}]")
    return value


def _strict_string(value: object, label: str) -> str:
    _require(type(value) is str and bool(value), f"{label} must be a nonempty string")
    return cast(str, value)


def _strict_sha256(value: object, label: str) -> str:
    _require(_is_sha256(value), f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


@dataclass(frozen=True, slots=True)
class HiddenRegimeExecutionClassification:
    """Content-derived managed-boundary classification for one world."""

    sensitivity: ExecutionSensitivity
    manifest_name: str | None
    manifest_payload_sha256: str | None
    match_kind: str
    final_tail_extension_steps: int | None

    @property
    def sensitive(self) -> bool:
        return self.sensitivity != "ordinary"


def _manifest_digest(manifest: object) -> str:
    return canonical_sha256(cast(Any, manifest).to_dict())


def classify_hidden_regime_world(
    world: HiddenRegimeWorldConfig,
) -> HiddenRegimeExecutionClassification:
    """Classify exact registered world content without trusting provenance labels."""

    if not isinstance(world, HiddenRegimeWorldConfig):
        raise TypeError("world must be a HiddenRegimeWorldConfig")
    for name, manifest in HIDDEN_REGIME_STRUCTURAL_MANIFESTS.items():
        same_nonlength_content = (
            world.segment_regimes == manifest.segment_regimes
            and world.regime_permutations == manifest.regime_permutations
        )
        if not same_nonlength_content:
            continue
        if world.segment_lengths == manifest.segment_lengths:
            return HiddenRegimeExecutionClassification(
                "protected",
                name,
                _manifest_digest(manifest),
                "exact_structural_manifest",
                0,
            )
        if world.segment_lengths[:-1] == manifest.segment_lengths[:-1]:
            extension = world.segment_lengths[-1] - manifest.segment_lengths[-1]
            if 1 <= extension <= 15:
                return HiddenRegimeExecutionClassification(
                    "protected",
                    name,
                    _manifest_digest(manifest),
                    "registered_structural_final_tail_extension",
                    extension,
                )
    for name, manifest in HIDDEN_REGIME_CALIBRATION_MANIFESTS.items():
        if (
            world.segment_lengths == manifest.segment_lengths
            and world.segment_regimes == manifest.segment_regimes
            and world.regime_permutations == manifest.regime_permutations
        ):
            return HiddenRegimeExecutionClassification(
                "calibration",
                name,
                _manifest_digest(manifest),
                "exact_calibration_manifest",
                0,
            )
    return HiddenRegimeExecutionClassification(
        "ordinary",
        None,
        None,
        "not_an_exact_managed_manifest",
        None,
    )


def hidden_regime_world_requires_managed_execution(world: HiddenRegimeWorldConfig) -> bool:
    """Return whether the managed entry point must not execute this world ordinarily."""

    return classify_hidden_regime_world(world).sensitive


def build_calibration_execution_genesis(
    *,
    source_archive_sha256: str,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    """Build the deterministic, outcome-free zero-entry ledger genesis."""

    for label, digest in (
        ("source_archive_sha256", source_archive_sha256),
        ("source_manifest_sha256", source_manifest_sha256),
        ("runtime_identity_sha256", runtime_identity_sha256),
    ):
        _strict_sha256(digest, label)
    initial_inventory = {
        "schema": CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "expected_case_count": N_MATCHED_CASES,
        "started_case_indices": [],
        "completed_case_indices": [],
        "finalized_case_indices": [],
        "learner_interrupted_case_indices": [],
        "post_audit_unfinalized_case_indices": [],
        "started_record_count": 0,
        "completed_record_count": 0,
        "finalized_record_count": 0,
        "managed_execution_attempt_count": 0,
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": True,
    }
    body: dict[str, object] = {
        "schema": CALIBRATION_EXECUTION_GENESIS_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "calibration_outcomes_observed": False,
        "protected_execution_permitted": False,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "source_archive_sha256": source_archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "expected_case_count": N_MATCHED_CASES,
        "case_directory_format": "cases/case-{case_index:03d}",
        "started_record_file": _STARTED_FILE,
        "completed_record_file": _COMPLETED_FILE,
        "finalized_record_file": _FINALIZED_FILE,
        "replay_record_file_format": "replay-{attempt_index:06d}.json",
        "final_case_shard_schema": CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
        "initial_inventory": initial_inventory,
        "initial_inventory_sha256": canonical_sha256(initial_inventory),
        "crash_consumption_rule": CRASH_CONSUMPTION_RULE,
        "protected_execution_policy": PROTECTED_EXECUTION_POLICY,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
        "zip_provenance_policy": ZIP_PROVENANCE_POLICY,
        "shard_finalization_policy": SHARD_FINALIZATION_POLICY,
        "replay_accounting_policy": REPLAY_ACCOUNTING_POLICY,
    }
    return _payload_with_digest(body, "genesis_sha256")


def require_valid_calibration_execution_genesis(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate and normalize one deterministic genesis payload."""

    body = _validate_payload_digest(
        payload,
        digest_field="genesis_sha256",
        label="execution genesis",
    )
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "calibration_outcomes_observed",
            "protected_execution_permitted",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "expected_case_count",
            "case_directory_format",
            "started_record_file",
            "completed_record_file",
            "finalized_record_file",
            "replay_record_file_format",
            "final_case_shard_schema",
            "initial_inventory",
            "initial_inventory_sha256",
            "crash_consumption_rule",
            "protected_execution_policy",
            "managed_boundary_scope",
            "zip_provenance_policy",
            "shard_finalization_policy",
            "replay_accounting_policy",
        },
        "execution genesis",
    )
    _require(body["schema"] == CALIBRATION_EXECUTION_GENESIS_SCHEMA, "genesis schema differs")
    _require(body["development_only"] is True, "genesis is not development-only")
    for field in (
        "scientific_promotion_allowed",
        "calibration_outcomes_observed",
        "protected_execution_permitted",
    ):
        _require(body[field] is False, f"genesis {field} must be false")
    _require(
        body["protocol_payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "genesis protocol digest differs",
    )
    _require(
        body["seed_snapshot_sha256"] == SEED_SNAPSHOT_SHA256,
        "genesis seed snapshot differs",
    )
    for field in (
        "source_archive_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
        "initial_inventory_sha256",
    ):
        _strict_sha256(body[field], f"genesis.{field}")
    _require(body["expected_case_count"] == N_MATCHED_CASES, "genesis case count differs")
    _require(
        body["case_directory_format"] == "cases/case-{case_index:03d}",
        "genesis case path format differs",
    )
    _require(body["started_record_file"] == _STARTED_FILE, "genesis start file differs")
    _require(body["completed_record_file"] == _COMPLETED_FILE, "genesis completion file differs")
    _require(body["finalized_record_file"] == _FINALIZED_FILE, "genesis finalization file differs")
    _require(
        body["replay_record_file_format"] == "replay-{attempt_index:06d}.json",
        "genesis replay file format differs",
    )
    _require(
        body["final_case_shard_schema"] == CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
        "genesis final case-shard schema differs",
    )
    _require(body["zip_provenance_policy"] == ZIP_PROVENANCE_POLICY, "genesis ZIP policy differs")
    _require(
        body["shard_finalization_policy"] == SHARD_FINALIZATION_POLICY,
        "genesis finalization policy differs",
    )
    _require(
        body["replay_accounting_policy"] == REPLAY_ACCOUNTING_POLICY,
        "genesis replay policy differs",
    )
    expected = build_calibration_execution_genesis(
        source_archive_sha256=cast(str, body["source_archive_sha256"]),
        source_manifest_sha256=cast(str, body["source_manifest_sha256"]),
        runtime_identity_sha256=cast(str, body["runtime_identity_sha256"]),
    )
    _require(
        cast(dict[str, object], _normalized_json(dict(payload))) == expected,
        "genesis differs from the deterministic zero-entry payload",
    )
    return expected


def calibration_execution_genesis_receipt_binding(
    genesis: Mapping[str, object],
) -> dict[str, object]:
    """Return the exact readiness-body binding for a pristine genesis."""

    normalized = require_valid_calibration_execution_genesis(genesis)
    return {
        "schema": CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
        "genesis_sha256": normalized["genesis_sha256"],
        "protocol_payload_sha256": normalized["protocol_payload_sha256"],
        "seed_snapshot_sha256": normalized["seed_snapshot_sha256"],
        "source_archive_sha256": normalized["source_archive_sha256"],
        "source_manifest_sha256": normalized["source_manifest_sha256"],
        "runtime_identity_sha256": normalized["runtime_identity_sha256"],
        "initial_inventory_sha256": normalized["initial_inventory_sha256"],
        "initial_started_record_count": 0,
        "initial_completed_record_count": 0,
        "initial_finalized_record_count": 0,
        "initial_managed_execution_attempt_count": 0,
        "final_case_shard_schema": CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
        "initial_protected_record_count": 0,
        "protected_execution_permitted": False,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
        "zip_provenance_policy": ZIP_PROVENANCE_POLICY,
        "shard_finalization_policy": SHARD_FINALIZATION_POLICY,
        "replay_accounting_policy": REPLAY_ACCOUNTING_POLICY,
    }


@dataclass(frozen=True, slots=True)
class PublishedCalibrationExecutionLedger:
    """Paths for one new-only content-addressed execution ledger."""

    directory: Path
    genesis_path: Path
    cases_directory: Path
    genesis_sha256: str


def _open_directory_without_symlink_ancestors(path: Path, label: str) -> int:
    """Open an absolute directory one O_NOFOLLOW component at a time."""

    absolute = path.absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise HiddenRegimeExecutionGovernanceError(
                        f"{label} traverses a symlink or non-directory component"
                    ) from error
                if error.errno == errno.ENOENT:
                    raise HiddenRegimeExecutionGovernanceError(
                        f"{label} is missing: {absolute}"
                    ) from error
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        _require(stat.S_ISDIR(os.fstat(descriptor).st_mode), f"{label} is not a directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _directory_without_symlink(path: Path, label: str) -> os.stat_result:
    descriptor = _open_directory_without_symlink_ancestors(path, label)
    try:
        return os.fstat(descriptor)
    finally:
        os.close(descriptor)


def _directory_members(path: Path, label: str) -> list[str]:
    descriptor = _open_directory_without_symlink_ancestors(path, label)
    try:
        return sorted(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def _atomic_install_checkpoint(stage: str, name: str) -> None:
    """Expose deterministic fault points without changing the atomic install protocol."""

    del stage, name


def _link_anonymous_file_no_replace(
    descriptor: int,
    directory_fd: int,
    name: str,
) -> None:
    """Atomically give a fully sealed anonymous inode its final, new-only name."""

    try:
        linkat = cast(Any, ctypes.CDLL(None, use_errno=True).linkat)
    except AttributeError as error:  # pragma: no cover - Linux exposes linkat
        raise HiddenRegimeExecutionGovernanceError(
            "the platform lacks linkat required for atomic immutable-file installation"
        ) from error
    ctypes.set_errno(0)
    result = int(
        linkat(
            descriptor,
            b"",
            directory_fd,
            os.fsencode(name),
            _AT_EMPTY_PATH,
        )
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), name)
    raise OSError(error_number, os.strerror(error_number), name)


def _rename_no_replace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically move a completed inode or tree without replacing a destination."""

    try:
        renameat2 = cast(Any, ctypes.CDLL(None, use_errno=True).renameat2)
    except AttributeError as error:  # pragma: no cover - Linux exposes renameat2
        raise HiddenRegimeExecutionGovernanceError(
            "the platform lacks renameat2 required for atomic ledger publication"
        ) from error
    ctypes.set_errno(0)
    result = int(
        renameat2(
            source_directory_fd,
            os.fsencode(source_name),
            destination_directory_fd,
            os.fsencode(destination_name),
            _RENAME_NOREPLACE,
        )
    )
    if result == 0:
        return
    error_number = ctypes.get_errno() or errno.EIO
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    raise OSError(error_number, os.strerror(error_number), destination_name)


def atomic_install_new_immutable(
    directory_fd: int,
    name: str,
    raw: bytes,
    *,
    max_bytes: int,
    label: str,
) -> None:
    """Atomically install one complete 0444 file at a previously absent plain name."""

    _require(
        bool(name) and name not in {".", ".."} and "/" not in name and "\x00" not in name,
        f"{label} name is invalid",
    )
    _require(
        type(max_bytes) is int and max_bytes > 0,
        f"{label} maximum size is invalid",
    )
    _require(type(label) is str and bool(label), "immutable publication label is invalid")
    _require(type(raw) is bytes and bool(raw), f"{label} bytes are invalid")
    _require(len(raw) <= max_bytes, f"{label} exceeds the size limit")
    anonymous_flag = getattr(os, "O_TMPFILE", None)
    _require(
        type(anonymous_flag) is int,
        f"the platform lacks O_TMPFILE required for atomic {label} installation",
    )
    flags = os.O_WRONLY | cast(int, anonymous_flag) | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(".", flags, 0o600, dir_fd=directory_fd)
    except OSError as error:
        raise HiddenRegimeExecutionGovernanceError(
            "the destination filesystem cannot create the anonymous inode required for atomic "
            f"{label} installation"
        ) from error
    try:
        opened = os.fstat(descriptor)
        _require(
            stat.S_ISREG(opened.st_mode)
            and stat.S_IMODE(opened.st_mode) == 0o600
            and opened.st_nlink == 0,
            f"anonymous {label} staging inode is invalid",
        )
        _atomic_install_checkpoint("anonymous_stage_opened", name)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing execution ledger record")
            view = view[written:]
        _atomic_install_checkpoint("stage_bytes_written", name)
        os.fchmod(descriptor, 0o444)
        _atomic_install_checkpoint("stage_mode_sealed", name)
        os.fsync(descriptor)
        staged = os.fstat(descriptor)
        _require(
            stat.S_ISREG(staged.st_mode)
            and stat.S_IMODE(staged.st_mode) == 0o444
            and staged.st_nlink == 0
            and staged.st_size == len(raw),
            f"sealed anonymous {label} staging inode is invalid",
        )
        _atomic_install_checkpoint("stage_data_synced", name)
        _link_anonymous_file_no_replace(descriptor, directory_fd, name)
        installed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(
            stat.S_ISREG(installed.st_mode)
            and stat.S_IMODE(installed.st_mode) == 0o444
            and installed.st_nlink == 1
            and installed.st_size == len(raw)
            and (installed.st_dev, installed.st_ino) == (staged.st_dev, staged.st_ino),
            f"atomically installed {label} is invalid",
        )
        _atomic_install_checkpoint("final_name_installed", name)
        os.fsync(directory_fd)
        _atomic_install_checkpoint("directory_synced", name)
    finally:
        os.close(descriptor)


def _write_new_immutable(directory_fd: int, name: str, raw: bytes) -> None:
    """Install one exact execution-ledger record through the reusable atomic primitive."""

    _require(
        name in {"genesis.json", _STARTED_FILE, _COMPLETED_FILE, _FINALIZED_FILE}
        or _REPLAY_FILE_PATTERN.fullmatch(name) is not None,
        "execution ledger record name is invalid",
    )
    atomic_install_new_immutable(
        directory_fd,
        name,
        raw,
        max_bytes=_MAX_RECORD_BYTES,
        label="execution ledger record",
    )


def _ledger_initialization_checkpoint(stage: str, genesis_sha256: str) -> None:
    """Expose deterministic whole-ledger publication fault points for validation."""

    del stage, genesis_sha256


def _remove_stale_ledger_stage(
    root_fd: int,
    stage_name: str,
    *,
    expected_case_names: set[str],
) -> None:
    """Remove only the exact, inert partial tree reserved for one ledger initialization."""

    try:
        stage_status = os.stat(stage_name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    _require(
        stat.S_ISDIR(stage_status.st_mode) and stat.S_IMODE(stage_status.st_mode) == 0o700,
        "stale execution-ledger staging path is not an exact private directory",
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    stage_fd = os.open(stage_name, directory_flags, dir_fd=root_fd)
    try:
        stage_members = set(os.listdir(stage_fd))
        _require(
            stage_members <= {"cases", "genesis.json"},
            "stale execution-ledger staging directory contains an unknown member",
        )
        if "genesis.json" in stage_members:
            genesis_status = os.stat("genesis.json", dir_fd=stage_fd, follow_symlinks=False)
            _require(
                stat.S_ISREG(genesis_status.st_mode)
                and stat.S_IMODE(genesis_status.st_mode) == 0o444
                and genesis_status.st_nlink == 1
                and genesis_status.st_size <= _MAX_RECORD_BYTES,
                "stale staged execution genesis is not a complete immutable file",
            )
            os.unlink("genesis.json", dir_fd=stage_fd)
        if "cases" in stage_members:
            cases_fd = os.open("cases", directory_flags, dir_fd=stage_fd)
            try:
                case_names = set(os.listdir(cases_fd))
                _require(
                    case_names <= expected_case_names,
                    "stale staged execution cases contain an unknown member",
                )
                for case_name in sorted(case_names):
                    case_fd = os.open(case_name, directory_flags, dir_fd=cases_fd)
                    try:
                        _require(
                            stat.S_IMODE(os.fstat(case_fd).st_mode) == 0o700
                            and not os.listdir(case_fd),
                            "stale staged execution case directory is not exact and empty",
                        )
                    finally:
                        os.close(case_fd)
                    os.rmdir(case_name, dir_fd=cases_fd)
            finally:
                os.close(cases_fd)
            os.rmdir("cases", dir_fd=stage_fd)
        _require(not os.listdir(stage_fd), "stale execution-ledger staging cleanup is incomplete")
    finally:
        os.close(stage_fd)
    os.rmdir(stage_name, dir_fd=root_fd)
    os.fsync(root_fd)


def initialize_calibration_execution_ledger(
    publication_root: Path,
    genesis: Mapping[str, object],
    *,
    authorize_initialization: bool,
) -> PublishedCalibrationExecutionLedger:
    """Publish a pristine ledger at ``<root>/<genesis_sha256>`` exactly once."""

    _require(authorize_initialization is True, "ledger initialization requires authorization")
    normalized = require_valid_calibration_execution_genesis(genesis)
    root = publication_root.absolute()
    _directory_without_symlink(root, "ledger publication root")
    digest = cast(str, normalized["genesis_sha256"])
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    root_fd = _open_directory_without_symlink_ancestors(root, "ledger publication root")
    stage_name = f".{digest}{_LEDGER_STAGE_SUFFIX}"
    expected_names = {f"{_CASE_DIRECTORY_PREFIX}{index:03d}" for index in range(N_MATCHED_CASES)}
    ledger_fd: int | None = None
    cases_fd: int | None = None
    directory_published = False
    try:
        fcntl.flock(root_fd, fcntl.LOCK_EX)
        try:
            os.stat(digest, dir_fd=root_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"refusing to overwrite execution ledger: {root / digest}")
        _remove_stale_ledger_stage(
            root_fd,
            stage_name,
            expected_case_names=expected_names,
        )
        try:
            os.mkdir(stage_name, 0o700, dir_fd=root_fd)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite execution ledger: {root / digest}"
            ) from error
        _ledger_initialization_checkpoint("staging_directory_created", digest)
        ledger_fd = os.open(stage_name, flags, dir_fd=root_fd)
        os.mkdir("cases", 0o700, dir_fd=ledger_fd)
        _ledger_initialization_checkpoint("cases_directory_created", digest)
        cases_fd = os.open("cases", flags, dir_fd=ledger_fd)
        for case_index in range(N_MATCHED_CASES):
            os.mkdir(f"{_CASE_DIRECTORY_PREFIX}{case_index:03d}", 0o700, dir_fd=cases_fd)
        _ledger_initialization_checkpoint("case_directories_created", digest)
        _write_new_immutable(ledger_fd, "genesis.json", canonical_json_bytes(normalized))
        _ledger_initialization_checkpoint("genesis_installed", digest)
        os.fsync(cases_fd)
        os.fsync(ledger_fd)
        _ledger_initialization_checkpoint("staging_tree_synced", digest)
        os.close(cases_fd)
        cases_fd = None
        os.close(ledger_fd)
        ledger_fd = None
        try:
            _rename_no_replace(root_fd, stage_name, root_fd, digest)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite execution ledger: {root / digest}"
            ) from error
        directory_published = True
        _ledger_initialization_checkpoint("final_directory_published", digest)
        os.fsync(root_fd)
        _ledger_initialization_checkpoint("publication_root_synced", digest)
    finally:
        if cases_fd is not None:
            os.close(cases_fd)
        if ledger_fd is not None:
            os.close(ledger_fd)
        try:
            if not directory_published:
                _remove_stale_ledger_stage(
                    root_fd,
                    stage_name,
                    expected_case_names=expected_names,
                )
        finally:
            try:
                fcntl.flock(root_fd, fcntl.LOCK_UN)
            finally:
                os.close(root_fd)
    published = PublishedCalibrationExecutionLedger(
        directory=root / digest,
        genesis_path=root / digest / "genesis.json",
        cases_directory=root / digest / "cases",
        genesis_sha256=digest,
    )
    snapshot = snapshot_calibration_execution_inventory(published.directory)
    _require(snapshot["pristine"] is True, "new execution ledger is not pristine")
    return published


def _open_immutable_bytes(path: Path, label: str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = _open_directory_without_symlink_ancestors(path.parent, f"{label} parent")
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as error:
        os.close(parent_fd)
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise HiddenRegimeExecutionGovernanceError(f"symlinked {label} is forbidden") from error
        raise
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        _require(stat.S_IMODE(before.st_mode) == 0o444, f"{label} mode is not 0444")
        _require(before.st_nlink == 1, f"{label} has multiple hard links")
        _require(before.st_size <= max_bytes, f"{label} exceeds the size limit")
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            _require(bool(chunk), f"{label} changed while read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        locator = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (locator.st_dev, locator.st_ino, locator.st_size, locator.st_mtime_ns),
            f"{label} changed or was replaced while read",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _open_immutable_json(path: Path, label: str) -> dict[str, object]:
    return _strict_json(
        _open_immutable_bytes(path, label, max_bytes=_MAX_RECORD_BYTES),
        label,
    )


def _load_ledger_genesis(directory: Path) -> dict[str, object]:
    absolute = directory.absolute()
    _directory_without_symlink(absolute, "execution ledger")
    _require(
        _directory_members(absolute, "execution ledger") == ["cases", "genesis.json"],
        "execution ledger root inventory differs",
    )
    payload = _open_immutable_json(absolute / "genesis.json", "execution genesis")
    normalized = require_valid_calibration_execution_genesis(payload)
    _require(
        absolute.name == normalized["genesis_sha256"],
        "ledger directory is not content addressed by its genesis",
    )
    _directory_without_symlink(absolute / "cases", "execution cases directory")
    return normalized


def _case_directory(directory: Path, case_index: int) -> Path:
    _strict_int(case_index, "case_index", maximum=N_MATCHED_CASES - 1)
    path = directory.absolute() / "cases" / f"{_CASE_DIRECTORY_PREFIX}{case_index:03d}"
    _directory_without_symlink(path, f"execution case {case_index} directory")
    return path


@contextmanager
def _case_mutation_critical_section(
    case_directory: Path,
    case_index: int,
) -> Iterator[int]:
    """Serialize cooperating case mutations across threads and worker processes."""

    descriptor = _open_directory_without_symlink_ancestors(
        case_directory,
        f"execution case {case_index} directory",
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield descriptor
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _validated_case_member_names(case_directory: Path, case_index: int) -> set[str]:
    members = set(_directory_members(case_directory, f"execution case {case_index} directory"))
    _require(
        all(
            member in {_STARTED_FILE, _COMPLETED_FILE, _FINALIZED_FILE}
            or _REPLAY_FILE_PATTERN.fullmatch(member) is not None
            for member in members
        ),
        f"execution case {case_index} contains an unknown member",
    )
    _require(
        _COMPLETED_FILE not in members or _STARTED_FILE in members,
        f"execution case {case_index} completed without a start",
    )
    _require(
        _FINALIZED_FILE not in members or _COMPLETED_FILE in members,
        f"execution case {case_index} finalized without a completion",
    )
    _require(
        not any(_REPLAY_FILE_PATTERN.fullmatch(member) for member in members)
        or _STARTED_FILE in members,
        f"execution case {case_index} has a replay without an initial start",
    )
    return members


def _validate_seed_pair_payload(value: object) -> dict[str, object]:
    _require(type(value) is dict, "case binding seed_pair must be a plain object")
    payload = cast(dict[str, object], value)
    _exact_keys(payload, {"namespace", "index", "world_seed", "learner_seed"}, "seed_pair")
    _require(
        payload["namespace"] == CONSUMED_CALIBRATION_NAMESPACE,
        "case binding seed namespace differs",
    )
    _strict_int(payload["index"], "seed_pair.index", maximum=29)
    _strict_int(payload["world_seed"], "seed_pair.world_seed", maximum=0xFFFFFFFF)
    _strict_int(payload["learner_seed"], "seed_pair.learner_seed", maximum=0xFFFFFFFF)
    return payload


_CASE_BINDING_FIELDS = {
    "protocol_payload_sha256",
    "seed_snapshot_sha256",
    "genesis_sha256",
    "readiness_receipt_sha256",
    "source_archive_sha256",
    "source_manifest_sha256",
    "runtime_identity_sha256",
    "zip_provenance_binding_sha256",
    "zip_provenance_attestation_sha256",
    "case_index",
    "manifest_name",
    "manifest_payload_sha256",
    "configuration_sha256",
    "case_request_binding_sha256",
    "condition",
    "seed_pair",
}


def _require_case_binding(value: object) -> dict[str, object]:
    _require(type(value) is dict, "case binding must be a plain object")
    binding = cast(dict[str, object], value)
    _exact_keys(binding, _CASE_BINDING_FIELDS, "case binding")
    _require(
        binding["protocol_payload_sha256"] == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "case binding protocol differs",
    )
    _require(binding["seed_snapshot_sha256"] == SEED_SNAPSHOT_SHA256, "case binding seeds differ")
    for field in (
        "genesis_sha256",
        "readiness_receipt_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
        "zip_provenance_binding_sha256",
        "zip_provenance_attestation_sha256",
        "manifest_payload_sha256",
        "configuration_sha256",
        "case_request_binding_sha256",
    ):
        _strict_sha256(binding[field], f"case binding.{field}")
    case_index = _strict_int(binding["case_index"], "case binding.case_index", maximum=239)
    design = _frozen_design()
    case = design.cases[case_index]
    _require(binding["manifest_name"] == case.manifest_name, "case manifest differs")
    _require(binding["condition"] == case.condition, "case condition differs")
    expected_manifest_digest = _manifest_digest(
        HIDDEN_REGIME_CALIBRATION_MANIFESTS[case.manifest_name]
    )
    _require(
        binding["manifest_payload_sha256"] == expected_manifest_digest,
        "case manifest digest differs",
    )
    pair = _validate_seed_pair_payload(binding["seed_pair"])
    _require(
        pair
        == {
            "namespace": CONSUMED_CALIBRATION_NAMESPACE,
            "index": case.seed_index,
            "world_seed": case.world_seed,
            "learner_seed": case.learner_seed,
        },
        "case seed pair differs from the frozen case",
    )
    return binding


def _started_record(
    binding: Mapping[str, object],
    *,
    attempt_request_payload_sha256: str,
    exact_replay_consent: bool,
) -> dict[str, object]:
    _strict_sha256(attempt_request_payload_sha256, "initial attempt request digest")
    _require(type(exact_replay_consent) is bool, "initial exact-replay consent is not boolean")
    body = {
        "schema": CALIBRATION_EXECUTION_STARTED_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_state": "started_case_consumed",
        "case_binding": dict(binding),
        "case_binding_sha256": canonical_sha256(binding),
        "attempt_request_payload_sha256": attempt_request_payload_sha256,
        "exact_replay_consent": exact_replay_consent,
        "crash_consumption_rule": CRASH_CONSUMPTION_RULE,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    return _payload_with_digest(body, "started_record_sha256")


def require_valid_calibration_execution_started_record(
    payload: Mapping[str, object],
    *,
    expected_genesis_sha256: str | None = None,
) -> dict[str, object]:
    """Strictly validate one immutable case-consumption record."""

    body = _validate_payload_digest(
        payload,
        digest_field="started_record_sha256",
        label="started record",
    )
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "execution_state",
            "case_binding",
            "case_binding_sha256",
            "attempt_request_payload_sha256",
            "exact_replay_consent",
            "crash_consumption_rule",
            "managed_boundary_scope",
        },
        "started record",
    )
    _require(body["schema"] == CALIBRATION_EXECUTION_STARTED_SCHEMA, "started schema differs")
    _require(body["development_only"] is True, "started record is not development-only")
    _require(body["scientific_promotion_allowed"] is False, "started record allows promotion")
    _require(body["execution_state"] == "started_case_consumed", "started state differs")
    binding = _require_case_binding(body["case_binding"])
    _require(
        body["case_binding_sha256"] == canonical_sha256(binding),
        "started case binding digest differs",
    )
    attempt_request_digest = _strict_sha256(
        body["attempt_request_payload_sha256"],
        "started attempt request digest",
    )
    _require(type(body["exact_replay_consent"]) is bool, "started replay consent differs")
    if expected_genesis_sha256 is not None:
        _strict_sha256(expected_genesis_sha256, "expected_genesis_sha256")
        _require(
            binding["genesis_sha256"] == expected_genesis_sha256,
            "started record belongs to another genesis",
        )
    expected = _started_record(
        binding,
        attempt_request_payload_sha256=attempt_request_digest,
        exact_replay_consent=cast(bool, body["exact_replay_consent"]),
    )
    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    _require(normalized == expected, "started record is not deterministic")
    return expected


def _replay_started_record(
    binding: Mapping[str, object],
    *,
    attempt_index: int,
    execution_mode: ExecutionMode,
    prior_started_record_sha256: str,
    prior_completed_record_sha256: str | None,
    zip_provenance_attestation_sha256: str,
    attempt_request_payload_sha256: str,
    exact_replay_consent: bool,
) -> dict[str, object]:
    _strict_int(
        attempt_index,
        "replay attempt index",
        minimum=1,
        maximum=_MAX_ATTEMPTS_PER_CASE - 1,
    )
    _require(
        execution_mode in {"exact_replay_after_interruption", "exact_replay_after_completion"},
        "replay record mode differs",
    )
    _strict_sha256(prior_started_record_sha256, "replay prior start digest")
    if prior_completed_record_sha256 is not None:
        _strict_sha256(prior_completed_record_sha256, "replay prior completion digest")
    _strict_sha256(zip_provenance_attestation_sha256, "replay ZIP provenance digest")
    _strict_sha256(attempt_request_payload_sha256, "replay attempt request digest")
    _require(exact_replay_consent is True, "a replay attempt requires explicit replay consent")
    body = {
        "schema": CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_state": "exact_replay_started",
        "case_binding": dict(binding),
        "case_binding_sha256": canonical_sha256(binding),
        "attempt_index": attempt_index,
        "execution_mode": execution_mode,
        "prior_started_record_sha256": prior_started_record_sha256,
        "prior_completed_record_sha256": prior_completed_record_sha256,
        "zip_provenance_attestation_sha256": zip_provenance_attestation_sha256,
        "attempt_request_payload_sha256": attempt_request_payload_sha256,
        "exact_replay_consent": exact_replay_consent,
        "replay_accounting_policy": REPLAY_ACCOUNTING_POLICY,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    return _payload_with_digest(body, "replay_started_record_sha256")


def require_valid_calibration_execution_replay_started_record(
    payload: Mapping[str, object],
    *,
    expected_started: Mapping[str, object],
    expected_attempt_index: int,
) -> dict[str, object]:
    """Strictly validate one immutable explicit-replay attempt record."""

    body = _validate_payload_digest(
        payload,
        digest_field="replay_started_record_sha256",
        label="replay started record",
    )
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "execution_state",
            "case_binding",
            "case_binding_sha256",
            "attempt_index",
            "execution_mode",
            "prior_started_record_sha256",
            "prior_completed_record_sha256",
            "zip_provenance_attestation_sha256",
            "attempt_request_payload_sha256",
            "exact_replay_consent",
            "replay_accounting_policy",
            "managed_boundary_scope",
        },
        "replay started record",
    )
    _require(
        body["schema"] == CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA,
        "replay started schema differs",
    )
    _require(body["development_only"] is True, "replay record is not development-only")
    _require(body["scientific_promotion_allowed"] is False, "replay record allows promotion")
    _require(body["execution_state"] == "exact_replay_started", "replay state differs")
    started = require_valid_calibration_execution_started_record(expected_started)
    binding = _require_case_binding(body["case_binding"])
    _require(binding == started["case_binding"], "replay and initial start bindings differ")
    _require(
        body["case_binding_sha256"] == canonical_sha256(binding), "replay binding digest differs"
    )
    attempt_index = _strict_int(
        body["attempt_index"],
        "replay attempt index",
        minimum=1,
        maximum=_MAX_ATTEMPTS_PER_CASE - 1,
    )
    _require(attempt_index == expected_attempt_index, "replay attempt index is noncontiguous")
    mode = cast(ExecutionMode, body["execution_mode"])
    prior_started = _strict_sha256(
        body["prior_started_record_sha256"],
        "replay prior start digest",
    )
    prior_completed_value = body["prior_completed_record_sha256"]
    prior_completed = (
        None
        if prior_completed_value is None
        else _strict_sha256(prior_completed_value, "replay prior completion digest")
    )
    provenance_digest = _strict_sha256(
        body["zip_provenance_attestation_sha256"],
        "replay ZIP provenance digest",
    )
    attempt_request_digest = _strict_sha256(
        body["attempt_request_payload_sha256"],
        "replay attempt request digest",
    )
    _require(body["exact_replay_consent"] is True, "replay record lacks explicit consent")
    _require(
        provenance_digest == binding["zip_provenance_attestation_sha256"],
        "replay ZIP provenance differs from the initial managed attempt",
    )
    _require(
        prior_started == started["started_record_sha256"],
        "replay prior start differs from initial start",
    )
    _require(body["replay_accounting_policy"] == REPLAY_ACCOUNTING_POLICY, "replay policy differs")
    _require(
        body["managed_boundary_scope"] == MANAGED_EXECUTION_BOUNDARY_SCOPE, "replay scope differs"
    )
    expected = _replay_started_record(
        binding,
        attempt_index=attempt_index,
        execution_mode=mode,
        prior_started_record_sha256=prior_started,
        prior_completed_record_sha256=prior_completed,
        zip_provenance_attestation_sha256=provenance_digest,
        attempt_request_payload_sha256=attempt_request_digest,
        exact_replay_consent=True,
    )
    _require(
        cast(dict[str, object], _normalized_json(dict(payload))) == expected,
        "replay started record is not deterministic",
    )
    return expected


def _case_attempt_rows(
    case_directory: Path,
    started: Mapping[str, object],
) -> list[dict[str, object]]:
    normalized_started = require_valid_calibration_execution_started_record(started)
    rows: list[dict[str, object]] = [
        {
            "attempt_index": 0,
            "execution_mode": "first_execution",
            "attempt_record_schema": CALIBRATION_EXECUTION_STARTED_SCHEMA,
            "attempt_record_sha256": normalized_started["started_record_sha256"],
            "attempt_request_payload_sha256": normalized_started[
                "attempt_request_payload_sha256"
            ],
            "exact_replay_consent": normalized_started["exact_replay_consent"],
            "zip_provenance_attestation_sha256": cast(
                dict[str, object],
                normalized_started["case_binding"],
            )["zip_provenance_attestation_sha256"],
        }
    ]
    replay_names = sorted(
        name
        for name in _directory_members(case_directory, "execution case directory")
        if _REPLAY_FILE_PATTERN.fullmatch(name)
    )
    for expected_attempt_index, name in enumerate(replay_names, start=1):
        match = _REPLAY_FILE_PATTERN.fullmatch(name)
        assert match is not None
        _require(
            int(match.group("attempt_index")) == expected_attempt_index,
            "replay attempt file sequence has a gap or duplicate",
        )
        record = require_valid_calibration_execution_replay_started_record(
            _open_immutable_json(case_directory / name, "replay started record"),
            expected_started=normalized_started,
            expected_attempt_index=expected_attempt_index,
        )
        rows.append(
            {
                "attempt_index": expected_attempt_index,
                "execution_mode": record["execution_mode"],
                "attempt_record_schema": CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA,
                "attempt_record_sha256": record["replay_started_record_sha256"],
                "attempt_request_payload_sha256": record["attempt_request_payload_sha256"],
                "exact_replay_consent": record["exact_replay_consent"],
                "zip_provenance_attestation_sha256": record["zip_provenance_attestation_sha256"],
            }
        )
    return rows


def _attempt_binding_from_rows(
    case_index: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    _strict_int(case_index, "attempt binding case index", maximum=N_MATCHED_CASES - 1)
    _require(bool(rows), "a started case must have at least one managed attempt")
    return {
        "case_index": case_index,
        "managed_execution_attempt_count": len(rows),
        "attempt_records_sha256": canonical_sha256(rows),
        "attempts": rows,
    }


def calibration_case_attempt_binding(
    inventory: Mapping[str, object],
    case_index: int,
) -> dict[str, object]:
    """Return and strictly validate one case's append-only managed-attempt binding."""

    _strict_int(case_index, "case_index", maximum=N_MATCHED_CASES - 1)
    records = _plain_list(inventory.get("attempt_records"), "inventory attempt records")
    matches = [
        _plain_mapping(item, "inventory attempt record")
        for item in records
        if _plain_mapping(item, "inventory attempt record").get("case_index") == case_index
    ]
    _require(len(matches) == 1, "case has no unique managed-attempt binding")
    binding = matches[0]
    _exact_keys(
        binding,
        {
            "case_index",
            "managed_execution_attempt_count",
            "attempt_records_sha256",
            "attempts",
        },
        "case attempt binding",
    )
    attempts = _plain_list(binding["attempts"], "case attempts")
    count = _strict_int(
        binding["managed_execution_attempt_count"],
        "managed execution attempt count",
        minimum=1,
        maximum=_MAX_ATTEMPTS_PER_CASE,
    )
    _require(len(attempts) == count, "managed execution attempt count differs")
    for expected_index, item in enumerate(attempts):
        row = _plain_mapping(item, "managed attempt")
        _exact_keys(
            row,
            {
                "attempt_index",
                "execution_mode",
                "attempt_record_schema",
                "attempt_record_sha256",
                "attempt_request_payload_sha256",
                "exact_replay_consent",
                "zip_provenance_attestation_sha256",
            },
            "managed attempt",
        )
        _require(row["attempt_index"] == expected_index, "managed attempt order differs")
        _require(
            row["execution_mode"]
            == ("first_execution" if expected_index == 0 else row["execution_mode"]),
            "initial managed attempt mode differs",
        )
        if expected_index > 0:
            _require(
                row["execution_mode"]
                in {"exact_replay_after_interruption", "exact_replay_after_completion"},
                "replay managed attempt mode differs",
            )
        _require(
            row["attempt_record_schema"]
            == (
                CALIBRATION_EXECUTION_STARTED_SCHEMA
                if expected_index == 0
                else CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA
            ),
            "managed attempt record schema differs",
        )
        _strict_sha256(row["attempt_record_sha256"], "managed attempt record digest")
        _strict_sha256(
            row["attempt_request_payload_sha256"],
            "managed attempt request digest",
        )
        _require(type(row["exact_replay_consent"]) is bool, "managed replay consent differs")
        if expected_index > 0:
            _require(row["exact_replay_consent"] is True, "managed replay lacks consent")
        _strict_sha256(
            row["zip_provenance_attestation_sha256"],
            "managed attempt ZIP provenance digest",
        )
    _require(
        binding["attempt_records_sha256"] == canonical_sha256(attempts),
        "managed attempt-record digest differs",
    )
    return cast(dict[str, object], _normalized_json(binding))


def _component_outcome_sha256(
    *,
    case_binding_sha256: str,
    started_record_sha256: str,
    summary_sha256: str,
    resource_sha256: str,
    primitive_trace_sha256: str,
    final_state_sha256: str,
    executed_steps: int,
) -> str:
    return canonical_sha256(
        {
            "schema": CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
            "case_binding_sha256": case_binding_sha256,
            "started_record_sha256": started_record_sha256,
            "summary_digest_schema": CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
            "summary_sha256": summary_sha256,
            "resource_digest_schema": CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
            "resource_sha256": resource_sha256,
            "primitive_trace_digest_schema": (CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA),
            "primitive_trace_sha256": primitive_trace_sha256,
            "final_state_digest_schema": CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA,
            "final_state_sha256": final_state_sha256,
            "executed_steps": executed_steps,
        }
    )


def _completed_record(
    binding: Mapping[str, object],
    *,
    started_record_sha256: str,
    summary_sha256: str,
    resource_sha256: str,
    primitive_trace_sha256: str,
    final_state_sha256: str,
    executed_steps: int,
) -> dict[str, object]:
    case_binding_sha256 = canonical_sha256(binding)
    outcome_sha256 = _component_outcome_sha256(
        case_binding_sha256=case_binding_sha256,
        started_record_sha256=started_record_sha256,
        summary_sha256=summary_sha256,
        resource_sha256=resource_sha256,
        primitive_trace_sha256=primitive_trace_sha256,
        final_state_sha256=final_state_sha256,
        executed_steps=executed_steps,
    )
    body = {
        "schema": CALIBRATION_EXECUTION_COMPLETED_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_state": "learner_execution_completed",
        "case_binding": dict(binding),
        "case_binding_sha256": case_binding_sha256,
        "started_record_sha256": started_record_sha256,
        "summary_digest_schema": CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
        "summary_sha256": summary_sha256,
        "resource_digest_schema": CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
        "resource_sha256": resource_sha256,
        "primitive_trace_digest_schema": CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
        "primitive_trace_sha256": primitive_trace_sha256,
        "final_state_digest_schema": CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA,
        "final_state_sha256": final_state_sha256,
        "outcome_digest_schema": CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
        "outcome_sha256": outcome_sha256,
        "executed_steps": executed_steps,
        "raw_trace_persisted_by_governance": False,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    return _payload_with_digest(body, "completed_record_sha256")


def require_valid_calibration_execution_completed_record(
    payload: Mapping[str, object],
    *,
    expected_started: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Strictly validate one deterministic learner-completion record."""

    body = _validate_payload_digest(
        payload,
        digest_field="completed_record_sha256",
        label="completed record",
    )
    _exact_keys(
        body,
        {
            "schema",
            "development_only",
            "scientific_promotion_allowed",
            "execution_state",
            "case_binding",
            "case_binding_sha256",
            "started_record_sha256",
            "summary_digest_schema",
            "summary_sha256",
            "resource_digest_schema",
            "resource_sha256",
            "primitive_trace_digest_schema",
            "primitive_trace_sha256",
            "final_state_digest_schema",
            "final_state_sha256",
            "outcome_digest_schema",
            "outcome_sha256",
            "executed_steps",
            "raw_trace_persisted_by_governance",
            "managed_boundary_scope",
        },
        "completed record",
    )
    _require(body["schema"] == CALIBRATION_EXECUTION_COMPLETED_SCHEMA, "completion schema differs")
    _require(body["development_only"] is True, "completion is not development-only")
    _require(body["scientific_promotion_allowed"] is False, "completion allows promotion")
    _require(body["execution_state"] == "learner_execution_completed", "completion state differs")
    binding = _require_case_binding(body["case_binding"])
    _require(
        body["case_binding_sha256"] == canonical_sha256(binding),
        "completion case binding digest differs",
    )
    started_digest = _strict_sha256(body["started_record_sha256"], "started_record_sha256")
    summary_digest = _strict_sha256(body["summary_sha256"], "summary_sha256")
    resource_digest = _strict_sha256(body["resource_sha256"], "resource_sha256")
    trace_digest = _strict_sha256(body["primitive_trace_sha256"], "primitive_trace_sha256")
    final_state_digest = _strict_sha256(body["final_state_sha256"], "final_state_sha256")
    _strict_sha256(body["outcome_sha256"], "outcome_sha256")
    _require(
        body["summary_digest_schema"] == CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
        "completion summary digest schema differs",
    )
    _require(
        body["resource_digest_schema"] == CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
        "completion resource digest schema differs",
    )
    _require(
        body["primitive_trace_digest_schema"]
        == CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
        "completion primitive-trace digest schema differs",
    )
    _require(
        body["final_state_digest_schema"] == CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA,
        "completion final-state digest schema differs",
    )
    _require(
        body["outcome_digest_schema"] == CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA,
        "completion outcome digest schema differs",
    )
    executed_steps = _strict_int(body["executed_steps"], "executed_steps", maximum=10**9)
    _require(body["raw_trace_persisted_by_governance"] is False, "completion persists raw trace")
    if expected_started is not None:
        started = require_valid_calibration_execution_started_record(expected_started)
        _require(
            started_digest == started["started_record_sha256"],
            "completion does not bind the supplied started record",
        )
        _require(binding == started["case_binding"], "completion and started bindings differ")
    expected = _completed_record(
        binding,
        started_record_sha256=started_digest,
        summary_sha256=summary_digest,
        resource_sha256=resource_digest,
        primitive_trace_sha256=trace_digest,
        final_state_sha256=final_state_digest,
        executed_steps=executed_steps,
    )
    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    _require(normalized == expected, "completed record is not deterministic")
    return expected


def snapshot_calibration_execution_inventory(directory: Path) -> dict[str, object]:
    """Read and strictly validate the deterministic per-case ledger inventory."""

    genesis = _load_ledger_genesis(directory)
    cases_directory = directory.absolute() / "cases"
    expected_names = [f"{_CASE_DIRECTORY_PREFIX}{index:03d}" for index in range(N_MATCHED_CASES)]
    actual_names = _directory_members(cases_directory, "execution cases directory")
    _require(actual_names == expected_names, "execution case directory inventory differs")
    started_records: list[dict[str, object]] = []
    completed_records: list[dict[str, object]] = []
    finalized_records: list[dict[str, object]] = []
    attempt_records: list[dict[str, object]] = []
    learner_interrupted: list[int] = []
    post_audit_unfinalized: list[int] = []
    for case_index, name in enumerate(expected_names):
        case_directory = cases_directory / name
        _directory_without_symlink(case_directory, f"execution case {case_index} directory")
        members = _validated_case_member_names(case_directory, case_index)
        started: dict[str, object] | None = None
        completed: dict[str, object] | None = None
        attempts: list[dict[str, object]] = []
        if _STARTED_FILE in members:
            started = require_valid_calibration_execution_started_record(
                _open_immutable_json(case_directory / _STARTED_FILE, "started record"),
                expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
            )
            binding = cast(dict[str, object], started["case_binding"])
            _require(
                binding["case_index"] == case_index,
                "started record is in the wrong case path",
            )
            started_records.append(
                {
                    "case_index": case_index,
                    "started_record_sha256": started["started_record_sha256"],
                    "case_binding_sha256": started["case_binding_sha256"],
                    "case_request_binding_sha256": binding[
                        "case_request_binding_sha256"
                    ],
                    "zip_provenance_binding_sha256": binding["zip_provenance_binding_sha256"],
                    "zip_provenance_attestation_sha256": binding[
                        "zip_provenance_attestation_sha256"
                    ],
                }
            )
            attempts = _case_attempt_rows(case_directory, started)
            attempt_records.append(_attempt_binding_from_rows(case_index, attempts))
        if _COMPLETED_FILE in members:
            assert started is not None
            completed = require_valid_calibration_execution_completed_record(
                _open_immutable_json(case_directory / _COMPLETED_FILE, "completed record"),
                expected_started=started,
            )
            completed_records.append(
                {
                    "case_index": case_index,
                    "completed_record_sha256": completed["completed_record_sha256"],
                    "started_record_sha256": completed["started_record_sha256"],
                    "summary_sha256": completed["summary_sha256"],
                    "resource_sha256": completed["resource_sha256"],
                    "primitive_trace_sha256": completed["primitive_trace_sha256"],
                    "final_state_sha256": completed["final_state_sha256"],
                    "outcome_sha256": completed["outcome_sha256"],
                }
            )
        elif started is not None:
            learner_interrupted.append(case_index)
        if _FINALIZED_FILE in members:
            assert started is not None and completed is not None
            finalized = require_valid_calibration_execution_finalized_record(
                _open_immutable_json(case_directory / _FINALIZED_FILE, "finalized record"),
                expected_started=started,
                expected_completed=completed,
                expected_attempt_rows=attempts,
            )
            finalized_records.append(
                {
                    "case_index": case_index,
                    "finalized_record_sha256": finalized["finalized_record_sha256"],
                    "started_record_sha256": finalized["started_record_sha256"],
                    "completed_record_sha256": finalized["completed_record_sha256"],
                    "managed_execution_attempt_count": finalized["managed_execution_attempt_count"],
                    "attempt_records_sha256": finalized["attempt_records_sha256"],
                    "shard_payload_sha256": finalized["shard_payload_sha256"],
                    "shard_canonical_sha256": finalized["shard_canonical_sha256"],
                    "trace_audit_report_sha256": finalized["trace_audit_report_sha256"],
                    "trace_audit_input_binding_sha256": finalized[
                        "trace_audit_input_binding_sha256"
                    ],
                    "final_state_sha256": finalized["final_state_sha256"],
                }
            )
        elif completed is not None:
            post_audit_unfinalized.append(case_index)
    started_indices = [cast(int, record["case_index"]) for record in started_records]
    completed_indices = [cast(int, record["case_index"]) for record in completed_records]
    finalized_indices = [cast(int, record["case_index"]) for record in finalized_records]
    body: dict[str, object] = {
        "schema": CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "genesis_sha256": genesis["genesis_sha256"],
        "expected_case_count": N_MATCHED_CASES,
        "started_case_indices": started_indices,
        "completed_case_indices": completed_indices,
        "finalized_case_indices": finalized_indices,
        "learner_interrupted_case_indices": learner_interrupted,
        "post_audit_unfinalized_case_indices": post_audit_unfinalized,
        "started_record_count": len(started_records),
        "completed_record_count": len(completed_records),
        "finalized_record_count": len(finalized_records),
        "managed_execution_attempt_count": sum(
            cast(int, record["managed_execution_attempt_count"]) for record in attempt_records
        ),
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": not started_records and not completed_records and not finalized_records,
        "started_records": started_records,
        "completed_records": completed_records,
        "finalized_records": finalized_records,
        "attempt_records": attempt_records,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
        "zip_provenance_policy": ZIP_PROVENANCE_POLICY,
        "shard_finalization_policy": SHARD_FINALIZATION_POLICY,
        "replay_accounting_policy": REPLAY_ACCOUNTING_POLICY,
    }
    return _payload_with_digest(body, "inventory_sha256")


def require_valid_calibration_execution_inventory(
    snapshot: Mapping[str, object],
    directory: Path,
) -> dict[str, object]:
    """Require a supplied inventory to equal a fresh strict ledger snapshot."""

    body = _validate_payload_digest(
        snapshot,
        digest_field="inventory_sha256",
        label="execution inventory",
    )
    _exact_keys(
        body,
        {
            "schema",
            "genesis_sha256",
            "expected_case_count",
            "started_case_indices",
            "completed_case_indices",
            "finalized_case_indices",
            "learner_interrupted_case_indices",
            "post_audit_unfinalized_case_indices",
            "started_record_count",
            "completed_record_count",
            "finalized_record_count",
            "managed_execution_attempt_count",
            "protected_started_record_count",
            "protected_completed_record_count",
            "pristine",
            "started_records",
            "completed_records",
            "finalized_records",
            "attempt_records",
            "managed_boundary_scope",
            "zip_provenance_policy",
            "shard_finalization_policy",
            "replay_accounting_policy",
        },
        "execution inventory",
    )
    current = snapshot_calibration_execution_inventory(directory)
    normalized = cast(dict[str, object], _normalized_json(dict(snapshot)))
    _require(normalized == current, "execution inventory snapshot is stale or tampered")
    return current


def _plain_mapping(value: object, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a plain object")
    return cast(dict[str, object], value)


def _plain_list(value: object, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a plain array")
    return cast(list[object], value)


def validate_completed_calibration_ledger_snapshot(
    snapshot: Mapping[str, object],
    shards_by_case: Mapping[int, Mapping[str, object]],
) -> dict[str, object]:
    """Join a canonical 240-case completed inventory to compact case shards.

    The caller first obtains the inventory through
    :func:`require_valid_calibration_execution_inventory`, which binds it to
    immutable files.  This second validator is directory-independent so the
    aggregate can bind that canonical inventory.  It reconstructs every
    started and completed record digest from shard request, frozen case,
    configuration, readiness, and exact outcome bindings.
    """

    body = _validate_payload_digest(
        snapshot,
        digest_field="inventory_sha256",
        label="completed execution inventory",
    )
    _exact_keys(
        body,
        {
            "schema",
            "genesis_sha256",
            "expected_case_count",
            "started_case_indices",
            "completed_case_indices",
            "finalized_case_indices",
            "learner_interrupted_case_indices",
            "post_audit_unfinalized_case_indices",
            "started_record_count",
            "completed_record_count",
            "finalized_record_count",
            "managed_execution_attempt_count",
            "protected_started_record_count",
            "protected_completed_record_count",
            "pristine",
            "started_records",
            "completed_records",
            "finalized_records",
            "attempt_records",
            "managed_boundary_scope",
            "zip_provenance_policy",
            "shard_finalization_policy",
            "replay_accounting_policy",
        },
        "completed execution inventory",
    )
    _require(
        body["schema"] == CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "completed inventory schema differs",
    )
    genesis_sha256 = _strict_sha256(body["genesis_sha256"], "inventory genesis")
    expected_indices = list(range(N_MATCHED_CASES))
    _require(body["expected_case_count"] == N_MATCHED_CASES, "inventory case count differs")
    _require(body["started_case_indices"] == expected_indices, "inventory starts are incomplete")
    _require(
        body["completed_case_indices"] == expected_indices,
        "inventory completions are incomplete",
    )
    _require(
        body["finalized_case_indices"] == expected_indices, "inventory finalizations are incomplete"
    )
    _require(
        body["learner_interrupted_case_indices"] == [],
        "inventory contains learner-interrupted cases",
    )
    _require(
        body["post_audit_unfinalized_case_indices"] == [],
        "inventory contains completed but unfinalized cases",
    )
    _require(body["started_record_count"] == N_MATCHED_CASES, "inventory start count differs")
    _require(
        body["completed_record_count"] == N_MATCHED_CASES,
        "inventory completion count differs",
    )
    _require(
        body["finalized_record_count"] == N_MATCHED_CASES,
        "inventory finalization count differs",
    )
    _strict_int(
        body["managed_execution_attempt_count"],
        "inventory managed execution attempt count",
        minimum=N_MATCHED_CASES,
        maximum=N_MATCHED_CASES * _MAX_ATTEMPTS_PER_CASE,
    )
    _require(
        body["protected_started_record_count"] == 0
        and body["protected_completed_record_count"] == 0,
        "inventory contains protected execution records",
    )
    _require(body["pristine"] is False, "completed inventory claims to be pristine")
    _require(
        body["managed_boundary_scope"] == MANAGED_EXECUTION_BOUNDARY_SCOPE,
        "inventory managed-boundary scope differs",
    )
    _require(body["zip_provenance_policy"] == ZIP_PROVENANCE_POLICY, "inventory ZIP policy differs")
    _require(
        body["shard_finalization_policy"] == SHARD_FINALIZATION_POLICY,
        "inventory finalization policy differs",
    )
    _require(
        body["replay_accounting_policy"] == REPLAY_ACCOUNTING_POLICY,
        "inventory replay policy differs",
    )
    _require(
        set(shards_by_case) == set(range(N_MATCHED_CASES))
        and all(type(index) is int for index in shards_by_case),
        "completed shard case index set differs",
    )

    started_items = tuple(
        _plain_mapping(item, "started inventory record")
        for item in _plain_list(body["started_records"], "started inventory records")
    )
    completed_items = tuple(
        _plain_mapping(item, "completed inventory record")
        for item in _plain_list(body["completed_records"], "completed inventory records")
    )
    finalized_items = tuple(
        _plain_mapping(item, "finalized inventory record")
        for item in _plain_list(body["finalized_records"], "finalized inventory records")
    )
    attempt_items = tuple(
        _plain_mapping(item, "attempt inventory record")
        for item in _plain_list(body["attempt_records"], "attempt inventory records")
    )
    _require(len(started_items) == N_MATCHED_CASES, "started inventory record count differs")
    _require(
        len(completed_items) == N_MATCHED_CASES,
        "completed inventory record count differs",
    )
    _require(len(finalized_items) == N_MATCHED_CASES, "finalized inventory record count differs")
    _require(len(attempt_items) == N_MATCHED_CASES, "attempt inventory record count differs")
    started_by_case: dict[int, dict[str, object]] = {}
    completed_by_case: dict[int, dict[str, object]] = {}
    finalized_by_case: dict[int, dict[str, object]] = {}
    attempt_by_case: dict[int, dict[str, object]] = {}
    for expected_index, item in enumerate(started_items):
        _exact_keys(
            item,
            {
                "case_index",
                "started_record_sha256",
                "case_binding_sha256",
                "case_request_binding_sha256",
                "zip_provenance_binding_sha256",
                "zip_provenance_attestation_sha256",
            },
            "started inventory record",
        )
        case_index = _strict_int(item["case_index"], "started case index", maximum=239)
        _require(case_index == expected_index, "started inventory records are not ordered")
        _strict_sha256(item["started_record_sha256"], "started record digest")
        _strict_sha256(item["case_binding_sha256"], "started case binding digest")
        _strict_sha256(
            item["case_request_binding_sha256"],
            "started case request binding digest",
        )
        _strict_sha256(item["zip_provenance_binding_sha256"], "started ZIP binding digest")
        _strict_sha256(item["zip_provenance_attestation_sha256"], "started ZIP attestation digest")
        started_by_case[case_index] = item
    for expected_index, item in enumerate(completed_items):
        _exact_keys(
            item,
            {
                "case_index",
                "completed_record_sha256",
                "started_record_sha256",
                "summary_sha256",
                "resource_sha256",
                "primitive_trace_sha256",
                "final_state_sha256",
                "outcome_sha256",
            },
            "completed inventory record",
        )
        case_index = _strict_int(item["case_index"], "completed case index", maximum=239)
        _require(case_index == expected_index, "completed inventory records are not ordered")
        for field in (
            "completed_record_sha256",
            "started_record_sha256",
            "summary_sha256",
            "resource_sha256",
            "primitive_trace_sha256",
            "final_state_sha256",
            "outcome_sha256",
        ):
            _strict_sha256(item[field], f"completed inventory {field}")
        completed_by_case[case_index] = item
    for expected_index, item in enumerate(finalized_items):
        _exact_keys(
            item,
            {
                "case_index",
                "finalized_record_sha256",
                "started_record_sha256",
                "completed_record_sha256",
                "managed_execution_attempt_count",
                "attempt_records_sha256",
                "shard_payload_sha256",
                "shard_canonical_sha256",
                "trace_audit_report_sha256",
                "trace_audit_input_binding_sha256",
                "final_state_sha256",
            },
            "finalized inventory record",
        )
        case_index = _strict_int(item["case_index"], "finalized case index", maximum=239)
        _require(case_index == expected_index, "finalized inventory records are not ordered")
        for field in (
            "finalized_record_sha256",
            "started_record_sha256",
            "completed_record_sha256",
            "attempt_records_sha256",
            "shard_payload_sha256",
            "shard_canonical_sha256",
            "trace_audit_report_sha256",
            "trace_audit_input_binding_sha256",
            "final_state_sha256",
        ):
            _strict_sha256(item[field], f"finalized inventory {field}")
        _strict_int(
            item["managed_execution_attempt_count"],
            "finalized attempt count",
            minimum=1,
            maximum=_MAX_ATTEMPTS_PER_CASE,
        )
        finalized_by_case[case_index] = item
    for expected_index, item in enumerate(attempt_items):
        case_index = _strict_int(item.get("case_index"), "attempt case index", maximum=239)
        _require(case_index == expected_index, "attempt inventory records are not ordered")
        attempt_by_case[case_index] = calibration_case_attempt_binding(body, case_index)
    _require(
        sum(cast(int, item["managed_execution_attempt_count"]) for item in attempt_by_case.values())
        == body["managed_execution_attempt_count"],
        "inventory aggregate attempt count differs",
    )

    design = _frozen_design()
    for case_index in range(N_MATCHED_CASES):
        shard = _plain_mapping(shards_by_case[case_index], f"case shard {case_index}")
        _validate_payload_digest(
            shard,
            digest_field="payload_sha256",
            label=f"case shard {case_index}",
        )
        case_payload = _plain_mapping(shard.get("case"), f"case shard {case_index}.case")
        case = design.cases[case_index]
        _require(case_payload == case.to_payload(), "shard differs from its frozen case")
        _require(
            shard.get("protocol_payload_sha256") == CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "shard protocol digest differs",
        )
        _require(shard.get("seed_snapshot_sha256") == SEED_SNAPSHOT_SHA256, "shard seeds differ")
        request_binding_digest = _strict_sha256(
            shard.get("case_request_binding_sha256"),
            "shard case request binding digest",
        )
        configuration_digest = _strict_sha256(
            shard.get("configuration_sha256"),
            "shard configuration digest",
        )
        configuration_payload = _plain_mapping(
            shard.get("configuration"),
            "shard configuration",
        )
        _require(
            configuration_digest == canonical_sha256(configuration_payload),
            "shard configuration digest does not match its direct payload",
        )
        readiness = _plain_mapping(
            shard.get("readiness_binding"),
            "shard readiness binding",
        )
        receipt_digest = _strict_sha256(
            readiness.get("readiness_receipt_sha256"),
            "shard readiness receipt digest",
        )
        source_archive_digest = _strict_sha256(
            readiness.get("source_archive_sha256"),
            "shard source archive digest",
        )
        source_manifest_digest = _strict_sha256(
            readiness.get("source_manifest_sha256"),
            "shard source manifest digest",
        )
        runtime_digest = _strict_sha256(
            readiness.get("runtime_identity_sha256"),
            "shard runtime identity digest",
        )
        governance_binding = _plain_mapping(
            readiness.get(READINESS_EXECUTION_GOVERNANCE_FIELD),
            "shard execution governance binding",
        )
        _require(
            governance_binding.get("genesis_sha256") == genesis_sha256,
            "shard readiness belongs to another execution genesis",
        )
        _require(
            governance_binding.get("source_archive_sha256") == source_archive_digest
            and governance_binding.get("source_manifest_sha256") == source_manifest_digest
            and governance_binding.get("runtime_identity_sha256") == runtime_digest,
            "shard readiness source/runtime identities differ",
        )
        case_binding: dict[str, object] = {
            "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
            "genesis_sha256": genesis_sha256,
            "readiness_receipt_sha256": receipt_digest,
            "source_archive_sha256": source_archive_digest,
            "source_manifest_sha256": source_manifest_digest,
            "runtime_identity_sha256": runtime_digest,
            "zip_provenance_binding_sha256": started_by_case[case_index][
                "zip_provenance_binding_sha256"
            ],
            "zip_provenance_attestation_sha256": started_by_case[case_index][
                "zip_provenance_attestation_sha256"
            ],
            "case_index": case_index,
            "manifest_name": case.manifest_name,
            "manifest_payload_sha256": _manifest_digest(
                HIDDEN_REGIME_CALIBRATION_MANIFESTS[case.manifest_name]
            ),
            "configuration_sha256": configuration_digest,
            "case_request_binding_sha256": request_binding_digest,
            "condition": case.condition,
            "seed_pair": {
                "namespace": CONSUMED_CALIBRATION_NAMESPACE,
                "index": case.seed_index,
                "world_seed": case.world_seed,
                "learner_seed": case.learner_seed,
            },
        }
        normalized_binding = _require_case_binding(
            cast(dict[str, object], _normalized_json(case_binding))
        )
        initial_attempt = _plain_mapping(
            _plain_list(attempt_by_case[case_index]["attempts"], "case attempts")[0],
            "initial case attempt",
        )
        expected_started = _started_record(
            normalized_binding,
            attempt_request_payload_sha256=cast(
                str,
                initial_attempt["attempt_request_payload_sha256"],
            ),
            exact_replay_consent=cast(bool, initial_attempt["exact_replay_consent"]),
        )
        started_inventory = started_by_case[case_index]
        _require(
            started_inventory["case_binding_sha256"] == expected_started["case_binding_sha256"],
            "shard request/case binding differs from immutable start",
        )
        _require(
            started_inventory["case_request_binding_sha256"] == request_binding_digest,
            "shard case-request binding differs from immutable start",
        )
        _require(
            started_inventory["started_record_sha256"] == expected_started["started_record_sha256"],
            "shard request/case differs from immutable started record",
        )
        completed_inventory = completed_by_case[case_index]
        _require(
            completed_inventory["started_record_sha256"]
            == expected_started["started_record_sha256"],
            "completion does not join its immutable start",
        )
        summary_payload = _plain_mapping(shard.get("summary"), "shard summary")
        summary_digest = _strict_sha256(shard.get("summary_sha256"), "shard summary digest")
        _require(
            summary_digest == canonical_sha256(summary_payload),
            "shard summary digest does not match its direct payload",
        )
        resource_payload = _plain_mapping(shard.get("resource"), "shard resource")
        resource_digest = _strict_sha256(
            shard.get("resource_sha256"),
            "shard resource digest",
        )
        _require(
            resource_digest == canonical_sha256(resource_payload),
            "shard resource digest does not match its direct payload",
        )
        primitive_trace = _plain_mapping(
            shard.get("primitive_trace"),
            "shard primitive trace binding",
        )
        trace_digest = _strict_sha256(
            primitive_trace.get("sha256"),
            "shard primitive trace digest",
        )
        _require(
            completed_inventory["summary_sha256"] == summary_digest
            and completed_inventory["resource_sha256"] == resource_digest
            and completed_inventory["primitive_trace_sha256"] == trace_digest,
            "shard summary/resource/trace digests differ from immutable completion",
        )
        executed_steps = _strict_int(
            shard.get("executed_steps"),
            "shard executed steps",
            maximum=10**9,
        )
        expected_completed = _completed_record(
            normalized_binding,
            started_record_sha256=cast(str, expected_started["started_record_sha256"]),
            summary_sha256=summary_digest,
            resource_sha256=resource_digest,
            primitive_trace_sha256=trace_digest,
            final_state_sha256=cast(str, completed_inventory["final_state_sha256"]),
            executed_steps=executed_steps,
        )
        _require(
            completed_inventory["completed_record_sha256"]
            == expected_completed["completed_record_sha256"],
            "shard execution length/outcome differs from immutable completion",
        )
        execution_binding = _plain_mapping(
            shard.get("execution_record_binding"),
            "shard execution record binding",
        )
        attempt_binding = attempt_by_case[case_index]
        expected_execution_binding = {
            "case_index": case_index,
            "genesis_sha256": genesis_sha256,
            "started_record_sha256": expected_started["started_record_sha256"],
            "completed_record_sha256": expected_completed["completed_record_sha256"],
            "summary_sha256": summary_digest,
            "resource_sha256": resource_digest,
            "primitive_trace_sha256": trace_digest,
            "final_state_sha256": completed_inventory["final_state_sha256"],
            "outcome_sha256": expected_completed["outcome_sha256"],
            "managed_execution_attempt_count": attempt_binding["managed_execution_attempt_count"],
            "attempt_records_sha256": attempt_binding["attempt_records_sha256"],
            "zip_provenance_binding_sha256": normalized_binding["zip_provenance_binding_sha256"],
            "zip_provenance_attestation_sha256": normalized_binding[
                "zip_provenance_attestation_sha256"
            ],
        }
        _require(
            execution_binding == expected_execution_binding,
            "shard execution binding differs from immutable ledger",
        )
        finalized_inventory = finalized_by_case[case_index]
        _require(
            finalized_inventory["started_record_sha256"]
            == expected_started["started_record_sha256"]
            and finalized_inventory["completed_record_sha256"]
            == expected_completed["completed_record_sha256"],
            "finalization does not join immutable start/completion",
        )
        _require(
            finalized_inventory["final_state_sha256"]
            == completed_inventory["final_state_sha256"],
            "finalization does not join immutable final learner state",
        )
        _require(
            finalized_inventory["managed_execution_attempt_count"]
            == attempt_binding["managed_execution_attempt_count"]
            and finalized_inventory["attempt_records_sha256"]
            == attempt_binding["attempt_records_sha256"],
            "finalization does not join immutable attempt ledger",
        )
        _require(
            finalized_inventory["shard_payload_sha256"] == shard["payload_sha256"]
            and finalized_inventory["shard_canonical_sha256"] == canonical_sha256(shard),
            "finalization differs from exact final shard payload",
        )
        audit_payload = _plain_mapping(shard.get("audit"), "shard audit")
        _require(
            finalized_inventory["trace_audit_report_sha256"]
            == audit_payload.get("trace_audit_report_sha256"),
            "finalization differs from full trace-audit report digest",
        )
    return cast(dict[str, object], _normalized_json(dict(snapshot)))


@dataclass(frozen=True, slots=True)
class CalibrationExecutionAuthorization:
    """Process-sealed authorization for one exact managed case invocation."""

    payload: dict[str, object]
    seal: str
    zip_provenance_capability: CalibrationZipProvenanceCapability


@dataclass(frozen=True, slots=True)
class CalibrationZipProvenanceCapability:
    """Opaque identity-registered proof of an exact ZIP-only worker environment."""

    payload: dict[str, object]
    seal: str
    nonce: str

    def __reduce__(self) -> NoReturn:
        raise TypeError("ZIP provenance capabilities are process-local and nonserializable")

    def __copy__(self) -> NoReturn:
        raise TypeError("ZIP provenance capabilities cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        del memo
        raise TypeError("ZIP provenance capabilities cannot be copied")


@dataclass(frozen=True, slots=True)
class ManagedCalibrationExecutionTicket:
    """Process-sealed proof that a start record exists before learner execution."""

    ledger_directory: Path
    case_binding: dict[str, object]
    started_record: dict[str, object]
    execution_mode: ExecutionMode
    attempt_index: int
    attempt_record_sha256: str
    seal: str


@dataclass(frozen=True, slots=True)
class _CompletedRunCapability:
    """Process-local proof that this exact run object completed the latest attempt."""

    result: object
    payload: dict[str, object]
    seal: str


_ZIP_PROVENANCE_CAPABILITIES: dict[str, CalibrationZipProvenanceCapability] = {}
_COMPLETED_RUN_CAPABILITIES: dict[tuple[str, int], _CompletedRunCapability] = {}


def _process_instance_binding() -> dict[str, object]:
    """Identify this exact interpreter instance, including across POSIX fork."""

    return {
        "pid": os.getpid(),
        "process_start_nonce": _PROCESS_START_NONCE,
    }


def _seal(kind: str, payload: object) -> str:
    return hmac.new(
        _PROCESS_SEAL_KEY,
        kind.encode("ascii")
        + b"\0"
        + canonical_json_bytes(_process_instance_binding())
        + b"\0"
        + canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()


def _reset_process_local_capabilities_after_fork() -> None:
    """Re-key a fork child and discard every capability inherited from its parent."""

    global _PROCESS_SEAL_KEY, _PROCESS_START_NONCE
    _PROCESS_SEAL_KEY = secrets.token_bytes(32)
    _PROCESS_START_NONCE = secrets.token_hex(32)
    _ZIP_PROVENANCE_CAPABILITIES.clear()
    _COMPLETED_RUN_CAPABILITIES.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_process_local_capabilities_after_fork)


def _zip_provenance_binding(
    *,
    readiness_receipt_sha256: str,
    source_archive_sha256: str,
    source_manifest_sha256: str,
    runtime_identity_sha256: str,
) -> dict[str, object]:
    body = {
        "schema": CALIBRATION_ZIP_PROVENANCE_BINDING_SCHEMA,
        "readiness_receipt_sha256": readiness_receipt_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "zip_provenance_policy": ZIP_PROVENANCE_POLICY,
    }
    for field in (
        "readiness_receipt_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
        "runtime_identity_sha256",
    ):
        _strict_sha256(body[field], f"ZIP provenance binding.{field}")
    return _payload_with_digest(body, "zip_provenance_binding_sha256")


def _currently_loaded_zip_project_modules(archive_path: Path, archive: bytes) -> list[object]:
    absolute = archive_path.absolute()
    archive_prefix = f"{absolute.as_posix()}/"
    rows: list[object] = []
    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as source_zip:
            member_names = source_zip.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        raise HiddenRegimeExecutionGovernanceError("ZIP provenance archive is invalid") from error
    _require(len(member_names) == len(set(member_names)), "ZIP provenance archive has duplicates")
    members = set(member_names)
    for name, module in sorted(sys.modules.items()):
        if name != "alberta_framework" and not name.startswith("alberta_framework."):
            continue
        loader = getattr(module, "__loader__", None)
        origin = getattr(module, "__file__", None)
        _require(
            isinstance(loader, zipimport.zipimporter), f"project module is not zipimport: {name}"
        )
        _require(
            type(origin) is str and origin.startswith(archive_prefix),
            f"project module origin is outside exact source ZIP: {name}",
        )
        member = cast(str, origin)[len(archive_prefix) :]
        _require(member in members, f"project module origin is not a source ZIP member: {name}")
        rows.append({"module": name, "archive_member": member})
    _require(bool(rows), "ZIP provenance found no loaded project modules")
    return rows


def _verify_exact_zip_worker_environment(
    *,
    source_archive_path: Path,
    source_archive: bytes,
    expected_source_archive_sha256: str,
    runtime_python_binding: Mapping[str, object],
    runtime_path_policy: object,
) -> dict[str, object]:
    """Independently verify the live interpreter is rooted in one exact ZIP."""

    _strict_sha256(expected_source_archive_sha256, "expected source archive digest")
    absolute = source_archive_path.absolute()
    on_disk = _open_immutable_bytes(
        absolute,
        "ZIP provenance source archive",
        max_bytes=_MAX_SOURCE_ARCHIVE_BYTES,
    )
    _require(on_disk == source_archive, "ZIP provenance archive bytes differ from supplied bytes")
    _require(
        hashlib.sha256(on_disk).hexdigest() == expected_source_archive_sha256,
        "ZIP provenance source archive digest differs from readiness",
    )
    _require(not os.listdir(Path.cwd()), "ZIP provenance working directory is not empty")
    _require(sys.dont_write_bytecode is True, "ZIP provenance bytecode writes are not disabled")
    pycache_value = sys.pycache_prefix
    _require(
        type(pycache_value) is str and bool(pycache_value) and os.path.isabs(pycache_value),
        "ZIP provenance command-line pycache prefix is missing",
    )
    xoptions = getattr(sys, "_xoptions", {})
    _require(
        type(xoptions) is dict and xoptions.get("pycache_prefix") == pycache_value,
        "ZIP provenance pycache prefix was not supplied on the command line",
    )
    pycache_prefix = Path(cast(str, pycache_value)).absolute()
    _directory_without_symlink(pycache_prefix, "ZIP provenance pycache prefix")
    _require(
        not _directory_members(pycache_prefix, "ZIP provenance pycache prefix"),
        "ZIP provenance pycache prefix is not fresh and empty",
    )
    cwd = Path.cwd().absolute()
    readiness_directory = absolute.parent

    def overlaps(first: Path, second: Path) -> bool:
        common = Path(os.path.commonpath((first.as_posix(), second.as_posix())))
        return common == first or common == second

    _require(
        not overlaps(pycache_prefix, cwd)
        and not overlaps(pycache_prefix, readiness_directory)
        and not overlaps(pycache_prefix, absolute),
        "ZIP provenance pycache prefix overlaps a bound source or working path",
    )
    _require(sys.flags.no_site == 1, "ZIP provenance automatic site initialization is enabled")
    forbidden_site_modules = tuple(
        name
        for name in ("site", "sitecustomize", "usercustomize", "_virtualenv")
        if name in sys.modules
    )
    _require(
        not forbidden_site_modules,
        "ZIP provenance found a loaded site or .pth customization module",
    )
    expected_prefix = Path(
        _strict_string(runtime_python_binding.get("prefix"), "runtime prefix binding")
    ).absolute()
    expected_exec_prefix = Path(
        _strict_string(runtime_python_binding.get("exec_prefix"), "runtime exec-prefix binding")
    ).absolute()
    purelib = Path(
        _strict_string(runtime_python_binding.get("purelib"), "runtime purelib binding")
    ).absolute()
    platlib = Path(
        _strict_string(runtime_python_binding.get("platlib"), "runtime platlib binding")
    ).absolute()
    stdlib_search_raw = _plain_list(
        runtime_python_binding.get("no_site_stdlib_search_paths"),
        "runtime no-site stdlib search paths",
    )
    _require(
        all(type(path) is str and os.path.isabs(path) for path in stdlib_search_raw),
        "runtime no-site stdlib search paths are not absolute strings",
    )
    stdlib_search_paths = [
        Path(cast(str, path)).absolute().as_posix() for path in stdlib_search_raw
    ]
    _require(
        len(stdlib_search_paths) == len(set(stdlib_search_paths)),
        "runtime no-site stdlib search paths contain duplicates",
    )
    _require(
        Path(sys.prefix).absolute() == expected_prefix,
        "ZIP provenance runtime prefix differs from readiness",
    )
    _require(
        Path(sys.exec_prefix).absolute() == expected_exec_prefix,
        "ZIP provenance runtime exec-prefix differs from readiness",
    )
    expected_dependency_paths = list(dict.fromkeys((purelib.as_posix(), platlib.as_posix())))
    for index, path in enumerate((purelib, platlib)):
        _directory_without_symlink(path, f"ZIP provenance dependency path {index}")
    expected_search_paths = [
        absolute.as_posix(),
        *stdlib_search_paths,
        *expected_dependency_paths,
    ]
    _require(all(type(entry) is str and bool(entry) for entry in sys.path), "sys.path differs")
    actual_search_paths = [Path(entry).absolute().as_posix() for entry in sys.path]
    _require(
        actual_search_paths == expected_search_paths,
        "ZIP provenance exact source/stdlib/dependency search paths differ from readiness",
    )
    runtime_path_policy_sha256 = canonical_sha256(runtime_path_policy)
    _require(bool(sys.path), "ZIP provenance sys.path is empty")
    _require(
        Path(sys.path[0]).absolute() == absolute,
        "ZIP provenance source archive is not sys.path[0]",
    )
    mutable_project_paths: list[str] = []
    for entry in sys.path[1:]:
        if not entry:
            continue
        candidate = Path(entry)
        if candidate.is_dir() and (candidate / "alberta_framework").is_dir():
            mutable_project_paths.append(candidate.absolute().as_posix())
    _require(
        not mutable_project_paths,
        "ZIP provenance found a mutable project source path",
    )
    module_rows = _currently_loaded_zip_project_modules(absolute, on_disk)
    return _canonical_zip_provenance_environment(
        source_archive_path=absolute,
        expected_source_archive_sha256=expected_source_archive_sha256,
        expected_prefix=expected_prefix,
        expected_exec_prefix=expected_exec_prefix,
        expected_dependency_paths=expected_dependency_paths,
        stdlib_search_paths=stdlib_search_paths,
        exact_runtime_search_paths=expected_search_paths,
        runtime_path_policy_sha256=runtime_path_policy_sha256,
        module_rows=module_rows,
    )


def _canonical_zip_provenance_environment(
    *,
    source_archive_path: Path,
    expected_source_archive_sha256: str,
    expected_prefix: Path,
    expected_exec_prefix: Path,
    expected_dependency_paths: list[str],
    stdlib_search_paths: list[str],
    exact_runtime_search_paths: list[str],
    runtime_path_policy_sha256: str,
    module_rows: list[object],
) -> dict[str, object]:
    """Project a physically verified worker environment without its staging locator."""

    _strict_sha256(expected_source_archive_sha256, "expected source archive digest")
    _strict_sha256(runtime_path_policy_sha256, "runtime path policy digest")
    physical_archive_path = source_archive_path.absolute().as_posix()
    _require(
        exact_runtime_search_paths
        == [physical_archive_path, *stdlib_search_paths, *expected_dependency_paths],
        "ZIP provenance canonical projection differs from verified search paths",
    )
    canonical_search_paths: list[dict[str, str]] = [
        {
            "kind": "source_archive",
            "locator": ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
        },
        *({"kind": "stdlib", "path": path} for path in stdlib_search_paths),
        *({"kind": "dependency", "path": path} for path in expected_dependency_paths),
    ]
    return {
        "source_archive_locator": ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
        "source_archive_sha256": expected_source_archive_sha256,
        "source_zip_first": True,
        "sole_project_source_path": True,
        "fresh_empty_working_directory": True,
        "dont_write_bytecode": True,
        "command_line_pycache_prefix": True,
        "pycache_prefix_fresh_empty_nonsymlink": True,
        "pycache_prefix_outside_bound_paths": True,
        "no_site": True,
        "site_or_pth_customization_module_count": 0,
        "runtime_prefix_sha256": canonical_sha256(expected_prefix.absolute().as_posix()),
        "runtime_exec_prefix_sha256": canonical_sha256(
            expected_exec_prefix.absolute().as_posix()
        ),
        "dependency_search_paths_sha256": canonical_sha256(expected_dependency_paths),
        "no_site_stdlib_search_paths_sha256": canonical_sha256(stdlib_search_paths),
        "canonical_runtime_search_paths_sha256": canonical_sha256(canonical_search_paths),
        "runtime_path_policy_sha256": runtime_path_policy_sha256,
        "project_module_loader": "zipimport.zipimporter",
        "project_module_count": len(module_rows),
        "project_modules_sha256": canonical_sha256(module_rows),
    }


def _zip_provenance_attestation_payload(
    *,
    binding: Mapping[str, object],
    environment: Mapping[str, object],
) -> dict[str, object]:
    return _payload_with_digest(
        {
            "schema": CALIBRATION_ZIP_PROVENANCE_SCHEMA,
            "binding": binding,
            "environment": environment,
            "zip_provenance_policy": ZIP_PROVENANCE_POLICY,
        },
        "zip_provenance_attestation_sha256",
    )


def attest_calibration_zip_provenance(
    *,
    readiness_bundle: object,
    readiness_source_archive: bytes,
    source_archive_path: Path,
) -> CalibrationZipProvenanceCapability:
    """Mint an opaque capability only after exact live ZIP provenance validation."""

    validated = _validate_readiness_bundle(readiness_bundle, readiness_source_archive)
    receipt_body = _plain_mapping(
        cast(Any, validated).payload.get("body"),
        "readiness body",
    )
    runtime_identity = _plain_mapping(
        receipt_body.get("runtime_identity"),
        "readiness runtime identity",
    )
    runtime_python = _plain_mapping(
        runtime_identity.get("python"),
        "readiness runtime Python binding",
    )
    worker_execution = _plain_mapping(
        receipt_body.get("worker_execution"),
        "readiness worker execution binding",
    )
    _require(worker_execution.get("no_site_flag") == "-S", "readiness does not require -S")
    runtime_path_policy = worker_execution.get("runtime_path_policy")
    _require(runtime_path_policy is not None, "readiness has no runtime path policy")
    environment = _verify_exact_zip_worker_environment(
        source_archive_path=source_archive_path,
        source_archive=readiness_source_archive,
        expected_source_archive_sha256=cast(Any, validated).source_archive_sha256,
        runtime_python_binding=runtime_python,
        runtime_path_policy=runtime_path_policy,
    )
    binding = _zip_provenance_binding(
        readiness_receipt_sha256=cast(Any, validated).receipt_sha256,
        source_archive_sha256=cast(Any, validated).source_archive_sha256,
        source_manifest_sha256=cast(Any, validated).source_manifest_sha256,
        runtime_identity_sha256=cast(Any, validated).runtime_identity_sha256,
    )
    payload = _zip_provenance_attestation_payload(binding=binding, environment=environment)
    nonce = secrets.token_hex(32)
    capability = CalibrationZipProvenanceCapability(
        payload=payload,
        seal=_seal("calibration-zip-provenance-capability-v1", {"nonce": nonce, **payload}),
        nonce=nonce,
    )
    _ZIP_PROVENANCE_CAPABILITIES[nonce] = capability
    return capability


def _require_zip_provenance_capability(
    capability: object,
    *,
    readiness_bundle: object | None = None,
) -> CalibrationZipProvenanceCapability:
    _require(
        type(capability) is CalibrationZipProvenanceCapability,
        "execution authorization requires a ZIP provenance capability",
    )
    result = cast(CalibrationZipProvenanceCapability, capability)
    _require(
        _ZIP_PROVENANCE_CAPABILITIES.get(result.nonce) is result,
        "ZIP provenance capability is forged, copied, or from another process",
    )
    _require(
        hmac.compare_digest(
            result.seal,
            _seal(
                "calibration-zip-provenance-capability-v1",
                {"nonce": result.nonce, **result.payload},
            ),
        ),
        "ZIP provenance capability seal is invalid",
    )
    body = _validate_payload_digest(
        result.payload,
        digest_field="zip_provenance_attestation_sha256",
        label="ZIP provenance attestation",
    )
    _exact_keys(
        body,
        {"schema", "binding", "environment", "zip_provenance_policy"},
        "ZIP provenance attestation",
    )
    _require(body["schema"] == CALIBRATION_ZIP_PROVENANCE_SCHEMA, "ZIP provenance schema differs")
    _require(
        body["zip_provenance_policy"] == ZIP_PROVENANCE_POLICY, "ZIP provenance policy differs"
    )
    binding = _plain_mapping(body["binding"], "ZIP provenance binding")
    binding_body = _validate_payload_digest(
        binding,
        digest_field="zip_provenance_binding_sha256",
        label="ZIP provenance binding",
    )
    _exact_keys(
        binding_body,
        {
            "schema",
            "readiness_receipt_sha256",
            "source_archive_sha256",
            "source_manifest_sha256",
            "runtime_identity_sha256",
            "zip_provenance_policy",
        },
        "ZIP provenance binding",
    )
    expected = _zip_provenance_binding(
        readiness_receipt_sha256=cast(str, binding_body["readiness_receipt_sha256"]),
        source_archive_sha256=cast(str, binding_body["source_archive_sha256"]),
        source_manifest_sha256=cast(str, binding_body["source_manifest_sha256"]),
        runtime_identity_sha256=cast(str, binding_body["runtime_identity_sha256"]),
    )
    _require(binding == expected, "ZIP provenance binding is not deterministic")
    environment = _plain_mapping(body["environment"], "ZIP provenance environment")
    _require(
        environment.get("source_archive_locator") == ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR,
        "ZIP provenance source archive locator differs",
    )
    _require(
        "source_archive_path" not in environment,
        "ZIP provenance environment leaks a physical source archive path",
    )
    _strict_sha256(
        environment.get("canonical_runtime_search_paths_sha256"),
        "ZIP canonical runtime search-path digest",
    )
    _strict_sha256(environment.get("project_modules_sha256"), "ZIP module inventory digest")
    if readiness_bundle is not None:
        _require(
            expected["readiness_receipt_sha256"]
            == getattr(readiness_bundle, "receipt_sha256", None)
            and expected["source_archive_sha256"]
            == getattr(readiness_bundle, "source_archive_sha256", None)
            and expected["source_manifest_sha256"]
            == getattr(readiness_bundle, "source_manifest_sha256", None)
            and expected["runtime_identity_sha256"]
            == getattr(readiness_bundle, "runtime_identity_sha256", None),
            "ZIP provenance capability belongs to another readiness receipt",
        )
    return result


def _config_sha256(config: object) -> str:
    return calibration_execution_configuration_sha256(config)


def _expected_case_config(case: object) -> object:
    # Lazy import avoids making the development evaluator depend on governance
    # while its own module is still initializing.
    from alberta_framework.evaluation.hidden_regime_signaling_development import (
        HiddenRegimeDevelopmentConfig,
    )

    design = _frozen_design()
    base = design.base_config_binding
    manifest = HIDDEN_REGIME_CALIBRATION_MANIFESTS[cast(Any, case).manifest_name]
    learner = SlotSignalingConfig(
        learning_rate=float(base.learning_rate_decimal),
        epsilon=float(base.epsilon_decimal),
        relevance_rate=float(base.relevance_rate_decimal),
        lease_length=base.lease_length,
        confirmation_steps=base.confirmation_steps,
        durable_retrieval_threshold=float(base.durable_retrieval_threshold_decimal),
        candidate_confirmation_threshold=float(base.candidate_confirmation_threshold_decimal),
        candidate_confirmation_leases=base.candidate_confirmation_leases,
        scratch_training_leases_before_retest=base.scratch_training_leases_before_retest,
        writable_lru_ablation=base.writable_lru_ablation,
        durable_write_policy=base.requested_durable_write_policy,
        replacement_target_policy=base.requested_replacement_target_policy,
    )
    return HiddenRegimeDevelopmentConfig(
        world=manifest.to_world_config(repeat_schedule=False),
        learner=learner,
        metric_window=base.metric_window,
    )


def _validate_readiness_bundle(bundle: object, source_archive: bytes) -> object:
    # Revalidate the receipt and archive here rather than trusting the publicly
    # constructible ValidatedReadinessBundle dataclass alone.
    from alberta_framework.evaluation.hidden_regime_calibration_readiness import (
        ValidatedReadinessBundle,
        require_validated_readiness_receipt,
    )

    _require(type(bundle) is ValidatedReadinessBundle, "readiness bundle has the wrong type")
    _require(type(source_archive) is bytes, "readiness source archive must be exact bytes")
    validated = require_validated_readiness_receipt(
        cast(Any, bundle).payload,
        source_archive,
        recheck_current=False,
        recheck_runtime=True,
    )
    _require(validated == bundle, "readiness bundle differs after strict revalidation")
    body = cast(dict[str, object], cast(Any, validated).payload["body"])
    authorization = cast(dict[str, object], body.get("authorization"))
    _require(type(authorization) is dict, "readiness receipt has no authorization object")
    _require(
        authorization.get("ready_for_calibration") is True,
        "readiness receipt is not ready for calibration",
    )
    _require(
        authorization.get("protected_candidate_execution_permitted") is False,
        "readiness receipt does not forbid protected execution",
    )
    return validated


def _binding_from_inputs(
    *,
    ledger_directory: Path,
    genesis: Mapping[str, object],
    readiness_bundle: object,
    case_index: int,
    condition: object,
    seed_pair: object,
    config: object,
    case_request_binding_sha256: str,
    zip_provenance_capability: CalibrationZipProvenanceCapability,
) -> dict[str, object]:
    design = _frozen_design()
    _strict_int(case_index, "case_index", maximum=N_MATCHED_CASES - 1)
    case = design.cases[case_index]
    _require(condition == case.condition, "condition differs from the frozen case")
    expected_config = _expected_case_config(case)
    _require(config == expected_config, "configuration differs from the frozen case")
    classification = classify_hidden_regime_world(cast(Any, config).world)
    _require(classification.sensitivity == "calibration", "case world is not calibration-managed")
    _require(classification.manifest_name == case.manifest_name, "case world manifest differs")
    supplied_pair = {
        "namespace": getattr(seed_pair, "namespace", None),
        "index": getattr(seed_pair, "index", None),
        "world_seed": getattr(seed_pair, "world_seed", None),
        "learner_seed": getattr(seed_pair, "learner_seed", None),
    }
    expected_pair = {
        "namespace": CONSUMED_CALIBRATION_NAMESPACE,
        "index": case.seed_index,
        "world_seed": case.world_seed,
        "learner_seed": case.learner_seed,
    }
    _require(supplied_pair == expected_pair, "seed pair differs from the frozen case")
    receipt_payload = cast(dict[str, object], cast(Any, readiness_bundle).payload)
    readiness_body = cast(dict[str, object], receipt_payload["body"])
    expected_governance = calibration_execution_genesis_receipt_binding(genesis)
    _require(
        readiness_body.get(READINESS_EXECUTION_GOVERNANCE_FIELD) == expected_governance,
        "readiness receipt is not bound to this pristine execution genesis",
    )
    source_snapshot = cast(dict[str, object], readiness_body["source_snapshot"])
    archive_binding = cast(dict[str, object], source_snapshot["archive"])
    provenance_body = _validate_payload_digest(
        zip_provenance_capability.payload,
        digest_field="zip_provenance_attestation_sha256",
        label="ZIP provenance attestation",
    )
    provenance_binding = _plain_mapping(
        provenance_body["binding"],
        "ZIP provenance binding",
    )
    binding: dict[str, object] = {
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "genesis_sha256": genesis["genesis_sha256"],
        "readiness_receipt_sha256": cast(Any, readiness_bundle).receipt_sha256,
        "source_archive_sha256": cast(Any, readiness_bundle).source_archive_sha256,
        "source_manifest_sha256": cast(Any, readiness_bundle).source_manifest_sha256,
        "runtime_identity_sha256": cast(Any, readiness_bundle).runtime_identity_sha256,
        "zip_provenance_binding_sha256": provenance_binding["zip_provenance_binding_sha256"],
        "zip_provenance_attestation_sha256": zip_provenance_capability.payload[
            "zip_provenance_attestation_sha256"
        ],
        "case_index": case_index,
        "manifest_name": case.manifest_name,
        "manifest_payload_sha256": classification.manifest_payload_sha256,
        "configuration_sha256": _config_sha256(config),
        "case_request_binding_sha256": case_request_binding_sha256,
        "condition": case.condition,
        "seed_pair": supplied_pair,
    }
    _require(
        archive_binding.get("sha256") == binding["source_archive_sha256"],
        "readiness source archive identity differs",
    )
    _require(
        genesis["source_archive_sha256"] == binding["source_archive_sha256"]
        and genesis["source_manifest_sha256"] == binding["source_manifest_sha256"]
        and genesis["runtime_identity_sha256"] == binding["runtime_identity_sha256"],
        "genesis source/runtime identities differ from readiness",
    )
    _require_case_binding(cast(dict[str, object], _normalized_json(binding)))
    _require(
        ledger_directory.absolute().name == genesis["genesis_sha256"],
        "authorization ledger path differs from genesis content address",
    )
    return cast(dict[str, object], _normalized_json(binding))


def issue_calibration_execution_authorization(
    *,
    ledger_directory: Path,
    readiness_bundle: object,
    readiness_source_archive: bytes,
    zip_provenance_capability: object,
    case_index: int,
    condition: object,
    seed_pair: object,
    config: object,
    case_request_binding_sha256: str,
    attempt_request_payload_sha256: str,
    explicit_acknowledgement: str,
    allow_exact_replay: bool = False,
) -> CalibrationExecutionAuthorization:
    """Issue a process-local seal for one exact frozen calibration case."""

    _require(
        explicit_acknowledgement == EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        "exact managed calibration execution acknowledgement is required",
    )
    _require(type(allow_exact_replay) is bool, "allow_exact_replay must be a strict boolean")
    _strict_sha256(case_request_binding_sha256, "case_request_binding_sha256")
    _strict_sha256(attempt_request_payload_sha256, "attempt_request_payload_sha256")
    validated = _validate_readiness_bundle(readiness_bundle, readiness_source_archive)
    checked_provenance = _require_zip_provenance_capability(
        zip_provenance_capability,
        readiness_bundle=validated,
    )
    genesis = _load_ledger_genesis(ledger_directory)
    binding = _binding_from_inputs(
        ledger_directory=ledger_directory,
        genesis=genesis,
        readiness_bundle=validated,
        case_index=case_index,
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        case_request_binding_sha256=case_request_binding_sha256,
        zip_provenance_capability=checked_provenance,
    )
    case_directory = _case_directory(ledger_directory, case_index)
    started_path = case_directory / _STARTED_FILE
    completed_path = case_directory / _COMPLETED_FILE
    case_members = _validated_case_member_names(case_directory, case_index)
    prior_started: str | None = None
    prior_completed: str | None = None
    prior_attempt_rows: list[dict[str, object]] = []
    provenance_attestation_sha256 = cast(
        str,
        checked_provenance.payload["zip_provenance_attestation_sha256"],
    )
    if _STARTED_FILE in case_members:
        started = require_valid_calibration_execution_started_record(
            _open_immutable_json(started_path, "started record"),
            expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
        )
        _require(started["case_binding"] == binding, "consumed case binding differs")
        prior_attempt_rows = _case_attempt_rows(case_directory, started)
        prior_started = cast(str, started["started_record_sha256"])
        if _COMPLETED_FILE in case_members:
            completed = require_valid_calibration_execution_completed_record(
                _open_immutable_json(completed_path, "completed record"),
                expected_started=started,
            )
            prior_completed = cast(str, completed["completed_record_sha256"])
            mode: ExecutionMode = "exact_replay_after_completion"
        else:
            mode = "exact_replay_after_interruption"
        if not allow_exact_replay:
            raise HiddenRegimeCaseConsumedError(
                "calibration case is consumed; only explicit exact replay/resume is permitted"
            )
        _require(
            _FINALIZED_FILE not in case_members,
            "finalized calibration cases cannot be replayed",
        )
    else:
        _require(_COMPLETED_FILE not in case_members, "completion exists without a start record")
        _require(_FINALIZED_FILE not in case_members, "finalization exists without a start record")
        mode = "first_execution"
    attempt_index = len(prior_attempt_rows) if prior_attempt_rows else 0
    _require(attempt_index < _MAX_ATTEMPTS_PER_CASE, "case replay attempt limit reached")
    prior_attempt_records_sha256 = canonical_sha256(prior_attempt_rows)
    payload: dict[str, object] = {
        "schema": CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA,
        "ledger_directory": ledger_directory.absolute().as_posix(),
        "case_binding": binding,
        "case_binding_sha256": canonical_sha256(binding),
        "execution_mode": mode,
        "attempt_index": attempt_index,
        "prior_attempt_records_sha256": prior_attempt_records_sha256,
        "prior_started_record_sha256": prior_started,
        "prior_completed_record_sha256": prior_completed,
        "zip_provenance_binding_sha256": binding["zip_provenance_binding_sha256"],
        "zip_provenance_attestation_sha256": provenance_attestation_sha256,
        "attempt_request_payload_sha256": attempt_request_payload_sha256,
        "exact_replay_consent": allow_exact_replay,
        "explicit_acknowledgement": explicit_acknowledgement,
        "authorization_scope": PROCESS_LOCAL_AUTHORIZATION_SCOPE,
        "crash_consumption_rule": CRASH_CONSUMPTION_RULE,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    normalized = cast(dict[str, object], _normalized_json(payload))
    return CalibrationExecutionAuthorization(
        payload=normalized,
        seal=_seal("calibration-execution-authorization-v3", normalized),
        zip_provenance_capability=checked_provenance,
    )


def _require_authorization(
    authorization: object,
    *,
    condition: object,
    seed_pair: object,
    config: object,
) -> CalibrationExecutionAuthorization:
    _require(
        type(authorization) is CalibrationExecutionAuthorization,
        "exact calibration worlds require a sealed execution authorization",
    )
    result = cast(CalibrationExecutionAuthorization, authorization)
    checked_provenance = _require_zip_provenance_capability(
        result.zip_provenance_capability,
    )
    _require(
        hmac.compare_digest(
            result.seal,
            _seal("calibration-execution-authorization-v3", result.payload),
        ),
        "calibration execution authorization seal is invalid",
    )
    payload = result.payload
    _exact_keys(
        payload,
        {
            "schema",
            "ledger_directory",
            "case_binding",
            "case_binding_sha256",
            "execution_mode",
            "attempt_index",
            "prior_attempt_records_sha256",
            "prior_started_record_sha256",
            "prior_completed_record_sha256",
            "zip_provenance_binding_sha256",
            "zip_provenance_attestation_sha256",
            "attempt_request_payload_sha256",
            "exact_replay_consent",
            "explicit_acknowledgement",
            "authorization_scope",
            "crash_consumption_rule",
            "managed_boundary_scope",
        },
        "execution authorization",
    )
    _require(
        payload["schema"] == CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA,
        "authorization schema differs",
    )
    _require(
        payload["authorization_scope"] == PROCESS_LOCAL_AUTHORIZATION_SCOPE,
        "authorization process-local scope differs",
    )
    binding = _require_case_binding(payload["case_binding"])
    _require(
        payload["case_binding_sha256"] == canonical_sha256(binding),
        "authorization case binding digest differs",
    )
    provenance_body = _validate_payload_digest(
        checked_provenance.payload,
        digest_field="zip_provenance_attestation_sha256",
        label="ZIP provenance attestation",
    )
    provenance_binding = _plain_mapping(
        provenance_body["binding"],
        "ZIP provenance binding",
    )
    _require(
        payload["zip_provenance_attestation_sha256"]
        == checked_provenance.payload["zip_provenance_attestation_sha256"]
        == binding["zip_provenance_attestation_sha256"],
        "authorization ZIP provenance attestation differs",
    )
    _require(
        payload["zip_provenance_binding_sha256"]
        == provenance_binding["zip_provenance_binding_sha256"]
        == binding["zip_provenance_binding_sha256"],
        "authorization ZIP provenance binding differs",
    )
    _strict_int(
        payload["attempt_index"],
        "authorization attempt index",
        maximum=_MAX_ATTEMPTS_PER_CASE - 1,
    )
    _strict_sha256(
        payload["prior_attempt_records_sha256"],
        "authorization prior attempt records digest",
    )
    _strict_sha256(
        payload["attempt_request_payload_sha256"],
        "authorization attempt request digest",
    )
    _require(type(payload["exact_replay_consent"]) is bool, "authorization consent differs")
    if payload["execution_mode"] != "first_execution":
        _require(payload["exact_replay_consent"] is True, "replay authorization lacks consent")
    _require(condition == binding["condition"], "authorization condition substitution")
    supplied_pair = {
        "namespace": getattr(seed_pair, "namespace", None),
        "index": getattr(seed_pair, "index", None),
        "world_seed": getattr(seed_pair, "world_seed", None),
        "learner_seed": getattr(seed_pair, "learner_seed", None),
    }
    _require(supplied_pair == binding["seed_pair"], "authorization seed substitution")
    _require(
        _config_sha256(config) == binding["configuration_sha256"],
        "authorization config substitution",
    )
    classification = classify_hidden_regime_world(cast(Any, config).world)
    _require(classification.sensitivity == "calibration", "authorization world substitution")
    _require(
        classification.manifest_name == binding["manifest_name"],
        "authorization manifest substitution",
    )
    _require(
        classification.manifest_payload_sha256 == binding["manifest_payload_sha256"],
        "authorization manifest digest substitution",
    )
    return result


def begin_managed_hidden_regime_execution(
    *,
    condition: object,
    seed_pair: object,
    config: object,
    authorization: object | None,
) -> ManagedCalibrationExecutionTicket | None:
    """Guard the main evaluator and consume a calibration case before execution."""

    world = getattr(config, "world", None)
    if not isinstance(world, HiddenRegimeWorldConfig):
        raise TypeError("config.world must be a HiddenRegimeWorldConfig")
    classification = classify_hidden_regime_world(world)
    if classification.sensitivity == "ordinary":
        _require(
            authorization is None,
            "a managed calibration authorization cannot be substituted onto an ordinary world",
        )
        return None
    if classification.sensitivity == "protected":
        raise HiddenRegimeProtectedExecutionError(
            "protected structural worlds and registered tail probes have no "
            "learner-execution issuer"
        )
    checked = _require_authorization(
        authorization,
        condition=condition,
        seed_pair=seed_pair,
        config=config,
    )
    payload = checked.payload
    ledger_directory = Path(_strict_string(payload["ledger_directory"], "ledger_directory"))
    genesis = _load_ledger_genesis(ledger_directory)
    binding = cast(dict[str, object], payload["case_binding"])
    _require(
        binding["genesis_sha256"] == genesis["genesis_sha256"],
        "authorization genesis differs",
    )
    case_index = cast(int, binding["case_index"])
    case_directory = _case_directory(ledger_directory, case_index)
    started_path = case_directory / _STARTED_FILE
    expected_started: dict[str, object] | None = None
    mode = cast(ExecutionMode, payload["execution_mode"])
    attempt_index = cast(int, payload["attempt_index"])
    _require(
        mode
        in {
            "first_execution",
            "exact_replay_after_interruption",
            "exact_replay_after_completion",
        },
        "authorization execution mode differs",
    )
    with _case_mutation_critical_section(case_directory, case_index) as case_fd:
        current_members = _validated_case_member_names(case_directory, case_index)
        if mode == "first_execution":
            expected_started = _started_record(
                binding,
                attempt_request_payload_sha256=cast(
                    str,
                    payload["attempt_request_payload_sha256"],
                ),
                exact_replay_consent=cast(bool, payload["exact_replay_consent"]),
            )
            _require(attempt_index == 0, "first execution attempt index differs")
            _require(
                payload["prior_attempt_records_sha256"] == canonical_sha256([]),
                "first execution has prior attempt records",
            )
            _require(
                payload["prior_started_record_sha256"] is None
                and payload["prior_completed_record_sha256"] is None,
                "first execution has prior record identities",
            )
            if current_members:
                raise HiddenRegimeCaseConsumedError(
                    "concurrent or stale first-execution authorization found a consumed case"
                )
            try:
                _write_new_immutable(case_fd, _STARTED_FILE, canonical_json_bytes(expected_started))
            except FileExistsError as error:
                raise HiddenRegimeCaseConsumedError(
                    "concurrent or stale first-execution authorization lost the atomic start race"
                ) from error
            os.fsync(case_fd)
            attempt_record_sha256 = cast(str, expected_started["started_record_sha256"])
        else:
            actual_started = require_valid_calibration_execution_started_record(
                _open_immutable_json(started_path, "started record"),
                expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
            )
            _require(actual_started["case_binding"] == binding, "replay start binding differs")
            expected_started = actual_started
            prior_attempt_rows = _case_attempt_rows(case_directory, actual_started)
            _require(
                attempt_index == len(prior_attempt_rows),
                "replay authorization attempt index is stale",
            )
            _require(
                payload["prior_attempt_records_sha256"] == canonical_sha256(prior_attempt_rows),
                "replay authorization prior attempt ledger is stale",
            )
            _require(
                payload["prior_started_record_sha256"] == actual_started["started_record_sha256"],
                "replay authorization start digest differs",
            )
            completed_path = case_directory / _COMPLETED_FILE
            if mode == "exact_replay_after_completion":
                actual_completed = require_valid_calibration_execution_completed_record(
                    _open_immutable_json(completed_path, "completed record"),
                    expected_started=actual_started,
                )
                _require(
                    payload["prior_completed_record_sha256"]
                    == actual_completed["completed_record_sha256"],
                    "replay authorization completion digest differs",
                )
            else:
                _require(
                    payload["prior_completed_record_sha256"] is None,
                    "interrupted replay authorization has a prior completion",
                )
                _require(
                    _COMPLETED_FILE not in current_members,
                    "interrupted replay case is already completed",
                )
            _require(
                _FINALIZED_FILE not in current_members,
                "finalized calibration cases cannot be replayed",
            )
            replay_record = _replay_started_record(
                binding,
                attempt_index=attempt_index,
                execution_mode=mode,
                prior_started_record_sha256=cast(str, actual_started["started_record_sha256"]),
                prior_completed_record_sha256=cast(
                    str | None, payload["prior_completed_record_sha256"]
                ),
                zip_provenance_attestation_sha256=cast(
                    str,
                    payload["zip_provenance_attestation_sha256"],
                ),
                attempt_request_payload_sha256=cast(
                    str,
                    payload["attempt_request_payload_sha256"],
                ),
                exact_replay_consent=cast(bool, payload["exact_replay_consent"]),
            )
            replay_name = f"replay-{attempt_index:06d}.json"
            try:
                _write_new_immutable(case_fd, replay_name, canonical_json_bytes(replay_record))
            except FileExistsError as error:
                raise HiddenRegimeCaseConsumedError(
                    "concurrent or stale replay authorization lost the atomic attempt race"
                ) from error
            os.fsync(case_fd)
            attempt_record_sha256 = cast(str, replay_record["replay_started_record_sha256"])
    assert expected_started is not None
    ticket_payload = {
        "ledger_directory": ledger_directory.absolute().as_posix(),
        "case_binding": binding,
        "started_record_sha256": expected_started["started_record_sha256"],
        "execution_mode": mode,
        "attempt_index": attempt_index,
        "attempt_record_sha256": attempt_record_sha256,
    }
    return ManagedCalibrationExecutionTicket(
        ledger_directory=ledger_directory.absolute(),
        case_binding=binding,
        started_record=expected_started,
        execution_mode=mode,
        attempt_index=attempt_index,
        attempt_record_sha256=attempt_record_sha256,
        seal=_seal("managed-calibration-execution-ticket-v2", ticket_payload),
    )


def _exact_json_value(value: object, *, label: str) -> object:
    """Match the calibration shard's exact float-hex JSON normalization."""

    if value is None or type(value) in (str, int, bool):
        return value
    if isinstance(value, (float, np.floating)) and not isinstance(value, bool):
        number = float(value)
        _require(math.isfinite(number), f"{label} contains a non-finite float")
        return number.hex()
    if isinstance(value, np.integer):
        return int(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _exact_json_value(dataclasses.asdict(value), label=label)
    if isinstance(value, (tuple, list)):
        return [
            _exact_json_value(item, label=f"{label}[{index}]") for index, item in enumerate(value)
        ]
    if type(value) is dict:
        result: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            _require(type(key) is str, f"{label} contains a non-string key")
            result[cast(str, key)] = _exact_json_value(
                item,
                label=f"{label}.{key}",
            )
        return result
    _fail(f"{label} contains unsupported type {type(value).__name__}")


def _result_payload(value: object, label: str) -> object:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
    return _exact_json_value(value, label=label)


def calibration_execution_summary_sha256(summary: object) -> str:
    """Hash the same exact float-hex summary payload persisted by a shard."""

    return canonical_sha256(_result_payload(summary, "summary"))


def calibration_execution_configuration_sha256(config: object) -> str:
    """Hash the same exact float-hex configuration payload persisted by a shard."""

    return canonical_sha256(_result_payload(config, "configuration"))


def calibration_execution_resource_sha256(resource: object) -> str:
    """Hash the same exact float-hex resource payload persisted by a shard."""

    return canonical_sha256(_result_payload(resource, "resource"))


def calibration_execution_primitive_trace_sha256(trace: object) -> str:
    """Hash trace schema, array metadata, and exact C-order bytes without persistence."""

    from alberta_framework.evaluation.hidden_regime_signaling_development import (
        HIDDEN_REGIME_TRACE_SCHEMA,
    )

    _require(
        dataclasses.is_dataclass(trace) and not isinstance(trace, type),
        "result trace is invalid",
    )
    trace_dataclass = cast(Any, trace)
    hasher = hashlib.sha256()
    hasher.update(CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(HIDDEN_REGIME_TRACE_SCHEMA.encode("ascii"))
    for field in dataclasses.fields(trace_dataclass):
        array = np.ascontiguousarray(np.asarray(getattr(trace_dataclass, field.name)))
        _require(array.dtype.kind != "O", f"trace.{field.name} has object dtype")
        header = {
            "field": field.name,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
        }
        hasher.update(canonical_json_bytes(header))
        hasher.update(b"\0")
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def calibration_execution_final_state_sha256(state: object) -> str:
    """Hash the exact final learner-state pytree structure, metadata, and bytes."""

    import jax

    leaves, tree_definition = jax.tree_util.tree_flatten(state)
    _require(bool(leaves), "result final learner state has no pytree leaves")
    hasher = hashlib.sha256()
    hasher.update(CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(
        canonical_json_bytes(
            {
                "leaf_count": len(leaves),
                "tree_definition": str(tree_definition),
            }
        )
    )
    hasher.update(b"\0")
    for index, leaf in enumerate(leaves):
        array = np.ascontiguousarray(np.asarray(leaf))
        _require(array.dtype.kind != "O", f"final_state leaf {index} has object dtype")
        hasher.update(
            canonical_json_bytes(
                {
                    "leaf_index": index,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "nbytes": int(array.nbytes),
                }
            )
        )
        hasher.update(b"\0")
        hasher.update(array.tobytes(order="C"))
    return hasher.hexdigest()


def _validated_result_component_digests(
    result: object,
    binding: Mapping[str, object],
) -> dict[str, object]:
    _require(getattr(result, "condition", None) == binding["condition"], "result condition differs")
    result_pair = getattr(result, "seed_pair", None)
    pair_payload = {
        "namespace": getattr(result_pair, "namespace", None),
        "index": getattr(result_pair, "index", None),
        "world_seed": getattr(result_pair, "world_seed", None),
        "learner_seed": getattr(result_pair, "learner_seed", None),
    }
    _require(pair_payload == binding["seed_pair"], "result seed pair differs")
    configuration_sha256 = _config_sha256(getattr(result, "config", None))
    _require(
        configuration_sha256 == binding["configuration_sha256"],
        "result configuration differs",
    )
    executed_steps = getattr(getattr(result, "summary", None), "num_steps", None)
    _require(type(executed_steps) is int and executed_steps > 0, "result step count is invalid")
    return {
        "configuration_sha256": configuration_sha256,
        "summary_sha256": calibration_execution_summary_sha256(
            getattr(result, "summary", None)
        ),
        "resource_sha256": calibration_execution_resource_sha256(
            getattr(result, "resource", None)
        ),
        "primitive_trace_sha256": calibration_execution_primitive_trace_sha256(
            getattr(result, "trace", None)
        ),
        "final_state_sha256": calibration_execution_final_state_sha256(
            getattr(result, "final_state", None)
        ),
        "executed_steps": executed_steps,
    }


def _register_completed_run_capability(
    result: object,
    ticket: ManagedCalibrationExecutionTicket,
    completed: Mapping[str, object],
    components: Mapping[str, object],
) -> None:
    payload: dict[str, object] = {
        "ledger_directory": ticket.ledger_directory.absolute().as_posix(),
        "case_index": ticket.case_binding["case_index"],
        "case_binding_sha256": canonical_sha256(ticket.case_binding),
        "started_record_sha256": ticket.started_record["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "execution_mode": ticket.execution_mode,
        "attempt_index": ticket.attempt_index,
        "attempt_record_sha256": ticket.attempt_record_sha256,
        "configuration_sha256": components["configuration_sha256"],
        "summary_sha256": components["summary_sha256"],
        "resource_sha256": components["resource_sha256"],
        "primitive_trace_sha256": components["primitive_trace_sha256"],
        "final_state_sha256": components["final_state_sha256"],
        "executed_steps": components["executed_steps"],
        "result_identity": id(result),
    }
    normalized = cast(dict[str, object], _normalized_json(payload))
    key = (
        ticket.ledger_directory.absolute().as_posix(),
        cast(int, ticket.case_binding["case_index"]),
    )
    _COMPLETED_RUN_CAPABILITIES[key] = _CompletedRunCapability(
        result=result,
        payload=normalized,
        seal=_seal("completed-calibration-run-capability-v1", normalized),
    )


def _require_completed_run_capability(
    result: object,
    *,
    ledger_directory: Path,
    binding: Mapping[str, object],
    completed: Mapping[str, object],
    attempt_rows: list[dict[str, object]],
    components: Mapping[str, object],
) -> _CompletedRunCapability:
    key = (ledger_directory.absolute().as_posix(), cast(int, binding["case_index"]))
    capability = _COMPLETED_RUN_CAPABILITIES.get(key)
    _require(
        capability is not None and capability.result is result,
        "finalization requires the exact process-local completed run object",
    )
    assert capability is not None
    _require(
        hmac.compare_digest(
            capability.seal,
            _seal("completed-calibration-run-capability-v1", capability.payload),
        ),
        "completed-run capability seal is invalid",
    )
    _require(bool(attempt_rows), "completed run has no managed attempt")
    latest_attempt = attempt_rows[-1]
    expected = {
        "ledger_directory": ledger_directory.absolute().as_posix(),
        "case_index": binding["case_index"],
        "case_binding_sha256": canonical_sha256(binding),
        "started_record_sha256": completed["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "execution_mode": latest_attempt["execution_mode"],
        "attempt_index": latest_attempt["attempt_index"],
        "attempt_record_sha256": latest_attempt["attempt_record_sha256"],
        "configuration_sha256": components["configuration_sha256"],
        "summary_sha256": components["summary_sha256"],
        "resource_sha256": components["resource_sha256"],
        "primitive_trace_sha256": components["primitive_trace_sha256"],
        "final_state_sha256": components["final_state_sha256"],
        "executed_steps": components["executed_steps"],
        "result_identity": id(result),
    }
    _require(capability.payload == expected, "completed-run capability is stale or differs")
    return capability


def complete_managed_hidden_regime_execution(
    ticket: ManagedCalibrationExecutionTicket | None,
    result: object,
) -> dict[str, object] | None:
    """Publish deterministic completion after a managed learner run returns."""

    if ticket is None:
        return None
    _require(type(ticket) is ManagedCalibrationExecutionTicket, "execution ticket type differs")
    ticket_payload = {
        "ledger_directory": ticket.ledger_directory.absolute().as_posix(),
        "case_binding": ticket.case_binding,
        "started_record_sha256": ticket.started_record["started_record_sha256"],
        "execution_mode": ticket.execution_mode,
        "attempt_index": ticket.attempt_index,
        "attempt_record_sha256": ticket.attempt_record_sha256,
    }
    _require(
        hmac.compare_digest(
            ticket.seal,
            _seal("managed-calibration-execution-ticket-v2", ticket_payload),
        ),
        "execution ticket seal is invalid",
    )
    started = require_valid_calibration_execution_started_record(ticket.started_record)
    binding = cast(dict[str, object], started["case_binding"])
    _require(binding == ticket.case_binding, "execution ticket case binding differs")
    components = _validated_result_component_digests(result, binding)
    case_index = cast(int, binding["case_index"])
    case_directory = _case_directory(ticket.ledger_directory, case_index)
    completed_path = case_directory / _COMPLETED_FILE
    with _case_mutation_critical_section(case_directory, case_index) as case_fd:
        completion_members = _validated_case_member_names(case_directory, case_index)
        _require(
            _FINALIZED_FILE not in completion_members,
            "a finalized case cannot accept a stale learner completion",
        )
        actual_started = require_valid_calibration_execution_started_record(
            _open_immutable_json(case_directory / _STARTED_FILE, "started record")
        )
        _require(actual_started == started, "execution ticket start differs from immutable start")
        attempt_rows = _case_attempt_rows(case_directory, actual_started)
        _require(
            len(attempt_rows) == ticket.attempt_index + 1,
            "execution ticket is not the latest immutable attempt",
        )
        _require(
            attempt_rows[ticket.attempt_index]["attempt_record_sha256"]
            == ticket.attempt_record_sha256,
            "execution ticket attempt record differs",
        )
        _require(
            attempt_rows[ticket.attempt_index]["execution_mode"] == ticket.execution_mode,
            "execution ticket attempt mode differs",
        )
        expected = _completed_record(
            binding,
            started_record_sha256=cast(str, started["started_record_sha256"]),
            summary_sha256=cast(str, components["summary_sha256"]),
            resource_sha256=cast(str, components["resource_sha256"]),
            primitive_trace_sha256=cast(str, components["primitive_trace_sha256"]),
            final_state_sha256=cast(str, components["final_state_sha256"]),
            executed_steps=cast(int, components["executed_steps"]),
        )
        if _COMPLETED_FILE in completion_members:
            existing = require_valid_calibration_execution_completed_record(
                _open_immutable_json(completed_path, "completed record"),
                expected_started=started,
            )
            _require(existing == expected, "exact replay outcome differs from immutable completion")
            completed = existing
        else:
            try:
                _write_new_immutable(case_fd, _COMPLETED_FILE, canonical_json_bytes(expected))
            except FileExistsError:
                existing = require_valid_calibration_execution_completed_record(
                    _open_immutable_json(completed_path, "completed record"),
                    expected_started=started,
                )
                _require(existing == expected, "concurrent replay completion outcome differs")
                completed = existing
            else:
                os.fsync(case_fd)
                completed = expected
        _register_completed_run_capability(result, ticket, completed, components)
        return completed


def _require_trace_audit_report_matches_shard(
    trace_audit_report: object,
    audit: Mapping[str, object],
    summary: Mapping[str, object],
) -> str:
    """Independently join the full in-memory audit to its compact shard projection."""

    from alberta_framework.evaluation.hidden_regime_trace_audit import (
        EVIDENCE_BOUNDARY,
        HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
        HiddenRegimeTraceAuditReport,
    )

    _require(
        type(trace_audit_report) is HiddenRegimeTraceAuditReport,
        "finalization requires the exact full trace-audit report type",
    )
    report = cast(Any, trace_audit_report)
    _require(report.schema == HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA, "trace-audit schema differs")
    _require(report.valid is True, "trace-audit report is invalid")
    _require(tuple(report.mismatches) == (), "trace-audit report has mismatches")
    _require(
        tuple(report.unobserved_transition_fields) == (),
        "trace-audit report leaves transition fields unobserved",
    )
    _require(report.evidence_boundary == EVIDENCE_BOUNDARY, "trace-audit evidence boundary differs")
    report_payload = _result_payload(report.to_dict(), "trace audit report")
    report_digest = canonical_sha256(report_payload)
    _require(
        audit.get("trace_audit_report_sha256") == report_digest,
        "compact audit differs from the full trace-audit report",
    )
    direct_fields = (
        "valid",
        "expected_steps",
        "rows_checked",
        "helper_transitions_checked",
        "beneficiary_transitions_checked",
        "world_transitions_checked",
        "commit_lineages_checked",
        "recurrence_records_checked",
        "retention_aggregate_fields_checked",
        "summary_fields_checked",
        "resource_fields_checked",
    )
    for field in direct_fields:
        _require(audit.get(field) == getattr(report, field), f"compact audit {field} differs")
    _require(audit.get("mismatch_count") == 0, "compact audit mismatch count differs")
    _require(
        audit.get("mismatches_sha256") == canonical_sha256([]),
        "compact audit mismatch digest differs",
    )
    contractions = list(report.accepted_float32_contractions)
    _require(
        type(audit.get("accepted_float32_contraction_count")) is int
        and audit.get("accepted_float32_contraction_count") == len(contractions),
        "compact audit contraction count differs",
    )
    _require(
        audit.get("accepted_float32_contractions_sha256") == canonical_sha256(contractions),
        "compact audit contraction list digest differs",
    )
    _require(audit.get("unobserved_transition_fields") == [], "compact audit omits observed fields")
    _require(
        audit.get("evidence_boundary_sha256")
        == hashlib.sha256(EVIDENCE_BOUNDARY.encode("utf-8")).hexdigest(),
        "compact audit evidence-boundary digest differs",
    )
    _require(audit.get("lineage_oracle_valid") is True, "compact lineage audit is invalid")
    _require(
        audit.get("lineage_oracle_mismatches_sha256") == canonical_sha256([]),
        "compact lineage audit mismatch digest differs",
    )
    _require(
        audit.get("audited_summary_sha256") == canonical_sha256(summary),
        "compact audit summary binding differs",
    )
    return report_digest


def _trace_audit_input_binding(
    *,
    binding: Mapping[str, object],
    completed: Mapping[str, object],
    trace_audit_report_sha256: str,
) -> dict[str, object]:
    from alberta_framework.evaluation.hidden_regime_signaling_development import (
        HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
        HIDDEN_REGIME_TRACE_SCHEMA,
    )
    from alberta_framework.evaluation.hidden_regime_trace_audit import (
        HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
    )

    body = {
        "schema": CALIBRATION_EXECUTION_TRACE_AUDIT_BINDING_SCHEMA,
        "audit_input_schema": HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
        "development_schema": HIDDEN_REGIME_DEVELOPMENT_SCHEMA,
        "primitive_trace_schema": HIDDEN_REGIME_TRACE_SCHEMA,
        "case_binding_sha256": canonical_sha256(binding),
        "configuration_digest_schema": CALIBRATION_EXECUTION_CONFIGURATION_DIGEST_SCHEMA,
        "configuration_sha256": binding["configuration_sha256"],
        "summary_digest_schema": CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA,
        "summary_sha256": completed["summary_sha256"],
        "resource_digest_schema": CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA,
        "resource_sha256": completed["resource_sha256"],
        "primitive_trace_digest_schema": CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA,
        "primitive_trace_sha256": completed["primitive_trace_sha256"],
        "final_state_digest_schema": CALIBRATION_EXECUTION_FINAL_STATE_DIGEST_SCHEMA,
        "final_state_sha256": completed["final_state_sha256"],
        "executed_steps": completed["executed_steps"],
        "trace_audit_report_sha256": trace_audit_report_sha256,
        "raw_trace_persisted": False,
        "final_state_persisted": False,
        "ephemeral_input_reauditable_from_ledger": False,
    }
    return _payload_with_digest(body, "trace_audit_input_binding_sha256")


def _require_trace_audit_input_binding(
    payload: Mapping[str, object],
    *,
    binding: Mapping[str, object],
    completed: Mapping[str, object],
    trace_audit_report_sha256: str,
) -> dict[str, object]:
    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    body = _validate_payload_digest(
        normalized,
        digest_field="trace_audit_input_binding_sha256",
        label="trace-audit input binding",
    )
    _exact_keys(
        body,
        {
            "schema",
            "audit_input_schema",
            "development_schema",
            "primitive_trace_schema",
            "case_binding_sha256",
            "configuration_digest_schema",
            "configuration_sha256",
            "summary_digest_schema",
            "summary_sha256",
            "resource_digest_schema",
            "resource_sha256",
            "primitive_trace_digest_schema",
            "primitive_trace_sha256",
            "final_state_digest_schema",
            "final_state_sha256",
            "executed_steps",
            "trace_audit_report_sha256",
            "raw_trace_persisted",
            "final_state_persisted",
            "ephemeral_input_reauditable_from_ledger",
        },
        "trace-audit input binding",
    )
    expected = _trace_audit_input_binding(
        binding=binding,
        completed=completed,
        trace_audit_report_sha256=trace_audit_report_sha256,
    )
    _require(normalized == expected, "trace-audit input binding differs from completion")
    return expected


def _trace_audit_report_from_payload(payload: Mapping[str, object]) -> object:
    """Reconstruct the exact full audit report stored beside a finalized shard."""

    from alberta_framework.evaluation.hidden_regime_trace_audit import (
        HiddenRegimeTraceAuditReport,
    )

    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    _exact_keys(
        normalized,
        {
            "schema",
            "valid",
            "expected_steps",
            "rows_checked",
            "helper_transitions_checked",
            "beneficiary_transitions_checked",
            "world_transitions_checked",
            "commit_lineages_checked",
            "recurrence_records_checked",
            "retention_aggregate_fields_checked",
            "summary_fields_checked",
            "resource_fields_checked",
            "mismatches",
            "accepted_float32_contractions",
            "unobserved_transition_fields",
            "evidence_boundary",
        },
        "stored full trace-audit report",
    )
    _require(type(normalized["valid"]) is bool, "stored trace-audit validity is not boolean")
    integer_fields = (
        "expected_steps",
        "rows_checked",
        "helper_transitions_checked",
        "beneficiary_transitions_checked",
        "world_transitions_checked",
        "commit_lineages_checked",
        "recurrence_records_checked",
        "retention_aggregate_fields_checked",
        "summary_fields_checked",
        "resource_fields_checked",
    )
    for field in integer_fields:
        _strict_int(normalized[field], f"stored trace-audit {field}", maximum=10**9)

    def string_tuple(field: str) -> tuple[str, ...]:
        values = _plain_list(normalized[field], f"stored trace-audit {field}")
        _require(all(type(value) is str for value in values), f"stored trace-audit {field} differs")
        return tuple(cast(list[str], values))

    report = HiddenRegimeTraceAuditReport(
        valid=cast(bool, normalized["valid"]),
        expected_steps=cast(int, normalized["expected_steps"]),
        rows_checked=cast(int, normalized["rows_checked"]),
        helper_transitions_checked=cast(int, normalized["helper_transitions_checked"]),
        beneficiary_transitions_checked=cast(int, normalized["beneficiary_transitions_checked"]),
        world_transitions_checked=cast(int, normalized["world_transitions_checked"]),
        mismatches=string_tuple("mismatches"),
        commit_lineages_checked=cast(int, normalized["commit_lineages_checked"]),
        recurrence_records_checked=cast(int, normalized["recurrence_records_checked"]),
        retention_aggregate_fields_checked=cast(
            int,
            normalized["retention_aggregate_fields_checked"],
        ),
        summary_fields_checked=cast(int, normalized["summary_fields_checked"]),
        resource_fields_checked=cast(int, normalized["resource_fields_checked"]),
        accepted_float32_contractions=string_tuple("accepted_float32_contractions"),
        unobserved_transition_fields=string_tuple("unobserved_transition_fields"),
        evidence_boundary=_strict_string(
            normalized["evidence_boundary"],
            "stored trace-audit evidence boundary",
        ),
        schema=_strict_string(normalized["schema"], "stored trace-audit schema"),
    )
    _require(
        cast(dict[str, object], _normalized_json(report.to_dict())) == normalized,
        "stored full trace-audit report is not canonical",
    )
    return report


def _validate_final_shard_payload(
    shard_payload: Mapping[str, object],
    *,
    binding: Mapping[str, object],
    started: Mapping[str, object],
    completed: Mapping[str, object],
    attempt_binding: Mapping[str, object],
    trace_audit_report: object | None,
) -> tuple[dict[str, object], str]:
    normalized = cast(dict[str, object], _normalized_json(dict(shard_payload)))
    body = _validate_payload_digest(
        normalized,
        digest_field="payload_sha256",
        label="final case shard",
    )
    _require(
        body.get("schema") == CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA,
        "final case shard schema differs",
    )
    _require(body.get("development_only") is True, "final case shard is not development-only")
    for field in (
        "scientific_promotion_allowed",
        "claim_accepted",
        "thresholds_frozen",
        "promotion_artifact",
    ):
        _require(body.get(field) is False, f"final case shard {field} must be false")
    case_payload = _plain_mapping(body.get("case"), "final shard case")
    case_index = _strict_int(case_payload.get("case_index"), "final shard case index", maximum=239)
    design = _frozen_design()
    case = design.cases[case_index]
    _require(case_payload == case.to_payload(), "final shard differs from the frozen case")
    _require(
        body.get("protocol_payload_sha256") == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "final shard protocol differs",
    )
    _require(body.get("seed_snapshot_sha256") == SEED_SNAPSHOT_SHA256, "final shard seeds differ")
    _require(
        body.get("case_request_binding_sha256") == binding["case_request_binding_sha256"],
        "final shard case-request binding differs",
    )
    config_payload = _plain_mapping(body.get("configuration"), "final shard configuration")
    config_digest = _strict_sha256(
        body.get("configuration_sha256"), "final shard configuration digest"
    )
    _require(
        config_digest == canonical_sha256(config_payload),
        "final shard configuration digest differs",
    )
    _require(
        config_digest == binding["configuration_sha256"],
        "final shard configuration differs from start",
    )
    readiness = _plain_mapping(body.get("readiness_binding"), "final shard readiness binding")
    _require(
        readiness.get("readiness_receipt_sha256") == binding["readiness_receipt_sha256"],
        "final shard readiness differs",
    )
    for field in ("source_archive_sha256", "source_manifest_sha256", "runtime_identity_sha256"):
        _require(readiness.get(field) == binding[field], f"final shard readiness {field} differs")
    governance = _plain_mapping(
        readiness.get(READINESS_EXECUTION_GOVERNANCE_FIELD),
        "final shard execution governance binding",
    )
    _require(
        governance.get("genesis_sha256") == binding["genesis_sha256"], "final shard genesis differs"
    )
    summary = _plain_mapping(body.get("summary"), "final shard summary")
    summary_digest = _strict_sha256(body.get("summary_sha256"), "final shard summary digest")
    resource = _plain_mapping(body.get("resource"), "final shard resource")
    resource_digest = _strict_sha256(body.get("resource_sha256"), "final shard resource digest")
    _require(
        summary_digest == canonical_sha256(summary), "final shard summary payload digest differs"
    )
    _require(
        resource_digest == canonical_sha256(resource), "final shard resource payload digest differs"
    )
    primitive_trace = _plain_mapping(body.get("primitive_trace"), "final shard trace binding")
    trace_digest = _strict_sha256(primitive_trace.get("sha256"), "final shard trace digest")
    _require(primitive_trace.get("persisted") is False, "final shard claims a persisted raw trace")
    _require(
        summary_digest == completed["summary_sha256"]
        and resource_digest == completed["resource_sha256"]
        and trace_digest == completed["primitive_trace_sha256"],
        "final shard components differ from immutable completion",
    )
    executed_steps = _strict_int(
        body.get("executed_steps"), "final shard executed steps", maximum=10**9
    )
    _require(executed_steps == completed["executed_steps"], "final shard execution length differs")
    attempt_count = _strict_int(
        body.get("managed_execution_attempt_count"),
        "final shard managed execution attempt count",
        minimum=1,
        maximum=_MAX_ATTEMPTS_PER_CASE,
    )
    _require(
        attempt_count == attempt_binding["managed_execution_attempt_count"],
        "final shard managed execution attempt count differs",
    )
    _require(body.get("unique_completed_outcome_count") == 1, "final shard outcome count differs")
    expected_execution_binding = {
        "case_index": case_index,
        "genesis_sha256": binding["genesis_sha256"],
        "started_record_sha256": started["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "summary_sha256": summary_digest,
        "resource_sha256": resource_digest,
        "primitive_trace_sha256": trace_digest,
        "final_state_sha256": completed["final_state_sha256"],
        "outcome_sha256": completed["outcome_sha256"],
        "managed_execution_attempt_count": attempt_count,
        "attempt_records_sha256": attempt_binding["attempt_records_sha256"],
        "zip_provenance_binding_sha256": binding["zip_provenance_binding_sha256"],
        "zip_provenance_attestation_sha256": binding["zip_provenance_attestation_sha256"],
    }
    _require(
        _plain_mapping(body.get("execution_record_binding"), "final shard execution binding")
        == expected_execution_binding,
        "final shard execution binding differs from immutable ledger",
    )
    audit = _plain_mapping(body.get("audit"), "final shard audit")
    if trace_audit_report is None:
        trace_audit_digest = _strict_sha256(
            audit.get("trace_audit_report_sha256"),
            "final shard trace-audit report digest",
        )
        _require(audit.get("valid") is True, "final shard compact audit is invalid")
        _require(audit.get("mismatch_count") == 0, "final shard compact audit has mismatches")
        _require(
            audit.get("mismatches_sha256") == canonical_sha256([]),
            "final shard audit mismatch digest differs",
        )
        _require(
            audit.get("lineage_oracle_valid") is True,
            "final shard compact lineage audit is invalid",
        )
        _require(
            audit.get("lineage_oracle_mismatches_sha256") == canonical_sha256([]),
            "final shard compact lineage mismatch digest differs",
        )
        _require(
            audit.get("audited_summary_sha256") == summary_digest,
            "final shard audit summary differs",
        )
    else:
        trace_audit_digest = _require_trace_audit_report_matches_shard(
            trace_audit_report,
            audit,
            summary,
        )
    return normalized, trace_audit_digest


def _finalized_record(
    shard_payload: Mapping[str, object],
    *,
    binding: Mapping[str, object],
    started: Mapping[str, object],
    completed: Mapping[str, object],
    attempt_binding: Mapping[str, object],
    trace_audit_report_sha256: str,
    trace_audit_report_payload: Mapping[str, object],
    trace_audit_input_binding: Mapping[str, object],
) -> dict[str, object]:
    normalized_shard = cast(dict[str, object], _normalized_json(dict(shard_payload)))
    normalized_audit_input = cast(
        dict[str, object],
        _normalized_json(dict(trace_audit_input_binding)),
    )
    body = {
        "schema": CALIBRATION_EXECUTION_FINALIZED_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_state": "post_audit_case_shard_finalized",
        "case_index": binding["case_index"],
        "genesis_sha256": binding["genesis_sha256"],
        "case_binding_sha256": canonical_sha256(binding),
        "started_record_sha256": started["started_record_sha256"],
        "completed_record_sha256": completed["completed_record_sha256"],
        "managed_execution_attempt_count": attempt_binding["managed_execution_attempt_count"],
        "attempt_records_sha256": attempt_binding["attempt_records_sha256"],
        "case_request_binding_sha256": binding["case_request_binding_sha256"],
        "summary_sha256": completed["summary_sha256"],
        "resource_sha256": completed["resource_sha256"],
        "primitive_trace_sha256": completed["primitive_trace_sha256"],
        "final_state_sha256": completed["final_state_sha256"],
        "outcome_sha256": completed["outcome_sha256"],
        "zip_provenance_binding_sha256": binding["zip_provenance_binding_sha256"],
        "zip_provenance_attestation_sha256": binding["zip_provenance_attestation_sha256"],
        "final_shard_digest_schema": CALIBRATION_EXECUTION_FINAL_SHARD_DIGEST_SCHEMA,
        "shard_payload_sha256": normalized_shard["payload_sha256"],
        "shard_canonical_sha256": canonical_sha256(normalized_shard),
        "trace_audit_report_sha256": trace_audit_report_sha256,
        "trace_audit_report": dict(trace_audit_report_payload),
        "trace_audit_input_binding_sha256": normalized_audit_input[
            "trace_audit_input_binding_sha256"
        ],
        "trace_audit_input_binding": normalized_audit_input,
        "raw_trace_persisted": False,
        "final_state_persisted": False,
        "shard_payload": normalized_shard,
        "shard_finalization_policy": SHARD_FINALIZATION_POLICY,
        "replay_accounting_policy": REPLAY_ACCOUNTING_POLICY,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    return _payload_with_digest(body, "finalized_record_sha256")


def require_valid_calibration_execution_finalized_record(
    payload: Mapping[str, object],
    *,
    expected_started: Mapping[str, object],
    expected_completed: Mapping[str, object],
    expected_attempt_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Strictly validate one immutable post-audit compact-shard finalization."""

    body = _validate_payload_digest(
        payload, digest_field="finalized_record_sha256", label="finalized record"
    )
    started = require_valid_calibration_execution_started_record(expected_started)
    completed = require_valid_calibration_execution_completed_record(
        expected_completed,
        expected_started=started,
    )
    binding = cast(dict[str, object], started["case_binding"])
    attempt_binding = _attempt_binding_from_rows(
        cast(int, binding["case_index"]), expected_attempt_rows
    )
    shard = _plain_mapping(body.get("shard_payload"), "finalized shard payload")
    stored_audit_payload = _plain_mapping(
        body.get("trace_audit_report"),
        "stored full trace-audit report",
    )
    stored_audit_report = _trace_audit_report_from_payload(stored_audit_payload)
    normalized_shard, trace_audit_digest = _validate_final_shard_payload(
        shard,
        binding=binding,
        started=started,
        completed=completed,
        attempt_binding=attempt_binding,
        trace_audit_report=stored_audit_report,
    )
    stored_audit_input = _plain_mapping(
        body.get("trace_audit_input_binding"),
        "stored trace-audit input binding",
    )
    validated_audit_input = _require_trace_audit_input_binding(
        stored_audit_input,
        binding=binding,
        completed=completed,
        trace_audit_report_sha256=trace_audit_digest,
    )
    _require(
        body.get("trace_audit_input_binding_sha256")
        == validated_audit_input["trace_audit_input_binding_sha256"],
        "finalized trace-audit input binding digest differs",
    )
    expected = _finalized_record(
        normalized_shard,
        binding=binding,
        started=started,
        completed=completed,
        attempt_binding=attempt_binding,
        trace_audit_report_sha256=trace_audit_digest,
        trace_audit_report_payload=stored_audit_payload,
        trace_audit_input_binding=validated_audit_input,
    )
    _require(
        cast(dict[str, object], _normalized_json(dict(payload))) == expected,
        "finalized record is not deterministic or differs from its shard",
    )
    return expected


def _require_authorization_matches_latest_attempt(
    authorization_payload: Mapping[str, object],
    *,
    started: Mapping[str, object],
    completed: Mapping[str, object],
    attempt_rows: list[dict[str, object]],
) -> None:
    _require(bool(attempt_rows), "finalizer case has no managed attempt")
    latest = attempt_rows[-1]
    attempt_index = cast(int, latest["attempt_index"])
    mode = latest["execution_mode"]
    _require(
        authorization_payload["attempt_index"] == attempt_index,
        "finalizer authorization is not the latest managed attempt",
    )
    _require(
        authorization_payload["execution_mode"] == mode,
        "finalizer authorization mode differs from the latest managed attempt",
    )
    _require(
        authorization_payload["prior_attempt_records_sha256"]
        == canonical_sha256(attempt_rows[:-1]),
        "finalizer authorization prior attempt ledger differs",
    )
    _require(
        authorization_payload["zip_provenance_attestation_sha256"]
        == latest["zip_provenance_attestation_sha256"],
        "finalizer authorization ZIP provenance differs from latest attempt",
    )
    _require(
        authorization_payload["attempt_request_payload_sha256"]
        == latest["attempt_request_payload_sha256"],
        "finalizer authorization request differs from latest attempt",
    )
    _require(
        authorization_payload["exact_replay_consent"] == latest["exact_replay_consent"],
        "finalizer authorization consent differs from latest attempt",
    )
    if attempt_index == 0:
        _require(
            authorization_payload["prior_started_record_sha256"] is None
            and authorization_payload["prior_completed_record_sha256"] is None,
            "first-attempt finalizer authorization has prior records",
        )
        return
    _require(
        authorization_payload["prior_started_record_sha256"]
        == started["started_record_sha256"],
        "finalizer authorization prior start differs",
    )
    if mode == "exact_replay_after_completion":
        _require(
            authorization_payload["prior_completed_record_sha256"]
            == completed["completed_record_sha256"],
            "finalizer authorization prior completion differs",
        )
    else:
        _require(
            authorization_payload["prior_completed_record_sha256"] is None,
            "interrupted-replay finalizer authorization has a prior completion",
        )


def finalize_calibration_case_shard(
    authorization: object,
    *,
    ledger_directory: Path,
    shard_payload: Mapping[str, object],
    run_result: object,
) -> dict[str, object]:
    """Re-audit the exact completed run and atomically finalize its compact shard."""

    normalized_shard = cast(dict[str, object], _normalized_json(dict(shard_payload)))
    shard_body = _validate_payload_digest(
        normalized_shard,
        digest_field="payload_sha256",
        label="final case shard",
    )
    case_payload = _plain_mapping(shard_body.get("case"), "final shard case")
    case_index = _strict_int(case_payload.get("case_index"), "final shard case index", maximum=239)
    case = _frozen_design().cases[case_index]
    from alberta_framework.evaluation.hidden_regime_signaling_development import (
        HiddenRegimeSeedPair,
    )

    seed_pair = HiddenRegimeSeedPair(
        namespace=CONSUMED_CALIBRATION_NAMESPACE,
        index=case.seed_index,
        world_seed=case.world_seed,
        learner_seed=case.learner_seed,
    )
    checked = _require_authorization(
        authorization,
        condition=case.condition,
        seed_pair=seed_pair,
        config=_expected_case_config(case),
    )
    _require(
        Path(cast(str, checked.payload["ledger_directory"])).absolute()
        == ledger_directory.absolute(),
        "finalizer authorization ledger differs",
    )
    binding = cast(dict[str, object], checked.payload["case_binding"])
    _require(binding["case_index"] == case_index, "finalizer authorization case differs")
    genesis = _load_ledger_genesis(ledger_directory)
    _require(binding["genesis_sha256"] == genesis["genesis_sha256"], "finalizer genesis differs")
    case_directory = _case_directory(ledger_directory, case_index)
    finalized_path = case_directory / _FINALIZED_FILE
    with _case_mutation_critical_section(case_directory, case_index) as case_fd:
        finalization_members = _validated_case_member_names(case_directory, case_index)
        _require(
            _STARTED_FILE in finalization_members and _COMPLETED_FILE in finalization_members,
            "finalization requires immutable start and completion records",
        )
        started = require_valid_calibration_execution_started_record(
            _open_immutable_json(case_directory / _STARTED_FILE, "started record"),
            expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
        )
        _require(started["case_binding"] == binding, "finalizer start binding differs")
        completed = require_valid_calibration_execution_completed_record(
            _open_immutable_json(case_directory / _COMPLETED_FILE, "completed record"),
            expected_started=started,
        )
        attempt_rows = _case_attempt_rows(case_directory, started)
        attempt_binding = _attempt_binding_from_rows(case_index, attempt_rows)
        _require_authorization_matches_latest_attempt(
            checked.payload,
            started=started,
            completed=completed,
            attempt_rows=attempt_rows,
        )
        if _FINALIZED_FILE in finalization_members:
            existing = require_valid_calibration_execution_finalized_record(
                _open_immutable_json(finalized_path, "finalized record"),
                expected_started=started,
                expected_completed=completed,
                expected_attempt_rows=attempt_rows,
            )
            _require(
                existing["shard_payload"] == normalized_shard,
                "finalized shard replay differs from immutable finalization",
            )
            return existing

        components = _validated_result_component_digests(run_result, binding)
        _require(
            components["summary_sha256"] == completed["summary_sha256"]
            and components["resource_sha256"] == completed["resource_sha256"]
            and components["primitive_trace_sha256"] == completed["primitive_trace_sha256"]
            and components["final_state_sha256"] == completed["final_state_sha256"]
            and components["executed_steps"] == completed["executed_steps"],
            "finalizer run components differ from immutable completion",
        )
        capability = _require_completed_run_capability(
            run_result,
            ledger_directory=ledger_directory,
            binding=binding,
            completed=completed,
            attempt_rows=attempt_rows,
            components=components,
        )
        from alberta_framework.evaluation.hidden_regime_factorial_calibration import (
            validate_calibration_case_shard,
        )
        from alberta_framework.evaluation.hidden_regime_trace_audit import (
            audit_hidden_regime_run_result,
        )

        strict_shard = validate_calibration_case_shard(normalized_shard)
        _require(
            cast(dict[str, object], _normalized_json(strict_shard)) == normalized_shard,
            "strict final shard validation changed its payload",
        )
        trace_audit_report = audit_hidden_regime_run_result(cast(Any, run_result))
        post_audit_components = _validated_result_component_digests(run_result, binding)
        _require(
            post_audit_components == components,
            "finalizer run components changed during trace audit",
        )
        normalized_shard, trace_audit_digest = _validate_final_shard_payload(
            normalized_shard,
            binding=binding,
            started=started,
            completed=completed,
            attempt_binding=attempt_binding,
            trace_audit_report=trace_audit_report,
        )
        report_payload = cast(
            dict[str, object],
            _result_payload(
                cast(Any, trace_audit_report).to_dict(),
                "trace audit report",
            ),
        )
        audit_input_binding = _trace_audit_input_binding(
            binding=binding,
            completed=completed,
            trace_audit_report_sha256=trace_audit_digest,
        )
        expected = _finalized_record(
            normalized_shard,
            binding=binding,
            started=started,
            completed=completed,
            attempt_binding=attempt_binding,
            trace_audit_report_sha256=trace_audit_digest,
            trace_audit_report_payload=report_payload,
            trace_audit_input_binding=audit_input_binding,
        )
        try:
            _write_new_immutable(case_fd, _FINALIZED_FILE, canonical_json_bytes(expected))
        except FileExistsError:
            existing = require_valid_calibration_execution_finalized_record(
                _open_immutable_json(finalized_path, "finalized record"),
                expected_started=started,
                expected_completed=completed,
                expected_attempt_rows=attempt_rows,
            )
            _require(existing == expected, "concurrent finalization differs")
            return existing
        os.fsync(case_fd)
        capability_key = (ledger_directory.absolute().as_posix(), case_index)
        registered = _COMPLETED_RUN_CAPABILITIES.get(capability_key)
        if registered is capability:
            del _COMPLETED_RUN_CAPABILITIES[capability_key]
        return expected


def load_finalized_calibration_case_shard(
    ledger_directory: Path,
    case_index: int,
) -> dict[str, object]:
    """Recover the immutable compact shard after a finalized-worker publication crash."""

    genesis = _load_ledger_genesis(ledger_directory)
    case_directory = _case_directory(ledger_directory, case_index)
    started = require_valid_calibration_execution_started_record(
        _open_immutable_json(case_directory / _STARTED_FILE, "started record"),
        expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
    )
    completed = require_valid_calibration_execution_completed_record(
        _open_immutable_json(case_directory / _COMPLETED_FILE, "completed record"),
        expected_started=started,
    )
    attempts = _case_attempt_rows(case_directory, started)
    finalized = require_valid_calibration_execution_finalized_record(
        _open_immutable_json(case_directory / _FINALIZED_FILE, "finalized record"),
        expected_started=started,
        expected_completed=completed,
        expected_attempt_rows=attempts,
    )
    return cast(dict[str, object], _normalized_json(finalized["shard_payload"]))


__all__ = [
    "CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA",
    "CALIBRATION_EXECUTION_COMPLETED_SCHEMA",
    "CALIBRATION_EXECUTION_FINALIZED_SCHEMA",
    "CALIBRATION_EXECUTION_FINAL_SHARD_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_FINAL_CASE_SHARD_SCHEMA",
    "CALIBRATION_EXECUTION_CONFIGURATION_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA",
    "CALIBRATION_EXECUTION_GENESIS_SCHEMA",
    "CALIBRATION_EXECUTION_INVENTORY_SCHEMA",
    "CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_STARTED_SCHEMA",
    "CALIBRATION_EXECUTION_REPLAY_STARTED_SCHEMA",
    "CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA",
    "CRASH_CONSUMPTION_RULE",
    "EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT",
    "MANAGED_EXECUTION_BOUNDARY_SCOPE",
    "PROTECTED_EXECUTION_POLICY",
    "PROCESS_LOCAL_AUTHORIZATION_SCOPE",
    "REPLAY_ACCOUNTING_POLICY",
    "READINESS_EXECUTION_GOVERNANCE_FIELD",
    "CalibrationExecutionAuthorization",
    "CalibrationZipProvenanceCapability",
    "HiddenRegimeCaseConsumedError",
    "HiddenRegimeExecutionClassification",
    "HiddenRegimeExecutionGovernanceError",
    "HiddenRegimeProtectedExecutionError",
    "ManagedCalibrationExecutionTicket",
    "PublishedCalibrationExecutionLedger",
    "atomic_install_new_immutable",
    "begin_managed_hidden_regime_execution",
    "attest_calibration_zip_provenance",
    "build_calibration_execution_genesis",
    "calibration_execution_configuration_sha256",
    "calibration_execution_genesis_receipt_binding",
    "calibration_execution_primitive_trace_sha256",
    "calibration_execution_resource_sha256",
    "calibration_execution_summary_sha256",
    "calibration_case_attempt_binding",
    "canonical_json_bytes",
    "canonical_sha256",
    "classify_hidden_regime_world",
    "complete_managed_hidden_regime_execution",
    "finalize_calibration_case_shard",
    "hidden_regime_world_requires_managed_execution",
    "initialize_calibration_execution_ledger",
    "issue_calibration_execution_authorization",
    "require_valid_calibration_execution_completed_record",
    "require_valid_calibration_execution_finalized_record",
    "require_valid_calibration_execution_genesis",
    "require_valid_calibration_execution_inventory",
    "require_valid_calibration_execution_started_record",
    "require_valid_calibration_execution_replay_started_record",
    "load_finalized_calibration_case_shard",
    "snapshot_calibration_execution_inventory",
    "validate_completed_calibration_ledger_snapshot",
    "SHARD_FINALIZATION_POLICY",
    "ZIP_PROVENANCE_POLICY",
    "ZIP_PROVENANCE_SOURCE_ARCHIVE_LOCATOR",
]
