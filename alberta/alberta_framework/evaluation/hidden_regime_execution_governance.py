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

import dataclasses
import errno
import functools
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping
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

CALIBRATION_EXECUTION_GENESIS_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-genesis.v1"
)
CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-genesis-receipt-binding.v1"
)
CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-authorization.v1"
)
CALIBRATION_EXECUTION_STARTED_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-started.v1"
)
CALIBRATION_EXECUTION_COMPLETED_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-completed.v1"
)
CALIBRATION_EXECUTION_INVENTORY_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-inventory.v1"
)
CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA = (
    "alberta.hidden-regime-factorial.execution-outcome.component-bundle.v1"
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
    "handoff is unsupported"
)

type ExecutionSensitivity = Literal["ordinary", "calibration", "protected"]
type ExecutionMode = Literal[
    "first_execution",
    "exact_replay_after_interruption",
    "exact_replay_after_completion",
]

_PROCESS_SEAL_KEY = secrets.token_bytes(32)
_SHA256_LENGTH = 64
_MAX_RECORD_BYTES = 4 * 1024 * 1024
_CASE_DIRECTORY_PREFIX = "case-"
_STARTED_FILE = "started.json"
_COMPLETED_FILE = "completed.json"


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
        "interrupted_case_indices": [],
        "started_record_count": 0,
        "completed_record_count": 0,
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
        "initial_inventory": initial_inventory,
        "initial_inventory_sha256": canonical_sha256(initial_inventory),
        "crash_consumption_rule": CRASH_CONSUMPTION_RULE,
        "protected_execution_policy": PROTECTED_EXECUTION_POLICY,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
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
            "initial_inventory",
            "initial_inventory_sha256",
            "crash_consumption_rule",
            "protected_execution_policy",
            "managed_boundary_scope",
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
        "initial_protected_record_count": 0,
        "protected_execution_permitted": False,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
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


def _write_new_immutable(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing execution ledger record")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    ledger_fd: int | None = None
    cases_fd: int | None = None
    try:
        try:
            os.mkdir(digest, 0o700, dir_fd=root_fd)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite execution ledger: {root / digest}"
            ) from error
        ledger_fd = os.open(digest, flags, dir_fd=root_fd)
        os.mkdir("cases", 0o700, dir_fd=ledger_fd)
        cases_fd = os.open("cases", flags, dir_fd=ledger_fd)
        for case_index in range(N_MATCHED_CASES):
            os.mkdir(f"{_CASE_DIRECTORY_PREFIX}{case_index:03d}", 0o700, dir_fd=cases_fd)
        _write_new_immutable(ledger_fd, "genesis.json", canonical_json_bytes(normalized))
        os.fsync(cases_fd)
        os.fsync(ledger_fd)
        os.fsync(root_fd)
    finally:
        if cases_fd is not None:
            os.close(cases_fd)
        if ledger_fd is not None:
            os.close(ledger_fd)
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


def _open_immutable_json(path: Path, label: str) -> dict[str, object]:
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
        _require(before.st_size <= _MAX_RECORD_BYTES, f"{label} exceeds the size limit")
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
        return _strict_json(b"".join(chunks), label)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _load_ledger_genesis(directory: Path) -> dict[str, object]:
    absolute = directory.absolute()
    _directory_without_symlink(absolute, "execution ledger")
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
    "case_index",
    "manifest_name",
    "manifest_payload_sha256",
    "configuration_sha256",
    "request_payload_sha256",
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
        "manifest_payload_sha256",
        "configuration_sha256",
        "request_payload_sha256",
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


def _started_record(binding: Mapping[str, object]) -> dict[str, object]:
    body = {
        "schema": CALIBRATION_EXECUTION_STARTED_SCHEMA,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "execution_state": "started_case_consumed",
        "case_binding": dict(binding),
        "case_binding_sha256": canonical_sha256(binding),
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
    if expected_genesis_sha256 is not None:
        _strict_sha256(expected_genesis_sha256, "expected_genesis_sha256")
        _require(
            binding["genesis_sha256"] == expected_genesis_sha256,
            "started record belongs to another genesis",
        )
    expected = _started_record(binding)
    normalized = cast(dict[str, object], _normalized_json(dict(payload)))
    _require(normalized == expected, "started record is not deterministic")
    return expected


def _component_outcome_sha256(
    *,
    case_binding_sha256: str,
    started_record_sha256: str,
    summary_sha256: str,
    resource_sha256: str,
    primitive_trace_sha256: str,
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
            "primitive_trace_digest_schema": (
                CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA
            ),
            "primitive_trace_sha256": primitive_trace_sha256,
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
    executed_steps: int,
) -> dict[str, object]:
    case_binding_sha256 = canonical_sha256(binding)
    outcome_sha256 = _component_outcome_sha256(
        case_binding_sha256=case_binding_sha256,
        started_record_sha256=started_record_sha256,
        summary_sha256=summary_sha256,
        resource_sha256=resource_sha256,
        primitive_trace_sha256=primitive_trace_sha256,
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
    interrupted: list[int] = []
    for case_index, name in enumerate(expected_names):
        case_directory = cases_directory / name
        _directory_without_symlink(case_directory, f"execution case {case_index} directory")
        members = _directory_members(case_directory, f"execution case {case_index} directory")
        _require(
            set(members).issubset({_STARTED_FILE, _COMPLETED_FILE}),
            f"execution case {case_index} contains an unknown member",
        )
        _require(
            _COMPLETED_FILE not in members or _STARTED_FILE in members,
            f"execution case {case_index} completed without a start",
        )
        started: dict[str, object] | None = None
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
                }
            )
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
                    "outcome_sha256": completed["outcome_sha256"],
                }
            )
        elif started is not None:
            interrupted.append(case_index)
    started_indices = [cast(int, record["case_index"]) for record in started_records]
    completed_indices = [cast(int, record["case_index"]) for record in completed_records]
    body: dict[str, object] = {
        "schema": CALIBRATION_EXECUTION_INVENTORY_SCHEMA,
        "genesis_sha256": genesis["genesis_sha256"],
        "expected_case_count": N_MATCHED_CASES,
        "started_case_indices": started_indices,
        "completed_case_indices": completed_indices,
        "interrupted_case_indices": interrupted,
        "started_record_count": len(started_records),
        "completed_record_count": len(completed_records),
        "protected_started_record_count": 0,
        "protected_completed_record_count": 0,
        "pristine": not started_records and not completed_records,
        "started_records": started_records,
        "completed_records": completed_records,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
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
            "interrupted_case_indices",
            "started_record_count",
            "completed_record_count",
            "protected_started_record_count",
            "protected_completed_record_count",
            "pristine",
            "started_records",
            "completed_records",
            "managed_boundary_scope",
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
            "interrupted_case_indices",
            "started_record_count",
            "completed_record_count",
            "protected_started_record_count",
            "protected_completed_record_count",
            "pristine",
            "started_records",
            "completed_records",
            "managed_boundary_scope",
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
    _require(body["interrupted_case_indices"] == [], "inventory contains interrupted cases")
    _require(body["started_record_count"] == N_MATCHED_CASES, "inventory start count differs")
    _require(
        body["completed_record_count"] == N_MATCHED_CASES,
        "inventory completion count differs",
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
    _require(len(started_items) == N_MATCHED_CASES, "started inventory record count differs")
    _require(
        len(completed_items) == N_MATCHED_CASES,
        "completed inventory record count differs",
    )
    started_by_case: dict[int, dict[str, object]] = {}
    completed_by_case: dict[int, dict[str, object]] = {}
    for expected_index, item in enumerate(started_items):
        _exact_keys(
            item,
            {"case_index", "started_record_sha256", "case_binding_sha256"},
            "started inventory record",
        )
        case_index = _strict_int(item["case_index"], "started case index", maximum=239)
        _require(case_index == expected_index, "started inventory records are not ordered")
        _strict_sha256(item["started_record_sha256"], "started record digest")
        _strict_sha256(item["case_binding_sha256"], "started case binding digest")
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
            "outcome_sha256",
        ):
            _strict_sha256(item[field], f"completed inventory {field}")
        completed_by_case[case_index] = item

    design = _frozen_design()
    for case_index in range(N_MATCHED_CASES):
        shard = _plain_mapping(shards_by_case[case_index], f"case shard {case_index}")
        case_payload = _plain_mapping(shard.get("case"), f"case shard {case_index}.case")
        case = design.cases[case_index]
        _require(case_payload == case.to_payload(), "shard differs from its frozen case")
        _require(
            shard.get("protocol_payload_sha256") == CALIBRATION_DESIGN_PAYLOAD_SHA256,
            "shard protocol digest differs",
        )
        _require(shard.get("seed_snapshot_sha256") == SEED_SNAPSHOT_SHA256, "shard seeds differ")
        request_digest = _strict_sha256(
            shard.get("request_payload_sha256"),
            "shard request payload digest",
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
            "case_index": case_index,
            "manifest_name": case.manifest_name,
            "manifest_payload_sha256": _manifest_digest(
                HIDDEN_REGIME_CALIBRATION_MANIFESTS[case.manifest_name]
            ),
            "configuration_sha256": configuration_digest,
            "request_payload_sha256": request_digest,
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
        expected_started = _started_record(normalized_binding)
        started_inventory = started_by_case[case_index]
        _require(
            started_inventory["case_binding_sha256"]
            == expected_started["case_binding_sha256"],
            "shard request/case binding differs from immutable start",
        )
        _require(
            started_inventory["started_record_sha256"]
            == expected_started["started_record_sha256"],
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
        _require(
            execution_binding
            == {
                "case_index": case_index,
                "genesis_sha256": genesis_sha256,
                "started_record_sha256": expected_started["started_record_sha256"],
                "completed_record_sha256": expected_completed["completed_record_sha256"],
                "summary_sha256": summary_digest,
                "resource_sha256": resource_digest,
                "primitive_trace_sha256": trace_digest,
                "outcome_sha256": expected_completed["outcome_sha256"],
            },
            "shard execution binding differs from immutable ledger",
        )
    return cast(dict[str, object], _normalized_json(dict(snapshot)))


@dataclass(frozen=True, slots=True)
class CalibrationExecutionAuthorization:
    """Process-sealed authorization for one exact managed case invocation."""

    payload: dict[str, object]
    seal: str


@dataclass(frozen=True, slots=True)
class ManagedCalibrationExecutionTicket:
    """Process-sealed proof that a start record exists before learner execution."""

    ledger_directory: Path
    case_binding: dict[str, object]
    started_record: dict[str, object]
    execution_mode: ExecutionMode
    seal: str


def _seal(kind: str, payload: object) -> str:
    return hmac.new(
        _PROCESS_SEAL_KEY,
        kind.encode("ascii") + b"\0" + canonical_json_bytes(payload),
        hashlib.sha256,
    ).hexdigest()


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
    request_payload_sha256: str,
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
    binding: dict[str, object] = {
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "genesis_sha256": genesis["genesis_sha256"],
        "readiness_receipt_sha256": cast(Any, readiness_bundle).receipt_sha256,
        "source_archive_sha256": cast(Any, readiness_bundle).source_archive_sha256,
        "source_manifest_sha256": cast(Any, readiness_bundle).source_manifest_sha256,
        "runtime_identity_sha256": cast(Any, readiness_bundle).runtime_identity_sha256,
        "case_index": case_index,
        "manifest_name": case.manifest_name,
        "manifest_payload_sha256": classification.manifest_payload_sha256,
        "configuration_sha256": _config_sha256(config),
        "request_payload_sha256": request_payload_sha256,
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
    case_index: int,
    condition: object,
    seed_pair: object,
    config: object,
    request_payload_sha256: str,
    explicit_acknowledgement: str,
    allow_exact_replay: bool = False,
) -> CalibrationExecutionAuthorization:
    """Issue a process-local seal for one exact frozen calibration case."""

    _require(
        explicit_acknowledgement == EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT,
        "exact managed calibration execution acknowledgement is required",
    )
    _require(type(allow_exact_replay) is bool, "allow_exact_replay must be a strict boolean")
    _strict_sha256(request_payload_sha256, "request_payload_sha256")
    validated = _validate_readiness_bundle(readiness_bundle, readiness_source_archive)
    genesis = _load_ledger_genesis(ledger_directory)
    binding = _binding_from_inputs(
        ledger_directory=ledger_directory,
        genesis=genesis,
        readiness_bundle=validated,
        case_index=case_index,
        condition=condition,
        seed_pair=seed_pair,
        config=config,
        request_payload_sha256=request_payload_sha256,
    )
    case_directory = _case_directory(ledger_directory, case_index)
    started_path = case_directory / _STARTED_FILE
    completed_path = case_directory / _COMPLETED_FILE
    prior_started: str | None = None
    prior_completed: str | None = None
    if started_path.exists():
        started = require_valid_calibration_execution_started_record(
            _open_immutable_json(started_path, "started record"),
            expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
        )
        _require(started["case_binding"] == binding, "consumed case binding differs")
        prior_started = cast(str, started["started_record_sha256"])
        if completed_path.exists():
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
    else:
        _require(not completed_path.exists(), "completion exists without a start record")
        mode = "first_execution"
    payload: dict[str, object] = {
        "schema": CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA,
        "ledger_directory": ledger_directory.absolute().as_posix(),
        "case_binding": binding,
        "case_binding_sha256": canonical_sha256(binding),
        "execution_mode": mode,
        "prior_started_record_sha256": prior_started,
        "prior_completed_record_sha256": prior_completed,
        "explicit_acknowledgement": explicit_acknowledgement,
        "authorization_scope": PROCESS_LOCAL_AUTHORIZATION_SCOPE,
        "crash_consumption_rule": CRASH_CONSUMPTION_RULE,
        "managed_boundary_scope": MANAGED_EXECUTION_BOUNDARY_SCOPE,
    }
    normalized = cast(dict[str, object], _normalized_json(payload))
    return CalibrationExecutionAuthorization(
        payload=normalized,
        seal=_seal("calibration-execution-authorization-v1", normalized),
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
    _require(
        hmac.compare_digest(
            result.seal,
            _seal("calibration-execution-authorization-v1", result.payload),
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
            "prior_started_record_sha256",
            "prior_completed_record_sha256",
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
    expected_started = _started_record(binding)
    mode = cast(ExecutionMode, payload["execution_mode"])
    _require(
        mode
        in {
            "first_execution",
            "exact_replay_after_interruption",
            "exact_replay_after_completion",
        },
        "authorization execution mode differs",
    )
    if mode == "first_execution":
        case_fd = _open_directory_without_symlink_ancestors(
            case_directory,
            f"execution case {case_index} directory",
        )
        try:
            try:
                _write_new_immutable(case_fd, _STARTED_FILE, canonical_json_bytes(expected_started))
            except FileExistsError as error:
                raise HiddenRegimeCaseConsumedError(
                    "concurrent or stale first-execution authorization lost the atomic start race"
                ) from error
            os.fsync(case_fd)
        finally:
            os.close(case_fd)
    else:
        actual_started = require_valid_calibration_execution_started_record(
            _open_immutable_json(started_path, "started record"),
            expected_genesis_sha256=cast(str, genesis["genesis_sha256"]),
        )
        _require(actual_started == expected_started, "replay start record binding differs")
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
            _require(not completed_path.exists(), "interrupted replay case is already completed")
    ticket_payload = {
        "ledger_directory": ledger_directory.absolute().as_posix(),
        "case_binding": binding,
        "started_record_sha256": expected_started["started_record_sha256"],
        "execution_mode": mode,
    }
    return ManagedCalibrationExecutionTicket(
        ledger_directory=ledger_directory.absolute(),
        case_binding=binding,
        started_record=expected_started,
        execution_mode=mode,
        seal=_seal("managed-calibration-execution-ticket-v1", ticket_payload),
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
            _exact_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
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
    }
    _require(
        hmac.compare_digest(
            ticket.seal,
            _seal("managed-calibration-execution-ticket-v1", ticket_payload),
        ),
        "execution ticket seal is invalid",
    )
    started = require_valid_calibration_execution_started_record(ticket.started_record)
    binding = cast(dict[str, object], started["case_binding"])
    _require(getattr(result, "condition", None) == binding["condition"], "result condition differs")
    result_pair = getattr(result, "seed_pair", None)
    pair_payload = {
        "namespace": getattr(result_pair, "namespace", None),
        "index": getattr(result_pair, "index", None),
        "world_seed": getattr(result_pair, "world_seed", None),
        "learner_seed": getattr(result_pair, "learner_seed", None),
    }
    _require(pair_payload == binding["seed_pair"], "result seed pair differs")
    _require(
        _config_sha256(getattr(result, "config", None)) == binding["configuration_sha256"],
        "result configuration differs",
    )
    executed_steps = getattr(getattr(result, "summary", None), "num_steps", None)
    _require(type(executed_steps) is int and executed_steps > 0, "result step count is invalid")
    summary_digest = calibration_execution_summary_sha256(getattr(result, "summary", None))
    resource_digest = calibration_execution_resource_sha256(getattr(result, "resource", None))
    trace_digest = calibration_execution_primitive_trace_sha256(getattr(result, "trace", None))
    expected = _completed_record(
        binding,
        started_record_sha256=cast(str, started["started_record_sha256"]),
        summary_sha256=summary_digest,
        resource_sha256=resource_digest,
        primitive_trace_sha256=trace_digest,
        executed_steps=cast(int, executed_steps),
    )
    case_index = cast(int, binding["case_index"])
    case_directory = _case_directory(ticket.ledger_directory, case_index)
    completed_path = case_directory / _COMPLETED_FILE
    if completed_path.exists():
        existing = require_valid_calibration_execution_completed_record(
            _open_immutable_json(completed_path, "completed record"),
            expected_started=started,
        )
        _require(existing == expected, "exact replay outcome differs from immutable completion")
        return existing
    case_fd = _open_directory_without_symlink_ancestors(
        case_directory,
        f"execution case {case_index} directory",
    )
    try:
        try:
            _write_new_immutable(case_fd, _COMPLETED_FILE, canonical_json_bytes(expected))
        except FileExistsError:
            existing = require_valid_calibration_execution_completed_record(
                _open_immutable_json(completed_path, "completed record"),
                expected_started=started,
            )
            _require(existing == expected, "concurrent replay completion outcome differs")
            return existing
        os.fsync(case_fd)
    finally:
        os.close(case_fd)
    return expected


__all__ = [
    "CALIBRATION_EXECUTION_AUTHORIZATION_SCHEMA",
    "CALIBRATION_EXECUTION_COMPLETED_SCHEMA",
    "CALIBRATION_EXECUTION_CONFIGURATION_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA",
    "CALIBRATION_EXECUTION_GENESIS_SCHEMA",
    "CALIBRATION_EXECUTION_INVENTORY_SCHEMA",
    "CALIBRATION_EXECUTION_OUTCOME_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_PRIMITIVE_TRACE_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_RESOURCE_DIGEST_SCHEMA",
    "CALIBRATION_EXECUTION_STARTED_SCHEMA",
    "CALIBRATION_EXECUTION_SUMMARY_DIGEST_SCHEMA",
    "CRASH_CONSUMPTION_RULE",
    "EXECUTION_AUTHORIZATION_ACKNOWLEDGEMENT",
    "MANAGED_EXECUTION_BOUNDARY_SCOPE",
    "PROTECTED_EXECUTION_POLICY",
    "PROCESS_LOCAL_AUTHORIZATION_SCOPE",
    "READINESS_EXECUTION_GOVERNANCE_FIELD",
    "CalibrationExecutionAuthorization",
    "HiddenRegimeCaseConsumedError",
    "HiddenRegimeExecutionClassification",
    "HiddenRegimeExecutionGovernanceError",
    "HiddenRegimeProtectedExecutionError",
    "ManagedCalibrationExecutionTicket",
    "PublishedCalibrationExecutionLedger",
    "begin_managed_hidden_regime_execution",
    "build_calibration_execution_genesis",
    "calibration_execution_configuration_sha256",
    "calibration_execution_genesis_receipt_binding",
    "calibration_execution_primitive_trace_sha256",
    "calibration_execution_resource_sha256",
    "calibration_execution_summary_sha256",
    "canonical_json_bytes",
    "canonical_sha256",
    "classify_hidden_regime_world",
    "complete_managed_hidden_regime_execution",
    "hidden_regime_world_requires_managed_execution",
    "initialize_calibration_execution_ledger",
    "issue_calibration_execution_authorization",
    "require_valid_calibration_execution_completed_record",
    "require_valid_calibration_execution_genesis",
    "require_valid_calibration_execution_inventory",
    "require_valid_calibration_execution_started_record",
    "snapshot_calibration_execution_inventory",
    "validate_completed_calibration_ledger_snapshot",
]
