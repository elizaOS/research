"""Pure-stdlib, fail-closed ledger for one-shot development executions.

The primitive records only execution authority.  It does not import a runner,
create a campaign directory, issue a development root, execute a panel, write
an experiment artifact, assess a result, or promote scientific evidence.

On a local POSIX filesystem, creating ``started.json`` with ``O_EXCL`` is the
linearization point that consumes the sole attempt.  A crash after that point
leaves the attempt consumed even if the file is empty or incomplete.  Files
are never replaced, repaired, deleted, or retried by this module.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, cast

DEVELOPMENT_ONLY: Final = True
ROOT_ISSUANCE_AUTHORIZED: Final = False
PANEL_EXECUTION_AUTHORIZED: Final = False
EXPERIMENT_OUTPUT_WRITES_ALLOWED: Final = False
LEDGER_WRITES_ONLY: Final = True
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
RETRY_OR_RECOVERY_AUTHORIZED: Final = False
CROSS_PROCESS_REPLAY_PREVENTED_ON_LOCAL_POSIX_FILESYSTEM: Final = True

GENESIS_FILENAME: Final = "genesis.json"
STARTED_FILENAME: Final = "started.json"
TERMINAL_FILENAME: Final = "terminal.json"
GENESIS_SCHEMA: Final = "alberta.one-shot-development-ledger.genesis.v1"
STARTED_SCHEMA: Final = "alberta.one-shot-development-ledger.started.v1"
TERMINAL_SCHEMA: Final = "alberta.one-shot-development-ledger.terminal.v1"
MAX_RECORD_BYTES: Final = 65_536
_RECORD_FILENAMES: Final = frozenset(
    {GENESIS_FILENAME, STARTED_FILENAME, TERMINAL_FILENAME}
)

LedgerState = Literal["issued-unused", "consumed-pending", "consumed-terminal"]
TerminalStatus = Literal["completed", "failed"]


class LedgerError(RuntimeError):
    """Base class for a fail-closed ledger error."""


class AttemptAlreadyConsumedError(LedgerError):
    """Raised when ``started.json`` already exists in any form."""


def _exact_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be a non-empty exact string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _exact_string(value, name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


@dataclasses.dataclass(frozen=True, slots=True)
class OneShotDevelopmentLedgerSpec:
    """Complete immutable bindings needed before the sole attempt is issued."""

    campaign: str
    development_root: int
    development_root_hex: str
    protocol_config_sha256: str
    control_protocol_config_sha256: str
    runtime_config_sha256: str
    consumed_history_sha256: str
    key_manifest_sha256: str
    stream_sha256: str
    cadence_bound_stream_sha256: str
    source_envelope_sha256: str
    execution_source_closure_sha256: str
    bootstrap_sha256: str
    ledger_primitive_sha256: str
    declared_loader_sha256: str
    acknowledgement: str

    def __post_init__(self) -> None:
        _exact_string(self.campaign, "campaign")
        if type(self.development_root) is not int or not 0 <= self.development_root <= 0xFFFFFFFF:
            raise ValueError("development_root must be an exact unsigned 32-bit integer")
        if type(self.development_root_hex) is not str or self.development_root_hex != (
            f"0x{self.development_root:08X}"
        ):
            raise ValueError("development_root_hex does not encode development_root")
        for field in (
            "protocol_config_sha256",
            "control_protocol_config_sha256",
            "runtime_config_sha256",
            "consumed_history_sha256",
            "key_manifest_sha256",
            "stream_sha256",
            "cadence_bound_stream_sha256",
            "source_envelope_sha256",
            "execution_source_closure_sha256",
            "bootstrap_sha256",
            "ledger_primitive_sha256",
            "declared_loader_sha256",
        ):
            _sha256(getattr(self, field), field)
        own_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if self.ledger_primitive_sha256 != own_sha256:
            raise ValueError("ledger_primitive_sha256 does not bind the executing source bytes")
        _exact_string(self.acknowledgement, "acknowledgement")


def _validate_json_tree(value: object, path: str = "record") -> None:
    if value is None or type(value) in {str, int, bool}:
        return
    if type(value) is list:
        for index, item in enumerate(cast(list[object], value)):
            _validate_json_tree(item, f"{path}[{index}]")
        return
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            if type(key) is not str:
                raise TypeError(f"{path} contains a non-string key")
            _validate_json_tree(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains unsupported exact type {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return the ledger's exact ASCII JSON representation."""

    _validate_json_tree(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_json_sha256(value: object) -> str:
    """Return SHA-256 of the exact canonical JSON bytes."""

    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _seal_record(body: Mapping[str, object], hash_field: str) -> dict[str, object]:
    if type(body) is not dict:
        raise TypeError("record body must be an exact dict")
    if hash_field in body:
        raise ValueError(f"record body already contains {hash_field}")
    exact = dict(body)
    return {**exact, hash_field: canonical_json_sha256(exact)}


def genesis_record(spec: OneShotDevelopmentLedgerSpec) -> dict[str, object]:
    """Build the deterministic genesis record for an already chosen root."""

    body: dict[str, object] = {
        "schema": GENESIS_SCHEMA,
        "campaign": spec.campaign,
        "attempt_index": 1,
        "attempts_authorized": 1,
        "development_root": spec.development_root,
        "development_root_hex": spec.development_root_hex,
        "bindings": {
            "protocol_config_sha256": spec.protocol_config_sha256,
            "control_protocol_config_sha256": spec.control_protocol_config_sha256,
            "runtime_config_sha256": spec.runtime_config_sha256,
            "consumed_history_sha256": spec.consumed_history_sha256,
            "key_manifest_sha256": spec.key_manifest_sha256,
            "stream_sha256": spec.stream_sha256,
            "cadence_bound_stream_sha256": spec.cadence_bound_stream_sha256,
            "source_envelope_sha256": spec.source_envelope_sha256,
            "execution_source_closure_sha256": spec.execution_source_closure_sha256,
            "bootstrap_sha256": spec.bootstrap_sha256,
            "ledger_primitive_sha256": spec.ledger_primitive_sha256,
            "declared_loader_sha256": spec.declared_loader_sha256,
        },
        "acknowledgement": spec.acknowledgement,
        "authority": {
            "development_only": True,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
            "ledger_writes_only": True,
            "retry_or_recovery_authorized": False,
        },
    }
    return _seal_record(body, "genesis_sha256")


def started_record(spec: OneShotDevelopmentLedgerSpec) -> dict[str, object]:
    """Build the deterministic record written at the consumption point."""

    genesis = genesis_record(spec)
    body: dict[str, object] = {
        "schema": STARTED_SCHEMA,
        "campaign": spec.campaign,
        "attempt_index": 1,
        "attempts_authorized": 1,
        "attempts_consumed": 1,
        "root_consumed": True,
        "attempt_consumed_before_evaluator_import": True,
        "retry_or_recovery_authorized": False,
        "development_root": spec.development_root,
        "development_root_hex": spec.development_root_hex,
        "genesis_sha256": genesis["genesis_sha256"],
        "acknowledgement": spec.acknowledgement,
    }
    return _seal_record(body, "started_sha256")


def terminal_record(
    spec: OneShotDevelopmentLedgerSpec,
    *,
    status: TerminalStatus,
    panel_completed: bool,
    report_sha256: str | None,
    failure_sha256: str | None,
) -> dict[str, object]:
    """Build one deterministic terminal record after validation or failure."""

    if status not in {"completed", "failed"}:
        raise ValueError("status must be completed or failed")
    if type(panel_completed) is not bool:
        raise TypeError("panel_completed must be an exact bool")
    if status == "completed":
        if not panel_completed:
            raise ValueError("completed status requires a completed panel")
        report = _sha256(report_sha256, "report_sha256")
        if failure_sha256 is not None:
            raise ValueError("completed status cannot contain failure_sha256")
        failure = None
    else:
        if report_sha256 is not None:
            raise ValueError("failed status cannot contain report_sha256")
        report = None
        failure = _sha256(failure_sha256, "failure_sha256")
    genesis = genesis_record(spec)
    started = started_record(spec)
    body: dict[str, object] = {
        "schema": TERMINAL_SCHEMA,
        "campaign": spec.campaign,
        "attempt_index": 1,
        "attempts_authorized": 1,
        "attempts_consumed": 1,
        "root_consumed": True,
        "retry_or_recovery_authorized": False,
        "development_root": spec.development_root,
        "development_root_hex": spec.development_root_hex,
        "genesis_sha256": genesis["genesis_sha256"],
        "started_sha256": started["started_sha256"],
        "status": status,
        "panel_completed": panel_completed,
        "report_sha256": report,
        "failure_sha256": failure,
        "scientific_promotion_allowed": False,
        "evidence_authorized": False,
    }
    return _seal_record(body, "terminal_sha256")


def _validated_directory(directory: Path) -> Path:
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    if not directory.is_absolute():
        raise ValueError("directory must be absolute")
    if directory.is_symlink():
        raise ValueError("ledger directory must not be a symlink")
    try:
        resolved = directory.resolve(strict=True)
    except OSError as error:
        raise LedgerError(f"ledger directory cannot be resolved: {directory}") from error
    if resolved != directory or not resolved.is_dir():
        raise ValueError("ledger directory must be an exact resolved directory")
    return resolved


def _record_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    if candidate.is_symlink():
        raise LedgerError(f"ledger record must not be a symlink: {filename}")
    return candidate


def _require_known_entries(directory: Path) -> None:
    unexpected = sorted(
        entry.name for entry in directory.iterdir() if entry.name not in _RECORD_FILENAMES
    )
    if unexpected:
        raise LedgerError("ledger directory contains unexpected entries: " + ", ".join(unexpected))


def _exclusive_write(directory: Path, filename: str, record: Mapping[str, object]) -> None:
    payload = canonical_json(dict(record)).encode("ascii") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    path = _record_path(directory, filename)
    try:
        descriptor = os.open(path, flags, 0o400)
    except FileExistsError:
        raise
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("ledger write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _pairs_without_duplicates(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LedgerError(f"ledger record contains duplicate key: {key}")
        result[key] = value
    return result


def _read_record(directory: Path, filename: str) -> dict[str, object]:
    path = _record_path(directory, filename)
    try:
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise LedgerError(f"missing ledger record: {filename}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise LedgerError(f"ledger record is not a regular file: {filename}")
    if stat.S_IMODE(metadata.st_mode) != 0o444:
        raise LedgerError(f"ledger record permissions are not exact read-only mode: {filename}")
    if metadata.st_size <= 0 or metadata.st_size > MAX_RECORD_BYTES:
        raise LedgerError(f"ledger record has invalid byte length: {filename}")
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
        parsed = json.loads(text, object_pairs_hook=_pairs_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LedgerError(f"ledger record is not exact ASCII JSON: {filename}") from error
    if type(parsed) is not dict:
        raise LedgerError(f"ledger record is not an exact JSON object: {filename}")
    record = cast(dict[str, object], parsed)
    try:
        expected_bytes = canonical_json(record).encode("ascii") + b"\n"
    except (TypeError, ValueError) as error:
        raise LedgerError(f"ledger record has invalid values: {filename}") from error
    if raw != expected_bytes:
        raise LedgerError(f"ledger record bytes are not canonical: {filename}")
    return record


def initialize_genesis(directory: Path, spec: OneShotDevelopmentLedgerSpec) -> dict[str, object]:
    """Create genesis once in an already-created, otherwise-empty directory."""

    resolved = _validated_directory(directory)
    if any(resolved.iterdir()):
        raise LedgerError("genesis requires an otherwise-empty ledger directory")
    record = genesis_record(spec)
    try:
        _exclusive_write(resolved, GENESIS_FILENAME, record)
    except FileExistsError as error:
        raise LedgerError("genesis already exists and is never replaced") from error
    return record


def validate_genesis(directory: Path, spec: OneShotDevelopmentLedgerSpec) -> dict[str, object]:
    """Require byte-canonical genesis equal to the complete supplied spec."""

    resolved = _validated_directory(directory)
    _require_known_entries(resolved)
    observed = _read_record(resolved, GENESIS_FILENAME)
    expected = genesis_record(spec)
    if observed != expected:
        raise LedgerError("genesis record does not equal the supplied one-shot spec")
    return observed


def inspect_state(directory: Path) -> LedgerState:
    """Classify state by existence; malformed files remain fail-closed consumed."""

    resolved = _validated_directory(directory)
    _require_known_entries(resolved)
    genesis = _record_path(resolved, GENESIS_FILENAME)
    started = _record_path(resolved, STARTED_FILENAME)
    terminal = _record_path(resolved, TERMINAL_FILENAME)
    if not genesis.exists():
        raise LedgerError("ledger genesis is missing")
    if terminal.exists() and not started.exists():
        raise LedgerError("terminal record exists without a started record")
    if terminal.exists():
        return "consumed-terminal"
    if started.exists():
        return "consumed-pending"
    return "issued-unused"


def consume_attempt(
    directory: Path,
    spec: OneShotDevelopmentLedgerSpec,
    *,
    acknowledgement: str,
) -> dict[str, object]:
    """Atomically consume the sole attempt before any evaluator import."""

    if type(acknowledgement) is not str or acknowledgement != spec.acknowledgement:
        raise LedgerError("execution acknowledgement does not exactly match genesis")
    resolved = _validated_directory(directory)
    validate_genesis(resolved, spec)
    if _record_path(resolved, TERMINAL_FILENAME).exists():
        raise AttemptAlreadyConsumedError("terminal record proves the attempt was consumed")
    record = started_record(spec)
    try:
        _exclusive_write(resolved, STARTED_FILENAME, record)
    except FileExistsError as error:
        raise AttemptAlreadyConsumedError(
            "started record exists; the sole attempt remains consumed"
        ) from error
    return record


def validate_started(directory: Path, spec: OneShotDevelopmentLedgerSpec) -> dict[str, object]:
    """Validate a complete started record without weakening existence semantics."""

    resolved = _validated_directory(directory)
    validate_genesis(resolved, spec)
    observed = _read_record(resolved, STARTED_FILENAME)
    if observed != started_record(spec):
        raise LedgerError("started record does not equal the supplied one-shot spec")
    return observed


def record_terminal(
    directory: Path,
    spec: OneShotDevelopmentLedgerSpec,
    *,
    status: TerminalStatus,
    panel_completed: bool,
    report_sha256: str | None = None,
    failure_sha256: str | None = None,
) -> dict[str, object]:
    """Create the sole terminal record; never repair or replace one."""

    resolved = _validated_directory(directory)
    validate_started(resolved, spec)
    record = terminal_record(
        spec,
        status=status,
        panel_completed=panel_completed,
        report_sha256=report_sha256,
        failure_sha256=failure_sha256,
    )
    try:
        _exclusive_write(resolved, TERMINAL_FILENAME, record)
    except FileExistsError as error:
        raise LedgerError("terminal record already exists and is never replaced") from error
    return record


def validate_terminal(
    directory: Path,
    spec: OneShotDevelopmentLedgerSpec,
    *,
    status: TerminalStatus,
    panel_completed: bool,
    report_sha256: str | None = None,
    failure_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the terminal record against its complete expected outcome."""

    resolved = _validated_directory(directory)
    validate_started(resolved, spec)
    observed = _read_record(resolved, TERMINAL_FILENAME)
    expected = terminal_record(
        spec,
        status=status,
        panel_completed=panel_completed,
        report_sha256=report_sha256,
        failure_sha256=failure_sha256,
    )
    if observed != expected:
        raise LedgerError("terminal record does not equal the supplied one-shot outcome")
    return observed


__all__ = [
    "AttemptAlreadyConsumedError",
    "CROSS_PROCESS_REPLAY_PREVENTED_ON_LOCAL_POSIX_FILESYSTEM",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXPERIMENT_OUTPUT_WRITES_ALLOWED",
    "GENESIS_FILENAME",
    "GENESIS_SCHEMA",
    "LEDGER_WRITES_ONLY",
    "LedgerError",
    "LedgerState",
    "OneShotDevelopmentLedgerSpec",
    "PANEL_EXECUTION_AUTHORIZED",
    "RETRY_OR_RECOVERY_AUTHORIZED",
    "ROOT_ISSUANCE_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "STARTED_FILENAME",
    "STARTED_SCHEMA",
    "TERMINAL_FILENAME",
    "TERMINAL_SCHEMA",
    "TerminalStatus",
    "canonical_json",
    "canonical_json_sha256",
    "consume_attempt",
    "genesis_record",
    "initialize_genesis",
    "inspect_state",
    "record_terminal",
    "started_record",
    "terminal_record",
    "validate_genesis",
    "validate_started",
    "validate_terminal",
]
