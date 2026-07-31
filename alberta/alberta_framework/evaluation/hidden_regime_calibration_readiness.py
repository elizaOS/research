"""Fail-closed readiness receipts for hidden-regime factorial calibration.

This module cannot run a calibration.  It prepares a source/runtime-bound
draft, derives certification records by running an exact list of tests only
after explicit authorization, and publishes a finalized receipt only after a
second explicit authorization.  A future calibration runner may be bound, but
only when its source module already exists.

The receipt and its deterministic source ZIP are content addressed.  The ZIP
is executable source, not merely a hash list: a worker launched through
``execute_bound_calibration_worker`` starts in an empty directory with the ZIP
as the first and sole project source path, and rejects any project module whose
loader or ``__file__`` does not originate inside that ZIP.

Nothing here constructs a hidden-regime world, advances a calibration seed,
derives a protected seed namespace, observes a learner outcome, freezes a
threshold, or promotes a scientific claim.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import hmac
import importlib.metadata
import io
import json
import math
import os
import platform
import secrets
import stat
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn, cast

from alberta_framework.evaluation.hidden_regime_checkpoint import (
    HIDDEN_REGIME_CHECKPOINT_SCHEMA,
    HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_execution_governance import (
    CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
    MANAGED_EXECUTION_BOUNDARY_SCOPE,
    READINESS_EXECUTION_GOVERNANCE_FIELD,
    build_calibration_execution_genesis,
    calibration_execution_genesis_receipt_binding,
    require_valid_calibration_execution_genesis,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
    BOUND_PRIMITIVE_TRACE_SCHEMA,
    CALIBRATION_DESIGN_PAYLOAD_SHA256,
    CALIBRATION_READINESS_RECEIPT_SCHEMA,
    CONSUMED_CALIBRATION_NAMESPACE,
    DESIGN_ENVELOPE_SCHEMA,
    DESIGN_SCHEMA,
    N_MATCHED_CASES,
    PROTOCOL_STATUS,
    SEED_SNAPSHOT_SHA256,
    calibration_design_payload,
)
from alberta_framework.evaluation.hidden_regime_factorial_protocol import (
    canonical_sha256 as _protocol_canonical_sha256,
)
from alberta_framework.evaluation.hidden_regime_lineage_oracle import (
    HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_summary_oracle import (
    HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_trace_audit import (
    HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
    HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
)
from alberta_framework.evaluation.hidden_regime_world_oracle import (
    HIDDEN_REGIME_WORLD_ORACLE_SCHEMA,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    SLOT_ROLE_TRANSITION_ORACLE_SCHEMA,
)
from alberta_framework.streams.hidden_regime_signaling import (
    HIDDEN_REGIME_MANIFEST_USE_LEDGER,
    PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED,
    PROTECTED_CANDIDATE_PARTITION,
)

READINESS_SOURCE_SCHEMA = "alberta.hidden-regime-calibration.source-closure.v1"
READINESS_ARCHIVE_SCHEMA = "alberta.hidden-regime-calibration.source-archive.v1"
READINESS_RUNTIME_SCHEMA = "alberta.hidden-regime-calibration.runtime-identity.v1"
READINESS_CERTIFICATION_SCHEMA = "alberta.hidden-regime-calibration.certification.v1"
READINESS_ENVELOPE_SCHEMA = "alberta.hidden-regime-calibration.readiness-envelope.v1"
READINESS_STATUS = "calibration_ready_outcomes_unexecuted"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCK_FILES = (Path("pyproject.toml"), Path("uv.lock"))
_MAX_SOURCE_FILE_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = stat.S_IFREG | 0o444
_PROCESS_SEAL_KEY = secrets.token_bytes(32)
_CALIBRATION_RUNNER_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_calibration"
_EXECUTION_GOVERNANCE_MODULE = "alberta_framework.evaluation.hidden_regime_execution_governance"
_SUMMARY_ORACLE_MODULE = "alberta_framework.evaluation.hidden_regime_summary_oracle"
_ALLOWED_WORKER_ENTRYPOINT_MODES = (
    "--worker-case-v1",
    "--worker-preflight-v1",
)

_BASE_SOURCE_ROOT_MODULES = (
    "alberta_framework.evaluation.hidden_regime_calibration_readiness",
    "alberta_framework.evaluation.hidden_regime_factorial_protocol",
    "alberta_framework.evaluation.hidden_regime_signaling_development",
    "alberta_framework.evaluation.hidden_regime_trace_audit",
    "alberta_framework.evaluation.hidden_regime_checkpoint",
    "alberta_framework.evaluation.slot_signaling_lifecycle_oracle",
    "alberta_framework.evaluation.hidden_regime_world_oracle",
    "alberta_framework.evaluation.hidden_regime_lineage_oracle",
    _CALIBRATION_RUNNER_MODULE,
    _EXECUTION_GOVERNANCE_MODULE,
    _SUMMARY_ORACLE_MODULE,
)

_ENVIRONMENT_PREFIXES = (
    "JAX_",
    "XLA_",
    "CUDA_",
    "NVIDIA_",
    "OMP_",
    "MKL_",
    "TF_",
)
_ENVIRONMENT_NAMES = (
    "LD_LIBRARY_PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
)
_KEY_DISTRIBUTIONS = (
    "alberta-framework",
    "chex",
    "jax",
    "jaxlib",
    "numpy",
    "pytest",
    "scipy",
)


class ReadinessError(RuntimeError):
    """A readiness contract, integrity, or publication check failed."""


@dataclass(frozen=True, slots=True)
class CertificationSpec:
    """One exact, outcome-free readiness certification test group."""

    certification_id: str
    node_ids: tuple[str, ...]

    @property
    def semantic_command(self) -> tuple[str, ...]:
        """Return the portable command bound into a receipt."""

        return (
            "{runtime_python}",
            "-I",
            "-c",
            "{readiness_certification_harness_v1}",
            "{verified_extracted_source_root}",
            *self.node_ids,
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        )


CERTIFICATION_SPECS = (
    CertificationSpec(
        "complete_factorial_protocol_manifest_recurrence_gate_and_digest_contract",
        ("tests/test_hidden_regime_factorial_protocol.py",),
    ),
    CertificationSpec(
        "complete_development_producer_lineage_serialization_and_actual_transition_contract",
        ("tests/test_hidden_regime_signaling_development.py",),
    ),
    CertificationSpec(
        "complete_independent_generation_lineage_oracle",
        ("tests/test_hidden_regime_lineage_oracle.py",),
    ),
    CertificationSpec(
        "complete_role_world_summary_and_lineage_trace_audit",
        ("tests/test_hidden_regime_trace_audit.py",),
    ),
    CertificationSpec(
        "complete_independent_summary_and_resource_oracle",
        ("tests/test_hidden_regime_summary_oracle.py",),
    ),
    CertificationSpec(
        "managed_execution_authorization_consumption_and_protected_boundary",
        ("tests/test_hidden_regime_execution_governance.py",),
    ),
    CertificationSpec(
        "factorial_runner_worker_shard_ledger_coordinator_and_publication",
        ("tests/test_hidden_regime_factorial_calibration.py",),
    ),
    CertificationSpec(
        "checkpoint_resume_and_decentralized_role_bit_exact_equivalence",
        (
            "tests/test_hidden_regime_checkpoint.py",
            "tests/test_slot_signaling_lifecycle_oracle.py",
            "tests/test_hidden_regime_world_oracle.py",
            "tests/test_slot_signaling_agent.py",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ReadinessDraft:
    """Prepared source/protocol/runtime context, not an authorization receipt."""

    base_body: dict[str, object]
    source_archive: bytes
    repository_root: Path
    seal: str


@dataclass(frozen=True, slots=True)
class VerifiedCertificationBundle:
    """Process-sealed records emitted only by the certification verifier."""

    records: tuple[dict[str, object], ...]
    source_manifest_sha256: str
    runtime_identity_sha256: str
    protocol_payload_sha256: str
    seal: str


@dataclass(frozen=True, slots=True)
class PreparedReadinessReceipt:
    """Final canonical receipt bytes and the exact bound source archive."""

    payload: dict[str, object]
    source_archive: bytes
    repository_root: Path
    seal: str


@dataclass(frozen=True, slots=True)
class ReadinessValidation:
    """Fail-closed validation result."""

    valid: bool
    ready_for_calibration: bool
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedReadinessReceipt:
    """Paths of one immutable, content-addressed publication."""

    directory: Path
    receipt_path: Path
    source_archive_path: Path
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedReadinessBundle:
    """Strictly validated identities consumed by a bound calibration runner."""

    payload: dict[str, object]
    receipt_sha256: str
    source_archive_sha256: str
    source_manifest_sha256: str
    runtime_identity_sha256: str
    calibration_runner_module: str
    execution_genesis_sha256: str


def _fail(message: str) -> NoReturn:
    raise ReadinessError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def _is_strict_int(value: object) -> bool:
    return type(value) is int


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_value(value: object, *, location: str = "$") -> None:
    if value is None or type(value) in (str, int, bool):
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
    """Return the receipt's ASCII, integer-only canonical JSON encoding."""

    _validate_json_value(value)
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    """Hash canonical receipt JSON."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_loads(raw: bytes) -> dict[str, object]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReadinessError("receipt JSON is not ASCII") from exc

    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ReadinessError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ReadinessError(f"non-finite JSON constant is forbidden: {value}")

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ReadinessError("receipt JSON is invalid") from exc
    _require(type(parsed) is dict, "receipt JSON must contain one plain object")
    result = cast(dict[str, object], parsed)
    _require(raw == canonical_json_bytes(result), "receipt JSON bytes are not canonical")
    return result


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _expect_dict(value: object, label: str) -> dict[str, object]:
    _require(type(value) is dict, f"{label} must be a plain object")
    return cast(dict[str, object], value)


def _expect_list(value: object, label: str) -> list[object]:
    _require(type(value) is list, f"{label} must be a plain array")
    return cast(list[object], value)


def _expect_exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys differ from the exact schema")


def _seal(kind: str, payload: object, repository_root: Path) -> str:
    preimage = b"|".join(
        (
            kind.encode("ascii"),
            canonical_json_bytes(payload),
            os.fsencode(repository_root.absolute()),
        )
    )
    return hmac.new(_PROCESS_SEAL_KEY, preimage, hashlib.sha256).hexdigest()


def _read_source_file(path: Path, repository_root: Path) -> bytes:
    """Read one stable regular non-symlink source member under the repository root."""

    root = repository_root.resolve(strict=True)
    try:
        relative = path.relative_to(repository_root)
    except ValueError as exc:
        raise ReadinessError("source member is outside the repository root") from exc
    _require(".." not in relative.parts, "source member traverses a parent")
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise ReadinessError(f"source member is missing: {relative.as_posix()}") from exc
    _require(stat.S_ISREG(before.st_mode), f"source member is not regular: {relative.as_posix()}")
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(root), f"source member escapes root: {relative.as_posix()}")
    _require(before.st_size <= _MAX_SOURCE_FILE_BYTES, f"source member is too large: {relative}")
    raw = path.read_bytes()
    after = path.lstat()
    _require(
        (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ),
        f"source member changed while read: {relative.as_posix()}",
    )
    return raw


def _module_candidates(repository_root: Path, module: str) -> tuple[Path, Path]:
    relative = Path(*module.split("."))
    return (
        repository_root / relative.with_suffix(".py"),
        repository_root / relative / "__init__.py",
    )


def _module_path(repository_root: Path, module: str) -> Path | None:
    if module != "alberta_framework" and not module.startswith("alberta_framework."):
        return None
    for candidate in _module_candidates(repository_root, module):
        try:
            mode = candidate.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISREG(mode):
            return candidate
        _fail(f"local module is not a regular file: {candidate.relative_to(repository_root)}")
    return None


def _module_name(repository_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(repository_root)
    if relative.name == "__init__.py":
        return ".".join(relative.parent.parts), True
    return ".".join(relative.with_suffix("").parts), False


def _parent_packages(module: str) -> set[str]:
    parts = module.split(".")
    return {".".join(parts[:index]) for index in range(1, len(parts))}


def _resolve_local_imports(repository_root: Path, path: Path, raw: bytes) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=path.as_posix())
    except (SyntaxError, UnicodeError) as exc:
        raise ReadinessError(f"cannot parse source member {path}: {exc}") from exc
    module, is_package = _module_name(repository_root, path)
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("alberta_framework"):
                    _require(
                        _module_path(repository_root, alias.name) is not None,
                        f"local source import is missing: {alias.name}",
                    )
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
                if base.startswith("alberta_framework"):
                    _require(
                        _module_path(repository_root, base) is not None,
                        f"local source import is missing: {base}",
                    )
                candidates.append(base)
                candidates.extend(f"{base}.{alias.name}" for alias in node.names)
        for candidate in candidates:
            parts = candidate.split(".")
            while parts:
                possible = ".".join(parts)
                if _module_path(repository_root, possible) is not None:
                    found.add(possible)
                    found.update(_parent_packages(possible))
                    break
                parts.pop()
    return found


def _certification_source_paths() -> tuple[Path, ...]:
    paths = {Path("tests/conftest.py")}
    for spec in CERTIFICATION_SPECS:
        paths.update(Path(node_id.split("::", 1)[0]) for node_id in spec.node_ids)
    return tuple(sorted(paths, key=Path.as_posix))


def _validate_certification_node_sources(repository_root: Path) -> None:
    functions_by_path: dict[Path, set[str]] = {}
    for spec in CERTIFICATION_SPECS:
        for node_id in spec.node_ids:
            locator_text, separator, function_name = node_id.partition("::")
            _require(
                not separator or (separator == "::" and bool(function_name)),
                f"invalid node ID: {node_id}",
            )
            locator = Path(locator_text)
            if locator not in functions_by_path:
                raw = _read_source_file(repository_root / locator, repository_root)
                try:
                    tree = ast.parse(raw.decode("utf-8"), filename=locator.as_posix())
                except (SyntaxError, UnicodeError) as exc:
                    raise ReadinessError(f"cannot parse certification source {locator}") from exc
                functions_by_path[locator] = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                }
            if separator:
                _require(
                    function_name in functions_by_path[locator],
                    f"certification node is absent from its source: {node_id}",
                )
            else:
                _require(
                    any(name.startswith("test_") for name in functions_by_path[locator]),
                    f"full-file certification source contains no tests: {node_id}",
                )


def _build_source_bundle(
    repository_root: Path,
) -> tuple[dict[str, object], bytes]:
    repository_root = repository_root.absolute()
    _require(repository_root.is_dir(), "repository root must be a directory")
    _validate_certification_node_sources(repository_root)
    roots = _BASE_SOURCE_ROOT_MODULES
    _require(
        _module_path(repository_root, _CALIBRATION_RUNNER_MODULE) is not None,
        "mandatory calibration runner module does not exist",
    )
    _require(
        _module_path(repository_root, _EXECUTION_GOVERNANCE_MODULE) is not None,
        "mandatory execution governance module does not exist",
    )
    _require(
        _module_path(repository_root, _SUMMARY_ORACLE_MODULE) is not None,
        "mandatory independent summary oracle module does not exist",
    )

    pending = set(roots)
    pending.update(parent for module in roots for parent in _parent_packages(module))
    visited: set[str] = set()
    bytes_by_module: dict[str, bytes] = {}
    path_by_module: dict[str, Path] = {}
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        path = _module_path(repository_root, module)
        _require(path is not None, f"source closure module is missing: {module}")
        assert path is not None
        raw = _read_source_file(path, repository_root)
        visited.add(module)
        path_by_module[module] = path
        bytes_by_module[module] = raw
        pending.update(_resolve_local_imports(repository_root, path, raw) - visited)

    source_entries: list[dict[str, object]] = []
    archive_members: dict[str, bytes] = {}
    for module in sorted(visited):
        path = path_by_module[module]
        locator = path.relative_to(repository_root).as_posix()
        raw = bytes_by_module[module]
        source_entries.append(
            {
                "module": module,
                "locator": locator,
                "byte_size": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
        archive_members[locator] = raw

    support_entries: list[dict[str, object]] = []
    support_paths = (
        *((path, "dependency_lock") for path in _LOCK_FILES),
        *((path, "certification_source") for path in _certification_source_paths()),
    )
    for relative, role in support_paths:
        locator = relative.as_posix()
        _require(locator not in archive_members, f"duplicate archive member: {locator}")
        raw = _read_source_file(repository_root / relative, repository_root)
        support_entries.append(
            {
                "locator": locator,
                "role": role,
                "byte_size": len(raw),
                "sha256": _sha256_bytes(raw),
            }
        )
        archive_members[locator] = raw

    manifest: dict[str, object] = {
        "schema": READINESS_SOURCE_SCHEMA,
        "closure_kind": "static_transitive_local_python_imports",
        "repository_subtree": "research/alberta",
        "root_modules": list(roots),
        "calibration_runner_module": _CALIBRATION_RUNNER_MODULE,
        "files": source_entries,
        "support_files": support_entries,
    }
    archive = _deterministic_source_zip(archive_members)
    return manifest, archive


def _deterministic_source_zip(members: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as zf:
        zf.comment = b""
        for locator in sorted(members):
            pure = PurePosixPath(locator)
            _require(
                not pure.is_absolute()
                and ".." not in pure.parts
                and "\\" not in locator
                and pure.as_posix() == locator,
                f"unsafe ZIP member locator: {locator}",
            )
            info = zipfile.ZipInfo(locator, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _ZIP_FILE_MODE << 16
            info.extra = b""
            info.comment = b""
            buffer_data = members[locator]
            zf.writestr(info, buffer_data)
    raw = buffer.getvalue()
    _require(len(raw) <= _MAX_ARCHIVE_BYTES, "source archive exceeds the size limit")
    return raw


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _distribution_inventory_sha256() -> tuple[int, str]:
    entries: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or "unknown"
        version = distribution.version or "unknown"
        metadata_text = distribution.read_text("METADATA") or ""
        record_text = distribution.read_text("RECORD") or ""
        direct_url_text = distribution.read_text("direct_url.json") or ""
        entries.append(
            {
                "name": name.casefold().replace("_", "-"),
                "version": version,
                "metadata_sha256": _sha256_bytes(metadata_text.encode("utf-8")),
                "record_sha256": _sha256_bytes(record_text.encode("utf-8")),
                "direct_url_sha256": _sha256_bytes(direct_url_text.encode("utf-8")),
            }
        )
    entries.sort(key=lambda item: tuple(item.values()))
    return len(entries), canonical_sha256(entries)


def _runtime_value(value: object) -> str | int | bool | None:
    if value is None or type(value) in (str, int, bool):
        return cast(str | int | bool | None, value)
    if type(value) is float:
        _require(math.isfinite(value), "runtime configuration is non-finite")
        return repr(value)
    return repr(value)


def _build_runtime_identity() -> dict[str, object]:
    import jax

    inventory_count, inventory_sha256 = _distribution_inventory_sha256()
    environment_names = sorted(
        name
        for name in os.environ
        if name in _ENVIRONMENT_NAMES or name.startswith(_ENVIRONMENT_PREFIXES)
    )
    environment = [
        {
            "name": name,
            "present": True,
            "value_sha256": _sha256_bytes(os.environ[name].encode("utf-8", "surrogateescape")),
            "value_length": len(os.environ[name]),
        }
        for name in environment_names
    ]
    executable = Path(sys.executable).resolve(strict=True)
    config_values = {
        name: _runtime_value(value) for name, value in sorted(jax.config.values.items())
    }
    devices: list[dict[str, object]] = []
    for device in jax.devices():
        devices.append(
            {
                "id": int(device.id),
                "process_index": int(device.process_index),
                "platform": str(device.platform),
                "device_kind": str(device.device_kind),
                "local_hardware_id": int(getattr(device, "local_hardware_id", device.id)),
            }
        )
    return {
        "schema": READINESS_RUNTIME_SCHEMA,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "hexversion": sys.hexversion,
            "cache_tag": sys.implementation.cache_tag,
            "byteorder": sys.byteorder,
            "executable_sha256": _sha256_file(executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version_sha256": _sha256_bytes(platform.version().encode("utf-8")),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
            "cpu_count": os.cpu_count(),
        },
        "dependencies": {
            "key_versions": {name: _distribution_version(name) for name in _KEY_DISTRIBUTIONS},
            "installed_distribution_count": inventory_count,
            "installed_distribution_inventory_sha256": inventory_sha256,
        },
        "jax": {
            "default_backend": str(jax.default_backend()),
            "enable_x64": bool(jax.config.jax_enable_x64),
            "config_sha256": canonical_sha256(config_values),
            "devices": devices,
        },
        "environment": environment,
    }


def _protocol_binding() -> dict[str, object]:
    payload = calibration_design_payload()
    _require(
        _protocol_canonical_sha256(payload) == CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "factorial protocol payload differs from its frozen digest",
    )
    manifest_bindings = cast(list[object], payload["manifest_bindings"])
    recurrence_bindings = cast(list[object], payload["recurrence_eligibility_bindings"])
    gate_matrix_sha256 = payload["gate_matrix_sha256"]
    _require(_is_sha256(gate_matrix_sha256), "protocol gate matrix digest is invalid")
    return {
        "receipt_schema": CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "design_schema": DESIGN_SCHEMA,
        "design_envelope_schema": DESIGN_ENVELOPE_SCHEMA,
        "protocol_status": PROTOCOL_STATUS,
        "protocol_payload_sha256": CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": SEED_SNAPSHOT_SHA256,
        "manifest_bindings": manifest_bindings,
        "manifest_bindings_sha256": _protocol_canonical_sha256(manifest_bindings),
        "recurrence_eligibility_sha256": _protocol_canonical_sha256(recurrence_bindings),
        "gate_matrix_sha256": gate_matrix_sha256,
        "development_summary_schema": BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace_schema": BOUND_PRIMITIVE_TRACE_SCHEMA,
        "consumed_calibration_namespace_sha256": _sha256_bytes(
            CONSUMED_CALIBRATION_NAMESPACE.encode("ascii")
        ),
        "matched_case_count": N_MATCHED_CASES,
    }


def _component_schema_binding() -> dict[str, object]:
    return {
        "development_summary": BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace": BOUND_PRIMITIVE_TRACE_SCHEMA,
        "trace_audit_input": HIDDEN_REGIME_TRACE_AUDIT_INPUT_SCHEMA,
        "trace_audit_report": HIDDEN_REGIME_TRACE_AUDIT_REPORT_SCHEMA,
        "checkpoint": HIDDEN_REGIME_CHECKPOINT_SCHEMA,
        "trace_chunk": HIDDEN_REGIME_TRACE_CHUNK_SCHEMA,
        "role_lifecycle_oracle": SLOT_ROLE_TRANSITION_ORACLE_SCHEMA,
        "world_oracle": HIDDEN_REGIME_WORLD_ORACLE_SCHEMA,
        "lineage_oracle": HIDDEN_REGIME_LINEAGE_ORACLE_SCHEMA,
        "summary_oracle": HIDDEN_REGIME_SUMMARY_ORACLE_SCHEMA,
        "execution_genesis_binding": CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
    }


def _protected_guard() -> dict[str, object]:
    entries = tuple(
        entry
        for entry in HIDDEN_REGIME_MANIFEST_USE_LEDGER.values()
        if entry.use_partition == PROTECTED_CANDIDATE_PARTITION
    )
    _require(bool(entries), "protected-candidate ledger is empty")
    _require(
        PROTECTED_CANDIDATE_LEARNER_OUTCOMES_EXECUTED is False,
        "protected-candidate outcome constant is no longer false",
    )
    _require(
        all(entry.learner_outcomes_executed is False for entry in entries),
        "protected-candidate outcome ledger is no longer uniformly false",
    )
    return {
        "scope": "source_literals_only_not_managed_or_external_execution_history",
        "learner_outcome_constant": False,
        "ledger_all_false": True,
        "ledger_entry_count": len(entries),
        "execution_absence_attested": False,
    }


def _base_body(
    source_manifest: dict[str, object],
    source_archive: bytes,
    runtime_identity: dict[str, object],
    protocol_binding: dict[str, object],
) -> dict[str, object]:
    source_manifest_sha256 = canonical_sha256(source_manifest)
    archive_binding = {
        "schema": READINESS_ARCHIVE_SCHEMA,
        "format": "zip-stored-deterministic-v1",
        "file_name": "source.zip",
        "byte_size": len(source_archive),
        "sha256": _sha256_bytes(source_archive),
        "member_count": len(cast(list[object], source_manifest["files"]))
        + len(cast(list[object], source_manifest["support_files"])),
        "member_timestamp": list(_ZIP_TIMESTAMP),
        "member_mode_octal": "100444",
    }
    genesis = build_calibration_execution_genesis(
        source_archive_sha256=cast(str, archive_binding["sha256"]),
        source_manifest_sha256=source_manifest_sha256,
        runtime_identity_sha256=canonical_sha256(runtime_identity),
    )
    validated_genesis = require_valid_calibration_execution_genesis(genesis)
    governance_binding = calibration_execution_genesis_receipt_binding(validated_genesis)
    _require(
        governance_binding["protocol_payload_sha256"]
        == protocol_binding["protocol_payload_sha256"],
        "execution governance protocol binding differs",
    )
    _require(
        governance_binding["seed_snapshot_sha256"] == protocol_binding["seed_snapshot_sha256"],
        "execution governance seed binding differs",
    )
    runner = source_manifest["calibration_runner_module"]
    return {
        "receipt_schema": CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "envelope_schema": READINESS_ENVELOPE_SCHEMA,
        "status": READINESS_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "protocol_binding": protocol_binding,
        "component_schema_binding": _component_schema_binding(),
        "source_snapshot": {
            "manifest": source_manifest,
            "manifest_sha256": source_manifest_sha256,
            "archive": archive_binding,
        },
        "runtime_identity": runtime_identity,
        "runtime_identity_sha256": canonical_sha256(runtime_identity),
        READINESS_EXECUTION_GOVERNANCE_FIELD: governance_binding,
        "worker_execution": {
            "calibration_runner_module": runner,
            "entrypoint": "main" if runner is not None else None,
            "allowed_entrypoint_modes": list(_ALLOWED_WORKER_ENTRYPOINT_MODES),
            "isolated_flag": "-I",
            "working_directory": "fresh_empty_temporary_directory",
            "project_source_path": "content_addressed_source_zip_first_and_sole",
            "project_module_provenance_required": "zipimport_loader_and_file_inside_source_zip",
            "explicit_execution_authorization_required": True,
        },
        "source_literal_outcome_guard": _protected_guard(),
        "claim_scope": (
            "nonpromoting readiness for a finite consumed hidden-regime calibration factorial; "
            "not a calibration outcome, threshold freeze, protected evaluation, general "
            "continual-learning result, or Alberta Plan completion; a managed local execution "
            "ledger cannot prove that equivalent source or seeds were never executed in an "
            "external clone"
        ),
    }


def build_readiness_draft(
    *,
    repository_root: Path = _REPO_ROOT,
) -> ReadinessDraft:
    """Build a non-authorizing draft without running any certification or calibration."""

    root = repository_root.absolute()
    source_manifest, archive = _build_source_bundle(root)
    protocol_binding = _protocol_binding()
    runtime_identity = _build_runtime_identity()
    base_body = _base_body(source_manifest, archive, runtime_identity, protocol_binding)
    seal = _seal("readiness-draft-v1", base_body, root)
    return ReadinessDraft(base_body, archive, root, seal)


def _validate_draft_seal(draft: ReadinessDraft) -> None:
    expected = _seal("readiness-draft-v1", draft.base_body, draft.repository_root)
    _require(hmac.compare_digest(draft.seal, expected), "readiness draft seal is invalid")


_CERTIFICATION_BOOTSTRAP = r"""
import os
import sys

source_root, *pytest_argv = sys.argv[1:]
source_root = os.path.abspath(source_root)
if os.path.abspath(os.getcwd()) != source_root:
    raise SystemExit("certification cwd is not the extracted source root")

def contains_project(path):
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "alberta_framework"))
    )

sys.path[:] = [source_root] + [
    path
    for path in sys.path
    if path != source_root and not contains_project(path)
]
import pytest

exit_code = int(pytest.main(pytest_argv))
prefix = source_root + os.sep
for name, loaded in tuple(sys.modules.items()):
    if name != "alberta_framework" and not name.startswith("alberta_framework."):
        continue
    origin = getattr(loaded, "__file__", None)
    spec_origin = getattr(getattr(loaded, "__spec__", None), "origin", None)
    if not isinstance(origin, str) or not os.path.abspath(origin).startswith(prefix):
        raise SystemExit("certified project module origin is outside snapshot: " + name)
    if not isinstance(spec_origin, str) or not os.path.abspath(spec_origin).startswith(prefix):
        raise SystemExit("certified project module spec is outside snapshot: " + name)
if source_root not in sys.path:
    raise SystemExit("extracted snapshot left the certification source path")
for path in sys.path:
    if contains_project(path) and os.path.abspath(path) != source_root:
        raise SystemExit("mutable project path entered certification imports")
raise SystemExit(exit_code)
"""
_CERTIFICATION_BOOTSTRAP_SHA256 = _sha256_bytes(_CERTIFICATION_BOOTSTRAP.encode("utf-8"))


def _spec_payload() -> list[dict[str, object]]:
    return [
        {
            "certification_id": spec.certification_id,
            "node_ids": list(spec.node_ids),
            "command": list(spec.semantic_command),
            "harness_sha256": _CERTIFICATION_BOOTSTRAP_SHA256,
        }
        for spec in CERTIFICATION_SPECS
    ]


def _extract_verified_source_archive(draft: ReadinessDraft, destination: Path) -> None:
    source = _expect_dict(draft.base_body["source_snapshot"], "source snapshot")
    manifest = _validate_source_manifest_shape(source["manifest"])
    _validate_archive(draft.source_archive, manifest, source["archive"])
    _require(not destination.exists(), "source extraction destination already exists")
    destination.mkdir(mode=0o700)
    directories = {destination}
    with zipfile.ZipFile(io.BytesIO(draft.source_archive), "r") as zf:
        for info in zf.infolist():
            target = destination.joinpath(*PurePosixPath(info.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            directories.update((target.parent, *target.parents[:-1]))
            with target.open("xb") as handle:
                handle.write(zf.read(info))
            target.chmod(0o444)
    for directory in sorted(
        (item for item in directories if item.is_relative_to(destination)),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    _verify_extracted_source_tree(destination, manifest)


def _verify_extracted_source_tree(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    entries = [
        *_expect_list(manifest["files"], "source files"),
        *_expect_list(manifest["support_files"], "support files"),
    ]
    expected = {
        cast(str, _expect_dict(raw, "source entry")["locator"]): _expect_dict(raw, "source entry")
        for raw in entries
    }
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _require(actual_files == set(expected), "extracted certification source members differ")
    for locator, entry in expected.items():
        path = root / locator
        metadata = path.lstat()
        _require(stat.S_ISREG(metadata.st_mode), f"extracted source is not regular: {locator}")
        _require(
            stat.S_IMODE(metadata.st_mode) == 0o444, f"extracted source mode differs: {locator}"
        )
        raw = path.read_bytes()
        _require(len(raw) == entry["byte_size"], f"extracted source size differs: {locator}")
        _require(
            _sha256_bytes(raw) == entry["sha256"], f"extracted source digest differs: {locator}"
        )


def _make_extracted_tree_removable(root: Path) -> None:
    if not root.exists():
        return
    for directory, subdirectories, _files in os.walk(root, topdown=False):
        for subdirectory in subdirectories:
            Path(directory, subdirectory).chmod(0o700)
        Path(directory).chmod(0o700)


def run_readiness_certifications(
    draft: ReadinessDraft,
    *,
    authorize_certification_execution: bool,
    timeout_seconds_per_group: int = 1800,
) -> VerifiedCertificationBundle:
    """Run only the frozen certification node IDs and derive sealed records.

    This does not run a calibration runner.  The exact source, runtime,
    protocol, and uniformly-false protected ledger are checked before and after
    every subprocess group.
    """

    _require(
        authorize_certification_execution is True,
        "certification execution requires explicit authorization",
    )
    _require(
        _is_strict_int(timeout_seconds_per_group) and timeout_seconds_per_group > 0,
        "certification timeout must be a positive strict integer",
    )
    _validate_draft_seal(draft)
    source_snapshot = _expect_dict(draft.base_body["source_snapshot"], "source_snapshot")
    source_digest = cast(str, source_snapshot["manifest_sha256"])
    runtime_digest = cast(str, draft.base_body["runtime_identity_sha256"])
    protocol = _expect_dict(draft.base_body["protocol_binding"], "protocol_binding")
    protocol_digest = cast(str, protocol["protocol_payload_sha256"])
    current_manifest, current_archive = _build_source_bundle(draft.repository_root)
    _require(
        canonical_sha256(current_manifest) == source_digest,
        "source drift before certification",
    )
    _require(current_archive == draft.source_archive, "source archive drift before certification")
    _require(
        canonical_sha256(_build_runtime_identity()) == runtime_digest,
        "runtime drift before certification",
    )
    _require(_protocol_binding() == protocol, "protocol drift before certification")
    guard_before = _protected_guard()

    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="alberta-readiness-certification-") as temporary:
        extracted_root = Path(temporary) / "source"
        _extract_verified_source_archive(draft, extracted_root)
        try:
            for spec in CERTIFICATION_SPECS:
                actual_command = (
                    sys.executable,
                    "-I",
                    "-c",
                    _CERTIFICATION_BOOTSTRAP,
                    extracted_root.as_posix(),
                    *spec.semantic_command[5:],
                )
                environment = dict(os.environ)
                environment.pop("PYTHONPATH", None)
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                completed = subprocess.run(
                    actual_command,
                    cwd=extracted_root,
                    env=environment,
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds_per_group,
                )
                _verify_extracted_source_tree(extracted_root, current_manifest)
                stdout = bytes(completed.stdout)
                stderr = bytes(completed.stderr)
                status = "passed" if completed.returncode == 0 else "failed"
                record: dict[str, object] = {
                    "schema": READINESS_CERTIFICATION_SCHEMA,
                    "certification_id": spec.certification_id,
                    "node_ids": list(spec.node_ids),
                    "command": list(spec.semantic_command),
                    "harness_sha256": _CERTIFICATION_BOOTSTRAP_SHA256,
                    "status": status,
                    "exit_code": int(completed.returncode),
                    "stdout": {"byte_size": len(stdout), "sha256": _sha256_bytes(stdout)},
                    "stderr": {"byte_size": len(stderr), "sha256": _sha256_bytes(stderr)},
                    "source_manifest_sha256": source_digest,
                    "runtime_identity_sha256": runtime_digest,
                    "protocol_payload_sha256": protocol_digest,
                }
                if completed.returncode != 0:
                    _fail(
                        f"readiness certification failed: {spec.certification_id}; "
                        f"stdout={record['stdout']!r}; stderr={record['stderr']!r}"
                    )
                records.append(record)
        finally:
            _make_extracted_tree_removable(extracted_root)

    current_manifest, current_archive = _build_source_bundle(draft.repository_root)
    _require(
        canonical_sha256(current_manifest) == source_digest,
        "source drift after certification",
    )
    _require(current_archive == draft.source_archive, "source archive drift after certification")
    _require(
        canonical_sha256(_build_runtime_identity()) == runtime_digest,
        "runtime drift after certification",
    )
    _require(_protocol_binding() == protocol, "protocol drift after certification")
    _require(_protected_guard() == guard_before, "protected ledger changed during certification")

    record_tuple = tuple(records)
    seal_payload = {
        "records": list(record_tuple),
        "source_manifest_sha256": source_digest,
        "runtime_identity_sha256": runtime_digest,
        "protocol_payload_sha256": protocol_digest,
    }
    seal = _seal("readiness-certifications-v1", seal_payload, draft.repository_root)
    return VerifiedCertificationBundle(
        record_tuple,
        source_digest,
        runtime_digest,
        protocol_digest,
        seal,
    )


def _validate_certification_bundle(
    draft: ReadinessDraft,
    bundle: VerifiedCertificationBundle,
) -> None:
    payload = {
        "records": list(bundle.records),
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "runtime_identity_sha256": bundle.runtime_identity_sha256,
        "protocol_payload_sha256": bundle.protocol_payload_sha256,
    }
    expected_seal = _seal("readiness-certifications-v1", payload, draft.repository_root)
    _require(
        hmac.compare_digest(bundle.seal, expected_seal),
        "certification bundle seal is invalid",
    )
    source = _expect_dict(draft.base_body["source_snapshot"], "source_snapshot")
    protocol = _expect_dict(draft.base_body["protocol_binding"], "protocol_binding")
    _require(
        bundle.source_manifest_sha256 == source["manifest_sha256"],
        "certification source drift",
    )
    _require(
        bundle.runtime_identity_sha256 == draft.base_body["runtime_identity_sha256"],
        "certification runtime drift",
    )
    _require(
        bundle.protocol_payload_sha256 == protocol["protocol_payload_sha256"],
        "certification protocol drift",
    )
    _validate_certification_records(list(bundle.records), draft.base_body)


def _validate_certification_records(
    records_raw: list[object],
    base_body: Mapping[str, object],
) -> None:
    _require(len(records_raw) == len(CERTIFICATION_SPECS), "certification count differs")
    source = _expect_dict(base_body["source_snapshot"], "source_snapshot")
    protocol = _expect_dict(base_body["protocol_binding"], "protocol_binding")
    for index, (raw, spec) in enumerate(zip(records_raw, CERTIFICATION_SPECS, strict=True)):
        record = _expect_dict(raw, f"certifications[{index}]")
        _expect_exact_keys(
            record,
            {
                "schema",
                "certification_id",
                "node_ids",
                "command",
                "harness_sha256",
                "status",
                "exit_code",
                "stdout",
                "stderr",
                "source_manifest_sha256",
                "runtime_identity_sha256",
                "protocol_payload_sha256",
            },
            f"certifications[{index}]",
        )
        _require(record["schema"] == READINESS_CERTIFICATION_SCHEMA, "certification schema differs")
        _require(record["certification_id"] == spec.certification_id, "certification order differs")
        _require(record["node_ids"] == list(spec.node_ids), "certification node IDs differ")
        _require(record["command"] == list(spec.semantic_command), "certification command differs")
        _require(
            record["harness_sha256"] == _CERTIFICATION_BOOTSTRAP_SHA256,
            "certification harness digest differs",
        )
        _require(record["status"] == "passed", "certification status is not passed")
        _require(
            _is_strict_int(record["exit_code"]) and record["exit_code"] == 0,
            "certification exit code is not strict integer zero",
        )
        for stream_name in ("stdout", "stderr"):
            stream = _expect_dict(record[stream_name], f"certification {stream_name}")
            _expect_exact_keys(stream, {"byte_size", "sha256"}, stream_name)
            _require(
                _is_strict_int(stream["byte_size"]) and cast(int, stream["byte_size"]) >= 0,
                f"certification {stream_name} byte size is invalid",
            )
            _require(_is_sha256(stream["sha256"]), f"certification {stream_name} digest invalid")
        _require(
            record["source_manifest_sha256"] == source["manifest_sha256"],
            "certification source binding differs",
        )
        _require(
            record["runtime_identity_sha256"] == base_body["runtime_identity_sha256"],
            "certification runtime binding differs",
        )
        _require(
            record["protocol_payload_sha256"] == protocol["protocol_payload_sha256"],
            "certification protocol binding differs",
        )


def finalize_readiness_receipt(
    draft: ReadinessDraft,
    certifications: VerifiedCertificationBundle,
) -> PreparedReadinessReceipt:
    """Finalize an in-memory receipt; this function performs no publication."""

    _validate_draft_seal(draft)
    _validate_certification_bundle(draft, certifications)
    _require(
        _protected_guard() == draft.base_body["source_literal_outcome_guard"],
        "source literal guard drift before receipt finalization",
    )
    body = dict(draft.base_body)
    body["certification_contract"] = {
        "specifications": _spec_payload(),
        "specifications_sha256": canonical_sha256(_spec_payload()),
        "records": list(certifications.records),
        "records_sha256": canonical_sha256(list(certifications.records)),
        "all_required_certifications_passed": True,
    }
    body["authorization"] = {
        "ready_for_calibration": True,
        "calibration_execution_requires_separate_explicit_authorization": True,
        "calibration_outcomes_observed": False,
        "thresholds_frozen": False,
        "protected_candidate_execution_permitted": False,
        "scientific_promotion_permitted": False,
    }
    digest = canonical_sha256(body)
    payload: dict[str, object] = {
        "body": body,
        "receipt_sha256": digest,
    }
    seal = _seal("prepared-readiness-receipt-v1", payload, draft.repository_root)
    prepared = PreparedReadinessReceipt(payload, draft.source_archive, draft.repository_root, seal)
    validation = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=draft.repository_root,
        recheck_current=True,
        recheck_runtime=True,
    )
    _require(validation.valid, "; ".join(validation.errors))
    return prepared


def _validate_source_manifest_shape(manifest_raw: object) -> dict[str, object]:
    manifest = _expect_dict(manifest_raw, "source manifest")
    _expect_exact_keys(
        manifest,
        {
            "schema",
            "closure_kind",
            "repository_subtree",
            "root_modules",
            "calibration_runner_module",
            "files",
            "support_files",
        },
        "source manifest",
    )
    _require(manifest["schema"] == READINESS_SOURCE_SCHEMA, "source manifest schema differs")
    _require(
        manifest["closure_kind"] == "static_transitive_local_python_imports",
        "source closure kind differs",
    )
    _require(manifest["repository_subtree"] == "research/alberta", "source subtree differs")
    runner = manifest["calibration_runner_module"]
    _require(runner == _CALIBRATION_RUNNER_MODULE, "calibration runner binding is invalid")
    _require(
        manifest["root_modules"] == list(_BASE_SOURCE_ROOT_MODULES),
        "source root modules differ",
    )
    seen_locators: set[str] = set()
    file_items = _expect_list(manifest["files"], "source files")
    _require(bool(file_items), "source closure is empty")
    modules: list[str] = []
    for index, raw in enumerate(file_items):
        item = _expect_dict(raw, f"source files[{index}]")
        _expect_exact_keys(item, {"module", "locator", "byte_size", "sha256"}, "source file")
        module = item["module"]
        _require(type(module) is str and bool(module), "source module is invalid")
        module_text = cast(str, module)
        locator = cast(str, item["locator"])
        module_path = PurePosixPath(*module_text.split("."))
        _require(
            locator
            in {
                module_path.with_suffix(".py").as_posix(),
                (module_path / "__init__.py").as_posix(),
            },
            "source module and locator disagree",
        )
        modules.append(module_text)
        _validate_source_entry(item, seen_locators)
    _require(modules == sorted(modules), "source modules are not sorted")
    _require(len(modules) == len(set(modules)), "source modules are duplicated")
    support_items = _expect_list(manifest["support_files"], "support files")
    roles: list[tuple[str, str]] = []
    for index, raw in enumerate(support_items):
        item = _expect_dict(raw, f"support files[{index}]")
        _expect_exact_keys(item, {"locator", "role", "byte_size", "sha256"}, "support file")
        role = item["role"]
        _require(role in {"dependency_lock", "certification_source"}, "support role is invalid")
        _validate_source_entry(item, seen_locators)
        roles.append((cast(str, item["locator"]), cast(str, role)))
    expected_support = [
        *((path.as_posix(), "dependency_lock") for path in _LOCK_FILES),
        *((path.as_posix(), "certification_source") for path in _certification_source_paths()),
    ]
    _require(roles == expected_support, "support file set differs")
    return manifest


def _validate_source_entry(item: Mapping[str, object], seen: set[str]) -> None:
    locator = item["locator"]
    _require(type(locator) is str and bool(locator), "source locator is invalid")
    locator_text = cast(str, locator)
    pure = PurePosixPath(locator_text)
    _require(
        not pure.is_absolute()
        and ".." not in pure.parts
        and "\\" not in locator_text
        and pure.as_posix() == locator_text,
        "source locator is unsafe",
    )
    _require(locator_text not in seen, "source locator is duplicated")
    seen.add(locator_text)
    _require(
        _is_strict_int(item["byte_size"]) and cast(int, item["byte_size"]) >= 0,
        "source byte size is invalid",
    )
    _require(_is_sha256(item["sha256"]), "source digest is invalid")


def _archived_local_imports(
    module: str,
    locator: str,
    raw: bytes,
    available_modules: set[str],
) -> set[str]:
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=locator)
    except (SyntaxError, UnicodeError) as exc:
        raise ReadinessError(f"cannot parse archived source member {locator}") from exc
    is_package = locator.endswith("/__init__.py")
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()

    def include_exact(candidate: str, *, required: bool) -> None:
        if not candidate.startswith("alberta_framework"):
            return
        if candidate not in available_modules:
            _require(not required, f"archived local import is missing: {candidate}")
            return
        found.add(candidate)
        found.update(_parent_packages(candidate))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                include_exact(alias.name, required=alias.name.startswith("alberta_framework"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                package_parts = package.split(".") if package else []
                keep = len(package_parts) - node.level + 1
                _require(keep >= 0, f"invalid archived relative import in {locator}")
                base_parts = package_parts[:keep]
                if node.module:
                    base_parts.extend(node.module.split("."))
                base = ".".join(base_parts)
            else:
                base = node.module or ""
            if not base:
                continue
            include_exact(base, required=base.startswith("alberta_framework"))
            for alias in node.names:
                include_exact(f"{base}.{alias.name}", required=False)
    return found


def _validate_archived_source_closure(
    manifest: Mapping[str, object],
    members: Mapping[str, bytes],
) -> None:
    files = [
        _expect_dict(raw, "source file") for raw in _expect_list(manifest["files"], "source files")
    ]
    locator_by_module = {cast(str, item["module"]): cast(str, item["locator"]) for item in files}
    available = set(locator_by_module)
    pending = set(cast(list[str], manifest["root_modules"]))
    pending.update(parent for module in tuple(pending) for parent in _parent_packages(module))
    visited: set[str] = set()
    while pending:
        module = min(pending)
        pending.remove(module)
        if module in visited:
            continue
        _require(module in available, f"archived source root/import is missing: {module}")
        visited.add(module)
        locator = locator_by_module[module]
        pending.update(
            _archived_local_imports(
                module,
                locator,
                members[locator],
                available,
            )
            - visited
        )
    _require(visited == available, "archived source closure contains unreachable local modules")


def _validate_archive(
    archive: bytes,
    manifest: Mapping[str, object],
    archive_binding_raw: object,
) -> dict[str, bytes]:
    _require(len(archive) <= _MAX_ARCHIVE_BYTES, "source archive exceeds size limit")
    binding = _expect_dict(archive_binding_raw, "source archive binding")
    _expect_exact_keys(
        binding,
        {
            "schema",
            "format",
            "file_name",
            "byte_size",
            "sha256",
            "member_count",
            "member_timestamp",
            "member_mode_octal",
        },
        "source archive binding",
    )
    _require(binding["schema"] == READINESS_ARCHIVE_SCHEMA, "archive schema differs")
    _require(binding["format"] == "zip-stored-deterministic-v1", "archive format differs")
    _require(binding["file_name"] == "source.zip", "archive file name differs")
    _require(binding["byte_size"] == len(archive), "archive byte size differs")
    _require(binding["sha256"] == _sha256_bytes(archive), "archive digest differs")
    _require(binding["member_timestamp"] == list(_ZIP_TIMESTAMP), "archive timestamp differs")
    _require(binding["member_mode_octal"] == "100444", "archive member mode differs")

    entries = [
        *_expect_list(manifest["files"], "source files"),
        *_expect_list(manifest["support_files"], "support files"),
    ]
    expected = {
        cast(str, _expect_dict(raw, "source entry")["locator"]): _expect_dict(raw, "source entry")
        for raw in entries
    }
    _require(binding["member_count"] == len(expected), "archive member count differs")
    try:
        zf = zipfile.ZipFile(io.BytesIO(archive), "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ReadinessError("source archive is not a valid ZIP") from exc
    with zf:
        _require(zf.comment == b"", "source ZIP comment is not empty")
        infos = zf.infolist()
        names = [info.filename for info in infos]
        _require(names == sorted(expected), "source ZIP member order or set differs")
        _require(len(names) == len(set(names)), "source ZIP contains duplicate names")
        members: dict[str, bytes] = {}
        for info in infos:
            _require(info.date_time == _ZIP_TIMESTAMP, f"ZIP timestamp differs: {info.filename}")
            _require(info.compress_type == zipfile.ZIP_STORED, "ZIP compression is not stored")
            _require(info.create_system == 3, "ZIP creator system differs")
            _require(info.extra == b"" and info.comment == b"", "ZIP member metadata differs")
            _require((info.external_attr >> 16) == _ZIP_FILE_MODE, "ZIP member mode differs")
            entry = expected[info.filename]
            raw = zf.read(info)
            _require(info.file_size == entry["byte_size"] == len(raw), "ZIP member size differs")
            _require(_sha256_bytes(raw) == entry["sha256"], "ZIP member digest differs")
            members[info.filename] = raw
    _validate_archived_source_closure(manifest, members)
    return members


def _validate_runtime_shape(runtime_raw: object) -> dict[str, object]:
    runtime = _expect_dict(runtime_raw, "runtime identity")
    _expect_exact_keys(
        runtime,
        {"schema", "python", "platform", "dependencies", "jax", "environment"},
        "runtime identity",
    )
    _require(runtime["schema"] == READINESS_RUNTIME_SCHEMA, "runtime schema differs")
    python = _expect_dict(runtime["python"], "runtime python")
    _expect_exact_keys(
        python,
        {"implementation", "version", "hexversion", "cache_tag", "byteorder", "executable_sha256"},
        "runtime python",
    )
    _require(_is_sha256(python["executable_sha256"]), "runtime executable digest invalid")
    system = _expect_dict(runtime["platform"], "runtime platform")
    _expect_exact_keys(
        system,
        {"system", "release", "version_sha256", "machine", "libc", "cpu_count"},
        "runtime platform",
    )
    _require(_is_sha256(system["version_sha256"]), "platform version digest invalid")
    dependencies = _expect_dict(runtime["dependencies"], "runtime dependencies")
    _expect_exact_keys(
        dependencies,
        {"key_versions", "installed_distribution_count", "installed_distribution_inventory_sha256"},
        "runtime dependencies",
    )
    _require(
        _is_sha256(dependencies["installed_distribution_inventory_sha256"]),
        "dependency inventory digest invalid",
    )
    jax = _expect_dict(runtime["jax"], "runtime jax")
    _expect_exact_keys(jax, {"default_backend", "enable_x64", "config_sha256", "devices"}, "jax")
    _require(_is_sha256(jax["config_sha256"]), "JAX config digest invalid")
    _require(bool(_expect_list(jax["devices"], "JAX devices")), "JAX device list is empty")
    environment = _expect_list(runtime["environment"], "runtime environment")
    names: list[str] = []
    for raw in environment:
        item = _expect_dict(raw, "runtime environment item")
        _expect_exact_keys(item, {"name", "present", "value_sha256", "value_length"}, "environment")
        _require(type(item["name"]) is str, "environment name is invalid")
        _require(item["present"] is True, "environment presence marker differs")
        _require(_is_sha256(item["value_sha256"]), "environment value digest invalid")
        _require(_is_strict_int(item["value_length"]), "environment value length invalid")
        names.append(cast(str, item["name"]))
    _require(names == sorted(names) and len(names) == len(set(names)), "environment names differ")
    return runtime


def _validate_protocol_shape(protocol_raw: object) -> dict[str, object]:
    protocol = _expect_dict(protocol_raw, "protocol binding")
    _expect_exact_keys(
        protocol,
        {
            "receipt_schema",
            "design_schema",
            "design_envelope_schema",
            "protocol_status",
            "protocol_payload_sha256",
            "seed_snapshot_sha256",
            "manifest_bindings",
            "manifest_bindings_sha256",
            "recurrence_eligibility_sha256",
            "gate_matrix_sha256",
            "development_summary_schema",
            "primitive_trace_schema",
            "consumed_calibration_namespace_sha256",
            "matched_case_count",
        },
        "protocol binding",
    )
    for key in (
        "protocol_payload_sha256",
        "seed_snapshot_sha256",
        "manifest_bindings_sha256",
        "recurrence_eligibility_sha256",
        "gate_matrix_sha256",
        "consumed_calibration_namespace_sha256",
    ):
        _require(_is_sha256(protocol[key]), f"protocol digest is invalid: {key}")
    manifests = _expect_list(protocol["manifest_bindings"], "manifest bindings")
    _require(
        protocol["manifest_bindings_sha256"] == _protocol_canonical_sha256(manifests),
        "manifest binding digest differs",
    )
    _require(protocol["matched_case_count"] == N_MATCHED_CASES, "matched case count differs")
    return protocol


def _expected_execution_governance_binding(
    body: Mapping[str, object],
) -> dict[str, object]:
    source = _expect_dict(body["source_snapshot"], "source snapshot")
    archive = _expect_dict(source["archive"], "source archive")
    genesis = build_calibration_execution_genesis(
        source_archive_sha256=cast(str, archive["sha256"]),
        source_manifest_sha256=cast(str, source["manifest_sha256"]),
        runtime_identity_sha256=cast(str, body["runtime_identity_sha256"]),
    )
    validated = require_valid_calibration_execution_genesis(genesis)
    return calibration_execution_genesis_receipt_binding(validated)


def validate_readiness_receipt(
    payload: Mapping[str, object],
    source_archive: bytes,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = True,
    recheck_runtime: bool = True,
) -> ReadinessValidation:
    """Validate an in-memory receipt and source archive without running outcomes."""

    errors: list[str] = []
    try:
        _require(type(payload) is dict, "readiness payload must be a plain object")
        _expect_exact_keys(payload, {"body", "receipt_sha256"}, "readiness envelope")
        body = _expect_dict(payload["body"], "readiness body")
        _expect_exact_keys(
            body,
            {
                "receipt_schema",
                "envelope_schema",
                "status",
                "development_only",
                "scientific_promotion_allowed",
                "protocol_binding",
                "component_schema_binding",
                "source_snapshot",
                "runtime_identity",
                "runtime_identity_sha256",
                READINESS_EXECUTION_GOVERNANCE_FIELD,
                "worker_execution",
                "source_literal_outcome_guard",
                "claim_scope",
                "certification_contract",
                "authorization",
            },
            "readiness body",
        )
        _require(
            body["receipt_schema"] == CALIBRATION_READINESS_RECEIPT_SCHEMA,
            "receipt schema differs",
        )
        _require(body["envelope_schema"] == READINESS_ENVELOPE_SCHEMA, "envelope schema differs")
        _require(body["status"] == READINESS_STATUS, "readiness status differs")
        _require(body["development_only"] is True, "receipt must remain development-only")
        _require(body["scientific_promotion_allowed"] is False, "receipt cannot promote")
        _require(_is_sha256(payload["receipt_sha256"]), "receipt digest is invalid")
        _require(payload["receipt_sha256"] == canonical_sha256(body), "receipt body digest differs")

        protocol = _validate_protocol_shape(body["protocol_binding"])
        _require(
            body["component_schema_binding"] == _component_schema_binding(),
            "component schema binding differs",
        )
        source = _expect_dict(body["source_snapshot"], "source snapshot")
        _expect_exact_keys(source, {"manifest", "manifest_sha256", "archive"}, "source snapshot")
        manifest = _validate_source_manifest_shape(source["manifest"])
        _require(
            source["manifest_sha256"] == canonical_sha256(manifest),
            "source manifest digest differs",
        )
        _validate_archive(source_archive, manifest, source["archive"])
        runtime = _validate_runtime_shape(body["runtime_identity"])
        _require(
            body["runtime_identity_sha256"] == canonical_sha256(runtime),
            "runtime identity digest differs",
        )
        governance = _expect_dict(
            body[READINESS_EXECUTION_GOVERNANCE_FIELD],
            READINESS_EXECUTION_GOVERNANCE_FIELD,
        )
        _require(
            governance.get("schema") == CALIBRATION_EXECUTION_GENESIS_RECEIPT_BINDING_SCHEMA,
            "execution governance binding schema differs",
        )
        _require(
            governance == _expected_execution_governance_binding(body),
            "execution governance binding differs from pristine deterministic genesis",
        )
        _require(
            governance.get("managed_boundary_scope") == MANAGED_EXECUTION_BOUNDARY_SCOPE,
            "execution governance boundary scope differs",
        )
        worker = _expect_dict(body["worker_execution"], "worker execution")
        _expect_exact_keys(
            worker,
            {
                "calibration_runner_module",
                "entrypoint",
                "allowed_entrypoint_modes",
                "isolated_flag",
                "working_directory",
                "project_source_path",
                "project_module_provenance_required",
                "explicit_execution_authorization_required",
            },
            "worker execution",
        )
        runner = manifest["calibration_runner_module"]
        _require(worker["calibration_runner_module"] == runner, "worker runner binding differs")
        _require(
            worker["entrypoint"] == ("main" if runner is not None else None),
            "entrypoint differs",
        )
        _require(
            worker["allowed_entrypoint_modes"] == list(_ALLOWED_WORKER_ENTRYPOINT_MODES),
            "allowed worker entrypoint modes differ",
        )
        _require(worker["isolated_flag"] == "-I", "worker isolation flag differs")
        _require(
            worker["working_directory"] == "fresh_empty_temporary_directory",
            "worker working directory contract differs",
        )
        _require(
            worker["project_source_path"] == "content_addressed_source_zip_first_and_sole",
            "worker project source path contract differs",
        )
        _require(
            worker["project_module_provenance_required"]
            == "zipimport_loader_and_file_inside_source_zip",
            "worker provenance contract differs",
        )
        _require(worker["explicit_execution_authorization_required"] is True, "worker auth differs")

        guard = _expect_dict(body["source_literal_outcome_guard"], "source literal guard")
        _expect_exact_keys(
            guard,
            {
                "scope",
                "learner_outcome_constant",
                "ledger_all_false",
                "ledger_entry_count",
                "execution_absence_attested",
            },
            "source literal guard",
        )
        _require(
            guard["scope"] == "source_literals_only_not_managed_or_external_execution_history"
            and guard["learner_outcome_constant"] is False
            and guard["ledger_all_false"] is True
            and guard["execution_absence_attested"] is False,
            "source literal guard overclaims execution history",
        )
        certification = _expect_dict(body["certification_contract"], "certification contract")
        _expect_exact_keys(
            certification,
            {
                "specifications",
                "specifications_sha256",
                "records",
                "records_sha256",
                "all_required_certifications_passed",
            },
            "certification contract",
        )
        _require(certification["specifications"] == _spec_payload(), "certification specs differ")
        _require(
            certification["specifications_sha256"] == canonical_sha256(_spec_payload()),
            "certification specification digest differs",
        )
        records = _expect_list(certification["records"], "certification records")
        _validate_certification_records(records, body)
        _require(
            certification["records_sha256"] == canonical_sha256(records),
            "certification records digest differs",
        )
        _require(
            certification["all_required_certifications_passed"] is True,
            "certifications incomplete",
        )
        authorization = _expect_dict(body["authorization"], "authorization")
        _expect_exact_keys(
            authorization,
            {
                "ready_for_calibration",
                "calibration_execution_requires_separate_explicit_authorization",
                "calibration_outcomes_observed",
                "thresholds_frozen",
                "protected_candidate_execution_permitted",
                "scientific_promotion_permitted",
            },
            "authorization",
        )
        _require(
            authorization
            == {
                "ready_for_calibration": True,
                "calibration_execution_requires_separate_explicit_authorization": True,
                "calibration_outcomes_observed": False,
                "thresholds_frozen": False,
                "protected_candidate_execution_permitted": False,
                "scientific_promotion_permitted": False,
            },
            "authorization policy differs",
        )
        if recheck_current:
            root = repository_root.absolute()
            current_manifest, current_archive = _build_source_bundle(root)
            _require(
                current_manifest == manifest,
                "source closure no longer matches current source",
            )
            _require(
                current_archive == source_archive,
                "source archive no longer matches current source",
            )
            _require(
                _protocol_binding() == protocol,
                "protocol binding no longer matches current source",
            )
            _require(_protected_guard() == guard, "protected ledger no longer matches receipt")
        if recheck_runtime:
            _require(
                _build_runtime_identity() == runtime,
                "runtime/JAX/device/dependency/environment identity drift",
            )
    except (ReadinessError, KeyError, TypeError, ValueError, OSError) as exc:
        errors.append(str(exc))
    return ReadinessValidation(not errors, not errors, tuple(errors))


def require_validated_readiness_receipt(
    payload: Mapping[str, object],
    source_archive: bytes,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = False,
    recheck_runtime: bool = True,
) -> ValidatedReadinessBundle:
    """Return runner-consumable identities or raise on any receipt/ZIP defect.

    ``recheck_current=False`` is intentional for an isolated worker: the exact
    executable checkout is the already-validated ZIP, not whatever mutable
    checkout happens to exist on the host.  Callers issuing or publishing a
    receipt use ``recheck_current=True`` instead.
    """

    validation = validate_readiness_receipt(
        payload,
        source_archive,
        repository_root=repository_root,
        recheck_current=recheck_current,
        recheck_runtime=recheck_runtime,
    )
    _require(validation.valid, "; ".join(validation.errors))
    normalized = dict(payload)
    body = _expect_dict(normalized["body"], "readiness body")
    source = _expect_dict(body["source_snapshot"], "source snapshot")
    archive = _expect_dict(source["archive"], "source archive")
    worker = _expect_dict(body["worker_execution"], "worker execution")
    governance = _expect_dict(
        body[READINESS_EXECUTION_GOVERNANCE_FIELD],
        READINESS_EXECUTION_GOVERNANCE_FIELD,
    )
    return ValidatedReadinessBundle(
        payload=normalized,
        receipt_sha256=cast(str, normalized["receipt_sha256"]),
        source_archive_sha256=cast(str, archive["sha256"]),
        source_manifest_sha256=cast(str, source["manifest_sha256"]),
        runtime_identity_sha256=cast(str, body["runtime_identity_sha256"]),
        calibration_runner_module=cast(str, worker["calibration_runner_module"]),
        execution_genesis_sha256=cast(str, governance["genesis_sha256"]),
    )


def _open_directory_without_symlinks(path: Path) -> tuple[int, Path]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    raise ReadinessError(
                        f"symlinked readiness directory is forbidden: {absolute}"
                    ) from exc
                raise
            os.close(descriptor)
            descriptor = next_descriptor
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, absolute


def _open_immutable_regular(path: Path, *, max_bytes: int) -> bytes:
    parent_fd, absolute_parent = _open_directory_without_symlinks(path.parent)
    absolute = absolute_parent / path.name
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ReadinessError(f"symlinked readiness path is forbidden: {absolute}") from exc
        raise
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"readiness path is not regular: {path}")
        _require(stat.S_IMODE(before.st_mode) == 0o444, f"readiness file mode is not 0444: {path}")
        _require(before.st_nlink == 1, f"readiness file has multiple hard links: {path}")
        _require(before.st_size <= max_bytes, f"readiness file exceeds size limit: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= max_bytes, f"readiness file exceeds size limit: {path}")
        after = os.fstat(descriptor)
        locator = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (locator.st_dev, locator.st_ino, locator.st_size, locator.st_mtime_ns),
            f"readiness file changed or was replaced during read: {path}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def _write_new_immutable(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "short write while publishing readiness file")
            view = view[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_readiness_receipt(
    prepared: PreparedReadinessReceipt,
    publication_root: Path,
    *,
    authorize_publication: bool,
) -> PublishedReadinessReceipt:
    """Publish ``<root>/<digest>/{readiness.json,source.zip}`` new-only and 0444."""

    _require(authorize_publication is True, "readiness publication requires explicit authorization")
    expected_seal = _seal(
        "prepared-readiness-receipt-v1",
        prepared.payload,
        prepared.repository_root,
    )
    _require(hmac.compare_digest(prepared.seal, expected_seal), "prepared receipt seal is invalid")
    validation = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=prepared.repository_root,
        recheck_current=True,
        recheck_runtime=True,
    )
    _require(validation.valid, "; ".join(validation.errors))
    digest = cast(str, prepared.payload["receipt_sha256"])
    root_fd, root = _open_directory_without_symlinks(publication_root)
    flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd: int | None = None
    try:
        try:
            os.mkdir(digest, 0o700, dir_fd=root_fd)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite readiness receipt: {root / digest}"
            ) from exc
        directory_fd = os.open(digest, flags, dir_fd=root_fd)
        _write_new_immutable(directory_fd, "readiness.json", canonical_json_bytes(prepared.payload))
        _write_new_immutable(directory_fd, "source.zip", prepared.source_archive)
        os.fsync(directory_fd)
        os.chmod(digest, 0o555, dir_fd=root_fd, follow_symlinks=False)
        os.fsync(root_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(root_fd)
    directory = root / digest
    return PublishedReadinessReceipt(
        directory,
        directory / "readiness.json",
        directory / "source.zip",
        digest,
    )


def validate_published_readiness_receipt(
    directory: Path,
    *,
    repository_root: Path = _REPO_ROOT,
    recheck_current: bool = True,
    recheck_runtime: bool = True,
) -> ReadinessValidation:
    """Validate immutable modes, content address, canonical JSON, ZIP, and drift."""

    errors: list[str] = []
    try:
        absolute = directory.absolute()
        mode = absolute.lstat().st_mode
        _require(stat.S_ISDIR(mode), "readiness publication path is not a directory")
        _require(stat.S_IMODE(mode) == 0o555, "readiness publication directory mode is not 0555")
        _require(
            sorted(path.name for path in absolute.iterdir()) == ["readiness.json", "source.zip"],
            "readiness publication members differ",
        )
        receipt_raw = _open_immutable_regular(
            absolute / "readiness.json",
            max_bytes=_MAX_RECEIPT_BYTES,
        )
        archive = _open_immutable_regular(absolute / "source.zip", max_bytes=_MAX_ARCHIVE_BYTES)
        payload = _strict_json_loads(receipt_raw)
        digest = payload.get("receipt_sha256")
        _require(
            type(digest) is str and absolute.name == digest,
            "content-addressed directory differs",
        )
        validation = validate_readiness_receipt(
            payload,
            archive,
            repository_root=repository_root,
            recheck_current=recheck_current,
            recheck_runtime=recheck_runtime,
        )
        _require(validation.valid, "; ".join(validation.errors))
    except (ReadinessError, FileNotFoundError, NotADirectoryError, OSError, ValueError) as exc:
        errors.append(str(exc))
    return ReadinessValidation(not errors, not errors, tuple(errors))


_BOUND_WORKER_BOOTSTRAP = r"""
import hashlib
import importlib
import json
import os
import sys
import zipimport

(
    archive,
    receipt_path,
    expected_receipt,
    expected_archive,
    module_name,
    entrypoint,
    *argv,
) = sys.argv[1:]
archive = os.path.abspath(archive)
receipt_path = os.path.abspath(receipt_path)
if hashlib.sha256(open(archive, "rb").read()).hexdigest() != expected_archive:
    raise SystemExit("bound source archive digest mismatch")
payload = json.loads(open(receipt_path, "rb").read())
if payload.get("receipt_sha256") != expected_receipt:
    raise SystemExit("bound readiness receipt digest mismatch")

def contains_project(path):
    return (
        bool(path)
        and os.path.isdir(path)
        and os.path.exists(os.path.join(path, "alberta_framework"))
    )

sys.path[:] = [archive] + [
    path for path in sys.path if path != archive and not contains_project(path)
]
module = importlib.import_module(module_name)
prefix = archive + os.sep
for name, loaded in tuple(sys.modules.items()):
    if name != "alberta_framework" and not name.startswith("alberta_framework."):
        continue
    origin = getattr(loaded, "__file__", None)
    loader = getattr(loaded, "__loader__", None)
    if not isinstance(loader, zipimport.zipimporter):
        raise SystemExit("project module loader is not zipimport: " + name)
    if not isinstance(origin, str) or not os.path.abspath(origin).startswith(prefix):
        raise SystemExit("project module origin is outside bound ZIP: " + name)
if sys.path[0] != archive or any(contains_project(path) for path in sys.path[1:]):
    raise SystemExit("bound ZIP is not the first and sole project source path")
target = getattr(module, entrypoint, None)
if not callable(target):
    raise SystemExit("bound calibration entrypoint is not callable")
result = target(argv)
raise SystemExit(0 if result is None else int(result))
"""


def execute_bound_calibration_worker(
    directory: Path,
    arguments: Sequence[str] = (),
    *,
    authorize_calibration_execution: bool,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Explicitly execute a bound future runner from the ZIP, never the checkout."""

    _require(
        authorize_calibration_execution is True,
        "calibration worker execution requires explicit authorization",
    )
    _require(
        timeout_seconds is None or (_is_strict_int(timeout_seconds) and timeout_seconds > 0),
        "worker timeout must be null or a positive strict integer",
    )
    validation = validate_published_readiness_receipt(
        directory,
        recheck_current=False,
        recheck_runtime=True,
    )
    _require(validation.valid, "; ".join(validation.errors))
    receipt_path = directory.absolute() / "readiness.json"
    archive_path = directory.absolute() / "source.zip"
    payload = _strict_json_loads(
        _open_immutable_regular(receipt_path, max_bytes=_MAX_RECEIPT_BYTES)
    )
    body = _expect_dict(payload["body"], "readiness body")
    worker = _expect_dict(body["worker_execution"], "worker execution")
    module = worker["calibration_runner_module"]
    entrypoint = worker["entrypoint"]
    _require(type(module) is str and type(entrypoint) is str, "receipt binds no calibration runner")
    allowed_modes = _expect_list(worker["allowed_entrypoint_modes"], "allowed worker modes")
    _require(bool(arguments), "bound calibration worker requires an entrypoint mode")
    _require(
        arguments[0] in allowed_modes,
        "bound calibration worker entrypoint mode is not allowed by the receipt",
    )
    source = _expect_dict(body["source_snapshot"], "source snapshot")
    archive_binding = _expect_dict(source["archive"], "archive binding")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="alberta-hidden-regime-worker-") as temporary:
        _require(not any(Path(temporary).iterdir()), "worker temporary directory is not empty")
        command = (
            sys.executable,
            "-I",
            "-c",
            _BOUND_WORKER_BOOTSTRAP,
            archive_path.as_posix(),
            receipt_path.as_posix(),
            cast(str, payload["receipt_sha256"]),
            cast(str, archive_binding["sha256"]),
            cast(str, module),
            cast(str, entrypoint),
            *tuple(arguments),
        )
        return subprocess.run(
            command,
            cwd=temporary,
            env=environment,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )


__all__ = [
    "CALIBRATION_READINESS_RECEIPT_SCHEMA",
    "CERTIFICATION_SPECS",
    "PreparedReadinessReceipt",
    "PublishedReadinessReceipt",
    "READINESS_ARCHIVE_SCHEMA",
    "READINESS_CERTIFICATION_SCHEMA",
    "READINESS_ENVELOPE_SCHEMA",
    "READINESS_RUNTIME_SCHEMA",
    "READINESS_SOURCE_SCHEMA",
    "ReadinessDraft",
    "ReadinessError",
    "ReadinessValidation",
    "ValidatedReadinessBundle",
    "VerifiedCertificationBundle",
    "build_readiness_draft",
    "canonical_json_bytes",
    "canonical_sha256",
    "execute_bound_calibration_worker",
    "finalize_readiness_receipt",
    "publish_readiness_receipt",
    "require_validated_readiness_receipt",
    "run_readiness_certifications",
    "validate_published_readiness_receipt",
    "validate_readiness_receipt",
]
