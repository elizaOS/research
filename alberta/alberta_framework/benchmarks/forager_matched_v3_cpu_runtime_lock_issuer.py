"""Pure-content issuance bridge for a matched-v3 CPU runtime lock.

The bridge consumes only caller-supplied canonical bytes.  It delegates capture
manifest and wheelhouse receipt validation to their final parsers, maps their
retained identities into the runtime-lock contract, and emits canonical lock and
CAS-manifest bytes.  It performs no filesystem, network, archive, wheel, install,
execution, qualification, evidence, or promotion operation.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Never, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_runtime_lock as runtime_lock_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_wheelhouse as wheelhouse_contract,
)

CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_runtime_lock_issuance_envelope.v1"
)
CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_STATUS: Final = (
    "canonical_issuance_inputs_unexecuted_non_authorizing"
)
CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_CLASSIFICATION: Final = (
    "pure_content_provenance_envelope_non_authorizing"
)
CPU_RUNTIME_LOCK_ISSUANCE_RANGE_LIMITATION: Final = (
    "Ranged overlay dependency satisfaction is carried by the exact solver provenance and "
    "is not independently re-evaluated by this pure-content bridge."
)
CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_STATUS: Final = (
    "canonical_cas_mapping_unexecuted_non_authorizing"
)
CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_CLASSIFICATION: Final = (
    "pure_content_wheelhouse_cas_mapping_non_authorizing"
)

PRODUCTION_ROOT_PIN_COUNT: Final = 36

_MAX_ARTIFACT_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 40
_MAX_JSON_NODES: Final = 1_000_000
_MAX_TEXT_LENGTH: Final = 64 * 1024
_MAX_INTEGER: Final = 2**63 - 1
_MAX_PACKAGES: Final = 10_000

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_NAME_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_EXTRA_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_CAS_KEY_RE: Final = re.compile(
    r"sha256/(?P<prefix>[0-9a-f]{2})/(?P<sha256>[0-9a-f]{64})/"
    r"(?P<filename>[^/]+\.whl)\Z"
)
_REQUIREMENT_BODY_RE: Final = re.compile(
    r"\s*(?P<name>[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)"
    r"(?:\[(?P<extras>[A-Za-z0-9_.-]+(?:\s*,\s*[A-Za-z0-9_.-]+)*)\])?"
    r"\s*(?:\(\s*(?P<parenthesized>[^()]*)\s*\)|(?P<plain>[^()]*))?\s*\Z"
)
_MARKER_TOKEN_RE: Final = re.compile(
    r"\s*(?:"
    r"(?P<variable>implementation_name|implementation_version|os_name|platform_machine|"
    r"platform_python_implementation|platform_release|platform_system|platform_version|"
    r"python_full_version|python_version|sys_platform|extra)|"
    r"(?P<string>'[^'\\]*'|\"[^\"\\]*\")|"
    r"(?P<operator>===|==|!=|~=|<=|>=|<|>|not\s+in\b|in\b)|"
    r"(?P<boolean>and\b|or\b)|(?P<left>\()|(?P<right>\))"
    r")"
)
_TIMESTAMP_RE: Final = re.compile(
    r"(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
_PYPI_SOURCE_URL_RE: Final = re.compile(
    r"https://files\.pythonhosted\.org/packages/[0-9a-f]{2}/[0-9a-f]{2}/"
    r"[0-9a-f]{60}/(?P<filename>[A-Za-z0-9_.+-]+\.whl)\Z"
)


class ForagerMatchedV3CpuRuntimeLockIssuerError(ValueError):
    """A pure-content issuance input or derived artifact failed closed."""


@dataclass(frozen=True, slots=True)
class CpuRuntimeLockIssuanceArtifacts:
    """Detached canonical artifacts returned by one pure-content issuance."""

    runtime_lock_bytes: bytes
    runtime_lock_sha256: str
    cas_manifest_bytes: bytes
    cas_manifest_sha256: str
    issuance_envelope_bytes: bytes
    issuance_envelope_sha256: str
    capture_manifest_bytes: bytes
    capture_manifest_sha256: str
    wheelhouse_receipt_bytes: bytes
    wheelhouse_receipt_sha256: str
    root_pin_count: int
    root_pin_inventory_sha256: str


def _fail(message: str) -> Never:
    raise ForagerMatchedV3CpuRuntimeLockIssuerError(message)


def _raise_float(value: str) -> Never:
    _fail(f"issuer JSON contains a float {value!r}")


def _raise_constant(value: str) -> Never:
    _fail(f"issuer JSON contains a non-finite constant {value!r}")


def _parse_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("issuer JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"issuer JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _assert_plain_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _fail("issuer JSON exceeds its structure bound")
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                _fail("issuer JSON strings must be bounded printable ASCII")
            continue
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail("issuer JSON integer exceeds its value bound")
            continue
        if type(item) not in {dict, list}:
            _fail("issuer JSON contains a non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("issuer JSON contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    _fail("issuer JSON object key is not an exact string")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _canonical_compact(value: Any) -> bytes:
    _assert_plain_json(value)
    try:
        result = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(
            "issuer artifact is not canonical finite ASCII JSON"
        ) from exc
    if len(result) > _MAX_ARTIFACT_BYTES:
        _fail("issuer artifact exceeds its byte bound")
    return result


def _canonical_json(value: Any) -> bytes:
    if type(value) is not dict:
        _fail("canonical issuer artifact root must be one plain object")
    return _canonical_compact(value) + b"\n"


def _strict_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        _fail(f"{label} must be nonempty exact bytes within the byte bound")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        _fail(f"{label} must have one canonical trailing newline")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(f"{label} must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_raise_constant,
            parse_float=_raise_float,
            parse_int=_parse_int,
        )
    except ForagerMatchedV3CpuRuntimeLockIssuerError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(
            f"{label} is not bounded strict JSON"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} root must be one object")
    result = cast(dict[str, Any], value)
    _assert_plain_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        _fail(f"{label} bytes are not canonical")
    return result


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _array(value: Any, *, label: str) -> list[Any]:
    if type(value) is not list:
        _fail(f"{label} must be one array")
    return value


def _string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        _fail(f"{label} must be one nonempty exact string")
    return value


def _integer(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_INTEGER:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _sha256(value: Any, *, label: str) -> str:
    result = _string(value, label=label)
    if _SHA256_RE.fullmatch(result) is None or result == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return result


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _inventory(value: Any, label: str) -> str:
    return _hash(_canonical_json({label: value}))


def _verify_body(value: Mapping[str, Any], field: str, *, label: str) -> str:
    body = copy.deepcopy(dict(value))
    supplied = _sha256(body.pop(field, None), label=f"{label} {field}")
    observed = _hash(_canonical_json(body))
    if not hmac.compare_digest(supplied, observed):
        _fail(f"{label} {field} differs from its canonical body")
    return observed


def _authority_contract() -> tuple[dict[str, Any], list[Any]]:
    descriptor = runtime_lock_contract.cpu_runtime_lock_descriptor()
    return (
        cast(dict[str, Any], descriptor["claims"]),
        cast(list[Any], descriptor["limitations"]),
    )


def _validate_authority_fields(
    value: Mapping[str, Any],
    *,
    label: str,
    issuance_envelope: bool = False,
) -> None:
    claims, limitations = _authority_contract()
    if value["claims"] != claims or any(item is not False for item in claims.values()):
        _fail(f"{label} must keep every authority claim false")
    expected_limitations = (
        [*limitations, CPU_RUNTIME_LOCK_ISSUANCE_RANGE_LIMITATION]
        if issuance_envelope
        else limitations
    )
    if value["limitations"] != expected_limitations:
        _fail(f"{label} limitations differ from the runtime-lock contract")


def _validate_input_bindings(value: Any) -> dict[str, Any]:
    bindings = _exact(
        value,
        frozenset(
            {
                "capture_manifest_sha256",
                "capture_manifest_body_sha256",
                "wheelhouse_receipt_sha256",
                "wheelhouse_receipt_body_sha256",
                "wheelhouse_archive_sha256",
                "wheelhouse_archive_size_bytes",
                "wheelhouse_archive_inventory_sha256",
            }
        ),
        label="issuance input bindings",
    )
    for field in (
        "capture_manifest_sha256",
        "capture_manifest_body_sha256",
        "wheelhouse_receipt_sha256",
        "wheelhouse_receipt_body_sha256",
        "wheelhouse_archive_sha256",
        "wheelhouse_archive_inventory_sha256",
    ):
        _sha256(bindings[field], label=f"issuance {field}")
    _integer(
        bindings["wheelhouse_archive_size_bytes"],
        label="issuance wheelhouse archive size",
        minimum=1,
    )
    return bindings


def _validate_root_pin_inventory(value: Any) -> dict[str, Any]:
    inventory = _exact(
        value,
        frozenset({"pins", "pin_count", "inventory_sha256"}),
        label="root-pin inventory",
    )
    pins = _array(inventory["pins"], label="root-pin entries")
    if not 1 <= len(pins) <= _MAX_PACKAGES:
        _fail("root-pin inventory count is outside its bound")
    validated: list[dict[str, Any]] = []
    names: set[str] = set()
    requirements: set[str] = set()
    for index, raw_pin in enumerate(pins):
        pin = _exact(
            raw_pin,
            frozenset({"name", "version", "requirement", "selected_extras"}),
            label=f"root pin {index}",
        )
        name = _string(pin["name"], label=f"root pin {index} name")
        if _NAME_RE.fullmatch(name) is None or name in names:
            _fail("root-pin names must be unique canonical distribution names")
        names.add(name)
        _string(pin["version"], label=f"root pin {index} version")
        requirement = _string(pin["requirement"], label=f"root pin {index} requirement")
        if requirement in requirements:
            _fail("root-pin requirements must be unique")
        requirements.add(requirement)
        extras = _array(pin["selected_extras"], label=f"root pin {index} extras")
        if any(
            type(item) is not str or _EXTRA_RE.fullmatch(item) is None for item in extras
        ) or extras != sorted(set(cast(list[str], extras))):
            _fail("root-pin extras must be unique sorted canonical names")
        validated.append(pin)
    if [pin["requirement"] for pin in validated] != sorted(requirements):
        _fail("root-pin inventory must be sorted by exact requirement")
    if inventory["pin_count"] != len(validated):
        _fail("root-pin inventory count differs")
    _integer(inventory["pin_count"], label="root-pin count", minimum=1)
    expected = _inventory(validated, "pins")
    if _sha256(inventory["inventory_sha256"], label="root-pin inventory") != expected:
        _fail("root-pin inventory SHA-256 differs")
    return inventory


def _validate_resolution_inventory(value: Any, *, solver: Mapping[str, Any]) -> dict[str, Any]:
    inventory = _exact(
        value,
        frozenset(
            {
                "lock_format",
                "lock_size_bytes",
                "lock_sha256",
                "selected_wheels",
                "selected_wheel_count",
                "selected_wheel_inventory_sha256",
            }
        ),
        label="resolution inventory",
    )
    if inventory["lock_format"] != "uv_lock_toml":
        _fail("resolution inventory lock format differs")
    _integer(inventory["lock_size_bytes"], label="resolution lock size", minimum=1)
    lock_sha = _sha256(inventory["lock_sha256"], label="resolution lock")
    if solver.get("resolution_input_sha256") != lock_sha:
        _fail("resolution lock identity differs from solver resolution input")
    raw_entries = _array(inventory["selected_wheels"], label="selected-wheel inventory")
    if not 1 <= len(raw_entries) <= _MAX_PACKAGES:
        _fail("selected-wheel inventory count is outside its bound")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    filenames: set[str] = set()
    urls: set[str] = set()
    hashes: set[str] = set()
    for index, raw_entry in enumerate(raw_entries):
        entry = _exact(
            raw_entry,
            frozenset({"name", "version", "filename", "source_url", "size_bytes", "sha256"}),
            label=f"selected wheel {index}",
        )
        name = _string(entry["name"], label=f"selected wheel {index} name")
        filename = _string(entry["filename"], label=f"selected wheel {index} filename")
        source_url = _string(entry["source_url"], label=f"selected wheel {index} source URL")
        wheel_sha = _sha256(entry["sha256"], label=f"selected wheel {index}")
        _string(entry["version"], label=f"selected wheel {index} version")
        _integer(entry["size_bytes"], label=f"selected wheel {index} size", minimum=1)
        if (
            _NAME_RE.fullmatch(name) is None
            or not filename.endswith(".whl")
            or "/" in filename
            or "\\" in filename
            or name in names
            or filename in filenames
            or source_url in urls
            or wheel_sha in hashes
        ):
            _fail("selected-wheel identities or source URLs are duplicate or noncanonical")
        names.add(name)
        filenames.add(filename)
        urls.add(source_url)
        hashes.add(wheel_sha)
        entries.append(entry)
    if [entry["name"] for entry in entries] != sorted(names):
        _fail("selected-wheel inventory must be sorted by canonical name")
    if inventory["selected_wheel_count"] != len(entries):
        _fail("selected-wheel inventory count differs")
    _integer(
        inventory["selected_wheel_count"],
        label="selected-wheel count",
        minimum=1,
    )
    expected = _inventory(entries, "selected_wheels")
    if (
        _sha256(
            inventory["selected_wheel_inventory_sha256"],
            label="selected-wheel inventory",
        )
        != expected
    ):
        _fail("selected-wheel inventory SHA-256 differs")
    return inventory


def _validate_upstream_shape(value: Any) -> dict[str, Any]:
    upstream = _exact(
        value,
        frozenset(
            {
                "repository_id",
                "repository_url",
                "commit_git_sha1",
                "tree_git_sha1",
                "archive",
                "pyproject",
                "lock",
                "root_project_distribution",
                "root_project_installed",
            }
        ),
        label="issuance upstream provenance",
    )
    descriptor = cast(
        Mapping[str, Any],
        runtime_lock_contract.cpu_runtime_lock_descriptor()["upstream"],
    )
    expected_scalars = {
        "repository_id": descriptor["repository_id"],
        "repository_url": descriptor["repository_url"],
        "commit_git_sha1": descriptor["commit_git_sha1"],
        "tree_git_sha1": descriptor["tree_git_sha1"],
        "root_project_distribution": descriptor["root_project_distribution"],
        "root_project_installed": False,
    }
    if any(upstream[field] != expected for field, expected in expected_scalars.items()):
        _fail("issuance upstream repository or root-project identity differs")
    archive = _exact(
        upstream["archive"],
        frozenset({"size_bytes", "sha256"}),
        label="issuance upstream archive",
    )
    if (
        archive["size_bytes"] != descriptor["archive_size_bytes"]
        or archive["sha256"] != descriptor["archive_sha256"]
    ):
        _fail("issuance upstream archive identity differs")
    pyproject = _exact(
        upstream["pyproject"],
        frozenset({"path", "size_bytes", "sha256"}),
        label="issuance upstream pyproject",
    )
    if (
        pyproject["path"] != descriptor["pyproject_path"]
        or pyproject["size_bytes"] != descriptor["pyproject_size_bytes"]
        or pyproject["sha256"] != descriptor["pyproject_sha256"]
    ):
        _fail("issuance upstream pyproject identity differs")
    lock = _exact(
        upstream["lock"],
        frozenset({"path", "size_bytes", "sha256"}),
        label="issuance upstream lock",
    )
    if lock["path"] != descriptor["lock_path"] or lock["sha256"] != descriptor["lock_sha256"]:
        _fail("issuance upstream lock identity differs")
    _integer(lock["size_bytes"], label="issuance upstream lock size", minimum=1)
    return upstream


def _validate_overlay_shape(value: Any) -> dict[str, Any]:
    overlay = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "base_pyproject_sha256",
                "base_lock_sha256",
                "delta_format",
                "operations",
                "operation_count",
                "operations_sha256",
                "direct_requirements_sha256",
                "source_builds_allowed",
                "overlay_body_sha256",
            }
        ),
        label="issuance overlay",
    )
    descriptor = cast(
        Mapping[str, Any],
        runtime_lock_contract.cpu_runtime_lock_descriptor()["upstream"],
    )
    if (
        overlay["schema_version"] != runtime_lock_contract.CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION
        or overlay["base_pyproject_sha256"] != descriptor["pyproject_sha256"]
        or overlay["base_lock_sha256"] != descriptor["lock_sha256"]
        or overlay["delta_format"] != "canonical_json_operations_v1"
        or overlay["source_builds_allowed"] is not False
    ):
        _fail("issuance overlay schema, base identity, format, or source-build policy differs")
    operations = _array(overlay["operations"], label="issuance overlay operations")
    if not operations:
        _fail("issuance overlay operations may not be empty")
    validated_operations: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, raw_operation in enumerate(operations):
        operation = _exact(
            raw_operation,
            frozenset({"op", "path", "expected", "replacement", "operation_body_sha256"}),
            label=f"issuance overlay operation {index}",
        )
        op = operation["op"]
        path = operation["path"]
        if (
            op not in {"add", "remove", "replace"}
            or type(path) is not str
            or not path.startswith(("/pyproject/", "/lock/"))
            or path in paths
            or (
                op == "add"
                and (operation["expected"] is not None or operation["replacement"] is None)
            )
            or (
                op == "remove"
                and (operation["expected"] is None or operation["replacement"] is not None)
            )
            or (
                op == "replace"
                and (operation["expected"] is None or operation["replacement"] is None)
            )
        ):
            _fail("issuance overlay operation kind, path, or values differ")
        paths.add(path)
        _verify_body(
            operation,
            "operation_body_sha256",
            label=f"issuance overlay operation {index}",
        )
        validated_operations.append(operation)
    if validated_operations != sorted(
        validated_operations,
        key=lambda operation: (cast(str, operation["path"]), cast(str, operation["op"])),
    ):
        _fail("issuance overlay operations are not sorted by exact path and kind")
    if overlay["operation_count"] != len(operations):
        _fail("issuance overlay operation count differs")
    if _sha256(overlay["operations_sha256"], label="issuance overlay operations") != _inventory(
        operations, "operations"
    ):
        _fail("issuance overlay operations SHA-256 differs")
    _sha256(overlay["direct_requirements_sha256"], label="issuance direct requirements")
    _verify_body(overlay, "overlay_body_sha256", label="issuance overlay")
    return overlay


def _validate_solver_shape(value: Any) -> dict[str, Any]:
    solver = _exact(
        value,
        frozenset(
            {
                "informational_only",
                "argv",
                "argv_sha256",
                "environment",
                "environment_sha256",
                "interpreter_implementation",
                "interpreter_version",
                "interpreter_binary_sha256",
                "solver",
                "solver_version",
                "solver_binary_sha256",
                "marker_environment_sha256",
                "index_url",
                "index_capture_timestamp_utc",
                "resolution_input_sha256",
                "resolution_report_size_bytes",
                "resolution_report_sha256",
                "trusted_for_acceptance",
            }
        ),
        label="issuance solver provenance",
    )
    if (
        solver["informational_only"] is not True
        or solver["trusted_for_acceptance"] is not False
        or solver["interpreter_implementation"] != "CPython"
        or solver["interpreter_version"] != runtime_lock_contract.PRODUCTION_PYTHON_VERSION
        or solver["solver"] != "uv"
        or solver["index_url"] != "https://pypi.org/simple"
    ):
        _fail("issuance solver policy, interpreter, solver, or index identity differs")
    argv = _array(solver["argv"], label="issuance solver argv")
    environment = _array(solver["environment"], label="issuance solver environment")
    if not argv or any(type(item) is not str or not item for item in argv):
        _fail("issuance solver argv is invalid")
    if any(type(item) is not str or not item for item in environment):
        _fail("issuance solver environment is invalid")
    if environment != sorted(set(cast(list[str], environment))):
        _fail("issuance solver environment must be unique and sorted")
    if _sha256(solver["argv_sha256"], label="issuance solver argv") != _inventory(argv, "argv"):
        _fail("issuance solver argv SHA-256 differs")
    if _sha256(solver["environment_sha256"], label="issuance solver environment") != _inventory(
        environment, "environment"
    ):
        _fail("issuance solver environment SHA-256 differs")
    for field in (
        "interpreter_binary_sha256",
        "solver_binary_sha256",
        "marker_environment_sha256",
        "resolution_input_sha256",
        "resolution_report_sha256",
    ):
        _sha256(solver[field], label=f"issuance solver {field}")
    _integer(
        solver["resolution_report_size_bytes"],
        label="issuance resolution report size",
        minimum=1,
    )
    for field in (
        "interpreter_version",
        "solver_version",
        "index_capture_timestamp_utc",
    ):
        _string(solver[field], label=f"issuance solver {field}")
    if _TIMESTAMP_RE.fullmatch(cast(str, solver["index_capture_timestamp_utc"])) is None:
        _fail("issuance solver capture timestamp is not exact second-resolution UTC")
    return solver


def validate_cpu_runtime_lock_issuance_envelope(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one canonical issuance provenance envelope."""

    envelope = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "input_bindings",
                "upstream",
                "overlay_delta",
                "solver_provenance",
                "resolution_inventory",
                "root_pin_inventory",
                "claims",
                "limitations",
                "envelope_body_sha256",
            }
        ),
        label="issuance envelope",
    )
    if (
        envelope["schema_version"] != CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_SCHEMA_VERSION
        or envelope["status"] != CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_STATUS
        or envelope["classification"] != CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_CLASSIFICATION
    ):
        _fail("issuance envelope schema, status, or classification differs")
    _validate_input_bindings(envelope["input_bindings"])
    _validate_upstream_shape(envelope["upstream"])
    _validate_overlay_shape(envelope["overlay_delta"])
    solver = _validate_solver_shape(envelope["solver_provenance"])
    _validate_resolution_inventory(envelope["resolution_inventory"], solver=solver)
    _validate_root_pin_inventory(envelope["root_pin_inventory"])
    _validate_authority_fields(
        envelope,
        label="issuance envelope",
        issuance_envelope=True,
    )
    _verify_body(envelope, "envelope_body_sha256", label="issuance envelope")
    _assert_plain_json(envelope)
    return _strict_json(_canonical_json(envelope), label="issuance envelope")


def canonical_cpu_runtime_lock_issuance_envelope_bytes(value: Mapping[str, Any]) -> bytes:
    """Validate and canonically encode one issuance envelope."""

    detached = validate_cpu_runtime_lock_issuance_envelope(value)
    return _canonical_json(detached)


def parse_cpu_runtime_lock_issuance_envelope(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse canonical envelope bytes under an independent full-file digest pin."""

    expected = _sha256(expected_file_sha256, label="expected issuance envelope")
    if type(raw) is not bytes or not hmac.compare_digest(_hash(raw), expected):
        _fail("issuance envelope full-file SHA-256 differs")
    value = _strict_json(raw, label="issuance envelope")
    return validate_cpu_runtime_lock_issuance_envelope(value)


def _runtime_target(target: Mapping[str, Any]) -> dict[str, Any]:
    libc = cast(Mapping[str, Any], target["libc"])
    return {
        "implementation": target["implementation"],
        "python_version": target["python_version"],
        "python_tag": "cp312",
        "abi_tag": target["abi"],
        "os": "linux",
        "architecture": "x86_64",
        "platform": "linux-amd64",
        "libc_family": libc["family"],
        "libc_version": libc["version"],
        "cpu_only": True,
    }


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _specifier_signature(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    parts = tuple(re.sub(r"\s+", "", item) for item in value.split(","))
    if any(not item for item in parts) or len(parts) != len(set(parts)):
        _fail("requirement specifier is empty, duplicate, or incomplete")
    return tuple(sorted(parts))


def _marker_signature(value: str) -> tuple[Any, ...]:
    marker = value.strip()
    if not marker:
        _fail("requirement marker is empty")
    result: list[tuple[str, str]] = []
    offset = 0
    while offset < len(marker):
        matched = _MARKER_TOKEN_RE.match(marker, offset)
        if matched is None:
            _fail("requirement marker contains unsupported or trailing syntax")
        kind = cast(str, matched.lastgroup)
        token = matched.group(kind)
        if token is None:
            raise AssertionError("matched marker token has no value")
        if kind == "string":
            result.append((kind, token[1:-1]))
        elif kind == "operator":
            result.append((kind, re.sub(r"\s+", " ", token)))
        else:
            result.append((kind, token))
        offset = matched.end()
    if not result:
        _fail("requirement marker is incomplete")
    position = 0

    def parse_operand() -> tuple[str, str]:
        nonlocal position
        if position >= len(result) or result[position][0] not in {"variable", "string"}:
            _fail("requirement marker operand is invalid")
        operand = result[position]
        position += 1
        return operand

    def parse_comparison() -> tuple[Any, ...]:
        nonlocal position
        left = parse_operand()
        if position >= len(result) or result[position][0] != "operator":
            _fail("requirement marker comparison operator is invalid")
        operator = result[position][1]
        position += 1
        right = parse_operand()
        return ("comparison", left, operator, right)

    def parse_atom() -> tuple[Any, ...]:
        nonlocal position
        if position < len(result) and result[position][0] == "left":
            position += 1
            expression = parse_or()
            if position >= len(result) or result[position][0] != "right":
                _fail("requirement marker parentheses are unbalanced")
            position += 1
            return expression
        return parse_comparison()

    def parse_and() -> tuple[Any, ...]:
        nonlocal position
        operands = [parse_atom()]
        while (
            position < len(result)
            and result[position][0] == "boolean"
            and result[position][1] == "and"
        ):
            position += 1
            operands.append(parse_atom())
        return operands[0] if len(operands) == 1 else ("and", *operands)

    def parse_or() -> tuple[Any, ...]:
        nonlocal position
        operands = [parse_and()]
        while (
            position < len(result)
            and result[position][0] == "boolean"
            and result[position][1] == "or"
        ):
            position += 1
            operands.append(parse_and())
        return operands[0] if len(operands) == 1 else ("or", *operands)

    signature = parse_or()
    if position != len(result):
        _fail("requirement marker contains misplaced tokens")
    return signature


def _parse_requirement_text(raw: str) -> tuple[str, list[str], str, str | None]:
    requirement, separator, marker_text = raw.partition(";")
    matched = _REQUIREMENT_BODY_RE.fullmatch(requirement)
    if matched is None:
        _fail("requirement raw body is not fully reconstructible")
    name = _normalized_name(matched.group("name"))
    extras_text = matched.group("extras")
    extras = (
        []
        if extras_text is None
        else sorted(_normalized_name(item.strip()) for item in extras_text.split(","))
    )
    if len(extras) != len(set(extras)):
        _fail("requirement raw extras are duplicate after normalization")
    specifier = matched.group("parenthesized")
    if specifier is None:
        specifier = matched.group("plain")
    exact_specifier = "" if specifier is None else specifier.strip()
    marker = marker_text.strip() if separator else None
    if marker == "":
        _fail("requirement raw marker is empty")
    return name, extras, exact_specifier, marker


def _validated_requirement_record(
    record: Mapping[str, Any],
    *,
    exact_root: bool = False,
) -> str | None:
    raw = _string(record.get("raw"), label="requirement raw")
    name, extras, specifier, raw_marker = _parse_requirement_text(raw)
    structured_name = _string(record.get("name"), label="requirement name")
    structured_extras = record.get("extras")
    structured_specifier = record.get("specifier")
    structured_marker = record.get("marker")
    if (
        name != structured_name
        or type(structured_extras) is not list
        or extras != structured_extras
        or type(structured_specifier) is not str
        or _specifier_signature(specifier) != _specifier_signature(structured_specifier)
    ):
        _fail("requirement raw name, extras, or specifier differs from its structured record")
    if raw_marker is None:
        if structured_marker is not None:
            _fail("unconditional requirement raw value has a structured marker")
    elif type(structured_marker) is not str or _marker_signature(raw_marker) != _marker_signature(
        structured_marker
    ):
        _fail("requirement raw marker differs from its structured marker")
    if exact_root:
        exact_extras = "" if not extras else f"[{','.join(extras)}]"
        if raw_marker is not None or raw != f"{name}{exact_extras}{structured_specifier}":
            _fail("root requirement raw value is not its exact canonical structured pin")
    return raw_marker


def _compare_manifest_and_receipt(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    manifest_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    receipt_capture = cast(Mapping[str, Any], receipt["capture_manifest"])
    if (
        receipt_capture["full_file_sha256"] != manifest_sha256
        or receipt_capture["body_sha256"] != manifest["manifest_body_sha256"]
    ):
        _fail("receipt capture-manifest body or full-file identity differs")
    if receipt["target"] != manifest["target"]:
        _fail("receipt and capture-manifest target differ")
    if receipt["root_requirements"] != manifest["root_requirements"]:
        _fail("receipt and capture-manifest root requirements differ")
    manifest_wheels = cast(list[dict[str, Any]], manifest["wheels"])
    receipt_packages = cast(list[dict[str, Any]], receipt["packages"])
    by_filename: dict[str, dict[str, Any]] = {}
    source_urls: set[str] = set()
    for wheel in manifest_wheels:
        filename = cast(str, wheel["filename"])
        source_url = cast(str, wheel["origin_url"])
        if filename in by_filename:
            _fail("capture manifest repeats a wheel filename")
        if source_url in source_urls:
            _fail("capture manifest repeats an exact source URL")
        by_filename[filename] = wheel
        source_urls.add(source_url)
    by_name: dict[str, dict[str, Any]] = {}
    receipt_filenames: set[str] = set()
    for package in receipt_packages:
        name = cast(str, package["name"])
        filename = cast(str, package["filename"])
        if name in by_name or filename in receipt_filenames:
            _fail("receipt repeats a package name or wheel filename")
        captured = by_filename.get(filename)
        if captured is None or (
            captured["sha256"],
            captured["size_bytes"],
        ) != (
            package["sha256"],
            package["size_bytes"],
        ):
            _fail("receipt package wheel identity differs from capture-manifest wheel identity")
        by_name[name] = package
        receipt_filenames.add(filename)
    if receipt_filenames != set(by_filename):
        _fail("receipt and capture-manifest wheel inventories differ")
    return by_name, by_filename


def _derive_roots(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    packages_by_name: Mapping[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    closure = cast(Mapping[str, Any], receipt["closure"])
    root_records = cast(list[dict[str, Any]], closure["root_requirements"])
    roots = cast(list[str], manifest["root_requirements"])
    if len(root_records) != len(roots) or [item["raw"] for item in root_records] != roots:
        _fail("receipt closure root requirements differ from the capture manifest")
    pins: list[dict[str, Any]] = []
    direct_requirements: list[str] = []
    names: set[str] = set()
    for record in root_records:
        _validated_requirement_record(record, exact_root=True)
        name = cast(str, record["name"])
        package = packages_by_name.get(name)
        if package is None or name in names:
            _fail("root requirement target is absent or duplicated")
        names.add(name)
        version = cast(str, package["version"])
        if record["marker"] is not None or record["specifier"] != f"=={version}":
            _fail(
                "every root requirement must unconditionally and exactly pin its selected version"
            )
        extras = cast(list[str], record["extras"])
        provided = cast(list[str], cast(Mapping[str, Any], package["metadata"])["provides_extra"])
        if extras != sorted(set(extras)) or not set(extras).issubset(provided):
            _fail("root requirement extras are noncanonical or undeclared")
        pins.append(
            {
                "name": name,
                "requirement": record["raw"],
                "selected_extras": copy.deepcopy(extras),
                "version": version,
            }
        )
        direct_requirements.append(f"{name}=={version}")
    if [pin["requirement"] for pin in pins] != sorted(roots):
        _fail("root-pin records do not preserve sorted capture requirements")
    return pins, sorted(direct_requirements)


def _crosscheck_overlay_roots(
    overlay: Mapping[str, Any],
    pins: list[dict[str, Any]],
) -> None:
    dependency_operations = [
        cast(Mapping[str, Any], operation)
        for operation in cast(list[Any], overlay["operations"])
        if cast(Mapping[str, Any], operation)["path"] == "/pyproject/project/dependencies"
    ]
    if len(dependency_operations) != 1:
        _fail("overlay must carry exactly one project dependency replacement")
    operation = dependency_operations[0]
    if operation["op"] != "replace" or type(operation["replacement"]) is not list:
        _fail("overlay project dependencies must be one exact replacement array")
    replacement = operation["replacement"]
    if not replacement or any(type(item) is not str or not item for item in replacement):
        _fail("overlay project dependency replacement entries are invalid")
    overlay_roots: dict[str, tuple[list[str], str]] = {}
    for raw in cast(list[str], replacement):
        name, extras, specifier, marker = _parse_requirement_text(raw)
        if marker is not None or name in overlay_roots:
            _fail("overlay project dependencies are marked or duplicate after normalization")
        overlay_roots[name] = (extras, specifier)
    pin_roots = {
        cast(str, pin["name"]): (
            cast(list[str], pin["selected_extras"]),
            cast(str, pin["version"]),
        )
        for pin in pins
    }
    if set(overlay_roots) != set(pin_roots):
        _fail("overlay project dependency names differ from capture root pins")
    for name, (pin_extras, selected_version) in pin_roots.items():
        overlay_extras, specifier = overlay_roots[name]
        if overlay_extras != pin_extras:
            _fail("overlay project dependency extras differ from capture root pins")
        signature = _specifier_signature(specifier)
        if len(signature) == 1 and signature[0].startswith("=="):
            exact_version = signature[0].removeprefix("==")
            if "*" not in exact_version and exact_version != selected_version:
                _fail("overlay exact dependency version differs from its selected root pin")


def _requirement_marker(record: Mapping[str, Any]) -> str | None:
    return _validated_requirement_record(record)


def _derive_activity(
    receipt: Mapping[str, Any],
    packages_by_name: Mapping[str, dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[tuple[str, bytes], dict[str, Any]]]:
    closure = cast(Mapping[str, Any], receipt["closure"])
    activated_raw = cast(dict[str, Any], closure["activated_extras"])
    package_names = sorted(packages_by_name)
    if sorted(activated_raw) != package_names:
        _fail("receipt activated extras do not cover the exact package inventory")
    activated: dict[str, list[str]] = {}
    for name in package_names:
        extras = cast(list[str], activated_raw[name])
        provided = cast(
            list[str],
            cast(Mapping[str, Any], packages_by_name[name]["metadata"])["provides_extra"],
        )
        if extras != sorted(set(extras)) or not set(extras).issubset(provided):
            _fail("receipt activated extras are duplicate, unsorted, or undeclared")
        activated[name] = copy.deepcopy(extras)
    edges: dict[tuple[str, bytes], dict[str, Any]] = {}
    for raw_edge in cast(list[dict[str, Any]], closure["edges"]):
        source = cast(str, raw_edge["source"])
        target = cast(str, raw_edge["target"])
        if source not in packages_by_name or target not in packages_by_name:
            _fail("receipt dependency edge references an absent package")
        requirement = cast(dict[str, Any], raw_edge["requirement"])
        key = (source, _canonical_json(requirement))
        source_requirements = cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], packages_by_name[source]["metadata"])["requires_dist"],
        )
        if key in edges or not any(_canonical_json(item) == key[1] for item in source_requirements):
            _fail("receipt dependency edge is duplicate or absent from source METADATA")
        if requirement["name"] != target:
            _fail("receipt dependency edge target differs from its requirement")
        contexts = cast(list[str], raw_edge["active_contexts"])
        allowed_contexts = ["", *activated[source]]
        if (
            not contexts
            or contexts != sorted(set(contexts))
            or not set(contexts).issubset(allowed_contexts)
        ):
            _fail("receipt dependency edge active contexts are nonreconstructible")
        if requirement["marker"] is None and contexts != sorted(allowed_contexts):
            _fail("receipt unconditional contexts do not cover every activated source context")
        edges[key] = raw_edge

    derived_extras: dict[str, set[str]] = {name: set() for name in package_names}
    for root in cast(list[dict[str, Any]], closure["root_requirements"]):
        derived_extras[cast(str, root["name"])].update(cast(list[str], root["extras"]))
    for edge in edges.values():
        edge_requirement = cast(Mapping[str, Any], edge["requirement"])
        derived_extras[cast(str, edge["target"])].update(
            cast(list[str], edge_requirement["extras"])
        )
    if any(set(activated[name]) != derived_extras[name] for name in package_names):
        _fail("receipt activated extras differ from root and active-edge extras")

    for name, package in packages_by_name.items():
        requirements = cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], package["metadata"])["requires_dist"],
        )
        raw_seen: set[str] = set()
        for requirement in requirements:
            raw = cast(str, requirement["raw"])
            if raw in raw_seen:
                _fail("receipt repeats an exact Requires-Dist raw value")
            raw_seen.add(raw)
            key = (name, _canonical_json(requirement))
            if requirement["marker"] is None and key not in edges:
                _fail("receipt omits an unconditional dependency edge")
            _requirement_marker(requirement)
    return activated, edges


def _runtime_requirement(
    owner: str,
    record: Mapping[str, Any],
    edges: Mapping[tuple[str, bytes], dict[str, Any]],
    packages_by_name: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    key = (owner, _canonical_json(record))
    active = key in edges
    target = packages_by_name.get(cast(str, record["name"]))
    if active and target is None:
        _fail("active receipt dependency has no selected target version")
    return {
        "raw": record["raw"],
        "name": record["name"],
        "marker": _requirement_marker(record),
        "active": active,
        "selected_version": None if target is None or not active else target["version"],
    }


def _build_packages(
    packages_by_name: Mapping[str, dict[str, Any]],
    manifest_by_filename: Mapping[str, dict[str, Any]],
    activated: Mapping[str, list[str]],
    edges: Mapping[tuple[str, bytes], dict[str, Any]],
    direct_names: frozenset[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name in sorted(packages_by_name):
        source = packages_by_name[name]
        version = cast(str, source["version"])
        filename = cast(str, source["filename"])
        captured = manifest_by_filename[filename]
        source_metadata = cast(Mapping[str, Any], source["metadata"])
        source_wheel = cast(Mapping[str, Any], source["wheel"])
        source_record = cast(Mapping[str, Any], source["record"])
        requirements = [
            _runtime_requirement(name, record, edges, packages_by_name)
            for record in cast(list[dict[str, Any]], source_metadata["requires_dist"])
        ]
        requirements.sort(key=lambda item: cast(str, item["raw"]))
        metadata = {
            "path": source_metadata["path"],
            "size_bytes": source_metadata["size_bytes"],
            "sha256": source_metadata["sha256"],
            "metadata_version": source_metadata["metadata_version"],
            "name": source_metadata["name"],
            "version": source_metadata["version"],
            "requires_python": source_metadata["requires_python"],
            "provides_extra": copy.deepcopy(source_metadata["provides_extra"]),
            "requires_dist": requirements,
        }
        wheel_file = {
            "path": source_wheel["path"],
            "size_bytes": source_wheel["size_bytes"],
            "sha256": source_wheel["sha256"],
            "wheel_version": source_wheel["wheel_version"],
            "generator": source_wheel["generator"],
            "root_is_purelib": source_wheel["root_is_purelib"],
            "tags": copy.deepcopy(source_wheel["tags"]),
        }
        record = {
            "path": source_record["path"],
            "size_bytes": source_record["size_bytes"],
            "sha256": source_record["sha256"],
            "entry_count": source_record["entry_count"],
            "entries_sha256": source_record["entries_sha256"],
        }
        wheel: dict[str, Any] = {
            "filename": filename,
            "source_url": captured["origin_url"],
            "cas_key": (f"sha256/{source['sha256'][:2]}/{source['sha256']}/{filename}"),
            "size_bytes": source["size_bytes"],
            "sha256": source["sha256"],
            "tags": copy.deepcopy(source["tags"]),
            "metadata": metadata,
            "wheel": wheel_file,
            "record": record,
            "wheel_body_sha256": "0" * 64,
        }
        wheel["wheel_body_sha256"] = _verify_derived_body(wheel, "wheel_body_sha256")
        package: dict[str, Any] = {
            "name": name,
            "version": version,
            "direct": name in direct_names,
            "selected_extras": copy.deepcopy(activated[name]),
            "installation_kind": "wheel",
            "build_required": False,
            "wheels": [wheel],
            "package_body_sha256": "0" * 64,
        }
        package["package_body_sha256"] = _verify_derived_body(
            package,
            "package_body_sha256",
        )
        result.append(package)
    return result


def _verify_derived_body(value: Mapping[str, Any], field: str) -> str:
    body = copy.deepcopy(dict(value))
    body.pop(field, None)
    return _hash(_canonical_json(body))


def _build_closure(packages: list[dict[str, Any]]) -> dict[str, Any]:
    wheels = [cast(dict[str, Any], package["wheels"][0]) for package in packages]
    dependencies = [
        {"from": package["name"], **requirement}
        for package in packages
        for requirement in cast(
            list[dict[str, Any]],
            cast(Mapping[str, Any], cast(Mapping[str, Any], package["wheels"][0])["metadata"])[
                "requires_dist"
            ],
        )
    ]
    distributions = [
        {
            "name": package["name"],
            "version": package["version"],
            "direct": package["direct"],
            "package_body_sha256": package["package_body_sha256"],
        }
        for package in packages
    ]
    wheel_inventory = [
        {
            "name": package["name"],
            "filename": wheel["filename"],
            "size_bytes": wheel["size_bytes"],
            "sha256": wheel["sha256"],
            "cas_key": wheel["cas_key"],
            "wheel_body_sha256": wheel["wheel_body_sha256"],
        }
        for package, wheel in zip(packages, wheels, strict=True)
    ]
    closure: dict[str, Any] = {
        "distribution_count": len(packages),
        "wheel_count": len(wheels),
        "total_wheel_bytes": sum(cast(int, wheel["size_bytes"]) for wheel in wheels),
        "total_metadata_bytes": sum(
            cast(int, cast(Mapping[str, Any], wheel["metadata"])["size_bytes"]) for wheel in wheels
        ),
        "total_wheel_file_bytes": sum(
            cast(int, cast(Mapping[str, Any], wheel["wheel"])["size_bytes"]) for wheel in wheels
        ),
        "total_record_bytes": sum(
            cast(int, cast(Mapping[str, Any], wheel["record"])["size_bytes"]) for wheel in wheels
        ),
        "requires_dist_count": len(dependencies),
        "active_dependency_count": sum(
            1 for dependency in dependencies if dependency["active"] is True
        ),
        "distribution_inventory_sha256": _inventory(distributions, "distributions"),
        "wheel_inventory_sha256": _inventory(wheel_inventory, "wheels"),
        "dependency_inventory_sha256": _inventory(dependencies, "dependencies"),
        "packages_body_sha256": _inventory(packages, "packages"),
        "closure_body_sha256": "0" * 64,
    }
    closure["closure_body_sha256"] = _verify_derived_body(closure, "closure_body_sha256")
    return closure


def _selected_inventory(
    packages_by_name: Mapping[str, dict[str, Any]],
    manifest_by_filename: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "version": packages_by_name[name]["version"],
            "filename": packages_by_name[name]["filename"],
            "source_url": manifest_by_filename[cast(str, packages_by_name[name]["filename"])][
                "origin_url"
            ],
            "size_bytes": packages_by_name[name]["size_bytes"],
            "sha256": packages_by_name[name]["sha256"],
        }
        for name in sorted(packages_by_name)
    ]


def _archive_members_by_filename(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    archive = cast(Mapping[str, Any], receipt["archive"])
    members = cast(list[dict[str, Any]], archive["members"])
    result: dict[str, dict[str, Any]] = {}
    for member in members:
        filename = cast(str, member["filename"])
        if filename in result:
            _fail("wheelhouse receipt archive mapping repeats a filename")
        result[filename] = member
    return result


def _build_cas_manifest(
    packages: list[dict[str, Any]],
    receipt: Mapping[str, Any],
    *,
    receipt_sha256: str,
) -> dict[str, Any]:
    members = _archive_members_by_filename(receipt)
    archive = cast(Mapping[str, Any], receipt["archive"])
    entries: list[dict[str, Any]] = []
    wheel_inventory: list[dict[str, Any]] = []
    for package in packages:
        wheel = cast(dict[str, Any], package["wheels"][0])
        member = members.get(cast(str, wheel["filename"]))
        if member is None or (
            member["sha256"],
            member["size_bytes"],
        ) != (
            wheel["sha256"],
            wheel["size_bytes"],
        ):
            _fail("wheelhouse receipt archive mapping is absent or differs")
        entries.append(
            {
                "name": package["name"],
                "version": package["version"],
                "filename": wheel["filename"],
                "archive_name": member["archive_name"],
                "source_url": wheel["source_url"],
                "cas_key": wheel["cas_key"],
                "size_bytes": wheel["size_bytes"],
                "sha256": wheel["sha256"],
                "wheel_body_sha256": wheel["wheel_body_sha256"],
            }
        )
        wheel_inventory.append(
            {
                "name": package["name"],
                "filename": wheel["filename"],
                "size_bytes": wheel["size_bytes"],
                "sha256": wheel["sha256"],
                "cas_key": wheel["cas_key"],
                "wheel_body_sha256": wheel["wheel_body_sha256"],
            }
        )
    if set(members) != {cast(str, entry["filename"]) for entry in entries}:
        _fail("wheelhouse receipt archive mapping differs from the selected closure")
    claims, limitations = _authority_contract()
    manifest: dict[str, Any] = {
        "schema_version": runtime_lock_contract.CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION,
        "status": CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_STATUS,
        "classification": CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_CLASSIFICATION,
        "cas_layout": "sha256/first-two/full-digest/wheel-filename",
        "source_receipt": {
            "full_file_sha256": receipt_sha256,
            "body_sha256": receipt["receipt_body_sha256"],
        },
        "source_archive": {
            "filename": "wheelhouse.v1.tar",
            "format": "canonical_posix_ustar_uncompressed",
            "size_bytes": archive["size_bytes"],
            "sha256": archive["sha256"],
            "inventory_sha256": archive["inventory_sha256"],
        },
        "entries": entries,
        "entry_count": len(entries),
        "total_bytes": sum(cast(int, entry["size_bytes"]) for entry in entries),
        "entry_inventory_sha256": _inventory(entries, "entries"),
        "wheel_inventory_sha256": _inventory(wheel_inventory, "wheels"),
        "claims": copy.deepcopy(claims),
        "limitations": copy.deepcopy(limitations),
        "manifest_body_sha256": "0" * 64,
    }
    manifest["manifest_body_sha256"] = _verify_derived_body(
        manifest,
        "manifest_body_sha256",
    )
    return manifest


def _canonical_ustar_size(entries: list[dict[str, Any]]) -> int:
    size = sum(
        512 + cast(int, entry["size_bytes"]) + (-cast(int, entry["size_bytes"])) % 512
        for entry in entries
    )
    size += 2 * 512
    return size + (-size) % 10_240


def _wheel_filename_matches_identity(filename: str, name: str, version: str) -> bool:
    try:
        prefix, _python_tag, _abi_tag, _platform_tag = filename.removesuffix(".whl").rsplit("-", 3)
    except ValueError:
        return False
    expected = f"{name.replace('-', '_')}-{version.replace('-', '_')}"
    return (
        prefix.casefold() == expected.casefold()
        or re.fullmatch(
            re.escape(expected) + r"-[0-9][A-Za-z0-9_.]*",
            prefix,
            re.IGNORECASE,
        )
        is not None
    )


def validate_cpu_runtime_wheelhouse_cas_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and detach one issuer-produced CAS mapping manifest."""

    manifest = _exact(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "cas_layout",
                "source_receipt",
                "source_archive",
                "entries",
                "entry_count",
                "total_bytes",
                "entry_inventory_sha256",
                "wheel_inventory_sha256",
                "claims",
                "limitations",
                "manifest_body_sha256",
            }
        ),
        label="CAS manifest",
    )
    if (
        manifest["schema_version"]
        != runtime_lock_contract.CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION
        or manifest["status"] != CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_STATUS
        or manifest["classification"] != CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_CLASSIFICATION
        or manifest["cas_layout"] != "sha256/first-two/full-digest/wheel-filename"
    ):
        _fail("CAS manifest schema, status, classification, or layout differs")
    receipt = _exact(
        manifest["source_receipt"],
        frozenset({"full_file_sha256", "body_sha256"}),
        label="CAS source receipt",
    )
    _sha256(receipt["full_file_sha256"], label="CAS source receipt file")
    _sha256(receipt["body_sha256"], label="CAS source receipt body")
    archive = _exact(
        manifest["source_archive"],
        frozenset({"filename", "format", "size_bytes", "sha256", "inventory_sha256"}),
        label="CAS source archive",
    )
    if (
        archive["filename"] != "wheelhouse.v1.tar"
        or archive["format"] != "canonical_posix_ustar_uncompressed"
    ):
        _fail("CAS source archive identity differs")
    _integer(archive["size_bytes"], label="CAS source archive size", minimum=1)
    _sha256(archive["sha256"], label="CAS source archive")
    _sha256(archive["inventory_sha256"], label="CAS source archive inventory")
    raw_entries = _array(manifest["entries"], label="CAS entries")
    if not 1 <= len(raw_entries) <= _MAX_PACKAGES:
        _fail("CAS entry count is outside its bound")
    entries: list[dict[str, Any]] = []
    names: set[str] = set()
    filenames: set[str] = set()
    archive_names: set[str] = set()
    hashes: set[str] = set()
    urls: set[str] = set()
    wheel_inventory: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _exact(
            raw_entry,
            frozenset(
                {
                    "name",
                    "version",
                    "filename",
                    "archive_name",
                    "source_url",
                    "cas_key",
                    "size_bytes",
                    "sha256",
                    "wheel_body_sha256",
                }
            ),
            label=f"CAS entry {index}",
        )
        name = _string(entry["name"], label=f"CAS entry {index} name")
        filename = _string(entry["filename"], label=f"CAS entry {index} filename")
        archive_name = _string(entry["archive_name"], label=f"CAS entry {index} archive name")
        source_url = _string(entry["source_url"], label=f"CAS entry {index} source URL")
        wheel_sha = _sha256(entry["sha256"], label=f"CAS entry {index}")
        body_sha = _sha256(entry["wheel_body_sha256"], label=f"CAS entry {index} body")
        _string(entry["version"], label=f"CAS entry {index} version")
        _integer(entry["size_bytes"], label=f"CAS entry {index} size", minimum=1)
        cas_key = _string(entry["cas_key"], label=f"CAS entry {index} CAS key")
        cas_match = _CAS_KEY_RE.fullmatch(cas_key)
        source_match = _PYPI_SOURCE_URL_RE.fullmatch(source_url)
        if (
            _NAME_RE.fullmatch(name) is None
            or not _wheel_filename_matches_identity(
                filename,
                name,
                cast(str, entry["version"]),
            )
            or source_match is None
            or source_match.group("filename") != filename
            or archive_name != f"{wheel_sha}.whl"
            or cas_match is None
            or cas_match.group("prefix") != wheel_sha[:2]
            or cas_match.group("sha256") != wheel_sha
            or cas_match.group("filename") != filename
            or name in names
            or filename in filenames
            or archive_name in archive_names
            or wheel_sha in hashes
            or source_url in urls
        ):
            _fail("CAS entry identity, archive mapping, source URL, or CAS key differs")
        names.add(name)
        filenames.add(filename)
        archive_names.add(archive_name)
        hashes.add(wheel_sha)
        urls.add(source_url)
        entries.append(entry)
        wheel_inventory.append(
            {
                "name": name,
                "filename": filename,
                "size_bytes": entry["size_bytes"],
                "sha256": wheel_sha,
                "cas_key": cas_key,
                "wheel_body_sha256": body_sha,
            }
        )
    if [entry["name"] for entry in entries] != sorted(names):
        _fail("CAS entries must be sorted by canonical distribution name")
    archive_members = sorted(
        [
            {
                "archive_name": entry["archive_name"],
                "filename": entry["filename"],
                "mode": "0444",
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
            for entry in entries
        ],
        key=lambda member: cast(str, member["archive_name"]).encode("ascii"),
    )
    if archive["size_bytes"] != _canonical_ustar_size(entries) or archive[
        "inventory_sha256"
    ] != _hash(_canonical_compact(archive_members)):
        _fail("CAS source archive size or member inventory is not reconstructible")
    if manifest["entry_count"] != len(entries):
        _fail("CAS manifest entry count differs")
    if manifest["total_bytes"] != sum(cast(int, entry["size_bytes"]) for entry in entries):
        _fail("CAS manifest total bytes differ")
    if _sha256(manifest["entry_inventory_sha256"], label="CAS entry inventory") != _inventory(
        entries, "entries"
    ):
        _fail("CAS entry inventory SHA-256 differs")
    if _sha256(manifest["wheel_inventory_sha256"], label="CAS wheel inventory") != _inventory(
        wheel_inventory, "wheels"
    ):
        _fail("CAS wheel inventory SHA-256 differs")
    _validate_authority_fields(manifest, label="CAS manifest")
    _verify_body(manifest, "manifest_body_sha256", label="CAS manifest")
    return _strict_json(_canonical_json(manifest), label="CAS manifest")


def canonical_cpu_runtime_wheelhouse_cas_manifest_bytes(value: Mapping[str, Any]) -> bytes:
    """Validate and canonically encode one CAS mapping manifest."""

    detached = validate_cpu_runtime_wheelhouse_cas_manifest(value)
    return _canonical_json(detached)


def parse_cpu_runtime_wheelhouse_cas_manifest(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse canonical CAS-manifest bytes under an independent digest pin."""

    expected = _sha256(expected_file_sha256, label="expected CAS manifest")
    if type(raw) is not bytes or not hmac.compare_digest(_hash(raw), expected):
        _fail("CAS manifest full-file SHA-256 differs")
    value = _strict_json(raw, label="CAS manifest")
    return validate_cpu_runtime_wheelhouse_cas_manifest(value)


def _bind_envelope_inputs(
    envelope: Mapping[str, Any],
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    manifest_sha256: str,
    receipt_sha256: str,
) -> None:
    bindings = cast(Mapping[str, Any], envelope["input_bindings"])
    archive = cast(Mapping[str, Any], receipt["archive"])
    expected = {
        "capture_manifest_sha256": manifest_sha256,
        "capture_manifest_body_sha256": manifest["manifest_body_sha256"],
        "wheelhouse_receipt_sha256": receipt_sha256,
        "wheelhouse_receipt_body_sha256": receipt["receipt_body_sha256"],
        "wheelhouse_archive_sha256": archive["sha256"],
        "wheelhouse_archive_size_bytes": archive["size_bytes"],
        "wheelhouse_archive_inventory_sha256": archive["inventory_sha256"],
    }
    if bindings != expected:
        differing = sorted(key for key in expected if bindings.get(key) != expected[key])
        _fail("issuance envelope input binding differs: " + ", ".join(differing))


def _build_lock(
    manifest: Mapping[str, Any],
    receipt: Mapping[str, Any],
    envelope: Mapping[str, Any],
    *,
    packages_by_name: Mapping[str, dict[str, Any]],
    manifest_by_filename: Mapping[str, dict[str, Any]],
    receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    pins, direct_requirements = _derive_roots(
        manifest,
        receipt,
        packages_by_name,
    )
    _crosscheck_overlay_roots(
        cast(Mapping[str, Any], envelope["overlay_delta"]),
        pins,
    )
    root_inventory = cast(Mapping[str, Any], envelope["root_pin_inventory"])
    if root_inventory["pins"] != pins:
        _fail("issuance envelope root-pin inventory differs from capture/receipt roots")
    activated, edges = _derive_activity(receipt, packages_by_name)
    direct_names = frozenset(cast(str, pin["name"]) for pin in pins)
    packages = _build_packages(
        packages_by_name,
        manifest_by_filename,
        activated,
        edges,
        direct_names,
    )
    expected_selected = _selected_inventory(packages_by_name, manifest_by_filename)
    resolution_inventory = cast(Mapping[str, Any], envelope["resolution_inventory"])
    if resolution_inventory["selected_wheels"] != expected_selected:
        _fail("issuance selected-wheel inventory differs from capture/receipt identities")
    closure = _build_closure(packages)
    cas_manifest = _build_cas_manifest(packages, receipt, receipt_sha256=receipt_sha256)
    cas_raw = canonical_cpu_runtime_wheelhouse_cas_manifest_bytes(cas_manifest)
    cas_sha = _hash(cas_raw)
    archive = cast(Mapping[str, Any], receipt["archive"])
    claims, limitations = _authority_contract()
    marker_environment = copy.deepcopy(
        cast(Mapping[str, Any], cast(Mapping[str, Any], manifest["target"])["marker_environment"])
    )
    selected_extras = sorted({extra for values in activated.values() for extra in values})
    lock: dict[str, Any] = {
        "schema_version": runtime_lock_contract.CPU_RUNTIME_LOCK_SCHEMA_VERSION,
        "status": "future_content_lock_unexecuted_non_authorizing",
        "classification": runtime_lock_contract.CPU_RUNTIME_LOCK_CLASSIFICATION,
        "target": _runtime_target(cast(Mapping[str, Any], manifest["target"])),
        "upstream": copy.deepcopy(envelope["upstream"]),
        "overlay_delta": copy.deepcopy(envelope["overlay_delta"]),
        "solver_provenance": copy.deepcopy(envelope["solver_provenance"]),
        "resolution": {
            "selected_extras": selected_extras,
            "marker_environment": marker_environment,
            "marker_environment_sha256": _hash(_canonical_json(marker_environment)),
            "direct_requirements": direct_requirements,
        },
        "packages": packages,
        "closure": closure,
        "wheelhouse": {
            "schema_version": runtime_lock_contract.CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION,
            "cas_layout": "sha256/first-two/full-digest/wheel-filename",
            "manifest": {
                "filename": "wheelhouse.cas-manifest.v1.json",
                "size_bytes": len(cas_raw),
                "sha256": cas_sha,
                "body_sha256": cas_manifest["manifest_body_sha256"],
                "entry_count": cas_manifest["entry_count"],
                "total_bytes": cas_manifest["total_bytes"],
                "inventory_sha256": cas_manifest["wheel_inventory_sha256"],
            },
            "archive": {
                "filename": "wheelhouse.v1.tar",
                "format": "ustar",
                "size_bytes": archive["size_bytes"],
                "sha256": archive["sha256"],
                "manifest_sha256": cas_sha,
                "manifest_body_sha256": cas_manifest["manifest_body_sha256"],
            },
            "networkless_install_required": True,
        },
        "claims": copy.deepcopy(claims),
        "limitations": copy.deepcopy(limitations),
        "lock_body_sha256": "0" * 64,
    }
    lock["lock_body_sha256"] = _verify_derived_body(lock, "lock_body_sha256")
    return lock, cas_manifest, pins


def issue_matched_v3_cpu_runtime_lock(
    *,
    capture_manifest_raw: bytes,
    expected_capture_manifest_sha256: str,
    wheelhouse_receipt_raw: bytes,
    expected_wheelhouse_receipt_sha256: str,
    issuance_envelope_raw: bytes,
    expected_issuance_envelope_sha256: str,
) -> CpuRuntimeLockIssuanceArtifacts:
    """Issue canonical lock/CAS bytes from exact already-captured content identities."""

    manifest_sha = _sha256(
        expected_capture_manifest_sha256,
        label="expected capture-manifest file",
    )
    receipt_sha = _sha256(
        expected_wheelhouse_receipt_sha256,
        label="expected wheelhouse-receipt file",
    )
    envelope_sha = _sha256(
        expected_issuance_envelope_sha256,
        label="expected issuance-envelope file",
    )
    try:
        manifest = wheelhouse_contract.parse_cpu_wheel_capture_manifest(
            capture_manifest_raw,
            expected_file_sha256=manifest_sha,
        )
        receipt = wheelhouse_contract.parse_cpu_wheelhouse_receipt(
            wheelhouse_receipt_raw,
            expected_file_sha256=receipt_sha,
        )
        envelope = parse_cpu_runtime_lock_issuance_envelope(
            issuance_envelope_raw,
            expected_file_sha256=envelope_sha,
        )
    except ForagerMatchedV3CpuRuntimeLockIssuerError:
        raise
    except (ValueError, TypeError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(
            "a final manifest, receipt, or runtime provenance parser rejected issuance input"
        ) from exc
    packages_by_name, manifest_by_filename = _compare_manifest_and_receipt(
        manifest,
        receipt,
        manifest_sha256=manifest_sha,
    )
    _bind_envelope_inputs(
        envelope,
        manifest,
        receipt,
        manifest_sha256=manifest_sha,
        receipt_sha256=receipt_sha,
    )
    lock, cas_manifest, pins = _build_lock(
        manifest,
        receipt,
        envelope,
        packages_by_name=packages_by_name,
        manifest_by_filename=manifest_by_filename,
        receipt_sha256=receipt_sha,
    )
    try:
        validated_lock = runtime_lock_contract.validate_cpu_runtime_lock(lock)
        lock_raw = runtime_lock_contract.canonical_cpu_runtime_lock_bytes(validated_lock)
    except (ValueError, TypeError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(
            "derived CPU runtime lock failed its final generic validator"
        ) from exc
    cas_raw = canonical_cpu_runtime_wheelhouse_cas_manifest_bytes(cas_manifest)
    root_inventory = cast(Mapping[str, Any], envelope["root_pin_inventory"])
    return CpuRuntimeLockIssuanceArtifacts(
        runtime_lock_bytes=lock_raw,
        runtime_lock_sha256=_hash(lock_raw),
        cas_manifest_bytes=cas_raw,
        cas_manifest_sha256=_hash(cas_raw),
        issuance_envelope_bytes=bytes(issuance_envelope_raw),
        issuance_envelope_sha256=envelope_sha,
        capture_manifest_bytes=bytes(capture_manifest_raw),
        capture_manifest_sha256=manifest_sha,
        wheelhouse_receipt_bytes=bytes(wheelhouse_receipt_raw),
        wheelhouse_receipt_sha256=receipt_sha,
        root_pin_count=len(pins),
        root_pin_inventory_sha256=cast(str, root_inventory["inventory_sha256"]),
    )


def validate_production_cpu_runtime_lock_issuance(
    artifacts: CpuRuntimeLockIssuanceArtifacts,
    *,
    expected_root_pin_inventory_sha256: str,
    expected_selected_wheel_inventory_sha256: str,
    expected_resolution_lock_sha256: str,
    expected_resolution_lock_size_bytes: int,
) -> CpuRuntimeLockIssuanceArtifacts:
    """Apply the distinct target and caller-pinned production provenance gate."""

    if type(artifacts) is not CpuRuntimeLockIssuanceArtifacts:
        _fail("production issuance gate requires exact issuance artifacts")
    expected_root_sha = _sha256(
        expected_root_pin_inventory_sha256,
        label="expected production root-pin inventory",
    )
    expected_selected_sha = _sha256(
        expected_selected_wheel_inventory_sha256,
        label="expected production selected-wheel inventory",
    )
    expected_lock_sha = _sha256(
        expected_resolution_lock_sha256,
        label="expected production resolution lock",
    )
    expected_lock_size = _integer(
        expected_resolution_lock_size_bytes,
        label="expected production resolution lock size",
        minimum=1,
    )
    if (
        _hash(artifacts.runtime_lock_bytes) != artifacts.runtime_lock_sha256
        or _hash(artifacts.cas_manifest_bytes) != artifacts.cas_manifest_sha256
        or _hash(artifacts.issuance_envelope_bytes) != artifacts.issuance_envelope_sha256
        or _hash(artifacts.capture_manifest_bytes) != artifacts.capture_manifest_sha256
        or _hash(artifacts.wheelhouse_receipt_bytes) != artifacts.wheelhouse_receipt_sha256
    ):
        _fail("production issuance artifact full-file identity differs")
    reissued = issue_matched_v3_cpu_runtime_lock(
        capture_manifest_raw=artifacts.capture_manifest_bytes,
        expected_capture_manifest_sha256=artifacts.capture_manifest_sha256,
        wheelhouse_receipt_raw=artifacts.wheelhouse_receipt_bytes,
        expected_wheelhouse_receipt_sha256=artifacts.wheelhouse_receipt_sha256,
        issuance_envelope_raw=artifacts.issuance_envelope_bytes,
        expected_issuance_envelope_sha256=artifacts.issuance_envelope_sha256,
    )
    if reissued != artifacts:
        _fail("production issuance artifacts differ from exact pure-content reissuance")
    envelope = parse_cpu_runtime_lock_issuance_envelope(
        artifacts.issuance_envelope_bytes,
        expected_file_sha256=artifacts.issuance_envelope_sha256,
    )
    bindings = cast(Mapping[str, Any], envelope["input_bindings"])
    if (
        bindings["capture_manifest_sha256"] != artifacts.capture_manifest_sha256
        or bindings["wheelhouse_receipt_sha256"] != artifacts.wheelhouse_receipt_sha256
    ):
        _fail("production envelope and retained input identities differ")
    root_inventory = cast(Mapping[str, Any], envelope["root_pin_inventory"])
    if (
        artifacts.root_pin_count != PRODUCTION_ROOT_PIN_COUNT
        or root_inventory["pin_count"] != PRODUCTION_ROOT_PIN_COUNT
        or artifacts.root_pin_inventory_sha256 != expected_root_sha
        or root_inventory["inventory_sha256"] != expected_root_sha
    ):
        _fail("production issuance requires the explicitly bound exact 36-root-pin inventory")
    cas = parse_cpu_runtime_wheelhouse_cas_manifest(
        artifacts.cas_manifest_bytes,
        expected_file_sha256=artifacts.cas_manifest_sha256,
    )
    cas_receipt = cast(Mapping[str, Any], cas["source_receipt"])
    cas_archive = cast(Mapping[str, Any], cas["source_archive"])
    if (
        cas_receipt["full_file_sha256"] != artifacts.wheelhouse_receipt_sha256
        or cas_receipt["body_sha256"] != bindings["wheelhouse_receipt_body_sha256"]
        or cas_archive["sha256"] != bindings["wheelhouse_archive_sha256"]
        or cas_archive["size_bytes"] != bindings["wheelhouse_archive_size_bytes"]
        or cas_archive["inventory_sha256"] != bindings["wheelhouse_archive_inventory_sha256"]
    ):
        _fail("production CAS manifest and envelope input identities differ")
    lock = runtime_lock_contract.parse_cpu_runtime_lock(
        artifacts.runtime_lock_bytes,
        expected_file_sha256=artifacts.runtime_lock_sha256,
    )
    try:
        validated = runtime_lock_contract.validate_production_cpu_runtime_lock(lock)
    except (ValueError, TypeError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockIssuerError(
            "derived lock failed the distinct frozen production validator"
        ) from exc
    target = cast(Mapping[str, Any], validated["target"])
    packages = cast(list[Any], validated["packages"])
    resolution_inventory = cast(Mapping[str, Any], envelope["resolution_inventory"])
    if (
        len(packages) != runtime_lock_contract.PRODUCTION_DISTRIBUTION_COUNT
        or cas["entry_count"] != runtime_lock_contract.PRODUCTION_DISTRIBUTION_COUNT
        or resolution_inventory["selected_wheel_count"]
        != runtime_lock_contract.PRODUCTION_DISTRIBUTION_COUNT
        or resolution_inventory["selected_wheel_inventory_sha256"] != expected_selected_sha
        or resolution_inventory["lock_sha256"] != expected_lock_sha
        or resolution_inventory["lock_size_bytes"] != expected_lock_size
        or target["python_version"] != runtime_lock_contract.PRODUCTION_PYTHON_VERSION
        or target["platform"] != "linux-amd64"
        or target["libc_family"] != "glibc"
        or tuple(int(item) for item in cast(str, target["libc_version"]).split("."))
        < tuple(int(item) for item in runtime_lock_contract.PRODUCTION_MINIMUM_GLIBC.split("."))
    ):
        _fail("production issuance requires the exact 104-package frozen CPU target")
    manifest_identity = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], lock["wheelhouse"])["manifest"],
    )
    lock_archive = cast(
        Mapping[str, Any],
        cast(Mapping[str, Any], lock["wheelhouse"])["archive"],
    )
    if (
        manifest_identity["size_bytes"] != len(artifacts.cas_manifest_bytes)
        or manifest_identity["sha256"] != artifacts.cas_manifest_sha256
        or manifest_identity["body_sha256"] != cas["manifest_body_sha256"]
        or manifest_identity["entry_count"] != cas["entry_count"]
        or manifest_identity["total_bytes"] != cas["total_bytes"]
        or manifest_identity["inventory_sha256"] != cas["wheel_inventory_sha256"]
        or lock_archive["size_bytes"] != cas_archive["size_bytes"]
        or lock_archive["sha256"] != cas_archive["sha256"]
        or lock_archive["manifest_sha256"] != artifacts.cas_manifest_sha256
        or lock_archive["manifest_body_sha256"] != cas["manifest_body_sha256"]
    ):
        _fail("production lock, CAS manifest, or retained archive identities differ")
    return artifacts


__all__ = [
    "CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_CLASSIFICATION",
    "CPU_RUNTIME_LOCK_ISSUANCE_RANGE_LIMITATION",
    "CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_SCHEMA_VERSION",
    "CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_STATUS",
    "CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_CLASSIFICATION",
    "CPU_RUNTIME_WHEELHOUSE_CAS_MANIFEST_STATUS",
    "PRODUCTION_ROOT_PIN_COUNT",
    "CpuRuntimeLockIssuanceArtifacts",
    "ForagerMatchedV3CpuRuntimeLockIssuerError",
    "canonical_cpu_runtime_lock_issuance_envelope_bytes",
    "canonical_cpu_runtime_wheelhouse_cas_manifest_bytes",
    "issue_matched_v3_cpu_runtime_lock",
    "parse_cpu_runtime_lock_issuance_envelope",
    "parse_cpu_runtime_wheelhouse_cas_manifest",
    "validate_cpu_runtime_lock_issuance_envelope",
    "validate_cpu_runtime_wheelhouse_cas_manifest",
    "validate_production_cpu_runtime_lock_issuance",
]
