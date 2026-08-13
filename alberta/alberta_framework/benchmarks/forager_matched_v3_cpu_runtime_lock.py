"""Pure content contract for a future matched-v3 CPU runtime lock.

This module describes and validates canonical JSON.  It deliberately does not
resolve dependencies, download or open wheels, build a wheelhouse, inspect an
interpreter, install a distribution, issue a lock, or grant execution or
scientific authority.  The generic validator admits small synthetic closures
for tests.  :func:`validate_production_cpu_runtime_lock` adds the distinct
production cardinality and target gate; no production lock is embedded here.
"""

from __future__ import annotations

import hashlib
import hmac
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final, Never, cast
from urllib.parse import unquote, urlsplit

CPU_RUNTIME_LOCK_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_runtime_lock_descriptor.v1"
)
CPU_RUNTIME_LOCK_SCHEMA_VERSION: Final = "alberta.forager_matched_v3.cpu_runtime_lock.v1"
CPU_RUNTIME_LOCK_STATUS: Final = "schema_only_no_production_lock"
CPU_RUNTIME_LOCK_CLASSIFICATION: Final = "pure_content_unexecuted_non_authorizing"
CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_runtime_lock_overlay_delta.v1"
)
CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_runtime_wheelhouse_cas_manifest.v1"
)

PRODUCTION_DISTRIBUTION_COUNT: Final = 104
PRODUCTION_PYTHON_VERSION: Final = "3.12.3"
PRODUCTION_MINIMUM_GLIBC: Final = "2.28"

_MAX_ARTIFACT_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_NODES: Final = 1_000_000
_MAX_TEXT_LENGTH: Final = 16 * 1024
_MAX_INTEGER: Final = 2**63 - 1
_MAX_PACKAGES: Final = 10_000
_MAX_REQUIRES_DIST_PER_PACKAGE: Final = 10_000
_MAX_MARKER_NESTING: Final = 64

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_GIT_SHA1_RE: Final = re.compile(r"[0-9a-f]{40}\Z")
_NAME_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_EXTRA_RE: Final = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_PYTHON_VERSION_RE: Final = re.compile(r"3\.12\.(?:0|[1-9][0-9]{0,2})\Z")
_GLIBC_VERSION_RE: Final = re.compile(r"2\.(?:0|[1-9][0-9]{0,2})\Z")
_METADATA_VERSION_RE: Final = re.compile(r"[12]\.[0-9]+\Z")
_GENERATOR_RE: Final = re.compile(r"[ -~]{1,512}\Z")
_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*\Z")
_DIRECT_REQUIREMENT_RE: Final = re.compile(
    r"(?P<name>[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)=="
    r"(?P<version>[^\s,;@()\[\]]+)\Z"
)
_REQUIRES_DIST_NAME_RE: Final = re.compile(
    r"\s*(?P<name>[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*)"
    r"(?:\[(?P<extras>[A-Za-z0-9_.-]+(?:\s*,\s*[A-Za-z0-9_.-]+)*)\])?"
    r"\s*(?:\(\s*(?P<parenthesized>[^()]*)\s*\)|(?P<plain>[^()]*))?\s*\Z"
)
_MANYLINUX_RE: Final = re.compile(r"manylinux_2_(?P<minor>[0-9]+)_x86_64\Z")
_CPYTHON_TAG_RE: Final = re.compile(r"cp3(?P<minor>[0-9]{1,2})\Z")
_REQUIRES_PYTHON_PART_RE: Final = re.compile(
    r"(?P<operator>===|==|!=|~=|<=|>=|<|>)\s*"
    r"(?P<version>[0-9]+(?:\.[0-9]+){0,2})(?P<wildcard>\.\*)?\Z"
)
_SPECIFIER_PART_RE: Final = re.compile(r"(?P<operator>===|==|!=|~=|<=|>=|<|>)\s*(?P<version>\S+)\Z")
_PEP440_VERSION_RE: Final = re.compile(
    r"""
    v?
    (?:(?P<epoch>[0-9]+)!)?
    (?P<release>[0-9]+(?:\.[0-9]+)*)
    (?P<pre>
        [-_.]?(?P<pre_label>alpha|a|beta|b|preview|pre|c|rc)
        [-_.]?(?P<pre_number>[0-9]+)?
    )?
    (?P<post>
        (?:-(?P<post_number1>[0-9]+))
        |(?:[-_.]?(?P<post_label>post|rev|r)[-_.]?(?P<post_number2>[0-9]+)?)
    )?
    (?P<dev>[-_.]?dev[-_.]?(?P<dev_number>[0-9]+)?)?
    (?:\+(?P<local>[a-z0-9]+(?:[-_.][a-z0-9]+)*))?
    \Z
    """,
    re.IGNORECASE | re.VERBOSE,
)
_PEP440_WILDCARD_RE: Final = re.compile(r"v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*\.\*\Z", re.IGNORECASE)
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
_JSON_POINTER_RE: Final = re.compile(r"/(?:[^~/]|~[01])+(?:/(?:[^~/]|~[01])+)*\Z")
_TIMESTAMP_RE: Final = re.compile(
    r"(?:19|20)[0-9]{2}-(?:0[1-9]|1[0-2])-"
    r"(?:0[1-9]|[12][0-9]|3[01])T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z\Z"
)
_ENVIRONMENT_ENTRY_RE: Final = re.compile(r"[A-Z][A-Z0-9_]{0,127}=[ -~]{0,4096}\Z")
_FORBIDDEN_SELECTED_TOKEN_RE: Final = re.compile(
    r"(?:cuda|nvidia|rocm|jax[-_.]?cuda|cu(?:11|12|13))", re.IGNORECASE
)

_UPSTREAM_REPOSITORY_ID: Final = "continual-foragax-agents"
_UPSTREAM_REPOSITORY_URL: Final = "https://github.com/steventango/continual-foragax-agents"
_UPSTREAM_COMMIT: Final = "9710f60fa30da5badc451ad7ce3ff296d5070830"
_UPSTREAM_TREE: Final = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
_UPSTREAM_ARCHIVE_SHA256: Final = "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
_UPSTREAM_ARCHIVE_SIZE: Final = 314_961_920
_UPSTREAM_LOCK_SHA256: Final = "46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd"
_UPSTREAM_PYPROJECT_SIZE: Final = 1_927
_UPSTREAM_PYPROJECT_SHA256: Final = (
    "297500b39833ac8210240dd248f93a4f6a3dab4572f11185accecaca8ffed417"
)
_ROOT_PROJECT_DISTRIBUTION: Final = "continual-foragax-agents"
_MANDATORY_VERSIONS: Final = {
    "continual-foragax": "0.55.0",
    "jax": "0.11.0",
    "jaxlib": "0.11.0",
}
_LEGACY_MANYLINUX_GLIBC: Final = {
    "manylinux1_x86_64": (2, 5),
    "manylinux2010_x86_64": (2, 12),
    "manylinux2014_x86_64": (2, 17),
}


class ForagerMatchedV3CpuRuntimeLockError(ValueError):
    """A CPU runtime descriptor or lock failed closed."""


def _raise_float(value: str) -> Never:
    raise ForagerMatchedV3CpuRuntimeLockError(f"runtime-lock JSON contains float {value!r}")


def _raise_constant(value: str) -> Never:
    raise ForagerMatchedV3CpuRuntimeLockError(
        f"runtime-lock JSON contains non-finite constant {value!r}"
    )


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "runtime-lock JSON integer exceeds its lexical bound"
        )
    return int(value)


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ForagerMatchedV3CpuRuntimeLockError(
                f"runtime-lock JSON contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _assert_plain_unaliased_json(value: Any) -> None:
    pending: list[tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ForagerMatchedV3CpuRuntimeLockError("runtime lock exceeds its node bound")
        if depth > _MAX_JSON_DEPTH:
            raise ForagerMatchedV3CpuRuntimeLockError("runtime lock exceeds its depth bound")
        if type(item) is str:
            if len(item) > _MAX_TEXT_LENGTH or any(
                ord(character) < 0x20 or ord(character) > 0x7E for character in item
            ):
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "runtime-lock strings must be bounded printable ASCII"
                )
            continue
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) not in {dict, list}:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "runtime lock must contain exact JSON scalar and container types"
            )
        identity = id(item)
        if identity in seen:
            raise ForagerMatchedV3CpuRuntimeLockError("runtime lock contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    raise ForagerMatchedV3CpuRuntimeLockError(
                        "runtime-lock object keys must be exact strings"
                    )
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict:
        raise ForagerMatchedV3CpuRuntimeLockError("canonical runtime-lock root must be an object")
    _assert_plain_unaliased_json(value)
    try:
        result = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "runtime lock is not canonical finite ASCII JSON"
        ) from exc
    if len(result) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3CpuRuntimeLockError("runtime lock exceeds its byte bound")
    return result


def _strict_json_load(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_ARTIFACT_BYTES:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "runtime-lock input must be nonempty exact bytes within the byte bound"
        )
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "runtime lock must have one canonical trailing newline"
        )
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedV3CpuRuntimeLockError("runtime lock must be ASCII") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_raise_constant,
            parse_float=_raise_float,
            parse_int=_parse_bounded_int,
        )
    except ForagerMatchedV3CpuRuntimeLockError:
        raise
    except (RecursionError, json.JSONDecodeError, ValueError) as exc:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "runtime-lock input is not bounded strict JSON"
        ) from exc
    if type(value) is not dict:
        raise ForagerMatchedV3CpuRuntimeLockError("runtime-lock root must be an object")
    result = cast(dict[str, Any], value)
    _assert_plain_unaliased_json(result)
    if not hmac.compare_digest(_canonical_json(result), raw):
        raise ForagerMatchedV3CpuRuntimeLockError("runtime lock is not in canonical form")
    return result


def _exact_keys(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != expected:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _array(value: Any, label: str) -> list[Any]:
    if type(value) is not list:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if type(value) is not str or not value:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} must be a nonempty exact string")
    return value


def _bounded_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_INTEGER:
        raise ForagerMatchedV3CpuRuntimeLockError(
            f"{label} must be an exact integer in [{minimum}, {_MAX_INTEGER}]"
        )
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} must be an exact boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if _SHA256_RE.fullmatch(result) is None or result == "0" * 64:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} must be one nonzero lowercase SHA-256")
    return result


def _git_sha1(value: Any, label: str) -> str:
    result = _string(value, label)
    if _GIT_SHA1_RE.fullmatch(result) is None or result == "0" * 40:
        raise ForagerMatchedV3CpuRuntimeLockError(
            f"{label} must be one nonzero lowercase Git SHA-1"
        )
    return result


def _relative_path(value: Any, label: str) -> str:
    result = _string(value, label)
    if (
        _RELATIVE_PATH_RE.fullmatch(result) is None
        or any(part in {"", ".", ".."} for part in result.split("/"))
        or result.startswith("/")
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            f"{label} must be a canonical relative POSIX path"
        )
    return result


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _name(value: Any, label: str) -> str:
    result = _string(value, label)
    if _NAME_RE.fullmatch(result) is None or _normalized_name(result) != result:
        raise ForagerMatchedV3CpuRuntimeLockError(
            f"{label} must be a canonical PEP-503 distribution name"
        )
    return result


def _version(value: Any, label: str) -> str:
    result = _string(value, label)
    if len(result) > 512 or _PEP440_VERSION_RE.fullmatch(result) is None:
        raise ForagerMatchedV3CpuRuntimeLockError(
            f"{label} is not a bounded PEP 440 version identity"
        )
    return result


def _validate_pep440_specifier_set(value: str) -> None:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Requires-Dist specifier set is empty or incomplete"
        )
    for part in parts:
        match = _SPECIFIER_PART_RE.fullmatch(part)
        if match is None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist specifier syntax is invalid or incomplete"
            )
        operator = match.group("operator")
        version = match.group("version")
        if len(version) > 512:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist specifier version exceeds its bound"
            )
        if operator == "===":
            continue
        if version.endswith(".*"):
            if operator not in {"==", "!="} or _PEP440_WILDCARD_RE.fullmatch(version) is None:
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "Requires-Dist wildcard is invalid for its operator or version"
                )
            continue
        version_match = _PEP440_VERSION_RE.fullmatch(version)
        if version_match is None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist specifier contains an invalid PEP 440 version"
            )
        if version_match.group("local") is not None and operator not in {"==", "!="}:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist ordered or compatible specifier contains a local version"
            )
        if operator == "~=" and len(version_match.group("release").split(".")) < 2:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist compatible-release specifier needs two release segments"
            )


def _body_sha256(value: Mapping[str, Any], body_field: str) -> str:
    body = dict(value)
    supplied = body.pop(body_field, None)
    _sha256(supplied, body_field)
    calculated = hashlib.sha256(_canonical_json(body)).hexdigest()
    if not hmac.compare_digest(cast(str, supplied), calculated):
        raise ForagerMatchedV3CpuRuntimeLockError(f"{body_field} disagrees with canonical body")
    return calculated


def _forbidden_selected_token(value: str) -> bool:
    return _FORBIDDEN_SELECTED_TOKEN_RE.search(value) is not None


def _reject_forbidden_selected_token(value: str, label: str) -> None:
    if _forbidden_selected_token(value):
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label} contains a forbidden accelerator token")


def _json_strings(value: Any) -> list[str]:
    pending = [value]
    result: list[str] = []
    while pending:
        item = pending.pop()
        if type(item) is str:
            result.append(item)
        elif type(item) is list:
            pending.extend(item)
        elif type(item) is dict:
            for key, child in cast(dict[str, Any], item).items():
                result.append(key)
                pending.append(child)
    return result


def _version_tuple(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split("."))
    return cast(tuple[int, int, int], parts + (0,) * (3 - len(parts)))


def _requires_python_allows(value: str, python_version: str) -> bool:
    target = _version_tuple(python_version)
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        return False
    for part in parts:
        match = _REQUIRES_PYTHON_PART_RE.fullmatch(part)
        if match is None:
            return False
        requested = _version_tuple(match.group("version"))
        operator = match.group("operator")
        wildcard = match.group("wildcard") is not None
        if wildcard:
            prefix = tuple(int(item) for item in match.group("version").split("."))
            equal = target[: len(prefix)] == prefix
            if operator == "==" and not equal:
                return False
            if operator == "!=" and equal:
                return False
            if operator not in {"==", "!="}:
                return False
            continue
        compatible_release = target >= requested and target[0] == requested[0]
        if operator == "~=":
            requested_parts = match.group("version").split(".")
            if len(requested_parts) < 2:
                return False
            prefix_length = len(requested_parts) - 1
            compatible_release = compatible_release and (
                target[:prefix_length] == requested[:prefix_length]
            )
        accepted = {
            "===": target == requested,
            "==": target == requested,
            "!=": target != requested,
            "<=": target <= requested,
            ">=": target >= requested,
            "<": target < requested,
            ">": target > requested,
            "~=": compatible_release,
        }[operator]
        if not accepted:
            return False
    return True


def _parse_marker_expression(value: str) -> tuple[Any, ...]:
    """Parse one bounded marker expression without claiming its target truth."""

    offset = 0
    tokens: list[tuple[str, str]] = []
    while offset < len(value):
        match = _MARKER_TOKEN_RE.match(value, offset)
        if match is None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist marker contains unsupported or trailing syntax"
            )
        kind = cast(str, match.lastgroup)
        token = match.group(kind)
        if token is None:
            raise AssertionError("matched marker token has no value")
        tokens.append((kind, token))
        offset = match.end()
    if offset != len(value) or not tokens:
        raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist marker is incomplete")

    position = 0

    def parse_operand() -> tuple[str, str]:
        nonlocal position
        if position >= len(tokens) or tokens[position][0] not in {"variable", "string"}:
            raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist marker operand is invalid")
        result = tokens[position]
        position += 1
        return result

    def parse_comparison() -> tuple[Any, ...]:
        nonlocal position
        left = parse_operand()
        if position >= len(tokens) or tokens[position][0] != "operator":
            raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist marker operator is invalid")
        operator = tokens[position][1]
        position += 1
        right = parse_operand()
        return ("compare", left, operator, right)

    def parse_atom(depth: int) -> tuple[Any, ...]:
        nonlocal position
        if depth > _MAX_MARKER_NESTING:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist marker exceeds its nesting bound"
            )
        if position < len(tokens) and tokens[position][0] == "left":
            position += 1
            expression = parse_or(depth + 1)
            if position >= len(tokens) or tokens[position][0] != "right":
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "Requires-Dist marker parentheses are unbalanced"
                )
            position += 1
            return expression
        return parse_comparison()

    def parse_and(depth: int) -> tuple[Any, ...]:
        nonlocal position
        expressions = [parse_atom(depth)]
        while (
            position < len(tokens)
            and tokens[position][0] == "boolean"
            and tokens[position][1] == "and"
        ):
            position += 1
            expressions.append(parse_atom(depth))
        if len(expressions) == 1:
            return expressions[0]
        return ("and", tuple(expressions))

    def parse_or(depth: int) -> tuple[Any, ...]:
        nonlocal position
        expressions = [parse_and(depth)]
        while (
            position < len(tokens)
            and tokens[position][0] == "boolean"
            and tokens[position][1] == "or"
        ):
            position += 1
            expressions.append(parse_and(depth))
        if len(expressions) == 1:
            return expressions[0]
        return ("or", tuple(expressions))

    result = parse_or(0)
    if position != len(tokens):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Requires-Dist marker contains misplaced parentheses or boolean operators"
        )
    return result


def _validate_marker_syntax(value: str) -> None:
    """Require complete, balanced marker syntax without claiming marker truth."""

    _parse_marker_expression(value)


def _marker_guaranteed_extras(value: str) -> frozenset[str]:
    """Return extra equalities that every truth path through a marker requires."""

    def guaranteed(expression: tuple[Any, ...]) -> frozenset[str]:
        kind = expression[0]
        if kind == "compare":
            left = cast(tuple[str, str], expression[1])
            operator = cast(str, expression[2])
            right = cast(tuple[str, str], expression[3])
            if operator != "==":
                return frozenset()
            if left == ("variable", "extra") and right[0] == "string":
                return frozenset({_normalized_name(right[1][1:-1])})
            if right == ("variable", "extra") and left[0] == "string":
                return frozenset({_normalized_name(left[1][1:-1])})
            return frozenset()
        children = cast(tuple[tuple[Any, ...], ...], expression[1])
        child_values = [guaranteed(child) for child in children]
        if kind == "and":
            return frozenset().union(*child_values)
        if kind == "or":
            result = set(child_values[0])
            for values in child_values[1:]:
                result.intersection_update(values)
            return frozenset(result)
        raise AssertionError("validated marker expression has an unknown node")

    return guaranteed(_parse_marker_expression(value))


def _validate_requirement_body(value: str) -> tuple[str, tuple[str, ...]]:
    """Consume one complete non-URL PEP 508 name/extras/specifier body."""

    if (
        re.search(
            r"(?:\s@\s|(?:git|hg|svn|bzr)\+|file:|https?://|(?:^|\s)-e(?:\s|$))",
            value,
            re.IGNORECASE,
        )
        is not None
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Requires-Dist may not use VCS, URL, editable, or path sources"
        )
    match = _REQUIRES_DIST_NAME_RE.fullmatch(value)
    if match is None:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Requires-Dist requirement body is not fully consumed"
        )
    specifier = match.group("parenthesized")
    if specifier is None:
        specifier = match.group("plain")
    if specifier is not None and specifier.strip():
        _validate_pep440_specifier_set(specifier.strip())
    extras_text = match.group("extras")
    extras = (
        ()
        if extras_text is None
        else tuple(_normalized_name(extra.strip()) for extra in extras_text.split(","))
    )
    if len(extras) != len(set(extras)):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Requires-Dist extras must normalize to unique names"
        )
    return _normalized_name(match.group("name")), tuple(sorted(extras))


def _glibc_requirement(platform_tag: str) -> tuple[int, int] | None:
    if platform_tag == "any":
        return None
    if platform_tag in _LEGACY_MANYLINUX_GLIBC:
        return _LEGACY_MANYLINUX_GLIBC[platform_tag]
    match = _MANYLINUX_RE.fullmatch(platform_tag)
    if match is None:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheel platform tag must be pure-any or x86_64 glibc manylinux"
        )
    return (2, int(match.group("minor")))


def _tag_is_python_312_compatible(python_tag: str, abi_tag: str) -> bool:
    if python_tag in {"py3", "py312"}:
        return abi_tag == "none"
    if python_tag == "cp312":
        return abi_tag in {"cp312", "abi3"}
    match = _CPYTHON_TAG_RE.fullmatch(python_tag)
    return match is not None and int(match.group("minor")) <= 12 and abi_tag == "abi3"


def _validate_tags(
    filename: str,
    tags_value: Any,
    *,
    name: str,
    version: str,
    glibc_version: str,
) -> list[str]:
    tags = _array(tags_value, "wheel tags")
    if not tags or any(type(tag) is not str for tag in tags):
        raise ForagerMatchedV3CpuRuntimeLockError("wheel tags must be nonempty exact strings")
    exact_tags = cast(list[str], tags)
    if exact_tags != sorted(set(exact_tags)):
        raise ForagerMatchedV3CpuRuntimeLockError("wheel tags must be sorted and unique")
    stem = filename.removesuffix(".whl")
    try:
        prefix, python_field, abi_field, platform_field = stem.rsplit("-", 3)
    except ValueError as exc:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheel filename has no complete tag triple"
        ) from exc
    expected_prefix = f"{name.replace('-', '_')}-{version.replace('-', '_')}"
    if (
        prefix.casefold() != expected_prefix.casefold()
        and re.fullmatch(
            re.escape(expected_prefix) + r"-[0-9][A-Za-z0-9_.]*",
            prefix,
            re.IGNORECASE,
        )
        is None
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheel filename distribution, version, or build tag disagrees"
        )
    python_tags = python_field.split(".")
    abi_tags = abi_field.split(".")
    platform_tags = platform_field.split(".")
    if any(not item for item in (*python_tags, *abi_tags, *platform_tags)):
        raise ForagerMatchedV3CpuRuntimeLockError("wheel filename has an empty compressed tag")
    expected_tags = sorted(
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag, abi_tag, platform_tag in itertools.product(
            python_tags, abi_tags, platform_tags
        )
    )
    if exact_tags != expected_tags:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheel tags do not exactly expand the wheel filename tags"
        )
    target_glibc = tuple(int(item) for item in glibc_version.split("."))
    compatible = False
    for python_tag, abi_tag, platform_tag in (tag.split("-", 2) for tag in exact_tags):
        if abi_tag not in {"none", "abi3", "cp312"}:
            raise ForagerMatchedV3CpuRuntimeLockError("wheel contains a wrong ABI tag")
        glibc_requirement = _glibc_requirement(platform_tag)
        if glibc_requirement is not None and glibc_requirement > target_glibc:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "wheel manylinux tag exceeds the target glibc"
            )
        if platform_tag == "any" and abi_tag != "none":
            raise ForagerMatchedV3CpuRuntimeLockError("platform-any wheel must use the none ABI")
        compatible |= _tag_is_python_312_compatible(python_tag, abi_tag)
    if not compatible:
        raise ForagerMatchedV3CpuRuntimeLockError("wheel has no tag compatible with CPython 3.12")
    return exact_tags


def _validate_https_wheel_url(value: Any, filename: str) -> str:
    url = _string(value, "wheel source_url")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.pythonhosted.org"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/packages/")
        or unquote(parsed.path) != parsed.path
        or parsed.path.rsplit("/", 1)[-1] != filename
        or any(part in {"", ".", ".."} for part in parsed.path.split("/")[1:])
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheel source_url must be one canonical files.pythonhosted.org HTTPS wheel URL"
        )
    _reject_forbidden_selected_token(url, "wheel source_url")
    return url


def _validate_file_identity(value: Any, expected_path: str, label: str) -> dict[str, Any]:
    item = _exact_keys(value, frozenset({"path", "size_bytes", "sha256"}), label)
    if _relative_path(item["path"], f"{label}.path") != expected_path:
        raise ForagerMatchedV3CpuRuntimeLockError(f"{label}.path disagrees")
    _bounded_int(item["size_bytes"], f"{label}.size_bytes", minimum=1)
    _sha256(item["sha256"], f"{label}.sha256")
    return item


def _validate_requires_dist(
    value: Any,
    *,
    owner: str,
    provided_extras: frozenset[str],
    selected_extras: frozenset[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, ...]]]:
    items = _array(value, f"{owner} Requires-Dist")
    if len(items) > _MAX_REQUIRES_DIST_PER_PACKAGE:
        raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist inventory exceeds its bound")
    result: list[dict[str, Any]] = []
    requested_extras_by_item: list[tuple[str, ...]] = []
    raw_seen: set[str] = set()
    for index, raw_item in enumerate(items):
        label = f"{owner} Requires-Dist[{index}]"
        item = _exact_keys(
            raw_item,
            frozenset({"raw", "name", "marker", "active", "selected_version"}),
            label,
        )
        raw = _string(item["raw"], f"{label}.raw")
        if raw in raw_seen:
            raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist raw entry is duplicated")
        raw_seen.add(raw)
        requirement, separator, marker_text = raw.partition(";")
        raw_dependency_name, requested_extras = _validate_requirement_body(requirement)
        marker: str | None
        if separator:
            marker = marker_text.strip()
            if not marker or item["marker"] != marker:
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "Requires-Dist marker does not preserve the exact marker text"
                )
            _validate_marker_syntax(marker)
        else:
            marker = None
            if item["marker"] is not None:
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "unconditional Requires-Dist must use a null marker"
                )
        dependency_name = _name(item["name"], f"{label}.name")
        if dependency_name != raw_dependency_name:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Dist normalized name disagrees with its raw entry"
            )
        active = _boolean(item["active"], f"{label}.active")
        if marker is None and not active:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "unconditional Requires-Dist cannot be inactive"
            )
        if active:
            _version(item["selected_version"], f"{label}.selected_version")
            if _forbidden_selected_token(dependency_name):
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "active Requires-Dist selects a forbidden accelerator distribution"
                )
            if any(_forbidden_selected_token(extra) for extra in requested_extras):
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "active Requires-Dist selects a forbidden accelerator extra"
                )
        elif item["selected_version"] is not None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "inactive Requires-Dist must have a null selected_version"
            )
        if not active and _forbidden_selected_token(dependency_name):
            if marker is None:
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "forbidden accelerator dependency is not optional"
                )
            guaranteed_extras = _marker_guaranteed_extras(marker)
            if (
                not guaranteed_extras
                or not guaranteed_extras.issubset(provided_extras)
                or not guaranteed_extras.isdisjoint(selected_extras)
            ):
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "inactive accelerator metadata is not proven to be a declared unselected extra"
                )
        result.append(item)
        requested_extras_by_item.append(requested_extras)
    if [item["raw"] for item in result] != sorted(item["raw"] for item in result):
        raise ForagerMatchedV3CpuRuntimeLockError("Requires-Dist entries must be raw-sorted")
    return result, requested_extras_by_item


def _validate_wheel(
    value: Any,
    *,
    package_name: str,
    package_version: str,
    selected_extras: frozenset[str],
    python_version: str,
    glibc_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[str, ...]]]:
    wheel = _exact_keys(
        value,
        frozenset(
            {
                "filename",
                "source_url",
                "cas_key",
                "size_bytes",
                "sha256",
                "tags",
                "metadata",
                "wheel",
                "record",
                "wheel_body_sha256",
            }
        ),
        f"{package_name} selected wheel",
    )
    filename = _string(wheel["filename"], "wheel filename")
    if not filename.endswith(".whl") or "/" in filename or "\\" in filename:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "selected artifact must be one basename .whl, never an sdist or path"
        )
    _reject_forbidden_selected_token(filename, "wheel filename")
    wheel_sha256 = _sha256(wheel["sha256"], "wheel sha256")
    _bounded_int(wheel["size_bytes"], "wheel size_bytes", minimum=1)
    _validate_https_wheel_url(wheel["source_url"], filename)
    expected_cas_key = f"sha256/{wheel_sha256[:2]}/{wheel_sha256}/{filename}"
    if _relative_path(wheel["cas_key"], "wheel cas_key") != expected_cas_key:
        raise ForagerMatchedV3CpuRuntimeLockError("wheel CAS key disagrees with its content hash")
    tags = _validate_tags(
        filename,
        wheel["tags"],
        name=package_name,
        version=package_version,
        glibc_version=glibc_version,
    )
    dist_info = f"{package_name.replace('-', '_')}-{package_version}.dist-info"
    metadata = _exact_keys(
        wheel["metadata"],
        frozenset(
            {
                "path",
                "size_bytes",
                "sha256",
                "metadata_version",
                "name",
                "version",
                "requires_python",
                "provides_extra",
                "requires_dist",
            }
        ),
        f"{package_name} METADATA",
    )
    if (
        _relative_path(metadata["path"], "METADATA.path").casefold()
        != f"{dist_info}/METADATA".casefold()
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("METADATA path disagrees with wheel identity")
    _bounded_int(metadata["size_bytes"], "METADATA.size_bytes", minimum=1)
    _sha256(metadata["sha256"], "METADATA.sha256")
    if (
        _METADATA_VERSION_RE.fullmatch(_string(metadata["metadata_version"], "Metadata-Version"))
        is None
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("Metadata-Version is invalid")
    if _name(metadata["name"], "METADATA Name") != package_name:
        raise ForagerMatchedV3CpuRuntimeLockError("METADATA Name is noncanonical or disagrees")
    if _version(metadata["version"], "METADATA Version") != package_version:
        raise ForagerMatchedV3CpuRuntimeLockError("METADATA Version disagrees")
    requires_python_value = metadata["requires_python"]
    if requires_python_value is not None:
        requires_python = _string(requires_python_value, "Requires-Python")
        if not _requires_python_allows(requires_python, python_version):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Requires-Python does not admit the exact target interpreter"
            )
    provides_extra = _array(metadata["provides_extra"], "Provides-Extra")
    if any(type(extra) is not str for extra in provides_extra):
        raise ForagerMatchedV3CpuRuntimeLockError("Provides-Extra must contain exact strings")
    exact_provides_extra = cast(list[str], provides_extra)
    if exact_provides_extra != sorted(set(exact_provides_extra)):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "Provides-Extra entries must be sorted and unique"
        )
    for extra in exact_provides_extra:
        if _EXTRA_RE.fullmatch(extra) is None or _normalized_name(extra) != extra:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "Provides-Extra entries must be canonical names"
            )
    if not selected_extras.issubset(exact_provides_extra):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "selected package extras must be declared by Provides-Extra"
        )
    requires_dist, requested_extras = _validate_requires_dist(
        metadata["requires_dist"],
        owner=package_name,
        provided_extras=frozenset(exact_provides_extra),
        selected_extras=selected_extras,
    )
    wheel_metadata = _exact_keys(
        wheel["wheel"],
        frozenset(
            {
                "path",
                "size_bytes",
                "sha256",
                "wheel_version",
                "generator",
                "root_is_purelib",
                "tags",
            }
        ),
        f"{package_name} WHEEL",
    )
    if (
        _relative_path(wheel_metadata["path"], "WHEEL.path").casefold()
        != f"{dist_info}/WHEEL".casefold()
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("WHEEL path disagrees with wheel identity")
    _bounded_int(wheel_metadata["size_bytes"], "WHEEL.size_bytes", minimum=1)
    _sha256(wheel_metadata["sha256"], "WHEEL.sha256")
    if wheel_metadata["wheel_version"] != "1.0":
        raise ForagerMatchedV3CpuRuntimeLockError("WHEEL version must be 1.0")
    if _GENERATOR_RE.fullmatch(_string(wheel_metadata["generator"], "WHEEL generator")) is None:
        raise ForagerMatchedV3CpuRuntimeLockError("WHEEL generator is invalid")
    _boolean(wheel_metadata["root_is_purelib"], "WHEEL Root-Is-Purelib")
    if wheel_metadata["tags"] != tags:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "WHEEL Tag headers disagree with the selected filename tags"
        )
    record = _exact_keys(
        wheel["record"],
        frozenset({"path", "size_bytes", "sha256", "entry_count", "entries_sha256"}),
        f"{package_name} RECORD",
    )
    if _relative_path(record["path"], "RECORD.path").casefold() != f"{dist_info}/RECORD".casefold():
        raise ForagerMatchedV3CpuRuntimeLockError("RECORD path disagrees with wheel identity")
    _bounded_int(record["size_bytes"], "RECORD.size_bytes", minimum=1)
    _sha256(record["sha256"], "RECORD.sha256")
    _bounded_int(record["entry_count"], "RECORD.entry_count", minimum=1)
    _sha256(record["entries_sha256"], "RECORD.entries_sha256")
    _body_sha256(wheel, "wheel_body_sha256")
    return wheel, requires_dist, requested_extras


def _validate_target(value: Any) -> dict[str, Any]:
    target = _exact_keys(
        value,
        frozenset(
            {
                "implementation",
                "python_version",
                "python_tag",
                "abi_tag",
                "os",
                "architecture",
                "platform",
                "libc_family",
                "libc_version",
                "cpu_only",
            }
        ),
        "runtime target",
    )
    if (
        target["implementation"] != "CPython"
        or _PYTHON_VERSION_RE.fullmatch(_string(target["python_version"], "python_version")) is None
        or target["python_tag"] != "cp312"
        or target["abi_tag"] != "cp312"
        or target["os"] != "linux"
        or target["architecture"] != "x86_64"
        or target["platform"] != "linux-amd64"
        or target["libc_family"] != "glibc"
        or _GLIBC_VERSION_RE.fullmatch(_string(target["libc_version"], "libc_version")) is None
        or _boolean(target["cpu_only"], "target.cpu_only") is not True
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "target must be exact-patch CPython 3.12 on glibc Linux x86_64 CPU"
        )
    return target


def _validate_upstream(value: Any) -> dict[str, Any]:
    upstream = _exact_keys(
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
        "upstream binding",
    )
    if (
        upstream["repository_id"] != _UPSTREAM_REPOSITORY_ID
        or upstream["repository_url"] != _UPSTREAM_REPOSITORY_URL
        or _git_sha1(upstream["commit_git_sha1"], "upstream commit") != _UPSTREAM_COMMIT
        or _git_sha1(upstream["tree_git_sha1"], "upstream tree") != _UPSTREAM_TREE
        or upstream["root_project_distribution"] != _ROOT_PROJECT_DISTRIBUTION
        or _boolean(upstream["root_project_installed"], "root_project_installed") is not False
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("upstream source identity drifted")
    archive = _exact_keys(
        upstream["archive"], frozenset({"size_bytes", "sha256"}), "upstream archive"
    )
    if (
        _bounded_int(archive["size_bytes"], "upstream archive size", minimum=1)
        != _UPSTREAM_ARCHIVE_SIZE
        or _sha256(archive["sha256"], "upstream archive sha256") != _UPSTREAM_ARCHIVE_SHA256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("upstream archive identity drifted")
    pyproject = _validate_file_identity(
        upstream["pyproject"], "pyproject.toml", "upstream pyproject"
    )
    if (
        pyproject["size_bytes"] != _UPSTREAM_PYPROJECT_SIZE
        or pyproject["sha256"] != _UPSTREAM_PYPROJECT_SHA256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("upstream pyproject identity drifted")
    lock = _validate_file_identity(upstream["lock"], "uv.lock", "upstream lock")
    if lock["sha256"] != _UPSTREAM_LOCK_SHA256:
        raise ForagerMatchedV3CpuRuntimeLockError("upstream uv.lock identity drifted")
    return upstream


def _validate_direct_requirements(value: Any) -> dict[str, str]:
    requirements = _array(value, "direct requirements")
    if not requirements or any(type(item) is not str for item in requirements):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "direct requirements must be nonempty exact strings"
        )
    exact = cast(list[str], requirements)
    if exact != sorted(set(exact)):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "direct requirements must be sorted and duplicate-free"
        )
    result: dict[str, str] = {}
    for requirement in exact:
        _reject_forbidden_selected_token(requirement, "direct requirement")
        match = _DIRECT_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "direct requirements must be exact name==version registry pins"
            )
        name = _normalized_name(match.group("name"))
        if name != match.group("name") or name in result or name == _ROOT_PROJECT_DISTRIBUTION:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "direct requirement name is aliased, duplicated, or installs the root project"
            )
        result[name] = _version(match.group("version"), f"direct requirement {name} version")
    for name, version in _MANDATORY_VERSIONS.items():
        if result.get(name) != version:
            raise ForagerMatchedV3CpuRuntimeLockError(
                f"direct requirement {name} must be pinned to {version}"
            )
    return result


def _validate_resolution(
    value: Any, target: Mapping[str, Any]
) -> tuple[dict[str, str], str, frozenset[str]]:
    resolution = _exact_keys(
        value,
        frozenset(
            {
                "selected_extras",
                "marker_environment",
                "marker_environment_sha256",
                "direct_requirements",
            }
        ),
        "resolution",
    )
    extras = _array(resolution["selected_extras"], "resolution selected_extras")
    if any(type(extra) is not str for extra in extras):
        raise ForagerMatchedV3CpuRuntimeLockError("selected extras must be exact strings")
    exact_extras = cast(list[str], extras)
    if exact_extras != sorted(set(exact_extras)):
        raise ForagerMatchedV3CpuRuntimeLockError("selected extras must be sorted and unique")
    for extra in exact_extras:
        if _EXTRA_RE.fullmatch(extra) is None:
            raise ForagerMatchedV3CpuRuntimeLockError("selected extra is noncanonical")
        _reject_forbidden_selected_token(extra, "selected extra")
    marker = _exact_keys(
        resolution["marker_environment"],
        frozenset(
            {
                "implementation_name",
                "implementation_version",
                "os_name",
                "platform_machine",
                "platform_python_implementation",
                "platform_release",
                "platform_system",
                "platform_version",
                "python_full_version",
                "python_version",
                "sys_platform",
            }
        ),
        "marker environment",
    )
    expected_marker = {
        "implementation_name": "cpython",
        "implementation_version": target["python_version"],
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "python_full_version": target["python_version"],
        "python_version": "3.12",
        "sys_platform": "linux",
    }
    for field in ("platform_release", "platform_version"):
        _string(marker[field], f"marker_environment.{field}")
        expected_marker[field] = marker[field]
    expected_marker["platform_system"] = "Linux"
    if marker != expected_marker:
        raise ForagerMatchedV3CpuRuntimeLockError("marker environment disagrees with target")
    marker_environment_sha256 = hashlib.sha256(_canonical_json(marker)).hexdigest()
    if (
        _sha256(resolution["marker_environment_sha256"], "marker environment sha256")
        != marker_environment_sha256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("marker environment hash disagrees")
    direct = _validate_direct_requirements(resolution["direct_requirements"])
    return direct, marker_environment_sha256, frozenset(exact_extras)


def _validate_overlay(
    value: Any,
    *,
    pyproject_sha256: str,
    direct_requirements: Sequence[Any],
) -> None:
    overlay = _exact_keys(
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
        "overlay delta",
    )
    expected_direct_sha256 = hashlib.sha256(
        _canonical_json({"direct_requirements": list(direct_requirements)})
    ).hexdigest()
    if (
        overlay["schema_version"] != CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION
        or overlay["base_pyproject_sha256"] != pyproject_sha256
        or overlay["base_lock_sha256"] != _UPSTREAM_LOCK_SHA256
        or overlay["delta_format"] != "canonical_json_operations_v1"
        or overlay["direct_requirements_sha256"] != expected_direct_sha256
        or _boolean(overlay["source_builds_allowed"], "source_builds_allowed") is not False
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("overlay base or policy binding drifted")
    operations = _array(overlay["operations"], "overlay operations")
    if not operations:
        raise ForagerMatchedV3CpuRuntimeLockError("overlay operations may not be empty")
    validated_operations: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, raw_operation in enumerate(operations):
        operation = _exact_keys(
            raw_operation,
            frozenset({"op", "path", "expected", "replacement", "operation_body_sha256"}),
            f"overlay operation[{index}]",
        )
        op = _string(operation["op"], f"overlay operation[{index}].op")
        path = _string(operation["path"], f"overlay operation[{index}].path")
        if (
            op not in {"add", "remove", "replace"}
            or _JSON_POINTER_RE.fullmatch(path) is None
            or not path.startswith(("/pyproject/", "/lock/"))
            or path in paths
        ):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "overlay operation kind/path is invalid, duplicated, or out of scope"
            )
        paths.add(path)
        expected = operation["expected"]
        replacement = operation["replacement"]
        if (
            (op == "add" and (expected is not None or replacement is None))
            or (op == "remove" and (expected is None or replacement is not None))
            or (op == "replace" and (expected is None or replacement is None))
        ):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "overlay operation expected/replacement values disagree with its kind"
            )
        if op == "replace" and hmac.compare_digest(
            _canonical_json({"value": expected}),
            _canonical_json({"value": replacement}),
        ):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "overlay replace operation must change its exact value"
            )
        if replacement is not None:
            for text in _json_strings(replacement):
                _reject_forbidden_selected_token(text, "overlay replacement")
        _body_sha256(operation, "operation_body_sha256")
        validated_operations.append(operation)
    if validated_operations != sorted(
        validated_operations, key=lambda operation: (operation["path"], operation["op"])
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "overlay operations must be sorted by exact path and operation"
        )
    if overlay["operation_count"] != len(validated_operations):
        raise ForagerMatchedV3CpuRuntimeLockError("overlay operation count disagrees")
    _bounded_int(overlay["operation_count"], "overlay operation count", minimum=1)
    expected_operations_sha256 = hashlib.sha256(
        _canonical_json({"operations": validated_operations})
    ).hexdigest()
    if (
        _sha256(overlay["operations_sha256"], "overlay operations sha256")
        != expected_operations_sha256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("overlay operations hash disagrees")
    _body_sha256(overlay, "overlay_body_sha256")


def _validate_solver(
    value: Any,
    marker_environment_sha256: str,
    target: Mapping[str, Any],
) -> None:
    solver = _exact_keys(
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
        "solver provenance",
    )
    if (
        _boolean(solver["informational_only"], "solver informational_only") is not True
        or solver["solver"] != "uv"
        or solver["interpreter_implementation"] != "CPython"
        or solver["interpreter_version"] != target["python_version"]
        or solver["marker_environment_sha256"] != marker_environment_sha256
        or solver["index_url"] != "https://pypi.org/simple"
        or _boolean(solver["trusted_for_acceptance"], "solver trusted_for_acceptance") is not False
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "solver provenance must remain informational and target-bound"
        )
    solver_version = _version(solver["solver_version"], "solver version")
    _reject_forbidden_selected_token(solver_version, "solver version")
    argv = _array(solver["argv"], "solver argv")
    if not argv or any(type(argument) is not str or not argument for argument in argv):
        raise ForagerMatchedV3CpuRuntimeLockError("solver argv must contain nonempty exact strings")
    exact_argv = cast(list[str], argv)
    for argument in exact_argv:
        _reject_forbidden_selected_token(argument, "solver argv")
    expected_argv_sha256 = hashlib.sha256(_canonical_json({"argv": exact_argv})).hexdigest()
    if _sha256(solver["argv_sha256"], "solver argv sha256") != expected_argv_sha256:
        raise ForagerMatchedV3CpuRuntimeLockError("solver argv hash disagrees")
    environment = _array(solver["environment"], "solver environment")
    if not environment or any(type(entry) is not str for entry in environment):
        raise ForagerMatchedV3CpuRuntimeLockError("solver environment must contain exact strings")
    exact_environment = cast(list[str], environment)
    if exact_environment != sorted(set(exact_environment)):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "solver environment must be sorted and duplicate-free"
        )
    environment_names: set[str] = set()
    for entry in exact_environment:
        if _ENVIRONMENT_ENTRY_RE.fullmatch(entry) is None:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "solver environment entry must be canonical NAME=value"
            )
        environment_name = entry.partition("=")[0]
        if environment_name in environment_names:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "solver environment variable name is duplicated"
            )
        environment_names.add(environment_name)
        _reject_forbidden_selected_token(entry, "solver environment")
    expected_environment_sha256 = hashlib.sha256(
        _canonical_json({"environment": exact_environment})
    ).hexdigest()
    if (
        _sha256(solver["environment_sha256"], "solver environment sha256")
        != expected_environment_sha256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("solver environment hash disagrees")
    timestamp = _string(solver["index_capture_timestamp_utc"], "index capture timestamp")
    if _TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "index capture timestamp must be exact second-resolution UTC"
        )
    for field in (
        "interpreter_binary_sha256",
        "solver_binary_sha256",
        "resolution_input_sha256",
        "resolution_report_sha256",
    ):
        _sha256(solver[field], f"solver {field}")
    _bounded_int(solver["resolution_report_size_bytes"], "resolution report size", minimum=1)


def _validate_packages(
    value: Any,
    *,
    direct_requirements: Mapping[str, str],
    resolution_selected_extras: frozenset[str],
    python_version: str,
    glibc_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_packages = _array(value, "packages")
    if not 1 <= len(raw_packages) <= _MAX_PACKAGES:
        raise ForagerMatchedV3CpuRuntimeLockError("package count violates the generic bound")
    packages: list[dict[str, Any]] = []
    wheels: list[dict[str, Any]] = []
    dependencies: list[dict[str, Any]] = []
    names: set[str] = set()
    filenames: set[str] = set()
    hashes: set[str] = set()
    versions: dict[str, str] = {}
    selected_extras_by_name: dict[str, frozenset[str]] = {}
    selected_extras_union: set[str] = set()
    dependency_requested_extras: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for index, raw_package in enumerate(raw_packages):
        package = _exact_keys(
            raw_package,
            frozenset(
                {
                    "name",
                    "version",
                    "direct",
                    "selected_extras",
                    "installation_kind",
                    "build_required",
                    "wheels",
                    "package_body_sha256",
                }
            ),
            f"package[{index}]",
        )
        name = _name(package["name"], f"package[{index}].name")
        version = _version(package["version"], f"{name}.version")
        _reject_forbidden_selected_token(name, "selected distribution name")
        if name == _ROOT_PROJECT_DISTRIBUTION or name in names:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "root project or duplicate distribution appears in selected closure"
            )
        names.add(name)
        versions[name] = version
        direct = _boolean(package["direct"], f"{name}.direct")
        if direct is not (name in direct_requirements) or (
            direct and direct_requirements[name] != version
        ):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "package direct flag/version disagrees with direct requirements"
            )
        extras = _array(package["selected_extras"], f"{name}.selected_extras")
        if any(type(extra) is not str for extra in extras):
            raise ForagerMatchedV3CpuRuntimeLockError("package extras must be exact strings")
        exact_extras = cast(list[str], extras)
        if exact_extras != sorted(set(exact_extras)):
            raise ForagerMatchedV3CpuRuntimeLockError("package extras must be sorted and unique")
        for extra in exact_extras:
            if _EXTRA_RE.fullmatch(extra) is None:
                raise ForagerMatchedV3CpuRuntimeLockError("package extra is noncanonical")
            _reject_forbidden_selected_token(extra, "package selected extra")
        selected_extras_union.update(exact_extras)
        selected_extras_by_name[name] = frozenset(exact_extras)
        if (
            package["installation_kind"] != "wheel"
            or _boolean(package["build_required"], f"{name}.build_required") is not False
        ):
            raise ForagerMatchedV3CpuRuntimeLockError(
                "only selected registry wheels without builds are allowed"
            )
        selected_wheels = _array(package["wheels"], f"{name}.wheels")
        if len(selected_wheels) != 1:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "each distribution must select exactly one wheel"
            )
        wheel, requires_dist, requested_extras_by_requirement = _validate_wheel(
            selected_wheels[0],
            package_name=name,
            package_version=version,
            selected_extras=frozenset(exact_extras),
            python_version=python_version,
            glibc_version=glibc_version,
        )
        filename = cast(str, wheel["filename"])
        wheel_sha256 = cast(str, wheel["sha256"])
        if filename in filenames or wheel_sha256 in hashes:
            raise ForagerMatchedV3CpuRuntimeLockError(
                "wheel filename or content hash is duplicated"
            )
        filenames.add(filename)
        hashes.add(wheel_sha256)
        for dependency, edge_extras in zip(
            requires_dist, requested_extras_by_requirement, strict=True
        ):
            edge = {"from": name, **dependency}
            dependencies.append(edge)
            dependency_requested_extras.append((edge, edge_extras))
        _body_sha256(package, "package_body_sha256")
        packages.append(package)
        wheels.append(wheel)
    if [package["name"] for package in packages] != sorted(names):
        raise ForagerMatchedV3CpuRuntimeLockError("packages must be sorted by canonical name")
    for mandatory_name, mandatory_version in _MANDATORY_VERSIONS.items():
        if versions.get(mandatory_name) != mandatory_version:
            raise ForagerMatchedV3CpuRuntimeLockError(
                f"selected {mandatory_name} must be exactly {mandatory_version}"
            )
    if selected_extras_union != resolution_selected_extras:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "resolution selected extras disagree with package selected extras"
        )
    for dependency, edge_extras in dependency_requested_extras:
        if dependency["active"]:
            selected_name = cast(str, dependency["name"])
            if versions.get(selected_name) != dependency["selected_version"]:
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "active Requires-Dist edge is absent or version-inconsistent"
                )
            if not frozenset(edge_extras).issubset(selected_extras_by_name[selected_name]):
                raise ForagerMatchedV3CpuRuntimeLockError(
                    "active Requires-Dist requested extras are not selected on the target"
                )
    missing_direct = set(direct_requirements) - names
    if missing_direct:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "direct requirements are absent from the selected package closure: "
            + ", ".join(sorted(missing_direct))
        )
    active_edges: dict[str, set[str]] = {name: set() for name in names}
    for dependency in dependencies:
        if dependency["active"] is True:
            active_edges[cast(str, dependency["from"])].add(cast(str, dependency["name"]))
    reachable = set(direct_requirements)
    pending = list(reachable)
    while pending:
        source = pending.pop()
        for dependency_name in active_edges[source]:
            if dependency_name not in reachable:
                reachable.add(dependency_name)
                pending.append(dependency_name)
    unreachable = names - reachable
    if unreachable:
        raise ForagerMatchedV3CpuRuntimeLockError(
            "selected package closure contains unreachable distributions: "
            + ", ".join(sorted(unreachable))
        )
    return packages, wheels, dependencies


def _inventory_sha256(value: Any, label: str) -> str:
    return hashlib.sha256(_canonical_json({label: value})).hexdigest()


def _validate_closure(
    value: Any,
    packages: Sequence[dict[str, Any]],
    wheels: Sequence[dict[str, Any]],
    dependencies: Sequence[dict[str, Any]],
) -> None:
    closure = _exact_keys(
        value,
        frozenset(
            {
                "distribution_count",
                "wheel_count",
                "total_wheel_bytes",
                "total_metadata_bytes",
                "total_wheel_file_bytes",
                "total_record_bytes",
                "requires_dist_count",
                "active_dependency_count",
                "distribution_inventory_sha256",
                "wheel_inventory_sha256",
                "dependency_inventory_sha256",
                "packages_body_sha256",
                "closure_body_sha256",
            }
        ),
        "closure",
    )
    distribution_inventory = [
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
    expected = {
        "distribution_count": len(packages),
        "wheel_count": len(wheels),
        "total_wheel_bytes": sum(cast(int, wheel["size_bytes"]) for wheel in wheels),
        "total_metadata_bytes": sum(
            cast(int, cast(dict[str, Any], wheel["metadata"])["size_bytes"]) for wheel in wheels
        ),
        "total_wheel_file_bytes": sum(
            cast(int, cast(dict[str, Any], wheel["wheel"])["size_bytes"]) for wheel in wheels
        ),
        "total_record_bytes": sum(
            cast(int, cast(dict[str, Any], wheel["record"])["size_bytes"]) for wheel in wheels
        ),
        "requires_dist_count": len(dependencies),
        "active_dependency_count": sum(
            1 for dependency in dependencies if dependency["active"] is True
        ),
        "distribution_inventory_sha256": _inventory_sha256(distribution_inventory, "distributions"),
        "wheel_inventory_sha256": _inventory_sha256(wheel_inventory, "wheels"),
        "dependency_inventory_sha256": _inventory_sha256(dependencies, "dependencies"),
        "packages_body_sha256": _inventory_sha256(packages, "packages"),
    }
    for field, expected_value in expected.items():
        if closure[field] != expected_value:
            raise ForagerMatchedV3CpuRuntimeLockError(f"closure {field} disagrees")
    for field in (
        "distribution_count",
        "wheel_count",
        "total_wheel_bytes",
        "total_metadata_bytes",
        "total_wheel_file_bytes",
        "total_record_bytes",
        "requires_dist_count",
        "active_dependency_count",
    ):
        _bounded_int(closure[field], f"closure {field}")
    for field in (
        "distribution_inventory_sha256",
        "wheel_inventory_sha256",
        "dependency_inventory_sha256",
        "packages_body_sha256",
    ):
        _sha256(closure[field], f"closure {field}")
    _body_sha256(closure, "closure_body_sha256")


def _validate_wheelhouse(
    value: Any,
    *,
    closure: Mapping[str, Any],
) -> None:
    wheelhouse = _exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "cas_layout",
                "manifest",
                "archive",
                "networkless_install_required",
            }
        ),
        "wheelhouse",
    )
    if (
        wheelhouse["schema_version"] != CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION
        or wheelhouse["cas_layout"] != "sha256/first-two/full-digest/wheel-filename"
        or _boolean(wheelhouse["networkless_install_required"], "networkless_install_required")
        is not True
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("wheelhouse CAS policy drifted")
    manifest = _exact_keys(
        wheelhouse["manifest"],
        frozenset(
            {
                "filename",
                "size_bytes",
                "sha256",
                "body_sha256",
                "entry_count",
                "total_bytes",
                "inventory_sha256",
            }
        ),
        "wheelhouse manifest",
    )
    if _relative_path(manifest["filename"], "wheelhouse manifest filename") != (
        "wheelhouse.cas-manifest.v1.json"
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("wheelhouse manifest filename drifted")
    _bounded_int(manifest["size_bytes"], "wheelhouse manifest size", minimum=1)
    _sha256(manifest["sha256"], "wheelhouse manifest sha256")
    _sha256(manifest["body_sha256"], "wheelhouse manifest body sha256")
    if (
        manifest["entry_count"] != closure["wheel_count"]
        or manifest["total_bytes"] != closure["total_wheel_bytes"]
        or manifest["inventory_sha256"] != closure["wheel_inventory_sha256"]
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "wheelhouse manifest does not bind the exact wheel closure"
        )
    archive = _exact_keys(
        wheelhouse["archive"],
        frozenset(
            {
                "filename",
                "format",
                "size_bytes",
                "sha256",
                "manifest_sha256",
                "manifest_body_sha256",
            }
        ),
        "wheelhouse archive",
    )
    if (
        _relative_path(archive["filename"], "wheelhouse archive filename") != "wheelhouse.v1.tar"
        or archive["format"] != "ustar"
        or archive["manifest_sha256"] != manifest["sha256"]
        or archive["manifest_body_sha256"] != manifest["body_sha256"]
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("wheelhouse archive identity drifted")
    _bounded_int(archive["size_bytes"], "wheelhouse archive size", minimum=1)
    _sha256(archive["sha256"], "wheelhouse archive sha256")


def _claims() -> dict[str, bool]:
    return {
        "production_lock_exists": False,
        "wheel_bytes_verified": False,
        "wheelhouse_exists": False,
        "runtime_built": False,
        "runtime_qualified": False,
        "imports_qualified": False,
        "candidate_qualified": False,
        "benchmark_executed": False,
        "result_observed": False,
        "execution_ready": False,
        "execution_authorized": False,
        "ingestion_authorized": False,
        "scientific_promotion_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "The module validates declared content only and performs no filesystem or network I/O.",
        "No embedded value is a production lock, wheelhouse, runtime, or qualification receipt.",
        "Wheel, METADATA, WHEEL, and RECORD digests are bound but their bytes are not opened here.",
        (
            "Non-accelerator marker truth is recorded by the future lock and is not a full "
            "PEP 508 evaluator."
        ),
        (
            "Inactive CUDA or NVIDIA Requires-Dist metadata is allowed only for an "
            "unselected exact extra."
        ),
        "Informational solver provenance is neither trusted nor an acceptance authority.",
        (
            "Canonical validation grants no installation, execution, ingestion, or "
            "scientific authority."
        ),
    ]


def _descriptor() -> dict[str, Any]:
    return {
        "schema_version": CPU_RUNTIME_LOCK_DESCRIPTOR_SCHEMA_VERSION,
        "lock_schema_version": CPU_RUNTIME_LOCK_SCHEMA_VERSION,
        "status": CPU_RUNTIME_LOCK_STATUS,
        "classification": CPU_RUNTIME_LOCK_CLASSIFICATION,
        "production_contract": {
            "distribution_count": PRODUCTION_DISTRIBUTION_COUNT,
            "python_implementation": "CPython",
            "python_version": PRODUCTION_PYTHON_VERSION,
            "python_tag": "cp312",
            "abi_tag": "cp312",
            "os": "linux",
            "architecture": "x86_64",
            "platform": "linux-amd64",
            "libc_family": "glibc",
            "minimum_glibc": PRODUCTION_MINIMUM_GLIBC,
            "cpu_only": True,
            "one_selected_wheel_per_distribution": True,
            "generic_validator_allows_smaller_synthetic_closures": True,
            "separate_production_validator_required": True,
        },
        "wheel_policy": {
            "registry_origin": "https://files.pythonhosted.org",
            "artifact_kind": "wheel_only",
            "sdists_allowed": False,
            "vcs_allowed": False,
            "editable_allowed": False,
            "path_dependencies_allowed": False,
            "source_builds_allowed": False,
            "musllinux_allowed": False,
            "non_x86_64_allowed": False,
            "legacy_manylinux_tags": sorted(_LEGACY_MANYLINUX_GLIBC),
            "stable_abi_tags_before_cp312_allowed": True,
            "universal_py2_py3_wheels_allowed_when_py3_tag_present": True,
            "inactive_optional_accelerator_metadata_allowed": True,
            "active_accelerator_dependencies_allowed": False,
        },
        "lock_sections": [
            "schema_version",
            "status",
            "classification",
            "target",
            "upstream",
            "overlay_delta",
            "solver_provenance",
            "resolution",
            "packages",
            "closure",
            "wheelhouse",
            "claims",
            "limitations",
            "lock_body_sha256",
        ],
        "per_wheel_content_bindings": [
            "source_url",
            "cas_key",
            "size_bytes",
            "sha256",
            "tags",
            "METADATA_path_size_sha256",
            "Metadata-Version",
            "Name",
            "Version",
            "Requires-Python",
            "Provides-Extra",
            "Requires-Dist_raw_marker_activity",
            "WHEEL_path_size_sha256_fields_and_tags",
            "RECORD_path_size_sha256_entry_count_and_inventory",
            "wheel_body_sha256",
        ],
        "closure_policy": {
            "active_edges_must_resolve_to_selected_versions": True,
            "active_requested_extras_must_be_selected_on_target_distribution": True,
            "all_selected_distributions_reachable_from_direct_roots": True,
            "selected_extras_must_be_declared_by_provides_extra": True,
            "requires_dist_pep440_specifier_syntax_validated": True,
            "inactive_accelerator_extra_must_be_declared_and_unselected": True,
            "arbitrary_nonaccelerator_marker_truth_recomputed_here": False,
            "later_byte_level_marker_validation_required": True,
            "root_is_purelib_exact_install_scheme_declaration_bound": True,
            "wheel_tag_compatibility_validated_independently": True,
        },
        "overlay_policy": {
            "delta_format": "canonical_json_operations_v1",
            "operation_kinds": ["add", "remove", "replace"],
            "exact_sorted_operations_embedded": True,
            "per_operation_body_hash_recomputed": True,
            "operations_hash_recomputed": True,
            "overlay_body_hash_recomputed": True,
            "direct_requirements_hash_recomputed": True,
        },
        "solver_provenance_policy": {
            "informational_only": True,
            "argv_and_environment_embedded_and_hashed": True,
            "interpreter_and_solver_binaries_hashed": True,
            "index_and_capture_timestamp_bound": True,
            "resolution_input_and_report_hashed": True,
            "trusted_for_acceptance": False,
        },
        "mandatory_distributions": dict(_MANDATORY_VERSIONS),
        "upstream": {
            "repository_id": _UPSTREAM_REPOSITORY_ID,
            "repository_url": _UPSTREAM_REPOSITORY_URL,
            "commit_git_sha1": _UPSTREAM_COMMIT,
            "tree_git_sha1": _UPSTREAM_TREE,
            "archive_sha256": _UPSTREAM_ARCHIVE_SHA256,
            "archive_size_bytes": _UPSTREAM_ARCHIVE_SIZE,
            "pyproject_sha256": _UPSTREAM_PYPROJECT_SHA256,
            "pyproject_size_bytes": _UPSTREAM_PYPROJECT_SIZE,
            "lock_sha256": _UPSTREAM_LOCK_SHA256,
            "pyproject_path": "pyproject.toml",
            "lock_path": "uv.lock",
            "root_project_distribution": _ROOT_PROJECT_DISTRIBUTION,
            "root_project_installation_allowed": False,
        },
        "canonicalization": {
            "format": "json",
            "encoding": "ascii",
            "sort_keys": True,
            "ensure_ascii": True,
            "allow_nan": False,
            "allow_floats": False,
            "separators": [",", ":"],
            "trailing_newline": True,
            "duplicate_keys_rejected": True,
            "container_aliases_rejected": True,
            "unknown_fields_rejected": True,
            "maximum_bytes": _MAX_ARTIFACT_BYTES,
            "maximum_depth": _MAX_JSON_DEPTH,
            "maximum_nodes": _MAX_JSON_NODES,
        },
        "claims": _claims(),
        "limitations": _limitations(),
    }


_DESCRIPTOR_BYTES: Final = _canonical_json(_descriptor())
CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256: Final = (
    "31d4c5a101f441bc082bdaf9250050f7950440271e6360854d5faa9fcd7ff34a"
)
if not hmac.compare_digest(
    hashlib.sha256(_DESCRIPTOR_BYTES).hexdigest(), CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256
):
    raise AssertionError("matched-v3 CPU runtime lock descriptor identity drifted")


def _validate_lock(value: Mapping[str, Any]) -> None:
    lock = _exact_keys(
        value,
        frozenset(
            {
                "schema_version",
                "status",
                "classification",
                "target",
                "upstream",
                "overlay_delta",
                "solver_provenance",
                "resolution",
                "packages",
                "closure",
                "wheelhouse",
                "claims",
                "limitations",
                "lock_body_sha256",
            }
        ),
        "CPU runtime lock",
    )
    if (
        lock["schema_version"] != CPU_RUNTIME_LOCK_SCHEMA_VERSION
        or lock["status"] != "future_content_lock_unexecuted_non_authorizing"
        or lock["classification"] != CPU_RUNTIME_LOCK_CLASSIFICATION
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("CPU runtime lock identity drifted")
    target = _validate_target(lock["target"])
    upstream = _validate_upstream(lock["upstream"])
    direct, marker_environment_sha256, selected_extras = _validate_resolution(
        lock["resolution"], target
    )
    resolution = cast(dict[str, Any], lock["resolution"])
    pyproject = cast(dict[str, Any], upstream["pyproject"])
    _validate_overlay(
        lock["overlay_delta"],
        pyproject_sha256=cast(str, pyproject["sha256"]),
        direct_requirements=cast(list[Any], resolution["direct_requirements"]),
    )
    _validate_solver(lock["solver_provenance"], marker_environment_sha256, target)
    packages, wheels, dependencies = _validate_packages(
        lock["packages"],
        direct_requirements=direct,
        resolution_selected_extras=selected_extras,
        python_version=cast(str, target["python_version"]),
        glibc_version=cast(str, target["libc_version"]),
    )
    closure = cast(dict[str, Any], lock["closure"])
    _validate_closure(closure, packages, wheels, dependencies)
    _validate_wheelhouse(lock["wheelhouse"], closure=closure)
    if lock["claims"] != _claims() or any(item is not False for item in lock["claims"].values()):
        raise ForagerMatchedV3CpuRuntimeLockError("a CPU runtime lock claim became true")
    if lock["limitations"] != _limitations():
        raise ForagerMatchedV3CpuRuntimeLockError("CPU runtime lock limitations drifted")
    _body_sha256(lock, "lock_body_sha256")
    _assert_plain_unaliased_json(lock)
    _canonical_json(lock)


def cpu_runtime_lock_descriptor() -> dict[str, Any]:
    """Return a detached snapshot of the fixed schema descriptor."""

    return _strict_json_load(_DESCRIPTOR_BYTES)


def canonical_cpu_runtime_lock_descriptor_bytes() -> bytes:
    """Return the exact canonical descriptor bytes, including one newline."""

    return _DESCRIPTOR_BYTES


def cpu_runtime_lock_descriptor_sha256() -> str:
    """Return the descriptor's frozen content identity."""

    return CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256


def parse_cpu_runtime_lock_descriptor(raw: bytes) -> dict[str, Any]:
    """Parse only the exact fixed descriptor and return detached JSON."""

    value = _strict_json_load(raw)
    if value != _descriptor() or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256
    ):
        raise ForagerMatchedV3CpuRuntimeLockError("CPU runtime lock descriptor drifted")
    return value


def validate_cpu_runtime_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a generic lock, allowing small synthetic package closures."""

    if type(value) is not dict:
        raise ForagerMatchedV3CpuRuntimeLockError("CPU runtime lock must be a plain object")
    _assert_plain_unaliased_json(value)
    _validate_lock(value)
    return _strict_json_load(_canonical_json(value))


def validate_production_cpu_runtime_lock(value: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the separate 104-package CPython-3.12.3 production target gate."""

    detached = validate_cpu_runtime_lock(value)
    target = cast(dict[str, Any], detached["target"])
    packages = cast(list[Any], detached["packages"])
    if (
        len(packages) != PRODUCTION_DISTRIBUTION_COUNT
        or target["python_version"] != PRODUCTION_PYTHON_VERSION
        or target["platform"] != "linux-amd64"
        or target["libc_family"] != "glibc"
        or tuple(int(item) for item in cast(str, target["libc_version"]).split("."))
        < tuple(int(item) for item in PRODUCTION_MINIMUM_GLIBC.split("."))
    ):
        raise ForagerMatchedV3CpuRuntimeLockError(
            "production lock must contain exactly 104 distributions for the frozen "
            "CPython 3.12.3 linux-amd64 glibc>=2.28 CPU target"
        )
    return detached


def canonical_cpu_runtime_lock_bytes(value: Mapping[str, Any]) -> bytes:
    """Validate and canonically encode one generic lock."""

    validate_cpu_runtime_lock(value)
    return _canonical_json(value)


def cpu_runtime_lock_sha256(value: Mapping[str, Any]) -> str:
    """Return the full-file SHA-256 of one validated generic lock."""

    return hashlib.sha256(canonical_cpu_runtime_lock_bytes(value)).hexdigest()


def parse_cpu_runtime_lock(
    raw: bytes,
    *,
    expected_file_sha256: str,
    production: bool = False,
) -> dict[str, Any]:
    """Parse a canonical lock under an independent full-file digest pin."""

    expected = _sha256(expected_file_sha256, "expected runtime-lock file sha256")
    if type(raw) is not bytes or not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected):
        raise ForagerMatchedV3CpuRuntimeLockError("runtime-lock full-file digest disagrees")
    value = _strict_json_load(raw)
    if production:
        return validate_production_cpu_runtime_lock(value)
    return validate_cpu_runtime_lock(value)


__all__ = [
    "CPU_RUNTIME_LOCK_CLASSIFICATION",
    "CPU_RUNTIME_LOCK_DESCRIPTOR_SCHEMA_VERSION",
    "CPU_RUNTIME_LOCK_DESCRIPTOR_SHA256",
    "CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION",
    "CPU_RUNTIME_LOCK_SCHEMA_VERSION",
    "CPU_RUNTIME_LOCK_STATUS",
    "CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION",
    "ForagerMatchedV3CpuRuntimeLockError",
    "PRODUCTION_DISTRIBUTION_COUNT",
    "PRODUCTION_MINIMUM_GLIBC",
    "PRODUCTION_PYTHON_VERSION",
    "canonical_cpu_runtime_lock_bytes",
    "canonical_cpu_runtime_lock_descriptor_bytes",
    "cpu_runtime_lock_descriptor",
    "cpu_runtime_lock_descriptor_sha256",
    "cpu_runtime_lock_sha256",
    "parse_cpu_runtime_lock",
    "parse_cpu_runtime_lock_descriptor",
    "validate_cpu_runtime_lock",
    "validate_production_cpu_runtime_lock",
]
