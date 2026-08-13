"""Durable one-shot publication for the matched-v3 CPU OCI build.

This is the deliberately separate composition boundary above the pure build
plan, sealed context, and explicit executor.  It accepts a caller-pinned local
source snapshot and the one frozen production dependency/source lineage,
replays every input, creates a new-only durable intent addressed by the sealed
context-receipt SHA-256, and only then creates and consumes the executor's
single-use authorization.

An existing intent is never resumed or retried.  A successful call publishes
the execution receipt separately; a failed call publishes a bounded failure
receipt that distinguishes a known pre-start failure from uncertain image
state.  The implementation never tags, pulls, prunes, removes, or publishes an
OCI image and grants no qualification, evidence, or promotion authority.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Never, NoReturn, cast

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_context as context_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_plan as plan_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_runtime_lock_issuer as runtime_lock_issuer,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_wheelhouse as wheelhouse_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_source_publication as external_publication_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_source_bundle as local_bundle_contract,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_source_snapshot as local_snapshot_contract,
)

if TYPE_CHECKING:
    from alberta_framework.benchmarks import (
        forager_matched_v3_cpu_oci_build_executor as executor_contract,
    )

CPU_OCI_BUILD_PUBLICATION_INTENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_intent.v1"
)
CPU_OCI_BUILD_PUBLICATION_SUCCESS_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_publication.v1"
)
CPU_OCI_BUILD_PUBLICATION_FAILURE_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_failure.v1"
)
CPU_OCI_BUILD_REQUEST_MEASUREMENT_SCHEMA_VERSION: Final = (
    "alberta.forager_matched_v3.cpu_oci_build_request_measurement.v1"
)
CPU_OCI_BUILD_PUBLICATION_STATUS: Final = (
    "local_cpu_oci_build_receipt_published_unqualified_non_authorizing"
)

# Frozen r5 production inputs.  There is intentionally no "latest" discovery.
PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256: Final = (
    "f396944111366df1e243214547d17c5ed35d517f0508ecff7b4a2edec1e881a7"
)
PRODUCTION_WHEELHOUSE_ARCHIVE_SIZE_BYTES: Final = 573_061_120
PRODUCTION_WHEELHOUSE_RECEIPT_SHA256: Final = (
    "51dc757abc25f07347c0b7b1416a61e149e72707485e3a54fdfd31765a53a1c6"
)
PRODUCTION_WHEELHOUSE_RECEIPT_SIZE_BYTES: Final = 460_516
PRODUCTION_CAPTURE_MANIFEST_SHA256: Final = (
    "f4d674e88f2a29047a0296ca84432cb08d05b631f96c9e75653c31df25c7275d"
)
PRODUCTION_CAPTURE_MANIFEST_SIZE_BYTES: Final = 61_762
PRODUCTION_RUNTIME_LOCK_SHA256: Final = (
    "f4089e4631bc1a8817827a27ab58943968c634f2a3c54ea4f54385c2163a8641"
)
PRODUCTION_RUNTIME_LOCK_SIZE_BYTES: Final = 356_996
PRODUCTION_CAS_MANIFEST_SHA256: Final = (
    "e9ea3ee9faaecf09ba4367db47ab8fe7d281505b96099d3963743fbe9fc1cc46"
)
PRODUCTION_CAS_MANIFEST_SIZE_BYTES: Final = 72_679
PRODUCTION_ISSUANCE_ENVELOPE_SHA256: Final = (
    "30ee57e9df1e1805d7a338d250daf99849170a765fe66613466664af38421eae"
)
PRODUCTION_ISSUANCE_ENVELOPE_SIZE_BYTES: Final = 49_145
PRODUCTION_ROOT_PIN_INVENTORY_SHA256: Final = (
    "2f175de86b18b7d72772dd093902f801f423bd37393fde9133377528e4a12d47"
)
PRODUCTION_SELECTED_WHEEL_INVENTORY_SHA256: Final = (
    "8cbe5daa6a66e87672fce419cf40f2b6769fbceea8eca3ded7e33401b3a618e6"
)
PRODUCTION_RESOLUTION_LOCK_SHA256: Final = (
    "6f6127c1b4d970c432bf29f6c7e8e65230b966cbf6197cf4e462822e84ef725d"
)
PRODUCTION_RESOLUTION_LOCK_SIZE_BYTES: Final = 106_980
PRODUCTION_EXTERNAL_ARCHIVE_SHA256: Final = (
    "83a2a026bd053e6f75a6308b3e4e74e9051e96d7ba0d82e3da1b62c49b914a1f"
)
PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES: Final = 321_802_240
PRODUCTION_EXTERNAL_RECEIPT_SHA256: Final = (
    "8e77524275d952996888fccab13cc2abca210c431c8febc368328a815ac2c646"
)
PRODUCTION_EXTERNAL_RECEIPT_SIZE_BYTES: Final = 2_788_296
PRODUCTION_EXTERNAL_MEMBER_COUNT: Final = 10_946
PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256: Final = (
    "82338719a8df20f5dfe809b45e27cdceb039ea9064dd00d96df8d084376f889b"
)
PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256: Final = (
    "55ee1723f821da4cc21d92523378f014d0483c05b8d8f36bee4ac6cd1cdf7aba"
)
PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256: Final = (
    "76f5758e97dc8f5410047004ff589dae45bc0d24e44ea157239a40e1dd27323f"
)

_INTENT_STATUS: Final = "durable_intent_committed_before_executor_authorization"
_INTENT_CLASSIFICATION: Final = "one_shot_local_build_intent_non_authorizing"
_SUCCESS_CLASSIFICATION: Final = "durable_local_build_observation_non_authorizing"
_FAILURE_STATUS: Final = "local_cpu_oci_build_attempt_failed_non_authorizing"
_FAILURE_CLASSIFICATION: Final = "durable_local_build_failure_observation_non_authorizing"

_INTENT_FILENAME: Final = "intent.v1.json"
_SNAPSHOT_FILENAME: Final = "local-source-snapshot.v1.json"
_LOCAL_ARCHIVE_FILENAME: Final = "local-source.v1.tar"
_LOCAL_RECEIPT_FILENAME: Final = "local-source-receipt.v1.json"
_PLAN_FILENAME: Final = "oci-build-plan.v1.json"
_CONTEXT_RECEIPT_FILENAME: Final = "oci-build-context-receipt.v1.json"
_EXECUTION_RECEIPT_FILENAME: Final = "oci-build-execution-receipt.v1.json"
_PUBLICATION_RECEIPT_FILENAME: Final = "publication-receipt.v1.json"
_FAILURE_FILENAME: Final = "failure.v1.json"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_TYPE_RE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,255}\Z")
_SAFE_FILE_RE: Final = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}\Z")
_DURABLE_FAILURE_NOTE_RE: Final = re.compile(
    r"\Adurable matched-v3 OCI failure receipt: sha256:([0-9a-f]{64})\Z"
)
_FAILURE_PHASE_NOTE_RE: Final = re.compile(
    r"\Amatched-v3 OCI failure phase: "
    r"(pre_intent|intent_publication_uncertain_pre_start|authorization_failed_pre_start|"
    r"executor_failed_pre_start|executor_failed_uncertain|"
    r"success_publication_failed_after_build)\Z"
)
_IMAGE_UNCERTAINTY_NOTE_RE: Final = re.compile(
    r"\Amatched-v3 OCI image state uncertain: (true|false)\Z"
)
_INDETERMINATE_INTENT_DEFERRED_NOTE: Final = (
    "durable matched-v3 OCI failure receipt deferred: canonical intent link is unavailable "
    "or intent commit state is indeterminate"
)
_MAX_JSON_BYTES: Final = 32 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 64
_MAX_JSON_NODES: Final = 500_000
_MAX_JSON_TEXT_BYTES: Final = 2 * 1024 * 1024
_MAX_INTEGER: Final = 2**63 - 1
_READ_CHUNK_BYTES: Final = 1024 * 1024
_MIN_TIMEOUT_SECONDS: Final = 60
_MAX_TIMEOUT_SECONDS: Final = 21_600
_RENAME_NOREPLACE: Final = 1
_EXECUTION_ACKNOWLEDGEMENT: Final = (
    "AUTHORIZE ONE LOCAL MATCHED-V3 CPU OCI BUILD FROM THIS SEALED CONTEXT"
)


def _load_executor_contract() -> Any:
    """Import the execution authority only from an execution/replay path."""

    from alberta_framework.benchmarks import (
        forager_matched_v3_cpu_oci_build_executor as loaded_executor_contract,
    )

    return loaded_executor_contract


class ForagerMatchedV3CpuOciBuildPublicationError(RuntimeError):
    """The composition, durable intent, or receipt publication failed closed."""

    def __init__(self, message: str, *, image_state_uncertain: bool = False) -> None:
        super().__init__(message)
        self.image_state_uncertain = image_state_uncertain


class MatchedV3CpuOciBuildIntentExistsError(ForagerMatchedV3CpuOciBuildPublicationError):
    """The same sealed context already has a durable one-shot intent."""

    def __init__(self, message: str, *, context_receipt_sha256: str) -> None:
        self.context_receipt_sha256 = _require_sha256(
            context_receipt_sha256,
            label="existing intent context receipt",
        )
        super().__init__(message, image_state_uncertain=True)


class MatchedV3CpuOciBuildPublicationStateUncertainError(
    ForagerMatchedV3CpuOciBuildPublicationError
):
    """A publication syscall or postcommit replay escaped exact classification."""


class MatchedV3CpuOciBuildSuccessPublicationUncertainError(
    MatchedV3CpuOciBuildPublicationStateUncertainError
):
    """The build succeeded but durable success publication did not complete cleanly."""

    def __init__(
        self,
        message: str,
        *,
        context_receipt_sha256: str,
        execution_receipt_sha256: str,
        image_id: str,
    ) -> None:
        self.context_receipt_sha256 = _require_sha256(
            context_receipt_sha256,
            label="uncertain success context receipt",
        )
        self.execution_receipt_sha256 = _require_sha256(
            execution_receipt_sha256,
            label="uncertain success execution receipt",
        )
        self.image_id = _require_image_id(image_id, label="uncertain success image ID")
        super().__init__(message, image_state_uncertain=True)


def _fail(message: str, *, image_state_uncertain: bool = False) -> NoReturn:
    raise ForagerMatchedV3CpuOciBuildPublicationError(
        message,
        image_state_uncertain=image_state_uncertain,
    )


def _add_note_once(error: BaseException, note: str) -> None:
    if note not in getattr(error, "__notes__", ()):
        error.add_note(note)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be one nonzero lowercase SHA-256")
    return value


def _require_image_id(value: Any, *, label: str) -> str:
    if (
        type(value) is not str
        or _IMAGE_ID_RE.fullmatch(value) is None
        or value == "sha256:" + "0" * 64
    ):
        _fail(f"{label} must be one exact sha256 image ID")
    return value


def _require_integer(
    value: Any,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = _MAX_INTEGER,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        _fail(f"{label} must be one bounded exact integer")
    return value


def _claims() -> dict[str, bool]:
    return {
        "acceptance_authority_granted": False,
        "artifact_accepted": False,
        "candidate_qualified": False,
        "evidence_authority_granted": False,
        "execution_authority_granted": False,
        "image_published": False,
        "performance_claim_allowed": False,
        "qualification_authority_granted": False,
        "runtime_qualified": False,
        "scientific_evidence_created": False,
        "scientific_promotion_allowed": False,
        "universal_sota_claim_allowed": False,
    }


def _limitations() -> list[str]:
    return [
        "This publication records one local nonauthorizing OCI build observation only.",
        "The OCI image remains addressed only by its local daemon image ID.",
        (
            "The caller-owned publication root is a trusted same-UID retention boundary; "
            "descriptor checks detect observed locator swaps but cannot prevent deletion."
        ),
        "Every non-sticky group- or world-writable path ancestor is rejected.",
        "No tag, pull, prune, remove, push, registry, or image-publication operation is issued.",
        "A durable intent is one-shot and is never automatically resumed or retried.",
        "Daemon and BuildKit egress remain unobserved and unattested.",
        "No receipt grants qualification, evidence, promotion, performance, or SOTA authority.",
    ]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    raw = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    if len(raw) > _MAX_JSON_BYTES:
        _fail("canonical publication JSON exceeds its byte bound")
    return raw


def _raise_float(value: str) -> Never:
    _fail(f"publication JSON contains float {value!r}")


def _raise_constant(value: str) -> Never:
    _fail(f"publication JSON contains non-finite constant {value!r}")


def _parse_integer(value: str) -> int:
    if len(value.lstrip("-")) > 19:
        _fail("publication JSON integer exceeds its lexical bound")
    return int(value)


def _without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"publication JSON repeats object key {key!r}")
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
            _fail("publication JSON exceeds its structure bound")
        if type(item) is str:
            if len(item.encode("utf-8")) > _MAX_JSON_TEXT_BYTES or any(
                ord(character) < 0x20 and character not in "\n\r\t" for character in item
            ):
                _fail("publication JSON contains an invalid or oversized string")
            continue
        if item is None or type(item) in {bool, int}:
            if type(item) is int and not -_MAX_INTEGER <= item <= _MAX_INTEGER:
                _fail("publication JSON integer exceeds its value bound")
            continue
        if type(item) not in {dict, list}:
            _fail("publication JSON contains a non-JSON value")
        identity = id(item)
        if identity in seen:
            _fail("publication JSON contains a container alias")
        seen.add(identity)
        if type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        else:
            for key, child in cast(dict[Any, Any], item).items():
                if type(key) is not str:
                    _fail("publication JSON object key is not an exact string")
                pending.append((key, depth + 1))
                pending.append((child, depth + 1))


def _parse_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > _MAX_JSON_BYTES or not raw.endswith(b"\n"):
        _fail(f"{label} bytes are absent, oversized, or not newline terminated")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_without_duplicates,
            parse_float=_raise_float,
            parse_int=_parse_integer,
            parse_constant=_raise_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ForagerMatchedV3CpuOciBuildPublicationError(
            f"{label} is not strict canonical JSON"
        ) from exc
    if type(value) is not dict:
        _fail(f"{label} root is not one object")
    exact = cast(dict[str, Any], value)
    _assert_plain_json(exact)
    if _canonical_json(exact) != raw:
        _fail(f"{label} bytes are not canonical")
    return exact


def _exact(value: Any, fields: frozenset[str], *, label: str) -> dict[str, Any]:
    if type(value) is not dict or frozenset(value) != fields:
        _fail(f"{label} fields are not exact")
    return cast(dict[str, Any], value)


def _file_record(name: str, raw: bytes) -> dict[str, Any]:
    if _SAFE_FILE_RE.fullmatch(name) is None:
        _fail("publication file name is unsafe")
    return {"name": name, "sha256": _sha256(raw), "size_bytes": len(raw)}


def _validate_file_records(value: Any, *, expected_names: frozenset[str]) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != len(expected_names):
        _fail("publication closure file records are incomplete")
    result: list[dict[str, Any]] = []
    previous = b""
    for index, item in enumerate(value):
        record = _exact(
            item,
            frozenset({"name", "sha256", "size_bytes"}),
            label=f"publication closure file {index}",
        )
        name = record["name"]
        if type(name) is not str or name not in expected_names:
            _fail("publication closure file name differs")
        encoded = name.encode("ascii")
        if encoded <= previous:
            _fail("publication closure file records are not uniquely sorted")
        previous = encoded
        _require_sha256(record["sha256"], label=f"publication file {name}")
        _require_integer(record["size_bytes"], label=f"publication file {name} size", minimum=1)
        result.append(copy.deepcopy(record))
    return result


def _production_bindings() -> dict[str, Any]:
    return {
        "capture_manifest": {
            "sha256": PRODUCTION_CAPTURE_MANIFEST_SHA256,
            "size_bytes": PRODUCTION_CAPTURE_MANIFEST_SIZE_BYTES,
        },
        "external_source": {
            "archive_sha256": PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
            "archive_size_bytes": PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES,
            "receipt_sha256": PRODUCTION_EXTERNAL_RECEIPT_SHA256,
            "receipt_size_bytes": PRODUCTION_EXTERNAL_RECEIPT_SIZE_BYTES,
            "source_manifest_sha256": PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256,
            "source_tree_sha256": PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256,
            "staging_manifest_sha256": PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256,
        },
        "resolution": {
            "lock_sha256": PRODUCTION_RESOLUTION_LOCK_SHA256,
            "lock_size_bytes": PRODUCTION_RESOLUTION_LOCK_SIZE_BYTES,
            "root_pin_inventory_sha256": PRODUCTION_ROOT_PIN_INVENTORY_SHA256,
            "selected_wheel_inventory_sha256": (PRODUCTION_SELECTED_WHEEL_INVENTORY_SHA256),
        },
        "runtime_lock": {
            "cas_manifest_sha256": PRODUCTION_CAS_MANIFEST_SHA256,
            "cas_manifest_size_bytes": PRODUCTION_CAS_MANIFEST_SIZE_BYTES,
            "issuance_envelope_sha256": PRODUCTION_ISSUANCE_ENVELOPE_SHA256,
            "issuance_envelope_size_bytes": PRODUCTION_ISSUANCE_ENVELOPE_SIZE_BYTES,
            "runtime_lock_sha256": PRODUCTION_RUNTIME_LOCK_SHA256,
            "runtime_lock_size_bytes": PRODUCTION_RUNTIME_LOCK_SIZE_BYTES,
        },
        "wheelhouse": {
            "archive_sha256": PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256,
            "archive_size_bytes": PRODUCTION_WHEELHOUSE_ARCHIVE_SIZE_BYTES,
            "receipt_sha256": PRODUCTION_WHEELHOUSE_RECEIPT_SHA256,
            "receipt_size_bytes": PRODUCTION_WHEELHOUSE_RECEIPT_SIZE_BYTES,
        },
    }


@dataclass(frozen=True, slots=True)
class MatchedV3CpuOciBuildPublicationRequest:
    """Explicit paths plus the caller-pinned local source snapshot."""

    repository_root: Path
    artifact_root: Path
    publication_root: Path
    expected_snapshot_manifest_bytes: bytes
    expected_snapshot_manifest_sha256: str
    expected_snapshot_tree_sha256: str
    exact_acknowledgement: str
    timeout_seconds: int = 7200

    def __post_init__(self) -> None:
        concrete_path_type = type(Path())
        for label, value in (
            ("repository root", self.repository_root),
            ("artifact root", self.artifact_root),
            ("publication root", self.publication_root),
        ):
            if (
                type(value) is not concrete_path_type
                or not value.is_absolute()
                or value == Path("/")
            ):
                _fail(f"{label} must be one exact non-root absolute pathlib.Path")
            if any(part in {".", ".."} for part in value.parts):
                _fail(f"{label} contains a dot segment")
        manifest_sha = _require_sha256(
            self.expected_snapshot_manifest_sha256,
            label="expected local snapshot manifest",
        )
        tree_sha = _require_sha256(
            self.expected_snapshot_tree_sha256,
            label="expected local snapshot tree",
        )
        if (
            type(self.expected_snapshot_manifest_bytes) is not bytes
            or not self.expected_snapshot_manifest_bytes
            or _sha256(self.expected_snapshot_manifest_bytes) != manifest_sha
        ):
            _fail("expected local snapshot manifest bytes differ from their caller pin")
        parsed_manifest = local_snapshot_contract.parse_matched_v3_local_source_snapshot_manifest(
            self.expected_snapshot_manifest_bytes,
            expected_full_sha256=manifest_sha,
        )
        parsed_tree = cast(Mapping[str, Any], parsed_manifest["tree"])
        if not hmac.compare_digest(cast(str, parsed_tree["sha256"]), tree_sha):
            _fail("expected local snapshot tree pin differs from the manifest")
        if type(self.exact_acknowledgement) is not str or not hmac.compare_digest(
            self.exact_acknowledgement,
            _EXECUTION_ACKNOWLEDGEMENT,
        ):
            _fail("exact OCI build acknowledgement differs")
        _require_integer(
            self.timeout_seconds,
            label="OCI build timeout",
            minimum=_MIN_TIMEOUT_SECONDS,
            maximum=_MAX_TIMEOUT_SECONDS,
        )


@dataclass(frozen=True, slots=True)
class MeasuredMatchedV3CpuOciBuildRequest:
    """One nonauthorizing request-preparation snapshot written new-only."""

    manifest_path: Path
    manifest_sha256: str
    tree_sha256: str
    directory_count: int
    file_count: int
    total_size_bytes: int

    def __post_init__(self) -> None:
        concrete_path_type = type(Path())
        if (
            type(self.manifest_path) is not concrete_path_type
            or not self.manifest_path.is_absolute()
            or self.manifest_path == Path("/")
        ):
            _fail("measured snapshot manifest path is not one exact absolute path")
        _require_sha256(self.manifest_sha256, label="measured snapshot manifest")
        _require_sha256(self.tree_sha256, label="measured snapshot tree")
        _require_integer(
            self.directory_count,
            label="measured snapshot directory count",
            minimum=1,
            maximum=10_000,
        )
        _require_integer(
            self.file_count,
            label="measured snapshot file count",
            minimum=2,
            maximum=20_000,
        )
        _require_integer(
            self.total_size_bytes,
            label="measured snapshot total bytes",
            maximum=2 * 1024 * 1024 * 1024,
        )


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuOciBuild:
    """Durable intent and successful local-build receipt identities."""

    intent_directory: Path
    success_directory: Path
    context_receipt_sha256: str
    execution_receipt_sha256: str
    publication_receipt_sha256: str
    image_id: str

    def __post_init__(self) -> None:
        _require_sha256(self.context_receipt_sha256, label="published context receipt")
        _require_sha256(self.execution_receipt_sha256, label="published execution receipt")
        _require_sha256(self.publication_receipt_sha256, label="publication receipt")
        _require_image_id(self.image_id, label="published image ID")


@dataclass(frozen=True, slots=True)
class PublishedMatchedV3CpuOciBuildFailure:
    """One durable bounded failure receipt."""

    directory: Path
    receipt_sha256: str
    phase: str
    image_state_uncertain: bool


@dataclass(frozen=True, slots=True)
class _LoadedProductionInputs:
    issuance_artifacts: runtime_lock_issuer.CpuRuntimeLockIssuanceArtifacts
    wheelhouse_archive_bytes: bytes
    external_source_archive_bytes: bytes
    external_source_receipt_bytes: bytes


@dataclass(slots=True)
class _BuildAttemptState:
    context_receipt_sha256: str | None = None
    plan_sha256: str | None = None
    intent_sha256: str | None = None
    intent_bytes: bytes | None = None
    executor_module: Any | None = None
    execution: executor_contract.CpuOciBuildExecutionArtifacts | None = None
    failure_publication_attempted: bool = False


@dataclass(slots=True)
class _IntentCommitState:
    committed: bool = False


class _CanonicalIntentUnavailableError(ForagerMatchedV3CpuOciBuildPublicationError):
    """The deterministic intent route cannot support a linked failure receipt."""


@dataclass(slots=True)
class _AnchoredDirectoryChain:
    descriptors: list[int]
    components: list[str]
    metadata: list[os.stat_result]
    label: str

    @property
    def descriptor(self) -> int:
        if not self.descriptors:
            _fail(f"{self.label} descriptor chain is closed")
        return self.descriptors[-1]

    def verify(self) -> None:
        for index in range(1, len(self.descriptors)):
            parent = self.descriptors[index - 1]
            descriptor = self.descriptors[index]
            component = self.components[index]
            expected = self.metadata[index]
            try:
                located = os.stat(component, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(descriptor)
            except OSError as exc:
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    f"{self.label} locator changed while retained"
                ) from exc
            if (
                _directory_locator_identity(located) != _directory_locator_identity(expected)
                or _directory_locator_identity(opened) != _directory_locator_identity(expected)
                or not _directory_component_metadata_is_secure(expected)
            ):
                _fail(f"{self.label} locator changed while retained")

    def close(self) -> None:
        descriptors = self.descriptors
        self.descriptors = []
        failure: BaseException | None = None
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if failure is None:
                    failure = cleanup_error
                    failure.add_note(f"while closing {self.label} descriptor chain")
                else:
                    failure.add_note(
                        f"another {self.label} descriptor close also failed: {cleanup_error!r}"
                    )
        if failure is not None:
            raise failure


@dataclass(frozen=True, slots=True)
class _OpenRoot:
    path: Path
    descriptor: int
    identity: tuple[int, ...]
    chain: _AnchoredDirectoryChain


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _directory_locator_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def _directory_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if type(nofollow) is not int or type(directory) is not int:
        _fail("publication requires O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | directory | nofollow | getattr(os, "O_CLOEXEC", 0)


def _file_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if type(nofollow) is not int:
        _fail("publication requires O_NOFOLLOW")
    return os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)


def _close_descriptors(
    descriptors: Sequence[tuple[int, str]],
    *,
    primary: BaseException | None,
) -> None:
    cleanup_failures: list[tuple[str, BaseException]] = []
    seen: set[int] = set()
    for descriptor, label in descriptors:
        if descriptor < 0 or descriptor in seen:
            continue
        seen.add(descriptor)
        try:
            os.close(descriptor)
        except BaseException as cleanup_error:
            cleanup_failures.append((label, cleanup_error))
    if not cleanup_failures:
        return
    if primary is not None:
        for failed_label, failed_cleanup in cleanup_failures:
            primary.add_note(f"{failed_label} cleanup also failed: {failed_cleanup!r}")
        return
    failed_label, failed_cleanup = cleanup_failures[0]
    failure = ForagerMatchedV3CpuOciBuildPublicationError(f"{failed_label} cleanup failed")
    for additional_label, additional_error in cleanup_failures[1:]:
        failure.add_note(f"{additional_label} cleanup also failed: {additional_error!r}")
    raise failure from failed_cleanup


def _validate_path_component(component: str, *, label: str) -> str:
    try:
        encoded = component.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPublicationError(f"{label} is not exact ASCII") from exc
    if (
        not encoded
        or len(encoded) > 255
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or any(byte < 0x21 or byte > 0x7E for byte in encoded)
    ):
        _fail(f"{label} is not one safe exact path component")
    return component


def _directory_component_metadata_is_secure(metadata: os.stat_result) -> bool:
    effective_uid = os.geteuid()
    if metadata.st_uid not in {0, effective_uid}:
        return False
    if not metadata.st_mode & 0o022:
        return True
    return bool(metadata.st_uid == 0 and metadata.st_mode & stat.S_ISVTX)


def _open_anchored_directory_chain(
    path: Path,
    *,
    label: str,
) -> _AnchoredDirectoryChain:
    _validate_exact_absolute_path(path, label=label)
    if path.anchor != os.sep or os.path.abspath(str(path)) != str(path):
        _fail(f"{label} contains an alias or traversal")
    components = [path.anchor]
    components.extend(
        _validate_path_component(component, label=f"{label} component {index}")
        for index, component in enumerate(path.parts[1:])
    )
    descriptors: list[int] = []
    metadata: list[os.stat_result] = []
    pending_descriptor = -1
    try:
        pending_descriptor = os.open(path.anchor, _directory_flags())
        descriptors.append(pending_descriptor)
        pending_descriptor = -1
        anchor = os.fstat(descriptors[0])
        if (
            not stat.S_ISDIR(anchor.st_mode)
            or anchor.st_uid != 0
            or anchor.st_mode & 0o022
            or os.get_inheritable(descriptors[0])
        ):
            _fail(f"{label} filesystem anchor metadata differs")
        metadata.append(anchor)
        for component in components[1:]:
            parent = descriptors[-1]
            before = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if (
                not stat.S_ISDIR(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or not _directory_component_metadata_is_secure(before)
            ):
                _fail(f"{label} contains an insecure or non-directory component")
            pending_descriptor = os.open(component, _directory_flags(), dir_fd=parent)
            descriptors.append(pending_descriptor)
            pending_descriptor = -1
            opened = os.fstat(descriptors[-1])
            after = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if (
                _directory_locator_identity(before) != _directory_locator_identity(opened)
                or _directory_locator_identity(opened) != _directory_locator_identity(after)
                or os.get_inheritable(descriptors[-1])
            ):
                _fail(f"{label} changed while its descriptor chain was opened")
            metadata.append(opened)
    except OSError as exc:
        failure = ForagerMatchedV3CpuOciBuildPublicationError(
            f"{label} cannot be opened without following links"
        )
        if pending_descriptor >= 0:
            try:
                os.close(pending_descriptor)
            except BaseException as cleanup_error:
                failure.add_note(
                    f"untransferred {label} descriptor cleanup also failed: {cleanup_error!r}"
                )
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                failure.add_note(f"{label} descriptor cleanup also failed: {cleanup_error!r}")
        raise failure from exc
    except BaseException as failure:
        if pending_descriptor >= 0:
            try:
                os.close(pending_descriptor)
            except BaseException as cleanup_error:
                failure.add_note(
                    f"untransferred {label} descriptor cleanup also failed: {cleanup_error!r}"
                )
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                failure.add_note(f"{label} descriptor cleanup also failed: {cleanup_error!r}")
        raise
    chain = _AnchoredDirectoryChain(descriptors, components, metadata, label)
    try:
        chain.verify()
    except BaseException as failure:
        try:
            chain.close()
        except BaseException as cleanup_error:
            failure.add_note(f"{label} cleanup also failed: {cleanup_error!r}")
        raise
    return chain


@contextmanager
def _open_root(path: Path, *, label: str, mutable: bool) -> Iterator[_OpenRoot]:
    chain: _AnchoredDirectoryChain | None = None
    failure: BaseException | None = None
    try:
        chain = _open_anchored_directory_chain(path, label=label)
        descriptor = chain.descriptor
        opened = os.fstat(descriptor)
        if (
            opened.st_uid != os.geteuid()
            or not _directory_component_metadata_is_secure(opened)
            or (mutable and stat.S_IMODE(opened.st_mode) != 0o700)
        ):
            _fail(f"{label} metadata differs")
        identity = _directory_locator_identity(opened)
        yield _OpenRoot(path, descriptor, identity, chain)
        chain.verify()
        after = os.fstat(descriptor)
        if _directory_locator_identity(after) != identity:
            _fail(f"{label} identity changed while used")
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if chain is not None:
            try:
                chain.close()
            except BaseException as exc:
                if failure is not None:
                    failure.add_note(f"{label} descriptor-chain cleanup also failed: {exc}")
                else:
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        f"{label} descriptor-chain cleanup failed"
                    ) from exc


def _open_directory_at(parent: int, name: str, *, label: str) -> int:
    if _SAFE_FILE_RE.fullmatch(name) is None:
        _fail(f"{label} name is unsafe")
    descriptor = -1
    failure: BaseException | None = None
    try:
        descriptor = os.open(name, _directory_flags(), dir_fd=parent)
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_locator_identity(opened) != _directory_locator_identity(located)
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o022
            or os.get_inheritable(descriptor)
        ):
            _fail(f"{label} metadata differs")
        result = descriptor
        descriptor = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if failure is not None:
                    failure.add_note(f"{label} descriptor cleanup also failed: {exc}")
                else:
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        f"{label} descriptor cleanup failed"
                    ) from exc


def _ensure_directory_at(parent: int, name: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
        os.fsync(parent)
    except FileExistsError:
        pass
    descriptor = -1
    failure: BaseException | None = None
    try:
        descriptor = _open_directory_at(parent, name, label=f"publication {name} directory")
        opened = os.fstat(descriptor)
        if stat.S_IMODE(opened.st_mode) != 0o700:
            _fail(f"publication {name} directory mode differs")
        result = descriptor
        descriptor = -1
        return result
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as exc:
                if failure is not None:
                    failure.add_note(f"publication {name} descriptor cleanup also failed: {exc}")
                else:
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        f"publication {name} descriptor cleanup failed"
                    ) from exc


def _prepare_layout(root: _OpenRoot) -> None:
    for category in ("failures", "intents", "successes"):
        category_fd = _ensure_directory_at(root.descriptor, category)
        sha_fd = -1
        primary: BaseException | None = None
        try:
            sha_fd = _ensure_directory_at(category_fd, "sha256")
            _verify_publication_namespace(root, category, category_fd, sha_fd)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_descriptors(
                (
                    (sha_fd, f"publication {category} sha256 descriptor"),
                    (category_fd, f"publication {category} descriptor"),
                ),
                primary=primary,
            )
    os.fsync(root.descriptor)


def _read_descriptor(descriptor: int, *, expected_size: int, label: str) -> bytes:
    result = bytearray()
    while len(result) < expected_size:
        block = os.read(descriptor, min(_READ_CHUNK_BYTES, expected_size - len(result)))
        if not block:
            _fail(f"{label} ended early")
        result.extend(block)
    if os.read(descriptor, 1):
        _fail(f"{label} grew while read")
    return bytes(result)


def _read_pinned_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> bytes:
    expected_sha = _require_sha256(expected_sha256, label=f"expected {label}")
    expected_size = _require_integer(
        expected_size_bytes,
        label=f"expected {label} size",
        minimum=1,
        maximum=2 * 1024 * 1024 * 1024,
    )
    _validate_exact_absolute_path(path, label=label)
    name = _validate_path_component(path.name, label=f"{label} filename")
    chain = _open_anchored_directory_chain(path.parent, label=f"{label} parent")
    descriptor = -1
    failure: BaseException | None = None
    try:
        before = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _file_flags(), dir_fd=chain.descriptor)
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or stat.S_IMODE(opened.st_mode) != 0o444
            or opened.st_size != expected_size
            or _stat_identity(opened) != _stat_identity(before)
            or _stat_identity(opened) != _stat_identity(located)
            or os.get_inheritable(descriptor)
        ):
            _fail(f"{label} metadata differs")
        raw = _read_descriptor(descriptor, expected_size=expected_size, label=label)
        after = os.fstat(descriptor)
        located_after = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        chain.verify()
        if (
            _stat_identity(opened) != _stat_identity(after)
            or _stat_identity(opened) != _stat_identity(located_after)
            or not hmac.compare_digest(_sha256(raw), expected_sha)
        ):
            _fail(f"{label} identity changed or differs")
        return raw
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if failure is not None:
                    failure.add_note(
                        f"{label} file descriptor cleanup also failed: {cleanup_error!r}"
                    )
                else:
                    failure = ForagerMatchedV3CpuOciBuildPublicationError(
                        f"{label} file descriptor cleanup failed"
                    )
                    failure.__cause__ = cleanup_error
        try:
            chain.close()
        except BaseException as cleanup_error:
            if failure is not None:
                failure.add_note(
                    f"{label} parent descriptor cleanup also failed: {cleanup_error!r}"
                )
            else:
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    f"{label} parent descriptor cleanup failed"
                ) from cleanup_error
        if failure is not None and sys.exc_info()[0] is None:
            raise failure


def _artifact_path(root: Path, *parts: str) -> Path:
    if any(_SAFE_FILE_RE.fullmatch(part) is None for part in parts):
        _fail("production artifact path component is unsafe")
    return root.joinpath(*parts)


def _load_production_inputs(artifact_root: Path) -> _LoadedProductionInputs:
    """Replay only the one frozen r5 production lineage from an explicit root."""

    with _open_root(artifact_root, label="production artifact root", mutable=False):
        wheel_directory = _artifact_path(
            artifact_root,
            "wheelhouse-publications",
            "sha256",
            PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256,
        )
        wheelhouse_contract.validate_published_matched_v3_cpu_wheelhouse(
            wheel_directory,
            expected_receipt_sha256=PRODUCTION_WHEELHOUSE_RECEIPT_SHA256,
            expected_archive_sha256=PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256,
        )
        wheel_archive = _read_pinned_file(
            wheel_directory / "wheelhouse.v1.tar",
            expected_sha256=PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256,
            expected_size_bytes=PRODUCTION_WHEELHOUSE_ARCHIVE_SIZE_BYTES,
            label="production wheelhouse archive",
        )
        wheel_receipt = _read_pinned_file(
            wheel_directory / "receipt.v1.json",
            expected_sha256=PRODUCTION_WHEELHOUSE_RECEIPT_SHA256,
            expected_size_bytes=PRODUCTION_WHEELHOUSE_RECEIPT_SIZE_BYTES,
            label="production wheelhouse receipt",
        )

        external_directory = _artifact_path(
            artifact_root,
            "external-source-bundle-publications",
            "sha256",
            PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
        )
        external_receipt_value = (
            external_publication_contract.validate_published_matched_v3_external_source(
                external_directory,
                expected_receipt_sha256=PRODUCTION_EXTERNAL_RECEIPT_SHA256,
                expected_archive_sha256=PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
            )
        )
        external_archive = _read_pinned_file(
            external_directory / "external-source.v1.tar",
            expected_sha256=PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
            expected_size_bytes=PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES,
            label="production external source archive",
        )
        external_receipt = _read_pinned_file(
            external_directory / "receipt.v1.json",
            expected_sha256=PRODUCTION_EXTERNAL_RECEIPT_SHA256,
            expected_size_bytes=PRODUCTION_EXTERNAL_RECEIPT_SIZE_BYTES,
            label="production external source receipt",
        )
        external_archive_record = cast(Mapping[str, Any], external_receipt_value["archive"])
        external_source_record = cast(
            Mapping[str, Any], external_receipt_value["external_source_manifest"]
        )
        external_staging_record = cast(
            Mapping[str, Any], external_receipt_value["staging_manifest"]
        )
        if (
            external_archive_record["member_count"] != PRODUCTION_EXTERNAL_MEMBER_COUNT
            or external_source_record["full_file_sha256"]
            != PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256
            or external_source_record["source_tree_sha256"]
            != PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256
            or external_staging_record["full_file_sha256"]
            != PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256
        ):
            _fail("production external source provenance differs")

        capture_directory = _artifact_path(
            artifact_root,
            "capture-manifest-publications",
            "sha256",
            PRODUCTION_CAPTURE_MANIFEST_SHA256,
        )
        capture_manifest = _read_pinned_file(
            capture_directory / "manifest.v1.json",
            expected_sha256=PRODUCTION_CAPTURE_MANIFEST_SHA256,
            expected_size_bytes=PRODUCTION_CAPTURE_MANIFEST_SIZE_BYTES,
            label="production capture manifest",
        )
        runtime_directory = _artifact_path(
            artifact_root,
            "runtime-lock-publications",
            "sha256",
            PRODUCTION_RUNTIME_LOCK_SHA256,
        )
        runtime_lock = _read_pinned_file(
            runtime_directory / "runtime-lock.v1.json",
            expected_sha256=PRODUCTION_RUNTIME_LOCK_SHA256,
            expected_size_bytes=PRODUCTION_RUNTIME_LOCK_SIZE_BYTES,
            label="production runtime lock",
        )
        cas_manifest = _read_pinned_file(
            runtime_directory / "wheelhouse.cas-manifest.v1.json",
            expected_sha256=PRODUCTION_CAS_MANIFEST_SHA256,
            expected_size_bytes=PRODUCTION_CAS_MANIFEST_SIZE_BYTES,
            label="production wheel CAS manifest",
        )
        issuance_envelope = _read_pinned_file(
            runtime_directory / "issuance-envelope.v1.json",
            expected_sha256=PRODUCTION_ISSUANCE_ENVELOPE_SHA256,
            expected_size_bytes=PRODUCTION_ISSUANCE_ENVELOPE_SIZE_BYTES,
            label="production runtime issuance envelope",
        )
        issued = runtime_lock_issuer.issue_matched_v3_cpu_runtime_lock(
            capture_manifest_raw=capture_manifest,
            expected_capture_manifest_sha256=PRODUCTION_CAPTURE_MANIFEST_SHA256,
            wheelhouse_receipt_raw=wheel_receipt,
            expected_wheelhouse_receipt_sha256=PRODUCTION_WHEELHOUSE_RECEIPT_SHA256,
            issuance_envelope_raw=issuance_envelope,
            expected_issuance_envelope_sha256=PRODUCTION_ISSUANCE_ENVELOPE_SHA256,
        )
        if (
            issued.runtime_lock_bytes != runtime_lock
            or issued.cas_manifest_bytes != cas_manifest
            or issued.issuance_envelope_bytes != issuance_envelope
        ):
            _fail("pure reissuance differs from the frozen runtime publication")
        validated = runtime_lock_issuer.validate_production_cpu_runtime_lock_issuance(
            issued,
            expected_root_pin_inventory_sha256=PRODUCTION_ROOT_PIN_INVENTORY_SHA256,
            expected_selected_wheel_inventory_sha256=(PRODUCTION_SELECTED_WHEEL_INVENTORY_SHA256),
            expected_resolution_lock_sha256=PRODUCTION_RESOLUTION_LOCK_SHA256,
            expected_resolution_lock_size_bytes=PRODUCTION_RESOLUTION_LOCK_SIZE_BYTES,
        )
        return _LoadedProductionInputs(
            issuance_artifacts=validated,
            wheelhouse_archive_bytes=wheel_archive,
            external_source_archive_bytes=external_archive,
            external_source_receipt_bytes=external_receipt,
        )


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            _fail("publication write made no progress")
        view = view[written:]


def _open_publication_namespace(root: _OpenRoot, category: str) -> tuple[int, int]:
    category_fd = _open_directory_at(
        root.descriptor,
        category,
        label=f"publication {category} directory",
    )
    sha_fd = -1
    try:
        sha_fd = _open_directory_at(
            category_fd,
            "sha256",
            label=f"publication {category} sha256 directory",
        )
        _verify_publication_namespace(root, category, category_fd, sha_fd)
    except BaseException as primary:
        _close_descriptors(
            (
                (sha_fd, f"publication {category} sha256 descriptor"),
                (category_fd, f"publication {category} descriptor"),
            ),
            primary=primary,
        )
        raise
    return category_fd, sha_fd


def _verify_publication_namespace(
    root: _OpenRoot,
    category: str,
    category_fd: int,
    namespace_fd: int,
) -> None:
    if category not in {"failures", "intents", "successes"}:
        _fail("publication category is unsupported")
    root.chain.verify()
    root_opened = os.fstat(root.descriptor)
    category_opened = os.fstat(category_fd)
    category_located = os.stat(
        category,
        dir_fd=root.descriptor,
        follow_symlinks=False,
    )
    namespace_opened = os.fstat(namespace_fd)
    namespace_located = os.stat(
        "sha256",
        dir_fd=category_fd,
        follow_symlinks=False,
    )
    if (
        _directory_locator_identity(root_opened) != root.identity
        or _directory_locator_identity(category_opened)
        != _directory_locator_identity(category_located)
        or _directory_locator_identity(namespace_opened)
        != _directory_locator_identity(namespace_located)
        or category_opened.st_uid != os.geteuid()
        or namespace_opened.st_uid != os.geteuid()
        or stat.S_IMODE(category_opened.st_mode) != 0o700
        or stat.S_IMODE(namespace_opened.st_mode) != 0o700
        or os.get_inheritable(category_fd)
        or os.get_inheritable(namespace_fd)
    ):
        _fail(f"publication {category} namespace locator changed")


def _rename_new_only(parent: int, source: str, target: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail("renameat2 is required for atomic new-only publication")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent,
        source.encode("ascii"),
        parent,
        target.encode("ascii"),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), target)
        raise OSError(error, os.strerror(error), target)


def _reconcile_staging_rename(
    parent: int,
    source: str,
    target: str,
    staging_fd: int,
) -> str:
    """Classify a rename by comparing both names with the held staging inode."""

    held = os.fstat(staging_fd)
    held_identity = (held.st_dev, held.st_ino)

    def observe(name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return None

    source_metadata = observe(source)
    target_metadata = observe(target)
    source_is_held = (
        source_metadata is not None
        and (
            source_metadata.st_dev,
            source_metadata.st_ino,
        )
        == held_identity
    )
    target_is_held = (
        target_metadata is not None
        and (
            target_metadata.st_dev,
            target_metadata.st_ino,
        )
        == held_identity
    )
    if target_is_held and source_metadata is None:
        return "committed"
    if source_is_held and not target_is_held:
        return "not_committed" if target_metadata is None else "collision"
    return "uncertain"


def _validate_exact_absolute_path(path: Path, *, label: str) -> None:
    concrete_path_type = type(Path())
    if (
        type(path) is not concrete_path_type
        or not path.is_absolute()
        or path == Path("/")
        or any(part in {".", ".."} for part in path.parts)
    ):
        _fail(f"{label} must be one exact non-root absolute pathlib.Path")


def _write_snapshot_manifest_new_only_once(
    path: Path,
    raw: bytes,
    *,
    expected_sha256: str,
    publication_state: list[bool],
) -> None:
    """Write one measured manifest without replacing any existing pathname."""

    _validate_exact_absolute_path(path, label="snapshot manifest output")
    if _SAFE_FILE_RE.fullmatch(path.name) is None:
        _fail("snapshot manifest output filename is unsafe")
    expected = _require_sha256(expected_sha256, label="measured snapshot manifest")
    if type(raw) is not bytes or not raw or _sha256(raw) != expected:
        _fail("measured snapshot manifest bytes differ from their pin")
    staging_name = f"snapshot-staging-{secrets.token_hex(16)}"
    descriptor = -1
    staging_identity: tuple[int, int] | None = None
    renamed = False
    rename_attempted = False
    primary: BaseException | None = None
    with _open_root(path.parent, label="snapshot manifest output parent", mutable=False) as root:
        try:
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if type(nofollow) is not int:
                _fail("snapshot publication requires O_NOFOLLOW")
            descriptor = os.open(
                staging_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=root.descriptor,
            )
            opened = os.fstat(descriptor)
            staging_identity = (opened.st_dev, opened.st_ino)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) != 0o400
                or os.get_inheritable(descriptor)
            ):
                _fail("snapshot manifest staging metadata differs")
            _write_all(descriptor, raw)
            os.fchmod(descriptor, 0o400)
            os.fsync(descriptor)
            after_write = os.fstat(descriptor)
            if (
                after_write.st_dev,
                after_write.st_ino,
            ) != staging_identity or after_write.st_size != len(raw):
                _fail("snapshot manifest staging identity changed")
            rename_attempted = True
            try:
                _rename_new_only(root.descriptor, staging_name, path.name)
            except BaseException as rename_error:
                try:
                    rename_status = _reconcile_staging_rename(
                        root.descriptor,
                        staging_name,
                        path.name,
                        descriptor,
                    )
                except BaseException as reconciliation_error:
                    uncertainty = MatchedV3CpuOciBuildPublicationStateUncertainError(
                        "snapshot manifest rename failed and cannot be reconciled",
                        image_state_uncertain=False,
                    )
                    uncertainty.add_note(
                        "rename failure before reconciliation: "
                        f"{type(rename_error).__name__}: {rename_error}"
                    )
                    raise uncertainty from reconciliation_error
                if rename_status == "committed":
                    publication_state[0] = True
                    renamed = True
                    raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                        "snapshot manifest became visible although rename reported failure",
                        image_state_uncertain=False,
                    ) from rename_error
                if rename_status == "not_committed":
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest rename failed before publication"
                    ) from rename_error
                if isinstance(rename_error, FileExistsError) and rename_status == "collision":
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest output already exists; refusing replacement"
                    ) from rename_error
                raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                    "snapshot manifest rename escaped with uncertain visible state",
                    image_state_uncertain=False,
                ) from rename_error
            rename_status = _reconcile_staging_rename(
                root.descriptor,
                staging_name,
                path.name,
                descriptor,
            )
            if rename_status == "not_committed":
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    "snapshot manifest rename returned without publication"
                )
            if rename_status != "committed":
                raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                    "snapshot manifest rename returned without the exact visible file",
                    image_state_uncertain=False,
                )
            publication_state[0] = True
            renamed = True
            os.fsync(root.descriptor)
            replayed = _read_file_at(
                root.descriptor,
                path.name,
                expected_sha256=expected,
                expected_size_bytes=len(raw),
            )
            if replayed != raw:
                _fail("published snapshot manifest replay differs")
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_failures: list[BaseException] = []
            terminal_state: ForagerMatchedV3CpuOciBuildPublicationError | None = None
            if rename_attempted and not renamed and descriptor >= 0:
                try:
                    final_rename_status = _reconcile_staging_rename(
                        root.descriptor,
                        staging_name,
                        path.name,
                        descriptor,
                    )
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
                    final_rename_status = "uncertain"
                if final_rename_status == "committed":
                    publication_state[0] = True
                    renamed = True
                    terminal_state = MatchedV3CpuOciBuildPublicationStateUncertainError(
                        "snapshot manifest commit was discovered during cleanup",
                        image_state_uncertain=False,
                    )
                elif final_rename_status == "not_committed":
                    terminal_state = ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest was proven not committed during final cleanup"
                    )
                elif final_rename_status == "collision":
                    terminal_state = ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest output already exists; refusing replacement"
                    )
                else:
                    terminal_state = MatchedV3CpuOciBuildPublicationStateUncertainError(
                        "snapshot manifest rename retained uncertain visible state",
                        image_state_uncertain=False,
                    )
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if (
                not renamed
                and staging_identity is not None
                and not isinstance(
                    terminal_state,
                    MatchedV3CpuOciBuildPublicationStateUncertainError,
                )
            ):
                try:
                    located = os.stat(
                        staging_name,
                        dir_fd=root.descriptor,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(located.st_mode)
                        or (located.st_dev, located.st_ino) != staging_identity
                    ):
                        _fail("snapshot manifest staging name changed before cleanup")
                    os.unlink(staging_name, dir_fd=root.descriptor)
                    os.fsync(root.descriptor)
                except FileNotFoundError:
                    pass
                except BaseException as cleanup_error:
                    cleanup_failures.append(cleanup_error)
            if terminal_state is not None:
                for failed_cleanup in cleanup_failures:
                    terminal_state.add_note(
                        f"snapshot manifest cleanup also failed: {failed_cleanup!r}"
                    )
                if primary is not None and primary is not terminal_state:
                    raise terminal_state from primary
                raise terminal_state
            if cleanup_failures:
                if primary is not None:
                    for failed_cleanup in cleanup_failures:
                        primary.add_note(
                            f"snapshot manifest cleanup also failed: {failed_cleanup!r}"
                        )
                else:
                    raise ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest cleanup failed"
                    ) from cleanup_failures[0]


def _write_snapshot_manifest_new_only(path: Path, raw: bytes, *, expected_sha256: str) -> None:
    """Write new-only bytes and normalize every exact postcommit escape."""

    publication_state = [False]
    try:
        _write_snapshot_manifest_new_only_once(
            path,
            raw,
            expected_sha256=expected_sha256,
            publication_state=publication_state,
        )
    except BaseException as exc:
        if publication_state[0] and not isinstance(
            exc,
            MatchedV3CpuOciBuildPublicationStateUncertainError,
        ):
            raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                "snapshot manifest became visible but final durability or replay failed",
                image_state_uncertain=False,
            ) from exc
        raise


def _read_request_snapshot_manifest(path: Path, *, expected_sha256: str) -> bytes:
    _validate_exact_absolute_path(path, label="snapshot manifest input")
    expected = _require_sha256(expected_sha256, label="expected snapshot manifest")
    name = _validate_path_component(path.name, label="snapshot manifest input filename")
    chain = _open_anchored_directory_chain(
        path.parent,
        label="snapshot manifest input parent",
    )
    descriptor = -1
    primary: BaseException | None = None
    try:
        before = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        descriptor = os.open(name, _file_flags(), dir_fd=chain.descriptor)
        opened = os.fstat(descriptor)
        located = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o222
            or not 1 <= opened.st_size <= 16 * 1024 * 1024
            or _stat_identity(opened) != _stat_identity(before)
            or _stat_identity(opened) != _stat_identity(located)
            or os.get_inheritable(descriptor)
        ):
            _fail("snapshot manifest input metadata differs")
        raw = _read_descriptor(
            descriptor,
            expected_size=opened.st_size,
            label="snapshot manifest input",
        )
        after = os.fstat(descriptor)
        located_after = os.stat(name, dir_fd=chain.descriptor, follow_symlinks=False)
        chain.verify()
        if (
            _stat_identity(opened) != _stat_identity(after)
            or _stat_identity(opened) != _stat_identity(located_after)
            or not hmac.compare_digest(_sha256(raw), expected)
        ):
            _fail("snapshot manifest input identity changed or differs")
        local_snapshot_contract.parse_matched_v3_local_source_snapshot_manifest(
            raw,
            expected_full_sha256=expected,
        )
        return raw
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except BaseException as cleanup_error:
                if primary is not None:
                    primary.add_note(
                        f"snapshot manifest input cleanup also failed: {cleanup_error!r}"
                    )
                else:
                    primary = ForagerMatchedV3CpuOciBuildPublicationError(
                        "snapshot manifest input cleanup failed"
                    )
                    primary.__cause__ = cleanup_error
        try:
            chain.close()
        except BaseException as cleanup_error:
            if primary is not None:
                primary.add_note(
                    f"snapshot manifest input parent cleanup also failed: {cleanup_error!r}"
                )
            else:
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    "snapshot manifest input parent cleanup failed"
                ) from cleanup_error
        if primary is not None and sys.exc_info()[0] is None:
            raise primary


def measure_matched_v3_cpu_oci_build_request(
    *,
    repository_root: Path,
    manifest_output: Path,
) -> MeasuredMatchedV3CpuOciBuildRequest:
    """Prepare caller-carried local-source pins without loading execution authority."""

    _validate_exact_absolute_path(repository_root, label="repository root")
    measured = local_snapshot_contract.measure_matched_v3_local_source_snapshot(
        repository_root=repository_root,
    )
    _write_snapshot_manifest_new_only(
        manifest_output,
        measured.canonical_manifest_bytes,
        expected_sha256=measured.full_sha256,
    )
    return MeasuredMatchedV3CpuOciBuildRequest(
        manifest_path=manifest_output,
        manifest_sha256=measured.full_sha256,
        tree_sha256=measured.tree_sha256,
        directory_count=measured.directory_count,
        file_count=measured.file_count,
        total_size_bytes=measured.total_size_bytes,
    )


def _read_file_at(
    directory: int,
    name: str,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
) -> bytes:
    descriptor = os.open(name, _file_flags(), dir_fd=directory)
    primary: BaseException | None = None
    try:
        before = os.fstat(descriptor)
        located = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or before.st_size != expected_size_bytes
            or _stat_identity(before) != _stat_identity(located)
        ):
            _fail(f"published file metadata differs: {name}")
        raw = _read_descriptor(
            descriptor,
            expected_size=expected_size_bytes,
            label=f"published file {name}",
        )
        after = os.fstat(descriptor)
        located_after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(located_after)
            or _sha256(raw) != expected_sha256
        ):
            _fail(f"published file identity differs: {name}")
        return raw
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_descriptors(
            ((descriptor, f"published file {name} descriptor"),),
            primary=primary,
        )


def _read_unpinned_file_at(
    directory: int,
    name: str,
    *,
    maximum_size_bytes: int,
) -> bytes:
    descriptor = os.open(name, _file_flags(), dir_fd=directory)
    primary: BaseException | None = None
    try:
        before = os.fstat(descriptor)
        located = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o400
            or not 1 <= before.st_size <= maximum_size_bytes
            or _stat_identity(before) != _stat_identity(located)
        ):
            _fail(f"published file metadata differs: {name}")
        raw = _read_descriptor(
            descriptor,
            expected_size=before.st_size,
            label=f"published file {name}",
        )
        after = os.fstat(descriptor)
        located_after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if _stat_identity(before) != _stat_identity(after) or _stat_identity(
            before
        ) != _stat_identity(located_after):
            _fail(f"published file identity changed: {name}")
        return raw
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_descriptors(
            ((descriptor, f"published file {name} descriptor"),),
            primary=primary,
        )


def _replay_directory_fd(
    directory: int,
    files: Mapping[str, tuple[str, int]],
) -> dict[str, bytes]:
    observed = set(os.listdir(directory))
    if observed != set(files):
        _fail("published directory file set differs")
    result: dict[str, bytes] = {}
    for name in sorted(files, key=str.encode):
        expected_sha, expected_size = files[name]
        result[name] = _read_file_at(
            directory,
            name,
            expected_sha256=expected_sha,
            expected_size_bytes=expected_size,
        )
    return result


def _cleanup_staging(parent: int, staging_fd: int, staging_name: str) -> None:
    held = os.fstat(staging_fd)
    try:
        located = os.stat(staging_name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ForagerMatchedV3CpuOciBuildPublicationError(
            "publication staging name vanished before cleanup"
        ) from exc
    if not stat.S_ISDIR(located.st_mode) or (located.st_dev, located.st_ino) != (
        held.st_dev,
        held.st_ino,
    ):
        _fail("publication staging name no longer identifies the held directory")
    os.fchmod(staging_fd, 0o700)
    for name in os.listdir(staging_fd):
        if _SAFE_FILE_RE.fullmatch(name) is None:
            _fail("publication staging contains an unsafe entry")
        observed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
        if not stat.S_ISREG(observed.st_mode):
            _fail("publication staging contains a non-regular entry")
        os.unlink(name, dir_fd=staging_fd)
    os.fsync(staging_fd)
    located_after = os.stat(staging_name, dir_fd=parent, follow_symlinks=False)
    if (located_after.st_dev, located_after.st_ino) != (held.st_dev, held.st_ino):
        _fail("publication staging name changed during cleanup")
    os.rmdir(staging_name, dir_fd=parent)
    os.fsync(parent)


def _publish_files(
    root: _OpenRoot,
    *,
    category: str,
    address: str,
    files: Mapping[str, bytes],
    intent: bool = False,
    commit_state: _IntentCommitState | None = None,
) -> Path:
    _require_sha256(address, label=f"{category} publication address")
    if not files or any(_SAFE_FILE_RE.fullmatch(name) is None for name in files):
        _fail("publication file set is empty or unsafe")
    if commit_state is not None and (type(commit_state) is not _IntentCommitState or not intent):
        _fail("publication commit state is not an exact intent-only latch")
    category_fd, namespace_fd = _open_publication_namespace(root, category)
    staging_name = f"staging-{secrets.token_hex(16)}"
    staging_fd = -1
    renamed = False
    rename_attempted = False
    primary: BaseException | None = None
    try:
        os.mkdir(staging_name, 0o700, dir_fd=namespace_fd)
        staging_fd = _open_directory_at(
            namespace_fd,
            staging_name,
            label="publication staging directory",
        )
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        for name in sorted(files, key=str.encode):
            raw = files[name]
            if type(raw) is not bytes or not raw:
                _fail(f"publication file is not nonempty exact bytes: {name}")
            nofollow = getattr(os, "O_NOFOLLOW", None)
            if type(nofollow) is not int:
                _fail("publication requires O_NOFOLLOW for created files")
            descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | nofollow | getattr(os, "O_CLOEXEC", 0),
                0o400,
                dir_fd=staging_fd,
            )
            file_primary: BaseException | None = None
            try:
                _write_all(descriptor, raw)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
            except BaseException as exc:
                file_primary = exc
                raise
            finally:
                _close_descriptors(
                    ((descriptor, f"staged publication file {name}"),),
                    primary=file_primary,
                )
        expected = {name: (_sha256(raw), len(raw)) for name, raw in files.items()}
        _replay_directory_fd(staging_fd, expected)
        os.fchmod(staging_fd, 0o500)
        os.fsync(staging_fd)
        os.fsync(namespace_fd)
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        rename_attempted = True
        try:
            _rename_new_only(namespace_fd, staging_name, address)
        except BaseException as rename_error:
            try:
                rename_status = _reconcile_staging_rename(
                    namespace_fd,
                    staging_name,
                    address,
                    staging_fd,
                )
            except BaseException as reconciliation_error:
                uncertainty = MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} rename failed and its visible state cannot be reconciled",
                    image_state_uncertain=(category == "successes"),
                )
                uncertainty.add_note(
                    "rename failure before reconciliation: "
                    f"{type(rename_error).__name__}: {rename_error}"
                )
                raise uncertainty from reconciliation_error
            if rename_status == "committed":
                if commit_state is not None:
                    commit_state.committed = True
                renamed = True
                raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} publication became visible although rename reported failure",
                    image_state_uncertain=(category == "successes"),
                ) from rename_error
            if rename_status == "not_committed":
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    f"{category} publication rename failed before commit"
                ) from rename_error
            if not isinstance(rename_error, FileExistsError) or rename_status != "collision":
                raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} publication rename escaped with uncertain visible state",
                    image_state_uncertain=(category == "successes"),
                ) from rename_error
            collision: ForagerMatchedV3CpuOciBuildPublicationError
            if intent:
                collision = MatchedV3CpuOciBuildIntentExistsError(
                    "durable intent already exists; refusing every automatic retry",
                    context_receipt_sha256=address,
                )
            else:
                collision = ForagerMatchedV3CpuOciBuildPublicationError(
                    f"refusing to overwrite {category} publication {address}"
                )
            try:
                _cleanup_staging(namespace_fd, staging_fd, staging_name)
            except BaseException as cleanup_error:
                collision.add_note(
                    f"duplicate-publication staging cleanup also failed: {cleanup_error!r}"
                )
            closing_staging_fd = staging_fd
            staging_fd = -1
            _close_descriptors(
                ((closing_staging_fd, "duplicate publication staging directory"),),
                primary=collision,
            )
            raise collision from rename_error
        rename_status = _reconcile_staging_rename(
            namespace_fd,
            staging_name,
            address,
            staging_fd,
        )
        if rename_status == "not_committed":
            raise ForagerMatchedV3CpuOciBuildPublicationError(
                f"{category} rename returned without publication"
            )
        if rename_status != "committed":
            raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                f"{category} rename returned success without the exact visible address",
                image_state_uncertain=(category == "successes"),
            )
        if commit_state is not None:
            commit_state.committed = True
        renamed = True
        os.fsync(namespace_fd)
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        named = _open_directory_at(
            namespace_fd,
            address,
            label=f"published {category} directory",
        )
        named_primary: BaseException | None = None
        try:
            if (os.fstat(named).st_dev, os.fstat(named).st_ino) != (
                os.fstat(staging_fd).st_dev,
                os.fstat(staging_fd).st_ino,
            ) or stat.S_IMODE(os.fstat(named).st_mode) != 0o500:
                _fail("published destination differs from staged directory")
            _replay_directory_fd(named, expected)
        except BaseException as exc:
            named_primary = exc
            raise
        finally:
            _close_descriptors(
                ((named, f"published {category} replay directory"),),
                primary=named_primary,
            )
        os.fsync(namespace_fd)
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        _verify_addressed_directory(namespace_fd, address, staging_fd)
        return root.path / category / "sha256" / address
    except BaseException as exc:
        primary = exc
        if renamed and not isinstance(exc, MatchedV3CpuOciBuildPublicationStateUncertainError):
            raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                f"{category} publication became visible but final replay failed",
                image_state_uncertain=(category == "successes"),
            ) from exc
        raise
    finally:
        cleanup_failures: list[BaseException] = []
        terminal_state: ForagerMatchedV3CpuOciBuildPublicationError | None = None
        if rename_attempted and not renamed and staging_fd >= 0:
            try:
                final_rename_status = _reconcile_staging_rename(
                    namespace_fd,
                    staging_name,
                    address,
                    staging_fd,
                )
            except BaseException as exc:
                cleanup_failures.append(exc)
                final_rename_status = "uncertain"
            if final_rename_status == "committed":
                if commit_state is not None:
                    commit_state.committed = True
                renamed = True
                terminal_state = MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} publication commit was discovered during final cleanup",
                    image_state_uncertain=(category == "successes"),
                )
            elif final_rename_status == "not_committed":
                terminal_state = ForagerMatchedV3CpuOciBuildPublicationError(
                    f"{category} publication was proven not committed during final cleanup"
                )
            elif final_rename_status == "collision" and intent:
                terminal_state = MatchedV3CpuOciBuildIntentExistsError(
                    "durable intent already exists; refusing every automatic retry",
                    context_receipt_sha256=address,
                )
            else:
                terminal_state = MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} rename escape retained uncertain visible state",
                    image_state_uncertain=(category == "successes"),
                )
        try:
            _verify_publication_namespace(root, category, category_fd, namespace_fd)
            if renamed and staging_fd >= 0:
                _verify_addressed_directory(namespace_fd, address, staging_fd)
        except BaseException as exc:
            cleanup_failures.append(exc)
        if staging_fd >= 0:
            if not renamed and not isinstance(
                terminal_state,
                MatchedV3CpuOciBuildPublicationStateUncertainError,
            ):
                try:
                    _cleanup_staging(namespace_fd, staging_fd, staging_name)
                except BaseException as exc:
                    cleanup_failures.append(exc)
            try:
                os.close(staging_fd)
            except BaseException as exc:
                cleanup_failures.append(exc)
        for descriptor in (namespace_fd, category_fd):
            try:
                os.close(descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        if terminal_state is not None:
            for failure in cleanup_failures:
                terminal_state.add_note(
                    f"publication cleanup also failed: {type(failure).__name__}: {failure}"
                )
            if primary is not None and primary is not terminal_state:
                raise terminal_state from primary
            raise terminal_state
        if cleanup_failures:
            if primary is not None:
                for failure in cleanup_failures:
                    primary.add_note(
                        f"publication cleanup also failed: {type(failure).__name__}: {failure}"
                    )
            elif renamed:
                raise MatchedV3CpuOciBuildPublicationStateUncertainError(
                    f"{category} publication committed but namespace cleanup replay failed",
                    image_state_uncertain=(category == "successes"),
                ) from cleanup_failures[0]
            else:
                raise ForagerMatchedV3CpuOciBuildPublicationError(
                    "publication cleanup failed"
                ) from cleanup_failures[0]


def _build_intent_bytes(
    *,
    request: MatchedV3CpuOciBuildPublicationRequest,
    plan_artifacts: plan_contract.CpuOciBuildPlanArtifacts,
    context_capability: context_contract.RetainedMatchedV3CpuOciBuildContext,
    local_archive_bytes: bytes,
    local_receipt_bytes: bytes,
    local_receipt_sha256: str,
) -> tuple[bytes, dict[str, bytes]]:
    context_receipt_bytes = context_capability.receipt_bytes
    context_receipt_sha256 = context_capability.receipt_sha256
    closure = {
        _CONTEXT_RECEIPT_FILENAME: context_receipt_bytes,
        _LOCAL_ARCHIVE_FILENAME: local_archive_bytes,
        _LOCAL_RECEIPT_FILENAME: local_receipt_bytes,
        _PLAN_FILENAME: plan_artifacts.plan_bytes,
        _SNAPSHOT_FILENAME: request.expected_snapshot_manifest_bytes,
    }
    records = [_file_record(name, closure[name]) for name in sorted(closure, key=str.encode)]
    intent: dict[str, Any] = {
        "authorization": {
            "created_before_durable_intent": False,
            "executor_authorization_created": False,
            "executor_authorization_single_use_required": True,
            "intent_commit_precedes_authorization": True,
        },
        "claims": _claims(),
        "classification": _INTENT_CLASSIFICATION,
        "closure_files": records,
        "context": {
            "archive_sha256": context_capability.archive_sha256,
            "archive_size_bytes": context_capability.archive_size_bytes,
            "execution_projection_sha256": context_capability.execution_projection_sha256,
            "plan_sha256": context_capability.plan_sha256,
            "receipt_sha256": context_receipt_sha256,
            "receipt_size_bytes": len(context_receipt_bytes),
        },
        "immutable_inputs": _production_bindings(),
        "limitations": _limitations(),
        "local_source": {
            "archive_sha256": _sha256(local_archive_bytes),
            "archive_size_bytes": len(local_archive_bytes),
            "bundle_receipt_sha256": local_receipt_sha256,
            "bundle_receipt_size_bytes": len(local_receipt_bytes),
            "snapshot_manifest_sha256": request.expected_snapshot_manifest_sha256,
            "snapshot_tree_sha256": request.expected_snapshot_tree_sha256,
        },
        "plan": {
            "sha256": plan_artifacts.plan_sha256,
            "size_bytes": len(plan_artifacts.plan_bytes),
        },
        "schema_version": CPU_OCI_BUILD_PUBLICATION_INTENT_SCHEMA_VERSION,
        "status": _INTENT_STATUS,
        "timeout_seconds": request.timeout_seconds,
        "intent_body_sha256": "0" * 64,
    }
    body = copy.deepcopy(intent)
    body.pop("intent_body_sha256")
    intent["intent_body_sha256"] = _sha256(_canonical_json(body))
    raw = _canonical_json(intent)
    parse_matched_v3_cpu_oci_build_intent(raw, expected_file_sha256=_sha256(raw))
    return raw, closure


def parse_matched_v3_cpu_oci_build_intent(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse one canonical durable pre-authorization intent under an exact digest."""

    expected = _require_sha256(expected_file_sha256, label="expected build intent")
    if type(raw) is not bytes or _sha256(raw) != expected:
        _fail("build intent full-file SHA-256 differs")
    intent = _exact(
        _parse_canonical_json(raw, label="build intent"),
        frozenset(
            {
                "authorization",
                "claims",
                "classification",
                "closure_files",
                "context",
                "immutable_inputs",
                "intent_body_sha256",
                "limitations",
                "local_source",
                "plan",
                "schema_version",
                "status",
                "timeout_seconds",
            }
        ),
        label="build intent",
    )
    if (
        intent["schema_version"] != CPU_OCI_BUILD_PUBLICATION_INTENT_SCHEMA_VERSION
        or intent["status"] != _INTENT_STATUS
        or intent["classification"] != _INTENT_CLASSIFICATION
    ):
        _fail("build intent schema, status, or classification differs")
    authorization = _exact(
        intent["authorization"],
        frozenset(
            {
                "created_before_durable_intent",
                "executor_authorization_created",
                "executor_authorization_single_use_required",
                "intent_commit_precedes_authorization",
            }
        ),
        label="build intent authorization",
    )
    if authorization != {
        "created_before_durable_intent": False,
        "executor_authorization_created": False,
        "executor_authorization_single_use_required": True,
        "intent_commit_precedes_authorization": True,
    }:
        _fail("build intent authorization ordering differs")
    claims = _exact(intent["claims"], frozenset(_claims()), label="build intent claims")
    if claims != _claims() or any(claims.values()):
        _fail("build intent authority claim became true")
    if intent["limitations"] != _limitations():
        _fail("build intent limitations differ")
    _validate_file_records(
        intent["closure_files"],
        expected_names=frozenset(
            {
                _CONTEXT_RECEIPT_FILENAME,
                _LOCAL_ARCHIVE_FILENAME,
                _LOCAL_RECEIPT_FILENAME,
                _PLAN_FILENAME,
                _SNAPSHOT_FILENAME,
            }
        ),
    )
    context = _exact(
        intent["context"],
        frozenset(
            {
                "archive_sha256",
                "archive_size_bytes",
                "execution_projection_sha256",
                "plan_sha256",
                "receipt_sha256",
                "receipt_size_bytes",
            }
        ),
        label="build intent context",
    )
    for field in (
        "archive_sha256",
        "execution_projection_sha256",
        "plan_sha256",
        "receipt_sha256",
    ):
        _require_sha256(context[field], label=f"build intent context {field}")
    _require_integer(
        context["archive_size_bytes"],
        label="build intent context archive size",
        minimum=1,
    )
    _require_integer(
        context["receipt_size_bytes"],
        label="build intent context receipt size",
        minimum=1,
        maximum=_MAX_JSON_BYTES,
    )
    local = _exact(
        intent["local_source"],
        frozenset(
            {
                "archive_sha256",
                "archive_size_bytes",
                "bundle_receipt_sha256",
                "bundle_receipt_size_bytes",
                "snapshot_manifest_sha256",
                "snapshot_tree_sha256",
            }
        ),
        label="build intent local source",
    )
    for field in (
        "archive_sha256",
        "bundle_receipt_sha256",
        "snapshot_manifest_sha256",
        "snapshot_tree_sha256",
    ):
        _require_sha256(local[field], label=f"build intent local source {field}")
    _require_integer(local["archive_size_bytes"], label="local archive size", minimum=1)
    _require_integer(
        local["bundle_receipt_size_bytes"],
        label="local bundle receipt size",
        minimum=1,
        maximum=_MAX_JSON_BYTES,
    )
    plan = _exact(
        intent["plan"],
        frozenset({"sha256", "size_bytes"}),
        label="build intent plan",
    )
    _require_sha256(plan["sha256"], label="build intent plan")
    _require_integer(plan["size_bytes"], label="build intent plan size", minimum=1)
    if plan["sha256"] != context["plan_sha256"]:
        _fail("build intent plan and context binding differ")
    if intent["immutable_inputs"] != _production_bindings():
        _fail("build intent immutable production inputs differ")
    _require_integer(
        intent["timeout_seconds"],
        label="build intent timeout",
        minimum=_MIN_TIMEOUT_SECONDS,
        maximum=_MAX_TIMEOUT_SECONDS,
    )
    supplied_body = _require_sha256(intent["intent_body_sha256"], label="build intent body")
    body = copy.deepcopy(intent)
    body.pop("intent_body_sha256")
    if supplied_body != _sha256(_canonical_json(body)):
        _fail("build intent body digest differs")
    return copy.deepcopy(intent)


def _intent_expected_files(
    intent: Mapping[str, Any],
    intent_raw: bytes,
) -> dict[str, tuple[str, int]]:
    records = cast(list[Mapping[str, Any]], intent["closure_files"])
    result = {
        cast(str, record["name"]): (
            cast(str, record["sha256"]),
            cast(int, record["size_bytes"]),
        )
        for record in records
    }
    result[_INTENT_FILENAME] = (_sha256(intent_raw), len(intent_raw))
    return result


def _validate_intent_directory_fd(
    directory: int,
    *,
    expected_context_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    expected_context = _require_sha256(
        expected_context_receipt_sha256,
        label="expected intent context receipt",
    )
    names = set(os.listdir(directory))
    required_names = {
        _INTENT_FILENAME,
        _CONTEXT_RECEIPT_FILENAME,
        _LOCAL_ARCHIVE_FILENAME,
        _LOCAL_RECEIPT_FILENAME,
        _PLAN_FILENAME,
        _SNAPSHOT_FILENAME,
    }
    if names != required_names:
        _fail("intent publication file set differs")
    intent_stat = os.stat(_INTENT_FILENAME, dir_fd=directory, follow_symlinks=False)
    if not stat.S_ISREG(intent_stat.st_mode) or not 1 <= intent_stat.st_size <= _MAX_JSON_BYTES:
        _fail("intent publication manifest metadata differs")
    intent_raw = _read_unpinned_file_at(
        directory,
        _INTENT_FILENAME,
        maximum_size_bytes=_MAX_JSON_BYTES,
    )
    intent = parse_matched_v3_cpu_oci_build_intent(
        intent_raw,
        expected_file_sha256=_sha256(intent_raw),
    )
    context_record = cast(Mapping[str, Any], intent["context"])
    if context_record["receipt_sha256"] != expected_context:
        _fail("intent publication is addressed by a different context receipt")
    files = _replay_directory_fd(directory, _intent_expected_files(intent, intent_raw))

    local_record = cast(Mapping[str, Any], intent["local_source"])
    local_snapshot_contract.parse_matched_v3_local_source_snapshot_manifest(
        files[_SNAPSHOT_FILENAME],
        expected_full_sha256=cast(str, local_record["snapshot_manifest_sha256"]),
    )
    local_receipt = local_bundle_contract.parse_matched_v3_local_source_bundle_receipt(
        files[_LOCAL_RECEIPT_FILENAME],
        expected_receipt_sha256=cast(str, local_record["bundle_receipt_sha256"]),
    )
    # The verifier does not own its descriptor.
    archive_descriptor = os.open(_LOCAL_ARCHIVE_FILENAME, _file_flags(), dir_fd=directory)
    archive_primary: BaseException | None = None
    try:
        local_bundle_contract.verify_matched_v3_local_source_bundle_archive(
            descriptor=archive_descriptor,
            expected_archive_size_bytes=cast(int, local_record["archive_size_bytes"]),
            expected_archive_sha256=cast(str, local_record["archive_sha256"]),
            expected_receipt_bytes=files[_LOCAL_RECEIPT_FILENAME],
            expected_receipt_sha256=cast(str, local_record["bundle_receipt_sha256"]),
            expected_source_snapshot_manifest_sha256=cast(
                str, local_record["snapshot_manifest_sha256"]
            ),
            expected_source_snapshot_tree_sha256=cast(str, local_record["snapshot_tree_sha256"]),
        )
    except BaseException as exc:
        archive_primary = exc
        raise
    finally:
        _close_descriptors(
            ((archive_descriptor, "published local source archive descriptor"),),
            primary=archive_primary,
        )
    source_binding = cast(Mapping[str, Any], local_receipt["source_snapshot"])
    if (
        source_binding["manifest_sha256"] != local_record["snapshot_manifest_sha256"]
        or source_binding["tree_sha256"] != local_record["snapshot_tree_sha256"]
    ):
        _fail("intent local source snapshot binding differs")

    plan_record = cast(Mapping[str, Any], intent["plan"])
    parsed_plan = plan_contract.parse_cpu_oci_build_plan(
        files[_PLAN_FILENAME],
        expected_file_sha256=cast(str, plan_record["sha256"]),
    )
    parsed_context = context_contract.parse_matched_v3_cpu_oci_build_context_receipt(
        files[_CONTEXT_RECEIPT_FILENAME],
        expected_receipt_sha256=expected_context,
    )
    parsed_context_plan = cast(Mapping[str, Any], parsed_context["plan"])
    parsed_context_archive = cast(Mapping[str, Any], parsed_context["archive"])
    if (
        parsed_context_plan["full_file_sha256"] != plan_record["sha256"]
        or parsed_context_archive["sha256"] != context_record["archive_sha256"]
        or parsed_context_archive["size_bytes"] != context_record["archive_size_bytes"]
        or parsed_context["execution_projection_sha256"]
        != context_record["execution_projection_sha256"]
    ):
        _fail("intent plan or context closure differs")
    plan_bindings = cast(Mapping[str, Any], parsed_plan["bindings"])
    sources = cast(list[Mapping[str, Any]], plan_bindings["sources"])
    local_sources = [source for source in sources if source["role"] == "local_alberta"]
    if len(local_sources) != 1:
        _fail("intent plan does not contain one local source binding")
    external_sources = [source for source in sources if source["role"] == "external_foragax"]
    runtime_binding = cast(Mapping[str, Any], plan_bindings["runtime_lock"])
    wheel_binding = cast(Mapping[str, Any], plan_bindings["wheelhouse"])
    if len(external_sources) != 1 or (
        external_sources[0]["archive_sha256"] != PRODUCTION_EXTERNAL_ARCHIVE_SHA256
        or external_sources[0]["archive_size_bytes"] != PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES
        or external_sources[0]["receipt_sha256"] != PRODUCTION_EXTERNAL_RECEIPT_SHA256
        or external_sources[0]["receipt_size_bytes"] != PRODUCTION_EXTERNAL_RECEIPT_SIZE_BYTES
        or external_sources[0]["source_manifest_sha256"]
        != PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256
        or external_sources[0]["source_tree_sha256"] != PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256
        or external_sources[0]["staging_manifest_sha256"]
        != PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256
        or runtime_binding["sha256"] != PRODUCTION_RUNTIME_LOCK_SHA256
        or wheel_binding["archive_sha256"] != PRODUCTION_WHEELHOUSE_ARCHIVE_SHA256
        or wheel_binding["receipt_sha256"] != PRODUCTION_WHEELHOUSE_RECEIPT_SHA256
        or wheel_binding["cas_manifest_sha256"] != PRODUCTION_CAS_MANIFEST_SHA256
    ):
        _fail("intent plan differs from the frozen immutable production lineage")
    plan_local = local_sources[0]
    local_archive = cast(Mapping[str, Any], local_receipt["archive"])
    if (
        plan_local["archive_sha256"] != local_record["archive_sha256"]
        or plan_local["archive_size_bytes"] != local_record["archive_size_bytes"]
        or plan_local["receipt_sha256"] != local_record["bundle_receipt_sha256"]
        or plan_local["receipt_size_bytes"] != local_record["bundle_receipt_size_bytes"]
        or plan_local["source_manifest_sha256"] != local_record["snapshot_manifest_sha256"]
        or plan_local["source_tree_sha256"] != local_record["snapshot_tree_sha256"]
        or plan_local["member_count"] != local_archive["member_count"]
    ):
        _fail("intent local source triple differs from the parsed plan binding")
    context_members = cast(list[Mapping[str, Any]], parsed_context["members"])
    local_context_members = [
        member
        for member in context_members
        if member["path"] == "inputs/local-alberta-source.v1.tar"
    ]
    if len(local_context_members) != 1 or (
        local_context_members[0]["sha256"],
        local_context_members[0]["size_bytes"],
    ) != (local_record["archive_sha256"], local_record["archive_size_bytes"]):
        _fail("intent local source archive differs from the context member binding")
    return intent, files


def _publish_intent(
    root: _OpenRoot,
    *,
    intent_raw: bytes,
    closure: Mapping[str, bytes],
    expected_context_receipt_sha256: str,
    commit_state: _IntentCommitState,
) -> Path:
    if type(commit_state) is not _IntentCommitState or commit_state.committed:
        _fail("intent publication requires one fresh exact commit latch")
    intent = parse_matched_v3_cpu_oci_build_intent(
        intent_raw,
        expected_file_sha256=_sha256(intent_raw),
    )
    if cast(Mapping[str, Any], intent["context"])["receipt_sha256"] != (
        expected_context_receipt_sha256
    ):
        _fail("intent context address differs before publication")
    files = {**closure, _INTENT_FILENAME: intent_raw}
    destination = _publish_files(
        root,
        category="intents",
        address=expected_context_receipt_sha256,
        files=files,
        intent=True,
        commit_state=commit_state,
    )
    try:
        category_fd, namespace_fd = _open_publication_namespace(root, "intents")
        directory = -1
        primary: BaseException | None = None
        try:
            directory = _open_directory_at(
                namespace_fd,
                expected_context_receipt_sha256,
                label="published build intent",
            )
            _validate_intent_directory_fd(
                directory,
                expected_context_receipt_sha256=expected_context_receipt_sha256,
            )
            _verify_publication_namespace(root, "intents", category_fd, namespace_fd)
            _verify_addressed_directory(
                namespace_fd,
                expected_context_receipt_sha256,
                directory,
            )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_descriptors(
                (
                    (directory, "published intent directory"),
                    (namespace_fd, "intent sha256 namespace"),
                    (category_fd, "intent category"),
                ),
                primary=primary,
            )
    except BaseException as exc:
        raise MatchedV3CpuOciBuildPublicationStateUncertainError(
            "durable intent committed but its final semantic replay failed",
            image_state_uncertain=False,
        ) from exc
    return destination


def _build_success_receipt(
    *,
    context_receipt_sha256: str,
    intent_sha256: str,
    execution: executor_contract.CpuOciBuildExecutionArtifacts,
    timeout_seconds: int,
    executor_module: Any | None = None,
) -> tuple[bytes, str]:
    loaded_executor_contract = (
        _load_executor_contract() if executor_module is None else executor_module
    )
    context_sha = _require_sha256(
        context_receipt_sha256,
        label="success context receipt",
    )
    exact_intent = _require_sha256(intent_sha256, label="success intent")
    execution_receipt = loaded_executor_contract.parse_matched_v3_cpu_oci_build_execution_receipt(
        execution.receipt_bytes,
        expected_receipt_sha256=execution.receipt_sha256,
    )
    execution_context = cast(Mapping[str, Any], execution_receipt["context"])
    execution_build = cast(Mapping[str, Any], execution_receipt["build"])
    if (
        execution_context["receipt_sha256"] != context_sha
        or execution_build["image_id"] != execution.image_id
        or execution_build["timeout_seconds"] != timeout_seconds
    ):
        _fail("execution receipt differs from the durable intent")
    receipt: dict[str, Any] = {
        "claims": _claims(),
        "classification": _SUCCESS_CLASSIFICATION,
        "context_receipt_sha256": context_sha,
        "execution": {
            "image_id": execution.image_id,
            "receipt_sha256": execution.receipt_sha256,
            "receipt_size_bytes": len(execution.receipt_bytes),
            "timeout_seconds": timeout_seconds,
        },
        "intent_sha256": exact_intent,
        "limitations": _limitations(),
        "publication_body_sha256": "0" * 64,
        "schema_version": CPU_OCI_BUILD_PUBLICATION_SUCCESS_SCHEMA_VERSION,
        "status": CPU_OCI_BUILD_PUBLICATION_STATUS,
    }
    body = copy.deepcopy(receipt)
    body.pop("publication_body_sha256")
    receipt["publication_body_sha256"] = _sha256(_canonical_json(body))
    raw = _canonical_json(receipt)
    parse_matched_v3_cpu_oci_build_publication_receipt(
        raw,
        expected_file_sha256=_sha256(raw),
    )
    return raw, _sha256(raw)


def parse_matched_v3_cpu_oci_build_publication_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse one canonical successful receipt-publication record."""

    expected = _require_sha256(expected_file_sha256, label="expected publication receipt")
    if type(raw) is not bytes or _sha256(raw) != expected:
        _fail("publication receipt full-file SHA-256 differs")
    receipt = _exact(
        _parse_canonical_json(raw, label="build publication receipt"),
        frozenset(
            {
                "claims",
                "classification",
                "context_receipt_sha256",
                "execution",
                "intent_sha256",
                "limitations",
                "publication_body_sha256",
                "schema_version",
                "status",
            }
        ),
        label="build publication receipt",
    )
    if (
        receipt["schema_version"] != CPU_OCI_BUILD_PUBLICATION_SUCCESS_SCHEMA_VERSION
        or receipt["status"] != CPU_OCI_BUILD_PUBLICATION_STATUS
        or receipt["classification"] != _SUCCESS_CLASSIFICATION
    ):
        _fail("publication receipt schema, status, or classification differs")
    claims = _exact(receipt["claims"], frozenset(_claims()), label="publication claims")
    if claims != _claims() or any(claims.values()):
        _fail("publication receipt authority claim became true")
    if receipt["limitations"] != _limitations():
        _fail("publication receipt limitations differ")
    _require_sha256(receipt["context_receipt_sha256"], label="publication context receipt")
    _require_sha256(receipt["intent_sha256"], label="publication intent")
    execution = _exact(
        receipt["execution"],
        frozenset({"image_id", "receipt_sha256", "receipt_size_bytes", "timeout_seconds"}),
        label="publication execution",
    )
    _require_image_id(execution["image_id"], label="publication image ID")
    _require_sha256(execution["receipt_sha256"], label="publication execution receipt")
    _require_integer(
        execution["receipt_size_bytes"],
        label="publication execution receipt size",
        minimum=1,
        maximum=_MAX_JSON_BYTES,
    )
    _require_integer(
        execution["timeout_seconds"],
        label="publication timeout",
        minimum=_MIN_TIMEOUT_SECONDS,
        maximum=_MAX_TIMEOUT_SECONDS,
    )
    supplied = _require_sha256(
        receipt["publication_body_sha256"],
        label="publication receipt body",
    )
    body = copy.deepcopy(receipt)
    body.pop("publication_body_sha256")
    if supplied != _sha256(_canonical_json(body)):
        _fail("publication receipt body digest differs")
    return copy.deepcopy(receipt)


def _build_failure_receipt(
    *,
    phase: str,
    error: BaseException,
    context_receipt_sha256: str | None,
    intent_sha256: str | None,
    plan_sha256: str | None,
    authorization_created: bool,
    executor_invoked: bool,
    build_succeeded: bool,
    image_state_uncertain: bool,
    image_id: str | None = None,
    execution_receipt_sha256: str | None = None,
) -> tuple[bytes, str]:
    allowed_phases = {
        "pre_intent",
        "authorization_failed_pre_start",
        "executor_failed_pre_start",
        "executor_failed_uncertain",
        "success_publication_failed_after_build",
    }
    if phase not in allowed_phases:
        _fail("failure receipt phase is unsupported")
    error_type = type(error).__name__
    if _SAFE_TYPE_RE.fullmatch(error_type) is None:
        error_type = "UnclassifiedError"
    message = str(error).encode("utf-8", errors="replace")
    if len(message) > _MAX_JSON_TEXT_BYTES:
        message = message[:_MAX_JSON_TEXT_BYTES]
    if context_receipt_sha256 is not None:
        _require_sha256(context_receipt_sha256, label="failure context receipt")
    if intent_sha256 is not None:
        _require_sha256(intent_sha256, label="failure intent")
    if plan_sha256 is not None:
        _require_sha256(plan_sha256, label="failure plan")
    if image_id is not None:
        _require_image_id(image_id, label="failure image ID")
    if execution_receipt_sha256 is not None:
        _require_sha256(execution_receipt_sha256, label="failure execution receipt")
    if build_succeeded and (image_id is None or execution_receipt_sha256 is None):
        _fail("post-build publication failure lacks the successful build identity")
    receipt: dict[str, Any] = {
        "authorization_created": authorization_created,
        "build_succeeded": build_succeeded,
        "claims": _claims(),
        "classification": _FAILURE_CLASSIFICATION,
        "context_receipt_sha256": context_receipt_sha256,
        "error": {
            "message_sha256": _sha256(message),
            "message_size_bytes": len(message),
            "type": error_type,
        },
        "execution_receipt_sha256": execution_receipt_sha256,
        "executor_invoked": executor_invoked,
        "failure_body_sha256": "0" * 64,
        "image_id": image_id,
        "image_state_uncertain": image_state_uncertain,
        "intent_sha256": intent_sha256,
        "limitations": _limitations(),
        "phase": phase,
        "plan_sha256": plan_sha256,
        "schema_version": CPU_OCI_BUILD_PUBLICATION_FAILURE_SCHEMA_VERSION,
        "status": _FAILURE_STATUS,
    }
    body = copy.deepcopy(receipt)
    body.pop("failure_body_sha256")
    receipt["failure_body_sha256"] = _sha256(_canonical_json(body))
    raw = _canonical_json(receipt)
    parse_matched_v3_cpu_oci_build_failure_receipt(
        raw,
        expected_file_sha256=_sha256(raw),
    )
    return raw, _sha256(raw)


def parse_matched_v3_cpu_oci_build_failure_receipt(
    raw: bytes,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    """Parse one bounded failure receipt without granting retry authority."""

    expected = _require_sha256(expected_file_sha256, label="expected failure receipt")
    if type(raw) is not bytes or _sha256(raw) != expected:
        _fail("failure receipt full-file SHA-256 differs")
    receipt = _exact(
        _parse_canonical_json(raw, label="build failure receipt"),
        frozenset(
            {
                "authorization_created",
                "build_succeeded",
                "claims",
                "classification",
                "context_receipt_sha256",
                "error",
                "execution_receipt_sha256",
                "executor_invoked",
                "failure_body_sha256",
                "image_id",
                "image_state_uncertain",
                "intent_sha256",
                "limitations",
                "phase",
                "plan_sha256",
                "schema_version",
                "status",
            }
        ),
        label="build failure receipt",
    )
    if (
        receipt["schema_version"] != CPU_OCI_BUILD_PUBLICATION_FAILURE_SCHEMA_VERSION
        or receipt["status"] != _FAILURE_STATUS
        or receipt["classification"] != _FAILURE_CLASSIFICATION
    ):
        _fail("failure receipt schema, status, or classification differs")
    for field in (
        "authorization_created",
        "build_succeeded",
        "executor_invoked",
        "image_state_uncertain",
    ):
        if type(receipt[field]) is not bool:
            _fail(f"failure receipt {field} is not an exact boolean")
    claims = _exact(receipt["claims"], frozenset(_claims()), label="failure claims")
    if claims != _claims() or any(claims.values()):
        _fail("failure receipt authority claim became true")
    if receipt["limitations"] != _limitations():
        _fail("failure receipt limitations differ")
    allowed_phases = {
        "pre_intent",
        "authorization_failed_pre_start",
        "executor_failed_pre_start",
        "executor_failed_uncertain",
        "success_publication_failed_after_build",
    }
    if receipt["phase"] not in allowed_phases:
        _fail("failure receipt phase differs")
    for field in (
        "context_receipt_sha256",
        "intent_sha256",
        "plan_sha256",
        "execution_receipt_sha256",
    ):
        if receipt[field] is not None:
            _require_sha256(receipt[field], label=f"failure receipt {field}")
    if receipt["image_id"] is not None:
        _require_image_id(receipt["image_id"], label="failure receipt image ID")
    error = _exact(
        receipt["error"],
        frozenset({"message_sha256", "message_size_bytes", "type"}),
        label="failure error",
    )
    _require_sha256(error["message_sha256"], label="failure message")
    _require_integer(
        error["message_size_bytes"],
        label="failure message size",
        maximum=_MAX_JSON_TEXT_BYTES,
    )
    if type(error["type"]) is not str or _SAFE_TYPE_RE.fullmatch(error["type"]) is None:
        _fail("failure error type differs")
    if receipt["build_succeeded"] is True and (
        receipt["phase"] != "success_publication_failed_after_build"
        or receipt["image_id"] is None
        or receipt["execution_receipt_sha256"] is None
        or receipt["image_state_uncertain"] is not True
    ):
        _fail("successful-build publication failure semantics differ")
    phase = cast(str, receipt["phase"])
    has_context = receipt["context_receipt_sha256"] is not None
    has_intent = receipt["intent_sha256"] is not None
    has_plan = receipt["plan_sha256"] is not None
    has_image = receipt["image_id"] is not None
    has_execution = receipt["execution_receipt_sha256"] is not None
    state = (
        has_context,
        has_intent,
        has_plan,
        receipt["authorization_created"],
        receipt["executor_invoked"],
        receipt["build_succeeded"],
        receipt["image_state_uncertain"],
        has_image,
        has_execution,
    )
    expected_states = {
        "pre_intent": (False, False, False, False, False, False, False, False, False),
        "authorization_failed_pre_start": (
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ),
        "executor_failed_pre_start": (
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ),
        "executor_failed_uncertain": (
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            False,
            False,
        ),
        "success_publication_failed_after_build": (
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
        ),
    }
    if state != expected_states[phase]:
        _fail("failure receipt phase/state matrix differs")
    supplied = _require_sha256(receipt["failure_body_sha256"], label="failure body")
    body = copy.deepcopy(receipt)
    body.pop("failure_body_sha256")
    if supplied != _sha256(_canonical_json(body)):
        _fail("failure receipt body digest differs")
    return copy.deepcopy(receipt)


def _publish_failure(
    root: _OpenRoot,
    *,
    phase: str,
    error: BaseException,
    context_receipt_sha256: str | None,
    intent_sha256: str | None,
    plan_sha256: str | None,
    authorization_created: bool,
    executor_invoked: bool,
    build_succeeded: bool,
    image_state_uncertain: bool,
    image_id: str | None = None,
    execution_receipt_sha256: str | None = None,
    execution_receipt_bytes: bytes | None = None,
    timeout_seconds: int | None = None,
    executor_module: Any | None = None,
) -> PublishedMatchedV3CpuOciBuildFailure:
    raw, digest = _build_failure_receipt(
        phase=phase,
        error=error,
        context_receipt_sha256=context_receipt_sha256,
        intent_sha256=intent_sha256,
        plan_sha256=plan_sha256,
        authorization_created=authorization_created,
        executor_invoked=executor_invoked,
        build_succeeded=build_succeeded,
        image_state_uncertain=image_state_uncertain,
        image_id=image_id,
        execution_receipt_sha256=execution_receipt_sha256,
    )
    files = {_FAILURE_FILENAME: raw}
    if phase == "success_publication_failed_after_build":
        if (
            type(execution_receipt_bytes) is not bytes
            or not execution_receipt_bytes
            or execution_receipt_sha256 is None
            or _sha256(execution_receipt_bytes) != execution_receipt_sha256
            or timeout_seconds is None
        ):
            _fail("post-build failure lacks its exact execution receipt bytes")
        loaded_executor_contract = (
            _load_executor_contract() if executor_module is None else executor_module
        )
        parsed_execution = (
            loaded_executor_contract.parse_matched_v3_cpu_oci_build_execution_receipt(
                execution_receipt_bytes,
                expected_receipt_sha256=execution_receipt_sha256,
            )
        )
        parsed_context = cast(Mapping[str, Any], parsed_execution["context"])
        parsed_build = cast(Mapping[str, Any], parsed_execution["build"])
        if (
            parsed_context["receipt_sha256"] != context_receipt_sha256
            or parsed_build["image_id"] != image_id
            or parsed_build["timeout_seconds"] != timeout_seconds
        ):
            _fail("post-build failure execution receipt identity differs")
        files[_EXECUTION_RECEIPT_FILENAME] = execution_receipt_bytes
    elif execution_receipt_bytes is not None or timeout_seconds is not None:
        _fail("pre-success failure cannot carry an execution receipt")
    destination = _publish_files(
        root,
        category="failures",
        address=digest,
        files=files,
    )
    return PublishedMatchedV3CpuOciBuildFailure(
        directory=destination,
        receipt_sha256=digest,
        phase=phase,
        image_state_uncertain=image_state_uncertain,
    )


def _validate_success_directory_fd(
    directory: int,
    *,
    expected_execution_receipt_sha256: str,
    intent: Mapping[str, Any],
    intent_files: Mapping[str, bytes],
    executor_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes, str]:
    loaded_executor_contract = (
        _load_executor_contract() if executor_module is None else executor_module
    )
    expected_execution = _require_sha256(
        expected_execution_receipt_sha256,
        label="expected successful execution receipt",
    )
    names = set(os.listdir(directory))
    if names != {_EXECUTION_RECEIPT_FILENAME, _PUBLICATION_RECEIPT_FILENAME}:
        _fail("success publication file set differs")
    for name in names:
        metadata = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= _MAX_JSON_BYTES:
            _fail("success publication file metadata differs")
    execution_stat = os.stat(
        _EXECUTION_RECEIPT_FILENAME,
        dir_fd=directory,
        follow_symlinks=False,
    )
    execution_raw = _read_file_at(
        directory,
        _EXECUTION_RECEIPT_FILENAME,
        expected_sha256=expected_execution,
        expected_size_bytes=execution_stat.st_size,
    )
    execution_receipt = loaded_executor_contract.parse_matched_v3_cpu_oci_build_execution_receipt(
        execution_raw,
        expected_receipt_sha256=expected_execution,
    )
    publication_raw = _read_unpinned_file_at(
        directory,
        _PUBLICATION_RECEIPT_FILENAME,
        maximum_size_bytes=_MAX_JSON_BYTES,
    )
    publication_sha = _sha256(publication_raw)
    publication_receipt = parse_matched_v3_cpu_oci_build_publication_receipt(
        publication_raw,
        expected_file_sha256=publication_sha,
    )
    intent_context = cast(Mapping[str, Any], intent["context"])
    execution_context = cast(Mapping[str, Any], execution_receipt["context"])
    execution_build = cast(Mapping[str, Any], execution_receipt["build"])
    publication_execution = cast(Mapping[str, Any], publication_receipt["execution"])
    canonical_context = execution_context["canonical_receipt"]
    if type(canonical_context) is not str:
        _fail("execution receipt embedded context is not exact text")
    try:
        canonical_context_bytes = canonical_context.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ForagerMatchedV3CpuOciBuildPublicationError(
            "execution receipt embedded context is not exact ASCII"
        ) from exc
    if (
        canonical_context_bytes != intent_files[_CONTEXT_RECEIPT_FILENAME]
        or execution_context["receipt_sha256"] != intent_context["receipt_sha256"]
        or execution_context["plan_sha256"] != intent_context["plan_sha256"]
        or execution_build["timeout_seconds"] != intent["timeout_seconds"]
        or publication_receipt["context_receipt_sha256"] != intent_context["receipt_sha256"]
        or publication_receipt["intent_sha256"] != _sha256(intent_files[_INTENT_FILENAME])
        or publication_execution["receipt_sha256"] != expected_execution
        or publication_execution["receipt_size_bytes"] != len(execution_raw)
        or publication_execution["timeout_seconds"] != intent["timeout_seconds"]
        or publication_execution["image_id"] != execution_build["image_id"]
    ):
        _fail("success publication differs from its exact durable intent")
    return publication_receipt, execution_receipt, publication_raw, publication_sha


def _publish_success_impl(
    root: _OpenRoot,
    *,
    intent: Mapping[str, Any],
    intent_files: Mapping[str, bytes],
    execution: executor_contract.CpuOciBuildExecutionArtifacts,
    executor_module: Any,
) -> PublishedMatchedV3CpuOciBuild:
    context_record = cast(Mapping[str, Any], intent["context"])
    context_sha = cast(str, context_record["receipt_sha256"])
    intent_sha = _sha256(intent_files[_INTENT_FILENAME])
    publication_raw, publication_sha = _build_success_receipt(
        context_receipt_sha256=context_sha,
        intent_sha256=intent_sha,
        execution=execution,
        timeout_seconds=cast(int, intent["timeout_seconds"]),
        executor_module=executor_module,
    )
    destination = _publish_files(
        root,
        category="successes",
        address=execution.receipt_sha256,
        files={
            _EXECUTION_RECEIPT_FILENAME: execution.receipt_bytes,
            _PUBLICATION_RECEIPT_FILENAME: publication_raw,
        },
    )
    category_fd, namespace_fd = _open_publication_namespace(root, "successes")
    directory = -1
    primary: BaseException | None = None
    try:
        directory = _open_directory_at(
            namespace_fd,
            execution.receipt_sha256,
            label="published successful build",
        )
        replayed, execution_receipt, replayed_raw, replayed_sha = _validate_success_directory_fd(
            directory,
            expected_execution_receipt_sha256=execution.receipt_sha256,
            intent=intent,
            intent_files=intent_files,
            executor_module=executor_module,
        )
        _verify_publication_namespace(root, "successes", category_fd, namespace_fd)
        _verify_addressed_directory(
            namespace_fd,
            execution.receipt_sha256,
            directory,
        )
    except BaseException as exc:
        primary = exc
        raise MatchedV3CpuOciBuildSuccessPublicationUncertainError(
            "successful build publication committed but final replay failed",
            context_receipt_sha256=context_sha,
            execution_receipt_sha256=execution.receipt_sha256,
            image_id=execution.image_id,
        ) from exc
    finally:
        _close_descriptors(
            (
                (directory, "published successful build directory"),
                (namespace_fd, "successful build sha256 namespace"),
                (category_fd, "successful build category"),
            ),
            primary=primary,
        )
    replayed_execution = cast(Mapping[str, Any], execution_receipt["build"])
    replayed_publication_execution = cast(Mapping[str, Any], replayed["execution"])
    if (
        replayed_raw != publication_raw
        or replayed_sha != publication_sha
        or replayed_execution["image_id"] != execution.image_id
        or replayed_publication_execution["receipt_sha256"] != execution.receipt_sha256
    ):
        raise MatchedV3CpuOciBuildSuccessPublicationUncertainError(
            "successful build publication replay identity differs",
            context_receipt_sha256=context_sha,
            execution_receipt_sha256=execution.receipt_sha256,
            image_id=execution.image_id,
        )
    return PublishedMatchedV3CpuOciBuild(
        intent_directory=(root.path / "intents" / "sha256" / context_sha),
        success_directory=destination,
        context_receipt_sha256=context_sha,
        execution_receipt_sha256=execution.receipt_sha256,
        publication_receipt_sha256=publication_sha,
        image_id=execution.image_id,
    )


def _publish_success(
    root: _OpenRoot,
    *,
    intent: Mapping[str, Any],
    intent_files: Mapping[str, bytes],
    execution: executor_contract.CpuOciBuildExecutionArtifacts,
    executor_module: Any,
) -> PublishedMatchedV3CpuOciBuild:
    context_sha = cast(str, cast(Mapping[str, Any], intent["context"])["receipt_sha256"])
    try:
        return _publish_success_impl(
            root,
            intent=intent,
            intent_files=intent_files,
            execution=execution,
            executor_module=executor_module,
        )
    except MatchedV3CpuOciBuildSuccessPublicationUncertainError:
        raise
    except BaseException as exc:
        raise MatchedV3CpuOciBuildSuccessPublicationUncertainError(
            "successful build exists but its durable publication failed",
            context_receipt_sha256=context_sha,
            execution_receipt_sha256=execution.receipt_sha256,
            image_id=execution.image_id,
        ) from exc


def _open_addressed_directory(
    root: _OpenRoot,
    *,
    category: str,
    address: str,
) -> tuple[int, int, int]:
    _require_sha256(address, label=f"expected {category} address")
    category_fd, namespace_fd = _open_publication_namespace(root, category)
    directory_fd = -1
    try:
        directory_fd = _open_directory_at(
            namespace_fd,
            address,
            label=f"published {category} address",
        )
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        _verify_addressed_directory(namespace_fd, address, directory_fd)
    except BaseException as primary:
        _close_descriptors(
            (
                (directory_fd, "publication addressed directory"),
                (namespace_fd, "publication sha256 namespace"),
                (category_fd, "publication category"),
            ),
            primary=primary,
        )
        raise
    return category_fd, namespace_fd, directory_fd


def _verify_addressed_directory(namespace: int, address: str, directory: int) -> None:
    _require_sha256(address, label="published directory address")
    opened = os.fstat(directory)
    located = os.stat(address, dir_fd=namespace, follow_symlinks=False)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or _directory_locator_identity(opened) != _directory_locator_identity(located)
        or opened.st_uid != os.geteuid()
        or stat.S_IMODE(opened.st_mode) != 0o500
        or os.get_inheritable(directory)
    ):
        _fail("published addressed directory locator changed")


@contextmanager
def _retain_addressed_directory(
    root: _OpenRoot,
    *,
    category: str,
    address: str,
) -> Iterator[int]:
    category_fd, namespace_fd, directory_fd = _open_addressed_directory(
        root,
        category=category,
        address=address,
    )
    primary: BaseException | None = None
    try:
        yield directory_fd
        _verify_publication_namespace(root, category, category_fd, namespace_fd)
        _verify_addressed_directory(namespace_fd, address, directory_fd)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _close_descriptors(
            (
                (directory_fd, f"retained {category} address directory"),
                (namespace_fd, f"retained {category} sha256 namespace"),
                (category_fd, f"retained {category} category"),
            ),
            primary=primary,
        )


def _validate_exact_expected_intent_directory(
    directory: int,
    *,
    expected_context_receipt_sha256: str | None,
    expected_intent_sha256: str | None,
    expected_plan_sha256: str | None,
    expected_intent_bytes: bytes | None,
) -> None:
    context_sha = _require_sha256(
        expected_context_receipt_sha256,
        label="canonical intent context receipt",
    )
    intent_sha = _require_sha256(
        expected_intent_sha256,
        label="canonical intent manifest",
    )
    plan_sha = _require_sha256(
        expected_plan_sha256,
        label="canonical intent plan",
    )
    if (
        type(expected_intent_bytes) is not bytes
        or not expected_intent_bytes
        or _sha256(expected_intent_bytes) != intent_sha
    ):
        _fail("canonical expected intent bytes differ from their exact pin")
    expected_intent = parse_matched_v3_cpu_oci_build_intent(
        expected_intent_bytes,
        expected_file_sha256=intent_sha,
    )
    expected_context = cast(Mapping[str, Any], expected_intent["context"])
    expected_plan = cast(Mapping[str, Any], expected_intent["plan"])
    if (
        expected_context["receipt_sha256"] != context_sha
        or expected_context["plan_sha256"] != plan_sha
        or expected_plan["sha256"] != plan_sha
    ):
        _fail("canonical expected intent differs from its context or plan pins")
    observed_intent, observed_files = _validate_intent_directory_fd(
        directory,
        expected_context_receipt_sha256=context_sha,
    )
    try:
        expected_files = _intent_expected_files(expected_intent, expected_intent_bytes)
        if (
            observed_intent != expected_intent
            or observed_files.get(_INTENT_FILENAME) != expected_intent_bytes
            or set(observed_files) != set(expected_files)
        ):
            _fail("canonical intent or closure differs from the exact expected intent")
        for name, (expected_sha, expected_size) in expected_files.items():
            raw = observed_files[name]
            if len(raw) != expected_size or not hmac.compare_digest(
                _sha256(raw),
                expected_sha,
            ):
                _fail(f"canonical intent closure file differs: {name}")
    finally:
        observed_files.clear()


@contextmanager
def _retain_exact_canonical_intent(
    root: _OpenRoot,
    *,
    expected_context_receipt_sha256: str | None,
    expected_intent_sha256: str | None,
    expected_plan_sha256: str | None,
    expected_intent_bytes: bytes | None,
) -> Iterator[None]:
    """Retain and semantically replay the exact canonical intent around one operation."""

    try:
        context_sha = _require_sha256(
            expected_context_receipt_sha256,
            label="retained canonical intent context",
        )
        category_fd, namespace_fd, directory_fd = _open_addressed_directory(
            root,
            category="intents",
            address=context_sha,
        )
    except BaseException as exc:
        raise _CanonicalIntentUnavailableError(
            "canonical intent route cannot be freshly retained"
        ) from exc
    primary: BaseException | None = None
    try:
        try:
            _validate_exact_expected_intent_directory(
                directory_fd,
                expected_context_receipt_sha256=context_sha,
                expected_intent_sha256=expected_intent_sha256,
                expected_plan_sha256=expected_plan_sha256,
                expected_intent_bytes=expected_intent_bytes,
            )
        except BaseException as exc:
            failure = _CanonicalIntentUnavailableError(
                "canonical intent failed its fresh exact semantic replay"
            )
            primary = failure
            raise failure from exc

        body_error: BaseException | None = None
        try:
            yield
        except BaseException as exc:
            body_error = exc

        try:
            _validate_exact_expected_intent_directory(
                directory_fd,
                expected_context_receipt_sha256=context_sha,
                expected_intent_sha256=expected_intent_sha256,
                expected_plan_sha256=expected_plan_sha256,
                expected_intent_bytes=expected_intent_bytes,
            )
            _verify_publication_namespace(
                root,
                "intents",
                category_fd,
                namespace_fd,
            )
            _verify_addressed_directory(namespace_fd, context_sha, directory_fd)
        except BaseException as exc:
            failure = _CanonicalIntentUnavailableError(
                "canonical intent changed while retained across failure publication"
            )
            if body_error is not None:
                failure.add_note(
                    "linked failure publication also failed while canonical intent was retained: "
                    f"{type(body_error).__name__}: {body_error}"
                )
            primary = failure
            raise failure from exc
        if body_error is not None:
            primary = body_error
            raise body_error
    except BaseException as exc:
        if primary is None:
            primary = exc
        raise
    finally:
        try:
            _close_descriptors(
                (
                    (directory_fd, "exact canonical intent address"),
                    (namespace_fd, "exact canonical intent sha256 namespace"),
                    (category_fd, "exact canonical intent category"),
                ),
                primary=primary,
            )
        except BaseException as exc:
            raise _CanonicalIntentUnavailableError(
                "canonical intent descriptor retention cleanup failed"
            ) from exc


def _publish_failure_with_retained_canonical_intent(
    root: _OpenRoot,
    *,
    state: _BuildAttemptState,
    phase: str,
    error: BaseException,
    authorization_created: bool,
    executor_invoked: bool,
    build_succeeded: bool,
    image_state_uncertain: bool,
    image_id: str | None,
    execution_receipt_sha256: str | None,
    execution_receipt_bytes: bytes | None,
    timeout_seconds: int | None,
) -> PublishedMatchedV3CpuOciBuildFailure:
    with _retain_exact_canonical_intent(
        root,
        expected_context_receipt_sha256=state.context_receipt_sha256,
        expected_intent_sha256=state.intent_sha256,
        expected_plan_sha256=state.plan_sha256,
        expected_intent_bytes=state.intent_bytes,
    ):
        return _publish_failure(
            root,
            phase=phase,
            error=error,
            context_receipt_sha256=state.context_receipt_sha256,
            intent_sha256=state.intent_sha256,
            plan_sha256=state.plan_sha256,
            authorization_created=authorization_created,
            executor_invoked=executor_invoked,
            build_succeeded=build_succeeded,
            image_state_uncertain=image_state_uncertain,
            image_id=image_id,
            execution_receipt_sha256=execution_receipt_sha256,
            execution_receipt_bytes=execution_receipt_bytes,
            timeout_seconds=timeout_seconds,
            executor_module=(None if execution_receipt_bytes is None else state.executor_module),
        )


def validate_published_matched_v3_cpu_oci_build(
    publication_root: Path,
    *,
    artifact_root: Path,
    expected_context_receipt_sha256: str,
    expected_execution_receipt_sha256: str,
) -> PublishedMatchedV3CpuOciBuild:
    """Fresh-process replay of one exact intent, execution receipt, and context."""

    context_sha = _require_sha256(
        expected_context_receipt_sha256,
        label="expected published context receipt",
    )
    execution_sha = _require_sha256(
        expected_execution_receipt_sha256,
        label="expected published execution receipt",
    )
    concrete_path_type = type(Path())
    if type(artifact_root) is not concrete_path_type or not artifact_root.is_absolute():
        _fail("artifact root must be one exact absolute pathlib.Path")
    with _open_root(publication_root, label="build publication root", mutable=True) as root:
        with _retain_addressed_directory(
            root,
            category="intents",
            address=context_sha,
        ) as intent_directory:
            intent, intent_files = _validate_intent_directory_fd(
                intent_directory,
                expected_context_receipt_sha256=context_sha,
            )
        with _retain_addressed_directory(
            root,
            category="successes",
            address=execution_sha,
        ) as success_directory:
            publication, execution, publication_raw, publication_sha = (
                _validate_success_directory_fd(
                    success_directory,
                    expected_execution_receipt_sha256=execution_sha,
                    intent=intent,
                    intent_files=intent_files,
                )
            )

        production = _load_production_inputs(artifact_root)
        local_record = cast(Mapping[str, Any], intent["local_source"])
        local_receipt = local_bundle_contract.parse_matched_v3_local_source_bundle_receipt(
            intent_files[_LOCAL_RECEIPT_FILENAME],
            expected_receipt_sha256=cast(str, local_record["bundle_receipt_sha256"]),
        )
        local_archive_record = cast(Mapping[str, Any], local_receipt["archive"])
        rebuilt_plan = plan_contract.build_matched_v3_cpu_oci_build_plan(
            issuance_artifacts=production.issuance_artifacts,
            expected_root_pin_inventory_sha256=PRODUCTION_ROOT_PIN_INVENTORY_SHA256,
            expected_selected_wheel_inventory_sha256=(PRODUCTION_SELECTED_WHEEL_INVENTORY_SHA256),
            expected_resolution_lock_sha256=PRODUCTION_RESOLUTION_LOCK_SHA256,
            expected_resolution_lock_size_bytes=PRODUCTION_RESOLUTION_LOCK_SIZE_BYTES,
            wheelhouse_archive_bytes=production.wheelhouse_archive_bytes,
            external_foragax_source=plan_contract.CanonicalSourceBundleInput(
                archive_bytes=production.external_source_archive_bytes,
                expected_archive_sha256=PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
                expected_archive_size_bytes=PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES,
                expected_member_count=PRODUCTION_EXTERNAL_MEMBER_COUNT,
                receipt_bytes=production.external_source_receipt_bytes,
                expected_receipt_sha256=PRODUCTION_EXTERNAL_RECEIPT_SHA256,
                source_manifest_sha256=PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256,
                source_tree_sha256=PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256,
                staging_manifest_sha256=PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256,
            ),
            local_alberta_source=plan_contract.CanonicalSourceBundleInput(
                archive_bytes=intent_files[_LOCAL_ARCHIVE_FILENAME],
                expected_archive_sha256=cast(str, local_record["archive_sha256"]),
                expected_archive_size_bytes=cast(int, local_record["archive_size_bytes"]),
                expected_member_count=cast(int, local_archive_record["member_count"]),
                receipt_bytes=intent_files[_LOCAL_RECEIPT_FILENAME],
                expected_receipt_sha256=cast(str, local_record["bundle_receipt_sha256"]),
                source_manifest_sha256=cast(str, local_record["snapshot_manifest_sha256"]),
                source_tree_sha256=cast(str, local_record["snapshot_tree_sha256"]),
                staging_manifest_sha256=None,
            ),
        )
        if (
            rebuilt_plan.plan_bytes != intent_files[_PLAN_FILENAME]
            or rebuilt_plan.plan_sha256 != cast(Mapping[str, Any], intent["plan"])["sha256"]
        ):
            _fail("freshly rebuilt OCI plan differs from the publication")
        with context_contract.retain_matched_v3_cpu_oci_build_context(
            plan_bytes=intent_files[_PLAN_FILENAME],
            expected_plan_sha256=cast(str, cast(Mapping[str, Any], intent["plan"])["sha256"]),
            wheelhouse_archive_bytes=production.wheelhouse_archive_bytes,
            external_foragax_source_archive_bytes=production.external_source_archive_bytes,
            local_alberta_source_archive_bytes=intent_files[_LOCAL_ARCHIVE_FILENAME],
        ) as rebuilt_context:
            if (
                rebuilt_context.receipt_sha256 != context_sha
                or rebuilt_context.receipt_bytes != intent_files[_CONTEXT_RECEIPT_FILENAME]
                or rebuilt_context.archive_sha256
                != cast(Mapping[str, Any], intent["context"])["archive_sha256"]
            ):
                _fail("freshly reconstructed sealed context differs from the publication")
        execution_build = cast(Mapping[str, Any], execution["build"])
        publication_execution = cast(Mapping[str, Any], publication["execution"])
        image_id = cast(str, execution_build["image_id"])
        if publication_execution["image_id"] != image_id:
            _fail("published image identity differs during fresh replay")
        with _retain_addressed_directory(
            root,
            category="intents",
            address=context_sha,
        ) as final_intent_directory:
            final_intent, final_intent_files = _validate_intent_directory_fd(
                final_intent_directory,
                expected_context_receipt_sha256=context_sha,
            )
            if final_intent != intent or final_intent_files != intent_files:
                _fail("intent changed before final validator return")
            with _retain_addressed_directory(
                root,
                category="successes",
                address=execution_sha,
            ) as final_success_directory:
                (
                    final_publication,
                    final_execution,
                    final_publication_raw,
                    final_publication_sha,
                ) = _validate_success_directory_fd(
                    final_success_directory,
                    expected_execution_receipt_sha256=execution_sha,
                    intent=final_intent,
                    intent_files=final_intent_files,
                )
                if (
                    final_publication != publication
                    or final_execution != execution
                    or final_publication_raw != publication_raw
                    or final_publication_sha != publication_sha
                ):
                    _fail("success publication changed before final validator return")
                return PublishedMatchedV3CpuOciBuild(
                    intent_directory=root.path / "intents" / "sha256" / context_sha,
                    success_directory=root.path / "successes" / "sha256" / execution_sha,
                    context_receipt_sha256=context_sha,
                    execution_receipt_sha256=execution_sha,
                    publication_receipt_sha256=publication_sha,
                    image_id=image_id,
                )


def validate_published_matched_v3_cpu_oci_build_failure(
    publication_root: Path,
    *,
    expected_failure_receipt_sha256: str,
) -> PublishedMatchedV3CpuOciBuildFailure:
    """Replay one exact content-addressed bounded failure receipt."""

    expected = _require_sha256(
        expected_failure_receipt_sha256,
        label="expected published failure receipt",
    )
    with _open_root(publication_root, label="build publication root", mutable=True) as root:
        category, namespace, directory = _open_addressed_directory(
            root,
            category="failures",
            address=expected,
        )
        primary: BaseException | None = None
        try:
            names = set(os.listdir(directory))
            if _FAILURE_FILENAME not in names or not names <= {
                _FAILURE_FILENAME,
                _EXECUTION_RECEIPT_FILENAME,
            }:
                _fail("failure publication file set differs")
            raw = _read_unpinned_file_at(
                directory,
                _FAILURE_FILENAME,
                maximum_size_bytes=_MAX_JSON_BYTES,
            )
            if _sha256(raw) != expected:
                _fail("failure publication is not addressed by its receipt")
            receipt = parse_matched_v3_cpu_oci_build_failure_receipt(
                raw,
                expected_file_sha256=expected,
            )
            post_build = receipt["phase"] == "success_publication_failed_after_build"
            expected_names = {_FAILURE_FILENAME}
            execution_bytes: bytes | None = None
            parsed_execution: Mapping[str, Any] | None = None
            if post_build:
                expected_names.add(_EXECUTION_RECEIPT_FILENAME)
                execution_bytes = _read_unpinned_file_at(
                    directory,
                    _EXECUTION_RECEIPT_FILENAME,
                    maximum_size_bytes=_MAX_JSON_BYTES,
                )
                if _sha256(execution_bytes) != receipt["execution_receipt_sha256"]:
                    _fail("failure publication execution receipt address differs")
                parsed_execution = (
                    _load_executor_contract().parse_matched_v3_cpu_oci_build_execution_receipt(
                        execution_bytes,
                        expected_receipt_sha256=cast(
                            str,
                            receipt["execution_receipt_sha256"],
                        ),
                    )
                )
            if names != expected_names:
                _fail("failure publication phase-dependent file set differs")
            context_sha = receipt["context_receipt_sha256"]
            if context_sha is not None:
                intent_category, intent_namespace, intent_directory = _open_addressed_directory(
                    root,
                    category="intents",
                    address=cast(str, context_sha),
                )
                intent_primary: BaseException | None = None
                try:
                    intent, intent_files = _validate_intent_directory_fd(
                        intent_directory,
                        expected_context_receipt_sha256=cast(str, context_sha),
                    )
                    intent_plan = cast(Mapping[str, Any], intent["plan"])
                    if (
                        _sha256(intent_files[_INTENT_FILENAME]) != receipt["intent_sha256"]
                        or intent_plan["sha256"] != receipt["plan_sha256"]
                    ):
                        _fail("failure receipt differs from its durable intent")
                    if parsed_execution is not None:
                        execution_context = cast(
                            Mapping[str, Any],
                            parsed_execution["context"],
                        )
                        execution_build = cast(
                            Mapping[str, Any],
                            parsed_execution["build"],
                        )
                        canonical_context = execution_context["canonical_receipt"]
                        if type(canonical_context) is not str:
                            _fail("failure execution context is not exact text")
                        try:
                            canonical_context_bytes = canonical_context.encode("ascii")
                        except UnicodeEncodeError as exc:
                            raise ForagerMatchedV3CpuOciBuildPublicationError(
                                "failure execution context is not exact ASCII"
                            ) from exc
                        if (
                            canonical_context_bytes != intent_files[_CONTEXT_RECEIPT_FILENAME]
                            or execution_context["receipt_sha256"] != context_sha
                            or execution_build["image_id"] != receipt["image_id"]
                            or execution_build["timeout_seconds"] != intent["timeout_seconds"]
                        ):
                            _fail("failure execution receipt differs from its durable intent")
                    _verify_publication_namespace(
                        root,
                        "intents",
                        intent_category,
                        intent_namespace,
                    )
                    _verify_addressed_directory(
                        intent_namespace,
                        cast(str, context_sha),
                        intent_directory,
                    )
                except BaseException as exc:
                    intent_primary = exc
                    raise
                finally:
                    _close_descriptors(
                        (
                            (intent_directory, "failure-linked intent directory"),
                            (intent_namespace, "failure-linked intent sha256 namespace"),
                            (intent_category, "failure-linked intent category"),
                        ),
                        primary=intent_primary,
                    )
            _verify_publication_namespace(root, "failures", category, namespace)
            _verify_addressed_directory(namespace, expected, directory)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            _close_descriptors(
                (
                    (directory, "published failure directory"),
                    (namespace, "failure sha256 namespace"),
                    (category, "failure category"),
                ),
                primary=primary,
            )
        return PublishedMatchedV3CpuOciBuildFailure(
            directory=root.path / "failures" / "sha256" / expected,
            receipt_sha256=expected,
            phase=cast(str, receipt["phase"]),
            image_state_uncertain=cast(bool, receipt["image_state_uncertain"]),
        )


def _execute_and_publish_matched_v3_cpu_oci_build(
    request: MatchedV3CpuOciBuildPublicationRequest,
    state: _BuildAttemptState,
) -> PublishedMatchedV3CpuOciBuild:
    """Execute at most one build after committing its deterministic durable intent."""

    if type(request) is not MatchedV3CpuOciBuildPublicationRequest:
        raise TypeError("build publication requires the exact request type")
    context_sha: str | None = None
    plan_sha: str | None = None
    intent_sha: str | None = None
    authorization_created = False
    executor_invoked = False
    execution: executor_contract.CpuOciBuildExecutionArtifacts | None = None
    intent_commit_state = _IntentCommitState()
    with _open_root(
        request.publication_root,
        label="build publication root",
        mutable=True,
    ) as publication_root:
        _prepare_layout(publication_root)
        try:
            production = _load_production_inputs(request.artifact_root)
            local_snapshot_contract.verify_matched_v3_local_source_snapshot(
                repository_root=request.repository_root,
                expected_canonical_manifest_bytes=request.expected_snapshot_manifest_bytes,
                expected_full_sha256=request.expected_snapshot_manifest_sha256,
            )
            with local_bundle_contract.retain_matched_v3_local_source_bundle(
                repository_root=request.repository_root,
                expected_canonical_snapshot_manifest_bytes=(
                    request.expected_snapshot_manifest_bytes
                ),
                expected_snapshot_manifest_sha256=(request.expected_snapshot_manifest_sha256),
                expected_snapshot_tree_sha256=request.expected_snapshot_tree_sha256,
            ) as local_bundle:
                local_archive_bytes = local_bundle.read_archive_bytes()
                local_receipt_bytes = local_bundle.receipt_bytes
                local_receipt_sha = local_bundle.receipt_sha256
                local_receipt = local_bundle.receipt()
                local_archive_record = cast(Mapping[str, Any], local_receipt["archive"])
                external_source = plan_contract.CanonicalSourceBundleInput(
                    archive_bytes=production.external_source_archive_bytes,
                    expected_archive_sha256=PRODUCTION_EXTERNAL_ARCHIVE_SHA256,
                    expected_archive_size_bytes=PRODUCTION_EXTERNAL_ARCHIVE_SIZE_BYTES,
                    expected_member_count=PRODUCTION_EXTERNAL_MEMBER_COUNT,
                    receipt_bytes=production.external_source_receipt_bytes,
                    expected_receipt_sha256=PRODUCTION_EXTERNAL_RECEIPT_SHA256,
                    source_manifest_sha256=PRODUCTION_EXTERNAL_SOURCE_MANIFEST_SHA256,
                    source_tree_sha256=PRODUCTION_EXTERNAL_SOURCE_TREE_SHA256,
                    staging_manifest_sha256=PRODUCTION_EXTERNAL_STAGING_MANIFEST_SHA256,
                )
                local_source = plan_contract.CanonicalSourceBundleInput(
                    archive_bytes=local_archive_bytes,
                    expected_archive_sha256=local_bundle.archive_sha256,
                    expected_archive_size_bytes=local_bundle.archive_size_bytes,
                    expected_member_count=cast(int, local_archive_record["member_count"]),
                    receipt_bytes=local_receipt_bytes,
                    expected_receipt_sha256=local_receipt_sha,
                    source_manifest_sha256=request.expected_snapshot_manifest_sha256,
                    source_tree_sha256=request.expected_snapshot_tree_sha256,
                    staging_manifest_sha256=None,
                )
                plan_artifacts = plan_contract.build_matched_v3_cpu_oci_build_plan(
                    issuance_artifacts=production.issuance_artifacts,
                    expected_root_pin_inventory_sha256=(PRODUCTION_ROOT_PIN_INVENTORY_SHA256),
                    expected_selected_wheel_inventory_sha256=(
                        PRODUCTION_SELECTED_WHEEL_INVENTORY_SHA256
                    ),
                    expected_resolution_lock_sha256=PRODUCTION_RESOLUTION_LOCK_SHA256,
                    expected_resolution_lock_size_bytes=(PRODUCTION_RESOLUTION_LOCK_SIZE_BYTES),
                    wheelhouse_archive_bytes=production.wheelhouse_archive_bytes,
                    external_foragax_source=external_source,
                    local_alberta_source=local_source,
                )
                plan_sha = plan_artifacts.plan_sha256
                state.plan_sha256 = plan_sha
                with context_contract.retain_matched_v3_cpu_oci_build_context(
                    plan_bytes=plan_artifacts.plan_bytes,
                    expected_plan_sha256=plan_artifacts.plan_sha256,
                    wheelhouse_archive_bytes=production.wheelhouse_archive_bytes,
                    external_foragax_source_archive_bytes=(
                        production.external_source_archive_bytes
                    ),
                    local_alberta_source_archive_bytes=local_archive_bytes,
                ) as context_capability:
                    context_sha = context_capability.receipt_sha256
                    state.context_receipt_sha256 = context_sha
                    executor_contract = _load_executor_contract()
                    state.executor_module = executor_contract
                    if (
                        executor_contract.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT
                        != _EXECUTION_ACKNOWLEDGEMENT
                    ):
                        _fail("executor acknowledgement contract differs")
                    local_snapshot_contract.verify_matched_v3_local_source_snapshot(
                        repository_root=request.repository_root,
                        expected_canonical_manifest_bytes=(
                            request.expected_snapshot_manifest_bytes
                        ),
                        expected_full_sha256=request.expected_snapshot_manifest_sha256,
                    )
                    intent_raw, closure = _build_intent_bytes(
                        request=request,
                        plan_artifacts=plan_artifacts,
                        context_capability=context_capability,
                        local_archive_bytes=local_archive_bytes,
                        local_receipt_bytes=local_receipt_bytes,
                        local_receipt_sha256=local_receipt_sha,
                    )
                    intent_sha = _sha256(intent_raw)
                    state.intent_sha256 = intent_sha
                    state.intent_bytes = intent_raw
                    intent = parse_matched_v3_cpu_oci_build_intent(
                        intent_raw,
                        expected_file_sha256=intent_sha,
                    )
                    _publish_intent(
                        publication_root,
                        intent_raw=intent_raw,
                        closure=closure,
                        expected_context_receipt_sha256=context_sha,
                        commit_state=intent_commit_state,
                    )
                    with _retain_addressed_directory(
                        publication_root,
                        category="intents",
                        address=context_sha,
                    ) as retained_intent_directory:
                        retained_intent, retained_intent_files = _validate_intent_directory_fd(
                            retained_intent_directory,
                            expected_context_receipt_sha256=context_sha,
                        )
                        if retained_intent != intent or retained_intent_files != {
                            **closure,
                            _INTENT_FILENAME: intent_raw,
                        }:
                            _fail("retained durable intent differs before authorization")
                        retained_intent_files.clear()
                        terminal_intent_files = {
                            _CONTEXT_RECEIPT_FILENAME: closure[_CONTEXT_RECEIPT_FILENAME],
                            _INTENT_FILENAME: intent_raw,
                        }
                        closure.clear()
                        local_archive_bytes = b""
                        local_bundle.close()
                        del local_source
                        del external_source
                        del production
                        authorization = executor_contract.authorize_matched_v3_cpu_oci_build(
                            context_capability=context_capability,
                            exact_acknowledgement=request.exact_acknowledgement,
                        )
                        authorization_created = True
                        executor_invoked = True
                        execution = executor_contract.execute_matched_v3_cpu_oci_build(
                            context_capability=context_capability,
                            authorization=authorization,
                            timeout_seconds=request.timeout_seconds,
                        )
                        state.execution = execution
                        post_execution_intent, post_execution_files = _validate_intent_directory_fd(
                            retained_intent_directory,
                            expected_context_receipt_sha256=context_sha,
                        )
                        if (
                            post_execution_intent != intent
                            or post_execution_files[_INTENT_FILENAME] != intent_raw
                            or post_execution_files[_CONTEXT_RECEIPT_FILENAME]
                            != terminal_intent_files[_CONTEXT_RECEIPT_FILENAME]
                        ):
                            _fail("retained durable intent differs after execution")
                        post_execution_files.clear()
                        return _publish_success(
                            publication_root,
                            intent=intent,
                            intent_files=terminal_intent_files,
                            execution=execution,
                            executor_module=executor_contract,
                        )
        except MatchedV3CpuOciBuildIntentExistsError:
            # The existing deterministic intent is the durable no-retry fence.
            raise
        except BaseException as exc:
            if execution is not None:
                phase = "success_publication_failed_after_build"
                uncertain = True
                build_succeeded = True
            elif executor_invoked:
                exact_uncertainty = getattr(exc, "image_state_uncertain", None)
                uncertain = exact_uncertainty if type(exact_uncertainty) is bool else True
                phase = "executor_failed_uncertain" if uncertain else "executor_failed_pre_start"
                build_succeeded = False
            elif intent_commit_state.committed:
                phase = "authorization_failed_pre_start"
                uncertain = False
                build_succeeded = False
            elif (
                context_sha is not None
                and plan_sha is not None
                and intent_sha is not None
                and isinstance(
                    exc,
                    MatchedV3CpuOciBuildPublicationStateUncertainError,
                )
            ):
                phase = "intent_publication_uncertain_pre_start"
                uncertain = False
                build_succeeded = False
            else:
                phase = "pre_intent"
                uncertain = False
                build_succeeded = False
            if phase == "intent_publication_uncertain_pre_start":
                _add_note_once(exc, f"matched-v3 OCI failure phase: {phase}")
                _add_note_once(exc, "matched-v3 OCI image state uncertain: false")
                _add_note_once(exc, _INDETERMINATE_INTENT_DEFERRED_NOTE)
                state.failure_publication_attempted = True
                raise
            post_intent = phase != "pre_intent"
            if post_intent:
                try:
                    state.failure_publication_attempted = True
                    failure = _publish_failure_with_retained_canonical_intent(
                        publication_root,
                        state=state,
                        phase=phase,
                        error=exc,
                        authorization_created=authorization_created,
                        executor_invoked=executor_invoked,
                        build_succeeded=build_succeeded,
                        image_state_uncertain=uncertain,
                        image_id=None if execution is None else execution.image_id,
                        execution_receipt_sha256=(
                            None if execution is None else execution.receipt_sha256
                        ),
                        execution_receipt_bytes=(
                            None if execution is None else execution.receipt_bytes
                        ),
                        timeout_seconds=(None if execution is None else request.timeout_seconds),
                    )
                except _CanonicalIntentUnavailableError:
                    if not authorization_created and not executor_invoked and execution is None:
                        phase = "intent_publication_uncertain_pre_start"
                        uncertain = False
                    _add_note_once(exc, f"matched-v3 OCI failure phase: {phase}")
                    _add_note_once(
                        exc,
                        f"matched-v3 OCI image state uncertain: {'true' if uncertain else 'false'}",
                    )
                    _add_note_once(exc, _INDETERMINATE_INTENT_DEFERRED_NOTE)
                except BaseException as publication_error:
                    _add_note_once(exc, f"matched-v3 OCI failure phase: {phase}")
                    _add_note_once(
                        exc,
                        f"matched-v3 OCI image state uncertain: {'true' if uncertain else 'false'}",
                    )
                    exc.add_note(
                        "failure-receipt publication also failed: "
                        f"{type(publication_error).__name__}: {publication_error}"
                    )
                else:
                    _add_note_once(exc, f"matched-v3 OCI failure phase: {phase}")
                    _add_note_once(
                        exc,
                        f"matched-v3 OCI image state uncertain: {'true' if uncertain else 'false'}",
                    )
                    exc.add_note(
                        f"durable matched-v3 OCI failure receipt: sha256:{failure.receipt_sha256}"
                    )
            else:
                _add_note_once(exc, f"matched-v3 OCI failure phase: {phase}")
                _add_note_once(exc, "matched-v3 OCI image state uncertain: false")
                try:
                    state.failure_publication_attempted = True
                    failure = _publish_failure(
                        publication_root,
                        phase=phase,
                        error=exc,
                        context_receipt_sha256=None,
                        intent_sha256=None,
                        plan_sha256=None,
                        authorization_created=False,
                        executor_invoked=False,
                        build_succeeded=False,
                        image_state_uncertain=False,
                    )
                    exc.add_note(
                        f"durable matched-v3 OCI failure receipt: sha256:{failure.receipt_sha256}"
                    )
                except BaseException as publication_error:
                    exc.add_note(
                        "failure-receipt publication also failed: "
                        f"{type(publication_error).__name__}: {publication_error}"
                    )
            raise


def execute_and_publish_matched_v3_cpu_oci_build(
    request: MatchedV3CpuOciBuildPublicationRequest,
) -> PublishedMatchedV3CpuOciBuild:
    """Execute once and normalize every post-success escape to rich uncertainty."""

    state = _BuildAttemptState()
    try:
        return _execute_and_publish_matched_v3_cpu_oci_build(request, state)
    except MatchedV3CpuOciBuildSuccessPublicationUncertainError:
        raise
    except BaseException as exc:
        execution = state.execution
        if execution is None:
            raise
        _add_note_once(
            exc,
            "matched-v3 OCI failure phase: success_publication_failed_after_build",
        )
        _add_note_once(exc, "matched-v3 OCI image state uncertain: true")
        if not state.failure_publication_attempted:
            state.failure_publication_attempted = True
            root_opened = False
            try:
                with _open_root(
                    request.publication_root,
                    label="build publication root after successful build",
                    mutable=True,
                ) as publication_root:
                    root_opened = True
                    failure = _publish_failure_with_retained_canonical_intent(
                        publication_root,
                        state=state,
                        phase="success_publication_failed_after_build",
                        error=exc,
                        authorization_created=True,
                        executor_invoked=True,
                        build_succeeded=True,
                        image_state_uncertain=True,
                        image_id=execution.image_id,
                        execution_receipt_sha256=execution.receipt_sha256,
                        execution_receipt_bytes=execution.receipt_bytes,
                        timeout_seconds=request.timeout_seconds,
                    )
                exc.add_note(
                    f"durable matched-v3 OCI failure receipt: sha256:{failure.receipt_sha256}"
                )
            except _CanonicalIntentUnavailableError:
                _add_note_once(exc, _INDETERMINATE_INTENT_DEFERRED_NOTE)
            except BaseException as publication_error:
                if not root_opened:
                    _add_note_once(exc, _INDETERMINATE_INTENT_DEFERRED_NOTE)
                else:
                    exc.add_note(
                        "failure-receipt publication after root-exit failure also failed: "
                        f"{type(publication_error).__name__}: {publication_error}"
                    )
        rich = MatchedV3CpuOciBuildSuccessPublicationUncertainError(
            "a successful local OCI build escaped its complete publication boundary",
            context_receipt_sha256=cast(str, state.context_receipt_sha256),
            execution_receipt_sha256=execution.receipt_sha256,
            image_id=execution.image_id,
        )
        for note in getattr(exc, "__notes__", ()):
            rich.add_note(note)
        raise rich from exc


def _cli_path(value: str) -> Path:
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path("/")
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise argparse.ArgumentTypeError(
            "path must be absolute, non-root, and contain no dot segments"
        )
    return path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alberta-forager-matched-v3-cpu-oci-build",
        description=(
            "Prepare explicit local-source pins or consume preexisting pins for one "
            "durably journaled matched-v3 CPU OCI build."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    measure = subparsers.add_parser(
        "measure",
        help="write a new-only nonauthorizing local-source snapshot manifest",
    )
    measure.add_argument("--repository-root", required=True, type=_cli_path)
    measure.add_argument("--snapshot-manifest-output", required=True, type=_cli_path)

    execute = subparsers.add_parser(
        "execute",
        help="execute once from preexisting exact snapshot bytes and caller-carried pins",
    )
    execute.add_argument("--repository-root", required=True, type=_cli_path)
    execute.add_argument("--artifact-root", required=True, type=_cli_path)
    execute.add_argument("--publication-root", required=True, type=_cli_path)
    execute.add_argument("--snapshot-manifest", required=True, type=_cli_path)
    execute.add_argument("--snapshot-manifest-sha256", required=True)
    execute.add_argument("--snapshot-tree-sha256", required=True)
    execute.add_argument(
        "--exact-acknowledgement",
        required=True,
        help="the executor's exact one-build acknowledgement text",
    )
    execute.add_argument("--timeout-seconds", type=int, default=7200)
    return parser


def _bounded_error_text(error: BaseException) -> str:
    raw = str(error).encode("utf-8", errors="replace")[:8192]
    return raw.decode("utf-8", errors="replace")


def _cli_error_record(error: BaseException) -> dict[str, Any]:
    durable_failure: str | None = None
    classified_phase: str | None = None
    classified_uncertainty: bool | None = None
    for note in getattr(error, "__notes__", ()):
        if type(note) is not str:
            continue
        match = _DURABLE_FAILURE_NOTE_RE.fullmatch(note)
        if match is not None:
            durable_failure = _require_sha256(
                match.group(1),
                label="CLI durable failure receipt",
            )
        phase_match = _FAILURE_PHASE_NOTE_RE.fullmatch(note)
        if phase_match is not None:
            classified_phase = phase_match.group(1)
        uncertainty_match = _IMAGE_UNCERTAINTY_NOTE_RE.fullmatch(note)
        if uncertainty_match is not None:
            classified_uncertainty = uncertainty_match.group(1) == "true"
    error_type = type(error).__name__
    if _SAFE_TYPE_RE.fullmatch(error_type) is None:
        error_type = "UnclassifiedError"
    uncertain = getattr(error, "image_state_uncertain", False)
    if type(uncertain) is not bool:
        uncertain = True
    if classified_uncertainty is not None:
        uncertain = classified_uncertainty
    context_sha = getattr(error, "context_receipt_sha256", None)
    execution_sha = getattr(error, "execution_receipt_sha256", None)
    image_id = getattr(error, "image_id", None)
    return {
        "context_receipt_sha256": (
            _require_sha256(context_sha, label="CLI uncertain context")
            if context_sha is not None
            else None
        ),
        "durable_failure_receipt_sha256": durable_failure,
        "error": {
            "message": _bounded_error_text(error),
            "type": error_type,
        },
        "execution_receipt_sha256": (
            _require_sha256(execution_sha, label="CLI uncertain execution receipt")
            if execution_sha is not None
            else None
        ),
        "image_id": (
            _require_image_id(image_id, label="CLI uncertain image ID")
            if image_id is not None
            else None
        ),
        "image_state_uncertain": uncertain,
        "phase": classified_phase,
        "retry_authorized": False,
        "schema_version": "alberta.forager_matched_v3.cpu_oci_build_cli_error.v1",
        "status": "cpu_oci_build_command_failed_non_authorizing",
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI with an authority-free measurement mode and a separate execution mode."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "measure":
            measured = measure_matched_v3_cpu_oci_build_request(
                repository_root=arguments.repository_root,
                manifest_output=arguments.snapshot_manifest_output,
            )
            output: dict[str, Any] = {
                "claims": _claims(),
                "directory_count": measured.directory_count,
                "file_count": measured.file_count,
                "manifest_path": str(measured.manifest_path),
                "schema_version": CPU_OCI_BUILD_REQUEST_MEASUREMENT_SCHEMA_VERSION,
                "snapshot_manifest_sha256": measured.manifest_sha256,
                "snapshot_tree_sha256": measured.tree_sha256,
                "status": "local_source_snapshot_measured_unqualified_non_authorizing",
                "total_size_bytes": measured.total_size_bytes,
            }
        else:
            manifest_bytes = _read_request_snapshot_manifest(
                arguments.snapshot_manifest,
                expected_sha256=arguments.snapshot_manifest_sha256,
            )
            published = execute_and_publish_matched_v3_cpu_oci_build(
                MatchedV3CpuOciBuildPublicationRequest(
                    repository_root=arguments.repository_root,
                    artifact_root=arguments.artifact_root,
                    publication_root=arguments.publication_root,
                    expected_snapshot_manifest_bytes=manifest_bytes,
                    expected_snapshot_manifest_sha256=(arguments.snapshot_manifest_sha256),
                    expected_snapshot_tree_sha256=arguments.snapshot_tree_sha256,
                    exact_acknowledgement=arguments.exact_acknowledgement,
                    timeout_seconds=arguments.timeout_seconds,
                )
            )
            output = {
                "context_receipt_sha256": published.context_receipt_sha256,
                "execution_receipt_sha256": published.execution_receipt_sha256,
                "image_id": published.image_id,
                "publication_receipt_sha256": published.publication_receipt_sha256,
                "schema_version": CPU_OCI_BUILD_PUBLICATION_SUCCESS_SCHEMA_VERSION,
                "status": CPU_OCI_BUILD_PUBLICATION_STATUS,
            }
    except Exception as exc:
        sys.stderr.buffer.write(_canonical_json(_cli_error_record(exc)))
        return 2
    sys.stdout.buffer.write(_canonical_json(output))
    return 0


__all__ = [
    "CPU_OCI_BUILD_PUBLICATION_FAILURE_SCHEMA_VERSION",
    "CPU_OCI_BUILD_PUBLICATION_INTENT_SCHEMA_VERSION",
    "CPU_OCI_BUILD_PUBLICATION_STATUS",
    "CPU_OCI_BUILD_PUBLICATION_SUCCESS_SCHEMA_VERSION",
    "CPU_OCI_BUILD_REQUEST_MEASUREMENT_SCHEMA_VERSION",
    "ForagerMatchedV3CpuOciBuildPublicationError",
    "MatchedV3CpuOciBuildIntentExistsError",
    "MatchedV3CpuOciBuildPublicationRequest",
    "MatchedV3CpuOciBuildPublicationStateUncertainError",
    "MatchedV3CpuOciBuildSuccessPublicationUncertainError",
    "MeasuredMatchedV3CpuOciBuildRequest",
    "PublishedMatchedV3CpuOciBuild",
    "PublishedMatchedV3CpuOciBuildFailure",
    "execute_and_publish_matched_v3_cpu_oci_build",
    "main",
    "measure_matched_v3_cpu_oci_build_request",
    "parse_matched_v3_cpu_oci_build_failure_receipt",
    "parse_matched_v3_cpu_oci_build_intent",
    "parse_matched_v3_cpu_oci_build_publication_receipt",
    "validate_published_matched_v3_cpu_oci_build",
    "validate_published_matched_v3_cpu_oci_build_failure",
]


if __name__ == "__main__":
    raise SystemExit(main())
