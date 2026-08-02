"""Reference production trust resolver for matched-current Forager campaigns.

Every authority path in the matched campaign terminates at an externally
configured :data:`~alberta_framework.benchmarks.forager_matched_executor.TrustResolver`
callable; the tree itself only ships the ``content_only_unendorsed_v1``
placeholder identity.  This module supplies a REFERENCE resolver so the
campaign authority flow can be exercised end to end against real signed
material.  It is deliberately not wired as a default anywhere: constructing
it requires explicit anchor, key, and receipt paths that only an operator
key ceremony can produce.

Mechanism.  The pinned execution venv does not provide the ``cryptography``
package, so the asymmetric Ed25519 variant is not implementable here without
a new dependency; this reference uses the stdlib symmetric equivalent —
detached HMAC-SHA256 tags over canonical receipt bytes with an explicit
domain-separation prefix.  The trust chain is:

* a **trust-anchor document** (public JSON) names the anchor identity, the
  ceremony key id, and the SHA-256 of the raw key bytes;
* a **key file** (private, ``0600``, single-link regular file) holds the raw
  hex key whose digest must match the anchor document exactly;
* a **verification receipt** per subject — one canonical JSON document under
  ``<receipt_directory>/<verification_subject_sha256>.receipt.json`` carrying
  every content digest of the subject, an issuance validity window, and the
  detached HMAC tag over the unsigned canonical bytes.

The resolver refuses (raises :class:`ForagerMatchedTrustError`) on any
mismatch: missing or malformed anchor/key/receipt, anchor identity drift,
the reserved unendorsed placeholder identity, wrong key, tampered digests,
bad signature, subject drift between the request and the receipt, and
receipts outside their validity window (stale or not yet valid).  Nothing in
this module weakens the executor's own closure checks —
``resolve_authenticated_bindings`` still re-verifies the returned bindings
against the request.

Receipt issuance (:func:`issue_matched_verification_receipt`) is the
ceremony-side primitive.  It exists so operators and tests can produce
receipts against a provisioned anchor; calling it is always an explicit
out-of-band act, never part of evidence parsing or plan construction.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

from alberta_framework.benchmarks.forager_matched_evidence import (
    AuthenticatedEvidenceBindings,
)
from alberta_framework.benchmarks.forager_matched_executor import (
    VerificationRequest,
    canonical_json_bytes,
    decode_strict_json,
)

MATCHED_TRUST_ANCHOR_SCHEMA_VERSION: Final = "alberta.forager_matched_trust_anchor.v1"
MATCHED_TRUST_RECEIPT_SCHEMA_VERSION: Final = "alberta.forager_matched_trust_receipt.v1"
TRUST_RECEIPT_SIGNATURE_DOMAIN: Final = b"alberta.forager_matched_trust_receipt.v1\x00"
TRUST_ANCHOR_ALGORITHM: Final = "hmac-sha256"
RECEIPT_FILE_SUFFIX: Final = ".receipt.json"
UNENDORSED_PLACEHOLDER_IDENTITY: Final = "content_only_unendorsed_v1"

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_KEY_HEX_RE: Final = re.compile(r"[0-9a-f]{64,128}\Z")
_MAX_DOCUMENT_BYTES: Final = 64 * 1024
_MAX_UNIX_TIME: Final = 2**63 - 1

Stage = Literal["open_tuning", "sealed_evaluation"]

_RECEIPT_DIGEST_FIELDS: Final = (
    "protocol_sha256",
    "qualification_manifest_sha256",
    "score_evidence_sha256",
    "source_manifest_sha256",
    "executor_manifest_sha256",
    "execution_closure_sha256",
    "verification_subject_sha256",
)


class ForagerMatchedTrustError(ValueError):
    """A trust anchor, ceremony key, or verification receipt failed closed."""


def _default_clock() -> int:
    return int(time.time())


def _string(value: Any, label: str, *, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum or "\x00" in value:
        raise ForagerMatchedTrustError(f"{label} must be a bounded non-empty string")
    return value


def _identifier(value: Any, label: str) -> str:
    result = _string(value, label, maximum=128)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ForagerMatchedTrustError(f"{label} must be a portable identifier")
    return result


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label, maximum=64)
    if _SHA256_RE.fullmatch(result) is None:
        raise ForagerMatchedTrustError(f"{label} must be a lowercase SHA-256")
    return result


def _unix_time(value: Any, label: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_UNIX_TIME:
        raise ForagerMatchedTrustError(f"{label} must be an integer Unix time")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ForagerMatchedTrustError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ForagerMatchedTrustError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _read_bounded_regular(
    path: Path,
    label: str,
    *,
    require_private: bool = False,
) -> bytes:
    """Read one bounded, single-link regular file without following symlinks.

    :param path: file to read.
    :param label: human-readable subject for error messages.
    :param require_private: when true, refuse group/other permission bits —
        used for the ceremony key so a world-readable key fails closed.
    """
    if not isinstance(path, Path):
        raise ForagerMatchedTrustError(f"{label} path must be a Path")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ForagerMatchedTrustError(f"{label} could not be opened: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ForagerMatchedTrustError(
                f"{label} must be a single-link regular file"
            )
        if metadata.st_size > _MAX_DOCUMENT_BYTES:
            raise ForagerMatchedTrustError(f"{label} exceeds the byte bound")
        if require_private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise ForagerMatchedTrustError(
                f"{label} must not be accessible to group or other users"
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_DOCUMENT_BYTES:
                raise ForagerMatchedTrustError(f"{label} exceeds the byte bound")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@dataclass(frozen=True, slots=True)
class TrustAnchorDocument:
    """Public identity of one ceremony anchor: who signs, with which key."""

    trust_anchor_identity: str
    key_id: str
    key_sha256: str

    def __post_init__(self) -> None:
        _identifier(self.trust_anchor_identity, "trust anchor identity")
        if self.trust_anchor_identity == UNENDORSED_PLACEHOLDER_IDENTITY:
            raise ForagerMatchedTrustError(
                "trust anchor identity must not be the reserved unendorsed placeholder"
            )
        _identifier(self.key_id, "trust anchor key_id")
        _sha256(self.key_sha256, "trust anchor key_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MATCHED_TRUST_ANCHOR_SCHEMA_VERSION,
            "algorithm": TRUST_ANCHOR_ALGORITHM,
            "trust_anchor_identity": self.trust_anchor_identity,
            "key_id": self.key_id,
            "key_sha256": self.key_sha256,
        }


def load_trust_anchor_document(path: Path) -> TrustAnchorDocument:
    """Load and strictly validate one public trust-anchor document."""
    raw = _read_bounded_regular(path, "trust anchor document")
    payload = _object(decode_strict_json(raw), "trust anchor document")
    _exact_keys(
        payload,
        {"schema_version", "algorithm", "trust_anchor_identity", "key_id", "key_sha256"},
        "trust anchor document",
    )
    if payload["schema_version"] != MATCHED_TRUST_ANCHOR_SCHEMA_VERSION:
        raise ForagerMatchedTrustError("trust anchor schema_version is unsupported")
    if payload["algorithm"] != TRUST_ANCHOR_ALGORITHM:
        raise ForagerMatchedTrustError("trust anchor algorithm is unsupported")
    return TrustAnchorDocument(
        trust_anchor_identity=_identifier(
            payload["trust_anchor_identity"],
            "trust anchor identity",
        ),
        key_id=_identifier(payload["key_id"], "trust anchor key_id"),
        key_sha256=_sha256(payload["key_sha256"], "trust anchor key_sha256"),
    )


def load_trust_anchor_key(path: Path, anchor: TrustAnchorDocument) -> bytes:
    """Load the private ceremony key and bind it to the anchor's key digest."""
    if type(anchor) is not TrustAnchorDocument:
        raise ForagerMatchedTrustError("anchor must be a TrustAnchorDocument")
    raw = _read_bounded_regular(path, "trust anchor key file", require_private=True)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ForagerMatchedTrustError("trust anchor key file is not ASCII hex") from exc
    text = text[:-1] if text.endswith("\n") else text
    if _KEY_HEX_RE.fullmatch(text) is None or len(text) % 2:
        raise ForagerMatchedTrustError(
            "trust anchor key file must hold 32..64 raw key bytes as lowercase hex"
        )
    key = bytes.fromhex(text)
    if not hmac.compare_digest(hashlib.sha256(key).hexdigest(), anchor.key_sha256):
        raise ForagerMatchedTrustError(
            "trust anchor key digest does not match the anchor document"
        )
    return key


def _receipt_signature(key: bytes, unsigned: Mapping[str, Any]) -> str:
    message = TRUST_RECEIPT_SIGNATURE_DOMAIN + canonical_json_bytes(unsigned)
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def matched_trust_receipt_path(receipt_directory: Path, subject_sha256: str) -> Path:
    """Content-addressed receipt location for one verification subject."""
    if not isinstance(receipt_directory, Path):
        raise ForagerMatchedTrustError("receipt directory must be a Path")
    subject = _sha256(subject_sha256, "verification subject SHA-256")
    return receipt_directory / f"{subject}{RECEIPT_FILE_SUFFIX}"


def issue_matched_verification_receipt(
    request: VerificationRequest,
    *,
    anchor_path: Path,
    key_path: Path,
    issued_at_unix: int,
    not_after_unix: int,
) -> dict[str, Any]:
    """Ceremony-side primitive: sign one verification subject into a receipt.

    This is an explicit operator act.  It never runs as part of evidence
    parsing, plan construction, or resolution, and it refuses when the
    request's anchor identity is not the provisioned anchor.

    :returns: the full canonical receipt document including its detached
        HMAC-SHA256 tag; the caller persists it at
        :func:`matched_trust_receipt_path`.
    """
    if type(request) is not VerificationRequest:
        raise ForagerMatchedTrustError("request must be a VerificationRequest")
    issued_at = _unix_time(issued_at_unix, "receipt issued_at_unix")
    not_after = _unix_time(not_after_unix, "receipt not_after_unix")
    if issued_at > not_after:
        raise ForagerMatchedTrustError(
            "receipt validity window is empty (issued_at_unix > not_after_unix)"
        )
    anchor = load_trust_anchor_document(anchor_path)
    if request.trust_anchor_identity != anchor.trust_anchor_identity:
        raise ForagerMatchedTrustError(
            "request trust anchor identity differs from the provisioned anchor"
        )
    key = load_trust_anchor_key(key_path, anchor)
    unsigned: dict[str, Any] = {
        "schema_version": MATCHED_TRUST_RECEIPT_SCHEMA_VERSION,
        "trust_anchor_identity": anchor.trust_anchor_identity,
        "key_id": anchor.key_id,
        "stage": request.stage,
        "protocol_sha256": request.protocol_sha256,
        "qualification_manifest_sha256": request.qualification_manifest_sha256,
        "score_evidence_sha256": request.score_evidence_sha256,
        "source_manifest_sha256": request.source_manifest_sha256,
        "executor_manifest_sha256": request.executor_manifest_sha256,
        "execution_closure_sha256": request.execution_closure_sha256,
        "verification_subject_sha256": request.verification_subject_sha256,
        "issued_at_unix": issued_at,
        "not_after_unix": not_after,
    }
    return {**unsigned, "signature_hmac_sha256": _receipt_signature(key, unsigned)}


def _parse_receipt_document(raw: bytes) -> dict[str, Any]:
    payload = _object(decode_strict_json(raw), "verification receipt")
    _exact_keys(
        payload,
        {
            "schema_version",
            "trust_anchor_identity",
            "key_id",
            "stage",
            "issued_at_unix",
            "not_after_unix",
            "signature_hmac_sha256",
            *_RECEIPT_DIGEST_FIELDS,
        },
        "verification receipt",
    )
    if payload["schema_version"] != MATCHED_TRUST_RECEIPT_SCHEMA_VERSION:
        raise ForagerMatchedTrustError("verification receipt schema_version is unsupported")
    _identifier(payload["trust_anchor_identity"], "verification receipt trust_anchor_identity")
    _identifier(payload["key_id"], "verification receipt key_id")
    if payload["stage"] not in {"open_tuning", "sealed_evaluation"}:
        raise ForagerMatchedTrustError("verification receipt stage is unsupported")
    for name in _RECEIPT_DIGEST_FIELDS:
        _sha256(payload[name], f"verification receipt {name}")
    issued_at = _unix_time(payload["issued_at_unix"], "verification receipt issued_at_unix")
    not_after = _unix_time(payload["not_after_unix"], "verification receipt not_after_unix")
    if issued_at > not_after:
        raise ForagerMatchedTrustError("verification receipt validity window is empty")
    signature = _string(
        payload["signature_hmac_sha256"],
        "verification receipt signature",
        maximum=64,
    )
    if _SHA256_RE.fullmatch(signature) is None:
        raise ForagerMatchedTrustError(
            "verification receipt signature must be a lowercase HMAC-SHA256 hex tag"
        )
    return payload


@dataclass(frozen=True, slots=True)
class SignedReceiptTrustResolver:
    """Reference :data:`TrustResolver`: signed-receipt lookup under one anchor.

    Every call re-reads the anchor document, key file, and receipt from disk
    so revocation (deleting a receipt or rotating the key) takes effect
    immediately; nothing is cached across resolutions.  The ``clock`` field
    exists so the freshness gate is testable; production callers keep the
    default wall clock.
    """

    anchor_path: Path
    key_path: Path
    receipt_directory: Path
    clock: Callable[[], int] = field(default=_default_clock)

    def __post_init__(self) -> None:
        for name, value in (
            ("anchor_path", self.anchor_path),
            ("key_path", self.key_path),
            ("receipt_directory", self.receipt_directory),
        ):
            if not isinstance(value, Path):
                raise ForagerMatchedTrustError(f"resolver {name} must be a Path")
        if not callable(self.clock):
            raise ForagerMatchedTrustError("resolver clock must be callable")

    def __call__(self, request: VerificationRequest) -> AuthenticatedEvidenceBindings:
        if type(request) is not VerificationRequest:
            raise ForagerMatchedTrustError("request must be a VerificationRequest")
        anchor = load_trust_anchor_document(self.anchor_path)
        if request.trust_anchor_identity != anchor.trust_anchor_identity:
            raise ForagerMatchedTrustError(
                "request trust anchor identity differs from the provisioned anchor"
            )
        key = load_trust_anchor_key(self.key_path, anchor)
        receipt_path = matched_trust_receipt_path(
            self.receipt_directory,
            request.verification_subject_sha256,
        )
        raw = _read_bounded_regular(receipt_path, "verification receipt")
        document = _parse_receipt_document(raw)
        unsigned = {
            name: value
            for name, value in document.items()
            if name != "signature_hmac_sha256"
        }
        expected_signature = _receipt_signature(key, unsigned)
        if not hmac.compare_digest(document["signature_hmac_sha256"], expected_signature):
            raise ForagerMatchedTrustError(
                "verification receipt signature does not verify under the anchor key"
            )
        if (
            document["trust_anchor_identity"] != anchor.trust_anchor_identity
            or document["key_id"] != anchor.key_id
        ):
            raise ForagerMatchedTrustError(
                "verification receipt names a different anchor or ceremony key"
            )
        expected_subject = {
            "stage": request.stage,
            "protocol_sha256": request.protocol_sha256,
            "qualification_manifest_sha256": request.qualification_manifest_sha256,
            "score_evidence_sha256": request.score_evidence_sha256,
            "source_manifest_sha256": request.source_manifest_sha256,
            "executor_manifest_sha256": request.executor_manifest_sha256,
            "execution_closure_sha256": request.execution_closure_sha256,
            "verification_subject_sha256": request.verification_subject_sha256,
        }
        drifted = [
            name for name, value in expected_subject.items() if document[name] != value
        ]
        if drifted:
            raise ForagerMatchedTrustError(
                "verification receipt endorses a different subject: " + ", ".join(drifted)
            )
        now = _unix_time(self.clock(), "resolver clock reading")
        if now < document["issued_at_unix"]:
            raise ForagerMatchedTrustError("verification receipt is not yet valid")
        if now > document["not_after_unix"]:
            raise ForagerMatchedTrustError("verification receipt has expired")
        return AuthenticatedEvidenceBindings(
            stage=cast(Stage, document["stage"]),
            protocol_sha256=document["protocol_sha256"],
            score_evidence_sha256=document["score_evidence_sha256"],
            source_manifest_sha256=document["source_manifest_sha256"],
            executor_manifest_sha256=document["executor_manifest_sha256"],
            qualification_manifest_sha256=document["qualification_manifest_sha256"],
            execution_closure_sha256=document["execution_closure_sha256"],
            trust_anchor_identity=document["trust_anchor_identity"],
            verification_subject_sha256=document["verification_subject_sha256"],
            verification_receipt_sha256=hashlib.sha256(
                canonical_json_bytes(document)
            ).hexdigest(),
        )
