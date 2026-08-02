"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_trust`.

The reference resolver is the first in-tree production TrustResolver; these
tests attack every refusal edge the module promises: missing or malformed
anchor documents, the reserved unendorsed placeholder identity, key files
that are world-readable or bound to the wrong digest, receipts that are
missing, unsigned, tampered, signed under a rotated key, replayed for a
different subject, expired, or not yet valid — all must fail closed — while
a ceremony-issued receipt must resolve to bindings that survive the
executor's own ``resolve_authenticated_bindings`` closure check.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_executor as executor
from alberta_framework.benchmarks import forager_matched_trust as trust

pytestmark = pytest.mark.unit

ANCHOR_IDENTITY = "alberta.matched.reference-anchor.test.v1"
KEY_ID = "ceremony-key-1"
ISSUED_AT = 1_000
NOT_AFTER = 2_000
CLOCK_IN_WINDOW = 1_500


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(executor.canonical_json_bytes(value)).hexdigest()


def _request(
    anchor: str = ANCHOR_IDENTITY,
    *,
    protocol_label: str = "protocol",
) -> executor.VerificationRequest:
    digests = {
        "protocol_sha256": _sha(protocol_label),
        "qualification_manifest_sha256": _sha("qualification"),
        "score_evidence_sha256": _sha("scores"),
        "source_manifest_sha256": _sha("source"),
        "executor_manifest_sha256": _sha("executor"),
        "execution_closure_sha256": _sha("closure"),
    }
    subject = _canonical_sha256(
        {
            "schema_version": evidence.MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION,
            "stage": "open_tuning",
            **digests,
            "trust_anchor_identity": anchor,
        }
    )
    return executor.VerificationRequest(
        stage="open_tuning",
        **digests,
        trust_anchor_identity=anchor,
        verification_subject_sha256=subject,
        qualification_authority_boundary={
            "endorsement_created": False,
            "endorsements_at_seal": 0,
            "gpu_qualified": False,
            "performance_claim": False,
            "seed_class": "open_development",
            "trust_profile_created": False,
            "trust_profiles_at_seal": 0,
        },
        rng_parity_qualification_status=(
            "content_complete_external_executor_receipt_unverified"
        ),
    )


def _write_key(path: Path, key: bytes) -> None:
    path.write_text(key.hex() + "\n")
    path.chmod(0o600)


def _write_anchor(path: Path, *, identity: str, key: bytes, key_id: str = KEY_ID) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": trust.MATCHED_TRUST_ANCHOR_SCHEMA_VERSION,
                "algorithm": trust.TRUST_ANCHOR_ALGORITHM,
                "trust_anchor_identity": identity,
                "key_id": key_id,
                "key_sha256": hashlib.sha256(key).hexdigest(),
            }
        )
    )


def _provision(
    tmp_path: Path,
    request: executor.VerificationRequest,
    *,
    key: bytes | None = None,
    clock_now: int = CLOCK_IN_WINDOW,
) -> tuple[trust.SignedReceiptTrustResolver, dict[str, Any]]:
    """Run a miniature key ceremony and issue one receipt for ``request``."""
    key = hashlib.sha256(b"deterministic-test-key").digest() if key is None else key
    key_path = tmp_path / "anchor.key"
    anchor_path = tmp_path / "anchor.json"
    receipts = tmp_path / "receipts"
    receipts.mkdir(exist_ok=True)
    _write_key(key_path, key)
    _write_anchor(anchor_path, identity=request.trust_anchor_identity, key=key)
    document = trust.issue_matched_verification_receipt(
        request,
        anchor_path=anchor_path,
        key_path=key_path,
        issued_at_unix=ISSUED_AT,
        not_after_unix=NOT_AFTER,
    )
    trust.matched_trust_receipt_path(
        receipts,
        request.verification_subject_sha256,
    ).write_bytes(executor.canonical_json_bytes(document))
    resolver = trust.SignedReceiptTrustResolver(
        anchor_path=anchor_path,
        key_path=key_path,
        receipt_directory=receipts,
        clock=lambda: clock_now,
    )
    return resolver, document


def _rewrite_receipt(
    resolver: trust.SignedReceiptTrustResolver,
    request: executor.VerificationRequest,
    document: dict[str, Any],
) -> None:
    trust.matched_trust_receipt_path(
        resolver.receipt_directory,
        request.verification_subject_sha256,
    ).write_bytes(executor.canonical_json_bytes(document))


class TestHappyPath:
    def test_resolver_returns_authenticated_bindings(self, tmp_path: Path) -> None:
        request = _request()
        resolver, document = _provision(tmp_path, request)
        bindings = resolver(request)
        assert type(bindings) is evidence.AuthenticatedEvidenceBindings
        assert bindings.stage == request.stage
        assert bindings.protocol_sha256 == request.protocol_sha256
        assert bindings.trust_anchor_identity == ANCHOR_IDENTITY
        assert bindings.verification_subject_sha256 == request.verification_subject_sha256
        assert bindings.verification_receipt_sha256 == hashlib.sha256(
            executor.canonical_json_bytes(document)
        ).hexdigest()

    def test_bindings_survive_executor_closure_check(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        bindings = executor.resolve_authenticated_bindings(request, resolver)
        assert bindings.execution_closure_sha256 == request.execution_closure_sha256

    def test_boundary_clock_instants_are_valid(self, tmp_path: Path) -> None:
        request = _request()
        for instant in (ISSUED_AT, NOT_AFTER):
            resolver, _ = _provision(tmp_path, request, clock_now=instant)
            assert resolver(request).stage == "open_tuning"


class TestRequestGate:
    def test_non_request_input_is_refused(self, tmp_path: Path) -> None:
        resolver, _ = _provision(tmp_path, _request())
        with pytest.raises(trust.ForagerMatchedTrustError, match="VerificationRequest"):
            resolver("not-a-request")  # type: ignore[arg-type]

    def test_issue_refuses_non_request_input(self, tmp_path: Path) -> None:
        request = _request()
        _provision(tmp_path, request)
        with pytest.raises(trust.ForagerMatchedTrustError, match="VerificationRequest"):
            trust.issue_matched_verification_receipt(
                {"stage": "open_tuning"},  # type: ignore[arg-type]
                anchor_path=tmp_path / "anchor.json",
                key_path=tmp_path / "anchor.key",
                issued_at_unix=ISSUED_AT,
                not_after_unix=NOT_AFTER,
            )

    def test_issue_refuses_empty_validity_window(self, tmp_path: Path) -> None:
        request = _request()
        _provision(tmp_path, request)
        with pytest.raises(trust.ForagerMatchedTrustError, match="validity window"):
            trust.issue_matched_verification_receipt(
                request,
                anchor_path=tmp_path / "anchor.json",
                key_path=tmp_path / "anchor.key",
                issued_at_unix=NOT_AFTER,
                not_after_unix=ISSUED_AT,
            )


class TestAnchorFailures:
    def test_missing_anchor_document_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        resolver.anchor_path.unlink()
        with pytest.raises(trust.ForagerMatchedTrustError, match="could not be opened"):
            resolver(request)

    def test_anchor_identity_mismatch_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        key = bytes.fromhex(resolver.key_path.read_text().strip())
        _write_anchor(resolver.anchor_path, identity="some.other.anchor.v1", key=key)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="differs from the provisioned anchor",
        ):
            resolver(request)

    def test_unendorsed_placeholder_identity_is_refused(self, tmp_path: Path) -> None:
        request = _request(trust.UNENDORSED_PLACEHOLDER_IDENTITY)
        key = hashlib.sha256(b"deterministic-test-key").digest()
        key_path = tmp_path / "anchor.key"
        anchor_path = tmp_path / "anchor.json"
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        _write_key(key_path, key)
        _write_anchor(
            anchor_path,
            identity=trust.UNENDORSED_PLACEHOLDER_IDENTITY,
            key=key,
        )
        resolver = trust.SignedReceiptTrustResolver(
            anchor_path=anchor_path,
            key_path=key_path,
            receipt_directory=receipts,
            clock=lambda: CLOCK_IN_WINDOW,
        )
        with pytest.raises(trust.ForagerMatchedTrustError, match="placeholder"):
            resolver(request)

    def test_wrong_anchor_schema_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        payload = json.loads(resolver.anchor_path.read_text())
        payload["schema_version"] = "alberta.forager_matched_trust_anchor.v999"
        resolver.anchor_path.write_text(json.dumps(payload))
        with pytest.raises(trust.ForagerMatchedTrustError, match="schema_version"):
            resolver(request)


class TestKeyFailures:
    def test_missing_key_file_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        resolver.key_path.unlink()
        with pytest.raises(trust.ForagerMatchedTrustError, match="could not be opened"):
            resolver(request)

    def test_key_digest_mismatch_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        _write_key(resolver.key_path, secrets.token_bytes(32))
        with pytest.raises(trust.ForagerMatchedTrustError, match="key digest"):
            resolver(request)

    def test_world_readable_key_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        resolver.key_path.chmod(0o644)
        with pytest.raises(trust.ForagerMatchedTrustError, match="group or other"):
            resolver(request)

    def test_short_key_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        short = secrets.token_bytes(16)
        resolver.key_path.write_text(short.hex() + "\n")
        resolver.key_path.chmod(0o600)
        with pytest.raises(trust.ForagerMatchedTrustError, match="32..64 raw key bytes"):
            resolver(request)

    def test_rotated_key_invalidates_old_receipt(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        rotated = hashlib.sha256(b"rotated-ceremony-key").digest()
        _write_key(resolver.key_path, rotated)
        _write_anchor(resolver.anchor_path, identity=ANCHOR_IDENTITY, key=rotated)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="signature does not verify",
        ):
            resolver(request)


class TestReceiptFailures:
    def test_missing_receipt_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        trust.matched_trust_receipt_path(
            resolver.receipt_directory,
            request.verification_subject_sha256,
        ).unlink()
        with pytest.raises(trust.ForagerMatchedTrustError, match="could not be opened"):
            resolver(request)

    def test_tampered_digest_breaks_the_signature(self, tmp_path: Path) -> None:
        request = _request()
        resolver, document = _provision(tmp_path, request)
        tampered = {**document, "score_evidence_sha256": _sha("forged-scores")}
        _rewrite_receipt(resolver, request, tampered)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="signature does not verify",
        ):
            resolver(request)

    def test_tampered_expiry_breaks_the_signature(self, tmp_path: Path) -> None:
        request = _request()
        resolver, document = _provision(tmp_path, request, clock_now=NOT_AFTER + 500)
        forged = {**document, "not_after_unix": NOT_AFTER + 1_000}
        _rewrite_receipt(resolver, request, forged)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="signature does not verify",
        ):
            resolver(request)

    def test_unsigned_receipt_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, document = _provision(tmp_path, request)
        unsigned = {
            name: value
            for name, value in document.items()
            if name != "signature_hmac_sha256"
        }
        _rewrite_receipt(resolver, request, unsigned)
        with pytest.raises(trust.ForagerMatchedTrustError, match="keys differ"):
            resolver(request)

    def test_receipt_replayed_for_another_subject_is_refused(self, tmp_path: Path) -> None:
        endorsed = _request()
        resolver, document = _provision(tmp_path, endorsed)
        replayed = _request(protocol_label="different-protocol")
        assert (
            replayed.verification_subject_sha256 != endorsed.verification_subject_sha256
        )
        _rewrite_receipt(resolver, replayed, document)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="endorses a different subject",
        ):
            resolver(replayed)

    def test_resigned_receipt_for_another_subject_is_refused(self, tmp_path: Path) -> None:
        endorsed = _request()
        resolver, _ = _provision(tmp_path, endorsed)
        replayed = _request(protocol_label="different-protocol")
        fresh = trust.issue_matched_verification_receipt(
            replayed,
            anchor_path=resolver.anchor_path,
            key_path=resolver.key_path,
            issued_at_unix=ISSUED_AT,
            not_after_unix=NOT_AFTER,
        )
        _rewrite_receipt(resolver, endorsed, fresh)
        with pytest.raises(
            trust.ForagerMatchedTrustError,
            match="endorses a different subject",
        ):
            resolver(endorsed)

    def test_expired_receipt_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request, clock_now=NOT_AFTER + 1)
        with pytest.raises(trust.ForagerMatchedTrustError, match="expired"):
            resolver(request)

    def test_not_yet_valid_receipt_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request, clock_now=ISSUED_AT - 1)
        with pytest.raises(trust.ForagerMatchedTrustError, match="not yet valid"):
            resolver(request)

    def test_symlinked_receipt_is_refused(self, tmp_path: Path) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        receipt = trust.matched_trust_receipt_path(
            resolver.receipt_directory,
            request.verification_subject_sha256,
        )
        aside = tmp_path / "aside.json"
        aside.write_bytes(receipt.read_bytes())
        receipt.unlink()
        receipt.symlink_to(aside)
        with pytest.raises(trust.ForagerMatchedTrustError, match="could not be opened"):
            resolver(request)


class TestSealBundleCompatibility:
    def test_resolver_matches_the_trust_resolver_callable_contract(
        self, tmp_path: Path
    ) -> None:
        request = _request()
        resolver, _ = _provision(tmp_path, request)
        # The seal module consumes any executor.TrustResolver via
        # resolve_authenticated_bindings; this is the exact call shape
        # authenticate_forager_matched_seal_bundle performs.
        bindings = executor.resolve_authenticated_bindings(request, resolver)
        assert bindings.to_dict()["schema_version"] == (
            evidence.AUTHENTICATED_EVIDENCE_BINDINGS_SCHEMA_VERSION
        )
