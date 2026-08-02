"""Contract tests for :mod:`alberta_framework.benchmarks.forager_matched_evidence`.

The evidence module is the narrow bridge between the frozen protocol and the
statistics layer: it parses one canonical self-hashed score-evidence bundle,
validates every bound identity against the protocol, deterministically ranks
open-tuning candidates, and constructs the statistics contract for the
open-to-sealed transition.  The tests attack each of those steps: identity
drift, rehashed rerank/statistic forgery, cross-domain digest reuse, stale
authenticated bindings, hostile or non-canonical JSON, symlinked and
ABA-swapped score files — all must fail closed — while selection itself must
be deterministic, replayable, and able to reverse a mean ranking under the
conservative CI-endpoint statistic.

Fixture helpers used by sibling suites (``test_forager_matched_seal``):
``_open_fixture()`` returns ``(open payload, parsed protocol, score bundle)``
for a two-group panel with hand-picked score vectors, and ``_sealed_fixture()``
extends it through selection and the sealed transition.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_evidence as evidence
from alberta_framework.benchmarks import forager_matched_protocol as protocol
from alberta_framework.benchmarks import forager_matched_statistics as statistics
from tests import test_forager_matched_protocol as protocol_fixtures

pytestmark = pytest.mark.integration


def _canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_QUALIFICATION_MANIFEST_SHA256 = _sha("shared-qualification-manifest")


def _rehash(payload: dict[str, Any]) -> dict[str, Any]:
    unsigned = copy.deepcopy(payload)
    unsigned.pop("payload_sha256", None)
    payload["payload_sha256"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    return payload


def _open_payload_with_two_alberta_candidates() -> dict[str, Any]:
    payload = protocol_fixtures._payload()
    payload["selection_plan"]["statistic_implementation_sha256"] = (
        evidence.MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
    )
    payload["selection_plan"]["bootstrap_rng_implementation_sha256"] = (
        evidence.MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
    )
    payload["analysis_plan"]["primary"]["implementation_sha256"] = (
        statistics.PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
    )
    payload["analysis_plan"]["secondary"]["implementation_sha256"] = (
        statistics.SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
    )
    second = protocol_fixtures._candidate(
        "alberta_route",
        stratum="alberta_learning",
        implementation_kind="alberta_causal_map",
        entrypoint_family="alberta_single_seed_worker",
        selection_group="alberta",
        eligible=True,
        analysis_role="inferential",
        exclusion_reasons=[],
    )
    payload["candidates"].insert(1, second)
    payload["selection_plan"]["groups"][0]["candidate_ids"] = [
        "alberta_causal",
        "alberta_route",
    ]
    return payload


def _score_payload(
    frozen: protocol.ForagerMatchedProtocol,
    candidate_ids: tuple[str, ...],
    *,
    score_by_candidate: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, Any]:
    score_by_candidate = score_by_candidate or {}
    candidates: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        candidate = frozen.candidate_index[candidate_id]
        values = score_by_candidate.get(
            candidate_id,
            tuple(float(index + 1) for index in range(len(frozen.active_seeds))),
        )
        assert len(values) == len(frozen.active_seeds)
        candidates.append(
            {
                "candidate_id": candidate_id,
                "capability_descriptor_sha256": (
                    protocol.candidate_capability_descriptor_sha256(candidate)
                ),
                "capability_qualification_receipt_sha256": (
                    candidate.runtime_binding.capability_qualification_receipt_sha256
                ),
                "execution_receipt_sha256": _sha(f"execution:{frozen.stage}:{candidate_id}"),
                "records": [
                    {
                        "seed": seed,
                        "score_hex": score.hex(),
                        "raw_artifact_sha256": _sha(f"raw:{frozen.stage}:{candidate_id}:{seed}"),
                        "reward_trace_sha256": _sha(f"trace:{frozen.stage}:{candidate_id}:{seed}"),
                        "scoring_record_sha256": _sha(
                            f"scoring:{frozen.stage}:{candidate_id}:{seed}"
                        ),
                    }
                    for seed, score in zip(
                        frozen.active_seeds,
                        values,
                        strict=True,
                    )
                ],
            }
        )
    return _rehash(
        {
            "schema_version": evidence.MATCHED_SCORE_EVIDENCE_SCHEMA_VERSION,
            "stage": frozen.stage,
            "protocol_sha256": frozen.protocol_sha256,
            "active_seeds": list(frozen.active_seeds),
            "horizon": frozen.horizon,
            "metric": frozen.analysis_plan.metric,
            "metric_implementation_sha256": (frozen.analysis_plan.metric_implementation_sha256),
            "task_identity_sha256": frozen.task.task_identity_sha256,
            "environment_rng_schedule_sha256": (frozen.task.environment_rng_schedule_sha256),
            "runtime_profile_sha256": frozen.runtime.runtime_profile_sha256,
            "source_evidence_sha256": _sha(f"source:{frozen.stage}"),
            "executor_evidence_sha256": _sha(f"executor:{frozen.stage}"),
            "qualification_manifest_sha256": _QUALIFICATION_MANIFEST_SHA256,
            "candidate_scores": candidates,
        }
    )


def _authenticated(
    frozen: protocol.ForagerMatchedProtocol,
    payload: dict[str, Any],
) -> evidence.AuthenticatedEvidenceBindings:
    return evidence.AuthenticatedEvidenceBindings(
        stage=frozen.stage,
        protocol_sha256=frozen.protocol_sha256,
        score_evidence_sha256=payload["payload_sha256"],
        source_manifest_sha256=payload["source_evidence_sha256"],
        executor_manifest_sha256=payload["executor_evidence_sha256"],
        qualification_manifest_sha256=payload["qualification_manifest_sha256"],
        execution_closure_sha256=(evidence.matched_execution_closure_sha256(frozen, payload)),
        trust_anchor_identity=(frozen.runtime.qualification_trust_anchor_identity),
        verification_subject_sha256=(evidence.matched_verification_subject_sha256(frozen, payload)),
        verification_receipt_sha256=_sha(f"verification:{frozen.stage}:{frozen.protocol_sha256}"),
    )


def _open_fixture() -> tuple[
    dict[str, Any],
    protocol.ForagerMatchedProtocol,
    dict[str, Any],
]:
    payload = _open_payload_with_two_alberta_candidates()
    frozen = protocol.parse_forager_matched_protocol(payload)
    candidate_ids = tuple(
        candidate_id
        for group in frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    scores = _score_payload(
        frozen,
        candidate_ids,
        score_by_candidate={
            "alberta_causal": (1.0, 1.0),
            "alberta_route": (3.0, 3.0),
            "external_dqn": (2.0, 2.0),
            "isolated_rtu": (2.5, 2.5),
        },
    )
    return payload, frozen, scores


def _sealed_fixture() -> tuple[
    protocol.ForagerMatchedProtocol,
    protocol.ForagerMatchedProtocol,
    protocol.ForagerMatchedSelectionResult,
    evidence.SelectionComputation,
    dict[str, Any],
    dict[str, Any],
    evidence.AuthenticatedEvidenceBindings,
    evidence.AuthenticatedEvidenceBindings,
]:
    open_payload, open_protocol, open_scores = _open_fixture()
    open_bindings = _authenticated(open_protocol, open_scores)
    selection = evidence.compute_open_selection(
        open_protocol,
        open_scores,
        authenticated_bindings=open_bindings,
    )
    sealed_payload, _ = protocol_fixtures._sealed_payload(
        open_payload,
        selection.selection_result.to_dict(),
    )
    sealed_protocol = protocol.parse_forager_matched_protocol(sealed_payload)
    transition = protocol.validate_sealed_protocol_transition(
        open_protocol,
        sealed_protocol,
        selection.selection_result,
        selection.selection_result.selection_result_sha256,
    )
    sealed_scores = _score_payload(
        sealed_protocol,
        transition.evaluation_candidate_ids,
        score_by_candidate={
            candidate_id: tuple(
                float(index + offset) for index in range(len(sealed_protocol.active_seeds))
            )
            for offset, candidate_id in enumerate(
                transition.evaluation_candidate_ids,
                start=1,
            )
        },
    )
    sealed_bindings = _authenticated(sealed_protocol, sealed_scores)
    return (
        open_protocol,
        sealed_protocol,
        selection.selection_result,
        selection,
        open_scores,
        sealed_scores,
        open_bindings,
        sealed_bindings,
    )


def test_score_evidence_round_trip_is_canonical_and_self_hashed() -> None:
    _, _, payload = _open_fixture()
    raw = _canonical(payload)
    parsed = evidence.parse_matched_score_evidence(
        raw,
        expected_payload_sha256=payload["payload_sha256"],
    )

    assert parsed.canonical_bytes == raw
    assert parsed.to_dict() == payload
    assert parsed.qualification_manifest_sha256 == payload["qualification_manifest_sha256"]
    assert parsed.schema_version == "alberta.forager_matched_score_evidence.v2"
    assert evidence.parse_matched_score_evidence(payload) == parsed
    assert evidence.parse_matched_score_evidence(raw.decode("ascii")) == parsed


def test_qualification_manifest_digest_is_required_exact_and_lowercase() -> None:
    _, _, payload = _open_fixture()
    missing = copy.deepcopy(payload)
    del missing["qualification_manifest_sha256"]
    _rehash(missing)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="keys differ"):
        evidence.parse_matched_score_evidence(missing)

    uppercase = copy.deepcopy(payload)
    uppercase["qualification_manifest_sha256"] = (payload["qualification_manifest_sha256"].upper())
    _rehash(uppercase)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="lowercase SHA-256"):
        evidence.parse_matched_score_evidence(uppercase)

    extra = copy.deepcopy(payload)
    extra["qualification_manifest_digest"] = extra["qualification_manifest_sha256"]
    _rehash(extra)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="keys differ"):
        evidence.parse_matched_score_evidence(extra)

    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = "alberta.forager_matched_score_evidence.v1"
    _rehash(legacy)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="unsupported"):
        evidence.parse_matched_score_evidence(legacy)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__("stage", "sealed_evaluation"),
        lambda payload: payload.__setitem__("metric", "other_metric"),
        lambda payload: payload["active_seeds"].reverse(),
        lambda payload: payload["candidate_scores"].reverse(),
        lambda payload: payload["candidate_scores"][0]["records"].reverse(),
        lambda payload: payload["candidate_scores"][0].__setitem__(
            "capability_descriptor_sha256", "0" * 64
        ),
        lambda payload: payload["candidate_scores"][0].__setitem__(
            "capability_qualification_receipt_sha256", "0" * 64
        ),
    ],
)
def test_protocol_validation_rejects_every_bound_identity_drift(
    mutator: Any,
) -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    tampered = copy.deepcopy(payload)
    mutator(tampered)
    _rehash(tampered)

    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        evidence.validate_score_evidence_against_protocol(
            frozen,
            tampered,
            authenticated_bindings=bindings,
        )


def test_direct_evidence_objects_cannot_bypass_validation() -> None:
    _, _, payload = _open_fixture()
    parsed = evidence.parse_matched_score_evidence(payload)

    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        replace(parsed, payload_sha256="0" * 64)
    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        replace(parsed.candidate_scores[0].records[0], score=cast(Any, 1))
    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        replace(parsed.candidate_scores[0], records=cast(Any, []))


def test_noncanonical_and_hostile_json_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, payload = _open_fixture()
    pretty = json.dumps(payload, indent=2)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="canonical"):
        evidence.parse_matched_score_evidence(pretty)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="duplicate"):
        evidence.decode_strict_json('{"a":1,"a":2}')
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="nesting"):
        evidence.decode_strict_json("[" * 70 + "0" + "]" * 70)
    monkeypatch.setattr(evidence, "_MAX_JSON_BYTES", 8)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="byte"):
        evidence.decode_strict_json(b'{"long":1}')


def test_loader_requires_regular_nonsymlink_canonical_file(
    tmp_path: Path,
) -> None:
    _, _, payload = _open_fixture()
    source = tmp_path / "scores.json"
    source.write_bytes(_canonical(payload))
    parsed = evidence.load_matched_score_evidence(
        source,
        expected_payload_sha256=payload["payload_sha256"],
    )
    assert parsed.to_dict() == payload

    symlink = tmp_path / "scores-link.json"
    os.symlink(source, symlink)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="symlink"):
        evidence.load_matched_score_evidence(
            symlink,
            expected_payload_sha256=payload["payload_sha256"],
        )

    fifo = tmp_path / "scores.fifo"
    os.mkfifo(fifo)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="regular file"):
        evidence.load_matched_score_evidence(
            fifo,
            expected_payload_sha256=payload["payload_sha256"],
        )


def test_selection_result_loader_requires_stable_canonical_digest_bound_file(
    tmp_path: Path,
) -> None:
    open_protocol = protocol.parse_forager_matched_protocol(protocol_fixtures._payload())
    result = protocol.parse_forager_matched_selection_result(
        protocol_fixtures._selection_result_payload(open_protocol)
    )
    source = tmp_path / "selection-result.json"
    source.write_bytes(result.canonical_bytes)

    loaded = evidence.load_forager_matched_selection_result(
        source,
        expected_selection_result_sha256=result.selection_result_sha256,
    )
    assert loaded == result

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="expected digest"):
        evidence.load_forager_matched_selection_result(
            source,
            expected_selection_result_sha256="0" * 64,
        )

    source.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="canonical"):
        evidence.load_forager_matched_selection_result(
            source,
            expected_selection_result_sha256=result.selection_result_sha256,
        )

    source.write_bytes(result.canonical_bytes)
    symlink = tmp_path / "selection-result-link.json"
    os.symlink(source, symlink)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="symlink"):
        evidence.load_forager_matched_selection_result(
            symlink,
            expected_selection_result_sha256=result.selection_result_sha256,
        )

    hardlink = tmp_path / "selection-result-hardlink.json"
    os.link(source, hardlink)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="single-link"):
        evidence.load_forager_matched_selection_result(
            hardlink,
            expected_selection_result_sha256=result.selection_result_sha256,
        )


def test_selection_result_loader_rejects_pathname_aba_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_protocol = protocol.parse_forager_matched_protocol(protocol_fixtures._payload())
    result = protocol.parse_forager_matched_selection_result(
        protocol_fixtures._selection_result_payload(open_protocol)
    )
    source = tmp_path / "selection-result.json"
    replacement = tmp_path / "selection-result-replacement.json"
    parked = tmp_path / "selection-result-parked.json"
    source.write_bytes(result.canonical_bytes)
    replacement.write_bytes(result.canonical_bytes)
    source_inode = source.stat().st_ino
    replacement_inode = replacement.stat().st_ino
    assert source_inode != replacement_inode

    real_open = os.open
    swapped = False

    def aba_open(path: Any, flags: int, mode: int = 0o777) -> int:
        nonlocal swapped
        if Path(path) == source and not swapped:
            swapped = True
            os.rename(source, parked)
            os.rename(replacement, source)
            descriptor = real_open(path, flags, mode)
            os.rename(source, replacement)
            os.rename(parked, source)
            return descriptor
        return real_open(path, flags, mode)

    monkeypatch.setattr(os, "open", aba_open)
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="changed before"):
        evidence.load_forager_matched_selection_result(
            source,
            expected_selection_result_sha256=result.selection_result_sha256,
        )
    assert swapped is True
    assert source.stat().st_ino == source_inode
    assert replacement.stat().st_ino == replacement_inode


def test_parser_normalizes_nonjson_exception_edges() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        evidence.decode_strict_json("\ud800")
    overflow = copy.deepcopy(payload)
    overflow["candidate_scores"][0]["records"][0]["score_hex"] = "0x1p+999999999"
    _rehash(overflow)
    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        evidence.parse_matched_score_evidence(overflow)

    cyclic: dict[str, Any] = {}
    cyclic["cycle"] = cyclic
    selection = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )
    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        evidence.parse_matched_selection_report(
            cyclic,
            open_protocol=frozen,
            open_evidence=payload,
            authenticated_bindings=bindings,
            selection_result=selection.selection_result,
            expected_payload_sha256=selection.report_sha256,
        )


def test_score_hex_must_be_canonical_and_finite() -> None:
    _, _, payload = _open_fixture()
    for invalid in ("1.0", "0X1.0000000000000P+0", "nan", "inf"):
        tampered = copy.deepcopy(payload)
        tampered["candidate_scores"][0]["records"][0]["score_hex"] = invalid
        _rehash(tampered)
        with pytest.raises(evidence.ForagerMatchedEvidenceError):
            evidence.parse_matched_score_evidence(tampered)


def test_chunked_selection_bootstrap_matches_one_shot_pcg64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = (1.0, 3.0, -2.0, 7.0)
    resamples = 257
    seed = 91
    confidence = 0.9
    monkeypatch.setattr(evidence, "_BOOTSTRAP_CHUNK_ELEMENTS", 7)

    mean, lower = evidence._bootstrap_lower_endpoint(
        values,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    expected = np.mean(array[indices], axis=1, dtype=np.float64)

    assert mean == float(np.mean(array, dtype=np.float64))
    assert lower == float(
        np.quantile(
            expected,
            (1.0 - confidence) / 2.0,
            method="linear",
        )
    )


def test_open_selection_is_deterministic_bound_and_recursively_immutable() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    first = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )
    second = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )

    assert first.selection_result == second.selection_result
    assert first.canonical_report_bytes == second.canonical_report_bytes
    assert first.selection_result.ranked_groups[0].ranked_candidate_ids == (
        "alberta_route",
        "alberta_causal",
    )
    assert b'"records"' not in first.canonical_report_bytes
    assert b'"score_hex"' not in first.canonical_report_bytes
    assert first.report["raw_scores_embedded"] is False
    with pytest.raises(TypeError):
        cast(dict[str, Any], first.report)["changed"] = True
    with pytest.raises(TypeError):
        cast(dict[str, Any], first.report["groups"][0])["changed"] = True
    assert (
        evidence.parse_matched_selection_report(
            first.canonical_report_bytes,
            open_protocol=frozen,
            open_evidence=payload,
            authenticated_bindings=bindings,
            selection_result=first.selection_result,
            expected_payload_sha256=first.report_sha256,
        )
        == first.report
    )


def test_conservative_selection_can_reverse_the_mean_ranking() -> None:
    open_payload = _open_payload_with_two_alberta_candidates()
    frozen = protocol.parse_forager_matched_protocol(open_payload)
    candidate_ids = tuple(
        candidate_id
        for group in frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    scores = _score_payload(
        frozen,
        candidate_ids,
        score_by_candidate={
            "alberta_causal": (0.0, 10.0),
            "alberta_route": (4.0, 4.0),
            "external_dqn": (2.0, 2.0),
            "isolated_rtu": (3.0, 3.0),
        },
    )
    bindings = _authenticated(frozen, scores)
    selection = evidence.compute_open_selection(
        frozen,
        scores,
        authenticated_bindings=bindings,
    )

    assert selection.selection_result.ranked_groups[0].ranked_candidate_ids == (
        "alberta_route",
        "alberta_causal",
    )
    group = selection.report["groups"][0]
    statistics_by_id = {item["candidate_id"]: item for item in group["candidate_statistics"]}
    assert statistics_by_id["alberta_causal"]["mean_hex"] == float(5).hex()
    assert statistics_by_id["alberta_causal"]["selection_statistic_hex"] == (float(0).hex())
    assert statistics_by_id["alberta_route"]["mean_hex"] == float(4).hex()
    assert statistics_by_id["alberta_route"]["selection_statistic_hex"] == (float(4).hex())


def test_selection_tie_break_uses_candidate_id_not_input_order() -> None:
    open_payload = _open_payload_with_two_alberta_candidates()
    open_payload["candidates"][0], open_payload["candidates"][1] = (
        open_payload["candidates"][1],
        open_payload["candidates"][0],
    )
    open_payload["selection_plan"]["groups"][0]["candidate_ids"] = [
        "alberta_route",
        "alberta_causal",
    ]
    frozen = protocol.parse_forager_matched_protocol(open_payload)
    candidate_ids = tuple(
        candidate_id
        for group in frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    scores = _score_payload(
        frozen,
        candidate_ids,
        score_by_candidate={
            "alberta_causal": (2.0, 2.0),
            "alberta_route": (2.0, 2.0),
        },
    )
    selection = evidence.compute_open_selection(
        frozen,
        scores,
        authenticated_bindings=_authenticated(frozen, scores),
    )

    assert selection.selection_result.ranked_groups[0].ranked_candidate_ids == (
        "alberta_causal",
        "alberta_route",
    )


def test_selection_requires_externally_expected_score_digest() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="authenticated"):
        evidence.compute_open_selection(
            frozen,
            payload,
            authenticated_bindings=replace(
                bindings,
                score_evidence_sha256="0" * 64,
            ),
        )


def test_selection_rejects_unknown_implementation_digests() -> None:
    open_payload, _, _ = _open_fixture()
    open_payload["selection_plan"]["statistic_implementation_sha256"] = "0" * 64
    frozen = protocol.parse_forager_matched_protocol(open_payload)
    candidate_ids = tuple(
        candidate_id
        for group in frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    scores = _score_payload(frozen, candidate_ids)
    bindings = _authenticated(frozen, scores)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="implementation"):
        evidence.compute_open_selection(
            frozen,
            scores,
            authenticated_bindings=bindings,
        )


def test_selection_implementation_descriptors_verify_their_digests() -> None:
    assert (
        hashlib.sha256(
            _canonical(evidence.matched_selection_bootstrap_rng_implementation_descriptor())
        ).hexdigest()
        == evidence.MATCHED_SELECTION_BOOTSTRAP_RNG_IMPLEMENTATION_SHA256
    )
    assert (
        hashlib.sha256(
            _canonical(evidence.matched_selection_statistic_implementation_descriptor())
        ).hexdigest()
        == evidence.MATCHED_SELECTION_STATISTIC_IMPLEMENTATION_SHA256
    )


def test_selection_fails_if_cached_semantic_descriptor_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    monkeypatch.setitem(
        evidence._SELECTION_STATISTIC_IMPLEMENTATION_DESCRIPTOR,
        "tie_break",
        "runtime_drift",
    )

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="descriptors drifted"):
        evidence.compute_open_selection(
            frozen,
            payload,
            authenticated_bindings=bindings,
        )


def test_mean_selection_does_not_run_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_payload = _open_payload_with_two_alberta_candidates()
    open_payload["selection_plan"]["statistic"] = "mean"
    frozen = protocol.parse_forager_matched_protocol(open_payload)
    candidate_ids = tuple(
        candidate_id
        for group in frozen.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    scores = _score_payload(frozen, candidate_ids)
    bindings = _authenticated(frozen, scores)

    def _unexpected_bootstrap(*args: Any, **kwargs: Any) -> tuple[float, float]:
        raise AssertionError("mean selection must not bootstrap")

    monkeypatch.setattr(evidence, "_bootstrap_lower_endpoint", _unexpected_bootstrap)
    result = evidence.compute_open_selection(
        frozen,
        scores,
        authenticated_bindings=bindings,
    )
    assert result.selection_result.ranked_groups


def test_selection_computation_cannot_be_constructed_without_replay() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    selection = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="authenticated"):
        evidence.SelectionComputation(
            selection_result=selection.selection_result,
            report=selection.report,
            _factory_token=object(),
        )


def test_selection_report_mapping_rejects_non_string_keys() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    selection = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )
    hostile: dict[Any, Any] = dict(selection.report)
    hostile[1] = "silently-colliding-key"

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="string keys"):
        evidence.parse_matched_selection_report(
            hostile,
            open_protocol=frozen,
            open_evidence=payload,
            authenticated_bindings=bindings,
            selection_result=selection.selection_result,
            expected_payload_sha256=selection.report_sha256,
        )


def test_selection_report_rejects_rehashed_group_drift() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    selection = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )
    report = json.loads(selection.canonical_report_bytes)
    report["groups"][0]["ranked_candidate_ids"].reverse()
    group = report["groups"][0]
    group_body = dict(group)
    del group_body["ranking_evidence_sha256"]
    group["ranking_evidence_sha256"] = hashlib.sha256(_canonical(group_body)).hexdigest()
    _rehash(report)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="result"):
        evidence.parse_matched_selection_report(
            report,
            open_protocol=frozen,
            open_evidence=payload,
            authenticated_bindings=bindings,
            selection_result=selection.selection_result,
            expected_payload_sha256=report["payload_sha256"],
        )


def test_build_rejects_fully_rehashed_rerank_and_statistic_forgery() -> None:
    open_payload, open_protocol, open_scores = _open_fixture()
    open_bindings = _authenticated(open_protocol, open_scores)
    selection = evidence.compute_open_selection(
        open_protocol,
        open_scores,
        authenticated_bindings=open_bindings,
    )
    forged_report = json.loads(selection.canonical_report_bytes)
    alberta_group = forged_report["groups"][0]
    alberta_group["ranked_candidate_ids"].reverse()
    causal_statistic = next(
        item
        for item in alberta_group["candidate_statistics"]
        if item["candidate_id"] == "alberta_causal"
    )
    causal_statistic["mean_hex"] = float(100).hex()
    causal_statistic["selection_statistic_hex"] = float(100).hex()
    group_body = dict(alberta_group)
    group_body.pop("ranking_evidence_sha256")
    alberta_group["ranking_evidence_sha256"] = hashlib.sha256(_canonical(group_body)).hexdigest()

    forged_result_payload = selection.selection_result.to_dict()
    forged_result_payload["ranked_groups"][0]["ranked_candidate_ids"].reverse()
    forged_result_payload["ranked_groups"][0]["ranking_evidence_sha256"] = alberta_group[
        "ranking_evidence_sha256"
    ]
    forged_result = protocol.parse_forager_matched_selection_result(forged_result_payload)
    forged_report["selection_result_sha256"] = forged_result.selection_result_sha256
    _rehash(forged_report)

    sealed_payload, _ = protocol_fixtures._sealed_payload(
        open_payload,
        forged_result.to_dict(),
    )
    sealed_protocol = protocol.parse_forager_matched_protocol(sealed_payload)
    transition = protocol.validate_sealed_protocol_transition(
        open_protocol,
        sealed_protocol,
        forged_result,
        forged_result.selection_result_sha256,
    )
    sealed_scores = _score_payload(
        sealed_protocol,
        transition.evaluation_candidate_ids,
    )
    sealed_bindings = _authenticated(sealed_protocol, sealed_scores)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="does not replay"):
        evidence.build_statistics_contract(
            open_protocol,
            sealed_protocol,
            forged_result,
            forged_report,
            open_scores,
            sealed_scores,
            open_authenticated_bindings=open_bindings,
            evaluation_authenticated_bindings=sealed_bindings,
            expected_selection_report_sha256=forged_report["payload_sha256"],
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__(
            "qualification_manifest_sha256", _sha("new-qualification")
        ),
        lambda payload: payload.__setitem__("source_evidence_sha256", _sha("new-source")),
        lambda payload: payload.__setitem__("executor_evidence_sha256", _sha("new-executor")),
        lambda payload: payload["candidate_scores"][0].__setitem__(
            "execution_receipt_sha256", _sha("new-execution")
        ),
        lambda payload: payload["candidate_scores"][0]["records"][0].__setitem__(
            "raw_artifact_sha256", _sha("new-raw")
        ),
        lambda payload: payload["candidate_scores"][0]["records"][0].__setitem__(
            "reward_trace_sha256", _sha("new-trace")
        ),
        lambda payload: payload["candidate_scores"][0]["records"][0].__setitem__(
            "scoring_record_sha256", _sha("new-scoring")
        ),
    ],
)
def test_stale_authenticated_bindings_reject_every_artifact_domain(
    mutator: Any,
) -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    tampered = copy.deepcopy(payload)
    mutator(tampered)
    _rehash(tampered)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="authenticated"):
        evidence.compute_open_selection(
            frozen,
            tampered,
            authenticated_bindings=bindings,
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.__setitem__(
            "qualification_manifest_sha256",
            payload["source_evidence_sha256"],
        ),
        lambda payload: payload["candidate_scores"][0]["records"][0].__setitem__(
            "raw_artifact_sha256",
            payload["source_evidence_sha256"],
        ),
        lambda payload: payload["candidate_scores"][0]["records"][0].__setitem__(
            "reward_trace_sha256",
            payload["candidate_scores"][0]["records"][0]["scoring_record_sha256"],
        ),
        lambda payload: payload["candidate_scores"][0].__setitem__(
            "execution_receipt_sha256",
            payload["executor_evidence_sha256"],
        ),
    ],
)
def test_execution_closure_rejects_cross_domain_digest_reuse(
    mutator: Any,
) -> None:
    _, frozen, payload = _open_fixture()
    tampered = copy.deepcopy(payload)
    mutator(tampered)
    _rehash(tampered)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="artifact domains"):
        evidence.matched_execution_closure_sha256(frozen, tampered)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("stage", "sealed_evaluation"),
        ("protocol_sha256", "0" * 64),
        ("score_evidence_sha256", "1" * 64),
        ("source_manifest_sha256", "2" * 64),
        ("executor_manifest_sha256", "3" * 64),
        ("qualification_manifest_sha256", "4" * 64),
        ("execution_closure_sha256", "5" * 64),
        ("trust_anchor_identity", "different_trust_anchor"),
    ],
)
def test_binding_subject_rejects_underlying_field_drift(
    field: str,
    replacement: str,
) -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)

    with pytest.raises(
        evidence.ForagerMatchedEvidenceError,
        match="verification subject",
    ):
        cast(Any, replace)(bindings, **{field: replacement})


def test_qualification_manifest_digest_changes_both_hash_domains() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    assert evidence.MATCHED_EXECUTION_CLOSURE_SCHEMA_VERSION.endswith(".v2")
    assert evidence.MATCHED_VERIFICATION_SUBJECT_SCHEMA_VERSION.endswith(".v2")
    original_closure = evidence.matched_execution_closure_sha256(frozen, payload)
    original_subject = evidence.matched_verification_subject_sha256(frozen, payload)
    changed = copy.deepcopy(payload)
    changed["qualification_manifest_sha256"] = _sha("different-qualification-manifest")
    _rehash(changed)

    assert evidence.matched_execution_closure_sha256(frozen, changed) != original_closure
    assert evidence.matched_verification_subject_sha256(frozen, changed) != original_subject
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="authenticated"):
        evidence.validate_score_evidence_against_protocol(
            frozen,
            changed,
            authenticated_bindings=bindings,
        )


def test_authenticated_bindings_v2_serializes_exact_qualification_digest() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)

    assert bindings.to_dict() == {
        "schema_version": "alberta.forager_authenticated_evidence_bindings.v2",
        "stage": frozen.stage,
        "protocol_sha256": frozen.protocol_sha256,
        "score_evidence_sha256": payload["payload_sha256"],
        "source_manifest_sha256": payload["source_evidence_sha256"],
        "executor_manifest_sha256": payload["executor_evidence_sha256"],
        "qualification_manifest_sha256": payload["qualification_manifest_sha256"],
        "execution_closure_sha256": bindings.execution_closure_sha256,
        "trust_anchor_identity": frozen.runtime.qualification_trust_anchor_identity,
        "verification_subject_sha256": bindings.verification_subject_sha256,
        "verification_receipt_sha256": bindings.verification_receipt_sha256,
    }
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="lowercase SHA-256"):
        replace(bindings, qualification_manifest_sha256="A" * 64)


def test_changed_external_receipt_cannot_validate_a_published_report() -> None:
    _, frozen, payload = _open_fixture()
    bindings = _authenticated(frozen, payload)
    selection = evidence.compute_open_selection(
        frozen,
        payload,
        authenticated_bindings=bindings,
    )
    changed_receipt = replace(
        bindings,
        verification_receipt_sha256=_sha("different-verification-receipt"),
    )

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="does not replay"):
        evidence.parse_matched_selection_report(
            selection.canonical_report_bytes,
            open_protocol=frozen,
            open_evidence=payload,
            authenticated_bindings=changed_receipt,
            selection_result=selection.selection_result,
            expected_payload_sha256=selection.report_sha256,
        )


def test_build_statistics_contract_binds_exact_sealed_transition() -> None:
    (
        open_protocol,
        sealed_protocol,
        selection_result,
        selection,
        open_scores,
        score_payload,
        open_bindings,
        sealed_bindings,
    ) = _sealed_fixture()

    contract, transition, parsed = evidence.build_statistics_contract(
        open_protocol,
        sealed_protocol,
        selection_result,
        selection.report,
        open_scores,
        score_payload,
        open_authenticated_bindings=open_bindings,
        evaluation_authenticated_bindings=sealed_bindings,
        expected_selection_report_sha256=selection.report_sha256,
    )

    assert parsed.payload_sha256 == score_payload["payload_sha256"]
    assert tuple(method.method_id for method in contract.methods) == tuple(
        candidate_id
        for candidate_id in transition.evaluation_candidate_ids
        if sealed_protocol.candidate_index[candidate_id].pairing.eligible
        and sealed_protocol.candidate_index[candidate_id].pairing.analysis_role == "inferential"
    )
    assert contract.primary_comparison.hypothesis_id == (
        sealed_protocol.primary_hypothesis.hypothesis_id
    )
    assert tuple(
        comparison.hypothesis_id for comparison in contract.secondary_comparisons
    ) == tuple(hypothesis.hypothesis_id for hypothesis in sealed_protocol.secondary_hypotheses)
    assert contract.evidence.sealed_protocol_sha256 == (sealed_protocol.protocol_sha256)
    assert contract.common_seeds == sealed_protocol.evaluation_seeds
    assert tuple(
        diagnostic.candidate_id for diagnostic in contract.fixed_descriptive_diagnostics
    ) == ("exact_ppo", "search_oracle")


    assert tuple(
        diagnostic.exclusion_reasons for diagnostic in contract.fixed_descriptive_diagnostics
    ) == (
        ("shared_agent_environment_rng",),
        ("privileged_observation_access",),
    )
    assert contract.primary_analysis_implementation_sha256 == (
        statistics.PRIMARY_BOOTSTRAP_IMPLEMENTATION_SHA256
    )
    assert contract.secondary_analysis_implementation_sha256 == (
        statistics.SECONDARY_SIGN_FLIP_HOLM_IMPLEMENTATION_SHA256
    )
    assert contract.evidence.score_evidence_sha256 == score_payload["payload_sha256"]
    assert contract.evidence.execution_closure_sha256 == (sealed_bindings.execution_closure_sha256)
    assert contract.evidence.authenticated_bindings_sha256 == (sealed_bindings.bindings_sha256)
    assert contract.evidence.external_verification_subject_sha256 == (
        sealed_bindings.verification_subject_sha256
    )
    assert contract.evidence.external_verification_receipt_sha256 == (
        sealed_bindings.verification_receipt_sha256
    )
    assert contract.evidence.selection_result_sha256 == (selection_result.selection_result_sha256)
    assert contract.evidence.selection_report_sha256 == selection.report_sha256

    analysis = statistics.analyze_matched_scores(contract)
    assert statistics.load_canonical_result(analysis.canonical_json(), contract) == analysis


def test_build_statistics_contract_rejects_cross_stage_qualification_drift() -> None:
    (
        open_protocol,
        sealed_protocol,
        selection_result,
        selection,
        open_scores,
        sealed_scores,
        open_bindings,
        _sealed_bindings,
    ) = _sealed_fixture()
    drifted_scores = copy.deepcopy(sealed_scores)
    drifted_scores["qualification_manifest_sha256"] = _sha(
        "different-sealed-qualification-manifest"
    )
    _rehash(drifted_scores)
    drifted_bindings = _authenticated(sealed_protocol, drifted_scores)

    with pytest.raises(
        evidence.ForagerMatchedEvidenceError,
        match="share one exact qualification manifest",
    ):
        evidence.build_statistics_contract(
            open_protocol,
            sealed_protocol,
            selection_result,
            selection.report,
            open_scores,
            drifted_scores,
            open_authenticated_bindings=open_bindings,
            evaluation_authenticated_bindings=drifted_bindings,
            expected_selection_report_sha256=selection.report_sha256,
        )


def test_build_statistics_contract_rejects_digest_or_transition_drift() -> None:
    (
        open_protocol,
        sealed_protocol,
        selection_result,
        selection,
        open_scores,
        score_payload,
        open_bindings,
        sealed_bindings,
    ) = _sealed_fixture()
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="authenticated"):
        evidence.build_statistics_contract(
            open_protocol,
            sealed_protocol,
            selection_result,
            selection.report,
            open_scores,
            score_payload,
            open_authenticated_bindings=open_bindings,
            evaluation_authenticated_bindings=replace(
                sealed_bindings,
                score_evidence_sha256="0" * 64,
            ),
            expected_selection_report_sha256=selection.report_sha256,
        )
    wrong_result = replace(
        selection_result,
        open_protocol_sha256="0" * 64,
    )
    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="does not replay"):
        evidence.build_statistics_contract(
            open_protocol,
            sealed_protocol,
            wrong_result,
            selection.report,
            open_scores,
            score_payload,
            open_authenticated_bindings=open_bindings,
            evaluation_authenticated_bindings=sealed_bindings,
            expected_selection_report_sha256=selection.report_sha256,
        )


@pytest.mark.parametrize("analysis_section", ["primary", "secondary"])
def test_build_rejects_unknown_analysis_implementation_digest(
    analysis_section: str,
) -> None:
    open_payload = _open_payload_with_two_alberta_candidates()
    open_payload["analysis_plan"][analysis_section]["implementation_sha256"] = "a" * 64
    open_protocol = protocol.parse_forager_matched_protocol(open_payload)
    tuning_ids = tuple(
        candidate_id
        for group in open_protocol.selection_plan.groups
        for candidate_id in group.candidate_ids
    )
    open_scores = _score_payload(open_protocol, tuning_ids)
    open_bindings = _authenticated(open_protocol, open_scores)
    selection = evidence.compute_open_selection(
        open_protocol,
        open_scores,
        authenticated_bindings=open_bindings,
    )
    sealed_payload, _ = protocol_fixtures._sealed_payload(
        open_payload,
        selection.selection_result.to_dict(),
    )
    sealed_protocol = protocol.parse_forager_matched_protocol(sealed_payload)
    transition = protocol.validate_sealed_protocol_transition(
        open_protocol,
        sealed_protocol,
        selection.selection_result,
        selection.selection_result.selection_result_sha256,
    )
    sealed_scores = _score_payload(
        sealed_protocol,
        transition.evaluation_candidate_ids,
    )
    sealed_bindings = _authenticated(sealed_protocol, sealed_scores)

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="implementation"):
        evidence.build_statistics_contract(
            open_protocol,
            sealed_protocol,
            selection.selection_result,
            selection.report,
            open_scores,
            sealed_scores,
            open_authenticated_bindings=open_bindings,
            evaluation_authenticated_bindings=sealed_bindings,
            expected_selection_report_sha256=selection.report_sha256,
        )


def test_sealed_validation_requires_explicit_candidate_panel() -> None:
    (
        _,
        sealed_protocol,
        _,
        _,
        _,
        score_payload,
        _,
        sealed_bindings,
    ) = _sealed_fixture()

    with pytest.raises(evidence.ForagerMatchedEvidenceError, match="explicit"):
        evidence.validate_score_evidence_against_protocol(
            sealed_protocol,
            score_payload,
            authenticated_bindings=sealed_bindings,
        )


@pytest.mark.parametrize(
    "candidate_ids",
    [
        "alberta_route",
        object(),
        [1],
        ["alberta_route", "alberta_route"],
    ],
)
def test_expected_candidate_ids_fail_closed_on_malformed_values(
    candidate_ids: Any,
) -> None:
    (
        _,
        sealed_protocol,
        _,
        _,
        _,
        score_payload,
        _,
        sealed_bindings,
    ) = _sealed_fixture()

    with pytest.raises(evidence.ForagerMatchedEvidenceError):
        evidence.validate_score_evidence_against_protocol(
            sealed_protocol,
            score_payload,
            authenticated_bindings=sealed_bindings,
            expected_candidate_ids=candidate_ids,
        )
