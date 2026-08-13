from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_candidate_universe as universe
from alberta_framework.benchmarks import forager_matched_v3_configuration_plan as plan
from alberta_framework.benchmarks import forager_matched_v3_development_schedule as subject
from alberta_framework.benchmarks import forager_matched_v3_protocol as protocol


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _finalize_registry(body: dict[str, Any]) -> bytes:
    payload = dict(body)
    payload["registry_body_sha256"] = _sha(_canonical(body))
    return _canonical(payload)


def _registry_body(*, block_count: int = 2, colliding: bool = False) -> dict[str, Any]:
    candidates = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    blocks: list[dict[str, Any]] = []
    for block_ordinal in range(block_count):
        environment_seed = 71 if colliding else 1000 + block_ordinal
        blocks.append(
            {
                "block_ordinal": block_ordinal,
                "block_id": f"development_block_{block_ordinal:04d}",
                "derivation_payload_sha256": _sha(
                    f"derivation-{block_ordinal}".encode("ascii")
                ),
                "environment_seed": environment_seed,
                "agent_seeds": [
                    {
                        "candidate_id": candidate_id,
                        "namespace": f"agent/{candidate_id}",
                        "seed": 73 if colliding else 2000 + block_ordinal * 100 + ordinal,
                    }
                    for ordinal, candidate_id in enumerate(candidates)
                ],
            }
        )
    return {
        "schema_version": subject.DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION,
        "classification": "provisioned_open_development_seeds_nonpromoting",
        "stage": "development_selection_v3",
        "candidate_universe_sha256": (
            universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256
        ),
        "candidate_order": list(candidates),
        "derivation_schema_version": (
            subject.DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION
        ),
        "derivation_domain": subject.DEVELOPMENT_SEED_DERIVATION_DOMAIN,
        "provider_receipt": {
            "schema_version": subject.DEVELOPMENT_SEED_PROVIDER_RECEIPT_SCHEMA_VERSION,
            "file_sha256": _sha(b"development-provider-receipt-file"),
            "body_sha256": _sha(b"development-provider-receipt-body"),
        },
        "blocks": blocks,
        "claims": {
            "confirmatory_or_held_out": False,
            "execution_authorized": False,
            "scientific_evidence": False,
            "scientific_promotion_allowed": False,
            "universal_sota_claim": False,
        },
        "limitations": [
            "The registry parser does not issue seeds or authenticate the provider receipt bytes.",
            "The registry is open development material and is never a held-out seed source.",
            "Content and body hashes establish byte identity, not randomness provenance.",
        ],
    }


def _registry(*, block_count: int = 2, colliding: bool = False) -> subject.DevelopmentSeedRegistry:
    raw = _finalize_registry(_registry_body(block_count=block_count, colliding=colliding))
    return subject.parse_development_seed_registry(
        raw,
        expected_registry_file_sha256=_sha(raw),
    )


def _binding(label: str, *, schema: str | None = None) -> subject.ContentBinding:
    return subject.ContentBinding(
        schema_version=schema or f"alberta.forager_matched_v3.{label}.v1",
        file_sha256=_sha(f"{label}-file".encode("ascii")),
        body_sha256=_sha(f"{label}-body".encode("ascii")),
    )


def _execution_bindings() -> dict[str, subject.ContentBinding]:
    return {
        candidate_id: _binding(f"execution_{candidate_id}")
        for candidate_id in universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    }


def _retry() -> subject.RetryPolicyBinding:
    return subject.RetryPolicyBinding(
        schema_version=subject.DEVELOPMENT_RETRY_POLICY_SCHEMA_VERSION,
        sha256=_sha(b"frozen-development-retry-policy"),
    )


def _schedule_fixture(
    *, block_count: int = 2, colliding: bool = False
) -> tuple[
    subject.DevelopmentSchedule,
    subject.DevelopmentSeedRegistry,
    subject.ContentBinding,
    dict[str, subject.ContentBinding],
    subject.RetryPolicyBinding,
]:
    registry = _registry(block_count=block_count, colliding=colliding)
    qualification = _binding("qualification_manifest")
    executions = _execution_bindings()
    retry = _retry()
    schedule = subject.build_development_schedule(
        seed_registry=registry,
        qualification_manifest=qualification,
        candidate_execution_bindings=executions,
        retry_policy=retry,
    )
    return schedule, registry, qualification, executions, retry


def _rehash_schedule(payload: dict[str, Any]) -> bytes:
    body = dict(payload)
    body.pop("schedule_body_sha256", None)
    rewritten = dict(body)
    rewritten["schedule_body_sha256"] = _sha(_canonical(body))
    return _canonical(rewritten)


def _rehash_cell_and_schedule(payload: dict[str, Any], cell_ordinal: int) -> bytes:
    cell = cast(dict[str, Any], payload["cells"][cell_ordinal])
    cell_body = dict(cell)
    cell_body.pop("cell_id", None)
    cell_body.pop("cell_body_sha256", None)
    cell_digest = _sha(_canonical(cell_body))
    cell["cell_id"] = f"cell_{cell_ordinal:016x}_{cell_digest}"
    cell["cell_body_sha256"] = cell_digest
    return _rehash_schedule(payload)


def _parse_schedule(
    raw: bytes,
    registry: subject.DevelopmentSeedRegistry,
    qualification: subject.ContentBinding,
    executions: dict[str, subject.ContentBinding],
    retry: subject.RetryPolicyBinding,
) -> subject.DevelopmentSchedule:
    return subject.parse_development_schedule(
        raw,
        expected_schedule_file_sha256=_sha(raw),
        seed_registry=registry,
        qualification_manifest=qualification,
        candidate_execution_bindings=executions,
        retry_policy=retry,
    )


def test_exact_schemas_and_pure_non_authorizing_surface() -> None:
    assert subject.DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION == (
        "alberta.forager_matched_v3.development_seed_registry.v1"
    )
    assert subject.DEVELOPMENT_CELL_SCHEMA_VERSION == (
        "alberta.forager_matched_v3.development_cell.v1"
    )
    assert subject.DEVELOPMENT_SCHEDULE_SCHEMA_VERSION == (
        "alberta.forager_matched_v3.development_schedule.v1"
    )
    assert subject.DEVELOPMENT_RETRY_POLICY_SCHEMA_VERSION == (
        "alberta.forager_matched_v3.development_retry_policy.v1"
    )
    assert len(
        {
            str(subject.DEVELOPMENT_SEED_DERIVATION_SCHEMA_VERSION),
            str(protocol.TRIAL_BLOCK_DERIVATION_SCHEMA_VERSION),
        }
    ) == 2
    assert len(
        {
            str(subject.DEVELOPMENT_SEED_DERIVATION_DOMAIN),
            str(protocol.TRIAL_BLOCK_DERIVATION_DOMAIN),
        }
    ) == 2
    forbidden = {
        "build_development_seed_registry",
        "derive_development_seeds",
        "execute_development_schedule",
        "issue_execution_authority",
        "load_development_results",
        "promote_development_results",
    }
    assert forbidden.isdisjoint(subject.__all__)


def test_registry_requires_full_file_digest_and_replays_body_digest() -> None:
    raw = _finalize_registry(_registry_body())
    parsed = subject.parse_development_seed_registry(
        raw,
        expected_registry_file_sha256=_sha(raw),
    )
    assert parsed.file_sha256 == _sha(raw)
    assert parsed.registry_body_sha256 == json.loads(raw)["registry_body_sha256"]
    assert subject.canonical_development_seed_registry_bytes(parsed) == raw
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            raw,
            expected_registry_file_sha256=_sha(b"different full file"),
        )


def test_registry_is_deep_detached_and_immutable() -> None:
    body = _registry_body()
    raw = _finalize_registry(body)
    parsed = subject.parse_development_seed_registry(
        raw,
        expected_registry_file_sha256=_sha(raw),
    )
    cast(dict[str, Any], body["blocks"][0])["environment_seed"] = 7
    cast(dict[str, Any], cast(dict[str, Any], body["blocks"][0])["agent_seeds"][0])[
        "seed"
    ] = 8
    assert parsed.blocks[0].environment_seed == 1000
    assert parsed.blocks[0].agent_seeds[0].seed == 2000
    with pytest.raises(FrozenInstanceError):
        parsed.file_sha256 = _sha(b"mutation")  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        parsed.blocks[0].environment_seed = 9  # type: ignore[misc]


def test_registry_preserves_numeric_seed_collisions_as_distinct_records() -> None:
    parsed = _registry(block_count=3, colliding=True)
    assert [block.environment_seed for block in parsed.blocks] == [71, 71, 71]
    assert all(seed.seed == 73 for block in parsed.blocks for seed in block.agent_seeds)
    assert tuple(block.block_id for block in parsed.blocks) == (
        "development_block_0000",
        "development_block_0001",
        "development_block_0002",
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", "alberta.forager_matched_v3.held_out_seed_registry.v1"),
        (
            "derivation_schema_version",
            protocol.TRIAL_BLOCK_DERIVATION_SCHEMA_VERSION,
        ),
        ("derivation_domain", protocol.TRIAL_BLOCK_DERIVATION_DOMAIN),
        ("classification", "held_out_confirmatory_seeds"),
        ("stage", "confirmatory_evaluation_v3"),
    ],
)
def test_registry_rejects_coherently_rehashed_held_out_substitutions(
    field: str, replacement: str
) -> None:
    body = _registry_body()
    body[field] = replacement
    raw = _finalize_registry(body)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            raw,
            expected_registry_file_sha256=_sha(raw),
        )


def test_registry_rejects_held_out_provider_receipt_schema() -> None:
    body = _registry_body()
    cast(dict[str, Any], body["provider_receipt"])["schema_version"] = (
        "alberta.forager_matched_v3.held_out_seed_provider_receipt.v1"
    )
    raw = _finalize_registry(body)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            raw,
            expected_registry_file_sha256=_sha(raw),
        )


def test_registry_rejects_duplicate_keys_noncanonical_bytes_and_missing_newline() -> None:
    raw = _finalize_registry(_registry_body())
    duplicate = raw.replace(
        b'{"blocks":',
        b'{"blocks":[],"blocks":',
        1,
    )
    variants = (
        duplicate,
        b" " + raw,
        raw[:-1],
        raw.replace(b"false", b"false ", 1),
    )
    for changed in variants:
        with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
            subject.parse_development_seed_registry(
                changed,
                expected_registry_file_sha256=_sha(changed),
            )


@pytest.mark.parametrize(
    "replacement",
    [b"NaN", b"Infinity", b"-Infinity", b"1.0", b"true"],
)
def test_registry_rejects_nonfinite_float_and_bool_integer_seeds(replacement: bytes) -> None:
    raw = _finalize_registry(_registry_body())
    changed = raw.replace(b'"environment_seed":1000', b'"environment_seed":' + replacement, 1)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            changed,
            expected_registry_file_sha256=_sha(changed),
        )


def test_registry_rejects_oversized_and_huge_integer_documents() -> None:
    oversized = b"{" + b" " * (8 * 1024 * 1024) + b"}\n"
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            oversized,
            expected_registry_file_sha256=_sha(oversized),
        )
    raw = _finalize_registry(_registry_body())
    huge = raw.replace(b'"environment_seed":1000', b'"environment_seed":' + b"9" * 21, 1)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            huge,
            expected_registry_file_sha256=_sha(huge),
        )


@pytest.mark.parametrize(
    "mutation",
    ["candidate_order", "agent_order", "namespace", "block_ordinal", "block_id", "claims"],
)
def test_registry_rejects_order_membership_namespace_identity_and_claim_drift(
    mutation: str,
) -> None:
    body = _registry_body()
    blocks = cast(list[dict[str, Any]], body["blocks"])
    if mutation == "candidate_order":
        cast(list[str], body["candidate_order"])[0:2] = reversed(
            cast(list[str], body["candidate_order"])[0:2]
        )
    elif mutation == "agent_order":
        seeds = cast(list[dict[str, Any]], blocks[0]["agent_seeds"])
        seeds[0], seeds[1] = seeds[1], seeds[0]
    elif mutation == "namespace":
        cast(list[dict[str, Any]], blocks[0]["agent_seeds"])[0]["namespace"] = "agent/wrong"
    elif mutation == "block_ordinal":
        blocks[0]["block_ordinal"] = 1
    elif mutation == "block_id":
        blocks[1]["block_id"] = blocks[0]["block_id"]
    else:
        cast(dict[str, bool], body["claims"])["execution_authorized"] = True
    raw = _finalize_registry(body)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            raw,
            expected_registry_file_sha256=_sha(raw),
        )


def test_registry_has_no_default_block_count_but_requires_a_nonempty_bounded_set() -> None:
    assert len(_registry(block_count=1).blocks) == 1
    assert len(_registry(block_count=3).blocks) == 3
    raw = _finalize_registry(_registry_body(block_count=0))
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_seed_registry(
            raw,
            expected_registry_file_sha256=_sha(raw),
        )


def test_content_and_retry_bindings_are_strict_frozen_values() -> None:
    binding = _binding("content")
    retry = _retry()
    assert binding.to_payload() == {
        "schema_version": "alberta.forager_matched_v3.content.v1",
        "file_sha256": _sha(b"content-file"),
        "body_sha256": _sha(b"content-body"),
    }
    with pytest.raises(FrozenInstanceError):
        binding.body_sha256 = _sha(b"changed")  # type: ignore[misc]
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.ContentBinding("held out schema", _sha(b"f"), _sha(b"b"))
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.ContentBinding("valid.schema", "0" * 64, _sha(b"b"))
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.RetryPolicyBinding("alberta.wrong.v1", retry.sha256)


def test_builder_emits_exact_block_major_cartesian_schedule() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=3)
    candidates = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS
    assert schedule.candidate_order == candidates
    assert len(candidates) == 25
    assert len(schedule.cells) == 75
    assert schedule.block_order == tuple(block.block_id for block in registry.blocks)
    assert schedule.candidate_universe.to_payload() == {
        "schema_version": universe.FORAGER_MATCHED_V3_DEVELOPMENT_UNIVERSE_SCHEMA_VERSION,
        "sha256": universe.MATCHED_V3_DEVELOPMENT_UNIVERSE_SHA256,
    }
    assert schedule.configuration_plan.to_payload() == {
        "schema_version": plan.CONFIGURATION_PLAN_SCHEMA_VERSION,
        "sha256": plan.MATCHED_V3_CONFIGURATION_PLAN_SHA256,
    }
    assert schedule.cumulative_reward_metric.to_payload() == {
        "schema_version": protocol.CUMULATIVE_REWARD_METRIC_SCHEMA_VERSION,
        "sha256": protocol.CUMULATIVE_REWARD_METRIC_SHA256,
    }
    assert schedule.seed_registry == subject.ContentBinding(
        subject.DEVELOPMENT_SEED_REGISTRY_SCHEMA_VERSION,
        registry.file_sha256,
        registry.registry_body_sha256,
    )
    assert schedule.qualification_manifest == qualification
    assert schedule.retry_policy == retry
    for ordinal, cell in enumerate(schedule.cells):
        block_ordinal, candidate_ordinal = divmod(ordinal, 25)
        assert cell.ordinal == ordinal
        assert cell.block_ordinal == block_ordinal
        assert cell.candidate_ordinal == candidate_ordinal
        assert cell.candidate_id == candidates[candidate_ordinal]
        assert cell.block_id == registry.blocks[block_ordinal].block_id
        assert cell.agent_seed_namespace == f"agent/{cell.candidate_id}"
        assert cell.analysis_role == "inferential"
        assert cell.candidate_execution_binding_sha256 == _sha(
            _canonical(executions[cell.candidate_id].to_payload())
        )
        assert cell.cell_id == f"cell_{ordinal:016x}_{cell.cell_body_sha256}"


def test_all_cell_configuration_digests_use_exact_plan_record_bytes_without_newline() -> None:
    schedule, *_ = _schedule_fixture(block_count=2)
    expected_by_candidate: dict[str, str] = {}
    newline_digest_by_candidate: dict[str, str] = {}
    for candidate_id in schedule.candidate_order:
        record = plan.configuration_record(candidate_id)
        exact_plan_bytes = plan._canonical_bytes(record)
        assert not exact_plan_bytes.endswith(b"\n")
        expected_by_candidate[candidate_id] = _sha(exact_plan_bytes)
        newline_digest_by_candidate[candidate_id] = _sha(exact_plan_bytes + b"\n")
        assert (
            expected_by_candidate[candidate_id]
            != newline_digest_by_candidate[candidate_id]
        )
    assert {
        cell.candidate_id: cell.configuration_record_sha256
        for cell in schedule.cells[:25]
    } == expected_by_candidate
    for cell in schedule.cells:
        assert cell.configuration_record_sha256 == expected_by_candidate[cell.candidate_id]
        assert (
            cell.configuration_record_sha256
            != newline_digest_by_candidate[cell.candidate_id]
        )


def test_schedule_is_canonical_newline_terminated_and_requires_full_digest() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture()
    raw = subject.canonical_development_schedule_bytes(schedule)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert raw == _canonical(schedule.to_payload())
    assert schedule.file_sha256 == _sha(raw)
    parsed = _parse_schedule(raw, registry, qualification, executions, retry)
    assert parsed == schedule
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.parse_development_schedule(
            raw,
            expected_schedule_file_sha256=_sha(b"wrong schedule file"),
            seed_registry=registry,
            qualification_manifest=qualification,
            candidate_execution_bindings=executions,
            retry_policy=retry,
        )


def test_schedule_preserves_colliding_seeds_as_distinct_cells() -> None:
    schedule, *_ = _schedule_fixture(block_count=2, colliding=True)
    assert len(schedule.cells) == 50
    assert {cell.environment_seed for cell in schedule.cells} == {71}
    assert {cell.agent_seed for cell in schedule.cells} == {73}
    assert len({cell.cell_id for cell in schedule.cells}) == 50
    assert subject.scheduled_cell(schedule, schedule.cells[0].cell_id) is schedule.cells[0]
    assert subject.scheduled_cell(schedule, schedule.cells[25].cell_id) is schedule.cells[25]


@pytest.mark.parametrize(
    "cell_id",
    [
        True,
        "",
        "cell_0000000000000000_" + "A" * 64,
        "cell_000000000000000_" + "a" * 64,
        "cell_0000000000000000_" + "a" * 63,
        "cell_0000000000000019_" + "a" * 64,
        "cell_0000000000000000_" + "a" * 64,
    ],
)
def test_scheduled_cell_rejects_non_string_malformed_out_of_range_and_unknown_ids(
    cell_id: object,
) -> None:
    schedule, *_ = _schedule_fixture(block_count=1)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.scheduled_cell(schedule, cast(str, cell_id))


def test_scheduled_cell_rejects_an_ordinal_digest_cross_wiring() -> None:
    schedule, *_ = _schedule_fixture(block_count=1)
    first = schedule.cells[0]
    cross_wired = f"cell_{1:016x}_{first.cell_body_sha256}"
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.scheduled_cell(schedule, cross_wired)


@pytest.mark.parametrize(
    "binding_key",
    [
        "candidate_universe",
        "configuration_plan",
        "cumulative_reward_metric",
        "seed_registry",
        "qualification_manifest",
        "retry_policy",
        "candidate_execution_bindings_sha256",
    ],
)
def test_parser_rejects_coherently_rehashed_top_level_binding_mutations(
    binding_key: str,
) -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    payload = cast(
        dict[str, Any], json.loads(subject.canonical_development_schedule_bytes(schedule))
    )
    if binding_key == "candidate_execution_bindings_sha256":
        payload[binding_key] = _sha(b"changed execution binding set")
    else:
        binding = cast(dict[str, Any], payload[binding_key])
        digest_key = "sha256" if "sha256" in binding else "body_sha256"
        binding[digest_key] = _sha(f"changed-{binding_key}".encode("ascii"))
    raw = _rehash_schedule(payload)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        _parse_schedule(raw, registry, qualification, executions, retry)


@pytest.mark.parametrize(
    "cell_field",
    [
        "configuration_record_sha256",
        "candidate_execution_binding_sha256",
        "derivation_payload_sha256",
        "environment_seed",
        "agent_seed",
        "candidate_id",
        "candidate_ordinal",
        "development_selection_group",
    ],
)
def test_parser_rejects_coherently_rehashed_cell_mutations(cell_field: str) -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    payload = cast(
        dict[str, Any], json.loads(subject.canonical_development_schedule_bytes(schedule))
    )
    cell = cast(dict[str, Any], payload["cells"][0])
    if cell_field.endswith("sha256"):
        cell[cell_field] = _sha(f"changed-{cell_field}".encode("ascii"))
    elif cell_field in {"environment_seed", "agent_seed", "candidate_ordinal"}:
        cell[cell_field] = cast(int, cell[cell_field]) + 1
    elif cell_field == "candidate_id":
        cell[cell_field] = universe.MATCHED_V3_DEVELOPMENT_INFERENTIAL_CANDIDATE_IDS[1]
    else:
        cell[cell_field] = "wrong_group"
    raw = _rehash_cell_and_schedule(payload, 0)
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        _parse_schedule(raw, registry, qualification, executions, retry)


def test_parser_rejects_swapped_missing_or_duplicated_cells_after_rehash() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    for mutation in ("swap", "missing", "duplicate"):
        payload = cast(
            dict[str, Any], json.loads(subject.canonical_development_schedule_bytes(schedule))
        )
        cells = cast(list[dict[str, Any]], payload["cells"])
        if mutation == "swap":
            cells[0], cells[1] = cells[1], cells[0]
        elif mutation == "missing":
            cells.pop()
        else:
            cells[-1] = dict(cells[0])
        raw = _rehash_schedule(payload)
        with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
            _parse_schedule(raw, registry, qualification, executions, retry)


def test_parser_requires_the_caller_expected_qualification_and_execution_bindings() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    raw = subject.canonical_development_schedule_bytes(schedule)
    changed_qualification = _binding("different_qualification_manifest")
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        _parse_schedule(raw, registry, changed_qualification, executions, retry)
    changed_executions = dict(executions)
    changed_executions[schedule.candidate_order[0]] = _binding("different_execution")
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        _parse_schedule(raw, registry, qualification, changed_executions, retry)


def test_parser_rejects_a_coherent_different_registry_binding() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    raw = subject.canonical_development_schedule_bytes(schedule)
    different_body = _registry_body(block_count=1)
    cast(list[dict[str, Any]], different_body["blocks"])[0]["environment_seed"] = 99
    different_raw = _finalize_registry(different_body)
    different_registry = subject.parse_development_seed_registry(
        different_raw,
        expected_registry_file_sha256=_sha(different_raw),
    )
    assert different_registry.file_sha256 != registry.file_sha256
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        _parse_schedule(raw, different_registry, qualification, executions, retry)


def test_schedule_values_are_deep_detached_from_binding_mapping_and_payloads() -> None:
    schedule, _, _, executions, _ = _schedule_fixture(block_count=1)
    first_id = schedule.candidate_order[0]
    first_digest = schedule.cells[0].candidate_execution_binding_sha256
    executions[first_id] = _binding("post_build_mutation")
    assert schedule.cells[0].candidate_execution_binding_sha256 == first_digest
    payload = schedule.to_payload()
    cast(dict[str, Any], payload["qualification_manifest"])["body_sha256"] = _sha(
        b"payload mutation"
    )
    cast(list[dict[str, Any]], payload["cells"])[0]["agent_seed"] = 17
    assert schedule.qualification_manifest.body_sha256 != _sha(b"payload mutation")
    assert schedule.cells[0].agent_seed != 17


def test_internal_canonicalizer_rejects_aliases_cycles_bool_int_confusion_and_nonfinite() -> None:
    shared: dict[str, object] = {"value": 1}
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject._canonical_json_bytes(
            {"a": shared, "b": shared}, label="aliased", maximum=4096
        )
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject._canonical_json_bytes(
            cycle, label="cycle", maximum=4096
        )
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject._canonical_json_bytes(
            {"value": float("nan")}, label="nan", maximum=4096
        )


def test_schedule_parser_rejects_duplicate_noncanonical_and_bool_integer_bytes() -> None:
    schedule, registry, qualification, executions, retry = _schedule_fixture(block_count=1)
    raw = subject.canonical_development_schedule_bytes(schedule)
    duplicate = raw.replace(
        b'{"block_order":',
        b'{"block_order":[],"block_order":',
        1,
    )
    bool_ordinal = raw.replace(b'"block_ordinal":0', b'"block_ordinal":false', 1)
    for changed in (duplicate, b" " + raw, raw[:-1], bool_ordinal):
        with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
            subject.parse_development_schedule(
                changed,
                expected_schedule_file_sha256=_sha(changed),
                seed_registry=registry,
                qualification_manifest=qualification,
                candidate_execution_bindings=executions,
                retry_policy=retry,
            )


def test_build_rejects_missing_extra_or_non_content_execution_bindings() -> None:
    registry = _registry(block_count=1)
    qualification = _binding("qualification")
    retry = _retry()
    bindings = _execution_bindings()
    bindings.pop(next(iter(bindings)))
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.build_development_schedule(
            seed_registry=registry,
            qualification_manifest=qualification,
            candidate_execution_bindings=bindings,
            retry_policy=retry,
        )
    bindings = _execution_bindings()
    bindings["extra"] = _binding("extra")
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.build_development_schedule(
            seed_registry=registry,
            qualification_manifest=qualification,
            candidate_execution_bindings=bindings,
            retry_policy=retry,
        )
    invalid = _execution_bindings()
    invalid[next(iter(invalid))] = cast(subject.ContentBinding, object())
    with pytest.raises(subject.ForagerMatchedV3DevelopmentScheduleError):
        subject.build_development_schedule(
            seed_registry=registry,
            qualification_manifest=qualification,
            candidate_execution_bindings=invalid,
            retry_policy=retry,
        )
