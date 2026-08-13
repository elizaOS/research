"""Focused contracts for the non-executable D-mapping-never-seen twin."""

from __future__ import annotations

import dataclasses
import hashlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.generated_class_d_mapping_twin as twin_module
from alberta_framework.evaluation.generated_class_d_mapping_twin import (
    D_MAPPING_NEVER_SEEN_TWIN,
    D_MAPPING_TWIN_DERANGEMENT_NAMESPACE,
    D_MAPPING_TWIN_OBSERVATION_NAMESPACE,
    REFERENCE_TRUE_MAPPING,
    SHAM_TRUE_MAPPING,
    DMappingTwinConstructionError,
    DMappingTwinContract,
    DMappingTwinDataset,
    build_d_mapping_never_seen_contract,
    build_d_mapping_twin_arms,
    build_d_mapping_twin_dataset,
    choose_cyclic_value_derangement,
    learner_view,
    measure_d_mapping_twin_array_nbytes,
    raw_d_mapping_twin_diagnostics,
    require_d_mapping_twin_runner_authorized,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    build_generated_class_recurrence_v0_protocol,
    derive_expression_manifest,
    evaluate_expression,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def twin_contract() -> DMappingTwinContract:
    return build_d_mapping_never_seen_contract()


@pytest.fixture(scope="module")
def twin_dataset(twin_contract: DMappingTwinContract) -> DMappingTwinDataset:
    return build_d_mapping_twin_dataset(twin_contract)


def _float32_bits(values: jnp.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _float32_sha256(values: jnp.ndarray) -> str:
    bits = _float32_bits(values).astype(">u4", copy=False)
    return hashlib.sha256(bits.tobytes(order="C")).hexdigest()


def test_contract_is_bound_to_stable_recurrence_and_has_no_authority(
    twin_contract: DMappingTwinContract,
) -> None:
    recurrence = build_generated_class_recurrence_v0_protocol()

    assert twin_contract.development_only
    assert not twin_contract.execution_authorized
    assert not twin_contract.runner_authorized
    assert not twin_contract.campaign_authorized
    assert not twin_contract.evidence_authorized
    assert not twin_contract.artifact_writes_authorized
    assert not twin_contract.scientific_promotion_allowed
    assert twin_contract.recurrence_schema == recurrence.schema
    assert twin_contract.expression_manifest_sha256 == (
        recurrence.expression_manifest_sha256
    )
    assert twin_contract.phase_length_manifest_sha256 == (
        recurrence.phase_length_manifest_sha256
    )
    assert twin_contract.phase_order == recurrence.phase_order
    assert twin_contract.phase_lengths == recurrence.phase_lengths
    assert twin_contract.first_d_phase_index == 3
    assert twin_contract.second_d_phase_index == 7
    assert twin_contract.second_d_is_first_true_d_phase_experience
    assert twin_contract.designated_first_d_true_mapping_exposures == 0
    assert twin_contract.preregistered_evaluator_only_noncausal_permutation
    with pytest.raises(DMappingTwinConstructionError, match="runner is unauthorized"):
        require_d_mapping_twin_runner_authorized(twin_contract)


def test_rng_namespaces_shapes_and_fixed_observations_are_exact(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    rng = twin_contract.rng_contract
    observations = twin_dataset.observations

    assert rng.observation_namespace == D_MAPPING_TWIN_OBSERVATION_NAMESPACE
    assert rng.derangement_namespace == D_MAPPING_TWIN_DERANGEMENT_NAMESPACE
    assert rng.observation_namespace != rng.derangement_namespace
    assert rng.observation_key_data != rng.derangement_key_data
    assert rng.prng_impl == "threefry2x32"
    assert rng.logical_named_rng_streams == 2
    assert rng.logical_rng_draw_sites == 2
    assert rng.builder_construction_draw_invocations == 2
    assert rng.builder_validation_replay_draw_invocations == 2
    assert rng.builder_total_draw_invocations == 4
    assert rng.per_consumer_validation_replay_draw_invocations == 2
    assert rng.learner_rng_draw_invocations_owned == 0
    assert rng.rng_accounting_scope == (
        "twin_owned_keys_only_bound_recurrence_rng_is_upstream"
    )
    assert rng.observation_uint32_words_per_draw == observations.size
    assert rng.derangement_candidate_shifts_per_draw == twin_contract.first_d_length - 1
    assert twin_contract.observation_sampling == (
        "iid_fixed_grid_from_named_threefry_uint32_bits"
    )
    assert observations.shape == (
        twin_contract.total_steps,
        twin_contract.input_dim,
    )
    assert observations.dtype == jnp.float32
    np.testing.assert_array_equal(
        np.asarray(observations * jnp.float32(64.0)),
        np.asarray(observations * jnp.float32(64.0), dtype=np.int32),
    )
    rebuilt = build_d_mapping_twin_dataset(twin_contract)
    np.testing.assert_array_equal(_float32_bits(rebuilt.observations), _float32_bits(observations))
    assert rebuilt.observation_sha256 == twin_dataset.observation_sha256


def test_rng_draw_site_invocation_accounting_is_literal(
    monkeypatch: pytest.MonkeyPatch,
    twin_contract: DMappingTwinContract,
) -> None:
    counts = {"observation": 0, "derangement": 0, "target_rows": 0}
    original_observations = twin_module._fixed_observations
    original_derangement = twin_module.choose_cyclic_value_derangement
    original_target_evaluation = twin_module._evaluate_expression_batch

    def counted_observations(contract: DMappingTwinContract) -> jnp.ndarray:
        counts["observation"] += 1
        return original_observations(contract)

    def counted_derangement(
        values: jnp.ndarray,
        key: jnp.ndarray,
    ) -> tuple[object, object, object]:
        counts["derangement"] += 1
        return original_derangement(values, key)

    def counted_target_evaluation(
        expression: object,
        observations: jnp.ndarray,
    ) -> jnp.ndarray:
        counts["target_rows"] += observations.shape[0]
        return original_target_evaluation(expression, observations)

    monkeypatch.setattr(twin_module, "_fixed_observations", counted_observations)
    monkeypatch.setattr(
        twin_module,
        "choose_cyclic_value_derangement",
        counted_derangement,
    )
    monkeypatch.setattr(
        twin_module,
        "_evaluate_expression_batch",
        counted_target_evaluation,
    )
    dataset = build_d_mapping_twin_dataset(twin_contract)
    assert counts == {
        "observation": 2,
        "derangement": 2,
        "target_rows": 2 * twin_contract.total_steps,
    }
    assert counts["observation"] + counts["derangement"] == (
        twin_contract.rng_contract.builder_total_draw_invocations
    )
    assert counts["target_rows"] == (
        twin_contract.operation_contract.builder_total_work.true_target_rows_evaluated
    )

    learner_view(dataset, REFERENCE_TRUE_MAPPING)
    assert counts == {
        "observation": 3,
        "derangement": 3,
        "target_rows": 3 * twin_contract.total_steps,
    }
    assert (
        counts["observation"]
        + counts["derangement"]
        - twin_contract.rng_contract.builder_total_draw_invocations
        == twin_contract.rng_contract.per_consumer_validation_replay_draw_invocations
    )
    rows_before_diagnostics = counts["target_rows"]
    raw_d_mapping_twin_diagnostics(dataset)
    assert counts["target_rows"] - rows_before_diagnostics == (
        twin_contract.operation_contract.per_consumer_validation_replay_work.true_target_rows_evaluated
        + twin_contract.operation_contract.raw_diagnostics_post_validation_target_rows_evaluated
    )


def test_first_d_uses_a_fixed_point_free_full_value_derangement(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    start = twin_contract.first_d_start
    stop = twin_contract.first_d_stop
    reference = _float32_bits(twin_dataset.reference_targets[start:stop])
    twin = _float32_bits(twin_dataset.twin_targets[start:stop])
    source_indices = np.asarray(twin_dataset.first_d_source_indices)
    audit = twin_dataset.derangement_audit

    assert 0 < audit.cyclic_shift < twin_contract.first_d_length
    assert source_indices.dtype == np.int32
    np.testing.assert_array_equal(
        np.sort(source_indices),
        np.arange(twin_contract.first_d_length, dtype=np.int32),
    )
    assert np.count_nonzero(source_indices == np.arange(source_indices.size)) == 0
    np.testing.assert_array_equal(twin, reference[source_indices])
    np.testing.assert_array_equal(np.sort(twin), np.sort(reference))
    assert np.count_nonzero(twin == reference) == 0
    assert audit.index_fixed_point_count == 0
    assert audit.exact_equal_value_pair_count == 0
    assert audit.exact_changed_value_pair_count == twin_contract.first_d_length
    assert audit.true_value_multiset_sha256 == audit.twin_value_multiset_sha256


def test_non_d_sham_and_second_d_streams_are_bit_identical(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    reference = _float32_bits(twin_dataset.reference_targets)
    sham = _float32_bits(twin_dataset.sham_targets)
    twin = _float32_bits(twin_dataset.twin_targets)
    first_d_mask = np.zeros(twin_contract.total_steps, dtype=np.bool_)
    first_d_mask[twin_contract.first_d_start : twin_contract.first_d_stop] = True

    np.testing.assert_array_equal(sham, reference)
    np.testing.assert_array_equal(twin[~first_d_mask], reference[~first_d_mask])
    np.testing.assert_array_equal(
        twin[twin_contract.second_d_start : twin_contract.second_d_stop],
        reference[twin_contract.second_d_start : twin_contract.second_d_stop],
    )


def test_reference_d_targets_are_exact_and_second_d_is_first_true_d_phase(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_d = next(target.expression for target in manifest.targets if target.name == "D")
    indices = (
        twin_contract.first_d_start,
        twin_contract.first_d_stop - 1,
        twin_contract.second_d_start,
        twin_contract.second_d_stop - 1,
    )

    for index in indices:
        expected = evaluate_expression(target_d, twin_dataset.observations[index])
        assert _float32_bits(expected).item() == _float32_bits(
            twin_dataset.reference_targets[index]
        ).item()
    assert np.count_nonzero(
        _float32_bits(
            twin_dataset.twin_targets[
                twin_contract.first_d_start : twin_contract.first_d_stop
            ]
        )
        == _float32_bits(
            twin_dataset.reference_targets[
                twin_contract.first_d_start : twin_contract.first_d_stop
            ]
        )
    ) == 0


def test_learner_view_contains_only_raw_features_and_scalar_targets(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    assert twin_contract.learner_input_fields == ("raw_features",)
    assert twin_contract.learner_target_fields == ("scalar_target",)
    assert twin_contract.evaluator_only_fields == (
        "phase_label",
        "phase_boundary",
        "arm_name",
        "twin_flag",
        "target_mapping_mode",
    )
    forbidden = ("phase", "label", "boundary", "twin", "arm", "mapping")
    assert all(
        all(token not in field for token in forbidden)
        for field in twin_contract.learner_input_fields
    )
    reference_observations, reference_targets = learner_view(
        twin_dataset,
        REFERENCE_TRUE_MAPPING,
    )
    twin_observations, twin_targets = learner_view(
        twin_dataset,
        D_MAPPING_NEVER_SEEN_TWIN,
    )
    assert reference_observations is twin_observations is twin_dataset.observations
    assert reference_targets is twin_dataset.reference_targets
    assert twin_targets is twin_dataset.twin_targets


def test_reference_sham_and_twin_are_resource_and_operation_paired(
    twin_contract: DMappingTwinContract,
) -> None:
    arms = build_d_mapping_twin_arms(twin_contract)

    assert tuple(arm.name for arm in arms) == (
        REFERENCE_TRUE_MAPPING,
        SHAM_TRUE_MAPPING,
        D_MAPPING_NEVER_SEEN_TWIN,
    )
    assert len({arm.pairing_manifest_sha256 for arm in arms}) == 1
    assert len({arm.resource_contract for arm in arms}) == 1
    assert len({arm.operation_contract for arm in arms}) == 1
    assert all(arm.derangement_computed for arm in arms)
    assert tuple(arm.derangement_applied for arm in arms) == (False, False, True)
    assert all(not arm.execution_authorized for arm in arms)
    assert all(not arm.evidence_authorized for arm in arms)
    assert all(not arm.scientific_promotion_allowed for arm in arms)
    assert twin_contract.paired_agent_initial_key_required
    assert twin_contract.paired_agent_rng_call_count_required
    assert not twin_contract.paired_agent_rng_audit_implemented


def test_exact_persistent_array_bytes_and_raw_diagnostics_have_no_gate(
    twin_contract: DMappingTwinContract,
    twin_dataset: DMappingTwinDataset,
) -> None:
    resources = twin_contract.resource_contract
    expected = 4 * (
        twin_contract.total_steps * twin_contract.input_dim
        + 3 * twin_contract.total_steps
        + twin_contract.first_d_length
    )

    assert resources.observation_shape == (
        twin_contract.total_steps,
        twin_contract.input_dim,
    )
    assert resources.target_shape == (twin_contract.total_steps,)
    assert resources.derangement_index_shape == (twin_contract.first_d_length,)
    assert resources.observation_dtype == "float32"
    assert resources.target_dtype == "float32"
    assert resources.derangement_index_dtype == "int32"
    assert resources.persistent_array_nbytes == expected
    assert measure_d_mapping_twin_array_nbytes(twin_dataset) == expected

    operations = twin_contract.operation_contract
    logical = operations.logical_construction_work
    replay = operations.builder_validation_replay_work
    total = operations.builder_total_work
    assert replay == logical == operations.per_consumer_validation_replay_work
    assert logical.observation_uint32_words_generated == (
        twin_contract.total_steps * twin_contract.input_dim
    )
    assert logical.true_target_rows_evaluated == twin_contract.total_steps
    assert logical.cyclic_shift_candidates_audited == twin_contract.first_d_length - 1
    assert logical.cyclic_value_comparisons == (
        twin_contract.first_d_length * (twin_contract.first_d_length - 1)
    )
    assert total.observation_uint32_words_generated == (
        2 * logical.observation_uint32_words_generated
    )
    assert total.true_target_rows_evaluated == 2 * logical.true_target_rows_evaluated
    assert total.cyclic_shift_candidates_audited == (
        2 * logical.cyclic_shift_candidates_audited
    )
    assert total.cyclic_value_comparisons == 2 * logical.cyclic_value_comparisons
    assert operations.raw_diagnostics_post_validation_target_rows_evaluated == (
        twin_contract.second_d_start
    )
    assert operations.accounting_scope == (
        "named_rng_target_and_derangement_dimensions_not_flop_or_hlo_total"
    )
    assert not operations.flop_or_hlo_equivalence_claimed

    diagnostics = raw_d_mapping_twin_diagnostics(twin_dataset)
    assert diagnostics.first_d_changed_value_rows == twin_contract.first_d_length
    assert diagnostics.first_d_equal_value_rows == 0
    assert diagnostics.non_first_d_target_bit_mismatches == 0
    assert diagnostics.second_d_target_bit_mismatches == 0
    assert diagnostics.reference_sham_target_bit_mismatches == 0
    assert diagnostics.incidental_non_d_pre_second_d_true_value_matches == 2
    diagnostic_fields = {field.name for field in dataclasses.fields(diagnostics)}
    assert diagnostic_fields.isdisjoint(
        {"accepted", "passed", "status", "threshold", "promotion_allowed"}
    )


def test_constant_target_block_fails_closed_instead_of_fake_derangement() -> None:
    values = jnp.ones((8,), dtype=jnp.float32)

    with pytest.raises(DMappingTwinConstructionError, match="no full value derangement"):
        choose_cyclic_value_derangement(values, jr.key(9_991))


def test_validator_rejects_self_hashed_forged_observations(
    twin_dataset: DMappingTwinDataset,
) -> None:
    forged_observations = twin_dataset.observations.at[0, 0].add(jnp.float32(1.0 / 64.0))
    forged = dataclasses.replace(
        twin_dataset,
        observations=forged_observations,
        observation_sha256=_float32_sha256(forged_observations),
    )

    with pytest.raises(DMappingTwinConstructionError, match="named-key stream"):
        learner_view(forged, REFERENCE_TRUE_MAPPING)


def test_validator_rejects_self_hashed_forged_reference_mapping(
    twin_dataset: DMappingTwinDataset,
) -> None:
    # Mutate reference, sham, and twin together outside first D. This preserves
    # every old cross-arm mismatch diagnostic and updates all self-hashes.
    forged_reference = twin_dataset.reference_targets.at[0].add(jnp.float32(0.125))
    forged_sham = twin_dataset.sham_targets.at[0].add(jnp.float32(0.125))
    forged_twin = twin_dataset.twin_targets.at[0].add(jnp.float32(0.125))
    forged = dataclasses.replace(
        twin_dataset,
        reference_targets=forged_reference,
        sham_targets=forged_sham,
        twin_targets=forged_twin,
        reference_target_sha256=_float32_sha256(forged_reference),
        sham_target_sha256=_float32_sha256(forged_sham),
        twin_target_sha256=_float32_sha256(forged_twin),
    )

    with pytest.raises(DMappingTwinConstructionError, match="manifest"):
        learner_view(forged, REFERENCE_TRUE_MAPPING)


def _alternate_full_derangement(
    twin_dataset: DMappingTwinDataset,
) -> tuple[jnp.ndarray, jnp.ndarray, object]:
    contract = twin_dataset.contract
    first_d = twin_dataset.reference_targets[
        contract.first_d_start : contract.first_d_stop
    ]
    canonical_indices = np.asarray(twin_dataset.first_d_source_indices)
    for seed in range(1, 17):
        deranged, indices, audit = choose_cyclic_value_derangement(first_d, jr.key(seed))
        if not np.array_equal(np.asarray(indices), canonical_indices):
            return deranged, indices, audit
    raise AssertionError("failed to find a noncanonical full derangement for adversarial test")


def test_validator_rejects_alternate_self_consistent_source_derangement(
    twin_dataset: DMappingTwinDataset,
) -> None:
    contract = twin_dataset.contract
    deranged, indices, audit = _alternate_full_derangement(twin_dataset)
    forged_twin = twin_dataset.reference_targets.at[
        contract.first_d_start : contract.first_d_stop
    ].set(deranged)
    forged = dataclasses.replace(
        twin_dataset,
        twin_targets=forged_twin,
        first_d_source_indices=indices,
        derangement_audit=audit,
        twin_target_sha256=_float32_sha256(forged_twin),
    )

    with pytest.raises(DMappingTwinConstructionError, match="named-key choice"):
        learner_view(forged, D_MAPPING_NEVER_SEEN_TWIN)


def test_validator_rejects_forged_audit_even_when_arrays_and_hashes_are_canonical(
    twin_dataset: DMappingTwinDataset,
) -> None:
    audit = twin_dataset.derangement_audit
    forged_audit = dataclasses.replace(
        audit,
        selected_candidate_rank=(audit.selected_candidate_rank + 1)
        % audit.candidate_count_evaluated,
    )
    forged = dataclasses.replace(twin_dataset, derangement_audit=forged_audit)

    with pytest.raises(DMappingTwinConstructionError, match="audit differs"):
        learner_view(forged, D_MAPPING_NEVER_SEEN_TWIN)
