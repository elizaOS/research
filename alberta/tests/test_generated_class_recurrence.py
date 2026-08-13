"""Focused contracts for generated-class recurrence v0 substrate."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    PARENT_MODE_RESIDUAL_IMPRINT,
    CompositionalFeatureLearner,
    _compute_feature_values,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    FINITE_DEGREE_TWO_ARCHIVE_CEILING,
    FROZEN_LIFECYCLE,
    FULL_LIFECYCLE,
    PROTECTED_EXPRESSION_NAMESPACE,
    RANDOM_CURATION,
    ZERO_CANDIDATE_HEAD_CARRY,
    GeneratedClassProtocolNotReadyError,
    adaptation_window_mse,
    assert_whole_expression_manifests_disjoint,
    build_development_expression_manifest,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_controls,
    build_generated_class_v0_learner,
    compositional_jax_state_nbytes_formula,
    count_expression_occurrences,
    derive_expression_manifest,
    evaluate_expression,
    expression_digest,
    expression_topology_signature,
    gate_expression,
    measure_compositional_jax_state_nbytes,
    prequential_squared_loss,
    product_expression,
    raw_expression,
    recurrence_savings,
    require_generated_class_v0_executable,
    sum_expression,
    tanh_expression,
)

pytestmark = pytest.mark.unit


def test_exact_ast_grammar_and_commutative_whole_tree_identity() -> None:
    x0 = raw_expression(0)
    x1 = raw_expression(1)
    x2 = raw_expression(2)
    summ = sum_expression(x1, x0)
    product = product_expression(x1, x0)
    tanh = tanh_expression(x0, x1, theta0=2.0, theta1=-1.0)
    gate = gate_expression(x0, x1)
    observation = jnp.asarray((2.0, 3.0), dtype=jnp.float32)

    assert expression_digest(summ) == expression_digest(sum_expression(x0, x1))
    assert expression_digest(product) == expression_digest(product_expression(x0, x1))
    assert expression_digest(gate) != expression_digest(gate_expression(x1, x0))
    assert expression_digest(tanh) == expression_digest(
        tanh_expression(x1, x0, theta0=-1.0, theta1=2.0)
    )
    assert expression_digest(tanh) != expression_digest(
        tanh_expression(x1, x0, theta0=2.0, theta1=-1.0)
    )
    np.testing.assert_allclose(float(evaluate_expression(summ, observation)), 5.0)
    np.testing.assert_allclose(float(evaluate_expression(product, observation)), 6.0)
    np.testing.assert_allclose(
        float(evaluate_expression(tanh, observation)),
        np.tanh(1.0),
    )
    np.testing.assert_allclose(
        float(evaluate_expression(gate, observation)),
        2.0 / (1.0 + np.exp(-3.0)),
    )
    assert "bias" not in dataclasses.asdict(tanh)

    # The core clips every node, so commutative canonicalization must not flatten
    # associative-looking trees: these two sums have observably different values.
    left_grouped = sum_expression(sum_expression(x0, x1), x2)
    right_grouped = sum_expression(x0, sum_expression(x1, x2))
    assert expression_digest(left_grouped) != expression_digest(right_grouped)
    clipping_observation = jnp.asarray((10.0, 10.0, -10.0), dtype=jnp.float32)
    assert float(evaluate_expression(left_grouped, clipping_observation)) == 0.0
    assert float(evaluate_expression(right_grouped, clipping_observation)) == 10.0


@pytest.mark.parametrize("invalid_index", [True, -1, np.int32(0)])
def test_ast_raw_indices_fail_closed(invalid_index: object) -> None:
    error = ValueError if invalid_index == -1 else TypeError
    with pytest.raises(error):
        raw_expression(invalid_index)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_coefficient", [True, 1, np.float32(1.0)])
def test_ast_coefficient_types_fail_closed(invalid_coefficient: object) -> None:
    with pytest.raises(TypeError):
        tanh_expression(
            raw_expression(0),
            raw_expression(1),
            theta0=invalid_coefficient,  # type: ignore[arg-type]
            theta1=1.0,
        )


@pytest.mark.parametrize("invalid_coefficient", [float("nan"), float("inf"), 1e100])
def test_ast_nonfinite_float32_coefficients_fail_closed(
    invalid_coefficient: float,
) -> None:
    with pytest.raises(ValueError, match="finite float32"):
        tanh_expression(
            raw_expression(0),
            raw_expression(1),
            theta0=invalid_coefficient,
            theta1=1.0,
        )


def test_direct_core_sum_and_gate_forward_generation_and_promotion() -> None:
    observation = jnp.asarray((2.0, 3.0), dtype=jnp.float32)
    ops = jnp.asarray((OP_RAW, OP_RAW, OP_SUM, OP_GATED), dtype=jnp.int32)
    parents_a = jnp.asarray((0, 1, 0, 0), dtype=jnp.int32)
    parents_b = jnp.asarray((-1, -1, 1, 1), dtype=jnp.int32)
    theta = jnp.zeros((4, 2), dtype=jnp.float32)

    values = _compute_feature_values(ops, parents_a, parents_b, theta, observation)

    np.testing.assert_allclose(float(values[2]), 5.0)
    np.testing.assert_allclose(float(values[3]), 2.0 / (1.0 + np.exp(-3.0)))

    for op in (OP_SUM, OP_GATED):
        learner = CompositionalFeatureLearner(
            n_features=3,
            n_tasks=1,
            candidate_count=1,
            step_size_output=0.0,
            step_size_theta=0.0,
            utility_decay=0.99,
            replacement_interval=1,
            min_feature_age=0,
            candidate_min_age=0,
            promotion_margin=0.0,
            use_obgd=False,
            max_depth=3,
        )
        generated_op, _, _, _, generated_depth = learner._generate_one(
            jr.key(8_000 + op),
            jnp.asarray((0, 0, 1), dtype=jnp.int32),
            forced_op=jnp.asarray(op, dtype=jnp.int32),
        )
        assert int(generated_op) == op
        assert 1 <= int(generated_depth) <= 3

        state = learner.init(feature_dim=2, key=jr.key(9_000 + op)).replace(  # type: ignore[attr-defined]
            ops=jnp.asarray((OP_RAW, OP_RAW, OP_PRODUCT), dtype=jnp.int32),
            parent_a=jnp.asarray((0, 1, 0), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, 1), dtype=jnp.int32),
            depth=jnp.asarray((0, 0, 1), dtype=jnp.int32),
            utilities=jnp.asarray((10.0, 10.0, 0.0), dtype=jnp.float32),
            ages=jnp.asarray((10, 10, 10), dtype=jnp.int32),
            candidate_ops=jnp.asarray((op,), dtype=jnp.int32),
            candidate_parent_a=jnp.asarray((0,), dtype=jnp.int32),
            candidate_parent_b=jnp.asarray((1,), dtype=jnp.int32),
            candidate_theta=jnp.zeros((1, 2), dtype=jnp.float32),
            candidate_depth=jnp.asarray((1,), dtype=jnp.int32),
            candidate_utilities=jnp.asarray((100.0,), dtype=jnp.float32),
            candidate_ages=jnp.asarray((10,), dtype=jnp.int32),
        )
        result = learner.update(state, observation, jnp.asarray((0.0,), dtype=jnp.float32))
        assert int(result.replaced_slot) == 2
        assert int(result.promoted_candidate) == 0
        assert int(result.state.ops[2]) == op


def test_manifest_namespace_and_nonperiodic_one_head_life_are_fail_closed() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)

    assert protocol.development_only
    assert not protocol.execution_authorized
    assert not protocol.evidence_authorized
    assert not protocol.scientific_promotion_allowed
    assert protocol.n_tasks == 1
    assert protocol.context_id == 0
    assert not protocol.boundary_signal_exposed
    assert not protocol.task_specific_heads
    assert not protocol.resets_allowed
    assert protocol.phase_order == ("A", "B", "A", "D", "A", "C", "A", "D", "A")
    assert len(set(protocol.phase_lengths)) == len(protocol.phase_lengths)
    assert tuple(target.name for target in manifest.targets) == ("A", "B", "C", "D")
    assert len({target.digest for target in manifest.targets}) == 4
    assert all(target.parameter_free for target in manifest.targets)
    assert protocol.expression_manifest_sha256 == manifest.manifest_sha256
    assert protocol.evaluator_only_fields == ("phase_label", "phase_boundary")
    assert protocol.learner_observation_fields == ("raw_features",)
    with pytest.raises(PermissionError, match="protected"):
        derive_expression_manifest(PROTECTED_EXPRESSION_NAMESPACE)
    with pytest.raises(ValueError, match="overlap"):
        assert_whole_expression_manifests_disjoint(manifest, manifest)


def test_manifest_disjointness_rejects_alpha_renamed_topology_overlap() -> None:
    first_expression = sum_expression(raw_expression(0), raw_expression(1))
    renamed_expression = sum_expression(raw_expression(3), raw_expression(2))
    first = build_development_expression_manifest(
        f"{DEVELOPMENT_EXPRESSION_NAMESPACE}/alpha-a",
        (("X", first_expression),),
    )
    renamed = build_development_expression_manifest(
        f"{DEVELOPMENT_EXPRESSION_NAMESPACE}/alpha-b",
        (("Y", renamed_expression),),
    )

    assert expression_digest(first_expression) != expression_digest(renamed_expression)
    assert expression_topology_signature(
        first_expression
    ) == expression_topology_signature(renamed_expression)
    assert (
        first.targets[0].alpha_renamed_topology_signature
        == renamed.targets[0].alpha_renamed_topology_signature
    )
    with pytest.raises(ValueError, match="alpha-renamed topology overlap"):
        assert_whole_expression_manifests_disjoint(first, renamed)


def test_controls_are_capacity_matched_and_lifecycle_execution_is_blocked() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    controls = build_generated_class_v0_controls(protocol)

    assert tuple(control.name for control in controls) == (
        FULL_LIFECYCLE,
        RANDOM_CURATION,
        FROZEN_LIFECYCLE,
        ZERO_CANDIDATE_HEAD_CARRY,
        FINITE_DEGREE_TWO_ARCHIVE_CEILING,
    )
    assert len({control.resource_contract for control in controls}) == 1
    assert len({control.operation_contract for control in controls}) == 1
    assert {control.phase_length_manifest_sha256 for control in controls} == {
        protocol.phase_length_manifest_sha256
    }
    assert controls[-1].effective_max_depth == 1
    assert all(control.allocated_max_depth == 3 for control in controls)
    assert all(not control.evaluator_boundary_dependent for control in controls)
    assert "boundary" not in controls[3].intervention
    assert "identity_refresh" in controls[3].intervention
    first_four_configs = tuple(
        build_generated_class_v0_learner(control.name, protocol).to_config()
        for control in controls[:4]
    )
    assert all(config == first_four_configs[0] for config in first_four_configs)
    assert not protocol.lifecycle_prerequisites.causal_shadow_deletion_complete
    assert not protocol.lifecycle_prerequisites.matched_sham_scrub_complete
    assert not protocol.lifecycle_prerequisites.d_never_seen_twin_complete
    assert not protocol.lifecycle_prerequisites.post_scrub_generation_freeze_complete
    assert not protocol.lifecycle_prerequisites.fresh_reacquisition_generation_epoch_complete
    assert not (
        protocol.lifecycle_prerequisites.fresh_reacquisition_generation_key_namespace_complete
    )
    assert not protocol.lifecycle_prerequisites.candidate_identity_refresh_head_zero_complete
    with pytest.raises(GeneratedClassProtocolNotReadyError, match="shadow deletion"):
        require_generated_class_v0_executable(protocol)


def test_d_is_initially_absent_but_has_discrete_nonzero_generator_support() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_d = next(target for target in manifest.targets if target.name == "D")
    assert target_d.expression.left is not None
    assert target_d.expression.right is not None
    contract = protocol.reachability_contract
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    state = learner.init(protocol.input_dim, jr.key(44_001))

    assert contract.target_whole_tree_digest == target_d.whole_tree_digest
    assert contract.target_parameter_free
    assert contract.no_coefficient_tolerance
    assert count_expression_occurrences(state, target_d.expression) == (
        contract.exact_initial_active_occurrences_required,
        contract.exact_initial_candidate_occurrences_required,
    )
    assert contract.required_left_parent_digest == expression_digest(
        target_d.expression.left
    )
    assert contract.required_right_parent_digest == expression_digest(
        target_d.expression.right
    )
    assert count_expression_occurrences(state, target_d.expression.left)[0] >= 1
    assert count_expression_occurrences(state, target_d.expression.right)[0] >= 1

    op_probabilities = jax.nn.softmax(learner._op_logits())
    assert float(op_probabilities[OP_PRODUCT]) > 0.0
    assert float(op_probabilities[OP_GATED]) == contract.required_top_operation_probability
    assert contract.required_parent_choices_have_nonzero_support
    recursive_parent_mask = (state.depth >= 1) & (state.depth + 1 <= 3)
    recursive_parent_logits = learner._parent_logits(
        recursive_parent_mask,
        state.utilities,
        depth=state.depth,
        ages=state.ages,
        parent_mode=jnp.asarray(PARENT_MODE_RESIDUAL_IMPRINT, dtype=jnp.int32),
    )
    prerequisite_slots = np.flatnonzero(
        (np.asarray(state.ops) == OP_PRODUCT)
        & (np.asarray(state.parent_a) == 0)
        & (np.asarray(state.parent_b) == 0)
    )
    assert prerequisite_slots.size == 1
    assert float(jax.nn.softmax(recursive_parent_logits)[prerequisite_slots[0]]) > 0.0
    # The robust-recursive partner path is uniform over all shallow parents.
    assert int(state.depth[1]) == 0
    assert 1.0 / float(jnp.sum(state.depth == 0)) > 0.0


def test_schedule_has_conservative_curation_opportunity_margin() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    audit = protocol.curation_opportunity_audit

    assert audit.curation_interval == 32
    assert audit.conservative_lifecycle_lower_bound == 7
    assert audit.development_margin_multiplier >= 4
    assert audit.required_total_opportunities >= 28
    assert audit.opportunities_before_first_d >= 7
    assert audit.opportunities_in_first_d >= 7
    assert audit.opportunities_between_d_phases >= 7
    assert audit.opportunities_in_second_d >= 7
    assert audit.every_critical_window_meets_lower_bound
    assert audit.total_meets_development_margin
    assert audit.total_opportunities >= audit.required_total_opportunities


def test_exact_jax_state_bytes_match_independent_shape_formula() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    state = learner.init(protocol.input_dim, jr.key(12_345))

    measured = measure_compositional_jax_state_nbytes(state)
    expected = compositional_jax_state_nbytes_formula(
        protocol.active_slots,
        protocol.candidate_slots,
    )

    assert expected == (
        68 * protocol.active_slots
        + 4 * protocol.active_slots * protocol.candidate_slots
        + 80 * protocol.candidate_slots
        + 92
    )
    assert measured == expected == protocol.resource_contract.jax_state_nbytes
    assert protocol.resource_contract.host_timing_metadata_count == 2
    assert protocol.operation_contract.latency_measurement == (
        "structural_only_no_wall_clock_acceptance"
    )


def test_raw_prequential_adaptation_and_recurrence_metrics_have_no_gate() -> None:
    predictions = jnp.asarray((0.0, 1.0, 2.0, 1.0), dtype=jnp.float32)
    targets = jnp.asarray((1.0, 1.0, 0.0, 3.0), dtype=jnp.float32)
    losses = prequential_squared_loss(predictions, targets)

    np.testing.assert_array_equal(losses, np.asarray((1.0, 0.0, 4.0, 4.0)))
    windows = adaptation_window_mse(
        losses,
        phase_starts=(0, 2),
        phase_lengths=(2, 2),
        window=2,
    )
    assert windows == (0.5, 4.0)
    assert recurrence_savings(
        losses,
        first_start=2,
        recurrence_start=0,
        window=2,
    ) == 3.5


def test_phase_labels_are_not_part_of_any_learner_state_or_observation_contract() -> None:
    protocol = build_generated_class_recurrence_v0_protocol()
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    state = learner.init(protocol.input_dim, jr.key(77))
    state_fields = {field.name for field in dataclasses.fields(state)}

    assert "phase" not in state_fields
    assert "boundary" not in state_fields
    assert "context_id" not in state_fields
    assert protocol.evaluator_label_permutation_trajectory_invariant
    assert all("phase" not in field for field in protocol.learner_observation_fields)
    assert all("boundary" not in field for field in protocol.learner_observation_fields)
    assert all(isinstance(leaf, (jax.Array, float)) for leaf in jax.tree_util.tree_leaves(state))
