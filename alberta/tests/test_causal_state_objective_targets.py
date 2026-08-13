"""Contracts for learner-owned causal comprehensive-objective targets."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.causal_state_objective_targets import (
    CausalCumulantMode,
    CausalStateObjectiveAcceptedTransition,
    CausalStateObjectiveDecisionReceipt,
    CausalStateObjectiveTargetProducer,
    CausalStateObjectiveTargetProducerConfig,
    CausalStateObjectiveTargetProducerState,
    CausalStateObjectiveTargetScanInputs,
    load_causal_state_objective_target_checkpoint,
    measure_causal_state_objective_target_state_nbytes,
    run_causal_state_objective_target_scan,
    save_causal_state_objective_target_checkpoint,
)
from alberta_framework.core.comprehensive_state_objectives import (
    ComprehensiveStateObjectivesConfig,
)

pytestmark = pytest.mark.unit


def _objectives_config() -> ComprehensiveStateObjectivesConfig:
    return ComprehensiveStateObjectivesConfig(
        representation_dim=2,
        observation_target_dim=2,
        n_actions=2,
        gvf_discounts=(0.0, 0.5, 0.9),
        max_abs_control_target=100.0,
        max_abs_cumulant=100.0,
        max_abs_reward_target=100.0,
    )


def _producer(
    *,
    cumulant_mode: CausalCumulantMode = "environment_reward",
) -> CausalStateObjectiveTargetProducer:
    return CausalStateObjectiveTargetProducer(
        CausalStateObjectiveTargetProducerConfig(
            objectives_config=_objectives_config(),
            transition_owner_digest=tuple(range(8)),
            cumulant_mode=cumulant_mode,
            cumulant_owner_digest=tuple(range(8, 16)),
        )
    )


def _started(
    producer: CausalStateObjectiveTargetProducer,
) -> tuple[
    CausalStateObjectiveTargetProducerState,
    CausalStateObjectiveDecisionReceipt,
]:
    state = producer.init(jr.key(11))
    objectives_state = cast(Any, state.objectives_state).replace(
        value_weights=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        value_bias=jnp.asarray(0.5, dtype=jnp.float32),
        gvf_weights=jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
    )
    state = cast(Any, state).replace(objectives_state=objectives_state)
    cached = producer.cache_decision(
        state,
        observation=jnp.asarray([4.0, 5.0], dtype=jnp.float32),
        representation=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        action=jnp.asarray(1, dtype=jnp.int32),
        representation_revision_words=jnp.asarray([0, 3], dtype=jnp.uint32),
        lifecycle_identity_words=jnp.asarray([1, 2, 3, 4], dtype=jnp.uint32),
        decision_identity_words=jnp.asarray([5, 6, 7, 8], dtype=jnp.uint32),
    )
    assert bool(cached.cache_applied)
    return cached.state, cached.receipt


def _transition(
    producer: CausalStateObjectiveTargetProducer,
    state: CausalStateObjectiveTargetProducerState,
    **updates: jax.Array,
) -> CausalStateObjectiveAcceptedTransition:
    values: dict[str, jax.Array] = {
        "next_observation": jnp.asarray([7.0, 8.0], dtype=jnp.float32),
        "next_representation": jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        "next_representation_revision_words": jnp.asarray([0, 4], dtype=jnp.uint32),
        "reward": jnp.asarray(2.0, dtype=jnp.float32),
        "discount": jnp.asarray(0.8, dtype=jnp.float32),
        "terminated": jnp.asarray(False, dtype=jnp.bool_),
        "truncated": jnp.asarray(False, dtype=jnp.bool_),
        "bootstrap_valid": jnp.asarray(True, dtype=jnp.bool_),
    }
    values.update(updates)
    return producer.bind_accepted_transition(state, **values)


def test_config_is_strict_nonpromoting_and_owns_the_nested_objectives() -> None:
    producer = _producer()
    payload = producer.to_config()
    restored = CausalStateObjectiveTargetProducer.from_config(payload)
    assert restored.to_config() == payload
    assert payload["evidence_level"] == "L0"
    assert payload["outcome_status"] == "not_assessed"
    assert payload["scientific_promotion_allowed"] is False
    assert payload["target_authority"] == "learner-owned-causal-real-transition"
    assert payload["cumulant_mode"] == "environment_reward"
    assert payload["arbitrary_cumulant_causal_derivation_claimed"] is False

    malformed = dict(payload)
    malformed["scientific_promotion_allowed"] = True
    with pytest.raises(ValueError):
        CausalStateObjectiveTargetProducer.from_config(malformed)


def test_real_transition_derives_detached_targets_and_updates_separate_heads() -> None:
    producer = _producer()
    state, receipt = _started(producer)
    transition = _transition(producer, state)
    before = state.objectives_state
    result = producer.update(state, receipt, transition)

    assert bool(result.update_applied)
    chex.assert_trees_all_close(result.targets.next_observation, jnp.asarray([7.0, 8.0]))
    chex.assert_trees_all_close(result.targets.next_latent, jnp.asarray([3.0, 4.0]))
    assert float(result.targets.reward) == pytest.approx(2.0)
    assert not bool(result.targets.terminated)
    assert float(result.targets.discount) == pytest.approx(0.8)
    assert float(result.targets.effective_continuation) == pytest.approx(0.8)
    chex.assert_trees_all_close(
        result.targets.gvf_targets,
        jnp.asarray([2.0, 3.6, 7.04], dtype=jnp.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    assert float(result.targets.current_value) == pytest.approx(5.5)
    assert float(result.targets.bootstrap_value) == pytest.approx(11.5)
    assert float(result.targets.control_value_target) == pytest.approx(11.2)
    assert float(result.targets.selected_action_advantage_target) == pytest.approx(5.7)
    assert int(result.targets.inverse_action_label) == 1
    assert bool(result.targets.inverse_pair_valid)
    chex.assert_trees_all_close(result.objective_gvf_targets, result.targets.gvf_targets)
    assert not bool(result.state.pending_valid)
    assert not bool(result.state.objectives_state.pending_valid)
    assert not jnp.array_equal(
        result.state.objectives_state.observation_weights,
        before.observation_weights,
    )
    assert not jnp.array_equal(
        result.state.objectives_state.value_weights,
        before.value_weights,
    )

    fresh_state, fresh_receipt = _started(producer)

    def targets_from_reward(reward: jax.Array) -> jax.Array:
        bound = _transition(producer, fresh_state, reward=reward)
        update = producer.update(fresh_state, fresh_receipt, bound)
        return (
            update.targets.reward
            + update.targets.control_value_target
            + jnp.sum(update.targets.gvf_targets)
        )

    assert float(jax.grad(targets_from_reward)(jnp.asarray(2.0, dtype=jnp.float32))) == 0.0


def test_terminal_and_truncation_bootstrap_semantics_are_explicit() -> None:
    producer = _producer()
    terminal_state, terminal_receipt = _started(producer)
    terminal = _transition(
        producer,
        terminal_state,
        discount=jnp.asarray(0.9, dtype=jnp.float32),
        terminated=jnp.asarray(True, dtype=jnp.bool_),
        bootstrap_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    terminal_result = producer.update(terminal_state, terminal_receipt, terminal)
    assert bool(terminal_result.update_applied)
    assert float(terminal_result.targets.discount) == pytest.approx(0.9)
    assert float(terminal_result.targets.effective_continuation) == 0.0
    assert float(terminal_result.targets.control_value_target) == pytest.approx(2.0)
    chex.assert_trees_all_close(
        terminal_result.targets.gvf_targets,
        jnp.full((3,), 2.0, dtype=jnp.float32),
    )

    truncated_state, truncated_receipt = _started(producer)
    truncated = _transition(
        producer,
        truncated_state,
        truncated=jnp.asarray(True, dtype=jnp.bool_),
    )
    truncated_result = producer.update(truncated_state, truncated_receipt, truncated)
    assert bool(truncated_result.update_applied)
    assert not bool(truncated_result.targets.terminated)
    assert float(truncated_result.targets.effective_continuation) == pytest.approx(0.8)

    invalid_state, invalid_receipt = _started(producer)
    invalid = _transition(
        producer,
        invalid_state,
        truncated=jnp.asarray(True, dtype=jnp.bool_),
        bootstrap_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    rejected = producer.update(invalid_state, invalid_receipt, invalid)
    chex.assert_trees_all_equal(rejected.state, invalid_state)
    assert not bool(rejected.update_applied)


def test_stale_tampered_and_nonfinite_inputs_roll_back_bit_exactly() -> None:
    producer = _producer()
    state, receipt = _started(producer)
    transition = _transition(producer, state)
    tampered = dataclasses.replace(  # type: ignore[type-var]
        transition,
        reward=jnp.asarray(3.0, dtype=jnp.float32),
    )
    rejected = producer.update(state, receipt, tampered)
    chex.assert_trees_all_equal(rejected.state, state)
    assert not bool(rejected.transition_content_valid)

    nonfinite = producer.bind_accepted_transition(
        state,
        next_observation=jnp.asarray([jnp.nan, 8.0], dtype=jnp.float32),
        next_representation=jnp.asarray([3.0, 4.0], dtype=jnp.float32),
        next_representation_revision_words=jnp.asarray([0, 4], dtype=jnp.uint32),
        reward=jnp.asarray(2.0, dtype=jnp.float32),
        discount=jnp.asarray(0.8, dtype=jnp.float32),
        terminated=jnp.asarray(False, dtype=jnp.bool_),
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        bootstrap_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    rejected = producer.update(state, receipt, nonfinite)
    chex.assert_trees_all_equal(rejected.state, state)
    assert not bool(rejected.source_valid)

    stale = dataclasses.replace(  # type: ignore[type-var]
        transition,
        next_representation_revision_words=jnp.asarray([0, 2], dtype=jnp.uint32),
    )
    rejected = producer.update(state, receipt, stale)
    chex.assert_trees_all_equal(rejected.state, state)
    assert not bool(rejected.representation_revision_valid)


def test_arbitrary_cumulant_requires_content_bound_nonzero_provenance() -> None:
    producer = _producer(cumulant_mode="bound_optional")
    state, receipt = _started(producer)
    transition = _transition(producer, state)
    cumulant = producer.bind_optional_cumulant(
        state,
        value=jnp.asarray(5.0, dtype=jnp.float32),
        source_revision_words=jnp.asarray([0, 9], dtype=jnp.uint32),
        provenance_words=jnp.asarray([10, 11, 12, 13], dtype=jnp.uint32),
    )
    result = producer.update(state, receipt, transition, cumulant)
    assert bool(result.update_applied)
    assert float(result.targets.cumulant) == pytest.approx(5.0)
    chex.assert_trees_all_close(
        result.targets.gvf_targets,
        jnp.asarray([5.0, 6.6, 10.04], dtype=jnp.float32),
        rtol=1e-6,
        atol=1e-6,
    )

    state, receipt = _started(producer)
    transition = _transition(producer, state)
    cumulant = producer.bind_optional_cumulant(
        state,
        value=jnp.asarray(5.0, dtype=jnp.float32),
        source_revision_words=jnp.asarray([0, 9], dtype=jnp.uint32),
        provenance_words=jnp.asarray([10, 11, 12, 13], dtype=jnp.uint32),
    )
    tampered = dataclasses.replace(  # type: ignore[type-var]
        cumulant,
        value=jnp.asarray(6.0, dtype=jnp.float32),
    )
    rejected = producer.update(state, receipt, transition, tampered)
    chex.assert_trees_all_equal(rejected.state, state)
    assert not bool(rejected.cumulant_valid)


def test_exact_clocks_resource_partition_and_checkpoint(tmp_path: Path) -> None:
    producer = _producer()
    state = producer.init(jr.key(3))
    budget = producer.resource_budget(state)
    assert budget.total_state_nbytes == measure_causal_state_objective_target_state_nbytes(state)
    assert budget.objectives_state_nbytes + budget.producer_state_nbytes == (
        budget.total_state_nbytes
    )

    state, receipt = _started(producer)
    result = producer.update(state, receipt, _transition(producer, state))
    assert bool(result.update_applied)
    checkpoint = tmp_path / "causal-targets"
    save_causal_state_objective_target_checkpoint(producer, result.state, checkpoint)
    restored_producer, restored_state = load_causal_state_objective_target_checkpoint(checkpoint)
    assert restored_producer.to_config() == producer.to_config()
    chex.assert_trees_all_equal(restored_state, result.state)

    exhausted = cast(Any, producer.init(jr.key(4))).replace(
        decision_words=jnp.full((2,), jnp.uint32(2**32 - 1)),
        objectives_state=cast(Any, producer.init(jr.key(4)).objectives_state).replace(
            decision_words=jnp.full((2,), jnp.uint32(2**32 - 1))
        ),
    )
    cached = producer.cache_decision(
        exhausted,
        observation=jnp.asarray([4.0, 5.0], dtype=jnp.float32),
        representation=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        action=jnp.asarray(1, dtype=jnp.int32),
        representation_revision_words=jnp.asarray([0, 3], dtype=jnp.uint32),
        lifecycle_identity_words=jnp.asarray([1, 2, 3, 4], dtype=jnp.uint32),
        decision_identity_words=jnp.asarray([5, 6, 7, 8], dtype=jnp.uint32),
    )
    chex.assert_trees_all_equal(cached.state, exhausted)
    assert not bool(cached.cache_applied)


def test_scan_has_eager_jit_and_checkpoint_split_determinism() -> None:
    producer = _producer()
    inputs = CausalStateObjectiveTargetScanInputs(  # type: ignore[call-arg]
        current_observations=jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        current_representations=jnp.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=jnp.float32),
        actions=jnp.asarray([0, 1], dtype=jnp.int32),
        current_representation_revision_words=jnp.asarray([[0, 1], [0, 3]], dtype=jnp.uint32),
        lifecycle_identity_words=jnp.asarray([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=jnp.uint32),
        decision_identity_words=jnp.asarray([[2, 2, 2, 1], [2, 2, 2, 2]], dtype=jnp.uint32),
        next_observations=jnp.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=jnp.float32),
        next_representations=jnp.asarray([[0.4, 0.5], [0.6, 0.7]], dtype=jnp.float32),
        next_representation_revision_words=jnp.asarray([[0, 2], [0, 4]], dtype=jnp.uint32),
        rewards=jnp.asarray([1.0, -0.2], dtype=jnp.float32),
        discounts=jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        terminated=jnp.asarray([False, True], dtype=jnp.bool_),
        truncated=jnp.asarray([False, False], dtype=jnp.bool_),
        bootstrap_valid=jnp.asarray([True, False], dtype=jnp.bool_),
        optional_cumulants=jnp.zeros((2,), dtype=jnp.float32),
        optional_cumulant_available=jnp.zeros((2,), dtype=jnp.bool_),
        cumulant_source_revision_words=jnp.zeros((2, 2), dtype=jnp.uint32),
        cumulant_provenance_words=jnp.zeros((2, 4), dtype=jnp.uint32),
    )
    initial = producer.init(jr.key(8))
    eager = run_causal_state_objective_target_scan(producer, initial, inputs)
    compiled = jax.jit(
        lambda source, arrays: run_causal_state_objective_target_scan(producer, source, arrays)
    )(initial, inputs)
    chex.assert_trees_all_close(eager, compiled, rtol=1e-6, atol=1e-7)
    assert bool(jnp.all(eager.cache_applied))
    assert bool(jnp.all(eager.update_applied))
    chex.assert_trees_all_equal(
        eager.state.transition_words,
        jnp.asarray([0, 2], dtype=jnp.uint32),
    )
