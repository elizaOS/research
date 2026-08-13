# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Contracts for the leakage-safe latent-context expert learner."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.latent_context_experts import (
    LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA,
    LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES,
    LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES,
    LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA,
    LATENT_CONTEXT_EXPERT_RESULT_SCHEMA,
    LATENT_CONTEXT_EXPERT_STATE_SCHEMA,
    LatentContextExpertConfig,
    LatentContextExpertLearner,
    LatentContextExpertState,
    measure_latent_context_expert_state_nbytes,
    run_latent_context_expert_arrays,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _words(value: int) -> jax.Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _telemetry(value: int) -> jax.Array:
    return jnp.asarray(min(value, _INT32_MAX), dtype=jnp.int32)


def _learner(**overrides: Any) -> LatentContextExpertLearner:
    values: dict[str, Any] = {
        "input_dim": 2,
        "output_dim": 1,
        "max_experts": 3,
    }
    values.update(overrides)
    return LatentContextExpertLearner(LatentContextExpertConfig(**values))


def _state_at(learner: LatentContextExpertLearner, value: int) -> LatentContextExpertState:
    return learner.init().replace(
        step_count=_telemetry(value),
        step_words=_words(value),
    )


def _assert_rejected(result: Any, source: LatentContextExpertState) -> None:
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, source)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)
    chex.assert_trees_all_equal(result.prediction, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.error, jnp.zeros((1,), dtype=jnp.float32))
    chex.assert_trees_all_equal(result.expert_update_mask, jnp.zeros((3,), dtype=jnp.bool_))
    assert int(result.selected_next_expert) == -1


def test_design_record_credits_context_inference_without_novelty_claim() -> None:
    record = _learner().design_record

    assert record.conceptual_novelty_claimed is False
    assert record.prior_mechanism == "ContextInference active-only-freeze law"
    assert record.prior_module == "alberta_framework.core.context_inference"
    assert any("predict-before-outcome" in item for item in record.integration_scope)
    assert any("generic regression" in item for item in record.integration_scope)


def test_cached_owner_predicts_before_target_and_target_selects_only_next_owner() -> None:
    learner = _learner(input_dim=1, max_experts=2, selective_gating=True)
    initial = learner.init()
    state = initial.replace(
        params=initial.params.replace(
            expert_weights=jnp.asarray([[[1.0]], [[-1.0]]], dtype=jnp.float32),
        ),
        active_expert=jnp.asarray(0, dtype=jnp.int32),
    )
    observation = jnp.asarray([1.0], dtype=jnp.float32)
    cache = learner.predict(state, observation)
    result = learner.update(state, cache, jnp.asarray([-0.5], dtype=jnp.float32))

    assert bool(cache.valid)
    assert int(cache.owner_active_expert) == 0
    chex.assert_trees_all_equal(cache.prediction, jnp.asarray([1.0], dtype=jnp.float32))
    assert bool(result.update_applied)
    assert int(result.pre_update_owner) == 0
    assert int(result.evidence_best_expert) == 1
    assert int(result.selected_next_expert) == 1
    assert int(result.state.active_expert) == 1
    chex.assert_trees_all_equal(result.prediction, cache.prediction)
    chex.assert_trees_all_equal(result.error, jnp.asarray([-1.5], dtype=jnp.float32))
    chex.assert_trees_all_equal(
        result.expert_update_mask,
        jnp.asarray([False, True]),
    )
    # The target may route the next update, but cannot rewrite the current
    # prequential prediction or touch the dormant outgoing expert.
    chex.assert_trees_all_equal(
        result.state.params.expert_weights[0],
        state.params.expert_weights[0],
    )
    assert not np.array_equal(
        np.asarray(result.state.params.expert_weights[1]),
        np.asarray(state.params.expert_weights[1]),
    )


def test_exact_evidence_tie_retains_cached_current_owner() -> None:
    learner = _learner(input_dim=1, max_experts=2, selective_gating=True)
    state = learner.init().replace(active_expert=jnp.asarray(1, dtype=jnp.int32))
    cache = learner.predict(state, jnp.asarray([0.5], dtype=jnp.float32))
    result = learner.update(state, cache, jnp.asarray([1.0], dtype=jnp.float32))

    assert bool(result.update_applied)
    chex.assert_trees_all_equal(result.expert_losses, jnp.asarray([1.0, 1.0]))
    assert int(result.evidence_best_expert) == 1
    assert int(result.selected_next_expert) == 1
    chex.assert_trees_all_equal(result.expert_update_mask, jnp.asarray([False, True]))


def test_defaults_off_ablation_has_equal_state_work_and_one_commit() -> None:
    ordinary = _learner(selective_gating=True)
    ablation = _learner(selective_gating=False)
    ordinary_initial = ordinary.init()
    ablation_initial = ablation.init()

    chex.assert_trees_all_equal(ordinary_initial, ablation_initial)
    assert LatentContextExpertConfig(input_dim=1).selective_gating is False
    assert ordinary.resource_record() == ablation.resource_record()
    resources = ordinary.resource_record()
    assert resources.maximum_expert_predictions_per_update == 6
    assert resources.maximum_expert_losses_per_update == 3
    assert resources.maximum_candidate_gradients_per_update == 3
    assert resources.maximum_expert_subtree_commits_per_update == 1

    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    cache = ablation.predict(ablation_initial, observation)
    result = ablation.update(
        ablation_initial,
        cache,
        jnp.asarray([0.5], dtype=jnp.float32),
    )
    assert bool(result.update_applied)
    assert int(jnp.sum(result.expert_update_mask)) == 1
    assert int(result.selected_next_expert) == int(cache.owner_active_expert) == 0


def test_every_nonselected_expert_subtree_is_bit_exact_per_transaction() -> None:
    learner = _learner(selective_gating=True)
    initial = learner.init()
    state = initial.replace(
        params=initial.params.replace(
            expert_weights=jnp.asarray(
                [[[1.0], [0.0]], [[0.0], [1.0]], [[-1.0], [0.0]]],
                dtype=jnp.float32,
            ),
            expert_biases=jnp.asarray([[0.1], [0.2], [0.3]], dtype=jnp.float32),
        )
    )
    cache = learner.predict(state, jnp.asarray([1.0, 0.0], dtype=jnp.float32))
    result = learner.update(state, cache, jnp.asarray([-0.5], dtype=jnp.float32))

    assert bool(result.update_applied)
    selected = int(result.selected_next_expert)
    assert int(jnp.sum(result.expert_update_mask)) == 1
    for expert in range(learner.config.max_experts):
        if expert == selected:
            continue
        chex.assert_trees_all_equal(
            result.state.params.expert_weights[expert],
            state.params.expert_weights[expert],
        )
        chex.assert_trees_all_equal(
            result.state.params.expert_biases[expert],
            state.params.expert_biases[expert],
        )


def test_eager_jit_and_scan_are_deterministic_and_fixed_shape() -> None:
    learner = _learner(selective_gating=True)
    state = learner.init()
    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    target = jnp.asarray([0.5], dtype=jnp.float32)
    with jax.disable_jit():
        eager_cache = learner.predict(state, observation)
        eager = learner.update(state, eager_cache, target)
    compiled_cache = learner.predict(state, observation)
    compiled = learner.update(state, compiled_cache, target)
    chex.assert_trees_all_close(eager_cache, compiled_cache, rtol=1.0e-6, atol=1.0e-7)
    chex.assert_trees_all_close(eager, compiled, rtol=1.0e-6, atol=1.0e-7)

    observations = jnp.stack((observation, -observation, observation))
    targets = jnp.asarray([[0.5], [-0.5], [0.5]], dtype=jnp.float32)
    scanned = run_latent_context_expert_arrays(learner, observations, targets, state=state)
    chex.assert_shape(scanned.predictions, (3, 1))
    chex.assert_shape(scanned.expert_losses, (3, 3))
    chex.assert_shape(scanned.expert_update_mask, (3, 3))
    chex.assert_shape(scanned.pre_update_owner, (3,))
    chex.assert_shape(scanned.selected_next_expert, (3,))
    assert bool(jnp.all(scanned.update_applied))
    assert int(scanned.state.step_count) == 3
    chex.assert_trees_all_equal(
        jnp.sum(scanned.expert_update_mask, axis=1),
        jnp.ones((3,), dtype=jnp.int32),
    )


def test_exact_clock_carry_saturation_and_terminal_rollback() -> None:
    learner = _learner()
    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    target = jnp.asarray([0.5], dtype=jnp.float32)

    carry_state = _state_at(learner, _UINT32_MAX)
    carry = learner.update(carry_state, learner.predict(carry_state, observation), target)
    chex.assert_trees_all_equal(carry.pre_step_words, _words(_UINT32_MAX))
    chex.assert_trees_all_equal(carry.post_step_words, _words(1 << 32))
    chex.assert_trees_all_equal(carry.state.step_count, _telemetry(1 << 32))

    boundary_state = _state_at(learner, _INT32_MAX - 1)
    boundary = learner.update(
        boundary_state,
        learner.predict(boundary_state, observation),
        target,
    )
    beyond = learner.update(
        boundary.state,
        learner.predict(boundary.state, observation),
        target,
    )
    assert int(boundary.state.step_count) == _INT32_MAX
    assert int(beyond.state.step_count) == _INT32_MAX
    chex.assert_trees_all_equal(beyond.state.step_words, _words(_INT32_MAX + 1))

    terminal = _state_at(learner, _UINT64_MAX)
    exhausted = learner.update(terminal, learner.predict(terminal, observation), target)
    assert bool(exhausted.source_state_valid)
    assert not bool(exhausted.lifetime_capacity_available)
    _assert_rejected(exhausted, terminal)


def test_nonfinite_cache_state_target_and_candidate_roll_back_atomically() -> None:
    learner = _learner()
    source = learner.init()
    observation = jnp.asarray([0.25, -0.75], dtype=jnp.float32)
    target = jnp.asarray([0.5], dtype=jnp.float32)
    cache = learner.predict(source, observation)

    tampered = cache.replace(prediction=cache.prediction + 1.0)
    tampered_result = learner.update(source, tampered, target)
    assert not bool(tampered_result.cache_prediction_exact)
    _assert_rejected(tampered_result, source)

    stale_state = source.replace(
        step_count=jnp.asarray(1, dtype=jnp.int32),
        step_words=_words(1),
    )
    stale = learner.update(stale_state, cache, target)
    assert not bool(stale.cache_owner_valid)
    _assert_rejected(stale, stale_state)

    finite_tamper = source.replace(
        params=source.params.replace(
            expert_biases=source.params.expert_biases.at[2, 0].set(0.25)
        )
    )
    owner_tamper = learner.update(finite_tamper, cache, target)
    assert not bool(owner_tamper.cache_owner_valid)
    _assert_rejected(owner_tamper, finite_tamper)

    bad_target = learner.update(
        source,
        cache,
        jnp.asarray([jnp.nan], dtype=jnp.float32),
    )
    assert not bool(bad_target.target_valid)
    _assert_rejected(bad_target, source)

    corrupt_state = source.replace(
        params=source.params.replace(
            expert_biases=source.params.expert_biases.at[0, 0].set(jnp.inf)
        )
    )
    corrupt_cache = learner.predict(corrupt_state, observation)
    corrupt = learner.update(corrupt_state, corrupt_cache, target)
    assert not bool(corrupt.source_state_valid)
    _assert_rejected(corrupt, corrupt_state)

    maximum = np.finfo(np.float32).max
    huge_observation = jnp.full((2,), maximum, dtype=jnp.float32)
    huge_target = jnp.asarray([maximum], dtype=jnp.float32)
    huge_cache = learner.predict(source, huge_observation)
    candidate = learner.update(source, huge_cache, huge_target)
    _assert_rejected(candidate, source)


def test_strict_shapes_dtypes_config_and_resources() -> None:
    learner = _learner()
    state = learner.init()
    with pytest.raises(ValueError, match="observation"):
        learner.predict(state, jnp.zeros((3,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="observation"):
        learner.predict(state, jnp.zeros((2,), dtype=jnp.float16))
    cache = learner.predict(state, jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(ValueError, match="target"):
        learner.update(state, cache, jnp.zeros((), dtype=jnp.float32))
    with pytest.raises(TypeError, match="target"):
        learner.update(state, cache, jnp.zeros((1,), dtype=jnp.int32))
    with pytest.raises(ValueError, match="step_words"):
        learner.predict(
            state.replace(step_words=jnp.zeros((3,), dtype=jnp.uint32)),
            cache.observation,
        )
    with pytest.raises(TypeError, match="step_words"):
        learner.predict(
            state.replace(step_words=jnp.zeros((2,), dtype=jnp.int32)),
            cache.observation,
        )
    with pytest.raises(ValueError, match="grad_clip"):
        _learner(grad_clip=-1.0)

    payload = learner.config.to_config()
    assert payload["schema"] == LATENT_CONTEXT_EXPERT_CONFIG_SCHEMA
    assert payload["state_schema"] == LATENT_CONTEXT_EXPERT_STATE_SCHEMA
    assert payload["result_schema"] == LATENT_CONTEXT_EXPERT_RESULT_SCHEMA
    assert payload["resource_schema"] == LATENT_CONTEXT_EXPERT_RESOURCE_SCHEMA
    assert LatentContextExpertConfig.from_config(payload) == learner.config
    with pytest.raises(ValueError, match="fields"):
        LatentContextExpertConfig.from_config({**payload, "extra": 1})

    measured = measure_latent_context_expert_state_nbytes(state)
    resources = learner.resource_record(state)
    assert resources.state_nbytes == measured
    assert resources.parameter_nbytes + 16 == measured
    assert resources.exact_lifetime_identity_nbytes == (
        LATENT_CONTEXT_EXPERT_EXACT_LIFETIME_NBYTES
    )
    assert resources.lifetime_counter_nbytes == LATENT_CONTEXT_EXPERT_LIFETIME_COUNTER_NBYTES
    assert resources.replay_capacity == 0
    assert resources.maximum_stored_examples == 0
    assert resources.persistent_capacity_growth == 0
