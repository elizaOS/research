"""Source-bound STOMP adoption and live-option masking for OaK."""

from __future__ import annotations

from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.oak import (
    OaKAgent,
    OaKConfig,
    OaKState,
    measure_oak_state_nbytes,
)
from alberta_framework.core.options import STOMPAgent, STOMPConfig, STOMPState, SubtaskSpec

pytestmark = pytest.mark.unit


def _config(
    *,
    epsilon: float = 0.0,
    n_options: int = 2,
    planning_backups: int = 0,
) -> OaKConfig:
    return OaKConfig(
        stomp=STOMPConfig(
            subtask_specs=tuple(
                SubtaskSpec(
                    feature_index=index % 2,
                    threshold=100.0,
                    max_option_steps=3,
                )
                for index in range(n_options)
            ),
            observation_dim=2,
            n_primitive_actions=2,
            epsilon_base=epsilon,
            epsilon_option=0.0,
            option_planning_backups_per_step=planning_backups,
        ),
        utility_ema_decay=0.5,
    )


def _with_constant_head_values(state: OaKState, values: tuple[float, ...]) -> OaKState:
    learner = state.stomp_state.base_learner_state
    assert len(values) == len(learner.head_params.biases)
    head_params = learner.head_params.replace(
        weights=tuple(jnp.zeros_like(weight) for weight in learner.head_params.weights),
        biases=tuple(
            jnp.full_like(bias, value)
            for bias, value in zip(learner.head_params.biases, values, strict=True)
        ),
    )
    return state.replace(
        stomp_state=state.stomp_state.replace(
            base_learner_state=learner.replace(head_params=head_params)
        )
    )


def _trees_all_equal(actual: Any, expected: Any) -> None:
    chex.assert_trees_all_equal(actual, expected)


def test_update_delegates_one_stomp_evaluation_to_exact_adoption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = OaKAgent(_config())
    observation = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    source = agent.start(agent.init(jr.key(7)), observation)
    reward = jnp.asarray(0.3, dtype=jnp.float32)
    next_observation = jnp.asarray([-0.1, 0.8], dtype=jnp.float32)
    discount = jnp.asarray(0.9, dtype=jnp.float32)
    raw = agent.stomp_agent.update(
        source.stomp_state,
        reward,
        next_observation,
        discount,
    )
    adopted = agent.adopt_stomp_update(
        source,
        source_state=source,
        stomp_result=raw,
    )

    calls = 0
    original = STOMPAgent.update

    def counted(self: STOMPAgent, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(STOMPAgent, "update", counted)
    delegated = agent.update(source, reward, next_observation, discount)

    assert calls == 1
    assert bool(adopted.transaction_applied)
    assert bool(adopted.source_state_matches)
    assert bool(adopted.result_clock_binding_valid)
    assert bool(adopted.result_endpoint_binding_valid)
    assert bool(adopted.result_diagnostics_valid)
    assert not bool(adopted.derivation_recomputed)
    assert bool(adopted.caller_authority_required)
    assert not bool(adopted.caller_authenticated)
    _trees_all_equal(adopted.update, delegated)


def test_adoption_has_eager_jit_and_scan_parity() -> None:
    agent = OaKAgent(_config())
    observation = jnp.asarray([0.2, 0.4], dtype=jnp.float32)
    source = agent.start(agent.init(jr.key(11)), observation)
    raw = agent.stomp_agent.update(
        source.stomp_state,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )

    eager = agent.adopt_stomp_update(source, source_state=source, stomp_result=raw)
    compiled = jax.jit(
        lambda current, bound, result: agent.adopt_stomp_update(
            current,
            source_state=bound,
            stomp_result=result,
        )
    )(source, source, raw)

    def body(
        carry: OaKState,
        inputs: tuple[OaKState, Any],
    ) -> tuple[OaKState, Any]:
        bound, result = inputs
        adoption = agent.adopt_stomp_update(
            carry,
            source_state=bound,
            stomp_result=result,
        )
        return adoption.update.state, adoption

    _, scanned = jax.lax.scan(
        body,
        source,
        (
            jax.tree.map(lambda leaf: leaf[None], source),
            jax.tree.map(lambda leaf: leaf[None], raw),
        ),
    )

    _trees_all_equal(compiled, eager)
    _trees_all_equal(jax.tree.map(lambda leaf: leaf[0], scanned), eager)


@pytest.mark.parametrize(
    "tamper",
    ["source", "source_cache", "pre_clock", "endpoint", "diagnostic"],
)
def test_adoption_rejects_stale_or_tampered_provenance(tamper: str) -> None:
    agent = OaKAgent(_config())
    observation = jnp.asarray([0.2, 0.4], dtype=jnp.float32)
    source = agent.start(agent.init(jr.key(13)), observation)
    raw = agent.stomp_agent.update(
        source.stomp_state,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
    )
    bound = source
    if tamper == "source":
        bound = source.replace(utility_ema=source.utility_ema.at[0].set(-0.0))
    elif tamper == "source_cache":
        bound = source.replace(
            stomp_state=source.stomp_state.replace(
                base_last_obs=source.stomp_state.base_last_obs.at[0].add(
                    jnp.float32(1.0)
                )
            )
        )
    elif tamper == "pre_clock":
        raw = raw.replace(pre_step_words=raw.pre_step_words.at[1].add(jnp.uint32(1)))
    elif tamper == "endpoint":
        raw = raw.replace(
            primitive_action=(raw.primitive_action + jnp.int32(1))
            % jnp.int32(agent.config.n_primitive_actions)
        )
    else:
        raw = raw.replace(update_applied=jnp.asarray(False, dtype=jnp.bool_))

    adoption = agent.adopt_stomp_update(
        source,
        source_state=bound,
        stomp_result=raw,
    )

    assert not bool(adoption.transaction_applied)
    assert not bool(adoption.update.update_applied)
    assert not bool(adoption.update.nested_update_applied)
    _trees_all_equal(adoption.update.state, source)


def test_all_true_mask_preserves_legacy_start_update_and_rng_bits() -> None:
    agent = OaKAgent(_config(epsilon=0.35))
    initial = agent.init(jr.key(17))
    observation = jnp.asarray([0.2, -0.4], dtype=jnp.float32)
    all_true = jnp.ones((agent.config.stomp.n_total_actions,), dtype=jnp.bool_)

    legacy_start = agent.start(initial, observation)
    masked_start = agent.start(
        initial,
        observation,
        extended_action_mask=all_true,
    )
    _trees_all_equal(masked_start, legacy_start)
    _trees_all_equal(masked_start.stomp_state.rng_key, legacy_start.stomp_state.rng_key)

    reward = jnp.asarray(0.3, dtype=jnp.float32)
    next_observation = jnp.asarray([-0.1, 0.8], dtype=jnp.float32)
    discount = jnp.asarray(0.9, dtype=jnp.float32)
    legacy_update = agent.update(legacy_start, reward, next_observation, discount)
    masked_update = agent.update(
        legacy_start,
        reward,
        next_observation,
        discount,
        extended_action_mask=all_true,
    )
    _trees_all_equal(masked_update, legacy_update)
    _trees_all_equal(
        masked_update.state.stomp_state.rng_key,
        legacy_update.state.stomp_state.rng_key,
    )


@pytest.mark.parametrize("epsilon", [0.0, 1.0])
def test_cold_option_is_excluded_from_selection_and_real_bootstrap(epsilon: float) -> None:
    agent = OaKAgent(_config(epsilon=epsilon, n_options=1))
    initial = agent.init(jr.key(19))
    low = _with_constant_head_values(initial, (0.25, 0.0, -3.0))
    huge = _with_constant_head_values(initial, (0.25, 0.0, 1.0e6))
    mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    observation = jnp.asarray([0.1, 0.2], dtype=jnp.float32)
    low = agent.start(low, observation, extended_action_mask=mask)
    huge = agent.start(huge, observation, extended_action_mask=mask)

    assert int(low.stomp_state.executing_option) == -1
    assert int(huge.stomp_state.executing_option) == -1
    reward = jnp.asarray(0.4, dtype=jnp.float32)
    next_observation = jnp.asarray([-0.3, 0.7], dtype=jnp.float32)
    discount = jnp.asarray(0.8, dtype=jnp.float32)
    low_update = agent.update(
        low,
        reward,
        next_observation,
        discount,
        extended_action_mask=mask,
    )
    huge_update = agent.update(
        huge,
        reward,
        next_observation,
        discount,
        extended_action_mask=mask,
    )

    _trees_all_equal(huge_update.td_error, low_update.td_error)
    _trees_all_equal(huge_update.average_reward, low_update.average_reward)
    assert int(huge_update.state.stomp_state.executing_option) == -1
    assert int(huge_update.primitive_action) < agent.config.n_primitive_actions


def test_scan_threads_cold_option_masks_without_dispatch() -> None:
    agent = OaKAgent(_config(epsilon=1.0, n_options=1))
    initial = _with_constant_head_values(
        agent.init(jr.key(23)),
        (0.0, 0.0, 1.0e6),
    )
    mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    started = agent.start(
        initial,
        jnp.zeros((2,), dtype=jnp.float32),
        extended_action_mask=mask,
    )
    length = 4

    result = agent.scan(
        started,
        jnp.zeros((length,), dtype=jnp.float32),
        jnp.zeros((length, 2), dtype=jnp.float32),
        jnp.ones((length,), dtype=jnp.float32),
        extended_action_masks=jnp.broadcast_to(mask, (length, 3)),
    )

    assert bool(jnp.all(result.executing_options == -1))
    assert bool(jnp.all(result.primitive_actions < 2))
    assert int(result.state.stomp_state.executing_option) == -1


def test_invalid_dynamic_action_mask_is_an_exact_start_and_update_noop() -> None:
    agent = OaKAgent(_config())
    source = jax.jit(lambda value: value)(agent.init(jr.key(24)))
    invalid = jnp.asarray([False, True, True, True], dtype=jnp.bool_)
    observation = jnp.asarray([0.2, 0.4], dtype=jnp.float32)

    rejected_start = agent.start(
        source,
        observation,
        extended_action_mask=invalid,
    )
    _trees_all_equal(rejected_start, source)

    started = agent.start(source, observation)
    rejected_update = agent.update(
        started,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
        extended_action_mask=invalid,
    )
    assert not bool(rejected_update.update_applied)
    _trees_all_equal(rejected_update.state, started)


def test_cold_completed_model_is_excluded_from_option_planning() -> None:
    agent = OaKAgent(_config(n_options=1, planning_backups=2))
    source = agent.init(jr.key(25))
    models = source.stomp_state.option_models.replace(
        env_return_ema=jnp.asarray([1.0e6], dtype=jnp.float32),
        next_state_weights=jnp.full((1, 2, 2), 1.0e6, dtype=jnp.float32),
        n_completions=jnp.asarray([1], dtype=jnp.int32),
    )
    source = source.replace(
        stomp_state=source.stomp_state.replace(option_models=models)
    )
    mask = jnp.asarray([True, True, False], dtype=jnp.bool_)
    started = agent.start(
        source,
        jnp.asarray([0.1, 0.2], dtype=jnp.float32),
        extended_action_mask=mask,
    )

    result = agent.update(
        started,
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray([-0.3, 0.7], dtype=jnp.float32),
        jnp.asarray(0.8, dtype=jnp.float32),
        extended_action_mask=mask,
    )

    assert bool(result.update_applied)
    assert int(result.planning_backups) == 0


def _rebound_slot(source: STOMPState, slot: int) -> STOMPState:
    policies = source.option_policies.replace(
        q_weights=source.option_policies.q_weights.at[slot].add(jnp.float32(1.0)),
        traces=source.option_policies.traces.at[slot].set(
            jnp.zeros_like(source.option_policies.traces[slot])
        ),
        average_rewards=source.option_policies.average_rewards.at[slot].set(
            jnp.float32(0.25)
        ),
    )
    models = source.option_models.replace(
        cumreward_ema=source.option_models.cumreward_ema.at[slot].set(jnp.float32(0.5)),
        n_completions=source.option_models.n_completions.at[slot].set(jnp.int32(0)),
    )
    learner = source.base_learner_state
    head = 2 + slot
    weights = list(learner.head_params.weights)
    biases = list(learner.head_params.biases)
    weights[head] = weights[head] + jnp.float32(0.75)
    biases[head] = biases[head] - jnp.float32(0.25)
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=tuple(weights),
            biases=tuple(biases),
        )
    )
    return source.replace(
        base_learner_state=learner,
        option_policies=policies,
        option_models=models,
    )


def test_option_slot_rebind_zeroes_only_reset_oak_stats_and_preserves_rng() -> None:
    agent = OaKAgent(_config())
    source = jax.jit(lambda value: value)(agent.init(jr.key(29))).replace(
        execution_counts=jnp.asarray([1, 1], dtype=jnp.int32),
        cumulative_pseudo_rewards=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        utility_ema=jnp.asarray([0.2, 0.3], dtype=jnp.float32),
    )
    rebound = _rebound_slot(source.stomp_state, 0)
    reset = jnp.asarray([True, False], dtype=jnp.bool_)

    eager = agent.rebind_option_slots(source, rebound, reset)
    compiled = jax.jit(agent.rebind_option_slots)(source, rebound, reset)

    assert bool(eager.transaction_applied)
    assert bool(eager.source_quiescent)
    assert bool(eager.clocks_preserved)
    assert bool(eager.policy_rng_preserved)
    assert bool(eager.unchanged_slots_preserved)
    _trees_all_equal(compiled, eager)
    _trees_all_equal(eager.state.stomp_state, rebound)
    _trees_all_equal(eager.state.stomp_state.rng_key, source.stomp_state.rng_key)
    np.testing.assert_array_equal(np.asarray(eager.state.execution_counts), [0, 1])
    np.testing.assert_array_equal(
        np.asarray(eager.state.cumulative_pseudo_rewards),
        [0.0, 2.0],
    )
    _trees_all_equal(
        eager.state.utility_ema,
        jnp.asarray([0.0, 0.3], dtype=jnp.float32),
    )
    assert measure_oak_state_nbytes(eager.state) == measure_oak_state_nbytes(source)


@pytest.mark.parametrize("tamper", ["unchanged_slot", "rng", "clock", "active"])
def test_option_slot_rebind_tampering_is_a_full_noop(tamper: str) -> None:
    agent = OaKAgent(_config())
    source = jax.jit(lambda value: value)(agent.init(jr.key(31))).replace(
        execution_counts=jnp.asarray([1, 1], dtype=jnp.int32),
        cumulative_pseudo_rewards=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        utility_ema=jnp.asarray([0.2, 0.3], dtype=jnp.float32),
    )
    rebound = _rebound_slot(source.stomp_state, 0)
    if tamper == "unchanged_slot":
        rebound = rebound.replace(
            option_policies=rebound.option_policies.replace(
                q_weights=rebound.option_policies.q_weights.at[1, 0, 0].add(
                    jnp.float32(1.0)
                )
            )
        )
    elif tamper == "rng":
        rebound = rebound.replace(rng_key=jr.key(999))
    elif tamper == "clock":
        rebound = rebound.replace(step_words=jnp.asarray([0, 1], dtype=jnp.uint32))
    else:
        source = source.replace(
            stomp_state=source.stomp_state.replace(
                base_last_action=jnp.int32(2),
                last_primitive_action=jnp.int32(0),
                executing_option=jnp.int32(0),
                option_last_intra_action=jnp.int32(0),
            )
        )
        rebound = _rebound_slot(source.stomp_state, 0)

    result = agent.rebind_option_slots(
        source,
        rebound,
        jnp.asarray([True, False], dtype=jnp.bool_),
    )

    assert not bool(result.transaction_applied)
    _trees_all_equal(result.state, source)


def test_external_adoption_resource_budget_is_zero_recompute_and_zero_growth() -> None:
    agent = OaKAgent(_config())
    source = agent.init(jr.key(37))

    budget = agent.external_stomp_adoption_resource_budget(source)

    assert budget.persistent_state_nbytes_before == measure_oak_state_nbytes(source)
    assert budget.persistent_state_nbytes_after == measure_oak_state_nbytes(source)
    assert budget.persistent_state_growth_bytes == 0
    assert budget.stomp_update_evaluations_per_adopt == 0
    assert budget.stomp_update_evaluations_per_delegated_update == 1
    assert budget.derivation_recomputed_on_adopt is False
    assert budget.source_result_integrity_checked is True
    assert budget.caller_authority_required is True
    assert budget.caller_authenticated is False
