# mypy: disable-error-code="attr-defined,call-arg,no-untyped-call,type-var"
"""JIT/scan/resume contracts for the live authority-free STOMP observer.

These are L0 mechanism tests only.  They make no scientific, curation,
dispatch, replacement, or promotion claim.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Callable, Iterator

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.option_lifecycle_audit import (
    OptionLifecycleAudit,
    OptionLifecycleAuditConfig,
    option_semantic_digest,
)
from alberta_framework.core.options import STOMPAgent, STOMPConfig, SubtaskSpec
from alberta_framework.core.stomp_option_lifecycle import (
    STOMPOptionLifecycle,
    STOMPOptionLifecycleState,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SOURCE = option_semantic_digest({"source": "stomp-scan-integration"})
REPRESENTATION = option_semantic_digest({"representation": "obs2-v1"})
LIFECYCLE = jnp.asarray([0x51DE, 0xA17E], dtype=jnp.uint32)


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()


def _wrapper() -> STOMPOptionLifecycle:
    stomp = STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=100.0,
                    pseudo_reward_scale=1.0,
                    max_option_steps=16,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_step_size=0.05,
            base_avg_reward_step_size=0.01,
            base_trace_decay=0.5,
            option_step_size=0.05,
            option_avg_reward_step_size=0.01,
            option_trace_decay=0.5,
            option_gamma=0.9,
            option_model_decay=0.0,
            option_model_step_size=0.5,
            option_planning_backups_per_step=0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )
    audit = OptionLifecycleAudit(
        OptionLifecycleAuditConfig(
            n_options=1,
            n_contexts=1,
            outcome_dim=2,
            fixed_horizon=1,
            maintenance_budget=1,
            signature_scales=(1.0,) * 7,
            initiation_opportunity_floor=1,
            completion_evidence_floor=1,
            model_error_evidence_floor=1,
            comparison_treatment_evidence_floor=1,
            comparison_primitive_evidence_floor=1,
            signature_evidence_floor_per_context=1,
            redundancy_shared_context_floor=1,
            max_observations=32,
            max_planning_uses_per_observation=4,
            max_compute_cost_per_observation=10.0,
        )
    )
    return STOMPOptionLifecycle(stomp, audit)


def _active_initial_state(wrapper: STOMPOptionLifecycle) -> STOMPOptionLifecycleState:
    state = wrapper.init(
        jr.key(12),
        source_digest=SOURCE,
        representation_digest=REPRESENTATION,
        lifecycle_id=LIFECYCLE,
    )
    option_head = wrapper.stomp_agent.config.n_primitive_actions
    learner = state.stomp_state.base_learner_state
    weights = tuple(
        jnp.full(
            (1, 2),
            10.0 if index == option_head else -10.0,
            dtype=jnp.float32,
        )
        for index in range(wrapper.stomp_agent.config.n_total_actions)
    )
    learner = learner.replace(
        head_params=learner.head_params.replace(
            weights=weights,
            biases=tuple(
                jnp.zeros((1,), dtype=jnp.float32)
                for _ in range(wrapper.stomp_agent.config.n_total_actions)
            ),
        )
    )
    stomp = state.stomp_state.replace(base_learner_state=learner)
    state = wrapper._with_checksum(dataclasses.replace(state, stomp_state=stomp))
    started = wrapper.start(state, jnp.asarray([1.0, 1.0], dtype=jnp.float32))
    assert bool(started.applied)
    assert int(started.state.stomp_state.executing_option) == 0
    return started.state


def _scan_step(
    wrapper: STOMPOptionLifecycle,
) -> Callable[
    [STOMPOptionLifecycleState, tuple[jax.Array, jax.Array, jax.Array]],
    tuple[STOMPOptionLifecycleState, tuple[jax.Array, ...]],
]:
    def step(
        state: STOMPOptionLifecycleState,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[STOMPOptionLifecycleState, tuple[jax.Array, ...]]:
        reward, observation, discount = inputs
        result = wrapper.update(
            state,
            reward,
            observation,
            discount,
            context=jnp.int32(0),
        )
        facts = (
            result.transaction_applied,
            result.audit_applied,
            result.audit_unavailable_noop,
            result.audit_error,
            result.post_step_words,
        )
        return result.state, facts

    return step


def test_eager_jit_scan_raw_stomp_and_mid_option_resume_are_exact() -> None:
    wrapper = _wrapper()
    initial = _active_initial_state(wrapper)
    inputs = (
        jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        jnp.asarray(
            [[1.5, 1.0], [2.0, 1.0], [2.5, 1.0], [3.0, 1.0]],
            dtype=jnp.float32,
        ),
        jnp.asarray([0.9, 0.8, 0.7, 0.6], dtype=jnp.float32),
    )
    step = _scan_step(wrapper)

    eager_state = initial
    raw_state = initial.stomp_state
    eager_facts: list[tuple[jax.Array, ...]] = []
    eager_states: list[STOMPOptionLifecycleState] = []
    for reward, observation, discount in zip(*inputs, strict=True):
        eager_state, facts = step(eager_state, (reward, observation, discount))
        raw = wrapper.stomp_agent.update(raw_state, reward, observation, discount)
        assert bool(raw.update_applied)
        raw_state = raw.state
        eager_facts.append(facts)
        eager_states.append(eager_state)

    chex.assert_trees_all_equal(eager_state.stomp_state, raw_state)
    assert int(eager_state.stomp_state.option_steps) == 4
    assert int(eager_state.audit_state.active_steps) == 4
    assert int(eager_state.audit_state.observation_count) == 4

    scanned_state, scanned_facts = jax.jit(
        lambda state, xs: jax.lax.scan(step, state, xs)
    )(initial, inputs)
    chex.assert_trees_all_equal(scanned_state, eager_state)
    expected_facts = jax.tree.map(lambda *values: jnp.stack(values), *eager_facts)
    chex.assert_trees_all_equal(scanned_facts, expected_facts)
    assert bool(jnp.all(scanned_facts[0]))
    assert bool(jnp.all(scanned_facts[1]))
    assert not bool(jnp.any(scanned_facts[2]))

    midpoint = eager_states[1]
    payload = wrapper.checkpoint_payload(midpoint)
    resumed = wrapper.restore_checkpoint(
        copy.deepcopy(payload),
        expected_source_digest=SOURCE,
        expected_representation_digest=REPRESENTATION,
        expected_lifecycle_id=LIFECYCLE,
    )
    resumed_state, resumed_facts = jax.jit(
        lambda state, xs: jax.lax.scan(step, state, xs)
    )(resumed, tuple(value[2:] for value in inputs))
    chex.assert_trees_all_equal(resumed_state, eager_state)
    expected_resumed = jax.tree.map(
        lambda *values: jnp.stack(values),
        *eager_facts[2:],
    )
    chex.assert_trees_all_equal(resumed_facts, expected_resumed)
    np.testing.assert_array_equal(
        resumed_state.stomp_state.step_words,
        jnp.asarray([0, 4], dtype=jnp.uint32),
    )
