# mypy: disable-error-code="attr-defined,call-arg"
"""Exact-lifetime and nested-transaction contracts for option search control."""

from __future__ import annotations

from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework
import alberta_framework.core as alberta_core
from alberta_framework.core.multi_head_learner import MULTI_HEAD_MLP_STATE_SCHEMA
from alberta_framework.core.option_search_control import (
    OPTION_SEARCH_CONTROL_BASE_LEARNER_STATE_SCHEMA,
    OPTION_SEARCH_CONTROL_CONFIG_SCHEMA,
    OPTION_SEARCH_CONTROL_EXACT_IDENTITY_NBYTES,
    OptionSearchControl,
    OptionSearchControlConfig,
    migrate_legacy_option_search_control_config,
)
from alberta_framework.core.options import (
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
)

pytestmark = pytest.mark.unit

_ANCHOR = jnp.asarray((1.0, 0.0), dtype=jnp.float32)
_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _agent(*, base_step_size: float = 0.25) -> STOMPAgent:
    return STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(
                    feature_index=0,
                    threshold=1.0e6,
                    max_option_steps=8,
                ),
            ),
            observation_dim=2,
            n_primitive_actions=2,
            base_step_size=base_step_size,
            base_avg_reward_step_size=0.0,
            base_hidden_sizes=(),
            option_planning_backups_per_step=0,
            epsilon_base=0.0,
            epsilon_option=0.0,
        )
    )


def _supported_state(
    agent: STOMPAgent,
    *,
    target: float = 1.0,
) -> STOMPState:
    state = agent.start(agent.init(jr.key(7)), _ANCHOR)
    learner = state.base_learner_state.replace(
        head_params=state.base_learner_state.head_params.replace(
            weights=tuple(
                jnp.zeros_like(weight)
                for weight in state.base_learner_state.head_params.weights
            ),
            biases=tuple(
                jnp.zeros_like(bias)
                for bias in state.base_learner_state.head_params.biases
            ),
        )
    )
    models = state.option_models.replace(
        cumreward_ema=jnp.asarray((100.0,), dtype=jnp.float32),
        env_return_ema=jnp.asarray((target,), dtype=jnp.float32),
        duration_ema=jnp.ones((1,), dtype=jnp.float32),
        baseline_mass_ema=jnp.ones((1,), dtype=jnp.float32),
        discount_ema=jnp.zeros((1,), dtype=jnp.float32),
        next_state_weights=jnp.zeros((1, 2, 2), dtype=jnp.float32),
        n_completions=jnp.ones((1,), dtype=jnp.int32),
    )
    return cast(
        STOMPState,
        state.replace(
            base_learner_state=learner,
            option_models=models,
            base_average_reward=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )


def _with_base_clock(
    state: STOMPState,
    words: tuple[int, int],
) -> STOMPState:
    exact = (words[0] << 32) | words[1]
    telemetry = min(exact, _INT32_MAX)
    return cast(
        STOMPState,
        state.replace(
            base_learner_state=state.base_learner_state.replace(
                step_count=jnp.asarray(telemetry, dtype=jnp.int32),
                step_words=jnp.asarray(words, dtype=jnp.uint32),
            )
        ),
    )


def test_exact_clock_carries_with_eager_jit_and_scan_parity() -> None:
    agent = _agent()
    state = _with_base_clock(
        _supported_state(agent),
        (0, _UINT32_MAX - 1),
    )
    two_backup = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=2),
    )

    eager = two_backup.apply(state, _ANCHOR)
    compiled = jax.jit(two_backup.apply)(state, _ANCHOR)
    chex.assert_trees_all_equal(eager, compiled)
    chex.assert_trees_all_equal(
        eager.diagnostics.nested_pre_step_words,
        jnp.asarray(
            ((0, _UINT32_MAX - 1), (0, _UINT32_MAX)),
            dtype=jnp.uint32,
        ),
    )
    chex.assert_trees_all_equal(
        eager.diagnostics.nested_post_step_words,
        jnp.asarray(
            ((0, _UINT32_MAX), (1, 0)),
            dtype=jnp.uint32,
        ),
    )
    chex.assert_trees_all_equal(
        eager.state.base_learner_state.step_words,
        jnp.asarray((1, 0), dtype=jnp.uint32),
    )
    assert int(eager.state.base_learner_state.step_count) == _INT32_MAX
    assert bool(jnp.all(eager.diagnostics.nested_update_applied))
    assert bool(jnp.all(eager.diagnostics.nested_transaction_authenticated))
    assert bool(jnp.all(eager.diagnostics.applied))

    one_backup = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=1),
    )

    def body(
        carry: STOMPState,
        _: jax.Array,
    ) -> tuple[STOMPState, tuple[jax.Array, jax.Array]]:
        result = one_backup.apply(carry, _ANCHOR)
        return result.state, (
            result.diagnostics.base_pre_step_words,
            result.diagnostics.base_post_step_words,
        )

    scan_state, (scan_pre, scan_post) = jax.lax.scan(
        body,
        state,
        jnp.arange(2, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(scan_state, eager.state)
    chex.assert_trees_all_equal(
        scan_pre,
        eager.diagnostics.nested_pre_step_words,
    )
    chex.assert_trees_all_equal(
        scan_post,
        eager.diagnostics.nested_post_step_words,
    )


def test_terminal_exact_identity_is_a_bit_exact_noop() -> None:
    agent = _agent()
    state = _with_base_clock(
        _supported_state(agent),
        (_UINT32_MAX, _UINT32_MAX),
    )
    controller = OptionSearchControl(
        agent,
        OptionSearchControlConfig(backup_budget=2),
    )

    result = jax.jit(controller.apply)(state, _ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    assert bool(result.diagnostics.state_counters_valid)
    assert not bool(result.diagnostics.base_update_capacity_available)
    assert not bool(result.diagnostics.planner_inputs_valid)
    assert not bool(jnp.any(result.diagnostics.nested_update_applied))
    assert not bool(jnp.any(result.diagnostics.applied))
    terminal = jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32)
    chex.assert_trees_all_equal(result.diagnostics.base_pre_step_words, terminal)
    chex.assert_trees_all_equal(result.diagnostics.base_post_step_words, terminal)


class _RefusingLearner:
    """Delegate that exposes a staged state but explicitly refuses commit."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def predict(self, state: Any, observation: jax.Array) -> jax.Array:
        return cast(jax.Array, self._delegate.predict(state, observation))

    def update(
        self,
        state: Any,
        observation: jax.Array,
        targets: jax.Array,
    ) -> Any:
        result = self._delegate.update(state, observation, targets)
        return result.replace(
            state=state,
            post_step_words=state.step_words,
            update_applied=jnp.asarray(False, dtype=jnp.bool_),
        )


def test_nested_learner_refusal_prevents_wrapper_commit() -> None:
    agent = _agent()
    state = _supported_state(agent)
    agent._base_learner = cast(Any, _RefusingLearner(agent.base_learner))

    result = OptionSearchControl(agent).apply(state, _ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    assert bool(result.diagnostics.candidate_update_finite[0])
    assert bool(result.diagnostics.trace_isolation_preserved[0])
    assert not bool(result.diagnostics.nested_update_applied[0])
    assert bool(result.diagnostics.nested_transaction_authenticated[0])
    assert not bool(result.diagnostics.applied[0])
    chex.assert_trees_all_equal(
        result.diagnostics.base_pre_step_words,
        result.diagnostics.base_post_step_words,
    )


def test_nonfinite_nested_candidate_is_rejected_before_wrapper_commit() -> None:
    agent = _agent(base_step_size=4.0)
    state = _supported_state(agent, target=3.0e38)

    result = OptionSearchControl(agent).apply(state, _ANCHOR)

    chex.assert_trees_all_equal(result.state, state)
    # The hardened nested learner now rejects its own non-finite proposal and
    # returns the finite source state. The wrapper authenticates that refusal
    # and preserves it instead of treating a staged non-finite tree as applied.
    assert not bool(result.diagnostics.nested_update_applied[0])
    assert bool(result.diagnostics.nested_transaction_authenticated[0])
    assert bool(result.diagnostics.candidate_update_finite[0])
    assert not bool(result.diagnostics.applied[0])
    chex.assert_trees_all_equal(
        result.diagnostics.base_pre_step_words,
        result.diagnostics.base_post_step_words,
    )


def test_clock_corruption_and_malformed_identity_fail_closed() -> None:
    agent = _agent()
    state = _supported_state(agent)
    corrupted = cast(
        STOMPState,
        state.replace(
            base_learner_state=state.base_learner_state.replace(
                step_words=jnp.asarray((1, 0), dtype=jnp.uint32),
            )
        ),
    )
    refused = OptionSearchControl(agent).apply(corrupted, _ANCHOR)
    chex.assert_trees_all_equal(refused.state, corrupted)
    assert not bool(refused.diagnostics.state_counters_valid)
    assert not bool(refused.diagnostics.planner_inputs_valid)

    malformed = cast(
        STOMPState,
        state.replace(
            base_learner_state=state.base_learner_state.replace(
                step_words=jnp.zeros((1,), dtype=jnp.uint32),
            )
        ),
    )
    unavailable = OptionSearchControl(agent).apply(malformed, _ANCHOR)
    assert unavailable.state is malformed
    assert not bool(unavailable.diagnostics.base_state_static_contract_valid)
    assert not bool(
        unavailable.diagnostics.base_exact_identity_static_contract_valid
    )


def test_v1_config_migration_resource_identity_and_public_exports() -> None:
    config = OptionSearchControlConfig(backup_budget=2, min_model_completions=3)
    payload = config.to_config()
    assert payload["schema"] == OPTION_SEARCH_CONTROL_CONFIG_SCHEMA
    assert (
        payload["base_learner_state_schema"]
        == OPTION_SEARCH_CONTROL_BASE_LEARNER_STATE_SCHEMA
        == MULTI_HEAD_MLP_STATE_SCHEMA
    )
    assert OptionSearchControlConfig.from_config(payload) == config

    legacy = {
        "schema": "alberta.option-search-control.config.v1",
        "type": "OptionSearchControlConfig",
        "mechanism_status": "development_mechanism_only",
        "scientific_promotion_allowed": False,
        "backup_budget": 2,
        "min_model_completions": 3,
    }
    with pytest.raises(ValueError, match="explicit migration"):
        OptionSearchControlConfig.from_config(legacy)
    assert migrate_legacy_option_search_control_config(legacy) == config
    with pytest.raises(ValueError, match="fields are not exact"):
        migrate_legacy_option_search_control_config({**legacy, "extra": 0})
    with pytest.raises(ValueError, match="base-learner state schema"):
        OptionSearchControlConfig.from_config(
            {**payload, "base_learner_state_schema": "unsupported"}
        )

    agent = _agent()
    controller = OptionSearchControl(agent, config)
    budget = controller.resource_budget
    assert budget.persistent_state_bytes == 0
    assert (
        budget.nested_exact_lifetime_identity_bytes
        == OPTION_SEARCH_CONTROL_EXACT_IDENTITY_NBYTES
        == 8
    )
    diagnostics = controller.apply(_supported_state(agent), _ANCHOR).diagnostics
    measured_diagnostic_nbytes = sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(diagnostics)
        if isinstance(leaf, jax.Array)
    )
    assert (
        measured_diagnostic_nbytes
        == budget.max_diagnostic_payload_bytes_per_call
    )
    for namespace in (alberta_framework, alberta_core):
        assert (
            namespace.OPTION_SEARCH_CONTROL_BASE_LEARNER_STATE_SCHEMA
            == OPTION_SEARCH_CONTROL_BASE_LEARNER_STATE_SCHEMA
        )
        assert (
            namespace.OPTION_SEARCH_CONTROL_EXACT_IDENTITY_NBYTES
            == OPTION_SEARCH_CONTROL_EXACT_IDENTITY_NBYTES
        )
        assert (
            namespace.migrate_legacy_option_search_control_config
            is migrate_legacy_option_search_control_config
        )
        for name in (
            "OPTION_SEARCH_CONTROL_BASE_LEARNER_STATE_SCHEMA",
            "OPTION_SEARCH_CONTROL_EXACT_IDENTITY_NBYTES",
            "migrate_legacy_option_search_control_config",
        ):
            assert name in namespace.__all__
