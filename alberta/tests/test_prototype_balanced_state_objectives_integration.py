"""Process, boundary, scan, and checkpoint contracts for the WP3 adapter."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.balanced_state_objectives import (
    BalancedStateObjectives,
    BalancedStateObjectivesConfig,
)
from alberta_framework.core.checkpoints import save_checkpoint
from alberta_framework.core.oak import OaKConfig
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentConfig,
    PrototypeTransition,
)
from alberta_framework.core.prototype_balanced_state_objectives import (
    PrototypeBalancedObjectivesState,
    PrototypeBalancedStateObjectives,
    load_prototype_balanced_objectives_checkpoint,
    save_prototype_balanced_objectives_checkpoint,
)
from alberta_framework.core.state_builder import (
    OnlineGatedStateBuilder,
    OnlineGatedStateBuilderConfig,
    OnlineGatedStateBuilderState,
)

pytestmark = pytest.mark.integration

RAW_DIM = 2
FEATURE_DIM = 3
N_ACTIONS = 2


@pytest.fixture(autouse=True)
def _clear_jax_caches_after_test() -> Iterator[None]:
    yield
    jax.clear_caches()


def _adapter() -> PrototypeBalancedStateObjectives:
    builder = OnlineGatedStateBuilderConfig(
        observation_dim=RAW_DIM,
        n_actions=N_ACTIONS,
        hidden_dim=1,
        include_raw_observation=True,
        step_size=0.04,
        gradient_clip=8.0,
        initialization_scale=0.1,
    )
    prototype = PrototypeAgent(
        PrototypeAgentConfig(
            oak=OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=(SubtaskSpec(feature_index=0),),
                    observation_dim=FEATURE_DIM,
                    n_primitive_actions=N_ACTIONS,
                    base_hidden_sizes=(),
                    base_step_size=0.02,
                    option_step_size=0.02,
                    epsilon_base=0.0,
                    epsilon_option=0.0,
                )
            ),
            state_builder=builder,
        )
    )
    objectives = BalancedStateObjectives(
        BalancedStateObjectivesConfig(
            representation_dim=FEATURE_DIM,
            n_actions=N_ACTIONS,
            gvf_discounts=(0.2, 0.7, 0.95),
            gvf_step_size=0.03,
            inverse_step_size=0.04,
            initialization_scale=0.08,
            representation_gradient_clip=10.0,
        )
    )
    return PrototypeBalancedStateObjectives(prototype, objectives)


def _transition(
    state: PrototypeBalancedObjectivesState,
    next_observation: jax.Array,
    *,
    reward: jax.Array,
    discount: jax.Array,
    terminated: jax.Array | None = None,
    next_decision_observation: jax.Array | None = None,
) -> PrototypeTransition:
    prototype = state.prototype_state
    terminated_value = (
        jnp.asarray(False, dtype=jnp.bool_) if terminated is None else terminated
    )
    return PrototypeTransition(
        observation=prototype.current_raw_observation,
        action=prototype.current_action,
        decision_id=prototype.current_decision_id,
        reward=reward,
        discount=discount,
        terminated=terminated_value,
        truncated=jnp.asarray(False, dtype=jnp.bool_),
        next_observation=next_observation,
        next_decision_observation=(
            next_observation
            if next_decision_observation is None
            else next_decision_observation
        ),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_allclose(left: object, right: object) -> None:
    left = _materialize_keys(left)
    right = _materialize_keys(right)
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_leaf)
        right_array = np.asarray(right_leaf)
        if np.issubdtype(left_array.dtype, np.inexact):
            np.testing.assert_allclose(left_array, right_array, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(left_array, right_array)


def test_boundary_scores_bootstrap_owner_before_autoreset_next_decision() -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(jr.key(20)),
        jnp.asarray([0.2, -0.4], dtype=jnp.float32),
    ).state
    final_observation = jnp.asarray([0.7, 0.1], dtype=jnp.float32)
    reset_observation = jnp.asarray([-0.8, 0.5], dtype=jnp.float32)
    source_builder = state.prototype_state.state_builder_state
    assert type(source_builder) is OnlineGatedStateBuilderState
    builder = adapter.builder
    assert type(builder) is OnlineGatedStateBuilder
    expected_bootstrap = builder.update_with_status(
        source_builder,
        final_observation,
        state.prototype_state.current_action,
        jnp.asarray(0.4, dtype=jnp.float32),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    result = adapter.update_transition(
        state,
        _transition(
            state,
            final_observation,
            reward=jnp.asarray(0.4, dtype=jnp.float32),
            discount=jnp.asarray(0.0, dtype=jnp.float32),
            terminated=jnp.asarray(True, dtype=jnp.bool_),
            next_decision_observation=reset_observation,
        ),
    )
    assert bool(result.update_applied)
    np.testing.assert_allclose(
        result.bootstrap_representation,
        expected_bootstrap.representation,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.objective_update.next_representation_revision_words,
        [0, 2],
    )
    np.testing.assert_array_equal(
        result.state.prototype_state.observation_event_words,
        [0, 3],
    )
    np.testing.assert_array_equal(
        result.state.objectives_state.pending_representation_revision_words,
        [0, 3],
    )
    np.testing.assert_allclose(
        result.state.prototype_state.current_raw_observation,
        reset_observation,
        rtol=0.0,
        atol=0.0,
    )
    assert not np.array_equal(
        np.asarray(result.bootstrap_representation),
        np.asarray(result.state.prototype_state.current_representation),
    )


def _run_scan(
    adapter: PrototypeBalancedStateObjectives,
    initial: PrototypeBalancedObjectivesState,
    observations: jax.Array,
    rewards: jax.Array,
    discounts: jax.Array,
) -> tuple[PrototypeBalancedObjectivesState, tuple[jax.Array, ...]]:
    def body(
        state: PrototypeBalancedObjectivesState,
        inputs: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[PrototypeBalancedObjectivesState, tuple[jax.Array, ...]]:
        observation, reward, discount = inputs
        result = adapter.update_transition(
            state,
            _transition(
                state,
                observation,
                reward=reward,
                discount=discount,
            ),
        )
        return result.state, (
            result.update_applied,
            result.objective_update.balanced_loss,
            result.action,
            result.post_transaction_words,
        )

    return jax.lax.scan(body, initial, (observations, rewards, discounts))


def test_eager_jit_and_scan_preserve_state_clocks_and_outputs() -> None:
    adapter = _adapter()
    initial = adapter.start(
        adapter.init(jr.key(21)),
        jnp.asarray([0.1, -0.2], dtype=jnp.float32),
    ).state
    observations = jnp.asarray(
        [[0.2, 0.3], [-0.4, 0.5], [0.6, -0.1]],
        dtype=jnp.float32,
    )
    rewards = jnp.asarray([0.2, -0.1, 0.4], dtype=jnp.float32)
    discounts = jnp.asarray([0.9, 0.8, 1.0], dtype=jnp.float32)
    with jax.disable_jit():
        eager = _run_scan(adapter, initial, observations, rewards, discounts)
    compiled = jax.jit(_run_scan, static_argnums=(0,))(
        adapter,
        initial,
        observations,
        rewards,
        discounts,
    )
    _assert_tree_allclose(eager, compiled)
    final_state, outputs = compiled
    applied, _losses, _actions, transaction_words = outputs
    assert bool(jnp.all(applied))
    np.testing.assert_array_equal(
        transaction_words,
        [[0, 1], [0, 2], [0, 3]],
    )
    np.testing.assert_array_equal(final_state.transaction_words, [0, 3])
    assert bool(adapter.state_valid(final_state))


def test_checkpoint_resume_preserves_pending_owner_and_config(tmp_path: Path) -> None:
    adapter = _adapter()
    state = adapter.start(
        adapter.init(
            jr.key(22),
            lifecycle_id=jnp.asarray([11, 13], dtype=jnp.uint32),
        ),
        jnp.asarray([0.3, 0.6], dtype=jnp.float32),
    ).state
    first = adapter.update_transition(
        state,
        _transition(
            state,
            jnp.asarray([-0.2, 0.4], dtype=jnp.float32),
            reward=jnp.asarray(0.1, dtype=jnp.float32),
            discount=jnp.asarray(0.95, dtype=jnp.float32),
        ),
    )
    assert bool(first.update_applied)
    checkpoint = tmp_path / "prototype-balanced"
    save_prototype_balanced_objectives_checkpoint(adapter, first.state, checkpoint)
    restored_adapter, restored_state = load_prototype_balanced_objectives_checkpoint(
        checkpoint
    )
    assert restored_adapter.to_config() == adapter.to_config()
    chex.assert_trees_all_equal(
        _materialize_keys(restored_state),
        _materialize_keys(first.state),
    )

    next_observation = jnp.asarray([0.8, -0.7], dtype=jnp.float32)
    uninterrupted = adapter.update_transition(
        first.state,
        _transition(
            first.state,
            next_observation,
            reward=jnp.asarray(-0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.85, dtype=jnp.float32),
        ),
    )
    resumed = restored_adapter.update_transition(
        restored_state,
        _transition(
            restored_state,
            next_observation,
            reward=jnp.asarray(-0.2, dtype=jnp.float32),
            discount=jnp.asarray(0.85, dtype=jnp.float32),
        ),
    )
    assert bool(uninterrupted.update_applied)
    assert bool(resumed.update_applied)
    _assert_tree_allclose(uninterrupted.state, resumed.state)

    malformed = tmp_path / "malformed-prototype-balanced"
    save_checkpoint(
        first.state,
        malformed,
        metadata={"schema": "missing-required-fields"},
    )
    with pytest.raises(ValueError, match="manifest is not exact"):
        load_prototype_balanced_objectives_checkpoint(malformed)
