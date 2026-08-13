# mypy: disable-error-code="attr-defined,call-arg"
"""Compiled and scan integration contracts for ensemble short rollouts."""

from __future__ import annotations

from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.ensemble_short_rollouts import (
    EnsembleShortRolloutConfig,
    EnsembleShortRolloutPlanner,
    EnsembleShortRolloutState,
    RealStateRolloutAnchor,
    RolloutPolicyValueAuthority,
    RolloutSelectionMode,
)
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

pytestmark = pytest.mark.integration


def _system(
    mode: RolloutSelectionMode = "policy_directed",
) -> tuple[
    EnsembleShortRolloutPlanner,
    WorldModelEnsembleState,
    RolloutPolicyValueAuthority,
    EnsembleShortRolloutState,
]:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=0.95,
        hidden_sizes=(),
        step_size=0.05,
        sparsity=0.0,
        use_layer_norm=False,
        error_decay=0.8,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        progress_warmup_steps=2,
        change_calibration_steps=2,
        max_input_magnitude=1_000.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    ensemble = WorldModelEnsemble(
        WorldModelEnsembleConfig(
            model=model,
            signal_estimator=signals,
            ensemble_size=2,
            bootstrap_probability=0.5,
            residual_variance_warmup_steps=1,
            residual_variance_floor=1.0e-6,
        )
    )
    model_state = ensemble.init(jr.key(1, impl="threefry2x32"))
    members = []
    for member_index, member in enumerate(model_state.member_states):
        learner = member.learner_state
        weights = []
        biases = []
        for head_index in range(4):
            weight = jnp.zeros_like(learner.head_params.weights[head_index])
            output0 = (0.0, 0.0, 1.0, 0.5)[head_index]
            output1 = (
                (0.0, 0.0, 1.0, 0.5)
                if member_index == 0
                else (0.0, 0.0, 2.0, 0.5)
            )[head_index]
            weight = weight.at[0, 2].set(output0)
            weight = weight.at[0, 3].set(output1)
            weights.append(weight)
            biases.append(jnp.zeros((1,), dtype=jnp.float32))
        heads = learner.head_params.replace(
            weights=tuple(weights),
            biases=tuple(biases),
        )
        members.append(member.replace(learner_state=learner.replace(head_params=heads)))
    model_state = cast(
        WorldModelEnsembleState,
        model_state.replace(member_states=tuple(members)),
    )
    config = EnsembleShortRolloutConfig(
        selection_mode=mode,
        rollout_horizon=2,
        rollout_budget=1,
        require_residual_proxy_ready=False,
        max_epistemic_disagreement=100.0,
        max_residual_variance=100.0,
        max_proposal_calls=8,
        max_rollout_attempts=8,
        max_imagined_steps=16,
    )
    planner = EnsembleShortRolloutPlanner(ensemble, config)
    revision = jnp.asarray([0, 1], dtype=jnp.uint32)
    authority = planner.bind_authority(
        policy_weights=jnp.zeros((2, 2), dtype=jnp.float32),
        policy_bias=jnp.asarray([0.0, 0.0], dtype=jnp.float32),
        value_weights=jnp.zeros((2,), dtype=jnp.float32),
        value_bias=jnp.asarray(4.0, dtype=jnp.float32),
        action_support_counts=jnp.asarray([10, 10], dtype=jnp.int32),
        source_revision_words=revision,
        model_state=model_state,
        policy_revision_words=revision,
        value_revision_words=revision,
    )
    state = planner.init(
        jr.key(7, impl="threefry2x32"),
        model_state,
        authority,
    )
    return planner, model_state, authority, state


def _materialize_keys(tree: object) -> object:
    def materialize(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(materialize, tree)


@pytest.mark.parametrize("mode", ["policy_directed", "uncertainty_directed"])
def test_eager_and_explicit_jit_parity_for_both_selection_modes(
    mode: RolloutSelectionMode,
) -> None:
    planner, model_state, authority, state = _system(mode)
    anchor = planner.bind_real_anchor(
        jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        jnp.asarray([0, 1], dtype=jnp.uint32),
        authority,
    )
    with jax.disable_jit():
        eager = planner.propose(
            state,
            model_state,
            authority,
            anchor,
        )
    compiled = jax.jit(
        lambda lane_state: planner.propose(
            lane_state,
            model_state,
            authority,
            anchor,
        )
    )(state)
    chex.assert_trees_all_equal(
        _materialize_keys(compiled),
        _materialize_keys(eager),
    )


def test_lax_scan_matches_ordered_python_composition() -> None:
    planner, model_state, authority, state = _system()
    anchors = tuple(
        planner.bind_real_anchor(
            jnp.asarray([float(index), 0.0], dtype=jnp.float32),
            jnp.asarray([0, index], dtype=jnp.uint32),
            authority,
        )
        for index in (1, 2, 3)
    )
    stacked = cast(
        RealStateRolloutAnchor,
        jax.tree.map(lambda *values: jnp.stack(values), *anchors),
    )

    def step(
        lane_state: EnsembleShortRolloutState,
        anchor: RealStateRolloutAnchor,
    ) -> tuple[EnsembleShortRolloutState, jax.Array]:
        result = planner.propose(
            lane_state,
            model_state,
            authority,
            anchor,
        )
        return result.state, result.proposals.root_returns

    scanned_state, scanned_returns = jax.jit(
        lambda initial, items: jax.lax.scan(step, initial, items)
    )(state, stacked)
    python_state = state
    python_returns = []
    with jax.disable_jit():
        for anchor in anchors:
            python_state, root_return = step(python_state, anchor)
            python_returns.append(root_return)
    chex.assert_trees_all_equal(
        _materialize_keys(scanned_state),
        _materialize_keys(python_state),
    )
    chex.assert_trees_all_equal(
        scanned_returns,
        jnp.stack(python_returns),
    )
