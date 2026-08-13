"""Full-batch protected TD ownership beneath the Kondo actor."""

from __future__ import annotations

import copy
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.kondo_protected_td import (
    KONDO_PROTECTED_TD_SCHEMA,
    KondoProtectedTDBatch,
    KondoProtectedTDConfig,
    KondoProtectedTDLearner,
    KondoProtectedTDParameters,
    kondo_protected_td_backward_kernel,
)

pytestmark = pytest.mark.unit


def _config() -> KondoProtectedTDConfig:
    return KondoProtectedTDConfig(
        batch_size=2,
        feature_dim=3,
        action_count=2,
        learning_rate=0.1,
        max_updates=10,
    )


def _parameters() -> KondoProtectedTDParameters:
    return KondoProtectedTDParameters(
        reward_weight=jnp.asarray((0.2, -0.1, 0.3), dtype=jnp.float32),
        reward_bias=jnp.asarray(0.05, dtype=jnp.float32),
        cost_weight=jnp.asarray((-0.2, 0.4, 0.1), dtype=jnp.float32),
        cost_bias=jnp.asarray(-0.03, dtype=jnp.float32),
    )


def _batch() -> KondoProtectedTDBatch:
    return KondoProtectedTDBatch(
        current_features=jnp.asarray(
            ((1.0, 0.5, -0.25), (-0.5, 0.25, 1.5)),
            dtype=jnp.float32,
        ),
        next_features=jnp.asarray(
            ((0.75, -0.5, 0.5), (0.25, 1.0, -0.75)),
            dtype=jnp.float32,
        ),
        actions=jnp.asarray((1, 0), dtype=jnp.int32),
        decision_identities=jnp.asarray(
            ((1, 2, 3, 4), (5, 6, 7, 8)),
            dtype=jnp.uint32,
        ),
        rewards=jnp.asarray((0.8, -0.35), dtype=jnp.float32),
        discounts=jnp.asarray((0.9, 0.75), dtype=jnp.float32),
        costs=jnp.asarray((0.1, 0.4), dtype=jnp.float32),
    )


def _prediction(features: jax.Array, weight: jax.Array, bias: jax.Array) -> jax.Array:
    return features @ weight + bias


def test_config_is_strict_and_declares_no_actor_or_safety_authority() -> None:
    config = _config()
    payload = json.loads(json.dumps(config.to_config()))

    assert KONDO_PROTECTED_TD_SCHEMA == "alberta.kondo-protected-td.v1"
    assert KondoProtectedTDConfig.from_config(payload) == config
    assert payload["full_batch_rows"] == 2
    assert payload["reward_target"] == "reward-plus-discount-times-detached-next-V"
    assert payload["cost_target"] == "cost-plus-discount-times-detached-next-C"
    assert payload["actor_gradient_gated"] is False
    assert payload["cost_gradient_gated"] is False
    assert payload["random_draws_per_update"] == 0
    assert payload["safety_authority"] is False
    assert payload["promotion_authority"] is False

    with pytest.raises(ValueError, match="batch_size"):
        KondoProtectedTDConfig(
            batch_size=0,
            feature_dim=3,
            action_count=2,
            learning_rate=0.1,
            max_updates=10,
        )
    altered = dict(payload)
    altered["actor_gradient_gated"] = True
    with pytest.raises(ValueError, match="noncanonical|semantics"):
        KondoProtectedTDConfig.from_config(altered)


def test_two_head_td_targets_and_gradients_use_every_row_exactly_once() -> None:
    config = _config()
    learner = KondoProtectedTDLearner(config)
    parameters = _parameters()
    state = learner.init(parameters)
    batch = _batch()
    result = learner.step(state, batch)

    reward_baseline = _prediction(
        batch.current_features,
        parameters.reward_weight,
        parameters.reward_bias,
    )
    reward_bootstrap = _prediction(
        batch.next_features,
        parameters.reward_weight,
        parameters.reward_bias,
    )
    reward_target = batch.rewards + batch.discounts * reward_bootstrap
    cost_baseline = _prediction(
        batch.current_features,
        parameters.cost_weight,
        parameters.cost_bias,
    )
    cost_bootstrap = _prediction(
        batch.next_features,
        parameters.cost_weight,
        parameters.cost_bias,
    )
    cost_target = batch.costs + batch.discounts * cost_bootstrap
    reward_error = reward_baseline - reward_target
    cost_error = cost_baseline - cost_target
    expected_reward_weight_gradient = jnp.mean(
        reward_error[:, None] * batch.current_features,
        axis=0,
    )
    expected_cost_weight_gradient = jnp.mean(
        cost_error[:, None] * batch.current_features,
        axis=0,
    )

    np.testing.assert_array_equal(result.reward_baseline, reward_baseline)
    np.testing.assert_array_equal(result.reward_bootstrap, reward_bootstrap)
    np.testing.assert_array_equal(result.return_targets, reward_target)
    np.testing.assert_array_equal(result.cost_baseline, cost_baseline)
    np.testing.assert_array_equal(result.cost_bootstrap, cost_bootstrap)
    np.testing.assert_array_equal(result.cost_targets, cost_target)
    np.testing.assert_allclose(
        result.gradient.reward_weight,
        expected_reward_weight_gradient,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.gradient.cost_weight,
        expected_cost_weight_gradient,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.gradient.reward_bias,
        jnp.mean(reward_error),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.gradient.cost_bias,
        jnp.mean(cost_error),
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        result.state.parameters.reward_weight,
        parameters.reward_weight
        - jnp.asarray(config.learning_rate, dtype=jnp.float32)
        * result.gradient.reward_weight,
    )
    assert int(result.state.update_count) == 1
    assert int(result.full_batch_rows) == config.batch_size
    assert bool(result.full_batch_backward_executed)
    assert bool(result.transaction_applied)


def test_actor_payload_is_full_batch_owned_and_has_no_joy_alias() -> None:
    learner = KondoProtectedTDLearner(_config())
    batch = _batch()
    result = learner.step(learner.init(_parameters()), batch)

    np.testing.assert_array_equal(
        result.actor_inputs.critic_features,
        result.batch.current_features,
    )
    np.testing.assert_array_equal(
        result.actor_inputs.safety_features,
        result.batch.current_features,
    )
    np.testing.assert_array_equal(
        result.actor_inputs.baseline_predictions,
        result.reward_baseline,
    )
    np.testing.assert_array_equal(
        result.actor_inputs.return_targets,
        result.return_targets,
    )
    assert result.batch is batch
    assert not hasattr(result, "sparks_joy")
    assert not hasattr(result, "executed_actor_backward_mask")


def test_backward_kernel_matches_step_and_bootstrap_is_detached() -> None:
    parameters = _parameters()
    batch = _batch()
    eager = kondo_protected_td_backward_kernel(parameters, batch)
    compiled = jax.jit(kondo_protected_td_backward_kernel)(parameters, batch)

    np.testing.assert_array_equal(compiled.total_loss, eager.total_loss)
    np.testing.assert_array_equal(
        compiled.gradient.reward_weight,
        eager.gradient.reward_weight,
    )
    np.testing.assert_array_equal(
        compiled.gradient.cost_weight,
        eager.gradient.cost_weight,
    )

    changed_next = batch.replace(
        next_features=batch.next_features.at[0, 0].add(
            jnp.asarray(0.25, dtype=jnp.float32)
        )
    )
    changed = kondo_protected_td_backward_kernel(parameters, changed_next)
    assert not np.array_equal(changed.return_targets, eager.return_targets)
    # Targets change, but no gradient path enters through their bootstrap
    # prediction; a stop-gradient witness is explicit in the JAXPR.
    jaxpr = str(jax.make_jaxpr(kondo_protected_td_backward_kernel)(parameters, batch))
    assert "stop_gradient" in jaxpr


def test_checkpoint_resource_and_invalid_input_contracts_fail_closed() -> None:
    config = _config()
    learner = KondoProtectedTDLearner(config)
    state = learner.step(learner.init(_parameters()), _batch()).state
    payload = learner.checkpoint_payload(state)
    restored_learner, restored_state = KondoProtectedTDLearner.from_checkpoint_payload(
        copy.deepcopy(payload)
    )

    assert restored_learner.to_config() == learner.to_config()
    np.testing.assert_array_equal(
        restored_state.parameters.reward_weight,
        state.parameters.reward_weight,
    )
    np.testing.assert_array_equal(
        restored_state.parameters.cost_weight,
        state.parameters.cost_weight,
    )
    assert int(restored_state.update_count) == int(state.update_count)
    resources = learner.resource_declaration(state)
    assert resources.full_batch_rows == config.batch_size
    assert resources.parameter_count == 2 * (config.feature_dim + 1)
    assert resources.maximum_backwards_per_update == 1
    assert resources.random_draws_per_update == 0
    assert resources.persistent_state_nbytes > 0
    assert resources.checkpoint_supported is True

    tampered = copy.deepcopy(payload)
    tampered["state"]["parameters"]["cost_weight"]["float32_bits"][0] ^= 1
    with pytest.raises(ValueError, match="digest|canonical|invalid"):
        KondoProtectedTDLearner.from_checkpoint_payload(tampered)

    bad_action = _batch().replace(actions=jnp.asarray((2, 0), dtype=jnp.int32))
    with pytest.raises(ValueError, match="actions"):
        learner.step(state, bad_action)
    bad_feature = _batch().replace(
        current_features=_batch().current_features.at[0, 0].set(jnp.nan)
    )
    with pytest.raises(ValueError, match="finite"):
        learner.step(state, bad_feature)

    capped = state.replace(update_count=jnp.asarray(config.max_updates, dtype=jnp.int32))
    capped = learner.reseal_state(capped)
    with pytest.raises(OverflowError, match="max_updates"):
        learner.step(capped, _batch())
