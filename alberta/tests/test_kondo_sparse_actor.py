# mypy: disable-error-code="attr-defined,call-arg,type-var"
"""Unit contracts for the real Kondo-selected nonlinear actor backward."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.kondo_gate import KONDO_GATE_SCHEMA, KondoGateConfig
from alberta_framework.core.kondo_sparse_actor import (
    KONDO_SPARSE_ACTOR_SCHEMA,
    KondoActorBackwardBatch,
    KondoActorParameters,
    KondoActorProtectedInputs,
    KondoSparseActor,
    KondoSparseActorBatch,
    KondoSparseActorConfig,
    KondoSparseActorState,
    kondo_sparse_actor_source_sha256,
)

pytestmark = pytest.mark.unit


def _config(
    *,
    target_rate: float = 0.5,
    max_screenings: int = 100,
    mode: str = "top_k_rate",
    sparse_capacity: int | None = None,
) -> KondoSparseActorConfig:
    return KondoSparseActorConfig(
        feature_dim=3,
        hidden_dim=4,
        action_count=3,
        critic_dim=2,
        safety_dim=2,
        learning_rate=0.025,
        gate=KondoGateConfig(
            batch_size=6,
            mode=mode,  # type: ignore[arg-type]
            target_rate=target_rate,
            sparse_capacity=sparse_capacity,
            price=0.0,
            temperature=0.1,
            max_screenings=max_screenings,
        ),
    )


def _parameters() -> KondoActorParameters:
    return KondoActorParameters(
        hidden_weight=jnp.asarray(
            [
                [0.20, -0.10, 0.05, 0.30],
                [-0.15, 0.25, 0.10, -0.20],
                [0.05, 0.15, -0.30, 0.10],
            ],
            dtype=jnp.float32,
        ),
        hidden_bias=jnp.asarray([0.01, -0.02, 0.03, 0.04], dtype=jnp.float32),
        output_weight=jnp.asarray(
            [
                [0.30, -0.20, 0.10],
                [-0.10, 0.25, 0.15],
                [0.20, 0.05, -0.25],
                [-0.15, 0.10, 0.20],
            ],
            dtype=jnp.float32,
        ),
        output_bias=jnp.asarray([0.02, -0.03, 0.01], dtype=jnp.float32),
    )


def _actor(
    config: KondoSparseActorConfig | None = None,
) -> tuple[KondoSparseActor, KondoSparseActorState]:
    actor = KondoSparseActor(config or _config())
    state = actor.init(_parameters(), jr.key(11, impl="threefry2x32"))
    return actor, state


def _batch(
    actor: KondoSparseActor,
    state: KondoSparseActorState,
    *,
    forced: jax.Array | None = None,
    valid: jax.Array | None = None,
    advantages: jax.Array | None = None,
) -> KondoSparseActorBatch:
    features = jnp.asarray(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [0.5, -0.5, 1.0],
            [-1.0, 0.5, 0.25],
            [0.25, 1.0, -1.0],
            [0.75, -0.25, 0.5],
        ],
        dtype=jnp.float32,
    )
    actions = jnp.asarray([0, 1, 2, 0, 1, 2], dtype=jnp.int32)
    behavior = actor.behavior_log_probability(state, features, actions)
    desired_advantage = (
        jnp.asarray([1.0, -1.0, 2.0, 0.5, 1.0, -0.25], dtype=jnp.float32)
        if advantages is None
        else advantages
    )
    baseline = jnp.asarray([0.2, 0.1, -0.1, 0.3, 0.0, -0.2], dtype=jnp.float32)
    protected = KondoActorProtectedInputs(
        critic_features=jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 10.0,
        baseline_predictions=baseline,
        return_targets=baseline + desired_advantage,
        safety_features=jnp.arange(12, dtype=jnp.float32).reshape(6, 2) / 20.0,
    )
    revision = jnp.full(
        (6,),
        state.policy_revision,
        dtype=jnp.int32,
    )
    return KondoSparseActorBatch(
        actor_features=features,
        actions=actions,
        action_identity=actions,
        policy_revision=revision,
        behavior_log_probability=behavior,
        valid_mask=(
            jnp.ones((6,), dtype=jnp.bool_) if valid is None else valid
        ),
        force_keep_mask=(
            jnp.zeros((6,), dtype=jnp.bool_) if forced is None else forced
        ),
        protected=protected,
    )


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = (
            np.asarray(jr.key_data(lhs))
            if jax.dtypes.issubdtype(lhs.dtype, jax.dtypes.prng_key)
            else np.asarray(lhs)
        )
        rhs_array = (
            np.asarray(jr.key_data(rhs))
            if jax.dtypes.issubdtype(rhs.dtype, jax.dtypes.prng_key)
            else np.asarray(rhs)
        )
        np.testing.assert_array_equal(lhs_array, rhs_array)


def _assert_tree_bit_equal(left: object, right: object) -> None:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    assert len(left_leaves) == len(right_leaves)
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_array = np.asarray(lhs)
        rhs_array = np.asarray(rhs)
        assert lhs_array.dtype == rhs_array.dtype
        assert lhs_array.shape == rhs_array.shape
        assert lhs_array.tobytes(order="C") == rhs_array.tobytes(order="C")


def test_config_is_canonical_and_semantics_are_narrow() -> None:
    config = _config()
    payload = config.to_config()

    assert payload["schema"] == KONDO_SPARSE_ACTOR_SCHEMA
    gate_payload = cast(dict[str, object], payload["gate"])
    assert gate_payload["schema"] == KONDO_GATE_SCHEMA
    assert "backward_admission_intent_semantics" in gate_payload
    assert "sparks_joy_semantics" not in gate_payload
    assert payload["delight_semantics"] == "advantage-times-selected-action-surprisal"
    assert (
        payload["sparks_joy_semantics"]
        == "gradient-contribution-entered-executed-actor-backward"
    )
    assert payload["baseline_gradient_gated"] is False
    assert payload["critic_gradient_gated"] is False
    assert payload["safety_gradient_gated"] is False
    assert payload["wall_clock_claimed"] is False
    assert payload["efficacy_claimed"] is False
    assert payload["safety_claimed"] is False
    assert payload["evidence_promotion_claimed"] is False
    assert KondoSparseActorConfig.from_config(payload) == config
    assert KondoSparseActor.from_config(payload).to_config() == payload
    assert config.parameter_count == 31

    malformed = dict(payload)
    malformed["sparks_joy_semantics"] = "generic-gradient-quality"
    with pytest.raises(ValueError, match="sparks_joy_semantics"):
        KondoSparseActorConfig.from_config(malformed)


@pytest.mark.parametrize(
    "values",
    [
        {"feature_dim": 0},
        {"hidden_dim": 2049},
        {"action_count": True},
        {"critic_dim": 4097},
        {"safety_dim": 0},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.0},
    ],
)
def test_config_rejects_unbounded_or_nonfinite_values(values: dict[str, object]) -> None:
    base: dict[str, object] = {
        "feature_dim": 3,
        "hidden_dim": 4,
        "action_count": 3,
        "critic_dim": 2,
        "safety_dim": 2,
        "learning_rate": 0.025,
        "gate": KondoGateConfig(batch_size=6, target_rate=0.5),
    }
    base.update(values)
    with pytest.raises((TypeError, ValueError)):
        KondoSparseActorConfig(**base)  # type: ignore[arg-type]


def test_sparse_step_computes_paper_delight_then_real_capacity_backward() -> None:
    actor, state = _actor()
    batch = _batch(actor, state)
    protected_before = batch.protected

    result = actor.step(state, batch)

    assert bool(result.transaction_applied)
    assert bool(result.sparse_backward_used)
    assert not bool(result.full_shape_masked_backward_used)
    assert int(result.backward_batch_size) == actor.config.backward_capacity == 3
    assert int(result.backward_selected_count) == int(result.screen.selected_count)
    assert int(result.state.policy_revision) == 1
    assert int(result.state.actor_backward_count) == 1
    assert int(result.state.sparse_backward_count) == 1
    _assert_tree_bit_equal(
        result.advantage,
        batch.protected.return_targets - batch.protected.baseline_predictions,
    )
    _assert_tree_bit_equal(
        result.screen.delight,
        result.advantage * -result.current_action_log_probability,
    )
    assert bool(jnp.any(result.gradient.hidden_weight != 0.0))
    assert bool(jnp.any(result.gradient.output_weight != 0.0))
    assert int(result.protected_slots) == 6
    _assert_tree_equal(result.protected, protected_before)
    _assert_tree_equal(batch.protected, protected_before)
    assert result.protected_digest.shape == (8,)
    np.testing.assert_array_equal(result.sparks_joy, result.screen.selected_mask)
    np.testing.assert_array_equal(
        result.executed_actor_backward_mask,
        result.screen.selected_mask,
    )
    assert bool(result.backward_delight_exact)
    _assert_tree_bit_equal(
        result.executed_delight[result.sparks_joy],
        result.screen.delight[result.sparks_joy],
    )
    np.testing.assert_array_equal(
        result.executed_delight[~result.sparks_joy],
        jnp.zeros_like(result.executed_delight[~result.sparks_joy]),
    )


def test_rows_that_do_not_spark_joy_cannot_change_the_actor_gradient() -> None:
    actor, state = _actor()
    advantages = jnp.asarray(
        [10.0, 9.0, 8.0, -100.0, -100.0, -100.0],
        dtype=jnp.float32,
    )
    source = _batch(actor, state, advantages=advantages)
    changed_features = source.actor_features.at[3:].set(
        jnp.asarray(
            [
                [90.0, -80.0, 70.0],
                [-60.0, 50.0, -40.0],
                [30.0, -20.0, 10.0],
            ],
            dtype=jnp.float32,
        )
    )
    changed_actions = source.actions.at[3:].set(
        jnp.asarray([2, 0, 1], dtype=jnp.int32)
    )
    changed_advantage = jnp.asarray(
        [10.0, 9.0, 8.0, -1000.0, -900.0, -800.0],
        dtype=jnp.float32,
    )
    changed_protected = source.protected.replace(
        return_targets=source.protected.baseline_predictions + changed_advantage,
    )
    changed = source.replace(
        actor_features=changed_features,
        actions=changed_actions,
        action_identity=changed_actions,
        behavior_log_probability=actor.behavior_log_probability(
            state,
            changed_features,
            changed_actions,
        ),
        protected=changed_protected,
    )

    baseline = actor.step(state, source)
    perturbed = actor.step(state, changed)

    expected_mask = jnp.asarray(
        [True, True, True, False, False, False],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(baseline.sparks_joy, expected_mask)
    np.testing.assert_array_equal(perturbed.sparks_joy, expected_mask)
    assert bool(baseline.sparse_backward_used)
    assert bool(perturbed.sparse_backward_used)
    _assert_tree_bit_equal(baseline.gradient, perturbed.gradient)
    _assert_tree_bit_equal(baseline.actor_loss, perturbed.actor_loss)


def test_equal_delight_uses_lowest_source_indices_and_exact_action_identity() -> None:
    actor, state = _actor()
    batch = _batch(
        actor,
        state,
        advantages=jnp.zeros((6,), dtype=jnp.float32),
    )

    result = actor.step(state, batch)

    assert bool(result.transaction_applied)
    assert bool(result.action_identity_valid)
    np.testing.assert_array_equal(
        result.screen.selected_indices,
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        result.sparks_joy,
        jnp.asarray([True, True, True, False, False, False]),
    )


def test_full_shape_fallback_preserves_every_forced_sample() -> None:
    actor, state = _actor(_config(target_rate=0.25))
    forced = jnp.asarray([True, True, True, False, False, False], dtype=jnp.bool_)
    batch = _batch(actor, state, forced=forced)

    result = actor.step(state, batch)

    assert bool(result.transaction_applied)
    assert not bool(result.sparse_backward_used)
    assert bool(result.full_shape_masked_backward_used)
    assert bool(result.screen.full_shape_masked_backward_required)
    assert not bool(result.screen.capacity_sufficient)
    assert int(result.backward_batch_size) == 6
    assert int(result.backward_selected_count) == int(result.screen.selected_count)
    assert bool(jnp.all(result.sparks_joy[:3]))
    assert bool(jnp.all(result.sparks_joy[forced]))
    assert bool(result.backward_delight_exact)
    _assert_tree_bit_equal(
        result.executed_delight[result.sparks_joy],
        result.screen.delight[result.sparks_joy],
    )
    assert int(result.state.full_fallback_count) == 1

    direct = actor.full_shape_masked_backward(
        state.parameters,
        KondoActorBackwardBatch(
            actor_features=batch.actor_features,
            actions=batch.actions,
            advantage=result.advantage,
            sample_mask=result.sparks_joy,
        ),
    )
    _assert_tree_equal(result.gradient, direct.gradient)
    np.testing.assert_array_equal(result.actor_loss, direct.loss)


def test_full_fallback_rejected_rows_cannot_change_actor_gradient() -> None:
    actor, state = _actor(_config(target_rate=0.25))
    forced = jnp.asarray([True, True, True, False, False, False], dtype=jnp.bool_)
    advantages = jnp.asarray(
        [10.0, 9.0, 8.0, -100.0, -100.0, -100.0],
        dtype=jnp.float32,
    )
    source = _batch(actor, state, forced=forced, advantages=advantages)
    changed_features = source.actor_features.at[3:].set(
        jnp.asarray(
            [
                [90.0, -80.0, 70.0],
                [-60.0, 50.0, -40.0],
                [30.0, -20.0, 10.0],
            ],
            dtype=jnp.float32,
        )
    )
    changed_actions = source.actions.at[3:].set(
        jnp.asarray([2, 0, 1], dtype=jnp.int32)
    )
    changed_advantage = jnp.asarray(
        [10.0, 9.0, 8.0, -1000.0, -900.0, -800.0],
        dtype=jnp.float32,
    )
    changed_protected = source.protected.replace(
        return_targets=source.protected.baseline_predictions + changed_advantage,
    )
    changed = source.replace(
        actor_features=changed_features,
        actions=changed_actions,
        action_identity=changed_actions,
        behavior_log_probability=actor.behavior_log_probability(
            state,
            changed_features,
            changed_actions,
        ),
        protected=changed_protected,
    )

    baseline = actor.step(state, source)
    perturbed = actor.step(state, changed)

    expected_mask = jnp.asarray(
        [True, True, True, False, False, False],
        dtype=jnp.bool_,
    )
    np.testing.assert_array_equal(baseline.sparks_joy, expected_mask)
    np.testing.assert_array_equal(perturbed.sparks_joy, expected_mask)
    assert bool(baseline.full_shape_masked_backward_used)
    assert bool(perturbed.full_shape_masked_backward_used)
    _assert_tree_bit_equal(baseline.gradient, perturbed.gradient)
    _assert_tree_bit_equal(baseline.actor_loss, perturbed.actor_loss)


def test_bernoulli_survivor_overflow_uses_full_shape_fallback() -> None:
    actor, state = _actor(
        _config(mode="bernoulli_price", sparse_capacity=2)
    )
    advantages = jnp.full((6,), 100.0, dtype=jnp.float32)
    batch = _batch(actor, state, advantages=advantages)

    result = actor.step(state, batch)

    assert bool(result.transaction_applied)
    assert bool(result.full_shape_masked_backward_used)
    assert int(result.screen.selected_count) == 6
    assert bool(jnp.all(result.screen.selected_by_delight_gate))
    assert int(result.backward_batch_size) == 6


@pytest.mark.parametrize(
    "failure",
    [
        "action_identity",
        "action_domain",
        "policy_revision",
        "behavior_log_probability",
        "force_keep_subset",
        "nonfinite_protected",
        "no_valid_samples",
    ],
)
def test_dynamic_contract_failures_do_not_screen_or_backward(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor, state = _actor()
    batch = _batch(actor, state)
    if failure == "action_identity":
        batch = dataclasses.replace(batch, action_identity=batch.action_identity.at[0].set(2))
    elif failure == "action_domain":
        batch = dataclasses.replace(
            batch,
            actions=batch.actions.at[0].set(3),
            action_identity=batch.action_identity.at[0].set(3),
        )
    elif failure == "policy_revision":
        batch = dataclasses.replace(batch, policy_revision=batch.policy_revision.at[0].set(1))
    elif failure == "behavior_log_probability":
        batch = dataclasses.replace(
            batch,
            behavior_log_probability=batch.behavior_log_probability.at[0].add(1.0e-4),
        )
    elif failure == "force_keep_subset":
        batch = dataclasses.replace(
            batch,
            valid_mask=batch.valid_mask.at[0].set(False),
            force_keep_mask=batch.force_keep_mask.at[0].set(True),
        )
    elif failure == "nonfinite_protected":
        protected = dataclasses.replace(
            batch.protected,
            safety_features=batch.protected.safety_features.at[0, 0].set(jnp.nan),
        )
        batch = dataclasses.replace(batch, protected=protected)
    else:
        batch = dataclasses.replace(
            batch,
            valid_mask=jnp.zeros((6,), dtype=jnp.bool_),
        )

    def forbidden_backward(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("invalid Kondo transaction invoked an actor backward")

    monkeypatch.setattr(actor, "sparse_backward", forbidden_backward)
    monkeypatch.setattr(actor, "full_shape_masked_backward", forbidden_backward)
    result = actor.step(state, batch)

    assert not bool(result.transaction_applied)
    assert not bool(result.screen.transaction_applied)
    assert not bool(result.sparse_backward_used)
    assert not bool(result.full_shape_masked_backward_used)
    assert int(result.backward_batch_size) == 0
    assert int(result.backward_selected_count) == 0
    assert not bool(result.backward_delight_exact)
    assert not bool(jnp.any(result.sparks_joy))
    _assert_tree_equal(result.state, state)
    _assert_tree_equal(result.gradient, actor._zero_gradient())


def test_corrupt_state_and_exhausted_gate_fail_closed() -> None:
    actor, state = _actor(_config(max_screenings=1))
    corrupt = dataclasses.replace(
        state,
        policy_revision=jnp.asarray(1, dtype=jnp.int32),
        actor_backward_count=jnp.asarray(1, dtype=jnp.int32),
    )
    corrupt_batch = _batch(actor, state)

    rejected = actor.step(corrupt, corrupt_batch)

    assert not bool(rejected.transaction_applied)
    assert not bool(rejected.state_valid)
    assert not bool(jnp.any(rejected.sparks_joy))
    _assert_tree_equal(rejected.state, corrupt)

    first = actor.step(state, _batch(actor, state))
    assert bool(first.transaction_applied)
    exhausted_batch = _batch(actor, first.state)
    exhausted = actor.step(first.state, exhausted_batch)
    assert not bool(exhausted.transaction_applied)
    assert bool(exhausted.state_valid)
    assert not bool(exhausted.screen.transaction_applied)
    assert not bool(jnp.any(exhausted.sparks_joy))
    _assert_tree_equal(exhausted.state, first.state)


def test_executed_nonfinite_backward_sparks_joy_but_cannot_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution truth is independent of later gradient/update acceptance."""

    actor, state = _actor()
    batch = _batch(actor, state)
    original_backward = actor.sparse_backward
    call_count = 0

    def nonfinite_backward(
        parameters: KondoActorParameters,
        backward_batch: KondoActorBackwardBatch,
    ) -> object:
        nonlocal call_count
        call_count += 1
        result = original_backward(parameters, backward_batch)
        poisoned_gradient = dataclasses.replace(
            result.gradient,
            hidden_weight=result.gradient.hidden_weight.at[0, 0].set(jnp.nan),
        )
        return dataclasses.replace(
            result,
            gradient=poisoned_gradient,
            gradient_finite=jnp.asarray(False, dtype=jnp.bool_),
        )

    monkeypatch.setattr(actor, "sparse_backward", nonfinite_backward)
    result = actor.step(state, batch)

    assert call_count == 1
    assert bool(result.screen.transaction_applied)
    assert bool(result.sparse_backward_used)
    assert bool(jnp.any(result.sparks_joy))
    assert int(result.backward_selected_count) == int(result.screen.selected_count)
    assert bool(result.backward_delight_exact)
    np.testing.assert_array_equal(
        result.executed_actor_backward_mask, result.sparks_joy
    )
    np.testing.assert_array_equal(result.sparks_joy, result.screen.selected_mask)
    assert not bool(result.gradient_finite)
    assert not bool(result.transaction_applied)
    assert int(result.state.actor_backward_count) == int(state.actor_backward_count)
    _assert_tree_equal(result.state, state)


def test_executed_finite_backward_sparks_joy_when_parameter_candidate_overflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backward entry is independent of later finite-parameter acceptance."""

    actor, state = _actor(dataclasses.replace(_config(), learning_rate=4.0))
    batch = _batch(actor, state)
    original_backward = actor.sparse_backward

    def overflowing_update_candidate(
        parameters: KondoActorParameters,
        backward_batch: KondoActorBackwardBatch,
    ) -> object:
        result = original_backward(parameters, backward_batch)
        maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
        finite_gradient = jax.tree.map(
            lambda leaf: jnp.full_like(leaf, maximum),
            result.gradient,
        )
        return dataclasses.replace(
            result,
            gradient=finite_gradient,
            gradient_finite=jnp.asarray(True, dtype=jnp.bool_),
        )

    monkeypatch.setattr(actor, "sparse_backward", overflowing_update_candidate)
    result = actor.step(state, batch)

    assert bool(result.gradient_finite)
    assert not bool(result.updated_parameters_finite)
    assert not bool(result.transaction_applied)
    assert bool(jnp.any(result.sparks_joy))
    assert bool(result.backward_delight_exact)
    assert int(result.backward_selected_count) == int(result.screen.selected_count)
    _assert_tree_equal(result.state, state)


def test_bernoulli_zero_survivor_executes_finite_noop_backward() -> None:
    """An executed empty masked backward has no joyful sample contribution."""

    actor, state = _actor(
        _config(mode="bernoulli_price", sparse_capacity=3)
    )
    advantages = jnp.full((6,), -1.0e6, dtype=jnp.float32)
    result = actor.step(
        state,
        _batch(actor, state, advantages=advantages),
    )

    assert bool(result.screen.transaction_applied)
    assert int(result.screen.selected_count) == 0
    assert bool(result.sparse_backward_used)
    assert not bool(result.full_shape_masked_backward_used)
    assert not bool(jnp.any(result.sparks_joy))
    assert int(result.backward_selected_count) == 0
    assert bool(result.backward_delight_exact)
    np.testing.assert_array_equal(
        result.executed_delight,
        jnp.zeros_like(result.executed_delight),
    )
    assert bool(result.gradient_finite)
    assert bool(result.updated_parameters_finite)
    assert bool(result.transaction_applied)
    assert int(result.state.actor_backward_count) == 1
    _assert_tree_equal(result.state.parameters, state.parameters)


def test_only_typed_threefry_keys_are_accepted_for_gate_rng_ownership() -> None:
    actor = KondoSparseActor(_config())

    with pytest.raises(TypeError, match="threefry2x32"):
        actor.init(_parameters(), jr.key(0, impl="rbg"))


def test_sparse_backward_jaxpr_has_capacity_not_full_batch_leading_axis() -> None:
    actor, state = _actor()
    fixed = KondoActorBackwardBatch(
        actor_features=jnp.ones((3, 3), dtype=jnp.float32),
        actions=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        advantage=jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32),
        sample_mask=jnp.ones((3,), dtype=jnp.bool_),
    )

    result = actor.sparse_backward(state.parameters, fixed)
    assert bool(result.gradient_finite)
    assert int(result.selected_count) == 3
    jaxpr = str(jax.make_jaxpr(actor.sparse_backward)(state.parameters, fixed))
    assert "f32[3,3]" in jaxpr
    assert "f32[6,3]" not in jaxpr


def test_backward_kernel_is_exact_under_eager_jit_and_scan() -> None:
    actor, state = _actor()
    fixed = KondoActorBackwardBatch(
        actor_features=jnp.asarray(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, -0.5, 1.0]],
            dtype=jnp.float32,
        ),
        actions=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        advantage=jnp.asarray([1.0, -0.5, 0.25], dtype=jnp.float32),
        sample_mask=jnp.ones((3,), dtype=jnp.bool_),
    )

    eager = actor.sparse_backward(state.parameters, fixed)
    compiled = jax.jit(actor.sparse_backward)(state.parameters, fixed)
    _assert_tree_equal(eager, compiled)

    stacked = jax.tree_util.tree_map(lambda value: jnp.stack([value, value]), fixed)

    def body(
        parameters: KondoActorParameters,
        scan_batch: KondoActorBackwardBatch,
    ) -> tuple[KondoActorParameters, jax.Array]:
        backward = actor.sparse_backward(parameters, scan_batch)
        next_parameters = jax.tree_util.tree_map(
            lambda parameter, gradient: parameter - 0.025 * gradient,
            parameters,
            backward.gradient,
        )
        return next_parameters, backward.loss

    final_eager, losses_eager = jax.lax.scan(body, state.parameters, stacked)
    final_compiled, losses_compiled = jax.jit(lambda p, b: jax.lax.scan(body, p, b))(
        state.parameters,
        stacked,
    )
    _assert_tree_equal(final_eager, final_compiled)
    np.testing.assert_array_equal(losses_eager, losses_compiled)


def test_checkpoint_and_resource_accounting_are_exact() -> None:
    actor, state = _actor()
    result = actor.step(state, _batch(actor, state))
    payload = actor.checkpoint_payload(result.state)
    restored_actor, restored_state = KondoSparseActor.from_checkpoint_payload(payload)

    assert restored_actor.to_config() == actor.to_config()
    _assert_tree_equal(restored_state, result.state)
    resources = actor.resource_declaration(result.state)
    assert resources.full_forward_batch_size == 6
    assert resources.sparse_backward_capacity == 3
    assert resources.full_fallback_batch_size == 6
    assert resources.nonlinear_parameter_count == 31
    assert resources.protected_full_fidelity_slots_per_step == 6
    assert resources.persistent_state_bytes > 0
    assert resources.maximum_actor_backward_passes_per_step == 1
    assert resources.maximum_delight_products_per_step == 6
    assert resources.source_sha256 == kondo_sparse_actor_source_sha256()
    assert not resources.wall_clock_savings_claimed
    assert not resources.efficacy_claimed
    assert not resources.safety_claimed
    assert not resources.evidence_promotion_claimed

    tampered = copy.deepcopy(payload)
    tampered_state = cast(dict[str, Any], tampered["state"])
    tampered_state["policy_revision"] = 0
    with pytest.raises(ValueError, match="integrity"):
        KondoSparseActor.from_checkpoint_payload(tampered)
    tampered = copy.deepcopy(payload)
    tampered_state = cast(dict[str, Any], tampered["state"])
    parameters = cast(dict[str, Any], tampered_state["parameters"])
    hidden = cast(dict[str, Any], parameters["hidden_weight"])
    hidden["data_hex"] = "00" + cast(str, hidden["data_hex"])[2:]
    with pytest.raises(ValueError, match="integrity"):
        KondoSparseActor.from_checkpoint_payload(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor_features", jnp.zeros((5, 3), dtype=jnp.float32)),
        ("actions", jnp.zeros((6,), dtype=jnp.float32)),
        ("policy_revision", jnp.zeros((6,), dtype=jnp.uint32)),
        ("valid_mask", jnp.zeros((6,), dtype=jnp.int32)),
    ],
)
def test_static_batch_contracts_raise(field: str, value: jax.Array) -> None:
    actor, state = _actor()
    batch = _batch(actor, state)
    batch = dataclasses.replace(batch, **{field: value})
    with pytest.raises((TypeError, ValueError)):
        actor.step(state, batch)
