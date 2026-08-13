"""Exact-lifetime and atomicity contracts for the linear off-policy TD family."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import chex
import jax
import jax.numpy as jnp
import pytest

from alberta_framework.core.off_policy_td import (
    ETD_CONFIG_SCHEMA,
    ETD_STATE_SCHEMA,
    GRADIENT_TD_CONFIG_SCHEMA,
    GRADIENT_TD_STATE_SCHEMA,
    OFF_POLICY_TD_CONFIG_SCHEMA,
    OFF_POLICY_TD_LIFETIME_COUNTER_DELTA_NBYTES,
    OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES,
    OFF_POLICY_TD_LIFETIME_SEMANTICS,
    OFF_POLICY_TD_MAX_COMMITTED_UPDATES,
    OFF_POLICY_TD_RESOURCE_SCHEMA,
    OFF_POLICY_TD_STATE_SCHEMA,
    ETDLinearLearner,
    ETDState,
    GradientTDLinearLearner,
    GradientTDState,
    OffPolicyTDLinearLearner,
    OffPolicyTDResourceBudget,
    OffPolicyTDState,
    measure_etd_state_nbytes,
    measure_gradient_td_state_nbytes,
    measure_off_policy_td_state_nbytes,
    migrate_legacy_etd_config,
    migrate_legacy_etd_state,
    migrate_legacy_gradient_td_config,
    migrate_legacy_gradient_td_state,
    migrate_legacy_off_policy_td_config,
    migrate_legacy_off_policy_td_state,
    run_gradient_td_learning_loop,
)

Learner = OffPolicyTDLinearLearner | ETDLinearLearner | GradientTDLinearLearner
State = OffPolicyTDState | ETDState | GradientTDState

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _learner(kind: str) -> Learner:
    if kind == "td":
        return OffPolicyTDLinearLearner(step_size=0.05, trace_decay=0.4)
    if kind == "etd":
        return ETDLinearLearner(step_size=0.05, trace_decay=0.4)
    if kind == "gtd":
        return GradientTDLinearLearner(
            step_size=0.05,
            secondary_step_size=0.1,
            trace_decay=0.4,
        )
    raise AssertionError(kind)


def _update(
    learner: Learner,
    state: State,
    *,
    observation: jax.Array | None = None,
    reward: jax.Array | float = jnp.float32(1.0),
    next_observation: jax.Array | None = None,
    gamma: jax.Array | float = jnp.float32(0.8),
    rho: jax.Array | float = jnp.float32(1.0),
) -> Any:
    obs = jnp.asarray([1.0, -0.5], dtype=jnp.float32) if observation is None else observation
    nxt = (
        jnp.asarray([0.25, 0.75], dtype=jnp.float32)
        if next_observation is None
        else next_observation
    )
    return learner.update(state, obs, reward, nxt, gamma, rho)


def _legacy_mapping(state: State) -> dict[str, Any]:
    return {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(state)
        if field.name != "step_words"
    }


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_init_exposes_exact_zero_identity_and_finite_lifetime(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2)

    chex.assert_trees_all_equal(state.step_words, jnp.zeros(2, dtype=jnp.uint32))
    assert int(state.step_count) == 0
    assert bool(learner.state_valid(state))
    assert OFF_POLICY_TD_LIFETIME_SEMANTICS == "exact-uint64-fail-stop"
    assert OFF_POLICY_TD_MAX_COMMITTED_UPDATES == 2**64 - 1
    assert OFF_POLICY_TD_LIFETIME_COUNTER_NBYTES == 12
    assert OFF_POLICY_TD_LIFETIME_COUNTER_DELTA_NBYTES == 8


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
@pytest.mark.parametrize(
    ("pre_words", "pre_count", "post_words", "post_count"),
    [
        ((0, _INT32_MAX - 1), _INT32_MAX - 1, (0, _INT32_MAX), _INT32_MAX),
        ((0, _INT32_MAX), _INT32_MAX, (0, _INT32_MAX + 1), _INT32_MAX),
        ((0, _UINT32_MAX), _INT32_MAX, (1, 0), _INT32_MAX),
    ],
)
def test_exact_clock_crosses_int32_and_uint32_boundaries(
    kind: str,
    pre_words: tuple[int, int],
    pre_count: int,
    post_words: tuple[int, int],
    post_count: int,
) -> None:
    learner = _learner(kind)
    state = learner.init(2).replace(
        step_words=jnp.asarray(pre_words, dtype=jnp.uint32),
        step_count=jnp.asarray(pre_count, dtype=jnp.int32),
    )

    result = _update(learner, state)

    assert bool(result.update_applied)
    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    chex.assert_trees_all_equal(
        result.state.step_words,
        jnp.asarray(post_words, dtype=jnp.uint32),
    )
    assert int(result.state.step_count) == post_count


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_all_ones_lifetime_refuses_without_any_state_mutation(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2).replace(
        step_words=jnp.asarray((_UINT32_MAX, _UINT32_MAX), dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )

    result = _update(learner, state)

    assert not bool(result.update_applied)
    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    chex.assert_trees_all_equal(result.state, state)
    chex.assert_trees_all_equal(result.pre_step_words, result.post_step_words)


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_tampered_clock_rolls_back_parameters_traces_and_counter(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2).replace(
        step_words=jnp.asarray((0, 5), dtype=jnp.uint32),
        step_count=jnp.asarray(4, dtype=jnp.int32),
    )

    result = _update(learner, state)

    assert not bool(result.update_applied)
    assert not bool(result.lifetime_counter_valid)
    assert bool(result.source_valid)
    assert bool(result.state_valid)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
@pytest.mark.parametrize(
    "invalid_kwargs",
    [
        {"reward": jnp.asarray(jnp.nan, dtype=jnp.float32)},
        {"gamma": jnp.asarray(1.1, dtype=jnp.float32)},
        {"rho": jnp.asarray(-0.1, dtype=jnp.float32)},
        {"observation": jnp.asarray([jnp.inf, 0.0], dtype=jnp.float32)},
    ],
)
def test_invalid_transition_rolls_back_complete_learning_state(
    kind: str,
    invalid_kwargs: dict[str, Any],
) -> None:
    learner = _learner(kind)
    state = learner.init(2)

    result = _update(learner, state, **invalid_kwargs)

    assert not bool(result.source_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_nonfinite_candidate_rolls_back_finite_source_state(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2).replace(weights=jnp.full_like(learner.init(2).weights, 3e38))

    result = _update(
        learner,
        state,
        observation=jnp.ones(2, dtype=jnp.float32),
        reward=jnp.asarray(-3e38, dtype=jnp.float32),
        next_observation=jnp.zeros(2, dtype=jnp.float32),
        gamma=jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert bool(result.source_valid)
    assert bool(result.state_valid)
    assert not bool(result.candidate_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


def test_etd_negative_interest_is_diagnosed_and_rolled_back() -> None:
    learner = ETDLinearLearner()
    state = learner.init(2)
    result = learner.update(
        state,
        jnp.ones(2, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.zeros(2, dtype=jnp.float32),
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        interest=jnp.asarray(-1.0, dtype=jnp.float32),
    )

    assert not bool(result.source_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_nonfinite_source_state_is_diagnosed_and_never_repaired(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2).replace(weights=jnp.full_like(learner.init(2).weights, jnp.nan))

    result = _update(learner, state)

    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    chex.assert_trees_all_equal(result.state, state)


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_eager_and_jit_transactions_have_identical_verdicts(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2)
    with jax.disable_jit():
        eager = _update(learner, state)
    compiled = _update(learner, state)

    chex.assert_trees_all_close(eager.state, compiled.state)
    chex.assert_trees_all_equal(eager.post_step_words, compiled.post_step_words)
    assert bool(eager.update_applied) == bool(compiled.update_applied) is True


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_jitted_scan_carries_atomic_verdicts(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2)
    observations = jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5]],
        dtype=jnp.float32,
    )

    def scan(initial: State) -> tuple[State, jax.Array]:
        def step(carry: State, observation: jax.Array) -> tuple[State, jax.Array]:
            result = learner.update(
                carry,
                observation,
                jnp.asarray(0.25, dtype=jnp.float32),
                -observation,
                jnp.asarray(0.7, dtype=jnp.float32),
                jnp.asarray(1.0, dtype=jnp.float32),
            )
            return result.state, result.update_applied

        return jax.lax.scan(step, initial, observations)

    final_state, verdicts = jax.jit(scan)(state)

    chex.assert_trees_all_equal(verdicts, jnp.ones(3, dtype=jnp.bool_))
    chex.assert_trees_all_equal(final_state.step_words, jnp.asarray((0, 3), dtype=jnp.uint32))
    assert int(final_state.step_count) == 3


def test_gradient_array_loop_reports_per_event_rollback_and_continues() -> None:
    learner = GradientTDLinearLearner()
    observations = jnp.ones((4, 2), dtype=jnp.float32)
    result = run_gradient_td_learning_loop(
        learner,
        learner.init(2),
        observations,
        jnp.ones(4, dtype=jnp.float32),
        observations,
        jnp.zeros(4, dtype=jnp.float32),
        jnp.asarray([1.0, 1.0, -1.0, 1.0], dtype=jnp.float32),
    )

    chex.assert_trees_all_equal(
        result.updates_applied,
        jnp.asarray([True, True, False, True]),
    )
    chex.assert_trees_all_equal(result.state.step_words, jnp.asarray((0, 3), dtype=jnp.uint32))
    assert int(result.state.step_count) == 3


@pytest.mark.parametrize("kind", ["td", "etd", "gtd"])
def test_strict_state_and_input_manifests(kind: str) -> None:
    learner = _learner(kind)
    state = learner.init(2)
    with pytest.raises(TypeError, match="observation must have dtype float32"):
        _update(learner, state, observation=jnp.ones(2, dtype=jnp.int32))
    with pytest.raises(ValueError, match="observation must have shape"):
        _update(learner, state, observation=jnp.ones(3, dtype=jnp.float32))
    with pytest.raises(TypeError, match="rho must have dtype float32"):
        _update(learner, state, rho=jnp.asarray(1, dtype=jnp.int32))
    with pytest.raises(TypeError, match="lifetime counter words must have dtype uint32"):
        _update(learner, state.replace(step_words=jnp.zeros(2, dtype=jnp.int32)))


@pytest.mark.parametrize(
    "construct",
    [
        lambda: OffPolicyTDLinearLearner(step_size=True),
        lambda: OffPolicyTDLinearLearner(step_size=float("nan")),
        lambda: OffPolicyTDLinearLearner(retrace_clip=float("-inf")),
        lambda: ETDLinearLearner(trace_decay=float("inf")),
        lambda: GradientTDLinearLearner(secondary_step_size=float("nan")),
        lambda: GradientTDLinearLearner(ratio_clip=False),
    ],
)
def test_config_scalars_reject_boolean_and_nonfinite_values(
    construct: Callable[[], Learner],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        construct()


@pytest.mark.parametrize(
    ("learner", "config_schema", "state_schema", "migrate"),
    [
        (
            OffPolicyTDLinearLearner(step_size=0.02, trace_decay=0.3, retrace_clip=2.0),
            OFF_POLICY_TD_CONFIG_SCHEMA,
            OFF_POLICY_TD_STATE_SCHEMA,
            migrate_legacy_off_policy_td_config,
        ),
        (
            ETDLinearLearner(step_size=0.02, trace_decay=0.3),
            ETD_CONFIG_SCHEMA,
            ETD_STATE_SCHEMA,
            migrate_legacy_etd_config,
        ),
        (
            GradientTDLinearLearner(
                step_size=0.02,
                secondary_step_size=0.03,
                trace_decay=0.3,
                ratio_clip=2.0,
            ),
            GRADIENT_TD_CONFIG_SCHEMA,
            GRADIENT_TD_STATE_SCHEMA,
            migrate_legacy_gradient_td_config,
        ),
    ],
)
def test_versioned_config_roundtrip_tamper_rejection_and_explicit_migration(
    learner: Learner,
    config_schema: str,
    state_schema: str,
    migrate: Callable[[dict[str, Any]], dict[str, Any]],
) -> None:
    serialized = learner.to_config()
    assert serialized["schema"] == config_schema
    assert serialized["state_schema"] == state_schema
    assert type(learner).from_config(serialized).to_config() == serialized

    legacy = dict(serialized)
    legacy.pop("schema")
    legacy.pop("state_schema")
    with pytest.raises(ValueError, match="manifest"):
        type(learner).from_config(legacy)
    assert migrate(legacy) == serialized

    tampered = dict(serialized, schema="unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        type(learner).from_config(tampered)
    with pytest.raises(ValueError, match="manifest"):
        type(learner).from_config(dict(serialized, injected=True))


@pytest.mark.parametrize(
    ("learner", "migrate"),
    [
        (OffPolicyTDLinearLearner(), migrate_legacy_off_policy_td_state),
        (ETDLinearLearner(), migrate_legacy_etd_state),
        (GradientTDLinearLearner(), migrate_legacy_gradient_td_state),
    ],
)
def test_state_migration_accepts_only_unambiguous_exact_legacy_counter(
    learner: Learner,
    migrate: Callable[[dict[str, Any]], State],
) -> None:
    source = learner.init(2).replace(step_count=jnp.asarray(17, dtype=jnp.int32))
    legacy = _legacy_mapping(source)
    migrated = migrate(legacy)
    chex.assert_trees_all_equal(migrated.step_words, jnp.asarray((0, 17), dtype=jnp.uint32))
    assert bool(learner.state_valid(migrated))

    ambiguous = dict(legacy, step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32))
    with pytest.raises(ValueError, match="ambiguous"):
        migrate(ambiguous)
    with pytest.raises(ValueError, match="manifest"):
        migrate(dict(legacy, injected=True))


@pytest.mark.parametrize(
    ("learner", "measure", "expected"),
    [
        (OffPolicyTDLinearLearner(), measure_off_policy_td_state_nbytes, 44),
        (ETDLinearLearner(), measure_etd_state_nbytes, 52),
        (GradientTDLinearLearner(), measure_gradient_td_state_nbytes, 60),
    ],
)
def test_resource_budget_matches_concrete_state_and_rejects_tampering(
    learner: Learner,
    measure: Callable[[Any], int],
    expected: int,
) -> None:
    state = learner.init(3)
    budget = learner.resource_budget(3)

    assert budget.state_nbytes == expected
    assert measure(state) == expected
    serialized = budget.to_dict()
    assert serialized["schema"] == OFF_POLICY_TD_RESOURCE_SCHEMA
    assert OffPolicyTDResourceBudget.from_dict(serialized) == budget

    with pytest.raises(ValueError, match="inconsistent"):
        OffPolicyTDResourceBudget.from_dict(dict(serialized, state_nbytes=expected + 4))
    with pytest.raises(ValueError, match="manifest"):
        OffPolicyTDResourceBudget.from_dict(dict(serialized, injected=4))


def test_gradient_array_loop_strict_shapes_and_dtypes() -> None:
    learner = GradientTDLinearLearner()
    state = learner.init(2)
    arrays = jnp.ones((3, 2), dtype=jnp.float32)
    scalars = jnp.ones(3, dtype=jnp.float32)

    with pytest.raises(ValueError, match="next_observations"):
        run_gradient_td_learning_loop(
            learner,
            state,
            arrays,
            scalars,
            arrays[:2],
            scalars,
            scalars,
        )
    with pytest.raises(TypeError, match="rewards"):
        run_gradient_td_learning_loop(
            learner,
            state,
            arrays,
            jnp.ones(3, dtype=jnp.int32),
            arrays,
            scalars,
            scalars,
        )


def test_compatibility_metrics_and_one_step_numerics_are_preserved() -> None:
    observation = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    next_observation = jnp.zeros(2, dtype=jnp.float32)

    td = OffPolicyTDLinearLearner(step_size=0.1)
    td_result = td.update(
        td.init(2),
        observation,
        jnp.asarray(2.0, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(td_result.state.weights, jnp.asarray([0.2, 0.0]))
    assert td_result.metrics.shape == (5,)

    etd = ETDLinearLearner(step_size=0.1)
    etd_result = etd.update(
        etd.init(2),
        observation,
        jnp.asarray(2.0, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(etd_result.state.weights, jnp.asarray([0.2, 0.0]))
    assert etd_result.metrics.shape == (7,)

    gtd = GradientTDLinearLearner(step_size=0.1, secondary_step_size=0.2)
    gtd_result = gtd.update(
        gtd.init(2),
        observation,
        jnp.asarray(2.0, dtype=jnp.float32),
        next_observation,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    chex.assert_trees_all_close(gtd_result.state.weights, jnp.asarray([0.2, 0.0, 0.2]))
    assert gtd_result.metrics.shape == (6,)
