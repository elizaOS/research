"""Exact-lifetime and fail-closed contracts for the Step 1/2 learner families."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as public_api
import alberta_framework.core as core_api
from alberta_framework.core import learners as learner_module
from alberta_framework.core.learners import (
    LEARNER_LIFETIME_COUNTER_NBYTES,
    LINEAR_LEARNER_CONFIG_SCHEMA,
    LINEAR_LEARNER_STATE_SCHEMA,
    MLP_LEARNER_CONFIG_SCHEMA,
    MLP_LEARNER_STATE_SCHEMA,
    TD_LINEAR_LEARNER_CONFIG_SCHEMA,
    TD_LINEAR_LEARNER_STATE_SCHEMA,
    TRUE_ONLINE_TD_CONFIG_SCHEMA,
    TRUE_ONLINE_TD_STATE_SCHEMA,
    LinearLearner,
    MLPLearner,
    TDLinearLearner,
    TrueOnlineTDLearner,
    measure_learner_state_nbytes,
    migrate_legacy_linear_learner_config,
    migrate_legacy_linear_learner_state,
    migrate_legacy_mlp_learner_config,
    migrate_legacy_mlp_learner_state,
    migrate_legacy_td_linear_learner_config,
    migrate_legacy_td_linear_learner_state,
    migrate_legacy_true_online_td_config,
    migrate_legacy_true_online_td_state,
)
from alberta_framework.core.normalizers import EMANormalizer
from alberta_framework.core.optimizers import TDIDBD, Autostep
from alberta_framework.core.types import (
    LearnerState as LegacyLearnerState,
)
from alberta_framework.core.types import (
    MLPLearnerState as LegacyMLPLearnerState,
)
from alberta_framework.core.types import (
    TDLearnerState as LegacyTDLearnerState,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _payload_equal(left: Any, right: Any) -> None:
    """Compare persistent payload while ignoring host wall-clock metadata."""

    for field in dataclasses.fields(left):
        if field.name in {"birth_timestamp", "uptime_s"}:
            continue
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        left_leaves, left_structure = jax.tree_util.tree_flatten(left_value)
        right_leaves, right_structure = jax.tree_util.tree_flatten(right_value)
        assert str(left_structure) == str(right_structure)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
            np.testing.assert_array_equal(
                np.asarray(left_leaf),
                np.asarray(right_leaf),
                strict=True,
            )


def _dataclass_fields(value: Any) -> tuple[Any, ...]:
    return dataclasses.fields(value)


def _linear_case() -> tuple[Any, Any, Callable[[Any, bool], Any]]:
    learner = LinearLearner(
        optimizer=Autostep(initial_step_size=0.01),
        normalizer=EMANormalizer(),
    )
    state = learner.init(3)

    def update(current: Any, invalid: bool) -> Any:
        observation = (
            jnp.asarray([jnp.nan, 0.0, 1.0], dtype=jnp.float32)
            if invalid
            else jnp.ones(3)
        )
        return learner.update(current, observation, jnp.asarray([1.0], dtype=jnp.float32))

    return learner, state, update


def _mlp_case() -> tuple[Any, Any, Callable[[Any, bool], Any]]:
    learner = MLPLearner(
        hidden_sizes=(3,),
        optimizer=Autostep(initial_step_size=0.01),
        normalizer=EMANormalizer(),
        sparsity=0.0,
        use_layer_norm=False,
    )
    state = learner.init(3, jax.random.key(7))

    def update(current: Any, invalid: bool) -> Any:
        observation = (
            jnp.asarray([jnp.nan, 0.0, 1.0], dtype=jnp.float32)
            if invalid
            else jnp.ones(3)
        )
        return learner.update(current, observation, jnp.asarray([1.0], dtype=jnp.float32))

    return learner, state, update


def _td_case() -> tuple[Any, Any, Callable[[Any, bool], Any]]:
    learner = TDLinearLearner(optimizer=TDIDBD(initial_step_size=0.01))
    state = learner.init(3)

    def update(current: Any, invalid: bool) -> Any:
        observation = (
            jnp.asarray([jnp.nan, 0.0, 1.0], dtype=jnp.float32)
            if invalid
            else jnp.ones(3)
        )
        return learner.update(
            current,
            observation,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
        )

    return learner, state, update


def _true_online_case() -> tuple[Any, Any, Callable[[Any, bool], Any]]:
    learner = TrueOnlineTDLearner(step_size=0.01, trace_decay=0.5)
    state = learner.init(3)

    def update(current: Any, invalid: bool) -> Any:
        observation = (
            jnp.asarray([jnp.nan, 0.0, 1.0], dtype=jnp.float32)
            if invalid
            else jnp.ones(3)
        )
        return learner.update(
            current,
            observation,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float32),
            jnp.asarray(0.9, dtype=jnp.float32),
        )

    return learner, state, update


@pytest.mark.parametrize(
    "case_factory",
    [_linear_case, _mlp_case, _td_case, _true_online_case],
    ids=["linear", "mlp", "td-linear", "true-online-td"],
)
def test_exact_clock_crosses_low_word_and_telemetry_saturates(
    case_factory: Callable[[], tuple[Any, Any, Callable[[Any, bool], Any]]],
) -> None:
    learner, state, update = case_factory()
    del learner
    state = state.replace(
        step_words=jnp.asarray([0, _UINT32_MAX], dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = update(state, False)
    np.testing.assert_array_equal(result.pre_step_words, [0, _UINT32_MAX])
    np.testing.assert_array_equal(result.post_step_words, [1, 0])
    assert int(result.state.step_count) == _INT32_MAX
    assert bool(result.lifetime_counter_valid)
    assert bool(result.lifetime_capacity_available)
    assert bool(result.update_applied)


@pytest.mark.parametrize(
    "case_factory",
    [_linear_case, _mlp_case, _td_case, _true_online_case],
    ids=["linear", "mlp", "td-linear", "true-online-td"],
)
def test_all_ones_refuses_without_mutating_any_payload(
    case_factory: Callable[[], tuple[Any, Any, Callable[[Any, bool], Any]]],
) -> None:
    _learner, state, update = case_factory()
    exhausted = state.replace(
        step_words=jnp.full((2,), _UINT32_MAX, dtype=jnp.uint32),
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    result = update(exhausted, False)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    _payload_equal(result.state, exhausted)


@pytest.mark.parametrize(
    "case_factory",
    [_linear_case, _mlp_case, _td_case, _true_online_case],
    ids=["linear", "mlp", "td-linear", "true-online-td"],
)
def test_invalid_transition_rolls_back_parameters_optimizer_normalizer_and_clock(
    case_factory: Callable[[], tuple[Any, Any, Callable[[Any, bool], Any]]],
) -> None:
    _learner, state, update = case_factory()
    result = update(state, True)
    assert not bool(result.input_valid)
    assert not bool(result.update_applied)
    _payload_equal(result.state, state)


def test_finite_input_with_nonfinite_optimizer_candidate_rolls_back() -> None:
    learner = LinearLearner()
    state = learner.init(2)
    maximum = float(np.finfo(np.float32).max)
    huge = jnp.full((2,), maximum, dtype=jnp.float32)
    result = learner.update(state, huge, jnp.asarray([maximum], dtype=jnp.float32))
    assert bool(result.input_valid)
    assert not bool(result.candidate_state_valid)
    assert not bool(result.update_applied)
    _payload_equal(result.state, state)


@pytest.mark.parametrize(
    ("learner", "state"),
    [
        (LinearLearner(), LinearLearner().init(3)),
        (
            MLPLearner(hidden_sizes=(2,), sparsity=0.0),
            MLPLearner(hidden_sizes=(2,), sparsity=0.0).init(3, jax.random.key(0)),
        ),
        (TDLinearLearner(), TDLinearLearner().init(3)),
        (TrueOnlineTDLearner(), TrueOnlineTDLearner().init(3)),
    ],
    ids=["linear", "mlp", "td-linear", "true-online-td"],
)
def test_resource_budget_matches_measured_initialized_state(learner: Any, state: Any) -> None:
    budget = learner.resource_budget(3)
    assert budget.state_nbytes == measure_learner_state_nbytes(state)
    assert budget.lifecycle_counter_nbytes == LEARNER_LIFETIME_COUNTER_NBYTES
    assert budget.lifetime_identity_bits == 64
    assert budget.telemetry_saturation == _INT32_MAX
    assert budget.max_updates_per_call == 1
    assert budget.replay_capacity == 0


def test_strict_config_schemas_round_trip_and_reject_implicit_legacy() -> None:
    learners_and_schemas: list[tuple[Any, str, str]] = [
        (LinearLearner(), LINEAR_LEARNER_CONFIG_SCHEMA, LINEAR_LEARNER_STATE_SCHEMA),
        (
            MLPLearner(hidden_sizes=(2,), sparsity=0.0),
            MLP_LEARNER_CONFIG_SCHEMA,
            MLP_LEARNER_STATE_SCHEMA,
        ),
        (
            TDLinearLearner(),
            TD_LINEAR_LEARNER_CONFIG_SCHEMA,
            TD_LINEAR_LEARNER_STATE_SCHEMA,
        ),
        (
            TrueOnlineTDLearner(),
            TRUE_ONLINE_TD_CONFIG_SCHEMA,
            TRUE_ONLINE_TD_STATE_SCHEMA,
        ),
    ]
    for learner, config_schema, state_schema in learners_and_schemas:
        config = learner.to_config()
        assert config["schema"] == config_schema
        assert config["state_schema"] == state_schema
        restored = type(learner).from_config(config)
        assert restored.to_config() == config
        legacy = dict(config)
        legacy.pop("schema")
        with pytest.raises(ValueError, match="explicit migration"):
            type(learner).from_config(legacy)
        with pytest.raises(ValueError):
            type(learner).from_config({**config, "unexpected": 1})


def test_explicit_config_migrations_accept_only_exact_legacy_payloads() -> None:
    pairs: list[tuple[Any, Callable[[Mapping[str, Any]], dict[str, Any]]]] = [
        (LinearLearner(), migrate_legacy_linear_learner_config),
        (
            MLPLearner(hidden_sizes=(2,), sparsity=0.0),
            migrate_legacy_mlp_learner_config,
        ),
        (TDLinearLearner(), migrate_legacy_td_linear_learner_config),
        (TrueOnlineTDLearner(), migrate_legacy_true_online_td_config),
    ]
    for learner, migrate in pairs:
        current = learner.to_config()
        legacy = {
            key: value
            for key, value in current.items()
            if key not in {"schema", "state_schema"}
        }
        assert migrate(legacy) == current
        with pytest.raises(ValueError):
            migrate({**legacy, "unexpected": 1})


def test_representable_legacy_state_migrations_add_exact_words() -> None:
    linear = LinearLearner()
    linear_current = linear.init(2)
    linear_legacy = LegacyLearnerState(
        **{
            field.name: getattr(linear_current, field.name)
            for field in _dataclass_fields(LegacyLearnerState)
        }
    )
    linear_legacy = linear_legacy.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(7, dtype=jnp.int32)
    )
    linear_migrated = migrate_legacy_linear_learner_state(linear, linear_legacy)
    np.testing.assert_array_equal(linear_migrated.step_words, [0, 7])

    mlp = MLPLearner(hidden_sizes=(2,), sparsity=0.0)
    mlp_current = mlp.init(2, jax.random.key(0))
    mlp_legacy = LegacyMLPLearnerState(
        **{
            field.name: getattr(mlp_current, field.name)
            for field in _dataclass_fields(LegacyMLPLearnerState)
        }
    ).replace(step_count=jnp.asarray(8, dtype=jnp.int32))  # type: ignore[attr-defined]
    mlp_migrated = migrate_legacy_mlp_learner_state(mlp, mlp_legacy)
    np.testing.assert_array_equal(mlp_migrated.step_words, [0, 8])

    td = TDLinearLearner()
    td_current = td.init(2)
    td_legacy = LegacyTDLearnerState(
        **{
            field.name: getattr(td_current, field.name)
            for field in _dataclass_fields(LegacyTDLearnerState)
        }
    ).replace(step_count=jnp.asarray(9, dtype=jnp.int32))  # type: ignore[attr-defined]
    td_migrated = migrate_legacy_td_linear_learner_state(td, td_legacy)
    np.testing.assert_array_equal(td_migrated.step_words, [0, 9])

    true_online = TrueOnlineTDLearner()
    true_current = true_online.init(2)
    true_legacy = {
        field.name: getattr(true_current, field.name)
        for field in _dataclass_fields(true_current)
        if field.name != "step_words"
    }
    true_legacy["step_count"] = jnp.asarray(10, dtype=jnp.int32)
    true_migrated = migrate_legacy_true_online_td_state(true_online, true_legacy)
    np.testing.assert_array_equal(true_migrated.step_words, [0, 10])


def test_legacy_state_migration_rejects_unrepresentable_counter() -> None:
    learner = LinearLearner()
    current = learner.init(2)
    legacy = {
        field.name: getattr(current, field.name)
        for field in _dataclass_fields(LegacyLearnerState)
    }
    legacy["step_count"] = jnp.asarray(-1, dtype=jnp.int32)
    with pytest.raises(ValueError, match="negative"):
        migrate_legacy_linear_learner_state(learner, legacy)


def test_static_state_contract_rejects_wrong_exact_clock_dtype() -> None:
    learner = LinearLearner()
    malformed = learner.init(2).replace(  # type: ignore[attr-defined]
        step_words=jnp.zeros((2,), dtype=jnp.int32)
    )
    with pytest.raises(TypeError, match="uint32"):
        learner.state_is_valid(malformed)


def test_exact_learner_contracts_are_identical_across_public_surfaces() -> None:
    exported_names = (
        "LINEAR_LEARNER_CONFIG_SCHEMA",
        "LINEAR_LEARNER_STATE_SCHEMA",
        "MLP_LEARNER_CONFIG_SCHEMA",
        "MLP_LEARNER_STATE_SCHEMA",
        "TD_LINEAR_LEARNER_CONFIG_SCHEMA",
        "TD_LINEAR_LEARNER_STATE_SCHEMA",
        "TRUE_ONLINE_TD_CONFIG_SCHEMA",
        "TRUE_ONLINE_TD_STATE_SCHEMA",
        "LEARNER_EXACT_LIFETIME_IDENTITY_NBYTES",
        "LEARNER_LIFETIME_COUNTER_NBYTES",
        "LearnerResourceBudget",
        "migrate_legacy_linear_learner_config",
        "migrate_legacy_linear_learner_state",
        "migrate_legacy_mlp_learner_config",
        "migrate_legacy_mlp_learner_state",
        "migrate_legacy_td_linear_learner_config",
        "migrate_legacy_td_linear_learner_state",
        "migrate_legacy_true_online_td_config",
        "migrate_legacy_true_online_td_state",
    )
    for name in exported_names:
        source = getattr(learner_module, name)
        assert getattr(core_api, name) is source
        assert getattr(public_api, name) is source
        assert name in core_api.__all__
        assert name in public_api.__all__

    assert core_api.LearnerState is LegacyLearnerState
    assert public_api.LearnerState is LegacyLearnerState
    assert public_api.MLPLearnerState is LegacyMLPLearnerState
    assert core_api.TDLearnerState is LegacyTDLearnerState
    assert public_api.TDLearnerState is LegacyTDLearnerState
    assert learner_module.LearnerState is not LegacyLearnerState
    assert learner_module.MLPLearnerState is not LegacyMLPLearnerState
    assert learner_module.TDLearnerState is not LegacyTDLearnerState
