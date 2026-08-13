"""Exact-horizon contracts for secondary Horde actor-critic variants."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Literal, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.horde import HordeLearner
from alberta_framework.core.horde_actor_critic import (
    NONLINEAR_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
    NONLINEAR_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA,
    NONLINEAR_HORDE_ACTOR_CRITIC_STATE_SCHEMA,
    NONLINEAR_Q_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
    NONLINEAR_Q_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA,
    Q_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA,
    Q_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA,
    Q_HORDE_ACTOR_CRITIC_STATE_SCHEMA,
    SECONDARY_ACTOR_CRITIC_OUTER_CLOCK_DELTA_NBYTES,
    SECONDARY_ACTOR_CRITIC_OUTER_CLOCK_NBYTES,
    NonlinearHordeActorCriticAgent,
    NonlinearHordeActorCriticConfig,
    NonlinearHordeActorCriticState,
    NonlinearQHordeActorCriticAgent,
    NonlinearQHordeActorCriticConfig,
    QHordeActorCriticAgent,
    QHordeActorCriticConfig,
    QHordeActorCriticState,
    load_nonlinear_horde_actor_critic_checkpoint,
    load_nonlinear_q_horde_actor_critic_checkpoint,
    load_q_horde_actor_critic_checkpoint,
    measure_nonlinear_horde_actor_critic_state_nbytes,
    measure_q_horde_actor_critic_state_nbytes,
    migrate_legacy_nonlinear_horde_actor_critic_state,
    migrate_legacy_q_horde_actor_critic_state,
    save_nonlinear_horde_actor_critic_checkpoint,
    save_nonlinear_q_horde_actor_critic_checkpoint,
    save_q_horde_actor_critic_checkpoint,
)
from alberta_framework.core.normalizers import EMANormalizer
from alberta_framework.core.types import DemonType, GVFSpec, create_horde_spec

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_OBSERVATION = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
_PREVIOUS_OBSERVATION = jnp.asarray([1.0, 0.0], dtype=jnp.float32)

CaseName = Literal["q_linear", "nonlinear_value", "nonlinear_q"]


@dataclasses.dataclass(frozen=True)
class _SecondaryCase:
    name: CaseName
    agent: Any
    state: Any


def _control_critic(
    n_actions: int = 2,
    *,
    normalizer: EMANormalizer | None = None,
) -> HordeLearner:
    return HordeLearner(
        create_horde_spec(
            [
                GVFSpec(  # type: ignore[call-arg]
                    name=f"q_{action}",
                    demon_type=DemonType.CONTROL,
                    gamma=0.0,
                    lamda=0.0,
                    cumulant_index=-1,
                )
                for action in range(n_actions)
            ]
        ),
        hidden_sizes=(),
        step_size=0.03,
        normalizer=normalizer,
        use_layer_norm=False,
    )


def _value_critic(*, normalizer: EMANormalizer | None = None) -> HordeLearner:
    return HordeLearner(
        create_horde_spec(
            [
                GVFSpec(  # type: ignore[call-arg]
                    name="value",
                    demon_type=DemonType.PREDICTION,
                    gamma=0.9,
                    lamda=0.0,
                    cumulant_index=-1,
                )
            ]
        ),
        hidden_sizes=(),
        step_size=0.03,
        normalizer=normalizer,
        use_layer_norm=False,
    )


def _case(name: CaseName, *, normalized: bool = False) -> _SecondaryCase:
    agent: Any
    state: Any
    normalizer = EMANormalizer(decay=0.9) if normalized else None
    if name == "q_linear":
        agent = QHordeActorCriticAgent(
            QHordeActorCriticConfig(n_actions=2, gamma=0.9),
            _control_critic(normalizer=normalizer),
        )
        state = agent.init(2, jr.key(1))
    elif name == "nonlinear_value":
        agent = NonlinearHordeActorCriticAgent(
            NonlinearHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
                use_layer_norm=False,
            ),
            _value_critic(normalizer=normalizer),
        )
        state = agent.init(2, jr.key(2))
    else:
        agent = NonlinearQHordeActorCriticAgent(
            NonlinearQHordeActorCriticConfig(
                n_actions=2,
                hidden_sizes=(),
                actor_sparsity=0.0,
                use_layer_norm=False,
            ),
            _control_critic(normalizer=normalizer),
        )
        state = agent.init(2, jr.key(3))
    state = cast(
        QHordeActorCriticState | NonlinearHordeActorCriticState,
        cast(Any, state).replace(
            last_observation=_PREVIOUS_OBSERVATION,
            last_action=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return _SecondaryCase(name=name, agent=agent, state=state)


def _update(
    case: _SecondaryCase,
    state: QHordeActorCriticState | NonlinearHordeActorCriticState,
    *,
    terminal: bool,
) -> Any:
    if case.name == "q_linear":
        return case.agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            _OBSERVATION,
            jnp.asarray(float(terminal), dtype=jnp.float32),
        )
    if case.name == "nonlinear_value":
        return case.agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            _OBSERVATION,
            discount=jnp.asarray(0.0 if terminal else 0.9, dtype=jnp.float32),
        )
    return case.agent.update(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        _OBSERVATION,
        jnp.asarray(float(terminal), dtype=jnp.float32),
    )


def _replace_clock(
    state: QHordeActorCriticState | NonlinearHordeActorCriticState,
    words: tuple[int, int],
) -> QHordeActorCriticState | NonlinearHordeActorCriticState:
    word_array = jnp.asarray(words, dtype=jnp.uint32)
    critic_state = state.critic_state.replace(  # type: ignore[attr-defined]
        step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        step_words=word_array,
    )
    return cast(
        QHordeActorCriticState | NonlinearHordeActorCriticState,
        cast(Any, state).replace(
            critic_state=critic_state,
            step_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            step_words=word_array,
        ),
    )


def _assert_trees_equal(left: Any, right: Any) -> None:
    # MultiHeadMLPState retains these two legacy host-only diagnostics as Python
    # floats. A rejected jitted update materializes them as float32 scalars even
    # though every persistent JAX leaf is rolled back bit-for-bit. Canonicalize
    # only those diagnostics so this assertion remains exact for the transaction.
    def canonicalize_timing(state: Any) -> Any:
        critic_state = state.critic_state.replace(
            birth_timestamp=jnp.asarray(state.critic_state.birth_timestamp, dtype=jnp.float32),
            uptime_s=jnp.asarray(state.critic_state.uptime_s, dtype=jnp.float32),
        )
        return state.replace(critic_state=critic_state)

    chex.assert_trees_all_equal(canonicalize_timing(left), canonicalize_timing(right))


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_exact_clock_crosses_low_word_at_terminal_without_telemetry_wrap(
    name: CaseName,
) -> None:
    case = _case(name)
    state = _replace_clock(case.state, (1, _UINT32_MAX))
    result = _update(case, state, terminal=True)

    assert bool(result.update_applied)
    np.testing.assert_array_equal(result.pre_step_words, (1, _UINT32_MAX))
    np.testing.assert_array_equal(result.post_step_words, (2, 0))
    np.testing.assert_array_equal(result.state.step_words, (2, 0))
    np.testing.assert_array_equal(result.state.critic_state.step_words, (2, 0))
    assert int(result.state.step_count) == _INT32_MAX
    assert int(result.state.critic_state.step_count) == _INT32_MAX
    assert bool(result.critic_result.update_applied)
    if name == "q_linear":
        assert bool(jnp.all(result.state.actor_trace_weights == 0.0))
        assert bool(jnp.all(result.state.actor_trace_bias == 0.0))
    else:
        assert bool(jnp.all(result.state.actor_head_trace_w == 0.0))
        assert bool(jnp.all(result.state.actor_head_trace_b == 0.0))


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_q"])
def test_boolean_terminal_source_preserves_the_terminal_contract(name: CaseName) -> None:
    case = _case(name)
    result = case.agent.update(
        case.state,
        jnp.asarray(1.0, dtype=jnp.float32),
        _OBSERVATION,
        jnp.asarray(True),
    )

    assert bool(result.source_valid)
    assert bool(result.update_applied)
    if name == "q_linear":
        assert bool(jnp.all(result.state.actor_trace_weights == 0.0))
    else:
        assert bool(jnp.all(result.state.actor_head_trace_w == 0.0))


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_corrupt_clock_alignment_rolls_back_the_complete_transaction(
    name: CaseName,
) -> None:
    case = _case(name)
    state = _replace_clock(case.state, (4, 9))
    corrupt = cast(
        QHordeActorCriticState | NonlinearHordeActorCriticState,
        cast(Any, state).replace(
            step_words=jnp.asarray((4, 10), dtype=jnp.uint32),
        ),
    )
    result = _update(case, corrupt, terminal=False)

    assert not bool(result.critic_counter_aligned)
    assert not bool(result.state_valid)
    assert not bool(result.update_applied)
    assert not bool(result.critic_result.update_applied)
    _assert_trees_equal(result.state, corrupt)
    np.testing.assert_array_equal(result.post_step_words, corrupt.step_words)
    np.testing.assert_array_equal(
        result.critic_result.post_step_words,
        corrupt.critic_state.step_words,
    )
    np.testing.assert_array_equal(
        result.critic_result.post_step_words,
        result.critic_result.state.step_words,
    )


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_nested_normalizer_refusal_is_a_truthful_global_rollback(name: CaseName) -> None:
    case = _case(name, normalized=True)
    normalizer_state = case.state.critic_state.normalizer_state
    assert normalizer_state is not None
    corrupt_normalizer = normalizer_state.replace(
        sample_count=jnp.asarray(1, dtype=jnp.int32),
        sample_count_words=jnp.asarray((0, 1), dtype=jnp.uint32),
    )
    corrupt = cast(Any, case.state).replace(
        critic_state=case.state.critic_state.replace(
            normalizer_state=corrupt_normalizer,
        )
    )

    result = _update(case, corrupt, terminal=False)

    assert bool(result.critic_counter_aligned)
    assert not bool(result.lifetime_counter_valid)
    assert not bool(result.state_valid)
    assert not bool(result.candidate_state_valid)
    assert not bool(result.update_applied)
    assert not bool(result.critic_result.lifetime_counter_valid)
    assert not bool(result.critic_result.normalizer_counter_aligned)
    assert not bool(result.critic_result.update_applied)
    _assert_trees_equal(result.state, corrupt)
    np.testing.assert_array_equal(
        result.critic_result.post_step_words,
        corrupt.critic_state.step_words,
    )


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_exhausted_exact_clock_refuses_without_rng_or_critic_mutation(
    name: CaseName,
) -> None:
    case = _case(name)
    state = _replace_clock(case.state, (_UINT32_MAX, _UINT32_MAX))
    result = _update(case, state, terminal=True)

    assert bool(result.lifetime_counter_valid)
    assert not bool(result.lifetime_capacity_available)
    assert not bool(result.update_applied)
    assert not bool(result.critic_result.update_applied)
    _assert_trees_equal(result.state, state)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_invalid_terminal_or_discount_source_is_global_rollback(
    name: CaseName,
) -> None:
    case = _case(name)
    state = case.state
    if name == "nonlinear_value":
        result = case.agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            _OBSERVATION,
            discount=jnp.asarray(1.5, dtype=jnp.float32),
        )
    else:
        result = case.agent.update(
            state,
            jnp.asarray(1.0, dtype=jnp.float32),
            _OBSERVATION,
            jnp.asarray(2.0, dtype=jnp.float32),
        )

    assert not bool(result.source_valid)
    assert not bool(result.update_applied)
    assert not bool(result.critic_result.update_applied)
    _assert_trees_equal(result.state, state)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_finite_source_with_nonfinite_candidate_is_rejected_atomically(
    name: CaseName,
) -> None:
    case = _case(name)
    maximum = jnp.asarray(np.finfo(np.float32).max, dtype=jnp.float32)
    if name == "q_linear":
        corrupt = cast(Any, case.state).replace(
            actor_weights=jnp.full_like(case.state.actor_weights, maximum),
            last_observation=jnp.full_like(case.state.last_observation, maximum),
        )
    else:
        corrupt = cast(Any, case.state).replace(
            actor_head_w=jnp.full_like(case.state.actor_head_w, maximum),
            last_observation=jnp.full_like(case.state.last_observation, maximum),
        )
    result = _update(case, corrupt, terminal=False)

    assert bool(result.source_valid)
    assert bool(result.state_valid)
    assert not bool(result.candidate_state_valid)
    assert not bool(result.update_applied)
    assert not bool(result.critic_result.update_applied)
    _assert_trees_equal(result.state, corrupt)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_resource_accounting_is_exact_and_includes_outer_clock(name: CaseName) -> None:
    case = _case(name)
    budget = case.agent.resource_budget(case.state)
    measured = (
        measure_q_horde_actor_critic_state_nbytes(case.state)
        if name == "q_linear"
        else measure_nonlinear_horde_actor_critic_state_nbytes(case.state)
    )
    assert budget.persistent_state_nbytes == measured
    assert budget.actor_state_nbytes + budget.critic_state_nbytes == measured
    assert budget.outer_clock_nbytes == SECONDARY_ACTOR_CRITIC_OUTER_CLOCK_NBYTES
    assert budget.exact_clock_delta_nbytes == SECONDARY_ACTOR_CRITIC_OUTER_CLOCK_DELTA_NBYTES
    assert budget.outer_clock_nbytes == 12
    assert budget.exact_clock_delta_nbytes == 8


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_unsaturated_legacy_state_migration_authenticates_nested_clock(
    name: CaseName,
) -> None:
    case = _case(name)
    word_array = jnp.asarray((0, 7), dtype=jnp.uint32)
    critic_state = case.state.critic_state.replace(
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=word_array,
    )
    current = cast(Any, case.state).replace(
        critic_state=critic_state,
        step_count=jnp.asarray(7, dtype=jnp.int32),
        step_words=word_array,
    )
    legacy = {
        field.name: getattr(current, field.name)
        for field in dataclasses.fields(current)
        if field.name != "step_words"
    }
    legacy["critic_state"] = {
        field.name: getattr(critic_state, field.name)
        for field in dataclasses.fields(critic_state)
        if field.name != "step_words"
    }
    migrated: Any
    if name == "q_linear":
        migrated = migrate_legacy_q_horde_actor_critic_state(
            legacy,
            agent=case.agent,
        )
    else:
        migrated = migrate_legacy_nonlinear_horde_actor_critic_state(
            legacy,
            agent=case.agent,
        )
    np.testing.assert_array_equal(migrated.step_words, (0, 7))
    np.testing.assert_array_equal(migrated.critic_state.step_words, (0, 7))
    assert bool(case.agent.state_is_valid(migrated))

    legacy["step_count"] = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    migration = (
        migrate_legacy_q_horde_actor_critic_state
        if name == "q_linear"
        else migrate_legacy_nonlinear_horde_actor_critic_state
    )
    with pytest.raises(ValueError, match="saturated.*ambiguous"):
        migration(legacy, agent=case.agent)


def _save_case(case: _SecondaryCase, path: Path) -> None:
    if case.name == "q_linear":
        save_q_horde_actor_critic_checkpoint(case.agent, case.state, path)
    elif case.name == "nonlinear_value":
        save_nonlinear_horde_actor_critic_checkpoint(case.agent, case.state, path)
    else:
        save_nonlinear_q_horde_actor_critic_checkpoint(case.agent, case.state, path)


def _load_case(name: CaseName, path: Path) -> tuple[Any, Any]:
    if name == "q_linear":
        return load_q_horde_actor_critic_checkpoint(path)
    if name == "nonlinear_value":
        return load_nonlinear_horde_actor_critic_checkpoint(path)
    return load_nonlinear_q_horde_actor_critic_checkpoint(path)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_checkpoint_roundtrip_preserves_exact_resume(
    name: CaseName,
    tmp_path: Path,
) -> None:
    initial_case = _case(name)
    first = _update(initial_case, initial_case.state, terminal=False)
    case = dataclasses.replace(initial_case, state=first.state)
    path = tmp_path / name
    _save_case(case, path)
    restored_agent, restored_state = _load_case(name, path)
    _assert_trees_equal(restored_state, first.state)
    assert restored_agent.to_config() == case.agent.to_config()

    restored_case = _SecondaryCase(name=name, agent=restored_agent, state=restored_state)
    expected = _update(case, first.state, terminal=True)
    resumed = _update(restored_case, restored_state, terminal=True)
    _assert_trees_equal(resumed.state, expected.state)
    assert bool(resumed.update_applied)


def test_config_and_checkpoint_schemas_fail_closed(tmp_path: Path) -> None:
    cases = tuple(_case(name) for name in ("q_linear", "nonlinear_value", "nonlinear_q"))
    assert cases[0].agent.config.to_config()["schema"] == Q_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA
    assert cases[1].agent.config.to_config()["schema"] == NONLINEAR_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA
    assert (
        cases[2].agent.config.to_config()["schema"] == NONLINEAR_Q_HORDE_ACTOR_CRITIC_CONFIG_SCHEMA
    )
    assert cases[0].agent.to_config()["state_schema"] == Q_HORDE_ACTOR_CRITIC_STATE_SCHEMA
    assert cases[1].agent.to_config()["state_schema"] == NONLINEAR_HORDE_ACTOR_CRITIC_STATE_SCHEMA
    assert Q_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA.endswith(".v2")
    assert NONLINEAR_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA.endswith(".v2")
    assert NONLINEAR_Q_HORDE_ACTOR_CRITIC_CHECKPOINT_SCHEMA.endswith(".v2")

    malformed = cases[0].agent.to_config()
    malformed["state_schema"] = "alberta.q-horde-actor-critic-state.v1"
    with pytest.raises(ValueError, match="state schema"):
        QHordeActorCriticAgent.from_config(malformed)

    loaders = (
        QHordeActorCriticAgent.from_config,
        NonlinearHordeActorCriticAgent.from_config,
        NonlinearQHordeActorCriticAgent.from_config,
    )
    for case, loader in zip(cases, loaders, strict=True):
        missing = case.agent.to_config()
        missing.pop("state_schema")
        with pytest.raises(ValueError, match="field manifest"):
            loader(missing)

        extra = case.agent.to_config()
        extra["unknown"] = True
        with pytest.raises(ValueError, match="field manifest"):
            loader(extra)

        wrong_type = case.agent.to_config()
        wrong_type["type"] = "WrongAgent"
        with pytest.raises(ValueError, match="type is unsupported"):
            loader(wrong_type)

        missing_inner_schema = case.agent.to_config()
        missing_inner_schema["config"] = dict(missing_inner_schema["config"])
        missing_inner_schema["config"].pop("schema")
        with pytest.raises(ValueError, match="field manifest"):
            loader(missing_inner_schema)

    path = tmp_path / "q-checkpoint"
    _save_case(cases[0], path)
    with pytest.raises(ValueError, match="checkpoint schema"):
        load_nonlinear_horde_actor_critic_checkpoint(path)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_static_input_and_state_contracts_reject_wrong_shape_or_dtype(name: CaseName) -> None:
    case = _case(name)
    with pytest.raises(ValueError, match="observation must have shape"):
        if name == "nonlinear_value":
            case.agent.update(
                case.state,
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.zeros((3,), dtype=jnp.float32),
            )
        else:
            case.agent.update(
                case.state,
                jnp.asarray(1.0, dtype=jnp.float32),
                jnp.zeros((3,), dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            )

    corrupt = cast(Any, case.state).replace(step_words=case.state.step_words.astype(jnp.int32))
    with pytest.raises(TypeError, match="step_words must have dtype"):
        _update(case, corrupt, terminal=False)

    corrupt_observation = cast(Any, case.state).replace(
        last_observation=jnp.asarray(0.0, dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="last_observation must have rank 1"):
        _update(case, corrupt_observation, terminal=False)

    corrupt_key = cast(Any, case.state).replace(rng_key=jnp.zeros((2,), dtype=jnp.float32))
    with pytest.raises(TypeError, match="rng_key must be one Threefry"):
        _update(case, corrupt_key, terminal=False)


@pytest.mark.parametrize("name", ["q_linear", "nonlinear_value", "nonlinear_q"])
def test_init_requires_positive_feature_dimension_and_valid_key(name: CaseName) -> None:
    case = _case(name)
    with pytest.raises(ValueError, match="feature_dim must be a positive integer"):
        case.agent.init(0, jr.key(4))
    with pytest.raises(TypeError, match="key must be one Threefry"):
        case.agent.init(2, jnp.zeros((2,), dtype=jnp.float32))


def test_jit_and_scan_keep_wrapper_and_critic_clocks_aligned() -> None:
    case = _case("q_linear")
    state = _replace_clock(case.state, (3, _UINT32_MAX - 1))

    def step(carry: QHordeActorCriticState, terminal: jax.Array) -> tuple[Any, Any]:
        result = case.agent.update(
            carry,
            jnp.asarray(0.25, dtype=jnp.float32),
            _OBSERVATION,
            terminal,
        )
        return result.state, result.update_applied

    final_state, applied = jax.jit(
        lambda initial: jax.lax.scan(
            step,
            initial,
            jnp.asarray((0.0, 1.0), dtype=jnp.float32),
        )
    )(state)
    assert bool(jnp.all(applied))
    np.testing.assert_array_equal(final_state.step_words, (4, 0))
    np.testing.assert_array_equal(final_state.critic_state.step_words, (4, 0))
    assert int(final_state.step_count) == _INT32_MAX
