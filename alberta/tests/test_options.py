"""Tests for STOMP checkpoint payloads and state migration (core/options.py)."""

import chex
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.options import (
    STOMP_OPTION_MODEL_EXPANSION_FIELDS,
    STOMP_STATE_EXPANSION_FIELDS,
    STOMPAgent,
    STOMPConfig,
    STOMPState,
    SubtaskSpec,
    load_stomp_state_with_migration,
    stomp_state_to_checkpoint_payload,
)

OBS_DIM = 3
N_PRIMITIVE = 2
N_OPTIONS = 2


def _agent() -> STOMPAgent:
    return STOMPAgent(
        STOMPConfig(
            subtask_specs=(
                SubtaskSpec(feature_index=0),
                SubtaskSpec(feature_index=1),
            ),
            observation_dim=OBS_DIM,
            n_primitive_actions=N_PRIMITIVE,
        )
    )


def _state() -> STOMPState:
    agent = _agent()
    state = agent.init(jr.key(0))
    return agent.start(state, jnp.array([0.5, -0.25, 1.0], dtype=jnp.float32))


def _old_format_payload(state: STOMPState) -> dict:
    """Synthesize a pre-expansion checkpoint payload from a current state."""
    payload = stomp_state_to_checkpoint_payload(state)
    for name in STOMP_STATE_EXPANSION_FIELDS:
        del payload[name]
    for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
        del payload["option_models"][name]
    return payload


class TestStompCheckpointRoundTrip:
    """Current-format payloads must round-trip losslessly."""

    def test_new_format_round_trips(self) -> None:
        state = _state()

        restored = load_stomp_state_with_migration(
            stomp_state_to_checkpoint_payload(state)
        )

        chex.assert_trees_all_equal(restored, state)

    def test_migrated_state_round_trips(self) -> None:
        state = _state()
        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        restored = load_stomp_state_with_migration(
            stomp_state_to_checkpoint_payload(migrated)
        )

        chex.assert_trees_all_equal(restored, migrated)


class TestStompStateMigration:
    """Pre-expansion payloads gain zero-filled expansion fields."""

    def test_old_format_fills_option_model_zeros(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
            value = getattr(migrated.option_models, name)
            chex.assert_shape(value, (N_OPTIONS,))
            assert value.dtype == jnp.float32
            assert bool(jnp.all(value == 0.0))

    def test_old_format_fills_scalar_accumulator_zeros(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        for name in STOMP_STATE_EXPANSION_FIELDS:
            value = getattr(migrated, name)
            chex.assert_shape(value, ())
            assert value.dtype == jnp.float32
            assert float(value) == 0.0

    def test_old_format_preserves_shared_fields(self) -> None:
        state = _state()

        migrated = load_stomp_state_with_migration(_old_format_payload(state))

        chex.assert_trees_all_equal(
            migrated.base_learner_state, state.base_learner_state
        )
        chex.assert_trees_all_equal(migrated.option_policies, state.option_policies)
        chex.assert_trees_all_equal(
            migrated.option_models.cumreward_ema, state.option_models.cumreward_ema
        )
        chex.assert_trees_all_equal(
            migrated.option_models.discount_ema, state.option_models.discount_ema
        )
        chex.assert_trees_all_equal(migrated.step_count, state.step_count)
        chex.assert_trees_all_equal(migrated.executing_option, state.executing_option)

    def test_migrated_state_matches_fresh_init_priors(self) -> None:
        """Filled defaults equal a fresh agent's expansion-field priors."""
        agent = _agent()
        fresh = agent.init(jr.key(1))
        migrated = load_stomp_state_with_migration(_old_format_payload(_state()))

        for name in STOMP_OPTION_MODEL_EXPANSION_FIELDS:
            chex.assert_trees_all_equal(
                getattr(migrated.option_models, name),
                getattr(fresh.option_models, name),
            )


class TestStompMigrationFailsClosed:
    """Anything other than the two known templates must be rejected."""

    def test_missing_required_state_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["base_average_reward"]

        with pytest.raises(ValueError, match="base_average_reward"):
            load_stomp_state_with_migration(payload)

    def test_missing_required_option_model_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["option_models"]["cumreward_ema"]

        with pytest.raises(ValueError, match="cumreward_ema"):
            load_stomp_state_with_migration(payload)

    def test_unknown_state_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["not_a_field"] = jnp.array(0.0, dtype=jnp.float32)

        with pytest.raises(ValueError, match="not_a_field"):
            load_stomp_state_with_migration(payload)

    def test_unknown_option_model_field_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["option_models"]["not_a_field"] = jnp.zeros(
            N_OPTIONS, dtype=jnp.float32
        )

        with pytest.raises(ValueError, match="not_a_field"):
            load_stomp_state_with_migration(payload)

    def test_incomplete_option_policies_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        del payload["option_policies"]["traces"]

        with pytest.raises(ValueError, match="option-policy"):
            load_stomp_state_with_migration(payload)

    def test_non_mapping_option_models_raises(self) -> None:
        payload = stomp_state_to_checkpoint_payload(_state())
        payload["option_models"] = 3.0

        with pytest.raises(ValueError, match="option_models"):
            load_stomp_state_with_migration(payload)
