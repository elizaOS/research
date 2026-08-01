# mypy: disable-error-code="attr-defined,call-arg"
"""Contracts for atomic, model-only dual-replay rehearsal composition."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.checkpoints import load_checkpoint_metadata, save_checkpoint
from alberta_framework.core.dual_replay import DualReplayConfig
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    MECHANISM_STATUS,
    MODEL_REPLAY_REHEARSAL_SCHEMA,
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    RealModelReplayEvent,
    load_model_replay_rehearsal_checkpoint,
    save_model_replay_rehearsal_checkpoint,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import WorldModelEnsembleConfig

pytestmark = pytest.mark.unit

_INT32_MAX = 2_147_483_647


@pytest.fixture(autouse=True)
def _bounded_jax_execution(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.name == "test_jit_scan_matches_eager_sequence":
        yield
    else:
        with jax.disable_jit():
            yield


def _ensemble_config() -> WorldModelEnsembleConfig:
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
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsembleConfig(
        model=model,
        signal_estimator=signals,
        ensemble_size=2,
        bootstrap_probability=0.5,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=1,
        residual_variance_floor=1.0e-6,
    )


def _config(
    *,
    action_encoding: str = "one_hot",
    max_representation_lag: int = 0,
) -> ModelReplayRehearsalConfig:
    action_dim = 1 if action_encoding == "scalar_index" else 2
    replay = DualReplayConfig(
        total_capacity=6,
        short_term_capacity=3,
        observation_dim=2,
        action_dim=action_dim,
        short_term_sample_size=2,
        long_term_sample_size=2,
        long_term_policy="reservoir",
        max_representation_lag=max_representation_lag,
    )
    return ModelReplayRehearsalConfig(
        ensemble=_ensemble_config(),
        replay=replay,
        action_encoding=action_encoding,  # type: ignore[arg-type]
    )


def _event(
    index: int = 0,
    *,
    version: int = 0,
    terminated: bool = False,
    truncated: bool = False,
    discount: float | None = None,
    valid: bool = True,
) -> RealModelReplayEvent:
    observation = jnp.asarray(
        [0.1 + 0.02 * index, -0.2 + 0.01 * index],
        dtype=jnp.float32,
    )
    return RealModelReplayEvent(
        observation=observation,
        action=jnp.asarray(index % 2, dtype=jnp.int32),
        reward=jnp.asarray(0.3 - 0.01 * index, dtype=jnp.float32),
        discount=jnp.asarray(
            (0.0 if terminated else 0.9) if discount is None else discount,
            dtype=jnp.float32,
        ),
        terminated=jnp.asarray(terminated),
        truncated=jnp.asarray(truncated),
        next_observation=observation + jnp.asarray([0.05, -0.03], dtype=jnp.float32),
        representation_version=jnp.asarray(version, dtype=jnp.int32),
        provenance_id=jnp.asarray(100 + index, dtype=jnp.int32),
        source_id=jnp.asarray(7, dtype=jnp.int32),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(False),
        valid=jnp.asarray(valid),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def _assert_tree_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree.flatten(_materialize_keys(left))
    right_leaves, right_structure = jax.tree.flatten(_materialize_keys(right))
    assert str(left_structure) == str(right_structure)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_actions_and_resource_contract_are_strict() -> None:
    config = _config()
    payload = config.to_config()
    assert payload["schema"] == MODEL_REPLAY_REHEARSAL_SCHEMA
    assert payload["mechanism_status"] == MECHANISM_STATUS
    assert payload["accepted_scientific_evidence"] is False
    assert ModelReplayRehearsalConfig.from_config(payload) == config
    composer = ModelReplayRehearsal(config)
    assert ModelReplayRehearsal.from_config(composer.to_config()).config == config

    one_hot = composer.decode_action(jnp.asarray([0.0, 1.0], dtype=jnp.float32))
    assert bool(one_hot.valid)
    assert int(one_hot.action) == 1
    for corrupt in ([0.5, 0.5], [1.0, 1.0], [0.0, 0.0]):
        assert not bool(composer.decode_action(jnp.asarray(corrupt, jnp.float32)).valid)

    scalar = ModelReplayRehearsal(_config(action_encoding="scalar_index"))
    assert bool(scalar.decode_action(jnp.asarray([1.0], jnp.float32)).valid)
    for corrupt in ([1.5], [2.0], [float("nan")]):
        assert not bool(scalar.decode_action(jnp.asarray(corrupt, jnp.float32)).valid)

    budget = composer.resource_budget()
    assert budget.persistent_state_bytes == (
        budget.ensemble_state_bytes
        + budget.replay_state_bytes
        + budget.composer_accounting_bytes
    )
    assert budget.composer_accounting_bytes == 28
    assert budget.fixed_replay_quota == 4
    assert budget.replay_total_capacity == 6
    assert budget.max_real_model_update_candidates_per_event == 2
    assert budget.max_replay_model_update_candidates_per_event == 8
    assert budget.max_actor_updates_per_event == 0
    assert budget.max_critic_updates_per_event == 0
    assert budget.max_state_builder_updates_per_event == 0


def test_causal_real_update_record_then_fixed_quota_model_only_rehearsal() -> None:
    composer = ModelReplayRehearsal(_config())
    state = composer.init(jr.key(1))
    event = _event(1, truncated=True)
    real_only = composer.ensemble.update(
        state.ensemble_state,
        event.observation,
        event.action,
        event.reward,
        event.discount,
        event.next_observation,
    )
    result = composer.step(state, event)

    assert bool(result.diagnostics.transaction_applied)
    assert bool(result.real_signals_committed)
    assert bool(result.real_update_diagnostics.applied)
    assert result.real_observed_loss.shape == ()
    assert result.real_observed_loss.dtype == jnp.float32
    np.testing.assert_array_equal(result.real_observed_loss, real_only.observed_loss)
    np.testing.assert_array_equal(
        np.asarray(result.real_representation_gradient),
        np.asarray(real_only.representation_gradient),
    )
    assert bool(result.real_representation_gradient_valid)
    np.testing.assert_array_equal(
        np.asarray(result.trace.sample_valid),
        np.asarray([True, False, True, False]),
    )
    np.testing.assert_array_equal(result.trace.padding, ~result.trace.sample_valid)
    assert not bool(jnp.any(result.trace.fresh_evidence_observed))
    assert int(result.state.accepted_real_event_count) == 1
    assert int(result.state.rehearsal_attempt_count) == 4
    assert int(result.state.rehearsal_applied_count) == 2
    assert int(result.state.rehearsal_padding_count) == 2
    assert bool(composer.state_valid(result.state))

    ensemble_state = result.state.ensemble_state
    _assert_tree_equal(ensemble_state.signal_state, real_only.state.signal_state)
    np.testing.assert_array_equal(
        ensemble_state.residual_variances,
        real_only.state.residual_variances,
    )
    np.testing.assert_array_equal(
        jr.key_data(ensemble_state.bootstrap_key),
        jr.key_data(real_only.state.bootstrap_key),
    )
    np.testing.assert_array_equal(
        ensemble_state.member_update_counts,
        real_only.state.member_update_counts,
    )
    assert int(ensemble_state.event_count) == 1
    assert int(ensemble_state.replay_event_count) == 2

    short = result.state.replay_state.short_term
    long = result.state.replay_state.long_term
    np.testing.assert_array_equal(short.actions[0], jnp.asarray([0.0, 1.0]))
    assert bool(short.truncated[0]) and not bool(short.terminated[0])
    assert float(short.discounts[0]) > 0.0
    assert not bool(short.old_behavior_probability_available[0])
    assert not bool(short.old_behavior_logit_available[0])
    assert not bool(short.old_value_target_available[0])
    assert float(short.old_behavior_probabilities[0]) == 0.0
    assert float(short.old_behavior_logits[0]) == 0.0
    assert float(short.old_value_targets[0]) == 0.0
    np.testing.assert_array_equal(short.observations[0], event.observation)
    np.testing.assert_array_equal(short.next_observations[0], event.next_observation)
    np.testing.assert_array_equal(long.next_observations[0], event.next_observation)


def test_representation_version_filtering_keeps_fixed_padding() -> None:
    composer = ModelReplayRehearsal(_config(max_representation_lag=0))
    state = composer.step(composer.init(jr.key(3)), _event(0, version=0)).state
    result = composer.step(state, _event(1, version=1))

    assert bool(result.diagnostics.transaction_applied)
    assert int(result.diagnostics.stale_short_term_count) == 1
    assert int(result.diagnostics.stale_long_term_count) == 1
    assert int(jnp.sum(result.trace.sample_valid)) == 2
    assert int(jnp.sum(result.trace.padding)) == 2
    assert bool(jnp.all(result.trace.representation_versions[result.trace.sample_valid] == 1))
    assert int(result.state.rehearsal_attempt_count) == 8
    assert int(result.state.rehearsal_applied_count) == 4


def test_invalid_event_rolls_back_children_and_gates_real_learning_surface() -> None:
    composer = ModelReplayRehearsal(_config())
    state = composer.init(jr.key(5))
    event = _event(0, terminated=False, discount=0.0)
    result = composer.step(state, event)

    assert not bool(result.diagnostics.event_valid)
    assert not bool(result.diagnostics.transaction_applied)
    assert bool(result.real_update_diagnostics.applied)
    _assert_tree_equal(result.state.ensemble_state, state.ensemble_state)
    _assert_tree_equal(result.state.replay_state, state.replay_state)
    assert int(result.state.real_attempt_count) == 1
    assert int(result.state.rejected_real_event_count) == 1
    assert int(result.state.accepted_real_event_count) == 0
    assert not bool(result.real_signals_committed)
    assert not bool(result.real_signals.availability.input_valid)
    assert float(result.real_observed_loss) == 0.0
    assert not bool(result.real_representation_gradient_valid)
    np.testing.assert_array_equal(
        result.real_representation_gradient,
        jnp.zeros_like(result.real_representation_gradient),
    )
    assert bool(composer.state_valid(result.state))


def test_corrupt_replay_encoding_and_exhausted_counter_fail_closed() -> None:
    composer = ModelReplayRehearsal(_config())
    accepted = composer.step(composer.init(jr.key(7)), _event()).state
    replay = accepted.replay_state
    corrupt_short = replay.short_term.replace(
        actions=jnp.where(
            replay.short_term.valid[:, None],
            jnp.full_like(replay.short_term.actions, 0.5),
            replay.short_term.actions,
        )
    )
    corrupt_long = replay.long_term.replace(
        actions=jnp.where(
            replay.long_term.valid[:, None],
            jnp.full_like(replay.long_term.actions, 0.5),
            replay.long_term.actions,
        )
    )
    corrupt = accepted.replace(
        replay_state=replay.replace(short_term=corrupt_short, long_term=corrupt_long)
    )
    assert not bool(composer.state_valid(corrupt))
    corrupt_result = composer.step(corrupt, _event(1))
    _assert_tree_equal(corrupt_result.state, corrupt)
    assert not bool(corrupt_result.diagnostics.state_valid)

    initial = composer.init(jr.key(9))
    exhausted = initial.replace(
        real_attempt_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        rejected_real_event_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )
    assert bool(composer.state_valid(exhausted))
    exhausted_result = composer.step(exhausted, _event())
    assert not bool(exhausted_result.diagnostics.counter_available)
    _assert_tree_equal(exhausted_result.state, exhausted)


def test_checkpoint_roundtrip_resume_and_metadata_tamper_rejection(tmp_path: Path) -> None:
    composer = ModelReplayRehearsal(_config())
    state = composer.step(composer.init(jr.key(11)), _event()).state
    checkpoint = tmp_path / "model-replay"
    save_model_replay_rehearsal_checkpoint(composer, state, checkpoint)
    restored_composer, restored_state = load_model_replay_rehearsal_checkpoint(checkpoint)
    assert restored_composer.config == composer.config
    _assert_tree_equal(restored_state, state)

    expected = composer.step(state, _event(1))
    resumed = restored_composer.step(restored_state, _event(1))
    _assert_tree_equal(resumed, expected)

    metadata = load_checkpoint_metadata(checkpoint)
    tampered_budget = dict(metadata["resource_budget"])
    tampered_budget["composer_accounting_bytes"] = 32
    tampered_metadata = dict(metadata)
    tampered_metadata["resource_budget"] = tampered_budget
    tampered = tmp_path / "tampered"
    save_checkpoint(state, tampered, metadata=tampered_metadata)
    with pytest.raises(ValueError, match="resource budget does not match config"):
        load_model_replay_rehearsal_checkpoint(tampered)


def test_jit_scan_matches_eager_sequence() -> None:
    composer = ModelReplayRehearsal(_config())
    initial = composer.init(jr.key(13))
    events = jax.tree.map(lambda *values: jnp.stack(values), _event(0), _event(1))

    eager_state = initial
    eager_results = []
    for index in range(2):
        event = jax.tree.map(lambda value: value[index], events)
        result = composer.step(eager_state, event)
        eager_state = result.state
        eager_results.append(result)
    eager_stacked = jax.tree.map(lambda *values: jnp.stack(values), *eager_results)

    def run_scan(state: Any, event_batch: Any) -> tuple[Any, Any]:
        def body(carry: Any, event: Any) -> tuple[Any, Any]:
            result = composer.step(carry, event)
            return result.state, result

        return jax.lax.scan(body, state, event_batch)

    compiled_state, compiled_results = jax.jit(run_scan)(initial, events)
    _assert_tree_equal(compiled_state, eager_state)
    _assert_tree_equal(compiled_results, eager_stacked)
