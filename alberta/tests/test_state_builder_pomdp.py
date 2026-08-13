"""Tests for the explicitly development-only state-builder POMDP diagnostic."""

from __future__ import annotations

import math

import pytest

from alberta_framework.evaluation.state_builder_pomdp import (
    DEVELOPMENT_SEEDS,
    StateBuilderPomdpConfig,
    run_development_state_builder_pomdp,
)

pytestmark = pytest.mark.unit


def test_default_seed_partition_is_explicitly_development_only() -> None:
    config = StateBuilderPomdpConfig()
    payload = config.to_config()

    assert config.development_seeds == DEVELOPMENT_SEEDS
    assert payload["development_seeds"] == list(DEVELOPMENT_SEEDS)
    assert payload["evidence_scope"] == "development_only"
    assert payload["rtu_taylor_correction"] is False


def test_small_development_diagnostic_is_deterministic_and_truthfully_scoped() -> None:
    config = StateBuilderPomdpConfig(
        development_seeds=(0,),
        training_trials=80,
        scoring_trials=20,
        recurrent_hidden_dim=3,
    )
    first = run_development_state_builder_pomdp(config)
    second = run_development_state_builder_pomdp(config)
    payload = first.to_config()

    assert first == second
    assert payload["evidence_scope"] == "development_only"
    assert payload["accepted_scientific_evidence"] is False
    assert payload["schema"] == "alberta.state_builder_pomdp.development.v3"
    assert len(payload["limitations"]) >= 4
    assert 0.0 <= first.observation_only.mean_accuracy <= 1.0
    assert 0.0 <= first.fixed_trace.mean_accuracy <= 1.0
    assert 0.0 <= first.learned_gated.mean_accuracy <= 1.0
    assert 0.0 <= first.learned_full_gru.mean_accuracy <= 1.0
    assert 0.0 <= first.learned_rtu.mean_accuracy <= 1.0
    assert math.isfinite(first.observation_only.mean_squared_error)
    assert math.isfinite(first.fixed_trace.mean_squared_error)
    assert math.isfinite(first.learned_gated.mean_squared_error)
    assert math.isfinite(first.learned_full_gru.mean_squared_error)
    assert math.isfinite(first.learned_rtu.mean_squared_error)
    assert first.observation_only.builder_trainable_scalars == 0
    assert first.fixed_trace.builder_trainable_scalars == 0
    assert first.learned_gated.builder_trainable_scalars > 0
    assert first.learned_full_gru.builder_trainable_scalars > 0
    assert first.learned_rtu.builder_trainable_scalars > 0
    assert (
        first.learned_full_gru.builder_trainable_scalars
        > first.learned_gated.builder_trainable_scalars
    )
    assert (
        first.learned_rtu.builder_trainable_scalars
        < first.learned_full_gru.builder_trainable_scalars
    )
    assert (
        first.learned_rtu.builder_persistent_state_bytes
        < first.learned_full_gru.builder_persistent_state_bytes
    )
    for arm in (
        first.observation_only,
        first.fixed_trace,
        first.learned_gated,
        first.learned_full_gru,
        first.learned_rtu,
    ):
        assert arm.probe_trainable_scalars == arm.builder_output_scalars + 1
        assert (
            arm.total_persistent_state_bytes
            == arm.builder_persistent_state_bytes + 4 * arm.probe_trainable_scalars
        )
    assert first.learned_gated.mean_parameter_change_norm > 0.0
    assert first.learned_full_gru.mean_parameter_change_norm > 0.0
    assert first.learned_rtu.mean_parameter_change_norm > 0.0


def test_development_diagnostic_recurrent_config_is_validated_eagerly() -> None:
    with pytest.raises(ValueError, match="recurrent_hidden_dim"):
        StateBuilderPomdpConfig(recurrent_hidden_dim=0)

    for invalid in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="recurrent_step_size"):
            StateBuilderPomdpConfig(recurrent_step_size=invalid)
        with pytest.raises(ValueError, match="recurrent_gradient_clip"):
            StateBuilderPomdpConfig(recurrent_gradient_clip=invalid)
        with pytest.raises(ValueError, match="recurrent_initial_gate_bias"):
            StateBuilderPomdpConfig(recurrent_initial_gate_bias=invalid)
        with pytest.raises(ValueError, match="recurrent_initialization_scale"):
            StateBuilderPomdpConfig(recurrent_initialization_scale=invalid)
        with pytest.raises(ValueError, match="rtu radii"):
            StateBuilderPomdpConfig(rtu_r_min=invalid)
        with pytest.raises(ValueError, match="rtu_max_phase"):
            StateBuilderPomdpConfig(rtu_max_phase=invalid)
        with pytest.raises(ValueError, match="rtu_epsilon"):
            StateBuilderPomdpConfig(rtu_epsilon=invalid)

    with pytest.raises(ValueError, match="rtu radii"):
        StateBuilderPomdpConfig(rtu_r_min=0.9, rtu_r_max=0.2)
    with pytest.raises(ValueError, match="rtu_taylor_correction"):
        StateBuilderPomdpConfig(rtu_taylor_correction=1)  # type: ignore[arg-type]
