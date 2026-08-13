"""Static contracts for the frozen FastSlow recurrence diagnostic."""

from __future__ import annotations

from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.fast_slow import (
    FAST_SLOW_CONFIG_SCHEMA,
    FAST_SLOW_RESOURCE_SCHEMA,
    FAST_SLOW_RESULT_SCHEMA,
    FAST_SLOW_STATE_SCHEMA,
    FastSlowLearner,
)
from alberta_framework.evaluation.fast_slow_recurrence_development import (
    ARM_NAMES,
    DEVELOPMENT_ROOT_SEED,
    FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA,
    FastSlowRecurrenceProtocol,
    _arm_config,
    _ordinary_config,
    _source_arrays,
    _state_resource_payload,
    _state_sha256,
)

pytestmark = pytest.mark.unit


def test_protocol_is_one_frozen_consumed_metadata_free_life() -> None:
    protocol = FastSlowRecurrenceProtocol()
    payload = protocol.to_config()

    assert protocol.schema_version == FAST_SLOW_RECURRENCE_PROTOCOL_SCHEMA
    assert protocol.development_root_seed == DEVELOPMENT_ROOT_SEED == 20_260_802
    assert protocol.phase_steps == 512
    assert protocol.total_steps == 1536
    assert protocol.summary_window == 64
    assert payload["development_root_consumed"] is True
    assert payload["seed_or_hyperparameter_search_performed"] is False
    assert payload["target_mapping"] == {"A1": "x", "B": "-x", "A2": "x"}
    assert payload["learner_inputs"] == ["observation", "target"]
    assert payload["learner_metadata_exposed"] == []
    assert "unidentifiable" in cast(str, payload["switch_identifiability"])
    assert FastSlowRecurrenceProtocol.from_config(payload) == protocol

    with pytest.raises(ValueError, match="frozen"):
        FastSlowRecurrenceProtocol(phase_steps=513)
    with pytest.raises(ValueError, match="frozen"):
        FastSlowRecurrenceProtocol(development_root_seed=0)


def test_source_is_one_exact_float32_draw_and_targets_reconstruct() -> None:
    protocol = FastSlowRecurrenceProtocol()
    observations, targets, manifest, initialization_key = _source_arrays(protocol)
    host_observations = np.asarray(observations)
    host_targets = np.asarray(targets)

    assert observations.shape == targets.shape == (1536, 1)
    assert observations.dtype == targets.dtype == jnp.float32
    assert np.array_equal(host_targets[:512], host_observations[:512])
    assert np.array_equal(host_targets[512:1024], -host_observations[512:1024])
    assert np.array_equal(host_targets[1024:], host_observations[1024:])
    assert manifest["one_source_draw_call"] is True
    assert manifest["source_float32_values_drawn"] == 1536
    assert manifest["probe_random_draws"] == 0
    assert manifest["root_key_data"] == [
        int(value) for value in np.asarray(jr.key_data(jr.key(DEVELOPMENT_ROOT_SEED)))
    ]
    assert manifest["initialization_key_data"] == [
        int(value) for value in np.asarray(jr.key_data(initialization_key))
    ]
    assert len(cast(str, manifest["input_sha256"])) == 64
    assert len(cast(str, manifest["manifest_sha256"])) == 64


def test_two_arm_configs_differ_only_in_prescribed_step_sizes_and_match_state() -> None:
    protocol = FastSlowRecurrenceProtocol()
    ordinary = _ordinary_config(protocol)
    slow_only = _arm_config(protocol, ARM_NAMES[1])
    ordinary_payload = ordinary.to_config()
    slow_payload = slow_only.to_config()
    differences = {
        name for name in ordinary_payload if ordinary_payload[name] != slow_payload[name]
    }

    assert ordinary_payload == {
        "type": "FastSlowConfig",
        "schema": FAST_SLOW_CONFIG_SCHEMA,
        "state_schema": FAST_SLOW_STATE_SCHEMA,
        "result_schema": FAST_SLOW_RESULT_SCHEMA,
        "resource_schema": FAST_SLOW_RESOURCE_SCHEMA,
        "input_dim": 1,
        "output_dim": 1,
        "hidden_dim": 64,
        "encoder_step_size": 0.001,
        "slow_step_size": 0.01,
        "fast_step_size": 0.05,
        "gate_step_size": 0.01,
        "fast_decay": 0.98,
        "slow_weight_decay": 1.0,
        "gate_l2": 0.0,
        "grad_clip": 10.0,
        "init_scale": 1.0,
    }
    assert differences == {"fast_step_size", "gate_step_size"}
    assert slow_only.fast_step_size == 0.0
    assert slow_only.gate_step_size == 0.0

    key = jr.key(7)
    ordinary_state = FastSlowLearner(ordinary).init(key)
    slow_state = FastSlowLearner(slow_only).init(key)
    assert _state_sha256(ordinary_state) == _state_sha256(slow_state)
    assert _state_resource_payload(ordinary_state) == _state_resource_payload(slow_state)
    resources = _state_resource_payload(ordinary_state)
    assert resources["total_nbytes"] == 1304
    assert resources["step_words_nbytes"] == 8
    assert resources["lifetime_counter_nbytes"] == 12
    assert resources["step_count_dtype"] == "int32"
    assert resources["step_count_indefinite_operation_established"] is False
