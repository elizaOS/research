"""Static contracts for the bounded Prototype feature-memory recurrence lane."""

from __future__ import annotations

import math
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.core.experiential_memory_policy import (
    ExperientialMemoryAdvantageGateConfig,
)
from alberta_framework.core.prototype_feature_lifecycle import (
    PrototypeFeatureLifecycle,
)
from alberta_framework.evaluation.prototype_feature_memory_recurrence_development import (
    ACCEPTANCE_STATUS,
    ACCEPTED_SCIENTIFIC_EVIDENCE,
    DEVELOPMENT_ONLY,
    PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA,
    PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA,
    RECURRENCE_ARMS,
    SCIENTIFIC_PROMOTION_ALLOWED,
    PrototypeFeatureMemoryRecurrenceProtocol,
    _agent_config,
    _pytree_semantic_sha256,
)

pytestmark = pytest.mark.unit


def test_default_protocol_is_the_declared_three_by_512_nonpromoting_life() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol()
    payload = protocol.to_config()

    assert protocol.segment_length == 512
    assert protocol.total_steps == 1536
    assert protocol.arm_names == (
        "full",
        "memory_readout_blocked",
        "feature_promotion_blocked",
        "dual_blocked",
        "cue_masked_counterexample",
        "conservative_outcome_gate",
        "conservative_outcome_gate_cue_masked",
    )
    assert payload["schedule"] == ["A1", "B", "A2"]
    assert payload["schema_version"] == PROTOTYPE_FEATURE_MEMORY_RECURRENCE_PROTOCOL_SCHEMA
    assert DEVELOPMENT_ONLY
    assert not SCIENTIFIC_PROMOTION_ALLOWED
    assert ACCEPTANCE_STATUS == "not-assessed"
    assert not ACCEPTED_SCIENTIFIC_EVIDENCE
    assert PROTOTYPE_FEATURE_MEMORY_RECURRENCE_REPORT_SCHEMA.endswith(".v1")
    assert PrototypeFeatureMemoryRecurrenceProtocol.from_config(payload) == protocol


def test_protocol_supports_short_tests_without_changing_the_transaction_contract() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=2,
        active_pair_slots=2,
        memory_capacity=3,
        replacement_interval=1,
        metric_window=1,
        arm_names=("full", "memory_readout_blocked"),
    )

    assert protocol.total_steps == 6
    assert protocol.base_observation_dim == 8
    assert protocol.candidate_pair_slots == 28
    assert protocol.to_config()["preview_contract"] == (
        "every arm executes and discards one no-memory preview update per event"
    )
    assert protocol.to_config()["conservative_outcome_gate_contract"] == {
        "configured_arms": [
            "conservative_outcome_gate",
            "conservative_outcome_gate_cue_masked",
        ],
        "config": ExperientialMemoryAdvantageGateConfig().to_config(),
        "threshold_tuning_performed": False,
        "evidence_semantics": "local similarity-weighted immediate observed reward",
        "causal_or_delayed_return_claimed": False,
    }
    assert protocol.to_config()["world_model_contract"] == {
        "coordinates": "stable_base_only",
        "generated_pair_tail_modeled": False,
        "buffer_capacity": 1,
        "real_update_calls_per_event": 2,
        "committed_updates_per_event": 1,
    }


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"segment_length": 0}, "segment_length"),
        ({"metric_window": 513}, "metric_window"),
        ({"arm_names": ("memory_readout_blocked", "full")}, "canonical-order"),
        ({"arm_names": ("full", "full")}, "canonical-order"),
        ({"arm_names": ("unknown",)}, "unsupported"),
    ],
)
def test_protocol_rejects_noncanonical_variants(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        PrototypeFeatureMemoryRecurrenceProtocol(**kwargs)  # type: ignore[arg-type]


def test_protocol_roundtrip_rejects_field_claim_and_derived_value_tampering() -> None:
    payload = PrototypeFeatureMemoryRecurrenceProtocol().to_config()
    with pytest.raises(ValueError, match="fields"):
        PrototypeFeatureMemoryRecurrenceProtocol.from_config({**payload, "extra": 1})
    with pytest.raises(ValueError, match="development-only"):
        PrototypeFeatureMemoryRecurrenceProtocol.from_config(
            {**payload, "development_only": False}
        )
    with pytest.raises(ValueError, match="total_steps"):
        PrototypeFeatureMemoryRecurrenceProtocol.from_config(
            {**payload, "total_steps": 1535}
        )
    world_contract = cast(dict[str, object], payload["world_model_contract"])
    with pytest.raises(ValueError, match="canonical"):
        PrototypeFeatureMemoryRecurrenceProtocol.from_config(
            {
                **payload,
                "world_model_contract": {
                    **world_contract,
                    "generated_pair_tail_modeled": 0,
                },
            }
        )
    gate_contract = cast(
        dict[str, object], payload["conservative_outcome_gate_contract"]
    )
    with pytest.raises(ValueError, match="outcome-gate"):
        PrototypeFeatureMemoryRecurrenceProtocol.from_config(
            {
                **payload,
                "conservative_outcome_gate_contract": {
                    **gate_contract,
                    "threshold_tuning_performed": True,
                },
            }
        )


def test_arm_contract_has_matched_ablation_roles_and_explicit_counterexample() -> None:
    arms = {arm.name: arm for arm in RECURRENCE_ARMS}

    assert tuple(arms) == (
        "full",
        "memory_readout_blocked",
        "feature_promotion_blocked",
        "dual_blocked",
        "cue_masked_counterexample",
        "conservative_outcome_gate",
        "conservative_outcome_gate_cue_masked",
    )
    assert arms["full"].memory_readout_enabled
    assert not arms["memory_readout_blocked"].memory_readout_enabled
    assert not arms["feature_promotion_blocked"].feature_promotion_enabled
    assert not arms["dual_blocked"].memory_readout_enabled
    assert not arms["dual_blocked"].feature_promotion_enabled
    assert not arms["cue_masked_counterexample"].cue_visible
    assert "counterexample" in arms["cue_masked_counterexample"].role
    assert arms["conservative_outcome_gate"].conservative_outcome_gate_enabled
    assert arms[
        "conservative_outcome_gate_cue_masked"
    ].conservative_outcome_gate_enabled
    assert arms["conservative_outcome_gate"].cue_visible
    assert not arms["conservative_outcome_gate_cue_masked"].cue_visible
    assert all(
        not arms[name].conservative_outcome_gate_enabled
        for name in (
            "full",
            "memory_readout_blocked",
            "feature_promotion_blocked",
            "dual_blocked",
            "cue_masked_counterexample",
        )
    )


def test_conservative_outcome_gate_uses_exact_defaults_and_stays_defaults_off() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=2,
        replacement_interval=1,
        metric_window=1,
        arm_names=(
            "conservative_outcome_gate",
            "conservative_outcome_gate_cue_masked",
        ),
    )

    historical = _agent_config(protocol, feature_promotion_enabled=True)
    gated = _agent_config(
        protocol,
        feature_promotion_enabled=True,
        conservative_outcome_gate_enabled=True,
    )

    assert historical.experiential_memory_advantage_gate is None
    assert (
        gated.experiential_memory_advantage_gate
        == ExperientialMemoryAdvantageGateConfig()
    )
    assert gated.experiential_memory_advantage_gate.to_config() == (
        ExperientialMemoryAdvantageGateConfig().to_config()
    )


def test_blocked_feature_config_disables_replacement_but_keeps_fixed_candidate_work() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=2,
        active_pair_slots=2,
        memory_capacity=1,
        replacement_interval=1,
        metric_window=1,
        arm_names=("feature_promotion_blocked",),
    )
    enabled = _agent_config(protocol, feature_promotion_enabled=True)
    blocked = _agent_config(protocol, feature_promotion_enabled=False)
    enabled_lifecycle = enabled.prototype_feature_lifecycle
    blocked_lifecycle = blocked.prototype_feature_lifecycle
    assert enabled_lifecycle is not None
    assert blocked_lifecycle is not None

    assert enabled_lifecycle.replacement_interval == 1
    assert blocked_lifecycle.replacement_interval == 0
    assert blocked_lifecycle.active_pair_slots == enabled_lifecycle.active_pair_slots
    assert blocked_lifecycle.candidate_pair_slots == enabled_lifecycle.candidate_pair_slots
    assert blocked_lifecycle.max_observations == enabled_lifecycle.max_observations

    lifecycle = PrototypeFeatureLifecycle(blocked_lifecycle)
    state = lifecycle.init(jr.key(0)).learner_state
    initial_descriptors = (
        state.feature_left.tolist(),
        state.feature_right.tolist(),
    )
    observation = jnp.asarray(
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        dtype=jnp.float32,
    )
    targets = jnp.ones((3,), dtype=jnp.float32)
    for _ in range(2):
        result = lifecycle.learner.update(state, observation, targets)
        assert int(result.promoted_candidate) == -1
        assert int(result.replaced_slot) == -1
        state = result.state
    assert (state.feature_left.tolist(), state.feature_right.tolist()) == initial_descriptors


def test_memory_similarity_gate_is_fixed_from_visible_cue_key_separation() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=1,
        replacement_interval=1,
        metric_window=1,
        arm_names=("full",),
    )
    config = _agent_config(protocol, feature_promotion_enabled=True)
    memory = config.experiential_memory
    assert memory is not None

    total_key_dim = protocol.base_observation_dim + protocol.active_pair_slots
    expected = min(1.0, math.exp(-2.0 / float(total_key_dim)) + 1.0e-6)
    assert memory.min_similarity == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_semantic_digest_normalizes_only_nonlearning_wall_clock_telemetry() -> None:
    protocol = PrototypeFeatureMemoryRecurrenceProtocol(
        segment_length=1,
        active_pair_slots=2,
        memory_capacity=1,
        replacement_interval=1,
        metric_window=1,
        arm_names=("full",),
    )
    lifecycle_config = _agent_config(
        protocol,
        feature_promotion_enabled=True,
    ).prototype_feature_lifecycle
    assert lifecycle_config is not None
    state = PrototypeFeatureLifecycle(lifecycle_config).init(jr.key(0)).learner_state
    later_telemetry = cast(Any, state).replace(
        birth_timestamp=state.birth_timestamp + 1024.0,
        uptime_s=state.uptime_s + 10.0,
    )
    changed_learning_state = cast(Any, state).replace(
        output_weights=state.output_weights.at[0, 0].set(1.0)
    )

    assert _pytree_semantic_sha256(later_telemetry) == _pytree_semantic_sha256(state)
    assert _pytree_semantic_sha256(changed_learning_state) != _pytree_semantic_sha256(
        state
    )
