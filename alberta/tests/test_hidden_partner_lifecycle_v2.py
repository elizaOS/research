"""Adversarial tests for causal v2 critical-feature lifecycle semantics."""

from __future__ import annotations

import base64
import dataclasses
import functools
import zlib

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.evaluation.hidden_partner_development import (
    HiddenPartnerCondition,
    HiddenPartnerDevelopmentProtocol,
    HiddenPartnerDevelopmentRunner,
    HiddenPartnerSeedPair,
    derive_hidden_partner_seed_pairs,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_v2 import (
    CONFIRMATION_NAMESPACE,
    CONFIRMATION_NAMESPACE_STATUS,
    CRITICAL_RUN_PRIMITIVES_SCHEMA,
    CRITICAL_RUN_PRIMITIVES_V5_SCHEMA,
    HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA,
    HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA,
    LEASE_TUNING_GRID,
    LEASE_TUNING_NAMESPACE,
    LEASE_TUNING_NAMESPACE_STATUS,
    RECURRENT_ENTRY_WINDOW,
    RESERVED_CONFIRMATION_CANDIDATES,
    RESERVED_CONFIRMATION_CONTROL,
    _all_run_contracts_valid,
    _candidate_archive_contract_valid,
    _consumer_gate_contract_audit,
    _consumer_gate_contract_valid,
    _feature_memory_contract_audit,
    _float32_state_sequence_payload,
    audit_hidden_partner_lifecycle_v5,
    critical_run_primitives,
    critical_run_primitives_v5,
    summarize_critical_lifecycle_v2,
)
from alberta_framework.streams.hidden_partner_mapping import (
    HiddenPartnerMappingConfig,
)

_C_PAIR = (0, 2)
_D_PAIR = (4, 5)
_LEARNED_COLUMN = np.asarray((-1.0, 1.0), dtype=np.float32)
_ONLINE_NLL = float(np.logaddexp(0.0, -2.0))
_ENTRY_FROZEN_NLL = float(np.log(2.0))
_COLUMN_LEARNING_NLL_GAIN = _ENTRY_FROZEN_NLL - _ONLINE_NLL


def test_reserved_v4_grid_freezes_namespace_seeds_and_both_memory_gates() -> None:
    assert HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA.endswith("lifecycle.v4")
    assert LEASE_TUNING_NAMESPACE == (
        "hidden-partner-v0-dev-v4-evidence-gated-lease-grid-a-v1"
    )
    expected = (
        (2131788533, 2997887370),
        (1008919385, 1922381242),
        (2587588257, 3453966518),
        (1472702288, 2991945725),
        (887122462, 4238410811),
        (955828618, 1757296777),
        (514378265, 1408622614),
        (3257117759, 3849968162),
    )
    actual = derive_hidden_partner_seed_pairs(LEASE_TUNING_NAMESPACE, 8)
    assert tuple(
        (pair.stream_seed, pair.initialization_seed) for pair in actual
    ) == expected
    for cell in LEASE_TUNING_GRID:
        config = cell.agent_config()
        assert config.evidence_gated_feature_memory
        assert config.feature_evidence_confirmation_steps == 8
        assert config.evidence_gated_consumer_memory
        assert config.consumer_evidence_confirmation_steps == 8
        assert config.consumer_read_confirmation_steps == 1
        assert config.consumer_read_lease_steps == 32


def test_v5_confirmation_plan_is_exactly_one_reserved_unexecuted_candidate() -> None:
    assert HIDDEN_PARTNER_LIFECYCLE_V2_SCHEMA.endswith("lifecycle.v4")
    assert CRITICAL_RUN_PRIMITIVES_SCHEMA.endswith("critical-run-primitives.v3")
    assert HIDDEN_PARTNER_LIFECYCLE_V5_SCHEMA.endswith("lifecycle.v5")
    assert CRITICAL_RUN_PRIMITIVES_V5_SCHEMA.endswith("critical-run-primitives.v4")
    assert LEASE_TUNING_NAMESPACE_STATUS == "FORBIDDEN/UNEXECUTED"
    assert CONFIRMATION_NAMESPACE_STATUS == "FORBIDDEN/UNEXECUTED"
    assert "v5" in CONFIRMATION_NAMESPACE
    assert CONFIRMATION_NAMESPACE != LEASE_TUNING_NAMESPACE
    assert len(RESERVED_CONFIRMATION_CANDIDATES) == 1

    candidate = RESERVED_CONFIRMATION_CANDIDATES[0]
    config = candidate.agent_config()
    assert candidate.to_dict()["namespace"] == CONFIRMATION_NAMESPACE
    assert candidate.to_dict()["namespace_status"] == "FORBIDDEN/UNEXECUTED"
    assert config.active_utility_retention_grace_steps == 4_096
    assert config.active_utility_evidence_threshold == 0.1
    assert config.candidate_promotion_floor == 0.1
    assert config.feature_evidence_confirmation_steps == 24
    assert config.independent_relevance_probe
    assert config.candidate_promotion_confirmation_steps == 1
    assert config.candidate_reacquisition_confirmation_steps == 8
    assert config.consumer_evidence_confirmation_steps == 12
    assert config.consumer_read_confirmation_steps == 4
    assert config.consumer_read_lease_steps == 4
    control = RESERVED_CONFIRMATION_CONTROL.agent_config()
    assert control.to_config() == {
        **config.to_config(),
        "candidate_reacquisition_confirmation_steps": 1,
    }


@functools.lru_cache(maxsize=1)
def _base_result():
    protocol = HiddenPartnerDevelopmentProtocol(
        environment=HiddenPartnerMappingConfig(
            base_segment_lengths=(256,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        ),
        recovery_window=128,
        early_late_window=128,
    )
    seed = derive_hidden_partner_seed_pairs(
        "hidden-partner-lifecycle-v2-unit",
        1,
    )[0]
    return HiddenPartnerDevelopmentRunner("full", protocol).run(seed)


@functools.lru_cache(maxsize=1)
def _manual_v5_result():
    protocol = HiddenPartnerDevelopmentProtocol(
        environment=HiddenPartnerMappingConfig(
            base_segment_lengths=(256,) * 9,
            jitter_radius=0,
            partner_flip_probability=0.0,
        ),
        recovery_window=128,
        early_late_window=128,
    )
    condition = HiddenPartnerCondition(
        name="full",
        config=RESERVED_CONFIRMATION_CANDIDATES[0].agent_config(),
        isolated_question="manual unit-seed v5 trace audit",
    )
    seed = HiddenPartnerSeedPair(
        namespace="hidden-partner-lifecycle-v5-manual-unit",
        index=0,
        stream_seed=0x5100A11D,
        initialization_seed=0x5100B22E,
    )
    assert seed.namespace != CONFIRMATION_NAMESPACE
    return HiddenPartnerDevelopmentRunner(condition, protocol).run(seed)


def test_manual_v5_trace_fail_closes_on_current_conditional_probe_source() -> None:
    result = _manual_v5_result()
    audit = audit_hidden_partner_lifecycle_v5(result)

    assert audit.config_role == "frozen_candidate"
    assert not audit.probe_contract_valid
    assert not audit.candidate_reacquisition_contract_valid
    assert audit.durable_memory_contract_valid
    assert not audit.lifecycle_contract_valid
    assert audit.resource_contract_valid
    assert not audit.all_contracts_valid
    assert np.any(audit.feature_violation_bits)
    assert np.any(audit.candidate_violation_bits)
    assert np.any(audit.step_violation_bits)


def test_v5_critical_primitives_reject_current_incompatible_trace() -> None:
    result = _manual_v5_result()
    with pytest.raises(ValueError, match="fully valid independent host audit"):
        critical_run_primitives_v5(result)


def _replace_v5_trace_field(result, field: str, values: np.ndarray):
    return dataclasses.replace(
        result,
        trace=result.trace.replace(**{field: jnp.asarray(values)}),
    )


def test_v5_audit_rejects_noncontiguous_active_prefix() -> None:
    result = _manual_v5_result()
    active = np.asarray(result.trace.active, dtype=np.bool_).copy()
    active[0] = False
    audit = audit_hidden_partner_lifecycle_v5(
        _replace_v5_trace_field(result, "active", active)
    )

    assert not audit.all_contracts_valid
    assert np.all(audit.step_violation_bits)


@pytest.mark.parametrize("hostile", ("negative_zero", "nonfinite", "event", "max_age"))
def test_v5_audit_rejects_hostile_primitives(hostile: str) -> None:
    result = _manual_v5_result()
    if hostile == "negative_zero":
        values = np.asarray(result.trace.interaction_phi_pre, dtype=np.float32).copy()
        values[0, 0] = np.float32(-0.0)
        field = "interaction_phi_pre"
    elif hostile == "nonfinite":
        values = np.asarray(
            result.trace.interaction_candidate_promotion_signal,
            dtype=np.float32,
        ).copy()
        values[0, 0] = np.float32(np.nan)
        field = "interaction_candidate_promotion_signal"
    elif hostile == "event":
        values = np.asarray(result.trace.interaction_retired_left, dtype=np.int32).copy()
        values[0] = 0
        field = "interaction_retired_left"
    else:
        values = np.asarray(result.trace.candidate_ages_pre, dtype=np.int32).copy()
        values[0, 0] = np.iinfo(np.int32).max
        field = "candidate_ages_pre"
    audit = audit_hidden_partner_lifecycle_v5(
        _replace_v5_trace_field(result, field, values)
    )

    assert not audit.all_contracts_valid
    assert bool(audit.step_violation_bits[0]) or bool(
        np.any(audit.candidate_violation_bits[0])
    )


def test_v5_audit_pins_exact_resource_budget_and_config() -> None:
    result = _manual_v5_result()
    changed_resource = dataclasses.replace(
        result.initial_resource,
        raw_observation_dim=result.initial_resource.raw_observation_dim + 1,
    )
    resource_audit = audit_hidden_partner_lifecycle_v5(
        dataclasses.replace(result, initial_resource=changed_resource)
    )
    assert not resource_audit.resource_contract_valid
    assert not resource_audit.all_contracts_valid

    changed_config = dataclasses.replace(
        result.condition.config,
        candidate_reacquisition_confirmation_steps=7,
    )
    config_audit = audit_hidden_partner_lifecycle_v5(
        dataclasses.replace(
            result,
            condition=dataclasses.replace(result.condition, config=changed_config),
        )
    )
    assert config_audit.config_role is None
    assert not config_audit.all_contracts_valid


def _bounds(result) -> np.ndarray:
    return np.cumsum((0, *result.summary.segment_lengths), dtype=np.int64)


def _candidate_index(descriptors: np.ndarray, pair: tuple[int, int]) -> int:
    matches = np.all(
        descriptors[0] == np.asarray(pair, dtype=np.int32),
        axis=1,
    )
    assert np.sum(matches) == 1
    return int(np.flatnonzero(matches)[0])


def _reconstruct_enabled_feature_memory_trace(
    states: np.ndarray,
    evidence: np.ndarray,
    *,
    confirmation_steps: int,
) -> dict[str, np.ndarray]:
    """Build an independent identity-routed feature-memory trace."""
    steps, slots = evidence.shape
    assert states.shape == (steps + 1, slots, 2)
    live_pre = np.all((states[:-1] >= 0) & (states[:-1] < 12), axis=2)
    live_post = np.all((states[1:] >= 0) & (states[1:] < 12), axis=2)
    confirmed = np.zeros((steps, slots), dtype=np.bool_)
    streak_pre = np.zeros((steps, slots), dtype=np.int32)
    streak_post = np.zeros((steps, slots), dtype=np.int32)
    committed_pre = np.zeros((steps, slots), dtype=np.bool_)
    committed_post = np.zeros((steps, slots), dtype=np.bool_)
    streak = np.zeros((slots,), dtype=np.int32)
    committed = np.zeros((slots,), dtype=np.bool_)
    for step in range(steps):
        streak_pre[step] = streak
        committed_pre[step] = committed
        updated_streak = np.where(
            live_pre[step] & evidence[step],
            streak + 1,
            0,
        ).astype(np.int32)
        confirmed[step] = (
            live_pre[step]
            & evidence[step]
            & (updated_streak >= confirmation_steps)
        )
        updated_committed = live_pre[step] & (
            committed | confirmed[step]
        )
        next_streak = np.zeros_like(streak)
        next_committed = np.zeros_like(committed)
        for post_slot, descriptor in enumerate(states[step + 1]):
            if not live_post[step, post_slot]:
                continue
            sources = np.flatnonzero(
                np.all(states[step] == descriptor, axis=1)
            )
            if sources.size != 1:
                continue
            source = int(sources[0])
            next_streak[post_slot] = updated_streak[source]
            next_committed[post_slot] = updated_committed[source]
        streak_post[step] = next_streak
        committed_post[step] = next_committed
        streak = next_streak
        committed = next_committed
    weights = np.zeros((steps, 1, slots), dtype=np.float32)
    return {
        "confirmed": confirmed,
        "weights_pre": weights.copy(),
        "weights_post": weights.copy(),
        "streak_pre": streak_pre,
        "streak_post": streak_post,
        "committed_pre": committed_pre,
        "committed_post": committed_post,
    }


def _feature_memory_audit_case(
    *,
    enabled: bool = True,
) -> tuple[dict[str, np.ndarray], IntegratedHiddenPartnerConfig]:
    """Return a small independently constructed routing/bootstrap case."""
    pair_a = np.asarray((0, 1), dtype=np.int32)
    pair_b = np.asarray((2, 3), dtype=np.int32)
    pair_c = np.asarray((4, 5), dtype=np.int32)
    states = np.full((6, 12, 2), -1, dtype=np.int32)
    states[0:2, 0] = pair_a
    states[2:, 2] = pair_a
    states[0:3, 1] = pair_b
    states[3:, 1] = pair_c
    raw = np.zeros((5, 12), dtype=np.bool_)
    raw[0:2, 0] = True
    raw[0:2, 1] = True
    if enabled:
        memory = _reconstruct_enabled_feature_memory_trace(
            states,
            raw,
            confirmation_steps=2,
        )
    else:
        zeros_i32 = np.zeros_like(raw, dtype=np.int32)
        zeros_bool = np.zeros_like(raw, dtype=np.bool_)
        memory = {
            "confirmed": raw.copy(),
            "streak_pre": zeros_i32.copy(),
            "streak_post": zeros_i32.copy(),
            "committed_pre": zeros_bool.copy(),
            "committed_post": zeros_bool.copy(),
        }

    state_weights = np.zeros((6, 1, 12), dtype=np.float32)
    # A bootstraps, confirms, and follows its identity from slot 0 to slot 2.
    state_weights[1, 0, 0] = 1.0
    state_weights[2:, 0, 2] = 2.0
    # B confirms before replacement; C may start from a seeded candidate head
    # and remains plastic while it is uncommitted.
    state_weights[1, 0, 1] = 1.0
    state_weights[2, 0, 1] = 2.0
    state_weights[3, 0, 1] = 7.0
    state_weights[4, 0, 1] = 8.0
    state_weights[5, 0, 1] = 9.0
    if not enabled:
        state_weights[3:, 0, 2] = np.asarray((3.0, 4.0, 5.0))
    memory["weights_pre"] = state_weights[:-1].copy()
    memory["weights_post"] = state_weights[1:].copy()
    config = IntegratedHiddenPartnerConfig(
        evidence_gated_feature_memory=enabled,
        feature_evidence_confirmation_steps=2,
        active_utility_retention_grace_steps=16,
        active_utility_evidence_threshold=0.1,
    )
    values = {
        "deployed_pre": states[:-1],
        "deployed_post": states[1:],
        "shadow_pre": states[:-1].copy(),
        "shadow_post": states[1:].copy(),
        "raw_evidence": raw,
        "confirmed_evidence": memory["confirmed"],
        "output_weights_pre": memory["weights_pre"],
        "output_weights_post": memory["weights_post"],
        "streak_pre": memory["streak_pre"],
        "streak_post": memory["streak_post"],
        "committed_pre": memory["committed_pre"],
        "committed_post": memory["committed_post"],
    }
    return values, config


def _audit_feature_case(
    values: dict[str, np.ndarray],
    config: IntegratedHiddenPartnerConfig,
):
    return _feature_memory_contract_audit(
        values["deployed_pre"],
        values["deployed_post"],
        values["shadow_pre"],
        values["shadow_post"],
        values["raw_evidence"],
        values["confirmed_evidence"],
        values["output_weights_pre"],
        values["output_weights_post"],
        values["streak_pre"],
        values["streak_post"],
        values["committed_pre"],
        values["committed_post"],
        config,
    )


def _controlled_behavior_weight_states(
    descriptor_states: np.ndarray,
    write_gate: np.ndarray,
) -> np.ndarray:
    """Route fake behavior columns and learn only through an open gate."""
    steps, slots = write_gate.shape
    assert descriptor_states.shape == (steps + 1, slots, 2)
    weights = np.zeros((steps + 1, 2, slots), dtype=np.float32)
    for step in range(steps):
        for post_slot, descriptor in enumerate(descriptor_states[step + 1]):
            sources = np.flatnonzero(
                np.all(descriptor_states[step] == descriptor, axis=1)
            )
            if sources.size != 1 or descriptor[0] < 0:
                continue
            source = int(sources[0])
            weights[step + 1, :, post_slot] = weights[step, :, source]
            pair = tuple(int(value) for value in descriptor)
            if write_gate[step, source] and pair in (_C_PAIR, _D_PAIR):
                weights[step + 1, :, post_slot] = _LEARNED_COLUMN
    return weights


def _controlled_lifecycle(
    c_states: np.ndarray,
    d_states: np.ndarray,
    *,
    c_promotion_step: int | None,
    d_promotion_steps: tuple[int, ...],
    c_evidence_refresh_steps: tuple[int, ...],
    d_evidence_refresh_steps: tuple[int, ...],
    d_retirement_events: tuple[tuple[int, bool], ...],
):
    result = _base_result()
    trace = result.trace
    cycle_steps = result.summary.cycle_steps
    assert c_states.shape == d_states.shape == (cycle_steps + 1,)

    deployed_pre = np.asarray(trace.active_descriptors, dtype=np.int32).copy()
    deployed_post = np.asarray(
        trace.deployed_descriptors_post,
        dtype=np.int32,
    ).copy()
    states = np.concatenate(
        (deployed_pre[:cycle_steps], deployed_post[cycle_steps - 1 : cycle_steps]),
        axis=0,
    )
    for pair in (_C_PAIR, _D_PAIR):
        matches = np.all(
            states == np.asarray(pair, dtype=np.int32),
            axis=-1,
        )
        states[matches] = np.asarray((-1, -1), dtype=np.int32)
    states[c_states, 0] = np.asarray(_C_PAIR, dtype=np.int32)
    states[d_states, 1] = np.asarray(_D_PAIR, dtype=np.int32)

    candidates = np.asarray(trace.candidate_descriptors, dtype=np.int32)
    c_candidate = _candidate_index(candidates, _C_PAIR)
    d_candidate = _candidate_index(candidates, _D_PAIR)
    promoted = np.full(
        trace.interaction_promoted_candidate.shape,
        -1,
        dtype=np.int32,
    )
    if c_promotion_step is not None:
        promoted[c_promotion_step] = c_candidate
    for step in d_promotion_steps:
        promoted[step] = d_candidate

    evidence_refreshed = np.zeros(
        trace.interaction_evidence_refreshed.shape,
        dtype=np.bool_,
    )
    for pair, refresh_steps in (
        (_C_PAIR, c_evidence_refresh_steps),
        (_D_PAIR, d_evidence_refresh_steps),
    ):
        target = np.asarray(pair, dtype=np.int32)
        for step in refresh_steps:
            matches = np.all(states[step] == target, axis=1)
            assert np.sum(matches) == 1
            evidence_refreshed[step, int(np.flatnonzero(matches)[0])] = True
    config = dataclasses.replace(
        result.condition.config,
        evidence_gated_feature_memory=True,
        feature_evidence_confirmation_steps=1,
        active_utility_retention_grace_steps=4_096,
        active_utility_evidence_threshold=0.1,
    )
    feature_memory = _reconstruct_enabled_feature_memory_trace(
        states,
        evidence_refreshed,
        confirmation_steps=config.feature_evidence_confirmation_steps,
    )

    retired_slot = np.full(
        trace.interaction_retired_slot.shape,
        -1,
        dtype=np.int32,
    )
    retired_left = np.full(
        trace.interaction_retired_left.shape,
        -1,
        dtype=np.int32,
    )
    retired_right = np.full(
        trace.interaction_retired_right.shape,
        -1,
        dtype=np.int32,
    )
    reset_mask = np.zeros(
        trace.interaction_matching_candidate_reset_mask.shape,
        dtype=np.bool_,
    )
    reset_count = np.zeros(
        trace.interaction_matching_candidate_reset_count.shape,
        dtype=np.int32,
    )
    candidate_utilities_post = np.asarray(
        trace.candidate_utilities_post,
        dtype=np.float32,
    ).copy()
    candidate_weights_post = np.asarray(
        trace.candidate_output_weights_post,
        dtype=np.float32,
    ).copy()
    candidate_ages_post = np.asarray(
        trace.candidate_ages_post,
        dtype=np.int32,
    ).copy()
    for event_step, reset in d_retirement_events:
        retired_slot[event_step] = 1
        retired_left[event_step] = _D_PAIR[0]
        retired_right[event_step] = _D_PAIR[1]
        if reset:
            reset_mask[event_step, d_candidate] = True
            reset_count[event_step] = 1
            candidate_utilities_post[event_step, d_candidate] = 0.0
            candidate_weights_post[event_step, :, d_candidate] = 0.0
            candidate_ages_post[event_step, d_candidate] = 0

    intended_actions = np.ones(
        trace.partner_intended_action.shape,
        dtype=np.int32,
    )
    behavior_logits = np.broadcast_to(
        _LEARNED_COLUMN,
        trace.behavior_logits_preupdate.shape,
    ).copy()
    behavior_probabilities = np.exp(
        behavior_logits - np.max(behavior_logits, axis=1, keepdims=True)
    )
    behavior_probabilities /= np.sum(
        behavior_probabilities,
        axis=1,
        keepdims=True,
    )
    consumer_write_gate = np.ones(
        trace.consumer_write_gate_pre.shape,
        dtype=np.bool_,
    )
    behavior_pair_weight_states = _controlled_behavior_weight_states(
        states,
        consumer_write_gate,
    )
    behavior_pair_weights = behavior_pair_weight_states[:-1]
    behavior_pair_weights_post = behavior_pair_weight_states[1:]
    control_pair_weights_pre = np.zeros_like(
        np.asarray(trace.control_pair_weights_pre, dtype=np.float32)
    )
    control_pair_weights_post = np.zeros_like(
        np.asarray(trace.control_pair_weights_post, dtype=np.float32)
    )
    control_pair_trace_weights_pre = np.zeros_like(
        np.asarray(trace.control_pair_trace_weights_pre, dtype=np.float32)
    )
    control_pair_trace_weights_post = np.zeros_like(
        np.asarray(trace.control_pair_trace_weights_post, dtype=np.float32)
    )
    pair_features = np.zeros(
        trace.deployed_pair_features.shape,
        dtype=np.float32,
    )
    for step in range(cycle_steps):
        for pair in (_C_PAIR, _D_PAIR):
            matches = np.all(
                states[step] == np.asarray(pair, dtype=np.int32),
                axis=1,
            )
            if np.any(matches):
                slot = int(np.flatnonzero(matches)[0])
                pair_features[step, slot] = 1.0

    canonical_candidates = np.asarray(
        tuple(
            (left, right)
            for left in range(12)
            for right in range(left + 1, 12)
        ),
        dtype=np.int32,
    )
    candidate_descriptors = np.broadcast_to(
        canonical_candidates,
        trace.candidate_descriptors.shape,
    ).copy()
    candidate_descriptors_post = np.broadcast_to(
        canonical_candidates,
        trace.candidate_descriptors_post.shape,
    ).copy()
    consumer_mask_pre = np.all(states[:-1] >= 0, axis=2)
    consumer_mask_post = np.all(states[1:] >= 0, axis=2)
    pair_features[~consumer_mask_pre] = 0.0
    c_masked_nll = np.where(
        c_states[:-1],
        _COLUMN_LEARNING_NLL_GAIN,
        0.0,
    ).astype(np.float32)
    d_masked_nll = np.where(
        d_states[:-1],
        _COLUMN_LEARNING_NLL_GAIN,
        0.0,
    ).astype(np.float32)
    controlled_trace = trace.replace(
        active_descriptors=jnp.asarray(states[:-1]),
        deployed_descriptors_post=jnp.asarray(states[1:]),
        shadow_descriptors_pre=jnp.asarray(states[:-1]),
        shadow_descriptors_post=jnp.asarray(states[1:]),
        candidate_descriptors=jnp.asarray(candidate_descriptors),
        candidate_descriptors_post=jnp.asarray(candidate_descriptors_post),
        partner_intended_action=jnp.asarray(intended_actions),
        behavior_logits_preupdate=jnp.asarray(behavior_logits),
        behavior_probabilities=jnp.asarray(behavior_probabilities),
        behavior_nll=jnp.full_like(
            trace.behavior_nll,
            _ONLINE_NLL,
            dtype=jnp.float32,
        ),
        deployed_pair_features=jnp.asarray(pair_features),
        behavior_pair_weights_pre=jnp.asarray(behavior_pair_weights),
        behavior_pair_weights_post=jnp.asarray(behavior_pair_weights_post),
        control_pair_weights_pre=jnp.asarray(control_pair_weights_pre),
        control_pair_weights_post=jnp.asarray(control_pair_weights_post),
        control_pair_trace_weights_pre=jnp.asarray(
            control_pair_trace_weights_pre
        ),
        control_pair_trace_weights_post=jnp.asarray(
            control_pair_trace_weights_post
        ),
        behavior_intended_correct=jnp.ones_like(
            trace.behavior_intended_correct,
            dtype=jnp.bool_,
        ),
        reward=jnp.ones_like(trace.reward, dtype=jnp.float32),
        candidate_utilities_post=jnp.asarray(candidate_utilities_post),
        candidate_output_weights_post=jnp.asarray(candidate_weights_post),
        candidate_ages_post=jnp.asarray(candidate_ages_post),
        interaction_promoted_candidate=jnp.asarray(promoted),
        interaction_evidence_refreshed=jnp.asarray(evidence_refreshed),
        interaction_retention_evidence_refreshed=jnp.asarray(
            feature_memory["confirmed"]
        ),
        interaction_output_weights_pre=jnp.asarray(
            feature_memory["weights_pre"]
        ),
        interaction_output_weights_post=jnp.asarray(
            feature_memory["weights_post"]
        ),
        interaction_utility_evidence_streak_pre=jnp.asarray(
            feature_memory["streak_pre"]
        ),
        interaction_utility_evidence_streak_post=jnp.asarray(
            feature_memory["streak_post"]
        ),
        interaction_active_output_memory_committed_pre=jnp.asarray(
            feature_memory["committed_pre"]
        ),
        interaction_active_output_memory_committed_post=jnp.asarray(
            feature_memory["committed_post"]
        ),
        consumer_write_gate_pre=jnp.asarray(consumer_write_gate),
        consumer_active_mask_pre=jnp.asarray(consumer_mask_pre),
        consumer_active_mask_post=jnp.asarray(consumer_mask_post),
        interaction_retired_slot=jnp.asarray(retired_slot),
        interaction_retired_left=jnp.asarray(retired_left),
        interaction_retired_right=jnp.asarray(retired_right),
        interaction_matching_candidate_reset_mask=jnp.asarray(reset_mask),
        interaction_matching_candidate_reset_count=jnp.asarray(reset_count),
        c_masked_behavior_nll_increase=jnp.asarray(c_masked_nll),
        d_masked_behavior_nll_increase=jnp.asarray(d_masked_nll),
    )
    condition = dataclasses.replace(result.condition, config=config)
    return dataclasses.replace(
        result,
        condition=condition,
        trace=controlled_trace,
    )


def _successful_control():
    result = _base_result()
    ends = _bounds(result)
    steps = result.summary.cycle_steps
    c_states = np.zeros(steps + 1, dtype=np.bool_)
    d_states = np.zeros(steps + 1, dtype=np.bool_)
    c_promotion = int(ends[5])
    d_promotion = int(ends[3])
    d_retirement = int(ends[4])
    c_states[c_promotion + 1 :] = True
    d_states[d_promotion + 1 : d_retirement + 1] = True
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=c_promotion,
        d_promotion_steps=(d_promotion,),
        c_evidence_refresh_steps=(c_promotion + 1,),
        d_evidence_refresh_steps=(d_promotion + 1,),
        d_retirement_events=((d_retirement, True),),
    )
    return controlled, ends, c_states, d_states


def _reconstruct_enabled_consumer_trace(
    deployed_pre: np.ndarray,
    deployed_post: np.ndarray,
    evidence: np.ndarray,
    config: IntegratedHiddenPartnerConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the exact enabled-gate trace used by controlled public summaries."""
    steps, slots = evidence.shape
    live_pre = np.all(
        (deployed_pre >= 0) & (deployed_pre < 12),
        axis=2,
    )
    live_post = np.all(
        (deployed_post >= 0) & (deployed_post < 12),
        axis=2,
    )
    write_gate = np.zeros((steps, slots), dtype=np.bool_)
    idle_pre = np.zeros((steps, slots), dtype=np.int32)
    idle_post = np.zeros((steps, slots), dtype=np.int32)
    mask_pre = np.zeros((steps, slots), dtype=np.bool_)
    mask_post = np.zeros((steps, slots), dtype=np.bool_)
    streak = np.zeros((slots,), dtype=np.int32)
    idle = np.zeros((slots,), dtype=np.int32)
    mask = np.zeros((slots,), dtype=np.bool_)

    for step in range(steps):
        mask_pre[step] = mask
        idle_pre[step] = idle
        streak = np.where(
            live_pre[step] & evidence[step],
            streak + 1,
            0,
        ).astype(np.int32)
        read_acquire = (
            live_pre[step]
            & evidence[step]
            & (streak >= config.consumer_read_confirmation_steps)
        )
        write_gate[step] = (
            live_pre[step]
            & evidence[step]
            & (streak >= config.consumer_evidence_confirmation_steps)
        )
        idle = np.where(
            live_pre[step],
            np.where(evidence[step], 0, idle + 1),
            0,
        ).astype(np.int32)

        next_streak = np.zeros_like(streak)
        next_idle = np.zeros_like(idle)
        next_mask = np.zeros_like(mask)
        for post_slot, descriptor in enumerate(deployed_post[step]):
            if not live_post[step, post_slot]:
                continue
            sources = np.flatnonzero(
                np.all(deployed_pre[step] == descriptor, axis=1)
            )
            if sources.size != 1:
                continue
            source = int(sources[0])
            next_streak[post_slot] = streak[source]
            next_idle[post_slot] = idle[source]
            next_mask[post_slot] = bool(
                (mask[source] or read_acquire[source])
                and next_idle[post_slot] <= config.consumer_read_lease_steps
            )
        mask_post[step] = next_mask
        idle_post[step] = next_idle
        streak = next_streak
        idle = next_idle
        mask = next_mask

    return write_gate, idle_pre, idle_post, mask_pre, mask_post


def _consumer_write_audit_case() -> tuple[dict[str, np.ndarray], IntegratedHiddenPartnerConfig]:
    """Construct a small exact trace with one confirmed durable write."""
    config = IntegratedHiddenPartnerConfig(
        evidence_gated_consumer_memory=True,
        active_utility_retention_grace_steps=16,
        active_utility_evidence_threshold=0.1,
        consumer_read_confirmation_steps=1,
        consumer_evidence_confirmation_steps=2,
        consumer_read_lease_steps=2,
    )
    states = np.full((6, 12, 2), -1, dtype=np.int32)
    states[:, 0] = np.asarray((0, 1), dtype=np.int32)
    states[:3, 1] = np.asarray((2, 3), dtype=np.int32)
    states[3:, 1] = np.asarray((4, 5), dtype=np.int32)
    evidence = np.zeros((5, 12), dtype=np.bool_)
    evidence[:2, 0] = True
    write_gate, idle_pre, idle_post, mask_pre, mask_post = (
        _reconstruct_enabled_consumer_trace(
            states[:-1],
            states[1:],
            evidence,
            config,
        )
    )
    behavior = np.zeros((6, 2, 12), dtype=np.float32)
    control_q = np.zeros_like(behavior)
    control_trace = np.zeros_like(behavior)
    behavior[2:, :, 0] = np.asarray((1.0, 2.0), dtype=np.float32)
    control_q[2:, :, 0] = np.asarray((3.0, 4.0), dtype=np.float32)
    control_trace[2, :, 0] = np.asarray((5.0, 6.0), dtype=np.float32)
    values = {
        "deployed_pre": states[:-1],
        "deployed_post": states[1:],
        "evidence": evidence,
        "write_gate": write_gate,
        "idle_pre": idle_pre,
        "idle_post": idle_post,
        "mask_pre": mask_pre,
        "mask_post": mask_post,
        "representations": np.zeros((5, 12), dtype=np.float32),
        "behavior_pre": behavior[:-1],
        "behavior_post": behavior[1:],
        "control_q_pre": control_q[:-1],
        "control_q_post": control_q[1:],
        "control_trace_pre": control_trace[:-1],
        "control_trace_post": control_trace[1:],
    }
    return values, config


def _audit_consumer_case(
    values: dict[str, np.ndarray],
    config: IntegratedHiddenPartnerConfig,
):
    return _consumer_gate_contract_audit(
        values["deployed_pre"],
        values["deployed_post"],
        values["evidence"],
        values["write_gate"],
        values["idle_pre"],
        values["idle_post"],
        values["mask_pre"],
        values["mask_post"],
        values["representations"],
        values["behavior_pre"],
        values["behavior_post"],
        values["control_q_pre"],
        values["control_q_post"],
        values["control_trace_pre"],
        values["control_trace_post"],
        config,
    )


def _successful_enabled_consumer_control():
    controlled, ends, c_states, d_states = _successful_control()
    config = dataclasses.replace(
        controlled.condition.config,
        evidence_gated_consumer_memory=True,
        active_utility_retention_grace_steps=4_096,
        active_utility_evidence_threshold=0.1,
        consumer_read_confirmation_steps=2,
        consumer_evidence_confirmation_steps=3,
        consumer_read_lease_steps=2,
    )
    deployed_pre = np.asarray(
        controlled.trace.active_descriptors,
        dtype=np.int32,
    )
    deployed_post = np.asarray(
        controlled.trace.deployed_descriptors_post,
        dtype=np.int32,
    )
    evidence = np.zeros(
        controlled.trace.interaction_evidence_refreshed.shape,
        dtype=np.bool_,
    )
    for pair in (_C_PAIR, _D_PAIR):
        evidence |= np.all(
            deployed_pre == np.asarray(pair, dtype=np.int32),
            axis=2,
        )
    write_gate, idle_pre, idle_post, mask_pre, mask_post = (
        _reconstruct_enabled_consumer_trace(
        deployed_pre,
        deployed_post,
        evidence,
        config,
        )
    )
    states = np.concatenate((deployed_pre, deployed_post[-1:]), axis=0)
    feature_memory = _reconstruct_enabled_feature_memory_trace(
        states,
        evidence,
        confirmation_steps=config.feature_evidence_confirmation_steps,
    )
    behavior_weight_states = _controlled_behavior_weight_states(
        states,
        write_gate,
    )
    representations = np.asarray(
        controlled.trace.deployed_pair_features,
        dtype=np.float32,
    ).copy()
    representations[~mask_pre] = 0.0
    trace = controlled.trace.replace(
        interaction_evidence_refreshed=jnp.asarray(evidence),
        interaction_retention_evidence_refreshed=jnp.asarray(
            feature_memory["confirmed"]
        ),
        interaction_output_weights_pre=jnp.asarray(
            feature_memory["weights_pre"]
        ),
        interaction_output_weights_post=jnp.asarray(
            feature_memory["weights_post"]
        ),
        interaction_utility_evidence_streak_pre=jnp.asarray(
            feature_memory["streak_pre"]
        ),
        interaction_utility_evidence_streak_post=jnp.asarray(
            feature_memory["streak_post"]
        ),
        interaction_active_output_memory_committed_pre=jnp.asarray(
            feature_memory["committed_pre"]
        ),
        interaction_active_output_memory_committed_post=jnp.asarray(
            feature_memory["committed_post"]
        ),
        consumer_write_gate_pre=jnp.asarray(write_gate),
        consumer_read_idle_steps_pre=jnp.asarray(idle_pre),
        consumer_read_idle_steps_post=jnp.asarray(idle_post),
        consumer_active_mask_pre=jnp.asarray(mask_pre),
        consumer_active_mask_post=jnp.asarray(mask_post),
        deployed_pair_features=jnp.asarray(representations),
        behavior_pair_weights_pre=jnp.asarray(behavior_weight_states[:-1]),
        behavior_pair_weights_post=jnp.asarray(behavior_weight_states[1:]),
    )
    condition = dataclasses.replace(controlled.condition, config=config)
    return (
        dataclasses.replace(
            controlled,
            condition=condition,
            trace=trace,
        ),
        ends,
        c_states,
        d_states,
    )


def test_enabled_consumer_gate_reconstructs_confirmation_routing_and_lease() -> None:
    config = IntegratedHiddenPartnerConfig(
        evidence_gated_consumer_memory=True,
        active_utility_retention_grace_steps=16,
        active_utility_evidence_threshold=0.1,
        consumer_read_confirmation_steps=2,
        consumer_evidence_confirmation_steps=3,
        consumer_read_lease_steps=2,
    )
    pair_a = np.asarray((0, 1), dtype=np.int32)
    pair_b = np.asarray((2, 3), dtype=np.int32)
    pair_c = np.asarray((4, 6), dtype=np.int32)
    states = np.full((9, 12, 2), -1, dtype=np.int32)
    states[:3, 0] = pair_a
    states[:4, 1] = pair_b
    states[3:, 2] = pair_a
    states[4:, 1] = pair_c
    deployed_pre = states[:-1]
    deployed_post = states[1:]

    evidence = np.zeros((8, 12), dtype=np.bool_)
    evidence[0:2, 0:2] = True
    evidence[2, 0] = True
    evidence[4:7, 1] = True

    write_gate = np.zeros_like(evidence)
    write_gate[2, 0] = True
    write_gate[6, 1] = True
    mask_pre = np.zeros_like(evidence)
    mask_pre[2, (0, 1)] = True
    mask_pre[3, (1, 2)] = True
    mask_pre[4:6, 2] = True
    mask_pre[6:8, 1] = True
    mask_post = np.zeros_like(evidence)
    mask_post[1, (0, 1)] = True
    mask_post[2, (1, 2)] = True
    mask_post[3:5, 2] = True
    mask_post[5:8, 1] = True
    representations = np.zeros_like(evidence, dtype=np.float32)
    (
        reconstructed_write,
        idle_pre,
        idle_post,
        reconstructed_mask_pre,
        reconstructed_mask_post,
    ) = _reconstruct_enabled_consumer_trace(
        deployed_pre,
        deployed_post,
        evidence,
        config,
    )
    zero_weight_states = np.zeros((9, 2, 12), dtype=np.float32)
    assert np.array_equal(write_gate, reconstructed_write)
    assert np.array_equal(mask_pre, reconstructed_mask_pre)
    assert np.array_equal(mask_post, reconstructed_mask_post)

    assert _consumer_gate_contract_valid(
        deployed_pre,
        deployed_post,
        evidence,
        write_gate,
        idle_pre,
        idle_post,
        mask_pre,
        mask_post,
        representations,
        zero_weight_states[:-1],
        zero_weight_states[1:],
        zero_weight_states[:-1],
        zero_weight_states[1:],
        zero_weight_states[:-1],
        zero_weight_states[1:],
        config,
    )
    # A's acquired read lease follows its identity from slot 0 to slot 2.
    assert mask_pre[2, 0] and mask_post[2, 2]
    # C replaces an open B column but starts closed; it acquires read access
    # one evidence step before its stricter write confirmation.
    assert mask_pre[3, 1] and not mask_post[3, 1]
    assert mask_post[5, 1] and not write_gate[5, 1]
    assert write_gate[6, 1]
    # A remains readable for exactly two evidence-idle steps, then expires.
    assert mask_post[4, 2] and not mask_post[5, 2]


def test_consumer_write_audit_allows_exact_confirmed_writes() -> None:
    values, config = _consumer_write_audit_case()
    audit = _audit_consumer_case(values, config)

    assert values["write_gate"][1, 0]
    assert np.any(values["behavior_pre"][1, :, 0] != values["behavior_post"][1, :, 0])
    assert np.any(values["control_q_pre"][1, :, 0] != values["control_q_post"][1, :, 0])
    assert np.any(values["control_trace_post"][1, :, 0] != 0.0)
    assert audit.valid
    assert not np.any(audit.write_violation_bits)


@pytest.mark.parametrize(
    ("pre_field", "post_field"),
    (
        ("behavior_pre", "behavior_post"),
        ("control_q_pre", "control_q_post"),
    ),
)
def test_consumer_write_audit_rejects_hidden_closed_durable_write(
    pre_field: str,
    post_field: str,
) -> None:
    values, config = _consumer_write_audit_case()
    states = np.concatenate((values[pre_field][:1], values[post_field]), axis=0)
    states[4:, 0, 0] += np.float32(11.0)
    values[pre_field] = states[:-1]
    values[post_field] = states[1:]

    audit = _audit_consumer_case(values, config)

    assert audit.write_violation_bits[3, 0]
    assert not audit.valid


def test_consumer_write_audit_rejects_nonzero_closed_q_trace() -> None:
    values, config = _consumer_write_audit_case()
    states = np.concatenate(
        (values["control_trace_pre"][:1], values["control_trace_post"]),
        axis=0,
    )
    states[4:, 0, 0] = np.float32(7.0)
    values["control_trace_pre"] = states[:-1]
    values["control_trace_post"] = states[1:]

    audit = _audit_consumer_case(values, config)

    assert audit.write_violation_bits[3, 0]
    assert not audit.valid


def test_consumer_write_audit_rejects_negative_zero_trace_erasure() -> None:
    values, config = _consumer_write_audit_case()
    states = np.concatenate(
        (values["control_trace_pre"][:1], values["control_trace_post"]),
        axis=0,
    )
    states[4:, 0, 0] = np.float32(-0.0)
    assert states[4, 0, 0].view(np.uint32) == np.uint32(0x80000000)
    values["control_trace_pre"] = states[:-1]
    values["control_trace_post"] = states[1:]

    audit = _audit_consumer_case(values, config)

    assert audit.write_violation_bits[3, 0]
    assert not audit.valid


def test_consumer_write_audit_rejects_new_identity_inheritance() -> None:
    values, config = _consumer_write_audit_case()
    states = np.concatenate(
        (values["behavior_pre"][:1], values["behavior_post"]),
        axis=0,
    )
    # C first appears at state 3 in physical slot 1 and must not inherit B.
    states[3:, 0, 1] = np.float32(9.0)
    values["behavior_pre"] = states[:-1]
    values["behavior_post"] = states[1:]

    audit = _audit_consumer_case(values, config)

    assert audit.write_violation_bits[2, 1]
    assert not audit.valid


def test_consumer_write_audit_rejects_nonfinite_state() -> None:
    values, config = _consumer_write_audit_case()
    states = np.concatenate(
        (values["control_q_pre"][:1], values["control_q_post"]),
        axis=0,
    )
    states[4:, 0, 0] = np.float32(np.nan)
    values["control_q_pre"] = states[:-1]
    values["control_q_post"] = states[1:]

    audit = _audit_consumer_case(values, config)

    assert audit.write_violation_bits[3, 0]
    assert not audit.valid


def test_feature_memory_routes_identity_and_allows_uncommitted_bootstrap() -> None:
    values, config = _feature_memory_audit_case()
    audit = _audit_feature_case(values, config)

    assert audit.valid
    # A changes on its two bootstrap/confirmation writes and follows its
    # identity from slot 0 into slot 2.
    assert audit.identity_routed_head_changed[0, 0]
    assert audit.identity_routed_head_changed[1, 0]
    assert not audit.identity_routed_head_changed[2, 2]
    # C is a new uncommitted identity and may keep learning without evidence.
    assert not values["committed_post"][2, 1]
    assert audit.identity_routed_head_changed[3, 1]
    assert audit.identity_routed_head_changed[4, 1]


def test_feature_memory_rejects_unconfirmed_committed_head_write() -> None:
    values, config = _feature_memory_audit_case()
    weights_pre = values["output_weights_pre"].copy()
    weights_post = values["output_weights_post"].copy()
    # A is committed at pre-state 2.  Change it without evidence, then carry
    # the tampered value forward so only the protected write is at issue.
    weights_post[2:, 0, 2] = 3.0
    weights_pre[3:, 0, 2] = 3.0
    values["output_weights_pre"] = weights_pre
    values["output_weights_post"] = weights_post

    audit = _audit_feature_case(values, config)

    assert audit.identity_routed_head_changed[2, 2]
    assert audit.violation_bits[2, 2]
    assert not audit.valid


@pytest.mark.parametrize(
    ("field", "step", "slot"),
    (
        ("confirmed_evidence", 1, 0),
        ("streak_pre", 1, 0),
        ("streak_post", 1, 2),
        ("committed_pre", 2, 2),
        ("committed_post", 1, 2),
    ),
)
def test_feature_memory_rejects_state_or_confirmation_tampering(
    field: str,
    step: int,
    slot: int,
) -> None:
    values, config = _feature_memory_audit_case()
    tampered = values[field].copy()
    if np.issubdtype(tampered.dtype, np.bool_):
        tampered[step, slot] = ~tampered[step, slot]
    else:
        tampered[step, slot] += 1
    values[field] = tampered

    audit = _audit_feature_case(values, config)

    assert audit.violation_bits[step, slot]
    assert not audit.valid


def test_feature_memory_replacement_resets_streak_and_commitment() -> None:
    values, config = _feature_memory_audit_case()
    # B is committed before C replaces it in the same physical slot.
    assert values["committed_pre"][2, 1]
    assert not values["committed_post"][2, 1]
    assert values["streak_post"][2, 1] == 0
    assert _audit_feature_case(values, config).valid

    values["committed_post"] = values["committed_post"].copy()
    values["committed_post"][2, 1] = True
    assert not _audit_feature_case(values, config).valid


def test_disabled_feature_memory_has_explicit_legacy_semantics() -> None:
    values, config = _feature_memory_audit_case(enabled=False)
    audit = _audit_feature_case(values, config)

    assert audit.valid
    assert np.array_equal(values["confirmed_evidence"], values["raw_evidence"])
    assert not np.any(values["committed_pre"])
    assert not np.any(values["committed_post"])
    # With the mechanism disabled, an otherwise-protected A head can change.
    assert audit.identity_routed_head_changed[2, 2]


@pytest.mark.parametrize(
    "tampered_field",
    (
        "consumer_write_gate_pre",
        "consumer_read_idle_steps_pre",
        "consumer_read_idle_steps_post",
        "consumer_active_mask_post",
    ),
)
def test_consumer_gate_tampering_fails_contract_and_joint_success(
    tampered_field: str,
) -> None:
    controlled, _, _, _ = _successful_enabled_consumer_control()
    baseline = summarize_critical_lifecycle_v2(controlled)
    assert baseline.consumer_gate_contract_valid
    assert baseline.joint_memory_management_success

    values = np.asarray(getattr(controlled.trace, tampered_field)).copy()
    if tampered_field == "consumer_write_gate_pre":
        confirmed = np.argwhere(values)
        assert confirmed.size > 0
        step, slot = (int(value) for value in confirmed[0])
    elif tampered_field in (
        "consumer_read_idle_steps_pre",
        "consumer_read_idle_steps_post",
    ):
        step, slot = 0, 0
    else:
        step, slot = 0, 0
    if np.issubdtype(values.dtype, np.bool_):
        values[step, slot] = ~values[step, slot]
    else:
        values[step, slot] += 1
    trace = controlled.trace.replace(
        **{tampered_field: jnp.asarray(values)},
    )
    tampered = dataclasses.replace(controlled, trace=trace)
    lifecycle = summarize_critical_lifecycle_v2(tampered)

    assert not lifecycle.consumer_gate_contract_valid
    assert not lifecycle.joint_memory_management_success


def test_joint_success_requires_attributed_learning_use_and_retirement() -> None:
    controlled, ends, _, _ = _successful_control()
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.representation_link_contract_valid
    assert lifecycle.c_target_evidence_refresh_steps == (int(ends[5]) + 1,)
    assert lifecycle.c_acquisition_step == int(ends[5]) + 2
    assert np.isclose(lifecycle.c_first_late_online_nll, _ONLINE_NLL)
    assert np.isclose(
        lifecycle.c_first_late_entry_frozen_critical_nll,
        _ENTRY_FROZEN_NLL,
    )
    assert np.isclose(
        lifecycle.c_critical_column_learning_nll_gain,
        _COLUMN_LEARNING_NLL_GAIN,
    )
    assert lifecycle.c_critical_column_learning_positive_fraction == 1.0
    assert np.isclose(
        lifecycle.c_critical_column_target_created_share,
        1.0,
    )
    assert lifecycle.c_first_late_entry_frozen_critical_accuracy == 0.0
    assert lifecycle.c_critical_column_learning_accuracy_gain == 1.0
    assert lifecycle.c_task_learned
    assert lifecycle.c_continuously_survived
    assert lifecycle.c_retained_and_used
    assert lifecycle.d_target_evidence_refresh_steps == (int(ends[3]) + 1,)
    assert lifecycle.d_acquisition_step == int(ends[3]) + 2
    assert np.isclose(
        lifecycle.d_critical_column_learning_nll_gain,
        _COLUMN_LEARNING_NLL_GAIN,
    )
    assert lifecycle.d_critical_column_learning_positive_fraction == 1.0
    assert np.isclose(
        lifecycle.d_critical_column_target_created_share,
        1.0,
    )
    assert lifecycle.d_task_learned
    assert lifecycle.d_retirement_event_step == int(ends[4])
    assert lifecycle.d_retirement_step == int(ends[4]) + 1
    assert lifecycle.d_retirement_event_latency_steps == 0
    assert lifecycle.d_retirement_latency_steps == 1
    assert lifecycle.d_retirement_event_aligned
    assert lifecycle.d_linked_matching_candidate_reset_count == 1
    assert lifecycle.d_linked_candidate_utility_post == 0.0
    assert lifecycle.d_linked_candidate_head_linf_post == 0.0
    assert lifecycle.d_linked_candidate_age_post == 0
    assert lifecycle.d_learned_then_stably_retired
    assert lifecycle.feature_memory_enabled
    assert lifecycle.feature_memory_contract_valid
    assert lifecycle.joint_memory_management_success
    assert lifecycle.c_lifecycle_rle[-1].end_exclusive == (
        controlled.summary.cycle_steps + 1
    )


def test_grid_contract_predicate_includes_every_lifecycle_contract() -> None:
    controlled, _, _, _ = _successful_control()
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert _all_run_contracts_valid(controlled.summary, lifecycle)
    for field in (
        "representation_link_contract_valid",
        "consumer_gate_contract_valid",
        "feature_memory_enabled",
        "feature_memory_contract_valid",
        "candidate_archive_contract_valid",
    ):
        assert not _all_run_contracts_valid(
            controlled.summary,
            dataclasses.replace(lifecycle, **{field: False}),
        )


def test_joint_success_requires_feature_memory_to_be_enabled() -> None:
    controlled, _, _, _ = _successful_control()
    raw_evidence = np.asarray(
        controlled.trace.interaction_evidence_refreshed,
        dtype=np.bool_,
    )
    zeros_i32 = np.zeros_like(raw_evidence, dtype=np.int32)
    zeros_bool = np.zeros_like(raw_evidence, dtype=np.bool_)
    trace = controlled.trace.replace(
        interaction_retention_evidence_refreshed=jnp.asarray(raw_evidence),
        interaction_utility_evidence_streak_pre=jnp.asarray(zeros_i32),
        interaction_utility_evidence_streak_post=jnp.asarray(zeros_i32),
        interaction_active_output_memory_committed_pre=jnp.asarray(zeros_bool),
        interaction_active_output_memory_committed_post=jnp.asarray(zeros_bool),
    )
    config = dataclasses.replace(
        controlled.condition.config,
        evidence_gated_feature_memory=False,
    )
    controlled = dataclasses.replace(
        controlled,
        condition=dataclasses.replace(controlled.condition, config=config),
        trace=trace,
    )

    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert not lifecycle.feature_memory_enabled
    assert lifecycle.feature_memory_contract_valid
    assert lifecycle.c_retained_and_used
    assert lifecycle.d_learned_then_stably_retired
    assert not lifecycle.joint_memory_management_success


def _decode_float32_state_payload(payload: dict[str, object]) -> np.ndarray:
    shape = tuple(int(value) for value in payload["shape"])
    compressed = base64.b64decode(str(payload["data_base64"]))
    deltas = np.frombuffer(zlib.decompress(compressed), dtype="<u4").reshape(shape)
    bits = np.empty_like(deltas)
    bits[0] = deltas[0]
    for index in range(1, bits.shape[0]):
        bits[index] = deltas[index] ^ bits[index - 1]
    return bits.view("<f4")


def test_critical_primitives_include_exact_durable_state_audits() -> None:
    controlled, _, _, _ = _successful_control()
    primitives = critical_run_primitives(controlled)
    expected_shape = [controlled.summary.cycle_steps, 12]

    assert CRITICAL_RUN_PRIMITIVES_SCHEMA == (
        "alberta.hidden-partner-development.critical-run-primitives.v3"
    )
    assert primitives["schema_version"] == CRITICAL_RUN_PRIMITIVES_SCHEMA
    assert primitives["feature_memory_enabled"] is True
    for field in (
        "retention_evidence_refresh_bits",
        "feature_memory_committed_pre_bits",
        "feature_memory_committed_post_bits",
        "identity_routed_head_changed_bits",
        "feature_memory_contract_violation_bits",
        "consumer_write_contract_violation_bits",
    ):
        assert primitives[field]["shape"] == expected_shape
    for field, expected_tail in (
        ("feature_head_state_xor", [1, 12]),
        ("behavior_pair_weight_state_xor", [2, 12]),
        ("control_q_pair_weight_state_xor", [2, 12]),
        ("control_q_trace_state_xor", [2, 12]),
    ):
        payload = primitives[field]
        assert payload["shape"] == [controlled.summary.cycle_steps + 1, *expected_tail]
        assert {
            key: payload[key]
            for key in ("dtype", "byteorder", "delta", "codec")
        } == {
            "dtype": "float32",
            "byteorder": "little",
            "delta": "uint32-xor",
            "codec": "zlib",
        }
        assert np.all(np.isfinite(_decode_float32_state_payload(payload)))
    assert primitives["candidate_bank_state_rle"] == [
        {
            "start": 0,
            "end_exclusive": controlled.summary.cycle_steps + 1,
            "candidate_descriptors": np.asarray(
                controlled.trace.candidate_descriptors[0]
            ).tolist(),
        }
    ]


def test_float32_state_payload_preserves_exact_bits_and_requires_continuity() -> None:
    states = np.asarray(
        (
            ((0.0, -0.0),),
            ((1.0, -2.0),),
            ((3.5, 4.25),),
        ),
        dtype=np.float32,
    )
    payload = _float32_state_sequence_payload(states[:-1], states[1:])
    decoded = _decode_float32_state_payload(payload)
    assert np.array_equal(decoded.view("<u4"), states.view("<u4"))

    discontinuous = states[1:].copy()
    discontinuous[0, 0, 0] = np.float32(8.0)
    with pytest.raises(ValueError, match="bitwise continuous"):
        _float32_state_sequence_payload(states[:-1], discontinuous)


def test_candidate_archive_requires_every_exact_lexicographic_state() -> None:
    controlled, _, _, _ = _successful_control()
    candidate_pre = np.asarray(controlled.trace.candidate_descriptors, dtype=np.int32)
    candidate_post = np.asarray(
        controlled.trace.candidate_descriptors_post,
        dtype=np.int32,
    )
    assert _candidate_archive_contract_valid(candidate_pre, candidate_post)
    states = np.concatenate((candidate_pre[:1], candidate_post), axis=0)
    states[100:, 1] = np.asarray((0, 3), dtype=np.int32)
    trace = controlled.trace.replace(
        candidate_descriptors=jnp.asarray(states[:-1]),
        candidate_descriptors_post=jnp.asarray(states[1:]),
    )
    tampered = dataclasses.replace(controlled, trace=trace)

    lifecycle = summarize_critical_lifecycle_v2(tampered)

    assert not lifecycle.candidate_archive_contract_valid
    assert not lifecycle.joint_memory_management_success


def test_c_endpoint_presence_cannot_hide_one_intervening_gap() -> None:
    _, ends, c_states, d_states = _successful_control()
    gap_step = int(ends[6]) + 32
    c_states[gap_step] = False
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(int(ends[5]) + 1,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((int(ends[4]), True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_survival_gap_steps == 1
    assert lifecycle.c_first_survival_gap_step == gap_step
    assert lifecycle.c_evictions_after_acquisition == 1
    assert not lifecycle.c_continuously_survived
    assert not lifecycle.joint_memory_management_success


def test_promotion_without_pair_specific_target_evidence_is_not_acquired() -> None:
    _, ends, c_states, d_states = _successful_control()
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((int(ends[4]), True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_promotion_event_steps == (int(ends[5]),)
    assert lifecycle.c_target_evidence_refresh_steps == ()
    assert lifecycle.c_acquisition_step is None
    assert not lifecycle.c_task_learned
    assert not lifecycle.c_retained_and_used


def test_evidence_refresh_on_an_unrelated_slot_is_not_target_attribution() -> None:
    controlled, ends, _, _ = _successful_control()
    refresh_step = int(ends[5]) + 1
    evidence = np.asarray(
        controlled.trace.interaction_evidence_refreshed,
        dtype=np.bool_,
    ).copy()
    descriptors = np.asarray(
        controlled.trace.active_descriptors,
        dtype=np.int32,
    )
    c_matches = np.all(
        descriptors[refresh_step] == np.asarray(_C_PAIR, dtype=np.int32),
        axis=1,
    )
    c_slot = int(np.flatnonzero(c_matches)[0])
    unrelated_slot = (c_slot + 1) % descriptors.shape[1]
    assert tuple(descriptors[refresh_step, unrelated_slot]) != _C_PAIR
    evidence[refresh_step, c_slot] = False
    evidence[refresh_step, unrelated_slot] = True
    controlled = dataclasses.replace(
        controlled,
        trace=controlled.trace.replace(
            interaction_evidence_refreshed=jnp.asarray(evidence)
        ),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_promotion_event_steps == (int(ends[5]),)
    assert lifecycle.c_target_evidence_refresh_steps == ()
    assert lifecycle.c_acquisition_step is None
    assert not lifecycle.c_task_learned
    assert not lifecycle.c_retained_and_used
    assert lifecycle.d_task_learned


def test_inherited_critical_column_cannot_claim_target_created_learning() -> None:
    _, ends, c_states, d_states = _successful_control()
    first_c_start = int(ends[5])
    # Promote one transition earlier so the new column starts at exact zero,
    # then legitimately learns before the evaluation entry state.
    c_states[first_c_start - 1 :] = True
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=first_c_start - 2,
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(first_c_start,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((int(ends[4]), True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_acquisition_step == first_c_start + 1
    assert lifecycle.c_first_late_reward == 1.0
    assert lifecycle.c_first_late_intended_accuracy == 1.0
    assert lifecycle.c_first_late_masked_nll_increase > 0.0
    assert np.isclose(
        lifecycle.c_first_late_online_nll,
        _ONLINE_NLL,
    )
    assert np.isclose(
        lifecycle.c_first_late_entry_frozen_critical_nll,
        _ONLINE_NLL,
    )
    assert np.isclose(
        lifecycle.c_critical_column_learning_nll_gain,
        0.0,
    )
    assert lifecycle.c_critical_column_learning_positive_fraction == 0.0
    assert np.isclose(
        lifecycle.c_critical_column_target_created_share,
        0.0,
    )
    assert lifecycle.c_first_late_entry_frozen_critical_accuracy == 1.0
    assert lifecycle.c_critical_column_learning_accuracy_gain == 0.0
    assert not lifecycle.c_task_learned
    assert not lifecycle.c_retained_and_used
    assert not lifecycle.joint_memory_management_success


def test_old_d_retirement_cannot_validate_later_unrelated_disappearance() -> None:
    _, ends, c_states, d_states = _successful_control()
    old_event = int(ends[3]) + 64
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(int(ends[5]) + 1,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((old_event, True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.d_task_learned
    assert lifecycle.d_retirement_event_count == 1
    assert not lifecycle.d_retirement_event_aligned
    assert lifecycle.d_retirement_step is None
    assert not lifecycle.d_learned_then_stably_retired


def test_retiring_on_last_d_transition_is_not_post_exit_forgetting() -> None:
    _, ends, c_states, d_states = _successful_control()
    early_event = int(ends[4]) - 1
    d_states[int(ends[4])] = False
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(int(ends[5]) + 1,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((early_event, True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.d_task_learned
    assert not lifecycle.d_retirement_event_aligned
    assert not lifecycle.d_learned_then_stably_retired


def test_aligned_retirement_without_exact_candidate_reset_fails() -> None:
    _, ends, c_states, d_states = _successful_control()
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]),),
        c_evidence_refresh_steps=(int(ends[5]) + 1,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((int(ends[4]), False),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.d_task_learned
    assert not lifecycle.d_retirement_event_aligned
    assert not lifecycle.d_learned_then_stably_retired


def test_zero_recurrent_c_accuracy_fails_retained_use() -> None:
    controlled, ends, _, _ = _successful_control()
    correct = np.asarray(
        controlled.trace.behavior_intended_correct,
        dtype=np.bool_,
    ).copy()
    recurrent_start = int(ends[8])
    correct[
        recurrent_start : recurrent_start + RECURRENT_ENTRY_WINDOW
    ] = False
    controlled = dataclasses.replace(
        controlled,
        trace=controlled.trace.replace(
            behavior_intended_correct=jnp.asarray(correct)
        ),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_task_learned
    assert lifecycle.c_recurrent_early_intended_accuracy == 0.0
    assert not lifecycle.c_retained_and_used
    assert not lifecycle.joint_memory_management_success


def test_zero_initial_c_mask_effect_fails_feature_use_attribution() -> None:
    controlled, ends, _, _ = _successful_control()
    weights = np.asarray(
        controlled.trace.behavior_pair_weights_pre,
        dtype=np.float32,
    ).copy()
    weights[int(ends[5]) : int(ends[6])] = 0.0
    controlled = dataclasses.replace(
        controlled,
        trace=controlled.trace.replace(
            behavior_pair_weights_pre=jnp.asarray(weights)
        ),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.c_acquisition_step is not None
    assert not lifecycle.c_task_learned
    assert not lifecycle.c_retained_and_used


def test_final_transition_d_promotion_is_visible_and_breaks_stability() -> None:
    controlled, ends, c_states, d_states = _successful_control()
    final_event = controlled.summary.cycle_steps - 1
    d_states[-1] = True
    controlled = _controlled_lifecycle(
        c_states,
        d_states,
        c_promotion_step=int(ends[5]),
        d_promotion_steps=(int(ends[3]), final_event),
        c_evidence_refresh_steps=(int(ends[5]) + 1,),
        d_evidence_refresh_steps=(int(ends[3]) + 1,),
        d_retirement_events=((int(ends[4]), True),),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert lifecycle.d_repromotions_after_retirement == 1
    assert not lifecycle.d_absent_entire_final_window
    assert not lifecycle.d_learned_then_stably_retired
    assert lifecycle.d_lifecycle_rle[-1].deployed_slot == 1


def test_shadow_deployed_link_corruption_fails_representation_contract() -> None:
    controlled, _, _, _ = _successful_control()
    shadow = np.asarray(
        controlled.trace.shadow_descriptors_pre,
        dtype=np.int32,
    ).copy()
    shadow[100, 0] = np.asarray((-1, -1), dtype=np.int32)
    controlled = dataclasses.replace(
        controlled,
        trace=controlled.trace.replace(
            shadow_descriptors_pre=jnp.asarray(shadow)
        ),
    )
    lifecycle = summarize_critical_lifecycle_v2(controlled)

    assert not lifecycle.representation_link_contract_valid
    assert not lifecycle.c_task_learned
    assert not lifecycle.d_task_learned
    assert not lifecycle.joint_memory_management_success
