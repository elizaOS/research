"""Fail-closed tests for the forbidden evidence-lease tuning artifact."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import jax.random as jr
import numpy as np
import pytest

import alberta_framework.evaluation.hidden_partner_lease_tuning_artifact as artifact_module
import alberta_framework.evaluation.hidden_partner_lease_tuning_cli as cli_module
import alberta_framework.evaluation.hidden_partner_lifecycle_v2 as lifecycle_module
from alberta_framework.core.integrated_hidden_partner import (
    IntegratedHiddenPartnerConfig,
)
from alberta_framework.evaluation.hidden_partner_development import (
    HiddenPartnerFeatureSummary,
    HiddenPartnerRunSummary,
    HiddenPartnerSegmentSummary,
)
from alberta_framework.evaluation.hidden_partner_lease_tuning_artifact import (
    SOURCE_PATHS,
    _decode_float32_xor_state_payload,
    _expected_integrated_state_nbytes,
    lease_tuning_artifact_json,
    load_lease_tuning_artifact,
)
from alberta_framework.evaluation.hidden_partner_lease_tuning_artifact import (
    _build_lease_tuning_artifact_for_testing as build_lease_tuning_artifact,
)
from alberta_framework.evaluation.hidden_partner_lease_tuning_artifact import (
    _validate_lease_tuning_artifact_for_testing as validate_lease_tuning_artifact,
)
from alberta_framework.evaluation.hidden_partner_lease_tuning_cli import (
    DEFAULT_OUTPUT,
    main,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_v2 import (
    CRITICAL_RUN_PRIMITIVES_SCHEMA,
    FEATURE_LEARNING_WINDOW,
    LEASE_TUNING_GRID,
    LEASE_TUNING_SEED_COUNT,
    RECURRENT_ENTRY_WINDOW,
    CriticalLifecycleV2Summary,
    CriticalPairLifecycleInterval,
    run_evidence_lease_tuning_grid,
)
from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_REGIME_SCHEDULE,
    REGIME_NAMES,
    HiddenPartnerMappingWorld,
)


def _canonical_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def test_source_snapshot_declares_relevant_algorithm_closure() -> None:
    paths = {path.as_posix() for path in SOURCE_PATHS}

    assert not any(path.startswith("alberta_framework/benchmarks/") for path in paths)
    assert {
        "pyproject.toml",
        "uv.lock",
        "alberta_framework/core/integrated_hidden_partner.py",
        "alberta_framework/core/interaction_features.py",
        "alberta_framework/evaluation/hidden_partner_development.py",
        "alberta_framework/evaluation/hidden_partner_lifecycle_v2.py",
        "alberta_framework/evaluation/hidden_partner_lease_tuning_artifact.py",
        "alberta_framework/streams/hidden_partner_mapping.py",
    } <= paths


def test_cli_default_output_uses_reserved_v4_path() -> None:
    assert DEFAULT_OUTPUT == Path(
        "outputs/hidden_partner_development/lease_tuning_grid_a.v4.json"
    )


def test_forbidden_runner_has_no_seed_or_worker_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"seed_derivations": 0, "runner_constructions": 0}

    def derive_forbidden_seeds(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["seed_derivations"] += 1
        raise AssertionError("forbidden runner derived seeds")

    class ForbiddenRunner:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            calls["runner_constructions"] += 1
            raise AssertionError("forbidden runner constructed a worker")

    monkeypatch.setattr(
        lifecycle_module,
        "derive_hidden_partner_seed_pairs",
        derive_forbidden_seeds,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "HiddenPartnerDevelopmentRunner",
        ForbiddenRunner,
    )

    with pytest.raises(RuntimeError, match="FORBIDDEN/UNEXECUTED"):
        run_evidence_lease_tuning_grid()

    assert calls == {"seed_derivations": 0, "runner_constructions": 0}


def test_float32_xor_decoder_preserves_exact_bits() -> None:
    state_bits = np.asarray(
        (
            (0x00000000, 0x80000000),
            (0x3F800000, 0xBF800000),
            (0x40490FDB, 0x00000000),
        ),
        dtype="<u4",
    )
    states = state_bits.view("<f4")
    payload = _float32_xor_state_payload(states)
    errors: list[str] = []

    decoded = _decode_float32_xor_state_payload(
        payload,
        states.shape,
        "payload",
        errors,
    )

    assert not errors
    assert decoded is not None
    assert np.array_equal(decoded.view("<u4"), state_bits)


@pytest.mark.parametrize(
    ("mutation", "error_fragment"),
    (
        ("metadata", "dtype"),
        ("shape", "shape"),
        ("malformed_base64", "data_base64"),
        ("truncated_zlib", "EOF"),
        ("trailing_stream", "trailing data"),
        ("overlong_output", "byte length"),
    ),
)
def test_float32_xor_decoder_rejects_hostile_streams(
    mutation: str,
    error_fragment: str,
) -> None:
    states = np.zeros((3, 2), dtype=np.float32)
    payload = _float32_xor_state_payload(states)
    if mutation == "metadata":
        payload["dtype"] = "float64"
    elif mutation == "shape":
        payload["shape"] = [True, 2]
    elif mutation == "malformed_base64":
        payload["data_base64"] = "!!!!"
    else:
        compressed = base64.b64decode(cast(str, payload["data_base64"]))
        if mutation == "truncated_zlib":
            compressed = compressed[:-1]
        elif mutation == "trailing_stream":
            compressed += zlib.compress(b"trailing", level=9)
        else:
            compressed = zlib.compress(b"\x00" * 100, level=9)
        payload["data_base64"] = base64.b64encode(compressed).decode("ascii")
    errors: list[str] = []

    decoded = _decode_float32_xor_state_payload(
        payload,
        states.shape,
        "payload",
        errors,
    )

    if mutation == "metadata":
        assert errors
    else:
        assert decoded is None
    assert any(error_fragment in error for error in errors)


def test_float32_xor_decoder_rejects_noncanonical_base64() -> None:
    states = np.zeros((2, 1), dtype=np.float32)
    payload = _float32_xor_state_payload(states)
    encoded = cast(str, payload["data_base64"])
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    if encoded.endswith("=="):
        index = len(encoded) - 3
        value = alphabet.index(encoded[index])
        replacement = alphabet[(value & 0b110000) | ((value + 1) & 0b001111)]
    else:
        assert encoded.endswith("=")
        index = len(encoded) - 2
        value = alphabet.index(encoded[index])
        replacement = alphabet[(value & 0b111100) | ((value + 1) & 0b000011)]
    noncanonical = encoded[:index] + replacement + encoded[index + 1 :]
    assert base64.b64decode(noncanonical, validate=True) == base64.b64decode(
        encoded,
        validate=True,
    )
    payload["data_base64"] = noncanonical
    errors: list[str] = []

    decoded = _decode_float32_xor_state_payload(
        payload,
        states.shape,
        "payload",
        errors,
    )

    assert decoded is None
    assert any("not canonical" in error for error in errors)


def _segment_ends(segment_lengths: tuple[int, ...]) -> np.ndarray:
    return np.cumsum((0, *segment_lengths), dtype=np.int64)


def _binary_rewards(
    segment_lengths: tuple[int, ...],
    *,
    reward_permille: int,
    phase: int,
) -> np.ndarray:
    """Evenly distribute exact binary rewards without weak critical windows."""
    cycle_steps = sum(segment_lengths)
    indices = np.arange(cycle_steps, dtype=np.int64) + phase
    return (
        ((indices + 1) * reward_permille) // 1_000 > (indices * reward_permille) // 1_000
    ).astype(np.float64)


def _summary(
    seed_pair,
    segment_lengths: tuple[int, ...],
    rewards: np.ndarray,
) -> HiddenPartnerRunSummary:
    prior_by_regime: dict[int, int] = {}
    segments: list[HiddenPartnerSegmentSummary] = []
    segment_start = 0
    for index, (regime, length) in enumerate(
        zip(
            DEFAULT_REGIME_SCHEDULE,
            segment_lengths,
            strict=True,
        )
    ):
        prior = prior_by_regime.get(regime)
        segment_rewards = rewards[segment_start : segment_start + length]
        window = min(256, length)
        segments.append(
            HiddenPartnerSegmentSummary(
                segment_index=index,
                regime_id=regime,
                regime_name=REGIME_NAMES[regime],
                length=length,
                mean_reward=float(np.mean(segment_rewards)),
                early_reward=float(np.mean(segment_rewards[:window])),
                late_reward=float(np.mean(segment_rewards[-window:])),
                mean_behavior_nll=0.1,
                late_behavior_nll=0.1,
                mean_behavior_brier=0.1,
                intended_prediction_accuracy=0.9,
                late_intended_prediction_accuracy=0.9,
                mean_realized_regret=0.1,
                mean_expected_greedy_regret=0.1,
                recovery_steps=128,
                prior_same_regime_segment=prior,
                recurrent_early_to_prior_late_ratio=(None if prior is None else 1.0),
                recurrence_retained=None if prior is None else True,
            )
        )
        prior_by_regime[regime] = index
        segment_start += length

    cycle_steps = sum(segment_lengths)
    reward = float(np.mean(rewards))
    state_nbytes = _expected_integrated_state_nbytes()
    epsilon = IntegratedHiddenPartnerConfig().epsilon
    perfect_policy_reward = (1.0 - epsilon) * 0.95 + epsilon * 0.5
    ends = _segment_ends(segment_lengths)
    return HiddenPartnerRunSummary(
        condition="full",
        seed_pair=seed_pair,
        cycle_steps=cycle_steps,
        segment_lengths=segment_lengths,
        mean_reward=reward,
        normalized_control_score=(reward - 0.5) / (perfect_policy_reward - 0.5),
        mean_behavior_nll=0.1,
        mean_behavior_brier=0.1,
        behavior_actual_accuracy=0.85,
        behavior_intended_accuracy=0.9,
        planner_intended_accuracy=0.9,
        executed_intended_accuracy=0.9,
        mean_realized_counterfactual_regret=0.1,
        mean_expected_greedy_regret=0.1,
        model_intervention_rate=0.1,
        helpful_model_intervention_rate=0.05,
        harmful_model_intervention_rate=0.02,
        mean_world_reward_absolute_error=0.1,
        mean_world_outcome_squared_error=0.1,
        descriptor_transaction_count=20,
        counter_contract_valid=True,
        causal_contract_valid=True,
        all_finite=True,
        initial_state_nbytes=state_nbytes,
        final_state_nbytes=state_nbytes,
        resource_shape_matched=True,
        compilation_wall_seconds=0.0,
        execution_wall_seconds=1.0,
        mean_execution_microseconds_per_step=1e6 / cycle_steps,
        segments=tuple(segments),
        features=HiddenPartnerFeatureSummary(
            c_first_active_step=int(ends[5]),
            d_first_active_step=int(ends[3]),
            c_active_evictions=0,
            d_active_evictions=1,
            c_active_late_first_c=True,
            c_active_at_recurrent_c_entry=True,
            c_survived_first_to_recurrent_c=True,
            d_active_at_end_of_d=True,
            d_active_at_life_end=False,
            c_candidate_fraction=1.0,
            d_candidate_fraction=1.0,
        ),
    )


def _rle(
    state_count: int,
    first_live: int,
    end_live: int,
    *,
    slot: int,
    candidate: int,
) -> tuple[CriticalPairLifecycleInterval, ...]:
    intervals: list[CriticalPairLifecycleInterval] = []
    if first_live > 0:
        intervals.append(
            CriticalPairLifecycleInterval(
                start=0,
                end_exclusive=first_live,
                deployed_slot=-1,
                shadow_slot=-1,
                candidate_slot=candidate,
            )
        )
    intervals.append(
        CriticalPairLifecycleInterval(
            start=first_live,
            end_exclusive=end_live,
            deployed_slot=slot,
            shadow_slot=slot,
            candidate_slot=candidate,
        )
    )
    if end_live < state_count:
        intervals.append(
            CriticalPairLifecycleInterval(
                start=end_live,
                end_exclusive=state_count,
                deployed_slot=-1,
                shadow_slot=-1,
                candidate_slot=candidate,
            )
        )
    return tuple(intervals)


def _packed_bits(values: np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=np.bool_)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return {
        "shape": list(array.shape),
        "bitorder": "little",
        "data_base64": base64.b64encode(packed.tobytes()).decode("ascii"),
    }


def _float32_xor_state_payload(values: np.ndarray) -> dict[str, object]:
    states = np.ascontiguousarray(values, dtype="<f4")
    state_bits = states.view("<u4")
    deltas = np.empty_like(state_bits)
    deltas[0] = state_bits[0]
    deltas[1:] = state_bits[1:] ^ state_bits[:-1]
    compressed = zlib.compress(deltas.tobytes(order="C"), level=9)
    return {
        "shape": list(states.shape),
        "dtype": "float32",
        "byteorder": "little",
        "delta": "uint32-xor",
        "codec": "zlib",
        "data_base64": base64.b64encode(compressed).decode("ascii"),
    }


def _descriptor_bank(
    *,
    c_live: bool,
    d_live: bool,
) -> list[list[int]]:
    bank = [[-1, -1] for _ in range(12)]
    if c_live:
        bank[0] = [0, 2]
    if d_live:
        bank[1] = [4, 5]
    return bank


def _bank_state_rle(
    state_count: int,
    *,
    d_start: int,
    d_retirement_effective: int,
    c_start: int,
) -> list[dict[str, object]]:
    states = (
        (0, d_start, False, False),
        (d_start, d_retirement_effective, False, True),
        (d_retirement_effective, c_start, False, False),
        (c_start, state_count, True, False),
    )
    return [
        {
            "start": start,
            "end_exclusive": end,
            "deployed_descriptors": _descriptor_bank(
                c_live=c_live,
                d_live=d_live,
            ),
            "shadow_descriptors": _descriptor_bank(
                c_live=c_live,
                d_live=d_live,
            ),
        }
        for start, end, c_live, d_live in states
    ]


def _candidate_bank_state_rle(state_count: int) -> list[dict[str, object]]:
    return [
        {
            "start": 0,
            "end_exclusive": state_count,
            "candidate_descriptors": [
                [left, right]
                for left in range(12)
                for right in range(left + 1, 12)
            ],
        }
    ]


def _gate_events(window_start: int, window_end: int) -> np.ndarray:
    """Events whose routed mask opens every state in one decision window."""
    return np.arange(window_start - 1, window_end - 1, dtype=np.int64)


def _bank_state_arrays(
    state_count: int,
    *,
    d_start: int,
    d_retirement_effective: int,
    c_start: int,
) -> np.ndarray:
    states = np.full((state_count, 12, 2), -1, dtype=np.int32)
    states[d_start:d_retirement_effective, 1] = (4, 5)
    states[c_start:, 0] = (0, 2)
    return states


def _consumer_gate_traces(
    deployed_states: np.ndarray,
    evidence: np.ndarray,
    *,
    confirmation_steps: int,
    read_confirmation_steps: int,
    read_lease_steps: int,
) -> dict[str, np.ndarray]:
    """Reconstruct descriptor-routed streak, confirmed-write, and read leases."""
    cycle_steps = evidence.shape[0]
    shape = (cycle_steps, 12)
    streak_pre = np.zeros(shape, dtype=np.int32)
    streak_updated_pre = np.zeros(shape, dtype=np.int32)
    streak_post = np.zeros(shape, dtype=np.int32)
    confirmed_pre = np.zeros(shape, dtype=np.bool_)
    confirmed_post = np.zeros(shape, dtype=np.bool_)
    read_acquire_pre = np.zeros(shape, dtype=np.bool_)
    read_acquire_post = np.zeros(shape, dtype=np.bool_)
    mask_pre = np.zeros(shape, dtype=np.bool_)
    mask_post = np.zeros(shape, dtype=np.bool_)
    idle_post = np.zeros(shape, dtype=np.int32)
    current_streak = np.zeros((12,), dtype=np.int32)
    current_mask = np.zeros((12,), dtype=np.bool_)
    current_idle = np.zeros((12,), dtype=np.int32)

    for step in range(cycle_steps):
        deployed_pre = deployed_states[step]
        deployed_post = deployed_states[step + 1]
        live_pre = np.all(deployed_pre >= 0, axis=1)
        streak_pre[step] = current_streak
        mask_pre[step] = current_mask
        updated = np.where(
            live_pre & evidence[step],
            np.minimum(current_streak, np.iinfo(np.int32).max - 1) + 1,
            0,
        ).astype(np.int32)
        confirmed = live_pre & evidence[step] & (updated >= confirmation_steps)
        read_acquire = live_pre & evidence[step] & (updated >= read_confirmation_steps)
        updated_idle = np.where(
            live_pre,
            np.where(evidence[step], 0, current_idle + 1),
            0,
        ).astype(np.int32)
        streak_updated_pre[step] = updated
        confirmed_pre[step] = confirmed
        read_acquire_pre[step] = read_acquire

        next_streak = np.zeros((12,), dtype=np.int32)
        next_confirmed = np.zeros((12,), dtype=np.bool_)
        next_read_acquire = np.zeros((12,), dtype=np.bool_)
        next_mask = np.zeros((12,), dtype=np.bool_)
        next_idle = np.zeros((12,), dtype=np.int32)
        for post_slot, descriptor in enumerate(deployed_post):
            if descriptor[0] < 0:
                continue
            matches = np.all(deployed_pre == descriptor, axis=1)
            if np.sum(matches) != 1:
                continue
            source = int(np.argmax(matches))
            next_streak[post_slot] = updated[source]
            next_confirmed[post_slot] = confirmed[source]
            next_read_acquire[post_slot] = read_acquire[source]
            next_idle[post_slot] = updated_idle[source]
            next_mask[post_slot] = bool(
                (current_mask[source] or read_acquire[source])
                and next_idle[post_slot] <= read_lease_steps
            )
        streak_post[step] = next_streak
        confirmed_post[step] = next_confirmed
        read_acquire_post[step] = next_read_acquire
        mask_post[step] = next_mask
        idle_post[step] = next_idle
        current_streak = next_streak
        current_mask = next_mask
        current_idle = next_idle

    return {
        "streak_pre": streak_pre,
        "streak_updated_pre": streak_updated_pre,
        "streak_post": streak_post,
        "confirmed_pre": confirmed_pre,
        "confirmed_post": confirmed_post,
        "read_acquire_pre": read_acquire_pre,
        "read_acquire_post": read_acquire_post,
        "mask_pre": mask_pre,
        "mask_post": mask_post,
        "idle_post": idle_post,
    }


def _feature_memory_traces(
    shadow_states: np.ndarray,
    evidence: np.ndarray,
    *,
    confirmation_steps: int,
) -> dict[str, np.ndarray]:
    """Build a valid descriptor-routed confirmed-head trace for fake lives."""
    cycle_steps = evidence.shape[0]
    shape = (cycle_steps, 12)
    confirmed = np.zeros(shape, dtype=np.bool_)
    committed_pre = np.zeros(shape, dtype=np.bool_)
    committed_post = np.zeros(shape, dtype=np.bool_)
    head_changed = np.zeros(shape, dtype=np.bool_)
    head_states = np.zeros((cycle_steps + 1, 1, 12), dtype=np.float32)
    current_streak = np.zeros((12,), dtype=np.int32)
    current_committed = np.zeros((12,), dtype=np.bool_)

    for step in range(cycle_steps):
        shadow_pre = shadow_states[step]
        shadow_post = shadow_states[step + 1]
        live_pre = np.all(shadow_pre >= 0, axis=1)
        committed_pre[step] = current_committed
        updated_streak = np.where(
            live_pre & evidence[step],
            np.minimum(current_streak, np.iinfo(np.int32).max - 1) + 1,
            0,
        ).astype(np.int32)
        confirmed_step = (
            live_pre
            & evidence[step]
            & (updated_streak >= confirmation_steps)
        )
        confirmed[step] = confirmed_step
        updated_committed = live_pre & (
            current_committed | confirmed_step
        )

        next_streak = np.zeros((12,), dtype=np.int32)
        next_committed = np.zeros((12,), dtype=np.bool_)
        next_head = np.zeros((1, 12), dtype=np.float32)
        for post_slot, descriptor in enumerate(shadow_post):
            if descriptor[0] < 0:
                continue
            matches = np.all(shadow_pre == descriptor, axis=1)
            if np.sum(matches) != 1:
                continue
            source = int(np.argmax(matches))
            next_streak[post_slot] = updated_streak[source]
            next_committed[post_slot] = updated_committed[source]
            change_allowed = bool(
                evidence[step, source]
                and (
                    not current_committed[source]
                    or confirmed_step[source]
                )
            )
            head_changed[step, source] = change_allowed
            next_head[0, post_slot] = (
                head_states[step, 0, source]
                + np.float32(1.0 if change_allowed else 0.0)
            )
        committed_post[step] = next_committed
        head_states[step + 1] = next_head
        current_streak = next_streak
        current_committed = next_committed

    return {
        "confirmed": confirmed,
        "committed_pre": committed_pre,
        "committed_post": committed_post,
        "head_changed": head_changed,
        "head_states": head_states,
        "violations": np.zeros(shape, dtype=np.bool_),
    }


def _consumer_state_traces(
    deployed_states: np.ndarray,
    write_gate: np.ndarray,
    *,
    c_success: bool,
) -> dict[str, np.ndarray]:
    """Build exact routed downstream states satisfying the write contract."""
    cycle_steps = write_gate.shape[0]
    state_shape = (cycle_steps + 1, 2, 12)
    behavior = np.zeros(state_shape, dtype=np.float32)
    control_q = np.zeros(state_shape, dtype=np.float32)
    control_q_trace = np.zeros(state_shape, dtype=np.float32)

    behavior_increment = np.asarray((-1.0, 1.0), dtype=np.float32)
    q_increment = np.asarray((0.25, -0.25), dtype=np.float32)
    trace_value = np.asarray((0.5, -0.5), dtype=np.float32)
    for step in range(cycle_steps):
        deployed_pre = deployed_states[step]
        deployed_post = deployed_states[step + 1]
        for post_slot, descriptor in enumerate(deployed_post):
            if descriptor[0] < 0:
                continue
            matches = np.all(deployed_pre == descriptor, axis=1)
            if np.sum(matches) != 1:
                continue
            source = int(np.argmax(matches))
            gate_open = bool(write_gate[step, source])
            behavior_write = gate_open and (
                tuple(int(value) for value in descriptor) != (0, 2)
                or c_success
            )
            behavior[step + 1, :, post_slot] = behavior[
                step,
                :,
                source,
            ] + (behavior_increment if behavior_write else 0.0)
            control_q[step + 1, :, post_slot] = control_q[
                step,
                :,
                source,
            ] + (q_increment if gate_open else 0.0)
            if gate_open:
                control_q_trace[step + 1, :, post_slot] = trace_value

    return {
        "behavior": behavior,
        "control_q": control_q,
        "control_q_trace": control_q_trace,
        "violations": np.zeros((cycle_steps, 12), dtype=np.bool_),
    }


def _critical_window(
    *,
    pair: tuple[int, int],
    entry_step: int,
    window_start: int,
    deployed_states: np.ndarray,
    behavior_states: np.ndarray,
) -> dict[str, object]:
    target = np.asarray(pair, dtype=np.int32)
    entry_matches = np.all(deployed_states[entry_step] == target, axis=1)
    entry_margin = 0.0
    if np.any(entry_matches):
        entry_slot = int(np.argmax(entry_matches))
        entry_margin = float(
            behavior_states[entry_step, 1, entry_slot]
            - behavior_states[entry_step, 0, entry_slot]
        )
    rows: list[dict[str, object]] = []
    for step in range(
        window_start,
        window_start + FEATURE_LEARNING_WINDOW,
    ):
        matches = np.all(deployed_states[step] == target, axis=1)
        current_margin = 0.0
        activation = 0.0
        if np.any(matches):
            slot = int(np.argmax(matches))
            current_margin = float(
                behavior_states[step, 1, slot]
                - behavior_states[step, 0, slot]
            )
            activation = 1.0
        rows.append(
            {
                "step": step,
                "intended_action": 1,
                "online_logit_margin": current_margin,
                "critical_activation": activation,
                "current_critical_weight_margin": current_margin,
            }
        )
    return {
        "pair": list(pair),
        "entry_step": entry_step,
        "window_start": window_start,
        "window_end_exclusive": window_start + FEATURE_LEARNING_WINDOW,
        "entry_critical_weight_margin": entry_margin,
        "rows": rows,
    }


def _critical_primitives(
    segment_lengths: tuple[int, ...],
    rewards: np.ndarray,
    *,
    agent_config: IntegratedHiddenPartnerConfig,
    c_success: bool,
    retirement_event_latency: int,
) -> dict[str, object]:
    cycle_steps = sum(segment_lengths)
    state_count = cycle_steps + 1
    ends = _segment_ends(segment_lengths)
    d_start, d_end = int(ends[3]), int(ends[4])
    c_start, c_end = int(ends[5]), int(ends[6])
    recurrent_c_start = int(ends[8])
    retirement_effective = d_end + retirement_event_latency + 1

    evidence = np.zeros((cycle_steps, 12), dtype=np.bool_)
    d_events = np.unique(
        np.concatenate(
            (
                np.asarray([d_start], dtype=np.int64),
                _gate_events(d_end - FEATURE_LEARNING_WINDOW, d_end),
            )
        )
    )
    c_events = np.unique(
        np.concatenate(
            (
                np.asarray([c_start], dtype=np.int64),
                _gate_events(c_end - FEATURE_LEARNING_WINDOW, c_end),
                _gate_events(
                    recurrent_c_start,
                    recurrent_c_start + RECURRENT_ENTRY_WINDOW,
                ),
            )
        )
    )
    evidence[d_events, 1] = True
    evidence[c_events, 0] = True
    deployed_states = _bank_state_arrays(
        state_count,
        d_start=d_start,
        d_retirement_effective=retirement_effective,
        c_start=c_start,
    )
    consumer_gate = _consumer_gate_traces(
        deployed_states,
        evidence,
        confirmation_steps=agent_config.consumer_evidence_confirmation_steps,
        read_confirmation_steps=agent_config.consumer_read_confirmation_steps,
        read_lease_steps=agent_config.consumer_read_lease_steps,
    )
    feature_memory = _feature_memory_traces(
        deployed_states,
        evidence,
        confirmation_steps=agent_config.feature_evidence_confirmation_steps,
    )
    consumer_states = _consumer_state_traces(
        deployed_states,
        consumer_gate["confirmed_pre"],
        c_success=c_success,
    )
    no_step_violations = np.zeros((cycle_steps,), dtype=np.bool_)
    no_slot_violations = np.zeros((cycle_steps, 12), dtype=np.bool_)
    return {
        "schema_version": CRITICAL_RUN_PRIMITIVES_SCHEMA,
        "cycle_steps": cycle_steps,
        "reward_one_bits": _packed_bits(rewards == 1.0),
        "evidence_refresh_bits": _packed_bits(evidence),
        "retention_evidence_refresh_bits": _packed_bits(
            feature_memory["confirmed"]
        ),
        "feature_memory_committed_pre_bits": _packed_bits(
            feature_memory["committed_pre"]
        ),
        "feature_memory_committed_post_bits": _packed_bits(
            feature_memory["committed_post"]
        ),
        "identity_routed_head_changed_bits": _packed_bits(
            feature_memory["head_changed"]
        ),
        "feature_memory_contract_violation_bits": _packed_bits(
            feature_memory["violations"]
        ),
        "feature_memory_enabled": (
            agent_config.evidence_gated_feature_memory
        ),
        "feature_head_state_xor": _float32_xor_state_payload(
            feature_memory["head_states"]
        ),
        "consumer_write_gate_bits": _packed_bits(consumer_gate["confirmed_pre"]),
        "behavior_pair_weight_state_xor": _float32_xor_state_payload(
            consumer_states["behavior"]
        ),
        "control_q_pair_weight_state_xor": _float32_xor_state_payload(
            consumer_states["control_q"]
        ),
        "control_q_trace_state_xor": _float32_xor_state_payload(
            consumer_states["control_q_trace"]
        ),
        "consumer_write_contract_violation_bits": _packed_bits(
            consumer_states["violations"]
        ),
        "consumer_active_mask_pre_bits": _packed_bits(consumer_gate["mask_pre"]),
        "consumer_active_mask_post_bits": _packed_bits(consumer_gate["mask_post"]),
        "closed_consumer_read_violation_bits": _packed_bits(no_slot_violations),
        "representation_link_violation_bits": _packed_bits(no_step_violations),
        "counter_contract_violation_bits": _packed_bits(no_step_violations),
        "causal_contract_violation_bits": _packed_bits(no_step_violations),
        "finite_violation_bits": _packed_bits(no_step_violations),
        "bank_state_rle": _bank_state_rle(
            state_count,
            d_start=d_start,
            d_retirement_effective=retirement_effective,
            c_start=c_start,
        ),
        "candidate_bank_state_rle": _candidate_bank_state_rle(
            state_count
        ),
        "critical_windows": {
            "c_first_late": _critical_window(
                pair=(0, 2),
                entry_step=c_start,
                window_start=c_end - FEATURE_LEARNING_WINDOW,
                deployed_states=deployed_states,
                behavior_states=consumer_states["behavior"],
            ),
            "d_late": _critical_window(
                pair=(4, 5),
                entry_step=d_start,
                window_start=d_end - FEATURE_LEARNING_WINDOW,
                deployed_states=deployed_states,
                behavior_states=consumer_states["behavior"],
            ),
            "c_recurrent_early": _critical_window(
                pair=(0, 2),
                entry_step=c_start,
                window_start=recurrent_c_start,
                deployed_states=deployed_states,
                behavior_states=consumer_states["behavior"],
            ),
        },
    }


def _window_metrics(window: dict[str, object]) -> dict[str, float]:
    entry_margin = float(window["entry_critical_weight_margin"])
    rows = cast(list[dict[str, object]], window["rows"])
    online_losses: list[float] = []
    entry_losses: list[float] = []
    zero_losses: list[float] = []
    online_correct: list[float] = []
    entry_correct: list[float] = []
    for row in rows:
        action = int(row["intended_action"])
        sign = 2.0 * action - 1.0
        margin = float(row["online_logit_margin"])
        activation = float(row["critical_activation"])
        current_margin = float(row["current_critical_weight_margin"])
        zero_margin = margin - current_margin * activation
        frozen_margin = zero_margin + entry_margin * activation
        online_losses.append(float(np.logaddexp(0.0, -sign * margin)))
        zero_losses.append(float(np.logaddexp(0.0, -sign * zero_margin)))
        entry_losses.append(float(np.logaddexp(0.0, -sign * frozen_margin)))
        online_correct.append(float((1 if margin > 0.0 else 0) == action))
        entry_correct.append(float((1 if frozen_margin > 0.0 else 0) == action))
    online_array = np.asarray(online_losses, dtype=np.float64)
    entry_array = np.asarray(entry_losses, dtype=np.float64)
    zero_array = np.asarray(zero_losses, dtype=np.float64)
    learning_gain = entry_array - online_array
    mask_gain = zero_array - online_array
    online_accuracy = float(np.mean(online_correct))
    entry_accuracy = float(np.mean(entry_correct))
    mean_learning_gain = float(np.mean(learning_gain))
    mean_mask_gain = float(np.mean(mask_gain))
    return {
        "online_nll": float(np.mean(online_array)),
        "entry_nll": float(np.mean(entry_array)),
        "gain": mean_learning_gain,
        "positive_fraction": float(np.mean(learning_gain > 0.0)),
        "online_accuracy": online_accuracy,
        "entry_accuracy": entry_accuracy,
        "accuracy_gain": online_accuracy - entry_accuracy,
        "mask_gain": mean_mask_gain,
        "mask_positive_fraction": float(np.mean(mask_gain > 0.0)),
        "target_created_share": mean_learning_gain / max(mean_mask_gain, 1e-12),
    }


def _lifecycle(
    segment_lengths: tuple[int, ...],
    rewards: np.ndarray,
    primitives: dict[str, object],
    *,
    c_success: bool,
    retirement_event_latency: int,
) -> CriticalLifecycleV2Summary:
    cycle_steps = sum(segment_lengths)
    state_count = cycle_steps + 1
    ends = _segment_ends(segment_lengths)
    c_start, c_end = int(ends[5]), int(ends[6])
    d_start, d_exit = int(ends[3]), int(ends[4])
    recurrent_c_start = int(ends[8])
    c_promotion = c_start - 1
    c_acquisition = c_start + 1
    d_promotion = d_start - 1
    d_acquisition = d_start + 1
    retirement_event = d_exit + retirement_event_latency
    retirement_effective = retirement_event + 1
    post_live_steps = retirement_event_latency + 1
    windows = cast(dict[str, dict[str, object]], primitives["critical_windows"])
    c_metrics = _window_metrics(windows["c_first_late"])
    d_metrics = _window_metrics(windows["d_late"])
    c_recurrent_metrics = _window_metrics(windows["c_recurrent_early"])
    evidence_bits = primitives["evidence_refresh_bits"]
    evidence_bytes = base64.b64decode(evidence_bits["data_base64"])
    evidence = np.unpackbits(
        np.frombuffer(evidence_bytes, dtype=np.uint8),
        bitorder="little",
    )[: cycle_steps * 12].reshape((cycle_steps, 12))
    c_refreshes = tuple(int(step) for step in range(c_start, c_end) if evidence[step, 0])
    d_refreshes = tuple(int(step) for step in range(d_start, d_exit) if evidence[step, 1])
    c_first_reward = float(np.mean(rewards[c_end - FEATURE_LEARNING_WINDOW : c_end]))
    c_recurrent_reward = float(
        np.mean(rewards[recurrent_c_start : recurrent_c_start + RECURRENT_ENTRY_WINDOW])
    )
    d_late_reward = float(np.mean(rewards[d_exit - FEATURE_LEARNING_WINDOW : d_exit]))
    retention_ratio = (c_recurrent_reward - 0.5) / max(
        c_first_reward - 0.5,
        1e-7,
    )
    return CriticalLifecycleV2Summary(
        cycle_steps=cycle_steps,
        decision_state_count=state_count,
        representation_link_contract_valid=True,
        consumer_gate_contract_valid=True,
        feature_memory_enabled=True,
        feature_memory_contract_valid=True,
        c_shadow_deployed_mismatch_steps=0,
        d_shadow_deployed_mismatch_steps=0,
        c_promotion_event_steps=(c_promotion,),
        c_target_evidence_refresh_steps=c_refreshes,
        c_acquisition_step=c_acquisition,
        c_first_late_reward=c_first_reward,
        c_first_late_intended_accuracy=c_metrics["online_accuracy"],
        c_first_late_online_nll=c_metrics["online_nll"],
        c_first_late_entry_frozen_critical_nll=c_metrics["entry_nll"],
        c_critical_column_learning_nll_gain=c_metrics["gain"],
        c_critical_column_learning_positive_fraction=c_metrics["positive_fraction"],
        c_critical_column_target_created_share=c_metrics["target_created_share"],
        c_first_late_entry_frozen_critical_accuracy=c_metrics["entry_accuracy"],
        c_critical_column_learning_accuracy_gain=c_metrics["accuracy_gain"],
        c_first_late_masked_nll_increase=c_metrics["mask_gain"],
        c_first_late_masked_nll_positive_fraction=c_metrics["mask_positive_fraction"],
        c_task_learned=c_success,
        c_survival_end_exclusive=min(
            cycle_steps,
            recurrent_c_start + RECURRENT_ENTRY_WINDOW,
        ),
        c_survival_gap_steps=0,
        c_first_survival_gap_step=None,
        c_evictions_after_acquisition=0,
        c_repromotions_after_acquisition=0,
        c_continuously_survived=True,
        c_recurrent_early_reward=c_recurrent_reward,
        c_recurrent_early_excess_reward_retention=retention_ratio,
        c_recurrent_early_intended_accuracy=c_recurrent_metrics["online_accuracy"],
        c_recurrent_early_masked_nll_increase=c_recurrent_metrics["mask_gain"],
        c_recurrent_early_masked_nll_positive_fraction=(
            c_recurrent_metrics["mask_positive_fraction"]
        ),
        c_retained_and_used=c_success,
        d_promotion_event_steps=(d_promotion,),
        d_target_evidence_refresh_steps=d_refreshes,
        d_acquisition_step=d_acquisition,
        d_deployed_through_exit=True,
        d_late_reward=d_late_reward,
        d_late_intended_accuracy=d_metrics["online_accuracy"],
        d_late_online_nll=d_metrics["online_nll"],
        d_late_entry_frozen_critical_nll=d_metrics["entry_nll"],
        d_critical_column_learning_nll_gain=d_metrics["gain"],
        d_critical_column_learning_positive_fraction=d_metrics["positive_fraction"],
        d_critical_column_target_created_share=d_metrics["target_created_share"],
        d_late_entry_frozen_critical_accuracy=d_metrics["entry_accuracy"],
        d_critical_column_learning_accuracy_gain=d_metrics["accuracy_gain"],
        d_late_masked_nll_increase=d_metrics["mask_gain"],
        d_late_masked_nll_positive_fraction=d_metrics["mask_positive_fraction"],
        d_task_learned=True,
        d_retirement_event_step=retirement_event,
        d_retirement_step=retirement_effective,
        d_retirement_event_latency_steps=retirement_event_latency,
        d_retirement_latency_steps=retirement_event_latency + 1,
        d_post_exit_live_slot_steps=post_live_steps,
        d_post_exit_live_fraction=(post_live_steps / (cycle_steps - d_exit)),
        d_post_exit_promotion_count=0,
        d_repromotions_after_retirement=0,
        d_absent_entire_final_window=True,
        d_retirement_event_steps=(retirement_event,),
        d_retirement_event_reset_counts=(1,),
        d_retirement_event_candidate_utility_post=(0.0,),
        d_retirement_event_candidate_head_linf_post=(0.0,),
        d_retirement_event_candidate_age_post=(0,),
        d_retirement_event_count=1,
        d_matching_candidate_reset_count=1,
        d_linked_matching_candidate_reset_count=1,
        d_linked_candidate_utility_post=0.0,
        d_linked_candidate_head_linf_post=0.0,
        d_linked_candidate_age_post=0,
        d_retirement_event_aligned=True,
        d_learned_then_stably_retired=True,
        joint_memory_management_success=c_success,
        candidate_archive_contract_valid=True,
        c_candidate_utility_at_life_end=0.1,
        d_candidate_utility_at_life_end=0.0,
        c_lifecycle_rle=_rle(
            state_count,
            c_start,
            state_count,
            slot=0,
            candidate=1,
        ),
        d_lifecycle_rle=_rle(
            state_count,
            d_start,
            retirement_effective,
            slot=1,
            candidate=38,
        ),
    )


def _record(
    monkeypatch: pytest.MonkeyPatch,
    *,
    successes_per_cell: int = 6,
) -> dict[str, object]:
    cached = _RECORD_CACHE.get(successes_per_cell)
    if cached is not None:
        return copy.deepcopy(cached)
    reward_permille = (860, 865, 870, 875, 880, 900, 890, 860)

    class FakeRunner:
        def __init__(self, condition, protocol) -> None:
            self.protocol = protocol
            self.cell = next(
                cell for cell in LEASE_TUNING_GRID if cell.agent_config() == condition.config
            )

        def run(self, seed_pair):
            success_count = 5 if self.cell.index in {0, 7} else successes_per_cell
            environment = HiddenPartnerMappingWorld(self.protocol.environment).init(
                jr.key(seed_pair.stream_seed)
            )
            segment_lengths = tuple(
                int(value)
                for value in np.asarray(
                    environment.segment_lengths,
                    dtype=np.int64,
                )
            )
            rewards = _binary_rewards(
                segment_lengths,
                reward_permille=reward_permille[self.cell.index],
                phase=137 * seed_pair.index + 29 * self.cell.index,
            )
            c_success = seed_pair.index < success_count
            retirement_event_latency = 10 + self.cell.index
            primitives = _critical_primitives(
                segment_lengths,
                rewards,
                agent_config=self.cell.agent_config(),
                c_success=c_success,
                retirement_event_latency=retirement_event_latency,
            )
            return SimpleNamespace(
                summary=_summary(
                    seed_pair,
                    segment_lengths,
                    rewards,
                ),
                lifecycle=_lifecycle(
                    segment_lengths,
                    rewards,
                    primitives,
                    c_success=c_success,
                    retirement_event_latency=retirement_event_latency,
                ),
                primitives=primitives,
            )

    monkeypatch.setattr(
        lifecycle_module,
        "HiddenPartnerDevelopmentRunner",
        FakeRunner,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "summarize_critical_lifecycle_v2",
        lambda result: result.lifecycle,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "critical_run_primitives",
        lambda result: result.primitives,
    )
    monkeypatch.setattr(
        lifecycle_module,
        "LEASE_TUNING_NAMESPACE_STATUS",
        "EXECUTABLE",
    )
    try:
        record = run_evidence_lease_tuning_grid()
    finally:
        monkeypatch.setattr(
            lifecycle_module,
            "LEASE_TUNING_NAMESPACE_STATUS",
            "FORBIDDEN/UNEXECUTED",
        )
    _RECORD_CACHE[successes_per_cell] = copy.deepcopy(record)
    return record


def _operational() -> dict[str, object]:
    return {
        "argv": ["--run"],
        "generated_at_utc": "2026-07-30T12:00:00+00:00",
        "jax_backend": "cpu",
        "jax_device_count": 1,
        "jax_devices": ["TFRT_CPU_0"],
        "jax_version": "test",
        "jaxlib_version": "test",
        "numpy_version": "test",
        "platform": "test",
        "python_version": "3.12",
        "wall_seconds": 1.0,
    }


def _artifact(
    monkeypatch: pytest.MonkeyPatch,
    *,
    successes_per_cell: int = 6,
) -> dict[str, object]:
    cached = _ARTIFACT_CACHE.get(successes_per_cell)
    if cached is not None:
        return copy.deepcopy(cached)
    artifact = build_lease_tuning_artifact(
        _record(
            monkeypatch,
            successes_per_cell=successes_per_cell,
        ),
        operational_metadata=_operational(),
    )
    _ARTIFACT_CACHE[successes_per_cell] = copy.deepcopy(artifact)
    return artifact


def _resign(artifact: dict[str, object]) -> None:
    artifact["scientific_digest"]["sha256"] = _canonical_sha256(artifact["scientific_payload"])


def _flip_packed_bit(payload: dict[str, object], flat_index: int = 0) -> None:
    decoded = bytearray(base64.b64decode(payload["data_base64"]))
    decoded[flat_index // 8] ^= 1 << (flat_index % 8)
    payload["data_base64"] = base64.b64encode(decoded).decode("ascii")


def _unpack_payload(payload: dict[str, object]) -> np.ndarray:
    shape = tuple(cast(list[int], payload["shape"]))
    bit_count = math.prod(shape)
    return np.unpackbits(
        np.frombuffer(
            base64.b64decode(cast(str, payload["data_base64"])),
            dtype=np.uint8,
        ),
        bitorder="little",
    )[:bit_count].astype(np.bool_).reshape(shape)


def _unpack_float32_state_payload(payload: dict[str, object]) -> np.ndarray:
    shape = tuple(cast(list[int], payload["shape"]))
    errors: list[str] = []
    decoded = _decode_float32_xor_state_payload(
        payload,
        shape,
        "payload",
        errors,
    )
    assert decoded is not None
    assert not errors
    return decoded.copy()


_RECORD_CACHE: dict[int, dict[str, object]] = {}
_ARTIFACT_CACHE: dict[int, dict[str, object]] = {}


def test_artifact_round_trip_reconstructs_exact_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = _artifact(monkeypatch)
    validation = validate_lease_tuning_artifact(artifact)
    assert validation.valid, validation.errors
    assert validation.feasible_cell_selected
    payload = artifact["scientific_payload"]
    assert len(payload["runs"]) == (len(LEASE_TUNING_GRID) * LEASE_TUNING_SEED_COUNT)
    assert payload["selected_cell"]["cell_index"] == 5
    first_run = payload["runs"][0]
    primitives = first_run["critical_run_primitives"]
    assert set(primitives["critical_windows"]) == {
        "c_first_late",
        "d_late",
        "c_recurrent_early",
    }
    assert all(len(window["rows"]) == 128 for window in primitives["critical_windows"].values())
    assert primitives["candidate_bank_state_rle"] == _candidate_bank_state_rle(
        first_run["run_summary"]["cycle_steps"] + 1
    )
    assert primitives["feature_head_state_xor"]["shape"] == [
        first_run["run_summary"]["cycle_steps"] + 1,
        1,
        12,
    ]
    assert first_run["run_summary"]["initial_state_nbytes"] == (_expected_integrated_state_nbytes())

    path = tmp_path / "lease.json"
    path.write_text(
        lease_tuning_artifact_json(artifact),
        encoding="utf-8",
    )
    loaded = load_lease_tuning_artifact(path)
    assert validate_lease_tuning_artifact(loaded).valid


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("d_retirement_latency_steps", "bad"),
        ("d_repromotions_after_retirement", False),
        ("d_retirement_event_count", -1),
    ),
)
def test_malformed_lifecycle_scalars_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    artifact = copy.deepcopy(_artifact(monkeypatch))
    artifact["scientific_payload"]["runs"][0]["critical_lifecycle"][field] = value
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("critical_lifecycle", "c_first_late_online_nll", 0.25),
        (
            "critical_lifecycle",
            "c_critical_column_learning_positive_fraction",
            0.75,
        ),
        ("run_summary", "mean_reward", 0.99),
    ),
)
def test_resigned_derived_performance_aggregate_edits_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    field: str,
    value: float,
) -> None:
    artifact = _artifact(monkeypatch)
    artifact["scientific_payload"]["runs"][0][section][field] = value
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected


def test_resigned_reward_primitive_bit_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    packed = artifact["scientific_payload"]["runs"][0]["critical_run_primitives"]["reward_one_bits"]
    _flip_packed_bit(packed)
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected


def test_resigned_consumer_gate_mask_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    packed = artifact["scientific_payload"]["runs"][0]["critical_run_primitives"][
        "consumer_active_mask_post_bits"
    ]
    _flip_packed_bit(packed)
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("consumer_gate_contract_valid" in error for error in validation.errors)


@pytest.mark.parametrize(
    "primitive_field",
    (
        "retention_evidence_refresh_bits",
        "feature_memory_committed_pre_bits",
        "feature_memory_committed_post_bits",
        "feature_memory_contract_violation_bits",
    ),
)
def test_resigned_feature_memory_primitive_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    primitive_field: str,
) -> None:
    artifact = _artifact(monkeypatch)
    primitive = artifact["scientific_payload"]["runs"][0][
        "critical_run_primitives"
    ][primitive_field]
    _flip_packed_bit(primitive)
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any(
        "feature_memory_contract_valid" in error
        or primitive_field in error
        for error in validation.errors
    )


def test_feature_memory_allows_bootstrap_and_confirmed_head_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    head_changed = _unpack_payload(
        primitives["identity_routed_head_changed_bits"]
    )
    committed_pre = _unpack_payload(
        primitives["feature_memory_committed_pre_bits"]
    )
    confirmed = _unpack_payload(
        primitives["retention_evidence_refresh_bits"]
    )

    assert head_changed[d_start, 1]
    assert not committed_pre[d_start, 1]
    assert not confirmed[d_start, 1]
    confirmed_d_steps = np.flatnonzero(confirmed[:, 1])
    assert confirmed_d_steps.size > 0
    assert head_changed[int(confirmed_d_steps[0]), 1]
    assert validate_lease_tuning_artifact(artifact).valid


def test_resigned_true_feature_head_change_suppressed_to_zero_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    head_changed = _unpack_payload(
        primitives["identity_routed_head_changed_bits"]
    )
    assert head_changed[d_start, 1]

    _flip_packed_bit(
        primitives["identity_routed_head_changed_bits"],
        d_start * 12 + 1,
    )
    _resign(artifact)
    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any(
        "feature_memory_contract_valid" in error
        for error in validation.errors
    )


def test_resigned_hidden_unconfirmed_committed_head_write_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    target_step = int(ends[4]) - 1
    committed_pre = _unpack_payload(
        primitives["feature_memory_committed_pre_bits"]
    )
    confirmed = _unpack_payload(
        primitives["retention_evidence_refresh_bits"]
    )
    head_changed = _unpack_payload(
        primitives["identity_routed_head_changed_bits"]
    )
    assert committed_pre[target_step, 1]
    assert not confirmed[target_step, 1]
    assert not head_changed[target_step, 1]

    head_states = _unpack_float32_state_payload(
        primitives["feature_head_state_xor"]
    )
    head_states[target_step + 1, 0, 1] += np.float32(0.5)
    primitives["feature_head_state_xor"] = _float32_xor_state_payload(
        head_states
    )
    _resign(artifact)
    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any(
        "feature_memory_contract_valid" in error
        for error in validation.errors
    )


@pytest.mark.parametrize(
    "state_field",
    (
        "behavior_pair_weight_state_xor",
        "control_q_pair_weight_state_xor",
        "control_q_trace_state_xor",
    ),
)
def test_resigned_hidden_closed_consumer_write_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    state_field: str,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    write_gate = _unpack_payload(primitives["consumer_write_gate_bits"])
    assert not write_gate[d_start, 1]
    states = _unpack_float32_state_payload(primitives[state_field])
    states[d_start + 1, 0, 1] = np.float32(1.0)
    primitives[state_field] = _float32_xor_state_payload(states)
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("consumer_gate_contract_valid" in error for error in validation.errors)


def test_resigned_suppressed_consumer_violation_bit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    states = _unpack_float32_state_payload(
        primitives["behavior_pair_weight_state_xor"]
    )
    states[d_start + 1, 0, 1] = np.float32(1.0)
    primitives["behavior_pair_weight_state_xor"] = (
        _float32_xor_state_payload(states)
    )
    # Model a producer-detected violation followed by a resigned 1 -> 0
    # suppression: the exact numeric state must still expose the hidden write.
    violation_payload = primitives["consumer_write_contract_violation_bits"]
    _flip_packed_bit(violation_payload, d_start * 12 + 1)
    assert _unpack_payload(violation_payload)[d_start, 1]
    _flip_packed_bit(violation_payload, d_start * 12 + 1)
    assert not _unpack_payload(violation_payload)[d_start, 1]
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("consumer_gate_contract_valid" in error for error in validation.errors)


def test_resigned_negative_zero_closed_trace_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    states = _unpack_float32_state_payload(
        primitives["control_q_trace_state_xor"]
    )
    states[d_start + 1, 0, 1] = np.float32(-0.0)
    assert states.view("<u4")[d_start + 1, 0, 1] == 0x80000000
    primitives["control_q_trace_state_xor"] = (
        _float32_xor_state_payload(states)
    )
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("consumer_gate_contract_valid" in error for error in validation.errors)


@pytest.mark.parametrize(
    "inherited_value",
    (np.float32(1.0), np.float32(-0.0)),
    ids=("nonzero", "negative-zero"),
)
def test_resigned_new_identity_consumer_inheritance_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    inherited_value: np.float32,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    primitives = run["critical_run_primitives"]
    ends = _segment_ends(tuple(run["run_summary"]["segment_lengths"]))
    d_start = int(ends[3])
    states = _unpack_float32_state_payload(
        primitives["behavior_pair_weight_state_xor"]
    )
    states[d_start, 0, 1] = inherited_value
    if inherited_value == 0.0:
        assert states.view("<u4")[d_start, 0, 1] == 0x80000000
    primitives["behavior_pair_weight_state_xor"] = (
        _float32_xor_state_payload(states)
    )
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("consumer_gate_contract_valid" in error for error in validation.errors)


def test_resigned_feature_memory_enabled_flag_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    primitives = artifact["scientific_payload"]["runs"][0][
        "critical_run_primitives"
    ]
    primitives["feature_memory_enabled"] = False
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("feature_memory_enabled" in error for error in validation.errors)


def test_resigned_feature_confirmation_config_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    condition = artifact["scientific_payload"]["runs"][0][
        "condition_config"
    ]["agent_config"]
    condition["feature_evidence_confirmation_steps"] -= 1
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("condition config is invalid" in error for error in validation.errors)


def test_resigned_omitted_d_repromotion_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    cycle_steps = run["run_summary"]["cycle_steps"]
    repromotion_start = cycle_steps - 600
    repromotion_end = repromotion_start + 8

    bank_rle = run["critical_run_primitives"]["bank_state_rle"]
    final_bank = bank_rle.pop()
    assert final_bank["start"] < repromotion_start
    assert final_bank["end_exclusive"] == cycle_steps + 1
    for start, end, d_live in (
        (final_bank["start"], repromotion_start, False),
        (repromotion_start, repromotion_end, True),
        (repromotion_end, cycle_steps + 1, False),
    ):
        deployed = copy.deepcopy(final_bank["deployed_descriptors"])
        shadow = copy.deepcopy(final_bank["shadow_descriptors"])
        if d_live:
            deployed[1] = [4, 5]
            shadow[1] = [4, 5]
        bank_rle.append(
            {
                "start": start,
                "end_exclusive": end,
                "deployed_descriptors": deployed,
                "shadow_descriptors": shadow,
            }
        )

    d_rle = run["critical_lifecycle"]["d_lifecycle_rle"]
    final_d = d_rle.pop()
    assert final_d["deployed_slot"] == -1
    for start, end, slot in (
        (final_d["start"], repromotion_start, -1),
        (repromotion_start, repromotion_end, 1),
        (repromotion_end, cycle_steps + 1, -1),
    ):
        d_rle.append(
            {
                "start": start,
                "end_exclusive": end,
                "deployed_slot": slot,
                "shadow_slot": slot,
                "candidate_slot": 38,
            }
        )
    # The declared event list intentionally omits the new rising edge.
    assert len(run["critical_lifecycle"]["d_promotion_event_steps"]) == 1
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("d_promotion_event_steps do not reconstruct" in error for error in validation.errors)


@pytest.mark.parametrize("mutation", ("omission", "reorder", "change"))
def test_resigned_candidate_archive_corruption_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    artifact = _artifact(monkeypatch)
    archive = artifact["scientific_payload"]["runs"][0][
        "critical_run_primitives"
    ]["candidate_bank_state_rle"][0]["candidate_descriptors"]
    if mutation == "omission":
        archive.pop()
    elif mutation == "reorder":
        archive[0], archive[1] = archive[1], archive[0]
    else:
        archive[0] = [0, 3]
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("candidate_bank_state_rle" in error for error in validation.errors)


def test_resigned_wrong_seeded_segment_lengths_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    run = artifact["scientific_payload"]["runs"][0]
    summary = run["run_summary"]
    summary["segment_lengths"] = list(summary["segment_lengths"])
    summary["segment_lengths"][0] += 1
    summary["segment_lengths"][1] -= 1
    summary["segments"][0]["length"] += 1
    summary["segments"][1]["length"] -= 1
    cycle_steps = summary["cycle_steps"]
    mean_reward = (
        sum(segment["length"] * segment["mean_reward"] for segment in summary["segments"])
        / cycle_steps
    )
    summary["mean_reward"] = mean_reward
    epsilon = IntegratedHiddenPartnerConfig().epsilon
    perfect_policy_reward = (1.0 - epsilon) * 0.95 + epsilon * 0.5
    summary["normalized_control_score"] = (mean_reward - 0.5) / (perfect_policy_reward - 0.5)
    _resign(artifact)

    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any(
        "segment lengths do not match the exact seeded schedule" in error
        for error in validation.errors
    )


def test_validator_rejects_order_config_scope_rle_and_forged_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _artifact(monkeypatch)

    changed_index = copy.deepcopy(base)
    changed_index["scientific_payload"]["runs"][0]["cell_index"] = False
    _resign(changed_index)
    assert not validate_lease_tuning_artifact(changed_index).valid

    changed_condition = copy.deepcopy(base)
    changed_condition["scientific_payload"]["runs"][0]["condition_config"]["agent_config"][
        "retire_stale_features"
    ] = False
    _resign(changed_condition)
    assert not validate_lease_tuning_artifact(changed_condition).valid

    changed_scope = copy.deepcopy(base)
    changed_scope["scientific_payload"]["scope_limits"] = []
    _resign(changed_scope)
    assert not validate_lease_tuning_artifact(changed_scope).valid

    changed_rle = copy.deepcopy(base)
    changed_rle["scientific_payload"]["runs"][0]["critical_lifecycle"]["c_lifecycle_rle"][0][
        "end_exclusive"
    ] -= 1
    _resign(changed_rle)
    assert not validate_lease_tuning_artifact(changed_rle).valid

    forged = copy.deepcopy(base)
    lifecycle = forged["scientific_payload"]["runs"][0]["critical_lifecycle"]
    lifecycle["c_first_late_masked_nll_increase"] = 0.0
    lifecycle["c_first_late_masked_nll_positive_fraction"] = 0.0
    _resign(forged)
    validation = validate_lease_tuning_artifact(forged)
    assert not validation.valid
    assert any("does not reconstruct" in error for error in validation.errors)


def test_validator_reconstructs_aggregate_selection_digest_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _artifact(monkeypatch)

    aggregate = copy.deepcopy(base)
    aggregate["scientific_payload"]["aggregates"][5]["mean_reward"] = 0.1
    _resign(aggregate)
    assert not validate_lease_tuning_artifact(aggregate).valid

    selection = copy.deepcopy(base)
    selection["scientific_payload"]["selected_cell"] = selection["scientific_payload"][
        "aggregates"
    ][6]
    _resign(selection)
    assert not validate_lease_tuning_artifact(selection).valid

    digest = copy.deepcopy(base)
    digest["scientific_digest"]["sha256"] = "0" * 64
    assert not validate_lease_tuning_artifact(digest).valid

    source = copy.deepcopy(base)
    first_source = next(iter(source["scientific_payload"]["source_sha256"]))
    source["scientific_payload"]["source_sha256"][first_source] = "0" * 64
    _resign(source)
    assert not validate_lease_tuning_artifact(source).valid


def test_loader_rejects_exponent_overflow_and_huge_integer(
    tmp_path: Path,
) -> None:
    exponent = tmp_path / "exponent.json"
    exponent.write_text('{"value":1e400}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_lease_tuning_artifact(exponent)

    huge = tmp_path / "huge.json"
    huge.write_text('{"value":' + "9" * 100 + "}", encoding="utf-8")
    with pytest.raises(ValueError, match="size bound"):
        load_lease_tuning_artifact(huge)


def test_public_artifact_boundaries_reject_forbidden_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(monkeypatch)
    synthetic_artifact = build_lease_tuning_artifact(
        record,
        operational_metadata=_operational(),
    )

    calls = {"source_snapshots": 0, "seed_derivations": 0}

    def forbidden_source_snapshot(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["source_snapshots"] += 1
        raise AssertionError("public builder read the source closure")

    def forbidden_seed_derivation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["seed_derivations"] += 1
        raise AssertionError("public validator derived forbidden seeds")

    monkeypatch.setattr(artifact_module, "source_snapshot", forbidden_source_snapshot)
    monkeypatch.setattr(
        artifact_module,
        "derive_hidden_partner_seed_pairs",
        forbidden_seed_derivation,
    )

    with pytest.raises(RuntimeError, match="FORBIDDEN/UNEXECUTED"):
        artifact_module.build_lease_tuning_artifact(
            record,
            operational_metadata=_operational(),
        )
    validation = artifact_module.validate_lease_tuning_artifact(synthetic_artifact)

    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("FORBIDDEN/UNEXECUTED" in error for error in validation.errors)
    assert calls == {"source_snapshots": 0, "seed_derivations": 0}


def test_forbidden_cli_has_no_execution_or_write_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = {"source_snapshots": 0, "runner_calls": 0, "artifact_builds": 0}

    def forbidden_source_snapshot(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["source_snapshots"] += 1
        raise AssertionError("forbidden CLI read the source closure")

    def forbidden_runner(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["runner_calls"] += 1
        raise AssertionError("forbidden CLI entered the runner")

    def forbidden_builder(*args: object, **kwargs: object) -> None:
        del args, kwargs
        calls["artifact_builds"] += 1
        raise AssertionError("forbidden CLI entered the artifact builder")

    monkeypatch.setattr(cli_module, "source_snapshot", forbidden_source_snapshot)
    monkeypatch.setattr(cli_module, "run_evidence_lease_tuning_grid", forbidden_runner)
    monkeypatch.setattr(cli_module, "build_lease_tuning_artifact", forbidden_builder)

    output = tmp_path / "forbidden.json"
    assert main(["--run", "--output", str(output)]) == 2
    result = json.loads(capsys.readouterr().out)

    assert result["valid"] is False
    assert any("FORBIDDEN/UNEXECUTED" in error for error in result["errors"])
    assert not output.exists()
    assert calls == {"source_snapshots": 0, "runner_calls": 0, "artifact_builds": 0}


def test_huge_in_memory_wall_seconds_fails_without_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _artifact(monkeypatch)
    artifact["operational_metadata"]["wall_seconds"] = 10**10_000
    validation = validate_lease_tuning_artifact(artifact)
    assert not validation.valid
    assert not validation.feasible_cell_selected
    assert any("wall_seconds" in error for error in validation.errors)
