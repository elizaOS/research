"""Strict tests for the pure, nonexecuting v6 scan-plan geometry."""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    BASE_SEGMENT_LENGTHS,
    CYCLE_COUNT,
    ENTRY_WINDOW_STEPS,
    FINAL_WINDOW_STEPS,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA,
    JITTER_RADIUS,
    MAX_CYCLE_LENGTH,
    MAX_SCAN_STEPS,
    MIN_CYCLE_LENGTH,
    N_SEGMENTS,
    REGIME_SCHEDULE,
    TAIL_WINDOW_STEPS,
    HiddenPartnerLifecycleWorldV6ControlSuiteNotReadyError,
    build_hidden_partner_lifecycle_world_v6_scan_plan,
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    build_v6_control_suite_readiness,
    canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes,
    load_hidden_partner_lifecycle_world_v6_scan_plan,
    require_v6_control_suite_ready,
    v6_scheduled_active_mask,
    validate_hidden_partner_lifecycle_world_v6_scan_plan,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit


def _lengths(jitters: tuple[int, ...]) -> np.ndarray:
    return np.asarray(
        tuple(
            base + jitter
            for base, jitter in zip(BASE_SEGMENT_LENGTHS, jitters, strict=True)
        ),
        dtype=np.int32,
    )


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def test_static_scan_bounds_are_derived_from_the_exact_world_config() -> None:
    config = HiddenPartnerWorldFeedbackConfig()
    assert BASE_SEGMENT_LENGTHS == config.base_segment_lengths
    assert REGIME_SCHEDULE == config.regime_schedule
    assert JITTER_RADIUS == config.jitter_radius == 63
    assert N_SEGMENTS == len(config.regime_schedule) == 9
    assert MIN_CYCLE_LENGTH == sum(config.base_segment_lengths) - 9 * config.jitter_radius
    assert MAX_CYCLE_LENGTH == sum(config.base_segment_lengths) + 9 * config.jitter_radius
    assert MIN_CYCLE_LENGTH == 14_025
    assert MAX_CYCLE_LENGTH == 15_159
    assert MAX_SCAN_STEPS == CYCLE_COUNT * MAX_CYCLE_LENGTH == 30_318
    assert MAX_SCAN_STEPS <= np.iinfo(np.int32).max


@pytest.mark.parametrize(
    "jitters",
    (
        (-JITTER_RADIUS,) * N_SEGMENTS,
        (0,) * N_SEGMENTS,
        (JITTER_RADIUS,) * N_SEGMENTS,
        (-63, -31, -1, 0, 1, 17, 42, 62, 63),
    ),
)
def test_minimum_maximum_and_mixed_jitter_patterns_have_exact_prefix_masks(
    jitters: tuple[int, ...],
) -> None:
    segment_lengths = _lengths(jitters)
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(segment_lengths)
    expected_cycle_length = sum(int(value) for value in segment_lengths)
    mask = np.asarray(plan.scheduled_active)

    assert plan.cycle_length == expected_cycle_length
    assert plan.run_steps == CYCLE_COUNT * expected_cycle_length
    assert plan.segment_ends == tuple(
        int(value) for value in np.cumsum(segment_lengths, dtype=np.int32)
    )
    assert mask.shape == (MAX_SCAN_STEPS,)
    assert mask.dtype == np.bool_
    assert np.count_nonzero(mask) == plan.run_steps
    assert np.all(mask[: plan.run_steps])
    assert not np.any(mask[plan.run_steps :])
    assert bool(mask[plan.run_steps - 1])
    if plan.run_steps < MAX_SCAN_STEPS:
        assert not bool(mask[plan.run_steps])
    else:
        assert np.all(mask)


def test_exact_eighteen_segment_occurrences_and_final_window_are_reconstructed() -> None:
    segment_lengths = _lengths((-63, -31, -1, 0, 1, 17, 42, 62, 63))
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(segment_lengths)
    mask = np.asarray(plan.scheduled_active)

    assert len(plan.segment_occurrences) == CYCLE_COUNT * N_SEGMENTS == 18
    for occurrence_index, occurrence in enumerate(plan.segment_occurrences):
        cycle_index, segment_index = divmod(occurrence_index, N_SEGMENTS)
        relative_start = 0 if segment_index == 0 else plan.segment_ends[segment_index - 1]
        expected_start = cycle_index * plan.cycle_length + relative_start
        expected_end = cycle_index * plan.cycle_length + plan.segment_ends[segment_index]

        assert occurrence.occurrence_index == occurrence_index
        assert occurrence.cycle_index == cycle_index
        assert occurrence.segment_index == segment_index
        assert occurrence.regime_id == REGIME_SCHEDULE[segment_index]
        assert occurrence.start == expected_start
        assert occurrence.end_exclusive == expected_end
        assert occurrence.length == plan.segment_lengths[segment_index]
        assert occurrence.entry_window.kind == "entry"
        assert occurrence.entry_window.start == expected_start
        assert occurrence.entry_window.end_exclusive == expected_start + ENTRY_WINDOW_STEPS
        assert occurrence.entry_window.steps == ENTRY_WINDOW_STEPS
        assert occurrence.tail_window.kind == "tail"
        assert occurrence.tail_window.start == expected_end - TAIL_WINDOW_STEPS
        assert occurrence.tail_window.end_exclusive == expected_end
        assert occurrence.tail_window.steps == TAIL_WINDOW_STEPS
        assert (
            occurrence.start
            <= occurrence.entry_window.start
            < occurrence.entry_window.end_exclusive
            <= occurrence.end_exclusive
        )
        assert (
            occurrence.start
            <= occurrence.tail_window.start
            < occurrence.tail_window.end_exclusive
            <= occurrence.end_exclusive
        )
        assert np.all(mask[occurrence.entry_window.start : occurrence.entry_window.end_exclusive])
        assert np.all(mask[occurrence.tail_window.start : occurrence.tail_window.end_exclusive])

    final_occurrence = plan.segment_occurrences[-1]
    assert plan.final_window.kind == "final"
    assert plan.final_window.start == plan.run_steps - FINAL_WINDOW_STEPS
    assert plan.final_window.end_exclusive == plan.run_steps
    assert plan.final_window.steps == FINAL_WINDOW_STEPS
    assert final_occurrence.start <= plan.final_window.start
    assert plan.final_window.end_exclusive <= final_occurrence.end_exclusive
    assert np.all(mask[plan.final_window.start : plan.final_window.end_exclusive])


def test_scheduled_active_primitive_is_static_shape_and_jit_compatible() -> None:
    run_steps = jnp.asarray(2 * sum(BASE_SEGMENT_LENGTHS), dtype=jnp.int32)
    eager = v6_scheduled_active_mask(run_steps)
    compiled = jax.jit(v6_scheduled_active_mask)(run_steps)
    np.testing.assert_array_equal(compiled, eager)
    assert compiled.shape == (MAX_SCAN_STEPS,)
    assert compiled.dtype == jnp.bool_

    with pytest.raises(ValueError, match="scalar"):
        v6_scheduled_active_mask(jnp.asarray([1], dtype=jnp.int32))
    with pytest.raises(TypeError, match="int32"):
        v6_scheduled_active_mask(jnp.asarray(1.0, dtype=jnp.float32))


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (list(BASE_SEGMENT_LENGTHS), ValueError),
        (np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int64), TypeError),
        (np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.float32), TypeError),
        (np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.bool_), TypeError),
        (np.asarray(BASE_SEGMENT_LENGTHS[:-1], dtype=np.int32), ValueError),
        (np.asarray([BASE_SEGMENT_LENGTHS], dtype=np.int32), ValueError),
    ),
)
def test_segment_lengths_reject_coercion_wrong_dtype_and_wrong_shape(
    value: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_hidden_partner_lifecycle_world_v6_scan_plan(value)


def test_segment_length_bounds_are_exact_and_off_by_one_values_fail() -> None:
    lower = _lengths((-JITTER_RADIUS,) * N_SEGMENTS)
    upper = _lengths((JITTER_RADIUS,) * N_SEGMENTS)
    build_hidden_partner_lifecycle_world_v6_scan_plan(lower)
    build_hidden_partner_lifecycle_world_v6_scan_plan(upper)

    too_low = lower.copy()
    too_low[0] -= np.int32(1)
    with pytest.raises(ValueError, match="jitter bound"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(too_low)
    too_high = upper.copy()
    too_high[-1] += np.int32(1)
    with pytest.raises(ValueError, match="jitter bound"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(too_high)

    zero = lower.copy()
    zero[0] = np.int32(0)
    with pytest.raises(ValueError, match="positive"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(zero)


def test_segment_ends_require_exact_int32_shape_and_cumulative_values() -> None:
    lengths = np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    ends = np.cumsum(lengths, dtype=np.int32)
    build_hidden_partner_lifecycle_world_v6_scan_plan(lengths, segment_ends=ends)

    wrong_value = ends.copy()
    wrong_value[4] += np.int32(1)
    with pytest.raises(ValueError, match="exact cumulative"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(
            lengths,
            segment_ends=wrong_value,
        )
    with pytest.raises(TypeError, match="int32"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(
            lengths,
            segment_ends=ends.astype(np.int64),
        )
    with pytest.raises(ValueError, match="shape"):
        build_hidden_partner_lifecycle_world_v6_scan_plan(
            lengths,
            segment_ends=ends[:-1],
        )


def test_initialized_world_state_is_accepted_but_advanced_or_corrupt_state_fails() -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    from_state = build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(state)
    direct = build_hidden_partner_lifecycle_world_v6_scan_plan(
        state.segment_lengths,
        segment_ends=state.segment_ends,
    )
    assert from_state == direct

    advanced = state.replace(step_count=jnp.asarray(1, dtype=jnp.int32))
    with pytest.raises(ValueError, match="step-zero"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(advanced)
    wrong_end_dtype = state.replace(segment_ends=state.segment_ends.astype(jnp.float32))
    with pytest.raises(TypeError, match="int32"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_end_dtype)
    corrupt_ends = state.replace(segment_ends=state.segment_ends.at[-1].add(1))
    with pytest.raises(ValueError, match="exact cumulative"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(corrupt_ends)
    with pytest.raises(TypeError, match="HiddenPartnerWorldFeedbackState"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(object())  # type: ignore[arg-type]


def test_payload_is_authority_free_and_padding_sentinels_are_exact() -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    payload = plan.to_config()
    keys = _all_keys(payload)

    assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA
    assert payload["status"] == "NONEXECUTING_DEVELOPMENT_SCAN_PLAN"
    assert payload["development_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert "seed" not in keys
    assert "seed_namespace" not in keys
    assert "thresholds" not in keys
    assert "outcomes" not in keys

    active = payload["scheduled_active_contract"]
    assert isinstance(active, dict)
    assert active == {
        "dtype": "bool",
        "shape": [MAX_SCAN_STEPS],
        "expression": "scan_index < run_steps",
        "active_start": 0,
        "active_end_exclusive": plan.run_steps,
        "padding_start": plan.run_steps,
        "padding_end_exclusive": MAX_SCAN_STEPS,
        "true_count": plan.run_steps,
        "false_count": MAX_SCAN_STEPS - plan.run_steps,
        "is_one_contiguous_prefix": True,
    }
    padding = payload["padding_contract"]
    assert isinstance(padding, dict)
    assert padding["padding_is_execution"] is False
    assert padding["padding_is_rejection"] is False
    assert padding["padding_contributes_support"] is False
    assert padding["world_step_called"] is False
    assert padding["agent_decide_called"] is False
    assert padding["agent_update_called"] is False
    assert padding["carry_preserved_bit_exact"] is True
    assert padding["canonical_sentinels"] == {
        "bool": False,
        "int32": -1,
        "uint32": 0,
        "float32": "positive_zero_bit_pattern_0x00000000",
        "scheduled_active": False,
    }


def test_canonical_json_round_trip_is_exact_and_reconstructs_arrays() -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        _lengths((-63, -31, -1, 0, 1, 17, 42, 62, 63))
    )
    encoded = canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(plan)
    decoded = load_hidden_partner_lifecycle_world_v6_scan_plan(encoded)

    assert decoded == plan
    assert canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(decoded) == encoded
    assert encoded == _canonical_bytes(plan.to_config())
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded
    validate_hidden_partner_lifecycle_world_v6_scan_plan(decoded)

    tampered_plan = dataclasses.replace(plan, run_steps=plan.run_steps - 1)
    with pytest.raises(ValueError, match="reconstructed"):
        canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(tampered_plan)


def test_loader_rejects_noncanonical_duplicate_nonfinite_and_tampered_payloads() -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    encoded = canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(plan)
    payload = json.loads(encoded)

    with pytest.raises(ValueError, match="not canonical"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(encoded + b"\n")

    duplicate = encoded.replace(
        b'{"development_only":true',
        b'{"development_only":true,"development_only":true',
        1,
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(duplicate)

    nonfinite = encoded.replace(
        f'"run_steps":{plan.run_steps}'.encode(),
        b'"run_steps":NaN',
        1,
    )
    with pytest.raises(ValueError, match="cannot contain NaN"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(nonfinite)

    floating = encoded.replace(
        f'"run_steps":{plan.run_steps}'.encode(),
        f'"run_steps":{plan.run_steps}.0'.encode(),
        1,
    )
    with pytest.raises(ValueError, match="no floating-point"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(floating)

    assert isinstance(payload, dict)
    payload["extra"] = True
    with pytest.raises(ValueError, match="differs"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(_canonical_bytes(payload))


def test_loader_rejects_wrong_types_off_by_one_ends_and_int32_overflow() -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    payload = plan.to_config()

    wrong_authority_type = json.loads(_canonical_bytes(payload))
    wrong_authority_type["execution_authorized"] = 0
    with pytest.raises(ValueError, match="differs"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(
            _canonical_bytes(wrong_authority_type)
        )

    wrong_end = json.loads(_canonical_bytes(payload))
    wrong_end["geometry"]["segment_ends"][0] += 1
    with pytest.raises(ValueError, match="exact cumulative"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(_canonical_bytes(wrong_end))

    overflow = json.loads(_canonical_bytes(payload))
    overflow["geometry"]["segment_lengths"][0] = 2**31
    with pytest.raises(ValueError, match="fit in int32"):
        load_hidden_partner_lifecycle_world_v6_scan_plan(_canonical_bytes(overflow))


def test_current_control_readiness_is_exact_and_requirement_fails_visibly() -> None:
    readiness = build_v6_control_suite_readiness()
    assert readiness.primary_ready_count == 15
    assert readiness.primary_required_count == 15
    assert readiness.diagnostic_ready_count == 2
    assert readiness.diagnostic_required_count == 3
    assert not readiness.all_controls_ready
    assert len(readiness.blocked_controls) == 1
    blocked = readiness.blocked_controls[0]
    assert blocked.family == "diagnostic"
    assert blocked.name == "uniform_action"
    assert "balanced external focal-action intervention" in blocked.reason

    payload = readiness.to_config()
    assert payload["status"] == "DEVELOPMENT_CONTROL_SUITE_BLOCKED"
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert "seed" not in _all_keys(payload)
    assert "thresholds" not in _all_keys(payload)
    assert "outcomes" not in _all_keys(payload)

    with pytest.raises(
        HiddenPartnerLifecycleWorldV6ControlSuiteNotReadyError,
        match="diagnostic:uniform_action",
    ):
        require_v6_control_suite_ready()
