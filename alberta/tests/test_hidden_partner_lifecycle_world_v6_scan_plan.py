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
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_READINESS_SCHEMA,
    HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA,
    JITTER_RADIUS,
    MAX_CYCLE_LENGTH,
    MAX_SCAN_STEPS,
    MIN_CYCLE_LENGTH,
    N_SEGMENTS,
    REGIME_SCHEDULE,
    TAIL_WINDOW_STEPS,
    HiddenPartnerLifecycleWorldV6BlockedControl,
    HiddenPartnerLifecycleWorldV6ControlBinding,
    HiddenPartnerLifecycleWorldV6ControlSuiteReadiness,
    HiddenPartnerLifecycleWorldV6ScanPlan,
    HiddenPartnerLifecycleWorldV6SegmentOccurrence,
    HiddenPartnerLifecycleWorldV6Window,
    build_hidden_partner_lifecycle_world_v6_scan_plan,
    build_hidden_partner_lifecycle_world_v6_scan_plan_from_state,
    build_v6_control_suite_readiness,
    canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes,
    load_hidden_partner_lifecycle_world_v6_scan_plan,
    require_v6_control_suite_ready,
    v6_scheduled_active_mask,
    validate_hidden_partner_lifecycle_world_v6_scan_plan,
    validate_v6_control_suite_readiness,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackState,
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


def _synthetic_blocked_control() -> HiddenPartnerLifecycleWorldV6BlockedControl:
    return HiddenPartnerLifecycleWorldV6BlockedControl(
        family="diagnostic",
        name="uniform_action",
        reason="synthetic unavailable binding",
    )


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


_NUMERIC_PLAN_PATHS = (
    "segment_lengths",
    "segment_ends",
    "cycle_length",
    "run_steps",
    "occurrence.occurrence_index",
    "occurrence.cycle_index",
    "occurrence.segment_index",
    "occurrence.regime_id",
    "occurrence.start",
    "occurrence.end_exclusive",
    "occurrence.length",
    "entry_window.start",
    "entry_window.end_exclusive",
    "entry_window.steps",
    "tail_window.start",
    "tail_window.end_exclusive",
    "tail_window.steps",
    "final_window.start",
    "final_window.end_exclusive",
    "final_window.steps",
)


def _numeric_plan_value(
    plan: HiddenPartnerLifecycleWorldV6ScanPlan,
    path: str,
) -> int:
    occurrence = plan.segment_occurrences[0]
    if path == "segment_lengths":
        return plan.segment_lengths[0]
    if path == "segment_ends":
        return plan.segment_ends[0]
    if path in ("cycle_length", "run_steps"):
        return getattr(plan, path)
    if path.startswith("occurrence."):
        return getattr(occurrence, path.removeprefix("occurrence."))
    if path.startswith("entry_window."):
        return getattr(occurrence.entry_window, path.removeprefix("entry_window."))
    if path.startswith("tail_window."):
        return getattr(occurrence.tail_window, path.removeprefix("tail_window."))
    if path.startswith("final_window."):
        return getattr(plan.final_window, path.removeprefix("final_window."))
    raise AssertionError(f"unknown numeric plan path: {path}")


def _forge_numeric_plan_field(
    plan: HiddenPartnerLifecycleWorldV6ScanPlan,
    path: str,
    value: object,
) -> HiddenPartnerLifecycleWorldV6ScanPlan:
    if path == "segment_lengths":
        return dataclasses.replace(
            plan,
            segment_lengths=(value, *plan.segment_lengths[1:]),
        )
    if path == "segment_ends":
        return dataclasses.replace(
            plan,
            segment_ends=(value, *plan.segment_ends[1:]),
        )
    if path in ("cycle_length", "run_steps"):
        return dataclasses.replace(plan, **{path: value})

    occurrence = plan.segment_occurrences[0]
    if path.startswith("occurrence."):
        occurrence = dataclasses.replace(
            occurrence,
            **{path.removeprefix("occurrence."): value},
        )
    elif path.startswith("entry_window."):
        window = dataclasses.replace(
            occurrence.entry_window,
            **{path.removeprefix("entry_window."): value},
        )
        occurrence = dataclasses.replace(occurrence, entry_window=window)
    elif path.startswith("tail_window."):
        window = dataclasses.replace(
            occurrence.tail_window,
            **{path.removeprefix("tail_window."): value},
        )
        occurrence = dataclasses.replace(occurrence, tail_window=window)
    elif path.startswith("final_window."):
        window = dataclasses.replace(
            plan.final_window,
            **{path.removeprefix("final_window."): value},
        )
        return dataclasses.replace(plan, final_window=window)
    else:
        raise AssertionError(f"unknown numeric plan path: {path}")
    return dataclasses.replace(
        plan,
        segment_occurrences=(occurrence, *plan.segment_occurrences[1:]),
    )


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


def test_world_state_projection_requires_the_exact_state_class() -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))

    @dataclasses.dataclass(frozen=True)
    class StateSubclass(HiddenPartnerWorldFeedbackState):
        pass

    subclass = StateSubclass(
        **{
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(HiddenPartnerWorldFeedbackState)
        }
    )
    with pytest.raises(TypeError, match="exact HiddenPartnerWorldFeedbackState"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(subclass)


@pytest.mark.parametrize(
    "field",
    ("signal_key", "partner_key", "world_key", "cue_key", "outcome_key"),
)
def test_world_state_projection_rejects_every_malformed_prng_key(field: str) -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    wrong_dtype = state.replace(
        **{field: jnp.zeros((2,), dtype=jnp.float32)},
    )
    with pytest.raises(TypeError, match="scalar typed key or legacy uint32"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_dtype)

    wrong_shape = state.replace(
        **{field: jnp.stack((getattr(state, field), getattr(state, field)))},
    )
    with pytest.raises(ValueError, match="scalar typed JAX PRNG key"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_shape)


def test_world_state_projection_accepts_one_legacy_uint32_key_per_rng_stream() -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.PRNGKey(7))
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(state)
    assert plan.segment_lengths == tuple(int(value) for value in state.segment_lengths)


def test_world_state_projection_rejects_mixed_prng_key_representations() -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    mixed = state.replace(signal_key=jr.PRNGKey(1))
    with pytest.raises(ValueError, match="one representation and implementation"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(mixed)


@pytest.mark.parametrize("field", ("current_signals", "current_cues", "world_sign"))
def test_world_state_projection_rejects_nonfinite_sign_fields(field: str) -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    value = getattr(state, field)
    corrupt = state.replace(
        **{field: jnp.full_like(value, jnp.asarray(float("nan"), dtype=jnp.float32))},
    )
    with pytest.raises(ValueError, match="finite signs"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(corrupt)


@pytest.mark.parametrize("field", ("current_signals", "current_cues", "world_sign"))
def test_world_state_projection_rejects_out_of_domain_sign_fields(field: str) -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    value = getattr(state, field)
    corrupt = state.replace(**{field: jnp.zeros_like(value)})
    with pytest.raises(ValueError, match=r"exact -1 or \+1 signs"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(corrupt)


def test_world_state_projection_rejects_wrong_sign_shapes_and_dtypes() -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    wrong_signal_shape = state.replace(
        current_signals=jnp.ones((2,), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="current_signals must have exact shape"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_signal_shape)

    wrong_cue_dtype = state.replace(
        current_cues=jnp.ones((2,), dtype=jnp.int32),
    )
    with pytest.raises(TypeError, match="current_cues must have exact float32 dtype"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_cue_dtype)

    wrong_world_shape = state.replace(
        world_sign=jnp.ones((1,), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match="world_sign must have exact shape"):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(wrong_world_shape)


@pytest.mark.parametrize(
    ("corruption", "error", "message"),
    (
        ("outcome_shape", ValueError, "previous_outcome must have exact shape"),
        ("outcome_dtype", TypeError, "previous_outcome must have exact float32 dtype"),
        ("outcome_nonfinite", ValueError, "previous_outcome must be finite"),
        ("outcome_negative_zero", ValueError, "exact positive float32 zero"),
        ("outcome_domain", ValueError, "exact positive float32 zero"),
        ("partner_shape", ValueError, "previous_partner_action must have exact shape"),
        ("partner_dtype", TypeError, "previous_partner_action must have exact int32 dtype"),
        ("partner_domain", ValueError, "step-zero sentinel 0"),
        ("history_shape", ValueError, "has_partner_history must have exact shape"),
        ("history_dtype", TypeError, "has_partner_history must have exact bool dtype"),
        ("history_domain", ValueError, "has_partner_history must be false"),
        ("counter_shape", ValueError, "step_count must be scalar"),
        ("counter_dtype", TypeError, "step_count must have exact int32 dtype"),
        ("counter_domain", ValueError, "initialized step-zero"),
    ),
)
def test_world_state_projection_rejects_invalid_history_and_counter_fields(
    corruption: str,
    error: type[Exception],
    message: str,
) -> None:
    state = HiddenPartnerWorldFeedbackWorld().init(jr.key(7))
    corruptions = {
        "outcome_shape": state.replace(
            previous_outcome=jnp.asarray([0.0], dtype=jnp.float32),
        ),
        "outcome_dtype": state.replace(
            previous_outcome=jnp.asarray(0, dtype=jnp.int32),
        ),
        "outcome_nonfinite": state.replace(
            previous_outcome=jnp.asarray(float("nan"), dtype=jnp.float32),
        ),
        "outcome_negative_zero": state.replace(
            previous_outcome=jnp.asarray(-0.0, dtype=jnp.float32),
        ),
        "outcome_domain": state.replace(
            previous_outcome=jnp.asarray(1.0, dtype=jnp.float32),
        ),
        "partner_shape": state.replace(
            previous_partner_action=jnp.asarray([0], dtype=jnp.int32),
        ),
        "partner_dtype": state.replace(
            previous_partner_action=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        "partner_domain": state.replace(
            previous_partner_action=jnp.asarray(1, dtype=jnp.int32),
        ),
        "history_shape": state.replace(
            has_partner_history=jnp.asarray([False], dtype=jnp.bool_),
        ),
        "history_dtype": state.replace(
            has_partner_history=jnp.asarray(0, dtype=jnp.int32),
        ),
        "history_domain": state.replace(
            has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
        ),
        "counter_shape": state.replace(
            step_count=jnp.asarray([0], dtype=jnp.int32),
        ),
        "counter_dtype": state.replace(
            step_count=jnp.asarray(0.0, dtype=jnp.float32),
        ),
        "counter_domain": state.replace(
            step_count=jnp.asarray(-1, dtype=jnp.int32),
        ),
    }
    with pytest.raises(error, match=message):
        build_hidden_partner_lifecycle_world_v6_scan_plan_from_state(
            corruptions[corruption]
        )


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


@pytest.mark.parametrize("path", _NUMERIC_PLAN_PATHS)
@pytest.mark.parametrize("wrong_type", ("float", "bool", "numpy_integer"))
def test_in_memory_plan_rejects_every_wrong_numeric_field_type_before_casting(
    path: str,
    wrong_type: str,
) -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    original = _numeric_plan_value(plan, path)
    replacements: dict[str, object] = {
        "float": float(original),
        "bool": bool(original),
        "numpy_integer": np.int64(original),
    }
    forged = _forge_numeric_plan_field(plan, path, replacements[wrong_type])

    with pytest.raises(TypeError, match="exact built-in int"):
        validate_hidden_partner_lifecycle_world_v6_scan_plan(forged)
    with pytest.raises(TypeError, match="exact built-in int"):
        forged.to_config()
    with pytest.raises(TypeError, match="exact built-in int"):
        canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(forged)


@pytest.mark.parametrize("path", _NUMERIC_PLAN_PATHS)
def test_public_serializer_rejects_forged_proof_geometry(path: str) -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    forged = _forge_numeric_plan_field(
        plan,
        path,
        _numeric_plan_value(plan, path) + 1,
    )
    with pytest.raises(ValueError, match="reconstructed|exact cumulative"):
        forged.to_config()


@pytest.mark.parametrize(
    ("window_name", "kind"),
    (
        ("entry_window", "tail"),
        ("tail_window", "entry"),
        ("final_window", "entry"),
    ),
)
def test_window_kinds_must_match_their_exact_geometry_role(
    window_name: str,
    kind: str,
) -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    occurrence = plan.segment_occurrences[0]
    if window_name == "final_window":
        forged = dataclasses.replace(
            plan,
            final_window=dataclasses.replace(plan.final_window, kind=kind),
        )
    else:
        window = getattr(occurrence, window_name)
        forged_occurrence = dataclasses.replace(
            occurrence,
            **{window_name: dataclasses.replace(window, kind=kind)},
        )
        forged = dataclasses.replace(
            plan,
            segment_occurrences=(forged_occurrence, *plan.segment_occurrences[1:]),
        )
    with pytest.raises(ValueError, match="kind must be exactly"):
        forged.to_config()


@pytest.mark.parametrize("window_name", ("entry_window", "tail_window", "final_window"))
def test_equal_window_kind_string_subclasses_are_rejected(window_name: str) -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    occurrence = plan.segment_occurrences[0]
    if window_name == "final_window":
        forged = dataclasses.replace(
            plan,
            final_window=dataclasses.replace(
                plan.final_window,
                kind=np.str_(plan.final_window.kind),
            ),
        )
    else:
        window = getattr(occurrence, window_name)
        forged_occurrence = dataclasses.replace(
            occurrence,
            **{
                window_name: dataclasses.replace(
                    window,
                    kind=np.str_(window.kind),
                )
            },
        )
        forged = dataclasses.replace(
            plan,
            segment_occurrences=(forged_occurrence, *plan.segment_occurrences[1:]),
        )
    with pytest.raises(TypeError, match="kind must be an exact built-in str"):
        forged.to_config()


def test_plan_requires_exact_tuple_and_nested_dataclass_types() -> None:
    plan = build_hidden_partner_lifecycle_world_v6_scan_plan(
        np.asarray(BASE_SEGMENT_LENGTHS, dtype=np.int32)
    )
    occurrence = plan.segment_occurrences[0]

    with pytest.raises(TypeError, match="segment_lengths.*tuple"):
        dataclasses.replace(plan, segment_lengths=list(plan.segment_lengths)).to_config()
    with pytest.raises(TypeError, match="segment_ends.*tuple"):
        dataclasses.replace(plan, segment_ends=list(plan.segment_ends)).to_config()
    with pytest.raises(TypeError, match="segment_occurrences.*tuple"):
        dataclasses.replace(
            plan,
            segment_occurrences=list(plan.segment_occurrences),
        ).to_config()

    @dataclasses.dataclass(frozen=True)
    class WindowSubclass(HiddenPartnerLifecycleWorldV6Window):
        pass

    subclass_window = WindowSubclass(
        kind=occurrence.entry_window.kind,
        start=occurrence.entry_window.start,
        end_exclusive=occurrence.entry_window.end_exclusive,
        steps=occurrence.entry_window.steps,
    )
    forged_occurrence = dataclasses.replace(
        occurrence,
        entry_window=subclass_window,
    )
    with pytest.raises(TypeError, match="exact HiddenPartnerLifecycleWorldV6Window"):
        dataclasses.replace(
            plan,
            segment_occurrences=(forged_occurrence, *plan.segment_occurrences[1:]),
        ).to_config()

    @dataclasses.dataclass(frozen=True)
    class OccurrenceSubclass(HiddenPartnerLifecycleWorldV6SegmentOccurrence):
        pass

    subclass_occurrence = OccurrenceSubclass(
        occurrence_index=occurrence.occurrence_index,
        cycle_index=occurrence.cycle_index,
        segment_index=occurrence.segment_index,
        regime_id=occurrence.regime_id,
        start=occurrence.start,
        end_exclusive=occurrence.end_exclusive,
        length=occurrence.length,
        entry_window=occurrence.entry_window,
        tail_window=occurrence.tail_window,
    )
    with pytest.raises(
        TypeError,
        match="exact HiddenPartnerLifecycleWorldV6SegmentOccurrence",
    ):
        dataclasses.replace(
            plan,
            segment_occurrences=(subclass_occurrence, *plan.segment_occurrences[1:]),
        ).to_config()


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


def test_current_control_readiness_is_exact_ready_and_nonauthorizing() -> None:
    readiness = build_v6_control_suite_readiness()
    assert validate_v6_control_suite_readiness(readiness) == readiness
    assert readiness.primary_ready_count == 15
    assert readiness.primary_required_count == 15
    assert readiness.diagnostic_ready_count == 3
    assert readiness.diagnostic_required_count == 3
    assert readiness.all_controls_ready
    assert readiness.blocked_controls == ()
    assert len(readiness.control_matrix_sha256) == 64
    assert len(readiness.bindings) == 18

    payload = readiness.to_config()
    assert payload["schema"] == HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_READINESS_SCHEMA
    assert (
        HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_CONTROL_READINESS_SCHEMA
        == "alberta.hidden-partner-lifecycle-world.control-suite-readiness-development.v2"
    )
    assert payload["schema"] != HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_SCAN_PLAN_SCHEMA
    assert payload["status"] == "DEVELOPMENT_CONTROL_SUITE_READY"
    assert payload["development_only"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["all_controls_ready"] is True
    assert payload["control_matrix_schema"] == readiness.control_matrix_schema
    assert payload["control_matrix_sha256"] == readiness.control_matrix_sha256
    assert payload["blocked_controls"] == []
    binding_payloads = payload["bindings"]
    assert isinstance(binding_payloads, list)
    assert len(binding_payloads) == 18
    for binding in binding_payloads:
        assert isinstance(binding, dict)
        assert binding["execution_authorized"] is False
        assert binding["evidence_authorized"] is False
        assert binding["scientific_promotion_allowed"] is False
    assert "seed" not in _all_keys(payload)
    assert "thresholds" not in _all_keys(payload)
    assert "outcomes" not in _all_keys(payload)

    assert require_v6_control_suite_ready() == readiness


@pytest.mark.parametrize(
    "field",
    (
        "primary_ready_count",
        "primary_required_count",
        "diagnostic_ready_count",
        "diagnostic_required_count",
    ),
)
@pytest.mark.parametrize("wrong_type", ("float", "bool", "numpy_integer"))
def test_readiness_rejects_equal_or_coercible_wrong_count_types(
    field: str,
    wrong_type: str,
) -> None:
    readiness = build_v6_control_suite_readiness()
    original = getattr(readiness, field)
    replacements: dict[str, object] = {
        "float": float(original),
        "bool": bool(original),
        "numpy_integer": np.int64(original),
    }
    forged = dataclasses.replace(readiness, **{field: replacements[wrong_type]})
    with pytest.raises(TypeError, match="exact built-in int"):
        validate_v6_control_suite_readiness(forged)
    with pytest.raises(TypeError, match="exact built-in int"):
        forged.to_config()


def test_hand_built_count_only_readiness_cannot_report_or_validate_ready() -> None:
    live = build_v6_control_suite_readiness()
    forged = HiddenPartnerLifecycleWorldV6ControlSuiteReadiness(
        primary_ready_count=15,
        primary_required_count=15,
        diagnostic_ready_count=3,
        diagnostic_required_count=3,
        blocked_controls=(),
        control_matrix_schema=live.control_matrix_schema,
        control_matrix_sha256=live.control_matrix_sha256,
        bindings=(),
    )

    assert not forged.all_controls_ready
    with pytest.raises(ValueError, match="exact unblocked control order"):
        validate_v6_control_suite_readiness(forged)
    with pytest.raises(ValueError, match="exact unblocked control order"):
        forged.to_config()


def test_readiness_rejects_same_count_digest_and_binding_order_drift() -> None:
    readiness = build_v6_control_suite_readiness()
    wrong_matrix_digest = dataclasses.replace(
        readiness,
        control_matrix_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="current live bindings"):
        validate_v6_control_suite_readiness(wrong_matrix_digest)

    reordered = dataclasses.replace(
        readiness,
        bindings=(readiness.bindings[1], readiness.bindings[0], *readiness.bindings[2:]),
    )
    with pytest.raises(ValueError, match="exact unblocked control order"):
        validate_v6_control_suite_readiness(reordered)

    changed_hash = dataclasses.replace(readiness.bindings[0])
    object.__setattr__(changed_hash, "control_config_sha256", "0" * 64)
    semantic_drift = dataclasses.replace(
        readiness,
        bindings=(changed_hash, *readiness.bindings[1:]),
    )
    with pytest.raises(ValueError, match="current live bindings"):
        validate_v6_control_suite_readiness(semantic_drift)


def test_readiness_rejects_matrix_schema_binding_schema_and_phase_tampering() -> None:
    readiness = build_v6_control_suite_readiness()
    with pytest.raises(TypeError, match="matrix_schema.*exact built-in str"):
        validate_v6_control_suite_readiness(
            dataclasses.replace(
                readiness,
                control_matrix_schema=np.str_(readiness.control_matrix_schema),
            )
        )
    with pytest.raises(TypeError, match="matrix_sha256.*exact built-in str"):
        validate_v6_control_suite_readiness(
            dataclasses.replace(
                readiness,
                control_matrix_sha256=np.str_(readiness.control_matrix_sha256),
            )
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_v6_control_suite_readiness(
            dataclasses.replace(
                readiness,
                control_matrix_sha256=readiness.control_matrix_sha256.upper(),
            )
        )
    with pytest.raises(ValueError, match="matrix schema"):
        validate_v6_control_suite_readiness(
            dataclasses.replace(
                readiness,
                control_matrix_schema=readiness.control_matrix_schema + ".tampered",
            )
        )

    for field, value, message in (
        ("binding_schema", readiness.bindings[0].binding_schema + ".tampered", "binding schema"),
        ("bridge_schema", readiness.bindings[0].bridge_schema + ".tampered", "bridge schema"),
        ("initial_external_action", 1, "canonical zero"),
    ):
        forged_binding = dataclasses.replace(readiness.bindings[0])
        object.__setattr__(forged_binding, field, value)
        forged = dataclasses.replace(
            readiness,
            bindings=(forged_binding, *readiness.bindings[1:]),
        )
        with pytest.raises(ValueError, match=message):
            validate_v6_control_suite_readiness(forged)


def test_readiness_requires_exact_binding_tuple_and_nested_binding_type() -> None:
    readiness = build_v6_control_suite_readiness()
    with pytest.raises(TypeError, match="bindings.*tuple"):
        dataclasses.replace(
            readiness,
            bindings=list(readiness.bindings),
        ).to_config()

    @dataclasses.dataclass(frozen=True)
    class BindingSubclass(HiddenPartnerLifecycleWorldV6ControlBinding):
        pass

    binding = readiness.bindings[0]
    with pytest.raises(TypeError, match="exact HiddenPartnerLifecycleWorldV6ControlBinding"):
        BindingSubclass(
            **{
                field.name: getattr(binding, field.name)
                for field in dataclasses.fields(HiddenPartnerLifecycleWorldV6ControlBinding)
            }
        )


def test_forged_blocked_control_snapshot_cannot_serialize_or_change_authority() -> None:
    readiness = build_v6_control_suite_readiness()
    forged = dataclasses.replace(
        readiness,
        diagnostic_ready_count=readiness.diagnostic_required_count - 1,
        blocked_controls=(_synthetic_blocked_control(),),
    )
    assert not forged.all_controls_ready
    with pytest.raises(ValueError, match="bindings|current live bindings"):
        validate_v6_control_suite_readiness(forged)
    with pytest.raises(ValueError, match="bindings|current live bindings"):
        forged.to_config()

    payload = readiness.to_config()
    assert payload["all_controls_ready"] is True
    assert payload["execution_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["scientific_promotion_allowed"] is False


@pytest.mark.parametrize("field", ("family", "name", "reason"))
def test_blocked_control_requires_exact_builtin_string_fields(field: str) -> None:
    readiness = build_v6_control_suite_readiness()
    blocked = _synthetic_blocked_control()
    forged_blocked = dataclasses.replace(
        blocked,
        **{field: np.str_(getattr(blocked, field))},
    )
    forged_readiness = dataclasses.replace(
        readiness,
        blocked_controls=(forged_blocked,),
    )
    with pytest.raises(TypeError, match="exact built-in str"):
        forged_blocked.to_config()
    with pytest.raises(TypeError, match="exact built-in str"):
        forged_readiness.to_config()


def test_readiness_requires_exact_blocked_tuple_and_nested_dataclass_type() -> None:
    readiness = build_v6_control_suite_readiness()
    blocked = _synthetic_blocked_control()
    with pytest.raises(TypeError, match="blocked_controls.*tuple"):
        dataclasses.replace(
            readiness,
            blocked_controls=list(readiness.blocked_controls),
        ).to_config()

    @dataclasses.dataclass(frozen=True)
    class BlockedSubclass(HiddenPartnerLifecycleWorldV6BlockedControl):
        pass

    subclass_blocked = BlockedSubclass(
        family=blocked.family,
        name=blocked.name,
        reason=blocked.reason,
    )
    with pytest.raises(
        TypeError,
        match="exact HiddenPartnerLifecycleWorldV6BlockedControl",
    ):
        dataclasses.replace(
            readiness,
            blocked_controls=(subclass_blocked,),
        ).to_config()


def test_blocked_control_literal_and_nonempty_contracts_are_strict() -> None:
    blocked = _synthetic_blocked_control()
    with pytest.raises(ValueError, match="primary or diagnostic"):
        dataclasses.replace(blocked, family="other").to_config()
    with pytest.raises(ValueError, match="name must belong"):
        dataclasses.replace(blocked, name="").to_config()
    with pytest.raises(ValueError, match="reason must be non-empty"):
        dataclasses.replace(blocked, reason="").to_config()
