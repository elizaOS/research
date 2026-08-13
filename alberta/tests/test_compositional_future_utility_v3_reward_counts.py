"""Exact, authority-free reward-count projection for the bound v3 source."""

from __future__ import annotations

import dataclasses
import inspect
import json
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import CompositionalFeatureLearner
from alberta_framework.evaluation import (
    _compositional_future_utility_calibration_engine as engine,
)
from alberta_framework.evaluation import (
    _compositional_future_utility_v3_reward_counts as reward_counts,
)
from alberta_framework.evaluation import compositional_control_life_development as control
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_protocol as protocol,
)
from alberta_framework.evaluation import (
    compositional_future_utility_calibration_v3_source as source_binding,
)


@pytest.fixture(scope="module")
def bound_source() -> source_binding.BoundV3Source:
    return source_binding.build_bound_v3_source()


@pytest.fixture(scope="module")
def valid_events(bound_source: source_binding.BoundV3Source) -> SimpleNamespace:
    """Make a cheap exact trace without executing the 8,998-step learner."""

    source = bound_source.source
    observations = np.asarray(source.observations)
    phases = np.asarray(source.phase_indices)
    exploration = np.asarray(source.exploration_mask)
    random_actions = np.asarray(source.random_actions)
    steps = protocol.TOTAL_STEPS

    full_q = np.zeros((steps, protocol.ACTION_HEADS), dtype=np.float32)
    raw_q = np.zeros_like(full_q)
    greedy_action = np.zeros((steps,), dtype=np.int32)
    action = np.where(exploration, random_actions, greedy_action).astype(np.int32)
    target_value = np.empty((steps,), dtype=np.float32)
    for phase, indices in enumerate(protocol.PHASE_TARGET_RAW_INDICES):
        mask = phases == phase
        target_value[mask] = np.prod(
            observations[mask][:, indices],
            axis=1,
            dtype=np.float32,
        )
    multipliers = np.asarray((-1.0, 1.0), dtype=np.float32)
    executed_reward = multipliers[action] * target_value
    greedy_reward = multipliers[greedy_action] * target_value

    return SimpleNamespace(
        action=action,
        greedy_action=greedy_action,
        explored=exploration.copy(),
        target_value=target_value,
        executed_reward=executed_reward,
        greedy_reward=greedy_reward,
        executed_regret=np.float32(1.0) - executed_reward,
        greedy_regret=np.float32(1.0) - greedy_reward,
        full_q=full_q,
        raw_q=raw_q,
        behavior_q=full_q.copy(),
    )


def _manual_record(events: object, start: int, stop: int) -> tuple[int, ...]:
    scan = cast(Any, events)
    executed_reward = np.asarray(scan.executed_reward)[start:stop]
    greedy_reward = np.asarray(scan.greedy_reward)[start:stop]
    action = np.asarray(scan.action)[start:stop]
    greedy_action = np.asarray(scan.greedy_action)[start:stop]
    explored = np.asarray(scan.explored)[start:stop]
    steps = stop - start
    return (
        steps,
        int(np.count_nonzero(executed_reward == 1.0))
        - int(np.count_nonzero(executed_reward == -1.0)),
        int(np.count_nonzero(greedy_reward == 1.0))
        - int(np.count_nonzero(greedy_reward == -1.0)),
        int(np.count_nonzero(action == 1)),
        int(np.count_nonzero(greedy_action == 1)),
        int(np.count_nonzero(explored)),
    )


def _record_tuple(record: reward_counts.ExactRewardCountRecord) -> tuple[int, ...]:
    return tuple(getattr(record, field.name) for field in dataclasses.fields(record))


def _assert_no_floats(value: object) -> None:
    assert not isinstance(value, (float, np.floating))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            _assert_no_floats(getattr(value, field.name))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_floats(item)


def _assert_exact_json_native(value: object) -> None:
    assert not isinstance(value, (tuple, float, np.floating))
    if type(value) is dict:
        for key, item in cast(dict[object, object], value).items():
            assert type(key) is str
            _assert_exact_json_native(item)
    elif type(value) is list:
        for item in cast(list[object], value):
            _assert_exact_json_native(item)
    else:
        assert type(value) in {str, int, bool, type(None)}


def test_record_is_exactly_six_integers_with_only_derived_redundancies() -> None:
    record = reward_counts.ExactRewardCountRecord(
        steps=4,
        executed_reward_sum=0,
        greedy_reward_sum=2,
        executed_action_one_count=1,
        greedy_action_one_count=3,
        explored_count=2,
    )

    assert tuple(field.name for field in dataclasses.fields(record)) == (
        "steps",
        "executed_reward_sum",
        "greedy_reward_sum",
        "executed_action_one_count",
        "greedy_action_one_count",
        "explored_count",
    )
    assert all(type(value) is int for value in _record_tuple(record))
    assert record.executed_positive_reward_count == 2
    assert record.executed_negative_reward_count == 2
    assert record.greedy_positive_reward_count == 3
    assert record.greedy_negative_reward_count == 1
    assert record.executed_action_zero_count == 3
    assert record.greedy_action_zero_count == 1
    assert record.non_explored_count == 2


def test_record_to_config_has_only_the_exact_six_stored_integer_fields() -> None:
    record = reward_counts.ExactRewardCountRecord(
        steps=4,
        executed_reward_sum=0,
        greedy_reward_sum=2,
        executed_action_one_count=1,
        greedy_action_one_count=3,
        explored_count=2,
    )

    assert record.to_config() == {
        "steps": 4,
        "executed_reward_sum": 0,
        "greedy_reward_sum": 2,
        "executed_action_one_count": 1,
        "greedy_action_one_count": 3,
        "explored_count": 2,
    }
    assert not {
        "executed_positive_reward_count",
        "executed_negative_reward_count",
        "greedy_positive_reward_count",
        "greedy_negative_reward_count",
        "executed_action_zero_count",
        "greedy_action_zero_count",
        "non_explored_count",
    } & set(record.to_config())
    _assert_exact_json_native(record.to_config())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("steps", True, "exact integer"),
        ("steps", 0, "positive"),
        ("executed_reward_sum", 5, "range"),
        ("executed_reward_sum", 1, "parity"),
        ("greedy_reward_sum", -5, "range"),
        ("greedy_reward_sum", -1, "parity"),
        ("executed_action_one_count", 5, "bounds"),
        ("greedy_action_one_count", -1, "bounds"),
        ("explored_count", 5, "bounds"),
    ),
)
def test_record_rejects_type_range_parity_action_and_exploration_mutations(
    field: str,
    value: object,
    match: str,
) -> None:
    record = reward_counts.ExactRewardCountRecord(
        steps=4,
        executed_reward_sum=0,
        greedy_reward_sum=0,
        executed_action_one_count=2,
        greedy_action_one_count=2,
        explored_count=2,
    )

    with pytest.raises((TypeError, ValueError), match=match):
        dataclasses.replace(record, **cast(Any, {field: value}))


def test_public_v3_wrapper_owns_source_arrays_and_all_task_semantics() -> None:
    signature = inspect.signature(reward_counts.project_v3_exact_reward_counts)

    assert tuple(signature.parameters) == ("bound_source", "events")
    assert {
        "observations",
        "phase_indices",
        "exploration_mask",
        "random_actions",
        "phase_target_raw_indices",
        "action_reward_multipliers",
        "composed_readout_enabled",
    }.isdisjoint(signature.parameters)
    assert reward_counts.DEVELOPMENT_ONLY is True
    assert reward_counts.EXECUTION_AUTHORIZED is False
    assert reward_counts.PANEL_EXECUTION_AUTHORIZED is False
    assert reward_counts.ROOT_ISSUANCE_AUTHORIZED is False
    assert reward_counts.OUTPUT_WRITES_ALLOWED is False
    assert reward_counts.EVIDENCE_AUTHORIZED is False
    assert reward_counts.SCIENTIFIC_PROMOTION_ALLOWED is False


def test_public_v3_wrapper_passes_only_bound_arrays_and_fixed_semantics(
    monkeypatch: pytest.MonkeyPatch,
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    original = engine.validate_future_utility_experience_semantics
    captured: dict[str, object] = {}

    def capture(
        geometry: engine.FutureUtilityEndpointGeometry,
        events: object,
        **kwargs: object,
    ) -> dict[str, object]:
        captured["geometry"] = geometry
        captured["events"] = events
        captured.update(kwargs)
        return original(geometry, events, **cast(Any, kwargs))

    monkeypatch.setattr(
        engine,
        "validate_future_utility_experience_semantics",
        capture,
    )

    reward_counts.project_v3_exact_reward_counts(bound_source, valid_events)

    geometry = cast(engine.FutureUtilityEndpointGeometry, captured["geometry"])
    assert geometry.phase_order == protocol.PHASE_ORDER
    assert geometry.phase_lengths == protocol.PHASE_LENGTHS
    assert captured["events"] is not valid_events
    assert np.array_equal(
        cast(Any, captured["events"]).executed_reward,
        valid_events.executed_reward,
    )
    assert np.array_equal(
        cast(Any, captured["observations"]),
        bound_source.source.observations,
    )
    assert np.array_equal(
        cast(Any, captured["phase_indices"]),
        bound_source.source.phase_indices,
    )
    assert np.array_equal(
        cast(Any, captured["exploration_mask"]),
        bound_source.source.exploration_mask,
    )
    assert np.array_equal(
        cast(Any, captured["random_actions"]),
        bound_source.source.random_actions,
    )
    assert captured["phase_target_raw_indices"] == protocol.PHASE_TARGET_RAW_INDICES
    assert captured["action_reward_multipliers"] == (-1.0, 1.0)
    assert captured["composed_readout_enabled"] is True


def test_public_v3_projection_has_exact_lifetime_phase_and_window_counts(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    projection = reward_counts.project_v3_exact_reward_counts(
        bound_source,
        valid_events,
    )

    assert projection.phase_order == protocol.PHASE_ORDER
    assert _record_tuple(projection.lifetime) == _manual_record(
        valid_events,
        0,
        protocol.TOTAL_STEPS,
    )
    assert len(projection.whole_phases) == 10
    assert len(projection.entry_windows) == 10
    assert len(projection.tail_windows) == 10
    assert tuple(record.steps for record in projection.whole_phases) == (
        protocol.PHASE_LENGTHS
    )
    assert {record.steps for record in projection.entry_windows} == {64}
    assert {record.steps for record in projection.tail_windows} == {64}

    for index, (start, stop) in enumerate(
        zip(
            protocol.PHASE_BOUNDARIES[:-1],
            protocol.PHASE_BOUNDARIES[1:],
            strict=True,
        )
    ):
        assert _record_tuple(projection.whole_phases[index]) == _manual_record(
            valid_events,
            start,
            stop,
        )
        assert _record_tuple(projection.entry_windows[index]) == _manual_record(
            valid_events,
            start,
            start + 64,
        )
        assert _record_tuple(projection.tail_windows[index]) == _manual_record(
            valid_events,
            stop - 64,
            stop,
        )

    for field in dataclasses.fields(reward_counts.ExactRewardCountRecord):
        assert getattr(projection.lifetime, field.name) == sum(
            getattr(record, field.name) for record in projection.whole_phases
        )
    assert projection.experience_semantics_validated is True
    assert projection.development_only is True
    assert projection.execution_authorized is False
    assert projection.output_writes_allowed is False
    assert projection.evidence_authorized is False
    assert projection.scientific_promotion_allowed is False
    _assert_no_floats(projection)


def test_projection_to_config_is_exact_json_native_and_roundtrips_losslessly(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    projection = reward_counts.project_v3_exact_reward_counts(
        bound_source,
        valid_events,
    )

    payload = projection.to_config()
    assert reward_counts.REWARD_COUNT_SCHEMA == (
        "alberta.compositional-future-utility-calibration-v3-cadence-separated."
        "exact-reward-counts.v1"
    )
    assert "REWARD_COUNT_SCHEMA" in reward_counts.__all__
    assert tuple(payload) == (
        "schema",
        "phase_order",
        "lifetime",
        "whole_phases",
        "entry_windows",
        "tail_windows",
        "experience_semantics_validated",
        "development_only",
        "execution_authorized",
        "output_writes_allowed",
        "evidence_authorized",
        "scientific_promotion_allowed",
    )
    assert payload["schema"] == reward_counts.REWARD_COUNT_SCHEMA
    assert payload["phase_order"] == list(protocol.PHASE_ORDER)
    assert payload["lifetime"] == projection.lifetime.to_config()
    assert payload["whole_phases"] == [
        record.to_config() for record in projection.whole_phases
    ]
    assert payload["entry_windows"] == [
        record.to_config() for record in projection.entry_windows
    ]
    assert payload["tail_windows"] == [
        record.to_config() for record in projection.tail_windows
    ]
    _assert_exact_json_native(payload)
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_projection_to_config_returns_fresh_lists_and_dicts(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    projection = reward_counts.project_v3_exact_reward_counts(
        bound_source,
        valid_events,
    )
    expected = projection.to_config()
    mutated = projection.to_config()
    cast(list[str], mutated["phase_order"])[0] = "mutated"
    cast(dict[str, object], mutated["lifetime"])["steps"] = 1
    cast(list[dict[str, object]], mutated["whole_phases"])[0]["steps"] = 1

    assert projection.to_config() == expected


def test_projection_rejects_a_lifetime_not_equal_to_the_whole_phase_sum(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    projection = reward_counts.project_v3_exact_reward_counts(
        bound_source,
        valid_events,
    )
    lifetime = projection.lifetime
    changed_count = (
        lifetime.executed_action_one_count - 1
        if lifetime.executed_action_one_count
        else 1
    )
    altered_lifetime = dataclasses.replace(
        lifetime,
        executed_action_one_count=changed_count,
    )

    with pytest.raises(ValueError, match="lifetime.*whole phases"):
        dataclasses.replace(projection, lifetime=altered_lifetime)


def test_validator_failure_precedes_any_reward_count_projection(
    monkeypatch: pytest.MonkeyPatch,
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    class ValidationStoppedError(RuntimeError):
        pass

    def stop_validation(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise ValidationStoppedError("mandatory validator stopped projection")

    def forbidden_projection(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("unvalidated events reached projection")

    monkeypatch.setattr(
        engine,
        "validate_future_utility_experience_semantics",
        stop_validation,
    )
    monkeypatch.setattr(
        reward_counts,
        "_project_exact_reward_count_records",
        forbidden_projection,
    )

    with pytest.raises(ValidationStoppedError, match="mandatory validator"):
        reward_counts.project_v3_exact_reward_counts(bound_source, valid_events)


@pytest.mark.parametrize(
    "field",
    (
        "executed_reward",
        "greedy_reward",
        "action",
        "greedy_action",
        "explored",
        "behavior_q",
    ),
)
def test_public_projection_rejects_mutated_experience_before_counting(
    field: str,
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    payload = dict(vars(valid_events))
    altered = np.array(payload[field], copy=True)
    if field in {"executed_reward", "greedy_reward"}:
        altered[0] = -altered[0]
    elif field in {"action", "greedy_action"}:
        altered[0] = 1 - altered[0]
    elif field == "explored":
        altered[0] = not altered[0]
    else:
        altered[0, 0] = np.float32(1.0)
    payload[field] = altered

    with pytest.raises(RuntimeError):
        reward_counts.project_v3_exact_reward_counts(
            bound_source,
            SimpleNamespace(**payload),
        )


def test_public_projection_rejects_an_unbound_source(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    with pytest.raises(TypeError, match="BoundV3Source"):
        reward_counts.project_v3_exact_reward_counts(
            cast(Any, bound_source.source),
            valid_events,
        )


@pytest.mark.integration
def test_short_production_scan_projects_exact_64_step_windows() -> None:
    short_protocol = control.CompositionalControlLifeProtocol(
        phase_lengths=(64, 65, 66, 67, 68, 69, 70, 71, 72, 73),
        epsilon=0.1,
        entry_window=64,
        tail_window=64,
    )
    source = control.build_bound_compositional_control_life_source(
        short_protocol,
        observation_key=jr.key(201),
        exploration_key=jr.key(202),
        random_action_key=jr.key(203),
        learner_key=jr.key(204),
    )
    learner = CompositionalFeatureLearner.from_config(
        control.learner_config_for_arm(protocol.LEFT_PACK_SOURCE_ARM)
    )
    execution = control.execute_compositional_control_life_arm(
        short_protocol,
        learner,
        source.learner_key,
        source.observations,
        source.phase_indices,
        source.exploration_mask,
        source.random_actions,
        composed_readout_enabled=True,
    )
    geometry = engine.FutureUtilityEndpointGeometry(
        phase_order=protocol.PHASE_ORDER,
        phase_lengths=short_protocol.phase_lengths,
        target_names=protocol.TARGET_NAMES,
        curation_interval=protocol.CURATION_INTERVAL,
    )

    projection = reward_counts._validated_exact_reward_count_projection(
        geometry,
        source,
        execution.events,
        phase_target_raw_indices=protocol.PHASE_TARGET_RAW_INDICES,
        action_reward_multipliers=(-1.0, 1.0),
        composed_readout_enabled=True,
        entry_window=64,
        tail_window=64,
    )

    assert projection.lifetime.steps == 685
    assert tuple(record.steps for record in projection.whole_phases) == (
        short_protocol.phase_lengths
    )
    assert all(record.steps == 64 for record in projection.entry_windows)
    assert all(record.steps == 64 for record in projection.tail_windows)
    assert _record_tuple(projection.lifetime) == _manual_record(
        execution.events,
        0,
        short_protocol.total_steps,
    )
    assert all(
        getattr(projection.lifetime, field.name)
        == sum(getattr(record, field.name) for record in projection.whole_phases)
        for field in dataclasses.fields(reward_counts.ExactRewardCountRecord)
    )
    _assert_no_floats(projection)


def test_jax_event_arrays_are_projected_without_float_records(
    bound_source: source_binding.BoundV3Source,
    valid_events: SimpleNamespace,
) -> None:
    jax_events = SimpleNamespace(
        **{name: jnp.asarray(value) for name, value in vars(valid_events).items()}
    )

    projection = reward_counts.project_v3_exact_reward_counts(
        bound_source,
        jax_events,
    )

    assert projection.lifetime.steps == protocol.TOTAL_STEPS
    _assert_no_floats(projection)
