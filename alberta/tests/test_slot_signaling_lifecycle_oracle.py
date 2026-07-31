"""Independent host-oracle tests for slot-signaling lifecycle transitions."""

import copy
import dataclasses
import json

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.slot_signaling_agent import (
    DURABLE_WRITE_SELECTIVE,
    DURABLE_WRITE_WRITABLE,
    REPLACEMENT_TARGET_EVIDENCE,
    REPLACEMENT_TARGET_LRU,
    SLOT_DURABLE,
    SLOT_SCRATCH,
    SLOT_VACANT,
    SlotRoleState,
    SlotRoleUpdate,
    SlotSignalingAgent,
    SlotSignalingConfig,
    slot_signaling_keys,
)
from alberta_framework.evaluation.slot_signaling_lifecycle_oracle import (
    INPUT_DELIVERED_MESSAGE,
    INPUT_PRIVATE_CUE,
    ROLE_BENEFICIARY,
    ROLE_HELPER,
    SlotLifecycleOracleInputError,
    SlotRoleStateSnapshot,
    SlotRoleTransitionRecord,
    reconstruct_slot_role_transition,
    validate_slot_role_state_continuity,
    validate_slot_role_state_snapshot,
    validate_slot_role_transition,
)

pytestmark = pytest.mark.unit


def _initial_role(config: SlotSignalingConfig, seed: int = 71) -> SlotRoleState:
    agent = SlotSignalingAgent(config)
    return agent.init(slot_signaling_keys(jr.key(seed))).helper


def _record_actual_transition(
    config: SlotSignalingConfig,
    old: SlotRoleState,
    *,
    private_input: int = 0,
    reward: float = 1.0,
    external_value_write: bool = True,
    lifecycle_write: bool = True,
    role: str = ROLE_HELPER,
) -> tuple[SlotRoleTransitionRecord, SlotRoleUpdate]:
    agent = SlotSignalingAgent(config)
    if role == ROLE_HELPER:
        decision = agent.select_helper(old, jnp.int32(private_input))
    else:
        decision = agent.select_beneficiary(old, jnp.int32(private_input))
    update = agent.update_role(
        old,
        decision,
        jnp.float32(reward),
        value_write=external_value_write,
        lifecycle_write=lifecycle_write,
    )
    record = SlotRoleTransitionRecord.from_runtime(
        role=role,  # type: ignore[arg-type]
        config=config,
        old_state=old,
        decision=decision,
        reward=jnp.float32(reward),
        external_value_write=external_value_write,
        lifecycle_write=lifecycle_write,
        update=update,
    )
    return record, update


def _replacement_state(config: SlotSignalingConfig) -> SlotRoleState:
    old = _initial_role(config)
    return dataclasses.replace(
        old,
        values=jnp.arange(36, dtype=jnp.float32).reshape(4, 3, 3) / jnp.float32(100.0),
        status=jnp.asarray(
            (SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE),
            dtype=jnp.int32,
        ),
        generation=jnp.asarray((0, 10, 20, 30), dtype=jnp.int32),
        failed_leases=jnp.asarray((0, 5, 1, 0), dtype=jnp.int32),
        idle_leases=jnp.asarray((0, 1, 9, 0), dtype=jnp.int32),
        next_generation=jnp.asarray(31, dtype=jnp.int32),
    )


def _factorial_config(write_policy: str, replacement_policy: str) -> SlotSignalingConfig:
    return SlotSignalingConfig(
        learning_rate=0.2,
        epsilon=0.0,
        relevance_rate=0.2,
        lease_length=1,
        confirmation_steps=1,
        durable_retrieval_threshold=0.5,
        candidate_confirmation_threshold=0.75,
        candidate_confirmation_leases=1,
        scratch_training_leases_before_retest=2,
        durable_write_policy=write_policy,  # type: ignore[arg-type]
        replacement_target_policy=replacement_policy,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("write_policy", "replacement_policy", "expected_target"),
    (
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE, 1),
        (DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_LRU, 2),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_EVIDENCE, 1),
        (DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU, 2),
    ),
)
def test_oracle_matches_atomic_replacement_for_all_factorial_axes(
    write_policy: str,
    replacement_policy: str,
    expected_target: int,
) -> None:
    config = _factorial_config(write_policy, replacement_policy)
    old = _replacement_state(config)
    old_snapshot = SlotRoleStateSnapshot.from_state(old)
    record, update = _record_actual_transition(config, old, private_input=1)

    validation = validate_slot_role_transition(record)
    assert validation.valid, validation.mismatches
    assert validation.expected == reconstruct_slot_role_transition(record)
    assert record.config.durable_write_policy == write_policy
    assert record.config.replacement_target_policy == replacement_policy
    assert int(update.committed_slot) == expected_target
    assert int(update.retired_slot) == expected_target
    assert int(update.retired_generation) == old_snapshot.generation[expected_target]
    assert record.new_state.generation[expected_target] == old_snapshot.next_generation
    assert record.new_state.status[expected_target] == SLOT_DURABLE
    assert record.new_state.values[expected_target] != old_snapshot.values[expected_target]
    assert record.new_state.values[0] == tuple((0.0, 0.0, 0.0) for _ in range(3))
    assert record.new_state.relevance_mass[0] == 0.0
    assert record.new_state.candidate_successful_leases == 0
    assert record.new_state.next_generation == old_snapshot.next_generation + 1


@pytest.mark.parametrize(
    ("write_policy", "expected_write"),
    (
        (DURABLE_WRITE_SELECTIVE, False),
        (DURABLE_WRITE_WRITABLE, True),
    ),
)
def test_oracle_separates_durable_write_policy_from_replacement_policy(
    write_policy: str,
    expected_write: bool,
) -> None:
    config = SlotSignalingConfig(
        epsilon=0.0,
        lease_length=2,
        confirmation_steps=1,
        durable_write_policy=write_policy,  # type: ignore[arg-type]
        replacement_target_policy=REPLACEMENT_TARGET_EVIDENCE,
    )
    old = dataclasses.replace(
        _initial_role(config),
        status=jnp.asarray(
            (SLOT_SCRATCH, SLOT_DURABLE, SLOT_VACANT, SLOT_VACANT),
            dtype=jnp.int32,
        ),
        generation=jnp.asarray((0, 1, 0, 0), dtype=jnp.int32),
        active_slot=jnp.asarray(1, dtype=jnp.int32),
    )
    record, update = _record_actual_transition(config, old)
    validation = validate_slot_role_transition(record)

    assert validation.valid, validation.mismatches
    assert bool(update.value_write) is expected_write
    assert record.new_state.values == record.old_state.values if not expected_write else (
        record.new_state.values != record.old_state.values
    )
    assert record.new_state.relevance_mass[1] == 1.0
    assert record.new_state.lease_offset == 1
    assert record.new_state.lease_reward_sum == 1.0


def test_oracle_reconstructs_exhaustive_durable_search_and_scratch_reset() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    role = dataclasses.replace(
        _initial_role(config),
        status=jnp.asarray(
            (SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_DURABLE),
            dtype=jnp.int32,
        ),
        generation=jnp.asarray((0, 1, 2, 3), dtype=jnp.int32),
        active_slot=jnp.asarray(1, dtype=jnp.int32),
        failed_leases=jnp.asarray((7, 0, 0, 0), dtype=jnp.int32),
        search_cursor=jnp.asarray(2, dtype=jnp.int32),
    )
    expected_lifecycle = (
        (2, 2, 3, (7, 1, 0, 0)),
        (3, 1, 1, (7, 1, 1, 0)),
        (0, 0, 1, (0, 1, 1, 1)),
    )
    for expected_active, expected_remaining, expected_cursor, expected_failed in (
        expected_lifecycle
    ):
        record, update = _record_actual_transition(config, role, reward=0.0)
        validation = validate_slot_role_transition(record)
        assert validation.valid, validation.mismatches
        assert record.new_state.active_slot == expected_active
        assert record.new_state.remaining_durable_tests == expected_remaining
        assert record.new_state.search_cursor == expected_cursor
        assert record.new_state.failed_leases == expected_failed
        role = update.state


def test_oracle_reconstructs_failed_scratch_residency_and_retest() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    role = dataclasses.replace(
        _initial_role(config),
        status=jnp.asarray(
            (SLOT_SCRATCH, SLOT_DURABLE, SLOT_DURABLE, SLOT_VACANT),
            dtype=jnp.int32,
        ),
        generation=jnp.asarray((0, 1, 2, 0), dtype=jnp.int32),
    )

    first, first_update = _record_actual_transition(config, role, reward=0.0)
    second, _ = _record_actual_transition(config, first_update.state, reward=0.0)

    assert validate_slot_role_transition(first).valid
    assert first.diagnostics.scratch_failed_leases_pre == 0
    assert first.diagnostics.scratch_failed_leases_post == 1
    assert not first.diagnostics.scratch_retest_started
    assert first.new_state.active_slot == 0
    assert validate_slot_role_transition(second).valid
    assert second.diagnostics.scratch_failed_leases_pre == 1
    assert second.diagnostics.scratch_failed_leases_post == 0
    assert second.diagnostics.scratch_retest_started
    assert second.new_state.active_slot == 1
    assert second.new_state.remaining_durable_tests == 2
    assert second.new_state.search_cursor == 2


def test_oracle_reconstructs_consecutive_candidate_confirmation_gate() -> None:
    config = dataclasses.replace(
        _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE),
        candidate_confirmation_leases=2,
    )
    role = _initial_role(config)

    first, first_update = _record_actual_transition(config, role, reward=1.0)
    second, _ = _record_actual_transition(config, first_update.state, reward=1.0)

    assert validate_slot_role_transition(first).valid
    assert first.diagnostics.candidate_lease_success
    assert first.diagnostics.committed_slot == -1
    assert first.new_state.candidate_successful_leases == 1
    assert validate_slot_role_transition(second).valid
    assert second.diagnostics.candidate_lease_success
    assert second.diagnostics.committed_slot == 1
    assert second.new_state.candidate_successful_leases == 0


@pytest.mark.parametrize(
    ("old_mass", "old_mean", "expected_mass", "expected_mean"),
    (
        (4.0, 0.25, 5.0, 0.4),
        (20.0, 0.25, 21.0, 0.325),
        (16_777_216.0, 0.25, 16_777_216.0, 0.325),
    ),
)
def test_oracle_reconstructs_bias_corrected_relevance_and_mass_saturation(
    old_mass: float,
    old_mean: float,
    expected_mass: float,
    expected_mean: float,
) -> None:
    config = SlotSignalingConfig(
        epsilon=1.0,
        relevance_rate=0.1,
        lease_length=32,
        confirmation_steps=8,
        durable_write_policy=DURABLE_WRITE_SELECTIVE,
        replacement_target_policy=REPLACEMENT_TARGET_EVIDENCE,
    )
    role = dataclasses.replace(
        _initial_role(config),
        relevance_mass=jnp.asarray((old_mass, 0.0, 0.0, 0.0), dtype=jnp.float32),
        relevance_mean=jnp.asarray((old_mean, 0.0, 0.0, 0.0), dtype=jnp.float32),
    )
    record, _ = _record_actual_transition(config, role, private_input=2, reward=1.0)
    validation = validate_slot_role_transition(record)
    assert validation.valid, validation.mismatches
    assert record.new_state.relevance_mass[0] == expected_mass
    assert record.new_state.relevance_mean[0] == float(np.float32(expected_mean))
    assert record.new_state.lease_offset == 1
    assert record.new_state.lease_reward_sum == 1.0


def test_oracle_reconstructs_vacancy_fill_and_generation_exhaustion() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    vacancy = dataclasses.replace(
        _initial_role(config),
        values=jnp.arange(36, dtype=jnp.float32).reshape(4, 3, 3),
        status=jnp.asarray(
            (SLOT_SCRATCH, SLOT_DURABLE, SLOT_VACANT, SLOT_VACANT),
            dtype=jnp.int32,
        ),
        generation=jnp.asarray((0, 4, 0, 0), dtype=jnp.int32),
        next_generation=jnp.asarray(5, dtype=jnp.int32),
    )
    filled, _ = _record_actual_transition(config, vacancy)
    assert validate_slot_role_transition(filled).valid
    assert filled.diagnostics.committed_slot == 2
    assert filled.diagnostics.retired_slot == -1
    assert filled.new_state.generation == (0, 4, 5, 0)

    exhausted = dataclasses.replace(
        _replacement_state(config),
        next_generation=jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32),
    )
    blocked, _ = _record_actual_transition(config, exhausted)
    assert validate_slot_role_transition(blocked).valid
    assert blocked.diagnostics.generation_exhausted
    assert blocked.diagnostics.committed_slot == -1
    assert blocked.diagnostics.retired_slot == -1
    assert blocked.new_state.status == blocked.old_state.status
    assert blocked.new_state.generation == blocked.old_state.generation
    assert blocked.new_state.next_generation == np.iinfo(np.int32).max
    assert blocked.new_state.active_slot == 0


def test_strict_json_round_trip_and_named_role_streams() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    helper_record, _ = _record_actual_transition(config, _replacement_state(config))
    beneficiary_record, _ = _record_actual_transition(
        config,
        _replacement_state(config),
        role=ROLE_BENEFICIARY,
    )
    for record, stream in (
        (helper_record, INPUT_PRIVATE_CUE),
        (beneficiary_record, INPUT_DELIVERED_MESSAGE),
    ):
        payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
        restored = SlotRoleTransitionRecord.from_dict(payload)
        assert restored.to_dict() == payload
        assert restored.input_stream == stream
        assert validate_slot_role_transition(restored).valid

    extra = helper_record.to_dict()
    extra["unregistered"] = 1
    with pytest.raises(SlotLifecycleOracleInputError, match="noncanonical keys"):
        SlotRoleTransitionRecord.from_dict(extra)
    wrong_stream = helper_record.to_dict()
    wrong_stream["input_stream"] = INPUT_DELIVERED_MESSAGE
    with pytest.raises(SlotLifecycleOracleInputError, match="requires input stream"):
        SlotRoleTransitionRecord.from_dict(wrong_stream)
    wrong_boolean = helper_record.to_dict()
    wrong_boolean["lifecycle_write"] = 1
    with pytest.raises(SlotLifecycleOracleInputError, match="must be boolean"):
        SlotRoleTransitionRecord.from_dict(wrong_boolean)


def _bump_float(value: float) -> float:
    return float(np.float32(value) + np.float32(0.125))


def _tampered_state(snapshot: SlotRoleStateSnapshot, field: str) -> SlotRoleStateSnapshot:
    if field == "values":
        values = [[list(row) for row in matrix] for matrix in snapshot.values]
        values[0][0][0] = _bump_float(values[0][0][0])
        replacement = tuple(
            tuple(tuple(row) for row in matrix) for matrix in values
        )
    elif field in {"relevance_mean", "relevance_mass"}:
        items = list(getattr(snapshot, field))
        items[0] = items[0] + (1.0 if field == "relevance_mass" else 0.125)
        replacement = tuple(items)
    elif field in {"failed_leases", "idle_leases", "generation"}:
        items = list(getattr(snapshot, field))
        items[1] += 1
        replacement = tuple(items)
    elif field == "status":
        items = list(snapshot.status)
        items[3] = SLOT_VACANT
        replacement = tuple(items)
    elif field == "active_slot":
        replacement = 3 if snapshot.active_slot != 3 else 2
    elif field == "lease_offset":
        replacement = snapshot.lease_offset + 1
    elif field == "lease_reward_sum":
        replacement = _bump_float(snapshot.lease_reward_sum)
    elif field == "remaining_durable_tests":
        replacement = snapshot.remaining_durable_tests + 1
    elif field == "search_cursor":
        replacement = (snapshot.search_cursor % 3) + 1
    elif field == "candidate_successful_leases":
        replacement = snapshot.candidate_successful_leases + 1
    elif field == "next_generation":
        replacement = snapshot.next_generation + 1
    elif field == "key_data":
        replacement = (snapshot.key_data[0] ^ 1, snapshot.key_data[1])
    else:  # pragma: no cover - test helper contract
        raise AssertionError(field)
    return dataclasses.replace(snapshot, **{field: replacement})


@pytest.mark.parametrize(
    "field",
    (
        "values",
        "relevance_mean",
        "relevance_mass",
        "failed_leases",
        "idle_leases",
        "status",
        "generation",
        "active_slot",
        "lease_offset",
        "lease_reward_sum",
        "remaining_durable_tests",
        "search_cursor",
        "candidate_successful_leases",
        "next_generation",
        "key_data",
    ),
)
def test_every_persistent_state_family_is_fail_closed_under_single_field_tampering(
    field: str,
) -> None:
    config = _factorial_config(DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU)
    record, _ = _record_actual_transition(config, _replacement_state(config))
    tampered = dataclasses.replace(record, new_state=_tampered_state(record.new_state, field))
    validation = validate_slot_role_transition(tampered)
    assert not validation.valid
    assert validation.mismatches


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("decision", "slot"),
        ("decision", "private_input"),
        ("decision", "action"),
        ("decision", "selected_value"),
        ("decision", "next_key_data"),
        ("diagnostics", "value_pre"),
        ("diagnostics", "candidate_value"),
        ("diagnostics", "value_post"),
        ("diagnostics", "value_write"),
        ("diagnostics", "lease_boundary"),
        ("diagnostics", "lease_reward_mean"),
        ("diagnostics", "relevance_ready"),
        ("diagnostics", "durable_relevant"),
        ("diagnostics", "candidate_relevant"),
        ("diagnostics", "candidate_lease_success"),
        ("diagnostics", "scratch_failed_leases_pre"),
        ("diagnostics", "scratch_failed_leases_post"),
        ("diagnostics", "scratch_retest_started"),
        ("diagnostics", "generation_exhausted"),
        ("diagnostics", "committed_slot"),
        ("diagnostics", "committed_generation"),
        ("diagnostics", "retired_slot"),
        ("diagnostics", "retired_generation"),
    ),
)
def test_decision_key_consumption_and_diagnostics_reject_single_field_tampering(
    section: str,
    field: str,
) -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    record, _ = _record_actual_transition(config, _replacement_state(config))
    original = getattr(record, section)
    value = getattr(original, field)
    if isinstance(value, tuple):
        replacement = (value[0] ^ 1, value[1])
    elif type(value) is bool:
        replacement = not value
    elif type(value) is int:
        replacement = value + 1
    else:
        replacement = _bump_float(value)
    tampered_section = dataclasses.replace(original, **{field: replacement})
    tampered = dataclasses.replace(record, **{section: tampered_section})
    validation = validate_slot_role_transition(tampered)
    assert not validation.valid
    assert validation.mismatches


def test_external_and_lifecycle_permits_are_independently_reconstructed() -> None:
    config = _factorial_config(DURABLE_WRITE_WRITABLE, REPLACEMENT_TARGET_LRU)
    old = _replacement_state(config)
    no_value_write, _ = _record_actual_transition(
        config,
        old,
        external_value_write=False,
        lifecycle_write=True,
    )
    no_lifecycle, _ = _record_actual_transition(
        config,
        old,
        external_value_write=True,
        lifecycle_write=False,
    )
    for record in (no_value_write, no_lifecycle):
        assert validate_slot_role_transition(record).valid
    assert not no_value_write.diagnostics.value_write
    assert no_value_write.diagnostics.committed_slot >= 1
    assert no_lifecycle.diagnostics.value_write
    assert no_lifecycle.diagnostics.committed_slot == -1
    assert no_lifecycle.new_state.candidate_successful_leases == 1


def test_serialized_old_state_tampering_cannot_be_hidden_by_rehash_or_replay() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    record, _ = _record_actual_transition(config, _replacement_state(config))
    payload = copy.deepcopy(record.to_dict())
    old_state = payload["old_state"]
    assert isinstance(old_state, dict)
    old_state["failed_leases"][1] = 0
    tampered = SlotRoleTransitionRecord.from_dict(payload)
    validation = validate_slot_role_transition(tampered)
    assert not validation.valid
    assert validation.expected is not None
    assert validation.mismatches


def test_state_snapshots_validate_directly_and_adjacent_records_require_full_continuity() -> None:
    config = _factorial_config(DURABLE_WRITE_SELECTIVE, REPLACEMENT_TARGET_EVIDENCE)
    first, first_update = _record_actual_transition(config, _replacement_state(config))
    second, _ = _record_actual_transition(config, first_update.state, reward=0.0)

    assert validate_slot_role_state_snapshot(first.old_state, first.config).valid
    assert validate_slot_role_state_snapshot(first.new_state, first.config).valid
    assert validate_slot_role_state_continuity(first.new_state, second.old_state).valid

    invalid_status = dataclasses.replace(
        first.new_state,
        status=(SLOT_VACANT, *first.new_state.status[1:]),
    )
    assert not validate_slot_role_state_snapshot(invalid_status, first.config).valid
    changed_key = dataclasses.replace(
        second.old_state,
        key_data=(second.old_state.key_data[0] ^ 1, second.old_state.key_data[1]),
    )
    continuity = validate_slot_role_state_continuity(first.new_state, changed_key)
    assert not continuity.valid
    assert continuity.mismatches == ("continuity.key_data[0]",)
