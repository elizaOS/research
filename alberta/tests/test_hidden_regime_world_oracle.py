"""Independent host-oracle tests for the hidden-regime world."""

import copy
import dataclasses
import json

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_regime_world_oracle import (
    HIDDEN_REGIME_WORLD_ORACLE_SCHEMA,
    PRE_DELIVERED,
    HiddenRegimeWorldOracleInputError,
    HiddenRegimeWorldStateSnapshot,
    HiddenRegimeWorldTransitionRecord,
    reconstruct_hidden_regime_world_transition,
    validate_hidden_regime_world_state_continuity,
    validate_hidden_regime_world_state_snapshot,
    validate_hidden_regime_world_transition,
)
from alberta_framework.streams.hidden_regime_signaling import (
    CONSTANT_ONE_TERNARY_CHANNEL,
    CONSTANT_TWO_TERNARY_CHANNEL,
    CONSTANT_ZERO_TERNARY_CHANNEL,
    DIRECT_TERNARY_CHANNEL,
    SHUFFLED_TERNARY_CHANNEL,
    HiddenRegimeSignalingWorld,
    HiddenRegimeTransition,
    HiddenRegimeWorldConfig,
    HiddenRegimeWorldState,
    hidden_regime_world_keys,
)

pytestmark = pytest.mark.unit


def _tiny_config(*, repeat_schedule: bool = False) -> HiddenRegimeWorldConfig:
    return HiddenRegimeWorldConfig(
        segment_lengths=(2, 1, 2),
        segment_regimes=(0, 1, 0),
        regime_permutations=((0, 1, 2), (2, 0, 1)),
        repeat_schedule=repeat_schedule,
    )


def _initial_state(config: HiddenRegimeWorldConfig) -> HiddenRegimeWorldState:
    return HiddenRegimeSignalingWorld(config).init(hidden_regime_world_keys(jr.key(23)))


def _record(
    config: HiddenRegimeWorldConfig,
    old_state: HiddenRegimeWorldState,
    delivery_contract: str,
    transition: HiddenRegimeTransition,
    new_state: HiddenRegimeWorldState,
) -> HiddenRegimeWorldTransitionRecord:
    return HiddenRegimeWorldTransitionRecord.from_runtime(
        config=config,
        old_state=old_state,
        delivery_contract=delivery_contract,  # type: ignore[arg-type]
        transition=transition,
        new_state=new_state,
    )


def _pre_delivered_transition(
    config: HiddenRegimeWorldConfig,
    old_state: HiddenRegimeWorldState,
    *,
    helper_message: int = 0,
    delivered_message: int = 2,
    beneficiary_action: int = 1,
) -> tuple[HiddenRegimeWorldTransitionRecord, HiddenRegimeWorldState]:
    world = HiddenRegimeSignalingWorld(config)
    transition, new_state = world.step_with_delivery(
        old_state,
        jnp.int32(helper_message),
        jnp.int32(delivered_message),
        jnp.int32(beneficiary_action),
    )
    return _record(config, old_state, PRE_DELIVERED, transition, new_state), new_state


def test_pre_delivered_contract_preserves_explicit_symbol_but_direct_is_strict() -> None:
    config = _tiny_config()
    old = _initial_state(config)
    record, _ = _pre_delivered_transition(
        config,
        old,
        helper_message=0,
        delivered_message=2,
    )

    validation = validate_hidden_regime_world_transition(record)
    assert validation.valid, validation.mismatches
    assert validation.expected == reconstruct_hidden_regime_world_transition(record)
    assert record.helper_message == 0
    assert record.delivered_message == 2

    falsely_direct = dataclasses.replace(
        record,
        delivery_contract=DIRECT_TERNARY_CHANNEL,
    )
    direct_validation = validate_hidden_regime_world_transition(falsely_direct)
    assert not direct_validation.valid
    assert direct_validation.mismatches == ("delivered_message",)


@pytest.mark.parametrize(
    "channel",
    (
        DIRECT_TERNARY_CHANNEL,
        CONSTANT_ZERO_TERNARY_CHANNEL,
        CONSTANT_ONE_TERNARY_CHANNEL,
        CONSTANT_TWO_TERNARY_CHANNEL,
        SHUFFLED_TERNARY_CHANNEL,
    ),
)
def test_named_channel_contracts_match_runtime_and_advance_both_keys(
    channel: str,
) -> None:
    config = _tiny_config()
    world = HiddenRegimeSignalingWorld(config)
    old = _initial_state(config)
    transition, new_state = world.step(
        old,
        jnp.int32(1),
        jnp.int32(0),
        channel=channel,  # type: ignore[arg-type]
    )
    record = _record(config, old, channel, transition, new_state)
    validation = validate_hidden_regime_world_transition(record)

    assert validation.valid, validation.mismatches
    assert record.new_state.cue_key_data != record.old_state.cue_key_data
    assert record.new_state.channel_key_data != record.old_state.channel_key_data
    assert validation.expected is not None
    assert record.delivered_message == validation.expected.delivered_message


def test_channel_key_advances_identically_even_when_channel_does_not_draw() -> None:
    config = _tiny_config()
    world = HiddenRegimeSignalingWorld(config)
    old = _initial_state(config)
    post_keys: set[tuple[int, ...]] = set()
    post_cues: set[tuple[tuple[int, ...], int]] = set()
    for channel in (
        DIRECT_TERNARY_CHANNEL,
        CONSTANT_ZERO_TERNARY_CHANNEL,
        SHUFFLED_TERNARY_CHANNEL,
    ):
        transition, new_state = world.step(
            old,
            jnp.int32(2),
            jnp.int32(1),
            channel=channel,
        )
        record = _record(config, old, channel, transition, new_state)
        assert validate_hidden_regime_world_transition(record).valid
        post_keys.add(record.new_state.channel_key_data)
        post_cues.add((record.new_state.cue_key_data, record.new_state.cue))
    assert len(post_keys) == 1
    assert len(post_cues) == 1


@pytest.mark.parametrize(
    ("repeat_schedule", "expected_position"),
    ((False, 4), (True, 0)),
)
def test_oracle_reconstructs_hold_and_repeat_cursors(
    repeat_schedule: bool,
    expected_position: int,
) -> None:
    config = _tiny_config(repeat_schedule=repeat_schedule)
    old = dataclasses.replace(
        _initial_state(config),
        schedule_position=jnp.asarray(4, dtype=jnp.int32),
        step_count=jnp.asarray(19, dtype=jnp.int32),
        cue=jnp.asarray(1, dtype=jnp.int32),
    )
    record, _ = _pre_delivered_transition(config, old, beneficiary_action=1)
    validation = validate_hidden_regime_world_transition(record)

    assert validation.valid, validation.mismatches
    assert record.diagnostics.oracle_segment_index == 2
    assert record.diagnostics.oracle_segment_step == 1
    assert record.diagnostics.oracle_regime_id == 0
    assert record.diagnostics.oracle_target == 1
    assert record.diagnostics.reward == 1.0
    assert not record.diagnostics.terminated
    assert record.diagnostics.discount == 1.0
    assert record.new_state.schedule_position == expected_position
    assert record.new_state.step_count == 20


def test_oracle_reconstructs_segment_boundary_and_saturated_global_counter() -> None:
    config = _tiny_config()
    old = dataclasses.replace(
        _initial_state(config),
        schedule_position=jnp.asarray(2, dtype=jnp.int32),
        step_count=jnp.asarray(np.iinfo(np.int32).max, dtype=jnp.int32),
        cue=jnp.asarray(2, dtype=jnp.int32),
    )
    record, _ = _pre_delivered_transition(config, old, beneficiary_action=1)
    validation = validate_hidden_regime_world_transition(record)

    assert validation.valid, validation.mismatches
    assert record.diagnostics.oracle_step_count == np.iinfo(np.int32).max
    assert record.diagnostics.oracle_segment_index == 1
    assert record.diagnostics.oracle_segment_step == 0
    assert record.diagnostics.oracle_regime_id == 1
    assert record.diagnostics.oracle_target == 1
    assert record.new_state.step_count == np.iinfo(np.int32).max
    assert record.new_state.schedule_position == 3


def test_strict_json_round_trip_and_schedule_permutation_binding() -> None:
    config = _tiny_config()
    old = dataclasses.replace(
        _initial_state(config),
        schedule_position=jnp.asarray(2, dtype=jnp.int32),
        cue=jnp.asarray(2, dtype=jnp.int32),
    )
    record, _ = _pre_delivered_transition(config, old, beneficiary_action=1)
    payload = json.loads(json.dumps(record.to_dict(), allow_nan=False))
    restored = HiddenRegimeWorldTransitionRecord.from_dict(payload)
    assert restored.to_dict() == payload
    assert validate_hidden_regime_world_transition(restored).valid

    extra = copy.deepcopy(payload)
    extra["extra"] = True
    with pytest.raises(HiddenRegimeWorldOracleInputError, match="noncanonical keys"):
        HiddenRegimeWorldTransitionRecord.from_dict(extra)
    wrong_schema = copy.deepcopy(payload)
    wrong_schema["schema"] = HIDDEN_REGIME_WORLD_ORACLE_SCHEMA + ".future"
    with pytest.raises(HiddenRegimeWorldOracleInputError, match="not supported"):
        HiddenRegimeWorldTransitionRecord.from_dict(wrong_schema)
    invalid_permutation = copy.deepcopy(payload)
    invalid_permutation["config"]["regime_permutations"][1] = [0, 0, 2]
    with pytest.raises(HiddenRegimeWorldOracleInputError, match="exactly 0, 1, 2"):
        HiddenRegimeWorldTransitionRecord.from_dict(invalid_permutation)

    changed_active_permutation = copy.deepcopy(payload)
    changed_active_permutation["config"]["regime_permutations"][1] = [1, 2, 0]
    changed = HiddenRegimeWorldTransitionRecord.from_dict(changed_active_permutation)
    validation = validate_hidden_regime_world_transition(changed)
    assert not validation.valid
    assert "diagnostics.oracle_target" in validation.mismatches
    assert "diagnostics.reward" in validation.mismatches


def _tamper_state(
    state: HiddenRegimeWorldStateSnapshot,
    field: str,
) -> HiddenRegimeWorldStateSnapshot:
    if field == "cue_key_data":
        value: object = (state.cue_key_data[0] ^ 1, state.cue_key_data[1])
    elif field == "channel_key_data":
        value = (state.channel_key_data[0] ^ 1, state.channel_key_data[1])
    elif field == "cue":
        value = (state.cue + 1) % 3
    elif field == "step_count":
        value = state.step_count + 1
    elif field == "schedule_position":
        value = state.schedule_position + 1
    else:  # pragma: no cover - test-helper contract
        raise AssertionError(field)
    return dataclasses.replace(state, **{field: value})


@pytest.mark.parametrize(
    "field",
    ("cue_key_data", "channel_key_data", "cue", "step_count", "schedule_position"),
)
def test_every_world_state_field_rejects_single_field_tampering(field: str) -> None:
    config = _tiny_config()
    record, _ = _pre_delivered_transition(config, _initial_state(config))
    tampered = dataclasses.replace(record, new_state=_tamper_state(record.new_state, field))
    validation = validate_hidden_regime_world_transition(tampered)
    assert not validation.valid
    assert validation.mismatches


@pytest.mark.parametrize(
    "field",
    (
        "observation_helper_cue",
        "reward",
        "next_observation_helper_cue",
        "terminated",
        "discount",
        "oracle_step_count",
        "oracle_segment_index",
        "oracle_segment_step",
        "oracle_regime_id",
        "oracle_target",
    ),
)
def test_every_diagnostic_rejects_single_field_tampering(field: str) -> None:
    config = _tiny_config()
    record, _ = _pre_delivered_transition(config, _initial_state(config))
    original = getattr(record.diagnostics, field)
    if type(original) is bool:
        replacement: object = not original
    elif type(original) is int:
        replacement = original + 1
    else:
        replacement = float(np.float32(original) + np.float32(0.25))
    diagnostics = dataclasses.replace(record.diagnostics, **{field: replacement})
    tampered = dataclasses.replace(record, diagnostics=diagnostics)
    validation = validate_hidden_regime_world_transition(tampered)
    assert not validation.valid
    assert f"diagnostics.{field}" in validation.mismatches


def test_action_and_direct_message_tampering_changes_reconstructed_transition() -> None:
    config = _tiny_config()
    world = HiddenRegimeSignalingWorld(config)
    old = dataclasses.replace(_initial_state(config), cue=jnp.asarray(0, dtype=jnp.int32))
    transition, new_state = world.step(
        old,
        jnp.int32(1),
        jnp.int32(0),
        channel=DIRECT_TERNARY_CHANNEL,
    )
    record = _record(config, old, DIRECT_TERNARY_CHANNEL, transition, new_state)
    assert validate_hidden_regime_world_transition(record).valid

    changed_helper = dataclasses.replace(record, helper_message=2)
    helper_validation = validate_hidden_regime_world_transition(changed_helper)
    assert not helper_validation.valid
    assert helper_validation.mismatches == ("delivered_message",)
    changed_action = dataclasses.replace(record, beneficiary_action=1)
    action_validation = validate_hidden_regime_world_transition(changed_action)
    assert not action_validation.valid
    assert "diagnostics.reward" in action_validation.mismatches


def test_state_validation_and_bit_exact_continuity_are_standalone() -> None:
    config = _tiny_config()
    first, next_state = _pre_delivered_transition(config, _initial_state(config))
    second, _ = _pre_delivered_transition(config, next_state)

    assert validate_hidden_regime_world_state_snapshot(first.old_state, first.config).valid
    assert validate_hidden_regime_world_state_snapshot(first.new_state, first.config).valid
    assert validate_hidden_regime_world_state_continuity(
        first.new_state, second.old_state
    ).valid

    changed = dataclasses.replace(
        second.old_state,
        channel_key_data=(
            second.old_state.channel_key_data[0] ^ 1,
            second.old_state.channel_key_data[1],
        ),
    )
    continuity = validate_hidden_regime_world_state_continuity(first.new_state, changed)
    assert not continuity.valid
    assert continuity.mismatches == ("continuity.channel_key_data[0]",)
    outside = dataclasses.replace(first.old_state, schedule_position=99)
    assert not validate_hidden_regime_world_state_snapshot(outside, first.config).valid


def test_oracle_matches_jit_produced_runtime_transition() -> None:
    config = _tiny_config(repeat_schedule=True)
    world = HiddenRegimeSignalingWorld(config)
    old = dataclasses.replace(
        _initial_state(config),
        schedule_position=jnp.asarray(4, dtype=jnp.int32),
    )

    @jax.jit
    def compiled_step(
        state: HiddenRegimeWorldState,
    ) -> tuple[HiddenRegimeTransition, HiddenRegimeWorldState]:
        return world.step_with_delivery(
            state,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    transition, new_state = compiled_step(old)
    record = _record(config, old, PRE_DELIVERED, transition, new_state)
    validation = validate_hidden_regime_world_transition(record)
    assert validation.valid, validation.mismatches
    assert record.new_state.schedule_position == 0
