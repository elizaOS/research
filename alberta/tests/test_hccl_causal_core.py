"""Contracts for the development-only HCCL world/event-receipt rung."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import alberta_framework
import alberta_framework.streams as streams_api
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS,
    HCCL_CAUSAL_CORE_L2_PROFILE,
    HCCL_CAUSAL_CORE_L2_SCHEDULE,
    HCCL_CAUSAL_CORE_L3_PROFILE,
    HCCL_CAUSAL_CORE_L3_SCHEDULE,
    HCCL_CAUSAL_CORE_REGIME_NAMES,
    HCCL_CAUSAL_CORE_SCHEDULE,
    HCCL_CAUSAL_CORE_SMOKE_PROFILE,
    HCCL_CAUSAL_CORE_SMOKE_SCHEDULE,
    HCCL_CAUSAL_CORE_STATUS,
    HCCL_REGIME_A,
    HCCL_REGIME_B,
    HCCL_REGIME_C,
    HCCL_REGIME_D,
    HCCLCausalCoreConfig,
    HCCLCausalCoreStepResult,
    HCCLCausalCoreWorld,
    load_hccl_causal_core_checkpoint,
    measure_hccl_causal_core_state_nbytes,
    run_hccl_causal_core_scan,
    save_hccl_causal_core_checkpoint,
)

pytestmark = pytest.mark.integration


def _world() -> HCCLCausalCoreWorld:
    return HCCLCausalCoreWorld(HCCLCausalCoreConfig())


def _step(
    world: HCCLCausalCoreWorld, state: Any, actions: tuple[int, int]
) -> HCCLCausalCoreStepResult:
    receipt = world.prepare_event(state)
    return world.step(
        state,
        receipt,
        jnp.asarray(actions, dtype=jnp.int32),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )


def _materialize_keys(tree: object) -> object:
    def convert(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            dtype, jax.dtypes.prng_key
        ):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree.map(convert, tree)


def test_development_resolution_config_is_strict_public_and_non_authoritative() -> None:
    config = HCCLCausalCoreConfig()
    world = HCCLCausalCoreWorld(config)
    payload = world.to_config()

    assert payload["mechanism_status"] == HCCL_CAUSAL_CORE_STATUS
    assert payload["mechanism_status"] == "l0-development-world-receipt-only-not_assessed"
    assert payload["initial_hidden_sign_distribution"] == "fair-bernoulli-minus1-plus1"
    assert payload["prng_implementation"] == "threefry2x32"
    assert payload["world_flip_probability"] == 0.03
    assert payload["cue_flip_probabilities"] == [0.25, 0.35]
    assert payload["outcome_flip_probability"] == 0.15
    assert payload["outcome_flip_scope"] == "convention-factor-P-only"
    assert payload["nuisance_distribution"] == "standard-normal-five-channels-per-agent"
    assert payload["tv_nuisance_variance_multiplier"] == 10.0
    assert payload["partner_velocity_observation_noise_std"] == 0.01
    assert payload["partner_velocity_noise_units"] == "physical-velocity-before-normalization"
    assert payload["event_draw_counts"] == HCCL_CAUSAL_CORE_EVENT_DRAW_COUNTS
    assert payload["agent_implementation_present"] is False
    assert payload["benchmark_execution_authorized"] is False
    assert payload["artifact_authorized"] is False
    assert payload["threshold_authorized"] is False
    assert payload["evidence_authorized"] is False
    assert payload["promotion_authorized"] is False
    assert payload["output_writes_authorized"] is False
    assert payload["seed_reservation_or_consumption_authorized"] is False
    assert HCCLCausalCoreWorld.from_config(payload).to_config() == payload
    assert HCCLCausalCoreWorld.from_json(world.to_json()).to_config() == payload
    assert alberta_framework.HCCLCausalCoreWorld is HCCLCausalCoreWorld
    assert streams_api.HCCLCausalCoreConfig is HCCLCausalCoreConfig

    malformed = dict(payload)
    malformed["partner_velocity_observation_noise_std"] = 0.02
    with pytest.raises(ValueError, match="resolution|config|fixed"):
        HCCLCausalCoreWorld.from_config(malformed)

    for name, equal_but_wrong_type in (
        ("agent_implementation_present", 0),
        ("maximum_committed_transitions", 8998.0),
        ("world_limit", 1),
    ):
        malformed = dict(payload)
        malformed[name] = equal_but_wrong_type
        with pytest.raises(ValueError, match="resolution|config|fixed"):
            HCCLCausalCoreWorld.from_config(malformed)

    with pytest.raises(ValueError, match="maximum_committed_transitions"):
        HCCLCausalCoreConfig(maximum_committed_transitions=8998.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON|duplicate|strict"):
        HCCLCausalCoreWorld.from_json('{"schema":"a","schema":"a"}')
    with pytest.raises(ValueError, match="JSON|strict"):
        HCCLCausalCoreWorld.from_json('{"schema":NaN}')


def test_canonical_config_bytes_and_integrity_tags_remain_frozen() -> None:
    world = _world()
    encoded = json.dumps(
        world.to_config(),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert len(encoded) == 3418
    assert hashlib.sha256(encoded).hexdigest() == (
        "22ba12ef34ef6e0b74869758d674210322de871160ca901c518aaace0afa3a77"
    )

    state = world.init(jr.key(7))
    event = world.prepare_event(state)
    proposal = world.propose(state, event, jnp.asarray([0, 1], dtype=jnp.int32))
    assert tuple(int(word) for word in state.content_tag_words) == (
        3767168205,
        2975764106,
        3260425921,
        4083227264,
    )
    assert tuple(int(word) for word in event.content_tag_words) == (
        773270647,
        1787952051,
        1923442694,
        3689835258,
    )
    assert tuple(int(word) for word in proposal.content_tag_words) == (
        2467927846,
        33154454,
        1849048428,
        33164812,
    )


def test_canonical_420_prefix_is_not_the_true_smoke_schedule() -> None:
    canonical = _world()
    smoke = HCCLCausalCoreWorld(HCCLCausalCoreConfig.mechanics_smoke())

    assert {
        canonical.evaluator_regime_name_for_step(step) for step in range(420)
    } == {"A"}
    assert tuple(smoke.evaluator_regime_name_for_step(step) for step in range(420)) != (
        "A",
    ) * 420
    assert smoke.config.schedule_profile == HCCL_CAUSAL_CORE_SMOKE_PROFILE


def test_smoke_profile_exact_boundaries_capacity_and_authentication() -> None:
    expected = (
        ("A", 0, 33),
        ("B", 33, 68),
        ("A", 68, 105),
        ("D", 105, 144),
        ("A", 144, 185),
        ("C", 185, 228),
        ("A", 228, 273),
        ("B", 273, 320),
        ("C", 320, 369),
        ("A", 369, 420),
    )
    assert HCCL_CAUSAL_CORE_SMOKE_SCHEDULE == expected
    smoke = HCCLCausalCoreWorld(HCCLCausalCoreConfig.mechanics_smoke())
    payload = smoke.to_config()
    assert payload["schedule_profile"] == HCCL_CAUSAL_CORE_SMOKE_PROFILE
    assert payload["entry_window_steps"] == 16
    assert payload["tail_window_steps"] == 16
    assert payload["maximum_committed_transitions"] == 420
    assert payload["schedule"] == [
        {"regime": name, "start": start, "end": end} for name, start, end in expected
    ]
    assert HCCLCausalCoreWorld.from_config(payload).to_config() == payload
    for name, start, end in expected:
        assert smoke.evaluator_regime_name_for_step(start) == name
        assert smoke.evaluator_regime_name_for_step(end - 1) == name
    with pytest.raises(ValueError, match="step"):
        smoke.evaluator_regime_name_for_step(420)

    near_capacity = smoke.reseal_state(
        dataclasses.replace(  # type: ignore[type-var]
            smoke.init(jr.key(29)),
            step_words=jnp.asarray([0, 419], dtype=jnp.uint32),
            history_available=jnp.asarray(True, dtype=jnp.bool_),
            previous_action_signs=jnp.asarray([-1.0, 1.0], dtype=jnp.float32),
        )
    )
    final = _step(smoke, near_capacity, (0, 1))
    assert bool(final.update_applied)
    chex.assert_trees_all_equal(final.post_step_words, jnp.asarray([0, 420], jnp.uint32))
    rejected = _step(smoke, final.state, (0, 1))
    assert not bool(rejected.lifetime_capacity_available)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, final.state)
    assert smoke.resource_budget().maximum_committed_transitions == 420

    canonical = _world()
    canonical_state = canonical.init(jr.key(31))
    smoke_state = smoke.init(jr.key(31))
    assert not bool(canonical.state_valid(smoke_state))
    assert not bool(smoke.state_valid(canonical_state))
    assert not bool(jnp.all(canonical_state.content_tag_words == smoke_state.content_tag_words))

    malformed = dict(payload)
    malformed["entry_window_steps"] = 15
    with pytest.raises(ValueError, match="config|resolution"):
        HCCLCausalCoreWorld.from_config(malformed)


@pytest.mark.parametrize(
    ("config", "profile", "cycles", "events", "schedule"),
    (
        (
            HCCLCausalCoreConfig.core_l2(),
            HCCL_CAUSAL_CORE_L2_PROFILE,
            8,
            71_984,
            HCCL_CAUSAL_CORE_L2_SCHEDULE,
        ),
        (
            HCCLCausalCoreConfig.core_l3(),
            HCCL_CAUSAL_CORE_L3_PROFILE,
            112,
            1_007_776,
            HCCL_CAUSAL_CORE_L3_SCHEDULE,
        ),
    ),
)
def test_long_life_profiles_have_exact_continuous_coverage_and_one_d(
    config: HCCLCausalCoreConfig,
    profile: str,
    cycles: int,
    events: int,
    schedule: tuple[tuple[str, int, int], ...],
) -> None:
    world = HCCLCausalCoreWorld(config)
    payload = world.to_config()
    canonical_lengths = tuple(end - start for _name, start, end in HCCL_CAUSAL_CORE_SCHEDULE)

    assert config.schedule_profile == profile
    assert config.maximum_committed_transitions == events == 8_998 * cycles
    assert world.schedule is schedule
    assert len(schedule) == 10 * cycles
    assert schedule[0][1] == 0
    assert schedule[-1][2] == events
    assert sum(name == "D" for name, _start, _end in schedule) == 1
    assert payload["schedule_profile"] == profile
    assert payload["cycle_count"] == cycles
    assert payload["reset_callbacks_present"] is False
    assert payload["boundary_callbacks_present"] is False
    assert payload["cycle_reseeding_present"] is False
    assert HCCLCausalCoreWorld.from_config(payload).to_config() == payload
    assert HCCLCausalCoreWorld.from_json(world.to_json()).to_config() == payload

    previous_end = 0
    for index, (name, start, end) in enumerate(schedule):
        cycle_index, segment_index = divmod(index, 10)
        canonical_name = HCCL_CAUSAL_CORE_SCHEDULE[segment_index][0]
        expected_name = "A" if cycle_index > 0 and canonical_name == "D" else canonical_name
        assert (start, end - start) == (previous_end, canonical_lengths[segment_index])
        assert name == expected_name
        previous_end = end
    assert world.evaluator_regime_name_for_step(2_395) == "D"
    assert world.evaluator_regime_name_for_step(8_998 + 2_395) == "A"
    assert world.evaluator_regime_name_for_step(events - 1) == "A"
    with pytest.raises(ValueError, match="step"):
        world.evaluator_regime_name_for_step(events)


def test_every_world_profile_has_distinct_config_and_state_identity() -> None:
    worlds = (
        HCCLCausalCoreWorld(HCCLCausalCoreConfig()),
        HCCLCausalCoreWorld(HCCLCausalCoreConfig.mechanics_smoke()),
        HCCLCausalCoreWorld(HCCLCausalCoreConfig.core_l2()),
        HCCLCausalCoreWorld(HCCLCausalCoreConfig.core_l3()),
    )
    states = tuple(world.init(jr.key(31)) for world in worlds)
    profiles = tuple(world.config.schedule_profile for world in worlds)
    config_schemas = tuple(cast(str, world.to_config()["schema"]) for world in worlds)
    state_schemas = tuple(cast(str, world.to_config()["state_schema"]) for world in worlds)
    tags = tuple(tuple(int(word) for word in state.content_tag_words) for state in states)

    assert profiles == (
        HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
        HCCL_CAUSAL_CORE_SMOKE_PROFILE,
        HCCL_CAUSAL_CORE_L2_PROFILE,
        HCCL_CAUSAL_CORE_L3_PROFILE,
    )
    assert len(set(config_schemas)) == len(worlds)
    assert len(set(state_schemas)) == len(worlds)
    assert len(set(tags)) == len(worlds)
    for owner_index, owner in enumerate(worlds):
        for state_index, state in enumerate(states):
            assert bool(owner.state_valid(state)) is (owner_index == state_index)

    l2_payload = worlds[2].to_config()
    crossed = dict(l2_payload)
    crossed["schedule_profile"] = HCCL_CAUSAL_CORE_L3_PROFILE
    with pytest.raises(ValueError, match="config|resolution"):
        HCCLCausalCoreWorld.from_config(crossed)
    with pytest.raises(ValueError, match="maximum_committed_transitions"):
        HCCLCausalCoreConfig(
            maximum_committed_transitions=71_984,
            schedule_profile=HCCL_CAUSAL_CORE_L3_PROFILE,
        )


@pytest.mark.parametrize(
    "config",
    (HCCLCausalCoreConfig.core_l2(), HCCLCausalCoreConfig.core_l3()),
)
def test_long_life_last_clock_applies_once_then_rejects_without_reset(
    config: HCCLCausalCoreConfig,
) -> None:
    world = HCCLCausalCoreWorld(config)
    final_source_clock = config.maximum_committed_transitions - 1
    near_capacity = world.reseal_state(
        dataclasses.replace(  # type: ignore[type-var]
            world.init(jr.key(37)),
            step_words=jnp.asarray([0, final_source_clock], dtype=jnp.uint32),
            history_available=jnp.asarray(True, dtype=jnp.bool_),
            previous_action_signs=jnp.asarray([-1.0, 1.0], dtype=jnp.float32),
        )
    )

    final = _step(world, near_capacity, (0, 1))
    assert bool(final.update_applied)
    chex.assert_trees_all_equal(
        final.post_step_words,
        jnp.asarray([0, config.maximum_committed_transitions], dtype=jnp.uint32),
    )
    rejected = _step(world, final.state, (0, 1))
    assert not bool(rejected.lifetime_capacity_available)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, final.state)

    cycle_boundary = world.reseal_state(
        dataclasses.replace(  # type: ignore[type-var]
            world.init(jr.key(41)),
            positions=jnp.asarray([0.2, -0.3], dtype=jnp.float32),
            velocities=jnp.asarray([0.04, -0.06], dtype=jnp.float32),
            step_words=jnp.asarray([0, 8_998], dtype=jnp.uint32),
            history_available=jnp.asarray(True, dtype=jnp.bool_),
            previous_action_signs=jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        )
    )
    boundary_receipt = world.prepare_event(cycle_boundary)
    after_boundary = world.step(
        cycle_boundary,
        boundary_receipt,
        jnp.asarray([1, 0], dtype=jnp.int32),
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert bool(after_boundary.update_applied)
    chex.assert_trees_all_equal(
        after_boundary.pre_step_words, jnp.asarray([0, 8_998], dtype=jnp.uint32)
    )
    chex.assert_trees_all_equal(
        after_boundary.post_step_words, jnp.asarray([0, 8_999], dtype=jnp.uint32)
    )
    for state_key_name, receipt_key_name in (
        ("world_key", "next_world_key"),
        ("cue_key", "next_cue_key"),
        ("outcome_key", "next_outcome_key"),
        ("nuisance_key", "next_nuisance_key"),
        ("partner_velocity_key", "next_partner_velocity_key"),
    ):
        source_key = getattr(cycle_boundary, state_key_name)
        next_key = getattr(boundary_receipt, receipt_key_name)
        committed_key = getattr(after_boundary.state, state_key_name)
        assert not bool(jnp.all(jr.key_data(source_key) == jr.key_data(next_key)))
        chex.assert_trees_all_equal(jr.key_data(committed_key), jr.key_data(next_key))
    assert not bool(
        jnp.all(
            after_boundary.state.positions
            == jnp.asarray([-0.5, 0.5], dtype=jnp.float32)
        )
    )


def test_exact_evaluator_hidden_schedule_and_genesis_observation() -> None:
    world = _world()
    expected = (
        ("A", 0, 769),
        ("B", 769, 1566),
        ("A", 1566, 2395),
        ("D", 2395, 3252),
        ("A", 3252, 4135),
        ("C", 4135, 5046),
        ("A", 5046, 5987),
        ("B", 5987, 6958),
        ("C", 6958, 7967),
        ("A", 7967, 8998),
    )
    assert HCCL_CAUSAL_CORE_SCHEDULE == expected
    assert HCCL_CAUSAL_CORE_REGIME_NAMES == ("A", "B", "C", "D")
    for name, start, end in expected:
        assert world.evaluator_regime_name_for_step(start) == name
        assert world.evaluator_regime_name_for_step(end - 1) == name
    with pytest.raises(ValueError, match="step"):
        world.evaluator_regime_name_for_step(8998)

    state = world.init(jr.key(7))
    observation = world.observe(state)
    assert observation.shape == (2, 16)
    np.testing.assert_array_equal(np.asarray(observation[:, 4:8]), np.zeros((2, 4)))
    np.testing.assert_array_equal(np.asarray(observation[:, 8]), np.zeros((2,)))
    assert set(np.asarray(observation[:, 9:11]).reshape(-1).tolist()) <= {-1.0, 1.0}
    assert world.learner_observation_fields == (
        "normalized_own_position",
        "normalized_relative_position",
        "normalized_own_velocity",
        "normalized_noisy_partner_velocity",
        "previous_own_action_sign",
        "previous_partner_action_sign",
        "previous_task_score",
        "previous_own_net_reward",
        "history_available",
        "noisy_hidden_sign_cue_0",
        "noisy_hidden_sign_cue_1",
        "nuisance_0_tv_sensitive",
        "nuisance_1",
        "nuisance_2",
        "nuisance_3",
        "nuisance_4",
    )
    assert all(
        forbidden not in world.learner_observation_fields
        for forbidden in ("regime", "oracle", "hidden_sign", "noise_flip", "clock")
    )


def test_named_event_receipt_is_source_bound_immutable_and_fixed_draw() -> None:
    world = _world()
    state = world.init(jr.key(11))
    first = world.prepare_event(state)
    second = world.prepare_event(state)
    chex.assert_trees_all_equal(first, second)
    assert bool(world.event_receipt_valid(state, first))
    assert first.stream_names == (
        "world_transition",
        "next_cues",
        "outcome_factor",
        "next_nuisance",
        "next_partner_velocity_observation",
    )
    assert tuple(int(value) for value in first.draw_counts) == (1, 2, 1, 10, 2)
    assert first.nuisance_standard_normal.shape == (2, 5)
    assert first.partner_velocity_standard_normal.shape == (2,)
    assert first.next_cue_flipped.shape == (2,)

    stale_state = world.reseal_state(
        dataclasses.replace(  # type: ignore[type-var]
            state,
            step_words=jnp.asarray([0, 1], dtype=jnp.uint32),
            history_available=jnp.asarray(True, dtype=jnp.bool_),
            previous_action_signs=jnp.asarray([-1.0, 1.0], dtype=jnp.float32),
        )
    )
    assert not bool(world.event_receipt_valid(stale_state, first))
    tampered = dataclasses.replace(  # type: ignore[type-var]
        first,
        nuisance_standard_normal=first.nuisance_standard_normal.at[0, 0].add(1.0),
    )
    assert not bool(world.event_receipt_valid(state, tampered))


def test_pure_proposal_physics_factors_noise_scope_and_typed_signals() -> None:
    world = _world()
    state = world.init(jr.key(13))
    receipt = world.prepare_event(state)
    actions = jnp.asarray([0, 1], dtype=jnp.int32)
    proposal = world.propose(state, receipt, actions)
    duplicate = world.propose(state, receipt, actions)
    chex.assert_trees_all_equal(proposal, duplicate)
    assert bool(proposal.valid)
    np.testing.assert_array_equal(np.asarray(proposal.action_signs), [-1.0, 1.0])

    expected_velocity = np.clip(
        0.75 * np.asarray(state.velocities) + 0.15 * np.asarray([-1.0, 1.0]),
        -0.25,
        0.25,
    )
    expected_position = np.clip(
        np.asarray(state.positions) + expected_velocity,
        -1.0,
        1.0,
    )
    np.testing.assert_allclose(np.asarray(proposal.candidate_state.velocities), expected_velocity)
    np.testing.assert_allclose(np.asarray(proposal.candidate_state.positions), expected_position)

    z = float(state.hidden_sign)
    distance = abs(expected_position[0] - expected_position[1]) / 2.0
    target = 0.6 * z
    local = [
        0.2 * (1.0 - abs(position - target) / 1.6) + 0.8 * float(abs(position - target) <= 0.1)
        for position in expected_position
    ]
    expected_g = 0.5 * (1.0 - distance) + 0.25 * sum(local)
    expected_v = np.clip(
        1.0 - sum(abs(velocity - 0.8 * 0.25 * z) for velocity in expected_velocity) / (3.6 * 0.25),
        0.0,
        1.0,
    )
    expected_clean_p = float((-1.0 * 1.0) == z)
    expected_noisy_p = 1.0 - expected_clean_p if bool(receipt.outcome_flipped) else expected_clean_p
    assert float(proposal.factors.gathering) == pytest.approx(expected_g)
    assert float(proposal.factors.velocity) == pytest.approx(expected_v)
    assert float(proposal.factors.convention_clean) == expected_clean_p
    assert float(proposal.factors.convention_noisy) == expected_noisy_p
    assert float(world.task_score_for_regime(HCCL_REGIME_A, proposal.factors)) == pytest.approx(
        expected_g
    )
    assert float(world.task_score_for_regime(HCCL_REGIME_B, proposal.factors)) == pytest.approx(
        expected_v
    )
    assert float(world.task_score_for_regime(HCCL_REGIME_C, proposal.factors)) == pytest.approx(
        0.5 * (expected_g + expected_v)
    )
    assert float(world.task_score_for_regime(HCCL_REGIME_D, proposal.factors)) == expected_noisy_p
    assert float(proposal.signals.task_score) == pytest.approx(expected_g)
    chex.assert_trees_all_equal(proposal.signals.message_charge, jnp.zeros((2,), jnp.float32))
    chex.assert_trees_all_equal(proposal.signals.safety_cost, jnp.zeros((2,), jnp.float32))
    chex.assert_trees_all_equal(
        proposal.signals.net_reward,
        jnp.full((2,), proposal.signals.task_score, dtype=jnp.float32),
    )


def test_tv_variance_and_partner_velocity_noise_are_exact_observation_mechanics() -> None:
    world = _world()
    state = world.init(jr.key(17))
    first = _step(world, state, (0, 0))
    assert bool(first.update_applied)
    receipt = world.prepare_event(first.state)
    proposal = world.propose(
        first.state,
        receipt,
        jnp.asarray([0, 1], dtype=jnp.int32),
    )
    assert float(proposal.candidate_state.positions[0]) < -0.8
    expected_tv = float(receipt.nuisance_standard_normal[0, 0]) * np.sqrt(10.0)
    assert float(proposal.candidate_state.current_nuisance[0, 0]) == pytest.approx(expected_tv)
    np.testing.assert_array_equal(
        np.asarray(proposal.candidate_state.current_nuisance[0, 1:]),
        np.asarray(receipt.nuisance_standard_normal[0, 1:]),
    )
    expected_partner_velocity = (
        float(proposal.candidate_state.velocities[1])
        + 0.01 * float(receipt.partner_velocity_standard_normal[0])
    ) / 0.25
    assert float(proposal.next_observation[0, 3]) == pytest.approx(expected_partner_velocity)
    assert float(proposal.next_observation[0, 11]) == pytest.approx(expected_tv)


def test_commit_rollback_retry_and_capacity_are_bit_exact() -> None:
    world = _world()
    initial = world.init(jr.key(19))
    receipt = world.prepare_event(initial)
    actions = jnp.asarray([1, 0], dtype=jnp.int32)
    proposal = world.propose(initial, receipt, actions)
    direct = world.commit(
        initial,
        receipt,
        proposal,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    rejected = world.commit(
        initial,
        receipt,
        proposal,
        downstream_candidate_valid=jnp.asarray(False, dtype=jnp.bool_),
    )
    assert bool(direct.update_applied)
    assert not bool(rejected.update_applied)
    chex.assert_trees_all_equal(rejected.state, initial)
    retry_receipt = world.prepare_event(rejected.state)
    chex.assert_trees_all_equal(retry_receipt, receipt)
    retry_proposal = world.propose(rejected.state, retry_receipt, actions)
    retry = world.commit(
        rejected.state,
        retry_receipt,
        retry_proposal,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    chex.assert_trees_all_equal(retry, direct)

    exhausted = world.reseal_state(
        dataclasses.replace(  # type: ignore[type-var]
            direct.state,
            step_words=jnp.asarray([0, 8998], dtype=jnp.uint32),
        )
    )
    exhausted_receipt = world.prepare_event(exhausted)
    exhausted_result = world.step(
        exhausted,
        exhausted_receipt,
        actions,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    assert not bool(exhausted_result.lifetime_capacity_available)
    assert not bool(exhausted_result.update_applied)
    chex.assert_trees_all_equal(exhausted_result.state, exhausted)


def test_eager_jit_scan_resources_and_in_memory_checkpoint() -> None:
    world = _world()
    initial = world.init(jr.key(23))
    receipt = world.prepare_event(initial)
    actions = jnp.asarray([1, 1], dtype=jnp.int32)
    eager = world.step(
        initial,
        receipt,
        actions,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    compiled = jax.jit(
        lambda state, event, action: world.step(
            state,
            event,
            action,
            downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
        )
    )(initial, receipt, actions)
    chex.assert_trees_all_close(
        _materialize_keys(eager),
        _materialize_keys(compiled),
        rtol=1e-6,
        atol=1e-7,
    )

    action_rows = jnp.asarray([[1, 1], [0, 1], [0, 0]], dtype=jnp.int32)
    gates = jnp.asarray([True, False, True], dtype=jnp.bool_)
    scan_eager = run_hccl_causal_core_scan(world, initial, action_rows, gates)
    scan_compiled = jax.jit(run_hccl_causal_core_scan, static_argnums=(0,))(
        world,
        initial,
        action_rows,
        gates,
    )
    chex.assert_trees_all_close(
        _materialize_keys(scan_eager),
        _materialize_keys(scan_compiled),
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(np.asarray(scan_eager.update_applied), [True, False, True])
    np.testing.assert_array_equal(
        np.asarray(scan_eager.post_step_words),
        [[0, 1], [0, 1], [0, 2]],
    )

    budget = world.resource_budget(initial)
    assert budget.persistent_state_nbytes == measure_hccl_causal_core_state_nbytes(initial)
    assert budget.event_draws_per_receipt == 16
    assert budget.maximum_committed_transitions == 8998
    assert budget.output_write_calls == 0
    checkpoint = save_hccl_causal_core_checkpoint(world, eager.state)
    restored_world, restored_state = load_hccl_causal_core_checkpoint(checkpoint)
    assert restored_world.to_config() == world.to_config()
    chex.assert_trees_all_equal(restored_state, eager.state)

    tampered = dataclasses.replace(
        checkpoint,
        state=cast(Any, checkpoint.state).replace(
            positions=checkpoint.state.positions.at[0].add(jnp.float32(0.1))
        ),
    )
    with pytest.raises(ValueError, match="checkpoint"):
        load_hccl_causal_core_checkpoint(tampered)

    wrong_fixed_type = dataclasses.replace(
        cast(Any, checkpoint),
        output_writes_authorized=0,
    )
    with pytest.raises(ValueError, match="output_writes_authorized"):
        load_hccl_causal_core_checkpoint(wrong_fixed_type)

    wrong_state_bytes_type = dataclasses.replace(
        cast(Any, checkpoint),
        state_nbytes=float(checkpoint.state_nbytes),
    )
    with pytest.raises(TypeError, match="state_nbytes"):
        load_hccl_causal_core_checkpoint(wrong_state_bytes_type)

    wrong_config_type = dict(checkpoint.config)
    wrong_config_type["maximum_committed_transitions"] = 8998.0
    with pytest.raises(ValueError, match="resolution|config|fixed"):
        load_hccl_causal_core_checkpoint(
            dataclasses.replace(checkpoint, config=wrong_config_type)
        )
