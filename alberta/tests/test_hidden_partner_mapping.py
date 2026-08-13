"""Causal contract tests for the uncued hidden-partner mapping life."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_BASE_SEGMENT_LENGTHS,
    DEFAULT_JITTER_RADIUS,
    DEFAULT_REGIME_SCHEDULE,
    HAS_PARTNER_HISTORY_INDEX,
    HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION,
    NEGATIVE_ACTION,
    NUISANCE_1_INDEX,
    NUISANCE_2_INDEX,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    POSITIVE_ACTION,
    PREVIOUS_OUTCOME_INDEX,
    PREVIOUS_PARTNER_ACTION_INDEX,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_D,
    U_INDEX,
    V_INDEX,
    X_INDEX,
    HiddenPartnerMappingConfig,
    HiddenPartnerMappingState,
    HiddenPartnerMappingTransition,
    HiddenPartnerMappingWorld,
)


def _small_config(
    *,
    lengths: tuple[int, ...] = (2, 3, 2, 2, 2, 3, 2, 3, 3),
    jitter_radius: int = 0,
    flip_probability: float = 0.0,
) -> HiddenPartnerMappingConfig:
    return HiddenPartnerMappingConfig(
        base_segment_lengths=lengths,
        jitter_radius=jitter_radius,
        partner_flip_probability=flip_probability,
    )


def _tree_nbytes(tree: object) -> int:
    return sum(int(leaf.nbytes) for leaf in jax.tree_util.tree_leaves(tree))


def _at_exact_step(
    state: HiddenPartnerMappingState,
    step: int,
) -> HiddenPartnerMappingState:
    """Move a test fixture to one coherent exact non-negative lifetime."""

    return state.replace(
        step_count=jnp.asarray(min(step, 2**31 - 1), dtype=jnp.int32),
        step_words=jnp.asarray((step >> 32, step & (2**32 - 1)), dtype=jnp.uint32),
        previous_outcome=jnp.asarray(0.0 if step == 0 else 1.0, dtype=jnp.float32),
        previous_partner_action=jnp.asarray(0 if step == 0 else 1, dtype=jnp.int32),
        has_partner_history=jnp.asarray(step > 0, dtype=jnp.bool_),
    )


def test_default_contract_and_canonical_config_round_trip() -> None:
    config = HiddenPartnerMappingConfig()
    world = HiddenPartnerMappingWorld(config)

    assert config.contract_version == HIDDEN_PARTNER_MAPPING_CONTRACT_VERSION
    assert config.regime_schedule == (
        REGIME_A,
        REGIME_B,
        REGIME_A,
        REGIME_D,
        REGIME_A,
        REGIME_C,
        REGIME_A,
        REGIME_B,
        REGIME_C,
    )
    assert config.regime_schedule == DEFAULT_REGIME_SCHEDULE
    assert config.base_segment_lengths == DEFAULT_BASE_SEGMENT_LENGTHS
    assert config.jitter_radius == DEFAULT_JITTER_RADIUS
    assert world.observation_dim == OBSERVATION_DIM == 8
    assert world.feature_dim == OBSERVATION_DIM
    assert world.n_actions == 2
    assert world.n_segments == 9
    assert OBSERVATION_FIELDS == (
        "x",
        "previous_joint_outcome",
        "previous_partner_action",
        "has_partner_history",
        "u",
        "v",
        "nuisance_1",
        "nuisance_2",
    )

    encoded = config.canonical_json()
    assert encoded == config.canonical_json()
    assert "NaN" not in encoded
    assert json.loads(encoded) == config.to_config()
    assert HiddenPartnerMappingConfig.from_config(json.loads(encoded)) == config

    world_payload = world.to_config()
    restored = HiddenPartnerMappingWorld.from_config(
        json.loads(
            json.dumps(
                world_payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    )
    assert restored.config == config
    assert restored.to_config() == world_payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contract_version": "hidden-partner-mapping-v1"},
        {"regime_schedule": DEFAULT_REGIME_SCHEDULE[:-1]},
        {"regime_schedule": (False,) + DEFAULT_REGIME_SCHEDULE[1:]},
        {"regime_schedule": list(DEFAULT_REGIME_SCHEDULE)},
        {"base_segment_lengths": DEFAULT_BASE_SEGMENT_LENGTHS[:-1]},
        {"base_segment_lengths": list(DEFAULT_BASE_SEGMENT_LENGTHS)},
        {"base_segment_lengths": (True,) + DEFAULT_BASE_SEGMENT_LENGTHS[1:]},
        {"base_segment_lengths": (1,) * 9, "jitter_radius": 1},
        {"jitter_radius": -1},
        {"jitter_radius": True},
        {"partner_flip_probability": -0.01},
        {"partner_flip_probability": 1.01},
        {"partner_flip_probability": float("nan")},
        {"partner_flip_probability": float("inf")},
        {"partner_flip_probability": True},
        {"base_segment_lengths": (300_000_000,) * 9},
    ],
)
def test_config_validation_is_strict(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HiddenPartnerMappingConfig(**kwargs)


def test_config_deserialization_rejects_schema_or_type_drift() -> None:
    payload = HiddenPartnerMappingConfig().to_config()

    missing = dict(payload)
    missing.pop("jitter_radius")
    with pytest.raises(ValueError, match="schema"):
        HiddenPartnerMappingConfig.from_config(missing)

    extra = dict(payload)
    extra["future_oracle"] = True
    with pytest.raises(ValueError, match="schema"):
        HiddenPartnerMappingConfig.from_config(extra)

    wrong_type = dict(payload)
    wrong_type["type"] = "OtherConfig"
    with pytest.raises(ValueError, match="type"):
        HiddenPartnerMappingConfig.from_config(wrong_type)

    with pytest.raises(TypeError, match="HiddenPartnerMappingConfig"):
        HiddenPartnerMappingWorld(config={})  # type: ignore[arg-type]


def test_seeded_jitter_is_reproducible_bounded_and_evaluator_only() -> None:
    world = HiddenPartnerMappingWorld()
    state = world.init(jr.key(5))
    repeated = world.init(jr.key(5))
    chex.assert_trees_all_equal(state, repeated)

    base = jnp.asarray(DEFAULT_BASE_SEGMENT_LENGTHS, dtype=jnp.int32)
    assert jnp.all(state.segment_lengths >= base - DEFAULT_JITTER_RADIUS)
    assert jnp.all(state.segment_lengths <= base + DEFAULT_JITTER_RADIUS)
    assert jnp.all(state.segment_lengths > 0)
    chex.assert_trees_all_equal(
        state.segment_ends,
        jnp.cumsum(state.segment_lengths, dtype=jnp.int32),
    )

    schedules = [
        tuple(map(int, world.init(jr.key(seed)).segment_lengths.tolist())) for seed in range(8)
    ]
    assert len(set(schedules)) > 1

    # The complete seeded schedule is environment/evaluator state, never part
    # of the fixed-width ordinary observation.
    observation = world.observe(state)
    chex.assert_shape(observation, (OBSERVATION_DIM,))
    assert observation.size < state.segment_lengths.size
    chex.assert_trees_all_equal(
        observation,
        jnp.asarray(
            (
                state.current_signals[0],
                state.previous_outcome,
                state.previous_partner_action,
                state.has_partner_history,
                state.current_signals[1],
                state.current_signals[2],
                state.current_signals[3],
                state.current_signals[4],
            ),
            dtype=jnp.float32,
        ),
    )


def test_initial_observation_has_exact_birth_semantics_and_ranges() -> None:
    world = HiddenPartnerMappingWorld(_small_config())
    state = world.init(jr.key(6))
    observation = world.observe(state)

    assert int(state.step_count) == 0
    assert not bool(state.has_partner_history)
    assert float(observation[PREVIOUS_OUTCOME_INDEX]) == 0.0
    assert float(observation[PREVIOUS_PARTNER_ACTION_INDEX]) == 0.0
    assert float(observation[HAS_PARTNER_HISTORY_INDEX]) == 0.0
    for index in (
        X_INDEX,
        U_INDEX,
        V_INDEX,
        NUISANCE_1_INDEX,
        NUISANCE_2_INDEX,
    ):
        assert float(observation[index]) in (-1.0, 1.0)
    chex.assert_tree_all_finite(observation)


def test_hidden_schedule_is_exact_and_repeats_without_resetting() -> None:
    lengths = (2, 3, 2, 2, 2, 3, 2, 3, 3)
    world = HiddenPartnerMappingWorld(_small_config(lengths=lengths))
    state = world.init(jr.key(7))
    cycle_length = sum(lengths)
    expected_one_cycle = [
        regime
        for regime, length in zip(DEFAULT_REGIME_SCHEDULE, lengths, strict=True)
        for _ in range(length)
    ]

    regimes: list[int] = []
    segments: list[int] = []
    segment_steps: list[int] = []
    switched: list[bool] = []
    terminated: list[bool] = []
    discounts: list[float] = []
    final_transition: HiddenPartnerMappingTransition | None = None
    for _ in range(cycle_length + 5):
        transition, state = world.step(
            state,
            jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32),
        )
        regimes.append(int(transition.oracle.regime_id))
        segments.append(int(transition.oracle.segment_index))
        segment_steps.append(int(transition.oracle.segment_step))
        switched.append(bool(transition.oracle.schedule_switched))
        terminated.append(bool(transition.terminated))
        discounts.append(float(transition.discount))
        final_transition = transition

    assert regimes == expected_one_cycle + expected_one_cycle[:5]
    assert segments[:cycle_length] == [
        segment for segment, length in enumerate(lengths) for _ in range(length)
    ]
    assert segment_steps[:cycle_length] == [step for length in lengths for step in range(length)]
    expected_switches = [
        (step + 1) in set(jnp.cumsum(jnp.asarray(lengths)).tolist()) for step in range(cycle_length)
    ]
    assert switched[:cycle_length] == expected_switches
    assert int(state.step_count) == cycle_length + 5
    assert bool(state.has_partner_history)
    assert not any(terminated)
    assert discounts == [1.0] * (cycle_length + 5)
    assert final_transition is not None
    assert int(final_transition.oracle.cycle_index) == 1
    # Recurrence never restores birth markers or clears temporal state.
    assert float(final_transition.observation[HAS_PARTNER_HISTORY_INDEX]) == 1.0
    assert float(final_transition.next_observation[HAS_PARTNER_HISTORY_INDEX]) == 1.0
    assert float(final_transition.next_observation[PREVIOUS_PARTNER_ACTION_INDEX]) in (
        -1.0,
        1.0,
    )


def test_hidden_task_and_imminent_boundary_do_not_leak_into_observation() -> None:
    world = HiddenPartnerMappingWorld(_small_config(lengths=(3,) * 9, jitter_radius=0))
    base = world.init(jr.key(8))

    # Identical ordinary state at hidden A and hidden B gives bitwise-identical
    # pre-action input, even though the partner mapping is opposite.
    hidden_a = _at_exact_step(base, 6)
    hidden_b = _at_exact_step(base, 3)
    chex.assert_trees_all_equal(world.observe(hidden_a), world.observe(hidden_b))
    a_transition, _ = world.step(
        hidden_a,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    b_transition, _ = world.step(
        hidden_b,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    assert int(a_transition.oracle.regime_id) == REGIME_A
    assert int(b_transition.oracle.regime_id) == REGIME_B
    assert float(a_transition.oracle.partner_intended_sign) == -float(
        b_transition.oracle.partner_intended_sign
    )

    # A second leakage pair has the same current A mapping and ordinary state,
    # but different evaluator-only schedules make only one transition cross
    # into B.  Current action, reward, and next ordinary observation remain
    # identical; only the oracle reports the imminent boundary.
    long_first = _at_exact_step(base, 1).replace(
        segment_lengths=jnp.asarray((3,) * 9, dtype=jnp.int32),
        segment_ends=jnp.cumsum(jnp.asarray((3,) * 9, dtype=jnp.int32)),
    )
    short_first_lengths = jnp.asarray((2,) + (3,) * 8, dtype=jnp.int32)
    short_first = long_first.replace(
        segment_lengths=short_first_lengths,
        segment_ends=jnp.cumsum(short_first_lengths, dtype=jnp.int32),
    )
    chex.assert_trees_all_equal(
        world.observe(long_first),
        world.observe(short_first),
    )
    stays_a, _ = world.step(
        long_first,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    enters_b, _ = world.step(
        short_first,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    assert not bool(stays_a.oracle.schedule_switched)
    assert bool(enters_b.oracle.schedule_switched)
    assert int(stays_a.oracle.next_regime_id) == REGIME_A
    assert int(enters_b.oracle.next_regime_id) == REGIME_B
    chex.assert_trees_all_equal(stays_a.observation, enters_b.observation)
    chex.assert_trees_all_equal(stays_a.partner_action, enters_b.partner_action)
    chex.assert_trees_all_equal(stays_a.reward, enters_b.reward)
    chex.assert_trees_all_equal(
        stays_a.next_observation,
        enters_b.next_observation,
    )


def test_partner_action_is_simultaneous_and_causally_independent_of_focal_action() -> None:
    world = HiddenPartnerMappingWorld(_small_config(flip_probability=0.5))
    state = world.init(jr.key(9))

    negative, negative_state = world.step(
        state,
        jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32),
    )
    positive, positive_state = world.step(
        state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )

    chex.assert_trees_all_equal(negative.partner_action, positive.partner_action)
    chex.assert_trees_all_equal(
        negative.oracle.partner_intended_action,
        positive.oracle.partner_intended_action,
    )
    chex.assert_trees_all_equal(
        negative.oracle.partner_flipped,
        positive.oracle.partner_flipped,
    )
    chex.assert_trees_all_equal(negative_state.partner_key, positive_state.partner_key)
    chex.assert_trees_all_equal(negative_state.signal_key, positive_state.signal_key)
    chex.assert_trees_all_equal(
        negative_state.current_signals,
        positive_state.current_signals,
    )
    chex.assert_trees_all_equal(
        negative_state.previous_partner_action,
        positive_state.previous_partner_action,
    )
    assert float(negative.reward + positive.reward) == pytest.approx(1.0)
    chex.assert_trees_all_equal(
        negative.reward,
        negative.oracle.counterfactual_rewards[NEGATIVE_ACTION],
    )
    chex.assert_trees_all_equal(
        positive.reward,
        positive.oracle.counterfactual_rewards[POSITIVE_ACTION],
    )

    # Own action changes only the causally downstream outcome channel in the
    # next ordinary observation.
    next_negative_without_outcome = negative.next_observation.at[PREVIOUS_OUTCOME_INDEX].set(0.0)
    next_positive_without_outcome = positive.next_observation.at[PREVIOUS_OUTCOME_INDEX].set(0.0)
    chex.assert_trees_all_equal(
        next_negative_without_outcome,
        next_positive_without_outcome,
    )
    assert float(negative.next_observation[PREVIOUS_OUTCOME_INDEX]) == -float(
        positive.next_observation[PREVIOUS_OUTCOME_INDEX]
    )


@pytest.mark.parametrize(
    ("segment_index", "expected_regime", "expected_intended_sign"),
    [
        (0, REGIME_A, 1.0),
        (1, REGIME_B, -1.0),
        (3, REGIME_D, -1.0),
        (5, REGIME_C, -1.0),
    ],
)
def test_regime_formulas_and_reward_timing(
    segment_index: int,
    expected_regime: int,
    expected_intended_sign: float,
) -> None:
    world = HiddenPartnerMappingWorld(_small_config(lengths=(2,) * 9, flip_probability=0.0))
    state = _at_exact_step(
        world.init(jr.key(10)),
        18 + 2 * segment_index,
    ).replace(
        current_signals=jnp.asarray(
            (1.0, 1.0, -1.0, 1.0, -1.0),
            dtype=jnp.float32,
        ),
        previous_outcome=jnp.asarray(-1.0, dtype=jnp.float32),
        previous_partner_action=jnp.asarray(-1, dtype=jnp.int32),
        has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
    )
    transition, next_state = world.step(
        state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )

    assert int(transition.oracle.regime_id) == expected_regime
    assert float(transition.oracle.partner_intended_sign) == expected_intended_sign
    assert float(transition.oracle.partner_action_sign) == expected_intended_sign
    assert float(transition.observation[PREVIOUS_OUTCOME_INDEX]) == -1.0
    expected_outcome = expected_intended_sign  # focal +1
    expected_reward = (1.0 + expected_outcome) / 2.0
    assert float(transition.outcome) == expected_outcome
    assert float(transition.reward) == expected_reward
    assert float(transition.next_observation[PREVIOUS_OUTCOME_INDEX]) == expected_outcome
    assert float(next_state.previous_outcome) == expected_outcome
    assert (
        float(transition.next_observation[PREVIOUS_PARTNER_ACTION_INDEX]) == expected_intended_sign
    )


def test_boundary_reward_uses_current_regime_not_next_regime() -> None:
    world = HiddenPartnerMappingWorld(_small_config(lengths=(2,) * 9, flip_probability=0.0))
    state = _at_exact_step(world.init(jr.key(11)), 1).replace(
        current_signals=jnp.asarray(
            (1.0, 1.0, 1.0, -1.0, -1.0),
            dtype=jnp.float32,
        ),
    )
    transition, _ = world.step(
        state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )

    assert int(transition.oracle.regime_id) == REGIME_A
    assert int(transition.oracle.next_regime_id) == REGIME_B
    assert bool(transition.oracle.schedule_switched)
    assert float(transition.oracle.partner_intended_sign) == 1.0
    assert float(transition.reward) == 1.0


def test_partner_flip_and_named_rng_streams_are_isolated() -> None:
    no_flip_world = HiddenPartnerMappingWorld(_small_config(jitter_radius=0, flip_probability=0.0))
    all_flip_world = HiddenPartnerMappingWorld(_small_config(jitter_radius=0, flip_probability=1.0))
    no_flip_state = no_flip_world.init(jr.key(12))
    all_flip_state = all_flip_world.init(jr.key(12))
    chex.assert_trees_all_equal(no_flip_state, all_flip_state)

    no_flip, no_flip_next = no_flip_world.step(
        no_flip_state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    all_flip, all_flip_next = all_flip_world.step(
        all_flip_state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    assert not bool(no_flip.oracle.partner_flipped)
    assert bool(all_flip.oracle.partner_flipped)
    assert float(no_flip.oracle.partner_action_sign) == -float(all_flip.oracle.partner_action_sign)
    chex.assert_trees_all_equal(
        no_flip_next.current_signals,
        all_flip_next.current_signals,
    )
    chex.assert_trees_all_equal(
        no_flip_next.segment_lengths,
        all_flip_next.segment_lengths,
    )
    chex.assert_trees_all_equal(no_flip_next.signal_key, all_flip_next.signal_key)

    # Altering only the schedule-jitter configuration cannot perturb the
    # separately named ordinary-signal or partner RNG streams.
    jitter_world = HiddenPartnerMappingWorld(
        _small_config(
            lengths=(4,) * 9,
            jitter_radius=2,
            flip_probability=0.0,
        )
    )
    fixed_world = HiddenPartnerMappingWorld(
        _small_config(
            lengths=(4,) * 9,
            jitter_radius=0,
            flip_probability=0.0,
        )
    )
    jitter_state = jitter_world.init(jr.key(13))
    fixed_state = fixed_world.init(jr.key(13))
    chex.assert_trees_all_equal(
        jitter_state.current_signals,
        fixed_state.current_signals,
    )
    chex.assert_trees_all_equal(jitter_state.signal_key, fixed_state.signal_key)
    chex.assert_trees_all_equal(jitter_state.partner_key, fixed_state.partner_key)


def test_fixed_state_resource_accounting_matches_array_leaves() -> None:
    world = HiddenPartnerMappingWorld()
    state = world.init(jr.key(14))
    budget = world.resource_budget

    assert budget.to_dict() == {
        "state_schema": "alberta.hidden-partner-mapping.state.v2",
        "observation_float32_scalars": 8,
        "persistent_float32_scalars": 6,
        "persistent_int32_scalars": 20,
        "persistent_bool_scalars": 1,
        "exact_identity_uint32_scalars": 2,
        "exact_identity_nbytes": 8,
        "lifetime_identity_bits": 64,
        "telemetry_saturation": 2_147_483_647,
        "rng_uint32_scalars": 4,
        "persistent_state_scalars": 33,
        "state_nbytes": 129,
        "trainable_scalars": 0,
        "replay_capacity": 0,
    }
    assert _tree_nbytes(state) == budget.state_nbytes
    _, next_state = world.step(
        state,
        jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32),
    )
    assert _tree_nbytes(next_state) == budget.state_nbytes


def test_state_is_an_immutable_fixed_shape_pytree() -> None:
    world = HiddenPartnerMappingWorld()
    state = world.init(jr.key(15))
    with pytest.raises(FrozenInstanceError):
        state.step_count = jnp.asarray(3, dtype=jnp.int32)

    leaves, structure = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)
    assert isinstance(rebuilt, HiddenPartnerMappingState)
    chex.assert_trees_all_equal(rebuilt, state)


def test_action_static_contract_rejects_wrong_shape_or_dtype() -> None:
    world = HiddenPartnerMappingWorld(_small_config())
    state = world.init(jr.key(16))
    with pytest.raises(ValueError, match="scalar"):
        world.step(state, jnp.asarray([NEGATIVE_ACTION], dtype=jnp.int32))
    with pytest.raises(TypeError, match="integer"):
        world.step(state, jnp.asarray(0.0, dtype=jnp.float32))


def test_jit_scan_and_vmap_remain_continuing_finite_and_in_range() -> None:
    world = HiddenPartnerMappingWorld(
        _small_config(
            lengths=(4, 5, 4, 3, 4, 5, 4, 5, 5),
            jitter_radius=1,
            flip_probability=0.05,
        )
    )
    num_steps = 96
    actions = jnp.mod(jnp.arange(num_steps, dtype=jnp.int32), 2)

    @jax.jit
    def rollout(
        initial_state: HiddenPartnerMappingState,
        action_ids: jax.Array,
    ) -> tuple[HiddenPartnerMappingState, HiddenPartnerMappingTransition]:
        def scan_step(
            state: HiddenPartnerMappingState,
            action: jax.Array,
        ) -> tuple[HiddenPartnerMappingState, HiddenPartnerMappingTransition]:
            transition, next_state = world.step(state, action)
            return next_state, transition

        return jax.lax.scan(scan_step, initial_state, action_ids)

    initial = jax.jit(world.init)(jr.key(17))
    final_state, transitions = rollout(initial, actions)

    assert int(final_state.step_count) == num_steps
    assert not bool(jnp.any(transitions.terminated))
    chex.assert_trees_all_equal(
        transitions.discount,
        jnp.ones((num_steps,), dtype=jnp.float32),
    )
    assert jnp.all((transitions.focal_action >= 0) & (transitions.focal_action < 2))
    assert jnp.all((transitions.partner_action >= 0) & (transitions.partner_action < 2))
    assert jnp.all((transitions.reward == 0.0) | (transitions.reward == 1.0))
    assert jnp.all((transitions.outcome == -1.0) | (transitions.outcome == 1.0))
    assert jnp.all((transitions.observation >= -1.0) & (transitions.observation <= 1.0))
    assert jnp.all((transitions.next_observation >= -1.0) & (transitions.next_observation <= 1.0))
    chex.assert_shape(transitions.observation, (num_steps, OBSERVATION_DIM))
    chex.assert_shape(transitions.next_observation, (num_steps, OBSERVATION_DIM))
    chex.assert_shape(transitions.oracle.counterfactual_rewards, (num_steps, 2))
    chex.assert_tree_all_finite(
        (
            transitions.observation,
            transitions.next_observation,
            transitions.reward,
            transitions.outcome,
            transitions.discount,
            transitions.oracle.counterfactual_rewards,
            final_state.current_signals,
        )
    )

    keys = jr.split(jr.key(18), 4)
    batched_states = jax.vmap(world.init)(keys)
    batched_actions = jnp.asarray(
        (NEGATIVE_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION, POSITIVE_ACTION),
        dtype=jnp.int32,
    )
    batched_transitions, batched_next_states = jax.jit(jax.vmap(world.step))(
        batched_states, batched_actions
    )
    chex.assert_shape(batched_transitions.observation, (4, OBSERVATION_DIM))
    chex.assert_shape(batched_transitions.reward, (4,))
    chex.assert_shape(batched_next_states.current_signals, (4, 5))
    chex.assert_trees_all_equal(
        batched_next_states.step_count,
        jnp.ones((4,), dtype=jnp.int32),
    )
