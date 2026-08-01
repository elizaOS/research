"""Causal contracts for the recurrent hidden-partner world-feedback stream."""

from __future__ import annotations

import dataclasses
import json
from fractions import Fraction
from itertools import product
from typing import Any

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.streams.hidden_partner_mapping import (
    DEFAULT_BASE_SEGMENT_LENGTHS,
    DEFAULT_JITTER_RADIUS,
    DEFAULT_REGIME_SCHEDULE,
    NEGATIVE_ACTION,
    POSITIVE_ACTION,
    REGIME_A,
    REGIME_B,
    REGIME_C,
    REGIME_D,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    DEFAULT_CUE_FLIP_PROBABILITIES,
    DEFAULT_OUTCOME_FLIP_PROBABILITY,
    DEFAULT_WORLD_FLIP_PROBABILITY,
    HAS_PARTNER_HISTORY_INDEX,
    HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION,
    OBSERVATION_DIM,
    OBSERVATION_FIELDS,
    PREVIOUS_OUTCOME_INDEX,
    PREVIOUS_PARTNER_ACTION_INDEX,
    U_INDEX,
    V_INDEX,
    X_INDEX,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackState,
    HiddenPartnerWorldFeedbackTransition,
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit


def _small_config(
    *,
    lengths: tuple[int, ...] = (2, 3, 2, 2, 2, 3, 2, 3, 3),
    jitter_radius: int = 0,
    partner_flip_probability: float = 0.0,
    world_flip_probability: float = 0.0,
    cue_flip_probability: float = 0.0,
    outcome_flip_probability: float = 0.0,
) -> HiddenPartnerWorldFeedbackConfig:
    return HiddenPartnerWorldFeedbackConfig(
        base_segment_lengths=lengths,
        jitter_radius=jitter_radius,
        partner_flip_probability=partner_flip_probability,
        world_flip_probability=world_flip_probability,
        cue_flip_probabilities=(cue_flip_probability, cue_flip_probability),
        outcome_flip_probability=outcome_flip_probability,
    )


def _tree_nbytes(tree: object) -> int:
    return sum(int(getattr(leaf, "nbytes", 0)) for leaf in jax.tree_util.tree_leaves(tree))


def _unwrap_prng_keys(tree: object) -> object:
    def unwrap(value: object) -> object:
        dtype = getattr(value, "dtype", None)
        if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
            return jr.key_data(value)  # type: ignore[arg-type]
        return value

    return jax.tree_util.tree_map(unwrap, tree)


def test_default_contract_roundtrip_and_same_shape_observation() -> None:
    config = HiddenPartnerWorldFeedbackConfig()
    world = HiddenPartnerWorldFeedbackWorld(config)

    assert config.contract_version == HIDDEN_PARTNER_WORLD_FEEDBACK_CONTRACT_VERSION
    assert config.regime_schedule == DEFAULT_REGIME_SCHEDULE
    assert config.base_segment_lengths == DEFAULT_BASE_SEGMENT_LENGTHS
    assert config.jitter_radius == DEFAULT_JITTER_RADIUS
    assert config.partner_flip_probability == pytest.approx(0.05)
    assert config.world_flip_probability == DEFAULT_WORLD_FLIP_PROBABILITY == pytest.approx(0.03)
    assert config.cue_flip_probabilities == DEFAULT_CUE_FLIP_PROBABILITIES == (0.25, 0.35)
    assert (
        config.outcome_flip_probability == DEFAULT_OUTCOME_FLIP_PROBABILITY == pytest.approx(0.15)
    )
    assert world.observation_dim == world.feature_dim == OBSERVATION_DIM == 8
    assert world.n_actions == 2
    assert world.n_segments == 9
    assert OBSERVATION_FIELDS == (
        "x",
        "previous_contextual_outcome",
        "previous_partner_action",
        "has_partner_history",
        "u",
        "v",
        "world_cue_1",
        "world_cue_2",
    )

    encoded = config.canonical_json()
    assert "NaN" not in encoded
    assert json.loads(encoded) == config.to_config()
    assert HiddenPartnerWorldFeedbackConfig.from_config(json.loads(encoded)) == config
    rational = HiddenPartnerWorldFeedbackConfig(
        partner_flip_probability=Fraction(1, 20),
        world_flip_probability=Fraction(3, 100),
        cue_flip_probabilities=(Fraction(1, 4), Fraction(7, 20)),
        outcome_flip_probability=Fraction(3, 20),
    )
    assert rational == config
    assert HiddenPartnerWorldFeedbackConfig.from_config(rational.to_config()) == rational
    payload = world.to_config()
    restored = HiddenPartnerWorldFeedbackWorld.from_config(
        json.loads(json.dumps(payload, allow_nan=False, sort_keys=True))
    )
    assert restored.config == config
    assert restored.to_config() == payload
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.world_flip_probability = 0.5  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contract_version": "hidden-partner-world-feedback-v0"},
        {"regime_schedule": DEFAULT_REGIME_SCHEDULE[:-1]},
        {"regime_schedule": list(DEFAULT_REGIME_SCHEDULE)},
        {"regime_schedule": (False,) + DEFAULT_REGIME_SCHEDULE[1:]},
        {"base_segment_lengths": DEFAULT_BASE_SEGMENT_LENGTHS[:-1]},
        {"base_segment_lengths": list(DEFAULT_BASE_SEGMENT_LENGTHS)},
        {"base_segment_lengths": (True,) + DEFAULT_BASE_SEGMENT_LENGTHS[1:]},
        {"base_segment_lengths": (1,) * 9, "jitter_radius": 1},
        {"jitter_radius": -1},
        {"jitter_radius": True},
        {"partner_flip_probability": -0.01},
        {"partner_flip_probability": True},
        {"world_flip_probability": 1.01},
        {"world_flip_probability": float("nan")},
        {"world_flip_probability": True},
        {"cue_flip_probabilities": [-0.01, 0.35]},
        {"cue_flip_probabilities": (0.25,)},
        {"cue_flip_probabilities": (-0.01, 0.35)},
        {"cue_flip_probabilities": (0.25, float("inf"))},
        {"cue_flip_probabilities": (0.25, True)},
        {"outcome_flip_probability": -0.01},
        {"outcome_flip_probability": float("nan")},
        {"outcome_flip_probability": True},
        {"base_segment_lengths": (300_000_000,) * 9},
    ],
)
def test_config_validation_is_strict(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        HiddenPartnerWorldFeedbackConfig(**kwargs)


def test_config_deserialization_rejects_schema_and_type_drift() -> None:
    payload = HiddenPartnerWorldFeedbackConfig().to_config()
    for malformed in (
        {key: value for key, value in payload.items() if key != "cue_flip_probabilities"},
        {**payload, "future_oracle": True},
        {**payload, "type": "OtherConfig"},
    ):
        with pytest.raises(ValueError):
            HiddenPartnerWorldFeedbackConfig.from_config(malformed)
    with pytest.raises(TypeError, match="HiddenPartnerWorldFeedbackConfig"):
        HiddenPartnerWorldFeedbackWorld(config={})  # type: ignore[arg-type]


def test_seeded_initial_state_is_bounded_task_oracle_free_and_exactly_accounted() -> None:
    world = HiddenPartnerWorldFeedbackWorld()
    state = world.init(jr.key(5))
    chex.assert_trees_all_equal(state, world.init(jr.key(5)))
    observation = world.observe(state)

    assert observation.shape == (OBSERVATION_DIM,)
    assert observation.dtype == jnp.float32
    assert float(observation[PREVIOUS_OUTCOME_INDEX]) == 0.0
    assert float(observation[PREVIOUS_PARTNER_ACTION_INDEX]) == 0.0
    assert float(observation[HAS_PARTNER_HISTORY_INDEX]) == 0.0
    for index in (X_INDEX, U_INDEX, V_INDEX, CUE_1_INDEX, CUE_2_INDEX):
        assert float(observation[index]) in (-1.0, 1.0)
    assert float(state.world_sign) in (-1.0, 1.0)
    assert observation.size < state.segment_lengths.size
    budget = world.resource_budget
    assert budget.observation_float32_scalars == 8
    assert budget.trainable_scalars == 0
    assert budget.replay_capacity == 0
    assert budget.state_nbytes == _tree_nbytes(state)
    assert budget.state_nbytes == 149


def test_contextual_reward_depends_on_world_state_after_conditioning_joint_action() -> None:
    world = HiddenPartnerWorldFeedbackWorld(_small_config(cue_flip_probability=0.5))
    base = world.init(jr.key(6)).replace(
        current_signals=jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        current_cues=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
    )
    positive_world = base.replace(world_sign=jnp.asarray(1.0, dtype=jnp.float32))
    negative_world = base.replace(world_sign=jnp.asarray(-1.0, dtype=jnp.float32))

    positive, _ = world.step(positive_world, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))
    negative, _ = world.step(negative_world, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))

    chex.assert_trees_all_equal(positive.focal_action, negative.focal_action)
    chex.assert_trees_all_equal(positive.partner_action, negative.partner_action)
    assert float(positive.reward) == 1.0
    assert float(negative.reward) == 0.0
    assert float(positive.outcome) == -float(negative.outcome)
    assert float(positive.reward) == (1.0 + float(positive.outcome)) / 2.0
    assert float(negative.reward) == (1.0 + float(negative.outcome)) / 2.0
    assert float(positive.oracle.world_sign) == 1.0
    assert float(negative.oracle.world_sign) == -1.0
    assert int(positive.oracle.full_information_optimal_focal_action) != int(
        negative.oracle.full_information_optimal_focal_action
    )


def test_partner_and_world_rng_are_action_independent_but_outcomes_are_causal() -> None:
    world = HiddenPartnerWorldFeedbackWorld(
        _small_config(
            partner_flip_probability=0.5,
            world_flip_probability=0.5,
            cue_flip_probability=0.5,
            outcome_flip_probability=0.5,
        )
    )
    state = world.init(jr.key(7))
    negative, negative_state = world.step(state, jnp.asarray(NEGATIVE_ACTION, dtype=jnp.int32))
    positive, positive_state = world.step(state, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))

    for left, right in (
        (negative.partner_action, positive.partner_action),
        (negative.oracle.partner_flipped, positive.oracle.partner_flipped),
        (negative.oracle.world_flipped, positive.oracle.world_flipped),
        (negative.oracle.outcome_flipped, positive.oracle.outcome_flipped),
        (negative_state.signal_key, positive_state.signal_key),
        (negative_state.partner_key, positive_state.partner_key),
        (negative_state.world_key, positive_state.world_key),
        (negative_state.cue_key, positive_state.cue_key),
        (negative_state.outcome_key, positive_state.outcome_key),
        (negative_state.current_signals, positive_state.current_signals),
        (negative_state.current_cues, positive_state.current_cues),
        (negative_state.world_sign, positive_state.world_sign),
    ):
        chex.assert_trees_all_equal(left, right)
    assert float(negative.reward + positive.reward) == pytest.approx(1.0)
    chex.assert_trees_all_equal(
        negative.reward,
        negative.oracle.counterfactual_rewards[NEGATIVE_ACTION],
    )
    chex.assert_trees_all_equal(
        positive.reward,
        positive.oracle.counterfactual_rewards[POSITIVE_ACTION],
    )
    chex.assert_trees_all_equal(
        negative.oracle.counterfactual_rewards,
        positive.oracle.counterfactual_rewards,
    )
    masked_negative = negative.next_observation.at[PREVIOUS_OUTCOME_INDEX].set(0.0)
    masked_positive = positive.next_observation.at[PREVIOUS_OUTCOME_INDEX].set(0.0)
    chex.assert_trees_all_equal(masked_negative, masked_positive)


def test_outcome_noise_is_sampled_independently_and_preserves_reward_semantics() -> None:
    world = HiddenPartnerWorldFeedbackWorld(_small_config(outcome_flip_probability=1.0))
    state = world.init(jr.key(71)).replace(
        world_sign=jnp.asarray(1.0, dtype=jnp.float32),
        current_signals=jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        current_cues=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
    )

    transition, next_state = world.step(
        state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )

    assert bool(transition.oracle.outcome_flipped)
    assert float(transition.oracle.noiseless_contextual_outcome) == 1.0
    assert float(transition.outcome) == -1.0
    assert float(transition.reward) == 0.0
    assert float(transition.reward) == (1.0 + float(transition.outcome)) / 2.0
    assert float(next_state.previous_outcome) == -1.0


@pytest.mark.parametrize("world_sign", (-1.0, 1.0))
@pytest.mark.parametrize("focal_action", (NEGATIVE_ACTION, POSITIVE_ACTION))
@pytest.mark.parametrize("partner_sign", (-1.0, 1.0))
@pytest.mark.parametrize("outcome_flipped", (False, True))
def test_contextual_truth_table_exhausts_world_action_partner_and_noise_signs(
    world_sign: float,
    focal_action: int,
    partner_sign: float,
    outcome_flipped: bool,
) -> None:
    world = HiddenPartnerWorldFeedbackWorld(
        _small_config(outcome_flip_probability=float(outcome_flipped))
    )
    state = world.init(jr.key(72)).replace(
        world_sign=jnp.asarray(world_sign, dtype=jnp.float32),
        current_signals=jnp.asarray((partner_sign, 1.0, 1.0), dtype=jnp.float32),
    )
    transition, _ = world.step(
        state,
        jnp.asarray(focal_action, dtype=jnp.int32),
    )
    focal_sign = 2.0 * focal_action - 1.0
    outcome_noise_sign = -1.0 if outcome_flipped else 1.0
    expected = world_sign * focal_sign * partner_sign * outcome_noise_sign

    assert float(transition.oracle.partner_action_sign) == partner_sign
    assert bool(transition.oracle.outcome_flipped) is outcome_flipped
    assert float(transition.outcome) == expected
    assert float(transition.outcome) * focal_sign * partner_sign == world_sign * outcome_noise_sign
    assert float(transition.reward) == (1.0 + expected) / 2.0


def test_changing_one_noise_configuration_leaves_other_rng_streams_unchanged() -> None:
    key = jr.key(73)
    action = jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32)

    cue_worlds = (
        HiddenPartnerWorldFeedbackWorld(_small_config(cue_flip_probability=0.0)),
        HiddenPartnerWorldFeedbackWorld(_small_config(cue_flip_probability=1.0)),
    )
    cue_states = tuple(world.init(key) for world in cue_worlds)
    cue_steps = tuple(
        world.step(state, action) for world, state in zip(cue_worlds, cue_states, strict=True)
    )
    for left, right in (
        (cue_states[0].current_signals, cue_states[1].current_signals),
        (cue_states[0].world_sign, cue_states[1].world_sign),
        (cue_steps[0][0].oracle.partner_flipped, cue_steps[1][0].oracle.partner_flipped),
        (cue_steps[0][0].oracle.world_flipped, cue_steps[1][0].oracle.world_flipped),
        (cue_steps[0][0].oracle.outcome_flipped, cue_steps[1][0].oracle.outcome_flipped),
        (cue_steps[0][1].current_signals, cue_steps[1][1].current_signals),
        (cue_steps[0][1].partner_key, cue_steps[1][1].partner_key),
        (cue_steps[0][1].world_key, cue_steps[1][1].world_key),
        (cue_steps[0][1].outcome_key, cue_steps[1][1].outcome_key),
    ):
        chex.assert_trees_all_equal(left, right)

    outcome_worlds = (
        HiddenPartnerWorldFeedbackWorld(_small_config(outcome_flip_probability=0.0)),
        HiddenPartnerWorldFeedbackWorld(_small_config(outcome_flip_probability=1.0)),
    )
    outcome_steps = tuple(world.step(world.init(key), action) for world in outcome_worlds)
    for left, right in (
        (outcome_steps[0][0].oracle.partner_flipped, outcome_steps[1][0].oracle.partner_flipped),
        (outcome_steps[0][0].oracle.world_flipped, outcome_steps[1][0].oracle.world_flipped),
        (
            outcome_steps[0][0].oracle.next_world_cue_flipped,
            outcome_steps[1][0].oracle.next_world_cue_flipped,
        ),
        (outcome_steps[0][1].current_signals, outcome_steps[1][1].current_signals),
        (outcome_steps[0][1].current_cues, outcome_steps[1][1].current_cues),
        (outcome_steps[0][1].signal_key, outcome_steps[1][1].signal_key),
        (outcome_steps[0][1].cue_key, outcome_steps[1][1].cue_key),
    ):
        chex.assert_trees_all_equal(left, right)

    hazard_worlds = (
        HiddenPartnerWorldFeedbackWorld(_small_config(world_flip_probability=0.0)),
        HiddenPartnerWorldFeedbackWorld(_small_config(world_flip_probability=1.0)),
    )
    hazard_steps = tuple(world.step(world.init(key), action) for world in hazard_worlds)
    for left, right in (
        (hazard_steps[0][0].oracle.partner_flipped, hazard_steps[1][0].oracle.partner_flipped),
        (hazard_steps[0][0].oracle.outcome_flipped, hazard_steps[1][0].oracle.outcome_flipped),
        (
            hazard_steps[0][0].oracle.next_world_cue_flipped,
            hazard_steps[1][0].oracle.next_world_cue_flipped,
        ),
        (hazard_steps[0][1].current_signals, hazard_steps[1][1].current_signals),
        (hazard_steps[0][1].partner_key, hazard_steps[1][1].partner_key),
        (hazard_steps[0][1].outcome_key, hazard_steps[1][1].outcome_key),
    ):
        chex.assert_trees_all_equal(left, right)


def test_world_flip_occurs_after_current_outcome_and_before_next_cues() -> None:
    world = HiddenPartnerWorldFeedbackWorld(
        _small_config(world_flip_probability=1.0, cue_flip_probability=0.0)
    )
    state = world.init(jr.key(8)).replace(
        world_sign=jnp.asarray(1.0, dtype=jnp.float32),
        current_cues=jnp.asarray((1.0, 1.0), dtype=jnp.float32),
        current_signals=jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
    )
    transition, next_state = world.step(state, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))

    assert float(transition.oracle.world_sign) == 1.0
    assert bool(transition.oracle.world_flipped)
    assert float(transition.oracle.next_world_sign) == -1.0
    assert float(next_state.world_sign) == -1.0
    np.testing.assert_array_equal(next_state.current_cues, np.asarray((-1.0, -1.0)))
    np.testing.assert_array_equal(
        transition.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))],
        np.asarray((-1.0, -1.0)),
    )
    assert float(transition.outcome) == 1.0
    assert float(transition.reward) == 1.0


def test_same_current_observation_can_require_opposite_actions_due_to_recurrent_history() -> None:
    # With uninformative cues, both latent states and this identical current
    # observation are reachable; the learner's own prior focal action disambiguates
    # their noisy feedback histories.
    world = HiddenPartnerWorldFeedbackWorld(_small_config(cue_flip_probability=0.5))
    base = world.init(jr.key(9)).replace(
        current_signals=jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
        current_cues=jnp.asarray((1.0, -1.0), dtype=jnp.float32),
        previous_outcome=jnp.asarray(1.0, dtype=jnp.float32),
        previous_partner_action=jnp.asarray(1, dtype=jnp.int32),
        has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
    )
    positive_history = base.replace(world_sign=jnp.asarray(1.0, dtype=jnp.float32))
    negative_history = base.replace(world_sign=jnp.asarray(-1.0, dtype=jnp.float32))
    chex.assert_trees_all_equal(
        world.observe(positive_history),
        world.observe(negative_history),
    )

    positive, _ = world.step(positive_history, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))
    negative, _ = world.step(negative_history, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))
    assert int(positive.oracle.full_information_optimal_focal_action) != int(
        negative.oracle.full_information_optimal_focal_action
    )
    assert float(positive.reward) != float(negative.reward)


def test_full_information_oracle_accounts_for_noise_polarity_and_ties() -> None:
    inverted_world = HiddenPartnerWorldFeedbackWorld(
        _small_config(
            partner_flip_probability=0.8,
            outcome_flip_probability=0.1,
        )
    )
    state = inverted_world.init(jr.key(91)).replace(
        world_sign=jnp.asarray(1.0, dtype=jnp.float32),
        current_signals=jnp.asarray((1.0, 1.0, 1.0), dtype=jnp.float32),
    )
    transition, _ = inverted_world.step(
        state,
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    assert float(transition.oracle.full_information_action_margin) == pytest.approx(0.48)
    assert not bool(transition.oracle.full_information_action_tied)
    assert int(transition.oracle.full_information_optimal_focal_action) == NEGATIVE_ACTION

    tied_world = HiddenPartnerWorldFeedbackWorld(
        _small_config(
            partner_flip_probability=0.5,
            outcome_flip_probability=0.1,
        )
    )
    tied, _ = tied_world.step(
        tied_world.init(jr.key(92)),
        jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32),
    )
    assert float(tied.oracle.full_information_action_margin) == 0.0
    assert bool(tied.oracle.full_information_action_tied)


@pytest.mark.parametrize(
    ("segment_index", "expected_regime", "expected_partner_sign"),
    [
        (0, REGIME_A, 1.0),
        (1, REGIME_B, -1.0),
        (3, REGIME_D, -1.0),
        (5, REGIME_C, -1.0),
    ],
)
def test_partner_regime_formulas_preserve_c_and_d_feature_questions(
    segment_index: int,
    expected_regime: int,
    expected_partner_sign: float,
) -> None:
    world = HiddenPartnerWorldFeedbackWorld(_small_config(lengths=(2,) * 9))
    state = world.init(jr.key(10)).replace(
        current_signals=jnp.asarray((1.0, 1.0, -1.0), dtype=jnp.float32),
        previous_partner_action=jnp.asarray(-1, dtype=jnp.int32),
        previous_outcome=jnp.asarray(1.0, dtype=jnp.float32),
        has_partner_history=jnp.asarray(True, dtype=jnp.bool_),
        world_sign=jnp.asarray(1.0, dtype=jnp.float32),
        step_count=jnp.asarray(2 * segment_index, dtype=jnp.int32),
    )
    transition, _ = world.step(state, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))

    assert int(transition.oracle.regime_id) == expected_regime
    assert float(transition.oracle.partner_intended_sign) == expected_partner_sign
    assert float(transition.oracle.partner_action_sign) == expected_partner_sign


def test_hidden_schedule_and_world_state_do_not_leak_or_reset_at_cycle_boundary() -> None:
    lengths = (2,) * 9
    world = HiddenPartnerWorldFeedbackWorld(
        _small_config(
            lengths=lengths,
            world_flip_probability=0.0,
            cue_flip_probability=0.0,
        )
    )
    state = world.init(jr.key(11)).replace(
        world_sign=jnp.asarray(-1.0, dtype=jnp.float32),
        current_cues=jnp.asarray((-1.0, -1.0), dtype=jnp.float32),
    )
    hidden_a = state.replace(step_count=jnp.asarray(0, dtype=jnp.int32))
    hidden_b = state.replace(step_count=jnp.asarray(2, dtype=jnp.int32))
    chex.assert_trees_all_equal(world.observe(hidden_a), world.observe(hidden_b))

    cycle_length = sum(lengths)
    transition: HiddenPartnerWorldFeedbackTransition | None = None
    for _ in range(cycle_length + 1):
        transition, state = world.step(state, jnp.asarray(POSITIVE_ACTION, dtype=jnp.int32))
    assert transition is not None
    assert int(transition.oracle.cycle_index) == 1
    assert float(state.world_sign) == -1.0
    assert bool(state.has_partner_history)
    assert not bool(transition.terminated)
    assert float(transition.discount) == 1.0


def test_bayes_headroom_is_preregisterable_and_nontrivial() -> None:
    hazard = Fraction(3, 100)
    cue_1_error = Fraction(1, 4)
    cue_2_error = Fraction(7, 20)
    outcome_error = Fraction(3, 20)
    signs = (-1, 1)

    def bsc(observed: int, latent: int, error: Fraction) -> Fraction:
        return 1 - error if observed == latent else error

    # Learner-visible order at decision 2:
    # C0 -> D1 -> hazard -> C1 -> D2 -> hazard -> C2.  D is the
    # action/partner-corrected contextual outcome, a noisy measurement of z.
    history_mass: dict[tuple[int, ...], list[Fraction]] = {}
    for values in product(signs, repeat=11):
        z_0, z_1, z_2, c_10, c_20, d_1, c_11, c_21, d_2, c_12, c_22 = values
        probability = Fraction(1, 2)
        probability *= bsc(c_10, z_0, cue_1_error)
        probability *= bsc(c_20, z_0, cue_2_error)
        probability *= bsc(d_1, z_0, outcome_error)
        probability *= bsc(z_1, z_0, hazard)
        probability *= bsc(c_11, z_1, cue_1_error)
        probability *= bsc(c_21, z_1, cue_2_error)
        probability *= bsc(d_2, z_1, outcome_error)
        probability *= bsc(z_2, z_1, hazard)
        probability *= bsc(c_12, z_2, cue_1_error)
        probability *= bsc(c_22, z_2, cue_2_error)
        history = (c_10, c_20, d_1, c_11, c_21, d_2, c_12, c_22)
        posterior_joint = history_mass.setdefault(history, [Fraction(0), Fraction(0)])
        posterior_joint[0 if z_2 == -1 else 1] += probability

    full_history_expected_abs_mean = sum(
        abs(positive - negative) for negative, positive in history_mass.values()
    )
    cue_only_expected_abs_mean = Fraction(1, 2)
    latest_outcome_after_hazard_expected_abs_mean = Fraction(329, 500)
    latest_outcome_and_cues_expected_abs_mean = Fraction(13_593, 20_000)

    assert Fraction(str(DEFAULT_WORLD_FLIP_PROBABILITY)) == hazard
    assert tuple(Fraction(str(value)) for value in DEFAULT_CUE_FLIP_PROBABILITIES) == (
        cue_1_error,
        cue_2_error,
    )
    assert Fraction(str(DEFAULT_OUTCOME_FLIP_PROBABILITY)) == outcome_error
    assert full_history_expected_abs_mean == Fraction(2_647_614_237, 3_200_000_000)
    assert cue_only_expected_abs_mean < latest_outcome_after_hazard_expected_abs_mean
    assert (
        latest_outcome_after_hazard_expected_abs_mean
        < latest_outcome_and_cues_expected_abs_mean
        < full_history_expected_abs_mean
        < 1
    )
    assert (1 + full_history_expected_abs_mean) / 2 == Fraction(
        5_847_614_237,
        6_400_000_000,
    )


def test_step_is_jittable_and_scan_keeps_fixed_state_shape() -> None:
    world = HiddenPartnerWorldFeedbackWorld(_small_config())
    initial = world.init(jr.key(12))
    actions = jnp.asarray(
        (NEGATIVE_ACTION, POSITIVE_ACTION, POSITIVE_ACTION, NEGATIVE_ACTION),
        dtype=jnp.int32,
    )

    def scan_step(
        state: HiddenPartnerWorldFeedbackState,
        action: jax.Array,
    ) -> tuple[HiddenPartnerWorldFeedbackState, HiddenPartnerWorldFeedbackTransition]:
        transition, next_state = world.step(state, action)
        return next_state, transition

    eager = jax.lax.scan(scan_step, initial, actions)
    compiled = jax.jit(lambda state: jax.lax.scan(scan_step, state, actions))(initial)
    chex.assert_trees_all_close(
        _unwrap_prng_keys(eager),
        _unwrap_prng_keys(compiled),
        atol=0.0,
        rtol=0.0,
    )
    assert int(compiled[0].step_count) == len(actions)
    assert _tree_nbytes(compiled[0]) == world.resource_budget.state_nbytes


def test_action_static_contract_and_dynamic_invalidity_are_explicit() -> None:
    world = HiddenPartnerWorldFeedbackWorld(_small_config())
    state = world.init(jr.key(13))
    with pytest.raises(ValueError, match="scalar"):
        world.step(state, jnp.asarray([POSITIVE_ACTION], dtype=jnp.int32))
    with pytest.raises(TypeError, match="integer"):
        world.step(state, jnp.asarray(POSITIVE_ACTION, dtype=jnp.float32))
    transition, next_state = world.step(state, jnp.asarray(7, dtype=jnp.int32))
    assert bool(jnp.isnan(transition.reward))
    assert bool(jnp.isnan(transition.outcome))
    chex.assert_trees_all_equal(
        _unwrap_prng_keys(next_state),
        _unwrap_prng_keys(state),
    )
