"""Exact learner-visible Bayes oracle for the noisy hidden-world stream."""

from __future__ import annotations

import json
from fractions import Fraction
from itertools import product

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.evaluation.hidden_partner_world_filter import (
    HiddenPartnerWorldBayesFilter,
    HiddenPartnerWorldFilterConfig,
    HiddenPartnerWorldFilterState,
    HiddenPartnerWorldRewardCells,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    CUE_1_INDEX,
    CUE_2_INDEX,
    HiddenPartnerWorldFeedbackConfig,
    HiddenPartnerWorldFeedbackWorld,
)

pytestmark = pytest.mark.unit

_SIGNS = (-1, 1)


def _bsc(observed: int, latent: int, error: Fraction) -> Fraction:
    return 1 - error if observed == latent else error


def _exact_history_posterior(
    observations: tuple[int, int, int, int, int, int, int, int],
) -> tuple[Fraction, Fraction]:
    c_10, c_20, d_1, c_11, c_21, d_2, c_12, c_22 = observations
    hazard = Fraction(3, 100)
    cue_1_error = Fraction(1, 4)
    cue_2_error = Fraction(7, 20)
    outcome_error = Fraction(3, 20)
    negative = Fraction(0)
    positive = Fraction(0)
    for z_0, z_1, z_2 in product(_SIGNS, repeat=3):
        probability = Fraction(1, 2)
        probability *= _bsc(c_10, z_0, cue_1_error)
        probability *= _bsc(c_20, z_0, cue_2_error)
        probability *= _bsc(d_1, z_0, outcome_error)
        probability *= _bsc(z_1, z_0, hazard)
        probability *= _bsc(c_11, z_1, cue_1_error)
        probability *= _bsc(c_21, z_1, cue_2_error)
        probability *= _bsc(d_2, z_1, outcome_error)
        probability *= _bsc(z_2, z_1, hazard)
        probability *= _bsc(c_12, z_2, cue_1_error)
        probability *= _bsc(c_22, z_2, cue_2_error)
        if z_2 == 1:
            positive += probability
        else:
            negative += probability
    evidence = negative + positive
    return (positive - negative) / evidence, evidence


def test_filter_config_is_strict_canonical_and_roundtrips() -> None:
    config = HiddenPartnerWorldFilterConfig()
    encoded = config.canonical_json()

    assert json.loads(encoded) == config.to_config()
    assert HiddenPartnerWorldFilterConfig.from_config(json.loads(encoded)) == config
    assert config.world_flip_probability == 0.03
    assert config.cue_flip_probabilities == (0.25, 0.35)
    assert config.outcome_flip_probability == 0.15
    rational = HiddenPartnerWorldFilterConfig(
        world_flip_probability=Fraction(3, 100),
        cue_flip_probabilities=(Fraction(1, 4), Fraction(7, 20)),
        outcome_flip_probability=Fraction(3, 20),
    )
    assert rational == config
    assert all(type(value) is float for value in rational.cue_flip_probabilities)
    assert type(rational.world_flip_probability) is float
    assert type(rational.outcome_flip_probability) is float
    assert HiddenPartnerWorldFilterConfig.from_config(rational.to_config()) == rational
    assert (
        HiddenPartnerWorldFilterConfig.from_world_config(HiddenPartnerWorldFeedbackConfig())
        == config
    )
    with pytest.raises(ValueError):
        HiddenPartnerWorldFilterConfig.from_world_config(
            HiddenPartnerWorldFeedbackConfig(outcome_flip_probability=0.0)
        )

    for malformed in (
        {**config.to_config(), "future": 1},
        {key: value for key, value in config.to_config().items() if key != "type"},
        {**config.to_config(), "type": "OtherFilter"},
    ):
        with pytest.raises(ValueError):
            HiddenPartnerWorldFilterConfig.from_config(malformed)
    for kwargs in (
        {"world_flip_probability": True},
        {"world_flip_probability": -0.1},
        {"cue_flip_probabilities": [0.25, 0.35]},
        {"cue_flip_probabilities": (0.25,)},
        {"cue_flip_probabilities": (0.0, 0.35)},
        {"cue_flip_probabilities": (1e-50, 0.35)},
        {"outcome_flip_probability": 0.5},
        {"outcome_flip_probability": 1e-50},
    ):
        with pytest.raises(ValueError):
            HiddenPartnerWorldFilterConfig(**kwargs)


def test_recursive_filter_matches_fraction_oracle_for_all_256_histories() -> None:
    world_filter = HiddenPartnerWorldBayesFilter()
    total_evidence = Fraction(0)
    expected_abs_mean = Fraction(0)

    for observations in product(_SIGNS, repeat=8):
        c_10, c_20, d_1, c_11, c_21, d_2, c_12, c_22 = observations
        state = world_filter.initialize(jnp.asarray((c_10, c_20), dtype=jnp.float32))
        first = world_filter.advance(
            state,
            jnp.asarray(d_1, dtype=jnp.float32),
            jnp.asarray((c_11, c_21), dtype=jnp.float32),
        )
        second = world_filter.advance(
            first.state,
            jnp.asarray(d_2, dtype=jnp.float32),
            jnp.asarray((c_12, c_22), dtype=jnp.float32),
        )
        exact_mean, evidence = _exact_history_posterior(observations)

        assert bool(state.valid & first.valid & second.valid)
        assert float(second.state.posterior_mean) == pytest.approx(
            float(exact_mean),
            abs=2e-6,
        )
        total_evidence += evidence
        expected_abs_mean += evidence * abs(exact_mean)

    assert total_evidence == 1
    assert expected_abs_mean == Fraction(2_647_614_237, 3_200_000_000)


def test_reward_cells_and_partner_marginal_define_exact_decision_fidelity() -> None:
    world_filter = HiddenPartnerWorldBayesFilter()
    cells = world_filter.expected_reward_cells(jnp.asarray(0.6, dtype=jnp.float32))
    expected = np.asarray(((0.71, 0.29), (0.29, 0.71)), dtype=np.float32)
    np.testing.assert_allclose(cells.rewards, expected, atol=1e-7, rtol=0.0)
    assert bool(cells.valid)

    decision = world_filter.marginalize_partner(
        cells,
        jnp.asarray((0.05, 0.95), dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        decision.expected_rewards,
        np.asarray((0.311, 0.689), dtype=np.float32),
        atol=1e-6,
        rtol=0.0,
    )
    assert int(decision.greedy_action) == 1
    assert float(decision.optimal_value) == pytest.approx(0.689, abs=1e-6)
    assert float(decision.action_regrets[1]) == pytest.approx(0.0)
    assert float(decision.action_regrets[0]) == pytest.approx(0.378, abs=1e-6)
    assert float(decision.action_margin) == pytest.approx(0.378, abs=1e-6)
    assert not bool(decision.tied)
    assert bool(decision.valid)

    asymmetric = HiddenPartnerWorldRewardCells(
        rewards=jnp.asarray(((0.1, 0.3), (0.6, 0.9)), dtype=jnp.float32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    asymmetric_decision = world_filter.marginalize_partner(
        asymmetric,
        jnp.asarray((0.25, 0.75), dtype=jnp.float32),
    )
    np.testing.assert_allclose(
        asymmetric_decision.expected_rewards,
        np.asarray((0.25, 0.825), dtype=np.float32),
        atol=1e-7,
        rtol=0.0,
    )
    assert int(asymmetric_decision.greedy_action) == 1

    tied = world_filter.marginalize_partner(
        world_filter.expected_reward_cells(jnp.asarray(0.0, dtype=jnp.float32)),
        jnp.asarray((0.5, 0.5), dtype=jnp.float32),
    )
    assert bool(tied.tied)
    assert float(tied.action_margin) == 0.0
    assert int(tied.greedy_action) == 0


def test_stream_transitions_feed_the_same_per_history_recursive_posterior() -> None:
    world = HiddenPartnerWorldFeedbackWorld()
    world_filter = HiddenPartnerWorldBayesFilter()
    world_state = world.init(jr.key(2026))
    initial_observation = world.observe(world_state)
    cues_0 = tuple(
        int(value) for value in initial_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
    )
    filter_state = world_filter.initialize(
        initial_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
    )
    history: list[int] = [*cues_0]

    for action in (0, 1):
        transition, world_state = world.step(
            world_state,
            jnp.asarray(action, dtype=jnp.int32),
        )
        focal_sign = 2 * int(transition.focal_action) - 1
        partner_sign = 2 * int(transition.partner_action) - 1
        corrected_outcome = int(transition.outcome) * focal_sign * partner_sign
        next_cues = transition.next_observation[jnp.asarray((CUE_1_INDEX, CUE_2_INDEX))]
        history.extend((corrected_outcome, *(int(value) for value in next_cues)))
        filter_state = world_filter.advance(
            filter_state,
            jnp.asarray(corrected_outcome, dtype=jnp.float32),
            next_cues,
        ).state

    exact_mean, _ = _exact_history_posterior(tuple(history))  # type: ignore[arg-type]
    assert bool(filter_state.valid)
    assert float(filter_state.posterior_mean) == pytest.approx(float(exact_mean), abs=2e-6)


def test_invalid_dynamic_evidence_fails_closed_and_jit_matches_eager() -> None:
    world_filter = HiddenPartnerWorldBayesFilter()
    state = world_filter.initialize(jnp.asarray((1.0, -1.0), dtype=jnp.float32))
    invalid = world_filter.advance(
        state,
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray((1.0, 1.0), dtype=jnp.float32),
    )
    assert not bool(invalid.valid)
    assert not bool(invalid.state.valid)
    chex.assert_trees_all_equal(invalid.state.posterior_mean, state.posterior_mean)
    chex.assert_trees_all_equal(invalid.state.step_count, state.step_count)

    eager = world_filter.advance(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((-1.0, 1.0), dtype=jnp.float32),
    )
    compiled = jax.jit(world_filter.advance)(
        state,
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray((-1.0, 1.0), dtype=jnp.float32),
    )
    chex.assert_trees_all_close(eager, compiled, atol=0.0, rtol=0.0)
    assert isinstance(compiled.state, HiddenPartnerWorldFilterState)

    malformed_state = state.replace(posterior_mean=jnp.asarray((0.0, 0.0), dtype=jnp.float32))
    with pytest.raises(ValueError, match="state.posterior_mean"):
        world_filter.advance(
            malformed_state,
            jnp.asarray(1.0, dtype=jnp.float32),
            jnp.asarray((1.0, -1.0), dtype=jnp.float32),
        )
    malformed_cells = HiddenPartnerWorldRewardCells(
        rewards=jnp.ones((4,), dtype=jnp.float32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    with pytest.raises(ValueError, match="reward_cells.rewards"):
        world_filter.marginalize_partner(
            malformed_cells,
            jnp.asarray((0.5, 0.5), dtype=jnp.float32),
        )


def test_partner_probability_tolerance_is_normalized_before_marginalization() -> None:
    world_filter = HiddenPartnerWorldBayesFilter()
    cells = world_filter.expected_reward_cells(jnp.asarray(1.0, dtype=jnp.float32))
    probabilities = jnp.asarray((0.05, 0.9500008), dtype=jnp.float32)
    decision = world_filter.marginalize_partner(cells, probabilities)

    assert bool(decision.valid)
    assert bool(jnp.all(decision.expected_rewards >= 0.0))
    assert bool(jnp.all(decision.expected_rewards <= 1.0))
    assert 0.0 <= float(decision.optimal_value) <= 1.0
