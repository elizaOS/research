"""Unit contracts for the bounded WP5.6 prospective exploration selector."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.prospective_exploration import (
    PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY,
    PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA,
    PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL,
    PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS,
    PROSPECTIVE_EXPLORATION_OUTCOME_STATUS,
    PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM,
    PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY,
    PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE,
    PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED,
    ExplorationCandidateBatch,
    ProspectiveExploration,
    ProspectiveExplorationConfig,
    measure_prospective_exploration_state_nbytes,
)

pytestmark = pytest.mark.unit


def _digest(index: int) -> tuple[int, ...]:
    return (index, index + 1, index + 2, index + 3, 0, 0, 0, 0)


def _config(**overrides: object) -> ProspectiveExplorationConfig:
    values: dict[str, object] = {
        "n_actions": 4,
        "candidate_budget": 3,
        "mode": "expected_improvement_surprisal",
        "epsilon": 0.0,
        "host_surprisal_cap": 4.0,
        "max_expected_improvement": 100.0,
        "max_ensemble_disagreement": 100.0,
        "max_information_gain": 100.0,
        "max_learning_progress": 100.0,
        "source_owner_digest": _digest(10),
        "host_policy_owner_digest": _digest(20),
        "candidate_owner_digest": _digest(30),
        "score_owner_digest": _digest(40),
        "safety_owner_digest": _digest(50),
    }
    values.update(overrides)
    return ProspectiveExplorationConfig(**values)  # type: ignore[arg-type]


def _batch(
    config: ProspectiveExplorationConfig,
    *,
    event: int = 1,
    expected_improvement: tuple[float, float, float] = (1.0, 2.0, 1.0),
    ensemble_disagreement: tuple[float, float, float] = (3.0, 2.0, 1.0),
    information_gain: tuple[float, float, float] = (1.0, 3.0, 2.0),
    learning_progress: tuple[float, float, float] = (1.0, 2.0, 3.0),
    safety: tuple[bool, bool, bool] = (True, True, True),
    host_safe: bool = True,
) -> ExplorationCandidateBatch:
    assert config.candidate_budget == 3
    source = jnp.asarray([0, event], dtype=jnp.uint32)
    return ExplorationCandidateBatch(
        candidate_actions=jnp.asarray([0, 1, 2], dtype=jnp.int32),
        candidate_identity_words=jnp.asarray(
            [[event, 101], [event, 102], [event, 103]],
            dtype=jnp.uint32,
        ),
        candidate_valid=jnp.asarray([True, True, True], dtype=jnp.bool_),
        host_policy=jnp.asarray([0.6, 0.3, 0.1, 0.0], dtype=jnp.float32),
        host_action=jnp.asarray(0, dtype=jnp.int32),
        expected_improvement=jnp.asarray(expected_improvement, dtype=jnp.float32),
        ensemble_disagreement=jnp.asarray(ensemble_disagreement, dtype=jnp.float32),
        information_gain=jnp.asarray(information_gain, dtype=jnp.float32),
        learning_progress=jnp.asarray(learning_progress, dtype=jnp.float32),
        candidate_safety_allowed=jnp.asarray(safety, dtype=jnp.bool_),
        host_action_safety_allowed=jnp.asarray(host_safe, dtype=jnp.bool_),
        source_event_words=source,
        candidate_source_event_words=source,
        score_source_event_words=source,
        host_policy_source_event_words=source,
        safety_source_event_words=source,
        host_policy_revision_words=jnp.asarray([0, 5], dtype=jnp.uint32),
        candidate_revision_words=jnp.asarray([0, 7], dtype=jnp.uint32),
        score_revision_words=jnp.asarray([0, event], dtype=jnp.uint32),
        safety_revision_words=jnp.asarray([0, 9], dtype=jnp.uint32),
        source_owner_digest=jnp.asarray(config.source_owner_digest, dtype=jnp.uint32),
        host_policy_owner_digest=jnp.asarray(
            config.host_policy_owner_digest,
            dtype=jnp.uint32,
        ),
        candidate_owner_digest=jnp.asarray(
            config.candidate_owner_digest,
            dtype=jnp.uint32,
        ),
        score_owner_digest=jnp.asarray(config.score_owner_digest, dtype=jnp.uint32),
        safety_owner_digest=jnp.asarray(config.safety_owner_digest, dtype=jnp.uint32),
        causal_pre_decision_attested=jnp.asarray(True, dtype=jnp.bool_),
    )


def _assert_tree_equal(left: object, right: object) -> None:
    chex.assert_trees_all_equal(left, right)


def test_config_is_strict_and_discloses_theory_and_authority_boundaries() -> None:
    config = _config()
    payload = config.to_config()

    assert payload["evidence_level"] == PROSPECTIVE_EXPLORATION_EVIDENCE_LEVEL == "L0"
    assert payload["outcome_status"] == PROSPECTIVE_EXPLORATION_OUTCOME_STATUS == "not_assessed"
    assert payload["revealed_value_equivalence"] == (
        PROSPECTIVE_EXPLORATION_REVEALED_VALUE_EQUIVALENCE
    )
    assert payload["noisy_bandit_semantics"] == PROSPECTIVE_EXPLORATION_NOISY_BANDIT_SEMANTICS
    assert payload["candidate_budget_contract"] == (
        "same-fixed-budget-for-all-comparator-modes"
    )
    assert payload["candidate_generation_before_safety_shield"] is True
    assert payload["causal_attestation_is_proof"] is False
    assert PROSPECTIVE_EXPLORATION_ACTION_DISPATCH_AUTHORITY is False
    assert PROSPECTIVE_EXPLORATION_POLICY_OVERRIDE_AUTHORITY is False
    assert PROSPECTIVE_EXPLORATION_PHYSICAL_SAFETY_CLAIM is False
    assert PROSPECTIVE_EXPLORATION_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert ProspectiveExplorationConfig.from_config(payload) == config

    legacy_payload = dict(payload)
    legacy_payload.update(
        type="DelightfulExploration",
        schema="alberta.delightful-exploration.config.v1",
        mode="prospective_delight",
    )
    assert ProspectiveExplorationConfig.from_config(legacy_payload) == config

    forged = dict(payload)
    forged["outcome_status"] = "accepted"
    with pytest.raises(ValueError, match="outcome_status"):
        ProspectiveExplorationConfig.from_config(forged)
    with pytest.raises((TypeError, ValueError)):
        _config(epsilon=1)
    with pytest.raises(ValueError):
        _config(candidate_budget=5)
    with pytest.raises(ValueError):
        _config(n_actions=1, candidate_budget=1)
    with pytest.raises(ValueError):
        _config(mode="unknown")
    with pytest.raises(ValueError, match="distinct"):
        _config(safety_owner_digest=_digest(10))


def test_prospective_score_matches_expected_improvement_times_capped_surprisal() -> None:
    config = _config(host_surprisal_cap=2.0)
    controller = ProspectiveExploration(config)
    state = controller.init(jr.key(1))
    batch = _batch(config, expected_improvement=(1.0, 2.0, 1.5))
    result = controller.select(state, batch)

    expected_surprisal = np.minimum(-np.log(np.asarray([0.6, 0.3, 0.1])), 2.0)
    expected_score = np.asarray([1.0, 2.0, 1.5]) * expected_surprisal
    np.testing.assert_allclose(result.host_relative_surprisal, expected_surprisal, rtol=2e-6)
    np.testing.assert_allclose(
        result.expected_improvement_surprisal_score,
        expected_score,
        rtol=2e-6,
    )
    assert int(result.selected_index) == int(np.argmax(expected_score)) == 2
    assert float(result.selected_expected_improvement_surprisal_score) == pytest.approx(
        expected_score[2],
        rel=2e-6,
    )


def test_zero_host_probability_is_finite_and_exactly_capped() -> None:
    config = _config(host_surprisal_cap=3.0)
    controller = ProspectiveExploration(config)
    batch = dataclasses.replace(
        _batch(config),
        candidate_actions=jnp.asarray([0, 1, 3], dtype=jnp.int32),
    )
    result = controller.select(controller.init(jr.key(2)), batch)

    assert bool(result.decision_applied)
    assert float(result.host_relative_surprisal[2]) == 3.0
    assert bool(
        jnp.all(jnp.isfinite(result.expected_improvement_surprisal_score))
    )


@pytest.mark.parametrize(
    ("mode", "epsilon", "expected_index"),
    [
        ("epsilon_greedy", 0.0, 1),
        ("ensemble_disagreement", 0.0, 0),
        ("information_gain", 0.0, 1),
        ("learning_progress", 0.0, 2),
    ],
)
def test_deterministic_comparators_rank_their_declared_signal(
    mode: str,
    epsilon: float,
    expected_index: int,
) -> None:
    config = _config(mode=mode, epsilon=epsilon)
    controller = ProspectiveExploration(config)
    result = controller.select(controller.init(jr.key(3)), _batch(config))
    assert bool(result.decision_applied)
    assert int(result.selected_index) == expected_index


def test_random_and_forced_random_epsilon_are_reproducible_and_ignore_padding() -> None:
    for mode, epsilon in (("random", 0.0), ("epsilon_greedy", 1.0)):
        config = _config(mode=mode, epsilon=epsilon)
        controller = ProspectiveExploration(config)
        state = controller.init(jr.key(4))
        base = _batch(config)
        padded = dataclasses.replace(
            base,
            candidate_actions=jnp.asarray([0, 1, -1], dtype=jnp.int32),
            candidate_identity_words=jnp.asarray([[1, 101], [1, 102], [0, 0]], dtype=jnp.uint32),
            candidate_valid=jnp.asarray([True, True, False], dtype=jnp.bool_),
            expected_improvement=jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32),
            ensemble_disagreement=jnp.asarray([3.0, 2.0, 0.0], dtype=jnp.float32),
            information_gain=jnp.asarray([1.0, 3.0, 0.0], dtype=jnp.float32),
            learning_progress=jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32),
            candidate_safety_allowed=jnp.asarray([True, True, False], dtype=jnp.bool_),
        )
        first = controller.select(state, padded)
        second = controller.select(state, padded)
        _assert_tree_equal(first, second)
        assert int(first.selected_index) in (0, 1)


def test_candidate_generation_precedes_hard_shield_and_unsafe_host_fails_closed() -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    state = controller.init(jr.key(5))
    # Candidate 1 has the highest prospective score. The shield rejects it,
    # so the independently shielded host action is the only proposal.
    rejected = controller.select(
        state,
        _batch(config, safety=(True, False, True), host_safe=True),
    )
    accepted = controller.select(
        state,
        _batch(config, safety=(True, True, True), host_safe=True),
    )
    unavailable = controller.select(
        state,
        _batch(config, safety=(True, False, True), host_safe=False),
    )

    assert int(rejected.selected_index) == int(accepted.selected_index) == 1
    assert bool(rejected.candidate_generated)
    assert not bool(rejected.candidate_passed_hard_shield)
    assert bool(rejected.host_fallback_used)
    assert int(rejected.proposed_executable_action) == 0
    assert bool(accepted.candidate_override_proposed)
    assert int(accepted.proposed_executable_action) == 1
    assert not bool(unavailable.proposed_executable_action_available)
    assert int(unavailable.proposed_executable_action) == -1


@pytest.mark.parametrize("invalid_kind", ["owner", "causal", "policy", "score", "source"])
def test_invalid_dynamic_receipts_are_atomic_noops(invalid_kind: str) -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    state = controller.init(jr.key(6))
    batch = _batch(config)
    if invalid_kind == "owner":
        batch = dataclasses.replace(
            batch,
            score_owner_digest=jnp.asarray(_digest(99), dtype=jnp.uint32),
        )
    elif invalid_kind == "causal":
        batch = dataclasses.replace(
            batch,
            causal_pre_decision_attested=jnp.asarray(False, dtype=jnp.bool_),
        )
    elif invalid_kind == "policy":
        batch = dataclasses.replace(
            batch,
            host_policy=jnp.asarray([0.5, 0.3, 0.1, 0.0], dtype=jnp.float32),
        )
    elif invalid_kind == "score":
        batch = dataclasses.replace(
            batch,
            information_gain=jnp.asarray([1.0, jnp.nan, 2.0], dtype=jnp.float32),
        )
    else:
        batch = dataclasses.replace(
            batch,
            score_source_event_words=jnp.asarray([0, 2], dtype=jnp.uint32),
        )
    result = controller.select(state, batch)

    assert not bool(result.decision_applied)
    assert int(result.selected_index) == -1
    assert not bool(result.proposed_executable_action_available)
    _assert_tree_equal(result.state, state)


def test_stale_source_and_revision_rollback_are_atomic_noops() -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    initial = controller.init(jr.key(7))
    first = controller.select(initial, _batch(config, event=2))
    assert bool(first.decision_applied)

    stale = controller.select(first.state, _batch(config, event=2))
    rolled_back = controller.select(
        first.state,
        dataclasses.replace(
            _batch(config, event=3),
            host_policy_revision_words=jnp.asarray([0, 4], dtype=jnp.uint32),
        ),
    )
    assert not bool(stale.causal_binding_valid)
    assert not bool(rolled_back.causal_binding_valid)
    _assert_tree_equal(stale.state, first.state)
    _assert_tree_equal(rolled_back.state, first.state)


def test_duplicate_and_noncanonical_padded_candidates_fail_closed() -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    state = controller.init(jr.key(8))
    base = _batch(config)
    duplicate = dataclasses.replace(
        base,
        candidate_actions=jnp.asarray([0, 0, 2], dtype=jnp.int32),
    )
    bad_padding = dataclasses.replace(
        base,
        candidate_actions=jnp.asarray([0, 1, -1], dtype=jnp.int32),
        candidate_identity_words=jnp.asarray([[1, 101], [1, 102], [0, 0]], dtype=jnp.uint32),
        candidate_valid=jnp.asarray([True, True, False], dtype=jnp.bool_),
        expected_improvement=jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32),
        ensemble_disagreement=jnp.asarray([3.0, 2.0, 0.0], dtype=jnp.float32),
        information_gain=jnp.asarray([1.0, 3.0, 0.0], dtype=jnp.float32),
        learning_progress=jnp.asarray([1.0, 2.0, 0.0], dtype=jnp.float32),
        candidate_safety_allowed=jnp.asarray([True, True, True], dtype=jnp.bool_),
    )
    for batch in (duplicate, bad_padding):
        result = controller.select(state, batch)
        assert not bool(result.candidate_batch_valid)
        assert not bool(result.decision_applied)
        _assert_tree_equal(result.state, state)


def test_exact_clock_carries_and_fails_stop_at_uint64_capacity() -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    initial = controller.init(jr.key(9))
    near_carry = dataclasses.replace(
        initial,
        decision_words=jnp.asarray([0, 0xFFFFFFFF], dtype=jnp.uint32),
    )
    carried = controller.select(near_carry, _batch(config))
    np.testing.assert_array_equal(carried.state.decision_words, [1, 0])

    maximum = dataclasses.replace(
        initial,
        decision_words=jnp.asarray([0xFFFFFFFF, 0xFFFFFFFF], dtype=jnp.uint32),
    )
    exhausted = controller.select(maximum, _batch(config))
    assert not bool(exhausted.lifetime_capacity_available)
    assert not bool(exhausted.decision_applied)
    _assert_tree_equal(exhausted.state, maximum)


def test_typed_threefry_key_is_required() -> None:
    controller = ProspectiveExploration(_config())
    with pytest.raises(TypeError, match="typed Threefry"):
        controller.init(jr.PRNGKey(1))
    with pytest.raises(TypeError, match="typed Threefry"):
        controller.init(jr.key(1, impl="rbg"))


def test_resource_partition_is_exact_and_modes_have_the_same_fixed_budget() -> None:
    budgets = []
    for mode in (
        "expected_improvement_surprisal",
        "random",
        "epsilon_greedy",
        "ensemble_disagreement",
        "information_gain",
        "learning_progress",
    ):
        controller = ProspectiveExploration(_config(mode=mode))
        state = controller.init(jr.key(10))
        budget = controller.resource_budget(state)
        assert budget.total_state_nbytes == measure_prospective_exploration_state_nbytes(state)
        assert budget.fixed_candidate_budget == 3
        assert budget.logical_uniform_draws_per_decision == 4
        assert "not-a-measured-device-peak" in budget.temporary_bytes_scope
        budgets.append(budget)
    assert len({budget.total_state_nbytes for budget in budgets}) == 1
    assert len({budget.candidate_metric_scalars_per_decision for budget in budgets}) == 1


def test_eager_and_jit_selection_are_bit_exact() -> None:
    config = _config(mode="epsilon_greedy", epsilon=0.35)
    controller = ProspectiveExploration(config)
    state = controller.init(jr.key(11))
    batch = _batch(config)
    eager = controller.select(state, batch)
    compiled = jax.jit(controller.select)(state, batch)
    _assert_tree_equal(eager, compiled)


def test_checkpoint_roundtrip_is_construction_bound(tmp_path: Path) -> None:
    config = _config()
    controller = ProspectiveExploration(config)
    state = controller.select(controller.init(jr.key(12)), _batch(config)).state
    path = tmp_path / "exploration.ckpt"
    controller.save_checkpoint(state, path)
    metadata = controller.checkpoint_metadata(path)
    restored = controller.load_checkpoint(controller.init(jr.key(0)), path)

    assert metadata["schema"] == PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA
    assert metadata["construction"] == config.to_config()
    assert metadata["resource_budget"] == dataclasses.asdict(
        controller.resource_budget(state)
    )
    _assert_tree_equal(restored, state)

    incompatible = ProspectiveExploration(_config(mode="random"))
    with pytest.raises(ValueError, match="construction"):
        incompatible.checkpoint_metadata(path)
