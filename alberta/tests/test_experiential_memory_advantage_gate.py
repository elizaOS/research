# mypy: disable-error-code="call-arg"
"""Conservative action-conditioned authority for experiential memory."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework as alberta
import alberta_framework.core as core
from alberta_framework.core.experiential_memory import (
    ExperientialMemory,
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
    ExperientialMemoryState,
)
from alberta_framework.core.experiential_memory_policy import (
    EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA,
    EXPERIENTIAL_MEMORY_POLICY_SCHEMA,
    ExperientialMemoryAdvantageGate,
    ExperientialMemoryAdvantageGateConfig,
    ExperientialMemoryPolicy,
)

pytestmark = pytest.mark.unit


def _memory_config(**overrides: Any) -> ExperientialMemoryConfig:
    values: dict[str, Any] = {
        "capacity": 4,
        "observation_dim": 2,
        "key_dim": 2,
        "action_dim": 2,
        "outcome_dim": 1,
        "top_k": 4,
        "min_neighbors": 1,
        "distance_scale": 1.0,
        "min_similarity": 0.0,
        "min_effective_reliability": 0.01,
        "max_uncertainty": 1.0,
        "max_safety_cost": 1.0,
        "max_age": 100,
        "staleness_scale": 1_000_000.0,
        "utility_decay": 1.0,
        "eviction_utility_weight": 1.0,
        "eviction_recency_weight": 1.0,
        "recency_scale": 100.0,
    }
    values.update(overrides)
    return ExperientialMemoryConfig(**values)


def _entry(
    provenance_id: int,
    *,
    action: tuple[float, float],
    reward: float,
) -> ExperientialMemoryEntry:
    zero = jnp.zeros((2,), dtype=jnp.float32)
    return ExperientialMemoryEntry(
        observation=zero,
        key=zero,
        action=jnp.asarray(action, dtype=jnp.float32),
        outcome=jnp.asarray([reward], dtype=jnp.float32),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        uncertainty=jnp.asarray(0.0, dtype=jnp.float32),
        uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
        safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
        safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
        reliability=jnp.asarray(1.0, dtype=jnp.float32),
        utility=jnp.asarray(max(reward, 0.0), dtype=jnp.float32),
        utility_available=jnp.asarray(True, dtype=jnp.bool_),
        representation_version=jnp.asarray(1, dtype=jnp.int32),
        valid=jnp.asarray(True, dtype=jnp.bool_),
        age=jnp.asarray(0, dtype=jnp.int32),
        provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
        source_id=jnp.asarray(1, dtype=jnp.int32),
    )


def _seed(
    memory: ExperientialMemory,
    rows: tuple[tuple[tuple[float, float], float], ...],
) -> ExperientialMemoryState:
    state = memory.init()
    for provenance_id, (action, reward) in enumerate(rows):
        result = memory.write(
            state,
            _entry(provenance_id, action=action, reward=reward),
        )
        assert bool(result.wrote)
        state = result.state
    return state


def _proposal(
    memory: ExperientialMemory,
    state: ExperientialMemoryState,
) -> Any:
    return ExperientialMemoryPolicy(memory).propose(
        state,
        jnp.zeros((2,), dtype=jnp.float32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(0.0, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        jnp.ones((2,), dtype=jnp.bool_),
    )


def _assert_trees_equal(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert str(left_tree) == str(right_tree)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_config_roundtrip_exports_resources_and_legacy_policy_schema() -> None:
    memory = ExperientialMemory(_memory_config())
    config = ExperientialMemoryAdvantageGateConfig(
        min_action_support=2,
        min_action_weight_mass=0.2,
        min_reward_advantage=0.25,
    )
    payload = config.to_config()

    assert payload["schema"] == EXPERIENTIAL_MEMORY_ADVANTAGE_GATE_CONFIG_SCHEMA
    assert ExperientialMemoryAdvantageGateConfig.from_config(payload) == config
    gate = ExperientialMemoryAdvantageGate.from_config(memory, payload)
    assert gate.to_config() == payload
    assert alberta.ExperientialMemoryAdvantageGate is ExperientialMemoryAdvantageGate
    assert core.ExperientialMemoryAdvantageGateConfig is ExperientialMemoryAdvantageGateConfig
    assert ExperientialMemoryPolicy(memory).to_config()["schema"] == (
        EXPERIENTIAL_MEMORY_POLICY_SCHEMA
    )
    resources = gate.resource_declaration()
    assert resources.to_config() == {
        "n_actions": 2,
        "top_k": 4,
        "neighbor_action_values_interpreted": 8,
        "neighbor_reward_values_interpreted": 4,
        "neighbor_weight_values_interpreted": 4,
        "owned_persistent_state_bytes": 0,
        "random_draws_per_assessment": 0,
    }

    with pytest.raises(ValueError, match="positive exact int32"):
        ExperientialMemoryAdvantageGateConfig(min_action_support=0)
    with pytest.raises(ValueError, match="non-negative"):
        ExperientialMemoryAdvantageGateConfig(min_reward_advantage=-0.1)
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        ExperientialMemoryAdvantageGateConfig(min_action_weight_mass=0.0)
    with pytest.raises(ValueError, match="top_k"):
        ExperientialMemoryAdvantageGate(
            memory,
            ExperientialMemoryAdvantageGateConfig(min_action_support=5),
        )


@pytest.mark.parametrize(
    ("rows", "minimum_advantage", "expected_allowed", "expected_advantage"),
    [
        (
            (
                ((0.0, 1.0), 0.0),
                ((0.0, 1.0), 0.0),
                ((0.0, 1.0), 0.0),
                ((1.0, 0.0), 1.0),
            ),
            0.0,
            False,
            -1.0,
        ),
        (
            (
                ((0.0, 1.0), 1.0),
                ((0.0, 1.0), 1.0),
                ((0.0, 1.0), 1.0),
                ((1.0, 0.0), 0.0),
            ),
            0.0,
            True,
            1.0,
        ),
        (
            (
                ((0.0, 1.0), 1.0),
                ((0.0, 1.0), 1.0),
                ((0.0, 1.0), 1.0),
                ((1.0, 0.0), 1.0),
            ),
            0.0,
            False,
            0.0,
        ),
        (
            (
                ((0.0, 1.0), 0.6),
                ((0.0, 1.0), 0.6),
                ((0.0, 1.0), 0.6),
                ((1.0, 0.0), 0.5),
            ),
            0.2,
            False,
            0.1,
        ),
    ],
)
def test_gate_requires_supported_strict_local_reward_advantage(
    rows: tuple[tuple[tuple[float, float], float], ...],
    minimum_advantage: float,
    expected_allowed: bool,
    expected_advantage: float,
) -> None:
    memory = ExperientialMemory(_memory_config())
    state = _seed(memory, rows)
    proposal = _proposal(memory, state)
    assert bool(proposal.available)
    assert int(proposal.action) == 1
    gate = ExperientialMemoryAdvantageGate(
        memory,
        ExperientialMemoryAdvantageGateConfig(
            min_action_support=1,
            min_reward_advantage=float(minimum_advantage),
        ),
    )

    diagnostics = gate.assess(
        state,
        proposal,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert bool(diagnostics.evidence_valid)
    assert bool(diagnostics.support_ready)
    assert int(diagnostics.base_support_count) == 1
    assert int(diagnostics.proposed_support_count) == 3
    assert float(diagnostics.reward_advantage) == pytest.approx(
        expected_advantage,
        abs=1.0e-6,
    )
    assert bool(diagnostics.replacement_allowed) is expected_allowed


def test_missing_base_support_and_fractional_actions_fail_closed() -> None:
    memory = ExperientialMemory(_memory_config())
    gate = ExperientialMemoryAdvantageGate(
        memory,
        ExperientialMemoryAdvantageGateConfig(),
    )
    no_base = _seed(
        memory,
        tuple(((0.0, 1.0), 1.0) for _ in range(4)),
    )
    no_base_proposal = _proposal(memory, no_base)
    missing = gate.assess(
        no_base,
        no_base_proposal,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert bool(missing.evidence_valid)
    assert int(missing.base_support_count) == 0
    assert not bool(missing.support_ready)
    assert not bool(missing.replacement_allowed)

    fractional = _seed(
        memory,
        (
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((0.5, 0.5), 0.0),
        ),
    )
    fractional_proposal = _proposal(memory, fractional)
    invalid = gate.assess(
        fractional,
        fractional_proposal,
        jnp.asarray(0, dtype=jnp.int32),
    )
    assert not bool(invalid.evidence_valid)
    assert not bool(invalid.replacement_allowed)
    assert not bool(jnp.all(invalid.neighbor_action_one_hot))


def test_raw_neighbor_count_cannot_bypass_minimum_action_weight_mass() -> None:
    memory = ExperientialMemory(_memory_config())
    state = _seed(
        memory,
        (
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 0.0),
        ),
    )
    proposal = _proposal(memory, state)
    gate = ExperientialMemoryAdvantageGate(
        memory,
        ExperientialMemoryAdvantageGateConfig(
            min_action_support=1,
            min_action_weight_mass=0.3,
            min_reward_advantage=0.0,
        ),
    )

    diagnostics = gate.assess(
        state,
        proposal,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert int(diagnostics.base_support_count) == 1
    assert int(diagnostics.proposed_support_count) == 3
    assert float(diagnostics.base_action_weight_mass) < 0.3
    assert float(diagnostics.proposed_action_weight_mass) > 0.7
    assert not bool(diagnostics.weight_mass_ready)
    assert not bool(diagnostics.support_ready)
    assert not bool(diagnostics.replacement_allowed)


def test_assessment_is_eager_jit_scan_exact_and_nonmutating() -> None:
    memory = ExperientialMemory(_memory_config())
    state = _seed(
        memory,
        (
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((0.0, 1.0), 1.0),
            ((1.0, 0.0), 0.0),
        ),
    )
    proposal = _proposal(memory, state)
    gate = ExperientialMemoryAdvantageGate(
        memory,
        ExperientialMemoryAdvantageGateConfig(),
    )
    base_action = jnp.asarray(0, dtype=jnp.int32)
    before = jax.tree.map(lambda value: np.asarray(value).copy(), state)

    eager = gate.assess(state, proposal, base_action)
    compiled = jax.jit(gate.assess)(state, proposal, base_action)
    _assert_trees_equal(eager, compiled)

    def body(
        carry: ExperientialMemoryState,
        action: jax.Array,
    ) -> tuple[ExperientialMemoryState, Any]:
        return carry, gate.assess(carry, proposal, action)

    final_state, scanned = jax.jit(
        lambda initial: jax.lax.scan(
            body,
            initial,
            jnp.asarray([0, 0], dtype=jnp.int32),
        )
    )(state)
    _assert_trees_equal(jax.tree.map(lambda value: value[0], scanned), eager)
    _assert_trees_equal(jax.tree.map(lambda value: value[1], scanned), eager)
    _assert_trees_equal(final_state, before)
    _assert_trees_equal(state, before)
