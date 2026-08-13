# mypy: disable-error-code="attr-defined,call-arg"
"""Exact finite-horizon contracts for Prototype feature-utility auditing."""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.prototype_feature_utility import (
    PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA,
    PROTOTYPE_FEATURE_UTILITY_COUNTER_DELTA_NBYTES,
    PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES,
    PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_DELTA_NBYTES,
    PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_NBYTES,
    PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA,
    PROTOTYPE_FEATURE_UTILITY_TELEMETRY_COUNTER_NBYTES,
    PrototypeFeatureUtilityAuditor,
    PrototypeFeatureUtilityConfig,
    PrototypeFeatureUtilityEvent,
    PrototypeFeatureUtilityState,
    measure_prototype_feature_utility_state_nbytes,
    migrate_legacy_prototype_feature_utility_config,
    migrate_legacy_prototype_feature_utility_state,
    prototype_feature_utility_counter_nbytes,
    prototype_feature_utility_lifetime_counter_nbytes,
)
from alberta_framework.core.prototype_feature_utility_curation import (
    PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA,
    PrototypeFeatureUtilityCurationConfig,
    PrototypeFeatureUtilityCurationPolicy,
    migrate_legacy_prototype_feature_utility_curation_config,
)

pytestmark = pytest.mark.unit

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1
_UINT64_MAX = 2**64 - 1


def _config(*, max_observations: int = _UINT64_MAX) -> PrototypeFeatureUtilityConfig:
    return PrototypeFeatureUtilityConfig(
        base_feature_dim=4,
        active_pair_slots=1,
        candidate_pair_slots=1,
        managed_horde_demons=1,
        utility_decay=0.75,
        shadow_step_size=0.2,
        second_moment_decay=0.5,
        scale_epsilon=1.0e-6,
        max_observations=max_observations,
    )


def _auditor_state(
    *,
    max_observations: int = _UINT64_MAX,
    generation_words: tuple[int, int] = (0, 3),
) -> tuple[PrototypeFeatureUtilityAuditor, PrototypeFeatureUtilityState]:
    auditor = PrototypeFeatureUtilityAuditor(
        _config(max_observations=max_observations)
    )
    generation_telemetry = (
        generation_words[1]
        if generation_words[0] == 0 and generation_words[1] < _INT32_MAX
        else _INT32_MAX
    )
    state = auditor.init(
        active_descriptors=jnp.asarray([[0, 1]], dtype=jnp.int32),
        candidate_descriptors=jnp.asarray([[0, 2]], dtype=jnp.int32),
        semantic_generation=jnp.asarray(generation_telemetry, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray(generation_words, dtype=jnp.uint32),
    )
    return auditor, state


def _event(
    state: PrototypeFeatureUtilityState,
    *,
    semantic_generation: jax.Array | None = None,
    semantic_generation_words: jax.Array | None = None,
    candidate_descriptors: jax.Array | None = None,
) -> PrototypeFeatureUtilityEvent:
    base = jnp.asarray([2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    active_value = base[0] * base[1]
    return PrototypeFeatureUtilityEvent(
        base_observation=base,
        augmented_observation=jnp.concatenate((base, active_value[None])),
        targets=jnp.asarray([2.0, 3.0], dtype=jnp.float32),
        predictions=jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        target_available=jnp.asarray([True, True], dtype=jnp.bool_),
        active_consumer_tail_weights=jnp.asarray([[0.2], [-0.1]], dtype=jnp.float32),
        semantic_generation=(
            state.semantic_generation
            if semantic_generation is None
            else semantic_generation
        ),
        semantic_generation_words=(
            state.semantic_generation_words
            if semantic_generation_words is None
            else semantic_generation_words
        ),
        active_descriptors=state.active_descriptors,
        candidate_descriptors=(
            state.candidate_descriptors
            if candidate_descriptors is None
            else candidate_descriptors
        ),
    )


def _assert_tree_exact(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def _assert_tree_close(left: Any, right: Any) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if jnp.issubdtype(left_leaf.dtype, jnp.inexact):
            np.testing.assert_allclose(left_leaf, right_leaf, rtol=2e-6, atol=2e-7)
        else:
            np.testing.assert_array_equal(left_leaf, right_leaf)


def _legacy_state(state: PrototypeFeatureUtilityState) -> dict[str, Any]:
    return {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(cast(Any, state))
        if field.name not in {"semantic_generation_words", "observation_words"}
    }


def test_v2_schema_default_and_exact_resource_delta_are_public() -> None:
    auditor, state = _auditor_state()
    config_payload = auditor.config.to_config()
    assert auditor.config.max_observations == _UINT64_MAX
    assert config_payload["schema_version"] == PROTOTYPE_FEATURE_UTILITY_CONFIG_SCHEMA
    assert config_payload["state_schema"] == PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA
    assert PrototypeFeatureUtilityConfig.from_config(config_payload) == auditor.config

    budget = auditor.resource_budget()
    assert budget.telemetry_counter_nbytes == 2 * 4
    assert budget.exact_counter_nbytes == 2 * 8
    assert budget.counter_delta_nbytes == 16
    assert budget.counter_nbytes == 24
    assert budget.persistent_state_nbytes == measure_prototype_feature_utility_state_nbytes(
        state
    )
    assert PROTOTYPE_FEATURE_UTILITY_TELEMETRY_COUNTER_NBYTES == 4
    assert PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_DELTA_NBYTES == 8
    assert PROTOTYPE_FEATURE_UTILITY_LIFETIME_COUNTER_NBYTES == 12
    assert PROTOTYPE_FEATURE_UTILITY_COUNTER_DELTA_NBYTES == 16
    assert PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES == 24
    assert prototype_feature_utility_lifetime_counter_nbytes() == 12
    assert prototype_feature_utility_counter_nbytes() == 24

    exported = (
        "PROTOTYPE_FEATURE_UTILITY_STATE_SCHEMA",
        "PROTOTYPE_FEATURE_UTILITY_COUNTER_NBYTES",
        "measure_prototype_feature_utility_state_nbytes",
        "migrate_legacy_prototype_feature_utility_state",
        "prototype_feature_utility_counter_nbytes",
    )
    for module_name in ("alberta_framework.core", "alberta_framework"):
        module = importlib.import_module(module_name)
        for name in exported:
            assert getattr(module, name) is not None
            assert module.__all__.count(name) == 1


def test_observation_words_carry_exactly_in_eager_jit_and_scan() -> None:
    auditor, initial = _auditor_state()
    state = initial.replace(
        observation_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        observation_words=jnp.asarray([0, _UINT32_MAX - 1], dtype=jnp.uint32),
    )
    assert bool(auditor.state_valid(state))
    event = _event(state)

    first = auditor.observe(state, event)
    second = auditor.observe(first.state, event)
    np.testing.assert_array_equal(first.state.observation_words, [0, _UINT32_MAX])
    np.testing.assert_array_equal(second.state.observation_words, [1, 0])
    assert int(first.state.observation_count) == _INT32_MAX
    assert int(second.state.observation_count) == _INT32_MAX
    assert bool(first.diagnostics.transaction_applied)
    assert bool(second.diagnostics.transaction_applied)

    compiled = jax.jit(auditor.observe)(first.state, event)
    _assert_tree_close(compiled, second)

    events = jax.tree.map(lambda leaf: jnp.stack((leaf, leaf)), event)

    def step(
        carry: PrototypeFeatureUtilityState,
        item: PrototypeFeatureUtilityEvent,
    ) -> tuple[PrototypeFeatureUtilityState, jax.Array]:
        result = auditor.observe(carry, item)
        return result.state, result.diagnostics.observation_words_after

    scanned, words = jax.lax.scan(step, state, events)
    _assert_tree_close(scanned, second.state)
    np.testing.assert_array_equal(
        words,
        jnp.asarray(
            [[0, _UINT32_MAX], [1, 0]],
            dtype=jnp.uint32,
        ),
    )


def test_generation_carry_and_all_ones_terminals_never_wrap() -> None:
    auditor, state = _auditor_state(generation_words=(0, _UINT32_MAX))
    new_active = jnp.asarray([[1, 2]], dtype=jnp.int32)
    new_candidates = jnp.asarray([[0, 3]], dtype=jnp.int32)
    rebound = auditor.rebind(
        state,
        active_descriptors=new_active,
        candidate_descriptors=new_candidates,
        semantic_generation=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray([1, 0], dtype=jnp.uint32),
    )
    assert bool(rebound.diagnostics.transaction_applied)
    np.testing.assert_array_equal(rebound.state.semantic_generation_words, [1, 0])
    assert int(rebound.state.semantic_generation) == _INT32_MAX
    _assert_tree_exact(
        jax.jit(auditor.rebind)(
            state,
            active_descriptors=new_active,
            candidate_descriptors=new_candidates,
            semantic_generation=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
            semantic_generation_words=jnp.asarray([1, 0], dtype=jnp.uint32),
        ),
        rebound,
    )

    terminal_generation = rebound.state.replace(
        semantic_generation=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray(
            [_UINT32_MAX, _UINT32_MAX], dtype=jnp.uint32
        ),
    )
    rejected_rebind = auditor.rebind(
        terminal_generation,
        active_descriptors=state.active_descriptors,
        candidate_descriptors=state.candidate_descriptors,
        semantic_generation=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        semantic_generation_words=jnp.asarray(
            [_UINT32_MAX, _UINT32_MAX], dtype=jnp.uint32
        ),
    )
    _assert_tree_exact(rejected_rebind.state, terminal_generation)
    assert not bool(rejected_rebind.diagnostics.generation_capacity_available)
    assert not bool(rejected_rebind.diagnostics.transaction_applied)

    terminal_observation = terminal_generation.replace(
        observation_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        observation_words=jnp.asarray(
            [_UINT32_MAX, _UINT32_MAX], dtype=jnp.uint32
        ),
    )
    rejected_observe = auditor.observe(
        terminal_observation,
        _event(terminal_observation),
    )
    _assert_tree_exact(rejected_observe.state, terminal_observation)
    assert bool(rejected_observe.diagnostics.capacity_capped)
    assert not bool(rejected_observe.diagnostics.transaction_applied)


def test_corrupt_state_event_and_candidate_contracts_roll_back_atomically() -> None:
    auditor, state = _auditor_state()
    event = _event(state)
    corrupt_state = state.replace(
        observation_words=jnp.asarray([0, 1], dtype=jnp.uint32),
    )
    state_rejection = jax.jit(auditor.observe)(corrupt_state, event)
    _assert_tree_exact(state_rejection.state, corrupt_state)
    assert not bool(state_rejection.diagnostics.state_values_valid)
    assert not bool(state_rejection.diagnostics.transaction_applied)

    corrupt_event = event.replace(
        semantic_generation_words=jnp.asarray([0, 4], dtype=jnp.uint32),
    )
    event_rejection = auditor.observe(state, corrupt_event)
    _assert_tree_exact(event_rejection.state, state)
    assert not bool(event_rejection.diagnostics.event_values_valid)
    assert not bool(event_rejection.diagnostics.transaction_applied)

    invalid_candidate = event.replace(
        candidate_descriptors=jnp.asarray([[3, 3]], dtype=jnp.int32),
    )
    candidate_rejection = auditor.observe(state, invalid_candidate)
    _assert_tree_exact(candidate_rejection.state, state)
    assert not bool(candidate_rejection.diagnostics.event_descriptors_valid)
    assert not bool(candidate_rejection.diagnostics.transaction_applied)


def test_legacy_config_and_state_migration_never_invent_saturated_history() -> None:
    auditor, state = _auditor_state(max_observations=20)
    legacy_config = auditor.config.to_config()
    legacy_config.pop("state_schema")
    legacy_config["schema_version"] = "alberta.prototype-feature-utility.config.v1"
    with pytest.raises(ValueError, match="schema|keys"):
        PrototypeFeatureUtilityConfig.from_config(legacy_config)
    assert (
        migrate_legacy_prototype_feature_utility_config(legacy_config)
        == auditor.config
    )
    with pytest.raises(ValueError, match="int32-safe"):
        migrate_legacy_prototype_feature_utility_config(
            {**legacy_config, "max_observations": _UINT64_MAX}
        )

    observed = auditor.observe(state, _event(state)).state
    migrated = migrate_legacy_prototype_feature_utility_state(
        auditor,
        _legacy_state(observed),
    )
    _assert_tree_exact(migrated, observed)
    with pytest.raises(ValueError, match="field manifest"):
        migrate_legacy_prototype_feature_utility_state(
            auditor,
            {**_legacy_state(observed), "extra": jnp.asarray(0)},
        )
    saturated_observation = _legacy_state(observed)
    saturated_observation["observation_count"] = jnp.asarray(
        _INT32_MAX, dtype=jnp.int32
    )
    with pytest.raises(ValueError, match="saturated.*observation_count"):
        migrate_legacy_prototype_feature_utility_state(
            auditor,
            saturated_observation,
        )
    saturated_generation = _legacy_state(observed)
    saturated_generation["semantic_generation"] = jnp.asarray(
        _INT32_MAX, dtype=jnp.int32
    )
    with pytest.raises(ValueError, match="saturated.*semantic_generation"):
        migrate_legacy_prototype_feature_utility_state(
            auditor,
            saturated_generation,
        )

    curation = PrototypeFeatureUtilityCurationConfig(minimum_task_evidence=3)
    legacy_curation = curation.to_config()
    legacy_curation["schema_version"] = (
        "alberta.prototype-feature-utility-curation.config.v1"
    )
    with pytest.raises(ValueError, match="schema_version"):
        PrototypeFeatureUtilityCurationConfig.from_config(legacy_curation)
    assert (
        migrate_legacy_prototype_feature_utility_curation_config(
            legacy_curation
        )
        == curation
    )
    assert curation.SCHEMA_VERSION == PROTOTYPE_FEATURE_UTILITY_CURATION_CONFIG_SCHEMA


def test_curation_uses_exact_high_word_generation_and_observation_capacity() -> None:
    auditor, state = _auditor_state(generation_words=(1, 2))
    state = state.replace(
        observation_count=jnp.asarray(_INT32_MAX, dtype=jnp.int32),
        observation_words=jnp.asarray([1, 5], dtype=jnp.uint32),
        active_task_evidence_counts=jnp.ones_like(
            state.active_task_evidence_counts
        ),
        candidate_task_evidence_counts=jnp.ones_like(
            state.candidate_task_evidence_counts
        ),
    )
    assert bool(auditor.state_valid(state))
    policy = PrototypeFeatureUtilityCurationPolicy(
        auditor.config,
        PrototypeFeatureUtilityCurationConfig(minimum_task_evidence=1),
    )
    ranked = policy.rank(
        state,
        source_semantic_generation=state.semantic_generation,
        source_semantic_generation_words=state.semantic_generation_words,
        source_active_descriptors=state.active_descriptors,
        source_candidate_descriptors=state.candidate_descriptors,
    )
    assert bool(ranked.override.enabled)
    assert bool(ranked.diagnostics.source_generation_words_match)
    np.testing.assert_array_equal(ranked.diagnostics.observation_words, [1, 5])
    np.testing.assert_array_equal(
        ranked.diagnostics.maximum_observation_words,
        [_UINT32_MAX, _UINT32_MAX],
    )
    assert int(ranked.diagnostics.maximum_observations) == _INT32_MAX

    wrong_words = policy.rank(
        state,
        source_semantic_generation=state.semantic_generation,
        source_semantic_generation_words=jnp.asarray([1, 3], dtype=jnp.uint32),
        source_active_descriptors=state.active_descriptors,
        source_candidate_descriptors=state.candidate_descriptors,
    )
    assert not bool(wrong_words.override.enabled)
    assert not bool(wrong_words.diagnostics.source_generation_words_match)
    assert bool(wrong_words.diagnostics.stale_state_generation)

    terminal = state.replace(
        observation_words=jnp.asarray(
            [_UINT32_MAX, _UINT32_MAX], dtype=jnp.uint32
        )
    )
    terminal_rank = jax.jit(policy.rank)(
        terminal,
        source_semantic_generation=terminal.semantic_generation,
        source_semantic_generation_words=terminal.semantic_generation_words,
        source_active_descriptors=terminal.active_descriptors,
        source_candidate_descriptors=terminal.candidate_descriptors,
    )
    assert bool(terminal_rank.diagnostics.transaction_valid)
    assert bool(terminal_rank.diagnostics.observation_capacity_capped)
    assert not bool(terminal_rank.override.enabled)
