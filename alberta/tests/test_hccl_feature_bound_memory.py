"""Focused contracts for HCCL R35 feature-bound learned memory."""

from __future__ import annotations

import dataclasses
import json
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.core.experiential_memory import (
    ExperientialMemoryConfig,
    ExperientialMemoryEntry,
)
from alberta_framework.core.hccl_feature_bound_memory import (
    HCCL_FEATURE_BOUND_MEMORY_ACTION_DISPATCH_AUTHORITY,
    HCCL_FEATURE_BOUND_MEMORY_CONFIG_SCHEMA,
    HCCL_FEATURE_BOUND_MEMORY_COUNTERFACTUAL_FEEDBACK_AUTHENTICATED,
    HCCL_FEATURE_BOUND_MEMORY_EVIDENCE_AUTHORITY,
    HCCL_FEATURE_BOUND_MEMORY_FULL_INTEGRATION_CLAIMED,
    HCCL_FEATURE_BOUND_MEMORY_OUTER_TRANSACTION_AUTHORITY,
    HCCL_FEATURE_BOUND_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED,
    HCCLFeatureBoundMemory,
    HCCLFeatureBoundMemoryConfig,
    HCCLFeatureBoundMemoryRebindResult,
    HCCLFeatureBoundMemorySettleResult,
    HCCLFeatureBoundMemoryState,
    HCCLFeatureBoundMemoryStepResult,
)
from alberta_framework.core.hccl_feature_consumer_route import (
    HCCLFeatureBirthLedger,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteResult,
)
from alberta_framework.core.learned_experiential_memory_controller import (
    LearnedExperientialMemoryControllerConfig,
    LearnedExperientialMemoryControllerState,
    LearnedExperientialMemoryFeedback,
)

pytestmark = pytest.mark.unit

_PHYSICAL_DIM = 16
_CONTEXT_START = 16
_FAST_START = 19
_PAIR_START = 23
_R35 = 35
_CAPACITY = 64


def _chex_replace[T](value: T, /, **changes: object) -> T:
    return cast(T, cast(Any, value).replace(**changes))


def _pairs(*live: tuple[int, int]) -> jax.Array:
    values = np.full((12, 2), -1, dtype=np.int32)
    for index, descriptor in enumerate(live):
        values[index] = descriptor
    return jnp.asarray(values, dtype=jnp.int32)


def _admissions(*slots: int) -> jax.Array:
    values = np.zeros((12,), dtype=np.bool_)
    values[list(slots)] = True
    return jnp.asarray(values, dtype=jnp.bool_)


def _route(agent_index: int = 0) -> HCCLFeatureConsumerRoute:
    return HCCLFeatureConsumerRoute(agent_index=agent_index)


def _genesis(route: HCCLFeatureConsumerRoute) -> HCCLFeatureBirthLedger:
    return route.init(
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
    )


def _controller_config(
    *,
    capacity: int = _CAPACITY,
    observation_dim: int = _R35,
    key_dim: int = _R35,
    outcome_dim: int = _R35,
) -> LearnedExperientialMemoryControllerConfig:
    return LearnedExperientialMemoryControllerConfig(
        memory=ExperientialMemoryConfig(
            capacity=capacity,
            observation_dim=observation_dim,
            key_dim=key_dim,
            action_dim=2,
            outcome_dim=outcome_dim,
            top_k=2,
            min_neighbors=1,
            distance_scale=100.0,
            min_similarity=0.0,
            min_effective_reliability=1.0e-6,
            max_uncertainty=1.0,
            max_safety_cost=1.0,
            max_age=10_000,
            staleness_scale=10_000.0,
            utility_decay=1.0,
            eviction_utility_weight=1.0,
            eviction_recency_weight=1.0,
            recency_scale=100.0,
        ),
        admission_step_size=0.1,
        retention_step_size=0.1,
        admission_threshold=0.0,
        initial_admission_bias=0.0,
        max_abs_admission_weight=8.0,
        max_abs_counterfactual_delta=1.0,
        retention_prior=0.5,
    )


def _config(
    *,
    agent_index: int = 0,
    controller: LearnedExperientialMemoryControllerConfig | None = None,
) -> HCCLFeatureBoundMemoryConfig:
    return HCCLFeatureBoundMemoryConfig(
        agent_index=agent_index,
        controller=_controller_config() if controller is None else controller,
    )


def _encode(base: jax.Array, ledger: HCCLFeatureBirthLedger) -> jax.Array:
    parents = ledger.parents[_PAIR_START:]
    active = ledger.active[_PAIR_START:]
    safe_left = jnp.clip(parents[:, 0], 0, _PHYSICAL_DIM - 1)
    safe_right = jnp.clip(parents[:, 1], 0, _PHYSICAL_DIM - 1)
    products = base[safe_left] * base[safe_right]
    products = jnp.where(active, products, jnp.float32(0.0))
    return jnp.concatenate((base, products)).astype(jnp.float32)


def _entry(
    ledger: HCCLFeatureBirthLedger,
    *,
    observation_base: jax.Array,
    outcome_base: jax.Array,
    provenance_id: int,
    source_id: int = 0,
    action_index: int = 0,
) -> ExperientialMemoryEntry:
    observation = _encode(observation_base, ledger)
    outcome = _encode(outcome_base, ledger)
    return cast(
        ExperientialMemoryEntry,
        cast(Any, ExperientialMemoryEntry)(
            observation=observation,
            key=observation,
            action=jax.nn.one_hot(
                jnp.asarray(action_index, dtype=jnp.int32),
                2,
                dtype=jnp.float32,
            ),
            outcome=outcome,
            reward=jnp.asarray(float(provenance_id) / 10.0, dtype=jnp.float32),
            uncertainty=jnp.asarray(0.1, dtype=jnp.float32),
            uncertainty_available=jnp.asarray(True, dtype=jnp.bool_),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(True, dtype=jnp.bool_),
            reliability=jnp.asarray(1.0, dtype=jnp.float32),
            utility=jnp.asarray(0.5, dtype=jnp.float32),
            utility_available=jnp.asarray(True, dtype=jnp.bool_),
            representation_version=jnp.asarray(0, dtype=jnp.int32),
            valid=jnp.asarray(True, dtype=jnp.bool_),
            age=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=jnp.asarray(provenance_id, dtype=jnp.int32),
            source_id=jnp.asarray(source_id, dtype=jnp.int32),
        ),
    )


def _bases(offset: float) -> tuple[jax.Array, jax.Array]:
    observation = np.linspace(-1.5 + offset, 2.0 + offset, _PAIR_START).astype(
        np.float32
    )
    outcome = np.linspace(2.5 + offset, -0.75 + offset, _PAIR_START).astype(
        np.float32
    )
    observation[_CONTEXT_START + 2] = 0.0
    outcome[_CONTEXT_START + 2] = 0.0
    return jnp.asarray(observation), jnp.asarray(outcome)


def _populated_controller_state(
    adapter: HCCLFeatureBoundMemory,
    ledger: HCCLFeatureBirthLedger,
) -> tuple[
    LearnedExperientialMemoryControllerState,
    tuple[tuple[jax.Array, jax.Array], ...],
]:
    controller = adapter.controller
    state = controller.init()
    first_bases = _bases(0.0)
    first_entry = _entry(
        ledger,
        observation_base=first_bases[0],
        outcome_base=first_bases[1],
        provenance_id=11,
    )
    first = controller.step(
        state,
        first_entry.key,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        first_entry,
    )
    assert bool(first.wrote)
    second_bases = _bases(0.25)
    second_entry = _entry(
        ledger,
        observation_base=second_bases[0],
        outcome_base=second_bases[1],
        provenance_id=12,
    )
    second = controller.step(
        first.state,
        first_entry.key,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        second_entry,
    )
    assert bool(second.wrote)
    assert bool(second.state.pending.available)
    assert bool(controller.state_valid(second.state))
    return second.state, (first_bases, second_bases)


def _assert_tree_bit_exact(left: object, right: object) -> None:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    assert left_tree == right_tree  # type: ignore[operator]
    assert len(left_leaves) == len(right_leaves)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = np.ascontiguousarray(np.asarray(left_leaf))
        right_array = np.ascontiguousarray(np.asarray(right_leaf))
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes() == right_array.tobytes()


def _assert_nonrepresentation_controller_exact(
    before: LearnedExperientialMemoryControllerState,
    after: LearnedExperientialMemoryControllerState,
) -> None:
    _assert_tree_bit_exact(before.pending, after.pending)
    for field in dataclasses.fields(cast(Any, before)):
        if field.name in {"memory", "pending"}:
            continue
        _assert_tree_bit_exact(getattr(before, field.name), getattr(after, field.name))
    for field in dataclasses.fields(cast(Any, before.memory)):
        if field.name == "entries":
            continue
        _assert_tree_bit_exact(
            getattr(before.memory, field.name),
            getattr(after.memory, field.name),
        )
    for field in dataclasses.fields(cast(Any, before.memory.entries)):
        if field.name in {
            "observations",
            "keys",
            "outcomes",
            "representation_versions",
        }:
            continue
        _assert_tree_bit_exact(
            getattr(before.memory.entries, field.name),
            getattr(after.memory.entries, field.name),
        )


def _successor(
    route: HCCLFeatureConsumerRoute,
    source: HCCLFeatureBirthLedger,
) -> HCCLFeatureConsumerRouteResult:
    return route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.asarray(
            ((0, 1), (0, 0), (0, 0)),
            dtype=jnp.uint32,
        ),
        pair_descriptors=_pairs((1, 3), (0, 1), (2, 3)),
        pair_admission_mask=_admissions(2),
    )


def _bound_step(
    adapter: HCCLFeatureBoundMemory,
    state: HCCLFeatureBoundMemoryState,
    ledger: HCCLFeatureBirthLedger,
    *,
    offset: float,
    provenance_id: int,
    source_id: int = 0,
    action_index: int = 0,
) -> HCCLFeatureBoundMemoryStepResult:
    observation_base, outcome_base = _bases(offset)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=provenance_id,
        source_id=source_id,
        action_index=action_index,
    )
    return adapter.step(
        state,
        entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )


def _state_with_pending(
    adapter: HCCLFeatureBoundMemory,
    ledger: HCCLFeatureBirthLedger,
) -> tuple[HCCLFeatureBoundMemoryState, HCCLFeatureBoundMemoryStepResult]:
    source = adapter.init(ledger)
    first = _bound_step(
        adapter,
        source,
        ledger,
        offset=0.0,
        provenance_id=101,
    )
    assert bool(first.diagnostics.transaction_applied)
    assert not bool(first.state.controller_state.pending.available)
    second = _bound_step(
        adapter,
        first.state,
        ledger,
        offset=0.2,
        provenance_id=102,
        action_index=1,
    )
    assert bool(second.diagnostics.transaction_applied)
    assert bool(second.state.controller_state.pending.available)
    return first.state, second


def test_config_is_strict_fixed_geometry_and_declares_narrow_authority() -> None:
    config = _config()
    payload = json.loads(json.dumps(config.to_config()))
    restored = HCCLFeatureBoundMemoryConfig.from_config(payload)

    assert restored == config
    assert payload["schema"] == HCCL_FEATURE_BOUND_MEMORY_CONFIG_SCHEMA
    assert payload["agent_index"] == 0
    assert payload["memory_capacity"] == 64
    assert payload["representation_order"] == [
        "physical16",
        "context3",
        "fast4",
        "pair12",
    ]
    assert payload["full_integration_claimed"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert HCCL_FEATURE_BOUND_MEMORY_FULL_INTEGRATION_CLAIMED is False
    assert HCCL_FEATURE_BOUND_MEMORY_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert HCCL_FEATURE_BOUND_MEMORY_COUNTERFACTUAL_FEEDBACK_AUTHENTICATED is False
    assert HCCL_FEATURE_BOUND_MEMORY_ACTION_DISPATCH_AUTHORITY is False
    assert HCCL_FEATURE_BOUND_MEMORY_OUTER_TRANSACTION_AUTHORITY is False
    assert HCCL_FEATURE_BOUND_MEMORY_EVIDENCE_AUTHORITY is False
    assert HCCLFeatureBoundMemory.from_config(payload).to_config() == payload

    bool_alias = dict(payload)
    bool_alias["full_integration_claimed"] = 0
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLFeatureBoundMemoryConfig.from_config(bool_alias)
    integer_alias = dict(payload)
    integer_alias["pair_products_per_rebind"] = float(12 * 64 * 2)
    with pytest.raises(ValueError, match="noncanonical"):
        HCCLFeatureBoundMemoryConfig.from_config(integer_alias)

    with pytest.raises(ValueError, match="capacity must equal 64"):
        _config(controller=_controller_config(capacity=63))
    with pytest.raises(ValueError, match="observation_dim must equal 35"):
        _config(controller=_controller_config(observation_dim=34))
    with pytest.raises(ValueError, match="key_dim must equal 35"):
        _config(controller=_controller_config(key_dim=34))
    with pytest.raises(ValueError, match="outcome_dim must equal 35"):
        _config(controller=_controller_config(outcome_dim=34))


def test_init_binds_empty_rows_and_populated_pending_to_exact_ledger() -> None:
    route = _route()
    ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    empty = adapter.init(ledger)

    assert bool(adapter.state_valid(empty, ledger))
    assert not bool(jnp.any(empty.row_ledger_content_tokens))
    assert not bool(jnp.any(empty.row_semantic_generation_words))
    assert not bool(jnp.any(empty.pending_ledger_content_token))
    assert not bool(jnp.any(empty.pending_semantic_generation_words))

    controller_state, _ = _populated_controller_state(adapter, ledger)
    state = adapter.init(ledger, controller_state)
    valid = np.asarray(controller_state.memory.entries.valid)
    assert int(np.sum(valid)) == 2
    assert bool(adapter.state_valid(state, ledger))
    np.testing.assert_array_equal(
        state.row_ledger_content_tokens[valid],
        np.broadcast_to(np.asarray(ledger.content_token), (2, 32)),
    )
    np.testing.assert_array_equal(
        state.row_semantic_generation_words[valid],
        np.zeros((2, 2), dtype=np.uint32),
    )
    np.testing.assert_array_equal(
        state.pending_ledger_content_token,
        ledger.content_token,
    )
    np.testing.assert_array_equal(
        state.pending_semantic_generation_words,
        ledger.semantic_generation_words,
    )


def test_populated_bare_controller_cannot_launder_a_post_genesis_ledger() -> None:
    route = _route()
    genesis = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    controller_state, _ = _populated_controller_state(adapter, genesis)
    advanced = route.prepare_successor(
        genesis,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    ).ledger

    assert bool(route.ledger_valid(advanced))
    np.testing.assert_array_equal(advanced.semantic_generation_words, (0, 0))
    with pytest.raises(ValueError, match="populated bare controller"):
        adapter.init(advanced, controller_state)

    empty_advanced = adapter.init(advanced)
    assert bool(adapter.state_valid(empty_advanced, advanced))


def test_cross_agent_row_provenance_cannot_be_bound_as_local_memory() -> None:
    route = _route(agent_index=1)
    ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config(agent_index=1))
    observation_base, outcome_base = _bases(0.0)
    foreign_entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=41,
    )
    assert int(foreign_entry.source_id) == 0
    written = adapter.controller.step(
        adapter.controller.init(),
        foreign_entry.key,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        foreign_entry,
    )
    assert bool(written.wrote)

    with pytest.raises(ValueError, match="not encoded for the ledger"):
        adapter.init(ledger, written.state)


def test_rebind_reencodes_rows_stamps_pending_and_preserves_controller_time() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    controller_state, bases = _populated_controller_state(adapter, source_ledger)
    source = adapter.init(source_ledger, controller_state)
    route_result = _successor(route, source_ledger)
    before = source.controller_state

    result = adapter.rebind(source, source_ledger, route_result)

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(result.diagnostics.complete_source_returned)
    assert bool(result.diagnostics.route_result_valid)
    assert bool(result.diagnostics.route_transaction_applied)
    assert bool(result.diagnostics.candidate_values_finite)
    assert bool(result.diagnostics.candidate_controller_state_valid)
    assert bool(result.diagnostics.candidate_state_valid)
    assert int(result.diagnostics.valid_rows_before) == 2
    assert int(result.diagnostics.valid_rows_reencoded) == 2
    assert int(result.diagnostics.valid_rows_after) == 2
    assert int(result.diagnostics.context_newborn_slots) == 1
    assert int(result.work.observation_key_pair_products_evaluated) == 12 * 64
    assert int(result.work.outcome_pair_products_evaluated) == 12 * 64
    assert int(result.work.pair_products_evaluated) == 2 * 12 * 64
    assert int(result.work.controller_clock_advances) == 0
    assert int(result.work.memory_step_advances) == 0
    assert int(result.work.insertion_clock_mutations) == 0
    assert int(result.work.rng_draws) == 0
    _assert_nonrepresentation_controller_exact(before, result.state.controller_state)
    np.testing.assert_array_equal(
        result.diagnostics.controller_transaction_words_before,
        result.diagnostics.controller_transaction_words_after,
    )
    np.testing.assert_array_equal(
        result.diagnostics.memory_step_words_before,
        result.diagnostics.memory_step_words_after,
    )
    np.testing.assert_array_equal(
        before.memory.entries.insertion_step_words,
        result.state.controller_state.memory.entries.insertion_step_words,
    )

    destination = route_result.ledger
    entries = result.state.controller_state.memory.entries
    valid = np.asarray(entries.valid)
    for index, (observation_base, outcome_base) in enumerate(bases):
        expected_observation_base = observation_base.at[_CONTEXT_START].set(0.0)
        expected_outcome_base = outcome_base.at[_CONTEXT_START].set(0.0)
        expected_observation = _encode(expected_observation_base, destination)
        expected_outcome = _encode(expected_outcome_base, destination)
        np.testing.assert_array_equal(entries.observations[index], expected_observation)
        np.testing.assert_array_equal(entries.keys[index], expected_observation)
        np.testing.assert_array_equal(entries.outcomes[index], expected_outcome)
        for representation in (
            entries.observations,
            entries.keys,
            entries.outcomes,
        ):
            for context_index in (_CONTEXT_START, _CONTEXT_START + 2):
                bits = np.asarray(
                    representation[index, context_index],
                    dtype=np.float32,
                ).view(np.uint32)
                assert int(bits) == 0
        assert int(entries.representation_versions[index]) == 1
    np.testing.assert_array_equal(
        result.state.row_ledger_content_tokens[valid],
        np.broadcast_to(np.asarray(destination.content_token), (2, 32)),
    )
    np.testing.assert_array_equal(
        result.state.row_semantic_generation_words[valid],
        np.broadcast_to(np.asarray((0, 1), dtype=np.uint32), (2, 2)),
    )
    np.testing.assert_array_equal(
        result.state.pending_ledger_content_token,
        destination.content_token,
    )
    np.testing.assert_array_equal(
        result.state.pending_semantic_generation_words,
        destination.semantic_generation_words,
    )
    assert bool(adapter.state_valid(result.state, destination))


def test_pure_pair_permutation_reorders_tails_without_generation_change() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    controller_state, _ = _populated_controller_state(adapter, source_ledger)
    source = adapter.init(source_ledger, controller_state)
    route_result = route.prepare_successor(
        source_ledger,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((1, 3), (0, 1), (0, 2)),
        pair_admission_mask=_admissions(),
    )

    result = adapter.rebind(source, source_ledger, route_result)

    assert bool(result.diagnostics.transaction_applied)
    assert not bool(route_result.witness.semantic_bank_changed)
    np.testing.assert_array_equal(
        result.state.feature_ledger.semantic_generation_words,
        (0, 0),
    )
    valid = result.state.controller_state.memory.entries.valid
    assert bool(
        jnp.all(
            result.state.controller_state.memory.entries.representation_versions[
                valid
            ]
            == 0
        )
    )
    assert int(result.work.pair_products_evaluated) == 1536


def test_ledger_pending_row_and_selected_result_tamper_fail_closed() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    controller_state, _ = _populated_controller_state(adapter, source_ledger)
    source = adapter.init(source_ledger, controller_state)
    route_result = _successor(route, source_ledger)

    pending_tamper = _chex_replace(
        source,
        pending_ledger_content_token=source.pending_ledger_content_token.at[0].set(
            source.pending_ledger_content_token[0] ^ jnp.uint8(1)
        )
    )
    rejected_pending = adapter.rebind(pending_tamper, source_ledger, route_result)
    assert not bool(rejected_pending.diagnostics.source_state_valid)
    assert bool(rejected_pending.diagnostics.reencode_attempted)
    _assert_tree_bit_exact(rejected_pending.state, pending_tamper)

    row_tamper = _chex_replace(
        source,
        row_semantic_generation_words=source.row_semantic_generation_words.at[
            0, 1
        ].set(jnp.uint32(9))
    )
    rejected_row = adapter.rebind(row_tamper, source_ledger, route_result)
    assert not bool(rejected_row.diagnostics.source_state_valid)
    assert bool(rejected_row.diagnostics.reencode_attempted)
    _assert_tree_bit_exact(rejected_row.state, row_tamper)

    forged_selection = _chex_replace(route_result, ledger=source_ledger)
    rejected_route = adapter.rebind(source, source_ledger, forged_selection)
    assert not bool(rejected_route.diagnostics.route_result_valid)
    assert bool(rejected_route.diagnostics.reencode_attempted)
    _assert_tree_bit_exact(rejected_route.state, source)

    advanced_source = route_result.ledger
    advanced_route = route.prepare_successor(
        advanced_source,
        destination_source_clock_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.asarray(
            ((0, 1), (0, 0), (0, 0)),
            dtype=jnp.uint32,
        ),
        pair_descriptors=_pairs((1, 3), (0, 1), (2, 3)),
        pair_admission_mask=_admissions(),
    )
    rejected_ledger = adapter.rebind(source, advanced_source, advanced_route)
    assert not bool(rejected_ledger.diagnostics.source_ledger_matches)
    assert bool(rejected_ledger.diagnostics.reencode_attempted)
    _assert_tree_bit_exact(rejected_ledger.state, source)


def test_exact_product_kernel_executes_two_full_capacity_encodings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(source_ledger)
    original = adapter._encode_rows
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def counted(
        base: jax.Array,
        ledger: HCCLFeatureBirthLedger,
    ) -> tuple[jax.Array, jax.Array]:
        encoded, products = original(base, ledger)
        calls.append((tuple(base.shape), tuple(products.shape)))
        return encoded, products

    monkeypatch.setattr(adapter, "_encode_rows", counted)

    result = adapter.rebind(source, source_ledger, _successor(route, source_ledger))

    assert calls == [((_CAPACITY, _PAIR_START), (_CAPACITY, 12))] * 2
    assert int(result.work.observation_key_pair_products_evaluated) == 64 * 12
    assert int(result.work.outcome_pair_products_evaluated) == 64 * 12
    assert int(result.work.pair_products_evaluated) == 1536


def test_resealed_representation_tamper_fails_semantics_and_rebind() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    controller_state, _ = _populated_controller_state(adapter, source_ledger)
    source = adapter.init(source_ledger, controller_state)
    entries = source.controller_state.memory.entries
    observations = entries.observations.at[0, _PAIR_START].add(jnp.float32(1.0))
    keys = entries.keys.at[0, _PAIR_START].add(jnp.float32(1.0))
    tampered_controller = _chex_replace(
        source.controller_state,
        memory=_chex_replace(
            source.controller_state.memory,
            entries=_chex_replace(
                entries,
                observations=observations,
                keys=keys,
            ),
        )
    )
    resealed = adapter._seal_state(
        _chex_replace(source, controller_state=tampered_controller)
    )

    assert bool(adapter.controller.state_valid(tampered_controller))
    assert not bool(adapter.state_valid(resealed))
    rejected = adapter.rebind(resealed, source_ledger, _successor(route, source_ledger))
    assert not bool(rejected.diagnostics.source_state_valid)
    _assert_tree_bit_exact(rejected.state, resealed)


def test_nonfinite_destination_pair_product_rejects_with_exact_work() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    observation = jnp.ones((_PAIR_START,), dtype=jnp.float32)
    observation = observation.at[0].set(jnp.float32(2.0e19))
    observation = observation.at[1].set(jnp.float32(1.0e-20))
    observation = observation.at[2].set(jnp.float32(1.0e-20))
    observation = observation.at[4].set(jnp.float32(2.0e19))
    observation = observation.at[_CONTEXT_START + 2].set(jnp.float32(0.0))
    entry = _entry(
        source_ledger,
        observation_base=observation,
        outcome_base=observation,
        provenance_id=31,
    )
    controller = adapter.controller
    written = controller.step(
        controller.init(),
        entry.key,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    source = adapter.init(source_ledger, written.state)
    route_result = route.prepare_successor(
        source_ledger,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (0, 4)),
        pair_admission_mask=_admissions(2),
    )

    result = adapter.rebind(source, source_ledger, route_result)

    assert bool(result.diagnostics.reencode_attempted)
    assert not bool(result.diagnostics.candidate_values_finite)
    assert not bool(result.diagnostics.transaction_applied)
    assert int(result.work.pair_products_evaluated) == 1536
    _assert_tree_bit_exact(result.state, source)


def test_static_contract_frozen_result_and_monolithic_jit_rejection() -> None:
    route = _route()
    source_ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(source_ledger)
    route_result = _successor(route, source_ledger)
    wrong_shape = _chex_replace(
        source,
        row_ledger_content_tokens=jnp.zeros((63, 32), dtype=jnp.uint8)
    )
    with pytest.raises(ValueError, match="row_ledger_content_tokens"):
        adapter.state_valid(wrong_shape)

    malformed_ledger = _chex_replace(
        route_result.ledger,
        active=jnp.ones((34,), dtype=jnp.bool_),
    )
    malformed_route = _chex_replace(route_result, ledger=malformed_ledger)
    with pytest.raises(ValueError, match=r"ledger\.active"):
        adapter.rebind(source, source_ledger, malformed_route)

    result: HCCLFeatureBoundMemoryRebindResult = adapter.rebind(
        source,
        source_ledger,
        route_result,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.state.config_token = jnp.zeros((32,), dtype=jnp.uint8)

    compiled = jax.jit(
        lambda current: adapter.rebind(
            current,
            source_ledger,
            route_result,
        ).state.controller_state.transaction_words
    )
    with pytest.raises(TypeError, match="host/eager-only"):
        compiled(source)


def test_bound_step_derives_version_calls_once_and_stamps_write_and_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    raw_entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=101,
    )
    raw_entry = _chex_replace(
        raw_entry,
        representation_version=jnp.asarray(777, dtype=jnp.int32),
    )
    original = adapter.controller.step
    calls: list[tuple[int, int]] = []

    def counted(*args: object) -> object:
        calls.append((int(cast(Any, args[2])), int(cast(Any, args[5]).representation_version)))
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "step", counted)
    first = adapter.step(
        source,
        raw_entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        raw_entry,
    )

    assert calls == [(0, 0)]
    assert bool(first.diagnostics.transaction_applied)
    assert bool(first.diagnostics.donor_result_valid)
    assert int(first.diagnostics.derived_representation_version) == 0
    assert int(first.work.controller_step_calls) == 1
    assert int(first.work.controller_query_kernels) == 1
    assert int(first.work.controller_write_attempts) == 1
    assert int(first.work.step_candidate_reconstructions) == 1
    assert int(first.work.memory_write_validation_replays) == 1
    assert int(first.work.representations_validated) == 4
    assert int(first.work.representation_pair_products_evaluated) == 48
    assert int(first.work.donor_query_replays) == 1
    assert int(first.work.rng_draws) == 0
    assert not bool(first.state.controller_state.pending.available)
    valid = np.asarray(first.state.controller_state.memory.entries.valid)
    assert int(np.sum(valid)) == 1
    assert int(first.state.controller_state.memory.entries.representation_versions[0]) == 0
    np.testing.assert_array_equal(
        first.state.row_ledger_content_tokens[valid],
        np.broadcast_to(np.asarray(ledger.content_token), (1, 32)),
    )
    np.testing.assert_array_equal(
        first.state.row_semantic_generation_words[valid],
        np.zeros((1, 2), dtype=np.uint32),
    )
    assert bool(adapter.state_valid(first.state, ledger))

    second = _bound_step(
        adapter,
        first.state,
        ledger,
        offset=0.2,
        provenance_id=102,
        action_index=1,
    )
    assert calls == [(0, 0), (0, 0)]
    assert bool(second.diagnostics.transaction_applied)
    assert bool(second.controller_result.diagnostics.pending_created)
    assert bool(second.state.controller_state.pending.available)
    np.testing.assert_array_equal(
        second.state.pending_ledger_content_token,
        ledger.content_token,
    )
    np.testing.assert_array_equal(
        second.state.pending_semantic_generation_words,
        ledger.semantic_generation_words,
    )
    assert not bool(second.diagnostics.action_dispatch_authority)
    assert not bool(second.diagnostics.outer_transaction_authority)
    assert not bool(second.diagnostics.evidence_authority)


def test_bound_settle_calls_once_restamps_and_step_state_rebinds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    ledger = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    _, pending_step = _state_with_pending(adapter, ledger)
    pending_state = pending_step.state
    feedback = cast(Any, LearnedExperientialMemoryFeedback)(
        transaction_words=pending_state.controller_state.pending.transaction_words,
        retrieval_used=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(0.5, dtype=jnp.float32),
    )
    original = adapter.controller.settle
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "settle", counted)
    settled: HCCLFeatureBoundMemorySettleResult = adapter.settle(
        pending_state,
        feedback,
    )

    assert calls == 1
    assert bool(settled.diagnostics.transaction_applied)
    assert bool(settled.diagnostics.donor_result_valid)
    assert not bool(settled.state.controller_state.pending.available)
    assert not bool(jnp.any(settled.state.pending_ledger_content_token))
    assert not bool(jnp.any(settled.state.pending_semantic_generation_words))
    assert int(settled.work.controller_settle_calls) == 1
    assert int(settled.work.settlement_candidate_reconstructions) == 1
    assert int(settled.work.retention_identity_checks) == 2
    assert int(settled.work.controller_step_calls) == 0
    assert not bool(settled.diagnostics.counterfactual_feedback_authenticated)
    assert not bool(settled.diagnostics.action_dispatch_authority)
    assert not bool(settled.diagnostics.outer_transaction_authority)
    assert not bool(settled.diagnostics.evidence_authority)
    assert bool(adapter.state_valid(settled.state, ledger))

    route_result = _successor(route, ledger)
    rebound = adapter.rebind(pending_state, ledger, route_result)
    assert bool(rebound.diagnostics.transaction_applied)
    assert bool(rebound.state.controller_state.pending.available)
    np.testing.assert_array_equal(
        rebound.state.pending_ledger_content_token,
        route_result.ledger.content_token,
    )
    np.testing.assert_array_equal(
        rebound.state.pending_semantic_generation_words,
        route_result.ledger.semantic_generation_words,
    )
    assert bool(adapter.state_valid(rebound.state, route_result.ledger))


def test_step_derives_nonzero_version_from_rebound_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    genesis = _genesis(route)
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(genesis)
    route_result = _successor(route, genesis)
    rebound = adapter.rebind(source, genesis, route_result)
    assert bool(rebound.diagnostics.transaction_applied)
    np.testing.assert_array_equal(
        rebound.state.feature_ledger.semantic_generation_words,
        (0, 1),
    )
    observation_base, outcome_base = _bases(0.3)
    entry = _entry(
        route_result.ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=151,
    )
    entry = _chex_replace(
        entry,
        representation_version=jnp.asarray(999, dtype=jnp.int32),
    )
    original = adapter.controller.step
    versions: list[int] = []

    def counted(*args: object) -> object:
        versions.append(int(cast(Any, args[2])))
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "step", counted)
    result = adapter.step(
        rebound.state,
        entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    assert versions == [1]
    assert bool(result.diagnostics.transaction_applied)
    assert int(result.diagnostics.derived_representation_version) == 1
    assert int(result.state.controller_state.memory.entries.representation_versions[0]) == 1


def test_resealed_pending_ledger_stamp_tamper_rejects_settlement_bit_exact() -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    _, pending_step = _state_with_pending(adapter, ledger)
    source = pending_step.state
    feedback = cast(Any, LearnedExperientialMemoryFeedback)(
        transaction_words=source.controller_state.pending.transaction_words,
        retrieval_used=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(0.25, dtype=jnp.float32),
    )
    tampered = adapter._seal_state(
        _chex_replace(
            source,
            pending_semantic_generation_words=jnp.asarray(
                (0, 9),
                dtype=jnp.uint32,
            ),
        )
    )
    assert not bool(adapter.state_valid(tampered))

    rejected = adapter.settle(tampered, feedback)

    assert not bool(rejected.diagnostics.source_state_valid)
    assert not bool(rejected.diagnostics.pending_stamp_matches_current_ledger)
    assert bool(rejected.diagnostics.donor_result_valid)
    assert bool(rejected.diagnostics.donor_transaction_applied)
    assert not bool(rejected.diagnostics.transaction_applied)
    _assert_tree_bit_exact(rejected.state, tampered)


def test_missing_and_stale_feedback_call_once_each_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    empty = adapter.init(ledger)
    original = adapter.controller.settle
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "settle", counted)
    missing_feedback = cast(Any, LearnedExperientialMemoryFeedback)(
        transaction_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        retrieval_used=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(0.25, dtype=jnp.float32),
    )
    missing = adapter.settle(empty, missing_feedback)
    assert calls == 1
    assert not bool(missing.diagnostics.pending_available)
    assert bool(missing.diagnostics.donor_result_valid)
    assert not bool(missing.diagnostics.donor_transaction_applied)
    assert bool(missing.diagnostics.complete_source_returned)
    _assert_tree_bit_exact(missing.state, empty)

    _, pending_step = _state_with_pending(adapter, ledger)
    stale_feedback = _chex_replace(
        missing_feedback,
        transaction_words=jnp.asarray((9, 9), dtype=jnp.uint32),
    )
    stale = adapter.settle(pending_step.state, stale_feedback)
    assert calls == 2
    assert bool(stale.diagnostics.pending_available)
    assert bool(stale.diagnostics.donor_result_valid)
    assert not bool(stale.diagnostics.donor_transaction_applied)
    assert bool(stale.diagnostics.complete_source_returned)
    _assert_tree_bit_exact(stale.state, pending_step.state)


def test_step_semantic_action_source_and_provenance_tamper_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=201,
    )
    original = adapter.controller.step
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "step", counted)

    def rejected(
        query: jax.Array,
        candidate: ExperientialMemoryEntry,
    ) -> HCCLFeatureBoundMemoryStepResult:
        result = adapter.step(
            source,
            query,
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            candidate,
        )
        assert not bool(result.diagnostics.transaction_applied)
        assert bool(result.diagnostics.complete_source_returned)
        _assert_tree_bit_exact(result.state, source)
        return result

    bad_query_pair = entry.key.at[_PAIR_START].add(jnp.float32(1.0))
    assert not bool(rejected(bad_query_pair, entry).diagnostics.query_representation_valid)

    negative_zero = jnp.asarray(-0.0, dtype=jnp.float32)
    bad_query_inactive = entry.key.at[_CONTEXT_START + 2].set(negative_zero)
    assert not bool(
        rejected(bad_query_inactive, entry).diagnostics.query_representation_valid
    )

    bad_observation = entry.observation.at[_PAIR_START].add(jnp.float32(1.0))
    bad_pair_entry = _chex_replace(
        entry,
        observation=bad_observation,
        key=bad_observation,
    )
    assert not bool(
        rejected(entry.key, bad_pair_entry).diagnostics.entry_observation_valid
    )

    mismatched_key = _chex_replace(
        entry,
        key=entry.key.at[0].add(jnp.float32(0.25)),
    )
    assert not bool(
        rejected(entry.key, mismatched_key).diagnostics.entry_key_matches_observation
    )

    fractional_action = _chex_replace(
        entry,
        action=jnp.asarray((0.5, 0.5), dtype=jnp.float32),
    )
    assert not bool(
        rejected(entry.key, fractional_action).diagnostics.entry_action_is_categorical_one_hot
    )

    foreign = _chex_replace(entry, source_id=jnp.asarray(1, dtype=jnp.int32))
    assert not bool(rejected(entry.key, foreign).diagnostics.entry_source_is_local)

    negative_provenance = _chex_replace(
        entry,
        provenance_id=jnp.asarray(-1, dtype=jnp.int32),
    )
    assert not bool(
        rejected(entry.key, negative_provenance).diagnostics.entry_provenance_nonnegative
    )

    bad_outcome = entry.outcome.at[_CONTEXT_START + 2].set(negative_zero)
    bad_outcome_entry = _chex_replace(entry, outcome=bad_outcome)
    assert not bool(
        rejected(entry.key, bad_outcome_entry).diagnostics.entry_outcome_valid
    )
    assert calls == 8


def test_pending_blocks_second_step_with_one_donor_call_and_exact_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    _, pending_step = _state_with_pending(adapter, ledger)
    original = adapter.controller.step
    calls = 0

    def counted(*args: object) -> object:
        nonlocal calls
        calls += 1
        return original(*cast(Any, args))

    monkeypatch.setattr(adapter.controller, "step", counted)
    blocked = _bound_step(
        adapter,
        pending_step.state,
        ledger,
        offset=0.4,
        provenance_id=103,
    )

    assert calls == 1
    assert bool(blocked.controller_result.diagnostics.pending_blocked)
    assert bool(blocked.diagnostics.donor_result_valid)
    assert not bool(blocked.diagnostics.donor_transaction_applied)
    assert not bool(blocked.diagnostics.transaction_applied)
    assert int(blocked.work.controller_query_kernels) == 1
    assert int(blocked.work.controller_write_attempts) == 0
    assert int(blocked.work.step_candidate_reconstructions) == 0
    assert int(blocked.work.memory_write_validation_replays) == 0
    assert int(blocked.work.donor_query_replays) == 0
    _assert_tree_bit_exact(blocked.state, pending_step.state)


def test_step_and_settle_reject_coherent_valid_donor_state_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=301,
    )
    original_step = adapter.controller.step
    step_calls = 0

    def tampered_step(*args: object) -> object:
        nonlocal step_calls
        step_calls += 1
        donor = original_step(*cast(Any, args))
        tampered_state = _chex_replace(
            donor.state,
            admission_weights=donor.state.admission_weights.at[0].add(
                jnp.float32(0.125)
            ),
        )
        assert bool(adapter.controller.state_valid(tampered_state))
        return _chex_replace(donor, state=tampered_state)

    monkeypatch.setattr(adapter.controller, "step", tampered_step)
    rejected_step = adapter.step(
        source,
        entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    assert step_calls == 1
    assert not bool(rejected_step.diagnostics.donor_result_valid)
    assert bool(rejected_step.diagnostics.candidate_state_valid)
    _assert_tree_bit_exact(rejected_step.state, source)

    monkeypatch.setattr(adapter.controller, "step", original_step)
    _, pending_step = _state_with_pending(adapter, ledger)
    feedback = cast(Any, LearnedExperientialMemoryFeedback)(
        transaction_words=pending_step.state.controller_state.pending.transaction_words,
        retrieval_used=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(0.5, dtype=jnp.float32),
    )
    original_settle = adapter.controller.settle
    settle_calls = 0

    def tampered_settle(*args: object) -> object:
        nonlocal settle_calls
        settle_calls += 1
        donor = original_settle(*cast(Any, args))
        tampered_state = _chex_replace(
            donor.state,
            admission_weights=donor.state.admission_weights.at[0].add(
                jnp.float32(0.0625)
            ),
        )
        assert bool(adapter.controller.state_valid(tampered_state))
        return _chex_replace(donor, state=tampered_state)

    monkeypatch.setattr(adapter.controller, "settle", tampered_settle)
    rejected_settle = adapter.settle(pending_step.state, feedback)
    assert settle_calls == 1
    assert not bool(rejected_settle.diagnostics.donor_result_valid)
    assert bool(rejected_settle.diagnostics.candidate_state_valid)
    _assert_tree_bit_exact(rejected_settle.state, pending_step.state)


def test_step_reconstruction_rejects_unrelated_row_payload_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    first = _bound_step(
        adapter,
        adapter.init(ledger),
        ledger,
        offset=0.0,
        provenance_id=351,
    )
    assert bool(first.diagnostics.transaction_applied)
    original = adapter.controller.step
    calls = 0

    def tampered(*args: object) -> object:
        nonlocal calls
        calls += 1
        donor = original(*cast(Any, args))
        entries = donor.state.memory.entries
        tampered_entries = _chex_replace(
            entries,
            rewards=entries.rewards.at[0].add(jnp.float32(0.125)),
        )
        tampered_state = _chex_replace(
            donor.state,
            memory=_chex_replace(donor.state.memory, entries=tampered_entries),
        )
        assert bool(adapter.controller.state_valid(tampered_state))
        return _chex_replace(donor, state=tampered_state)

    monkeypatch.setattr(adapter.controller, "step", tampered)
    rejected = _bound_step(
        adapter,
        first.state,
        ledger,
        offset=0.2,
        provenance_id=352,
    )
    assert calls == 1
    assert not bool(rejected.diagnostics.donor_result_valid)
    assert bool(rejected.diagnostics.candidate_state_valid)
    _assert_tree_bit_exact(rejected.state, first.state)


def test_step_query_replay_rejects_coherent_retrieval_diagnostic_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=375,
    )
    original = adapter.controller.step
    calls = 0

    def tampered(*args: object) -> object:
        nonlocal calls
        calls += 1
        donor = original(*cast(Any, args))
        fixed = _chex_replace(
            donor.fixed_store_retrieval,
            query_valid=~donor.fixed_store_retrieval.query_valid,
        )
        learned = _chex_replace(
            donor.retrieval,
            query_valid=~donor.retrieval.query_valid,
        )
        return _chex_replace(
            donor,
            fixed_store_retrieval=fixed,
            retrieval=learned,
        )

    monkeypatch.setattr(adapter.controller, "step", tampered)
    rejected = adapter.step(
        source,
        entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    assert calls == 1
    assert not bool(rejected.diagnostics.donor_result_valid)
    assert int(rejected.work.donor_query_replays) == 1
    assert int(rejected.work.step_candidate_reconstructions) == 0
    _assert_tree_bit_exact(rejected.state, source)


def test_operation_static_contracts_and_stored_action_reseal_rejection() -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=401,
    )

    with pytest.raises(TypeError, match="query_key"):
        adapter.step(
            source,
            entry.key.astype(jnp.int32),
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            entry,
        )
    with pytest.raises(ValueError, match="query_key"):
        adapter.step(
            source,
            entry.key[:-1],
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            entry,
        )
    with pytest.raises(TypeError, match="query_uncertainty"):
        adapter.step(
            source,
            entry.key,
            cast(Any, 0.1),
            jnp.asarray(True, dtype=jnp.bool_),
            entry,
        )
    malformed_entry = _chex_replace(
        entry,
        action=jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32),
    )
    with pytest.raises(ValueError, match=r"entry\.action"):
        adapter.step(
            source,
            entry.key,
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            malformed_entry,
        )
    with pytest.raises(TypeError, match="feedback"):
        adapter.settle(source, cast(Any, None))
    malformed_feedback = cast(Any, LearnedExperientialMemoryFeedback)(
        transaction_words=jnp.asarray((0, 1), dtype=jnp.int32),
        retrieval_used=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_available=jnp.asarray(True, dtype=jnp.bool_),
        counterfactual_delta=jnp.asarray(0.1, dtype=jnp.float32),
    )
    with pytest.raises(TypeError, match=r"feedback\.transaction_words"):
        adapter.settle(source, malformed_feedback)

    written = _bound_step(
        adapter,
        source,
        ledger,
        offset=0.0,
        provenance_id=401,
    ).state
    entries = written.controller_state.memory.entries
    tampered_entries = _chex_replace(
        entries,
        actions=entries.actions.at[0].set(
            jnp.asarray((0.5, 0.5), dtype=jnp.float32)
        ),
    )
    tampered_controller = _chex_replace(
        written.controller_state,
        memory=_chex_replace(
            written.controller_state.memory,
            entries=tampered_entries,
        ),
    )
    assert bool(adapter.controller.state_valid(tampered_controller))
    resealed = adapter._seal_state(
        _chex_replace(written, controller_state=tampered_controller)
    )
    assert not bool(adapter.state_valid(resealed))


def test_step_executes_four_semantic_pair_checks_and_is_not_jittable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _genesis(_route())
    adapter = HCCLFeatureBoundMemory(_config())
    source = adapter.init(ledger)
    observation_base, outcome_base = _bases(0.0)
    entry = _entry(
        ledger,
        observation_base=observation_base,
        outcome_base=outcome_base,
        provenance_id=501,
    )
    original = adapter._representation_semantics_valid
    calls: list[tuple[int, ...]] = []

    def counted(
        representation: jax.Array,
        current_ledger: HCCLFeatureBirthLedger,
    ) -> bool:
        calls.append(tuple(representation.shape))
        return original(representation, current_ledger)

    monkeypatch.setattr(adapter, "_representation_semantics_valid", counted)
    result = adapter.step(
        source,
        entry.key,
        jnp.asarray(0.1, dtype=jnp.float32),
        jnp.asarray(True, dtype=jnp.bool_),
        entry,
    )
    assert bool(result.diagnostics.transaction_applied)
    assert calls == [(_R35,)] * 4
    assert int(result.work.representation_pair_products_evaluated) == 4 * 12

    compiled = jax.jit(
        lambda current: adapter.step(
            current,
            entry.key,
            jnp.asarray(0.1, dtype=jnp.float32),
            jnp.asarray(True, dtype=jnp.bool_),
            entry,
        ).state.controller_state.transaction_words
    )
    with pytest.raises(TypeError, match="host/eager-only"):
        compiled(source)
