# mypy: disable-error-code="attr-defined"
"""Focused contracts for the HCCL semantic-birth route witness."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import alberta_framework.core.hccl_feature_consumer_route as route_module
from alberta_framework.core.hccl_feature_consumer_route import (
    HCCL_FEATURE_BIRTH_LEDGER_SCHEMA,
    HCCL_FEATURE_CONSUMER_ROUTE_CONFIG_SCHEMA,
    HCCL_FEATURE_CONSUMER_ROUTE_FULL_CONSUMER_ROUTING_CLAIMED,
    HCCL_FEATURE_CONSUMER_ROUTE_SCIENTIFIC_PROMOTION_ALLOWED,
    HCCL_FEATURE_ROUTE_WITNESS_SCHEMA,
    HCCLFeatureBirthEvent,
    HCCLFeatureBirthLedger,
    HCCLFeatureConsumerRoute,
    HCCLFeatureConsumerRouteResult,
    HCCLFeatureKind,
)

_PHYSICAL_DIM = 16
_CONTEXT_START = 16
_FAST_START = 19
_PAIR_START = 23
_TOTAL_DIM = 35


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


def _genesis(
    route: HCCLFeatureConsumerRoute | None = None,
) -> HCCLFeatureBirthLedger:
    owner = _route() if route is None else route
    return owner.init(
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
    )


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


@pytest.mark.unit
def test_genesis_is_an_exact_agent_bound_35_slot_ledger() -> None:
    route = _route()
    ledger = _genesis(route)

    assert bool(route.ledger_valid(ledger))
    assert ledger.kind.shape == (_TOTAL_DIM,)
    assert ledger.descriptor.shape == (_TOTAL_DIM, 2)
    assert ledger.birth_words.shape == (_TOTAL_DIM, 2)
    assert ledger.birth_source_words.shape == (_TOTAL_DIM, 2)
    assert ledger.parents.shape == (_TOTAL_DIM, 2)
    assert ledger.birth_event.shape == (_TOTAL_DIM,)
    assert ledger.active.shape == (_TOTAL_DIM,)
    assert ledger.content_token.shape == (32,)
    assert ledger.schema_digest.shape == (32,)
    assert int(ledger.agent_index) == 0
    np.testing.assert_array_equal(ledger.source_clock_words, (0, 0))
    np.testing.assert_array_equal(ledger.semantic_generation_words, (0, 0))
    np.testing.assert_array_equal(
        ledger.kind[:_PHYSICAL_DIM],
        int(HCCLFeatureKind.PHYSICAL),
    )
    np.testing.assert_array_equal(
        ledger.kind[_CONTEXT_START:_FAST_START],
        int(HCCLFeatureKind.CONTEXT),
    )
    np.testing.assert_array_equal(
        ledger.kind[_FAST_START:_PAIR_START],
        int(HCCLFeatureKind.FAST),
    )
    np.testing.assert_array_equal(
        ledger.kind[_PAIR_START:],
        int(HCCLFeatureKind.PAIR),
    )
    assert int(jnp.sum(ledger.active)) == 25
    np.testing.assert_array_equal(
        ledger.birth_event[ledger.active],
        int(HCCLFeatureBirthEvent.GENESIS),
    )

    payload = route.to_config()
    assert payload["schema"] == HCCL_FEATURE_CONSUMER_ROUTE_CONFIG_SCHEMA
    assert payload["ledger_schema"] == HCCL_FEATURE_BIRTH_LEDGER_SCHEMA
    assert payload["witness_schema"] == HCCL_FEATURE_ROUTE_WITNESS_SCHEMA
    assert payload["full_consumer_routing_claimed"] is False
    assert payload["scientific_promotion_allowed"] is False
    assert payload["explicit_pair_admission_mask_required"] is True
    assert payload["pure_pair_permutation_advances_generation"] is False
    assert payload["history_owned"] is False
    assert payload["replay_detection_claimed"] is False
    assert HCCL_FEATURE_CONSUMER_ROUTE_FULL_CONSUMER_ROUTING_CLAIMED is False
    assert HCCL_FEATURE_CONSUMER_ROUTE_SCIENTIFIC_PROMOTION_ALLOWED is False
    assert HCCLFeatureConsumerRoute.from_config(payload).to_config() == payload


@pytest.mark.unit
def test_survivor_move_newborn_and_retirement_use_complete_birth_identity() -> None:
    route = _route()
    source = _genesis(route)

    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((1, 3), (0, 1), (2, 3)),
        pair_admission_mask=_admissions(2),
    )

    assert bool(result.witness.transaction_applied)
    assert bool(route.ledger_valid(result.ledger))
    assert bool(
        route.witness_integrity_valid(
            source,
            result.candidate_ledger,
            result.witness,
        )
    )
    assert bool(route.result_integrity_valid(source, result))
    assert not bool(route.result_integrity_valid(source, result.replace(ledger=source)))
    np.testing.assert_array_equal(
        result.witness.requested_pair_descriptors,
        _pairs((1, 3), (0, 1), (2, 3)),
    )
    route_map = result.witness.route_map
    assert int(route_map.source_slots[_PAIR_START]) == _PAIR_START + 2
    assert int(route_map.source_slots[_PAIR_START + 1]) == _PAIR_START
    assert int(route_map.source_slots[_PAIR_START + 2]) == -1
    assert bool(route_map.survivor_mask[_PAIR_START])
    assert bool(route_map.survivor_mask[_PAIR_START + 1])
    assert bool(route_map.newborn_mask[_PAIR_START + 2])
    assert bool(route_map.retired_mask[_PAIR_START + 1])
    assert bool(route_map.inactive_mask[-1])
    assert bool(route_map.unique_full_identity_matches)
    assert bool(route_map.unique_source_identity_use)
    assert int(route_map.survivor_count) == 24
    assert int(route_map.newborn_count) == 1
    assert int(route_map.retired_count) == 1
    assert int(route_map.inactive_count) == 10

    destination = result.ledger
    np.testing.assert_array_equal(
        destination.birth_words[_PAIR_START],
        source.birth_words[_PAIR_START + 2],
    )
    np.testing.assert_array_equal(
        destination.birth_source_words[_PAIR_START],
        source.birth_source_words[_PAIR_START + 2],
    )
    np.testing.assert_array_equal(
        destination.birth_words[_PAIR_START + 2],
        (0, 1),
    )
    np.testing.assert_array_equal(
        destination.birth_source_words[_PAIR_START + 2],
        (0, 1),
    )
    np.testing.assert_array_equal(
        destination.parents[_PAIR_START + 2],
        (2, 3),
    )
    assert int(destination.birth_event[_PAIR_START + 2]) == int(
        HCCLFeatureBirthEvent.PAIR_ADMISSION
    )
    np.testing.assert_array_equal(destination.semantic_generation_words, (0, 1))

    work = result.witness.work
    assert int(work.source_ledger_validations) == 1
    assert int(work.successor_ledger_candidates) == 1
    assert int(work.pair_descriptor_identity_comparisons) == 12 * 12
    assert int(work.full_birth_identity_comparisons) == 35 * 35
    assert int(work.destination_slot_classifications) == 35
    assert int(work.source_slot_retirement_classifications) == 35
    assert int(work.ledger_content_digest_evaluations) == 3
    assert int(work.witness_content_digest_evaluations) == 1
    assert int(work.consumer_route_evaluations) == 0
    assert int(work.rng_draws) == 0


@pytest.mark.unit
def test_work_receipt_matches_spied_digest_and_identity_matrix_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    route = _route()
    source = _genesis(route)
    original_digest = route_module._digest_tree
    original_pair_matrix = HCCLFeatureConsumerRoute._pair_descriptor_identity_matrix
    original_full_matrix = HCCLFeatureConsumerRoute._full_identity_matrix
    ledger_digests = 0
    witness_digests = 0
    pair_matrix_sizes: list[int] = []
    full_matrix_sizes: list[int] = []

    def counted_digest(schema: str, *values: object) -> jax.Array:
        nonlocal ledger_digests, witness_digests
        if schema == HCCL_FEATURE_BIRTH_LEDGER_SCHEMA:
            ledger_digests += 1
        if schema == HCCL_FEATURE_ROUTE_WITNESS_SCHEMA:
            witness_digests += 1
        return original_digest(schema, *values)

    def counted_pair_matrix(
        ledger: HCCLFeatureBirthLedger,
        descriptors: jax.Array,
    ) -> np.ndarray:
        matrix = original_pair_matrix(ledger, descriptors)
        pair_matrix_sizes.append(int(matrix.size))
        return matrix

    def counted_full_matrix(
        source_ledger: HCCLFeatureBirthLedger,
        destination_ledger: HCCLFeatureBirthLedger,
    ) -> np.ndarray:
        matrix = original_full_matrix(source_ledger, destination_ledger)
        full_matrix_sizes.append(int(matrix.size))
        return matrix

    with monkeypatch.context() as local:
        local.setattr(route_module, "_digest_tree", counted_digest)
        local.setattr(
            HCCLFeatureConsumerRoute,
            "_pair_descriptor_identity_matrix",
            staticmethod(counted_pair_matrix),
        )
        local.setattr(
            HCCLFeatureConsumerRoute,
            "_full_identity_matrix",
            staticmethod(counted_full_matrix),
        )
        result = route.prepare_successor(
            source,
            destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
            context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
            context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
            pair_descriptors=_pairs((1, 3), (0, 1), (2, 3)),
            pair_admission_mask=_admissions(2),
        )

    work = result.witness.work
    assert pair_matrix_sizes == [12 * 12]
    assert full_matrix_sizes == [35 * 35]
    assert ledger_digests == int(work.ledger_content_digest_evaluations) == 3
    assert witness_digests == int(work.witness_content_digest_evaluations) == 1
    assert sum(pair_matrix_sizes) == int(work.pair_descriptor_identity_comparisons)
    assert sum(full_matrix_sizes) == int(work.full_birth_identity_comparisons)


@pytest.mark.unit
def test_explicit_same_step_pair_readmission_forces_a_new_birth() -> None:
    route = _route()
    source = _genesis(route)

    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(0),
    )

    slot = _PAIR_START
    assert bool(result.witness.transaction_applied)
    assert bool(result.witness.requested_pair_admission_mask[0])
    assert bool(result.witness.semantic_bank_changed)
    assert bool(result.witness.route_map.newborn_mask[slot])
    assert bool(result.witness.route_map.retired_mask[slot])
    assert int(result.witness.route_map.source_slots[slot]) == -1
    np.testing.assert_array_equal(result.ledger.birth_words[slot], (0, 1))
    np.testing.assert_array_equal(result.ledger.birth_source_words[slot], (0, 1))
    assert int(result.ledger.birth_event[slot]) == int(
        HCCLFeatureBirthEvent.PAIR_ADMISSION
    )
    np.testing.assert_array_equal(result.ledger.semantic_generation_words, (0, 1))
    assert bool(route.result_integrity_valid(source, result))


@pytest.mark.unit
def test_pair_slot_permutation_is_generation_neutral_and_all_survivors() -> None:
    route = _route()
    source = _genesis(route)

    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((1, 3), (0, 1), (0, 2)),
        pair_admission_mask=_admissions(),
    )

    assert bool(result.witness.transaction_applied)
    assert not bool(result.witness.semantic_bank_changed)
    np.testing.assert_array_equal(result.ledger.semantic_generation_words, (0, 0))
    np.testing.assert_array_equal(
        result.witness.route_map.source_slots[_PAIR_START : _PAIR_START + 3],
        (_PAIR_START + 2, _PAIR_START, _PAIR_START + 1),
    )
    assert bool(
        jnp.all(
            result.witness.route_map.survivor_mask[
                _PAIR_START : _PAIR_START + 3
            ]
        )
    )
    assert not bool(jnp.any(result.witness.route_map.newborn_mask[_PAIR_START:]))
    assert not bool(jnp.any(result.witness.route_map.retired_mask[_PAIR_START:]))
    assert bool(result.witness.route_map.unique_full_identity_matches)
    assert bool(result.witness.route_map.unique_source_identity_use)
    assert bool(route.result_integrity_valid(source, result))


@pytest.mark.unit
def test_new_pair_descriptor_without_explicit_admission_fails_closed() -> None:
    route = _route()
    source = _genesis(route)
    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (2, 3), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert not bool(result.witness.destination_inputs_valid)
    assert not bool(result.witness.transaction_applied)
    _assert_tree_bit_exact(result.ledger, source)
    assert bool(route.result_integrity_valid(source, result))


@pytest.mark.unit
def test_same_descriptor_reintroduced_after_retirement_is_a_new_birth() -> None:
    route = _route()
    genesis = _genesis(route)
    removed = route.prepare_successor(
        genesis,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    assert bool(removed.witness.transaction_applied)

    reintroduced = route.prepare_successor(
        removed.ledger,
        destination_source_clock_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (1, 3), (0, 2)),
        pair_admission_mask=_admissions(2),
    )

    assert bool(reintroduced.witness.transaction_applied)
    slot = _PAIR_START + 2
    assert bool(reintroduced.witness.route_map.newborn_mask[slot])
    assert int(reintroduced.witness.route_map.source_slots[slot]) == -1
    np.testing.assert_array_equal(reintroduced.ledger.descriptor[slot], (0, 2))
    np.testing.assert_array_equal(reintroduced.ledger.birth_words[slot], (0, 2))
    np.testing.assert_array_equal(
        reintroduced.ledger.birth_source_words[slot],
        (0, 2),
    )
    assert int(reintroduced.ledger.birth_event[slot]) == int(
        HCCLFeatureBirthEvent.PAIR_ADMISSION
    )
    assert not np.array_equal(
        np.asarray(reintroduced.ledger.birth_words[slot]),
        np.asarray(genesis.birth_words[_PAIR_START + 1]),
    )


@pytest.mark.unit
@pytest.mark.parametrize("field_name", ["birth_words", "birth_source_words"])
def test_resealed_zero_pair_admission_identity_is_rejected(
    field_name: str,
) -> None:
    route = _route()
    source = _genesis(route)
    admitted = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(0),
    )
    values = getattr(admitted.ledger, field_name).at[_PAIR_START].set(
        jnp.zeros((2,), dtype=jnp.uint32)
    )
    tampered = admitted.ledger.replace(**{field_name: values})
    resealed = route._seal_ledger(tampered)

    assert not bool(route.ledger_valid(resealed))
    rejected = route.prepare_successor(
        resealed,
        destination_source_clock_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    assert not bool(rejected.witness.source_ledger_valid)
    assert not bool(rejected.witness.transaction_applied)
    _assert_tree_bit_exact(rejected.ledger, resealed)
    assert bool(route.result_integrity_valid(resealed, rejected))


@pytest.mark.unit
def test_context_rebirth_is_newborn_and_never_transfers_h2_parameters() -> None:
    route = _route()
    source = _genesis(route)
    context_births = jnp.asarray(((0, 1), (0, 0), (0, 0)), dtype=jnp.uint32)

    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=context_births,
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert bool(result.witness.transaction_applied)
    slot = _CONTEXT_START
    assert bool(result.witness.route_map.newborn_mask[slot])
    assert int(result.witness.route_map.source_slots[slot]) == -1
    assert bool(result.witness.route_map.retired_mask[slot])
    np.testing.assert_array_equal(result.ledger.birth_words[slot], (0, 1))
    np.testing.assert_array_equal(result.ledger.birth_source_words[slot], (0, 1))
    assert int(result.ledger.birth_event[slot]) == int(
        HCCLFeatureBirthEvent.CONTEXT_ALLOCATION
    )
    assert int(result.witness.route_map.source_slots[slot + 1]) == slot + 1


@pytest.mark.unit
@pytest.mark.parametrize(
    "mutator",
    [
        lambda state: state.replace(
            birth_words=state.birth_words.at[_PAIR_START].set(
                jnp.asarray((0, 9), dtype=jnp.uint32)
            )
        ),
        lambda state: state.replace(
            birth_source_words=state.birth_source_words.at[_PAIR_START].set(
                jnp.asarray((0, 9), dtype=jnp.uint32)
            )
        ),
        lambda state: state.replace(
            parents=state.parents.at[_PAIR_START].set(
                jnp.asarray((2, 3), dtype=jnp.int32)
            )
        ),
        lambda state: state.replace(
            birth_event=state.birth_event.at[_PAIR_START].set(
                jnp.asarray(HCCLFeatureBirthEvent.PAIR_ADMISSION, dtype=jnp.int32)
            )
        ),
        lambda state: state.replace(
            source_clock_words=jnp.asarray((0, 3), dtype=jnp.uint32)
        ),
    ],
)
def test_source_birth_record_or_clock_tamper_fails_closed(
    mutator: Callable[[object], object],
) -> None:
    route = _route()
    source = _genesis(route)
    forged = mutator(source)

    result = route.prepare_successor(
        forged,  # type: ignore[arg-type]
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert not bool(result.witness.source_ledger_valid)
    assert not bool(result.witness.transaction_applied)
    _assert_tree_bit_exact(result.ledger, forged)
    assert bool(
        route.witness_integrity_valid(
            forged,  # type: ignore[arg-type]
            result.candidate_ledger,
            result.witness,
        )
    )


@pytest.mark.unit
def test_duplicate_destination_and_foreign_agent_source_fail_closed() -> None:
    route = _route()
    source = _genesis(route)
    duplicate = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 1), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    assert not bool(duplicate.witness.destination_inputs_valid)
    assert not bool(duplicate.witness.transaction_applied)
    assert bool(duplicate.witness.route_map.unique_full_identity_matches)
    assert not bool(duplicate.witness.route_map.unique_source_identity_use)
    assert not bool(duplicate.witness.route_map_valid)
    _assert_tree_bit_exact(duplicate.ledger, source)
    assert bool(
        route.witness_integrity_valid(
            source,
            duplicate.candidate_ledger,
            duplicate.witness,
        )
    )

    foreign_route = _route(agent_index=1)
    foreign = foreign_route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    assert not bool(foreign.witness.source_ledger_valid)
    assert not bool(foreign.witness.transaction_applied)
    _assert_tree_bit_exact(foreign.ledger, source)
    assert bool(
        foreign_route.witness_integrity_valid(
            source,
            foreign.candidate_ledger,
            foreign.witness,
        )
    )


@pytest.mark.unit
def test_skipped_source_clock_returns_source_with_an_integral_rejection() -> None:
    route = _route()
    source = _genesis(route)
    skipped = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 2), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert not bool(skipped.witness.source_clock_is_successor)
    assert not bool(skipped.witness.transaction_applied)
    assert bool(skipped.witness.complete_source_returned)
    _assert_tree_bit_exact(skipped.ledger, source)
    assert bool(route.result_integrity_valid(source, skipped))
    assert bool(
        route.witness_integrity_valid(
            source,
            skipped.candidate_ledger,
            skipped.witness,
        )
    )


@pytest.mark.unit
def test_scrubbed_invalid_context_request_remains_exactly_auditable() -> None:
    route = _route()
    source = _genesis(route)
    requested_births = jnp.zeros((3, 2), dtype=jnp.uint32).at[2].set(
        jnp.asarray((0, 1), dtype=jnp.uint32)
    )
    rejected = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=requested_births,
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert not bool(rejected.witness.destination_inputs_valid)
    assert not bool(rejected.witness.transaction_applied)
    np.testing.assert_array_equal(
        rejected.candidate_ledger.birth_words[_CONTEXT_START + 2],
        (0, 0),
    )
    _assert_tree_bit_exact(rejected.ledger, source)
    assert bool(
        route.witness_integrity_valid(
            source,
            rejected.candidate_ledger,
            rejected.witness,
        )
    )


@pytest.mark.unit
def test_no_semantic_change_advances_only_source_clock() -> None:
    route = _route()
    source = _genesis(route)
    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )

    assert bool(result.witness.transaction_applied)
    assert not bool(result.witness.semantic_bank_changed)
    np.testing.assert_array_equal(result.ledger.semantic_generation_words, (0, 0))
    np.testing.assert_array_equal(result.ledger.source_clock_words, (0, 1))
    assert int(result.witness.route_map.newborn_count) == 0
    assert int(result.witness.route_map.retired_count) == 0
    assert int(result.witness.route_map.survivor_count) == 25


@pytest.mark.unit
def test_witness_tamper_is_detected_and_monolithic_jit_is_rejected() -> None:
    route = _route()
    source = _genesis(route)
    result = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    forged = result.witness.replace(
        route_map=result.witness.route_map.replace(
            source_slots=result.witness.route_map.source_slots.at[0].set(-1)
        )
    )
    assert not bool(
        route.witness_integrity_valid(source, result.candidate_ledger, forged)
    )
    forged_request = result.witness.replace(
        requested_pair_descriptors=result.witness.requested_pair_descriptors.at[
            0
        ].set(jnp.asarray((2, 3), dtype=jnp.int32))
    )
    assert not bool(
        route.witness_integrity_valid(
            source,
            result.candidate_ledger,
            forged_request,
        )
    )
    forged_admission = result.witness.replace(
        requested_pair_admission_mask=(
            result.witness.requested_pair_admission_mask.at[0].set(True)
        )
    )
    assert not bool(
        route.witness_integrity_valid(
            source,
            result.candidate_ledger,
            forged_admission,
        )
    )

    compiled = jax.jit(
        lambda clock: route.prepare_successor(
            source,
            destination_source_clock_words=clock,
            context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
            context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
            pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
            pair_admission_mask=_admissions(),
        ).ledger.source_clock_words
    )
    with pytest.raises(TypeError, match="host/eager-only"):
        compiled(jnp.asarray((0, 1), dtype=jnp.uint32))


@pytest.mark.unit
def test_result_and_ledger_dataclasses_are_frozen() -> None:
    route = _route()
    source = _genesis(route)
    result: HCCLFeatureConsumerRouteResult = route.prepare_successor(
        source,
        destination_source_clock_words=jnp.asarray((0, 1), dtype=jnp.uint32),
        context_active=jnp.asarray((True, True, False), dtype=jnp.bool_),
        context_birth_words=jnp.zeros((3, 2), dtype=jnp.uint32),
        pair_descriptors=_pairs((0, 1), (0, 2), (1, 3)),
        pair_admission_mask=_admissions(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(result.ledger, "agent_index", jnp.int32(1))
