"""Focused tests for the external generated-feature identity ledger."""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any, cast

import numpy as np
import pytest

import alberta_framework.evaluation.generated_birth_identity_ledger as ledger_module
from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
)
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    CANDIDATE_OVERDEPTH_REGENERATION_CHANNEL,
    CANDIDATE_PARENT_REBOUND_CHANNEL,
    CASCADE_ACTIVE_REFILL_CHANNEL,
    DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
    GENERATED_BIRTH_IDENTITY_CHANNELS,
    GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST,
    GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA,
    ORDINARY_CANDIDATE_REFRESH_CHANNEL,
    POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL,
    RAW_SOURCE_IDENTITY_CHANNEL,
    GeneratedBirthIdentityEvent,
    GeneratedBirthIdentityLedgerConfig,
    GeneratedBirthIdentityLedgerConstructionError,
    GeneratedBirthIdentityLedgerState,
    build_generated_birth_identity_event,
    build_generated_birth_identity_transaction,
    derive_generated_birth_identity,
    generated_birth_identity_event_sha256,
    generated_birth_identity_ledger_state_sha256,
    generated_birth_identity_transaction_sha256,
    initialize_generated_birth_identity_ledger,
    validate_generated_birth_identity_transaction,
)

pytestmark = pytest.mark.unit


def _config() -> GeneratedBirthIdentityLedgerConfig:
    return GeneratedBirthIdentityLedgerConfig(
        namespace="generated-class-v0-paired-development",
        active_slots=7,
        candidate_slots=3,
        raw_feature_slots=2,
        max_depth=4,
        learn_generator_resources=True,
    )


def _descriptor_arrays(
    config: GeneratedBirthIdentityLedgerConfig,
) -> dict[str, np.ndarray]:
    candidate_count = config.candidate_slots
    return {
        "active_parent_a": np.asarray((0, 1, 0, 1, 2, 3, 4), dtype=np.int32),
        "active_parent_b": np.asarray((-1, -1, 1, 0, 1, 2, 3), dtype=np.int32),
        "active_ops": np.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM, OP_PRODUCT, OP_GATED, OP_PRODUCT),
            dtype=np.int32,
        ),
        "active_depth": np.asarray((0, 0, 1, 1, 2, 2, 3), dtype=np.int32),
        "active_generator_policy": np.zeros((config.active_slots,), dtype=np.int32),
        "candidate_parent_a": np.asarray((5, 3, 6), dtype=np.int32)[:candidate_count],
        "candidate_parent_b": np.asarray((1, 2, 3), dtype=np.int32)[:candidate_count],
        "candidate_ops": np.asarray((OP_PRODUCT, OP_SUM, OP_GATED), dtype=np.int32)[
            :candidate_count
        ],
        "candidate_depth": np.asarray((3, 2, 4), dtype=np.int32)[:candidate_count],
        "candidate_generator_policy": np.zeros((candidate_count,), dtype=np.int32),
    }


def _state(
    *,
    config: GeneratedBirthIdentityLedgerConfig | None = None,
    seed: int = 402,
    learner_step: int = 17,
    changes: dict[str, np.ndarray] | None = None,
) -> GeneratedBirthIdentityLedgerState:
    resolved_config = _config() if config is None else config
    descriptors = _descriptor_arrays(resolved_config)
    if changes is not None:
        descriptors.update(changes)
    return initialize_generated_birth_identity_ledger(
        resolved_config,
        paired_development_life_seed=seed,
        learner_step=learner_step,
        active_parent_a=descriptors["active_parent_a"],
        active_parent_b=descriptors["active_parent_b"],
        active_ops=descriptors["active_ops"],
        active_depth=descriptors["active_depth"],
        active_generator_policy=descriptors["active_generator_policy"],
        candidate_parent_a=descriptors["candidate_parent_a"],
        candidate_parent_b=descriptors["candidate_parent_b"],
        candidate_ops=descriptors["candidate_ops"],
        candidate_depth=descriptors["candidate_depth"],
        candidate_generator_policy=descriptors["candidate_generator_policy"],
    )


def _event(
    config: GeneratedBirthIdentityLedgerConfig,
    pre: GeneratedBirthIdentityLedgerState,
    *,
    learner_step: int = 18,
    generator_policy_sampled: bool | None = None,
    generator_policy_id: int = 1,
    active_parent_a: np.ndarray | None = None,
    active_parent_b: np.ndarray | None = None,
    active_ops: np.ndarray | None = None,
    active_depth: np.ndarray | None = None,
    active_generator_policy: np.ndarray | None = None,
    candidate_staged_parent_a: np.ndarray | None = None,
    candidate_staged_parent_b: np.ndarray | None = None,
    candidate_staged_ops: np.ndarray | None = None,
    candidate_staged_depth: np.ndarray | None = None,
    candidate_staged_generator_policy: np.ndarray | None = None,
    candidate_parent_a: np.ndarray | None = None,
    candidate_parent_b: np.ndarray | None = None,
    candidate_ops: np.ndarray | None = None,
    candidate_depth: np.ndarray | None = None,
    candidate_generator_policy: np.ndarray | None = None,
    promotion_active_slot: int = -1,
    promotion_candidate_slot: int = -1,
    direct_active_replacement_slot: int = -1,
    cascade_refill_mask: np.ndarray | None = None,
    ordinary_candidate_refresh_slot: int = -1,
    post_promotion_candidate_refresh_slot: int | None = None,
) -> GeneratedBirthIdentityEvent:
    resolved_candidate_parent_a = (
        pre.candidate_parent_a if candidate_parent_a is None else candidate_parent_a
    )
    resolved_candidate_parent_b = (
        pre.candidate_parent_b if candidate_parent_b is None else candidate_parent_b
    )
    resolved_candidate_ops = pre.candidate_ops if candidate_ops is None else candidate_ops
    resolved_candidate_depth = pre.candidate_depth if candidate_depth is None else candidate_depth
    resolved_candidate_policy = (
        pre.candidate_generator_policy
        if candidate_generator_policy is None
        else candidate_generator_policy
    )
    refresh_slot = (
        ordinary_candidate_refresh_slot
        if ordinary_candidate_refresh_slot >= 0
        else promotion_candidate_slot
    )
    default_staged_parent_a = np.array(pre.candidate_parent_a, copy=True)
    default_staged_parent_b = np.array(pre.candidate_parent_b, copy=True)
    default_staged_ops = np.array(pre.candidate_ops, copy=True)
    default_staged_depth = np.array(pre.candidate_depth, copy=True)
    default_staged_policy = np.array(pre.candidate_generator_policy, copy=True)
    if refresh_slot >= 0:
        if candidate_generator_policy is None:
            resolved_candidate_policy = np.array(resolved_candidate_policy, copy=True)
            resolved_candidate_policy[refresh_slot] = generator_policy_id
        default_staged_parent_a[refresh_slot] = resolved_candidate_parent_a[refresh_slot]
        default_staged_parent_b[refresh_slot] = resolved_candidate_parent_b[refresh_slot]
        default_staged_ops[refresh_slot] = resolved_candidate_ops[refresh_slot]
        default_staged_depth[refresh_slot] = resolved_candidate_depth[refresh_slot]
        default_staged_policy[refresh_slot] = resolved_candidate_policy[refresh_slot]
    return build_generated_birth_identity_event(
        config,
        pre,
        learner_step=learner_step,
        generator_policy_sampled=(
            config.learn_generator_resources
            if generator_policy_sampled is None
            else generator_policy_sampled
        ),
        generator_policy_id=generator_policy_id,
        active_parent_a=pre.active_parent_a if active_parent_a is None else active_parent_a,
        active_parent_b=pre.active_parent_b if active_parent_b is None else active_parent_b,
        active_ops=pre.active_ops if active_ops is None else active_ops,
        active_depth=pre.active_depth if active_depth is None else active_depth,
        active_generator_policy=(
            pre.active_generator_policy
            if active_generator_policy is None
            else active_generator_policy
        ),
        candidate_staged_parent_a=(
            default_staged_parent_a
            if candidate_staged_parent_a is None
            else candidate_staged_parent_a
        ),
        candidate_staged_parent_b=(
            default_staged_parent_b
            if candidate_staged_parent_b is None
            else candidate_staged_parent_b
        ),
        candidate_staged_ops=(
            default_staged_ops if candidate_staged_ops is None else candidate_staged_ops
        ),
        candidate_staged_depth=(
            default_staged_depth if candidate_staged_depth is None else candidate_staged_depth
        ),
        candidate_staged_generator_policy=(
            default_staged_policy
            if candidate_staged_generator_policy is None
            else candidate_staged_generator_policy
        ),
        candidate_parent_a=resolved_candidate_parent_a,
        candidate_parent_b=resolved_candidate_parent_b,
        candidate_ops=resolved_candidate_ops,
        candidate_depth=resolved_candidate_depth,
        candidate_generator_policy=resolved_candidate_policy,
        promotion_active_slot=promotion_active_slot,
        promotion_candidate_slot=promotion_candidate_slot,
        direct_active_replacement_slot=direct_active_replacement_slot,
        cascade_refill_mask=cascade_refill_mask,
        ordinary_candidate_refresh_slot=ordinary_candidate_refresh_slot,
        post_promotion_candidate_refresh_slot=post_promotion_candidate_refresh_slot,
    )


def _promotion_event(
    config: GeneratedBirthIdentityLedgerConfig,
    pre: GeneratedBirthIdentityLedgerState,
) -> GeneratedBirthIdentityEvent:
    active_parent_a = np.array(pre.active_parent_a, copy=True)
    active_parent_b = np.array(pre.active_parent_b, copy=True)
    active_ops = np.array(pre.active_ops, copy=True)
    active_depth = np.array(pre.active_depth, copy=True)
    active_provenance = np.array(pre.active_generator_policy, copy=True)
    # Slot 4 transfers candidate 1 exactly.  In the pre graph, slot 6 is its
    # only transitive descendant; slot 5 is deliberately a nondescendant.
    active_parent_a[4], active_parent_b[4] = 3, 2
    active_ops[4], active_depth[4], active_provenance[4] = OP_SUM, 2, 0
    active_parent_a[6], active_parent_b[6] = 0, 1
    active_ops[6], active_depth[6], active_provenance[6] = OP_PRODUCT, 1, 1

    candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    candidate_ops = np.array(pre.candidate_ops, copy=True)
    candidate_depth = np.array(pre.candidate_depth, copy=True)
    candidate_provenance = np.array(pre.candidate_generator_policy, copy=True)
    candidate_parent_a[1] = 6
    candidate_ops[1], candidate_depth[1], candidate_provenance[1] = OP_PRODUCT, 2, 1
    candidate_depth[2] = 2
    candidate_staged_depth = np.array(pre.candidate_depth, copy=True)
    candidate_staged_depth[1] = 4
    return _event(
        config,
        pre,
        promotion_active_slot=4,
        promotion_candidate_slot=1,
        cascade_refill_mask=np.asarray(
            (False, False, False, False, False, False, True),
            dtype=np.bool_,
        ),
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_provenance,
        candidate_staged_parent_a=candidate_parent_a,
        candidate_staged_ops=candidate_ops,
        candidate_staged_depth=candidate_staged_depth,
        candidate_staged_generator_policy=candidate_provenance,
        candidate_parent_a=candidate_parent_a,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_provenance,
    )


def _overdepth_promotion_case() -> tuple[
    GeneratedBirthIdentityLedgerConfig,
    GeneratedBirthIdentityLedgerState,
    GeneratedBirthIdentityEvent,
]:
    config = GeneratedBirthIdentityLedgerConfig(
        namespace="generated-overdepth-development",
        active_slots=5,
        candidate_slots=2,
        raw_feature_slots=2,
        max_depth=2,
        learn_generator_resources=True,
    )
    pre = initialize_generated_birth_identity_ledger(
        config,
        paired_development_life_seed=811,
        learner_step=9,
        active_parent_a=np.asarray((0, 1, 0, 0, 0), dtype=np.int32),
        active_parent_b=np.asarray((-1, -1, 1, 1, 1), dtype=np.int32),
        active_ops=np.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT, OP_PRODUCT),
            dtype=np.int32,
        ),
        active_depth=np.asarray((0, 0, 1, 1, 1), dtype=np.int32),
        active_generator_policy=np.zeros((5,), dtype=np.int32),
        candidate_parent_a=np.asarray((2, 4), dtype=np.int32),
        candidate_parent_b=np.asarray((3, 1), dtype=np.int32),
        candidate_ops=np.asarray((OP_SUM, OP_PRODUCT), dtype=np.int32),
        candidate_depth=np.asarray((2, 2), dtype=np.int32),
        candidate_generator_policy=np.zeros((2,), dtype=np.int32),
    )

    active_parent_a = np.array(pre.active_parent_a, copy=True)
    active_parent_b = np.array(pre.active_parent_b, copy=True)
    active_ops = np.array(pre.active_ops, copy=True)
    active_depth = np.array(pre.active_depth, copy=True)
    active_policy = np.array(pre.active_generator_policy, copy=True)
    active_parent_a[4], active_parent_b[4] = 2, 3
    active_ops[4], active_depth[4], active_policy[4] = OP_SUM, 2, 0

    staged_parent_a = np.array(pre.candidate_parent_a, copy=True)
    staged_parent_b = np.array(pre.candidate_parent_b, copy=True)
    staged_ops = np.array(pre.candidate_ops, copy=True)
    staged_depth = np.array(pre.candidate_depth, copy=True)
    staged_policy = np.array(pre.candidate_generator_policy, copy=True)
    staged_parent_a[0], staged_parent_b[0] = 0, 1
    staged_ops[0], staged_depth[0], staged_policy[0] = OP_PRODUCT, 1, 1

    final_parent_a = np.array(staged_parent_a, copy=True)
    final_parent_b = np.array(staged_parent_b, copy=True)
    final_ops = np.array(staged_ops, copy=True)
    final_depth = np.array(staged_depth, copy=True)
    final_policy = np.array(staged_policy, copy=True)
    final_parent_a[1], final_parent_b[1] = 0, 1
    final_ops[1], final_depth[1], final_policy[1] = OP_SUM, 1, 1

    event = build_generated_birth_identity_event(
        config,
        pre,
        learner_step=10,
        generator_policy_sampled=True,
        generator_policy_id=1,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_policy,
        candidate_staged_parent_a=staged_parent_a,
        candidate_staged_parent_b=staged_parent_b,
        candidate_staged_ops=staged_ops,
        candidate_staged_depth=staged_depth,
        candidate_staged_generator_policy=staged_policy,
        candidate_parent_a=final_parent_a,
        candidate_parent_b=final_parent_b,
        candidate_ops=final_ops,
        candidate_depth=final_depth,
        candidate_generator_policy=final_policy,
        promotion_active_slot=4,
        promotion_candidate_slot=0,
    )
    return config, pre, event


def _post_refresh_overdepth_overlap_case() -> tuple[
    GeneratedBirthIdentityLedgerConfig,
    GeneratedBirthIdentityLedgerState,
    GeneratedBirthIdentityEvent,
]:
    config = GeneratedBirthIdentityLedgerConfig(
        namespace="generated-post-refresh-overdepth-development",
        active_slots=6,
        candidate_slots=2,
        raw_feature_slots=2,
        max_depth=2,
        learn_generator_resources=True,
    )
    pre = initialize_generated_birth_identity_ledger(
        config,
        paired_development_life_seed=812,
        learner_step=11,
        active_parent_a=np.asarray((0, 1, 0, 0, 0, 4), dtype=np.int32),
        active_parent_b=np.asarray((-1, -1, 1, 1, -1, 1), dtype=np.int32),
        active_ops=np.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT, OP_RAW, OP_PRODUCT),
            dtype=np.int32,
        ),
        active_depth=np.asarray((0, 0, 1, 1, 0, 1), dtype=np.int32),
        active_generator_policy=np.zeros((6,), dtype=np.int32),
        candidate_parent_a=np.asarray((2, 2), dtype=np.int32),
        candidate_parent_b=np.asarray((3, 1), dtype=np.int32),
        candidate_ops=np.asarray((OP_SUM, OP_PRODUCT), dtype=np.int32),
        candidate_depth=np.asarray((2, 2), dtype=np.int32),
        candidate_generator_policy=np.zeros((2,), dtype=np.int32),
    )
    active_parent_a = np.array(pre.active_parent_a, copy=True)
    active_parent_b = np.array(pre.active_parent_b, copy=True)
    active_ops = np.array(pre.active_ops, copy=True)
    active_depth = np.array(pre.active_depth, copy=True)
    active_policy = np.array(pre.active_generator_policy, copy=True)
    active_parent_a[4], active_parent_b[4] = 2, 3
    active_ops[4], active_depth[4], active_policy[4] = OP_SUM, 2, 0
    active_parent_a[5], active_parent_b[5] = 2, 3
    active_ops[5], active_depth[5], active_policy[5] = OP_PRODUCT, 2, 1

    staged_parent_a = np.array(pre.candidate_parent_a, copy=True)
    staged_parent_b = np.array(pre.candidate_parent_b, copy=True)
    staged_ops = np.array(pre.candidate_ops, copy=True)
    staged_depth = np.array(pre.candidate_depth, copy=True)
    staged_policy = np.array(pre.candidate_generator_policy, copy=True)
    staged_parent_a[0], staged_parent_b[0] = 5, 1
    staged_ops[0], staged_depth[0], staged_policy[0] = OP_PRODUCT, 2, 1

    final_parent_a = np.array(staged_parent_a, copy=True)
    final_parent_b = np.array(staged_parent_b, copy=True)
    final_ops = np.array(staged_ops, copy=True)
    final_depth = np.array(staged_depth, copy=True)
    final_policy = np.array(staged_policy, copy=True)
    final_parent_a[0], final_parent_b[0] = 0, 1
    final_ops[0], final_depth[0], final_policy[0] = OP_SUM, 1, 1
    event = build_generated_birth_identity_event(
        config,
        pre,
        learner_step=12,
        generator_policy_sampled=True,
        generator_policy_id=1,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_policy,
        candidate_staged_parent_a=staged_parent_a,
        candidate_staged_parent_b=staged_parent_b,
        candidate_staged_ops=staged_ops,
        candidate_staged_depth=staged_depth,
        candidate_staged_generator_policy=staged_policy,
        candidate_parent_a=final_parent_a,
        candidate_parent_b=final_parent_b,
        candidate_ops=final_ops,
        candidate_depth=final_depth,
        candidate_generator_policy=final_policy,
        promotion_active_slot=4,
        promotion_candidate_slot=0,
        cascade_refill_mask=np.asarray(
            (False, False, False, False, False, True),
            dtype=np.bool_,
        ),
    )
    return config, pre, event


def _rehash_event(
    event: GeneratedBirthIdentityEvent,
    **changes: object,
) -> GeneratedBirthIdentityEvent:
    typed_changes = cast(
        Any,
        {
            name: _readonly(value) if type(value) is np.ndarray else value
            for name, value in changes.items()
        },
    )
    changed = dataclasses.replace(
        event,
        **typed_changes,
        integrity_sha256="0" * 64,
    )
    return dataclasses.replace(
        changed,
        integrity_sha256=generated_birth_identity_event_sha256(changed),
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(value)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


def _rehash_state(
    state: GeneratedBirthIdentityLedgerState,
    **changes: object,
) -> GeneratedBirthIdentityLedgerState:
    changed = dataclasses.replace(
        state,
        **cast(Any, changes),
        integrity_sha256="0" * 64,
    )
    return dataclasses.replace(
        changed,
        integrity_sha256=generated_birth_identity_ledger_state_sha256(changed),
    )


def _rehash_assignments(
    assignments: ledger_module.GeneratedBirthIdentityAssignments,
    **changes: object,
) -> ledger_module.GeneratedBirthIdentityAssignments:
    changed = dataclasses.replace(
        assignments,
        **cast(Any, changes),
        integrity_sha256="0" * 64,
    )
    return dataclasses.replace(
        changed,
        integrity_sha256=ledger_module._assignments_sha256(changed),
    )


def _rehash_audit_transaction(
    transaction: ledger_module.GeneratedBirthIdentityTransaction,
    **changes: object,
) -> ledger_module.GeneratedBirthIdentityTransaction:
    audit = dataclasses.replace(
        transaction.audit,
        **cast(Any, changes),
        transaction_sha256="0" * 64,
    )
    changed = dataclasses.replace(transaction, audit=audit)
    return dataclasses.replace(
        changed,
        audit=dataclasses.replace(
            audit,
            transaction_sha256=generated_birth_identity_transaction_sha256(changed),
        ),
    )


def test_initial_state_is_fixed_shape_external_and_exactly_accounted() -> None:
    config = _config()
    state = _state()

    assert state.schema == GENERATED_BIRTH_IDENTITY_LEDGER_SCHEMA
    assert state.active_identity.shape == (config.active_slots, 32)
    assert state.active_parent_identity_snapshot.shape == (
        config.active_slots,
        2,
        32,
    )
    assert state.candidate_identity.shape == (config.candidate_slots, 32)
    assert state.candidate_parent_identity_snapshot.shape == (
        config.candidate_slots,
        2,
        32,
    )
    assert state.active_identity.dtype == np.uint8
    assert state.candidate_identity.dtype == np.uint8
    assert state.candidate_parent_identity_snapshot.dtype == np.uint8
    assert np.any(state.active_parent_identity_snapshot[0, 0])
    assert not np.any(state.active_parent_identity_snapshot[0, 1])
    assert not np.array_equal(
        state.active_parent_identity_snapshot[0, 0],
        state.active_parent_identity_snapshot[1, 0],
    )
    assert state.active_ops.dtype == np.int32
    assert state.active_depth.dtype == np.int32
    assert state.active_generator_policy.dtype == np.int32
    assert state.candidate_ops.dtype == np.int32
    assert state.active_generator_policy_sampled.dtype == np.bool_
    assert state.candidate_generator_policy_sampled.dtype == np.bool_
    assert not np.any(state.active_generator_policy_sampled)
    assert not np.any(state.candidate_generator_policy_sampled)
    assert state.persistent_array_nbytes == 117 * (config.active_slots + config.candidate_slots)
    assert state.expected_persistent_array_nbytes_formula == (
        "117 * (active_slots + candidate_slots)"
    )
    assert state.integrity_sha256 == generated_birth_identity_ledger_state_sha256(state)
    assert not state.active_identity.flags.writeable
    assert not state.active_parent_identity_snapshot.flags.writeable
    assert not state.candidate_identity.flags.writeable
    assert not state.candidate_parent_identity_snapshot.flags.writeable
    for immutable in (
        state.active_identity,
        state.active_parent_a,
        state.active_generator_policy_sampled,
        state.candidate_identity,
        state.candidate_generator_policy_sampled,
    ):
        with pytest.raises(ValueError):
            immutable.setflags(write=True)

    assert config.runner_side_state_only
    assert not config.compositional_feature_state_fields_added
    assert not config.caller_supplied_events_authenticated
    assert config.public_core_event_trace_required_for_authentication
    assert config.public_core_event_trace_available
    assert not config.public_core_event_trace_consumed
    assert config.structural_lifetime_descriptor_complete
    assert config.active_parent_snapshot_complete
    assert config.raw_source_identity_bound
    assert not config.theta_bound_to_structural_lifetime_identity
    assert config.theta_may_adapt_within_structural_lifetime
    assert not config.exact_functional_expression_identity_claimed
    assert not config.cryptographic_collision_impossibility_claimed
    assert not config.dead_identity_history_retained
    assert not config.historical_global_uniqueness_claimed
    assert not config.lifecycle_prerequisite_complete
    assert not config.execution_authorized
    assert not config.runner_authorized
    assert not config.artifact_writes_authorized
    assert not config.evidence_authorized
    assert not config.scientific_promotion_allowed


def test_id_domain_is_target_outcome_blind_and_collision_separated() -> None:
    parameters = inspect.signature(derive_generated_birth_identity).parameters
    assert "target" not in parameters
    assert "targets" not in parameters
    assert "outcome" not in parameters
    assert "reward" not in parameters

    namespace = _config().namespace
    identities = {
        derive_generated_birth_identity(
            namespace=namespace,
            paired_development_life_seed=seed,
            learner_step=step,
            event_channel=channel,
            slot=slot,
            ordinal=ordinal,
        )
        for seed, step, channel, slot, ordinal in (
            (402, 18, DIRECT_ACTIVE_REPLACEMENT_CHANNEL, 4, 0),
            (403, 18, DIRECT_ACTIVE_REPLACEMENT_CHANNEL, 4, 0),
            (402, 19, DIRECT_ACTIVE_REPLACEMENT_CHANNEL, 4, 0),
            (402, 18, CASCADE_ACTIVE_REFILL_CHANNEL, 4, 0),
            (402, 18, DIRECT_ACTIVE_REPLACEMENT_CHANNEL, 5, 0),
            (402, 18, DIRECT_ACTIVE_REPLACEMENT_CHANNEL, 4, 1),
        )
    }
    assert len(identities) == 6
    assert all(type(identity) is bytes and len(identity) == 32 for identity in identities)
    other_namespace_identity = derive_generated_birth_identity(
        namespace=f"{namespace}-other",
        paired_development_life_seed=402,
        learner_step=18,
        event_channel=DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
        slot=4,
        ordinal=0,
    )
    assert other_namespace_identity not in identities

    target_a = {"target": "D", "outcome": 1.0}
    target_b = {"target": "A", "outcome": -999.0}
    identity_a = derive_generated_birth_identity(
        namespace=namespace,
        paired_development_life_seed=402,
        learner_step=18,
        event_channel=ORDINARY_CANDIDATE_REFRESH_CHANNEL,
        slot=1,
        ordinal=0,
    )
    identity_b = derive_generated_birth_identity(
        namespace=namespace,
        paired_development_life_seed=402,
        learner_step=18,
        event_channel=ORDINARY_CANDIDATE_REFRESH_CHANNEL,
        slot=1,
        ordinal=0,
    )
    assert target_a != target_b
    assert identity_a == identity_b


def test_event_hash_and_byte_fields_are_one_frozen_exact_enumeration() -> None:
    field_names = tuple(field.name for field in dataclasses.fields(GeneratedBirthIdentityEvent))
    assert field_names == (
        *ledger_module._EVENT_SCALAR_FIELD_NAMES,
        *ledger_module._EVENT_ARRAY_FIELD_NAMES,
        "integrity_sha256",
    )
    assert len(ledger_module._EVENT_ARRAY_FIELD_NAMES) == len(
        set(ledger_module._EVENT_ARRAY_FIELD_NAMES)
    )
    assert len(ledger_module._EVENT_ARRAY_FIELD_NAMES) == 23


def test_raw_source_tokens_are_stable_and_separated_from_every_live_bank() -> None:
    config = _config()
    state = _state()
    token_0 = derive_generated_birth_identity(
        namespace=config.namespace,
        paired_development_life_seed=402,
        learner_step=0,
        event_channel=RAW_SOURCE_IDENTITY_CHANNEL,
        slot=0,
        ordinal=0,
    )
    token_0_again = derive_generated_birth_identity(
        namespace=config.namespace,
        paired_development_life_seed=402,
        learner_step=0,
        event_channel=RAW_SOURCE_IDENTITY_CHANNEL,
        slot=0,
        ordinal=0,
    )
    token_1 = derive_generated_birth_identity(
        namespace=config.namespace,
        paired_development_life_seed=402,
        learner_step=0,
        event_channel=RAW_SOURCE_IDENTITY_CHANNEL,
        slot=1,
        ordinal=0,
    )
    other_seed = derive_generated_birth_identity(
        namespace=config.namespace,
        paired_development_life_seed=403,
        learner_step=0,
        event_channel=RAW_SOURCE_IDENTITY_CHANNEL,
        slot=0,
        ordinal=0,
    )
    assert token_0 == token_0_again
    assert len({token_0, token_1, other_seed}) == 3
    assert state.active_parent_identity_snapshot[0, 0].tobytes() == token_0
    assert state.active_parent_identity_snapshot[1, 0].tobytes() == token_1
    live = {
        row.tobytes()
        for row in np.concatenate((state.active_identity, state.candidate_identity), axis=0)
    }
    assert token_0 not in live
    assert token_1 not in live

    transaction = build_generated_birth_identity_transaction(
        config,
        state,
        _event(config, state),
    )
    assert transaction.post_state.active_parent_identity_snapshot[0, 0].tobytes() == token_0


def test_minimal_candidate_free_shape_has_exact_v3_byte_accounting() -> None:
    config = GeneratedBirthIdentityLedgerConfig(
        namespace="generated-minimal-candidate-free-development",
        active_slots=2,
        candidate_slots=0,
        raw_feature_slots=1,
        max_depth=1,
        learn_generator_resources=False,
    )
    empty_i32 = np.zeros((0,), dtype=np.int32)
    pre = initialize_generated_birth_identity_ledger(
        config,
        paired_development_life_seed=5,
        learner_step=0,
        active_parent_a=np.asarray((0, 0), dtype=np.int32),
        active_parent_b=np.asarray((-1, 0), dtype=np.int32),
        active_ops=np.asarray((OP_RAW, OP_PRODUCT), dtype=np.int32),
        active_depth=np.asarray((0, 1), dtype=np.int32),
        active_generator_policy=np.zeros((2,), dtype=np.int32),
        candidate_parent_a=empty_i32,
        candidate_parent_b=empty_i32,
        candidate_ops=empty_i32,
        candidate_depth=empty_i32,
        candidate_generator_policy=empty_i32,
    )
    event = build_generated_birth_identity_event(
        config,
        pre,
        learner_step=1,
        generator_policy_sampled=False,
        generator_policy_id=0,
        active_parent_a=pre.active_parent_a,
        active_parent_b=pre.active_parent_b,
        active_ops=pre.active_ops,
        active_depth=pre.active_depth,
        active_generator_policy=pre.active_generator_policy,
        candidate_staged_parent_a=empty_i32,
        candidate_staged_parent_b=empty_i32,
        candidate_staged_ops=empty_i32,
        candidate_staged_depth=empty_i32,
        candidate_staged_generator_policy=empty_i32,
        candidate_parent_a=empty_i32,
        candidate_parent_b=empty_i32,
        candidate_ops=empty_i32,
        candidate_depth=empty_i32,
        candidate_generator_policy=empty_i32,
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)

    for name in ledger_module._EVENT_ARRAY_FIELD_NAMES:
        with pytest.raises(ValueError):
            getattr(event, name).setflags(write=True)
    for field in dataclasses.fields(transaction.assignments):
        value = getattr(transaction.assignments, field.name)
        if type(value) is np.ndarray:
            with pytest.raises(ValueError):
                value.setflags(write=True)
    assert pre.persistent_array_nbytes == 117 * 2
    assert transaction.assignments.persistent_array_nbytes == 32 * (3 * 2)
    assert transaction.audit.event_fixed_array_nbytes == 23 * 2
    assert transaction.audit.expected_event_fixed_array_nbytes_formula == (
        "23 * active_slots + 45 * candidate_slots"
    )
    assert transaction.post_state.candidate_identity.shape == (0, 32)
    with pytest.raises(ValueError):
        transaction.assignments.cascade_active_birth_identity.setflags(write=True)


def test_promotion_transfer_refresh_then_cascade_rebound_order_is_exact() -> None:
    config = _config()
    pre = _state()
    event = _promotion_event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    post = transaction.post_state
    assignments = transaction.assignments

    # Promotion transfers an existing identity; it is not a new proposal.
    assert np.array_equal(post.active_identity[4], pre.candidate_identity[1])
    assert np.array_equal(
        assignments.promotion_transfer_active_identity[4],
        pre.candidate_identity[1],
    )
    assert np.array_equal(
        post.active_parent_identity_snapshot[4],
        pre.candidate_parent_identity_snapshot[1],
    )

    # The source refresh occurs before cascade.  Because it names cascade slot
    # 6, the final rebound identity must win and its snapshot must bind the
    # final cascade identity rather than the pre-cascade identity.
    assert bool(event.post_promotion_candidate_refresh_mask[1])
    assert bool(event.candidate_rebound_mask[1])
    assert np.any(assignments.post_promotion_candidate_birth_identity[1])
    assert np.any(assignments.candidate_rebound_identity[1])
    assert np.array_equal(post.candidate_identity[1], assignments.candidate_rebound_identity[1])
    assert not np.array_equal(
        post.candidate_identity[1],
        assignments.post_promotion_candidate_birth_identity[1],
    )
    assert np.array_equal(
        post.candidate_parent_identity_snapshot[1, 0],
        post.active_identity[6],
    )
    assert np.array_equal(
        post.candidate_parent_identity_snapshot[0, 0],
        post.active_identity[5],
    )
    assert np.array_equal(
        post.candidate_parent_identity_snapshot[2, 0],
        post.active_identity[6],
    )
    assert not bool(event.candidate_rebound_mask[0])
    assert not np.any(assignments.candidate_rebound_identity[0])
    assert np.array_equal(post.candidate_identity[0], pre.candidate_identity[0])
    assert post.candidate_depth[0] == 3
    assert post.candidate_depth[1] == 2
    assert post.candidate_depth[2] == 2
    assert post.candidate_ops[0] == pre.candidate_ops[0]
    assert post.candidate_generator_policy[0] == pre.candidate_generator_policy[0]
    assert not bool(post.active_generator_policy_sampled[4])
    assert bool(post.active_generator_policy_sampled[6])
    assert bool(post.candidate_generator_policy_sampled[1])
    assert not bool(post.candidate_generator_policy_sampled[2])
    assert transaction.audit.post_promotion_refresh_then_cascade_then_candidate_resolution
    assert transaction.audit.just_refreshed_candidate_rebound_count == 1
    assert transaction.audit.candidate_overdepth_regeneration_count == 0
    assert transaction.audit.structural_lifetime_descriptor_complete
    assert transaction.audit.active_parent_snapshot_complete
    assert not transaction.audit.theta_bound_to_structural_lifetime_identity
    assert not transaction.audit.exact_functional_expression_identity_claimed
    assert not transaction.audit.transition_collision_observed
    assert not transaction.audit.dead_identity_history_retained
    assert not transaction.audit.historical_global_uniqueness_claimed
    assert transaction.audit.state_persistent_array_nbytes == 117 * (7 + 3)
    assert transaction.audit.assignment_persistent_array_nbytes == 32 * (3 * 7 + 4 * 3)
    assert transaction.audit.event_fixed_array_nbytes == 23 * 7 + 45 * 3
    assert validate_generated_birth_identity_transaction(
        transaction,
        config=config,
        pre_state=pre,
        event=event,
    ).valid


def test_overdepth_regeneration_is_separate_from_rebound_and_follows_refresh() -> None:
    config, pre, event = _overdepth_promotion_case()
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    assignments = transaction.assignments
    post = transaction.post_state

    assert bool(event.post_promotion_candidate_refresh_mask[0])
    assert not np.any(event.candidate_rebound_mask)
    assert bool(event.candidate_overdepth_regeneration_mask[1])
    assert np.any(assignments.post_promotion_candidate_birth_identity[0])
    assert np.any(assignments.candidate_overdepth_regeneration_identity[1])
    assert not np.any(assignments.candidate_rebound_identity)
    assert np.array_equal(
        post.candidate_identity[1],
        assignments.candidate_overdepth_regeneration_identity[1],
    )
    assert post.candidate_depth[1] == 1
    assert post.candidate_generator_policy[1] == event.generator_policy_id
    assert bool(post.candidate_generator_policy_sampled[1])
    assert not bool(post.active_generator_policy_sampled[4])
    assert transaction.audit.candidate_overdepth_regeneration_count == 1
    assert transaction.audit.just_refreshed_candidate_overdepth_regeneration_count == 0
    assert CANDIDATE_OVERDEPTH_REGENERATION_CHANNEL in GENERATED_BIRTH_IDENTITY_CHANNELS


def test_post_promotion_refresh_then_overdepth_repair_keeps_both_births() -> None:
    config, pre, event = _post_refresh_overdepth_overlap_case()
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    assignments = transaction.assignments

    assert bool(event.post_promotion_candidate_refresh_mask[0])
    assert bool(event.candidate_overdepth_regeneration_mask[0])
    assert not bool(event.candidate_rebound_mask[0])
    assert np.any(assignments.post_promotion_candidate_birth_identity[0])
    assert np.any(assignments.candidate_overdepth_regeneration_identity[0])
    assert not np.array_equal(
        assignments.post_promotion_candidate_birth_identity[0],
        assignments.candidate_overdepth_regeneration_identity[0],
    )
    assert np.array_equal(
        transaction.post_state.candidate_identity[0],
        assignments.candidate_overdepth_regeneration_identity[0],
    )
    assert transaction.audit.post_promotion_candidate_birth_count == 1
    assert transaction.audit.cascade_active_birth_count == 1
    assert transaction.audit.candidate_overdepth_regeneration_count == 1
    assert transaction.audit.just_refreshed_candidate_overdepth_regeneration_count == 1


def test_promotion_transfers_preexisting_sampled_policy_instead_of_current_policy() -> None:
    config = _config()
    genesis = _state()
    sampled_event = _event(
        config,
        genesis,
        ordinary_candidate_refresh_slot=1,
        generator_policy_id=1,
    )
    pre = build_generated_birth_identity_transaction(
        config,
        genesis,
        sampled_event,
    ).post_state
    assert bool(pre.candidate_generator_policy_sampled[1])
    assert pre.candidate_generator_policy[1] == 1

    active_parent_a = np.array(pre.active_parent_a, copy=True)
    active_parent_b = np.array(pre.active_parent_b, copy=True)
    active_ops = np.array(pre.active_ops, copy=True)
    active_depth = np.array(pre.active_depth, copy=True)
    active_policy = np.array(pre.active_generator_policy, copy=True)
    active_parent_a[4], active_parent_b[4] = 3, 2
    active_ops[4], active_depth[4], active_policy[4] = OP_SUM, 2, 1
    active_parent_a[6], active_parent_b[6] = 0, 1
    active_ops[6], active_depth[6], active_policy[6] = OP_PRODUCT, 1, 2

    candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    candidate_ops = np.array(pre.candidate_ops, copy=True)
    candidate_depth = np.array(pre.candidate_depth, copy=True)
    candidate_policy = np.array(pre.candidate_generator_policy, copy=True)
    candidate_parent_a[1], candidate_parent_b[1] = 0, 1
    candidate_ops[1], candidate_depth[1], candidate_policy[1] = OP_PRODUCT, 1, 2
    candidate_depth[2] = 2
    event = _event(
        config,
        pre,
        learner_step=19,
        generator_policy_id=2,
        promotion_active_slot=4,
        promotion_candidate_slot=1,
        cascade_refill_mask=np.asarray(
            (False, False, False, False, False, False, True),
            dtype=np.bool_,
        ),
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_policy,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_policy,
    )
    post = build_generated_birth_identity_transaction(config, pre, event).post_state
    assert post.active_generator_policy[4] == 1
    assert bool(post.active_generator_policy_sampled[4])
    assert post.active_generator_policy[4] != event.generator_policy_id
    assert post.active_generator_policy[6] == event.generator_policy_id
    assert bool(post.active_generator_policy_sampled[6])
    assert post.candidate_generator_policy[1] == event.generator_policy_id
    assert bool(post.candidate_generator_policy_sampled[1])


def test_direct_replacement_cascade_and_ordinary_refresh_channels_are_distinct() -> None:
    direct_config = GeneratedBirthIdentityLedgerConfig(
        namespace="generated-class-v0-direct-development",
        active_slots=7,
        candidate_slots=0,
        raw_feature_slots=2,
        max_depth=4,
        learn_generator_resources=True,
    )
    direct_pre = _state(config=direct_config)
    direct_parent_a = np.array(direct_pre.active_parent_a, copy=True)
    direct_parent_b = np.array(direct_pre.active_parent_b, copy=True)
    direct_ops = np.array(direct_pre.active_ops, copy=True)
    direct_depth = np.array(direct_pre.active_depth, copy=True)
    direct_provenance = np.array(direct_pre.active_generator_policy, copy=True)
    direct_parent_a[3], direct_parent_b[3] = 0, 1
    direct_parent_a[5], direct_parent_b[5] = 2, 3
    direct_parent_a[6], direct_parent_b[6] = 3, 4
    direct_ops[3] = OP_PRODUCT
    direct_ops[5:] = np.asarray((OP_GATED, OP_PRODUCT))
    direct_depth[3] = 1
    direct_depth[5:] = np.asarray((2, 3), dtype=np.int32)
    direct_provenance[3] = 1
    direct_provenance[5:] = np.asarray((1, 1), dtype=np.int32)

    direct_event = _event(
        direct_config,
        direct_pre,
        direct_active_replacement_slot=3,
        cascade_refill_mask=np.asarray(
            (False, False, False, False, False, True, True),
            dtype=np.bool_,
        ),
        active_parent_a=direct_parent_a,
        active_parent_b=direct_parent_b,
        active_ops=direct_ops,
        active_depth=direct_depth,
        active_generator_policy=direct_provenance,
    )
    direct_tx = build_generated_birth_identity_transaction(
        direct_config,
        direct_pre,
        direct_event,
    )
    assert np.array_equal(
        direct_tx.post_state.active_identity[3],
        direct_tx.assignments.direct_active_birth_identity[3],
    )
    assert not np.any(direct_tx.assignments.cascade_active_birth_identity[4])
    assert np.any(direct_tx.assignments.cascade_active_birth_identity[5])
    assert np.any(direct_tx.assignments.cascade_active_birth_identity[6])
    assert not np.array_equal(
        direct_tx.assignments.direct_active_birth_identity[3],
        direct_tx.assignments.cascade_active_birth_identity[5],
    )

    config = _config()
    pre = _state()
    candidate_ops = np.array(pre.candidate_ops, copy=True)
    candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    candidate_depth = np.array(pre.candidate_depth, copy=True)
    candidate_provenance = np.array(pre.candidate_generator_policy, copy=True)
    candidate_parent_a[2], candidate_parent_b[2] = 0, 1
    candidate_ops[2], candidate_depth[2], candidate_provenance[2] = OP_PRODUCT, 1, 1
    ordinary_event = _event(
        config,
        pre,
        ordinary_candidate_refresh_slot=2,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_provenance,
    )
    ordinary_tx = build_generated_birth_identity_transaction(config, pre, ordinary_event)
    expected = np.frombuffer(
        derive_generated_birth_identity(
            namespace=config.namespace,
            paired_development_life_seed=402,
            learner_step=18,
            event_channel=ORDINARY_CANDIDATE_REFRESH_CHANNEL,
            slot=2,
            ordinal=0,
        ),
        dtype=np.uint8,
    )
    assert np.array_equal(ordinary_tx.post_state.candidate_identity[2], expected)
    assert not np.any(ordinary_tx.assignments.post_promotion_candidate_birth_identity)
    assert not np.any(ordinary_event.candidate_rebound_mask)


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("stale_step", "learner_step"),
        ("stale_seed", "paired development life seed"),
        ("stale_channel", "channel manifest"),
        ("stale_index", "promotion active mask"),
        ("overlap_mask", "active event masks must be disjoint"),
        ("stale_rebound", "candidate rebound mask"),
    ),
)
def test_stale_event_fields_and_incompatible_masks_fail_closed(
    mutation: str,
    match: str,
) -> None:
    config = _config()
    pre = _state()
    event = _promotion_event(config, pre)

    if mutation == "stale_step":
        bad = _rehash_event(event, learner_step=19)
    elif mutation == "stale_seed":
        bad = _rehash_event(event, paired_development_life_seed=999)
    elif mutation == "stale_channel":
        channels = list(event.channel_manifest)
        channels[-1] = "target-conditioned-rebound"
        bad = _rehash_event(event, channel_manifest=tuple(channels))
    elif mutation == "stale_index":
        bad = _rehash_event(event, promotion_active_slot=5)
    elif mutation == "overlap_mask":
        cascade = np.array(event.cascade_refill_mask, copy=True)
        cascade[4] = True
        bad = _rehash_event(event, cascade_refill_mask=cascade)
    else:
        rebound = np.array(event.candidate_rebound_mask, copy=True)
        rebound[0] = ~rebound[0]
        bad = _rehash_event(event, candidate_rebound_mask=rebound)

    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match=match):
        build_generated_birth_identity_transaction(config, pre, bad)


@pytest.mark.parametrize("mutation", ("missing_descendant", "extra_nondescendant"))
def test_cascade_mask_must_equal_exact_pre_graph_descendant_closure(
    mutation: str,
) -> None:
    config = _config()
    pre = _state()
    event = _promotion_event(config, pre)
    cascade = np.array(event.cascade_refill_mask, copy=True)
    if mutation == "missing_descendant":
        cascade[6] = False
    else:
        cascade[5] = True
    bad = _rehash_event(event, cascade_refill_mask=cascade)

    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="exact pre-graph descendant closure",
    ):
        build_generated_birth_identity_transaction(config, pre, bad)


def test_multi_hop_pre_graph_closure_is_exact_and_cascade_parents_are_rejected() -> None:
    config = GeneratedBirthIdentityLedgerConfig(
        namespace="generated-multihop-cascade-development",
        active_slots=6,
        candidate_slots=0,
        raw_feature_slots=2,
        max_depth=4,
        learn_generator_resources=True,
    )
    empty = np.zeros((0,), dtype=np.int32)
    pre = initialize_generated_birth_identity_ledger(
        config,
        paired_development_life_seed=91,
        learner_step=3,
        active_parent_a=np.asarray((0, 1, 0, 2, 3, 4), dtype=np.int32),
        active_parent_b=np.asarray((-1, -1, 1, 1, 1, 1), dtype=np.int32),
        active_ops=np.asarray(
            (OP_RAW, OP_RAW, OP_PRODUCT, OP_PRODUCT, OP_PRODUCT, OP_PRODUCT),
            dtype=np.int32,
        ),
        active_depth=np.asarray((0, 0, 1, 2, 3, 4), dtype=np.int32),
        active_generator_policy=np.zeros((6,), dtype=np.int32),
        candidate_parent_a=empty,
        candidate_parent_b=empty,
        candidate_ops=empty,
        candidate_depth=empty,
        candidate_generator_policy=empty,
    )
    final_parent_a = np.array(pre.active_parent_a, copy=True)
    final_parent_b = np.array(pre.active_parent_b, copy=True)
    final_ops = np.array(pre.active_ops, copy=True)
    final_depth = np.array(pre.active_depth, copy=True)
    final_policy = np.array(pre.active_generator_policy, copy=True)
    for slot in (2, 3, 4, 5):
        final_parent_a[slot], final_parent_b[slot] = 0, 1
        final_ops[slot], final_depth[slot], final_policy[slot] = OP_PRODUCT, 1, 1
    event = build_generated_birth_identity_event(
        config,
        pre,
        learner_step=4,
        generator_policy_sampled=True,
        generator_policy_id=1,
        active_parent_a=final_parent_a,
        active_parent_b=final_parent_b,
        active_ops=final_ops,
        active_depth=final_depth,
        active_generator_policy=final_policy,
        candidate_staged_parent_a=empty,
        candidate_staged_parent_b=empty,
        candidate_staged_ops=empty,
        candidate_staged_depth=empty,
        candidate_staged_generator_policy=empty,
        candidate_parent_a=empty,
        candidate_parent_b=empty,
        candidate_ops=empty,
        candidate_depth=empty,
        candidate_generator_policy=empty,
        direct_active_replacement_slot=2,
        cascade_refill_mask=np.asarray(
            (False, False, False, True, True, True),
            dtype=np.bool_,
        ),
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    assert transaction.audit.cascade_active_birth_count == 3

    bad_parent_a = np.array(event.active_parent_a, copy=True)
    bad_depth = np.array(event.active_depth, copy=True)
    bad_parent_a[4] = 3
    bad_depth[4] = 2
    bad = _rehash_event(event, active_parent_a=bad_parent_a, active_depth=bad_depth)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="cascade refill parent cannot be another cascade-refilled slot",
    ):
        build_generated_birth_identity_transaction(config, pre, bad)


def test_stale_parent_snapshot_and_fully_rehashed_transaction_tampering_fail() -> None:
    config = _config()
    pre = _state()
    event = _event(config, pre, ordinary_candidate_refresh_slot=1)
    transaction = build_generated_birth_identity_transaction(config, pre, event)

    snapshots = np.array(
        transaction.post_state.candidate_parent_identity_snapshot,
        copy=True,
    )
    snapshots[0, 0] = transaction.post_state.active_identity[0]
    stale_snapshot_state = dataclasses.replace(
        transaction.post_state,
        candidate_parent_identity_snapshot=_readonly(snapshots),
        integrity_sha256="0" * 64,
    )
    stale_snapshot_state = dataclasses.replace(
        stale_snapshot_state,
        integrity_sha256=generated_birth_identity_ledger_state_sha256(stale_snapshot_state),
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="candidate parent identity snapshot is stale",
    ):
        validate_generated_birth_identity_transaction(
            dataclasses.replace(transaction, post_state=stale_snapshot_state),
            config=config,
            pre_state=pre,
            event=event,
        )

    candidate_identity = np.array(transaction.post_state.candidate_identity, copy=True)
    candidate_identity[0, 0] ^= np.uint8(1)
    tampered_state = dataclasses.replace(
        transaction.post_state,
        candidate_identity=_readonly(candidate_identity),
        integrity_sha256="0" * 64,
    )
    tampered_state = dataclasses.replace(
        tampered_state,
        integrity_sha256=generated_birth_identity_ledger_state_sha256(tampered_state),
    )
    tampered_audit = dataclasses.replace(
        transaction.audit,
        post_state_sha256=tampered_state.integrity_sha256,
        transaction_sha256="0" * 64,
    )
    tampered = dataclasses.replace(
        transaction,
        post_state=tampered_state,
        audit=tampered_audit,
    )
    tampered = dataclasses.replace(
        tampered,
        audit=dataclasses.replace(
            tampered.audit,
            transaction_sha256=generated_birth_identity_transaction_sha256(tampered),
        ),
    )
    assert tampered.audit.transaction_sha256 == generated_birth_identity_transaction_sha256(
        tampered
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="independent canonical rebuild",
    ):
        validate_generated_birth_identity_transaction(
            tampered,
            config=config,
            pre_state=pre,
            event=event,
        )


def test_collision_scope_includes_overwritten_intermediate_births(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    pre = _state()
    event = _promotion_event(config, pre)
    original = ledger_module._identity_array
    forged = np.arange(1, 33, dtype=np.uint8)
    pre_scope = {
        row.tobytes()
        for row in np.concatenate((pre.active_identity, pre.candidate_identity), axis=0)
    }
    assert forged.tobytes() not in pre_scope

    def collide_intermediate(**kwargs: object) -> np.ndarray:
        if kwargs["channel"] in {
            POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL,
            CANDIDATE_PARENT_REBOUND_CHANNEL,
        }:
            return np.array(forged, copy=True)
        return original(**cast(Any, kwargs))

    monkeypatch.setattr(ledger_module, "_identity_array", collide_intermediate)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="pre-live/raw/new-transition scope",
    ):
        build_generated_birth_identity_transaction(config, pre, event)


def test_rehashed_assignment_collision_with_pre_live_identity_fails_closed() -> None:
    config = _config()
    pre = _state()
    candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    candidate_ops = np.array(pre.candidate_ops, copy=True)
    candidate_depth = np.array(pre.candidate_depth, copy=True)
    candidate_policy = np.array(pre.candidate_generator_policy, copy=True)
    candidate_parent_a[2], candidate_parent_b[2] = 0, 1
    candidate_ops[2], candidate_depth[2], candidate_policy[2] = OP_PRODUCT, 1, 1
    event = _event(
        config,
        pre,
        ordinary_candidate_refresh_slot=2,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
        candidate_generator_policy=candidate_policy,
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    ordinary = np.array(
        transaction.assignments.ordinary_candidate_birth_identity,
        copy=True,
    )
    ordinary[2] = pre.active_identity[0]
    assignments = _rehash_assignments(
        transaction.assignments,
        ordinary_candidate_birth_identity=_readonly(ordinary),
    )
    tampered = dataclasses.replace(
        transaction,
        assignments=assignments,
        audit=dataclasses.replace(
            transaction.audit,
            assignments_sha256=assignments.integrity_sha256,
            transaction_sha256="0" * 64,
        ),
    )
    tampered = dataclasses.replace(
        tampered,
        audit=dataclasses.replace(
            tampered.audit,
            transaction_sha256=generated_birth_identity_transaction_sha256(tampered),
        ),
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="pre-live/raw/new-transition scope",
    ):
        validate_generated_birth_identity_transaction(
            tampered,
            config=config,
            pre_state=pre,
            event=event,
        )


def test_identity_array_noop_advances_step_and_is_still_strictly_bound() -> None:
    config = _config()
    pre = _state()
    event = _event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)

    assert transaction.audit.identity_array_noop_with_monotone_step_advance
    assert transaction.audit.applied_identity_event_count == 0
    assert np.array_equal(transaction.post_state.active_identity, pre.active_identity)
    assert np.array_equal(
        transaction.post_state.active_parent_identity_snapshot,
        pre.active_parent_identity_snapshot,
    )
    assert np.array_equal(transaction.post_state.candidate_identity, pre.candidate_identity)
    assert np.array_equal(
        transaction.post_state.candidate_parent_identity_snapshot,
        pre.candidate_parent_identity_snapshot,
    )
    assert transaction.post_state.learner_step == pre.learner_step + 1
    assert transaction.audit.channel_manifest == GENERATED_BIRTH_IDENTITY_CHANNELS
    validation = validate_generated_birth_identity_transaction(
        transaction,
        config=config,
        pre_state=pre,
        event=event,
    )
    assert validation.valid
    assert not validation.caller_supplied_events_authenticated
    assert not validation.execution_authorized
    assert not validation.runner_authorized
    assert not validation.artifact_writes_authorized
    assert not validation.evidence_authorized
    assert not validation.scientific_promotion_allowed


def test_strict_validators_reject_writable_state_event_and_assignment_arrays() -> None:
    config = _config()
    pre = _state()
    event = _promotion_event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)

    writable_pre = dataclasses.replace(
        pre,
        active_identity=np.array(pre.active_identity, copy=True),
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="active_identity must be an immutable read-only array",
    ):
        build_generated_birth_identity_transaction(config, writable_pre, event)

    writable_event = dataclasses.replace(
        event,
        cascade_refill_mask=np.array(event.cascade_refill_mask, copy=True),
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="cascade_refill_mask must be an immutable read-only array",
    ):
        build_generated_birth_identity_transaction(config, pre, writable_event)

    writable_assignments = dataclasses.replace(
        transaction.assignments,
        candidate_rebound_identity=np.array(
            transaction.assignments.candidate_rebound_identity,
            copy=True,
        ),
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="candidate_rebound_identity must be an immutable read-only array",
    ):
        validate_generated_birth_identity_transaction(
            dataclasses.replace(transaction, assignments=writable_assignments),
            config=config,
            pre_state=pre,
            event=event,
        )

    owned_readonly = np.array(event.cascade_refill_mask, copy=True)
    owned_readonly.flags.writeable = False
    forged_backing_event = dataclasses.replace(
        event,
        cascade_refill_mask=owned_readonly,
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="cascade_refill_mask must have immutable bytes backing",
    ):
        build_generated_birth_identity_transaction(config, pre, forged_backing_event)


def test_wrong_parent_shape_and_raw_slot_writes_are_rejected() -> None:
    config = _config()
    pre = _state()
    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match="candidate_parent_a"):
        _event(
            config,
            pre,
            candidate_parent_a=pre.candidate_parent_a[:-1],
        )
    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match="raw-prefix"):
        _event(
            config,
            pre,
            promotion_active_slot=1,
            promotion_candidate_slot=1,
        )


@pytest.mark.parametrize("bank", ("active", "candidate"))
@pytest.mark.parametrize("invalid_policy", (-1, 4))
def test_initial_provenance_is_bounded_by_public_policy_manifest(
    bank: str,
    invalid_policy: int,
) -> None:
    config = _config()
    field = f"{bank}_generator_policy"
    values = _descriptor_arrays(config)[field].copy()
    values[0 if bank == "candidate" else 3] = invalid_policy
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="outside the bound public policy manifest",
    ):
        _state(config=config, changes={field: values})


@pytest.mark.parametrize("bank", ("active", "candidate"))
def test_genesis_provenance_must_be_an_unsampled_fixed_placeholder(bank: str) -> None:
    config = _config()
    field = f"{bank}_generator_policy"
    values = _descriptor_arrays(config)[field].copy()
    values[0 if bank == "candidate" else 3] = 1
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match=f"genesis {bank} generator policies must be fixed placeholders",
    ):
        _state(config=config, changes={field: values})


def test_disabled_generator_resources_keep_refresh_provenance_unsampled() -> None:
    config = dataclasses.replace(
        _config(),
        namespace="generated-unsampled-policy-development",
        learn_generator_resources=False,
    )
    pre = _state(config=config)
    candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    candidate_ops = np.array(pre.candidate_ops, copy=True)
    candidate_depth = np.array(pre.candidate_depth, copy=True)
    candidate_parent_a[1], candidate_parent_b[1] = 0, 1
    candidate_ops[1], candidate_depth[1] = OP_SUM, 1
    event = _event(
        config,
        pre,
        generator_policy_sampled=False,
        generator_policy_id=0,
        ordinary_candidate_refresh_slot=1,
        candidate_parent_a=candidate_parent_a,
        candidate_parent_b=candidate_parent_b,
        candidate_ops=candidate_ops,
        candidate_depth=candidate_depth,
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    assert transaction.post_state.candidate_generator_policy[1] == 0
    assert not bool(transaction.post_state.candidate_generator_policy_sampled[1])


@pytest.mark.parametrize(
    "learn_resources,sampled,policy_id,match",
    (
        (True, False, 0, "sampled-policy flag"),
        (False, True, 1, "sampled-policy flag"),
        (False, False, 1, "fixed policy placeholder"),
    ),
)
def test_event_sampled_flag_and_placeholder_match_bound_learner_mode(
    learn_resources: bool,
    sampled: bool,
    policy_id: int,
    match: str,
) -> None:
    config = dataclasses.replace(
        _config(),
        learn_generator_resources=learn_resources,
    )
    pre = _state(config=config)
    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match=match):
        _event(
            config,
            pre,
            generator_policy_sampled=sampled,
            generator_policy_id=policy_id,
        )


@pytest.mark.parametrize("bank", ("active", "candidate"))
def test_event_provenance_is_bounded_by_public_policy_manifest(bank: str) -> None:
    config = _config()
    pre = _state()
    if bank == "active":
        values = np.array(pre.active_generator_policy, copy=True)
        values[3] = config.generator_policy_count
        kwargs = {"active_generator_policy": values}
    else:
        values = np.array(pre.candidate_generator_policy, copy=True)
        values[0] = config.generator_policy_count
        kwargs = {"candidate_generator_policy": values}
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="outside the bound public policy manifest",
    ):
        _event(config, pre, **cast(Any, kwargs))


def test_unchanged_active_parent_slot_aliasing_requires_a_cascade_birth() -> None:
    config = _config()
    pre = _state()
    aliased_parent_a = np.array(pre.active_parent_a, copy=True)
    aliased_parent_a[6] = 5
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="unchanged active structural descriptor changed at parent_a",
    ):
        _event(
            config,
            pre,
            active_parent_a=aliased_parent_a,
        )


def test_raw_active_and_candidate_source_swaps_are_not_rebounds() -> None:
    config = _config()
    active_ops = _descriptor_arrays(config)["active_ops"].copy()
    active_parent_a = _descriptor_arrays(config)["active_parent_a"].copy()
    active_parent_b = _descriptor_arrays(config)["active_parent_b"].copy()
    active_depth = _descriptor_arrays(config)["active_depth"].copy()
    candidate_depth = _descriptor_arrays(config)["candidate_depth"].copy()
    active_ops[5], active_parent_a[5], active_parent_b[5], active_depth[5] = (
        OP_RAW,
        0,
        -1,
        0,
    )
    candidate_depth[0] = 1
    raw_active_state = _state(
        changes={
            "active_ops": active_ops,
            "active_parent_a": active_parent_a,
            "active_parent_b": active_parent_b,
            "active_depth": active_depth,
            "candidate_depth": candidate_depth,
        }
    )
    swapped_active_source = np.array(raw_active_state.active_parent_a, copy=True)
    swapped_active_source[5] = 1
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="unchanged active structural descriptor changed at parent_a",
    ):
        _event(config, raw_active_state, active_parent_a=swapped_active_source)

    candidate_ops = _descriptor_arrays(config)["candidate_ops"].copy()
    candidate_parent_a = _descriptor_arrays(config)["candidate_parent_a"].copy()
    candidate_parent_b = _descriptor_arrays(config)["candidate_parent_b"].copy()
    candidate_depth = _descriptor_arrays(config)["candidate_depth"].copy()
    candidate_ops[0], candidate_parent_a[0], candidate_parent_b[0] = OP_RAW, 0, -1
    candidate_depth[0] = 0
    raw_candidate_state = _state(
        changes={
            "candidate_ops": candidate_ops,
            "candidate_parent_a": candidate_parent_a,
            "candidate_parent_b": candidate_parent_b,
            "candidate_depth": candidate_depth,
        }
    )
    swapped_candidate_source = np.array(raw_candidate_state.candidate_parent_a, copy=True)
    swapped_candidate_source[0] = 1
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="unrefreshed candidate local structural descriptor changed at parent_a",
    ):
        _event(
            config,
            raw_candidate_state,
            candidate_parent_a=swapped_candidate_source,
        )


@pytest.mark.parametrize("field", ("op", "depth", "provenance"))
def test_same_parent_op_and_stale_depth_or_provenance_fail_closed(field: str) -> None:
    config = _config()
    pre = _state()
    if field == "op":
        values = np.array(pre.active_ops, copy=True)
        values[3] = OP_PRODUCT
        match = "unchanged active structural descriptor changed at op"
    elif field == "depth":
        values = np.array(pre.active_depth, copy=True)
        values[3] += 1
        match = "depth is not exactly derived"
    else:
        values = np.array(pre.active_generator_policy, copy=True)
        values[3] = (values[3] + 1) % config.generator_policy_count
        match = "unchanged active structural descriptor changed at generator provenance"
    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match=match):
        if field == "op":
            _event(config, pre, active_ops=values)
        elif field == "depth":
            _event(config, pre, active_depth=values)
        else:
            _event(config, pre, active_generator_policy=values)


@pytest.mark.parametrize("field", ("op", "depth", "provenance"))
def test_unrefreshed_candidate_structural_mutation_fails_closed(field: str) -> None:
    config = _config()
    pre = _state()
    if field == "op":
        values = np.array(pre.candidate_ops, copy=True)
        values[0] = OP_SUM
        match = "unrefreshed candidate local structural descriptor changed at op"
    elif field == "depth":
        values = np.array(pre.candidate_depth, copy=True)
        values[0] += 1
        match = "depth is not exactly derived"
    else:
        values = np.array(pre.candidate_generator_policy, copy=True)
        values[0] = (values[0] + 1) % config.generator_policy_count
        match = "unrefreshed candidate local structural descriptor changed at generator provenance"
    with pytest.raises(GeneratedBirthIdentityLedgerConstructionError, match=match):
        if field == "op":
            _event(config, pre, candidate_ops=values)
        elif field == "depth":
            _event(config, pre, candidate_depth=values)
        else:
            _event(config, pre, candidate_generator_policy=values)


def test_raw_promotion_transfers_full_descriptor_and_provenance() -> None:
    config = _config()
    candidate_ops = _descriptor_arrays(config)["candidate_ops"].copy()
    candidate_parent_a = _descriptor_arrays(config)["candidate_parent_a"].copy()
    candidate_parent_b = _descriptor_arrays(config)["candidate_parent_b"].copy()
    candidate_depth = _descriptor_arrays(config)["candidate_depth"].copy()
    candidate_provenance = _descriptor_arrays(config)["candidate_generator_policy"].copy()
    candidate_ops[0], candidate_parent_a[0], candidate_parent_b[0] = OP_RAW, 1, -1
    candidate_depth[0], candidate_provenance[0] = 0, 0
    pre = _state(
        changes={
            "candidate_ops": candidate_ops,
            "candidate_parent_a": candidate_parent_a,
            "candidate_parent_b": candidate_parent_b,
            "candidate_depth": candidate_depth,
            "candidate_generator_policy": candidate_provenance,
        }
    )
    active_parent_a = np.array(pre.active_parent_a, copy=True)
    active_parent_b = np.array(pre.active_parent_b, copy=True)
    active_ops = np.array(pre.active_ops, copy=True)
    active_depth = np.array(pre.active_depth, copy=True)
    active_provenance = np.array(pre.active_generator_policy, copy=True)
    active_parent_a[5], active_parent_b[5], active_ops[5], active_depth[5] = (
        1,
        -1,
        OP_RAW,
        0,
    )
    active_provenance[5] = 0
    final_candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    final_candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    final_candidate_ops = np.array(pre.candidate_ops, copy=True)
    final_candidate_depth = np.array(pre.candidate_depth, copy=True)
    final_candidate_provenance = np.array(pre.candidate_generator_policy, copy=True)
    final_candidate_parent_a[0], final_candidate_parent_b[0] = 0, 1
    final_candidate_ops[0], final_candidate_depth[0] = OP_PRODUCT, 1
    final_candidate_provenance[0] = 1
    event = _event(
        config,
        pre,
        promotion_active_slot=5,
        promotion_candidate_slot=0,
        active_parent_a=active_parent_a,
        active_parent_b=active_parent_b,
        active_ops=active_ops,
        active_depth=active_depth,
        active_generator_policy=active_provenance,
        candidate_parent_a=final_candidate_parent_a,
        candidate_parent_b=final_candidate_parent_b,
        candidate_ops=final_candidate_ops,
        candidate_depth=final_candidate_depth,
        candidate_generator_policy=final_candidate_provenance,
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    post = transaction.post_state
    assert post.active_ops[5] == OP_RAW
    assert post.active_parent_a[5] == 1
    assert post.active_depth[5] == 0
    assert post.active_generator_policy[5] == 0
    assert np.array_equal(
        post.active_parent_identity_snapshot[5],
        pre.candidate_parent_identity_snapshot[0],
    )


def test_fresh_promotion_root_reference_does_not_trigger_rebound() -> None:
    config = _config()
    base = _descriptor_arrays(config)
    active_parent_a = base["active_parent_a"].copy()
    active_parent_b = base["active_parent_b"].copy()
    candidate_parent_a = base["candidate_parent_a"].copy()
    candidate_parent_b = base["candidate_parent_b"].copy()
    candidate_depth = base["candidate_depth"].copy()
    active_parent_a[6], active_parent_b[6] = 3, 2
    candidate_parent_a[0], candidate_parent_b[0], candidate_depth[0] = 2, 1, 2
    candidate_parent_a[2], candidate_parent_b[2], candidate_depth[2] = 3, 1, 2
    pre = _state(
        changes={
            "active_parent_a": active_parent_a,
            "active_parent_b": active_parent_b,
            "active_depth": np.asarray((0, 0, 1, 1, 2, 2, 2), dtype=np.int32),
            "candidate_parent_a": candidate_parent_a,
            "candidate_parent_b": candidate_parent_b,
            "candidate_depth": candidate_depth,
        }
    )
    final_active_parent_a = np.array(pre.active_parent_a, copy=True)
    final_active_parent_b = np.array(pre.active_parent_b, copy=True)
    final_active_ops = np.array(pre.active_ops, copy=True)
    final_active_depth = np.array(pre.active_depth, copy=True)
    final_active_provenance = np.array(pre.active_generator_policy, copy=True)
    final_active_parent_a[4], final_active_parent_b[4] = 3, 2
    final_active_ops[4], final_active_depth[4], final_active_provenance[4] = OP_SUM, 2, 0
    final_candidate_parent_a = np.array(pre.candidate_parent_a, copy=True)
    final_candidate_parent_b = np.array(pre.candidate_parent_b, copy=True)
    final_candidate_ops = np.array(pre.candidate_ops, copy=True)
    final_candidate_depth = np.array(pre.candidate_depth, copy=True)
    final_candidate_provenance = np.array(pre.candidate_generator_policy, copy=True)
    final_candidate_parent_a[1], final_candidate_parent_b[1] = 4, 1
    final_candidate_ops[1], final_candidate_depth[1] = OP_PRODUCT, 3
    final_candidate_provenance[1] = 1
    event = _event(
        config,
        pre,
        promotion_active_slot=4,
        promotion_candidate_slot=1,
        active_parent_a=final_active_parent_a,
        active_parent_b=final_active_parent_b,
        active_ops=final_active_ops,
        active_depth=final_active_depth,
        active_generator_policy=final_active_provenance,
        candidate_parent_a=final_candidate_parent_a,
        candidate_parent_b=final_candidate_parent_b,
        candidate_ops=final_candidate_ops,
        candidate_depth=final_candidate_depth,
        candidate_generator_policy=final_candidate_provenance,
    )
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    assert not np.any(event.candidate_rebound_mask)
    assert np.array_equal(
        transaction.post_state.candidate_identity[1],
        transaction.assignments.post_promotion_candidate_birth_identity[1],
    )


def test_direct_active_replacement_with_candidates_is_impossible() -> None:
    config = _config()
    pre = _state()
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="direct active replacement requires candidate_slots == 0",
    ):
        _event(config, pre, direct_active_replacement_slot=3)


def test_frozen_policy_manifest_matches_live_core_and_detects_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        "random_product_safe",
        "mutation_product_nominal",
        "residual_tanh",
        "residual_gated_aggressive",
    )
    assert GENERATED_BIRTH_IDENTITY_GENERATOR_POLICY_MANIFEST == expected
    assert getattr(ledger_module, "_CORE_GENERATOR_META_POLICY_NAMES") == expected
    monkeypatch.setattr(
        ledger_module,
        "_CORE_GENERATOR_META_POLICY_NAMES",
        ("drifted",),
    )
    with pytest.raises(ValueError, match="live public core generator-policy manifest drifted"):
        _config()


def test_every_config_boolean_rejects_int_string_and_numpy_bool() -> None:
    boolean_fields = tuple(
        field.name
        for field in dataclasses.fields(GeneratedBirthIdentityLedgerConfig)
        if field.type == "bool"
    )
    assert boolean_fields
    for name in boolean_fields:
        for forged in (0, 1, "true", np.bool_(True)):
            with pytest.raises(TypeError, match=f"{name} must be an exact Python boolean"):
                dataclasses.replace(_config(), **cast(Any, {name: forged}))


def test_every_external_event_audit_and_validation_boolean_is_exact() -> None:
    config = _config()
    pre = _state()
    event = _event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    validation = validate_generated_birth_identity_transaction(
        transaction,
        config=config,
        pre_state=pre,
        event=event,
    )

    for forged in (0, 1, "true", np.bool_(True)):
        bad_event = dataclasses.replace(
            event,
            **cast(Any, {"generator_policy_sampled": forged}),
        )
        with pytest.raises(
            GeneratedBirthIdentityLedgerConstructionError,
            match="event generator_policy_sampled must be an exact Python boolean",
        ):
            build_generated_birth_identity_transaction(config, pre, bad_event)

    audit_boolean_fields = tuple(
        field.name
        for field in dataclasses.fields(ledger_module.GeneratedBirthIdentityLedgerAudit)
        if field.type == "bool"
    )
    assert audit_boolean_fields
    for name in audit_boolean_fields:
        for forged in (0, 1, "true", np.bool_(True)):
            bad_audit = dataclasses.replace(
                transaction.audit,
                **cast(Any, {name: forged}),
            )
            with pytest.raises(
                GeneratedBirthIdentityLedgerConstructionError,
                match=f"audit {name} must be an exact Python boolean",
            ):
                validate_generated_birth_identity_transaction(
                    dataclasses.replace(transaction, audit=bad_audit),
                    config=config,
                    pre_state=pre,
                    event=event,
                )

    validation_boolean_fields = tuple(
        field.name
        for field in dataclasses.fields(ledger_module.GeneratedBirthIdentityValidation)
        if field.type == "bool"
    )
    assert validation_boolean_fields
    for name in validation_boolean_fields:
        for forged in (0, 1, "true", np.bool_(True)):
            with pytest.raises(
                GeneratedBirthIdentityLedgerConstructionError,
                match=f"validation {name} must be an exact Python boolean",
            ):
                dataclasses.replace(validation, **cast(Any, {name: forged}))


def test_exact_integer_type_rehash_forgeries_fail_before_hash_equality() -> None:
    config = _config()
    pre = _state()
    event = _event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)

    rehashed_state = _rehash_state(pre, learner_step=True)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="learner_step must be an exact Python integer",
    ):
        build_generated_birth_identity_transaction(config, rehashed_state, event)

    rehashed_event = _rehash_event(event, generator_policy_id=True)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="event generator_policy_id must be an exact Python integer",
    ):
        build_generated_birth_identity_transaction(config, pre, rehashed_event)

    rehashed_assignments = _rehash_assignments(
        transaction.assignments,
        persistent_array_nbytes=True,
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="assignment persistent_array_nbytes must be an exact Python integer",
    ):
        validate_generated_birth_identity_transaction(
            dataclasses.replace(transaction, assignments=rehashed_assignments),
            config=config,
            pre_state=pre,
            event=event,
        )

    rehashed_audit = _rehash_audit_transaction(
        transaction,
        applied_identity_event_count=True,
    )
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="audit applied_identity_event_count must be an exact Python integer",
    ):
        validate_generated_birth_identity_transaction(
            rehashed_audit,
            config=config,
            pre_state=pre,
            event=event,
        )


def test_every_external_integer_scalar_rejects_bool_float_and_numpy_integer() -> None:
    config = _config()
    for name in (
        "active_slots",
        "candidate_slots",
        "raw_feature_slots",
        "max_depth",
        "generator_policy_count",
    ):
        original = getattr(config, name)
        for forged in (True, float(original), np.int32(original), str(original)):
            with pytest.raises(TypeError, match=f"{name} must be an exact Python integer"):
                dataclasses.replace(config, **cast(Any, {name: forged}))

    pre = _state()
    event = _event(config, pre)
    transaction = build_generated_birth_identity_transaction(config, pre, event)
    for name in (
        "paired_development_life_seed",
        "learner_step",
        "persistent_array_nbytes",
    ):
        original = getattr(pre, name)
        for forged in (True, float(original), np.int32(original), str(original)):
            bad_state = dataclasses.replace(pre, **cast(Any, {name: forged}))
            with pytest.raises(
                GeneratedBirthIdentityLedgerConstructionError,
                match="exact Python integer",
            ):
                build_generated_birth_identity_transaction(config, bad_state, event)

    event_integer_fields = (
        "paired_development_life_seed",
        "learner_step",
        "generator_policy_id",
        "promotion_active_slot",
        "promotion_candidate_slot",
        "direct_active_replacement_slot",
        "ordinary_candidate_refresh_slot",
        "post_promotion_candidate_refresh_slot",
    )
    for name in event_integer_fields:
        original = getattr(event, name)
        for forged in (True, float(original), np.int32(original), str(original)):
            bad_event = dataclasses.replace(event, **cast(Any, {name: forged}))
            with pytest.raises(
                GeneratedBirthIdentityLedgerConstructionError,
                match="exact Python integer",
            ):
                build_generated_birth_identity_transaction(config, pre, bad_event)

    for forged in (True, 1.0, np.int32(1), "1"):
        assignments = dataclasses.replace(
            transaction.assignments,
            **cast(Any, {"persistent_array_nbytes": forged}),
        )
        with pytest.raises(
            GeneratedBirthIdentityLedgerConstructionError,
            match="assignment persistent_array_nbytes must be an exact Python integer",
        ):
            validate_generated_birth_identity_transaction(
                dataclasses.replace(transaction, assignments=assignments),
                config=config,
                pre_state=pre,
                event=event,
            )

    audit_integer_fields = tuple(
        field.name
        for field in dataclasses.fields(ledger_module.GeneratedBirthIdentityLedgerAudit)
        if field.type == "int"
    )
    assert audit_integer_fields
    for name in audit_integer_fields:
        original = getattr(transaction.audit, name)
        for forged in (True, float(original), np.int32(original), str(original)):
            audit = dataclasses.replace(
                transaction.audit,
                **cast(Any, {name: forged}),
            )
            with pytest.raises(
                GeneratedBirthIdentityLedgerConstructionError,
                match=f"audit {name} must be an exact Python integer",
            ):
                validate_generated_birth_identity_transaction(
                    dataclasses.replace(transaction, audit=audit),
                    config=config,
                    pre_state=pre,
                    event=event,
                )


def test_int32_step_exhaustion_and_bank_bounds_fail_explicitly() -> None:
    maximum = ledger_module._INT32_MAX
    config = _config()
    terminal = _state(learner_step=maximum)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="terminal learner_step is exhausted",
    ):
        _event(config, terminal, learner_step=maximum)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="event learner_step is outside",
    ):
        _event(config, _state(), learner_step=maximum + 1)
    with pytest.raises(
        GeneratedBirthIdentityLedgerConstructionError,
        match="promotion_active_slot is outside the fixed bank",
    ):
        _event(config, _state(), promotion_active_slot=config.active_slots)
    with pytest.raises(ValueError, match="combined bank size"):
        GeneratedBirthIdentityLedgerConfig(
            namespace="generated-oversized-banks-development",
            active_slots=maximum,
            candidate_slots=1,
            raw_feature_slots=1,
            max_depth=1,
            learn_generator_resources=False,
        )
    with pytest.raises(ValueError, match="active_slots"):
        dataclasses.replace(config, active_slots=maximum + 1)

    terminal_identity = derive_generated_birth_identity(
        namespace=config.namespace,
        paired_development_life_seed=0,
        learner_step=maximum,
        event_channel=DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
        slot=maximum,
        ordinal=maximum,
    )
    assert len(terminal_identity) == 32
    for name, value in (
        ("learner_step", maximum + 1),
        ("slot", maximum + 1),
        ("ordinal", maximum + 1),
    ):
        kwargs = {
            "namespace": config.namespace,
            "paired_development_life_seed": 0,
            "learner_step": 0,
            "event_channel": DIRECT_ACTIVE_REPLACEMENT_CHANNEL,
            "slot": 0,
            "ordinal": 0,
        }
        kwargs[name] = value
        with pytest.raises(
            GeneratedBirthIdentityLedgerConstructionError,
            match=f"{name} is outside",
        ):
            derive_generated_birth_identity(**cast(Any, kwargs))


def test_non_authority_configuration_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot grant authority"):
        dataclasses.replace(_config(), evidence_authorized=True)
    with pytest.raises(ValueError, match="canonical channel manifest"):
        dataclasses.replace(
            _config(),
            channel_manifest=(POST_PROMOTION_CANDIDATE_REFRESH_CHANNEL,),
        )
    with pytest.raises(ValueError, match="canonical schema"):
        dataclasses.replace(_config(), schema="forged")
    with pytest.raises(ValueError, match="cannot claim collision impossibility"):
        dataclasses.replace(_config(), cryptographic_collision_impossibility_claimed=True)
    with pytest.raises(ValueError, match="cannot grant authority"):
        dataclasses.replace(_config(), exact_functional_expression_identity_claimed=True)
    with pytest.raises(ValueError, match="cannot grant authority"):
        dataclasses.replace(_config(), public_core_event_trace_consumed=True)
    with pytest.raises(ValueError, match="public core policy count"):
        dataclasses.replace(_config(), generator_policy_count=5)
    with pytest.raises(ValueError, match="public core manifest"):
        dataclasses.replace(
            _config(),
            generator_policy_manifest=("forged", "forged", "forged", "forged"),
        )
    assert CANDIDATE_PARENT_REBOUND_CHANNEL in GENERATED_BIRTH_IDENTITY_CHANNELS
