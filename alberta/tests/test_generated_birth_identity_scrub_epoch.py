"""Focused tests for the external structural-scrub identity rollover."""

from __future__ import annotations

import dataclasses
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    OP_TANH,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation import generated_birth_identity_ledger as _ledger
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
)
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA,
    GeneratedBirthIdentityScrubEpochError,
    build_generated_birth_identity_scrub_epoch_transaction,
    generated_birth_identity_scrub_epoch_transaction_sha256,
    validate_generated_birth_identity_scrub_epoch_transaction,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    attach_generated_birth_identity_ledger_at_core_genesis,
    authenticate_generated_birth_identity_trace_by_source_replay,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    GeneratedClassScrubConfig,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    GeneratedExpression,
    product_expression,
    raw_expression,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    ExpandedExpressionLineagePlan,
    compile_expanded_expression_lineage_masks,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GeneratedReacquisitionEpochConfig,
)

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _BoundScrub:
    learner: CompositionalFeatureLearner
    config: GeneratedBirthIdentityLedgerV4Config
    pre: CompositionalFeatureState
    post: CompositionalFeatureState
    target: GeneratedExpression
    lineage_plan: ExpandedExpressionLineagePlan
    lineage_config: ExpandedExpressionLineageConfig
    scrub_config: GeneratedClassScrubConfig
    ledger: object


@pytest.fixture(scope="module")
def bound_scrub() -> _BoundScrub:
    learner = CompositionalFeatureLearner(
        n_features=8,
        n_tasks=1,
        candidate_count=4,
        replacement_interval=32,
        max_depth=3,
        use_obgd=False,
    )
    initialized = learner.init(3, jr.key(402))
    theta = (
        jnp.zeros((8, 2), dtype=jnp.float32)
        .at[6]
        .set(jnp.asarray((-1.0, 2.0), dtype=jnp.float32))
    )
    pre = cast(
        CompositionalFeatureState,
        initialized.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                (
                    OP_RAW,
                    OP_RAW,
                    OP_RAW,
                    OP_PRODUCT,
                    OP_SUM,
                    OP_GATED,
                    OP_TANH,
                    OP_PRODUCT,
                ),
                dtype=jnp.int32,
            ),
            parent_a=jnp.asarray((0, 1, 2, 0, 3, 1, 1, 4), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, -1, 1, 2, 2, 0, 5), dtype=jnp.int32),
            theta=theta,
            depth=jnp.asarray((0, 0, 0, 1, 2, 1, 1, 3), dtype=jnp.int32),
            candidate_ops=jnp.asarray(
                (OP_GATED, OP_PRODUCT, OP_RAW, OP_SUM),
                dtype=jnp.int32,
            ),
            candidate_parent_a=jnp.asarray((4, 6, 0, 4), dtype=jnp.int32),
            candidate_parent_b=jnp.asarray((0, 2, -1, 5), dtype=jnp.int32),
            candidate_theta=jnp.zeros((4, 2), dtype=jnp.float32),
            candidate_depth=jnp.asarray((3, 2, 0, 3), dtype=jnp.int32),
            feature_generator_policy=jnp.zeros((8,), dtype=jnp.int32),
            candidate_generator_policy=jnp.zeros((4,), dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.asarray((0, 0), dtype=jnp.uint32),
            replacement_phase=jnp.asarray(0, dtype=jnp.int32),
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    target = product_expression(raw_expression(0), raw_expression(1))
    lineage_config = ExpandedExpressionLineageConfig(
        feature_dim=3,
        active_slots=8,
        candidate_slots=4,
        n_tasks=1,
        generator_contexts=1,
        generator_policy_count=4,
    )
    scrub_config = GeneratedClassScrubConfig(
        feature_dim=3,
        active_slots=8,
        candidate_slots=4,
        n_tasks=1,
    )
    lineage_plan = compile_expanded_expression_lineage_masks(
        pre,
        target,
        config=lineage_config,
    )
    result = scrub_compositional_feature_state(
        pre,
        lineage_plan.active_mask,
        lineage_plan.candidate_mask,
        jnp.asarray(True, dtype=jnp.bool_),
        config=scrub_config,
    )
    assert bool(result.diagnostics.committed)
    config = GeneratedBirthIdentityLedgerV4Config(
        namespace="generated-scrub-rollover-focused-development",
        active_slots=8,
        candidate_slots=4,
        raw_feature_slots=3,
        max_depth=3,
        learn_generator_resources=False,
    )
    ledger = attach_generated_birth_identity_ledger_at_core_genesis(
        config,
        learner_pre_state=pre,
        paired_development_life_seed=101,
    )
    return _BoundScrub(
        learner=learner,
        config=config,
        pre=pre,
        post=result.state,
        target=target,
        lineage_plan=lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        ledger=ledger,
    )


def _build(bound: _BoundScrub):  # type: ignore[no-untyped-def]
    return build_generated_birth_identity_scrub_epoch_transaction(
        bound.config,
        bound.ledger,  # type: ignore[arg-type]
        bound.pre,
        bound.post,
        bound.target,
        bound.lineage_plan,
        lineage_config=bound.lineage_config,
        scrub_config=bound.scrub_config,
        epoch_config=GeneratedReacquisitionEpochConfig(),
    )


def _validate(transaction, bound: _BoundScrub):  # type: ignore[no-untyped-def]
    return validate_generated_birth_identity_scrub_epoch_transaction(
        transaction,
        config=bound.config,
        pre_ledger_state=bound.ledger,  # type: ignore[arg-type]
        pre_core_state=bound.pre,
        post_core_state=bound.post,
        target=bound.target,
        lineage_plan=bound.lineage_plan,
        lineage_config=bound.lineage_config,
        scrub_config=bound.scrub_config,
        epoch_config=GeneratedReacquisitionEpochConfig(),
    )


def test_rollover_retires_every_masked_identity_and_preserves_every_other_identity(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    validation = _validate(transaction, bound_scrub)
    active_mask = np.asarray(bound_scrub.lineage_plan.active_mask)
    candidate_mask = np.asarray(bound_scrub.lineage_plan.candidate_mask)
    pre_ledger = bound_scrub.ledger

    assert transaction.audit.schema == GENERATED_BIRTH_IDENTITY_SCRUB_EPOCH_SCHEMA
    assert validation.valid and validation.output_is_normal_v4_state
    assert transaction.audit.new_identity_count == int(
        np.count_nonzero(active_mask) + np.count_nonzero(candidate_mask)
    )
    assert np.all(
        np.any(
            transaction.post_ledger_state.active_identity[active_mask]
            != pre_ledger.active_identity[active_mask],  # type: ignore[attr-defined]
            axis=1,
        )
    )
    assert np.all(
        np.any(
            transaction.post_ledger_state.candidate_identity[candidate_mask]
            != pre_ledger.candidate_identity[candidate_mask],  # type: ignore[attr-defined]
            axis=1,
        )
    )
    np.testing.assert_array_equal(
        transaction.post_ledger_state.active_identity[~active_mask],
        pre_ledger.active_identity[~active_mask],  # type: ignore[attr-defined]
    )
    np.testing.assert_array_equal(
        transaction.post_ledger_state.candidate_identity[~candidate_mask],
        pre_ledger.candidate_identity[~candidate_mask],  # type: ignore[attr-defined]
    )
    assert not any(
        (
            validation.execution_authorized,
            validation.runner_authorized,
            validation.campaign_authorized,
            validation.artifact_writes_authorized,
            validation.threshold_authorized,
            validation.evidence_authorized,
            validation.scientific_promotion_allowed,
        )
    )


def test_normal_v4_output_authenticates_the_next_production_trace(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    observation = jnp.asarray((0.1, 0.2, 0.3), dtype=jnp.float32)
    targets = jnp.asarray((0.4,), dtype=jnp.float32)
    result = bound_scrub.learner.update(bound_scrub.post, observation, targets)
    result.state.step_count.block_until_ready()
    binding = authenticate_generated_birth_identity_trace_by_source_replay(
        bound_scrub.learner,
        bound_scrub.config,
        transaction.post_ledger_state,
        learner_pre_state=bound_scrub.post,
        learner_post_state=result.state,
        supplied_update_result=result,
        observation=observation,
        targets=targets,
    )

    assert binding.source_replay_authenticated
    assert binding.transaction.audit.pre_state_sha256 == (
        transaction.post_ledger_state.integrity_sha256
    )


def test_forged_post_core_state_and_lineage_plan_fail_closed(
    bound_scrub: _BoundScrub,
) -> None:
    forged_post = cast(
        CompositionalFeatureState,
        bound_scrub.post.replace(  # type: ignore[attr-defined]
            output_bias=bound_scrub.post.output_bias.at[0].add(jnp.float32(1.0))
        ),
    )
    with pytest.raises(Exception, match="transaction|validation|scrub"):
        build_generated_birth_identity_scrub_epoch_transaction(
            bound_scrub.config,
            bound_scrub.ledger,  # type: ignore[arg-type]
            bound_scrub.pre,
            forged_post,
            bound_scrub.target,
            bound_scrub.lineage_plan,
            lineage_config=bound_scrub.lineage_config,
            scrub_config=bound_scrub.scrub_config,
            epoch_config=GeneratedReacquisitionEpochConfig(),
        )

    forged_plan = dataclasses.replace(
        bound_scrub.lineage_plan,
        active_mask=bound_scrub.lineage_plan.active_mask.at[3].set(False),
    )
    with pytest.raises(Exception, match="validation|scrub"):
        build_generated_birth_identity_scrub_epoch_transaction(
            bound_scrub.config,
            bound_scrub.ledger,  # type: ignore[arg-type]
            bound_scrub.pre,
            bound_scrub.post,
            bound_scrub.target,
            forged_plan,
            lineage_config=bound_scrub.lineage_config,
            scrub_config=bound_scrub.scrub_config,
            epoch_config=GeneratedReacquisitionEpochConfig(),
        )


def test_forged_identity_assignment_ledger_and_epoch_plan_fail_closed(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    assignments = np.array(transaction.active_scrub_birth_identity, copy=True)
    assignments[np.flatnonzero(transaction.active_scrub_mask)[0], 0] ^= np.uint8(1)
    forged_assignment = dataclasses.replace(
        transaction,
        active_scrub_birth_identity=assignments,
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="self-hash"):
        _validate(forged_assignment, bound_scrub)

    forged_ledger = dataclasses.replace(
        transaction.post_ledger_state,
        integrity_sha256="0" * 64,
    )
    with pytest.raises(Exception, match="integrity"):
        _validate(
            dataclasses.replace(transaction, post_ledger_state=forged_ledger),
            bound_scrub,
        )

    forged_contract = dataclasses.replace(
        transaction.reacquisition_epoch_plan.contract,
        contract_sha256="f" * 64,
    )
    forged_epoch = dataclasses.replace(
        transaction.reacquisition_epoch_plan,
        contract=forged_contract,
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="self-hash"):
        _validate(
            dataclasses.replace(transaction, reacquisition_epoch_plan=forged_epoch),
            bound_scrub,
        )


def test_resealed_assignment_byte_tamper_fails_independent_canonical_rebuild(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    assignments = np.array(transaction.active_scrub_birth_identity, copy=True)
    assignments[np.flatnonzero(transaction.active_scrub_mask)[0], 0] ^= np.uint8(1)
    assignments.setflags(write=False)
    forged = dataclasses.replace(
        transaction,
        active_scrub_birth_identity=assignments,
    )
    forged = dataclasses.replace(
        forged,
        audit=dataclasses.replace(
            forged.audit,
            transaction_sha256=generated_birth_identity_scrub_epoch_transaction_sha256(
                forged
            ),
        ),
    )

    assert forged.audit.transaction_sha256 == (
        generated_birth_identity_scrub_epoch_transaction_sha256(forged)
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="canonical rebuild"):
        _validate(forged, bound_scrub)


def test_scrub_cannot_retain_masked_ids_even_with_resealed_audit(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    active = np.array(transaction.post_ledger_state.active_identity, copy=True)
    active_mask = np.asarray(transaction.active_scrub_mask)
    active[active_mask] = bound_scrub.ledger.active_identity[active_mask]  # type: ignore[attr-defined]
    structural = transaction.post_ledger_state.structural_state
    forged_structural = _ledger._make_state(  # noqa: SLF001
        _ledger._v4_to_v3_config(bound_scrub.config),  # noqa: SLF001
        seed=transaction.post_ledger_state.paired_development_life_seed,
        step=0,
        active_identity=active,
        active_parent_a=structural.active_parent_a,
        active_parent_b=structural.active_parent_b,
        active_ops=structural.active_ops,
        active_depth=structural.active_depth,
        active_generator_policy=structural.active_generator_policy,
        active_generator_policy_sampled=structural.active_generator_policy_sampled,
        candidate_identity=structural.candidate_identity,
        candidate_parent_a=structural.candidate_parent_a,
        candidate_parent_b=structural.candidate_parent_b,
        candidate_ops=structural.candidate_ops,
        candidate_depth=structural.candidate_depth,
        candidate_generator_policy=structural.candidate_generator_policy,
        candidate_generator_policy_sampled=(
            structural.candidate_generator_policy_sampled
        ),
    )
    forged_post = _ledger._make_v4_state(  # noqa: SLF001
        bound_scrub.config,
        step_words=transaction.post_ledger_state.step_words,
        structural_state=forged_structural,
    )
    forged = dataclasses.replace(transaction, post_ledger_state=forged_post)
    forged = dataclasses.replace(
        forged,
        audit=dataclasses.replace(
            forged.audit,
            transaction_sha256=generated_birth_identity_scrub_epoch_transaction_sha256(
                forged
            ),
        ),
    )

    assert forged.audit.transaction_sha256 == (
        generated_birth_identity_scrub_epoch_transaction_sha256(forged)
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="canonical rebuild"):
        _validate(forged, bound_scrub)


def test_epoch_life_seed_must_equal_the_ledger_life(bound_scrub: _BoundScrub) -> None:
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="life seed"):
        build_generated_birth_identity_scrub_epoch_transaction(
            bound_scrub.config,
            bound_scrub.ledger,  # type: ignore[arg-type]
            bound_scrub.pre,
            bound_scrub.post,
            bound_scrub.target,
            bound_scrub.lineage_plan,
            lineage_config=bound_scrub.lineage_config,
            scrub_config=bound_scrub.scrub_config,
            epoch_config=GeneratedReacquisitionEpochConfig(paired_life_seed=211),
        )


def test_resealed_wrong_mask_dtype_and_wrong_tuple_container_fail_exact_compare(
    bound_scrub: _BoundScrub,
) -> None:
    transaction = _build(bound_scrub)
    wrong_dtype = dataclasses.replace(
        transaction,
        active_scrub_mask=np.asarray(transaction.active_scrub_mask, dtype=np.uint8),
    )
    wrong_dtype = dataclasses.replace(
        wrong_dtype,
        audit=dataclasses.replace(
            wrong_dtype.audit,
            transaction_sha256=generated_birth_identity_scrub_epoch_transaction_sha256(
                wrong_dtype
            ),
        ),
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="exact type"):
        _validate(wrong_dtype, bound_scrub)

    wrong_container = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            transaction.audit,
            step_words_uint32=list(transaction.audit.step_words_uint32),
        ),
    )
    wrong_container = dataclasses.replace(
        wrong_container,
        audit=dataclasses.replace(
            wrong_container.audit,
            transaction_sha256=generated_birth_identity_scrub_epoch_transaction_sha256(
                wrong_container
            ),
        ),
    )
    with pytest.raises(GeneratedBirthIdentityScrubEpochError, match="exact type"):
        _validate(wrong_container, bound_scrub)
