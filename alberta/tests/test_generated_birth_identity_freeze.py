"""Cheap focused tests for due-slot generation-freeze identity chaining."""

from __future__ import annotations

import dataclasses
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.compositional_features import (
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
)
from alberta_framework.evaluation.generated_birth_identity_freeze import (
    CAUSAL_FREEZE_ARM,
    MATCHED_SHAM_FREEZE_ARM,
    GeneratedBirthIdentityFreezeDueInputs,
    GeneratedBirthIdentityFreezeEndpointInputs,
    GeneratedBirthIdentityFreezeEndpointTransaction,
    GeneratedBirthIdentityFreezeEndpointValidation,
    GeneratedBirthIdentityFreezeError,
    GeneratedBirthIdentityFreezeOrdinaryStep,
    GeneratedBirthIdentityFreezeTransaction,
    GeneratedBirthIdentityMatchedShamStart,
    GeneratedBirthIdentityPairedFreezeTransaction,
    build_generated_birth_identity_freeze_endpoint_transaction,
    build_generated_birth_identity_freeze_transaction,
    build_generated_birth_identity_matched_sham_start,
    build_generated_birth_identity_paired_freeze_transaction,
    generated_birth_identity_freeze_transaction_sha256,
    generated_birth_identity_matched_sham_start_sha256,
    validate_generated_birth_identity_freeze_endpoint_transaction,
    validate_generated_birth_identity_freeze_transaction,
    validate_generated_birth_identity_matched_sham_start,
    validate_generated_birth_identity_paired_freeze_transaction,
)
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
)
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    GeneratedBirthIdentityScrubEpochInputs,
    GeneratedBirthIdentityScrubEpochTransaction,
    build_generated_birth_identity_scrub_epoch_transaction,
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
    compile_expanded_expression_lineage_masks,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GeneratedReacquisitionEpochConfig,
)

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _DueCase:
    learner: CompositionalFeatureLearner
    config: GeneratedBirthIdentityLedgerV4Config
    scrub: GeneratedBirthIdentityScrubEpochTransaction
    inputs: GeneratedBirthIdentityScrubEpochInputs
    observation: Array
    targets: Array


@dataclasses.dataclass(frozen=True)
class _ArmEndpoint:
    due: GeneratedBirthIdentityFreezeTransaction
    due_inputs: GeneratedBirthIdentityFreezeDueInputs
    endpoint: GeneratedBirthIdentityFreezeEndpointTransaction
    endpoint_inputs: GeneratedBirthIdentityFreezeEndpointInputs
    endpoint_validation: GeneratedBirthIdentityFreezeEndpointValidation


@dataclasses.dataclass(frozen=True)
class _PairedCase:
    case: _DueCase
    sham_start: GeneratedBirthIdentityMatchedShamStart
    causal: _ArmEndpoint
    sham: _ArmEndpoint


def _ordinary_chain(
    case: _DueCase,
    start_core: CompositionalFeatureState,
    start_ledger: GeneratedBirthIdentityLedgerV4State,
    count: int,
) -> tuple[
    tuple[GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    CompositionalFeatureState,
    GeneratedBirthIdentityLedgerV4State,
]:
    steps: list[GeneratedBirthIdentityFreezeOrdinaryStep] = []
    core = start_core
    ledger = start_ledger
    for _ in range(count):
        result: CompositionalFeatureUpdateResult = case.learner.update(
            core,
            case.observation,
            case.targets,
        )
        result.state.step_words.block_until_ready()
        binding = authenticate_generated_birth_identity_trace_by_source_replay(
            case.learner,
            case.config,
            ledger,
            learner_pre_state=core,
            learner_post_state=result.state,
            supplied_update_result=result,
            observation=case.observation,
            targets=case.targets,
        )
        assert not bool(np.asarray(result.curation_trace.should_try_replace))
        assert not bool(np.asarray(result.curation_trace.has_event))
        steps.append(
            GeneratedBirthIdentityFreezeOrdinaryStep(
                learner_pre_state=core,
                supplied_update_result=result,
                observation=case.observation,
                targets=case.targets,
                binding=binding,
            )
        )
        core = result.state
        ledger = binding.transaction.post_state
    return tuple(steps), core, ledger


def _assert_exact_non_key_value(left: object, right: object) -> None:
    if isinstance(left, Array) or isinstance(left, np.ndarray):
        left_array = np.asarray(left)
        right_array = np.asarray(right)
        assert left_array.dtype == right_array.dtype
        assert left_array.shape == right_array.shape
        assert left_array.tobytes(order="C") == right_array.tobytes(order="C")
        return
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        assert type(left) is type(right)
        for field in dataclasses.fields(left):
            _assert_exact_non_key_value(
                getattr(left, field.name),
                getattr(right, field.name),
            )
        return
    assert type(left) is type(right)
    assert left == right


def _case(*, phase: int = 31, step: int = 0) -> _DueCase:
    learner = CompositionalFeatureLearner(
        n_features=4,
        n_tasks=1,
        candidate_count=1,
        replacement_interval=32,
        min_feature_age=0,
        candidate_min_age=0,
        max_depth=3,
        use_obgd=False,
    )
    initialized = learner.init(2, jr.key(991))
    genesis = cast(
        CompositionalFeatureState,
        initialized.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray((OP_RAW, OP_RAW, OP_PRODUCT, OP_SUM), dtype=jnp.int32),
            parent_a=jnp.asarray((0, 1, 0, 2), dtype=jnp.int32),
            parent_b=jnp.asarray((-1, -1, 1, 1), dtype=jnp.int32),
            theta=jnp.zeros((4, 2), dtype=jnp.float32),
            depth=jnp.asarray((0, 0, 1, 2), dtype=jnp.int32),
            candidate_ops=jnp.asarray((OP_PRODUCT,), dtype=jnp.int32),
            candidate_parent_a=jnp.asarray((2,), dtype=jnp.int32),
            candidate_parent_b=jnp.asarray((1,), dtype=jnp.int32),
            candidate_theta=jnp.zeros((1, 2), dtype=jnp.float32),
            candidate_depth=jnp.asarray((2,), dtype=jnp.int32),
            feature_generator_policy=jnp.zeros((4,), dtype=jnp.int32),
            candidate_generator_policy=jnp.zeros((1,), dtype=jnp.int32),
            ages=jnp.full((4,), 10, dtype=jnp.int32),
            candidate_ages=jnp.full((1,), 10, dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.asarray((0, 0), dtype=jnp.uint32),
            replacement_phase=jnp.asarray(0, dtype=jnp.int32),
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    config = GeneratedBirthIdentityLedgerV4Config(
        namespace="freeze-due-focused-development",
        active_slots=4,
        candidate_slots=1,
        raw_feature_slots=2,
        max_depth=3,
        learn_generator_resources=False,
    )
    ledger = attach_generated_birth_identity_ledger_at_core_genesis(
        config,
        learner_pre_state=genesis,
        paired_development_life_seed=101,
    )
    observation = jnp.asarray((0.2, -0.1), dtype=jnp.float32)
    targets = jnp.asarray((0.3,), dtype=jnp.float32)
    pre = genesis
    if step:
        assert phase == step % 32
        for _ in range(step):
            result = learner.update(pre, observation, targets)
            result.state.step_words.block_until_ready()
            binding = authenticate_generated_birth_identity_trace_by_source_replay(
                learner,
                config,
                ledger,
                learner_pre_state=pre,
                learner_post_state=result.state,
                supplied_update_result=result,
                observation=observation,
                targets=targets,
            )
            assert not bool(np.asarray(result.curation_trace.has_event))
            pre = result.state
            ledger = binding.transaction.post_state
    else:
        pre = cast(
            CompositionalFeatureState,
            genesis.replace(  # type: ignore[attr-defined]
                replacement_phase=jnp.asarray(phase, dtype=jnp.int32)
            ),
        )
    target: GeneratedExpression = product_expression(raw_expression(0), raw_expression(1))
    lineage_config = ExpandedExpressionLineageConfig(
        feature_dim=2,
        active_slots=4,
        candidate_slots=1,
        n_tasks=1,
        generator_contexts=1,
        generator_policy_count=4,
    )
    scrub_config = GeneratedClassScrubConfig(
        feature_dim=2,
        active_slots=4,
        candidate_slots=1,
        n_tasks=1,
    )
    lineage_plan = compile_expanded_expression_lineage_masks(
        pre,
        target,
        config=lineage_config,
    )
    scrub_result = scrub_compositional_feature_state(
        pre,
        lineage_plan.active_mask,
        lineage_plan.candidate_mask,
        jnp.asarray(True, dtype=jnp.bool_),
        config=scrub_config,
    )
    assert bool(scrub_result.diagnostics.committed)
    epoch_config = GeneratedReacquisitionEpochConfig()
    scrub = build_generated_birth_identity_scrub_epoch_transaction(
        config,
        ledger,
        pre,
        scrub_result.state,
        target,
        lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        epoch_config=epoch_config,
    )
    inputs = GeneratedBirthIdentityScrubEpochInputs(
        config=config,
        pre_ledger_state=ledger,
        pre_core_state=pre,
        post_core_state=scrub_result.state,
        target=target,
        lineage_plan=lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        epoch_config=epoch_config,
    )
    return _DueCase(
        learner=learner,
        config=config,
        scrub=scrub,
        inputs=inputs,
        observation=observation,
        targets=targets,
    )


def _build_arm_endpoint(
    case: _DueCase,
    *,
    start_core: CompositionalFeatureState,
    start_ledger: GeneratedBirthIdentityLedgerV4State,
    matched_sham_start: GeneratedBirthIdentityMatchedShamStart | None,
) -> _ArmEndpoint:
    interval = int(case.learner.to_config()["replacement_interval"])
    start_phase = int(np.asarray(start_core.replacement_phase))
    prefix_count = interval - start_phase - 1
    prefix, due_pre_core, due_pre_ledger = _ordinary_chain(
        case,
        start_core,
        start_ledger,
        prefix_count,
    )
    due_inputs = GeneratedBirthIdentityFreezeDueInputs(
        learner=case.learner,
        config=case.config,
        ledger_pre_state=due_pre_ledger,
        learner_pre_state=due_pre_core,
        observation=case.observation,
        targets=case.targets,
        scrub_rollover=case.scrub,
        scrub_inputs=case.inputs,
        prefix_steps=prefix,
        matched_sham_start=matched_sham_start,
    )
    due = build_generated_birth_identity_freeze_transaction(
        case.learner,
        case.config,
        due_pre_ledger,
        due_pre_core,
        case.observation,
        case.targets,
        case.scrub,
        case.inputs,
        prefix,
        matched_sham_start=matched_sham_start,
    )
    due_validation = validate_generated_birth_identity_freeze_transaction(
        due,
        learner=case.learner,
        config=case.config,
        ledger_pre_state=due_pre_ledger,
        learner_pre_state=due_pre_core,
        observation=case.observation,
        targets=case.targets,
        scrub_rollover=case.scrub,
        scrub_inputs=case.inputs,
        prefix_steps=prefix,
        matched_sham_start=matched_sham_start,
    )
    assert due_validation.valid
    suffix, end_core, end_ledger = _ordinary_chain(
        case,
        due.committed_core_state,
        due.carried_ledger_state,
        due.audit.suffix_update_count,
    )
    endpoint_inputs = GeneratedBirthIdentityFreezeEndpointInputs(
        due_transaction=due,
        due_inputs=due_inputs,
        freeze_end_core_state=end_core,
        freeze_end_ledger_state=end_ledger,
        suffix_steps=suffix,
    )
    endpoint = build_generated_birth_identity_freeze_endpoint_transaction(
        due,
        due_inputs,
        end_core,
        end_ledger,
        suffix,
    )
    endpoint_validation = validate_generated_birth_identity_freeze_endpoint_transaction(
        endpoint,
        due_transaction=due,
        due_inputs=due_inputs,
        freeze_end_core_state=end_core,
        freeze_end_ledger_state=end_ledger,
        suffix_steps=suffix,
    )
    assert endpoint_validation.valid
    return _ArmEndpoint(
        due=due,
        due_inputs=due_inputs,
        endpoint=endpoint,
        endpoint_inputs=endpoint_inputs,
        endpoint_validation=endpoint_validation,
    )


@pytest.fixture(scope="module")
def causal_due() -> tuple[_DueCase, GeneratedBirthIdentityFreezeTransaction]:
    case = _case()
    transaction = build_generated_birth_identity_freeze_transaction(
        case.learner,
        case.config,
        case.scrub.post_ledger_state,
        case.inputs.post_core_state,
        case.observation,
        case.targets,
        case.scrub,
        case.inputs,
        (),
    )
    return case, transaction


@pytest.fixture(scope="module")
def paired_case() -> _PairedCase:
    case = _case(phase=17, step=17)
    sham_start = build_generated_birth_identity_matched_sham_start(
        case.scrub,
        case.inputs,
    )
    causal = _build_arm_endpoint(
        case,
        start_core=case.inputs.post_core_state,
        start_ledger=case.scrub.post_ledger_state,
        matched_sham_start=None,
    )
    sham = _build_arm_endpoint(
        case,
        start_core=sham_start.start_core_state,
        start_ledger=sham_start.start_ledger_state,
        matched_sham_start=sham_start,
    )
    return _PairedCase(
        case=case,
        sham_start=sham_start,
        causal=causal,
        sham=sham,
    )


def test_zero_prefix_phase_derived_due_replays_both_branches(
    causal_due: tuple[_DueCase, GeneratedBirthIdentityFreezeTransaction],
) -> None:
    case, transaction = causal_due
    validation = validate_generated_birth_identity_freeze_transaction(
        transaction,
        learner=case.learner,
        config=case.config,
        ledger_pre_state=case.scrub.post_ledger_state,
        learner_pre_state=case.inputs.post_core_state,
        observation=case.observation,
        targets=case.targets,
        scrub_rollover=case.scrub,
        scrub_inputs=case.inputs,
        prefix_steps=(),
    )

    assert validation.valid
    assert transaction.audit.arm_mode == CAUSAL_FREEZE_ARM
    assert transaction.audit.phase_derived_due_pre_step == 0
    assert transaction.audit.phase_derived_due_post_step == 1
    assert transaction.audit.prefix_update_count == 0
    assert transaction.audit.suffix_update_count == 31
    assert transaction.audit.attempted_branch_abandoned
    assert transaction.audit.shadow_no_event_branch_carried
    assert transaction.audit.work.total_learner_update_calls_for_validated_transaction == 8


def test_matched_sham_executes_noncommitting_scrub_and_dual_due_work() -> None:
    case = _case()
    sham = build_generated_birth_identity_matched_sham_start(case.scrub, case.inputs)
    validate_generated_birth_identity_matched_sham_start(
        sham,
        causal_scrub=case.scrub,
        scrub_inputs=case.inputs,
    )
    transaction = build_generated_birth_identity_freeze_transaction(
        case.learner,
        case.config,
        sham.start_ledger_state,
        sham.start_core_state,
        case.observation,
        case.targets,
        case.scrub,
        case.inputs,
        (),
        matched_sham_start=sham,
    )

    assert transaction.audit.arm_mode == MATCHED_SHAM_FREEZE_ARM
    assert transaction.audit.matched_sham_scrub_work_executed
    assert transaction.audit.work.total_matched_sham_scrub_kernel_calls == 3
    assert transaction.audit.work.total_learner_update_calls_for_validated_transaction == 8
    assert not sham.audit.threshold_authorized


def test_missing_authenticated_prefix_fails_before_due_replay() -> None:
    case = _case(phase=30)
    _, due_pre_core, due_pre_ledger = _ordinary_chain(
        case,
        case.inputs.post_core_state,
        case.scrub.post_ledger_state,
        1,
    )
    with pytest.raises(GeneratedBirthIdentityFreezeError, match="step count"):
        build_generated_birth_identity_freeze_transaction(
            case.learner,
            case.config,
            due_pre_ledger,
            due_pre_core,
            case.observation,
            case.targets,
            case.scrub,
            case.inputs,
            (),
        )


def test_stale_nested_carried_ledger_bits_fail_strict_branch_validation(
    causal_due: tuple[_DueCase, GeneratedBirthIdentityFreezeTransaction],
) -> None:
    case, transaction = causal_due
    identity = np.array(
        transaction.carried_binding.transaction.post_state.candidate_identity,
        copy=True,
    )
    identity[0, 0] ^= np.uint8(1)
    identity.setflags(write=False)
    forged_structural = dataclasses.replace(
        transaction.carried_binding.transaction.post_state.structural_state,
        candidate_identity=identity,
    )
    forged_ledger = dataclasses.replace(
        transaction.carried_binding.transaction.post_state,
        structural_state=forged_structural,
    )
    forged_ledger_transaction = dataclasses.replace(
        transaction.carried_binding.transaction,
        post_state=forged_ledger,
    )
    forged_binding = dataclasses.replace(
        transaction.carried_binding,
        transaction=forged_ledger_transaction,
    )
    forged = dataclasses.replace(transaction, carried_binding=forged_binding)
    forged = dataclasses.replace(
        forged,
        audit=dataclasses.replace(
            forged.audit,
            transaction_sha256=generated_birth_identity_freeze_transaction_sha256(
                forged
            ),
        ),
    )

    with pytest.raises(Exception, match="integrity|canonical"):
        validate_generated_birth_identity_freeze_transaction(
            forged,
            learner=case.learner,
            config=case.config,
            ledger_pre_state=case.scrub.post_ledger_state,
            learner_pre_state=case.inputs.post_core_state,
            observation=case.observation,
            targets=case.targets,
            scrub_rollover=case.scrub,
            scrub_inputs=case.inputs,
            prefix_steps=(),
        )


def test_tuple_list_tamper_remains_kind_distinct_after_reseal(
    causal_due: tuple[_DueCase, GeneratedBirthIdentityFreezeTransaction],
) -> None:
    case, transaction = causal_due
    forged = dataclasses.replace(
        transaction,
        audit=dataclasses.replace(
            transaction.audit,
            planned_fresh_key_data_uint32=list(  # type: ignore[arg-type]
                transaction.audit.planned_fresh_key_data_uint32
            ),
        ),
    )
    forged = dataclasses.replace(
        forged,
        audit=dataclasses.replace(
            forged.audit,
            transaction_sha256=generated_birth_identity_freeze_transaction_sha256(
                forged
            ),
        ),
    )

    with pytest.raises(GeneratedBirthIdentityFreezeError, match="complete freeze"):
        validate_generated_birth_identity_freeze_transaction(
            forged,
            learner=case.learner,
            config=case.config,
            ledger_pre_state=case.scrub.post_ledger_state,
            learner_pre_state=case.inputs.post_core_state,
            observation=case.observation,
            targets=case.targets,
            scrub_rollover=case.scrub,
            scrub_inputs=case.inputs,
            prefix_steps=(),
        )


def test_resealed_bool_int_tamper_remains_exact_type_distinct() -> None:
    case = _case()
    receipt = build_generated_birth_identity_matched_sham_start(
        case.scrub,
        case.inputs,
    )
    forged = dataclasses.replace(
        receipt,
        audit=dataclasses.replace(
            receipt.audit,
            development_only=1,  # type: ignore[arg-type]
        ),
    )
    forged = dataclasses.replace(
        forged,
        audit=dataclasses.replace(
            forged.audit,
            transaction_sha256=generated_birth_identity_matched_sham_start_sha256(
                forged,
            ),
        ),
    )

    assert forged.audit.transaction_sha256 == (
        generated_birth_identity_matched_sham_start_sha256(forged)
    )
    with pytest.raises(GeneratedBirthIdentityFreezeError, match="matched sham"):
        validate_generated_birth_identity_matched_sham_start(
            forged,
            causal_scrub=case.scrub,
            scrub_inputs=case.inputs,
        )


@pytest.mark.slow
def test_phase_unaligned_nonzero_prefix_endpoint_and_fresh_key_only(
    paired_case: _PairedCase,
) -> None:
    arm = paired_case.causal
    validation = arm.endpoint_validation

    assert validation.valid
    assert arm.due.audit.freeze_start_step == 17
    assert arm.due.audit.prefix_update_count == 14
    assert arm.due.audit.phase_derived_due_pre_step == 31
    assert arm.due.audit.phase_derived_due_post_step == 32
    assert arm.due.audit.suffix_update_count == 17
    assert arm.endpoint.audit.freeze_end_step == 49
    assert arm.endpoint.audit.total_learner_update_calls_for_validated_endpoint == 168
    assert (
        arm.endpoint.audit.total_matched_sham_scrub_kernel_calls_for_validated_endpoint
        == 0
    )
    assert arm.endpoint.audit.fresh_key_is_only_endpoint_state_change
    for field in dataclasses.fields(CompositionalFeatureState):  # type: ignore[arg-type]
        if field.name != "key":
            _assert_exact_non_key_value(
                getattr(arm.endpoint.freeze_end_core_state, field.name),
                getattr(arm.endpoint.fresh_key_applied_core_state, field.name),
            )
    np.testing.assert_array_equal(
        jr.key_data(arm.endpoint.fresh_key_applied_core_state.key),
        jr.key_data(paired_case.case.scrub.reacquisition_epoch_plan.fresh_learner_key),
    )
    assert not np.array_equal(
        np.asarray(jr.key_data(arm.endpoint.freeze_end_core_state.key)),
        np.asarray(jr.key_data(arm.endpoint.fresh_key_applied_core_state.key)),
    )
    next_result = paired_case.case.learner.update(
        arm.endpoint.fresh_key_applied_core_state,
        paired_case.case.observation,
        paired_case.case.targets,
    )
    next_result.state.step_words.block_until_ready()
    next_binding = authenticate_generated_birth_identity_trace_by_source_replay(
        paired_case.case.learner,
        paired_case.case.config,
        arm.endpoint.carried_ledger_state,
        learner_pre_state=arm.endpoint.fresh_key_applied_core_state,
        learner_post_state=next_result.state,
        supplied_update_result=next_result,
        observation=paired_case.case.observation,
        targets=paired_case.case.targets,
    )
    assert next_binding.source_replay_authenticated


@pytest.mark.slow
def test_paired_endpoint_success_accounts_full_work_and_discards_sham(
    paired_case: _PairedCase,
) -> None:
    transaction: GeneratedBirthIdentityPairedFreezeTransaction = (
        build_generated_birth_identity_paired_freeze_transaction(
            paired_case.causal.endpoint,
            paired_case.causal.endpoint_inputs,
            paired_case.sham.endpoint,
            paired_case.sham.endpoint_inputs,
        )
    )
    validation = validate_generated_birth_identity_paired_freeze_transaction(
        transaction,
        causal_endpoint=paired_case.causal.endpoint,
        causal_inputs=paired_case.causal.endpoint_inputs,
        sham_endpoint=paired_case.sham.endpoint,
        sham_inputs=paired_case.sham.endpoint_inputs,
    )

    assert validation.valid
    assert transaction.audit.exact_crn_input_parity
    assert transaction.audit.causal_total_learner_update_calls == 238
    assert transaction.audit.sham_total_learner_update_calls == 238
    assert (
        transaction.audit.total_matched_sham_scrub_kernel_calls_for_validated_pair
        == 7
    )
    assert validation.causal_total_learner_update_calls_accounted == 238
    assert validation.sham_total_learner_update_calls_accounted == 238
    assert validation.total_matched_sham_scrub_kernel_calls_accounted == 7
    assert transaction.audit.sham_endpoint_state_discarded
    assert (
        transaction.causal_output_core_state
        is paired_case.causal.endpoint.fresh_key_applied_core_state
    )
    assert (
        transaction.causal_output_ledger_state
        is paired_case.causal.endpoint.carried_ledger_state
    )
    assert (
        transaction.causal_output_core_state
        is not paired_case.sham.endpoint.fresh_key_applied_core_state
    )


@pytest.mark.slow
def test_paired_rejects_one_key_and_one_shared_contract_field_mismatch(
    paired_case: _PairedCase,
) -> None:
    prefix = paired_case.causal.due_inputs.prefix_steps
    forged_pre = cast(
        CompositionalFeatureState,
        prefix[0].learner_pre_state.replace(key=jr.key(993)),  # type: ignore[attr-defined]
    )
    forged_prefix = (
        dataclasses.replace(prefix[0], learner_pre_state=forged_pre),
        *prefix[1:],
    )
    key_due_inputs = dataclasses.replace(
        paired_case.causal.due_inputs,
        prefix_steps=forged_prefix,
    )
    key_endpoint_inputs = dataclasses.replace(
        paired_case.causal.endpoint_inputs,
        due_inputs=key_due_inputs,
    )
    with pytest.raises(GeneratedBirthIdentityFreezeError):
        build_generated_birth_identity_paired_freeze_transaction(
            paired_case.causal.endpoint,
            key_endpoint_inputs,
            paired_case.sham.endpoint,
            paired_case.sham.endpoint_inputs,
        )

    contract = paired_case.case.scrub.reacquisition_epoch_plan.contract
    forged_contract = dataclasses.replace(
        contract,
        paired_life_seed=contract.paired_life_seed + 1,
    )
    forged_plan = dataclasses.replace(
        paired_case.case.scrub.reacquisition_epoch_plan,
        contract=forged_contract,
    )
    forged_scrub = dataclasses.replace(
        paired_case.case.scrub,
        reacquisition_epoch_plan=forged_plan,
    )
    contract_due_inputs = dataclasses.replace(
        paired_case.causal.due_inputs,
        scrub_rollover=forged_scrub,
    )
    contract_endpoint_inputs = dataclasses.replace(
        paired_case.causal.endpoint_inputs,
        due_inputs=contract_due_inputs,
    )
    with pytest.raises(Exception, match="hash|canonical|contract"):
        build_generated_birth_identity_paired_freeze_transaction(
            paired_case.causal.endpoint,
            contract_endpoint_inputs,
            paired_case.sham.endpoint,
            paired_case.sham.endpoint_inputs,
        )
