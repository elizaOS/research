"""Development-only generated-class recurrence execution contracts."""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from alberta_framework.core.compositional_features import (
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    attach_generated_birth_identity_ledger_at_core_genesis,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    FINITE_DEGREE_TWO_ARCHIVE_CEILING,
    FROZEN_LIFECYCLE,
    FULL_LIFECYCLE,
    RANDOM_CURATION,
    ZERO_CANDIDATE_HEAD_CARRY,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_learner,
    derive_expression_manifest,
    evaluate_expression,
)
from alberta_framework.evaluation.generated_class_recurrence_development import (
    D_NEVER_SEEN_TWIN,
    DECLARED_ARM_ORDER,
    GENERATED_CLASS_PAIRED_FREEZE_SCHEMA,
    GENERATED_CLASS_PAIRED_FREEZE_STATUS,
    GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS,
    GENERATION_FREEZE_UPDATES,
    MATCHED_SHAM_SCRUB,
    REALISTIC_FREEZE_ENDPOINT_STEP,
    REALISTIC_SCRUB_BOUNDARY_STEP,
    GeneratedClassPairedFreezeExecution,
    GeneratedClassRecurrenceAdapterUnavailableError,
    GeneratedClassRecurrenceDevelopmentError,
    build_generated_class_development_stream,
    build_generated_class_recurrence_development_plan,
    run_authenticated_generated_class_paired_scrub_freeze,
    run_tiny_generated_class_recurrence_replay,
    validate_authenticated_generated_class_paired_scrub_freeze,
    validate_generated_class_development_campaign,
    validate_generated_class_recurrence_development_plan,
)

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _PairedMechanicalCase:
    learner: CompositionalFeatureLearner
    config: GeneratedBirthIdentityLedgerV4Config
    genesis_core: CompositionalFeatureState
    genesis_ledger: GeneratedBirthIdentityLedgerV4State
    observations: tuple[Array, ...]
    targets: tuple[Array, ...]


def _paired_mechanical_case() -> _PairedMechanicalCase:
    """Build a deterministic D-present genesis for transaction-mechanics tests.

    This is deliberately not a scientific generated-class run: the exact D
    lineage is installed only to exercise the external scrub/freeze plumbing.
    The production development runner itself performs no feature injection.
    """

    protocol = build_generated_class_recurrence_v0_protocol()
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    initialized = learner.init(protocol.input_dim, jr.key(801))
    genesis = cast(
        CompositionalFeatureState,
        initialized.replace(  # type: ignore[attr-defined]
            ops=jnp.asarray(
                (
                    OP_RAW,
                    OP_RAW,
                    OP_RAW,
                    OP_RAW,
                    OP_PRODUCT,
                    OP_GATED,
                    OP_SUM,
                    OP_PRODUCT,
                    OP_PRODUCT,
                    OP_PRODUCT,
                    OP_PRODUCT,
                    OP_PRODUCT,
                    OP_PRODUCT,
                    OP_PRODUCT,
                ),
                dtype=jnp.int32,
            ),
            parent_a=jnp.asarray(
                (0, 1, 2, 3, 0, 4, 5, 0, 0, 0, 1, 1, 1, 2),
                dtype=jnp.int32,
            ),
            parent_b=jnp.asarray(
                (-1, -1, -1, -1, 0, 1, 2, 1, 2, 3, 1, 2, 3, 2),
                dtype=jnp.int32,
            ),
            theta=jnp.zeros((protocol.active_slots, 2), dtype=jnp.float32),
            depth=jnp.asarray(
                (0, 0, 0, 0, 1, 2, 3, 1, 1, 1, 1, 1, 1, 1),
                dtype=jnp.int32,
            ),
            candidate_ops=jnp.asarray(
                (
                    OP_GATED,
                    OP_PRODUCT,
                    OP_SUM,
                    OP_GATED,
                    OP_PRODUCT,
                    OP_SUM,
                    OP_GATED,
                    OP_PRODUCT,
                ),
                dtype=jnp.int32,
            ),
            candidate_parent_a=jnp.asarray(
                (5, 7, 8, 9, 10, 11, 12, 13),
                dtype=jnp.int32,
            ),
            candidate_parent_b=jnp.asarray(
                (2, 2, 3, 1, 2, 0, 3, 1),
                dtype=jnp.int32,
            ),
            candidate_theta=jnp.zeros(
                (protocol.candidate_slots, 2),
                dtype=jnp.float32,
            ),
            candidate_depth=jnp.asarray((3, 2, 2, 2, 2, 2, 2, 2), dtype=jnp.int32),
            feature_generator_policy=jnp.zeros(
                (protocol.active_slots,),
                dtype=jnp.int32,
            ),
            candidate_generator_policy=jnp.zeros(
                (protocol.candidate_slots,),
                dtype=jnp.int32,
            ),
            ages=jnp.zeros((protocol.active_slots,), dtype=jnp.int32),
            candidate_ages=jnp.zeros((protocol.candidate_slots,), dtype=jnp.int32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
            step_words=jnp.asarray((0, 0), dtype=jnp.uint32),
            replacement_phase=jnp.asarray(0, dtype=jnp.int32),
            birth_timestamp=jnp.asarray(0.0, dtype=jnp.float32),
            uptime_s=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    config = GeneratedBirthIdentityLedgerV4Config(
        namespace="generated-class-paired-freeze-mechanics",
        active_slots=protocol.active_slots,
        candidate_slots=protocol.candidate_slots,
        raw_feature_slots=protocol.input_dim,
        max_depth=protocol.allocated_max_depth,
        learn_generator_resources=False,
    )
    ledger = attach_generated_birth_identity_ledger_at_core_genesis(
        config,
        learner_pre_state=genesis,
        paired_development_life_seed=101,
    )
    observation_matrix = jr.normal(
        jr.key(802),
        (REALISTIC_FREEZE_ENDPOINT_STEP, protocol.input_dim),
        dtype=jnp.float32,
    )
    observations = tuple(
        observation_matrix[index] for index in range(observation_matrix.shape[0])
    )
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_a = next(target.expression for target in manifest.targets if target.name == "A")
    targets = tuple(
        jnp.expand_dims(evaluate_expression(target_a, observation), axis=0)
        for observation in observations
    )
    return _PairedMechanicalCase(
        learner=learner,
        config=config,
        genesis_core=genesis,
        genesis_ledger=ledger,
        observations=observations,
        targets=targets,
    )


def _tiny_plan() -> Any:
    return build_generated_class_recurrence_development_plan(
        phase_lengths=(1,) * 9,
        seeds=(7,),
    )


def test_plan_declares_exact_matched_seven_arm_nonpromoting_life() -> None:
    plan = _tiny_plan()

    assert plan.status == GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS
    assert plan.development_only
    assert not plan.scientific_evidence_authorized
    assert not plan.promotion_authorized
    assert not plan.artifact_writes_authorized
    assert not plan.thresholds_authorized
    assert plan.phase_order == ("A", "B", "A", "D", "A", "C", "A", "D", "A")
    assert plan.phase_starts == tuple(range(9))
    assert plan.total_steps == 9
    assert plan.tiny_noncurating_replay
    assert not plan.canonical_full_life
    assert tuple(arm.name for arm in plan.arms) == (
        FULL_LIFECYCLE,
        RANDOM_CURATION,
        FROZEN_LIFECYCLE,
        ZERO_CANDIDATE_HEAD_CARRY,
        FINITE_DEGREE_TWO_ARCHIVE_CEILING,
        MATCHED_SHAM_SCRUB,
        D_NEVER_SEEN_TWIN,
    ) == DECLARED_ARM_ORDER
    assert len({arm.persistent_jax_state_nbytes for arm in plan.arms}) == 1
    assert len({arm.allocated_active_slots for arm in plan.arms}) == 1
    assert len({arm.allocated_candidate_slots for arm in plan.arms}) == 1
    assert all(arm.allocated_max_depth == 3 for arm in plan.arms)
    assert plan.work.equal_persistent_capacity_declared
    assert plan.work.equal_logical_work_declared
    assert plan.work.total_learner_updates == 7 * 9
    assert plan.work.trace_authentication_attempts_per_arm == 9
    assert plan.work.artifact_bytes_written == 0
    assert plan.work.wall_clock_threshold is None
    assert plan.learner_observation_fields == ("raw_features",)
    assert not set(plan.learner_observation_fields) & set(plan.evaluator_only_fields)


def test_stream_is_uninterrupted_crn_and_d_twin_changes_only_evaluator_target() -> None:
    plan = _tiny_plan()
    full = build_generated_class_development_stream(plan, 7, FULL_LIFECYCLE)
    sham = build_generated_class_development_stream(plan, 7, MATCHED_SHAM_SCRUB)
    twin = build_generated_class_development_stream(plan, 7, D_NEVER_SEEN_TWIN)

    assert tuple(step.raw_features for step in full.steps) == tuple(
        step.raw_features for step in sham.steps
    ) == tuple(step.raw_features for step in twin.steps)
    assert tuple(step.phase_label for step in full.steps) == plan.phase_order
    assert tuple(step.step_index for step in full.steps) == tuple(range(9))
    assert all(step.phase_boundary for step in full.steps)
    d_steps = tuple(index for index, name in enumerate(plan.phase_order) if name == "D")
    assert d_steps == (3, 7)
    for index, (full_step, twin_step) in enumerate(
        zip(full.steps, twin.steps, strict=True)
    ):
        if index == d_steps[0]:
            assert full_step.presented_target_name == "D"
            assert twin_step.presented_target_name == "D_mapping_deranged"
        else:
            assert full_step.presented_target_name == twin_step.presented_target_name
            assert np.float32(full_step.target).tobytes() == np.float32(
                twin_step.target
            ).tobytes()
    assert full.observation_key_words_uint32 == twin.observation_key_words_uint32
    assert full.stream_sha256 != twin.stream_sha256


@pytest.mark.parametrize(
    "bad_lengths,error",
    [
        ((1,) * 8, ValueError),
        ((1,) * 8 + (True,), TypeError),
        ((1,) * 8 + (0,), ValueError),
    ],
)
def test_malformed_phase_manifests_fail_before_execution(
    bad_lengths: tuple[object, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_generated_class_recurrence_development_plan(
            phase_lengths=bad_lengths,  # type: ignore[arg-type]
            seeds=(7,),
        )


def test_authority_work_and_arm_tampering_fail_closed() -> None:
    plan = _tiny_plan()
    with pytest.raises(ValueError, match="authority"):
        validate_generated_class_recurrence_development_plan(
            dataclasses.replace(plan, scientific_evidence_authorized=True)
        )
    with pytest.raises(ValueError, match="seven-arm"):
        validate_generated_class_recurrence_development_plan(
            dataclasses.replace(plan, arms=tuple(reversed(plan.arms)))
        )
    with pytest.raises(ValueError, match="logical work"):
        validate_generated_class_recurrence_development_plan(
            dataclasses.replace(
                plan,
                work=dataclasses.replace(
                    plan.work,
                    equal_logical_work_declared=False,
                ),
            )
        )


def test_canonical_plan_has_no_accidental_tiny_or_default_execution() -> None:
    canonical = build_generated_class_recurrence_development_plan(seeds=(7,))

    assert canonical.canonical_full_life
    assert not canonical.tiny_noncurating_replay
    assert canonical.total_steps > canonical.replacement_interval
    assert REALISTIC_SCRUB_BOUNDARY_STEP + GENERATION_FREEZE_UPDATES == (
        REALISTIC_FREEZE_ENDPOINT_STEP
    )
    with pytest.raises(GeneratedClassRecurrenceDevelopmentError, match="explicit noncanonical"):
        run_tiny_generated_class_recurrence_replay(
            canonical,
            arm_names=(FULL_LIFECYCLE,),
        )


def test_paired_scrub_freeze_has_no_default_or_implicit_execution() -> None:
    with pytest.raises(TypeError, match="learner"):
        run_authenticated_generated_class_paired_scrub_freeze(
            cast(Any, object()),
            cast(Any, object()),
            cast(Any, object()),
            cast(Any, object()),
            (),
            (),
        )


def test_missing_trace_adapter_fails_before_any_update(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _tiny_plan()

    def unavailable(_name: str) -> Any:
        raise ImportError("hostile missing adapter")

    monkeypatch.setattr(
        "alberta_framework.evaluation.generated_class_recurrence_development."
        "importlib.import_module",
        unavailable,
    )
    with pytest.raises(
        GeneratedClassRecurrenceAdapterUnavailableError,
        match="source-replay",
    ):
        run_tiny_generated_class_recurrence_replay(
            plan,
            arm_names=(FULL_LIFECYCLE,),
        )


@pytest.fixture(scope="module")
def paired_mechanical_execution() -> GeneratedClassPairedFreezeExecution:
    case = _paired_mechanical_case()
    return run_authenticated_generated_class_paired_scrub_freeze(
        case.learner,
        case.config,
        case.genesis_core,
        case.genesis_ledger,
        case.observations,
        case.targets,
        paired_life_seed=101,
    )


@pytest.mark.slow
def test_realistic_paired_scrub_freeze_executes_both_paths_and_carries_only_causal(
    paired_mechanical_execution: GeneratedClassPairedFreezeExecution,
) -> None:
    execution = paired_mechanical_execution
    receipt = execution.receipt
    audit = receipt.audit
    causal_due = receipt.causal_endpoint_inputs.due_transaction
    sham_due = receipt.sham_endpoint_inputs.due_transaction
    causal_endpoint = receipt.paired_transaction.causal_endpoint
    sham_endpoint = receipt.paired_transaction.sham_endpoint

    assert audit.schema == GENERATED_CLASS_PAIRED_FREEZE_SCHEMA
    assert audit.status == GENERATED_CLASS_PAIRED_FREEZE_STATUS
    assert audit.genesis_step == 0
    assert audit.scrub_boundary_step == REALISTIC_SCRUB_BOUNDARY_STEP == 17
    assert audit.phase_derived_due_pre_step == 31
    assert audit.phase_derived_due_post_step == 32
    assert audit.freeze_endpoint_step == REALISTIC_FREEZE_ENDPOINT_STEP == 49
    assert audit.exogenous_step_count == 49
    assert len(receipt.genesis_prefix_steps) == 17
    assert len(receipt.causal_endpoint_inputs.due_inputs.prefix_steps) == 14
    assert len(receipt.sham_endpoint_inputs.due_inputs.prefix_steps) == 14
    assert len(receipt.causal_endpoint_inputs.suffix_steps) == 17
    assert len(receipt.sham_endpoint_inputs.suffix_steps) == 17

    assert receipt.scrub_transaction.audit.structural_scrub_valid
    assert receipt.matched_sham_start.audit.matched_sham_scrub_executed
    assert not receipt.matched_sham_start.audit.matched_sham_scrub_commit_requested
    assert receipt.matched_sham_start.audit.matched_sham_scrub_noop_validated
    assert audit.causal_scrub_committed
    assert audit.matched_sham_scrub_executed_noncommitting
    assert audit.scrubbed_candidate_heads_zero_at_identity_birth

    for due in (causal_due, sham_due):
        assert due.audit.attempted_event_authenticated
        assert due.audit.attempted_branch_abandoned
        assert due.audit.shadow_no_event_authenticated
        assert due.audit.shadow_no_event_branch_carried
        assert due.audit.ordinary_learning_preserved_bit_exactly
    for endpoint in (causal_endpoint, sham_endpoint):
        assert endpoint.audit.due_transaction_strictly_revalidated
        assert endpoint.audit.suffix_every_core_and_ledger_bit_source_replayed
        assert endpoint.audit.fresh_key_is_only_endpoint_state_change
        assert endpoint.audit.endpoint_core_ledger_pair_ready_for_next_trace

    assert receipt.paired_validation.valid
    assert receipt.paired_validation.causal_path_strictly_validated
    assert receipt.paired_validation.sham_path_strictly_validated
    assert receipt.paired_validation.exact_learner_work_parity
    assert receipt.paired_validation.matched_sham_work_actually_consumed
    assert receipt.paired_validation.sham_endpoint_state_discarded
    assert receipt.paired_validation.causal_output_ready_for_next_trace
    assert audit.exact_crn_input_parity
    assert audit.typed_key_checkpoints_bound
    assert audit.fresh_key_applied_only_at_endpoint
    assert audit.sham_endpoint_state_discarded
    assert audit.causal_output_only
    assert execution.core_state is receipt.paired_transaction.causal_output_core_state
    assert execution.ledger_state is receipt.paired_transaction.causal_output_ledger_state
    assert audit.fresh_key_words_uint32 != audit.freeze_end_pre_fresh_key_words_uint32

    accounting = audit.accounting
    assert accounting.genesis_prefix_direct_update_calls == 17
    assert accounting.genesis_prefix_source_replay_calls == 17
    assert accounting.causal_total_learner_update_calls == (
        accounting.sham_total_learner_update_calls
    )
    assert accounting.exact_learner_work_parity
    assert accounting.matched_sham_scrub_kernel_calls > 0
    assert accounting.measured_runtime_sample_count == 0
    assert not accounting.measured_runtime_parity_claimed
    assert accounting.wall_clock_threshold is None
    assert audit.development_only
    assert not any(
        (
            audit.execution_authorized,
            audit.runner_authorized,
            audit.campaign_authorized,
            audit.artifact_writes_authorized,
            audit.threshold_authorized,
            audit.evidence_authorized,
            audit.scientific_promotion_allowed,
        )
    )


@pytest.mark.slow
def test_paired_scrub_freeze_revalidates_and_nested_audit_tamper_fails_closed(
    paired_mechanical_execution: GeneratedClassPairedFreezeExecution,
) -> None:
    assert (
        validate_authenticated_generated_class_paired_scrub_freeze(
            paired_mechanical_execution
        )
        is paired_mechanical_execution
    )
    forged_audit = dataclasses.replace(
        paired_mechanical_execution.receipt.audit,
        exact_crn_input_parity=False,
    )
    forged_receipt = dataclasses.replace(
        paired_mechanical_execution.receipt,
        audit=forged_audit,
    )
    forged_execution = dataclasses.replace(
        paired_mechanical_execution,
        receipt=forged_receipt,
    )
    with pytest.raises(ValueError, match="receipt hash"):
        validate_authenticated_generated_class_paired_scrub_freeze(forged_execution)


@pytest.fixture(scope="module")
def exact_replays() -> tuple[Any, Any]:
    plan = _tiny_plan()
    selected = (FULL_LIFECYCLE, D_NEVER_SEEN_TWIN)
    first = run_tiny_generated_class_recurrence_replay(plan, arm_names=selected)
    second = run_tiny_generated_class_recurrence_replay(plan, arm_names=selected)
    return first, second


@pytest.mark.slow
def test_tiny_production_replay_is_bit_exact_and_every_step_authenticated(
    exact_replays: tuple[Any, Any],
) -> None:
    first, second = exact_replays

    assert first == second
    assert first.result_sha256 == second.result_sha256
    assert len(first.trials) == 2
    for trial in first.trials:
        assert trial.initial_persistent_state_nbytes == (
            trial.final_persistent_state_nbytes
        )
        assert len(trial.step_traces) == 9
        assert trial.total_identity_events_applied == 0
        assert not trial.lifecycle.lifecycle_complete
        assert trial.artifacts_written == 0
        assert not trial.evidence_authorized
        assert not trial.promotion_authorized
        for index, trace in enumerate(trial.step_traces):
            assert trace.step_index == index
            assert trace.learner_input_fields == ("raw_features", "target")
            assert "phase_label" not in trace.learner_input_fields
            assert "phase_boundary" not in trace.learner_input_fields
            assert trace.post_step_words_uint32 == (0, index + 1)
            assert trace.source_replay_authenticated
            assert not trace.curation_trace_has_event
            assert trace.accounting.logical_curation_event_count == 0
            assert trace.accounting.identity_events_applied == 0
            assert trace.accounting.persistent_state_nbytes_before == (
                trial.initial_persistent_state_nbytes
            )
            assert trace.accounting.persistent_state_nbytes_after == (
                trial.initial_persistent_state_nbytes
            )
            assert len(trace.ledger_transaction_sha256) == 64
            assert len(trace.occurrence_identity.ledger_state_sha256) == 64

    full = next(trial for trial in first.trials if trial.arm.name == FULL_LIFECYCLE)
    twin = next(trial for trial in first.trials if trial.arm.name == D_NEVER_SEEN_TWIN)
    assert full.lifecycle.target_d_exposure_count == 2
    assert not full.lifecycle.target_d_absent_before_second_d_by_evaluator_construction
    assert twin.lifecycle.target_d_exposure_count == 1
    assert twin.lifecycle.target_d_absent_before_second_d_by_evaluator_construction
    assert not twin.lifecycle.scrub_attempted
    assert not twin.lifecycle.scrub_committed
    assert "true D mapping intentionally withheld" in twin.lifecycle.incompleteness_reasons[0]


@pytest.mark.slow
def test_campaign_nested_tamper_and_malformed_trials_fail_closed(
    exact_replays: tuple[Any, Any],
) -> None:
    campaign, _ = exact_replays
    trial = campaign.trials[0]
    trace = trial.step_traces[0]
    forged_trace = dataclasses.replace(
        trace,
        prediction_before_update=trace.prediction_before_update + 1.0,
    )
    forged_trial = dataclasses.replace(
        trial,
        step_traces=(forged_trace, *trial.step_traces[1:]),
    )
    forged_campaign = dataclasses.replace(
        campaign,
        trials=(forged_trial, *campaign.trials[1:]),
    )
    with pytest.raises(ValueError, match="result hash"):
        validate_generated_class_development_campaign(forged_campaign)

    malformed = dataclasses.replace(
        campaign,
        trials=cast(Any, (object(),)),
    )
    with pytest.raises(TypeError, match="invalid type"):
        validate_generated_class_development_campaign(malformed)
