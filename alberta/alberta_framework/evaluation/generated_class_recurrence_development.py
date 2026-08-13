"""Executable, development-only generated-class recurrence instrumentation.

This module is the evaluator-owned vertical slice around the blocked
``generated_class_recurrence`` declaration.  It provides an uninterrupted
``A/B/A/D/A/C/A/D/A`` raw-feature stream, the five declared controls plus a
matched sham scrub and a D-never-presented twin, source-replay-authenticated
schema-v4 birth-identity sidecars, exact expanded-expression occurrence
snapshots, and raw prequential measurements.

The small runner is intentionally restricted to fewer than one production
curation interval.  It is useful for deterministic replay, pairing, label-
leakage, sidecar, and malformed-input tests; it cannot demonstrate a lifecycle.
The causal runner is separately fail-closed around authenticated external
scrub/epoch-rollover and generation-freeze adapters.  Neither runner writes an
artifact, applies a threshold, validates evidence, or promotes a claim.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import math
from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
)
from alberta_framework.evaluation import generated_birth_identity_freeze as freeze_module
from alberta_framework.evaluation import generated_birth_identity_scrub_epoch as scrub_epoch_module
from alberta_framework.evaluation.compositional_discovery_development import (
    ProductChainTrajectory,
    summarize_presence_history,
)
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    authenticate_generated_birth_identity_trace_by_source_replay,
)
from alberta_framework.evaluation.generated_class_d_mapping_twin import (
    build_d_mapping_never_seen_contract,
    build_d_mapping_twin_dataset,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    GeneratedClassScrubConfig,
    persistent_compositional_state_nbytes,
    scrub_compositional_feature_state,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    FINITE_DEGREE_TWO_ARCHIVE_CEILING,
    FROZEN_LIFECYCLE,
    FULL_LIFECYCLE,
    RANDOM_CURATION,
    ZERO_CANDIDATE_HEAD_CARRY,
    GeneratedClassRecurrenceV0Protocol,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_controls,
    build_generated_class_v0_learner,
    count_expression_occurrences,
    derive_expression_manifest,
    evaluate_expression,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    ExpandedExpressionLineagePlan,
    compile_expanded_expression_lineage_masks,
)
from alberta_framework.evaluation.generated_reacquisition_epoch import (
    GeneratedReacquisitionEpochConfig,
)

GENERATED_CLASS_RECURRENCE_DEVELOPMENT_SCHEMA: Final = (
    "alberta.generated-class-recurrence.development.v1"
)
GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS: Final = (
    "DEVELOPMENT_ONLY_DESCRIPTIVE_NO_EVIDENCE_OR_PROMOTION"
)
THREEFRY_IMPLEMENTATION: Final = "threefry2x32"

MATCHED_SHAM_SCRUB: Final = "matched_sham_scrub"
D_NEVER_SEEN_TWIN: Final = "d_never_seen_twin"

DECLARED_ARM_ORDER: Final = (
    FULL_LIFECYCLE,
    RANDOM_CURATION,
    FROZEN_LIFECYCLE,
    ZERO_CANDIDATE_HEAD_CARRY,
    FINITE_DEGREE_TWO_ARCHIVE_CEILING,
    MATCHED_SHAM_SCRUB,
    D_NEVER_SEEN_TWIN,
)

# The domains are stable development pairing domains, not evidence seeds.
STREAM_DOMAIN: Final = 0x47525354  # GRST
OBSERVATION_DOMAIN: Final = 0x4F425356  # OBSV
LEARNER_DOMAIN: Final = 0x4C524E52  # LRNR
# Shared with the fixed fresh-epoch development-life manifest.
DEFAULT_DEVELOPMENT_SEEDS: Final = (101,)
MAX_TINY_NONCURATING_STEPS: Final = 31
TRACE_LEDGER_NAMESPACE_PREFIX: Final = "generated-class-recurrence-development-v1"
GENERATED_CLASS_PAIRED_FREEZE_SCHEMA: Final = (
    "alberta.generated-class-paired-scrub-freeze.development.v0"
)
GENERATED_CLASS_PAIRED_FREEZE_STATUS: Final = (
    "DEVELOPMENT_AUTHENTICATED_PAIRED_CAUSAL_SHAM_NO_EVIDENCE"
)
REALISTIC_SCRUB_BOUNDARY_STEP: Final = 17
GENERATION_FREEZE_UPDATES: Final = 32
REALISTIC_FREEZE_ENDPOINT_STEP: Final = 49
_PAIRED_OPERATION_ACCOUNTING_SCOPE: Final = (
    "shared authenticated genesis prefix plus the stable paired receipt's exact "
    "causal/sham direct updates, source replays, due replays, endpoint "
    "revalidations, and matched sham scrub calls; no wall-clock equivalence"
)

ScrubMode = Literal["causal", "sham", "not_applicable"]


class GeneratedClassRecurrenceDevelopmentError(RuntimeError):
    """Fail-closed construction or execution error for this development lane."""


class GeneratedClassRecurrenceAdapterUnavailableError(
    GeneratedClassRecurrenceDevelopmentError
):
    """Raised before any intervention when a required authenticated adapter is absent."""


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassPairedFreezeAccounting:
    """Exact logical calls from stable receipts; measured runtime is out of scope."""

    genesis_prefix_direct_update_calls: int
    genesis_prefix_source_replay_calls: int
    causal_total_learner_update_calls: int
    sham_total_learner_update_calls: int
    matched_sham_scrub_kernel_calls: int
    exact_learner_work_parity: bool
    measured_runtime_sample_count: int
    measured_runtime_parity_claimed: bool
    wall_clock_threshold: float | None
    operation_accounting_scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassPairedFreezeAudit:
    """Compact binding of the complete stable causal/sham transaction graph."""

    schema: str
    status: str
    genesis_step: int
    scrub_boundary_step: int
    phase_derived_due_pre_step: int
    phase_derived_due_post_step: int
    freeze_endpoint_step: int
    exogenous_step_count: int
    exogenous_input_manifest_sha256: str
    genesis_prefix_transaction_sha256: tuple[str, ...]
    shared_core_source_sha256: str
    shared_ledger_source_sha256: str
    shared_learner_config_sha256: str
    shared_ledger_config_sha256: str
    genesis_key_words_uint32: tuple[int, int]
    scrub_boundary_key_words_uint32: tuple[int, int]
    due_pre_key_words_uint32: tuple[int, int]
    freeze_end_pre_fresh_key_words_uint32: tuple[int, int]
    fresh_key_words_uint32: tuple[int, int]
    scrub_rollover_transaction_sha256: str
    matched_sham_start_sha256: str
    causal_endpoint_transaction_sha256: str
    sham_endpoint_transaction_sha256: str
    paired_transaction_sha256: str
    causal_scrub_committed: bool
    matched_sham_scrub_executed_noncommitting: bool
    attempted_due_branches_authenticated_and_abandoned: bool
    shadow_due_branches_authenticated_and_carried: bool
    exact_crn_input_parity: bool
    typed_key_checkpoints_bound: bool
    fresh_key_applied_only_at_endpoint: bool
    scrubbed_candidate_heads_zero_at_identity_birth: bool
    sham_endpoint_state_discarded: bool
    causal_output_only: bool
    causal_output_core_state_sha256: str
    causal_output_ledger_state_sha256: str
    accounting: GeneratedClassPairedFreezeAccounting
    receipt_sha256: str
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassPairedFreezeReceipt:
    """Complete replay inputs/receipts needed for adversarial revalidation."""

    genesis_core_state: CompositionalFeatureState
    genesis_ledger_state: GeneratedBirthIdentityLedgerV4State
    genesis_prefix_steps: tuple[freeze_module.GeneratedBirthIdentityFreezeOrdinaryStep, ...]
    scrub_inputs: scrub_epoch_module.GeneratedBirthIdentityScrubEpochInputs
    scrub_transaction: scrub_epoch_module.GeneratedBirthIdentityScrubEpochTransaction
    matched_sham_start: freeze_module.GeneratedBirthIdentityMatchedShamStart
    causal_endpoint_inputs: freeze_module.GeneratedBirthIdentityFreezeEndpointInputs
    sham_endpoint_inputs: freeze_module.GeneratedBirthIdentityFreezeEndpointInputs
    paired_transaction: freeze_module.GeneratedBirthIdentityPairedFreezeTransaction
    paired_validation: freeze_module.GeneratedBirthIdentityPairedFreezeValidation
    audit: GeneratedClassPairedFreezeAudit


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassPairedFreezeExecution:
    """Only the causal fresh-key endpoint is carried; the sham remains in the receipt."""

    core_state: CompositionalFeatureState
    ledger_state: GeneratedBirthIdentityLedgerV4State
    receipt: GeneratedClassPairedFreezeReceipt


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassDevelopmentArm:
    """One capacity-matched arm declaration and its evaluator intervention."""

    name: str
    learner_control_name: str
    intervention: str
    scrub_mode: ScrubMode
    target_d_presented: bool
    first_d_true_mapping_presented: bool
    candidate_identity_refresh_head_zero: bool
    allocated_active_slots: int
    allocated_candidate_slots: int
    allocated_max_depth: int
    persistent_jax_state_nbytes: int
    learner_updates_per_step: int
    active_feature_evaluations_per_step: int
    candidate_feature_evaluations_per_step: int
    allocated_curation_decision_slots_per_step: int
    development_only: bool = True
    execution_authorized: bool = False
    evidence_authorized: bool = False
    scientific_promotion_allowed: bool = False


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassDevelopmentWorkAccounting:
    """Declared logical work; this is not a FLOP, latency, or peak-RAM claim."""

    paired_seed_count: int
    arm_count: int
    steps_per_arm: int
    total_learner_updates: int
    learner_updates_per_arm: int
    active_slot_exposures_per_arm: int
    candidate_slot_exposures_per_arm: int
    curation_decision_slots_per_arm: int
    trace_authentication_attempts_per_arm: int
    stream_float32_values_per_seed: int
    artifact_bytes_written: int
    wall_clock_threshold: float | None
    equal_persistent_capacity_declared: bool
    equal_logical_work_declared: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassRecurrenceDevelopmentPlan:
    """Inert declaration of either the canonical or an explicitly tiny life."""

    schema: str
    status: str
    development_only: bool
    scientific_evidence_authorized: bool
    promotion_authorized: bool
    artifact_writes_authorized: bool
    thresholds_authorized: bool
    protocol_schema: str
    expression_manifest_sha256: str
    canonical_phase_length_manifest_sha256: str
    phase_order: tuple[str, ...]
    phase_lengths: tuple[int, ...]
    phase_starts: tuple[int, ...]
    total_steps: int
    canonical_full_life: bool
    tiny_noncurating_replay: bool
    replacement_interval: int
    input_dim: int
    n_tasks: int
    learner_observation_fields: tuple[str, ...]
    evaluator_only_fields: tuple[str, ...]
    seeds: tuple[int, ...]
    arms: tuple[GeneratedClassDevelopmentArm, ...]
    work: GeneratedClassDevelopmentWorkAccounting


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassStreamStep:
    """Evaluator record; only ``raw_features`` and ``target`` reach the learner."""

    step_index: int
    phase_index: int
    phase_label: str
    phase_boundary: bool
    presented_target_name: str
    raw_features: tuple[float, ...]
    target: float


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassDevelopmentStream:
    """One exact paired stream for one seed/arm target-presentation rule."""

    root_seed_uint32: int
    arm_name: str
    observation_key_words_uint32: tuple[int, int]
    steps: tuple[GeneratedClassStreamStep, ...]
    stream_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassOccurrenceIdentitySnapshot:
    """Exact D occurrence/expanded-lineage masks bound to v4 birth identities."""

    active_exact_root_occurrences: int
    candidate_exact_root_occurrences: int
    expanded_target_present: bool
    expanded_active_mask: tuple[bool, ...]
    expanded_candidate_mask: tuple[bool, ...]
    expanded_active_birth_identities: tuple[tuple[int, str], ...]
    expanded_candidate_birth_identities: tuple[tuple[int, str], ...]
    active_lineage_has_nonzero_output_head: bool
    ledger_state_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassStepAccounting:
    """Raw production trace counts and concrete persistent-write accounting."""

    proposal_count: int
    root_change_count: int
    promotion_count: int
    cascade_refill_count: int
    candidate_refresh_count: int
    candidate_rebound_count: int
    candidate_overdepth_regeneration_count: int
    logical_curation_event_count: int
    identity_events_applied: int
    changed_persistent_array_leaf_count: int
    changed_persistent_array_bytes: int
    persistent_state_nbytes_before: int
    persistent_state_nbytes_after: int
    candidate_identity_refresh_heads_zero: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassStepTrace:
    """One raw prequential prediction followed by one authenticated online update."""

    step_index: int
    phase_index: int
    phase_label: str
    phase_boundary: bool
    presented_target_name: str
    learner_input_fields: tuple[str, ...]
    raw_features: tuple[float, ...]
    target: float
    prediction_before_update: float
    error_before_update: float
    prequential_squared_loss: float
    production_metrics: tuple[float, ...]
    post_step_words_uint32: tuple[int, int]
    curation_trace_has_event: bool
    source_replay_authenticated: bool
    trace_binding_schema: str
    ledger_transaction_sha256: str
    occurrence_identity: GeneratedClassOccurrenceIdentitySnapshot
    accounting: GeneratedClassStepAccounting


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassLifecycleTrace:
    """Threshold-free natural event indices and structural presence summaries."""

    target_d_exposure_count: int
    target_d_absent_before_second_d_by_evaluator_construction: bool
    d_initially_absent: bool
    d_first_exact_presence_step: int | None
    d_first_use_step: int | None
    d_exact_birth_steps: tuple[int, ...]
    d_exact_retirement_steps: tuple[int, ...]
    d_exact_reacquisition_steps: tuple[int, ...]
    active_presence: ProductChainTrajectory
    candidate_presence: ProductChainTrajectory
    scrub_attempted: bool
    scrub_committed: bool
    scrub_source_replay_boundary_authenticated: bool
    reacquisition_epoch_key_applied: bool
    generation_frozen_updates_executed: int
    lifecycle_complete: bool
    incompleteness_reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassDevelopmentTrial:
    """One in-memory arm/seed replay with no acceptance interpretation."""

    root_seed_uint32: int
    arm: GeneratedClassDevelopmentArm
    stream_sha256: str
    initial_persistent_state_nbytes: int
    final_persistent_state_nbytes: int
    step_traces: tuple[GeneratedClassStepTrace, ...]
    lifecycle: GeneratedClassLifecycleTrace
    total_prequential_squared_loss: float
    total_identity_events_applied: int
    total_changed_persistent_array_bytes: int
    artifacts_written: int
    evidence_authorized: bool
    promotion_authorized: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassDevelopmentCampaign:
    """Exact replay result for an explicit in-memory development run."""

    plan: GeneratedClassRecurrenceDevelopmentPlan
    trials: tuple[GeneratedClassDevelopmentTrial, ...]
    result_sha256: str
    artifacts_written: int
    evidence_authorized: bool
    promotion_authorized: bool


@dataclasses.dataclass(frozen=True, slots=True)
class _TraceLedgerAdapter:
    config_type: type[Any]
    attach: Callable[..., Any]
    authenticate: Callable[..., Any]


def _exact_positive_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_seed(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python integer")
    if not 0 <= value <= np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit uint32")
    return value


def _phase_starts(lengths: tuple[int, ...]) -> tuple[int, ...]:
    starts: list[int] = []
    cursor = 0
    for length in lengths:
        starts.append(cursor)
        cursor += length
    return tuple(starts)


def _arm_declarations(
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> tuple[GeneratedClassDevelopmentArm, ...]:
    controls = {control.name: control for control in build_generated_class_v0_controls(protocol)}
    declarations: list[GeneratedClassDevelopmentArm] = []
    for name in DECLARED_ARM_ORDER:
        base_name = (
            FULL_LIFECYCLE if name in {MATCHED_SHAM_SCRUB, D_NEVER_SEEN_TWIN} else name
        )
        control = controls[base_name]
        if name == MATCHED_SHAM_SCRUB:
            intervention = (
                "compile_and_validate_identical_expanded_lineage_scrub_then_commit_false"
            )
            scrub_mode: ScrubMode = "sham"
        elif name == D_NEVER_SEEN_TWIN:
            intervention = (
                "derange_first_D_mapping_without_learner_label_then_present_true_second_D"
            )
            scrub_mode = "not_applicable"
        else:
            intervention = control.intervention
            scrub_mode = "causal"
        declarations.append(
            GeneratedClassDevelopmentArm(
                name=name,
                learner_control_name=base_name,
                intervention=intervention,
                scrub_mode=scrub_mode,
                target_d_presented=True,
                first_d_true_mapping_presented=name != D_NEVER_SEEN_TWIN,
                candidate_identity_refresh_head_zero=(
                    name == ZERO_CANDIDATE_HEAD_CARRY
                ),
                allocated_active_slots=control.resource_contract.active_slots,
                allocated_candidate_slots=control.resource_contract.candidate_slots,
                allocated_max_depth=control.allocated_max_depth,
                persistent_jax_state_nbytes=control.resource_contract.jax_state_nbytes,
                learner_updates_per_step=control.operation_contract.learner_updates_per_step,
                active_feature_evaluations_per_step=(
                    control.operation_contract.active_feature_evaluations_per_step
                ),
                candidate_feature_evaluations_per_step=(
                    control.operation_contract.candidate_feature_evaluations_per_step
                ),
                allocated_curation_decision_slots_per_step=(
                    control.operation_contract.allocated_curation_decision_slots_per_step
                ),
            )
        )
    return tuple(declarations)


def build_generated_class_recurrence_development_plan(
    *,
    phase_lengths: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS,
) -> GeneratedClassRecurrenceDevelopmentPlan:
    """Build an inert canonical life or an explicitly small noncanonical replay."""

    protocol = build_generated_class_recurrence_v0_protocol()
    canonical = phase_lengths is None
    selected_lengths = protocol.phase_lengths if canonical else phase_lengths
    if type(selected_lengths) is not tuple:
        raise TypeError("phase_lengths must be an exact tuple")
    if len(selected_lengths) != len(protocol.phase_order):
        raise ValueError("phase_lengths must have one entry for every canonical phase")
    checked_lengths = tuple(
        _exact_positive_int(length, name=f"phase_lengths[{index}]")
        for index, length in enumerate(selected_lengths)
    )
    if type(seeds) is not tuple or not seeds:
        raise TypeError("seeds must be a non-empty exact tuple")
    checked_seeds = tuple(
        _validate_seed(seed, name=f"seeds[{index}]") for index, seed in enumerate(seeds)
    )
    if len(set(checked_seeds)) != len(checked_seeds):
        raise ValueError("development seeds must be unique")
    total_steps = sum(checked_lengths)
    tiny = not canonical and total_steps <= MAX_TINY_NONCURATING_STEPS
    arms = _arm_declarations(protocol)
    work = GeneratedClassDevelopmentWorkAccounting(
        paired_seed_count=len(checked_seeds),
        arm_count=len(arms),
        steps_per_arm=total_steps,
        total_learner_updates=len(checked_seeds) * len(arms) * total_steps,
        learner_updates_per_arm=total_steps,
        active_slot_exposures_per_arm=total_steps * protocol.active_slots,
        candidate_slot_exposures_per_arm=total_steps * protocol.candidate_slots,
        curation_decision_slots_per_arm=total_steps,
        trace_authentication_attempts_per_arm=total_steps,
        stream_float32_values_per_seed=total_steps * (protocol.input_dim + 1),
        artifact_bytes_written=0,
        wall_clock_threshold=None,
        equal_persistent_capacity_declared=(
            len({arm.persistent_jax_state_nbytes for arm in arms}) == 1
        ),
        equal_logical_work_declared=all(
            arm.learner_updates_per_step == 1
            and arm.active_feature_evaluations_per_step == protocol.active_slots
            and arm.candidate_feature_evaluations_per_step == protocol.candidate_slots
            and arm.allocated_curation_decision_slots_per_step == 1
            for arm in arms
        ),
    )
    plan = GeneratedClassRecurrenceDevelopmentPlan(
        schema=GENERATED_CLASS_RECURRENCE_DEVELOPMENT_SCHEMA,
        status=GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS,
        development_only=True,
        scientific_evidence_authorized=False,
        promotion_authorized=False,
        artifact_writes_authorized=False,
        thresholds_authorized=False,
        protocol_schema=protocol.schema,
        expression_manifest_sha256=protocol.expression_manifest_sha256,
        canonical_phase_length_manifest_sha256=(
            protocol.phase_length_manifest_sha256
        ),
        phase_order=protocol.phase_order,
        phase_lengths=checked_lengths,
        phase_starts=_phase_starts(checked_lengths),
        total_steps=total_steps,
        canonical_full_life=canonical,
        tiny_noncurating_replay=tiny,
        replacement_interval=protocol.curation_opportunity_audit.curation_interval,
        input_dim=protocol.input_dim,
        n_tasks=protocol.n_tasks,
        learner_observation_fields=protocol.learner_observation_fields,
        evaluator_only_fields=(
            "phase_index",
            "phase_label",
            "phase_boundary",
            "presented_target_name",
        ),
        seeds=checked_seeds,
        arms=arms,
        work=work,
    )
    return validate_generated_class_recurrence_development_plan(plan)


def validate_generated_class_recurrence_development_plan(
    plan: GeneratedClassRecurrenceDevelopmentPlan,
) -> GeneratedClassRecurrenceDevelopmentPlan:
    """Reject mutated authority, pairing, shape, or work declarations."""

    if type(plan) is not GeneratedClassRecurrenceDevelopmentPlan:
        raise TypeError("plan must be an exact GeneratedClassRecurrenceDevelopmentPlan")
    protocol = build_generated_class_recurrence_v0_protocol()
    if plan.schema != GENERATED_CLASS_RECURRENCE_DEVELOPMENT_SCHEMA:
        raise ValueError("development plan schema is not canonical")
    if plan.status != GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS:
        raise ValueError("development plan status is not canonical")
    if not plan.development_only or any(
        (
            plan.scientific_evidence_authorized,
            plan.promotion_authorized,
            plan.artifact_writes_authorized,
            plan.thresholds_authorized,
        )
    ):
        raise ValueError("development plan cannot grant evidence or external authority")
    if plan.protocol_schema != protocol.schema:
        raise ValueError("recurrence protocol schema drifted")
    if plan.expression_manifest_sha256 != protocol.expression_manifest_sha256:
        raise ValueError("expression manifest binding drifted")
    if plan.phase_order != protocol.phase_order or plan.phase_starts != _phase_starts(
        plan.phase_lengths
    ):
        raise ValueError("phase schedule is malformed")
    if plan.total_steps != sum(plan.phase_lengths):
        raise ValueError("total_steps does not match phase lengths")
    if plan.canonical_full_life != (plan.phase_lengths == protocol.phase_lengths):
        raise ValueError("canonical-life disclosure is false")
    expected_tiny = (
        not plan.canonical_full_life
        and plan.total_steps <= MAX_TINY_NONCURATING_STEPS
    )
    if plan.tiny_noncurating_replay != expected_tiny:
        raise ValueError("tiny noncurating disclosure is false")
    if plan.learner_observation_fields != ("raw_features",):
        raise ValueError("learner input contract must contain raw features only")
    if set(plan.learner_observation_fields) & set(plan.evaluator_only_fields):
        raise ValueError("evaluator metadata leaked into learner inputs")
    if tuple(arm.name for arm in plan.arms) != DECLARED_ARM_ORDER:
        raise ValueError("the seven-arm declaration is incomplete or reordered")
    if plan.arms != _arm_declarations(protocol):
        raise ValueError("one or more seven-arm intervention declarations were mutated")
    if not plan.work.equal_persistent_capacity_declared:
        raise ValueError("all arms must declare equal persistent capacity")
    if not plan.work.equal_logical_work_declared:
        raise ValueError("all arms must declare equal logical work")
    expected_updates = len(plan.seeds) * len(plan.arms) * plan.total_steps
    if plan.work.total_learner_updates != expected_updates:
        raise ValueError("declared total learner updates are stale")
    if plan.work.artifact_bytes_written != 0 or plan.work.wall_clock_threshold is not None:
        raise ValueError("development plan cannot declare writes or latency thresholds")
    return plan


def _key_words(key: Array) -> tuple[int, int]:
    if not jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):  # type: ignore[attr-defined]
        raise TypeError("key must be a typed JAX key")
    words = np.asarray(jr.key_data(key), dtype=np.uint32).reshape(-1)
    if words.shape != (2,):
        raise ValueError("Threefry key must contain exactly two uint32 words")
    return int(words[0]), int(words[1])


def _float32_array_record(value: Array, *, shape: tuple[int, ...], name: str) -> dict[str, object]:
    if not isinstance(value, Array):
        raise TypeError(f"{name} must be a JAX array")
    if value.dtype != jnp.float32 or value.shape != shape:
        raise TypeError(f"{name} must have exact float32 shape {shape}")
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return {
        "dtype": "float32",
        "shape": list(shape),
        "raw_hex": array.tobytes(order="C").hex(),
    }


def _paired_exogenous_manifest_sha256(
    observations: tuple[Array, ...],
    targets: tuple[Array, ...],
    *,
    input_dim: int,
    context_id: int,
) -> str:
    if type(observations) is not tuple or type(targets) is not tuple:
        raise TypeError("paired observations and targets must be exact tuples")
    if len(observations) != len(targets):
        raise ValueError("paired observations and targets must have equal length")
    payload = {
        "context_id": context_id,
        "rows": [
            {
                "step": index,
                "observation": _float32_array_record(
                    observation,
                    shape=(input_dim,),
                    name=f"observations[{index}]",
                ),
                "targets": _float32_array_record(
                    target,
                    shape=(1,),
                    name=f"targets[{index}]",
                ),
            }
            for index, (observation, target) in enumerate(
                zip(observations, targets, strict=True)
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _paired_audit_sha256(audit: GeneratedClassPairedFreezeAudit) -> str:
    payload = dataclasses.asdict(audit)
    payload.pop("receipt_sha256")
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _stream_payload(steps: Sequence[GeneratedClassStreamStep]) -> list[dict[str, object]]:
    return [
        {
            "step_index": step.step_index,
            "phase_index": step.phase_index,
            "phase_label": step.phase_label,
            "phase_boundary": step.phase_boundary,
            "presented_target_name": step.presented_target_name,
            "raw_features_float32_hex": [
                np.float32(value).tobytes().hex() for value in step.raw_features
            ],
            "target_float32_hex": np.float32(step.target).tobytes().hex(),
        }
        for step in steps
    ]


def build_generated_class_development_stream(
    plan: GeneratedClassRecurrenceDevelopmentPlan,
    root_seed_uint32: int,
    arm_name: str,
) -> GeneratedClassDevelopmentStream:
    """Derive one arm-independent observation stream and evaluator-only targets."""

    checked = validate_generated_class_recurrence_development_plan(plan)
    seed = _validate_seed(root_seed_uint32, name="root_seed_uint32")
    if type(arm_name) is not str:
        raise TypeError("arm_name must be an exact string")
    if arm_name not in DECLARED_ARM_ORDER:
        raise ValueError("unknown generated-class development arm")
    root = jr.key(seed, impl=THREEFRY_IMPLEMENTATION)
    stream_root = jr.fold_in(root, np.uint32(STREAM_DOMAIN))
    observation_key = jr.fold_in(stream_root, np.uint32(OBSERVATION_DOMAIN))
    canonical_reference_targets: Array | None = None
    canonical_twin_targets: Array | None = None
    if checked.canonical_full_life:
        twin_contract = build_d_mapping_never_seen_contract()
        dataset = build_d_mapping_twin_dataset(twin_contract)
        observations = dataset.observations
        canonical_reference_targets = dataset.reference_targets
        canonical_twin_targets = dataset.twin_targets
        observation_key_words = twin_contract.rng_contract.observation_key_data
    else:
        observations = jr.normal(
            observation_key,
            (checked.total_steps, checked.input_dim),
            dtype=jnp.float32,
        )
        observation_key_words = _key_words(observation_key)
    observations.block_until_ready()
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    targets = {target.name: target.expression for target in manifest.targets}
    steps: list[GeneratedClassStreamStep] = []
    cursor = 0
    for phase_index, (phase_label, length) in enumerate(
        zip(checked.phase_order, checked.phase_lengths, strict=True)
    ):
        for offset in range(length):
            observation = observations[cursor]
            is_withheld_first_d = (
                arm_name == D_NEVER_SEEN_TWIN
                and phase_label == "D"
                and phase_index == 3
            )
            presented_name = "D_mapping_deranged" if is_withheld_first_d else phase_label
            if canonical_reference_targets is not None:
                selected_targets = (
                    canonical_twin_targets
                    if arm_name == D_NEVER_SEEN_TWIN
                    else canonical_reference_targets
                )
                if selected_targets is None:
                    raise GeneratedClassRecurrenceDevelopmentError(
                        "canonical D-mapping twin targets are unavailable"
                    )
                target = selected_targets[cursor]
            elif is_withheld_first_d:
                # A one-row tiny D block cannot be deranged while preserving its
                # value multiset.  Substitute the preregistered A expression only
                # in this explicitly noncanonical diagnostics path.
                target = evaluate_expression(targets["A"], observation)
            else:
                target = evaluate_expression(targets[phase_label], observation)
            raw = tuple(float(value) for value in np.asarray(observation, dtype=np.float32))
            steps.append(
                GeneratedClassStreamStep(
                    step_index=cursor,
                    phase_index=phase_index,
                    phase_label=phase_label,
                    phase_boundary=offset == 0,
                    presented_target_name=presented_name,
                    raw_features=raw,
                    target=float(np.float32(target)),
                )
            )
            cursor += 1
    payload = {
        "root_seed_uint32": seed,
        "arm_name": arm_name,
        "observation_key_words_uint32": list(observation_key_words),
        "steps": _stream_payload(steps),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return GeneratedClassDevelopmentStream(
        root_seed_uint32=seed,
        arm_name=arm_name,
        observation_key_words_uint32=observation_key_words,
        steps=tuple(steps),
        stream_sha256=digest,
    )


def _load_trace_ledger_adapter() -> _TraceLedgerAdapter:
    """Resolve the source-pinned adapter at execution time, never at import time."""

    try:
        ledger = importlib.import_module(
            "alberta_framework.evaluation.generated_birth_identity_ledger"
        )
        binding = importlib.import_module(
            "alberta_framework.evaluation.generated_birth_identity_trace_binding"
        )
        config_type = getattr(ledger, "GeneratedBirthIdentityLedgerV4Config")
        attach = getattr(
            binding,
            "attach_generated_birth_identity_ledger_at_core_genesis",
        )
        authenticate = getattr(
            binding,
            "authenticate_generated_birth_identity_trace_by_source_replay",
        )
    except (AttributeError, ImportError) as exc:
        raise GeneratedClassRecurrenceAdapterUnavailableError(
            "schema-v4 source-replay trace binding is unavailable"
        ) from exc
    if not isinstance(config_type, type) or not callable(attach) or not callable(
        authenticate
    ):
        raise GeneratedClassRecurrenceAdapterUnavailableError(
            "schema-v4 source-replay trace binding has malformed public entry points"
        )
    return _TraceLedgerAdapter(
        config_type=config_type,
        attach=cast(Callable[..., Any], attach),
        authenticate=cast(Callable[..., Any], authenticate),
    )


def _build_arm_learner(
    arm: GeneratedClassDevelopmentArm,
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> CompositionalFeatureLearner:
    learner = build_generated_class_v0_learner(arm.learner_control_name, protocol)
    if arm.name == ZERO_CANDIDATE_HEAD_CARRY:
        config = learner.to_config()
        config["candidate_imprint_scale"] = 0.0
        learner = CompositionalFeatureLearner.from_config(config)
    if type(learner) is not CompositionalFeatureLearner:
        raise GeneratedClassRecurrenceDevelopmentError(
            "development arms must use the exact production learner class"
        )
    return learner


def _lineage_config(
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> ExpandedExpressionLineageConfig:
    return ExpandedExpressionLineageConfig(
        feature_dim=protocol.input_dim,
        active_slots=protocol.active_slots,
        candidate_slots=protocol.candidate_slots,
        n_tasks=protocol.n_tasks,
        generator_contexts=protocol.resource_contract.generator_resource_contexts,
        generator_policy_count=protocol.resource_contract.generator_policy_count,
    )


def _identity_hex_rows(value: object, *, expected_rows: int) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.dtype != np.uint8:
        raise GeneratedClassRecurrenceDevelopmentError(
            "birth-identity bank must be a uint8 matrix with one row per slot"
        )
    return tuple(np.ascontiguousarray(row).tobytes().hex() for row in array)


def _occurrence_identity_snapshot(
    state: CompositionalFeatureState,
    ledger_state: Any,
    target_d: Any,
    *,
    lineage_config: ExpandedExpressionLineageConfig,
) -> tuple[GeneratedClassOccurrenceIdentitySnapshot, ExpandedExpressionLineagePlan]:
    active_count, candidate_count = count_expression_occurrences(state, target_d)
    plan = compile_expanded_expression_lineage_masks(
        state,
        target_d,
        config=lineage_config,
    )
    active_mask = tuple(bool(value) for value in np.asarray(plan.active_mask, dtype=np.bool_))
    candidate_mask = tuple(
        bool(value) for value in np.asarray(plan.candidate_mask, dtype=np.bool_)
    )
    active_ids = _identity_hex_rows(
        ledger_state.active_identity,
        expected_rows=lineage_config.active_slots,
    )
    candidate_ids = _identity_hex_rows(
        ledger_state.candidate_identity,
        expected_rows=lineage_config.candidate_slots,
    )
    selected_active_ids = tuple(
        (slot, active_ids[slot]) for slot, selected in enumerate(active_mask) if selected
    )
    selected_candidate_ids = tuple(
        (slot, candidate_ids[slot])
        for slot, selected in enumerate(candidate_mask)
        if selected
    )
    active_weights = np.asarray(state.output_weights, dtype=np.float32)
    if active_weights.shape != (lineage_config.n_tasks, lineage_config.active_slots):
        raise GeneratedClassRecurrenceDevelopmentError(
            "active output-head shape drifted from the lineage contract"
        )
    mask_array = np.asarray(active_mask, dtype=np.bool_)
    nonzero_head = bool(
        np.any(active_weights[:, mask_array].view(np.uint32) != np.uint32(0))
    )
    ledger_sha = getattr(ledger_state, "integrity_sha256", None)
    if type(ledger_sha) is not str or len(ledger_sha) != 64:
        raise GeneratedClassRecurrenceDevelopmentError(
            "schema-v4 ledger state has a malformed integrity hash"
        )
    return (
        GeneratedClassOccurrenceIdentitySnapshot(
            active_exact_root_occurrences=active_count,
            candidate_exact_root_occurrences=candidate_count,
            expanded_target_present=plan.audit.pre_target_present,
            expanded_active_mask=active_mask,
            expanded_candidate_mask=candidate_mask,
            expanded_active_birth_identities=selected_active_ids,
            expanded_candidate_birth_identities=selected_candidate_ids,
            active_lineage_has_nonzero_output_head=nonzero_head,
            ledger_state_sha256=ledger_sha,
        ),
        plan,
    )


def _array_bits(value: Array) -> bytes:
    if jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):  # type: ignore[attr-defined]
        implementation = str(jr.key_impl(value)).encode("ascii")
        words = np.asarray(jr.key_data(value), dtype=np.uint32)
        return implementation + b"\x00" + np.ascontiguousarray(words).tobytes()
    return np.ascontiguousarray(np.asarray(value)).tobytes()


def _persistent_write_accounting(
    before: CompositionalFeatureState,
    after: CompositionalFeatureState,
) -> tuple[int, int]:
    before_with_paths = jax.tree_util.tree_flatten_with_path(before)[0]
    after_with_paths = jax.tree_util.tree_flatten_with_path(after)[0]
    before_leaves = [leaf for _, leaf in before_with_paths]
    after_leaves = [leaf for _, leaf in after_with_paths]
    if len(before_leaves) != len(after_leaves):
        raise GeneratedClassRecurrenceDevelopmentError("learner state tree shape changed")
    changed_count = 0
    changed_bytes = 0
    for (path, left), (right_path, right) in zip(
        before_with_paths,
        after_with_paths,
        strict=True,
    ):
        path_names = tuple(getattr(key, "name", None) for key in path)
        right_path_names = tuple(getattr(key, "name", None) for key in right_path)
        if path_names != right_path_names:
            raise GeneratedClassRecurrenceDevelopmentError("learner state paths changed")
        if path_names in {("birth_timestamp",), ("uptime_s",)}:
            continue
        if isinstance(left, Array) and isinstance(right, Array):
            if left.shape != right.shape or left.dtype != right.dtype:
                raise GeneratedClassRecurrenceDevelopmentError(
                    "learner persistent leaf shape or dtype changed"
                )
            if _array_bits(left) != _array_bits(right):
                changed_count += 1
                changed_bytes += int(right.nbytes)
        else:
            raise GeneratedClassRecurrenceDevelopmentError(
                "learner state contains an unsupported leaf type"
            )
    return changed_count, changed_bytes


def _scalar_int(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise GeneratedClassRecurrenceDevelopmentError(f"{name} must be an integer scalar")
    return int(array)


def _scalar_bool(value: object, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise GeneratedClassRecurrenceDevelopmentError(f"{name} must be a bool scalar")
    return bool(array)


def _authenticated_ordinary_steps(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    start_core_state: CompositionalFeatureState,
    start_ledger_state: GeneratedBirthIdentityLedgerV4State,
    observations: tuple[Array, ...],
    targets: tuple[Array, ...],
    *,
    context_id: int,
) -> tuple[
    tuple[freeze_module.GeneratedBirthIdentityFreezeOrdinaryStep, ...],
    CompositionalFeatureState,
    GeneratedBirthIdentityLedgerV4State,
]:
    """Execute one exact no-curation chain and retain every replay receipt."""

    if len(observations) != len(targets):
        raise ValueError("ordinary-chain observation/target lengths differ")
    core_state = start_core_state
    ledger_state = start_ledger_state
    steps: list[freeze_module.GeneratedBirthIdentityFreezeOrdinaryStep] = []
    for index, (observation, target) in enumerate(
        zip(observations, targets, strict=True)
    ):
        _float32_array_record(
            observation,
            shape=(config.raw_feature_slots,),
            name=f"ordinary observations[{index}]",
        )
        _float32_array_record(
            target,
            shape=(1,),
            name=f"ordinary targets[{index}]",
        )
        result: CompositionalFeatureUpdateResult = learner.update(
            core_state,
            observation,
            target,
            context_id=context_id,
        )
        result.state.step_words.block_until_ready()
        binding = authenticate_generated_birth_identity_trace_by_source_replay(
            learner,
            config,
            ledger_state,
            learner_pre_state=core_state,
            learner_post_state=result.state,
            supplied_update_result=result,
            observation=observation,
            targets=target,
            context_id=context_id,
        )
        if not binding.source_replay_authenticated or not binding.ledger_validation.valid:
            raise GeneratedClassRecurrenceDevelopmentError(
                "ordinary freeze row failed source-replay or v4 validation"
            )
        if _scalar_bool(
            result.curation_trace.should_try_replace,
            name="ordinary trace.should_try_replace",
        ) or _scalar_bool(
            result.curation_trace.has_event,
            name="ordinary trace.has_event",
        ):
            raise GeneratedClassRecurrenceDevelopmentError(
                "ordinary freeze prefix/suffix unexpectedly crossed a curation event"
            )
        steps.append(
            freeze_module.GeneratedBirthIdentityFreezeOrdinaryStep(
                learner_pre_state=core_state,
                supplied_update_result=result,
                observation=observation,
                targets=target,
                binding=binding,
                context_id=context_id,
            )
        )
        core_state = result.state
        ledger_state = binding.transaction.post_state
    return tuple(steps), core_state, ledger_state


@dataclasses.dataclass(frozen=True, slots=True)
class _GeneratedClassFreezeArmBuild:
    endpoint: freeze_module.GeneratedBirthIdentityFreezeEndpointTransaction
    inputs: freeze_module.GeneratedBirthIdentityFreezeEndpointInputs


def _build_generated_class_freeze_arm(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    scrub_transaction: scrub_epoch_module.GeneratedBirthIdentityScrubEpochTransaction,
    scrub_inputs: scrub_epoch_module.GeneratedBirthIdentityScrubEpochInputs,
    start_core_state: CompositionalFeatureState,
    start_ledger_state: GeneratedBirthIdentityLedgerV4State,
    observations: tuple[Array, ...],
    targets: tuple[Array, ...],
    *,
    matched_sham_start: freeze_module.GeneratedBirthIdentityMatchedShamStart | None,
    context_id: int,
) -> _GeneratedClassFreezeArmBuild:
    """Execute prefix, phase-derived dual due, suffix, and endpoint for one arm."""

    replacement_interval = learner.to_config().get("replacement_interval")
    if type(replacement_interval) is not int or replacement_interval != 32:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired freeze requires the canonical replacement interval"
        )
    start_phase = _scalar_int(
        start_core_state.replacement_phase,
        name="freeze start replacement_phase",
    )
    prefix_count = replacement_interval - start_phase - 1
    if prefix_count < 0:
        raise GeneratedClassRecurrenceDevelopmentError(
            "freeze start replacement phase is outside the canonical interval"
        )
    if len(observations) != GENERATION_FREEZE_UPDATES or len(targets) != (
        GENERATION_FREEZE_UPDATES
    ):
        raise ValueError("each freeze arm requires exactly 32 exogenous rows")
    prefix, due_pre_core, due_pre_ledger = _authenticated_ordinary_steps(
        learner,
        config,
        start_core_state,
        start_ledger_state,
        observations[:prefix_count],
        targets[:prefix_count],
        context_id=context_id,
    )
    due_observation = observations[prefix_count]
    due_targets = targets[prefix_count]
    due_inputs = freeze_module.GeneratedBirthIdentityFreezeDueInputs(
        learner=learner,
        config=config,
        ledger_pre_state=due_pre_ledger,
        learner_pre_state=due_pre_core,
        observation=due_observation,
        targets=due_targets,
        scrub_rollover=scrub_transaction,
        scrub_inputs=scrub_inputs,
        prefix_steps=prefix,
        matched_sham_start=matched_sham_start,
        context_id=context_id,
    )
    due_transaction = freeze_module.build_generated_birth_identity_freeze_transaction(
        learner,
        config,
        due_pre_ledger,
        due_pre_core,
        due_observation,
        due_targets,
        scrub_transaction,
        scrub_inputs,
        prefix,
        matched_sham_start=matched_sham_start,
        context_id=context_id,
    )
    due_validation = freeze_module.validate_generated_birth_identity_freeze_transaction(
        due_transaction,
        learner=learner,
        config=config,
        ledger_pre_state=due_pre_ledger,
        learner_pre_state=due_pre_core,
        observation=due_observation,
        targets=due_targets,
        scrub_rollover=scrub_transaction,
        scrub_inputs=scrub_inputs,
        prefix_steps=prefix,
        matched_sham_start=matched_sham_start,
        context_id=context_id,
    )
    if not due_validation.valid:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired freeze due transaction failed independent validation"
        )
    suffix_observations = observations[prefix_count + 1 :]
    suffix_targets = targets[prefix_count + 1 :]
    if len(suffix_observations) != due_transaction.audit.suffix_update_count:
        raise GeneratedClassRecurrenceDevelopmentError(
            "phase-derived freeze suffix count differs from supplied inputs"
        )
    suffix, end_core, end_ledger = _authenticated_ordinary_steps(
        learner,
        config,
        due_transaction.committed_core_state,
        due_transaction.carried_ledger_state,
        suffix_observations,
        suffix_targets,
        context_id=context_id,
    )
    endpoint_inputs = freeze_module.GeneratedBirthIdentityFreezeEndpointInputs(
        due_transaction=due_transaction,
        due_inputs=due_inputs,
        freeze_end_core_state=end_core,
        freeze_end_ledger_state=end_ledger,
        suffix_steps=suffix,
    )
    endpoint = freeze_module.build_generated_birth_identity_freeze_endpoint_transaction(
        due_transaction,
        due_inputs,
        end_core,
        end_ledger,
        suffix,
    )
    endpoint_validation = (
        freeze_module.validate_generated_birth_identity_freeze_endpoint_transaction(
            endpoint,
            due_transaction=due_transaction,
            due_inputs=due_inputs,
            freeze_end_core_state=end_core,
            freeze_end_ledger_state=end_ledger,
            suffix_steps=suffix,
        )
    )
    if not endpoint_validation.valid:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired freeze endpoint failed independent validation"
        )
    return _GeneratedClassFreezeArmBuild(endpoint=endpoint, inputs=endpoint_inputs)


def run_authenticated_generated_class_paired_scrub_freeze(
    learner: CompositionalFeatureLearner,
    config: GeneratedBirthIdentityLedgerV4Config,
    genesis_core_state: CompositionalFeatureState,
    genesis_ledger_state: GeneratedBirthIdentityLedgerV4State,
    observations: tuple[Array, ...],
    targets: tuple[Array, ...],
    *,
    paired_life_seed: int = 101,
    scrub_boundary_step: int = REALISTIC_SCRUB_BOUNDARY_STEP,
    context_id: int = 0,
) -> GeneratedClassPairedFreezeExecution:
    """Execute one explicit genesis→scrub→paired-freeze development receipt.

    The input state must already contain a naturally obtained expanded D
    lineage at the requested boundary.  This function never injects D, chooses
    a curation slot, or changes a threshold.  It executes the causal scrub and
    matched noncommitting scrub, both authenticated phase-derived due paths,
    both suffixes, and both fresh-key endpoints.  Only the causal endpoint is
    returned as live state; the complete sham path remains in the receipt.
    """

    if type(learner) is not CompositionalFeatureLearner:
        raise TypeError("learner must be an exact CompositionalFeatureLearner")
    if type(config) is not GeneratedBirthIdentityLedgerV4Config:
        raise TypeError("config must be an exact GeneratedBirthIdentityLedgerV4Config")
    if type(genesis_core_state) is not CompositionalFeatureState:
        raise TypeError("genesis_core_state must be an exact CompositionalFeatureState")
    if type(genesis_ledger_state) is not GeneratedBirthIdentityLedgerV4State:
        raise TypeError(
            "genesis_ledger_state must be an exact GeneratedBirthIdentityLedgerV4State"
        )
    if type(scrub_boundary_step) is not int:
        raise TypeError("scrub_boundary_step must be an exact Python integer")
    if scrub_boundary_step != REALISTIC_SCRUB_BOUNDARY_STEP:
        raise ValueError("this realistic development receipt fixes the scrub boundary at 17")
    if type(context_id) is not int or context_id != 0:
        raise ValueError("generated-class paired freeze fixes the learner context at zero")
    seed = _validate_seed(paired_life_seed, name="paired_life_seed")
    if len(observations) != REALISTIC_FREEZE_ENDPOINT_STEP or len(targets) != (
        REALISTIC_FREEZE_ENDPOINT_STEP
    ):
        raise ValueError("realistic paired receipt requires exactly 49 exogenous rows")
    protocol = build_generated_class_recurrence_v0_protocol()
    expected_learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    if learner.to_config() != expected_learner.to_config():
        raise ValueError("paired freeze learner is not the canonical full-lifecycle config")
    expected_config_shape = (
        protocol.active_slots,
        protocol.candidate_slots,
        protocol.input_dim,
        protocol.allocated_max_depth,
        False,
    )
    supplied_config_shape = (
        config.active_slots,
        config.candidate_slots,
        config.raw_feature_slots,
        config.max_depth,
        config.learn_generator_resources,
    )
    if supplied_config_shape != expected_config_shape:
        raise ValueError("paired freeze v4 config does not match generated-class resources")
    if (
        _scalar_int(genesis_core_state.step_count, name="genesis step_count") != 0
        or _scalar_int(
            genesis_core_state.replacement_phase,
            name="genesis replacement_phase",
        )
        != 0
        or tuple(
            int(value)
            for value in np.asarray(genesis_core_state.step_words, dtype=np.uint32)
        )
        != (0, 0)
    ):
        raise ValueError("paired freeze requires exact core genesis coordinates")
    if tuple(int(value) for value in genesis_ledger_state.step_words) != (0, 0):
        raise ValueError("paired freeze requires exact v4 ledger genesis words")
    exogenous_sha256 = _paired_exogenous_manifest_sha256(
        observations,
        targets,
        input_dim=protocol.input_dim,
        context_id=context_id,
    )
    genesis_prefix, boundary_core, boundary_ledger = _authenticated_ordinary_steps(
        learner,
        config,
        genesis_core_state,
        genesis_ledger_state,
        observations[:scrub_boundary_step],
        targets[:scrub_boundary_step],
        context_id=context_id,
    )
    if (
        _scalar_int(boundary_core.step_count, name="scrub boundary step_count")
        != scrub_boundary_step
        or _scalar_int(
            boundary_core.replacement_phase,
            name="scrub boundary replacement_phase",
        )
        != scrub_boundary_step
        or tuple(int(value) for value in boundary_ledger.step_words)
        != (0, scrub_boundary_step)
    ):
        raise GeneratedClassRecurrenceDevelopmentError(
            "authenticated genesis prefix did not reach exact step/phase/word coordinate 17"
        )
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_d = next(target.expression for target in manifest.targets if target.name == "D")
    lineage_config = _lineage_config(protocol)
    lineage_plan = compile_expanded_expression_lineage_masks(
        boundary_core,
        target_d,
        config=lineage_config,
    )
    if not lineage_plan.audit.pre_target_present or not lineage_plan.audit.nonempty_causal_plan:
        raise GeneratedClassRecurrenceDevelopmentError(
            "authenticated boundary has no expanded D lineage to scrub"
        )
    scrub_config = GeneratedClassScrubConfig(
        feature_dim=protocol.input_dim,
        active_slots=protocol.active_slots,
        candidate_slots=protocol.candidate_slots,
        n_tasks=protocol.n_tasks,
    )
    scrub_result = scrub_compositional_feature_state(
        boundary_core,
        lineage_plan.active_mask,
        lineage_plan.candidate_mask,
        jnp.asarray(True, dtype=jnp.bool_),
        config=scrub_config,
    )
    scrub_result.state.step_words.block_until_ready()
    if not _scalar_bool(
        scrub_result.diagnostics.committed,
        name="causal scrub committed",
    ):
        raise GeneratedClassRecurrenceDevelopmentError("causal D scrub did not commit")
    candidate_mask = np.asarray(lineage_plan.candidate_mask, dtype=np.bool_)
    candidate_heads = np.asarray(
        scrub_result.state.candidate_output_weights,
        dtype=np.float32,
    )
    scrubbed_candidate_heads_zero = bool(
        not np.any(candidate_mask)
        or np.all(candidate_heads[:, candidate_mask].view(np.uint32) == np.uint32(0))
    )
    if not scrubbed_candidate_heads_zero:
        raise GeneratedClassRecurrenceDevelopmentError(
            "scrubbed candidate identity birth retained a nonzero head"
        )
    epoch_config = GeneratedReacquisitionEpochConfig(paired_life_seed=seed)
    scrub_inputs = scrub_epoch_module.GeneratedBirthIdentityScrubEpochInputs(
        config=config,
        pre_ledger_state=boundary_ledger,
        pre_core_state=boundary_core,
        post_core_state=scrub_result.state,
        target=target_d,
        lineage_plan=lineage_plan,
        lineage_config=lineage_config,
        scrub_config=scrub_config,
        epoch_config=epoch_config,
    )
    scrub_transaction = (
        scrub_epoch_module.build_generated_birth_identity_scrub_epoch_transaction(
            config,
            boundary_ledger,
            boundary_core,
            scrub_result.state,
            target_d,
            lineage_plan,
            lineage_config=lineage_config,
            scrub_config=scrub_config,
            epoch_config=epoch_config,
        )
    )
    scrub_validation = (
        scrub_epoch_module.validate_generated_birth_identity_scrub_epoch_transaction_from_inputs(
            scrub_transaction,
            scrub_inputs,
        )
    )
    if not scrub_validation.valid or not scrub_validation.output_is_normal_v4_state:
        raise GeneratedClassRecurrenceDevelopmentError(
            "causal scrub rollover failed independent validation"
        )
    matched_sham_start = freeze_module.build_generated_birth_identity_matched_sham_start(
        scrub_transaction,
        scrub_inputs,
    )
    freeze_module.validate_generated_birth_identity_matched_sham_start(
        matched_sham_start,
        causal_scrub=scrub_transaction,
        scrub_inputs=scrub_inputs,
    )
    freeze_observations = observations[scrub_boundary_step:]
    freeze_targets = targets[scrub_boundary_step:]
    causal = _build_generated_class_freeze_arm(
        learner,
        config,
        scrub_transaction,
        scrub_inputs,
        scrub_inputs.post_core_state,
        scrub_transaction.post_ledger_state,
        freeze_observations,
        freeze_targets,
        matched_sham_start=None,
        context_id=context_id,
    )
    sham = _build_generated_class_freeze_arm(
        learner,
        config,
        scrub_transaction,
        scrub_inputs,
        matched_sham_start.start_core_state,
        matched_sham_start.start_ledger_state,
        freeze_observations,
        freeze_targets,
        matched_sham_start=matched_sham_start,
        context_id=context_id,
    )
    paired_transaction = (
        freeze_module.build_generated_birth_identity_paired_freeze_transaction(
            causal.endpoint,
            causal.inputs,
            sham.endpoint,
            sham.inputs,
        )
    )
    paired_validation = (
        freeze_module.validate_generated_birth_identity_paired_freeze_transaction(
            paired_transaction,
            causal_endpoint=causal.endpoint,
            causal_inputs=causal.inputs,
            sham_endpoint=sham.endpoint,
            sham_inputs=sham.inputs,
        )
    )
    if not paired_validation.valid or not paired_validation.causal_output_ready_for_next_trace:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired causal/sham freeze transaction failed independent validation"
        )
    paired_audit = paired_transaction.audit
    accounting = GeneratedClassPairedFreezeAccounting(
        genesis_prefix_direct_update_calls=len(genesis_prefix),
        genesis_prefix_source_replay_calls=len(genesis_prefix),
        causal_total_learner_update_calls=(
            paired_audit.causal_total_learner_update_calls
        ),
        sham_total_learner_update_calls=paired_audit.sham_total_learner_update_calls,
        matched_sham_scrub_kernel_calls=(
            paired_audit.total_matched_sham_scrub_kernel_calls_for_validated_pair
        ),
        exact_learner_work_parity=paired_audit.exact_learner_work_parity,
        measured_runtime_sample_count=0,
        measured_runtime_parity_claimed=False,
        wall_clock_threshold=None,
        operation_accounting_scope=_PAIRED_OPERATION_ACCOUNTING_SCOPE,
    )
    prefix_bindings = tuple(step.binding for step in genesis_prefix)
    first_binding = prefix_bindings[0]
    causal_due = causal.inputs.due_transaction
    causal_end = causal.inputs.freeze_end_core_state
    audit_without_hash = GeneratedClassPairedFreezeAudit(
        schema=GENERATED_CLASS_PAIRED_FREEZE_SCHEMA,
        status=GENERATED_CLASS_PAIRED_FREEZE_STATUS,
        genesis_step=0,
        scrub_boundary_step=scrub_boundary_step,
        phase_derived_due_pre_step=causal_due.audit.phase_derived_due_pre_step,
        phase_derived_due_post_step=causal_due.audit.phase_derived_due_post_step,
        freeze_endpoint_step=causal.endpoint.audit.freeze_end_step,
        exogenous_step_count=len(observations),
        exogenous_input_manifest_sha256=exogenous_sha256,
        genesis_prefix_transaction_sha256=tuple(
            binding.transaction.audit.transaction_sha256 for binding in prefix_bindings
        ),
        shared_core_source_sha256=first_binding.core_module_sha256,
        shared_ledger_source_sha256=first_binding.ledger_module_sha256,
        shared_learner_config_sha256=paired_audit.shared_learner_config_sha256,
        shared_ledger_config_sha256=paired_audit.shared_ledger_config_sha256,
        genesis_key_words_uint32=_key_words(genesis_core_state.key),
        scrub_boundary_key_words_uint32=_key_words(boundary_core.key),
        due_pre_key_words_uint32=_key_words(
            causal.inputs.due_inputs.learner_pre_state.key
        ),
        freeze_end_pre_fresh_key_words_uint32=_key_words(causal_end.key),
        fresh_key_words_uint32=_key_words(paired_transaction.causal_output_core_state.key),
        scrub_rollover_transaction_sha256=(
            scrub_transaction.audit.transaction_sha256
        ),
        matched_sham_start_sha256=matched_sham_start.audit.transaction_sha256,
        causal_endpoint_transaction_sha256=causal.endpoint.audit.transaction_sha256,
        sham_endpoint_transaction_sha256=sham.endpoint.audit.transaction_sha256,
        paired_transaction_sha256=paired_audit.transaction_sha256,
        causal_scrub_committed=True,
        matched_sham_scrub_executed_noncommitting=True,
        attempted_due_branches_authenticated_and_abandoned=True,
        shadow_due_branches_authenticated_and_carried=True,
        exact_crn_input_parity=paired_audit.exact_crn_input_parity,
        typed_key_checkpoints_bound=True,
        fresh_key_applied_only_at_endpoint=True,
        scrubbed_candidate_heads_zero_at_identity_birth=(
            scrubbed_candidate_heads_zero
        ),
        sham_endpoint_state_discarded=paired_audit.sham_endpoint_state_discarded,
        causal_output_only=True,
        causal_output_core_state_sha256=paired_audit.causal_output_core_state_sha256,
        causal_output_ledger_state_sha256=(
            paired_audit.causal_output_ledger_state_sha256
        ),
        accounting=accounting,
        receipt_sha256="",
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )
    audit = dataclasses.replace(
        audit_without_hash,
        receipt_sha256=_paired_audit_sha256(audit_without_hash),
    )
    receipt = GeneratedClassPairedFreezeReceipt(
        genesis_core_state=genesis_core_state,
        genesis_ledger_state=genesis_ledger_state,
        genesis_prefix_steps=genesis_prefix,
        scrub_inputs=scrub_inputs,
        scrub_transaction=scrub_transaction,
        matched_sham_start=matched_sham_start,
        causal_endpoint_inputs=causal.inputs,
        sham_endpoint_inputs=sham.inputs,
        paired_transaction=paired_transaction,
        paired_validation=paired_validation,
        audit=audit,
    )
    return GeneratedClassPairedFreezeExecution(
        core_state=paired_transaction.causal_output_core_state,
        ledger_state=paired_transaction.causal_output_ledger_state,
        receipt=receipt,
    )


def validate_authenticated_generated_class_paired_scrub_freeze(
    execution: GeneratedClassPairedFreezeExecution,
) -> GeneratedClassPairedFreezeExecution:
    """Adversarially replay the shared prefix and complete paired receipt graph."""

    if type(execution) is not GeneratedClassPairedFreezeExecution:
        raise TypeError("execution must be an exact GeneratedClassPairedFreezeExecution")
    receipt = execution.receipt
    if type(receipt) is not GeneratedClassPairedFreezeReceipt:
        raise TypeError("execution receipt must be an exact paired-freeze receipt")
    audit = receipt.audit
    if type(audit) is not GeneratedClassPairedFreezeAudit:
        raise TypeError("paired-freeze audit has a malformed type")
    if audit.receipt_sha256 != _paired_audit_sha256(audit):
        raise ValueError("paired-freeze receipt hash does not reconstruct")
    if audit.schema != GENERATED_CLASS_PAIRED_FREEZE_SCHEMA or audit.status != (
        GENERATED_CLASS_PAIRED_FREEZE_STATUS
    ):
        raise ValueError("paired-freeze audit schema or status drifted")
    if not audit.development_only or any(
        (
            audit.execution_authorized,
            audit.runner_authorized,
            audit.campaign_authorized,
            audit.artifact_writes_authorized,
            audit.threshold_authorized,
            audit.evidence_authorized,
            audit.scientific_promotion_allowed,
        )
    ):
        raise ValueError("paired-freeze receipt grants forbidden authority")
    scrub_validation = (
        scrub_epoch_module.validate_generated_birth_identity_scrub_epoch_transaction_from_inputs(
            receipt.scrub_transaction,
            receipt.scrub_inputs,
        )
    )
    if not scrub_validation.valid:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired receipt scrub transaction rejected during replay"
        )
    freeze_module.validate_generated_birth_identity_matched_sham_start(
        receipt.matched_sham_start,
        causal_scrub=receipt.scrub_transaction,
        scrub_inputs=receipt.scrub_inputs,
    )
    paired_validation = (
        freeze_module.validate_generated_birth_identity_paired_freeze_transaction(
            receipt.paired_transaction,
            causal_endpoint=receipt.paired_transaction.causal_endpoint,
            causal_inputs=receipt.causal_endpoint_inputs,
            sham_endpoint=receipt.paired_transaction.sham_endpoint,
            sham_inputs=receipt.sham_endpoint_inputs,
        )
    )
    if not paired_validation.valid:
        raise GeneratedClassRecurrenceDevelopmentError(
            "paired receipt failed stable paired-path replay"
        )
    if receipt.paired_validation != paired_validation:
        raise ValueError("stored paired validation differs from independent replay")
    learner = receipt.causal_endpoint_inputs.due_inputs.learner
    config = receipt.scrub_inputs.config
    core_state = receipt.genesis_core_state
    ledger_state = receipt.genesis_ledger_state
    prefix_transaction_hashes: list[str] = []
    prefix_core_sha: str | None = None
    prefix_ledger_sha: str | None = None
    for index, step in enumerate(receipt.genesis_prefix_steps):
        if type(step) is not freeze_module.GeneratedBirthIdentityFreezeOrdinaryStep:
            raise TypeError("genesis prefix contains a malformed ordinary receipt")
        if scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
            step.learner_pre_state
        ) != scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
            core_state
        ):
            raise ValueError(f"genesis prefix step {index} has a stale pre-state")
        replay = authenticate_generated_birth_identity_trace_by_source_replay(
            learner,
            config,
            ledger_state,
            learner_pre_state=core_state,
            learner_post_state=step.supplied_update_result.state,
            supplied_update_result=step.supplied_update_result,
            observation=step.observation,
            targets=step.targets,
            context_id=step.context_id,
        )
        if not replay.source_replay_authenticated or _scalar_bool(
            step.supplied_update_result.curation_trace.has_event,
            name="genesis prefix has_event",
        ):
            raise ValueError("genesis prefix is not an authenticated no-event chain")
        if replay.transaction.audit.transaction_sha256 != (
            step.binding.transaction.audit.transaction_sha256
        ):
            raise ValueError("genesis prefix stored binding differs from source replay")
        prefix_transaction_hashes.append(replay.transaction.audit.transaction_sha256)
        prefix_core_sha = replay.core_module_sha256
        prefix_ledger_sha = replay.ledger_module_sha256
        core_state = step.supplied_update_result.state
        ledger_state = replay.transaction.post_state
    if len(receipt.genesis_prefix_steps) != audit.scrub_boundary_step:
        raise ValueError("genesis prefix length differs from scrub boundary coordinate")
    if scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
        core_state
    ) != scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
        receipt.scrub_inputs.pre_core_state
    ) or ledger_state.integrity_sha256 != (
        receipt.scrub_inputs.pre_ledger_state.integrity_sha256
    ):
        raise ValueError("genesis prefix does not terminate at the scrub inputs")
    due_inputs = receipt.causal_endpoint_inputs.due_inputs
    observations = (
        *(step.observation for step in receipt.genesis_prefix_steps),
        *(step.observation for step in due_inputs.prefix_steps),
        due_inputs.observation,
        *(step.observation for step in receipt.causal_endpoint_inputs.suffix_steps),
    )
    target_rows = (
        *(step.targets for step in receipt.genesis_prefix_steps),
        *(step.targets for step in due_inputs.prefix_steps),
        due_inputs.targets,
        *(step.targets for step in receipt.causal_endpoint_inputs.suffix_steps),
    )
    if len(observations) != audit.exogenous_step_count:
        raise ValueError("paired receipt exogenous row count is stale")
    context_values = {
        *(int(np.asarray(step.context_id)) for step in receipt.genesis_prefix_steps),
        *(int(np.asarray(step.context_id)) for step in due_inputs.prefix_steps),
        int(np.asarray(due_inputs.context_id)),
        *(
            int(np.asarray(step.context_id))
            for step in receipt.causal_endpoint_inputs.suffix_steps
        ),
    }
    if context_values != {0}:
        raise ValueError("paired receipt contains noncanonical context inputs")
    exogenous_sha = _paired_exogenous_manifest_sha256(
        observations,
        target_rows,
        input_dim=config.raw_feature_slots,
        context_id=0,
    )
    if exogenous_sha != audit.exogenous_input_manifest_sha256:
        raise ValueError("paired receipt exogenous input manifest is stale")
    expected_prefix_hashes = tuple(prefix_transaction_hashes)
    paired_audit = receipt.paired_transaction.audit
    causal_due = receipt.causal_endpoint_inputs.due_transaction
    sham_due = receipt.sham_endpoint_inputs.due_transaction
    causal_end = receipt.causal_endpoint_inputs.freeze_end_core_state
    candidate_mask = np.asarray(
        receipt.scrub_inputs.lineage_plan.candidate_mask,
        dtype=np.bool_,
    )
    candidate_heads = np.asarray(
        receipt.scrub_inputs.post_core_state.candidate_output_weights,
        dtype=np.float32,
    )
    scrubbed_candidate_heads_zero = bool(
        not np.any(candidate_mask)
        or np.all(candidate_heads[:, candidate_mask].view(np.uint32) == np.uint32(0))
    )
    expected_accounting = GeneratedClassPairedFreezeAccounting(
        genesis_prefix_direct_update_calls=len(receipt.genesis_prefix_steps),
        genesis_prefix_source_replay_calls=len(receipt.genesis_prefix_steps),
        causal_total_learner_update_calls=(
            paired_audit.causal_total_learner_update_calls
        ),
        sham_total_learner_update_calls=paired_audit.sham_total_learner_update_calls,
        matched_sham_scrub_kernel_calls=(
            paired_audit.total_matched_sham_scrub_kernel_calls_for_validated_pair
        ),
        exact_learner_work_parity=paired_audit.exact_learner_work_parity,
        measured_runtime_sample_count=0,
        measured_runtime_parity_claimed=False,
        wall_clock_threshold=None,
        operation_accounting_scope=_PAIRED_OPERATION_ACCOUNTING_SCOPE,
    )
    expected_fields = (
        audit.genesis_step == 0,
        audit.scrub_boundary_step == REALISTIC_SCRUB_BOUNDARY_STEP,
        audit.phase_derived_due_pre_step == causal_due.audit.phase_derived_due_pre_step,
        audit.phase_derived_due_post_step == causal_due.audit.phase_derived_due_post_step,
        audit.freeze_endpoint_step == REALISTIC_FREEZE_ENDPOINT_STEP,
        audit.genesis_prefix_transaction_sha256 == expected_prefix_hashes,
        audit.shared_core_source_sha256 == prefix_core_sha,
        audit.shared_ledger_source_sha256 == prefix_ledger_sha,
        audit.shared_core_source_sha256 == paired_audit.shared_core_source_sha256,
        audit.shared_ledger_source_sha256 == paired_audit.shared_ledger_source_sha256,
        audit.shared_learner_config_sha256
        == paired_audit.shared_learner_config_sha256,
        audit.shared_ledger_config_sha256 == paired_audit.shared_ledger_config_sha256,
        audit.genesis_key_words_uint32 == _key_words(receipt.genesis_core_state.key),
        audit.scrub_boundary_key_words_uint32
        == _key_words(receipt.scrub_inputs.pre_core_state.key),
        audit.due_pre_key_words_uint32 == _key_words(due_inputs.learner_pre_state.key),
        audit.freeze_end_pre_fresh_key_words_uint32 == _key_words(causal_end.key),
        audit.fresh_key_words_uint32 == _key_words(execution.core_state.key),
        audit.scrub_rollover_transaction_sha256
        == receipt.scrub_transaction.audit.transaction_sha256,
        audit.matched_sham_start_sha256
        == receipt.matched_sham_start.audit.transaction_sha256,
        audit.causal_endpoint_transaction_sha256
        == receipt.paired_transaction.causal_endpoint.audit.transaction_sha256,
        audit.sham_endpoint_transaction_sha256
        == receipt.paired_transaction.sham_endpoint.audit.transaction_sha256,
        audit.paired_transaction_sha256 == paired_audit.transaction_sha256,
        audit.causal_scrub_committed
        and receipt.scrub_transaction.audit.structural_scrub_valid,
        audit.matched_sham_scrub_executed_noncommitting
        and receipt.matched_sham_start.audit.matched_sham_scrub_executed
        and not receipt.matched_sham_start.audit.matched_sham_scrub_commit_requested
        and receipt.matched_sham_start.audit.matched_sham_scrub_noop_validated,
        audit.attempted_due_branches_authenticated_and_abandoned
        and causal_due.audit.attempted_event_authenticated
        and causal_due.audit.attempted_branch_abandoned
        and sham_due.audit.attempted_event_authenticated
        and sham_due.audit.attempted_branch_abandoned,
        audit.shadow_due_branches_authenticated_and_carried
        and causal_due.audit.shadow_no_event_authenticated
        and causal_due.audit.shadow_no_event_branch_carried
        and sham_due.audit.shadow_no_event_authenticated
        and sham_due.audit.shadow_no_event_branch_carried,
        audit.exact_crn_input_parity == paired_audit.exact_crn_input_parity,
        audit.typed_key_checkpoints_bound,
        audit.fresh_key_applied_only_at_endpoint
        and receipt.paired_transaction.causal_endpoint.audit.fresh_key_is_only_endpoint_state_change
        and receipt.paired_transaction.sham_endpoint.audit.fresh_key_is_only_endpoint_state_change,
        audit.scrubbed_candidate_heads_zero_at_identity_birth
        and scrubbed_candidate_heads_zero,
        audit.sham_endpoint_state_discarded == paired_audit.sham_endpoint_state_discarded,
        audit.causal_output_only,
        audit.causal_output_core_state_sha256
        == paired_audit.causal_output_core_state_sha256,
        audit.causal_output_ledger_state_sha256
        == paired_audit.causal_output_ledger_state_sha256,
        audit.accounting == expected_accounting,
    )
    if not all(expected_fields):
        raise ValueError("paired-freeze audit does not reconstruct from stable receipts")
    output_core_sha = (
        scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
            execution.core_state
        )
    )
    canonical_core_sha = (
        scrub_epoch_module.generated_birth_identity_scrub_epoch_core_state_sha256(
            receipt.paired_transaction.causal_output_core_state
        )
    )
    if output_core_sha != canonical_core_sha or execution.ledger_state.integrity_sha256 != (
        receipt.paired_transaction.causal_output_ledger_state.integrity_sha256
    ):
        raise ValueError("execution does not carry only the paired causal output")
    return execution


def _trace_accounting(
    trace: CompositionalCurationTrace,
    binding: Any,
    before: CompositionalFeatureState,
    after: CompositionalFeatureState,
) -> GeneratedClassStepAccounting:
    if type(trace) is not CompositionalCurationTrace:
        raise GeneratedClassRecurrenceDevelopmentError(
            "production update did not expose the exact public curation trace"
        )
    changed_count, changed_bytes = _persistent_write_accounting(before, after)
    refresh_mask = (
        np.asarray(trace.candidate_refresh_mask, dtype=np.bool_)
        | np.asarray(trace.candidate_rebound_mask, dtype=np.bool_)
        | np.asarray(trace.candidate_overdepth_regeneration_mask, dtype=np.bool_)
    )
    candidate_weights = np.asarray(after.candidate_output_weights, dtype=np.float32)
    refresh_heads_zero = bool(
        not np.any(refresh_mask)
        or np.all(candidate_weights[:, refresh_mask].view(np.uint32) == np.uint32(0))
    )
    before_nbytes = persistent_compositional_state_nbytes(before)
    after_nbytes = persistent_compositional_state_nbytes(after)
    if before_nbytes != after_nbytes:
        raise GeneratedClassRecurrenceDevelopmentError(
            "production update changed persistent learner capacity"
        )
    return GeneratedClassStepAccounting(
        proposal_count=_scalar_int(trace.proposal_count, name="trace.proposal_count"),
        root_change_count=_scalar_int(
            trace.root_change_count,
            name="trace.root_change_count",
        ),
        promotion_count=_scalar_int(
            trace.promotion_count,
            name="trace.promotion_count",
        ),
        cascade_refill_count=_scalar_int(
            trace.cascade_refill_count,
            name="trace.cascade_refill_count",
        ),
        candidate_refresh_count=_scalar_int(
            trace.candidate_refresh_count,
            name="trace.candidate_refresh_count",
        ),
        candidate_rebound_count=_scalar_int(
            trace.candidate_rebound_count,
            name="trace.candidate_rebound_count",
        ),
        candidate_overdepth_regeneration_count=_scalar_int(
            trace.candidate_overdepth_regeneration_count,
            name="trace.candidate_overdepth_regeneration_count",
        ),
        logical_curation_event_count=_scalar_int(
            trace.logical_event_count,
            name="trace.logical_event_count",
        ),
        identity_events_applied=int(
            binding.transaction.audit.applied_identity_event_count
        ),
        changed_persistent_array_leaf_count=changed_count,
        changed_persistent_array_bytes=changed_bytes,
        persistent_state_nbytes_before=before_nbytes,
        persistent_state_nbytes_after=after_nbytes,
        candidate_identity_refresh_heads_zero=refresh_heads_zero,
    )


def _lifecycle_trace(
    arm: GeneratedClassDevelopmentArm,
    steps: Sequence[GeneratedClassStepTrace],
    *,
    initial_active_present: bool,
    initial_candidate_present: bool,
) -> GeneratedClassLifecycleTrace:
    active_presence = [initial_active_present]
    candidate_presence = [initial_candidate_present]
    exact_presence = [initial_active_present or initial_candidate_present]
    first_use: int | None = None
    d_exposures = 0
    for step in steps:
        snapshot = step.occurrence_identity
        active = snapshot.active_exact_root_occurrences > 0
        candidate = snapshot.candidate_exact_root_occurrences > 0
        active_presence.append(active)
        candidate_presence.append(candidate)
        exact_presence.append(active or candidate)
        if step.presented_target_name == "D":
            d_exposures += 1
            if (
                first_use is None
                and active
                and snapshot.active_lineage_has_nonzero_output_head
            ):
                first_use = step.step_index
    births = tuple(
        index - 1
        for index, (previous, current) in enumerate(
            zip(exact_presence[:-1], exact_presence[1:], strict=True),
            start=1,
        )
        if current and not previous
    )
    retirements = tuple(
        index - 1
        for index, (previous, current) in enumerate(
            zip(exact_presence[:-1], exact_presence[1:], strict=True),
            start=1,
        )
        if previous and not current
    )
    active_summary = summarize_presence_history(active_presence)
    candidate_summary = summarize_presence_history(candidate_presence)
    reasons: list[str] = []
    if arm.name == D_NEVER_SEEN_TWIN:
        reasons.append("true D mapping intentionally withheld until the second D phase")
    if not births:
        reasons.append("no natural exact-D birth observed")
    if first_use is None:
        reasons.append("no natural exact-D active use observed")
    if not retirements:
        reasons.append("no natural exact-D retirement observed")
    reasons.extend(
        (
            "tiny replay never attempts a causal scrub",
            "tiny replay never runs a generation-frozen occlusion epoch",
            "tiny replay never applies a fresh reacquisition key",
        )
    )
    return GeneratedClassLifecycleTrace(
        target_d_exposure_count=d_exposures,
        target_d_absent_before_second_d_by_evaluator_construction=(
            arm.name == D_NEVER_SEEN_TWIN
        ),
        d_initially_absent=not exact_presence[0],
        d_first_exact_presence_step=births[0] if births else None,
        d_first_use_step=first_use,
        d_exact_birth_steps=births,
        d_exact_retirement_steps=retirements,
        d_exact_reacquisition_steps=births[1:],
        active_presence=active_summary,
        candidate_presence=candidate_summary,
        scrub_attempted=False,
        scrub_committed=False,
        scrub_source_replay_boundary_authenticated=False,
        reacquisition_epoch_key_applied=False,
        generation_frozen_updates_executed=0,
        lifecycle_complete=False,
        incompleteness_reasons=tuple(reasons),
    )


def _run_tiny_trial(
    plan: GeneratedClassRecurrenceDevelopmentPlan,
    seed: int,
    arm: GeneratedClassDevelopmentArm,
    adapter: _TraceLedgerAdapter,
) -> GeneratedClassDevelopmentTrial:
    protocol = build_generated_class_recurrence_v0_protocol()
    learner = _build_arm_learner(arm, protocol)
    root = jr.key(seed, impl=THREEFRY_IMPLEMENTATION)
    learner_key = jr.fold_in(root, np.uint32(LEARNER_DOMAIN))
    state = learner.init(plan.input_dim, learner_key)
    initial_nbytes = persistent_compositional_state_nbytes(state)
    if initial_nbytes != arm.persistent_jax_state_nbytes:
        raise GeneratedClassRecurrenceDevelopmentError(
            "concrete learner state violates the declared persistent capacity"
        )
    ledger_config = adapter.config_type(
        namespace=f"{TRACE_LEDGER_NAMESPACE_PREFIX}-{arm.name}-{seed:08x}",
        active_slots=plan.arms[0].allocated_active_slots,
        candidate_slots=plan.arms[0].allocated_candidate_slots,
        raw_feature_slots=plan.input_dim,
        max_depth=learner.max_depth,
        learn_generator_resources=bool(learner.to_config()["learn_generator_resources"]),
    )
    ledger_state = adapter.attach(
        ledger_config,
        learner_pre_state=state,
        paired_development_life_seed=seed,
    )
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_d = next(target.expression for target in manifest.targets if target.name == "D")
    lineage_config = _lineage_config(protocol)
    initial_snapshot, _ = _occurrence_identity_snapshot(
        state,
        ledger_state,
        target_d,
        lineage_config=lineage_config,
    )
    if (
        initial_snapshot.active_exact_root_occurrences != 0
        or initial_snapshot.candidate_exact_root_occurrences != 0
    ):
        raise GeneratedClassRecurrenceDevelopmentError(
            "target D must be initially absent in every paired arm"
        )
    stream = build_generated_class_development_stream(plan, seed, arm.name)
    traces: list[GeneratedClassStepTrace] = []
    for stream_step in stream.steps:
        observation = jnp.asarray(stream_step.raw_features, dtype=jnp.float32)
        target = jnp.asarray((stream_step.target,), dtype=jnp.float32)
        pre_state = state
        result = learner.update(pre_state, observation, target)
        result.state.step_words.block_until_ready()
        binding = adapter.authenticate(
            learner,
            ledger_config,
            ledger_state,
            learner_pre_state=pre_state,
            learner_post_state=result.state,
            supplied_update_result=result,
            observation=observation,
            targets=target,
        )
        if not bool(binding.source_replay_authenticated):
            raise GeneratedClassRecurrenceDevelopmentError(
                "production update trace was not source-replay authenticated"
            )
        if not bool(binding.ledger_validation.valid):
            raise GeneratedClassRecurrenceDevelopmentError(
                "schema-v4 birth-identity transaction failed strict validation"
            )
        if any(
            (
                bool(binding.execution_authorized),
                bool(binding.runner_authorized),
                bool(binding.artifact_writes_authorized),
                bool(binding.evidence_authorized),
                bool(binding.scientific_promotion_allowed),
            )
        ):
            raise GeneratedClassRecurrenceDevelopmentError(
                "trace adapter unexpectedly granted external authority"
            )
        ledger_state = binding.transaction.post_state
        state = result.state
        occurrence, _ = _occurrence_identity_snapshot(
            state,
            ledger_state,
            target_d,
            lineage_config=lineage_config,
        )
        predictions = np.asarray(result.predictions, dtype=np.float32)
        errors = np.asarray(result.errors, dtype=np.float32)
        metrics = tuple(float(value) for value in np.asarray(result.metrics, dtype=np.float32))
        if predictions.shape != (1,) or errors.shape != (1,):
            raise GeneratedClassRecurrenceDevelopmentError(
                "one-head production prediction shape drifted"
            )
        prediction = float(predictions[0])
        error = float(errors[0])
        loss = float(np.float32(error) * np.float32(error))
        if not all(math.isfinite(value) for value in (prediction, error, loss, *metrics)):
            raise GeneratedClassRecurrenceDevelopmentError(
                "production update returned a nonfinite raw metric"
            )
        trace = result.curation_trace
        accounting = _trace_accounting(trace, binding, pre_state, state)
        if arm.candidate_identity_refresh_head_zero and not (
            accounting.candidate_identity_refresh_heads_zero
        ):
            raise GeneratedClassRecurrenceDevelopmentError(
                "zero-head control retained a candidate head across identity refresh"
            )
        post_words = tuple(
            int(value) for value in np.asarray(state.step_words, dtype=np.uint32)
        )
        if len(post_words) != 2:
            raise GeneratedClassRecurrenceDevelopmentError(
                "production lifetime counter did not expose two exact words"
            )
        traces.append(
            GeneratedClassStepTrace(
                step_index=stream_step.step_index,
                phase_index=stream_step.phase_index,
                phase_label=stream_step.phase_label,
                phase_boundary=stream_step.phase_boundary,
                presented_target_name=stream_step.presented_target_name,
                learner_input_fields=("raw_features", "target"),
                raw_features=stream_step.raw_features,
                target=stream_step.target,
                prediction_before_update=prediction,
                error_before_update=error,
                prequential_squared_loss=loss,
                production_metrics=metrics,
                post_step_words_uint32=post_words,
                curation_trace_has_event=_scalar_bool(
                    trace.has_event,
                    name="trace.has_event",
                ),
                source_replay_authenticated=True,
                trace_binding_schema=str(binding.schema),
                ledger_transaction_sha256=str(
                    binding.transaction.audit.transaction_sha256
                ),
                occurrence_identity=occurrence,
                accounting=accounting,
            )
        )
    if any(trace.curation_trace_has_event for trace in traces):
        raise GeneratedClassRecurrenceDevelopmentError(
            "tiny replay crossed a production curation event"
        )
    lifecycle = _lifecycle_trace(
        arm,
        traces,
        initial_active_present=False,
        initial_candidate_present=False,
    )
    return GeneratedClassDevelopmentTrial(
        root_seed_uint32=seed,
        arm=arm,
        stream_sha256=stream.stream_sha256,
        initial_persistent_state_nbytes=initial_nbytes,
        final_persistent_state_nbytes=persistent_compositional_state_nbytes(state),
        step_traces=tuple(traces),
        lifecycle=lifecycle,
        total_prequential_squared_loss=float(
            np.sum(
                [trace.prequential_squared_loss for trace in traces],
                dtype=np.float64,
            )
        ),
        total_identity_events_applied=sum(
            trace.accounting.identity_events_applied for trace in traces
        ),
        total_changed_persistent_array_bytes=sum(
            trace.accounting.changed_persistent_array_bytes for trace in traces
        ),
        artifacts_written=0,
        evidence_authorized=False,
        promotion_authorized=False,
    )


def _campaign_digest(
    plan: GeneratedClassRecurrenceDevelopmentPlan,
    trials: tuple[GeneratedClassDevelopmentTrial, ...],
) -> str:
    payload = {
        "schema": plan.schema,
        "phase_lengths": list(plan.phase_lengths),
        "seeds": list(plan.seeds),
        "trials": [dataclasses.asdict(trial) for trial in trials],
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def validate_generated_class_development_campaign(
    campaign: GeneratedClassDevelopmentCampaign,
) -> GeneratedClassDevelopmentCampaign:
    """Validate nested replay invariants and the complete in-memory result hash."""

    if type(campaign) is not GeneratedClassDevelopmentCampaign:
        raise TypeError("campaign must be an exact GeneratedClassDevelopmentCampaign")
    plan = validate_generated_class_recurrence_development_plan(campaign.plan)
    if type(campaign.trials) is not tuple or not campaign.trials:
        raise TypeError("campaign trials must be a non-empty exact tuple")
    if campaign.artifacts_written != 0 or any(
        (campaign.evidence_authorized, campaign.promotion_authorized)
    ):
        raise ValueError("development campaign cannot write artifacts or grant authority")
    seen: set[tuple[int, str]] = set()
    declared_arms = {arm.name: arm for arm in plan.arms}
    for trial_index, trial in enumerate(campaign.trials):
        if type(trial) is not GeneratedClassDevelopmentTrial:
            raise TypeError(f"campaign.trials[{trial_index}] has an invalid type")
        if trial.root_seed_uint32 not in plan.seeds:
            raise ValueError("trial seed is outside the plan")
        if trial.arm.name not in declared_arms or trial.arm != declared_arms[trial.arm.name]:
            raise ValueError("trial arm declaration does not match the plan")
        key = (trial.root_seed_uint32, trial.arm.name)
        if key in seen:
            raise ValueError("campaign contains a duplicate seed/arm trial")
        seen.add(key)
        if type(trial.step_traces) is not tuple or len(trial.step_traces) != plan.total_steps:
            raise ValueError("trial step trace cardinality does not match the plan")
        if trial.initial_persistent_state_nbytes != trial.arm.persistent_jax_state_nbytes:
            raise ValueError("trial initial capacity differs from its arm declaration")
        if trial.final_persistent_state_nbytes != trial.initial_persistent_state_nbytes:
            raise ValueError("trial persistent learner capacity changed")
        if trial.artifacts_written != 0 or any(
            (trial.evidence_authorized, trial.promotion_authorized)
        ):
            raise ValueError("trial cannot write artifacts or grant authority")
        stream = build_generated_class_development_stream(
            plan,
            trial.root_seed_uint32,
            trial.arm.name,
        )
        if trial.stream_sha256 != stream.stream_sha256:
            raise ValueError("trial stream binding is stale")
        for expected_step, trace in zip(stream.steps, trial.step_traces, strict=True):
            if type(trace) is not GeneratedClassStepTrace:
                raise TypeError("trial contains a malformed step trace")
            if (
                trace.step_index != expected_step.step_index
                or trace.phase_index != expected_step.phase_index
                or trace.phase_label != expected_step.phase_label
                or trace.phase_boundary != expected_step.phase_boundary
                or trace.presented_target_name != expected_step.presented_target_name
                or trace.raw_features != expected_step.raw_features
                or trace.target != expected_step.target
            ):
                raise ValueError("step trace does not match the paired evaluator stream")
            if trace.learner_input_fields != ("raw_features", "target"):
                raise ValueError("step trace claims evaluator metadata entered the learner")
            if not trace.source_replay_authenticated:
                raise ValueError("step trace lacks source-replay authentication")
            if len(trace.ledger_transaction_sha256) != 64:
                raise ValueError("step trace ledger transaction hash is malformed")
            expected_words = (0, trace.step_index + 1)
            if trace.post_step_words_uint32 != expected_words:
                raise ValueError("tiny replay lifetime words are not uninterrupted")
            if trace.accounting.persistent_state_nbytes_before != (
                trial.initial_persistent_state_nbytes
            ) or trace.accounting.persistent_state_nbytes_after != (
                trial.initial_persistent_state_nbytes
            ):
                raise ValueError("step accounting reports a persistent-capacity drift")
            expected_loss = float(
                np.float32(trace.error_before_update)
                * np.float32(trace.error_before_update)
            )
            if np.float32(trace.prequential_squared_loss).tobytes() != np.float32(
                expected_loss
            ).tobytes():
                raise ValueError("prequential squared loss is inconsistent with raw error")
        expected_total_loss = float(
            np.sum(
                [trace.prequential_squared_loss for trace in trial.step_traces],
                dtype=np.float64,
            )
        )
        if trial.total_prequential_squared_loss != expected_total_loss:
            raise ValueError("trial total prequential loss is stale")
        if trial.total_identity_events_applied != sum(
            trace.accounting.identity_events_applied for trace in trial.step_traces
        ):
            raise ValueError("trial identity-event total is stale")
        if trial.total_changed_persistent_array_bytes != sum(
            trace.accounting.changed_persistent_array_bytes
            for trace in trial.step_traces
        ):
            raise ValueError("trial persistent-write total is stale")
    expected_digest = _campaign_digest(plan, campaign.trials)
    if campaign.result_sha256 != expected_digest:
        raise ValueError("campaign result hash does not reconstruct")
    return campaign


def run_tiny_generated_class_recurrence_replay(
    plan: GeneratedClassRecurrenceDevelopmentPlan,
    *,
    arm_names: tuple[str, ...] = DECLARED_ARM_ORDER,
) -> GeneratedClassDevelopmentCampaign:
    """Run selected declared arms below the first possible curation transaction.

    The caller must explicitly supply a noncanonical plan.  There is no default
    execution and therefore no accidental full campaign under test or import.
    Every update, including a no-event update, is source-replayed and advances
    the strict schema-v4 birth-identity sidecar.  The default is the complete
    seven-arm declaration; an explicit subset exists for focused development
    diagnostics and does not change the plan's matched-work declaration.
    """

    checked = validate_generated_class_recurrence_development_plan(plan)
    if not checked.tiny_noncurating_replay:
        raise GeneratedClassRecurrenceDevelopmentError(
            "tiny replay requires an explicit noncanonical life of at most 31 steps"
        )
    if checked.total_steps >= checked.replacement_interval:
        raise GeneratedClassRecurrenceDevelopmentError(
            "tiny replay must end before the first production curation interval"
        )
    if type(arm_names) is not tuple or not arm_names:
        raise TypeError("arm_names must be a non-empty exact tuple")
    if not all(type(name) is str for name in arm_names):
        raise TypeError("arm_names must contain exact strings")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("arm_names must not contain duplicates")
    unknown = set(arm_names) - set(DECLARED_ARM_ORDER)
    if unknown:
        raise ValueError(f"unknown generated-class development arms: {sorted(unknown)!r}")
    selected_arms = tuple(arm for arm in checked.arms if arm.name in arm_names)
    if tuple(arm.name for arm in selected_arms) != tuple(
        name for name in DECLARED_ARM_ORDER if name in arm_names
    ):
        raise GeneratedClassRecurrenceDevelopmentError(
            "selected arms drifted from canonical declaration order"
        )
    adapter = _load_trace_ledger_adapter()
    trials = tuple(
        _run_tiny_trial(checked, seed, arm, adapter)
        for seed in checked.seeds
        for arm in selected_arms
    )
    # Common-random-number audit: raw observations are exactly shared across
    # all arms, while only the D-never target differs during D phases.
    for seed in checked.seeds:
        selected = [trial for trial in trials if trial.root_seed_uint32 == seed]
        reference = tuple(trace.raw_features for trace in selected[0].step_traces)
        if any(
            tuple(trace.raw_features for trace in trial.step_traces) != reference
            for trial in selected[1:]
        ):
            raise GeneratedClassRecurrenceDevelopmentError(
                "paired observation streams drifted across arms"
            )
    digest = _campaign_digest(checked, trials)
    campaign = GeneratedClassDevelopmentCampaign(
        plan=checked,
        trials=trials,
        result_sha256=digest,
        artifacts_written=0,
        evidence_authorized=False,
        promotion_authorized=False,
    )
    return validate_generated_class_development_campaign(campaign)


__all__ = [
    "DECLARED_ARM_ORDER",
    "DEFAULT_DEVELOPMENT_SEEDS",
    "D_NEVER_SEEN_TWIN",
    "GENERATED_CLASS_RECURRENCE_DEVELOPMENT_SCHEMA",
    "GENERATED_CLASS_RECURRENCE_DEVELOPMENT_STATUS",
    "GENERATED_CLASS_PAIRED_FREEZE_SCHEMA",
    "GENERATED_CLASS_PAIRED_FREEZE_STATUS",
    "GENERATION_FREEZE_UPDATES",
    "LEARNER_DOMAIN",
    "MATCHED_SHAM_SCRUB",
    "MAX_TINY_NONCURATING_STEPS",
    "OBSERVATION_DOMAIN",
    "REALISTIC_FREEZE_ENDPOINT_STEP",
    "REALISTIC_SCRUB_BOUNDARY_STEP",
    "STREAM_DOMAIN",
    "THREEFRY_IMPLEMENTATION",
    "GeneratedClassDevelopmentArm",
    "GeneratedClassDevelopmentCampaign",
    "GeneratedClassDevelopmentStream",
    "GeneratedClassDevelopmentTrial",
    "GeneratedClassDevelopmentWorkAccounting",
    "GeneratedClassLifecycleTrace",
    "GeneratedClassOccurrenceIdentitySnapshot",
    "GeneratedClassPairedFreezeAccounting",
    "GeneratedClassPairedFreezeAudit",
    "GeneratedClassPairedFreezeExecution",
    "GeneratedClassPairedFreezeReceipt",
    "GeneratedClassRecurrenceAdapterUnavailableError",
    "GeneratedClassRecurrenceDevelopmentError",
    "GeneratedClassRecurrenceDevelopmentPlan",
    "GeneratedClassStepAccounting",
    "GeneratedClassStepTrace",
    "GeneratedClassStreamStep",
    "build_generated_class_development_stream",
    "build_generated_class_recurrence_development_plan",
    "run_authenticated_generated_class_paired_scrub_freeze",
    "run_tiny_generated_class_recurrence_replay",
    "validate_generated_class_development_campaign",
    "validate_authenticated_generated_class_paired_scrub_freeze",
    "validate_generated_class_recurrence_development_plan",
]
