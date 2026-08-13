"""Nonexecuting matched scan plan for the hidden learning-partner bridge.

The plan binds every condition already exposed by
``hidden_learning_partner_planning_development`` to one exact default
configuration and one fixed set of paired exploratory seeds.  It declares raw
outputs and source-justified operation totals.  The separate matched runner,
cross-arm audit, execution-request/permit gate, and authenticated replay are
implemented, but this plan itself grants no execution, artifact, evidence, or
promotion authority.

Arm order is serialization metadata only.  The runner must initialize a
fresh state from the same per-seed root key for every condition and must audit
the named key chains across arms.  This module never executes the default
3,072-step life.  Execution remains blocked until an exact source-bound permit
is issued after a live strict host-quiescence check.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections import defaultdict
from typing import Literal, cast

import jax.random as jr
import numpy as np
from jax import Array

import alberta_framework.evaluation.hidden_learning_partner_planning_development as bridge_module
from alberta_framework.core.signaling_bandit import signaling_bandit_keys
from alberta_framework.evaluation.hidden_learning_partner_planning_development import (
    BEHAVIOR_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_MODELS_FROZEN,
    BOTH_ROLES_FROZEN,
    CONSTANT_ONE_DELIVERY,
    CONSTANT_ZERO_DELIVERY,
    GROUNDED_FROZEN,
    HELPER_FROZEN,
    HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA,
    JOINT_ADAPTIVE,
    MATCHED_CONDITIONS,
    PLANNER_NEVER_CONSUMED,
    SHUFFLED_DELIVERY,
    HiddenLearningPartnerPhaseDiagnostics,
    HiddenLearningPartnerPlanningBridge,
    HiddenLearningPartnerPlanningConfig,
    HiddenLearningPartnerPlanningMetrics,
    HiddenLearningPartnerPlanningResourceBudget,
    HiddenLearningPartnerPlanningState,
    HiddenLearningPartnerPlanningTrace,
    HiddenPlanningCondition,
    HiddenPlanningConditionSpec,
    condition_spec,
)
from alberta_framework.streams.learning_partner import learning_partner_world_keys

HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_SCHEMA = (
    "alberta.hidden-learning-partner-planning.scan-plan.development.v1"
)
HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_STATUS = (
    "RUNNER_AND_CRN_AUDIT_IMPLEMENTED_EXECUTION_PERMIT_REQUIRED"
)
DEVELOPMENT_SEED_NAMESPACE = (
    "alberta/hidden-learning-partner-planning/development-scan/v1/paired-root-seeds"
)

DEVELOPMENT_ONLY = True
EXECUTION_AUTHORIZED = False
RUNNER_AUTHORIZED = False
CAMPAIGN_AUTHORIZED = False
ARTIFACT_WRITES_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

PRIMARY_CONDITIONS: tuple[str, ...] = (
    JOINT_ADAPTIVE,
    HELPER_FROZEN,
    BENEFICIARY_FROZEN,
    BOTH_ROLES_FROZEN,
    BEHAVIOR_FROZEN,
    GROUNDED_FROZEN,
    BOTH_MODELS_FROZEN,
    PLANNER_NEVER_CONSUMED,
)
DIAGNOSTIC_CONDITIONS: tuple[str, ...] = (
    CONSTANT_ZERO_DELIVERY,
    CONSTANT_ONE_DELIVERY,
    SHUFFLED_DELIVERY,
)
CANONICAL_CONDITION_ORDER: tuple[str, ...] = (
    *PRIMARY_CONDITIONS,
    *DIAGNOSTIC_CONDITIONS,
)

PAIRED_DEVELOPMENT_SEEDS: tuple[int, ...] = (104_729, 130_363, 155_921, 196_613)

_WORLD_ROOT_RNG_TAG = 0x4850574C
_LEARNER_ROOT_RNG_TAG = 0x48504C52
_BEHAVIOR_RNG_TAG = 0x48504248
_GROUNDED_RNG_TAG = 0x48504752
_PLANNER_RNG_TAG = 0x4850504C
_INTERVENTION_RNG_TAG = 0x4850494E
_PRNG_IMPL = "threefry2x32"
_EXPECTED_BRIDGE_SCHEMA = "alberta.hidden-learning-partner-planning.development.v1"
_EXPECTED_SOURCE_RESOURCE_BYTES = (80, 48, 108, 32, 321)

ArmFamily = Literal["primary", "diagnostic"]
ArmRole = Literal["reference", "primary_intervention", "diagnostic"]
ContrastFamily = Literal["primary_causal", "diagnostic_only"]


class HiddenLearningPartnerPlanningScanPlanError(RuntimeError):
    """Raised when the nonexecuting plan does not satisfy its exact contract."""


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningKeyStream:
    """Ownership and transition rule for one key derived from a paired root."""

    name: str
    owner: str
    derivation: str
    initializer_or_state_path: str
    lifetime: str
    transition_rule: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningSeedBinding:
    """One exploratory seed and its exact pre-initialization named key words."""

    seed_index: int
    seed: int
    root_key_data: tuple[int, int]
    named_key_data: tuple[tuple[str, tuple[int, int]], ...]


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningSeedContract:
    """Fixed paired development seeds; never evidence or held-out seeds."""

    namespace: str
    prng_impl: str
    selection_method: str
    bindings: tuple[HiddenPlanningSeedBinding, ...]
    paired_across_every_condition: bool
    development_only: bool
    held_out: bool
    evidence_eligible: bool
    executed: bool
    seed_manifest_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningCommonRandomNumbers:
    """Exact common-random-number obligations for any future suite runner."""

    same_seed_set_every_arm: bool
    same_root_key_for_seed_every_arm: bool
    condition_is_key_derivation_input: bool
    arm_order_is_key_derivation_input: bool
    fresh_state_per_seed_condition: bool
    cross_arm_state_reuse_allowed: bool
    allowed_initial_state_difference_fields: tuple[str, ...]
    required_equal_named_key_streams: tuple[str, ...]
    required_cross_arm_trace_key_fields: tuple[str, ...]
    world_cue_stream_reconstruction_required: bool
    world_channel_stream_reconstruction_required: bool
    branch_invariant_persistent_key_advancement_required: bool
    shuffled_channel_output_binding_required: bool
    cross_arm_rng_audit_implemented: bool
    result_join_key: tuple[str, ...]
    serialization_order_semantics: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningNamedOperation:
    """One named bridge-level operation with an exact per-run formula."""

    name: str
    fixed_per_run: int
    per_transition: int
    per_run_total: int


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningExactChildClock:
    """Hash-bound initial/final words for one source-kernel child clock."""

    name: str
    words_state_path: str
    telemetry_state_path: str
    words_dtype: str
    words_shape: tuple[int, ...]
    initial_words: tuple[int, int]
    final_words: tuple[int, int]
    initial_telemetry: int
    final_telemetry: int


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningArm:
    """One canonical condition binding; this is not an executable run request."""

    serialization_index: int
    condition: str
    family: ArmFamily
    role: ArmRole
    contrast_id: str | None
    condition_spec: HiddenPlanningConditionSpec
    config_sha256: str
    config_token_hex: str
    seed_manifest_sha256: str
    resource_budget: HiddenLearningPartnerPlanningResourceBudget
    exact_child_clocks: tuple[HiddenPlanningExactChildClock, ...]
    named_operation_totals: tuple[HiddenPlanningNamedOperation, ...]
    evaluator_order_is_key_input: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningContrast:
    """A raw paired contrast declaration with no threshold or verdict."""

    contrast_id: str
    family: ContrastFamily
    reference_condition: str
    intervention_condition: str
    difference_direction: str
    interpretation: str
    requested_scalar_metric_fields: tuple[str, ...]
    phase_diagnostics_requested: bool
    causal_claim_scope: str
    evidence_authorized: bool
    scientific_promotion_allowed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningRequestedOutputs:
    """Threshold-free future in-memory records requested by the plan."""

    output_mode: str
    record_key_fields: tuple[str, ...]
    initial_state_fields: tuple[str, ...]
    final_state_fields: tuple[str, ...]
    trace_fields: tuple[str, ...]
    scalar_metric_fields: tuple[str, ...]
    phase_diagnostic_fields: tuple[str, ...]
    paired_difference_metric_fields: tuple[str, ...]
    resource_budget_requested: bool
    strict_run_validation_errors_requested: bool
    aggregate_statistics_requested: bool
    outcomes_present: bool
    thresholds_defined: bool
    artifact_output_requested: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningSuiteCounts:
    """Exact logical record and operation totals for the unexecuted scan."""

    primary_arm_count: int
    diagnostic_arm_count: int
    contrast_count: int
    paired_seed_count: int
    planned_run_count: int
    steps_per_run: int
    planned_transition_count: int
    initial_state_record_count: int
    final_state_record_count: int
    trace_row_count: int
    trace_fields_per_row: int
    metric_record_count: int
    scalar_metric_fields_per_record: int
    phase_diagnostic_container_count: int
    phase_rows_per_container: int
    phase_diagnostic_phase_row_count: int
    phase_diagnostic_field_count: int
    state_fields_per_snapshot: int
    persistent_state_bytes_per_run: int
    summed_logical_persistent_state_bytes: int
    suite_named_operation_totals: tuple[tuple[str, int], ...]
    named_operation_accounting_scope: str
    flop_or_hlo_equivalence_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenPlanningReadiness:
    """Structural readiness is separate from execution and host quiescence."""

    plan_validator_implemented: bool
    canonical_bridge_conditions_bound: bool
    paired_seed_and_key_contract_complete: bool
    named_operation_accounting_complete_in_declared_scope: bool
    ready_for_runner_implementation: bool
    suite_runner_implemented: bool
    cross_arm_rng_audit_implemented: bool
    execution_request_and_permit_implemented: bool
    authenticated_source_replay_implemented: bool
    default_life_executed: bool
    outcomes_present: bool
    quiescence_required: bool
    quiescence_checked: bool
    quiescence_verified: bool
    ready_for_execution: bool
    blockers: tuple[str, ...]
    quiescence_declaration: str


@dataclasses.dataclass(frozen=True, slots=True)
class HiddenLearningPartnerPlanningScanPlan:
    """Complete hash-bound declaration for a future matched development scan."""

    schema: str
    status: str
    bridge_schema: str
    development_only: bool
    execution_authorized: bool
    runner_authorized: bool
    campaign_authorized: bool
    artifact_writes_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    config: HiddenLearningPartnerPlanningConfig
    config_sha256: str
    life_steps: int
    key_streams: tuple[HiddenPlanningKeyStream, ...]
    key_manifest_sha256: str
    seed_contract: HiddenPlanningSeedContract
    common_random_numbers: HiddenPlanningCommonRandomNumbers
    resource_budget: HiddenLearningPartnerPlanningResourceBudget
    resource_budget_sha256: str
    arms: tuple[HiddenPlanningArm, ...]
    contrasts: tuple[HiddenPlanningContrast, ...]
    requested_outputs: HiddenPlanningRequestedOutputs
    counts: HiddenPlanningSuiteCounts
    readiness: HiddenPlanningReadiness
    thresholds: None
    outcomes: None
    artifact_output_path: None
    plan_sha256: str


_PAIRED_DIFFERENCE_METRICS: tuple[str, ...] = (
    "mean_reward",
    "behavior_mean_nll",
    "behavior_mean_brier",
    "grounded_reward_mse",
    "grounded_next_observation_mse",
    "planner_consumption_rate",
    "action_change_rate",
    "randomized_effect",
    "potential_effect",
)

_INTERPRETATIONS: dict[str, str] = {
    HELPER_FROZEN: "finite-life contribution of online helper value adaptation",
    BENEFICIARY_FROZEN: "finite-life contribution of online beneficiary value adaptation",
    BOTH_ROLES_FROZEN: "finite-life joint contribution of both role value learners",
    BEHAVIOR_FROZEN: "finite-life contribution of online beneficiary-behavior modeling",
    GROUNDED_FROZEN: "finite-life contribution of online grounded joint-world modeling",
    BOTH_MODELS_FROZEN: "finite-life joint contribution of both online predictive models",
    PLANNER_NEVER_CONSUMED: "finite-life contribution of randomized planner consumption",
    CONSTANT_ZERO_DELIVERY: "diagnostic constant-zero communication channel",
    CONSTANT_ONE_DELIVERY: "diagnostic constant-one communication channel",
    SHUFFLED_DELIVERY: "diagnostic message-destroying shuffled communication channel",
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _config_sha256(config: HiddenLearningPartnerPlanningConfig) -> str:
    return _sha256_json(config.to_dict())


def _resource_sha256(resource: HiddenLearningPartnerPlanningResourceBudget) -> str:
    return _sha256_json(dataclasses.asdict(resource))


def _key_words(key: Array) -> tuple[int, int]:
    data = np.asarray(jr.key_data(key), dtype=np.uint32)
    if data.shape != (2,) or str(jr.key_impl(key)) != _PRNG_IMPL:
        raise HiddenLearningPartnerPlanningScanPlanError(
            "named key is not an exact scalar Threefry key"
        )
    return int(data[0]), int(data[1])


def _require_bridge_rng_tags() -> None:
    expected = {
        "_WORLD_ROOT_RNG_TAG": _WORLD_ROOT_RNG_TAG,
        "_LEARNER_ROOT_RNG_TAG": _LEARNER_ROOT_RNG_TAG,
        "_BEHAVIOR_RNG_TAG": _BEHAVIOR_RNG_TAG,
        "_GROUNDED_RNG_TAG": _GROUNDED_RNG_TAG,
        "_PLANNER_RNG_TAG": _PLANNER_RNG_TAG,
        "_INTERVENTION_RNG_TAG": _INTERVENTION_RNG_TAG,
    }
    for name, value in expected.items():
        if getattr(bridge_module, name, None) != value:
            raise HiddenLearningPartnerPlanningScanPlanError(
                f"bridge named RNG tag drifted: {name}"
            )


def _key_streams() -> tuple[HiddenPlanningKeyStream, ...]:
    return (
        HiddenPlanningKeyStream(
            name="world.cue",
            owner="world",
            derivation="learning_partner_world_keys(fold_in(root,0x4850574c)).cue",
            initializer_or_state_path="world.cue_key",
            lifetime="persistent_after_world_initial_cue_draw",
            transition_rule="one split and one next-cue draw per accepted transition",
        ),
        HiddenPlanningKeyStream(
            name="world.channel",
            owner="world",
            derivation="learning_partner_world_keys(fold_in(root,0x4850574c)).channel",
            initializer_or_state_path="world.channel_key",
            lifetime="persistent",
            transition_rule=(
                "one key advance every transition; shuffled output must bind to the "
                "key-derived channel output, without claiming an observed draw marker"
            ),
        ),
        HiddenPlanningKeyStream(
            name="learner.helper",
            owner="helper_role",
            derivation="signaling_bandit_keys(fold_in(root,0x48504c52)).helper",
            initializer_or_state_path="learner.helper.key",
            lifetime="persistent",
            transition_rule="jr.split(key,4)[0] committed every transition even when frozen",
        ),
        HiddenPlanningKeyStream(
            name="learner.beneficiary",
            owner="beneficiary_role",
            derivation="signaling_bandit_keys(fold_in(root,0x48504c52)).beneficiary",
            initializer_or_state_path="learner.beneficiary.key",
            lifetime="persistent",
            transition_rule="jr.split(key,4)[0] committed every transition even when frozen",
        ),
        HiddenPlanningKeyStream(
            name="behavior.initialization",
            owner="learner_behavior_model",
            derivation="fold_in(root,0x48504248)",
            initializer_or_state_path="BehaviorModel.init input only",
            lifetime="initialization_only_no_persistent_model_key",
            transition_rule="no transition RNG",
        ),
        HiddenPlanningKeyStream(
            name="grounded.initialization",
            owner="learner_grounded_world_model",
            derivation="fold_in(root,0x48504752)",
            initializer_or_state_path="GroundedJointWorldModel.init input only",
            lifetime="initialization_only_no_persistent_model_key",
            transition_rule="no transition RNG",
        ),
        HiddenPlanningKeyStream(
            name="planner",
            owner="learner_planner_tie_break",
            derivation="fold_in(root,0x4850504c)",
            initializer_or_state_path="planner_key",
            lifetime="persistent",
            transition_rule="jr.split(key,2)[0] committed every transition",
        ),
        HiddenPlanningKeyStream(
            name="intervention",
            owner="evaluator_randomized_planner_gate",
            derivation="fold_in(root,0x4850494e)",
            initializer_or_state_path="intervention_key",
            lifetime="persistent",
            transition_rule="jr.split(key,2)[0] committed every transition",
        ),
    )


def _seed_binding(seed_index: int, seed: int) -> HiddenPlanningSeedBinding:
    root = jr.key(seed, impl=_PRNG_IMPL)
    world = learning_partner_world_keys(jr.fold_in(root, _WORLD_ROOT_RNG_TAG))
    learner = signaling_bandit_keys(jr.fold_in(root, _LEARNER_ROOT_RNG_TAG))
    named = (
        ("world.cue", _key_words(world.cue)),
        ("world.channel", _key_words(world.channel)),
        ("learner.helper", _key_words(learner.helper)),
        ("learner.beneficiary", _key_words(learner.beneficiary)),
        ("behavior.initialization", _key_words(jr.fold_in(root, _BEHAVIOR_RNG_TAG))),
        ("grounded.initialization", _key_words(jr.fold_in(root, _GROUNDED_RNG_TAG))),
        ("planner", _key_words(jr.fold_in(root, _PLANNER_RNG_TAG))),
        ("intervention", _key_words(jr.fold_in(root, _INTERVENTION_RNG_TAG))),
    )
    return HiddenPlanningSeedBinding(
        seed_index=seed_index,
        seed=seed,
        root_key_data=_key_words(root),
        named_key_data=named,
    )


def _seed_contract() -> HiddenPlanningSeedContract:
    bindings = tuple(
        _seed_binding(index, seed) for index, seed in enumerate(PAIRED_DEVELOPMENT_SEEDS)
    )
    payload = {
        "namespace": DEVELOPMENT_SEED_NAMESPACE,
        "prng_impl": _PRNG_IMPL,
        "selection_method": "fixed_unrun_exploratory_constants_no_rng_selection_draw",
        "bindings": [dataclasses.asdict(binding) for binding in bindings],
        "paired_across_every_condition": True,
        "development_only": True,
        "held_out": False,
        "evidence_eligible": False,
        "executed": False,
    }
    return HiddenPlanningSeedContract(
        namespace=DEVELOPMENT_SEED_NAMESPACE,
        prng_impl=_PRNG_IMPL,
        selection_method="fixed_unrun_exploratory_constants_no_rng_selection_draw",
        bindings=bindings,
        paired_across_every_condition=True,
        development_only=True,
        held_out=False,
        evidence_eligible=False,
        executed=False,
        seed_manifest_sha256=_sha256_json(payload),
    )


def _resource_budget() -> HiddenLearningPartnerPlanningResourceBudget:
    live_schema = getattr(bridge_module, "HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA", None)
    if live_schema != _EXPECTED_BRIDGE_SCHEMA or HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA != (
        _EXPECTED_BRIDGE_SCHEMA
    ):
        raise HiddenLearningPartnerPlanningScanPlanError(
            "source kernel schema is not the exact development.v1 contract"
        )
    source_resources = tuple(
        getattr(bridge_module, name, None)
        for name in (
            "_EXPECTED_SIGNALING_BYTES",
            "_EXPECTED_BEHAVIOR_BYTES",
            "_EXPECTED_GROUNDED_BYTES",
            "_EXPECTED_WORLD_BYTES",
            "_EXPECTED_TOTAL_BYTES",
        )
    )
    if source_resources != _EXPECTED_SOURCE_RESOURCE_BYTES or any(
        type(value) is not int for value in source_resources
    ):
        raise HiddenLearningPartnerPlanningScanPlanError(
            "source kernel persistent resources are not exactly 80/48/108/32/321 bytes"
        )
    signaling, behavior, grounded, world, declared_total = source_resources
    learner_model = signaling + behavior + grounded
    metadata = 8 + 8 + 32 + 1 + 4
    total = learner_model + world + metadata
    if total != declared_total or total != 321:
        raise HiddenLearningPartnerPlanningScanPlanError(
            "source kernel persistent resources do not sum to exactly 321 bytes"
        )
    return HiddenLearningPartnerPlanningResourceBudget(
        signaling_state_nbytes=signaling,
        behavior_state_nbytes=behavior,
        grounded_state_nbytes=grounded,
        learner_model_state_nbytes=learner_model,
        world_state_nbytes=world,
        planner_key_nbytes=8,
        intervention_key_nbytes=8,
        config_token_nbytes=32,
        valid_nbytes=1,
        step_count_nbytes=4,
        metadata_state_nbytes=metadata,
        total_state_nbytes=total,
        replay_capacity=0,
        exact_tree_match=True,
    )


def _word_pair(value: int) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= (2**64 - 1):
        raise HiddenLearningPartnerPlanningScanPlanError(
            "exact child-clock total is outside the uint64 range"
        )
    return (value >> 32, value & (2**32 - 1))


def _exact_child_clocks(
    spec: HiddenPlanningConditionSpec,
    *,
    steps: int,
) -> tuple[HiddenPlanningExactChildClock, ...]:
    behavior_writes = steps if spec.behavior_write else 0
    grounded_writes = steps if spec.grounded_write else 0
    return (
        HiddenPlanningExactChildClock(
            name="behavior",
            words_state_path="behavior.step_words",
            telemetry_state_path="behavior.step_count",
            words_dtype="uint32",
            words_shape=(2,),
            initial_words=(0, 0),
            final_words=_word_pair(behavior_writes),
            initial_telemetry=0,
            final_telemetry=behavior_writes,
        ),
        HiddenPlanningExactChildClock(
            name="grounded",
            words_state_path="grounded.update_words",
            telemetry_state_path="grounded.update_count",
            words_dtype="uint32",
            words_shape=(2,),
            initial_words=(0, 0),
            final_words=_word_pair(grounded_writes),
            initial_telemetry=0,
            final_telemetry=grounded_writes,
        ),
    )


def _operation_totals(
    condition: str,
    *,
    steps: int,
) -> tuple[HiddenPlanningNamedOperation, ...]:
    spec = condition_spec(condition)
    per_transition = (
        ("bridge_step_calls", 1),
        ("world_step_with_delivery_calls", 1),
        ("helper_policy_select_calls", 1),
        ("beneficiary_executed_policy_select_calls", 1),
        ("beneficiary_potential_policy_select_calls", 2),
        ("behavior_candidate_symbol_predict_calls", 2),
        ("behavior_update_proposal_opportunities", 1),
        ("grounded_joint_cell_predict_calls", 4),
        ("grounded_update_proposal_opportunities", 1),
        ("helper_value_update_proposal_opportunities", 1),
        ("beneficiary_value_update_proposal_opportunities", 1),
        ("helper_value_committed_writes_on_required_valid_trace", int(spec.helper_write)),
        (
            "beneficiary_value_committed_writes_on_required_valid_trace",
            int(spec.beneficiary_write),
        ),
        (
            "behavior_model_committed_writes_on_required_valid_trace",
            int(spec.behavior_write),
        ),
        (
            "grounded_model_committed_writes_on_required_valid_trace",
            int(spec.grounded_write),
        ),
        ("planner_tie_random_draws", 1),
        ("planner_consumption_gate_draws", 1),
        ("world_cue_draws", 1),
        ("world_channel_key_advances", 1),
        ("helper_policy_key_advances", 1),
        ("beneficiary_policy_key_advances", 1),
        ("planner_key_advances", 1),
        ("intervention_key_advances", 1),
        ("delivered_potential_outcome_rows", 2),
        ("shuffled_channel_output_bindings", int(spec.channel == "shuffled")),
    )
    fixed = (
        ("bridge_initialize_calls", 1),
        ("world_initial_cue_draws", 1),
        ("bridge_initializer_resource_contract_checks", 1),
        ("runner_initial_resource_measurements", 1),
        ("runner_final_resource_measurements", 1),
        ("runner_raw_metric_reconstructions", 1),
        ("strict_run_validator_calls_required", 1),
        ("validator_raw_metric_reconstructions_required", 1),
    )
    records = [
        HiddenPlanningNamedOperation(
            name=name,
            fixed_per_run=count,
            per_transition=0,
            per_run_total=count,
        )
        for name, count in fixed
    ]
    records.extend(
        HiddenPlanningNamedOperation(
            name=name,
            fixed_per_run=0,
            per_transition=count,
            per_run_total=count * steps,
        )
        for name, count in per_transition
    )
    return tuple(records)


def _condition_family(condition: str) -> ArmFamily:
    return "primary" if condition in PRIMARY_CONDITIONS else "diagnostic"


def _condition_role(condition: str) -> ArmRole:
    if condition == JOINT_ADAPTIVE:
        return "reference"
    if condition in PRIMARY_CONDITIONS:
        return "primary_intervention"
    return "diagnostic"


def _contrast_id(condition: str) -> str | None:
    return None if condition == JOINT_ADAPTIVE else f"{condition}_vs_{JOINT_ADAPTIVE}"


def _field_names(dataclass_type: object) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in dataclasses.fields(dataclass_type)  # type: ignore[arg-type]
    )


def _build_arm(
    condition: str,
    *,
    serialization_index: int,
    config: HiddenLearningPartnerPlanningConfig,
    config_sha256: str,
    seed_manifest_sha256: str,
    resource_budget: HiddenLearningPartnerPlanningResourceBudget,
) -> HiddenPlanningArm:
    spec = condition_spec(condition)
    bridge = HiddenLearningPartnerPlanningBridge(
        config,
        cast(HiddenPlanningCondition, condition),
    )
    token = np.asarray(bridge.config_token, dtype=np.uint8)
    if token.shape != (32,):
        raise HiddenLearningPartnerPlanningScanPlanError("bridge config token is not uint8[32]")
    return HiddenPlanningArm(
        serialization_index=serialization_index,
        condition=condition,
        family=_condition_family(condition),
        role=_condition_role(condition),
        contrast_id=_contrast_id(condition),
        condition_spec=spec,
        config_sha256=config_sha256,
        config_token_hex=token.tobytes().hex(),
        seed_manifest_sha256=seed_manifest_sha256,
        resource_budget=resource_budget,
        exact_child_clocks=_exact_child_clocks(spec, steps=config.num_steps),
        named_operation_totals=_operation_totals(condition, steps=config.num_steps),
        evaluator_order_is_key_input=False,
        execution_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


def _contrasts() -> tuple[HiddenPlanningContrast, ...]:
    records: list[HiddenPlanningContrast] = []
    for condition in CANONICAL_CONDITION_ORDER:
        if condition == JOINT_ADAPTIVE:
            continue
        primary = condition in PRIMARY_CONDITIONS
        records.append(
            HiddenPlanningContrast(
                contrast_id=cast(str, _contrast_id(condition)),
                family="primary_causal" if primary else "diagnostic_only",
                reference_condition=JOINT_ADAPTIVE,
                intervention_condition=condition,
                difference_direction="intervention_minus_reference_per_paired_seed",
                interpretation=_INTERPRETATIONS[condition],
                requested_scalar_metric_fields=_PAIRED_DIFFERENCE_METRICS,
                phase_diagnostics_requested=True,
                causal_claim_scope=(
                    "finite_unexecuted_development_intervention_only"
                    if primary
                    else "diagnostic_no_primary_causal_claim"
                ),
                evidence_authorized=False,
                scientific_promotion_allowed=False,
            )
        )
    return tuple(records)


def _requested_outputs() -> HiddenPlanningRequestedOutputs:
    state_fields = _field_names(HiddenLearningPartnerPlanningState)
    trace_fields = _field_names(HiddenLearningPartnerPlanningTrace)
    metric_fields = tuple(
        field.name
        for field in dataclasses.fields(HiddenLearningPartnerPlanningMetrics)
        if field.name != "phase_diagnostics"
    )
    phase_fields = tuple(
        field.name for field in dataclasses.fields(HiddenLearningPartnerPhaseDiagnostics)
    )
    return HiddenPlanningRequestedOutputs(
        output_mode="future_in_memory_per_seed_condition_raw_records_only",
        record_key_fields=("seed", "condition"),
        initial_state_fields=state_fields,
        final_state_fields=state_fields,
        trace_fields=trace_fields,
        scalar_metric_fields=metric_fields,
        phase_diagnostic_fields=phase_fields,
        paired_difference_metric_fields=_PAIRED_DIFFERENCE_METRICS,
        resource_budget_requested=True,
        strict_run_validation_errors_requested=True,
        aggregate_statistics_requested=False,
        outcomes_present=False,
        thresholds_defined=False,
        artifact_output_requested=False,
    )


def _suite_operation_totals(
    arms: tuple[HiddenPlanningArm, ...],
    *,
    seed_count: int,
) -> tuple[tuple[str, int], ...]:
    totals: defaultdict[str, int] = defaultdict(int)
    order: list[str] = []
    for arm in arms:
        for operation in arm.named_operation_totals:
            if operation.name not in totals:
                order.append(operation.name)
            totals[operation.name] += operation.per_run_total * seed_count
    return tuple((name, totals[name]) for name in order)


def _counts(
    arms: tuple[HiddenPlanningArm, ...],
    contrasts: tuple[HiddenPlanningContrast, ...],
    config: HiddenLearningPartnerPlanningConfig,
    seed_contract: HiddenPlanningSeedContract,
    resource_budget: HiddenLearningPartnerPlanningResourceBudget,
) -> HiddenPlanningSuiteCounts:
    run_count = len(arms) * len(seed_contract.bindings)
    transitions = run_count * config.num_steps
    return HiddenPlanningSuiteCounts(
        primary_arm_count=len(PRIMARY_CONDITIONS),
        diagnostic_arm_count=len(DIAGNOSTIC_CONDITIONS),
        contrast_count=len(contrasts),
        paired_seed_count=len(seed_contract.bindings),
        planned_run_count=run_count,
        steps_per_run=config.num_steps,
        planned_transition_count=transitions,
        initial_state_record_count=run_count,
        final_state_record_count=run_count,
        trace_row_count=transitions,
        trace_fields_per_row=len(_field_names(HiddenLearningPartnerPlanningTrace)),
        metric_record_count=run_count,
        scalar_metric_fields_per_record=len(
            tuple(
                field.name
                for field in dataclasses.fields(HiddenLearningPartnerPlanningMetrics)
                if field.name != "phase_diagnostics"
            )
        ),
        phase_diagnostic_container_count=run_count,
        phase_rows_per_container=config.n_phases,
        phase_diagnostic_phase_row_count=run_count * config.n_phases,
        phase_diagnostic_field_count=len(dataclasses.fields(HiddenLearningPartnerPhaseDiagnostics)),
        state_fields_per_snapshot=len(_field_names(HiddenLearningPartnerPlanningState)),
        persistent_state_bytes_per_run=resource_budget.total_state_nbytes,
        summed_logical_persistent_state_bytes=(run_count * resource_budget.total_state_nbytes),
        suite_named_operation_totals=_suite_operation_totals(
            arms,
            seed_count=len(seed_contract.bindings),
        ),
        named_operation_accounting_scope=(
            "selected_bridge_runner_calls_static_write_masks_and_key_advances_not_"
            "flop_hlo_or_all_nested_primitives"
        ),
        flop_or_hlo_equivalence_claimed=False,
    )


def _plan_sha256(plan: HiddenLearningPartnerPlanningScanPlan) -> str:
    payload = dataclasses.asdict(plan)
    payload.pop("plan_sha256")
    return _sha256_json(payload)


def _build_canonical_plan() -> HiddenLearningPartnerPlanningScanPlan:
    if CANONICAL_CONDITION_ORDER != tuple(MATCHED_CONDITIONS):
        raise HiddenLearningPartnerPlanningScanPlanError(
            "scan condition order differs from the bridge's complete matched surface"
        )
    _require_bridge_rng_tags()
    config = HiddenLearningPartnerPlanningConfig()
    if config.num_steps != 3_072:
        raise HiddenLearningPartnerPlanningScanPlanError(
            "canonical hidden planning life is no longer exactly 3,072 transitions"
        )
    config_digest = _config_sha256(config)
    streams = _key_streams()
    key_manifest = _sha256_json([dataclasses.asdict(stream) for stream in streams])
    seeds = _seed_contract()
    resource = _resource_budget()
    arms = tuple(
        _build_arm(
            condition,
            serialization_index=index,
            config=config,
            config_sha256=config_digest,
            seed_manifest_sha256=seeds.seed_manifest_sha256,
            resource_budget=resource,
        )
        for index, condition in enumerate(CANONICAL_CONDITION_ORDER)
    )
    contrasts = _contrasts()
    requested_outputs = _requested_outputs()
    common_random_numbers = HiddenPlanningCommonRandomNumbers(
        same_seed_set_every_arm=True,
        same_root_key_for_seed_every_arm=True,
        condition_is_key_derivation_input=False,
        arm_order_is_key_derivation_input=False,
        fresh_state_per_seed_condition=True,
        cross_arm_state_reuse_allowed=False,
        allowed_initial_state_difference_fields=("config_token",),
        required_equal_named_key_streams=tuple(stream.name for stream in streams),
        required_cross_arm_trace_key_fields=(
            "helper_key_before",
            "helper_key_after",
            "beneficiary_key_before",
            "beneficiary_key_after",
            "planner_key_before",
            "planner_key_after",
            "intervention_key_before",
            "intervention_key_after",
        ),
        world_cue_stream_reconstruction_required=True,
        world_channel_stream_reconstruction_required=True,
        branch_invariant_persistent_key_advancement_required=True,
        shuffled_channel_output_binding_required=True,
        cross_arm_rng_audit_implemented=True,
        result_join_key=("seed", "condition"),
        serialization_order_semantics=(
            "canonical_manifest_order_only_never_a_key_state_or_result_identity_input"
        ),
    )
    readiness = HiddenPlanningReadiness(
        plan_validator_implemented=True,
        canonical_bridge_conditions_bound=True,
        paired_seed_and_key_contract_complete=True,
        named_operation_accounting_complete_in_declared_scope=True,
        ready_for_runner_implementation=False,
        suite_runner_implemented=True,
        cross_arm_rng_audit_implemented=True,
        execution_request_and_permit_implemented=True,
        authenticated_source_replay_implemented=True,
        default_life_executed=False,
        outcomes_present=False,
        quiescence_required=True,
        quiescence_checked=False,
        quiescence_verified=False,
        ready_for_execution=False,
        blockers=(
            "exact_authenticated_execution_permit_not_issued",
            "external_host_quiescence_not_verified_live",
        ),
        quiescence_declaration=(
            "execution requires a fresh strict live load/runnable-process check bound into "
            "an in-process permit; a prose confirmation is insufficient"
        ),
    )
    provisional = HiddenLearningPartnerPlanningScanPlan(
        schema=HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_SCHEMA,
        status=HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_STATUS,
        bridge_schema=HIDDEN_LEARNING_PARTNER_PLANNING_SCHEMA,
        development_only=True,
        execution_authorized=False,
        runner_authorized=False,
        campaign_authorized=False,
        artifact_writes_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        config=config,
        config_sha256=config_digest,
        life_steps=config.num_steps,
        key_streams=streams,
        key_manifest_sha256=key_manifest,
        seed_contract=seeds,
        common_random_numbers=common_random_numbers,
        resource_budget=resource,
        resource_budget_sha256=_resource_sha256(resource),
        arms=arms,
        contrasts=contrasts,
        requested_outputs=requested_outputs,
        counts=_counts(arms, contrasts, config, seeds, resource),
        readiness=readiness,
        thresholds=None,
        outcomes=None,
        artifact_output_path=None,
        plan_sha256="",
    )
    return dataclasses.replace(provisional, plan_sha256=_plan_sha256(provisional))


def build_hidden_learning_partner_planning_scan_plan() -> HiddenLearningPartnerPlanningScanPlan:
    """Build the canonical declaration without initializing or stepping a life."""

    return _build_canonical_plan()


def validate_hidden_learning_partner_planning_scan_plan(plan: object) -> tuple[str, ...]:
    """Return every detected contract error without executing any bridge step."""

    if type(plan) is not HiddenLearningPartnerPlanningScanPlan:
        return ("plan must be an exact HiddenLearningPartnerPlanningScanPlan",)
    checked = plan
    canonical = _build_canonical_plan()
    errors: list[str] = []

    if checked.schema != canonical.schema or checked.bridge_schema != canonical.bridge_schema:
        errors.append("plan or bridge schema differs from the canonical binding")
    if checked.status != canonical.status:
        errors.append("plan status differs from the nonexecuting readiness status")
    authority = (
        checked.execution_authorized,
        checked.runner_authorized,
        checked.campaign_authorized,
        checked.artifact_writes_authorized,
        checked.evidence_authorized,
        checked.scientific_promotion_allowed,
    )
    if checked.development_only is not True or any(value is not False for value in authority):
        errors.append("plan carries execution, artifact, evidence, or promotion authority")
    if checked.thresholds is not None:
        errors.append("thresholds are forbidden in the development scan plan")
    if checked.outcomes is not None:
        errors.append("outcomes are forbidden in the nonexecuting scan plan")
    if checked.artifact_output_path is not None:
        errors.append("artifact output paths are forbidden in the scan plan")

    if type(checked.config) is not HiddenLearningPartnerPlanningConfig:
        errors.append("plan config has the wrong concrete type")
    elif (
        checked.config != canonical.config
        or checked.config_sha256 != _config_sha256(checked.config)
        or checked.life_steps != checked.config.num_steps
    ):
        errors.append("config or exact life length differs from the canonical default")

    if type(checked.arms) is not tuple:
        errors.append("arms must be an exact tuple")
        arms: tuple[object, ...] = ()
    else:
        arms = cast(tuple[object, ...], checked.arms)
    conditions = tuple(arm.condition for arm in arms if type(arm) is HiddenPlanningArm)
    if len(conditions) != len(arms):
        errors.append("every arm must have the exact HiddenPlanningArm type")
    if len(set(conditions)) != len(conditions):
        errors.append("scan plan contains duplicate condition arms")
    unsupported = tuple(
        condition for condition in conditions if condition not in MATCHED_CONDITIONS
    )
    if unsupported:
        errors.append("scan plan contains unsupported bridge conditions")
    missing = tuple(condition for condition in MATCHED_CONDITIONS if condition not in conditions)
    if missing:
        errors.append("scan plan is missing canonical bridge conditions")
    if conditions != CANONICAL_CONDITION_ORDER:
        errors.append("arm order differs; evaluator order must remain serialization-only")
    canonical_arms = {arm.condition: arm for arm in canonical.arms}
    for arm_object in arms:
        if type(arm_object) is not HiddenPlanningArm:
            continue
        arm = arm_object
        expected = canonical_arms.get(arm.condition)
        if expected is None:
            continue
        if arm != expected:
            errors.append(f"arm binding differs from canonical condition: {arm.condition}")
        if arm.exact_child_clocks != expected.exact_child_clocks:
            errors.append(f"arm exact child-clock binding differs: {arm.condition}")
        if arm.evaluator_order_is_key_input is not False:
            errors.append(f"arm order influences key derivation: {arm.condition}")
        if (
            arm.resource_budget != checked.resource_budget
            or arm.config_sha256 != checked.config_sha256
            or arm.seed_manifest_sha256 != checked.seed_contract.seed_manifest_sha256
        ):
            errors.append(f"arm resource/config/seed pairing mismatch: {arm.condition}")
        if any(
            value is not False
            for value in (
                arm.execution_authorized,
                arm.evidence_authorized,
                arm.scientific_promotion_allowed,
            )
        ):
            errors.append(f"arm carries forbidden authority: {arm.condition}")

    if checked.contrasts != canonical.contrasts:
        errors.append("primary causal contrasts or diagnostic declarations differ")
    contrast_conditions = tuple(contrast.intervention_condition for contrast in checked.contrasts)
    if len(set(contrast_conditions)) != len(contrast_conditions):
        errors.append("contrast declarations contain duplicate intervention arms")
    if set(contrast_conditions) != set(CANONICAL_CONDITION_ORDER[1:]):
        errors.append("contrast declarations are missing or add conditions")
    if any(
        contrast.evidence_authorized is not False
        or contrast.scientific_promotion_allowed is not False
        for contrast in checked.contrasts
    ):
        errors.append("a contrast carries evidence or promotion authority")

    if checked.seed_contract != canonical.seed_contract:
        errors.append("paired development seed or named key ownership contract differs")
    seeds = tuple(binding.seed for binding in checked.seed_contract.bindings)
    if len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        errors.append("paired development seeds must be unique non-negative integers")
    if checked.key_streams != canonical.key_streams:
        errors.append("named key-stream ownership differs")
    if checked.key_manifest_sha256 != canonical.key_manifest_sha256:
        errors.append("named key-stream manifest digest differs")
    if checked.common_random_numbers != canonical.common_random_numbers:
        errors.append("common-random-number or evaluator-order requirements differ")

    if (
        checked.resource_budget != canonical.resource_budget
        or checked.resource_budget_sha256 != _resource_sha256(checked.resource_budget)
    ):
        errors.append("persistent resource contract differs")
    if checked.requested_outputs != canonical.requested_outputs:
        errors.append("threshold-free raw output request differs")
    if checked.counts != canonical.counts:
        errors.append("transition/state/trace or named operation totals differ")
    if checked.readiness != canonical.readiness:
        errors.append("readiness or quiescence declaration differs")
    if (
        checked.readiness.ready_for_execution is not False
        or checked.readiness.quiescence_verified is not False
        or checked.readiness.default_life_executed is not False
    ):
        errors.append("readiness falsely claims execution, quiescence, or outcomes")

    try:
        actual_digest = _plan_sha256(checked)
    except (TypeError, ValueError) as exc:
        errors.append(f"plan cannot be canonically hashed: {exc}")
    else:
        if checked.plan_sha256 != actual_digest:
            errors.append("plan digest does not bind its complete contents")
    if checked != canonical:
        errors.append("plan differs from the complete canonical scan declaration")
    return tuple(dict.fromkeys(errors))


def require_valid_hidden_learning_partner_planning_scan_plan(
    plan: object,
) -> HiddenLearningPartnerPlanningScanPlan:
    """Return an exact plan or fail closed; this does not authorize execution."""

    errors = validate_hidden_learning_partner_planning_scan_plan(plan)
    if errors:
        raise HiddenLearningPartnerPlanningScanPlanError("; ".join(errors))
    return cast(HiddenLearningPartnerPlanningScanPlan, plan)


__all__ = [
    "ARTIFACT_WRITES_AUTHORIZED",
    "CAMPAIGN_AUTHORIZED",
    "CANONICAL_CONDITION_ORDER",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SEED_NAMESPACE",
    "DIAGNOSTIC_CONDITIONS",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_SCHEMA",
    "HIDDEN_LEARNING_PARTNER_PLANNING_SCAN_PLAN_STATUS",
    "HiddenLearningPartnerPlanningScanPlan",
    "HiddenLearningPartnerPlanningScanPlanError",
    "HiddenPlanningArm",
    "HiddenPlanningCommonRandomNumbers",
    "HiddenPlanningContrast",
    "HiddenPlanningExactChildClock",
    "HiddenPlanningKeyStream",
    "HiddenPlanningNamedOperation",
    "HiddenPlanningReadiness",
    "HiddenPlanningRequestedOutputs",
    "HiddenPlanningSeedBinding",
    "HiddenPlanningSeedContract",
    "HiddenPlanningSuiteCounts",
    "PAIRED_DEVELOPMENT_SEEDS",
    "PRIMARY_CONDITIONS",
    "RUNNER_AUTHORIZED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "build_hidden_learning_partner_planning_scan_plan",
    "require_valid_hidden_learning_partner_planning_scan_plan",
    "validate_hidden_learning_partner_planning_scan_plan",
]
