"""Fixed-seed, development-only probe of natural generated-D birth feasibility.

This module does **not** execute the v0 D-mapping-twin contract.  That contract
deliberately has no runner authority.  Instead, this module defines a fresh,
strictly narrower permit for one in-memory descriptive probe.  The permit binds
the canonical read-only dataset, the canonical full-lifecycle learner, seed 101,
the reference and D-mapping-never-seen arms, a fixed endpoint at the end of the
twin's first true-D phase, and exact dependency source hashes.

Every learner transition is independently source-replayed and attached to the
schema-v4 birth-identity ledger.  The probe never injects a feature or lineage,
chooses a curation slot, changes a learner threshold/configuration, searches a
seed/configuration, writes an artifact, or grants evidence/promotion authority.
Its result is a threshold-free description of natural structural births and
prequential use, including an honest zero-birth outcome if that is what occurs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Final

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core import compositional_features as compositional_module
from alberta_framework.core.compositional_features import (
    CompositionalCurationTrace,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
    CompositionalFeatureUpdateResult,
)
from alberta_framework.evaluation import (
    generated_birth_identity_ledger as ledger_module,
)
from alberta_framework.evaluation import (
    generated_birth_identity_scrub_epoch as scrub_epoch_module,
)
from alberta_framework.evaluation import (
    generated_birth_identity_trace_binding as trace_binding_module,
)
from alberta_framework.evaluation import (
    generated_class_d_mapping_twin as d_mapping_module,
)
from alberta_framework.evaluation import (
    generated_class_lifecycle_scrub as lifecycle_scrub_module,
)
from alberta_framework.evaluation import (
    generated_class_recurrence as recurrence_module,
)
from alberta_framework.evaluation import (
    generated_expression_lineage as lineage_module,
)
from alberta_framework.evaluation.generated_birth_identity_ledger import (
    GeneratedBirthIdentityLedgerV4Config,
    GeneratedBirthIdentityLedgerV4State,
)
from alberta_framework.evaluation.generated_birth_identity_scrub_epoch import (
    generated_birth_identity_scrub_epoch_core_state_sha256,
)
from alberta_framework.evaluation.generated_birth_identity_trace_binding import (
    GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA,
    GeneratedBirthIdentityTraceBinding,
    attach_generated_birth_identity_ledger_at_core_genesis,
    authenticate_generated_birth_identity_trace_by_source_replay,
)
from alberta_framework.evaluation.generated_class_d_mapping_twin import (
    D_MAPPING_NEVER_SEEN_TWIN,
    build_d_mapping_never_seen_contract,
    build_d_mapping_twin_dataset,
)
from alberta_framework.evaluation.generated_class_lifecycle_scrub import (
    persistent_compositional_state_nbytes,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    FULL_LIFECYCLE,
    GeneratedClassRecurrenceV0Protocol,
    GeneratedExpression,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_learner,
    derive_expression_manifest,
)
from alberta_framework.evaluation.generated_expression_lineage import (
    ExpandedExpressionLineageConfig,
    compile_expanded_expression_lineage_masks,
)

GENERATED_NATURAL_D_BIRTH_PROBE_SCHEMA: Final = (
    "alberta.generated-natural-d-birth-feasibility.development.v0"
)
GENERATED_NATURAL_D_BIRTH_PROBE_STATUS: Final = (
    "FIXED_SEED_IN_MEMORY_DEVELOPMENT_PROBE_NO_EVIDENCE_OR_PROMOTION"
)
GENERATED_NATURAL_D_BIRTH_RESULT_SCHEMA: Final = (
    "alberta.generated-natural-d-birth-feasibility-result.development.v0"
)
GENERATED_NATURAL_D_BIRTH_RESULT_STATUS: Final = (
    "DESCRIPTIVE_FIXED_SEED_RESULT_NO_ACCEPTANCE_THRESHOLD"
)

PROBE_SEED: Final = 101
PROBE_ARM_ORDER: Final = (FULL_LIFECYCLE, D_MAPPING_NEVER_SEEN_TWIN)
PROBE_STOP_POST_STEP: Final = 3_928
LEARNER_KEY_DOMAIN: Final = 0x4C524E52
THREEFRY_IMPLEMENTATION: Final = "threefry2x32"
LEDGER_NAMESPACE_PREFIX: Final = "generated-natural-d-birth-feasibility-v0"

OUTCOME_NO_NATURAL_D_BIRTH: Final = "VALID_DESCRIPTIVE_REJECTION_NO_NATURAL_D_BIRTH"
OUTCOME_CANDIDATE_ONLY: Final = "NATURAL_D_CANDIDATE_BIRTH_WITHOUT_ACTIVE_USE"
OUTCOME_ACTIVE_HEAD_USE: Final = "NATURAL_D_ACTIVE_BIRTH_AND_PREQUENTIAL_HEAD_USE"
OUTCOME_ACTIVE_CONTRIBUTION: Final = (
    "NATURAL_D_ACTIVE_BIRTH_AND_NONZERO_PREQUENTIAL_CONTRIBUTION"
)

_PINNED_UPSTREAM_CONTRACT_SHA256: Final = (
    "0d95ea1022aefcc334479b973133ba53443b716f32e48515dba39b6a2bc617bc"
)
_PINNED_OBSERVATION_SHA256: Final = (
    "ab94bf71dca223c504f2c14f80fcbfcf57e590be6cd8448fb395462a0cb9dba7"
)
_PINNED_REFERENCE_TARGET_SHA256: Final = (
    "129c75eb3cf96031e19ed9bb6654577233b349e308fd39ef46d440b9662e9f4f"
)
_PINNED_TWIN_TARGET_SHA256: Final = (
    "3c5242269e4e71b639210132d224c8dc8d395af7048de02dec12ed0e6f580593"
)
_PINNED_SOURCE_SHA256: Final = (
    (
        "alberta_framework.core.compositional_features",
        "e8024b45c3585f28799616dc0220ee3bbc61c75f5a1ca2fa427caf75181ac402",
    ),
    (
        "alberta_framework.evaluation.generated_birth_identity_ledger",
        "1008dc090d54d4a776e2681bbbaf8f20c01999839b1d7879137e00f728e85cdb",
    ),
    (
        "alberta_framework.evaluation.generated_birth_identity_scrub_epoch",
        "57337a73b47140f149f2afbc382c8bc0f7b6316361c99c8eba37172f69d4150c",
    ),
    (
        "alberta_framework.evaluation.generated_birth_identity_trace_binding",
        "520194f7a92b023a366249bd1d23583209fd210b3f02862bfb50b5be11bae10d",
    ),
    (
        "alberta_framework.evaluation.generated_class_d_mapping_twin",
        "47a25eeeac09ac0b34b32ce6e897b8859e63c77246bde544ddd96accf7d03593",
    ),
    (
        "alberta_framework.evaluation.generated_class_lifecycle_scrub",
        "d389ea4294ab354a47af010d76914c3f4e6e87f231b9ccd4101dd1e40741dba9",
    ),
    (
        "alberta_framework.evaluation.generated_class_recurrence",
        "fd2aec43d2afc97e63c686b6d008d279315d5017689c6b23ce47b94ab7da075e",
    ),
    (
        "alberta_framework.evaluation.generated_expression_lineage",
        "06933ca3843ad507b6ff5873da5c832cd3f150efcf4aa84a99d19ea53bd167f8",
    ),
)


class GeneratedNaturalDBirthProbeError(RuntimeError):
    """A fixed probe authority, pairing, transition, or resource check failed."""


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDBirthProbePermit:
    """Fresh authority for exactly one fixed, in-memory descriptive probe."""

    schema: str
    status: str
    development_only: bool
    executes_upstream_d_mapping_v0_contract: bool
    upstream_d_mapping_v0_execution_authorized: bool
    upstream_d_mapping_v0_runner_authorized: bool
    canonical_dataset_reused_read_only: bool
    fixed_in_memory_execution_authorized: bool
    fixed_source_replay_authorized: bool
    artifact_writes_authorized: bool
    outputs_authorized: bool
    search_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    fixed_seed_uint32: int
    fixed_arm_order: tuple[str, str]
    fixed_learner_control_name: str
    fixed_stop_post_step: int
    stop_is_complete_twin_first_true_d_exposure: bool
    learner_visible_fields: tuple[str, str]
    evaluator_only_fields: tuple[str, ...]
    feature_or_lineage_injection_authorized: bool
    evaluator_chosen_curation_slot_authorized: bool
    learner_config_change_authorized: bool
    upstream_contract_sha256: str
    observation_sha256: str
    reference_target_sha256: str
    twin_target_sha256: str
    learner_config_sha256: str
    dependency_source_sha256: tuple[tuple[str, str], ...]
    expected_transactions_per_arm: int
    expected_curation_opportunities_per_arm: int
    wall_clock_threshold: float | None
    permit_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDStructuralSnapshot:
    """Exact roots plus expanded lineages and their current birth identities."""

    post_step: int
    phase_index: int
    phase_label: str
    active_exact_root_count: int
    candidate_exact_root_count: int
    active_expanded_mask_count: int
    candidate_expanded_mask_count: int
    active_target_subtree_occurrences: int
    candidate_target_subtree_occurrences: int
    active_expanded_birth_identities: tuple[tuple[int, str], ...]
    candidate_expanded_birth_identities: tuple[tuple[int, str], ...]
    lineage_plan_sha256: str
    core_state_sha256: str
    ledger_state_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDCurationTransaction:
    """One natural production curation opportunity and its authenticated state."""

    post_step: int
    zero_based_step_index: int
    phase_index: int
    phase_label: str
    should_try_replace: bool
    has_event: bool
    proposal_count: int
    root_change_count: int
    promotion_count: int
    cascade_refill_count: int
    candidate_refresh_count: int
    candidate_rebound_count: int
    candidate_overdepth_regeneration_count: int
    logical_event_count: int
    applied_identity_event_count: int
    transaction_sha256: str
    snapshot: GeneratedNaturalDStructuralSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDFirstUse:
    """First true-D row where an active expanded lineage participates."""

    use_kind: str
    zero_based_step_index: int
    phase_index: int
    phase_label: str
    true_d_exposure_ordinal: int
    active_exact_root_count: int
    active_expanded_mask_count: int
    active_expanded_birth_identities: tuple[tuple[int, str], ...]
    nonzero_head_slot_count: int
    nonzero_contribution_slot_count: int
    total_contribution_float32_hex: str
    pre_core_state_sha256: str
    pre_ledger_state_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDBirthArmResult:
    """One fixed arm's authenticated natural structural and use history."""

    arm_name: str
    learner_control_name: str
    stream_target_sha256: str
    learner_config_sha256: str
    initial_core_state_sha256: str
    final_core_state_sha256: str
    initial_persistent_core_state_nbytes: int
    final_persistent_core_state_nbytes: int
    initial_persistent_ledger_state_nbytes: int
    final_persistent_ledger_state_nbytes: int
    authenticated_transaction_count: int
    source_replay_authenticated_transaction_count: int
    transaction_sha256s: tuple[str, ...]
    curation_opportunity_count: int
    curation_event_count: int
    curation_transactions: tuple[GeneratedNaturalDCurationTransaction, ...]
    total_applied_identity_event_count: int
    true_d_exposure_count: int
    deranged_d_exposure_count: int
    exact_any_birth_post_steps: tuple[int, ...]
    exact_active_count_increase_post_steps: tuple[int, ...]
    exact_candidate_count_increase_post_steps: tuple[int, ...]
    exact_active_count_decrease_post_steps: tuple[int, ...]
    exact_candidate_count_decrease_post_steps: tuple[int, ...]
    expanded_any_birth_post_steps: tuple[int, ...]
    expanded_active_identity_entry_post_steps: tuple[int, ...]
    expanded_candidate_identity_entry_post_steps: tuple[int, ...]
    expanded_active_identity_exit_post_steps: tuple[int, ...]
    expanded_candidate_identity_exit_post_steps: tuple[int, ...]
    first_prequential_head_use: GeneratedNaturalDFirstUse | None
    first_nonzero_prequential_contribution: GeneratedNaturalDFirstUse | None
    final_snapshot: GeneratedNaturalDStructuralSnapshot
    final_step_words_uint32: tuple[int, int]
    final_ledger_step_words_uint32: tuple[int, int]
    final_learner_key_words_uint32: tuple[int, int]
    every_transaction_source_replay_authenticated: bool
    persistent_capacity_unchanged: bool
    artifacts_written: int
    evidence_authorized: bool
    promotion_authorized: bool
    descriptive_outcome: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedNaturalDBirthProbeResult:
    """Paired fixed-seed result with no acceptance or promotion interpretation."""

    schema: str
    status: str
    permit: GeneratedNaturalDBirthProbePermit
    arm_results: tuple[GeneratedNaturalDBirthArmResult, ...]
    observation_prefix_sha256: str
    first_d_target_bit_mismatch_count: int
    non_first_d_target_bit_mismatch_count: int
    initial_core_state_bit_exact_across_arms: bool
    learner_key_bit_exact_across_arms_every_step: bool
    learner_visible_field_schema_equal_across_arms: bool
    raw_observation_values_bit_exact_across_arms: bool
    target_value_differences_confined_to_first_d: bool
    paired_steps_executed: int
    total_authenticated_transactions: int
    artifact_bytes_written: int
    thresholds_applied: int
    searches_executed: int
    evidence_authorized: bool
    promotion_authorized: bool
    result_sha256: str


@dataclasses.dataclass(slots=True)
class _ArmRuntime:
    name: str
    learner: CompositionalFeatureLearner
    state: CompositionalFeatureState
    ledger_config: GeneratedBirthIdentityLedgerV4Config
    ledger_state: GeneratedBirthIdentityLedgerV4State
    target_rows: Array
    target_sha256: str
    initial_core_sha256: str
    initial_core_nbytes: int
    initial_ledger_nbytes: int
    snapshot: GeneratedNaturalDStructuralSnapshot
    transaction_sha256s: list[str]
    curation_transactions: list[GeneratedNaturalDCurationTransaction]
    identity_event_count: int
    true_d_exposure_count: int
    deranged_d_exposure_count: int
    exact_any_birth_steps: list[int]
    exact_active_increase_steps: list[int]
    exact_candidate_increase_steps: list[int]
    exact_active_decrease_steps: list[int]
    exact_candidate_decrease_steps: list[int]
    expanded_any_birth_steps: list[int]
    expanded_active_entry_steps: list[int]
    expanded_candidate_entry_steps: list[int]
    expanded_active_exit_steps: list[int]
    expanded_candidate_exit_steps: list[int]
    first_head_use: GeneratedNaturalDFirstUse | None
    first_contribution_use: GeneratedNaturalDFirstUse | None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(module: ModuleType) -> str:
    source = inspect.getsourcefile(module)
    if source is None:
        raise GeneratedNaturalDBirthProbeError("a pinned dependency has no source path")
    path = Path(source).resolve()
    if not path.is_file():
        raise GeneratedNaturalDBirthProbeError("a pinned dependency source is not a file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _live_dependency_hashes() -> tuple[tuple[str, str], ...]:
    modules = (
        compositional_module,
        ledger_module,
        scrub_epoch_module,
        trace_binding_module,
        d_mapping_module,
        lifecycle_scrub_module,
        recurrence_module,
        lineage_module,
    )
    return tuple((module.__name__, _file_sha256(module)) for module in modules)


def _config_sha256(learner: CompositionalFeatureLearner) -> str:
    return _sha256(learner.to_config())


def _permit_payload(permit: GeneratedNaturalDBirthProbePermit) -> dict[str, object]:
    payload = dataclasses.asdict(permit)
    payload.pop("permit_sha256")
    return payload


def _permit_sha256(permit: GeneratedNaturalDBirthProbePermit) -> str:
    return _sha256(_permit_payload(permit))


def _canonical_permit() -> GeneratedNaturalDBirthProbePermit:
    protocol = build_generated_class_recurrence_v0_protocol()
    upstream = build_d_mapping_never_seen_contract()
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    if upstream.contract_sha256 != _PINNED_UPSTREAM_CONTRACT_SHA256:
        raise GeneratedNaturalDBirthProbeError("upstream D-mapping contract hash drifted")
    if upstream.second_d_stop != PROBE_STOP_POST_STEP:
        raise GeneratedNaturalDBirthProbeError("fixed probe endpoint is no longer second-D stop")
    permit = GeneratedNaturalDBirthProbePermit(
        schema=GENERATED_NATURAL_D_BIRTH_PROBE_SCHEMA,
        status=GENERATED_NATURAL_D_BIRTH_PROBE_STATUS,
        development_only=True,
        executes_upstream_d_mapping_v0_contract=False,
        upstream_d_mapping_v0_execution_authorized=False,
        upstream_d_mapping_v0_runner_authorized=False,
        canonical_dataset_reused_read_only=True,
        fixed_in_memory_execution_authorized=True,
        fixed_source_replay_authorized=True,
        artifact_writes_authorized=False,
        outputs_authorized=False,
        search_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        fixed_seed_uint32=PROBE_SEED,
        fixed_arm_order=PROBE_ARM_ORDER,
        fixed_learner_control_name=FULL_LIFECYCLE,
        fixed_stop_post_step=PROBE_STOP_POST_STEP,
        stop_is_complete_twin_first_true_d_exposure=True,
        learner_visible_fields=("raw_features", "target"),
        evaluator_only_fields=(
            "phase_index",
            "phase_label",
            "true_d_exposure_ordinal",
            "structural_identity_audit",
        ),
        feature_or_lineage_injection_authorized=False,
        evaluator_chosen_curation_slot_authorized=False,
        learner_config_change_authorized=False,
        upstream_contract_sha256=upstream.contract_sha256,
        observation_sha256=_PINNED_OBSERVATION_SHA256,
        reference_target_sha256=_PINNED_REFERENCE_TARGET_SHA256,
        twin_target_sha256=_PINNED_TWIN_TARGET_SHA256,
        learner_config_sha256=_config_sha256(learner),
        dependency_source_sha256=_PINNED_SOURCE_SHA256,
        expected_transactions_per_arm=PROBE_STOP_POST_STEP,
        expected_curation_opportunities_per_arm=(
            PROBE_STOP_POST_STEP
            // protocol.curation_opportunity_audit.curation_interval
        ),
        wall_clock_threshold=None,
        permit_sha256="",
    )
    return dataclasses.replace(permit, permit_sha256=_permit_sha256(permit))


def build_generated_natural_d_birth_probe_permit() -> GeneratedNaturalDBirthProbePermit:
    """Build the fresh fixed-seed permit without constructing or running a learner life."""

    return validate_generated_natural_d_birth_probe_permit(_canonical_permit())


def validate_generated_natural_d_birth_probe_permit(
    permit: GeneratedNaturalDBirthProbePermit,
) -> GeneratedNaturalDBirthProbePermit:
    """Reject any authority, source, input, seed, arm, or endpoint drift."""

    if type(permit) is not GeneratedNaturalDBirthProbePermit:
        raise TypeError("permit must be an exact GeneratedNaturalDBirthProbePermit")
    canonical = _canonical_permit()
    if permit != canonical:
        raise ValueError("natural-D birth probe permit is not the exact canonical permit")
    if permit.permit_sha256 != _permit_sha256(permit):
        raise ValueError("natural-D birth probe permit hash does not reconstruct")
    if _live_dependency_hashes() != permit.dependency_source_sha256:
        raise ValueError("natural-D birth probe dependency source hashes drifted")
    if (
        permit.executes_upstream_d_mapping_v0_contract
        or permit.upstream_d_mapping_v0_execution_authorized
        or permit.upstream_d_mapping_v0_runner_authorized
    ):
        raise ValueError("the fresh probe cannot claim to execute the upstream v0 contract")
    if not (
        permit.development_only
        and permit.canonical_dataset_reused_read_only
        and permit.fixed_in_memory_execution_authorized
        and permit.fixed_source_replay_authorized
    ):
        raise ValueError("the narrow in-memory development authority is incomplete")
    if any(
        (
            permit.artifact_writes_authorized,
            permit.outputs_authorized,
            permit.search_authorized,
            permit.threshold_authorized,
            permit.evidence_authorized,
            permit.scientific_promotion_allowed,
            permit.feature_or_lineage_injection_authorized,
            permit.evaluator_chosen_curation_slot_authorized,
            permit.learner_config_change_authorized,
        )
    ):
        raise ValueError("the natural-D birth permit grants forbidden authority")
    return permit


def _key_words(key: Array) -> tuple[int, int]:
    if not isinstance(key, Array) or not jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        key.dtype,
        jax.dtypes.prng_key,
    ):
        raise GeneratedNaturalDBirthProbeError("learner key is not a typed JAX key")
    if str(jr.key_impl(key)) != THREEFRY_IMPLEMENTATION:
        raise GeneratedNaturalDBirthProbeError("learner key implementation drifted")
    words = np.asarray(jr.key_data(key))
    if words.shape != (2,) or words.dtype != np.uint32:
        raise GeneratedNaturalDBirthProbeError("learner key does not contain uint32[2]")
    return int(words[0]), int(words[1])


def _words(value: object, *, name: str) -> tuple[int, int]:
    array = np.asarray(value)
    if array.shape != (2,) or array.dtype != np.uint32:
        raise GeneratedNaturalDBirthProbeError(f"{name} must be exact uint32[2]")
    return int(array[0]), int(array[1])


def _int_scalar(value: object, *, name: str) -> int:
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise GeneratedNaturalDBirthProbeError(f"{name} must be an integer scalar")
    return int(array)


def _bool_scalar(value: object, *, name: str) -> bool:
    array = np.asarray(value)
    if array.shape != () or array.dtype != np.bool_:
        raise GeneratedNaturalDBirthProbeError(f"{name} must be a bool scalar")
    return bool(array)


def _phase_coordinates(
    protocol: GeneratedClassRecurrenceV0Protocol,
    zero_based_step_index: int,
) -> tuple[int, str]:
    cursor = 0
    for phase_index, (label, length) in enumerate(
        zip(protocol.phase_order, protocol.phase_lengths, strict=True)
    ):
        if cursor <= zero_based_step_index < cursor + length:
            return phase_index, label
        cursor += length
    raise GeneratedNaturalDBirthProbeError("step index is outside the canonical schedule")


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


def _selected_identities(
    identities: object,
    mask: Array,
    *,
    slots: int,
    name: str,
) -> tuple[tuple[int, str], ...]:
    identity_array = np.asarray(identities)
    mask_array = np.asarray(mask)
    if identity_array.shape != (slots, 32) or identity_array.dtype != np.uint8:
        raise GeneratedNaturalDBirthProbeError(f"{name} identity bank is malformed")
    if mask_array.shape != (slots,) or mask_array.dtype != np.bool_:
        raise GeneratedNaturalDBirthProbeError(f"{name} lineage mask is malformed")
    return tuple(
        (slot, np.ascontiguousarray(identity_array[slot]).tobytes().hex())
        for slot in range(slots)
        if bool(mask_array[slot])
    )


def _snapshot(
    state: CompositionalFeatureState,
    ledger_state: GeneratedBirthIdentityLedgerV4State,
    target_d: GeneratedExpression,
    *,
    post_step: int,
    protocol: GeneratedClassRecurrenceV0Protocol,
    lineage_config: ExpandedExpressionLineageConfig,
) -> GeneratedNaturalDStructuralSnapshot:
    if post_step == 0:
        phase_index, phase_label = -1, "GENESIS"
    else:
        phase_index, phase_label = _phase_coordinates(protocol, post_step - 1)
    plan = compile_expanded_expression_lineage_masks(
        state,
        target_d,
        config=lineage_config,
    )
    audit = plan.audit
    ledger_sha = ledger_state.integrity_sha256
    if type(ledger_sha) is not str or len(ledger_sha) != 64:
        raise GeneratedNaturalDBirthProbeError("ledger state integrity hash is malformed")
    return GeneratedNaturalDStructuralSnapshot(
        post_step=post_step,
        phase_index=phase_index,
        phase_label=phase_label,
        active_exact_root_count=audit.active_exact_target_root_count,
        candidate_exact_root_count=audit.candidate_exact_target_root_count,
        active_expanded_mask_count=audit.active_mask_count,
        candidate_expanded_mask_count=audit.candidate_mask_count,
        active_target_subtree_occurrences=audit.active_target_subtree_occurrences,
        candidate_target_subtree_occurrences=audit.candidate_target_subtree_occurrences,
        active_expanded_birth_identities=_selected_identities(
            ledger_state.active_identity,
            plan.active_mask,
            slots=protocol.active_slots,
            name="active",
        ),
        candidate_expanded_birth_identities=_selected_identities(
            ledger_state.candidate_identity,
            plan.candidate_mask,
            slots=protocol.candidate_slots,
            name="candidate",
        ),
        lineage_plan_sha256=audit.plan_sha256,
        core_state_sha256=generated_birth_identity_scrub_epoch_core_state_sha256(state),
        ledger_state_sha256=ledger_sha,
    )


def _repeat_step_for_positive_delta(before: int, after: int, post_step: int) -> list[int]:
    return [post_step] * max(after - before, 0)


def _record_structural_deltas(
    runtime: _ArmRuntime,
    before: GeneratedNaturalDStructuralSnapshot,
    after: GeneratedNaturalDStructuralSnapshot,
) -> None:
    before_exact_any = before.active_exact_root_count + before.candidate_exact_root_count
    after_exact_any = after.active_exact_root_count + after.candidate_exact_root_count
    if before_exact_any == 0 and after_exact_any > 0:
        runtime.exact_any_birth_steps.append(after.post_step)
    runtime.exact_active_increase_steps.extend(
        _repeat_step_for_positive_delta(
            before.active_exact_root_count,
            after.active_exact_root_count,
            after.post_step,
        )
    )
    runtime.exact_candidate_increase_steps.extend(
        _repeat_step_for_positive_delta(
            before.candidate_exact_root_count,
            after.candidate_exact_root_count,
            after.post_step,
        )
    )
    runtime.exact_active_decrease_steps.extend(
        _repeat_step_for_positive_delta(
            after.active_exact_root_count,
            before.active_exact_root_count,
            after.post_step,
        )
    )
    runtime.exact_candidate_decrease_steps.extend(
        _repeat_step_for_positive_delta(
            after.candidate_exact_root_count,
            before.candidate_exact_root_count,
            after.post_step,
        )
    )
    before_expanded_any = (
        before.active_expanded_mask_count + before.candidate_expanded_mask_count
    )
    after_expanded_any = after.active_expanded_mask_count + after.candidate_expanded_mask_count
    if before_expanded_any == 0 and after_expanded_any > 0:
        runtime.expanded_any_birth_steps.append(after.post_step)
    before_active_ids = set(before.active_expanded_birth_identities)
    after_active_ids = set(after.active_expanded_birth_identities)
    before_candidate_ids = set(before.candidate_expanded_birth_identities)
    after_candidate_ids = set(after.candidate_expanded_birth_identities)
    # A ledger identity can transfer from candidate to active during promotion,
    # so bank membership changes are entries/exits, not necessarily new births.
    runtime.expanded_active_entry_steps.extend(
        [after.post_step] * len(after_active_ids - before_active_ids)
    )
    runtime.expanded_candidate_entry_steps.extend(
        [after.post_step] * len(after_candidate_ids - before_candidate_ids)
    )
    runtime.expanded_active_exit_steps.extend(
        [after.post_step] * len(before_active_ids - after_active_ids)
    )
    runtime.expanded_candidate_exit_steps.extend(
        [after.post_step] * len(before_candidate_ids - after_candidate_ids)
    )


def _float32_hex(value: np.float32) -> str:
    return np.asarray(value, dtype=np.float32).tobytes().hex()


def _maybe_record_first_use(
    runtime: _ArmRuntime,
    observation: Array,
    *,
    zero_based_step_index: int,
    phase_index: int,
    phase_label: str,
) -> None:
    if phase_label != "D":
        return
    is_true_d = not (
        runtime.name == D_MAPPING_NEVER_SEEN_TWIN and phase_index == 3
    )
    if not is_true_d:
        runtime.deranged_d_exposure_count += 1
        return
    runtime.true_d_exposure_count += 1
    identities = runtime.snapshot.active_expanded_birth_identities
    if not identities:
        return
    slots = np.asarray([slot for slot, _ in identities], dtype=np.int32)
    weights = np.asarray(runtime.state.output_weights, dtype=np.float32)[0, slots]
    nonzero_head = weights.view(np.uint32) & np.uint32(0x7FFFFFFF) != np.uint32(0)
    if not np.any(nonzero_head):
        return
    features = np.asarray(
        runtime.learner.constructed_features(runtime.state, observation),
        dtype=np.float32,
    )[slots]
    contributions = np.asarray(weights * features, dtype=np.float32)
    nonzero_contribution = (
        contributions.view(np.uint32) & np.uint32(0x7FFFFFFF) != np.uint32(0)
    )
    if not np.all(np.isfinite(contributions)):
        raise GeneratedNaturalDBirthProbeError("active D-lineage contribution is nonfinite")
    def use(use_kind: str) -> GeneratedNaturalDFirstUse:
        return GeneratedNaturalDFirstUse(
            use_kind=use_kind,
            zero_based_step_index=zero_based_step_index,
            phase_index=phase_index,
            phase_label=phase_label,
            true_d_exposure_ordinal=runtime.true_d_exposure_count,
            active_exact_root_count=runtime.snapshot.active_exact_root_count,
            active_expanded_mask_count=runtime.snapshot.active_expanded_mask_count,
            active_expanded_birth_identities=identities,
            nonzero_head_slot_count=int(np.count_nonzero(nonzero_head)),
            nonzero_contribution_slot_count=int(
                np.count_nonzero(nonzero_contribution)
            ),
            total_contribution_float32_hex=_float32_hex(
                np.sum(contributions, dtype=np.float32)
            ),
            pre_core_state_sha256=(
                generated_birth_identity_scrub_epoch_core_state_sha256(
                    runtime.state
                )
            ),
            pre_ledger_state_sha256=runtime.ledger_state.integrity_sha256,
        )

    if runtime.first_head_use is None:
        runtime.first_head_use = use(
            "nonzero_active_output_head_on_true_D_prequential_row"
        )
    if np.any(nonzero_contribution) and runtime.first_contribution_use is None:
        runtime.first_contribution_use = use(
            "nonzero_active_lineage_contribution_on_true_D_prequential_row"
        )


def _trace_count(trace: CompositionalCurationTrace, name: str) -> int:
    return _int_scalar(getattr(trace, name), name=f"curation_trace.{name}")


def _validate_binding(
    binding: GeneratedBirthIdentityTraceBinding,
    *,
    expected_pre_words: tuple[int, int],
    expected_post_words: tuple[int, int],
    expected_ledger_nbytes: int,
) -> None:
    if type(binding) is not GeneratedBirthIdentityTraceBinding:
        raise GeneratedNaturalDBirthProbeError("trace adapter returned a malformed binding")
    if binding.schema != GENERATED_BIRTH_IDENTITY_TRACE_BINDING_SCHEMA:
        raise GeneratedNaturalDBirthProbeError("trace-binding schema drifted")
    if not all(
        (
            binding.structural_trace_validated,
            binding.complete_result_bit_compared,
            binding.typed_prng_implementation_and_key_data_compared,
            binding.float_raw_bytes_compared_including_nan_payloads,
            binding.source_replay_authenticated,
            binding.ledger_validation.valid,
        )
    ):
        raise GeneratedNaturalDBirthProbeError("a transaction lacks full source replay")
    if any(
        (
            binding.execution_authorized,
            binding.runner_authorized,
            binding.artifact_writes_authorized,
            binding.evidence_authorized,
            binding.scientific_promotion_allowed,
        )
    ):
        raise GeneratedNaturalDBirthProbeError("a ledger binding granted external authority")
    audit = binding.transaction.audit
    if audit.pre_step_words != expected_pre_words or audit.post_step_words != (
        expected_post_words
    ):
        raise GeneratedNaturalDBirthProbeError("ledger audit exact clock drifted")
    if binding.transaction.post_state.persistent_array_nbytes != expected_ledger_nbytes:
        raise GeneratedNaturalDBirthProbeError("ledger persistent capacity changed")
    if audit.post_state_sha256 != binding.transaction.post_state.integrity_sha256:
        raise GeneratedNaturalDBirthProbeError("ledger post-state hash binding drifted")


def _advance_arm(
    runtime: _ArmRuntime,
    observation: Array,
    target: Array,
    *,
    zero_based_step_index: int,
    phase_index: int,
    phase_label: str,
    protocol: GeneratedClassRecurrenceV0Protocol,
    target_d: GeneratedExpression,
    lineage_config: ExpandedExpressionLineageConfig,
) -> None:
    post_step = zero_based_step_index + 1
    _maybe_record_first_use(
        runtime,
        observation,
        zero_based_step_index=zero_based_step_index,
        phase_index=phase_index,
        phase_label=phase_label,
    )
    pre_state = runtime.state
    pre_ledger = runtime.ledger_state
    result: CompositionalFeatureUpdateResult = runtime.learner.update(
        pre_state,
        observation,
        target,
        context_id=0,
    )
    result.state.step_words.block_until_ready()
    binding = authenticate_generated_birth_identity_trace_by_source_replay(
        runtime.learner,
        runtime.ledger_config,
        pre_ledger,
        learner_pre_state=pre_state,
        learner_post_state=result.state,
        supplied_update_result=result,
        observation=observation,
        targets=target,
        context_id=0,
    )
    expected_pre_words = (0, zero_based_step_index)
    expected_post_words = (0, post_step)
    _validate_binding(
        binding,
        expected_pre_words=expected_pre_words,
        expected_post_words=expected_post_words,
        expected_ledger_nbytes=runtime.initial_ledger_nbytes,
    )
    trace = result.curation_trace
    if _words(trace.pre_step_words, name="trace.pre_step_words") != expected_pre_words:
        raise GeneratedNaturalDBirthProbeError("curation trace pre-step words drifted")
    if _words(trace.post_step_words, name="trace.post_step_words") != expected_post_words:
        raise GeneratedNaturalDBirthProbeError("curation trace post-step words drifted")
    if _words(result.state.step_words, name="state.step_words") != expected_post_words:
        raise GeneratedNaturalDBirthProbeError("learner exact lifetime words drifted")
    if _words(binding.transaction.post_state.step_words, name="ledger.step_words") != (
        expected_post_words
    ):
        raise GeneratedNaturalDBirthProbeError("ledger exact lifetime words drifted")
    if _int_scalar(result.state.step_count, name="state.step_count") != post_step:
        raise GeneratedNaturalDBirthProbeError("short-horizon scalar telemetry drifted")
    interval = protocol.curation_opportunity_audit.curation_interval
    opportunity = post_step % interval == 0
    should_try = _bool_scalar(trace.should_try_replace, name="trace.should_try_replace")
    if should_try != opportunity:
        raise GeneratedNaturalDBirthProbeError("natural curation cadence drifted")
    has_event = _bool_scalar(trace.has_event, name="trace.has_event")
    transaction_sha = binding.transaction.audit.transaction_sha256
    if len(transaction_sha) != 64:
        raise GeneratedNaturalDBirthProbeError("ledger transaction hash is malformed")
    runtime.transaction_sha256s.append(transaction_sha)
    runtime.identity_event_count += binding.transaction.audit.applied_identity_event_count
    runtime.state = result.state
    runtime.ledger_state = binding.transaction.post_state
    if not opportunity:
        if has_event or binding.transaction.audit.applied_identity_event_count != 0:
            raise GeneratedNaturalDBirthProbeError(
                "a structural event occurred outside a canonical curation opportunity"
            )
        return
    if persistent_compositional_state_nbytes(runtime.state) != runtime.initial_core_nbytes:
        raise GeneratedNaturalDBirthProbeError("learner persistent capacity changed")
    post_snapshot = _snapshot(
        runtime.state,
        runtime.ledger_state,
        target_d,
        post_step=post_step,
        protocol=protocol,
        lineage_config=lineage_config,
    )
    _record_structural_deltas(runtime, runtime.snapshot, post_snapshot)
    runtime.snapshot = post_snapshot
    runtime.curation_transactions.append(
        GeneratedNaturalDCurationTransaction(
            post_step=post_step,
            zero_based_step_index=zero_based_step_index,
            phase_index=phase_index,
            phase_label=phase_label,
            should_try_replace=should_try,
            has_event=has_event,
            proposal_count=_trace_count(trace, "proposal_count"),
            root_change_count=_trace_count(trace, "root_change_count"),
            promotion_count=_trace_count(trace, "promotion_count"),
            cascade_refill_count=_trace_count(trace, "cascade_refill_count"),
            candidate_refresh_count=_trace_count(trace, "candidate_refresh_count"),
            candidate_rebound_count=_trace_count(trace, "candidate_rebound_count"),
            candidate_overdepth_regeneration_count=_trace_count(
                trace,
                "candidate_overdepth_regeneration_count",
            ),
            logical_event_count=_trace_count(trace, "logical_event_count"),
            applied_identity_event_count=(
                binding.transaction.audit.applied_identity_event_count
            ),
            transaction_sha256=transaction_sha,
            snapshot=post_snapshot,
        )
    )


def _array_sha256(value: Array, *, dtype: np.dtype[np.generic]) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _observation_prefix_sha256(observations: Array) -> str:
    array = np.asarray(observations)
    if array.shape[0] != PROBE_STOP_POST_STEP or array.dtype != np.float32:
        raise GeneratedNaturalDBirthProbeError("observation prefix shape or dtype drifted")
    return _array_sha256(observations, dtype=np.dtype(np.float32))


def _new_runtime(
    name: str,
    target_rows: Array,
    target_sha256: str,
    *,
    shared_genesis_state: CompositionalFeatureState,
    protocol: GeneratedClassRecurrenceV0Protocol,
    target_d: GeneratedExpression,
    lineage_config: ExpandedExpressionLineageConfig,
    expected_config_sha256: str,
) -> _ArmRuntime:
    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    if type(learner) is not CompositionalFeatureLearner:
        raise GeneratedNaturalDBirthProbeError("probe learner is not the exact base class")
    if _config_sha256(learner) != expected_config_sha256:
        raise GeneratedNaturalDBirthProbeError("canonical full-lifecycle config drifted")
    if type(shared_genesis_state) is not CompositionalFeatureState:
        raise TypeError("shared_genesis_state must be an exact CompositionalFeatureState")
    state = shared_genesis_state
    core_nbytes = persistent_compositional_state_nbytes(state)
    if core_nbytes != protocol.resource_contract.jax_state_nbytes:
        raise GeneratedNaturalDBirthProbeError("genesis violates canonical core resources")
    ledger_config = GeneratedBirthIdentityLedgerV4Config(
        namespace=f"{LEDGER_NAMESPACE_PREFIX}-{name}",
        active_slots=protocol.active_slots,
        candidate_slots=protocol.candidate_slots,
        raw_feature_slots=protocol.input_dim,
        max_depth=protocol.allocated_max_depth,
        learn_generator_resources=False,
    )
    ledger_state = attach_generated_birth_identity_ledger_at_core_genesis(
        ledger_config,
        learner_pre_state=state,
        paired_development_life_seed=PROBE_SEED,
    )
    snapshot = _snapshot(
        state,
        ledger_state,
        target_d,
        post_step=0,
        protocol=protocol,
        lineage_config=lineage_config,
    )
    if any(
        (
            snapshot.active_exact_root_count,
            snapshot.candidate_exact_root_count,
            snapshot.active_expanded_mask_count,
            snapshot.candidate_expanded_mask_count,
        )
    ):
        raise GeneratedNaturalDBirthProbeError("D must be absent at canonical genesis")
    return _ArmRuntime(
        name=name,
        learner=learner,
        state=state,
        ledger_config=ledger_config,
        ledger_state=ledger_state,
        target_rows=target_rows,
        target_sha256=target_sha256,
        initial_core_sha256=generated_birth_identity_scrub_epoch_core_state_sha256(state),
        initial_core_nbytes=core_nbytes,
        initial_ledger_nbytes=ledger_state.persistent_array_nbytes,
        snapshot=snapshot,
        transaction_sha256s=[],
        curation_transactions=[],
        identity_event_count=0,
        true_d_exposure_count=0,
        deranged_d_exposure_count=0,
        exact_any_birth_steps=[],
        exact_active_increase_steps=[],
        exact_candidate_increase_steps=[],
        exact_active_decrease_steps=[],
        exact_candidate_decrease_steps=[],
        expanded_any_birth_steps=[],
        expanded_active_entry_steps=[],
        expanded_candidate_entry_steps=[],
        expanded_active_exit_steps=[],
        expanded_candidate_exit_steps=[],
        first_head_use=None,
        first_contribution_use=None,
    )


def _outcome(runtime: _ArmRuntime) -> str:
    if not runtime.exact_any_birth_steps:
        return OUTCOME_NO_NATURAL_D_BIRTH
    if runtime.first_contribution_use is not None:
        return OUTCOME_ACTIVE_CONTRIBUTION
    if runtime.first_head_use is not None:
        return OUTCOME_ACTIVE_HEAD_USE
    return OUTCOME_CANDIDATE_ONLY


def _finish_arm(
    runtime: _ArmRuntime,
    *,
    protocol: GeneratedClassRecurrenceV0Protocol,
    target_d: GeneratedExpression,
    lineage_config: ExpandedExpressionLineageConfig,
) -> GeneratedNaturalDBirthArmResult:
    final_core_nbytes = persistent_compositional_state_nbytes(runtime.state)
    final_snapshot = _snapshot(
        runtime.state,
        runtime.ledger_state,
        target_d,
        post_step=PROBE_STOP_POST_STEP,
        protocol=protocol,
        lineage_config=lineage_config,
    )
    return GeneratedNaturalDBirthArmResult(
        arm_name=runtime.name,
        learner_control_name=FULL_LIFECYCLE,
        stream_target_sha256=runtime.target_sha256,
        learner_config_sha256=_config_sha256(runtime.learner),
        initial_core_state_sha256=runtime.initial_core_sha256,
        final_core_state_sha256=final_snapshot.core_state_sha256,
        initial_persistent_core_state_nbytes=runtime.initial_core_nbytes,
        final_persistent_core_state_nbytes=final_core_nbytes,
        initial_persistent_ledger_state_nbytes=runtime.initial_ledger_nbytes,
        final_persistent_ledger_state_nbytes=(
            runtime.ledger_state.persistent_array_nbytes
        ),
        authenticated_transaction_count=len(runtime.transaction_sha256s),
        source_replay_authenticated_transaction_count=len(runtime.transaction_sha256s),
        transaction_sha256s=tuple(runtime.transaction_sha256s),
        curation_opportunity_count=len(runtime.curation_transactions),
        curation_event_count=sum(item.has_event for item in runtime.curation_transactions),
        curation_transactions=tuple(runtime.curation_transactions),
        total_applied_identity_event_count=runtime.identity_event_count,
        true_d_exposure_count=runtime.true_d_exposure_count,
        deranged_d_exposure_count=runtime.deranged_d_exposure_count,
        exact_any_birth_post_steps=tuple(runtime.exact_any_birth_steps),
        exact_active_count_increase_post_steps=tuple(runtime.exact_active_increase_steps),
        exact_candidate_count_increase_post_steps=tuple(
            runtime.exact_candidate_increase_steps
        ),
        exact_active_count_decrease_post_steps=tuple(runtime.exact_active_decrease_steps),
        exact_candidate_count_decrease_post_steps=tuple(
            runtime.exact_candidate_decrease_steps
        ),
        expanded_any_birth_post_steps=tuple(runtime.expanded_any_birth_steps),
        expanded_active_identity_entry_post_steps=tuple(
            runtime.expanded_active_entry_steps
        ),
        expanded_candidate_identity_entry_post_steps=tuple(
            runtime.expanded_candidate_entry_steps
        ),
        expanded_active_identity_exit_post_steps=tuple(
            runtime.expanded_active_exit_steps
        ),
        expanded_candidate_identity_exit_post_steps=tuple(
            runtime.expanded_candidate_exit_steps
        ),
        first_prequential_head_use=runtime.first_head_use,
        first_nonzero_prequential_contribution=runtime.first_contribution_use,
        final_snapshot=final_snapshot,
        final_step_words_uint32=_words(runtime.state.step_words, name="final step_words"),
        final_ledger_step_words_uint32=_words(
            runtime.ledger_state.step_words,
            name="final ledger step_words",
        ),
        final_learner_key_words_uint32=_key_words(runtime.state.key),
        every_transaction_source_replay_authenticated=True,
        persistent_capacity_unchanged=(
            final_core_nbytes == runtime.initial_core_nbytes
            and runtime.ledger_state.persistent_array_nbytes
            == runtime.initial_ledger_nbytes
        ),
        artifacts_written=0,
        evidence_authorized=False,
        promotion_authorized=False,
        descriptive_outcome=_outcome(runtime),
    )


def _result_sha256(result: GeneratedNaturalDBirthProbeResult) -> str:
    payload = dataclasses.asdict(result)
    payload.pop("result_sha256")
    return _sha256(payload)


def run_generated_natural_d_birth_probe(
    permit: GeneratedNaturalDBirthProbePermit,
) -> GeneratedNaturalDBirthProbeResult:
    """Run exactly the permitted paired seed-101 in-memory descriptive probe."""

    checked = validate_generated_natural_d_birth_probe_permit(permit)
    protocol = build_generated_class_recurrence_v0_protocol()
    upstream = build_d_mapping_never_seen_contract()
    dataset = build_d_mapping_twin_dataset(upstream)
    if (
        dataset.observation_sha256 != checked.observation_sha256
        or dataset.reference_target_sha256 != checked.reference_target_sha256
        or dataset.twin_target_sha256 != checked.twin_target_sha256
    ):
        raise GeneratedNaturalDBirthProbeError("canonical read-only dataset hashes drifted")
    observations = dataset.observations[: checked.fixed_stop_post_step]
    reference_targets = dataset.reference_targets[: checked.fixed_stop_post_step]
    twin_targets = dataset.twin_targets[: checked.fixed_stop_post_step]
    if observations.shape != (checked.fixed_stop_post_step, protocol.input_dim):
        raise GeneratedNaturalDBirthProbeError("canonical observation prefix shape drifted")
    if reference_targets.shape != (checked.fixed_stop_post_step,) or twin_targets.shape != (
        checked.fixed_stop_post_step,
    ):
        raise GeneratedNaturalDBirthProbeError("canonical target prefix shape drifted")
    reference_bits = np.asarray(reference_targets, dtype=np.float32).view(np.uint32)
    twin_bits = np.asarray(twin_targets, dtype=np.float32).view(np.uint32)
    first_mask = np.zeros((checked.fixed_stop_post_step,), dtype=np.bool_)
    first_mask[upstream.first_d_start : upstream.first_d_stop] = True
    first_mismatch = int(np.count_nonzero(reference_bits[first_mask] != twin_bits[first_mask]))
    non_first_mismatch = int(
        np.count_nonzero(reference_bits[~first_mask] != twin_bits[~first_mask])
    )
    if first_mismatch != upstream.first_d_length or non_first_mismatch != 0:
        raise GeneratedNaturalDBirthProbeError("fixed D-twin target pairing drifted")
    root_key = jr.key(PROBE_SEED, impl=THREEFRY_IMPLEMENTATION)
    learner_key = jr.fold_in(root_key, np.uint32(LEARNER_KEY_DOMAIN))
    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target_d = next(target.expression for target in manifest.targets if target.name == "D")
    lineage_config = _lineage_config(protocol)
    genesis_learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    if _config_sha256(genesis_learner) != checked.learner_config_sha256:
        raise GeneratedNaturalDBirthProbeError("canonical genesis learner config drifted")
    shared_genesis_state = genesis_learner.init(protocol.input_dim, learner_key)
    runtimes = (
        _new_runtime(
            FULL_LIFECYCLE,
            reference_targets,
            dataset.reference_target_sha256,
            shared_genesis_state=shared_genesis_state,
            protocol=protocol,
            target_d=target_d,
            lineage_config=lineage_config,
            expected_config_sha256=checked.learner_config_sha256,
        ),
        _new_runtime(
            D_MAPPING_NEVER_SEEN_TWIN,
            twin_targets,
            dataset.twin_target_sha256,
            shared_genesis_state=shared_genesis_state,
            protocol=protocol,
            target_d=target_d,
            lineage_config=lineage_config,
            expected_config_sha256=checked.learner_config_sha256,
        ),
    )
    initial_parity = runtimes[0].initial_core_sha256 == runtimes[1].initial_core_sha256
    if not initial_parity:
        raise GeneratedNaturalDBirthProbeError("paired learner genesis states differ")
    key_parity = True
    for step_index in range(checked.fixed_stop_post_step):
        observation = observations[step_index]
        if observation.dtype != jnp.float32 or observation.shape != (protocol.input_dim,):
            raise GeneratedNaturalDBirthProbeError("learner observation shape/dtype drifted")
        phase_index, phase_label = _phase_coordinates(protocol, step_index)
        for runtime in runtimes:
            scalar_target = runtime.target_rows[step_index]
            target = jnp.expand_dims(scalar_target, axis=0)
            if target.dtype != jnp.float32 or target.shape != (1,):
                raise GeneratedNaturalDBirthProbeError("learner target shape/dtype drifted")
            _advance_arm(
                runtime,
                observation,
                target,
                zero_based_step_index=step_index,
                phase_index=phase_index,
                phase_label=phase_label,
                protocol=protocol,
                target_d=target_d,
                lineage_config=lineage_config,
            )
        step_key_parity = _key_words(runtimes[0].state.key) == _key_words(
            runtimes[1].state.key
        )
        key_parity = key_parity and step_key_parity
        if not step_key_parity:
            raise GeneratedNaturalDBirthProbeError("paired learner RNG call counts diverged")
    arm_results = tuple(
        _finish_arm(
            runtime,
            protocol=protocol,
            target_d=target_d,
            lineage_config=lineage_config,
        )
        for runtime in runtimes
    )
    result = GeneratedNaturalDBirthProbeResult(
        schema=GENERATED_NATURAL_D_BIRTH_RESULT_SCHEMA,
        status=GENERATED_NATURAL_D_BIRTH_RESULT_STATUS,
        permit=checked,
        arm_results=arm_results,
        observation_prefix_sha256=_observation_prefix_sha256(observations),
        first_d_target_bit_mismatch_count=first_mismatch,
        non_first_d_target_bit_mismatch_count=non_first_mismatch,
        initial_core_state_bit_exact_across_arms=initial_parity,
        learner_key_bit_exact_across_arms_every_step=key_parity,
        learner_visible_field_schema_equal_across_arms=True,
        raw_observation_values_bit_exact_across_arms=True,
        target_value_differences_confined_to_first_d=True,
        paired_steps_executed=checked.fixed_stop_post_step,
        total_authenticated_transactions=sum(
            arm.authenticated_transaction_count for arm in arm_results
        ),
        artifact_bytes_written=0,
        thresholds_applied=0,
        searches_executed=0,
        evidence_authorized=False,
        promotion_authorized=False,
        result_sha256="",
    )
    result = dataclasses.replace(result, result_sha256=_result_sha256(result))
    return validate_generated_natural_d_birth_probe_result(result)


def validate_generated_natural_d_birth_probe_result(
    result: GeneratedNaturalDBirthProbeResult,
) -> GeneratedNaturalDBirthProbeResult:
    """Validate the complete in-memory result and its per-transition hash bindings."""

    if type(result) is not GeneratedNaturalDBirthProbeResult:
        raise TypeError("result must be an exact GeneratedNaturalDBirthProbeResult")
    permit = validate_generated_natural_d_birth_probe_permit(result.permit)
    if result.schema != GENERATED_NATURAL_D_BIRTH_RESULT_SCHEMA or result.status != (
        GENERATED_NATURAL_D_BIRTH_RESULT_STATUS
    ):
        raise ValueError("natural-D result schema or status drifted")
    if tuple(arm.arm_name for arm in result.arm_results) != permit.fixed_arm_order:
        raise ValueError("natural-D result arm order drifted")
    if not all(
        (
            result.initial_core_state_bit_exact_across_arms,
            result.learner_key_bit_exact_across_arms_every_step,
            result.learner_visible_field_schema_equal_across_arms,
            result.raw_observation_values_bit_exact_across_arms,
            result.target_value_differences_confined_to_first_d,
        )
    ):
        raise ValueError("natural-D result pairing disclosure is false")
    if (
        result.paired_steps_executed != permit.fixed_stop_post_step
        or result.total_authenticated_transactions
        != len(result.arm_results) * permit.expected_transactions_per_arm
        or result.first_d_target_bit_mismatch_count != 421
        or result.non_first_d_target_bit_mismatch_count != 0
    ):
        raise ValueError("natural-D result fixed work or target pairing drifted")
    if any(
        (
            result.artifact_bytes_written,
            result.thresholds_applied,
            result.searches_executed,
            result.evidence_authorized,
            result.promotion_authorized,
        )
    ):
        raise ValueError("natural-D result claims forbidden output or authority")
    expected_targets = {
        FULL_LIFECYCLE: permit.reference_target_sha256,
        D_MAPPING_NEVER_SEEN_TWIN: permit.twin_target_sha256,
    }
    protocol = build_generated_class_recurrence_v0_protocol()
    for arm in result.arm_results:
        if type(arm) is not GeneratedNaturalDBirthArmResult:
            raise TypeError("natural-D result contains a malformed arm")
        if (
            arm.learner_control_name != FULL_LIFECYCLE
            or arm.learner_config_sha256 != permit.learner_config_sha256
            or arm.stream_target_sha256 != expected_targets[arm.arm_name]
        ):
            raise ValueError("natural-D arm config or target binding drifted")
        if (
            arm.authenticated_transaction_count != permit.expected_transactions_per_arm
            or arm.source_replay_authenticated_transaction_count
            != permit.expected_transactions_per_arm
            or len(arm.transaction_sha256s) != permit.expected_transactions_per_arm
            or not arm.every_transaction_source_replay_authenticated
        ):
            raise ValueError("natural-D arm source-replay cardinality drifted")
        if not all(len(value) == 64 for value in arm.transaction_sha256s):
            raise ValueError("natural-D arm contains a malformed transaction hash")
        expected_posts = tuple(
            range(
                protocol.curation_opportunity_audit.curation_interval,
                permit.fixed_stop_post_step + 1,
                protocol.curation_opportunity_audit.curation_interval,
            )
        )
        if (
            arm.curation_opportunity_count
            != permit.expected_curation_opportunities_per_arm
            or tuple(item.post_step for item in arm.curation_transactions)
            != expected_posts
        ):
            raise ValueError("natural-D arm curation opportunity schedule drifted")
        for item in arm.curation_transactions:
            if (
                not item.should_try_replace
                or item.zero_based_step_index != item.post_step - 1
                or item.transaction_sha256 != arm.transaction_sha256s[item.post_step - 1]
                or item.snapshot.post_step != item.post_step
            ):
                raise ValueError("natural-D curation receipt is not transaction-bound")
        if arm.curation_event_count != sum(
            item.has_event for item in arm.curation_transactions
        ) or arm.total_applied_identity_event_count != sum(
            item.applied_identity_event_count for item in arm.curation_transactions
        ):
            raise ValueError("natural-D curation or identity totals are stale")
        if (
            arm.initial_persistent_core_state_nbytes
            != protocol.resource_contract.jax_state_nbytes
            or arm.final_persistent_core_state_nbytes
            != arm.initial_persistent_core_state_nbytes
            or arm.final_persistent_ledger_state_nbytes
            != arm.initial_persistent_ledger_state_nbytes
            or not arm.persistent_capacity_unchanged
        ):
            raise ValueError("natural-D arm persistent capacity changed")
        expected_words = (0, permit.fixed_stop_post_step)
        if (
            arm.final_step_words_uint32 != expected_words
            or arm.final_ledger_step_words_uint32 != expected_words
            or arm.final_snapshot.post_step != permit.fixed_stop_post_step
        ):
            raise ValueError("natural-D arm final exact clocks drifted")
        if arm.artifacts_written != 0 or arm.evidence_authorized or arm.promotion_authorized:
            raise ValueError("natural-D arm grants forbidden authority")
        expected_outcome = (
            OUTCOME_NO_NATURAL_D_BIRTH
            if not arm.exact_any_birth_post_steps
            else (
                OUTCOME_ACTIVE_CONTRIBUTION
                if arm.first_nonzero_prequential_contribution is not None
                else (
                    OUTCOME_ACTIVE_HEAD_USE
                    if arm.first_prequential_head_use is not None
                    else OUTCOME_CANDIDATE_ONLY
                )
            )
        )
        if arm.descriptive_outcome != expected_outcome:
            raise ValueError("natural-D arm descriptive outcome is stale")
    if result.arm_results[0].initial_core_state_sha256 != (
        result.arm_results[1].initial_core_state_sha256
    ):
        raise ValueError("natural-D paired initial states are not bit exact")
    if result.arm_results[0].final_learner_key_words_uint32 != (
        result.arm_results[1].final_learner_key_words_uint32
    ):
        raise ValueError("natural-D paired final learner keys differ")
    if result.result_sha256 != _result_sha256(result):
        raise ValueError("natural-D result hash does not reconstruct")
    return result


__all__ = [
    "GENERATED_NATURAL_D_BIRTH_PROBE_SCHEMA",
    "GENERATED_NATURAL_D_BIRTH_PROBE_STATUS",
    "GENERATED_NATURAL_D_BIRTH_RESULT_SCHEMA",
    "GENERATED_NATURAL_D_BIRTH_RESULT_STATUS",
    "LEARNER_KEY_DOMAIN",
    "OUTCOME_ACTIVE_CONTRIBUTION",
    "OUTCOME_ACTIVE_HEAD_USE",
    "OUTCOME_CANDIDATE_ONLY",
    "OUTCOME_NO_NATURAL_D_BIRTH",
    "PROBE_ARM_ORDER",
    "PROBE_SEED",
    "PROBE_STOP_POST_STEP",
    "GeneratedNaturalDBirthArmResult",
    "GeneratedNaturalDBirthProbeError",
    "GeneratedNaturalDBirthProbePermit",
    "GeneratedNaturalDBirthProbeResult",
    "GeneratedNaturalDCurationTransaction",
    "GeneratedNaturalDFirstUse",
    "GeneratedNaturalDStructuralSnapshot",
    "build_generated_natural_d_birth_probe_permit",
    "run_generated_natural_d_birth_probe",
    "validate_generated_natural_d_birth_probe_permit",
    "validate_generated_natural_d_birth_probe_result",
]
