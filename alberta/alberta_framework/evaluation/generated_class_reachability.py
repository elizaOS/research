"""Negative analytic reachability audit for generated-class target D.

The generated-class recurrence declaration counts curation decisions, not all
fresh structural draws.  This module therefore does *not* manufacture an
acquisition probability for the unfinished lifecycle.  It derives one narrow,
state-independent ceiling for a candidate-only reacquisition epoch: when each
of the twelve second-D curation decisions applies exactly one canonical
``_generate_one`` candidate proposal and promotion cascades are disabled, the
probability of at least one exact D proposal is at most
``1 - (15 / 16) ** 12`` under the split-key categorical sampling model.
This uses the conditional-hazard chain rule, not cross-event independence: if
the hit hazard after every possible prior miss history is at most ``1 / 16``,
the probability of twelve consecutive misses is at least ``(15 / 16) ** 12``.

The bound is deliberately optimistic.  It assigns probability one both to the
required recursive parent and to proposal application.  The only remaining
factors are the gated-operation mass (exactly one quarter under the reviewed
prior) and the uniform
choice of raw parent ``x1`` (one of four).  It is an upper bound, not a power
calculation, outcome, threshold, or lower bound.

The real lifecycle is not covered by that calculation.  A promotion can
refresh one candidate and then refill a state-dependent set of active
descendants.  An active or cascade change can also force fresh regeneration of
each candidate whose rebound depth exceeds the fixed budget.  Conversely, an
existing D candidate can reappear by promotion without being born in the
audited window.  The current protocol has no
executable candidate-only/no-cascade reacquisition epoch or trace-bound
proposal/commit ledger.  A public learner trace and a development structural-
lifetime sidecar now exist, but no adapter or source replay binds them.
Whole-lifecycle D-birth and D-reacquisition probabilities remain unidentified.

This is development-only blocker analysis.  It executes no learner updates,
generated life, Monte Carlo sample, threshold, artifact, or evidence path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.compositional_features import (
    GENERATION_ROBUST_RECURSIVE,
    NUM_OPS,
    OP_GATED,
    OP_RAW,
    CompositionalFeatureLearner,
)
from alberta_framework.evaluation.generated_class_recurrence import (
    DEVELOPMENT_EXPRESSION_NAMESPACE,
    FULL_LIFECYCLE,
    GENERATED_CLASS_RECURRENCE_V0_SCHEMA,
    GeneratedExpression,
    build_generated_class_recurrence_v0_protocol,
    build_generated_class_v0_learner,
    count_expression_occurrences,
    derive_expression_manifest,
    expression_digest,
    measure_compositional_jax_state_nbytes,
)

GENERATED_CLASS_REACHABILITY_AUDIT_SCHEMA = (
    "alberta.generated-class-reachability-audit.development.v4"
)
GENERATED_CLASS_REACHABILITY_AUDIT_STATUS = (
    "DEVELOPMENT_BLOCKER_CANDIDATE_CHANNEL_UPPER_BOUND_ONLY"
)

_EXPECTED_OP_LOGITS_SOURCE_SHA256 = (
    "c6d1a059bd29d074596809716cc775df2396fb11ec1a5386dbc78f7dc148fd82"
)
_EXPECTED_COMPOSITIONAL_FEATURES_MODULE_BYTE_SHA256 = (
    "767f054bb3413b2408e664a17bcb8690a9f83018f638d6acfcfde2e9debf5b5a"
)
_EXPECTED_INIT_SOURCE_SHA256 = (
    "1ae246e6736ad932e798d21d5f7238477414ff908636289cdbfcd73f42f3ce94"
)
_EXPECTED_GENERATE_ONE_SOURCE_SHA256 = (
    "d6aafc8401b45ed915323997f7560d866e36fd42aac375d3db3ab660d625c960"
)
_EXPECTED_CURATION_STAGE_GUIDANCE_SOURCE_SHA256 = (
    "27c997e2478a7eef95ffd5adb3e592b085a313f3ce2fa1d78ab3330bccd0f725"
)
_EXPECTED_CASCADE_REPLACE_SOURCE_SHA256 = (
    "2994d446d0ff9a667ec67a304b92cf7ff003292fc278d62e4f66ae849fe836a2"
)
_EXPECTED_CASCADE_REPLACE_WITH_MASK_SOURCE_SHA256 = (
    "8a26857076565a2c8a4f064b079ac3a412b6194c4a2c4095925dec462951fa98"
)
_EXPECTED_UPDATE_SOURCE_SHA256 = (
    "9c86a5c984c9d7cbcb454543f87589c938d8d7873797c81aa057a71f0903b8eb"
)

_EXPECTED_OPERATION_PRIOR = (0.0, 0.25, 0.25, 0.25, 0.25)
_SECOND_D_PHASE_INDEX = 7
_EXPECTED_SECOND_D_OPPORTUNITIES = 12
_EXPECTED_RAW_PARENT_SLOTS = (0, 1, 2, 3)
_REQUIRED_RIGHT_RAW_INDEX = 1
_INITIALIZATION_KEY_DATA = (
    (0x41D2A7C3, 0x9B305E16),
    (0xD6048B2F, 0x237AC951),
)

_BLOCKERS = (
    "no candidate-only/no-promotion-cascade fresh reacquisition epoch",
    (
        "recursive-parent probability depends on utilities, feature values, "
        "residual credit, ages, and depth"
    ),
    "the required product(x0,x0) parent is replaceable under robust_recursive",
    "promotion-triggered descendant refills add a state-dependent number of active birth draws",
    (
        "active and cascade changes can add up to candidate-bank-size fresh "
        "overdepth-regeneration draws"
    ),
    "promotion can reintroduce a candidate identity born outside the audited window",
    (
        "public curation trace and structural-lifetime ledger sidecar are not yet "
        "bound by an adapter or independent source replay"
    ),
)

_MINIMUM_IDENTIFICATION_CONTRACT = (
    "freeze a named reacquisition interval and its included step endpoints",
    "allocate fresh unique Threefry generation keys and replay every split and draw",
    "record every candidate proposal, applied write, promotion, and active cascade refill",
    "record exact pre-draw structures and all parent-logit inputs for every draw",
    "bind required-parent presence and exact expression identity before and after every write",
    "define whether promotion of a pre-window candidate counts as birth or reacquisition",
    "validate the complete event ledger against independently replayed learner transitions",
)


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassReachabilityResourceContract:
    """Work performed by this deterministic audit, excluding test replays."""

    deterministic_initialization_probes: int
    initialization_key_impl: str
    initialization_key_data_uint32: tuple[tuple[int, int], ...]
    one_initialized_state_jax_nbytes: int
    peak_probe_state_jax_nbytes: int
    learner_updates_executed: int
    runtime_generate_one_calls_executed: int
    full_life_steps_executed: int
    monte_carlo_samples: int
    artifact_bytes_written: int
    wall_clock_threshold: float | None
    flop_or_hlo_equivalence_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassReachabilityOperationContract:
    """Abstract analytic operations; this is not a FLOP or latency claim."""

    phase_lengths_checked: int
    second_d_curation_events_enumerated: int
    candidate_proposal_channels_counted_per_curation: int
    probability_factors_per_candidate_proposal: int
    conditional_miss_product_exponent: int
    active_cascade_refill_channels_in_probability_bound: int
    candidate_overdepth_regeneration_channels_in_probability_bound: int
    max_active_cascade_refill_write_sites_per_curation: int
    max_candidate_overdepth_regeneration_write_sites_per_curation: int
    max_fresh_structural_write_sites_per_curation: int
    operation_accounting_scope: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassReachabilityAudit:
    """Fail-closed negative audit of the current D-reacquisition opportunity budget."""

    schema: str
    status: str
    development_only: bool
    protocol_schema: str
    phase_length_manifest_sha256: str
    target_name: str
    target_whole_tree_digest: str
    required_left_parent_digest: str
    required_right_parent_digest: str
    initial_target_active_occurrences: int
    initial_target_candidate_occurrences: int
    initial_left_parent_active_occurrences: int
    initial_right_parent_active_occurrences: int
    initialization_structure_key_invariant_observed: bool
    second_d_phase_index: int
    second_d_start_step_zero_based: int
    second_d_stop_step_zero_based_exclusive: int
    second_d_curation_step_counts_one_based: tuple[int, ...]
    second_d_curation_opportunities: int
    raw_parent_slots: tuple[int, ...]
    required_right_raw_index: int
    required_right_raw_slot: int
    robust_recursive_partner_rule: str
    split_key_categorical_model_used: bool
    conditional_hazard_chain_rule_used: bool
    cross_event_independence_assumed: bool
    operation_prior_float32: tuple[float, ...]
    op_logits_float32_bits: tuple[int, ...]
    op_probabilities_float32_bits: tuple[int, ...]
    zero_prior_operation_logits_are_negative_infinity: bool
    raw_operation_probability_is_exact_zero: bool
    gate_mass_upper_bound_numerator: int
    gate_mass_upper_bound_denominator: int
    required_right_probability_numerator: int
    required_right_probability_denominator: int
    optimistic_required_left_parent_probability_numerator: int
    optimistic_required_left_parent_probability_denominator: int
    optimistic_proposal_application_probability_numerator: int
    optimistic_proposal_application_probability_denominator: int
    per_candidate_proposal_d_birth_upper_bound_numerator: int
    per_candidate_proposal_d_birth_upper_bound_denominator: int
    candidate_channel_any_birth_upper_bound_numerator: int
    candidate_channel_any_birth_upper_bound_denominator: int
    candidate_channel_any_birth_upper_bound_float64: float
    candidate_channel_bound_scope: str
    candidate_channel_bound_applies_to_current_lifecycle: bool
    protocol_post_scrub_generation_freeze_complete: bool
    protocol_fresh_reacquisition_generation_epoch_complete: bool
    protocol_fresh_reacquisition_generation_key_namespace_complete: bool
    nontrivial_state_independent_lower_bound_derived: bool
    whole_lifecycle_d_birth_probability_identified: bool
    whole_lifecycle_d_reacquisition_probability_identified: bool
    adequate_reacquisition_probability_demonstrated: bool
    reviewed_compositional_features_module_byte_sha256: str
    reviewed_init_source_sha256: str
    reviewed_op_logits_source_sha256: str
    reviewed_generate_one_source_sha256: str
    reviewed_curation_stage_guidance_source_sha256: str
    reviewed_cascade_replace_source_sha256: str
    reviewed_cascade_replace_with_mask_source_sha256: str
    reviewed_update_source_sha256: str
    blockers: tuple[str, ...]
    minimum_identification_contract: tuple[str, ...]
    resource_contract: GeneratedClassReachabilityResourceContract
    operation_contract: GeneratedClassReachabilityOperationContract
    runner_execution_authorized: bool
    threshold_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool


def _source_sha256(method: Callable[..., object]) -> str:
    return hashlib.sha256(inspect.getsource(method).encode("utf-8")).hexdigest()


def _compositional_features_module_byte_sha256() -> str:
    source_path = inspect.getsourcefile(CompositionalFeatureLearner)
    if source_path is None:
        raise RuntimeError("compositional-feature learner source path is unavailable")
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def _uint32_bits(values: object) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.dtype != np.float32:
        raise TypeError("probability audit values must have dtype float32")
    return tuple(int(value) for value in array.view(np.uint32).reshape(-1))


def _typed_key(words: tuple[int, int]) -> jax.Array:
    data = jnp.asarray(words, dtype=jnp.uint32)
    key = jr.wrap_key_data(data, impl="threefry2x32")
    if tuple(int(value) for value in np.asarray(jr.key_data(key))) != words:
        raise RuntimeError("initialization key data failed an exact Threefry round trip")
    if str(jr.key_impl(key)) != "threefry2x32":
        raise RuntimeError("initialization key implementation is not Threefry")
    return cast(jax.Array, key)


def _assert_exact_target_shape(expression: GeneratedExpression) -> None:
    if expression.op != "gate" or expression.left is None or expression.right is None:
        raise RuntimeError("target D must be an ordered gated binary expression")
    left = expression.left
    right = expression.right
    if left.op != "product" or left.left is None or left.right is None:
        raise RuntimeError("target D left parent must be product(x0,x0)")
    if not (
        left.left.op == "raw"
        and left.left.raw_index == 0
        and left.right.op == "raw"
        and left.right.raw_index == 0
    ):
        raise RuntimeError("target D left parent identity changed")
    if right.op != "raw" or right.raw_index != _REQUIRED_RIGHT_RAW_INDEX:
        raise RuntimeError("target D right parent must be raw x1")


def _second_d_window(
    phase_order: tuple[str, ...],
    phase_lengths: tuple[int, ...],
    cadence: int,
) -> tuple[int, int, tuple[int, ...]]:
    if len(phase_order) != len(phase_lengths):
        raise RuntimeError("phase order and phase lengths differ")
    d_indices = tuple(index for index, name in enumerate(phase_order) if name == "D")
    if d_indices != (3, _SECOND_D_PHASE_INDEX):
        raise RuntimeError("generated-class schedule no longer has the reviewed D phases")
    start = sum(phase_lengths[: _SECOND_D_PHASE_INDEX])
    stop = start + phase_lengths[_SECOND_D_PHASE_INDEX]
    first_event = ((start // cadence) + 1) * cadence
    last_event = (stop // cadence) * cadence
    events = tuple(range(first_event, last_event + 1, cadence))
    independent_count = stop // cadence - start // cadence
    if len(events) != independent_count:
        raise RuntimeError("second-D cadence enumeration disagrees with integer accounting")
    return start, stop, events


def _structural_signature(state: object) -> tuple[bytes, ...]:
    names = (
        "ops",
        "parent_a",
        "parent_b",
        "depth",
        "candidate_ops",
        "candidate_parent_a",
        "candidate_parent_b",
        "candidate_depth",
    )
    return tuple(np.asarray(getattr(state, name)).tobytes() for name in names)


def _derive_audit() -> GeneratedClassReachabilityAudit:
    module_byte_sha256 = _compositional_features_module_byte_sha256()
    init_source_sha256 = _source_sha256(CompositionalFeatureLearner.__init__)
    op_logits_source_sha256 = _source_sha256(CompositionalFeatureLearner._op_logits)
    generate_one_source_sha256 = _source_sha256(
        CompositionalFeatureLearner._generate_one
    )
    curation_stage_guidance_source_sha256 = _source_sha256(
        CompositionalFeatureLearner._curation_stage_guidance
    )
    cascade_replace_source_sha256 = _source_sha256(
        CompositionalFeatureLearner._cascade_replace
    )
    cascade_replace_with_mask_source_sha256 = _source_sha256(
        CompositionalFeatureLearner._cascade_replace_with_mask
    )
    update_source_sha256 = _source_sha256(CompositionalFeatureLearner.update)
    if (
        module_byte_sha256
        != _EXPECTED_COMPOSITIONAL_FEATURES_MODULE_BYTE_SHA256
    ):
        raise RuntimeError(
            "reviewed compositional_features module bytes changed; "
            "reachability audit is stale"
        )
    if init_source_sha256 != _EXPECTED_INIT_SOURCE_SHA256:
        raise RuntimeError("reviewed __init__ source changed; reachability audit is stale")
    if op_logits_source_sha256 != _EXPECTED_OP_LOGITS_SOURCE_SHA256:
        raise RuntimeError("reviewed _op_logits source changed; reachability audit is stale")
    if generate_one_source_sha256 != _EXPECTED_GENERATE_ONE_SOURCE_SHA256:
        raise RuntimeError("reviewed _generate_one source changed; reachability audit is stale")
    if (
        curation_stage_guidance_source_sha256
        != _EXPECTED_CURATION_STAGE_GUIDANCE_SOURCE_SHA256
    ):
        raise RuntimeError(
            "reviewed _curation_stage_guidance source changed; "
            "reachability audit is stale"
        )
    if cascade_replace_source_sha256 != _EXPECTED_CASCADE_REPLACE_SOURCE_SHA256:
        raise RuntimeError(
            "reviewed _cascade_replace source changed; reachability audit is stale"
        )
    if (
        cascade_replace_with_mask_source_sha256
        != _EXPECTED_CASCADE_REPLACE_WITH_MASK_SOURCE_SHA256
    ):
        raise RuntimeError(
            "reviewed _cascade_replace_with_mask source changed; "
            "reachability audit is stale"
        )
    if update_source_sha256 != _EXPECTED_UPDATE_SOURCE_SHA256:
        raise RuntimeError("reviewed update source changed; reachability audit is stale")

    protocol = build_generated_class_recurrence_v0_protocol()
    if protocol.schema != GENERATED_CLASS_RECURRENCE_V0_SCHEMA:
        raise RuntimeError("generated-class protocol schema changed")
    if protocol.execution_authorized or protocol.evidence_authorized:
        raise RuntimeError("negative reachability audit cannot inspect an authorized protocol")
    if protocol.scientific_promotion_allowed:
        raise RuntimeError("development reachability substrate cannot allow promotion")
    prerequisites = protocol.lifecycle_prerequisites
    if (
        prerequisites.post_scrub_generation_freeze_complete
        or prerequisites.fresh_reacquisition_generation_epoch_complete
        or prerequisites.fresh_reacquisition_generation_key_namespace_complete
    ):
        raise RuntimeError("negative reachability audit is stale after lifecycle completion")
    if protocol.input_dim != 4 or protocol.active_slots != 14:
        raise RuntimeError("reviewed generated-class active-bank geometry changed")
    if protocol.candidate_slots != 8 or protocol.allocated_max_depth != 3:
        raise RuntimeError("reviewed generated-class candidate-bank geometry changed")
    if protocol.curation_opportunity_audit.curation_interval != 32:
        raise RuntimeError("reviewed generated-class curation cadence changed")

    start, stop, events = _second_d_window(
        protocol.phase_order,
        protocol.phase_lengths,
        protocol.curation_opportunity_audit.curation_interval,
    )
    if len(events) != _EXPECTED_SECOND_D_OPPORTUNITIES:
        raise RuntimeError("second-D candidate-proposal budget is no longer twelve")
    if (
        protocol.curation_opportunity_audit.opportunities_in_second_d
        != len(events)
    ):
        raise RuntimeError("protocol second-D opportunity count is not independently reproducible")

    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    target = next((item for item in manifest.targets if item.name == "D"), None)
    if target is None:
        raise RuntimeError("development expression manifest has no target D")
    _assert_exact_target_shape(target.expression)
    if target.expression.left is None or target.expression.right is None:
        raise RuntimeError("target D parent validation failed")

    learner = build_generated_class_v0_learner(FULL_LIFECYCLE, protocol)
    config = learner.to_config()
    if config["generation_strategy"] != GENERATION_ROBUST_RECURSIVE:
        raise RuntimeError("D reachability audit requires robust_recursive generation")
    if config["learn_generator_resources"] is not False:
        raise RuntimeError("D reachability audit requires an unforced operation draw")
    if config["replacement_interval"] != 32 or config["max_depth"] != 3:
        raise RuntimeError("learner cadence or depth differs from the protocol")
    operation_prior = tuple(float(value) for value in config["operation_prior"])
    if operation_prior != _EXPECTED_OPERATION_PRIOR:
        raise RuntimeError("reviewed four-way operation prior changed")

    logits = learner._op_logits()
    if logits.shape != (NUM_OPS,) or logits.dtype != jnp.float32:
        raise RuntimeError("operation logits no longer have the reviewed shape and dtype")
    prior = jnp.asarray(_EXPECTED_OPERATION_PRIOR, dtype=jnp.float32)
    expected_logits = jnp.where(
        prior > 0.0,
        jnp.log(prior),
        -jnp.inf,
    )
    if _uint32_bits(logits) != _uint32_bits(expected_logits):
        raise RuntimeError("operation logits do not preserve the reviewed exact-zero support")
    probabilities = jax.nn.softmax(logits)
    probability_bits = _uint32_bits(probabilities)
    zero_prior = prior == 0.0
    if not bool(jnp.all(jnp.isneginf(logits[zero_prior]))):
        raise RuntimeError("zero-prior operations do not all have negative-infinity logits")
    if float(probabilities[OP_RAW]) != 0.0:
        raise RuntimeError("zero-prior raw operation does not have exact zero support")
    composing = np.asarray(probabilities)[1:]
    if not np.array_equal(composing, np.full(4, np.float32(0.25))):
        raise RuntimeError("reviewed operation softmax is no longer four equal float32 masses")
    if float(probabilities[OP_GATED]) > 0.25:
        raise RuntimeError("gated-operation mass exceeds its analytic one-quarter ceiling")

    first_state = learner.init(protocol.input_dim, _typed_key(_INITIALIZATION_KEY_DATA[0]))
    second_state = learner.init(protocol.input_dim, _typed_key(_INITIALIZATION_KEY_DATA[1]))
    key_invariant = _structural_signature(first_state) == _structural_signature(
        second_state
    )
    if not key_invariant:
        raise RuntimeError("robust-recursive initialization structure depends on the key")
    measured_state_nbytes = measure_compositional_jax_state_nbytes(first_state)
    if measured_state_nbytes != protocol.resource_contract.jax_state_nbytes:
        raise RuntimeError("initialized state bytes differ from the protocol resource budget")

    ops = np.asarray(first_state.ops)
    depths = np.asarray(first_state.depth)
    parents_a = np.asarray(first_state.parent_a)
    raw_slots = tuple(
        int(index) for index in np.flatnonzero((ops == OP_RAW) & (depths == 0))
    )
    if raw_slots != _EXPECTED_RAW_PARENT_SLOTS:
        raise RuntimeError("robust-recursive shallow parent bank is not raw x0..x3")
    if tuple(int(parents_a[index]) for index in raw_slots) != raw_slots:
        raise RuntimeError("raw prefix does not preserve raw-variable identity")
    required_right_slot = raw_slots[_REQUIRED_RIGHT_RAW_INDEX]

    target_occurrences = count_expression_occurrences(first_state, target.expression)
    left_occurrences = count_expression_occurrences(first_state, target.expression.left)[0]
    right_occurrences = count_expression_occurrences(first_state, target.expression.right)[0]
    if target_occurrences != (0, 0):
        raise RuntimeError("target D must be absent from both initial banks")
    if left_occurrences != 1 or right_occurrences != 1:
        raise RuntimeError("target D must start with one exact occurrence of each parent")
    left_digest = expression_digest(target.expression.left)
    right_digest = expression_digest(target.expression.right)
    reachability = protocol.reachability_contract
    if reachability.target_name != "D":
        raise RuntimeError("protocol reachability target is no longer D")
    if reachability.target_whole_tree_digest != target.whole_tree_digest:
        raise RuntimeError("protocol D whole-tree digest changed")
    if reachability.required_top_operation != "gate":
        raise RuntimeError("protocol D top-operation declaration changed")
    if reachability.required_top_operation_probability != 0.25:
        raise RuntimeError("protocol D top-operation probability declaration changed")
    if not reachability.required_parent_choices_have_nonzero_support:
        raise RuntimeError("protocol no longer declares parent-choice support")
    if not reachability.initialization_structure_key_invariant:
        raise RuntimeError("protocol no longer declares key-invariant initialization")
    if reachability.required_left_parent_digest != left_digest:
        raise RuntimeError("protocol D-left-parent digest changed")
    if reachability.required_right_parent_digest != right_digest:
        raise RuntimeError("protocol D-right-parent digest changed")

    gate_mass = Fraction(1, 4)
    required_right_mass = Fraction(1, len(raw_slots))
    optimistic_left_mass = Fraction(1, 1)
    optimistic_application_mass = Fraction(1, 1)
    per_proposal_upper = (
        gate_mass
        * required_right_mass
        * optimistic_left_mass
        * optimistic_application_mass
    )
    any_birth_upper = 1 - (1 - per_proposal_upper) ** len(events)
    max_cascade_refills = protocol.active_slots - protocol.input_dim - 1
    max_candidate_overdepth_regenerations = protocol.candidate_slots
    max_fresh_write_sites = (
        1 + max_cascade_refills + max_candidate_overdepth_regenerations
    )

    resources = GeneratedClassReachabilityResourceContract(
        deterministic_initialization_probes=2,
        initialization_key_impl="threefry2x32",
        initialization_key_data_uint32=_INITIALIZATION_KEY_DATA,
        one_initialized_state_jax_nbytes=measured_state_nbytes,
        peak_probe_state_jax_nbytes=2 * measured_state_nbytes,
        learner_updates_executed=0,
        runtime_generate_one_calls_executed=0,
        full_life_steps_executed=0,
        monte_carlo_samples=0,
        artifact_bytes_written=0,
        wall_clock_threshold=None,
        flop_or_hlo_equivalence_claimed=False,
    )
    operations = GeneratedClassReachabilityOperationContract(
        phase_lengths_checked=len(protocol.phase_lengths),
        second_d_curation_events_enumerated=len(events),
        candidate_proposal_channels_counted_per_curation=1,
        probability_factors_per_candidate_proposal=4,
        conditional_miss_product_exponent=len(events),
        active_cascade_refill_channels_in_probability_bound=0,
        candidate_overdepth_regeneration_channels_in_probability_bound=0,
        max_active_cascade_refill_write_sites_per_curation=max_cascade_refills,
        max_candidate_overdepth_regeneration_write_sites_per_curation=(
            max_candidate_overdepth_regenerations
        ),
        max_fresh_structural_write_sites_per_curation=max_fresh_write_sites,
        operation_accounting_scope=(
            "abstract_candidate_proposal_hazard_only_no_flop_hlo_or_latency_claim"
        ),
    )
    return GeneratedClassReachabilityAudit(
        schema=GENERATED_CLASS_REACHABILITY_AUDIT_SCHEMA,
        status=GENERATED_CLASS_REACHABILITY_AUDIT_STATUS,
        development_only=True,
        protocol_schema=protocol.schema,
        phase_length_manifest_sha256=protocol.phase_length_manifest_sha256,
        target_name=target.name,
        target_whole_tree_digest=target.whole_tree_digest,
        required_left_parent_digest=left_digest,
        required_right_parent_digest=right_digest,
        initial_target_active_occurrences=target_occurrences[0],
        initial_target_candidate_occurrences=target_occurrences[1],
        initial_left_parent_active_occurrences=left_occurrences,
        initial_right_parent_active_occurrences=right_occurrences,
        initialization_structure_key_invariant_observed=key_invariant,
        second_d_phase_index=_SECOND_D_PHASE_INDEX,
        second_d_start_step_zero_based=start,
        second_d_stop_step_zero_based_exclusive=stop,
        second_d_curation_step_counts_one_based=events,
        second_d_curation_opportunities=len(events),
        raw_parent_slots=raw_slots,
        required_right_raw_index=_REQUIRED_RIGHT_RAW_INDEX,
        required_right_raw_slot=required_right_slot,
        robust_recursive_partner_rule=(
            "fallback_parent_b_uniform_over_depth_zero_eligible_active_slots"
        ),
        split_key_categorical_model_used=True,
        conditional_hazard_chain_rule_used=True,
        cross_event_independence_assumed=False,
        operation_prior_float32=operation_prior,
        op_logits_float32_bits=_uint32_bits(logits),
        op_probabilities_float32_bits=probability_bits,
        zero_prior_operation_logits_are_negative_infinity=True,
        raw_operation_probability_is_exact_zero=True,
        gate_mass_upper_bound_numerator=gate_mass.numerator,
        gate_mass_upper_bound_denominator=gate_mass.denominator,
        required_right_probability_numerator=required_right_mass.numerator,
        required_right_probability_denominator=required_right_mass.denominator,
        optimistic_required_left_parent_probability_numerator=(
            optimistic_left_mass.numerator
        ),
        optimistic_required_left_parent_probability_denominator=(
            optimistic_left_mass.denominator
        ),
        optimistic_proposal_application_probability_numerator=(
            optimistic_application_mass.numerator
        ),
        optimistic_proposal_application_probability_denominator=(
            optimistic_application_mass.denominator
        ),
        per_candidate_proposal_d_birth_upper_bound_numerator=(
            per_proposal_upper.numerator
        ),
        per_candidate_proposal_d_birth_upper_bound_denominator=(
            per_proposal_upper.denominator
        ),
        candidate_channel_any_birth_upper_bound_numerator=any_birth_upper.numerator,
        candidate_channel_any_birth_upper_bound_denominator=any_birth_upper.denominator,
        candidate_channel_any_birth_upper_bound_float64=float(any_birth_upper),
        candidate_channel_bound_scope=(
            "counterfactual_second_d_candidate_only_no_promotion_cascade_epoch"
        ),
        candidate_channel_bound_applies_to_current_lifecycle=False,
        protocol_post_scrub_generation_freeze_complete=(
            prerequisites.post_scrub_generation_freeze_complete
        ),
        protocol_fresh_reacquisition_generation_epoch_complete=(
            prerequisites.fresh_reacquisition_generation_epoch_complete
        ),
        protocol_fresh_reacquisition_generation_key_namespace_complete=(
            prerequisites.fresh_reacquisition_generation_key_namespace_complete
        ),
        nontrivial_state_independent_lower_bound_derived=False,
        whole_lifecycle_d_birth_probability_identified=False,
        whole_lifecycle_d_reacquisition_probability_identified=False,
        adequate_reacquisition_probability_demonstrated=False,
        reviewed_compositional_features_module_byte_sha256=module_byte_sha256,
        reviewed_init_source_sha256=init_source_sha256,
        reviewed_op_logits_source_sha256=op_logits_source_sha256,
        reviewed_generate_one_source_sha256=generate_one_source_sha256,
        reviewed_curation_stage_guidance_source_sha256=(
            curation_stage_guidance_source_sha256
        ),
        reviewed_cascade_replace_source_sha256=cascade_replace_source_sha256,
        reviewed_cascade_replace_with_mask_source_sha256=(
            cascade_replace_with_mask_source_sha256
        ),
        reviewed_update_source_sha256=update_source_sha256,
        blockers=_BLOCKERS,
        minimum_identification_contract=_MINIMUM_IDENTIFICATION_CONTRACT,
        resource_contract=resources,
        operation_contract=operations,
        runner_execution_authorized=False,
        threshold_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
    )


def build_generated_class_reachability_audit() -> GeneratedClassReachabilityAudit:
    """Build the canonical negative audit without running a generated life."""

    return _derive_audit()


def validate_generated_class_reachability_audit(
    audit: GeneratedClassReachabilityAudit,
) -> GeneratedClassReachabilityAudit:
    """Strictly reconstruct and validate every field of a development audit."""

    if type(audit) is not GeneratedClassReachabilityAudit:
        raise TypeError("audit must be an exact GeneratedClassReachabilityAudit")
    expected = _derive_audit()
    for field in dataclasses.fields(GeneratedClassReachabilityAudit):
        if getattr(audit, field.name) != getattr(expected, field.name):
            raise ValueError(f"generated-class reachability audit mismatch: {field.name}")
    return audit


__all__ = [
    "GENERATED_CLASS_REACHABILITY_AUDIT_SCHEMA",
    "GENERATED_CLASS_REACHABILITY_AUDIT_STATUS",
    "GeneratedClassReachabilityAudit",
    "GeneratedClassReachabilityOperationContract",
    "GeneratedClassReachabilityResourceContract",
    "build_generated_class_reachability_audit",
    "validate_generated_class_reachability_audit",
]
