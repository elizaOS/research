# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,union-attr"
"""Private compact event path for the primitive HCCL continual dyad.

This development-only runner seam deliberately targets bit-exact persistent
destination state and event/B/M/P/PP transcript equivalence with the audited
two-phase transaction.  It does *not* target audit-work equivalence.  The
ordinary transaction API remains the integrity and fault-injection reference;
this module calls the same learning donors once in process, checks their
returned commit flags, seals one destination, and publishes it only after all
operational checks succeed.

The kernel relies on explicitly enumerated private transaction composition
helpers.  That coupling is intentional and tested: a donor refactor must force
an operational-path review.  No artifact, evidence, benchmark, threshold,
seed, dispatch, or promotion authority is conferred here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.context_lineage_retention_seam import (
    ContextLineageRetentionPreparation,
    ContextLineageRetentionStepResult,
)
from alberta_framework.core.external_learned_state_live_memory_action_stack_adapter import (
    ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
)
from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
)
from alberta_framework.core.hccl_continual_dyad_transaction import (
    HCCLContinualDyadActionBinding,
    HCCLContinualDyadState,
    HCCLContinualDyadTransaction,
)
from alberta_framework.core.hccl_memory_credit_estimands import (
    derive_hccl_memory_credit_estimands,
)
from alberta_framework.core.hccl_world_attribution_adapter import (
    HCCLWorldAttributionAdapterResult,
)
from alberta_framework.core.prototype_factorized_partner_planner import (
    PrototypeFactorizedPartnerTransitionResult,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
)

HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA = (
    "alberta.hccl-continual-dyad-operational-work.v1"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA = (
    "alberta.hccl-continual-dyad-operational-transcript.v1"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA = (
    "alberta.hccl-continual-dyad-operational-result.v1"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_STATUS = (
    "l0-development-private-primitive-hccl-operational-runner"
)
HCCL_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL = "L0"
HCCL_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS = (
    "private-runner-only-not-a-public-transaction-api",
    "primitive-hccl-dyad-only",
    "trusted-in-process-donors-only",
    "audit-work-equivalence-is-explicitly-not-targeted",
    "fault-injection-must-use-the-two-phase-reference-api",
    "persistent-state-and-event-bmp-pp-transcript-equivalence-awaits-differential-run",
    "periodic-full-state-validation-is-runner-configured",
    "final-full-state-validation-is-mandatory",
    "host-eager-only",
    "no-output-artifact-threshold-seed-evidence-benchmark-or-promotion-authority",
)

# Every private transaction dependency used by the compact kernel.  Keep this
# exact and review it whenever the primitive transaction changes.
HCCL_CONTINUAL_DYAD_OPERATIONAL_PRIVATE_DEPENDENCIES = (
    "HCCLContinualDyadTransaction._causal_core_memory_event_inputs",
    "HCCLContinualDyadTransaction._composed_observation",
    "HCCLContinualDyadTransaction._hard_action_masks",
    "HCCLContinualDyadTransaction._horde_targets",
    "HCCLContinualDyadTransaction._make_binding",
    "HCCLContinualDyadTransaction._memory_feedback",
    "HCCLContinualDyadTransaction._planner_binding_words",
    "HCCLContinualDyadTransaction._prototype",
    "HCCLContinualDyadTransaction._seal_state",
    "HCCLContinualDyadTransaction._state_contract",
    "HCCLContinualDyadTransaction._transition",
)

_N_AGENTS = 2
_N_ACTIONS = 2
_PP_SLOT = HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER.index("PP-planner")
_TOKEN_NBYTES = 32


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact int")
    exact = value
    if exact < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return exact


def _exact_pair(value: object, *, name: str) -> tuple[int, int]:
    if type(value) is not tuple or len(cast(tuple[object, ...], value)) != _N_AGENTS:
        raise TypeError(f"{name} must be an exact two-int tuple")
    pair = cast(tuple[object, object], value)
    return (
        _exact_int(pair[0], name=f"{name}[0]"),
        _exact_int(pair[1], name=f"{name}[1]"),
    )


def _host_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray[Any, Any]:
    try:
        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a concrete array") from error
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return array


def _host_true(value: object, *, name: str) -> bool:
    array = _host_array(
        value,
        name=name,
        shape=(),
        dtype=np.dtype(np.bool_),
    )
    if not bool(array):
        raise ValueError(f"{name} must be exact True")
    return True


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(
        left_leaves
    ) != len(right_leaves):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if not hasattr(left_leaf, "shape") or not hasattr(right_leaf, "shape"):
            return False
        if (
            left_leaf.shape != right_leaf.shape
            or left_leaf.dtype != right_leaf.dtype
        ):
            return False
        typed_key = jax.dtypes.issubdtype(left_leaf.dtype, jax.dtypes.prng_key)
        left_value = jr.key_data(left_leaf) if typed_key else left_leaf
        right_value = jr.key_data(right_leaf) if typed_key else right_leaf
        left_array = np.asarray(jax.device_get(left_value))
        right_array = np.asarray(jax.device_get(right_value))
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
            or left_array.tobytes(order="C") != right_array.tobytes(order="C")
        ):
            return False
    return True


def _increment_clock_words(words: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    high = int(words[0])
    low = int(words[1])
    if high == 2**32 - 1 and low == 2**32 - 1:
        raise OverflowError("operational transcript clock exhausted uint64 capacity")
    low += 1
    if low == 2**32:
        high += 1
        low = 0
    return np.asarray((high, low), dtype=np.uint32)


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalWork:
    """Truthful logical calls made by one compact operational event."""

    schema: str
    world_event_preparations: int
    action_binding_constructions: int
    action_receipt_bindings: int
    memory_metadata_derivations: int
    memory_metadata_records: int
    context_preparations: tuple[int, int]
    hccl_stage_calls: int
    world_proposal_calls: int
    attribution_proposal_calls: int
    memory_credit_panel_derivations: int
    horde_target_derivations: int
    context_steps: tuple[int, int]
    memory_feedback_derivations: tuple[int, int]
    transition_constructions: tuple[int, int]
    action_stack_memory_preparations: tuple[int, int]
    planner_completed_transition_calls: int
    final_action_bindings: tuple[int, int]
    persistent_state_seals: int
    runner_checkpoint_state_validations: int
    through_memory_transaction_seals: int
    prepared_transaction_seals: int
    outer_preparation_receipts: int
    child_integrity_receipts: tuple[int, int]
    child_adoption_calls: tuple[int, int]
    outer_prepared_reconstructions: int
    child_finalization_reconstructions: tuple[int, int]
    audit_work_equivalence_targeted: bool
    persistent_state_and_transcript_bit_equivalence_targeted: bool

    def __post_init__(self) -> None:
        if self.schema != HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA:
            raise ValueError("operational work schema is unsupported")
        scalar_counts = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
            if field.name
            not in {
                "schema",
                "context_preparations",
                "context_steps",
                "memory_feedback_derivations",
                "transition_constructions",
                "action_stack_memory_preparations",
                "final_action_bindings",
                "child_integrity_receipts",
                "child_adoption_calls",
                "child_finalization_reconstructions",
                "audit_work_equivalence_targeted",
                "persistent_state_and_transcript_bit_equivalence_targeted",
            }
        }
        for name, value in scalar_counts.items():
            _exact_int(value, name=f"work.{name}")
        pair_names = (
            "context_preparations",
            "context_steps",
            "memory_feedback_derivations",
            "transition_constructions",
            "action_stack_memory_preparations",
            "final_action_bindings",
            "child_integrity_receipts",
            "child_adoption_calls",
            "child_finalization_reconstructions",
        )
        for name in pair_names:
            _exact_pair(getattr(self, name), name=f"work.{name}")
        exact_scalars = {
            "world_event_preparations": 1,
            "action_binding_constructions": 1,
            "action_receipt_bindings": 3,
            "memory_metadata_derivations": 1,
            "memory_metadata_records": 2,
            "context_preparations": (1, 1),
            "hccl_stage_calls": 1,
            "world_proposal_calls": 8,
            "attribution_proposal_calls": 8,
            "memory_credit_panel_derivations": 1,
            "horde_target_derivations": 1,
            "context_steps": (1, 1),
            "memory_feedback_derivations": (1, 1),
            "transition_constructions": (1, 1),
            "action_stack_memory_preparations": (1, 1),
            "planner_completed_transition_calls": 1,
            "final_action_bindings": (1, 1),
            "persistent_state_seals": 1,
            "through_memory_transaction_seals": 0,
            "prepared_transaction_seals": 0,
            "outer_preparation_receipts": 0,
            "child_integrity_receipts": (0, 0),
            "child_adoption_calls": (0, 0),
            "outer_prepared_reconstructions": 0,
            "child_finalization_reconstructions": (0, 0),
        }
        for name, expected in exact_scalars.items():
            if getattr(self, name) != expected:
                label = name.replace("_", " ")
                raise ValueError(f"operational {label} must equal {expected!r}")
        if self.runner_checkpoint_state_validations not in (0, 1):
            raise ValueError("operational checkpoint validation count must be zero or one")
        if type(self.audit_work_equivalence_targeted) is not bool:
            raise TypeError("audit_work_equivalence_targeted must be an exact bool")
        if self.audit_work_equivalence_targeted:
            raise ValueError("operational work cannot claim audit-work equivalence")
        if type(self.persistent_state_and_transcript_bit_equivalence_targeted) is not bool:
            raise TypeError(
                "persistent_state_and_transcript_bit_equivalence_targeted must be an exact bool"
            )
        if not self.persistent_state_and_transcript_bit_equivalence_targeted:
            raise ValueError("operational work must state its narrow equivalence target")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


def _make_operational_work(
    *,
    world_proposal_calls: int,
    attribution_proposal_calls: int,
    runner_checkpoint_state_validations: int,
) -> HCCLContinualDyadOperationalWork:
    world_calls = _exact_int(world_proposal_calls, name="world proposal calls")
    attribution_calls = _exact_int(
        attribution_proposal_calls,
        name="attribution proposal calls",
    )
    checkpoints = _exact_int(
        runner_checkpoint_state_validations,
        name="checkpoint validation count",
    )
    return HCCLContinualDyadOperationalWork(
        schema=HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA,
        world_event_preparations=1,
        action_binding_constructions=1,
        action_receipt_bindings=3,
        memory_metadata_derivations=1,
        memory_metadata_records=2,
        context_preparations=(1, 1),
        hccl_stage_calls=1,
        world_proposal_calls=world_calls,
        attribution_proposal_calls=attribution_calls,
        memory_credit_panel_derivations=1,
        horde_target_derivations=1,
        context_steps=(1, 1),
        memory_feedback_derivations=(1, 1),
        transition_constructions=(1, 1),
        action_stack_memory_preparations=(1, 1),
        planner_completed_transition_calls=1,
        final_action_bindings=(1, 1),
        persistent_state_seals=1,
        runner_checkpoint_state_validations=checkpoints,
        through_memory_transaction_seals=0,
        prepared_transaction_seals=0,
        outer_preparation_receipts=0,
        child_integrity_receipts=(0, 0),
        child_adoption_calls=(0, 0),
        outer_prepared_reconstructions=0,
        child_finalization_reconstructions=(0, 0),
        audit_work_equivalence_targeted=False,
        persistent_state_and_transcript_bit_equivalence_targeted=True,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalTranscript:
    """Compact exact evaluator transcript for the committed PP successor."""

    schema: str
    event: HCCLCausalCoreEventReceipt
    binding: HCCLContinualDyadActionBinding
    pp_proposal: HCCLCausalCoreProposal
    pre_transaction_words: Array
    post_transaction_words: Array

    def __post_init__(self) -> None:
        if self.schema != HCCL_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA:
            raise ValueError("operational transcript schema is unsupported")
        if type(self.event) is not HCCLCausalCoreEventReceipt:
            raise TypeError("operational transcript event has the wrong type")
        if type(self.binding) is not HCCLContinualDyadActionBinding:
            raise TypeError("operational transcript binding has the wrong type")
        if type(self.pp_proposal) is not HCCLCausalCoreProposal:
            raise TypeError("operational transcript PP proposal has the wrong type")
        pre = _host_array(
            self.pre_transaction_words,
            name="transcript.pre_transaction_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        post = _host_array(
            self.post_transaction_words,
            name="transcript.post_transaction_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        event_words = _host_array(
            self.event.source_step_words,
            name="transcript.event.source_step_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        proposal_words = _host_array(
            self.pp_proposal.source_step_words,
            name="transcript.pp_proposal.source_step_words",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        proposal_actions = _host_array(
            self.pp_proposal.joint_action_ids,
            name="transcript.pp_proposal.joint_action_ids",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.int32),
        )
        final_actions = _host_array(
            self.binding.final_actions,
            name="transcript.binding.final_actions",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.int32),
        )
        if not (
            np.array_equal(event_words, pre)
            and np.array_equal(proposal_words, pre)
            and np.array_equal(post, _increment_clock_words(pre))
            and np.array_equal(proposal_actions, final_actions)
        ):
            raise ValueError("operational event/B/M/P/PP transcript is not clock/action bound")
        if not np.array_equal(
            np.asarray(jax.device_get(self.pp_proposal.event_content_tag_words)),
            np.asarray(jax.device_get(self.event.content_tag_words)),
        ):
            raise ValueError("operational PP proposal names a foreign event")
        _host_true(self.pp_proposal.valid, name="transcript.pp_proposal.valid")
        regime = _host_array(
            self.pp_proposal.evaluator_regime_id,
            name="transcript.pp_proposal.evaluator_regime_id",
            shape=(),
            dtype=np.dtype(np.int32),
        )
        if not 0 <= int(regime) < 4:
            raise ValueError("operational PP evaluator regime lies outside the fixed domain")
        for name, value, shape in (
            ("task_score", self.pp_proposal.signals.task_score, ()),
            ("net_reward", self.pp_proposal.signals.net_reward, (_N_AGENTS,)),
            ("factors.gathering", self.pp_proposal.factors.gathering, ()),
            ("factors.velocity", self.pp_proposal.factors.velocity, ()),
            ("factors.convention_clean", self.pp_proposal.factors.convention_clean, ()),
            ("factors.convention_noisy", self.pp_proposal.factors.convention_noisy, ()),
        ):
            array = _host_array(
                value,
                name=f"transcript.pp_proposal.{name}",
                shape=shape,
                dtype=np.dtype(np.float32),
            )
            if not bool(np.all(np.isfinite(array))):
                raise ValueError(f"operational PP {name} must be finite")


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLContinualDyadOperationalEventResult:
    """One committed destination and its narrow operational transcript."""

    schema: str
    state: HCCLContinualDyadState
    transcript: HCCLContinualDyadOperationalTranscript
    work: HCCLContinualDyadOperationalWork
    update_applied: bool

    def __post_init__(self) -> None:
        if self.schema != HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA:
            raise ValueError("operational result schema is unsupported")
        if type(self.state) is not HCCLContinualDyadState:
            raise TypeError("operational result state has the wrong type")
        if type(self.transcript) is not HCCLContinualDyadOperationalTranscript:
            raise TypeError("operational result transcript has the wrong type")
        if type(self.work) is not HCCLContinualDyadOperationalWork:
            raise TypeError("operational result work has the wrong type")
        if type(self.update_applied) is not bool:
            raise TypeError("operational result update_applied must be an exact bool")
        if not self.update_applied:
            raise ValueError("an operational result may expose only a committed candidate")
        destination_words = _host_array(
            self.state.hccl_state.world_state.step_words,
            name="operational result destination clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        post_words = _host_array(
            self.transcript.post_transaction_words,
            name="operational result transcript post clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        if not np.array_equal(destination_words, post_words):
            raise ValueError("operational destination and PP transcript clocks differ")
        if not _tree_exact_equal(
            self.state.hccl_state.world_state,
            self.transcript.pp_proposal.candidate_state,
        ):
            raise ValueError("operational destination is not the exact committed PP world")


class HCCLContinualDyadOperationalError(RuntimeError):
    """Fail-closed operational error; the executor keeps its complete source."""

    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        super().__init__(f"primitive HCCL operational event failed during {stage}: {detail}")


def _require_hccl_commit(result: HCCLWorldAttributionAdapterResult) -> None:
    if type(result) is not HCCLWorldAttributionAdapterResult:
        raise TypeError("HCCL stage returned a malformed result")
    for name in (
        "source_state_valid",
        "world_source_clock_bound",
        "event_receipt_valid",
        "event_receipt_identity_bound",
        "action_receipt_identities_bound",
        "all_world_proposals_valid",
        "equal_action_world_payloads_bit_exact",
        "causal_core_signal_contract_valid",
        "world_duplicate_mm_bit_exact",
        "downstream_candidate_valid",
        "candidate_state_valid",
        "update_applied",
    ):
        _host_true(getattr(result, name), name=f"hccl_result.{name}")
    world_calls = int(
        _host_array(
            result.work.world_proposal_calls,
            name="hccl_result.work.world_proposal_calls",
            shape=(),
            dtype=np.dtype(np.int32),
        )
    )
    attribution_calls = int(
        _host_array(
            result.work.attribution_proposal_calls,
            name="hccl_result.work.attribution_proposal_calls",
            shape=(),
            dtype=np.dtype(np.int32),
        )
    )
    committed = int(
        _host_array(
            result.work.committed_pp_world_successors,
            name="hccl_result.work.committed_pp_world_successors",
            shape=(),
            dtype=np.dtype(np.int32),
        )
    )
    if (world_calls, attribution_calls, committed) != (8, 8, 1):
        raise ValueError("HCCL operational stage did not commit one exact eight-vertex PP row")


def _require_context_commit(result: ContextLineageRetentionStepResult, *, index: int) -> None:
    if type(result) is not ContextLineageRetentionStepResult:
        raise TypeError(f"agent {index} context donor returned a malformed result")
    for name in (
        "source_state_valid",
        "preparation_integrity_valid",
        "preparation_matches_source",
        "protection_binding_valid",
        "birth_binding_valid",
        "context_update_applied",
        "lineage_update_applied",
        "candidate_state_valid",
        "context_owner_committed",
        "lineage_owner_committed",
        "protection_snapshotted_before_outcome",
        "update_applied",
    ):
        _host_true(getattr(result, name), name=f"context_result[{index}].{name}")
    changed = _host_array(
        result.current_outcome_changed_current_eviction_protection,
        name=(
            f"context_result[{index}]."
            "current_outcome_changed_current_eviction_protection"
        ),
        shape=(),
        dtype=np.dtype(np.bool_),
    )
    if bool(changed):
        raise ValueError("current outcome changed pre-outcome eviction protection")


def _require_memory_commit(
    result: ExternalLearnedStateLiveMemoryActionStackMemoryPreparation,
    *,
    index: int,
) -> None:
    if type(result) is not ExternalLearnedStateLiveMemoryActionStackMemoryPreparation:
        raise TypeError(f"agent {index} memory donor returned a malformed preparation")
    _host_true(result.preflight_valid, name=f"memory_preparation[{index}].preflight_valid")
    _host_true(
        result.transition_final_action_exact,
        name=f"memory_preparation[{index}].transition_final_action_exact",
    )
    _host_true(
        result.preparation_valid,
        name=f"memory_preparation[{index}].preparation_valid",
    )


def _require_planner_commit(result: PrototypeFactorizedPartnerTransitionResult) -> None:
    if type(result) is not PrototypeFactorizedPartnerTransitionResult:
        raise TypeError("planner donor returned a malformed transition result")
    diagnostics = result.diagnostics
    _host_true(diagnostics.candidate_valid, name="planner_result.candidate_valid")
    _host_true(
        diagnostics.transaction_committed,
        name="planner_result.transaction_committed",
    )
    for name in (
        "source_cache_valid",
        "behavior_update_applied",
        "grounded_update_applied",
        "prediction_matches_cache",
        "candidate_clock_aligned",
        "candidate_generation_aligned",
        "next_observations_match",
    ):
        flags = _host_array(
            getattr(diagnostics, name),
            name=f"planner_result.{name}",
            shape=(_N_AGENTS,),
            dtype=np.dtype(np.bool_),
        )
        if not bool(np.all(flags)):
            raise ValueError(f"planner_result.{name} must be all true")
    _host_true(
        diagnostics.next_prepare.pair_committed,
        name="planner_result.next_prepare.pair_committed",
    )


def _require_finalization_commit(
    result: ExternalLearnedStateLiveMemoryActionStackFinalizedTransition,
    *,
    index: int,
) -> None:
    if type(result) is not ExternalLearnedStateLiveMemoryActionStackFinalizedTransition:
        raise TypeError(f"agent {index} final-action donor returned a malformed result")
    _host_true(
        result.finalization_valid,
        name=f"finalization[{index}].finalization_valid",
    )
    evaluations = int(
        _host_array(
            result.bind_work.final_action_binding_evaluations,
            name=f"finalization[{index}].bind_work.final_action_binding_evaluations",
            shape=(),
            dtype=np.dtype(np.int32),
        )
    )
    if evaluations != 1:
        raise ValueError(f"agent {index} final-action donor did not bind exactly once")


def _execute_operational_event(
    transaction: HCCLContinualDyadTransaction,
    state: HCCLContinualDyadState,
    next_hard_action_masks: Array,
) -> HCCLContinualDyadOperationalEventResult:
    """Execute one trusted in-process event without audit reconstruction.

    The private caller must own a source accepted by a previous exact runner
    checkpoint (or emitted by this kernel).  This function is intentionally
    not exported as a public transaction operation.
    """

    if type(transaction) is not HCCLContinualDyadTransaction:
        raise TypeError("transaction must be an exact primitive HCCL dyad transaction")
    transaction._state_contract(state)
    masks = transaction._hard_action_masks(
        next_hard_action_masks,
        name="next_hard_action_masks",
    )
    if _contains_tracer((state, masks)):
        raise TypeError("primitive HCCL operational execution is host/eager-only")

    event = transaction.hccl.world.prepare_event(state.hccl_state.world_state)
    binding = transaction._make_binding(state, event)
    event_inputs = transaction._causal_core_memory_event_inputs(event)
    agents = (state.agent_0_state, state.agent_1_state)
    contexts = (state.context_0_state, state.context_1_state)
    adapters = (transaction.agent_0, transaction.agent_1)

    context_preparations = tuple(
        transaction.context.prepare(
            contexts[index],
            jax.nn.one_hot(
                binding.final_actions[1 - index],
                _N_ACTIONS,
                dtype=jnp.float32,
            ),
            binding.final_actions[index],
        )
        for index in range(_N_AGENTS)
    )
    if not all(
        type(item) is ContextLineageRetentionPreparation
        for item in context_preparations
    ):
        raise TypeError("context donor returned a malformed preparation")

    hccl_result = transaction.hccl.stage(
        state.hccl_state,
        event,
        binding.base,
        binding.memory,
        binding.planner,
        downstream_candidate_valid=jnp.asarray(True, dtype=jnp.bool_),
    )
    _require_hccl_commit(hccl_result)
    pp_proposal = cast(
        HCCLCausalCoreProposal,
        jax.tree.map(lambda leaf: leaf[_PP_SLOT], hccl_result.world_proposals),
    )
    proposals = hccl_result.world_proposals
    # The estimand API accepts typed signal rows, not an indexable dataclass.
    credit_panel = derive_hccl_memory_credit_estimands(
        mm=cast(Any, jax.tree.map(lambda leaf: leaf[0], proposals.signals)),
        b0m1=cast(Any, jax.tree.map(lambda leaf: leaf[1], proposals.signals)),
        m0b1=cast(Any, jax.tree.map(lambda leaf: leaf[2], proposals.signals)),
        bb=cast(Any, jax.tree.map(lambda leaf: leaf[3], proposals.signals)),
    )
    _host_true(
        credit_panel.algebra.all_identities_hold,
        name="memory_credit_panel.algebra.all_identities_hold",
    )
    credits = (
        credit_panel.baseline_context_direct_effect.net_reward[0, 0],
        credit_panel.baseline_context_direct_effect.net_reward[1, 1],
    )
    horde_cumulants, horde_discounts = transaction._horde_targets(
        pp_proposal,
        pp_proposal.signals,
        binding.final_actions,
    )
    context_results = tuple(
        transaction.context.step(
            contexts[index],
            context_preparations[index],
            pp_proposal.signals.task_score,
        )
        for index in range(_N_AGENTS)
    )
    for index, context_result in enumerate(context_results):
        _require_context_commit(context_result, index=index)

    next_raw = jnp.stack(
        tuple(
            transaction._composed_observation(
                pp_proposal.next_observation[index],
                context_results[index].state,
            )
            for index in range(_N_AGENTS)
        )
    ).astype(jnp.float32)
    feedback = tuple(
        transaction._memory_feedback(agents[index], credits[index])
        for index in range(_N_AGENTS)
    )
    transitions = tuple(
        transaction._transition(
            agents[index],
            executed_action=binding.final_actions[index],
            reward=pp_proposal.signals.net_reward[index],
            next_observation=next_raw[index],
            horde_cumulants=horde_cumulants[index],
            horde_discounts=horde_discounts[index],
        )
        for index in range(_N_AGENTS)
    )
    memory_preparations = tuple(
        adapters[index].prepare_memory_transition(
            agents[index],
            transitions[index],
            event_inputs[index],
            masks[index],
            feedback[index],
            None,
            partner_policy_fusion_input=None,
            partner_policy_fusion_feedback=None,
            extended_action_mask=None,
        )
        for index in range(_N_AGENTS)
    )
    for index, memory_result in enumerate(memory_preparations):
        _require_memory_commit(memory_result, index=index)

    post_memory_prototypes = tuple(
        transaction._prototype(item.memory_candidate_state)
        for item in memory_preparations
    )
    planner_result = transaction.planner.completed_transition(
        state.planner_state,
        transaction._prototype(agents[0]),
        transaction._prototype(agents[1]),
        post_memory_prototypes[0],
        post_memory_prototypes[1],
        binding.final_actions,
        pp_proposal.signals.net_reward,
        jnp.stack(
            tuple(item.current_raw_observation for item in post_memory_prototypes)
        ).astype(jnp.float32),
        jnp.asarray(transaction.config.discount, dtype=jnp.float32),
        masks,
    )
    _require_planner_commit(planner_result)
    planner_words = transaction._planner_binding_words(
        planner_result.state,
        planner_result.prototype_agent_0,
        planner_result.prototype_agent_1,
    )
    planner_agents = (planner_result.state.agent_0, planner_result.state.agent_1)
    selected_prototypes = (
        planner_result.prototype_agent_0,
        planner_result.prototype_agent_1,
    )
    planner_before = tuple(
        jnp.where(
            planner_agents[index].cache.planner_consumed,
            planner_result.diagnostics.next_prepare.proposed_actions[index],
            memory_preparations[index]
            .memory_candidate_state.action_binding.memory_action,
        ).astype(jnp.int32)
        for index in range(_N_AGENTS)
    )
    finalizations = tuple(
        adapters[index].bind_final_action(
            memory_preparations[index],
            selected_prototypes[index],
            planner_action_before_mask=planner_before[index],
            planner_candidate_words=planner_words,
            planner_consumed=planner_agents[index].cache.planner_consumed,
        )
        for index in range(_N_AGENTS)
    )
    for index, finalization in enumerate(finalizations):
        _require_finalization_commit(finalization, index=index)

    candidate_state = transaction._seal_state(
        HCCLContinualDyadState(
            config_token=state.config_token,
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
            hccl_state=hccl_result.state,
            agent_0_state=finalizations[0].candidate_state,
            agent_1_state=finalizations[1].candidate_state,
            planner_state=planner_result.state,
            context_0_state=context_results[0].state,
            context_1_state=context_results[1].state,
        )
    )
    transcript = HCCLContinualDyadOperationalTranscript(
        schema=HCCL_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA,
        event=event,
        binding=binding,
        pp_proposal=pp_proposal,
        pre_transaction_words=hccl_result.pre_transaction_words,
        post_transaction_words=hccl_result.post_transaction_words,
    )
    work = _make_operational_work(
        world_proposal_calls=int(jax.device_get(hccl_result.work.world_proposal_calls)),
        attribution_proposal_calls=int(
            jax.device_get(hccl_result.work.attribution_proposal_calls)
        ),
        runner_checkpoint_state_validations=0,
    )
    return HCCLContinualDyadOperationalEventResult(
        schema=HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA,
        state=candidate_state,
        transcript=transcript,
        work=work,
        update_applied=True,
    )


class _HCCLContinualDyadOperationalExecutor:
    """Own and publish compact destinations with optional exact checkpoints."""

    __slots__ = (
        "_transaction",
        "_state",
        "_absolute_step",
        "_maximum_transitions",
        "_checkpoint_interval",
    )

    def __init__(
        self,
        transaction: HCCLContinualDyadTransaction,
        state: HCCLContinualDyadState,
        *,
        checkpoint_interval: int | None,
    ) -> None:
        if type(transaction) is not HCCLContinualDyadTransaction:
            raise TypeError("transaction must be an exact primitive HCCL dyad transaction")
        transaction._state_contract(state)
        if checkpoint_interval is not None:
            _exact_int(checkpoint_interval, name="checkpoint_interval", minimum=1)
        if _contains_tracer(state):
            raise TypeError("primitive HCCL operational executor is host/eager-only")
        try:
            _host_true(transaction.state_valid(state), name="initial state validity")
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalError(
                "initial-checkpoint",
                str(error),
            ) from error
        words = _host_array(
            state.hccl_state.world_state.step_words,
            name="initial state clock",
            shape=(2,),
            dtype=np.dtype(np.uint32),
        )
        if int(words[0]) != 0:
            raise ValueError("fixed primitive HCCL lives require a zero high clock word")
        maximum = _exact_int(
            transaction.config.hccl.world_config.maximum_committed_transitions,
            name="world maximum_committed_transitions",
            minimum=1,
        )
        if int(words[1]) > maximum:
            raise ValueError("initial state clock exceeds the configured fixed life")
        self._transaction = transaction
        self._state = state
        self._absolute_step = int(words[1])
        self._maximum_transitions = maximum
        self._checkpoint_interval = checkpoint_interval

    @property
    def state(self) -> HCCLContinualDyadState:
        return self._state

    @property
    def absolute_step(self) -> int:
        return self._absolute_step

    @property
    def checkpoint_interval(self) -> int | None:
        return self._checkpoint_interval

    def step(self, next_hard_action_masks: Array) -> HCCLContinualDyadOperationalEventResult:
        """Compute, check, then publish one complete destination or retain source."""

        if self._absolute_step >= self._maximum_transitions:
            raise HCCLContinualDyadOperationalError(
                "bounds",
                "configured fixed life has no remaining transition capacity",
            )
        source = self._state
        try:
            result = _execute_operational_event(
                self._transaction,
                source,
                next_hard_action_masks,
            )
        except HCCLContinualDyadOperationalError:
            raise
        except Exception as error:
            raise HCCLContinualDyadOperationalError("learning-donors", str(error)) from error
        if type(result) is not HCCLContinualDyadOperationalEventResult:
            raise HCCLContinualDyadOperationalError(
                "result-contract",
                "kernel returned a malformed operational result",
            )

        next_step = self._absolute_step + 1
        final_step = next_step == self._maximum_transitions
        periodic_checkpoint_due = (
            self._checkpoint_interval is not None
            and next_step % self._checkpoint_interval == 0
        )
        checkpoint_due = final_step or periodic_checkpoint_due
        if checkpoint_due:
            checkpoint_stage = "final-checkpoint" if final_step else "periodic-checkpoint"
            try:
                _host_true(
                    self._transaction.state_valid(result.state),
                    name=f"{checkpoint_stage} candidate validity",
                )
            except (AttributeError, RuntimeError, TypeError, ValueError) as error:
                raise HCCLContinualDyadOperationalError(
                    checkpoint_stage,
                    str(error),
                ) from error
            result = dataclasses.replace(
                result,
                work=dataclasses.replace(
                    result.work,
                    runner_checkpoint_state_validations=1,
                ),
            )

        # Publication is the only executor mutation and occurs after donor,
        # result, and optional full-checkpoint validation has succeeded.
        self._state = result.state
        self._absolute_step = next_step
        return result

    def validate_checkpoint(self) -> HCCLContinualDyadState:
        """Run one explicit exact validation without changing the owned state."""

        try:
            _host_true(
                self._transaction.state_valid(self._state),
                name="explicit operational checkpoint",
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise HCCLContinualDyadOperationalError(
                "explicit-checkpoint",
                str(error),
            ) from error
        return self._state


__all__ = (
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_EVIDENCE_LEVEL",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_LIMITATIONS",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_PRIVATE_DEPENDENCIES",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_RESULT_SCHEMA",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_STATUS",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_TRANSCRIPT_SCHEMA",
    "HCCL_CONTINUAL_DYAD_OPERATIONAL_WORK_SCHEMA",
    "HCCLContinualDyadOperationalError",
    "HCCLContinualDyadOperationalEventResult",
    "HCCLContinualDyadOperationalTranscript",
    "HCCLContinualDyadOperationalWork",
)
