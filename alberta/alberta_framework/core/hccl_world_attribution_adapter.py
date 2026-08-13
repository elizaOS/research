# mypy: disable-error-code="call-arg"
"""Atomic HCCL causal-core world to adjacent-cube attribution transaction.

This development-only seam owns exactly one causal-core world state and one
adjacent-cube attribution state.  It consumes an already-prepared world event
receipt and exact B/M/P action receipts, constructs the donor kernel's fixed
eight vertices, evaluates eight pure same-source world proposals, and gives
those exact proposals to the existing attribution kernel.  Only the PP world
successor (slot four) can be selected, together with the kernel successor, in
one composite transaction.

The transient eight-proposal PyTree is deliberately not persistent state.  A
failed source, receipt, proposal, duplicate-MM, typed-contrast, downstream, or
candidate-state gate selects the bit-exact composite source so the identical
event and action receipts can be retried.  Integrity tags are deterministic
bindings rather than authentication.

This module does not implement agents, execute the HCCL schedule, reserve or
consume protocol seeds, write outputs, define thresholds, validate artifacts,
or confer evidence, benchmark-execution, promotion, or claim authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.hccl_causal_attribution import (
    HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER,
    HCCLActionLayer,
    HCCLActionReceipt,
    HCCLCausalAttributionConfig,
    HCCLCausalAttributionKernel,
    HCCLCausalAttributionResult,
    HCCLCausalAttributionState,
    HCCLCausalSourceReceipt,
    HCCLExogenousReceipt,
    HCCLJointActionVertex,
    HCCLTypedSignals,
    measure_hccl_causal_attribution_state_nbytes,
)
from alberta_framework.streams.hccl_causal_core import (
    HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
    HCCLCausalCoreConfig,
    HCCLCausalCoreEventReceipt,
    HCCLCausalCoreProposal,
    HCCLCausalCoreState,
    HCCLCausalCoreWorld,
    measure_hccl_causal_core_state_nbytes,
)

HCCL_WORLD_ATTRIBUTION_ADAPTER_CONFIG_SCHEMA = (
    "alberta.hccl-world-attribution-adapter-config.v1"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_STATE_SCHEMA = (
    "alberta.hccl-world-attribution-adapter-state.v1"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_RESULT_SCHEMA = (
    "alberta.hccl-world-attribution-adapter-result.v1"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_RESOURCE_SCHEMA = (
    "alberta.hccl-world-attribution-adapter-resource.v1"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_CHECKPOINT_SCHEMA = (
    "alberta.hccl-world-attribution-adapter-checkpoint.v1"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS = (
    "l0-development-world-attribution-transaction-only"
)
HCCL_WORLD_ATTRIBUTION_ADAPTER_EVIDENCE_LEVEL = "L0"
HCCL_WORLD_ATTRIBUTION_ADAPTER_LIMITATIONS = (
    "single-prepared-event-and-immediate-same-prestate-proposals-only",
    "transient-eight-world-proposal-stack-not-persistent-state",
    "integrity-tags-bind-content-but-do-not-authenticate-it",
    "composite-stage-and-prebound-scan-are-host-eager-only",
    "jit-is-supported-only-at-the-smaller-donor-kernel-boundaries",
    "agent-component-identities-are-synthetic-world-bound-placeholders",
    "caller-labelled-action-layers-are-not-authenticated-agent-outputs",
    "invalid-host-preflight-still-evaluates-eight-pure-world-proposals",
    "no-agent-or-schedule-execution",
    "no-artifact-writer-threshold-seed-evidence-promotion-or-claim-authority",
)

_N_AGENTS = 2
_N_ACTION_LAYERS = 3
_N_PROPOSALS = 8
_PP_SLOT = 4
_SOURCE_DIM = 32
_EXOGENOUS_DIM = 16
_TRANSITION_DIM = 32
_FLOAT_BOUND = 1.0e6
_UINT32_MAX = 2**32 - 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_array(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    label: str,
) -> Array:
    if getattr(value, "shape", None) != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if getattr(value, "dtype", None) != dtype:
        raise TypeError(f"{label} must have dtype {dtype}")
    return jnp.asarray(value)


def _tree_select(condition: Array, yes: Any, no: Any) -> Any:
    return jax.tree.map(lambda left, right: jnp.where(condition, left, right), yes, no)


def _contains_tracer(value: Any) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _array_words(value: Array) -> Array:
    if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
        value.dtype, jax.dtypes.prng_key
    ):
        return jr.key_data(value)
    if value.dtype == jnp.dtype(jnp.float32):
        return jax.lax.bitcast_convert_type(value, jnp.uint32)
    if value.dtype == jnp.dtype(jnp.int32):
        return jax.lax.bitcast_convert_type(value, jnp.uint32)
    if value.dtype == jnp.dtype(jnp.bool_):
        return value.astype(jnp.uint32)
    if value.dtype == jnp.dtype(jnp.uint32):
        return value
    raise TypeError("transaction equality supports typed keys, float32, int32, bool, and uint32")


def _tree_exact_equal(left: Any, right: Any) -> Bool[Array, ""]:
    if type(left) is not type(right):
        return jnp.asarray(False, dtype=jnp.bool_)
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    equal = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = cast(Array, left_leaf)
        right_array = cast(Array, right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        equal = equal & jnp.all(_array_words(left_array) == _array_words(right_array))
    return equal


def _stack_tree(items: tuple[Any, ...]) -> Any:
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *items)


def _tree_at(tree: Any, index: int | Array) -> Any:
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _salted_identity(words: Array, salt: tuple[int, int, int, int]) -> Array:
    mixed = words ^ jnp.asarray(salt, dtype=jnp.uint32)
    fallback = jnp.asarray(salt, dtype=jnp.uint32)
    return jnp.where(jnp.any(mixed != 0), mixed, fallback).astype(jnp.uint32)


def _increment_low_clock(words: Array) -> Array:
    # The fixed world lifetime is 8,998, so the low word cannot overflow here.
    return jnp.stack((words[0], words[1] + jnp.uint32(1))).astype(jnp.uint32)


def _composite_link_words(
    world_state: HCCLCausalCoreState,
    attribution_state: HCCLCausalAttributionState,
    schedule_profile: str = HCCL_CAUSAL_CORE_CANONICAL_PROFILE,
) -> Array:
    """Bind the two current donor states against accidental same-clock crossing."""

    step = world_state.step_words
    step_mix = jnp.stack(
        (
            step[0],
            step[1],
            step[0] ^ jnp.uint32(0xA17E5EED),
            step[1] ^ jnp.uint32(0xC05A1EAF),
        )
    ).astype(jnp.uint32)
    mixed = (
        world_state.content_tag_words
        ^ jnp.roll(attribution_state.last_attribution_tag_words, 1)
        ^ step_mix
    )
    if schedule_profile != HCCL_CAUSAL_CORE_CANONICAL_PROFILE:
        mixed = mixed ^ jnp.asarray(
            (0x534D4F4B, 0x45343230, 0x41445431, 0x00000001), dtype=jnp.uint32
        )
    return _salted_identity(
        mixed,
        (0x434F4D50, 0x4C494E4B, 0x574F524C, 0x44415454),
    )


def _state_host_payload(state: HCCLWorldAttributionAdapterState) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for leaf in jax.tree.leaves(state):
        if not hasattr(leaf, "dtype"):
            continue
        array = cast(Array, leaf)
        if jax.dtypes.issubdtype(  # type: ignore[attr-defined]
            array.dtype, jax.dtypes.prng_key
        ):
            host = np.asarray(jr.key_data(array), dtype=np.uint32)
            dtype = "typed-threefry-key-uint32"
        else:
            host = np.asarray(array)
            dtype = str(host.dtype)
        payload.append(
            {
                "shape": list(host.shape),
                "dtype": dtype,
                "bytes_hex": np.ascontiguousarray(host).tobytes().hex(),
            }
        )
    return payload


@dataclasses.dataclass(frozen=True)
class HCCLWorldAttributionAdapterConfig:
    """Strict owner identity for the fixed donor composition."""

    proposal_owner_digest: tuple[int, ...]
    world_config: HCCLCausalCoreConfig = dataclasses.field(
        default_factory=HCCLCausalCoreConfig
    )

    def __post_init__(self) -> None:
        value = self.proposal_owner_digest
        if type(value) is not tuple or len(value) != 8:
            raise ValueError("proposal_owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(value):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"proposal_owner_digest[{index}] must be uint32")
        left, right = value[:4], value[4:]
        if not any(left) or not any(right) or left == right:
            raise ValueError("proposal owner must provide two distinct nonzero agent identities")
        if type(self.world_config) is not HCCLCausalCoreConfig:
            raise TypeError("world_config must be exact HCCLCausalCoreConfig")


@chex.dataclass(frozen=True)
class HCCLWorldAttributionAdapterState:
    """One world, one attribution state, and their deterministic integrity link."""

    world_state: HCCLCausalCoreState
    attribution_state: HCCLCausalAttributionState
    composite_link_words: UInt[Array, " 4"]


@chex.dataclass(frozen=True)
class HCCLWorldAttributionAdapterWork:
    world_proposal_calls: Int[Array, ""]
    attribution_proposal_calls: Int[Array, ""]
    designated_counterfactual_world_slots: Int[Array, ""]
    discarded_world_proposal_calls: Int[Array, ""]
    committed_pp_world_successors: Int[Array, ""]
    duplicate_mm_world_equality_checks: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLWorldAttributionAdapterResult:
    state: HCCLWorldAttributionAdapterState
    world_proposals: HCCLCausalCoreProposal
    attribution: HCCLCausalAttributionResult
    work: HCCLWorldAttributionAdapterWork
    pre_transaction_words: UInt[Array, " 2"]
    post_transaction_words: UInt[Array, " 2"]
    source_state_valid: Bool[Array, ""]
    world_source_clock_bound: Bool[Array, ""]
    event_receipt_valid: Bool[Array, ""]
    event_receipt_identity_bound: Bool[Array, ""]
    action_receipt_identities_bound: Bool[Array, ""]
    all_world_proposals_valid: Bool[Array, ""]
    equal_action_world_payloads_bit_exact: Bool[Array, ""]
    causal_core_signal_contract_valid: Bool[Array, ""]
    world_duplicate_mm_bit_exact: Bool[Array, ""]
    downstream_candidate_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]


@chex.dataclass(frozen=True)
class HCCLWorldAttributionScanResult:
    state: HCCLWorldAttributionAdapterState
    memory_total_task_score: Float[Array, " steps"]
    planner_total_task_score: Float[Array, " steps"]
    pp_minus_bb_task_score: Float[Array, " steps"]
    post_transaction_words: UInt[Array, "steps 2"]
    world_proposal_calls: Int[Array, " steps"]
    attribution_proposal_calls: Int[Array, " steps"]
    update_applied: Bool[Array, " steps"]


@dataclasses.dataclass(frozen=True)
class HCCLWorldAttributionResourceBudget:
    schema: str
    world_state_owners: int
    attribution_state_owners: int
    world_state_nbytes: int
    attribution_state_nbytes: int
    composite_link_nbytes: int
    total_persistent_state_nbytes: int
    event_receipt_nbytes: int
    world_proposal_nbytes: int
    max_transient_world_proposal_stack_nbytes: int
    max_world_proposal_calls_per_transaction: int
    max_attribution_proposal_calls_per_transaction: int
    designated_counterfactual_world_slots_per_transaction: int
    max_discarded_world_proposal_calls_per_transaction: int
    max_committed_world_successors_per_transaction: int
    maximum_committed_transactions: int
    output_write_calls: int
    artifact_bytes_written: int
    persistent_bytes_scope: str
    transient_bytes_scope: str

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class HCCLWorldAttributionCheckpoint:
    schema: str
    mechanism_status: str
    evidence_level: str
    output_writes_authorized: bool
    artifact_authorized: bool
    evidence_authorized: bool
    config: dict[str, object]
    config_sha256: str
    resource_budget: dict[str, object]
    state: HCCLWorldAttributionAdapterState
    state_nbytes: int
    state_sha256: str
    checkpoint_sha256: str


class HCCLWorldAttributionAdapter:
    """Fail-closed composite owner over the two existing donor states."""

    def __init__(self, config: HCCLWorldAttributionAdapterConfig):
        if type(config) is not HCCLWorldAttributionAdapterConfig:
            raise TypeError("config must be exact HCCLWorldAttributionAdapterConfig")
        self._config = config
        self._world = HCCLCausalCoreWorld(config.world_config)
        self._attribution = HCCLCausalAttributionKernel(
            HCCLCausalAttributionConfig(
                source_dim=_SOURCE_DIM,
                exogenous_dim=_EXOGENOUS_DIM,
                transition_dim=_TRANSITION_DIM,
                n_actions=2,
                proposal_owner_digest=config.proposal_owner_digest,
                max_transactions=config.world_config.maximum_committed_transitions,
                max_abs_source=_FLOAT_BOUND,
                max_abs_exogenous=_FLOAT_BOUND,
                max_abs_transition=_FLOAT_BOUND,
                max_abs_signal=_FLOAT_BOUND,
            )
        )
        self._owner = jnp.asarray(config.proposal_owner_digest, dtype=jnp.uint32)
        # Route both eager and enclosing-JIT calls through the same donor
        # compilation boundary.  This prevents backend fusion from producing
        # different float32 bits (and therefore different donor content tags)
        # for mixed-action vertices across the two execution surfaces.
        self._world_propose = jax.jit(self._world.propose)

    @property
    def config(self) -> HCCLWorldAttributionAdapterConfig:
        return self._config

    @property
    def world(self) -> HCCLCausalCoreWorld:
        return self._world

    @property
    def attribution(self) -> HCCLCausalAttributionKernel:
        return self._attribution

    def to_config(self) -> dict[str, object]:
        return {
            "type": "HCCLWorldAttributionAdapter",
            "schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_CONFIG_SCHEMA,
            "state_schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_STATE_SCHEMA,
            "result_schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_RESULT_SCHEMA,
            "resource_schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_RESOURCE_SCHEMA,
            "checkpoint_schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_CHECKPOINT_SCHEMA,
            "mechanism_status": HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS,
            "evidence_level": HCCL_WORLD_ATTRIBUTION_ADAPTER_EVIDENCE_LEVEL,
            "proposal_owner_digest": list(self._config.proposal_owner_digest),
            "world_config": self._world.to_config(),
            "attribution_config": self._attribution.to_config(),
            "source_projection": "current-two-by-sixteen-learner-observation-row-major",
            "event_projection": (
                "world-flip-two-cue-flips-outcome-flip-ten-nuisance-two-velocity-draws"
            ),
            "transition_projection": "candidate-next-two-by-sixteen-observation-row-major",
            "clock_encoding": "uint64-big-endian-two-uint32-words",
            "proposal_order": list(HCCL_CAUSAL_ATTRIBUTION_PROPOSAL_ORDER),
            "proposal_calls_per_valid_transaction": _N_PROPOSALS,
            "committed_world_successors_per_transaction": 1,
            "committed_world_successor_slot": _PP_SLOT,
            "world_state_owners": 1,
            "attribution_state_owners": 1,
            "composite_link_owners": 1,
            "composite_jit_supported": False,
            "prebound_scan_execution": "host-eager-python-loop",
            "supported_jit_boundaries": [
                "HCCLCausalCoreWorld.propose",
                "HCCLCausalAttributionKernel.stage",
            ],
            "agent_implementation_present": False,
            "agent_component_provenance_bound": False,
            "action_layer_provenance_bound": False,
            "synthetic_world_bound_source_placeholders": True,
            "schedule_execution_authorized": False,
            "artifact_authorized": False,
            "output_writes_authorized": False,
            "threshold_authorized": False,
            "seed_authority": False,
            "evidence_authorized": False,
            "promotion_authorized": False,
            "limitations": list(HCCL_WORLD_ATTRIBUTION_ADAPTER_LIMITATIONS),
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> HCCLWorldAttributionAdapter:
        if type(payload) is not dict:
            raise TypeError("config payload must be an exact dict")
        fields = cast(dict[str, object], payload)
        digest = fields.get("proposal_owner_digest")
        if type(digest) is not list:
            raise TypeError("proposal_owner_digest must serialize as a list")
        world_payload = fields.get("world_config")
        if type(world_payload) is not dict:
            raise TypeError("world_config must serialize as an exact dict")
        world = HCCLCausalCoreWorld.from_config(cast(dict[str, object], world_payload))
        candidate = cls(
            HCCLWorldAttributionAdapterConfig(
                tuple(digest),
                world_config=world.config,
            )
        )
        if _canonical_json_bytes(fields) != _canonical_json_bytes(candidate.to_config()):
            raise ValueError("HCCL world-attribution adapter config is unsupported")
        return candidate

    def init(self, key: Array) -> HCCLWorldAttributionAdapterState:
        world_state = self._world.init(key)
        attribution_state = self._attribution.init()
        return HCCLWorldAttributionAdapterState(
            world_state=world_state,
            attribution_state=attribution_state,
            composite_link_words=_composite_link_words(
                world_state,
                attribution_state,
                self._config.world_config.schedule_profile,
            ),
        )

    def _require_state_contract(self, state: HCCLWorldAttributionAdapterState) -> None:
        if type(state) is not HCCLWorldAttributionAdapterState:
            raise TypeError("state must be exact HCCLWorldAttributionAdapterState")
        self._world._require_state_contract(state.world_state)
        self._attribution._require_state_contract(state.attribution_state)
        _require_array(
            state.composite_link_words,
            shape=(4,),
            dtype=jnp.dtype(jnp.uint32),
            label="state.composite_link_words",
        )

    def state_valid(self, state: HCCLWorldAttributionAdapterState) -> Bool[Array, ""]:
        self._require_state_contract(state)
        return (
            self._world.state_valid(state.world_state)
            & self._attribution.state_valid(state.attribution_state)
            & jnp.all(
                state.world_state.step_words == state.attribution_state.transaction_words
            )
            & jnp.all(
                state.attribution_state.decision_words
                == state.attribution_state.transaction_words
            )
            & jnp.all(
                state.composite_link_words
                == _composite_link_words(
                    state.world_state,
                    state.attribution_state,
                    self._config.world_config.schedule_profile,
                )
            )
        )

    def _source(self, state: HCCLWorldAttributionAdapterState) -> HCCLCausalSourceReceipt:
        world_state = state.world_state
        attribution_state = state.attribution_state
        tag = world_state.content_tag_words
        step = world_state.step_words
        agent_identities = jnp.reshape(self._owner, (_N_AGENTS, 4))
        raw_identities = jnp.stack(
            (
                _salted_identity(tag, (0xA1B2C301, 0x10213243, 0x55667789, 0x90ABCDEF)),
                _salted_identity(tag, (0xA1B2C302, 0x10213244, 0x5566778A, 0x90ABCDF0)),
            )
        )
        slow_identities = jnp.stack(
            (
                _salted_identity(tag, (0x01020305, 0x11121315, 0x21222325, 0x31323335)),
                _salted_identity(tag, (0x41424345, 0x51525355, 0x61626365, 0x71727375)),
            )
        )
        feature_identities = jnp.stack(
            (
                _salted_identity(tag, (0x81828385, 0x91929395, 0xA1A2A3A5, 0xB1B2B3B5)),
                _salted_identity(tag, (0xC1C2C3C5, 0xD1D2D3D5, 0xE1E2E3E5, 0xF1F2F3F5)),
            )
        )
        rng_identities = jnp.stack(
            (
                _salted_identity(tag, (0x13579BDF, 0x2468ACE1, 0x31415927, 0x27182819)),
                _salted_identity(tag, (0x89ABCDEF, 0x76543211, 0x11235813, 0x21345591)),
            )
        )
        generations = jnp.stack((step, step)).astype(jnp.uint32)
        return self._attribution.bind_source(
            attribution_state,
            source_vector=jnp.reshape(self._world.observe(world_state), (_SOURCE_DIM,)),
            source_identity_words=tag,
            agent_identity_words=agent_identities,
            raw_observation_identity_words=raw_identities,
            fast_state_words=generations,
            slow_context_birth_words=slow_identities,
            feature_birth_words=feature_identities,
            memory_generation_words=generations,
            planner_model_words=generations,
            hard_mask_generation_words=generations,
            rng_receipt_identity_words=rng_identities,
        )

    def _exogenous(
        self,
        source: HCCLCausalSourceReceipt,
        event: HCCLCausalCoreEventReceipt,
    ) -> HCCLExogenousReceipt:
        payload = jnp.concatenate(
            (
                jnp.reshape(event.world_flipped.astype(jnp.float32), (1,)),
                event.next_cue_flipped.astype(jnp.float32),
                jnp.reshape(event.outcome_flipped.astype(jnp.float32), (1,)),
                jnp.reshape(event.nuisance_standard_normal, (10,)),
                event.partner_velocity_standard_normal,
            )
        ).astype(jnp.float32)
        return self._attribution.bind_exogenous(
            source,
            exogenous_identity_words=event.content_tag_words,
            exogenous_source_words=_increment_low_clock(event.source_step_words),
            payload=payload,
        )

    def bind_action_receipt(
        self,
        state: HCCLWorldAttributionAdapterState,
        event: HCCLCausalCoreEventReceipt,
        *,
        layer: HCCLActionLayer,
        actions_before_mask: Array,
        actions_after_mask: Array,
        hard_action_masks: Array,
        action_receipt_identity_words: Array,
    ) -> HCCLActionReceipt:
        """Bind one exact action receipt to the composite source and prepared event."""

        self._require_state_contract(state)
        self._world._require_event_contract(event)
        source = self._source(state)
        exogenous = self._exogenous(source, event)
        return self._attribution.bind_action_receipt(
            source,
            exogenous,
            layer=layer,
            actions_before_mask=actions_before_mask,
            actions_after_mask=actions_after_mask,
            hard_action_masks=hard_action_masks,
            action_receipt_identity_words=action_receipt_identity_words,
        )

    def _source_clock_bound(
        self,
        state: HCCLWorldAttributionAdapterState,
        event: HCCLCausalCoreEventReceipt,
        receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    ) -> Bool[Array, ""]:
        result = (
            jnp.all(event.source_state_tag_words == state.world_state.content_tag_words)
            & jnp.all(event.source_step_words == state.world_state.step_words)
            & jnp.all(
                state.world_state.step_words == state.attribution_state.transaction_words
            )
        )
        for receipt in receipts:
            result = (
                result
                & jnp.all(
                    receipt.source.source_identity_words == state.world_state.content_tag_words
                )
                & jnp.all(
                    receipt.source.source_transition_words == state.world_state.step_words
                )
                & jnp.all(receipt.source.decision_words == state.attribution_state.decision_words)
            )
        return result

    def _event_identity_bound(
        self,
        state: HCCLWorldAttributionAdapterState,
        event: HCCLCausalCoreEventReceipt,
        receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    ) -> Bool[Array, ""]:
        result = self._world.event_receipt_valid(state.world_state, event)
        for receipt in receipts:
            result = (
                result
                & jnp.all(receipt.exogenous_identity_words == event.content_tag_words)
                & jnp.all(
                    receipt.exogenous_content_tag_words
                    == self._exogenous(receipt.source, event).content_tag_words
                )
            )
        return result

    def _action_identities_bound(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        receipts: tuple[HCCLActionReceipt, HCCLActionReceipt, HCCLActionReceipt],
    ) -> Bool[Array, ""]:
        base, memory, planner = receipts
        return (
            self._attribution._action_valid(source, exogenous, base, HCCLActionLayer.BASE)
            & self._attribution._action_valid(
                source, exogenous, memory, HCCLActionLayer.MEMORY
            )
            & self._attribution._action_valid(
                source, exogenous, planner, HCCLActionLayer.PLANNER
            )
            & self._attribution._receipt_identities_distinct(receipts)
            & jnp.array_equal(base.hard_action_masks, memory.hard_action_masks)
            & jnp.array_equal(base.hard_action_masks, planner.hard_action_masks)
        )

    def _attribution_proposal(
        self,
        source: HCCLCausalSourceReceipt,
        exogenous: HCCLExogenousReceipt,
        vertex: HCCLJointActionVertex,
        world_proposals: HCCLCausalCoreProposal,
        slot: Array,
    ) -> Any:
        world_proposal = cast(HCCLCausalCoreProposal, _tree_at(world_proposals, slot))
        signals = world_proposal.signals
        typed = HCCLTypedSignals(
            task_score=signals.task_score,
            net_reward=signals.net_reward,
            safety_cost=signals.safety_cost,
            message_charge=signals.message_charge,
        )
        accepted = world_proposal.valid & jnp.all(
            world_proposal.joint_action_ids == vertex.actions
        )
        return self._attribution.bind_proposal(
            source,
            exogenous,
            vertex,
            candidate_transition=jnp.reshape(
                world_proposal.next_observation, (_TRANSITION_DIM,)
            ),
            signals=typed,
            accepted=accepted,
        )

    def _equal_action_payloads_valid(
        self,
        vertices: tuple[HCCLJointActionVertex, ...],
        proposals: tuple[HCCLCausalCoreProposal, ...],
    ) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for index, vertex in enumerate(vertices):
            for previous in range(index):
                same_actions = jnp.all(vertex.actions == vertices[previous].actions)
                same_payload = _tree_exact_equal(proposals[index], proposals[previous])
                valid = valid & (~same_actions | same_payload)
        return valid

    def _causal_core_signals_valid(
        self,
        proposals: tuple[HCCLCausalCoreProposal, ...],
    ) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for proposal in proposals:
            signals = proposal.signals
            expected_net = jnp.broadcast_to(signals.task_score, (_N_AGENTS,))
            valid = (
                valid
                & jnp.all(_array_words(signals.message_charge) == 0)
                & jnp.all(_array_words(signals.safety_cost) == 0)
                & jnp.all(_array_words(signals.net_reward) == _array_words(expected_net))
            )
        return valid

    def stage(
        self,
        state: HCCLWorldAttributionAdapterState,
        event: HCCLCausalCoreEventReceipt,
        base: HCCLActionReceipt,
        memory: HCCLActionReceipt,
        planner: HCCLActionReceipt,
        *,
        downstream_candidate_valid: Array,
    ) -> HCCLWorldAttributionAdapterResult:
        """Evaluate one fixed eight-proposal transaction and atomically select PP."""

        self._require_state_contract(state)
        self._world._require_event_contract(event)
        for receipt in (base, memory, planner):
            self._attribution._require_action_contract(receipt)
        downstream = _require_array(
            downstream_candidate_valid,
            shape=(),
            dtype=jnp.dtype(jnp.bool_),
            label="downstream_candidate_valid",
        )
        if _contains_tracer((state, event, base, memory, planner, downstream)):
            raise TypeError(
                "HCCL world-attribution composite stage is host/eager only; "
                "JIT the smaller world-proposal or attribution-kernel donor boundary"
            )
        source = self._source(state)
        exogenous = self._exogenous(source, event)
        receipts = (base, memory, planner)
        source_state_valid = self.state_valid(state)
        source_clock_bound = self._source_clock_bound(state, event, receipts)
        event_valid = self._world.event_receipt_valid(state.world_state, event)
        event_identity_bound = self._event_identity_bound(state, event, receipts)
        action_identities_bound = self._action_identities_bound(source, exogenous, receipts)

        vertices = self._attribution._vertices(base, memory, planner)
        proposal_items = tuple(
            self._world_propose(state.world_state, event, vertex.actions) for vertex in vertices
        )
        world_proposals = cast(HCCLCausalCoreProposal, _stack_tree(proposal_items))
        all_world_valid = jnp.all(
            jnp.stack(tuple(proposal.valid for proposal in proposal_items))
        )
        equal_action_payloads = self._equal_action_payloads_valid(
            vertices,
            proposal_items,
        )
        causal_core_signals = self._causal_core_signals_valid(proposal_items)
        duplicate_mm = _tree_exact_equal(proposal_items[0], proposal_items[7])
        composite_downstream = (
            downstream
            & source_state_valid
            & source_clock_bound
            & event_valid
            & event_identity_bound
            & action_identities_bound
            & all_world_valid
            & equal_action_payloads
            & causal_core_signals
            & duplicate_mm
        )

        def callback(
            source_row: HCCLCausalSourceReceipt,
            exogenous_row: HCCLExogenousReceipt,
            vertex: HCCLJointActionVertex,
            slot: Array,
        ) -> Any:
            return self._attribution_proposal(
                source_row,
                exogenous_row,
                vertex,
                world_proposals,
                slot,
            )

        attribution_result = self._attribution.stage(
            state.attribution_state,
            source,
            exogenous,
            base,
            memory,
            planner,
            callback,
            downstream_candidate_valid=composite_downstream,
        )
        pp = proposal_items[_PP_SLOT]
        candidate_world = pp.candidate_state
        candidate_attribution = attribution_result.state
        candidate = HCCLWorldAttributionAdapterState(
            world_state=candidate_world,
            attribution_state=candidate_attribution,
            composite_link_words=_composite_link_words(
                candidate_world,
                candidate_attribution,
                self._config.world_config.schedule_profile,
            ),
        )
        candidate_valid = self.state_valid(candidate)
        applied = attribution_result.update_applied & composite_downstream & candidate_valid
        final_state = cast(
            HCCLWorldAttributionAdapterState,
            _tree_select(applied, candidate, state),
        )
        return HCCLWorldAttributionAdapterResult(
            state=final_state,
            world_proposals=world_proposals,
            attribution=attribution_result,
            work=HCCLWorldAttributionAdapterWork(
                world_proposal_calls=jnp.asarray(_N_PROPOSALS, dtype=jnp.int32),
                attribution_proposal_calls=attribution_result.work.proposal_calls,
                designated_counterfactual_world_slots=jnp.asarray(
                    _N_PROPOSALS - 1, dtype=jnp.int32
                ),
                discarded_world_proposal_calls=(
                    jnp.asarray(_N_PROPOSALS, dtype=jnp.int32)
                    - applied.astype(jnp.int32)
                ),
                committed_pp_world_successors=applied.astype(jnp.int32),
                duplicate_mm_world_equality_checks=jnp.asarray(1, dtype=jnp.int32),
            ),
            pre_transaction_words=state.world_state.step_words,
            post_transaction_words=final_state.world_state.step_words,
            source_state_valid=source_state_valid,
            world_source_clock_bound=source_clock_bound,
            event_receipt_valid=event_valid,
            event_receipt_identity_bound=event_identity_bound,
            action_receipt_identities_bound=action_identities_bound,
            all_world_proposals_valid=all_world_valid,
            equal_action_world_payloads_bit_exact=equal_action_payloads,
            causal_core_signal_contract_valid=causal_core_signals,
            world_duplicate_mm_bit_exact=duplicate_mm,
            downstream_candidate_valid=downstream,
            candidate_state_valid=candidate_valid,
            update_applied=applied,
        )

    def resource_budget(
        self,
        state: HCCLWorldAttributionAdapterState | None = None,
    ) -> HCCLWorldAttributionResourceBudget:
        reference = self.init(jr.key(0)) if state is None else state
        self._require_state_contract(reference)
        world_budget = self._world.resource_budget(reference.world_state)
        attribution_budget = self._attribution.resource_budget(reference.attribution_state)
        world_bytes = measure_hccl_causal_core_state_nbytes(reference.world_state)
        attribution_bytes = measure_hccl_causal_attribution_state_nbytes(
            reference.attribution_state
        )
        return HCCLWorldAttributionResourceBudget(
            schema=HCCL_WORLD_ATTRIBUTION_ADAPTER_RESOURCE_SCHEMA,
            world_state_owners=1,
            attribution_state_owners=1,
            world_state_nbytes=world_bytes,
            attribution_state_nbytes=attribution_bytes,
            composite_link_nbytes=16,
            total_persistent_state_nbytes=world_bytes + attribution_bytes + 16,
            event_receipt_nbytes=world_budget.event_receipt_nbytes,
            world_proposal_nbytes=world_budget.proposal_nbytes,
            max_transient_world_proposal_stack_nbytes=(
                _N_PROPOSALS * world_budget.proposal_nbytes
            ),
            max_world_proposal_calls_per_transaction=_N_PROPOSALS,
            max_attribution_proposal_calls_per_transaction=(
                attribution_budget.max_proposal_calls_per_transaction
            ),
            designated_counterfactual_world_slots_per_transaction=_N_PROPOSALS - 1,
            max_discarded_world_proposal_calls_per_transaction=_N_PROPOSALS,
            max_committed_world_successors_per_transaction=1,
            maximum_committed_transactions=(
                self._config.world_config.maximum_committed_transitions
            ),
            output_write_calls=0,
            artifact_bytes_written=0,
            persistent_bytes_scope=(
                "one-world-state-plus-one-attribution-state-plus-composite-link"
            ),
            transient_bytes_scope=(
                "prepared-event-three-action-receipts-eight-world-proposals-and-"
                "attribution-staging; compiler-and-XLA-workspaces-excluded"
            ),
        )


def measure_hccl_world_attribution_state_nbytes(
    state: HCCLWorldAttributionAdapterState,
) -> int:
    if type(state) is not HCCLWorldAttributionAdapterState:
        raise TypeError("state must be exact HCCLWorldAttributionAdapterState")
    return (
        measure_hccl_causal_core_state_nbytes(state.world_state)
        + measure_hccl_causal_attribution_state_nbytes(state.attribution_state)
        + int(state.composite_link_words.size) * int(state.composite_link_words.dtype.itemsize)
    )


def _leading_steps(tree: Any, *, label: str) -> int:
    leaves = jax.tree.leaves(tree)
    if not leaves or getattr(leaves[0], "ndim", 0) < 1:
        raise ValueError(f"{label} must have a leading step dimension")
    steps = int(leaves[0].shape[0])
    if steps < 1:
        raise ValueError(f"{label} must contain at least one row")
    for leaf in leaves:
        if getattr(leaf, "ndim", 0) < 1 or leaf.shape[0] != steps:
            raise ValueError(f"{label} leaves must share one leading step dimension")
    return steps


def run_hccl_world_attribution_scan(
    adapter: HCCLWorldAttributionAdapter,
    state: HCCLWorldAttributionAdapterState,
    events: HCCLCausalCoreEventReceipt,
    base: HCCLActionReceipt,
    memory: HCCLActionReceipt,
    planner: HCCLActionReceipt,
    downstream_candidate_valid: Array,
) -> HCCLWorldAttributionScanResult:
    """Host-scan prepared rows; create no draws, agents, or composite JIT graph."""

    if type(adapter) is not HCCLWorldAttributionAdapter:
        raise TypeError("adapter must be exact HCCLWorldAttributionAdapter")
    adapter._require_state_contract(state)
    steps = _leading_steps(events, label="events")
    for label, value in (("base", base), ("memory", memory), ("planner", planner)):
        if _leading_steps(value, label=label) != steps:
            raise ValueError(f"{label} step count differs")
    _require_array(
        downstream_candidate_valid,
        shape=(steps,),
        dtype=jnp.dtype(jnp.bool_),
        label="downstream_candidate_valid",
    )
    adapter.world._require_event_contract(
        cast(HCCLCausalCoreEventReceipt, _tree_at(events, 0))
    )
    for receipt in (base, memory, planner):
        adapter.attribution._require_action_contract(
            cast(HCCLActionReceipt, _tree_at(receipt, 0))
        )

    carry = state
    memory_totals: list[Array] = []
    planner_totals: list[Array] = []
    pp_minus_bbs: list[Array] = []
    words: list[Array] = []
    world_calls: list[Array] = []
    attribution_calls: list[Array] = []
    applied: list[Array] = []
    for index in range(steps):
        event = cast(HCCLCausalCoreEventReceipt, _tree_at(events, index))
        base_row = cast(HCCLActionReceipt, _tree_at(base, index))
        memory_row = cast(HCCLActionReceipt, _tree_at(memory, index))
        planner_row = cast(HCCLActionReceipt, _tree_at(planner, index))
        result = adapter.stage(
            carry,
            event,
            base_row,
            memory_row,
            planner_row,
            downstream_candidate_valid=downstream_candidate_valid[index],
        )
        carry = result.state
        memory_totals.append(result.attribution.contrasts.memory_total.task_score)
        planner_totals.append(result.attribution.contrasts.planner_total.task_score)
        pp_minus_bbs.append(result.attribution.contrasts.pp_minus_bb.task_score)
        words.append(result.post_transaction_words)
        world_calls.append(result.work.world_proposal_calls)
        attribution_calls.append(result.work.attribution_proposal_calls)
        applied.append(result.update_applied)
    return HCCLWorldAttributionScanResult(
        state=carry,
        memory_total_task_score=jnp.stack(tuple(memory_totals)),
        planner_total_task_score=jnp.stack(tuple(planner_totals)),
        pp_minus_bb_task_score=jnp.stack(tuple(pp_minus_bbs)),
        post_transaction_words=jnp.stack(tuple(words)),
        world_proposal_calls=jnp.stack(tuple(world_calls)),
        attribution_proposal_calls=jnp.stack(tuple(attribution_calls)),
        update_applied=jnp.stack(tuple(applied)),
    )


def _checkpoint_digest(checkpoint: HCCLWorldAttributionCheckpoint) -> str:
    return _canonical_digest(
        {
            "schema": checkpoint.schema,
            "mechanism_status": checkpoint.mechanism_status,
            "evidence_level": checkpoint.evidence_level,
            "output_writes_authorized": checkpoint.output_writes_authorized,
            "artifact_authorized": checkpoint.artifact_authorized,
            "evidence_authorized": checkpoint.evidence_authorized,
            "config": checkpoint.config,
            "config_sha256": checkpoint.config_sha256,
            "resource_budget": checkpoint.resource_budget,
            "state_nbytes": checkpoint.state_nbytes,
            "state_sha256": checkpoint.state_sha256,
        }
    )


def save_hccl_world_attribution_checkpoint(
    adapter: HCCLWorldAttributionAdapter,
    state: HCCLWorldAttributionAdapterState,
) -> HCCLWorldAttributionCheckpoint:
    """Return a strict in-memory checkpoint and perform zero output writes."""

    if type(adapter) is not HCCLWorldAttributionAdapter:
        raise TypeError("adapter must be exact HCCLWorldAttributionAdapter")
    adapter._require_state_contract(state)
    if not bool(adapter.state_valid(state)):
        raise ValueError("cannot checkpoint invalid HCCL world-attribution state")
    copied = cast(HCCLWorldAttributionAdapterState, jax.tree.map(jnp.array, state))
    config = adapter.to_config()
    budget = adapter.resource_budget(copied).to_config()
    bare = HCCLWorldAttributionCheckpoint(
        schema=HCCL_WORLD_ATTRIBUTION_ADAPTER_CHECKPOINT_SCHEMA,
        mechanism_status=HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS,
        evidence_level=HCCL_WORLD_ATTRIBUTION_ADAPTER_EVIDENCE_LEVEL,
        output_writes_authorized=False,
        artifact_authorized=False,
        evidence_authorized=False,
        config=config,
        config_sha256=_canonical_digest(config),
        resource_budget=budget,
        state=copied,
        state_nbytes=measure_hccl_world_attribution_state_nbytes(copied),
        state_sha256=_canonical_digest(_state_host_payload(copied)),
        checkpoint_sha256="",
    )
    return dataclasses.replace(bare, checkpoint_sha256=_checkpoint_digest(bare))


def load_hccl_world_attribution_checkpoint(
    checkpoint: HCCLWorldAttributionCheckpoint,
) -> tuple[HCCLWorldAttributionAdapter, HCCLWorldAttributionAdapterState]:
    """Restore only a canonical in-memory composite transaction state."""

    if type(checkpoint) is not HCCLWorldAttributionCheckpoint:
        raise TypeError("checkpoint must be exact HCCLWorldAttributionCheckpoint")
    fixed = {
        "schema": HCCL_WORLD_ATTRIBUTION_ADAPTER_CHECKPOINT_SCHEMA,
        "mechanism_status": HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS,
        "evidence_level": HCCL_WORLD_ATTRIBUTION_ADAPTER_EVIDENCE_LEVEL,
        "output_writes_authorized": False,
        "artifact_authorized": False,
        "evidence_authorized": False,
    }
    for name, expected in fixed.items():
        actual = getattr(checkpoint, name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"checkpoint {name} differs")
    if type(checkpoint.config) is not dict:
        raise TypeError("checkpoint config must be an exact dict")
    if type(checkpoint.resource_budget) is not dict:
        raise TypeError("checkpoint resource budget must be an exact dict")
    for name in ("config_sha256", "state_sha256", "checkpoint_sha256"):
        if type(getattr(checkpoint, name)) is not str:
            raise TypeError(f"checkpoint {name} must be an exact str")
    adapter = HCCLWorldAttributionAdapter.from_config(checkpoint.config)
    if checkpoint.config_sha256 != _canonical_digest(checkpoint.config):
        raise ValueError("checkpoint config digest differs")
    adapter._require_state_contract(checkpoint.state)
    if type(checkpoint.state_nbytes) is not int:
        raise TypeError("checkpoint state_nbytes must be an exact int")
    if checkpoint.state_nbytes != measure_hccl_world_attribution_state_nbytes(checkpoint.state):
        raise ValueError("checkpoint state bytes differ")
    if checkpoint.state_sha256 != _canonical_digest(_state_host_payload(checkpoint.state)):
        raise ValueError("checkpoint state digest differs")
    expected_budget = adapter.resource_budget(checkpoint.state).to_config()
    if _canonical_json_bytes(checkpoint.resource_budget) != _canonical_json_bytes(
        expected_budget
    ):
        raise ValueError("checkpoint resource budget differs")
    if checkpoint.checkpoint_sha256 != _checkpoint_digest(checkpoint):
        raise ValueError("checkpoint digest differs")
    if not bool(adapter.state_valid(checkpoint.state)):
        raise ValueError("checkpoint state is invalid")
    restored = cast(
        HCCLWorldAttributionAdapterState,
        jax.tree.map(jnp.array, checkpoint.state),
    )
    return adapter, restored


__all__ = [
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_CHECKPOINT_SCHEMA",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_CONFIG_SCHEMA",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_EVIDENCE_LEVEL",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_LIMITATIONS",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_RESOURCE_SCHEMA",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_RESULT_SCHEMA",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_STATE_SCHEMA",
    "HCCL_WORLD_ATTRIBUTION_ADAPTER_STATUS",
    "HCCLWorldAttributionAdapter",
    "HCCLWorldAttributionAdapterConfig",
    "HCCLWorldAttributionAdapterResult",
    "HCCLWorldAttributionAdapterState",
    "HCCLWorldAttributionAdapterWork",
    "HCCLWorldAttributionCheckpoint",
    "HCCLWorldAttributionResourceBudget",
    "HCCLWorldAttributionScanResult",
    "load_hccl_world_attribution_checkpoint",
    "measure_hccl_world_attribution_state_nbytes",
    "run_hccl_world_attribution_scan",
    "save_hccl_world_attribution_checkpoint",
]
