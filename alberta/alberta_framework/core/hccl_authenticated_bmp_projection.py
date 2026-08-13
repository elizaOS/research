# mypy: disable-error-code="attr-defined,call-arg"
"""Generic authenticated-cache projection for one HCCL B/M/P decision.

This host/eager L0 seam installs two already-computed action proposals into a
live :class:`PrototypeAgent` cache.  The first projection is the learned-memory
layer ``M`` over the coordinator's freshly learned base action ``B``.  The
second is the planner layer ``P`` over that exact selected ``M`` Prototype.
Both projections use only the public
``PrototypeAgent.replace_cached_primitive_action`` operation; neither invokes a
learner, model, environment, replay path, or RNG.

The external-owner words are opaque integrity material.  This seam binds them
to its receipts but cannot prove that they came from a memory or planner.  A
composing outer transaction must validate the concrete typed memory/planner
results before supplying their digests.  The SHA-256 bindings are unkeyed and
do not authenticate callers, authorize dispatch, or grant evidence or
promotion authority.
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
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.external_learned_state_router_audit_coordinator import (
    ExternalLearnedStateRouterAuditCoordinator,
    ExternalLearnedStateRouterAuditCoordinatorState,
)
from alberta_framework.core.prototype_agent import (
    PrototypeAgent,
    PrototypeAgentState,
    PrototypeCachedPrimitiveActionReplacement,
)

HCCL_AUTHENTICATED_BMP_PROJECTION_CONFIG_SCHEMA = (
    "alberta.hccl-authenticated-bmp-projection.config.v1"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_MEMORY_SCHEMA = (
    "alberta.hccl-authenticated-bmp-projection.memory.v1"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_BINDING_SCHEMA = (
    "alberta.hccl-authenticated-bmp-projection.binding.v1"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_PREPARED_SCHEMA = (
    "alberta.hccl-authenticated-bmp-projection.prepared.v1"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_RECEIPT_SCHEMA = (
    "alberta.hccl-authenticated-bmp-projection.receipt.v1"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_STATUS = (
    "l0-development-generic-bmp-cache-projection"
)
HCCL_AUTHENTICATED_BMP_PROJECTION_SCIENTIFIC_PROMOTION_ALLOWED = False

_N_ACTIONS = 2
_DIGEST_WORDS = 8
_TOKEN_NBYTES = 32
_UINT32_MAX = 2**32 - 1


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _contains_tracer(value: object) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _require_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise TypeError(f"{name} must expose exact array metadata")
    array = cast(Array, value)
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(array.shape)}")
    if jnp.dtype(array.dtype) != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}; got {array.dtype}")
    return array


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _array_exact_equal(left: object, right: object) -> bool:
    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
        if not jax.dtypes.issubdtype(right_array.dtype, jax.dtypes.prng_key):
            return False
        left_array = jr.key_data(left_array)
        right_array = jr.key_data(right_array)
    left_host = np.ascontiguousarray(np.asarray(jax.device_get(left_array)))
    right_host = np.ascontiguousarray(np.asarray(jax.device_get(right_array)))
    return (
        left_host.dtype == right_host.dtype
        and left_host.shape == right_host.shape
        and left_host.tobytes(order="C") == right_host.tobytes(order="C")
    )


def _tree_exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree):
        return False
    if len(left_leaves) != len(right_leaves):
        return False
    return all(
        _array_exact_equal(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _digest_tree(schema: str, *values: object) -> UInt[Array, " 32"]:
    if _contains_tracer(values):
        raise TypeError("B/M/P projection integrity is host/eager-only")
    digest = hashlib.sha256(schema.encode("ascii"))
    for value in values:
        digest.update(type(value).__module__.encode("utf-8"))
        digest.update(type(value).__qualname__.encode("utf-8"))
        leaves, structure = jax.tree.flatten(value)
        digest.update(repr(structure).encode("utf-8"))
        digest.update(len(leaves).to_bytes(8, "big"))
        for leaf in leaves:
            array = jnp.asarray(leaf)
            if jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key):
                array = jr.key_data(array)
            host = np.ascontiguousarray(np.asarray(jax.device_get(array)))
            digest.update(str(host.dtype).encode("ascii"))
            digest.update(np.asarray(host.shape, dtype=np.int64).tobytes())
            digest.update(host.tobytes(order="C"))
    return jnp.asarray(tuple(digest.digest()), dtype=jnp.uint8)


def _owner_words_valid(words: Array) -> bool:
    return bool(np.any(np.asarray(jax.device_get(words), dtype=np.uint32)))


@dataclasses.dataclass(frozen=True, slots=True)
class HCCLAuthenticatedBMPProjectionConfig:
    """One opaque owner identity for a generic two-action projection seam."""

    owner_digest: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.owner_digest) is not tuple or len(self.owner_digest) != _DIGEST_WORDS:
            raise ValueError("owner_digest must be an exact eight-word tuple")
        for index, word in enumerate(self.owner_digest):
            if type(word) is not int or not 0 <= word <= _UINT32_MAX:
                raise ValueError(f"owner_digest[{index}] must be an exact uint32")
        if not any(self.owner_digest):
            raise ValueError("owner_digest must be nonzero")

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema": HCCL_AUTHENTICATED_BMP_PROJECTION_CONFIG_SCHEMA,
            "mechanism_status": HCCL_AUTHENTICATED_BMP_PROJECTION_STATUS,
            "n_actions": _N_ACTIONS,
            "owner_digest": list(self.owner_digest),
            "external_owner_words_opaque": True,
            "external_owner_words_caller_authenticated": False,
            "model_updates_per_projection": 0,
            "rng_draws_per_projection": 0,
            "dispatch_authority": False,
            "artifact_authority": False,
            "evidence_authority": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> HCCLAuthenticatedBMPProjectionConfig:
        if type(payload) is not dict:
            raise TypeError("B/M/P projection config must be an exact dict")
        owner = payload.get("owner_digest")
        if type(owner) is not list or not all(type(word) is int for word in owner):
            raise ValueError("owner_digest must serialize as an exact integer list")
        candidate = cls(owner_digest=tuple(cast(list[int], owner)))
        if _canonical_json_bytes(candidate.to_config()) != _canonical_json_bytes(payload):
            raise ValueError("B/M/P projection config is noncanonical")
        return candidate


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPMemoryProjection:
    """Exact B-to-M public Prototype replacement attempt."""

    source_coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState
    source_prototype_token: UInt[Array, " 32"]
    memory_prototype_token: UInt[Array, " 32"]
    external_owner_words: UInt[Array, " 8"]
    hard_action_mask: Bool[Array, " 2"]
    base_action: Int[Array, ""]
    proposed_action: Int[Array, ""]
    memory_action: Int[Array, ""]
    consumed: Bool[Array, ""]
    replacement: PrototypeCachedPrimitiveActionReplacement
    source_state_valid: Bool[Array, ""]
    external_owner_words_valid: Bool[Array, ""]
    replacement_candidate_committed: Bool[Array, ""]
    replacement_selected: Bool[Array, ""]
    selected_prototype_valid: Bool[Array, ""]
    phase_valid: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPActionBinding:
    """Persistent B/M/P identity for one selected coordinator cache."""

    config_token: UInt[Array, " 32"]
    decision_id: UInt[Array, " 4"]
    hard_action_mask: Bool[Array, " 2"]
    base_action: Int[Array, ""]
    memory_action_before_mask: Int[Array, ""]
    memory_action: Int[Array, ""]
    planner_action_before_mask: Int[Array, ""]
    final_action: Int[Array, ""]
    memory_consumed: Bool[Array, ""]
    planner_consumed: Bool[Array, ""]
    memory_external_owner_words: UInt[Array, " 8"]
    planner_external_owner_words: UInt[Array, " 8"]
    source_prototype_token: UInt[Array, " 32"]
    memory_prototype_token: UInt[Array, " 32"]
    final_prototype_token: UInt[Array, " 32"]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPProjectionWork:
    """Exact logical public-call schedule for one successful preparation."""

    source_coordinator_validations: Int[Array, ""]
    prototype_memory_replacement_calls: Int[Array, ""]
    prototype_planner_replacement_calls: Int[Array, ""]
    coordinator_candidate_assemblies: Int[Array, ""]
    coordinator_candidate_validations: Int[Array, ""]
    external_memory_owner_bindings: Int[Array, ""]
    external_planner_owner_bindings: Int[Array, ""]
    outer_commit_decisions: Int[Array, ""]
    learner_updates: Int[Array, ""]
    model_updates: Int[Array, ""]
    environment_proposals: Int[Array, ""]
    replay_updates: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPPreparedProjection:
    """Complete source-bound B/M/P candidate awaiting one integrity adoption."""

    memory_projection: HCCLAuthenticatedBMPMemoryProjection
    planner_external_owner_words: UInt[Array, " 8"]
    planner_proposed_action: Int[Array, ""]
    planner_consumed: Bool[Array, ""]
    planner_replacement: PrototypeCachedPrimitiveActionReplacement
    candidate_coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState
    binding: HCCLAuthenticatedBMPActionBinding
    work: HCCLAuthenticatedBMPProjectionWork
    memory_phase_valid: Bool[Array, ""]
    planner_external_owner_words_valid: Bool[Array, ""]
    external_owners_distinct: Bool[Array, ""]
    planner_replacement_candidate_committed: Bool[Array, ""]
    planner_replacement_selected: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    preparation_valid: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPProjectionIntegrityReceipt:
    """Unkeyed exact-content receipt for one prepared projection."""

    config_token: UInt[Array, " 32"]
    source_coordinator_token: UInt[Array, " 32"]
    prepared_content_token: UInt[Array, " 32"]
    integrity_bound: Bool[Array, ""]
    content_token: UInt[Array, " 32"]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPProjectionAdoptionWork:
    """Exact logical checks at the no-donor adoption boundary."""

    source_identity_checks: Int[Array, ""]
    prepared_integrity_checks: Int[Array, ""]
    receipt_integrity_checks: Int[Array, ""]
    candidate_state_validations: Int[Array, ""]
    binding_validations: Int[Array, ""]
    outer_commit_decisions: Int[Array, ""]
    prototype_replacement_calls: Int[Array, ""]
    learner_updates: Int[Array, ""]
    model_updates: Int[Array, ""]
    rng_draws: Int[Array, ""]


@chex.dataclass(frozen=True)
class HCCLAuthenticatedBMPProjectionResult:
    """Selected coordinator state or bit-exact source plus complete audit."""

    state: ExternalLearnedStateRouterAuditCoordinatorState
    prepared: HCCLAuthenticatedBMPPreparedProjection
    receipt: HCCLAuthenticatedBMPProjectionIntegrityReceipt
    adoption_work: HCCLAuthenticatedBMPProjectionAdoptionWork
    source_state_matches: Bool[Array, ""]
    prepared_content_valid: Bool[Array, ""]
    receipt_valid: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    binding_valid: Bool[Array, ""]
    update_applied: Bool[Array, ""]
    complete_source_returned: Bool[Array, ""]


def _projection_work() -> HCCLAuthenticatedBMPProjectionWork:
    zero = jnp.asarray(0, dtype=jnp.int32)
    one = jnp.asarray(1, dtype=jnp.int32)
    return HCCLAuthenticatedBMPProjectionWork(
        source_coordinator_validations=one,
        prototype_memory_replacement_calls=one,
        prototype_planner_replacement_calls=one,
        coordinator_candidate_assemblies=one,
        coordinator_candidate_validations=one,
        external_memory_owner_bindings=one,
        external_planner_owner_bindings=one,
        outer_commit_decisions=one,
        learner_updates=zero,
        model_updates=zero,
        environment_proposals=zero,
        replay_updates=zero,
        rng_draws=zero,
    )


class HCCLAuthenticatedBMPProjection:
    """Two-stage B-to-M-to-P projection over one live coordinator cache."""

    def __init__(
        self,
        coordinator: ExternalLearnedStateRouterAuditCoordinator,
        config: HCCLAuthenticatedBMPProjectionConfig,
    ) -> None:
        if type(coordinator) is not ExternalLearnedStateRouterAuditCoordinator:
            raise TypeError("coordinator must be an exact external coordinator")
        if type(config) is not HCCLAuthenticatedBMPProjectionConfig:
            raise TypeError("config must be an exact B/M/P projection config")
        prototype = coordinator.inner.prototype
        if type(prototype) is not PrototypeAgent:
            raise TypeError("coordinator must expose an exact PrototypeAgent")
        if prototype.config.oak.n_primitive_actions != _N_ACTIONS:
            raise ValueError("B/M/P projection requires exactly two primitive actions")
        self._coordinator = coordinator
        self._prototype = prototype
        self._config = config
        self._owner_words = jnp.asarray(config.owner_digest, dtype=jnp.uint32)
        self._config_token = jnp.asarray(
            tuple(
                hashlib.sha256(
                    _canonical_json_bytes(
                        {
                            "projection": config.to_config(),
                            "coordinator": coordinator.to_config(),
                        }
                    )
                ).digest()
            ),
            dtype=jnp.uint8,
        )

    @property
    def config(self) -> HCCLAuthenticatedBMPProjectionConfig:
        return self._config

    @property
    def coordinator(self) -> ExternalLearnedStateRouterAuditCoordinator:
        return self._coordinator

    @property
    def prototype(self) -> PrototypeAgent:
        return self._prototype

    def to_config(self) -> dict[str, object]:
        return {
            **self._config.to_config(),
            "coordinator": self._coordinator.to_config(),
        }

    @staticmethod
    def _prototype_state(
        coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState,
    ) -> PrototypeAgentState:
        return coordinator_state.inner_state.prototype_state

    def _memory_token(
        self,
        prepared: HCCLAuthenticatedBMPMemoryProjection,
    ) -> Array:
        bare = cast(
            HCCLAuthenticatedBMPMemoryProjection,
            prepared.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_tree(
            HCCL_AUTHENTICATED_BMP_PROJECTION_MEMORY_SCHEMA,
            self._config_token,
            self._owner_words,
            bare,
        )

    def _binding_token(self, binding: HCCLAuthenticatedBMPActionBinding) -> Array:
        bare = cast(
            HCCLAuthenticatedBMPActionBinding,
            binding.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_tree(
            HCCL_AUTHENTICATED_BMP_PROJECTION_BINDING_SCHEMA,
            self._owner_words,
            bare,
        )

    def _prepared_token(
        self,
        prepared: HCCLAuthenticatedBMPPreparedProjection,
    ) -> Array:
        bare = cast(
            HCCLAuthenticatedBMPPreparedProjection,
            prepared.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_tree(
            HCCL_AUTHENTICATED_BMP_PROJECTION_PREPARED_SCHEMA,
            self._config_token,
            self._owner_words,
            bare,
        )

    def _receipt_token(
        self,
        receipt: HCCLAuthenticatedBMPProjectionIntegrityReceipt,
    ) -> Array:
        bare = cast(
            HCCLAuthenticatedBMPProjectionIntegrityReceipt,
            receipt.replace(content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8)),
        )
        return _digest_tree(
            HCCL_AUTHENTICATED_BMP_PROJECTION_RECEIPT_SCHEMA,
            self._owner_words,
            bare,
        )

    @staticmethod
    def _hard_action_mask(value: object) -> Array:
        mask = _require_array(
            value,
            name="hard_action_mask",
            shape=(_N_ACTIONS,),
            dtype=jnp.bool_,
        )
        if not _host_bool(jnp.any(mask)):
            raise ValueError("hard_action_mask must admit at least one action")
        return mask

    @staticmethod
    def _external_owner_words(value: object, *, name: str) -> Array:
        return _require_array(
            value,
            name=name,
            shape=(_DIGEST_WORDS,),
            dtype=jnp.uint32,
        )

    @staticmethod
    def _action(value: object, *, name: str) -> Array:
        return _require_array(value, name=name, shape=(), dtype=jnp.int32)

    @staticmethod
    def _consumed(value: object, *, name: str) -> Array:
        return _require_array(value, name=name, shape=(), dtype=jnp.bool_)

    def prepare_memory(
        self,
        source: ExternalLearnedStateRouterAuditCoordinatorState,
        *,
        proposed_action: Array,
        hard_action_mask: Array,
        consumed: Array,
        external_owner_words: Array,
    ) -> HCCLAuthenticatedBMPMemoryProjection:
        """Evaluate exactly one public B-to-M cached-action replacement."""

        if type(source) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("source must be an exact coordinator state")
        action = self._action(proposed_action, name="memory.proposed_action")
        mask = self._hard_action_mask(hard_action_mask)
        selected = self._consumed(consumed, name="memory.consumed")
        owner = self._external_owner_words(
            external_owner_words,
            name="memory.external_owner_words",
        )
        if _contains_tracer((source, action, mask, selected, owner)):
            raise TypeError("B/M/P projection preparation is host/eager-only")

        source_valid = _host_bool(self._coordinator.state_valid(source))
        source_prototype = self._prototype_state(source)
        replacement = self._prototype.replace_cached_primitive_action(
            source_prototype,
            decision_id=source_prototype.current_decision_id,
            decision_observation=source_prototype.current_representation,
            proposed_action=action,
            safety_action_mask=mask,
        )
        consumed_value = _host_bool(selected)
        selected_prototype = replacement.state if consumed_value else source_prototype
        base_action = source_prototype.current_action
        memory_action = selected_prototype.current_action
        owner_valid = _owner_words_valid(owner) and not _array_exact_equal(
            owner,
            self._owner_words,
        )
        replacement_committed = _host_bool(replacement.committed)
        selected_valid = _host_bool(self._prototype.validate_state(selected_prototype))
        action_relation = (
            int(np.asarray(jax.device_get(memory_action)))
            in (
                int(np.asarray(jax.device_get(action))),
                int(np.asarray(jax.device_get(base_action))),
            )
            if consumed_value
            else int(np.asarray(jax.device_get(memory_action)))
            == int(np.asarray(jax.device_get(base_action)))
        )
        phase_valid = bool(
            source_valid
            and owner_valid
            and replacement_committed
            and selected_valid
            and action_relation
            and _host_bool(mask[jnp.clip(memory_action, 0, _N_ACTIONS - 1)])
        )
        bare = HCCLAuthenticatedBMPMemoryProjection(
            source_coordinator_state=source,
            source_prototype_token=_digest_tree("hccl-bmp-source-prototype-v1", source_prototype),
            memory_prototype_token=_digest_tree(
                "hccl-bmp-memory-prototype-v1",
                selected_prototype,
            ),
            external_owner_words=owner,
            hard_action_mask=mask,
            base_action=base_action,
            proposed_action=action,
            memory_action=memory_action,
            consumed=selected,
            replacement=replacement,
            source_state_valid=jnp.asarray(source_valid, dtype=jnp.bool_),
            external_owner_words_valid=jnp.asarray(owner_valid, dtype=jnp.bool_),
            replacement_candidate_committed=jnp.asarray(
                replacement_committed,
                dtype=jnp.bool_,
            ),
            replacement_selected=jnp.asarray(
                consumed_value and replacement_committed,
                dtype=jnp.bool_,
            ),
            selected_prototype_valid=jnp.asarray(selected_valid, dtype=jnp.bool_),
            phase_valid=jnp.asarray(phase_valid, dtype=jnp.bool_),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return cast(
            HCCLAuthenticatedBMPMemoryProjection,
            bare.replace(content_token=self._memory_token(bare)),
        )

    def _memory_projection_valid(
        self,
        prepared: HCCLAuthenticatedBMPMemoryProjection,
    ) -> bool:
        if type(prepared) is not HCCLAuthenticatedBMPMemoryProjection:
            return False
        try:
            source = prepared.source_coordinator_state
            source_prototype = self._prototype_state(source)
            replacement = prepared.replacement
            if type(replacement) is not PrototypeCachedPrimitiveActionReplacement:
                return False
            selected = (
                replacement.state if _host_bool(prepared.consumed) else source_prototype
            )
            source_valid = _host_bool(self._coordinator.state_valid(source))
            owner_valid = _owner_words_valid(
                prepared.external_owner_words
            ) and not _array_exact_equal(
                prepared.external_owner_words,
                self._owner_words,
            )
            replacement_committed = _host_bool(replacement.committed)
            selected_valid = _host_bool(self._prototype.validate_state(selected))
            consumed = _host_bool(prepared.consumed)
            base = int(np.asarray(jax.device_get(source_prototype.current_action)))
            proposed = int(np.asarray(jax.device_get(prepared.proposed_action)))
            memory_action = int(np.asarray(jax.device_get(selected.current_action)))
            action_relation = (
                memory_action in (proposed, base)
                if consumed
                else memory_action == base
            )
            phase_valid = bool(
                source_valid
                and owner_valid
                and replacement_committed
                and selected_valid
                and action_relation
                and _host_bool(
                    prepared.hard_action_mask[
                        jnp.clip(selected.current_action, 0, _N_ACTIONS - 1)
                    ]
                )
            )
            return bool(
                _array_exact_equal(
                    prepared.source_prototype_token,
                    _digest_tree("hccl-bmp-source-prototype-v1", source_prototype),
                )
                and _array_exact_equal(
                    prepared.memory_prototype_token,
                    _digest_tree("hccl-bmp-memory-prototype-v1", selected),
                )
                and int(np.asarray(jax.device_get(prepared.base_action))) == base
                and int(np.asarray(jax.device_get(prepared.memory_action)))
                == memory_action
                and _host_bool(prepared.source_state_valid) == source_valid
                and _host_bool(prepared.external_owner_words_valid) == owner_valid
                and _host_bool(prepared.replacement_candidate_committed)
                == replacement_committed
                and _host_bool(prepared.replacement_selected)
                == (consumed and replacement_committed)
                and _host_bool(prepared.selected_prototype_valid) == selected_valid
                and _host_bool(prepared.phase_valid) == phase_valid
                and _array_exact_equal(
                    prepared.content_token,
                    self._memory_token(prepared),
                )
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def _candidate_coordinator(
        self,
        source: ExternalLearnedStateRouterAuditCoordinatorState,
        selected_prototype: PrototypeAgentState,
    ) -> ExternalLearnedStateRouterAuditCoordinatorState:
        inner = source.inner_state.replace(prototype_state=selected_prototype)
        return cast(
            ExternalLearnedStateRouterAuditCoordinatorState,
            source.replace(
                inner_state=inner,
                current_action=selected_prototype.current_action,
                current_decision_id=selected_prototype.current_decision_id,
                cached_prototype_step_words=selected_prototype.step_words,
            ),
        )

    def _make_binding(
        self,
        memory: HCCLAuthenticatedBMPMemoryProjection,
        *,
        planner_action: Array,
        planner_consumed: Array,
        planner_owner_words: Array,
        final_prototype: PrototypeAgentState,
    ) -> HCCLAuthenticatedBMPActionBinding:
        bare = HCCLAuthenticatedBMPActionBinding(
            config_token=self._config_token,
            decision_id=final_prototype.current_decision_id,
            hard_action_mask=memory.hard_action_mask,
            base_action=memory.base_action,
            memory_action_before_mask=memory.proposed_action,
            memory_action=memory.memory_action,
            planner_action_before_mask=planner_action,
            final_action=final_prototype.current_action,
            memory_consumed=memory.consumed,
            planner_consumed=planner_consumed,
            memory_external_owner_words=memory.external_owner_words,
            planner_external_owner_words=planner_owner_words,
            source_prototype_token=memory.source_prototype_token,
            memory_prototype_token=memory.memory_prototype_token,
            final_prototype_token=_digest_tree(
                "hccl-bmp-final-prototype-v1",
                final_prototype,
            ),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return cast(
            HCCLAuthenticatedBMPActionBinding,
            bare.replace(content_token=self._binding_token(bare)),
        )

    def binding_valid(
        self,
        coordinator_state: ExternalLearnedStateRouterAuditCoordinatorState,
        binding: HCCLAuthenticatedBMPActionBinding,
    ) -> Bool[Array, ""]:
        """Validate the persisted final P cache and sealed B/M/P identities."""

        if type(coordinator_state) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("coordinator_state must be an exact coordinator state")
        if type(binding) is not HCCLAuthenticatedBMPActionBinding:
            raise TypeError("binding must be an exact B/M/P action binding")
        if _contains_tracer((coordinator_state, binding)):
            raise TypeError("B/M/P projection validity is host/eager-only")
        try:
            for name, shape, dtype in (
                ("config_token", (_TOKEN_NBYTES,), jnp.uint8),
                ("decision_id", (4,), jnp.uint32),
                ("hard_action_mask", (_N_ACTIONS,), jnp.bool_),
                ("base_action", (), jnp.int32),
                ("memory_action_before_mask", (), jnp.int32),
                ("memory_action", (), jnp.int32),
                ("planner_action_before_mask", (), jnp.int32),
                ("final_action", (), jnp.int32),
                ("memory_consumed", (), jnp.bool_),
                ("planner_consumed", (), jnp.bool_),
                ("memory_external_owner_words", (_DIGEST_WORDS,), jnp.uint32),
                ("planner_external_owner_words", (_DIGEST_WORDS,), jnp.uint32),
                ("source_prototype_token", (_TOKEN_NBYTES,), jnp.uint8),
                ("memory_prototype_token", (_TOKEN_NBYTES,), jnp.uint8),
                ("final_prototype_token", (_TOKEN_NBYTES,), jnp.uint8),
                ("content_token", (_TOKEN_NBYTES,), jnp.uint8),
            ):
                _require_array(
                    getattr(binding, name),
                    name=f"binding.{name}",
                    shape=shape,
                    dtype=dtype,
                )
        except (AttributeError, TypeError, ValueError):
            return jnp.asarray(False, dtype=jnp.bool_)
        actions = (
            binding.base_action,
            binding.memory_action,
            binding.final_action,
        )
        actions_in_range = all(
            0 <= int(np.asarray(jax.device_get(action))) < _N_ACTIONS
            for action in actions
        )
        before_in_range = all(
            0 <= int(np.asarray(jax.device_get(action))) < _N_ACTIONS
            for action in (
                binding.memory_action_before_mask,
                binding.planner_action_before_mask,
            )
        )
        mask = np.asarray(jax.device_get(binding.hard_action_mask), dtype=np.bool_)
        actions_safe = actions_in_range and all(
            bool(mask[int(np.asarray(jax.device_get(action)))]) for action in actions
        )
        memory_consumed = _host_bool(binding.memory_consumed)
        planner_consumed = _host_bool(binding.planner_consumed)
        base = int(np.asarray(jax.device_get(binding.base_action)))
        memory_before = int(
            np.asarray(jax.device_get(binding.memory_action_before_mask))
        )
        memory = int(np.asarray(jax.device_get(binding.memory_action)))
        planner_before = int(
            np.asarray(jax.device_get(binding.planner_action_before_mask))
        )
        final = int(np.asarray(jax.device_get(binding.final_action)))
        memory_relation = (
            memory in (memory_before, base)
            if memory_consumed
            else memory == base
        )
        planner_relation = (
            final in (planner_before, memory)
            if planner_consumed
            else final == memory
        )
        prototype = self._prototype_state(coordinator_state)
        valid = all(
            (
                _host_bool(self._coordinator.state_valid(coordinator_state)),
                _array_exact_equal(binding.config_token, self._config_token),
                _array_exact_equal(binding.decision_id, coordinator_state.current_decision_id),
                int(np.asarray(jax.device_get(coordinator_state.current_action))) == final,
                int(np.asarray(jax.device_get(prototype.current_action))) == final,
                bool(np.any(mask)),
                actions_in_range,
                before_in_range,
                actions_safe,
                memory_relation,
                planner_relation,
                _owner_words_valid(binding.memory_external_owner_words),
                _owner_words_valid(binding.planner_external_owner_words),
                not _array_exact_equal(
                    binding.memory_external_owner_words,
                    binding.planner_external_owner_words,
                ),
                not _array_exact_equal(
                    binding.memory_external_owner_words,
                    self._owner_words,
                ),
                not _array_exact_equal(
                    binding.planner_external_owner_words,
                    self._owner_words,
                ),
                _array_exact_equal(
                    binding.final_prototype_token,
                    _digest_tree("hccl-bmp-final-prototype-v1", prototype),
                ),
                _array_exact_equal(binding.content_token, self._binding_token(binding)),
            )
        )
        return jnp.asarray(valid, dtype=jnp.bool_)

    def prepare_planner(
        self,
        memory: HCCLAuthenticatedBMPMemoryProjection,
        *,
        proposed_action: Array,
        consumed: Array,
        external_owner_words: Array,
    ) -> HCCLAuthenticatedBMPPreparedProjection:
        """Evaluate exactly one public M-to-P replacement and seal a candidate."""

        if type(memory) is not HCCLAuthenticatedBMPMemoryProjection:
            raise TypeError("memory must be an exact B-to-M projection")
        action = self._action(proposed_action, name="planner.proposed_action")
        selected = self._consumed(consumed, name="planner.consumed")
        owner = self._external_owner_words(
            external_owner_words,
            name="planner.external_owner_words",
        )
        if _contains_tracer((memory, action, selected, owner)):
            raise TypeError("B/M/P projection preparation is host/eager-only")

        memory_valid = self._memory_projection_valid(memory)
        source = memory.source_coordinator_state
        source_prototype = self._prototype_state(source)
        memory_prototype = (
            memory.replacement.state
            if _host_bool(memory.consumed)
            else source_prototype
        )
        replacement = self._prototype.replace_cached_primitive_action(
            memory_prototype,
            decision_id=memory_prototype.current_decision_id,
            decision_observation=memory_prototype.current_representation,
            proposed_action=action,
            safety_action_mask=memory.hard_action_mask,
        )
        consumed_value = _host_bool(selected)
        final_prototype = replacement.state if consumed_value else memory_prototype
        owner_valid = _owner_words_valid(owner) and not _array_exact_equal(
            owner,
            self._owner_words,
        )
        owners_distinct = not _array_exact_equal(owner, memory.external_owner_words)
        replacement_committed = _host_bool(replacement.committed)
        action_relation = (
            int(np.asarray(jax.device_get(final_prototype.current_action)))
            in (
                int(np.asarray(jax.device_get(action))),
                int(np.asarray(jax.device_get(memory.memory_action))),
            )
            if consumed_value
            else int(np.asarray(jax.device_get(final_prototype.current_action)))
            == int(np.asarray(jax.device_get(memory.memory_action)))
        )
        candidate_attempt = self._candidate_coordinator(source, final_prototype)
        candidate_valid = _host_bool(self._coordinator.state_valid(candidate_attempt))
        binding = self._make_binding(
            memory,
            planner_action=action,
            planner_consumed=selected,
            planner_owner_words=owner,
            final_prototype=final_prototype,
        )
        binding_valid = _host_bool(self.binding_valid(candidate_attempt, binding))
        preparation_valid = bool(
            memory_valid
            and _host_bool(memory.phase_valid)
            and owner_valid
            and owners_distinct
            and replacement_committed
            and action_relation
            and candidate_valid
            and binding_valid
        )
        selected_candidate = candidate_attempt if preparation_valid else source
        bare = HCCLAuthenticatedBMPPreparedProjection(
            memory_projection=memory,
            planner_external_owner_words=owner,
            planner_proposed_action=action,
            planner_consumed=selected,
            planner_replacement=replacement,
            candidate_coordinator_state=selected_candidate,
            binding=binding,
            work=_projection_work(),
            memory_phase_valid=jnp.asarray(memory_valid, dtype=jnp.bool_),
            planner_external_owner_words_valid=jnp.asarray(
                owner_valid,
                dtype=jnp.bool_,
            ),
            external_owners_distinct=jnp.asarray(owners_distinct, dtype=jnp.bool_),
            planner_replacement_candidate_committed=jnp.asarray(
                replacement_committed,
                dtype=jnp.bool_,
            ),
            planner_replacement_selected=jnp.asarray(
                consumed_value and replacement_committed,
                dtype=jnp.bool_,
            ),
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            binding_valid=jnp.asarray(binding_valid, dtype=jnp.bool_),
            preparation_valid=jnp.asarray(preparation_valid, dtype=jnp.bool_),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return cast(
            HCCLAuthenticatedBMPPreparedProjection,
            bare.replace(content_token=self._prepared_token(bare)),
        )

    def _prepared_valid(
        self,
        source: ExternalLearnedStateRouterAuditCoordinatorState,
        prepared: HCCLAuthenticatedBMPPreparedProjection,
    ) -> bool:
        if type(prepared) is not HCCLAuthenticatedBMPPreparedProjection:
            return False
        try:
            if not _tree_exact_equal(source, prepared.memory_projection.source_coordinator_state):
                return False
            memory = prepared.memory_projection
            memory_valid = self._memory_projection_valid(memory)
            source_prototype = self._prototype_state(source)
            memory_prototype = (
                memory.replacement.state
                if _host_bool(memory.consumed)
                else source_prototype
            )
            replacement = prepared.planner_replacement
            if type(replacement) is not PrototypeCachedPrimitiveActionReplacement:
                return False
            planner_consumed = _host_bool(prepared.planner_consumed)
            final_prototype = (
                replacement.state if planner_consumed else memory_prototype
            )
            owner_valid = _owner_words_valid(
                prepared.planner_external_owner_words
            ) and not _array_exact_equal(
                prepared.planner_external_owner_words,
                self._owner_words,
            )
            owners_distinct = not _array_exact_equal(
                prepared.planner_external_owner_words,
                memory.external_owner_words,
            )
            replacement_committed = _host_bool(replacement.committed)
            planner_action = int(
                np.asarray(jax.device_get(prepared.planner_proposed_action))
            )
            memory_action = int(np.asarray(jax.device_get(memory.memory_action)))
            final_action = int(
                np.asarray(jax.device_get(final_prototype.current_action))
            )
            action_relation = (
                final_action in (planner_action, memory_action)
                if planner_consumed
                else final_action == memory_action
            )
            candidate_attempt = self._candidate_coordinator(source, final_prototype)
            candidate_valid = _host_bool(
                self._coordinator.state_valid(candidate_attempt)
            )
            expected_binding = self._make_binding(
                memory,
                planner_action=prepared.planner_proposed_action,
                planner_consumed=prepared.planner_consumed,
                planner_owner_words=prepared.planner_external_owner_words,
                final_prototype=final_prototype,
            )
            binding_valid = _host_bool(
                self.binding_valid(candidate_attempt, expected_binding)
            )
            preparation_valid = bool(
                memory_valid
                and _host_bool(memory.phase_valid)
                and owner_valid
                and owners_distinct
                and replacement_committed
                and action_relation
                and candidate_valid
                and binding_valid
            )
            expected_candidate = candidate_attempt if preparation_valid else source
            return bool(
                _tree_exact_equal(prepared.binding, expected_binding)
                and _tree_exact_equal(
                    prepared.candidate_coordinator_state,
                    expected_candidate,
                )
                and _tree_exact_equal(prepared.work, _projection_work())
                and _host_bool(prepared.memory_phase_valid) == memory_valid
                and _host_bool(prepared.planner_external_owner_words_valid)
                == owner_valid
                and _host_bool(prepared.external_owners_distinct) == owners_distinct
                and _host_bool(prepared.planner_replacement_candidate_committed)
                == replacement_committed
                and _host_bool(prepared.planner_replacement_selected)
                == (planner_consumed and replacement_committed)
                and _host_bool(prepared.candidate_state_valid) == candidate_valid
                and _host_bool(prepared.binding_valid) == binding_valid
                and _host_bool(prepared.preparation_valid) == preparation_valid
                and _array_exact_equal(
                    prepared.content_token,
                    self._prepared_token(prepared),
                )
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def integrity_receipt(
        self,
        prepared: HCCLAuthenticatedBMPPreparedProjection,
    ) -> HCCLAuthenticatedBMPProjectionIntegrityReceipt:
        """Bind one exact prepared projection without reevaluating a donor."""

        if type(prepared) is not HCCLAuthenticatedBMPPreparedProjection:
            raise TypeError("prepared must be an exact B/M/P projection")
        if _contains_tracer(prepared):
            raise TypeError("B/M/P projection receipts are host/eager-only")
        source = prepared.memory_projection.source_coordinator_state
        integrity = self._prepared_valid(source, prepared)
        bare = HCCLAuthenticatedBMPProjectionIntegrityReceipt(
            config_token=self._config_token,
            source_coordinator_token=_digest_tree("hccl-bmp-source-coordinator-v1", source),
            prepared_content_token=prepared.content_token,
            integrity_bound=jnp.asarray(integrity, dtype=jnp.bool_),
            content_token=jnp.zeros((_TOKEN_NBYTES,), dtype=jnp.uint8),
        )
        return cast(
            HCCLAuthenticatedBMPProjectionIntegrityReceipt,
            bare.replace(content_token=self._receipt_token(bare)),
        )

    def _receipt_valid(
        self,
        source: ExternalLearnedStateRouterAuditCoordinatorState,
        prepared: HCCLAuthenticatedBMPPreparedProjection,
        receipt: HCCLAuthenticatedBMPProjectionIntegrityReceipt,
    ) -> bool:
        if type(receipt) is not HCCLAuthenticatedBMPProjectionIntegrityReceipt:
            return False
        try:
            expected = self.integrity_receipt(prepared)
            return bool(
                _tree_exact_equal(receipt, expected)
                and _tree_exact_equal(source, prepared.memory_projection.source_coordinator_state)
                and _host_bool(receipt.integrity_bound)
            )
        except (AttributeError, IndexError, TypeError, ValueError):
            return False

    def adopt(
        self,
        source: ExternalLearnedStateRouterAuditCoordinatorState,
        prepared: HCCLAuthenticatedBMPPreparedProjection,
        receipt: HCCLAuthenticatedBMPProjectionIntegrityReceipt,
    ) -> HCCLAuthenticatedBMPProjectionResult:
        """Adopt B/M/P together or return the complete coordinator source."""

        if type(source) is not ExternalLearnedStateRouterAuditCoordinatorState:
            raise TypeError("source must be an exact coordinator state")
        if type(prepared) is not HCCLAuthenticatedBMPPreparedProjection:
            raise TypeError("prepared must be an exact B/M/P projection")
        if type(receipt) is not HCCLAuthenticatedBMPProjectionIntegrityReceipt:
            raise TypeError("receipt must be an exact B/M/P integrity receipt")
        if _contains_tracer((source, prepared, receipt)):
            raise TypeError("B/M/P projection adoption is host/eager-only")

        source_matches = _tree_exact_equal(
            source,
            prepared.memory_projection.source_coordinator_state,
        )
        prepared_valid = self._prepared_valid(source, prepared)
        receipt_valid = self._receipt_valid(source, prepared, receipt)
        candidate_valid = _host_bool(
            self._coordinator.state_valid(prepared.candidate_coordinator_state)
        )
        binding_valid = _host_bool(
            self.binding_valid(prepared.candidate_coordinator_state, prepared.binding)
        )
        applied = bool(
            source_matches
            and prepared_valid
            and receipt_valid
            and candidate_valid
            and binding_valid
            and _host_bool(prepared.preparation_valid)
        )
        selected = prepared.candidate_coordinator_state if applied else source
        zero = jnp.asarray(0, dtype=jnp.int32)
        one = jnp.asarray(1, dtype=jnp.int32)
        return HCCLAuthenticatedBMPProjectionResult(
            state=selected,
            prepared=prepared,
            receipt=receipt,
            adoption_work=HCCLAuthenticatedBMPProjectionAdoptionWork(
                source_identity_checks=one,
                prepared_integrity_checks=one,
                receipt_integrity_checks=one,
                candidate_state_validations=one,
                binding_validations=one,
                outer_commit_decisions=one,
                prototype_replacement_calls=zero,
                learner_updates=zero,
                model_updates=zero,
                rng_draws=zero,
            ),
            source_state_matches=jnp.asarray(source_matches, dtype=jnp.bool_),
            prepared_content_valid=jnp.asarray(prepared_valid, dtype=jnp.bool_),
            receipt_valid=jnp.asarray(receipt_valid, dtype=jnp.bool_),
            candidate_state_valid=jnp.asarray(candidate_valid, dtype=jnp.bool_),
            binding_valid=jnp.asarray(binding_valid, dtype=jnp.bool_),
            update_applied=jnp.asarray(applied, dtype=jnp.bool_),
            complete_source_returned=jnp.asarray(not applied, dtype=jnp.bool_),
        )


__all__ = (
    "HCCL_AUTHENTICATED_BMP_PROJECTION_BINDING_SCHEMA",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_CONFIG_SCHEMA",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_MEMORY_SCHEMA",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_PREPARED_SCHEMA",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_RECEIPT_SCHEMA",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_SCIENTIFIC_PROMOTION_ALLOWED",
    "HCCL_AUTHENTICATED_BMP_PROJECTION_STATUS",
    "HCCLAuthenticatedBMPActionBinding",
    "HCCLAuthenticatedBMPMemoryProjection",
    "HCCLAuthenticatedBMPPreparedProjection",
    "HCCLAuthenticatedBMPProjection",
    "HCCLAuthenticatedBMPProjectionAdoptionWork",
    "HCCLAuthenticatedBMPProjectionConfig",
    "HCCLAuthenticatedBMPProjectionIntegrityReceipt",
    "HCCLAuthenticatedBMPProjectionResult",
    "HCCLAuthenticatedBMPProjectionWork",
)
