# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Fixed-memory continual L0 lifecycle for pair features and linear consumers.

This module composes two existing mechanisms without changing either one:

* :class:`FixedBudgetInteractionLearner` proposes a fixed-width bank of
  pair-product features; and
* :class:`FeatureBankRouter` moves downstream linear state by descriptor
  identity when that bank changes.

The lifecycle owns only discovery and routing metadata.  Supplied
:class:`OaKState` and optional exact-linear :class:`MultiHeadMLPState`
consumers remain caller-owned.  At a caller-declared safe boundary, one
descriptor change is routed atomically through the linear base heads,
eligibility traces, intra-option policies, option-start cache, both axes of
every option-model matrix, and every managed Horde head weight and weight
trace.  Unsafe changes are deferred: ordinary causal feature learning is
retained, while the proposed descriptor mutation is rolled back to
``InteractionFeatureUpdateResult.pre_curation_state``.  A deferred mutation
is not queued; the learner may propose again only at a later curation
opportunity under its fixed replacement schedule.

This is development mechanism evidence only.  It does not establish a
benefit, an Alberta Plan completion claim, or eligibility for scientific
promotion.
"""

from __future__ import annotations

import dataclasses
import math
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.feature_bank_router import (
    FeatureBankRouteDiagnostics,
    FeatureBankRouter,
    FeatureBankRouterConfig,
    FeatureBankRouterState,
)
from alberta_framework.core.feature_discovery import GENERATOR_RANDOM
from alberta_framework.core.interaction_features import (
    FixedBudgetInteractionLearner,
    InteractionCurationPriorityOverride,
    InteractionFeatureState,
)
from alberta_framework.core.multi_head_learner import (
    MultiHeadMLPLearner,
    MultiHeadMLPState,
)
from alberta_framework.core.oak import OaKAgent, OaKConfig, OaKState
from alberta_framework.core.options import (
    STOMPConfig,
    SubtaskSpec,
    replace_dispatched_primitive_action,
)
from alberta_framework.core.types import LMSState

_LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v1"
)
_LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v2"
)
_LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA = (
    "alberta.prototype-feature-lifecycle.checkpoint.v1"
)
PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v3"
)
PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v4"
)
_PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v5"
)
_PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_HORDE_CONFIG_SCHEMA = (
    "alberta.prototype-feature-lifecycle.config.v6"
)
PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA = (
    "alberta.prototype-feature-lifecycle.checkpoint.v2"
)
PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA = (
    "alberta.prototype-feature-lifecycle.state.v2"
)
PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS = "development_mechanism_only"
PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED = False

_CONFIG_TYPE = "PrototypeFeatureLifecycleConfig"
_INT32_MAX = 2_147_483_647
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
PROTOTYPE_FEATURE_LIFECYCLE_TELEMETRY_COUNTER_NBYTES = 4
PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES = 12
PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_DELTA_NBYTES = 8
PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES = 48
PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_DELTA_NBYTES = 32
PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_NBYTES = 12
PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_DELTA_NBYTES = 8
_MAX_TOTAL_FEATURE_DIM = 4_096
_MAX_PAIR_SLOTS = 262_144
_MAX_AXIS_PRODUCT_SCALARS = 4_194_304
_MAX_MANAGED_CONSUMER_SCALARS = 8_388_608
_MAX_DESCRIPTOR_COMPARISON_CELLS = 4_194_304
_MAX_ENUMERATED_PAIR_SPACE = 65_536
_MAX_PYTHON_COLLECTION_LENGTH = 4_096


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int = _INT32_MAX,
) -> int:
    """Validate a Python integer without accepting booleans or coercions."""

    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be a strict integer in [{minimum}, {maximum}]"
        )
    return value


def _strict_float(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool = True,
) -> float:
    """Validate an exact Python float and a finite closed/half-open range."""

    if type(value) is not float:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must be a strict float in [{minimum}, {maximum}{bracket}"
        )
    upper_valid = value <= maximum if maximum_inclusive else value < maximum
    if not math.isfinite(value) or value < minimum or not upper_valid:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must be a strict float in [{minimum}, {maximum}{bracket}"
        )
    try:
        float32_value = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as error:
        raise ValueError(f"{name} must remain finite as float32") from error
    if not math.isfinite(float32_value) or (value != 0.0 and float32_value == 0.0):
        raise ValueError(f"{name} must remain finite and nonzero as float32")
    float32_upper_valid = (
        float32_value <= maximum
        if maximum_inclusive
        else float32_value < maximum
    )
    if float32_value < minimum or not float32_upper_valid:
        bracket = "]" if maximum_inclusive else ")"
        raise ValueError(
            f"{name} must remain in [{minimum}, {maximum}{bracket} as float32"
        )
    return value


@dataclasses.dataclass(frozen=True)
class PrototypeFeatureLifecycleConfig:
    """Static capacity, learner, and downstream-layout contract.

    ``option_subtask_feature_indices`` is a caller attestation.  OaKState does
    not persist SubtaskSpecs, so this standalone boundary cannot derive their
    indices from a supplied state.  Every attested index is restricted to the
    stable base prefix; an integrator must call
    :meth:`PrototypeFeatureLifecycle.require_compatible_oak_config` when it
    binds this lifecycle to the actual OaK configuration.
    """

    base_feature_dim: int
    active_pair_slots: int
    candidate_pair_slots: int
    n_tasks: int
    n_options: int
    n_primitive_actions: int
    option_subtask_feature_indices: tuple[int, ...]
    step_size_output: float = 0.03
    utility_decay: float = 0.995
    replacement_interval: int = 100
    min_feature_age: int = 50
    candidate_min_age: int = 25
    promotion_margin: float = 1.05
    scale_normalizer_decay: float = 0.99
    scale_normalizer_epsilon: float = 1.0e-6
    carry_survivors: bool = True
    # This is an optional explicit logical budget, not an int32 implementation
    # ceiling.  The default reaches the complete two-word lifetime and then
    # rejects the next transaction atomically.
    max_observations: int = _UINT64_MAX
    managed_horde_demons: int = 0
    pair_source_feature_dim: int | None = None

    def __post_init__(self) -> None:
        _strict_int(
            self.base_feature_dim,
            name="base_feature_dim",
            minimum=2,
            maximum=_MAX_TOTAL_FEATURE_DIM,
        )
        _strict_int(
            self.active_pair_slots,
            name="active_pair_slots",
            minimum=1,
            maximum=_MAX_PAIR_SLOTS,
        )
        _strict_int(
            self.candidate_pair_slots,
            name="candidate_pair_slots",
            minimum=0,
            maximum=_MAX_PAIR_SLOTS,
        )
        _strict_int(
            self.n_tasks,
            name="n_tasks",
            minimum=1,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        _strict_int(
            self.n_options,
            name="n_options",
            minimum=0,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        _strict_int(
            self.n_primitive_actions,
            name="n_primitive_actions",
            minimum=1,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        _strict_int(
            self.managed_horde_demons,
            name="managed_horde_demons",
            minimum=0,
            maximum=_MAX_PYTHON_COLLECTION_LENGTH,
        )
        if self.pair_source_feature_dim is not None:
            _strict_int(
                self.pair_source_feature_dim,
                name="pair_source_feature_dim",
                minimum=2,
                maximum=self.base_feature_dim - 1,
            )
        if (
            self.managed_horde_demons > 0
            and self.n_tasks != self.managed_horde_demons + 1
        ):
            raise ValueError(
                "managed Horde requires n_tasks == 1 + managed_horde_demons"
            )
        if (
            type(self.option_subtask_feature_indices) is not tuple
            or len(self.option_subtask_feature_indices) != self.n_options
        ):
            raise ValueError(
                "option_subtask_feature_indices must be an exact tuple with one "
                "entry per option"
            )
        for feature_index in self.option_subtask_feature_indices:
            if (
                type(feature_index) is not int
                or not 0 <= feature_index < self.base_feature_dim
            ):
                raise ValueError(
                    "every option subtask must index the stable base prefix"
                )
        _strict_int(
            self.replacement_interval,
            name="replacement_interval",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        if self.candidate_pair_slots == 0 and self.replacement_interval > 0:
            raise ValueError(
                "positive replacement_interval requires candidate_pair_slots > 0"
            )
        _strict_int(
            self.min_feature_age,
            name="min_feature_age",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        _strict_int(
            self.candidate_min_age,
            name="candidate_min_age",
            minimum=0,
            maximum=_INT32_MAX - 1,
        )
        _strict_int(
            self.max_observations,
            name="max_observations",
            minimum=1,
            maximum=_UINT64_MAX,
        )
        _strict_float(
            self.step_size_output,
            name="step_size_output",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.step_size_output == 0.0:
            raise ValueError("step_size_output must be positive")
        _strict_float(
            self.utility_decay,
            name="utility_decay",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _strict_float(
            self.promotion_margin,
            name="promotion_margin",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.promotion_margin == 0.0:
            raise ValueError("promotion_margin must be positive")
        _strict_float(
            self.scale_normalizer_decay,
            name="scale_normalizer_decay",
            minimum=0.0,
            maximum=1.0,
            maximum_inclusive=False,
        )
        _strict_float(
            self.scale_normalizer_epsilon,
            name="scale_normalizer_epsilon",
            minimum=0.0,
            maximum=float("inf"),
        )
        if self.scale_normalizer_epsilon == 0.0:
            raise ValueError("scale_normalizer_epsilon must be positive")
        if type(self.carry_survivors) is not bool:
            raise ValueError("carry_survivors must be a strict boolean")

        pair_source_dim = self.effective_pair_source_feature_dim
        pair_space = pair_source_dim * (pair_source_dim - 1) // 2
        if self.active_pair_slots > pair_space:
            raise ValueError("active_pair_slots exceeds the canonical pair space")
        if self.candidate_pair_slots > pair_space:
            raise ValueError("candidate_pair_slots exceeds the canonical pair space")
        if self.active_pair_slots**2 > _MAX_DESCRIPTOR_COMPARISON_CELLS:
            raise ValueError(
                "active descriptor comparison matrix exceeds the allocation ceiling"
            )
        if self.candidate_pair_slots**2 > _MAX_DESCRIPTOR_COMPARISON_CELLS:
            raise ValueError(
                "candidate descriptor comparison matrix exceeds the allocation ceiling"
            )
        if (
            self.candidate_pair_slots > 0
            and pair_space > _MAX_ENUMERATED_PAIR_SPACE
        ):
            raise ValueError(
                "all-pairs candidate enumeration exceeds the allocation ceiling"
            )
        if self.n_total_actions > _MAX_PYTHON_COLLECTION_LENGTH:
            raise ValueError(
                "linear base-head collection exceeds the allocation ceiling"
            )
        if self.total_feature_dim > _MAX_TOTAL_FEATURE_DIM:
            raise ValueError(
                "total_feature_dim exceeds the lifecycle allocation ceiling"
            )
        discovery_axis_product = self.n_tasks * (
            self.active_pair_slots + self.candidate_pair_slots
        )
        if discovery_axis_product > _MAX_AXIS_PRODUCT_SCALARS:
            raise ValueError(
                "task-by-pair discovery state exceeds the allocation ceiling"
            )
        option_model_scalars = (
            self.n_options * self.total_feature_dim * self.total_feature_dim
        )
        if option_model_scalars > _MAX_AXIS_PRODUCT_SCALARS:
            raise ValueError(
                "option-model feature matrix state exceeds the allocation ceiling"
            )
        input_groups = (
            2 * self.n_total_actions
            + 2 * self.n_options * self.n_primitive_actions
            + self.n_options * self.total_feature_dim
            + 1
            + 2 * self.managed_horde_demons
        )
        if input_groups * self.total_feature_dim > _MAX_MANAGED_CONSUMER_SCALARS:
            raise ValueError(
                "managed linear consumers exceed the allocation ceiling"
            )

    @property
    def total_feature_dim(self) -> int:
        """Width of ``[base prefix | discovered pair tail]``."""

        return self.base_feature_dim + self.active_pair_slots

    @property
    def effective_pair_source_feature_dim(self) -> int:
        """Width of the stable base prefix eligible to become pair parents."""

        if self.pair_source_feature_dim is None:
            return self.base_feature_dim
        return self.pair_source_feature_dim

    @property
    def n_total_actions(self) -> int:
        """Number of primitive plus option heads in the linear base learner."""

        return self.n_primitive_actions + self.n_options

    def to_config(self) -> dict[str, object]:
        """Return the exact JSON-compatible L0 configuration."""

        payload: dict[str, object] = {
            "schema": PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
            "state_schema": PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA,
            "type": _CONFIG_TYPE,
            "mechanism_status": PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "base_feature_dim": self.base_feature_dim,
            "active_pair_slots": self.active_pair_slots,
            "candidate_pair_slots": self.candidate_pair_slots,
            "n_tasks": self.n_tasks,
            "n_options": self.n_options,
            "n_primitive_actions": self.n_primitive_actions,
            "option_subtask_feature_indices": list(
                self.option_subtask_feature_indices
            ),
            "step_size_output": self.step_size_output,
            "utility_decay": self.utility_decay,
            "replacement_interval": self.replacement_interval,
            "min_feature_age": self.min_feature_age,
            "candidate_min_age": self.candidate_min_age,
            "promotion_margin": self.promotion_margin,
            "scale_normalizer_decay": self.scale_normalizer_decay,
            "scale_normalizer_epsilon": self.scale_normalizer_epsilon,
            "carry_survivors": self.carry_survivors,
            "max_observations": self.max_observations,
        }
        if self.managed_horde_demons > 0:
            payload["schema"] = PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA
            payload["managed_horde_demons"] = self.managed_horde_demons
        if self.pair_source_feature_dim is not None:
            payload["schema"] = (
                _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_HORDE_CONFIG_SCHEMA
                if self.managed_horde_demons > 0
                else _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_CONFIG_SCHEMA
            )
            payload["pair_source_feature_dim"] = self.pair_source_feature_dim
        return payload

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeFeatureLifecycleConfig:
        """Strictly reconstruct only the exact versioned mechanism schema."""

        payload = dict(config)
        expected_v1 = {
            "schema",
            "type",
            "mechanism_status",
            "scientific_promotion_allowed",
            "base_feature_dim",
            "active_pair_slots",
            "candidate_pair_slots",
            "n_tasks",
            "n_options",
            "n_primitive_actions",
            "option_subtask_feature_indices",
            "step_size_output",
            "utility_decay",
            "replacement_interval",
            "min_feature_age",
            "candidate_min_age",
            "promotion_margin",
            "scale_normalizer_decay",
            "scale_normalizer_epsilon",
            "carry_survivors",
            "max_observations",
        }
        schema = payload.get("schema")
        current_schema = schema in {
            PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
            PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA,
            _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_CONFIG_SCHEMA,
            _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_HORDE_CONFIG_SCHEMA,
        }
        if schema == _LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA:
            if set(payload) != expected_v1:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "non-Horde schema"
                )
        elif schema == PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA:
            if set(payload) != expected_v1 | {"state_schema"}:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "exact non-Horde schema"
                )
        elif schema == _LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA:
            if set(payload) != expected_v1 | {"managed_horde_demons"}:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "managed-Horde schema"
                )
            _strict_int(
                payload["managed_horde_demons"],
                name="managed_horde_demons",
                minimum=1,
                maximum=_MAX_PYTHON_COLLECTION_LENGTH,
            )
        elif schema == PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA:
            if set(payload) != expected_v1 | {
                "managed_horde_demons",
                "state_schema",
            }:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "exact managed-Horde schema"
                )
            _strict_int(
                payload["managed_horde_demons"],
                name="managed_horde_demons",
                minimum=1,
                maximum=_MAX_PYTHON_COLLECTION_LENGTH,
            )
        elif schema == _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_CONFIG_SCHEMA:
            if set(payload) != expected_v1 | {
                "pair_source_feature_dim",
                "state_schema",
            }:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "exact pair-source schema"
                )
            _strict_int(
                payload["pair_source_feature_dim"],
                name="pair_source_feature_dim",
                minimum=2,
                maximum=_MAX_TOTAL_FEATURE_DIM - 1,
            )
        elif schema == (
            _PROTOTYPE_FEATURE_LIFECYCLE_PAIR_SOURCE_HORDE_CONFIG_SCHEMA
        ):
            if set(payload) != expected_v1 | {
                "managed_horde_demons",
                "pair_source_feature_dim",
                "state_schema",
            }:
                raise ValueError(
                    "prototype feature lifecycle config fields do not match the "
                    "exact managed-Horde pair-source schema"
                )
            _strict_int(
                payload["managed_horde_demons"],
                name="managed_horde_demons",
                minimum=1,
                maximum=_MAX_PYTHON_COLLECTION_LENGTH,
            )
            _strict_int(
                payload["pair_source_feature_dim"],
                name="pair_source_feature_dim",
                minimum=2,
                maximum=_MAX_TOTAL_FEATURE_DIM - 1,
            )
        else:
            raise ValueError("unexpected prototype feature lifecycle config schema")
        if schema in {
            _LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA,
            _LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA,
        }:
            _strict_int(
                payload["max_observations"],
                name="legacy max_observations",
                minimum=1,
                maximum=_INT32_MAX - 1,
            )
        payload.pop("schema")
        if current_schema:
            if payload.pop("state_schema") != PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA:
                raise ValueError(
                    "prototype feature lifecycle state schema is unsupported"
                )
        if payload.pop("type") != _CONFIG_TYPE:
            raise ValueError("unexpected prototype feature lifecycle config type")
        if (
            payload.pop("mechanism_status")
            != PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
        ):
            raise ValueError("prototype feature lifecycle must remain mechanism-only")
        if payload.pop("scientific_promotion_allowed") is not False:
            raise ValueError("prototype feature lifecycle config cannot claim promotion")
        raw_subtask_indices = payload.get("option_subtask_feature_indices")
        if type(raw_subtask_indices) is not list or not all(
            type(index) is int for index in raw_subtask_indices
        ):
            raise ValueError(
                "serialized option_subtask_feature_indices must be a JSON integer list"
            )
        payload["option_subtask_feature_indices"] = tuple(raw_subtask_indices)
        return PrototypeFeatureLifecycleConfig(**cast(dict[str, Any], payload))


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleState:
    """Owned discovery state, exact identities, and saturating telemetry.

    The scalar int32 counts remain compatibility telemetry only. Scheduling,
    capacity, causal composition, and checkpoint identity use the big-endian
    ``uint32[2]`` words.  All four logical clocks have fixed shape.
    """

    learner_state: InteractionFeatureState
    router_state: FeatureBankRouterState
    observe_count: Int[Array, ""]
    observe_words: UInt[Array, " 2"]
    deferred_curation_count: Int[Array, ""]
    deferred_curation_words: UInt[Array, " 2"]
    committed_curation_count: Int[Array, ""]
    committed_curation_words: UInt[Array, " 2"]
    rolled_back_curation_count: Int[Array, ""]
    rolled_back_curation_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureConsumerBinding:
    """Exact feature-bank identity carried with all bound linear consumers.

    This is caller-owned metadata. A standalone caller must persist and
    restore it together with the OaK state and any managed Horde state;
    rebuilding it from a separately restored lifecycle state would be an
    attestation, not proof that their weights share that feature-bank identity.
    """

    semantic_generation: Int[Array, ""]
    semantic_generation_words: UInt[Array, " 2"]
    descriptors: Int[Array, " active_pair_slots 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleEvent:
    """One causal discovery observation and a caller-owned commit boundary.

    ``observation`` and ``targets`` are supplied directly to the interaction
    learner.  NaN targets are its documented missing-head sentinel.
    ``next_observation`` must correspond to the supplied OaK state's cached
    ``base_last_obs`` under the current descriptor bank.  The cache is rebuilt
    under a newly committed bank, preventing mixed-generation decisions.
    """

    observation: Float[Array, " base_feature_dim"]
    targets: Float[Array, " n_tasks"]
    next_observation: Float[Array, " base_feature_dim"]
    allow_curation: Bool[Array, ""]


@dataclasses.dataclass(frozen=True)
class PrototypeFeatureLifecycleResourceBudget:
    """Exact lifecycle-owned bytes and routed-axis logical work bounds.

    The managed-consumer byte fields cover only feature axes routed by this
    lifecycle.  They intentionally do not claim the caller-owned consumers'
    complete persistent footprints or Prototype's composition-only schema
    digest.
    """

    mechanism_status: str
    scientific_promotion_allowed: bool
    base_feature_slots: int
    pair_source_feature_slots: int
    canonical_pair_universe_slots: int
    active_pair_slots: int
    candidate_pair_slots: int
    managed_oak_feature_width: int
    learner_persistent_state_nbytes: int
    router_persistent_state_nbytes: int
    lifecycle_telemetry_counter_nbytes: int
    lifecycle_exact_counter_nbytes: int
    lifecycle_counter_delta_nbytes: int
    lifecycle_counter_nbytes: int
    lifecycle_state_nbytes: int
    consumer_binding_persistent_nbytes: int
    consumer_binding_generation_nbytes: int
    consumer_binding_generation_delta_nbytes: int
    internal_learner_template_nbytes: int
    internal_oak_template_nbytes: int
    internal_template_nbytes: int
    owned_persistent_state_nbytes: int
    managed_oak_consumer_nbytes: int
    rebuilt_base_cache_nbytes: int
    input_route_feature_groups: int
    output_route_feature_groups: int
    router_calls_per_observe: int
    router_calls_per_committed_curation: int
    max_active_pair_products_per_observe: int
    max_candidate_pair_products_per_observe: int
    max_observations: int
    managed_horde_demons: int = 0
    horde_persistent_state_nbytes: int = 0
    managed_horde_consumer_nbytes: int = 0
    managed_total_consumer_nbytes: int = 0
    internal_horde_template_nbytes: int = 0

    def to_config(self) -> dict[str, str | int | bool]:
        """Return an exact JSON-compatible resource record."""

        payload = dataclasses.asdict(self)
        if self.pair_source_feature_slots == self.base_feature_slots:
            payload.pop("pair_source_feature_slots")
            payload.pop("canonical_pair_universe_slots")
        if self.managed_horde_demons == 0:
            for field_name in (
                "managed_horde_demons",
                "horde_persistent_state_nbytes",
                "managed_horde_consumer_nbytes",
                "managed_total_consumer_nbytes",
                "internal_horde_template_nbytes",
            ):
                payload.pop(field_name)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class PrototypeFeatureLifecycleExternalTransactionResourceBudget:
    """Serialized logical PyTree bytes and fixed prepare/adopt work.

    Receipt trees embed their preparation.  The logical counts therefore count
    every serialized leaf occurrence, including shared in-memory references;
    they are not claims about allocator-level physical peak memory.
    """

    managed_horde_demons: int
    lifecycle_persistent_state_nbytes_before: int
    lifecycle_persistent_state_nbytes_after: int
    source_oak_state_nbytes: int
    source_horde_state_nbytes: int
    source_consumer_binding_nbytes: int
    prepared_route_logical_nbytes: int
    readiness_receipt_logical_nbytes: int
    simultaneous_logical_transient_nbytes: int
    learner_update_evaluations_per_prepare: int
    learner_update_evaluations_per_adopt: int
    learner_update_evaluations_per_transaction: int
    router_evaluations_per_prepare: int
    router_evaluations_per_adopt: int
    router_evaluations_per_transaction: int
    persistent_capacity_growth: int

    def to_config(self) -> dict[str, int]:
        """Return an exact JSON-compatible transient resource record."""

        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class PrototypePairGradientPullback:
    """Base-coordinate pullback of a gradient over the augmented vector."""

    gradient: Float[Array, " base_feature_dim"]
    valid: Bool[Array, ""]
    semantic_generation: Int[Array, ""]
    semantic_generation_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleDiagnostics:
    """Fixed-shape audit for one observe/defer/route transaction."""

    available: Bool[Array, ""]
    state_values_valid: Bool[Array, ""]
    oak_values_valid: Bool[Array, ""]
    consumer_binding_valid: Bool[Array, ""]
    event_values_valid: Bool[Array, ""]
    next_observation_matches_oak_cache: Bool[Array, ""]
    update_capacity_available: Bool[Array, ""]
    post_update_consumer_clock_valid: Bool[Array, ""]
    learner_update_rejected: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    curation_priority_override_enabled: Bool[Array, ""]
    curation_priority_override_applied: Bool[Array, ""]
    curation_selected_active_worst_slot: Int[Array, ""]
    curation_selected_promotion_candidate: Int[Array, ""]
    curation_selected_refresh_candidate: Int[Array, ""]
    curation_proposed: Bool[Array, ""]
    safe_curation_boundary: Bool[Array, ""]
    curation_deferred: Bool[Array, ""]
    routing_attempted: Bool[Array, ""]
    input_route_valid: Bool[Array, ""]
    output_route_valid: Bool[Array, ""]
    route_states_match: Bool[Array, ""]
    routed_values_finite: Bool[Array, ""]
    curation_committed: Bool[Array, ""]
    curation_rolled_back: Bool[Array, ""]
    postcondition_checked: Bool[Array, ""]
    postcondition_valid: Bool[Array, ""]
    postcondition_rolled_back: Bool[Array, ""]
    semantic_generation_before: Int[Array, ""]
    semantic_generation_after: Int[Array, ""]
    semantic_generation_words_before: UInt[Array, " 2"]
    semantic_generation_words_after: UInt[Array, " 2"]
    observe_words_before: UInt[Array, " 2"]
    observe_words_after: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleResult:
    """Owned state, caller-owned OaK state, and complete route audit."""

    state: PrototypeFeatureLifecycleState
    oak_state: OaKState
    consumer_binding: PrototypeFeatureConsumerBinding
    next_augmented_observation: Float[Array, " total_feature_dim"]
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    input_route_diagnostics: FeatureBankRouteDiagnostics
    output_route_diagnostics: FeatureBankRouteDiagnostics
    diagnostics: PrototypeFeatureLifecycleDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleHordeDiagnostics:
    """Additional audit for an atomically managed linear Horde consumer."""

    horde_state_values_valid: Bool[Array, ""]
    pre_step_parity_valid: Bool[Array, ""]
    post_step_parity_valid: Bool[Array, ""]
    lifecycle_capacity_capped: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleHordeResult:
    """Shared lifecycle result with post-update OaK and linear Horde state."""

    state: PrototypeFeatureLifecycleState
    oak_state: OaKState
    horde_state: MultiHeadMLPState
    consumer_binding: PrototypeFeatureConsumerBinding
    next_augmented_observation: Float[Array, " total_feature_dim"]
    predictions: Float[Array, " n_tasks"]
    errors: Float[Array, " n_tasks"]
    metrics: Float[Array, " 7"]
    input_route_diagnostics: FeatureBankRouteDiagnostics
    output_route_diagnostics: FeatureBankRouteDiagnostics
    diagnostics: PrototypeFeatureLifecycleDiagnostics
    horde_diagnostics: PrototypeFeatureLifecycleHordeDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecyclePreparedRoute:
    """One source-bound ordinary successor and routed destination candidate.

    The preparation is transient.  It captures every source/event/result leaf
    needed to bind an external readiness decision without evaluating the
    learner again during adoption.
    """

    source_state: PrototypeFeatureLifecycleState
    source_oak_state: OaKState
    source_consumer_binding: PrototypeFeatureConsumerBinding
    event: PrototypeFeatureLifecycleEvent
    curation_priority_override: InteractionCurationPriorityOverride | None
    ordinary_result: PrototypeFeatureLifecycleResult
    destination_result: PrototypeFeatureLifecycleResult
    internally_valid: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecyclePreparedHordeRoute:
    """Horde-managed source successor and routed destination candidate."""

    source_state: PrototypeFeatureLifecycleState
    source_oak_state: OaKState
    source_horde_state: MultiHeadMLPState
    source_consumer_binding: PrototypeFeatureConsumerBinding
    event: PrototypeFeatureLifecycleEvent
    curation_priority_override: InteractionCurationPriorityOverride | None
    ordinary_result: PrototypeFeatureLifecycleHordeResult
    destination_result: PrototypeFeatureLifecycleHordeResult
    internally_valid: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleExternalReadinessReceipt:
    """Exact unkeyed content binding for an externally trusted verdict."""

    prepared_route: PrototypeFeatureLifecyclePreparedRoute
    all_consumers_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleHordeExternalReadinessReceipt:
    """Exact unkeyed content binding for an externally trusted Horde verdict."""

    prepared_route: PrototypeFeatureLifecyclePreparedHordeRoute
    all_consumers_ready: Bool[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleAdoptionDiagnostics:
    """Source, receipt, external-veto, and exact-work adoption facts."""

    source_state_matches: Bool[Array, ""]
    source_oak_state_matches: Bool[Array, ""]
    source_horde_state_matches: Bool[Array, ""]
    source_consumer_binding_matches: Bool[Array, ""]
    receipt_matches_preparation: Bool[Array, ""]
    preparation_internally_valid: Bool[Array, ""]
    all_consumers_ready: Bool[Array, ""]
    destination_adopted: Bool[Array, ""]
    ordinary_update_retained: Bool[Array, ""]
    external_curation_rolled_back: Bool[Array, ""]
    transaction_applied: Bool[Array, ""]
    rejected: Bool[Array, ""]
    preparation_learner_update_evaluations: Int[Array, ""]
    adoption_learner_update_evaluations: Int[Array, ""]
    total_learner_update_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleAdoptionResult:
    """Legacy-shaped result plus audit of external readiness adoption."""

    result: PrototypeFeatureLifecycleResult
    diagnostics: PrototypeFeatureLifecycleAdoptionDiagnostics


@chex.dataclass(frozen=True)
class PrototypeFeatureLifecycleHordeAdoptionResult:
    """Horde legacy-shaped result plus external readiness adoption audit."""

    result: PrototypeFeatureLifecycleHordeResult
    diagnostics: PrototypeFeatureLifecycleAdoptionDiagnostics


def _array_has_contract(value: Any, shape: tuple[int, ...], dtype: Any) -> bool:
    """Return an exact noncoercing array shape/dtype predicate."""

    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and value.shape == shape
        and value.dtype == dtype
    )


def _require_array(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: Any,
) -> Array:
    """Reject shape and effective-dtype mismatches before indexed work."""

    if not hasattr(value, "shape") or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not hasattr(value, "dtype") or value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    return cast(Array, value)


def _static_tree_matches(value: Any, template: Any) -> bool:
    """Compare exact PyTree nodes and leaf contracts, tolerating host timers."""

    try:
        value_leaves, value_tree = jax.tree_util.tree_flatten(value)
        template_leaves, template_tree = jax.tree_util.tree_flatten(template)
    except (AttributeError, TypeError, ValueError):
        return False
    if (
        value_tree != template_tree  # type: ignore[operator]
        or len(value_leaves) != len(template_leaves)
    ):
        return False
    for actual, expected in zip(value_leaves, template_leaves, strict=True):
        if type(expected) is float:
            if type(actual) is float:
                continue
            if not (
                hasattr(actual, "shape")
                and hasattr(actual, "dtype")
                and actual.shape == ()
                and actual.dtype == jnp.float32
            ):
                return False
            continue
        if not (
            hasattr(actual, "shape")
            and hasattr(actual, "dtype")
            and hasattr(expected, "shape")
            and hasattr(expected, "dtype")
            and actual.shape == expected.shape
            and actual.dtype == expected.dtype
        ):
            return False
    return True


def _floating_tree_is_finite(value: Any) -> Bool[Array, ""]:
    """Return whether all inexact leaves, including host timers, are finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(array))
    return valid


def _tree_nbytes(value: Any) -> int:
    """Count physical bytes reported by every persistent PyTree leaf."""

    return sum(
        int(getattr(leaf, "nbytes", 0))
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _trees_exactly_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Return one JAX boolean for exact equality of matching array PyTrees."""

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if (
        left_tree != right_tree  # type: ignore[operator]
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        valid = valid & jnp.array_equal(jnp.asarray(left_leaf), jnp.asarray(right_leaf))
    return valid


def _trees_bit_exactly_equal(left: Any, right: Any) -> Bool[Array, ""]:
    """Bit-authenticate matching JAX PyTrees, including float32 signed zero."""

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    if (
        left_tree != right_tree  # type: ignore[operator]
        or len(left_leaves) != len(right_leaves)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        left_array = jnp.asarray(left_leaf)
        right_array = jnp.asarray(right_leaf)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return jnp.asarray(False, dtype=jnp.bool_)
        if jax.dtypes.issubdtype(left_array.dtype, jax.dtypes.prng_key):
            leaf_equal = jnp.array_equal(
                jr.key_data(left_array),
                jr.key_data(right_array),
            )
        elif left_array.dtype == jnp.dtype(jnp.float32):
            leaf_equal = jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint32),
                jax.lax.bitcast_convert_type(right_array, jnp.uint32),
            )
        elif left_array.dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
            leaf_equal = jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint16),
                jax.lax.bitcast_convert_type(right_array, jnp.uint16),
            )
        elif left_array.dtype == jnp.dtype(jnp.float64):
            leaf_equal = jnp.array_equal(
                jax.lax.bitcast_convert_type(left_array, jnp.uint64),
                jax.lax.bitcast_convert_type(right_array, jnp.uint64),
            )
        else:
            leaf_equal = jnp.array_equal(left_array, right_array)
        valid = valid & leaf_equal
    return valid


def _exact_json_tree_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like trees without Python's bool/int or int/float aliases."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_json_tree_equal(left[key], right[key])
            for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_json_tree_equal(left_value, right_value)
            for left_value, right_value in zip(
                left,
                right,
                strict=True,
            )
        )
    if type(left) is float:
        return struct.pack("!d", left) == struct.pack("!d", right)
    return bool(left == right)


def _float32_arrays_bit_exact(left: Array, right: Array) -> Bool[Array, ""]:
    """Compare float32 arrays by representation, distinguishing signed zero."""

    return jnp.array_equal(
        jax.lax.bitcast_convert_type(left, jnp.int32),
        jax.lax.bitcast_convert_type(right, jnp.int32),
    )


def _saturating_int32_increment(value: Array) -> Int[Array, ""]:
    """Advance compatibility telemetry without signed wraparound."""

    maximum = jnp.asarray(_INT32_MAX, dtype=jnp.int32)
    safe = jnp.minimum(jnp.maximum(value, 0), maximum - 1)
    return safe + jnp.asarray(1, dtype=jnp.int32)


def _checked_lifetime_words_increment(
    words: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Propose one exact increment without wrapping the all-ones identity."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifecycle lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifecycle lifetime words must have dtype uint32")
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    capacity = ~jnp.all(words == maximum)
    one = jnp.asarray(1, dtype=jnp.uint32)
    low = words[1] + one
    carry = (low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    proposed = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(capacity, proposed, words), capacity


def _checked_lifetime_words_add(
    left: Array,
    right: Array,
) -> tuple[UInt[Array, " 2"], Bool[Array, ""]]:
    """Add two exact identities, reporting rather than wrapping overflow."""

    for name, words in (("left", left), ("right", right)):
        if getattr(words, "shape", None) != (2,):
            raise ValueError(f"{name} lifecycle words must have shape (2,)")
        if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
            raise TypeError(f"{name} lifecycle words must have dtype uint32")
    low = left[1] + right[1]
    carry = (low < left[1]).astype(jnp.uint32)
    high_without_carry = left[0] + right[0]
    overflow_without_carry = high_without_carry < left[0]
    high = high_without_carry + carry
    overflow_with_carry = high < high_without_carry
    capacity = ~(overflow_without_carry | overflow_with_carry)
    proposed = jnp.stack((high, low)).astype(jnp.uint32)
    return jnp.where(capacity, proposed, left), capacity


def _lifetime_counter_valid(
    words: Array,
    telemetry: Array,
) -> Bool[Array, ""]:
    """Bind one exact identity to its saturating int32 telemetry."""

    if getattr(words, "shape", None) != (2,):
        raise ValueError("lifecycle lifetime words must have shape (2,)")
    if getattr(words, "dtype", None) != jnp.dtype(jnp.uint32):
        raise TypeError("lifecycle lifetime words must have dtype uint32")
    if getattr(telemetry, "shape", None) != ():
        raise ValueError("lifecycle telemetry must be scalar")
    if getattr(telemetry, "dtype", None) != jnp.dtype(jnp.int32):
        raise TypeError("lifecycle telemetry must have dtype int32")
    below_saturation = (words[0] == jnp.asarray(0, dtype=jnp.uint32)) & (
        words[1] < jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
    )
    return (telemetry >= 0) & jnp.where(
        below_saturation,
        telemetry == words[1].astype(jnp.int32),
        telemetry == jnp.asarray(_INT32_MAX, dtype=jnp.int32),
    )


def _lifetime_words_le(left: Array, right: Array) -> Bool[Array, ""]:
    """Lexicographic comparison of big-endian uint32 word pairs."""

    return (left[0] < right[0]) | (
        (left[0] == right[0]) & (left[1] <= right[1])
    )


def _python_uint64_words(value: int) -> UInt[Array, " 2"]:
    """Represent one already-validated Python uint64 as two JAX words."""

    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _curation_outcome_words_within_observations(
    deferred: Array,
    committed: Array,
    rolled_back: Array,
    observations: Array,
) -> Bool[Array, ""]:
    """Prove the disjoint persisted curation outcomes fit causal updates."""

    partial, partial_capacity = _checked_lifetime_words_add(deferred, committed)
    total, total_capacity = _checked_lifetime_words_add(partial, rolled_back)
    return partial_capacity & total_capacity & _lifetime_words_le(total, observations)


def _nonnegative_int32_sum_within(
    values: Array,
    limit: Array,
) -> Bool[Array, ""]:
    """Check a sum against a limit without allowing int32 wraparound."""

    if values.shape == (0,):
        return limit >= jnp.asarray(0, dtype=jnp.int32)

    def step(carry: tuple[Array, Array], value: Array) -> tuple[tuple[Array, Array], None]:
        total, valid = carry
        fits = (value >= 0) & (value <= limit - total)
        safe_value = jnp.where(fits, value, jnp.asarray(0, dtype=jnp.int32))
        return (total + safe_value, valid & fits), None

    (_, valid), _ = jax.lax.scan(
        step,
        (jnp.asarray(0, dtype=jnp.int32), jnp.asarray(True, dtype=jnp.bool_)),
        values,
    )
    return valid


class PrototypeFeatureLifecycle:
    """Standalone fixed-bank feature discovery and exact route transaction."""

    def __init__(self, config: PrototypeFeatureLifecycleConfig):
        if type(config) is not PrototypeFeatureLifecycleConfig:
            raise TypeError("config must be a PrototypeFeatureLifecycleConfig")
        self._config = config
        task_utility_weights = (
            (
                0.5,
                *(
                    0.5 / config.managed_horde_demons
                    for _ in range(config.managed_horde_demons)
                ),
            )
            if config.managed_horde_demons > 0
            else None
        )
        self._learner = FixedBudgetInteractionLearner(
            n_features=config.active_pair_slots,
            n_tasks=config.n_tasks,
            step_size_output=config.step_size_output,
            utility_decay=config.utility_decay,
            replacement_interval=config.replacement_interval,
            min_feature_age=config.min_feature_age,
            candidate_count=config.candidate_pair_slots,
            candidate_min_age=config.candidate_min_age,
            promotion_margin=config.promotion_margin,
            promotion_blend=1.0,
            generator_mix=(1.0, 0.0, 0.0),
            candidate_strategy="all_pairs",
            utility_aggregation="mean",
            utility_task_balancing="active",
            task_utility_weights=task_utility_weights,
            task_activity_decay=config.utility_decay,
            future_utility_mix=0.0,
            refresh_candidates=False,
            refresh_promoted_candidate=False,
            include_squares=False,
            use_obgd=False,
            scale_robust=True,
            scale_normalizer_decay=config.scale_normalizer_decay,
            scale_normalizer_epsilon=config.scale_normalizer_epsilon,
        )
        self._router = FeatureBankRouter(
            FeatureBankRouterConfig(
                base_dim=config.base_feature_dim,
                active_slots=config.active_pair_slots,
            )
        )
        self._learner_template = self._initial_learner_state(jr.key(0))
        self._oak_template = self._make_oak_template()
        self._horde_template = (
            MultiHeadMLPLearner(
                n_heads=config.managed_horde_demons,
                hidden_sizes=(),
                step_size=1.0,
            ).init(config.total_feature_dim, jr.key(0))
            if config.managed_horde_demons > 0
            else None
        )

    @property
    def config(self) -> PrototypeFeatureLifecycleConfig:
        """Return the immutable lifecycle configuration."""

        return self._config

    @property
    def learner(self) -> FixedBudgetInteractionLearner:
        """Return the unchanged fixed-budget interaction learner."""

        return self._learner

    @property
    def router(self) -> FeatureBankRouter:
        """Return the unchanged descriptor-identity router."""

        return self._router

    def to_config(self) -> dict[str, object]:
        """Serialize the exact lifecycle configuration."""

        return self._config.to_config()

    def require_compatible_oak_config(self, oak_config: OaKConfig) -> None:
        """Validate the caller attestation against an actual OaK config.

        This bind-time check cannot be recovered from ``OaKState`` later,
        because the state intentionally contains no SubtaskSpecs.
        """

        if type(oak_config) is not OaKConfig:
            raise TypeError("oak_config must be an OaKConfig")
        stomp = oak_config.stomp
        if type(stomp) is not STOMPConfig:
            raise TypeError("oak_config.stomp must be an exact STOMPConfig")
        specs = stomp.subtask_specs
        exact_specs = type(specs) is tuple and all(
            type(spec) is SubtaskSpec and type(spec.feature_index) is int
            for spec in specs
        )
        indices = tuple(spec.feature_index for spec in specs)
        compatible = (
            exact_specs
            and type(stomp.observation_dim) is int
            and type(stomp.n_primitive_actions) is int
            and type(stomp.base_hidden_sizes) is tuple
            and stomp.observation_dim == self._config.total_feature_dim
            and stomp.n_primitive_actions == self._config.n_primitive_actions
            and stomp.n_options == self._config.n_options
            and stomp.base_hidden_sizes == ()
            and indices == self._config.option_subtask_feature_indices
        )
        if not compatible:
            raise ValueError(
                "actual OaK config does not match the linear layout and subtask attestation"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
    ) -> PrototypeFeatureLifecycle:
        """Construct from the strict versioned configuration."""

        return PrototypeFeatureLifecycle(
            PrototypeFeatureLifecycleConfig.from_config(config)
        )

    def _canonical_active_descriptors(self) -> Array:
        pairs: list[tuple[int, int]] = []
        source_dim = self._config.effective_pair_source_feature_dim
        for left in range(source_dim):
            for right in range(left + 1, source_dim):
                pairs.append((left, right))
                if len(pairs) == self._config.active_pair_slots:
                    break
            if len(pairs) == self._config.active_pair_slots:
                break
        return jnp.asarray(
            pairs,
            dtype=jnp.int32,
        )

    def _initial_learner_state(self, key: Array) -> InteractionFeatureState:
        state = self._learner.init(
            feature_dim=self._config.effective_pair_source_feature_dim,
            key=key,
        )
        descriptors = self._canonical_active_descriptors()
        return cast(
            InteractionFeatureState,
            state.replace(
                feature_left=descriptors[:, 0],
                feature_right=descriptors[:, 1],
            ),
        )

    def _make_oak_template(self) -> OaKState:
        specs = tuple(
            SubtaskSpec(feature_index=feature_index)
            for feature_index in self._config.option_subtask_feature_indices
        )
        agent = OaKAgent(
            OaKConfig(
                stomp=STOMPConfig(
                    subtask_specs=specs,
                    observation_dim=self._config.total_feature_dim,
                    n_primitive_actions=self._config.n_primitive_actions,
                    base_hidden_sizes=(),
                )
            )
        )
        return agent.init(jr.key(0))

    def _augment_base_observation(
        self,
        learner_state: InteractionFeatureState,
        base_observation: Array,
    ) -> Array:
        """Append source-prefix pair products to the complete stable base."""

        source = base_observation[
            : self._config.effective_pair_source_feature_dim
        ]
        pairs = self._learner.constructed_features(learner_state, source)
        return jnp.concatenate((base_observation, pairs))

    def init(self, key: Array) -> PrototypeFeatureLifecycleState:
        """Initialize a unique canonical bank and zero lifecycle counters."""

        if not (
            hasattr(key, "shape")
            and hasattr(key, "dtype")
            and key.shape == ()
            and jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key)
        ):
            raise TypeError("key must be a scalar typed JAX PRNG key")
        learner_state = self._initial_learner_state(key)
        descriptors = jnp.stack(
            (learner_state.feature_left, learner_state.feature_right),
            axis=1,
        )
        zero = jnp.asarray(0, dtype=jnp.int32)
        zero_words = jnp.zeros((2,), dtype=jnp.uint32)
        return PrototypeFeatureLifecycleState(
            learner_state=learner_state,
            router_state=self._router.init(descriptors),
            observe_count=zero,
            observe_words=zero_words,
            deferred_curation_count=zero,
            deferred_curation_words=zero_words,
            committed_curation_count=zero,
            committed_curation_words=zero_words,
            rolled_back_curation_count=zero,
            rolled_back_curation_words=zero_words,
        )

    def init_bound(
        self,
        key: Array,
    ) -> tuple[PrototypeFeatureLifecycleState, PrototypeFeatureConsumerBinding]:
        """Initialize lifecycle state and its inseparable consumer identity."""

        state = self.init(key)
        return state, PrototypeFeatureConsumerBinding(
            semantic_generation=state.router_state.generation_count,
            semantic_generation_words=state.router_state.generation_words,
            descriptors=state.router_state.descriptors,
        )

    def _consumer_binding_static_contract_valid(self, binding: Any) -> bool:
        return (
            type(binding) is PrototypeFeatureConsumerBinding
            and _array_has_contract(
                binding.semantic_generation,
                (),
                jnp.int32,
            )
            and _array_has_contract(
                binding.semantic_generation_words,
                (2,),
                jnp.uint32,
            )
            and _array_has_contract(
                binding.descriptors,
                (self._config.active_pair_slots, 2),
                jnp.int32,
            )
        )

    def consumer_binding_valid(
        self,
        state: PrototypeFeatureLifecycleState,
        binding: PrototypeFeatureConsumerBinding,
    ) -> Bool[Array, ""]:
        """Return whether a caller-owned OaK binding exactly matches ``state``."""

        if not self._state_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        if not self._consumer_binding_static_contract_valid(binding):
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            (binding.semantic_generation >= 0)
            & _lifetime_counter_valid(
                binding.semantic_generation_words,
                binding.semantic_generation,
            )
            & (
                binding.semantic_generation
                == state.router_state.generation_count
            )
            & jnp.all(
                binding.semantic_generation_words
                == state.router_state.generation_words
            )
            & jnp.array_equal(
                binding.descriptors,
                state.router_state.descriptors,
            )
        )

    def _state_static_contract_valid(self, state: Any) -> bool:
        if type(state) is not PrototypeFeatureLifecycleState:
            return False
        template = PrototypeFeatureLifecycleState(
            learner_state=self._learner_template,
            router_state=self._router.init(self._canonical_active_descriptors()),
            observe_count=jnp.asarray(0, dtype=jnp.int32),
            observe_words=jnp.zeros((2,), dtype=jnp.uint32),
            deferred_curation_count=jnp.asarray(0, dtype=jnp.int32),
            deferred_curation_words=jnp.zeros((2,), dtype=jnp.uint32),
            committed_curation_count=jnp.asarray(0, dtype=jnp.int32),
            committed_curation_words=jnp.zeros((2,), dtype=jnp.uint32),
            rolled_back_curation_count=jnp.asarray(0, dtype=jnp.int32),
            rolled_back_curation_words=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return _static_tree_matches(state, template)

    def _oak_static_contract_valid(self, state: Any) -> bool:
        return type(state) is OaKState and _static_tree_matches(
            state,
            self._oak_template,
        )

    def _horde_static_contract_valid(self, state: Any) -> bool:
        """Return the exact supported linear-Horde structural contract."""

        template = self._horde_template
        if (
            self._config.managed_horde_demons == 0
            or template is None
            or type(state) is not MultiHeadMLPState
            or not _static_tree_matches(state, template)
        ):
            return False
        if not (
            type(state.trunk_params.weights) is tuple
            and type(state.trunk_params.biases) is tuple
            and type(state.trunk_optimizer_states) is tuple
            and type(state.trunk_traces) is tuple
            and type(state.hidden_unit_utilities) is tuple
            and type(state.head_params.weights) is tuple
            and type(state.head_params.biases) is tuple
            and type(state.head_optimizer_states) is tuple
            and type(state.head_traces) is tuple
            and state.normalizer_state is None
            and len(state.trunk_params.weights) == 0
            and len(state.trunk_params.biases) == 0
            and len(state.trunk_optimizer_states) == 0
            and len(state.trunk_traces) == 0
            and len(state.hidden_unit_utilities) == 0
            and len(state.head_params.weights)
            == self._config.managed_horde_demons
            and len(state.head_params.biases)
            == self._config.managed_horde_demons
            and len(state.head_optimizer_states)
            == self._config.managed_horde_demons
            and len(state.head_traces) == self._config.managed_horde_demons
        ):
            return False
        for head_index in range(self._config.managed_horde_demons):
            trace_pair = state.head_traces[head_index]
            optimizer_pair = state.head_optimizer_states[head_index]
            if not (
                type(trace_pair) is tuple
                and len(trace_pair) == 2
                and type(optimizer_pair) is tuple
                and len(optimizer_pair) == 2
                and type(optimizer_pair[0]) is LMSState
                and type(optimizer_pair[1]) is LMSState
            ):
                return False
        return True

    def horde_state_valid(self, state: MultiHeadMLPState) -> Bool[Array, ""]:
        """Validate one exact linear Horde consumer without coercion.

        This predicate is available only for a managed-Horde configuration.
        A static mismatch, including use on a legacy no-Horde lifecycle,
        returns a scalar false rather than reaching indexed work.
        """

        if not self._horde_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        step_sizes = jnp.stack(
            tuple(
                optimizer_state.step_size
                for optimizer_pair in state.head_optimizer_states
                for optimizer_state in optimizer_pair
            )
        )
        first_step_bits = jax.lax.bitcast_convert_type(
            step_sizes[0],
            jnp.int32,
        )
        step_sizes_equal = jnp.all(
            jax.lax.bitcast_convert_type(step_sizes, jnp.int32)
            == first_step_bits
        )
        return (
            _floating_tree_is_finite(state)
            & (jnp.asarray(state.birth_timestamp) >= 0.0)
            & (jnp.asarray(state.uptime_s) >= 0.0)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & jnp.all(step_sizes > 0.0)
            & step_sizes_equal
        )

    def state_valid(
        self,
        state: PrototypeFeatureLifecycleState,
    ) -> Bool[Array, ""]:
        """Validate exact structure, descriptor identity, values, and counters."""

        if not self._state_static_contract_valid(state):
            return jnp.asarray(False, dtype=jnp.bool_)
        learner = state.learner_state
        active_descriptors = jnp.stack(
            (learner.feature_left, learner.feature_right),
            axis=1,
        )
        active_validation = self._router.validate_descriptors(active_descriptors)
        router_validation = self._router.validate_descriptors(
            state.router_state.descriptors
        )
        canonical_descriptors = self._canonical_active_descriptors()
        canonical_row_distance = jnp.sum(
            jnp.any(active_descriptors != canonical_descriptors, axis=1),
            dtype=jnp.int32,
        )
        descriptor_history_valid = (
            canonical_row_distance
            <= state.router_state.generation_count
        )

        candidate_left = learner.candidate_left
        candidate_right = learner.candidate_right
        candidate_live = (
            (candidate_left >= 0)
            & (candidate_left < candidate_right)
            & (
                candidate_right
                < self._config.effective_pair_source_feature_dim
            )
        )
        candidate_descriptors = jnp.stack(
            (candidate_left, candidate_right),
            axis=1,
        )
        candidate_equal = jnp.all(
            candidate_descriptors[:, None, :] == candidate_descriptors[None, :, :],
            axis=-1,
        )
        candidate_duplicate = jnp.any(
            candidate_equal
            & ~jnp.eye(
                self._config.candidate_pair_slots,
                dtype=jnp.bool_,
            )
        )

        nonnegative_counters = (
            (learner.step_count >= 0)
            & jnp.all(learner.ages >= 0)
            & jnp.all(learner.candidate_ages >= 0)
            & jnp.all(learner.evidence_idle_steps >= 0)
            & jnp.all(learner.utility_evidence_streak >= 0)
            & jnp.all(learner.candidate_promotion_evidence_streak >= 0)
            & (state.router_state.route_count >= 0)
            & (state.router_state.generation_count >= 0)
            & (state.observe_count >= 0)
            & (state.deferred_curation_count >= 0)
            & (state.committed_curation_count >= 0)
            & (state.rolled_back_curation_count >= 0)
        )
        lifecycle_counters_valid = (
            _lifetime_counter_valid(state.observe_words, state.observe_count)
            & _lifetime_counter_valid(
                state.deferred_curation_words,
                state.deferred_curation_count,
            )
            & _lifetime_counter_valid(
                state.committed_curation_words,
                state.committed_curation_count,
            )
            & _lifetime_counter_valid(
                state.rolled_back_curation_words,
                state.rolled_back_curation_count,
            )
        )
        learner_counter_progress_valid = (
            jnp.all(learner.ages <= learner.step_count)
            & jnp.all(learner.candidate_ages <= learner.step_count)
            & jnp.all(learner.evidence_idle_steps <= learner.step_count)
            & jnp.all(learner.utility_evidence_streak <= learner.step_count)
            & jnp.all(
                learner.candidate_promotion_evidence_streak
                <= learner.step_count
            )
        )
        exact_counter_composition_valid = (
            _lifetime_words_le(
                state.observe_words,
                _python_uint64_words(self._config.max_observations),
            )
            & jnp.all(learner.step_words == state.observe_words)
            & (learner.step_count == state.observe_count)
            & _curation_outcome_words_within_observations(
                state.deferred_curation_words,
                state.committed_curation_words,
                state.rolled_back_curation_words,
                state.observe_words,
            )
            & (state.router_state.route_count == state.committed_curation_count)
            & jnp.all(
                state.router_state.route_words
                == state.committed_curation_words
            )
            & (
                state.router_state.generation_count
                == state.committed_curation_count
            )
            & jnp.all(
                state.router_state.generation_words
                == state.committed_curation_words
            )
        )
        learner_transaction_valid = self._learner._transaction_state_valid(
            learner,
            feature_dim=self._config.effective_pair_source_feature_dim,
        )
        provenance_valid = (
            jnp.all(learner.feature_parent_a == -1)
            & jnp.all(learner.feature_parent_b == -1)
            & jnp.all(learner.candidate_parent_a == -1)
            & jnp.all(learner.candidate_parent_b == -1)
            & jnp.all(learner.feature_generator == GENERATOR_RANDOM)
            & jnp.all(learner.candidate_generator == GENERATOR_RANDOM)
        )
        moments_valid = (
            jnp.all(learner.utilities >= 0.0)
            & jnp.all(learner.utilities <= 1.0)
            & jnp.all(learner.candidate_utilities >= 0.0)
            & jnp.all(learner.candidate_utilities <= 1.0)
            & jnp.all(learner.feature_second_moments >= 0.0)
            & jnp.all(learner.candidate_second_moments >= 0.0)
            & jnp.all(learner.target_second_moments >= 0.0)
            & jnp.all(learner.task_activity_ema >= 0.0)
            & jnp.all(learner.task_activity_ema <= 1.0)
        )
        timer_values_valid = (
            learner.birth_timestamp >= 0.0
        ) & (learner.uptime_s >= 0.0)
        fixed_disabled_substate_valid = (
            _float32_arrays_bit_exact(
                learner.relevance_probe_weights,
                jnp.zeros_like(learner.relevance_probe_weights),
            )
            & _float32_arrays_bit_exact(
                learner.relevance_probe_biases,
                learner.output_biases,
            )
            & jnp.all(learner.evidence_idle_steps == 0)
            & jnp.all(learner.utility_evidence_streak == 0)
            & jnp.all(learner.candidate_promotion_evidence_streak == 0)
            & ~jnp.any(learner.active_output_memory_committed)
            & ~jnp.any(learner.candidate_reacquisition_required)
        )
        return (
            active_validation.valid
            & jnp.all(active_validation.live_mask)
            & router_validation.valid
            & jnp.all(router_validation.live_mask)
            & jnp.all(candidate_live)
            & ~candidate_duplicate
            & jnp.array_equal(active_descriptors, state.router_state.descriptors)
            & descriptor_history_valid
            & _floating_tree_is_finite(learner)
            & nonnegative_counters
            & lifecycle_counters_valid
            & learner_counter_progress_valid
            & exact_counter_composition_valid
            & learner_transaction_valid
            & provenance_valid
            & moments_valid
            & timer_values_valid
            & fixed_disabled_substate_valid
        )

    def _oak_values_valid(self, state: OaKState) -> Bool[Array, ""]:
        """Apply the public STOMP ownership audit plus outer OaK checks."""

        stomp = state.stomp_state
        dispatch = replace_dispatched_primitive_action(
            stomp,
            stomp.base_last_obs,
            stomp.last_primitive_action,
            jnp.ones(
                (self._config.n_primitive_actions,),
                dtype=jnp.bool_,
            ),
        ).decision
        execution_limit = jnp.where(
            state.step_count < _INT32_MAX,
            state.step_count + jnp.asarray(1, dtype=jnp.int32),
            state.step_count,
        )
        option_completions = stomp.option_models.n_completions
        timer_values_valid = (
            jnp.asarray(stomp.base_learner_state.birth_timestamp) >= 0.0
        ) & (jnp.asarray(stomp.base_learner_state.uptime_s) >= 0.0)
        exact_clocks_valid = (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _lifetime_counter_valid(stomp.step_words, stomp.step_count)
            & _lifetime_counter_valid(
                stomp.base_learner_state.step_words,
                stomp.base_learner_state.step_count,
            )
            & jnp.all(state.step_words == stomp.step_words)
        )
        # Per-option completion/execution telemetry has no exact words.  Its
        # aggregate is causal while the outer exact clock remains int32-sized;
        # after that point, summing several saturated telemetry slots would
        # spuriously reject a valid continuing OaK history.
        summary_telemetry_identifiable = (
            state.step_words[0] == jnp.asarray(0, dtype=jnp.uint32)
        ) & (
            state.step_words[1]
            <= jnp.asarray(_INT32_MAX, dtype=jnp.uint32)
        )
        execution_summary_valid = jnp.where(
            summary_telemetry_identifiable,
            _nonnegative_int32_sum_within(
                state.execution_counts,
                execution_limit,
            ),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        completion_summary_valid = jnp.where(
            summary_telemetry_identifiable,
            _nonnegative_int32_sum_within(
                option_completions,
                state.step_count,
            ),
            jnp.asarray(True, dtype=jnp.bool_),
        )
        return (
            dispatch.state_valid
            & dispatch.observation_matches
            & exact_clocks_valid
            & _floating_tree_is_finite(state)
            & jnp.all(state.execution_counts >= 0)
            & execution_summary_valid
            & jnp.all(option_completions <= state.execution_counts)
            & completion_summary_valid
            & timer_values_valid
            & (state.step_count >= 0)
            & (state.step_count == stomp.step_count)
        )

    def augment(
        self,
        state: PrototypeFeatureLifecycleState,
        observation: Array,
    ) -> Float[Array, " total_feature_dim"]:
        """Return the fixed-width base prefix plus live pair products.

        Static violations raise before indexed work.  Dynamic invalidity is a
        finite all-zero fail-closed result so this method is safe inside JIT
        and scan composition.
        """

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        raw = _require_array(
            observation,
            name="observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        augmented = self._augment_base_observation(state.learner_state, raw)
        valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(augmented))
        )
        return jnp.where(valid, augmented, jnp.zeros_like(augmented))

    def pullback_pair_gradient(
        self,
        state: PrototypeFeatureLifecycleState,
        observation: Array,
        augmented_gradient: Array,
        expected_generation: Array,
        expected_bank_descriptors: Array,
        *,
        expected_generation_words: Array | None = None,
    ) -> PrototypePairGradientPullback:
        """Apply the chain rule only for one exact generation and bank.

        The optional word argument preserves an unsaturated legacy call path.
        Once generation telemetry saturates, callers must supply the exact
        words; an int32 generation plus descriptors is no longer a unique
        historical owner.
        """

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        raw = _require_array(
            observation,
            name="observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        gradient = _require_array(
            augmented_gradient,
            name="augmented_gradient",
            shape=(self._config.total_feature_dim,),
            dtype=jnp.float32,
        )
        generation = _require_array(
            expected_generation,
            name="expected_generation",
            shape=(),
            dtype=jnp.int32,
        )
        expected_descriptors = _require_array(
            expected_bank_descriptors,
            name="expected_bank_descriptors",
            shape=(self._config.active_pair_slots, 2),
            dtype=jnp.int32,
        )
        if expected_generation_words is None:
            generation_words = jnp.stack(
                (
                    jnp.asarray(0, dtype=jnp.uint32),
                    generation.astype(jnp.uint32),
                )
            )
            generation_words_available = (
                generation >= jnp.asarray(0, dtype=jnp.int32)
            ) & (generation < jnp.asarray(_INT32_MAX, dtype=jnp.int32))
        else:
            generation_words = _require_array(
                expected_generation_words,
                name="expected_generation_words",
                shape=(2,),
                dtype=jnp.uint32,
            )
            generation_words_available = _lifetime_counter_valid(
                generation_words,
                generation,
            )
        left = state.learner_state.feature_left
        right = state.learner_state.feature_right
        live = (
            (left >= 0)
            & (left < right)
            & (right < self._config.effective_pair_source_feature_dim)
        )
        safe_left = jnp.where(live, left, 0)
        safe_right = jnp.where(live, right, 0)
        pair_gradient = gradient[self._config.base_feature_dim :]
        pulled = gradient[: self._config.base_feature_dim]
        pulled = pulled.at[safe_left].add(
            jnp.where(live, pair_gradient * raw[safe_right], 0.0)
        )
        pulled = pulled.at[safe_right].add(
            jnp.where(live, pair_gradient * raw[safe_left], 0.0)
        )
        valid = (
            self.state_valid(state)
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(pulled))
            & generation_words_available
            & (generation == state.router_state.generation_count)
            & jnp.all(
                generation_words == state.router_state.generation_words
            )
            & jnp.array_equal(
                expected_descriptors,
                state.router_state.descriptors,
            )
        )
        return PrototypePairGradientPullback(
            gradient=jnp.where(valid, pulled, jnp.zeros_like(pulled)),
            valid=valid,
            semantic_generation=jnp.where(
                valid,
                state.router_state.generation_count,
                jnp.asarray(0, dtype=jnp.int32),
            ),
            semantic_generation_words=jnp.where(
                valid,
                state.router_state.generation_words,
                jnp.zeros((2,), dtype=jnp.uint32),
            ),
        )

    def resource_budget(
        self,
        state: PrototypeFeatureLifecycleState | None = None,
        horde_state: MultiHeadMLPState | None = None,
    ) -> PrototypeFeatureLifecycleResourceBudget:
        """Return exact owned bytes and routed consumer/work bounds."""

        measured = self.init(jr.key(0)) if state is None else state
        if not self._state_static_contract_valid(measured):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        width = self._config.total_feature_dim
        heads = self._config.n_total_actions
        options = self._config.n_options
        primitive_actions = self._config.n_primitive_actions
        oak_input_groups = (
            heads
            + heads
            + options * primitive_actions
            + options * primitive_actions
            + options * width
            + 1
        )
        horde_input_groups = 2 * self._config.managed_horde_demons
        input_groups = oak_input_groups + horde_input_groups
        if self._config.managed_horde_demons == 0:
            if horde_state is not None:
                raise ValueError("legacy lifecycle cannot account for Horde state")
            measured_horde = None
        else:
            measured_horde = self._horde_template if horde_state is None else horde_state
            if measured_horde is None or not self._horde_static_contract_valid(
                measured_horde
            ):
                raise ValueError("managed Horde state has an invalid static contract")

        output_groups = options * width
        managed_oak_consumer_scalars = oak_input_groups * width
        lifecycle_state_nbytes = _tree_nbytes(measured)
        consumer_binding_nbytes = _tree_nbytes(
            PrototypeFeatureConsumerBinding(
                semantic_generation=measured.router_state.generation_count,
                semantic_generation_words=measured.router_state.generation_words,
                descriptors=measured.router_state.descriptors,
            )
        )
        internal_learner_template_nbytes = _tree_nbytes(self._learner_template)
        internal_oak_template_nbytes = _tree_nbytes(self._oak_template)
        internal_horde_template_nbytes = (
            _tree_nbytes(self._horde_template)
            if self._horde_template is not None
            else 0
        )
        internal_template_nbytes = (
            internal_learner_template_nbytes
            + internal_oak_template_nbytes
            + internal_horde_template_nbytes
        )
        managed_oak_consumer_nbytes = 4 * (
            managed_oak_consumer_scalars + width
        )
        managed_horde_consumer_nbytes = 4 * horde_input_groups * width
        return PrototypeFeatureLifecycleResourceBudget(
            mechanism_status=PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            scientific_promotion_allowed=(
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            base_feature_slots=self._config.base_feature_dim,
            pair_source_feature_slots=(
                self._config.effective_pair_source_feature_dim
            ),
            canonical_pair_universe_slots=(
                self._config.effective_pair_source_feature_dim
                * (self._config.effective_pair_source_feature_dim - 1)
                // 2
            ),
            active_pair_slots=self._config.active_pair_slots,
            candidate_pair_slots=self._config.candidate_pair_slots,
            managed_oak_feature_width=width,
            learner_persistent_state_nbytes=_tree_nbytes(measured.learner_state),
            router_persistent_state_nbytes=_tree_nbytes(measured.router_state),
            lifecycle_telemetry_counter_nbytes=(
                4 * PROTOTYPE_FEATURE_LIFECYCLE_TELEMETRY_COUNTER_NBYTES
            ),
            lifecycle_exact_counter_nbytes=(
                4
                * (
                    PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES
                    - PROTOTYPE_FEATURE_LIFECYCLE_TELEMETRY_COUNTER_NBYTES
                )
            ),
            lifecycle_counter_delta_nbytes=(
                PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_DELTA_NBYTES
            ),
            lifecycle_counter_nbytes=PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES,
            lifecycle_state_nbytes=lifecycle_state_nbytes,
            consumer_binding_persistent_nbytes=consumer_binding_nbytes,
            consumer_binding_generation_nbytes=(
                PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_NBYTES
            ),
            consumer_binding_generation_delta_nbytes=(
                PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_DELTA_NBYTES
            ),
            internal_learner_template_nbytes=internal_learner_template_nbytes,
            internal_oak_template_nbytes=internal_oak_template_nbytes,
            internal_template_nbytes=internal_template_nbytes,
            owned_persistent_state_nbytes=(
                lifecycle_state_nbytes + internal_template_nbytes
            ),
            managed_oak_consumer_nbytes=managed_oak_consumer_nbytes,
            rebuilt_base_cache_nbytes=4 * width,
            input_route_feature_groups=input_groups,
            output_route_feature_groups=output_groups,
            # Both pure route candidates execute on every observe so the
            # method remains one fixed JIT/scan program.  Their state is
            # adopted only at a committed curation boundary.
            router_calls_per_observe=2,
            router_calls_per_committed_curation=2,
            # Current implementation evaluates the active bank for the old
            # cache audit, learner update, provisional committed cache,
            # candidate postcondition cache, and final returned cache.
            max_active_pair_products_per_observe=(
                5 * self._config.active_pair_slots
            ),
            max_candidate_pair_products_per_observe=(
                self._config.candidate_pair_slots
            ),
            max_observations=self._config.max_observations,
            managed_horde_demons=self._config.managed_horde_demons,
            horde_persistent_state_nbytes=(
                _tree_nbytes(measured_horde)
                if measured_horde is not None
                else 0
            ),
            managed_horde_consumer_nbytes=managed_horde_consumer_nbytes,
            managed_total_consumer_nbytes=(
                managed_oak_consumer_nbytes + managed_horde_consumer_nbytes
            ),
            internal_horde_template_nbytes=internal_horde_template_nbytes,
        )

    def unavailable_diagnostics(
        self,
        semantic_generation: Array,
    ) -> PrototypeFeatureLifecycleDiagnostics:
        """Return finite neutral diagnostics for an outer rejected branch.

        This constructor performs no lifecycle, OaK, route, or postcondition
        audit.  It exists so an integrating JAX ``lax.cond`` can keep a fixed
        diagnostics PyTree when a prerequisite outside this boundary rejects
        the call.
        """

        generation = _require_array(
            semantic_generation,
            name="semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        false = jnp.asarray(False, dtype=jnp.bool_)
        return PrototypeFeatureLifecycleDiagnostics(
            available=false,
            state_values_valid=false,
            oak_values_valid=false,
            consumer_binding_valid=false,
            event_values_valid=false,
            next_observation_matches_oak_cache=false,
            update_capacity_available=false,
            post_update_consumer_clock_valid=false,
            learner_update_rejected=false,
            transaction_applied=false,
            curation_priority_override_enabled=false,
            curation_priority_override_applied=false,
            curation_selected_active_worst_slot=jnp.asarray(
                -1,
                dtype=jnp.int32,
            ),
            curation_selected_promotion_candidate=jnp.asarray(
                -1,
                dtype=jnp.int32,
            ),
            curation_selected_refresh_candidate=jnp.asarray(
                -1,
                dtype=jnp.int32,
            ),
            curation_proposed=false,
            safe_curation_boundary=false,
            curation_deferred=false,
            routing_attempted=false,
            input_route_valid=false,
            output_route_valid=false,
            route_states_match=false,
            routed_values_finite=false,
            curation_committed=false,
            curation_rolled_back=false,
            postcondition_checked=false,
            postcondition_valid=false,
            postcondition_rolled_back=false,
            semantic_generation_before=generation,
            semantic_generation_after=generation,
            semantic_generation_words_before=jnp.zeros((2,), dtype=jnp.uint32),
            semantic_generation_words_after=jnp.zeros((2,), dtype=jnp.uint32),
            observe_words_before=jnp.zeros((2,), dtype=jnp.uint32),
            observe_words_after=jnp.zeros((2,), dtype=jnp.uint32),
        )

    def _input_consumers(
        self,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState | None = None,
    ) -> dict[str, Any]:
        stomp = oak_state.stomp_state
        consumers: dict[str, Any] = {
            "base_head_weights": stomp.base_learner_state.head_params.weights,
            "base_head_weight_traces": tuple(
                trace_pair[0]
                for trace_pair in stomp.base_learner_state.head_traces
            ),
            "option_policy_weights": stomp.option_policies.q_weights,
            "option_policy_traces": stomp.option_policies.traces,
            "option_model_input_weights": stomp.option_models.next_state_weights,
            "option_start_observation": stomp.option_start_obs,
        }
        if horde_state is not None:
            consumers["horde_head_weights"] = horde_state.head_params.weights
            consumers["horde_head_weight_traces"] = tuple(
                trace_pair[0] for trace_pair in horde_state.head_traces
            )
        return consumers

    def _routed_oak_state(
        self,
        old: OaKState,
        routed_inputs: dict[str, Any],
        routed_model: Array,
        next_augmented_observation: Array,
        commit: Array,
    ) -> OaKState:
        stomp = old.stomp_state
        learner = stomp.base_learner_state
        candidate_head_weights = cast(
            tuple[Array, ...],
            routed_inputs["base_head_weights"],
        )
        candidate_weight_traces = cast(
            tuple[Array, ...],
            routed_inputs["base_head_weight_traces"],
        )
        head_weights = tuple(
            jnp.where(commit, candidate, original)
            for candidate, original in zip(
                candidate_head_weights,
                learner.head_params.weights,
                strict=True,
            )
        )
        head_traces = tuple(
            (
                jnp.where(commit, candidate_weight, original_pair[0]),
                original_pair[1],
            )
            for candidate_weight, original_pair in zip(
                candidate_weight_traces,
                learner.head_traces,
                strict=True,
            )
        )
        next_learner = learner.replace(
            head_params=learner.head_params.replace(weights=head_weights),
            head_traces=head_traces,
        )
        policies = stomp.option_policies.replace(
            q_weights=jnp.where(
                commit,
                routed_inputs["option_policy_weights"],
                stomp.option_policies.q_weights,
            ),
            traces=jnp.where(
                commit,
                routed_inputs["option_policy_traces"],
                stomp.option_policies.traces,
            ),
        )
        models = stomp.option_models.replace(
            next_state_weights=jnp.where(
                commit,
                routed_model,
                stomp.option_models.next_state_weights,
            )
        )
        next_stomp = stomp.replace(
            base_learner_state=next_learner,
            base_last_obs=jnp.where(
                commit,
                next_augmented_observation,
                stomp.base_last_obs,
            ),
            option_policies=policies,
            option_models=models,
            option_start_obs=jnp.where(
                commit,
                routed_inputs["option_start_observation"],
                stomp.option_start_obs,
            ),
        )
        return cast(OaKState, old.replace(stomp_state=next_stomp))

    def _routed_horde_state(
        self,
        old: MultiHeadMLPState,
        routed_inputs: dict[str, Any],
        commit: Array,
    ) -> MultiHeadMLPState:
        """Adopt one atomic descriptor route on every linear Horde axis."""

        candidate_weights = cast(
            tuple[Array, ...],
            routed_inputs["horde_head_weights"],
        )
        candidate_weight_traces = cast(
            tuple[Array, ...],
            routed_inputs["horde_head_weight_traces"],
        )
        head_weights = tuple(
            jnp.where(commit, candidate, original)
            for candidate, original in zip(
                candidate_weights,
                old.head_params.weights,
                strict=True,
            )
        )
        head_traces = tuple(
            (
                jnp.where(commit, candidate, original_pair[0]),
                original_pair[1],
            )
            for candidate, original_pair in zip(
                candidate_weight_traces,
                old.head_traces,
                strict=True,
            )
        )
        return cast(
            MultiHeadMLPState,
            old.replace(
                head_params=old.head_params.replace(weights=head_weights),
                head_traces=head_traces,
            ),
        )

    def observe_and_route(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        curation_priority_override: (
            InteractionCurationPriorityOverride | None
        ) = None,
    ) -> PrototypeFeatureLifecycleResult:
        """Learn once, then defer or atomically route a descriptor mutation.

        Static mismatches raise before indexed work.  Dynamic invalidity is an
        exact no-op for both owned state and the supplied OaK state.  A route
        failure preserves ordinary learning via ``pre_curation_state`` and
        rolls back only the descriptor mutation.
        """

        if self._config.managed_horde_demons != 0:
            raise ValueError(
                "managed Horde lifecycle requires observe_and_route_with_horde"
            )
        prepared = self.prepare_observe_and_route(
            state,
            oak_state,
            consumer_binding,
            event,
            curation_priority_override=curation_priority_override,
        )
        return prepared.destination_result

    def prepare_observe_and_route(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        curation_priority_override: (
            InteractionCurationPriorityOverride | None
        ) = None,
    ) -> PrototypeFeatureLifecyclePreparedRoute:
        """Compute one ordinary successor and one routed candidate exactly once."""

        if self._config.managed_horde_demons != 0:
            raise ValueError(
                "managed Horde lifecycle requires prepare_observe_and_route_with_horde"
            )
        return cast(
            PrototypeFeatureLifecyclePreparedRoute,
            self._prepare_observe_and_route_impl(
                state,
                oak_state,
                consumer_binding,
                event,
                horde_state=None,
                curation_priority_override=curation_priority_override,
            ),
        )

    def observe_and_route_with_horde(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        curation_priority_override: (
            InteractionCurationPriorityOverride | None
        ) = None,
    ) -> PrototypeFeatureLifecycleHordeResult:
        """Route post-update OaK and linear-Horde consumers atomically.

        The caller must first update both consumers under the old feature bank.
        Their exact step words must equal the lifecycle observation identity
        plus one.  At a caller-configured budget below the exact terminal,
        equal OaK and Horde clocks may continue advancing while this method
        remains an audited no-op.  At the all-ones terminal every aligned
        component fails closed because no exact successor exists.
        """

        if self._config.managed_horde_demons == 0:
            raise ValueError(
                "legacy lifecycle does not manage a Horde consumer"
            )
        if not self._horde_static_contract_valid(horde_state):
            raise ValueError(
                "horde_state must satisfy the exact supported linear Horde "
                "static contract"
            )
        prepared = self.prepare_observe_and_route_with_horde(
            state,
            oak_state,
            horde_state,
            consumer_binding,
            event,
            curation_priority_override=curation_priority_override,
        )
        return prepared.destination_result

    def prepare_observe_and_route_with_horde(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        curation_priority_override: (
            InteractionCurationPriorityOverride | None
        ) = None,
    ) -> PrototypeFeatureLifecyclePreparedHordeRoute:
        """Compute Horde ordinary/routed candidates from one learner update."""

        if self._config.managed_horde_demons == 0:
            raise ValueError("legacy lifecycle does not manage a Horde consumer")
        if not self._horde_static_contract_valid(horde_state):
            raise ValueError(
                "horde_state must satisfy the exact supported linear Horde "
                "static contract"
            )
        return cast(
            PrototypeFeatureLifecyclePreparedHordeRoute,
            self._prepare_observe_and_route_impl(
                state,
                oak_state,
                consumer_binding,
                event,
                horde_state=horde_state,
                curation_priority_override=curation_priority_override,
            ),
        )

    def _validate_external_event_static_contract(
        self,
        event: PrototypeFeatureLifecycleEvent,
    ) -> None:
        if type(event) is not PrototypeFeatureLifecycleEvent:
            raise TypeError("prepared event must be a PrototypeFeatureLifecycleEvent")
        _require_array(
            event.observation,
            name="prepared.event.observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            event.targets,
            name="prepared.event.targets",
            shape=(self._config.n_tasks,),
            dtype=jnp.float32,
        )
        _require_array(
            event.next_observation,
            name="prepared.event.next_observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        _require_array(
            event.allow_curation,
            name="prepared.event.allow_curation",
            shape=(),
            dtype=jnp.bool_,
        )

    def _validate_external_override_static_contract(
        self,
        override: InteractionCurationPriorityOverride | None,
    ) -> None:
        if override is None:
            return
        if type(override) is not InteractionCurationPriorityOverride:
            raise TypeError("prepared curation priority override has the wrong type")
        _require_array(
            override.enabled,
            name="prepared.curation_priority_override.enabled",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            override.active_ranks,
            name="prepared.curation_priority_override.active_ranks",
            shape=(self._config.active_pair_slots,),
            dtype=jnp.float32,
        )
        _require_array(
            override.candidate_ranks,
            name="prepared.curation_priority_override.candidate_ranks",
            shape=(self._config.candidate_pair_slots,),
            dtype=jnp.float32,
        )

    def _validate_prepared_route_static_contract(
        self,
        prepared: PrototypeFeatureLifecyclePreparedRoute,
    ) -> None:
        if type(prepared) is not PrototypeFeatureLifecyclePreparedRoute:
            raise TypeError("prepared must be a PrototypeFeatureLifecyclePreparedRoute")
        if not self._state_static_contract_valid(prepared.source_state):
            raise ValueError("prepared source lifecycle static contract differs")
        if not self._oak_static_contract_valid(prepared.source_oak_state):
            raise ValueError("prepared source OaK static contract differs")
        if not self._consumer_binding_static_contract_valid(
            prepared.source_consumer_binding
        ):
            raise ValueError("prepared source binding static contract differs")
        self._validate_external_event_static_contract(prepared.event)
        self._validate_external_override_static_contract(
            prepared.curation_priority_override
        )
        if type(prepared.ordinary_result) is not PrototypeFeatureLifecycleResult:
            raise TypeError("prepared ordinary result has the wrong type")
        if type(prepared.destination_result) is not PrototypeFeatureLifecycleResult:
            raise TypeError("prepared destination result has the wrong type")
        _require_array(
            prepared.internally_valid,
            name="prepared.internally_valid",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            prepared.preparation_learner_update_evaluations,
            name="prepared.preparation_learner_update_evaluations",
            shape=(),
            dtype=jnp.int32,
        )

    def _validate_prepared_horde_route_static_contract(
        self,
        prepared: PrototypeFeatureLifecyclePreparedHordeRoute,
    ) -> None:
        if type(prepared) is not PrototypeFeatureLifecyclePreparedHordeRoute:
            raise TypeError(
                "prepared must be a PrototypeFeatureLifecyclePreparedHordeRoute"
            )
        if not self._state_static_contract_valid(prepared.source_state):
            raise ValueError("prepared source lifecycle static contract differs")
        if not self._oak_static_contract_valid(prepared.source_oak_state):
            raise ValueError("prepared source OaK static contract differs")
        if not self._horde_static_contract_valid(prepared.source_horde_state):
            raise ValueError("prepared source Horde static contract differs")
        if not self._consumer_binding_static_contract_valid(
            prepared.source_consumer_binding
        ):
            raise ValueError("prepared source binding static contract differs")
        self._validate_external_event_static_contract(prepared.event)
        self._validate_external_override_static_contract(
            prepared.curation_priority_override
        )
        if type(prepared.ordinary_result) is not PrototypeFeatureLifecycleHordeResult:
            raise TypeError("prepared ordinary Horde result has the wrong type")
        if type(prepared.destination_result) is not PrototypeFeatureLifecycleHordeResult:
            raise TypeError("prepared destination Horde result has the wrong type")
        _require_array(
            prepared.internally_valid,
            name="prepared.internally_valid",
            shape=(),
            dtype=jnp.bool_,
        )
        _require_array(
            prepared.preparation_learner_update_evaluations,
            name="prepared.preparation_learner_update_evaluations",
            shape=(),
            dtype=jnp.int32,
        )

    def external_readiness_receipt(
        self,
        prepared: PrototypeFeatureLifecyclePreparedRoute,
        all_consumers_ready: Array,
    ) -> PrototypeFeatureLifecycleExternalReadinessReceipt:
        """Bind one external all-consumer verdict to every preparation leaf."""

        self._validate_prepared_route_static_contract(prepared)
        ready = _require_array(
            all_consumers_ready,
            name="all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )
        return PrototypeFeatureLifecycleExternalReadinessReceipt(
            prepared_route=prepared,
            all_consumers_ready=ready,
        )

    def horde_external_readiness_receipt(
        self,
        prepared: PrototypeFeatureLifecyclePreparedHordeRoute,
        all_consumers_ready: Array,
    ) -> PrototypeFeatureLifecycleHordeExternalReadinessReceipt:
        """Bind one external verdict to every Horde preparation leaf."""

        self._validate_prepared_horde_route_static_contract(prepared)
        ready = _require_array(
            all_consumers_ready,
            name="all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )
        return PrototypeFeatureLifecycleHordeExternalReadinessReceipt(
            prepared_route=prepared,
            all_consumers_ready=ready,
        )

    def external_transaction_resource_budget(
        self,
        prepared: (
            PrototypeFeatureLifecyclePreparedRoute
            | PrototypeFeatureLifecyclePreparedHordeRoute
        ),
        receipt: (
            PrototypeFeatureLifecycleExternalReadinessReceipt
            | PrototypeFeatureLifecycleHordeExternalReadinessReceipt
        ),
    ) -> PrototypeFeatureLifecycleExternalTransactionResourceBudget:
        """Measure serialized logical leaves without changing checkpoint bytes."""

        if type(prepared) is PrototypeFeatureLifecyclePreparedRoute:
            if type(receipt) is not PrototypeFeatureLifecycleExternalReadinessReceipt:
                raise TypeError("legacy preparation requires a legacy readiness receipt")
            self._validate_prepared_route_static_contract(prepared)
            self._validate_prepared_route_static_contract(receipt.prepared_route)
            source_horde_nbytes = 0
            managed_horde_demons = 0
        elif type(prepared) is PrototypeFeatureLifecyclePreparedHordeRoute:
            if type(receipt) is not PrototypeFeatureLifecycleHordeExternalReadinessReceipt:
                raise TypeError("Horde preparation requires a Horde readiness receipt")
            self._validate_prepared_horde_route_static_contract(prepared)
            self._validate_prepared_horde_route_static_contract(receipt.prepared_route)
            source_horde_nbytes = _tree_nbytes(prepared.source_horde_state)
            managed_horde_demons = self._config.managed_horde_demons
        else:
            raise TypeError("prepared has the wrong exact lifecycle preparation type")
        prepared_nbytes = _tree_nbytes(prepared)
        receipt_nbytes = _tree_nbytes(receipt)
        lifecycle_nbytes = _tree_nbytes(prepared.source_state)
        return PrototypeFeatureLifecycleExternalTransactionResourceBudget(
            managed_horde_demons=managed_horde_demons,
            lifecycle_persistent_state_nbytes_before=lifecycle_nbytes,
            lifecycle_persistent_state_nbytes_after=lifecycle_nbytes,
            source_oak_state_nbytes=_tree_nbytes(prepared.source_oak_state),
            source_horde_state_nbytes=source_horde_nbytes,
            source_consumer_binding_nbytes=_tree_nbytes(
                prepared.source_consumer_binding
            ),
            prepared_route_logical_nbytes=prepared_nbytes,
            readiness_receipt_logical_nbytes=receipt_nbytes,
            simultaneous_logical_transient_nbytes=prepared_nbytes + receipt_nbytes,
            learner_update_evaluations_per_prepare=1,
            learner_update_evaluations_per_adopt=0,
            learner_update_evaluations_per_transaction=1,
            router_evaluations_per_prepare=2,
            router_evaluations_per_adopt=0,
            router_evaluations_per_transaction=2,
            persistent_capacity_growth=0,
        )

    def adopt_prepared_route(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        prepared: PrototypeFeatureLifecyclePreparedRoute,
        receipt: PrototypeFeatureLifecycleExternalReadinessReceipt,
    ) -> PrototypeFeatureLifecycleAdoptionResult:
        """Adopt a bound destination or retain its exact ordinary successor.

        Exact matching provides unkeyed integrity, not caller authentication.
        A party able to coordinate changes to both preparation and receipt is
        outside this primitive's trust boundary; adoption never rederives the
        learner result.
        """

        if self._config.managed_horde_demons != 0:
            raise ValueError("managed Horde lifecycle requires Horde adoption")
        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        if not self._oak_static_contract_valid(oak_state):
            raise ValueError("oak_state static contract differs")
        if not self._consumer_binding_static_contract_valid(consumer_binding):
            raise ValueError("consumer_binding static contract differs")
        self._validate_prepared_route_static_contract(prepared)
        if type(receipt) is not PrototypeFeatureLifecycleExternalReadinessReceipt:
            raise TypeError(
                "receipt must be a PrototypeFeatureLifecycleExternalReadinessReceipt"
            )
        self._validate_prepared_route_static_contract(receipt.prepared_route)
        _require_array(
            receipt.all_consumers_ready,
            name="receipt.all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )
        return self._adopt_prepared_route_impl(
            state,
            oak_state,
            consumer_binding,
            prepared,
            receipt,
        )

    def _adopt_prepared_route_impl(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        prepared: PrototypeFeatureLifecyclePreparedRoute,
        receipt: PrototypeFeatureLifecycleExternalReadinessReceipt,
    ) -> PrototypeFeatureLifecycleAdoptionResult:
        source_state_matches = _trees_bit_exactly_equal(state, prepared.source_state)
        source_oak_matches = _trees_bit_exactly_equal(
            oak_state,
            prepared.source_oak_state,
        )
        source_binding_matches = _trees_bit_exactly_equal(
            consumer_binding,
            prepared.source_consumer_binding,
        )
        receipt_matches = _trees_bit_exactly_equal(
            prepared,
            receipt.prepared_route,
        )
        exact_work = prepared.preparation_learner_update_evaluations == jnp.int32(1)
        preparation_valid = prepared.internally_valid & exact_work
        authenticated = (
            source_state_matches
            & source_oak_matches
            & source_binding_matches
            & receipt_matches
            & exact_work
        )
        applied = authenticated & preparation_valid
        selected = jax.lax.cond(
            receipt.all_consumers_ready,
            lambda _: prepared.destination_result,
            lambda _: prepared.ordinary_result,
            operand=None,
        )
        nested_applied = applied & selected.diagnostics.transaction_applied
        capacity_capped_noop, capacity_next_augmented = (
            self._external_capacity_capped_noop_encoding(
                state,
                oak_state,
                None,
                consumer_binding,
                prepared.event,
                selected.diagnostics,
                source_authenticated=authenticated,
            )
        )
        selected_state = jax.lax.cond(
            nested_applied,
            lambda _: selected.state,
            lambda _: state,
            operand=None,
        )
        selected_oak = jax.lax.cond(
            nested_applied,
            lambda _: selected.oak_state,
            lambda _: oak_state,
            operand=None,
        )
        selected_binding = jax.lax.cond(
            nested_applied,
            lambda _: selected.consumer_binding,
            lambda _: consumer_binding,
            operand=None,
        )
        result = selected.replace(
            state=selected_state,
            oak_state=selected_oak,
            consumer_binding=selected_binding,
            next_augmented_observation=jnp.where(
                nested_applied,
                selected.next_augmented_observation,
                jnp.where(
                    capacity_capped_noop,
                    capacity_next_augmented,
                    jnp.zeros_like(selected.next_augmented_observation),
                ),
            ),
            predictions=jnp.where(
                nested_applied,
                selected.predictions,
                jnp.full_like(selected.predictions, jnp.nan),
            ),
            errors=jnp.where(
                nested_applied,
                selected.errors,
                jnp.full_like(selected.errors, jnp.nan),
            ),
            metrics=jnp.where(
                nested_applied,
                selected.metrics,
                jnp.zeros_like(selected.metrics),
            ),
            diagnostics=selected.diagnostics.replace(
                transaction_applied=nested_applied,
                curation_committed=(
                    nested_applied & selected.diagnostics.curation_committed
                ),
                curation_rolled_back=(
                    nested_applied & selected.diagnostics.curation_rolled_back
                ),
                postcondition_valid=(
                    nested_applied & selected.diagnostics.postcondition_valid
                ),
                postcondition_rolled_back=(
                    nested_applied & selected.diagnostics.postcondition_rolled_back
                ),
                semantic_generation_after=(
                    selected_state.router_state.generation_count
                ),
                semantic_generation_words_after=(
                    selected_state.router_state.generation_words
                ),
                observe_words_after=selected_state.observe_words,
            ),
        )
        external_rollback = (
            nested_applied
            & ~receipt.all_consumers_ready
            & prepared.destination_result.diagnostics.curation_committed
        )
        zero_work = jnp.asarray(0, dtype=jnp.int32)
        return PrototypeFeatureLifecycleAdoptionResult(
            result=cast(PrototypeFeatureLifecycleResult, result),
            diagnostics=PrototypeFeatureLifecycleAdoptionDiagnostics(
                source_state_matches=source_state_matches,
                source_oak_state_matches=source_oak_matches,
                source_horde_state_matches=jnp.asarray(True, dtype=jnp.bool_),
                source_consumer_binding_matches=source_binding_matches,
                receipt_matches_preparation=receipt_matches,
                preparation_internally_valid=preparation_valid,
                all_consumers_ready=receipt.all_consumers_ready,
                destination_adopted=nested_applied & receipt.all_consumers_ready,
                ordinary_update_retained=nested_applied & ~receipt.all_consumers_ready,
                external_curation_rolled_back=external_rollback,
                transaction_applied=nested_applied,
                rejected=~nested_applied,
                preparation_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
                adoption_learner_update_evaluations=zero_work,
                total_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
            ),
        )

    def _external_capacity_capped_noop_encoding(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState | None,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        diagnostics: PrototypeFeatureLifecycleDiagnostics,
        *,
        source_authenticated: Array,
    ) -> tuple[Bool[Array, ""], Float[Array, " features"]]:
        """Derive the one safe output retained by an authenticated cap no-op."""

        safe_next_observation = jnp.where(
            jnp.isfinite(event.next_observation),
            event.next_observation,
            jnp.zeros_like(event.next_observation),
        )
        derived_next_augmented = self._augment_base_observation(
            state.learner_state,
            safe_next_observation,
        )
        expected_post_observe_words, exact_capacity_available = (
            _checked_lifetime_words_increment(state.observe_words)
        )
        configured_capacity_available = _lifetime_words_le(
            expected_post_observe_words,
            _python_uint64_words(self._config.max_observations),
        )
        configured_capacity_capped = (
            exact_capacity_available & ~configured_capacity_available
        )
        consumer_clocks_valid = (
            jnp.all(oak_state.step_words == oak_state.stomp_state.step_words)
            & _lifetime_words_le(state.observe_words, oak_state.step_words)
            & jnp.any(state.observe_words != oak_state.step_words)
        )
        horde_valid = jnp.asarray(True, dtype=jnp.bool_)
        if horde_state is not None:
            horde_valid = self.horde_state_valid(horde_state) & jnp.all(
                horde_state.step_words == oak_state.step_words
            )
        event_valid = (
            jnp.all(jnp.isfinite(event.observation))
            & jnp.all(jnp.isfinite(event.next_observation))
            & jnp.all(jnp.isfinite(event.targets) | jnp.isnan(event.targets))
        )
        no_route_leak = (
            ~diagnostics.transaction_applied
            & ~diagnostics.curation_proposed
            & ~diagnostics.curation_deferred
            & ~diagnostics.routing_attempted
            & ~diagnostics.curation_committed
            & ~diagnostics.curation_rolled_back
        )
        valid = (
            source_authenticated
            & configured_capacity_capped
            & self.state_valid(state)
            & self._oak_values_valid(oak_state)
            & horde_valid
            & self.consumer_binding_valid(state, consumer_binding)
            & event_valid
            & consumer_clocks_valid
            & jnp.all(jnp.isfinite(derived_next_augmented))
            & _float32_arrays_bit_exact(
                oak_state.stomp_state.base_last_obs,
                derived_next_augmented,
            )
            & diagnostics.available
            & diagnostics.state_values_valid
            & diagnostics.oak_values_valid
            & diagnostics.consumer_binding_valid
            & diagnostics.event_values_valid
            & diagnostics.next_observation_matches_oak_cache
            & ~diagnostics.update_capacity_available
            & diagnostics.post_update_consumer_clock_valid
            & diagnostics.learner_update_rejected
            & no_route_leak
        )
        return valid, derived_next_augmented

    def adopt_prepared_route_with_horde(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        prepared: PrototypeFeatureLifecyclePreparedHordeRoute,
        receipt: PrototypeFeatureLifecycleHordeExternalReadinessReceipt,
    ) -> PrototypeFeatureLifecycleHordeAdoptionResult:
        """Adopt or externally roll back a source-bound Horde route.

        The receipt is an unkeyed exact-content binding issued by a trusted
        external coordinator; coordinated preparation/receipt forgery is not
        caller-authenticated here and the learner is deliberately not rerun.
        """

        if self._config.managed_horde_demons == 0:
            raise ValueError("legacy lifecycle does not manage a Horde consumer")
        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        if not self._oak_static_contract_valid(oak_state):
            raise ValueError("oak_state static contract differs")
        if not self._horde_static_contract_valid(horde_state):
            raise ValueError("horde_state static contract differs")
        if not self._consumer_binding_static_contract_valid(consumer_binding):
            raise ValueError("consumer_binding static contract differs")
        self._validate_prepared_horde_route_static_contract(prepared)
        if type(receipt) is not PrototypeFeatureLifecycleHordeExternalReadinessReceipt:
            raise TypeError(
                "receipt must be a PrototypeFeatureLifecycleHordeExternalReadinessReceipt"
            )
        self._validate_prepared_horde_route_static_contract(receipt.prepared_route)
        _require_array(
            receipt.all_consumers_ready,
            name="receipt.all_consumers_ready",
            shape=(),
            dtype=jnp.bool_,
        )

        source_state_matches = _trees_bit_exactly_equal(state, prepared.source_state)
        source_oak_matches = _trees_bit_exactly_equal(
            oak_state,
            prepared.source_oak_state,
        )
        source_horde_matches = _trees_bit_exactly_equal(
            horde_state,
            prepared.source_horde_state,
        )
        source_binding_matches = _trees_bit_exactly_equal(
            consumer_binding,
            prepared.source_consumer_binding,
        )
        receipt_matches = _trees_bit_exactly_equal(
            prepared,
            receipt.prepared_route,
        )
        exact_work = prepared.preparation_learner_update_evaluations == jnp.int32(1)
        preparation_valid = prepared.internally_valid & exact_work
        authenticated = (
            source_state_matches
            & source_oak_matches
            & source_horde_matches
            & source_binding_matches
            & receipt_matches
            & exact_work
        )
        applied = authenticated & preparation_valid
        selected = jax.lax.cond(
            receipt.all_consumers_ready,
            lambda _: prepared.destination_result,
            lambda _: prepared.ordinary_result,
            operand=None,
        )
        nested_applied = applied & selected.diagnostics.transaction_applied
        capacity_capped_noop, capacity_next_augmented = (
            self._external_capacity_capped_noop_encoding(
                state,
                oak_state,
                horde_state,
                consumer_binding,
                prepared.event,
                selected.diagnostics,
                source_authenticated=authenticated,
            )
        )
        capacity_capped_noop = (
            capacity_capped_noop
            & selected.horde_diagnostics.horde_state_values_valid
            & selected.horde_diagnostics.pre_step_parity_valid
            & selected.horde_diagnostics.post_step_parity_valid
            & selected.horde_diagnostics.lifecycle_capacity_capped
        )
        selected_state = jax.lax.cond(
            nested_applied,
            lambda _: selected.state,
            lambda _: state,
            operand=None,
        )
        selected_oak = jax.lax.cond(
            nested_applied,
            lambda _: selected.oak_state,
            lambda _: oak_state,
            operand=None,
        )
        selected_horde = jax.lax.cond(
            nested_applied,
            lambda _: selected.horde_state,
            lambda _: horde_state,
            operand=None,
        )
        selected_binding = jax.lax.cond(
            nested_applied,
            lambda _: selected.consumer_binding,
            lambda _: consumer_binding,
            operand=None,
        )
        result = selected.replace(
            state=selected_state,
            oak_state=selected_oak,
            horde_state=selected_horde,
            consumer_binding=selected_binding,
            next_augmented_observation=jnp.where(
                nested_applied,
                selected.next_augmented_observation,
                jnp.where(
                    capacity_capped_noop,
                    capacity_next_augmented,
                    jnp.zeros_like(selected.next_augmented_observation),
                ),
            ),
            predictions=jnp.where(
                nested_applied,
                selected.predictions,
                jnp.full_like(selected.predictions, jnp.nan),
            ),
            errors=jnp.where(
                nested_applied,
                selected.errors,
                jnp.full_like(selected.errors, jnp.nan),
            ),
            metrics=jnp.where(
                nested_applied,
                selected.metrics,
                jnp.zeros_like(selected.metrics),
            ),
            diagnostics=selected.diagnostics.replace(
                transaction_applied=nested_applied,
                curation_committed=(
                    nested_applied & selected.diagnostics.curation_committed
                ),
                curation_rolled_back=(
                    nested_applied & selected.diagnostics.curation_rolled_back
                ),
                postcondition_valid=(
                    nested_applied & selected.diagnostics.postcondition_valid
                ),
                postcondition_rolled_back=(
                    nested_applied & selected.diagnostics.postcondition_rolled_back
                ),
                semantic_generation_after=(
                    selected_state.router_state.generation_count
                ),
                semantic_generation_words_after=(
                    selected_state.router_state.generation_words
                ),
                observe_words_after=selected_state.observe_words,
            ),
            horde_diagnostics=selected.horde_diagnostics.replace(
                post_step_parity_valid=(
                    nested_applied
                    & selected.horde_diagnostics.post_step_parity_valid
                ),
            ),
        )
        external_rollback = (
            nested_applied
            & ~receipt.all_consumers_ready
            & prepared.destination_result.diagnostics.curation_committed
        )
        zero_work = jnp.asarray(0, dtype=jnp.int32)
        return PrototypeFeatureLifecycleHordeAdoptionResult(
            result=cast(PrototypeFeatureLifecycleHordeResult, result),
            diagnostics=PrototypeFeatureLifecycleAdoptionDiagnostics(
                source_state_matches=source_state_matches,
                source_oak_state_matches=source_oak_matches,
                source_horde_state_matches=source_horde_matches,
                source_consumer_binding_matches=source_binding_matches,
                receipt_matches_preparation=receipt_matches,
                preparation_internally_valid=preparation_valid,
                all_consumers_ready=receipt.all_consumers_ready,
                destination_adopted=nested_applied & receipt.all_consumers_ready,
                ordinary_update_retained=nested_applied & ~receipt.all_consumers_ready,
                external_curation_rolled_back=external_rollback,
                transaction_applied=nested_applied,
                rejected=~nested_applied,
                preparation_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
                adoption_learner_update_evaluations=zero_work,
                total_learner_update_evaluations=(
                    prepared.preparation_learner_update_evaluations
                ),
            ),
        )

    def _prepare_observe_and_route_impl(
        self,
        state: PrototypeFeatureLifecycleState,
        oak_state: OaKState,
        consumer_binding: PrototypeFeatureConsumerBinding,
        event: PrototypeFeatureLifecycleEvent,
        *,
        horde_state: MultiHeadMLPState | None,
        curation_priority_override: InteractionCurationPriorityOverride | None,
    ) -> PrototypeFeatureLifecyclePreparedRoute | PrototypeFeatureLifecyclePreparedHordeRoute:
        """Shared fixed-shape preparation for legacy and Horde transactions."""

        if not self._state_static_contract_valid(state):
            raise ValueError("prototype feature lifecycle state has an invalid static contract")
        if type(consumer_binding) is not PrototypeFeatureConsumerBinding:
            raise TypeError(
                "consumer_binding must be a PrototypeFeatureConsumerBinding"
            )
        _require_array(
            consumer_binding.semantic_generation,
            name="consumer_binding.semantic_generation",
            shape=(),
            dtype=jnp.int32,
        )
        _require_array(
            consumer_binding.semantic_generation_words,
            name="consumer_binding.semantic_generation_words",
            shape=(2,),
            dtype=jnp.uint32,
        )
        _require_array(
            consumer_binding.descriptors,
            name="consumer_binding.descriptors",
            shape=(self._config.active_pair_slots, 2),
            dtype=jnp.int32,
        )
        if type(event) is not PrototypeFeatureLifecycleEvent:
            raise TypeError("event must be a PrototypeFeatureLifecycleEvent")
        observation = _require_array(
            event.observation,
            name="event.observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        targets = _require_array(
            event.targets,
            name="event.targets",
            shape=(self._config.n_tasks,),
            dtype=jnp.float32,
        )
        next_observation = _require_array(
            event.next_observation,
            name="event.next_observation",
            shape=(self._config.base_feature_dim,),
            dtype=jnp.float32,
        )
        allow_curation = _require_array(
            event.allow_curation,
            name="event.allow_curation",
            shape=(),
            dtype=jnp.bool_,
        )
        if not self._oak_static_contract_valid(oak_state):
            raise ValueError(
                "oak_state must satisfy the exact supported linear OaK static contract"
            )

        state_values_valid = self.state_valid(state)
        oak_values_valid = self._oak_values_valid(oak_state)
        horde_state_values_valid = (
            self.horde_state_valid(horde_state)
            if horde_state is not None
            else jnp.asarray(True, dtype=jnp.bool_)
        )
        consumer_binding_values_valid = self.consumer_binding_valid(
            state,
            consumer_binding,
        )
        event_values_valid = (
            jnp.all(jnp.isfinite(observation))
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(targets) | jnp.isnan(targets))
        )
        old_next_augmented = self.augment(state, next_observation)
        next_observation_matches_oak_cache = _float32_arrays_bit_exact(
            oak_state.stomp_state.base_last_obs,
            old_next_augmented,
        )
        expected_post_observe_words, exact_lifetime_capacity_available = (
            _checked_lifetime_words_increment(state.observe_words)
        )
        configured_capacity_available = _lifetime_words_le(
            expected_post_observe_words,
            _python_uint64_words(self._config.max_observations),
        )
        update_capacity_available = (
            exact_lifetime_capacity_available & configured_capacity_available
        )
        lifecycle_capacity_capped = ~update_capacity_available
        consumer_clocks_equal = (
            jnp.all(oak_state.step_words == oak_state.stomp_state.step_words)
        )
        consumer_clock_after_lifecycle = (
            _lifetime_words_le(state.observe_words, oak_state.step_words)
            & jnp.any(state.observe_words != oak_state.step_words)
        )
        configured_capacity_capped = (
            exact_lifetime_capacity_available
            & ~configured_capacity_available
        )
        expected_update_consumer_clock_valid = (
            jnp.all(oak_state.step_words == expected_post_observe_words)
            & jnp.all(
                oak_state.stomp_state.step_words
                == expected_post_observe_words
            )
        )
        post_update_consumer_clock_valid = jnp.where(
            update_capacity_available,
            expected_update_consumer_clock_valid,
            configured_capacity_capped
            & consumer_clocks_equal
            & consumer_clock_after_lifecycle,
        )
        pre_step_parity_valid = jnp.asarray(True, dtype=jnp.bool_)
        if horde_state is not None:
            pre_step_parity_valid = (
                post_update_consumer_clock_valid
                & jnp.all(horde_state.step_words == oak_state.step_words)
            )
        composition_valid = (
            state_values_valid
            & oak_values_valid
            & consumer_binding_values_valid
            & event_values_valid
            & next_observation_matches_oak_cache
            & update_capacity_available
            & post_update_consumer_clock_valid
        )
        if horde_state is not None:
            composition_valid = (
                composition_valid
                & horde_state_values_valid
                & pre_step_parity_valid
            )

        safe_observation = jnp.where(jnp.isfinite(observation), observation, 0.0)
        safe_targets = jnp.where(
            jnp.isfinite(targets) | jnp.isnan(targets),
            targets,
            jnp.nan,
        )
        learner_update = self._learner.update(
            state.learner_state,
            safe_observation[
                : self._config.effective_pair_source_feature_dim
            ],
            safe_targets,
            curation_priority_override=curation_priority_override,
        )
        deferred_words_incremented, deferred_capacity_available = (
            _checked_lifetime_words_increment(state.deferred_curation_words)
        )
        committed_words_incremented, committed_capacity_available = (
            _checked_lifetime_words_increment(state.committed_curation_words)
        )
        rolled_back_words_incremented, rolled_back_capacity_available = (
            _checked_lifetime_words_increment(state.rolled_back_curation_words)
        )
        outcome_capacity_available = (
            deferred_capacity_available
            & committed_capacity_available
            & rolled_back_capacity_available
        )
        learner_update_valid = (
            composition_valid
            & outcome_capacity_available
            & ~learner_update.update_rejected
            & jnp.all(
                learner_update.state.step_words
                == expected_post_observe_words
            )
            & jnp.all(
                learner_update.pre_curation_state.step_words
                == expected_post_observe_words
            )
        )
        proposed_descriptors = jnp.stack(
            (
                learner_update.state.feature_left,
                learner_update.state.feature_right,
            ),
            axis=1,
        )
        pre_curation_descriptors = jnp.stack(
            (
                learner_update.pre_curation_state.feature_left,
                learner_update.pre_curation_state.feature_right,
            ),
            axis=1,
        )
        curation_proposed = learner_update_valid & jnp.any(
            proposed_descriptors != pre_curation_descriptors
        )
        stomp = oak_state.stomp_state
        safe_curation_boundary = (
            allow_curation
            & (stomp.executing_option == -1)
            & (stomp.base_last_action < self._config.n_primitive_actions)
        )
        curation_deferred = (
            curation_proposed & ~safe_curation_boundary
        )
        routing_attempted = curation_proposed & safe_curation_boundary

        input_route = self._router.route(
            state.router_state,
            self._input_consumers(oak_state, horde_state),
            proposed_descriptors,
            carry_survivors=self._config.carry_survivors,
        )
        routed_inputs = cast(dict[str, Any], input_route.consumers)
        output_route = self._router.route(
            state.router_state,
            routed_inputs["option_model_input_weights"],
            proposed_descriptors,
            feature_axes=1,
            carry_survivors=self._config.carry_survivors,
        )
        route_states_match_raw = _trees_exactly_equal(
            input_route.state,
            output_route.state,
        )
        routed_values_finite_raw = (
            _floating_tree_is_finite(routed_inputs)
            & _floating_tree_is_finite(output_route.consumers)
        )
        route_valid = (
            input_route.diagnostics.valid
            & output_route.diagnostics.valid
            & route_states_match_raw
            & routed_values_finite_raw
            & input_route.diagnostics.descriptors_changed
            & output_route.diagnostics.descriptors_changed
        )
        provisional_curation_committed = routing_attempted & route_valid
        route_curation_rolled_back = routing_attempted & ~route_valid

        learned_state = jax.lax.cond(
            curation_proposed & ~provisional_curation_committed,
            lambda _: learner_update.pre_curation_state,
            lambda _: learner_update.state,
            operand=None,
        )
        next_learner_state = jax.lax.cond(
            learner_update_valid,
            lambda _: learned_state,
            lambda _: state.learner_state,
            operand=None,
        )
        next_observe_count = jnp.where(
            learner_update_valid,
            _saturating_int32_increment(state.observe_count),
            state.observe_count,
        )
        next_observe_words = jnp.where(
            learner_update_valid,
            expected_post_observe_words,
            state.observe_words,
        )
        next_deferred_count = jnp.where(
            curation_deferred,
            _saturating_int32_increment(state.deferred_curation_count),
            state.deferred_curation_count,
        )
        next_deferred_words = jnp.where(
            curation_deferred,
            deferred_words_incremented,
            state.deferred_curation_words,
        )
        next_committed_count = jnp.where(
            provisional_curation_committed,
            _saturating_int32_increment(state.committed_curation_count),
            state.committed_curation_count,
        )
        next_committed_words = jnp.where(
            provisional_curation_committed,
            committed_words_incremented,
            state.committed_curation_words,
        )
        next_rolled_back_count = jnp.where(
            route_curation_rolled_back,
            _saturating_int32_increment(state.rolled_back_curation_count),
            state.rolled_back_curation_count,
        )
        next_rolled_back_words = jnp.where(
            route_curation_rolled_back,
            rolled_back_words_incremented,
            state.rolled_back_curation_words,
        )
        candidate_router_state = input_route.state
        next_router_state = FeatureBankRouterState(
            descriptors=jnp.where(
                provisional_curation_committed,
                candidate_router_state.descriptors,
                state.router_state.descriptors,
            ),
            route_count=jnp.where(
                provisional_curation_committed,
                candidate_router_state.route_count,
                state.router_state.route_count,
            ),
            generation_count=jnp.where(
                provisional_curation_committed,
                candidate_router_state.generation_count,
                state.router_state.generation_count,
            ),
            route_words=jnp.where(
                provisional_curation_committed,
                candidate_router_state.route_words,
                state.router_state.route_words,
            ),
            generation_words=jnp.where(
                provisional_curation_committed,
                candidate_router_state.generation_words,
                state.router_state.generation_words,
            ),
        )
        candidate_state = PrototypeFeatureLifecycleState(
            learner_state=next_learner_state,
            router_state=next_router_state,
            observe_count=next_observe_count,
            observe_words=next_observe_words,
            deferred_curation_count=next_deferred_count,
            deferred_curation_words=next_deferred_words,
            committed_curation_count=next_committed_count,
            committed_curation_words=next_committed_words,
            rolled_back_curation_count=next_rolled_back_count,
            rolled_back_curation_words=next_rolled_back_words,
        )
        candidate_consumer_binding = PrototypeFeatureConsumerBinding(
            semantic_generation=next_router_state.generation_count,
            semantic_generation_words=next_router_state.generation_words,
            descriptors=next_router_state.descriptors,
        )

        safe_next_observation = jnp.where(
            jnp.isfinite(next_observation),
            next_observation,
            0.0,
        )
        committed_next_augmented = self._augment_base_observation(
            learner_update.state,
            safe_next_observation,
        )
        candidate_oak_state = self._routed_oak_state(
            oak_state,
            routed_inputs,
            cast(Array, output_route.consumers),
            committed_next_augmented,
            provisional_curation_committed,
        )
        candidate_horde_state = (
            self._routed_horde_state(
                horde_state,
                routed_inputs,
                provisional_curation_committed,
            )
            if horde_state is not None
            else None
        )
        candidate_next_augmented = self._augment_base_observation(
            next_learner_state,
            safe_next_observation,
        )
        candidate_state_valid = self.state_valid(candidate_state)
        candidate_oak_values_valid = self._oak_values_valid(candidate_oak_state)
        candidate_cache_valid = _float32_arrays_bit_exact(
            candidate_oak_state.stomp_state.base_last_obs,
            candidate_next_augmented,
        )
        candidate_consumer_clock_valid = (
            jnp.all(
                candidate_state.observe_words
                == candidate_oak_state.step_words
            )
            & jnp.all(
                candidate_state.observe_words
                == candidate_oak_state.stomp_state.step_words
            )
        )
        candidate_horde_values_valid = jnp.asarray(True, dtype=jnp.bool_)
        candidate_step_parity_valid = jnp.asarray(True, dtype=jnp.bool_)
        if candidate_horde_state is not None:
            candidate_horde_values_valid = self.horde_state_valid(
                candidate_horde_state
            )
            candidate_step_parity_valid = (
                jnp.all(
                    candidate_oak_state.step_words
                    == candidate_horde_state.step_words
                )
                & jnp.all(
                    candidate_state.observe_words
                    == candidate_oak_state.step_words
                )
            )
        candidate_postcondition_valid = (
            candidate_state_valid
            & candidate_oak_values_valid
            & self.consumer_binding_valid(
                candidate_state,
                candidate_consumer_binding,
            )
            & candidate_cache_valid
            & candidate_consumer_clock_valid
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(candidate_next_augmented))
        )
        if candidate_horde_state is not None:
            candidate_postcondition_valid = (
                candidate_postcondition_valid
                & candidate_horde_values_valid
                & candidate_step_parity_valid
            )
        postcondition_valid = learner_update_valid & candidate_postcondition_valid
        postcondition_rolled_back = learner_update_valid & ~candidate_postcondition_valid
        transaction_applied = learner_update_valid & candidate_postcondition_valid
        curation_committed = (
            provisional_curation_committed & candidate_postcondition_valid
        )
        curation_rolled_back = (
            route_curation_rolled_back
            | (postcondition_rolled_back & curation_proposed)
        )
        next_state = jax.lax.cond(
            postcondition_rolled_back,
            lambda _: state,
            lambda _: candidate_state,
            operand=None,
        )
        next_oak_state = self._routed_oak_state(
            oak_state,
            routed_inputs,
            cast(Array, output_route.consumers),
            committed_next_augmented,
            curation_committed,
        )
        next_horde_state = (
            self._routed_horde_state(
                horde_state,
                routed_inputs,
                curation_committed,
            )
            if horde_state is not None
            else None
        )
        next_consumer_binding = PrototypeFeatureConsumerBinding(
            semantic_generation=jnp.where(
                curation_committed,
                candidate_consumer_binding.semantic_generation,
                consumer_binding.semantic_generation,
            ),
            semantic_generation_words=jnp.where(
                curation_committed,
                candidate_consumer_binding.semantic_generation_words,
                consumer_binding.semantic_generation_words,
            ),
            descriptors=jnp.where(
                curation_committed,
                candidate_consumer_binding.descriptors,
                consumer_binding.descriptors,
            ),
        )
        final_next_augmented = self._augment_base_observation(
            next_state.learner_state,
            safe_next_observation,
        )
        final_next_valid = (
            self.state_valid(next_state)
            & jnp.all(jnp.isfinite(next_observation))
            & jnp.all(jnp.isfinite(final_next_augmented))
        )
        final_next_augmented = jnp.where(
            final_next_valid,
            final_next_augmented,
            jnp.zeros_like(final_next_augmented),
        )
        post_step_parity_valid = jnp.asarray(True, dtype=jnp.bool_)
        if next_horde_state is not None:
            post_step_parity_valid = (
                jnp.all(next_oak_state.step_words == next_horde_state.step_words)
                & jnp.where(
                    transaction_applied,
                    jnp.all(next_state.observe_words == next_oak_state.step_words),
                    post_update_consumer_clock_valid,
                )
            )

        diagnostics = PrototypeFeatureLifecycleDiagnostics(
            available=jnp.asarray(True, dtype=jnp.bool_),
            state_values_valid=state_values_valid,
            oak_values_valid=oak_values_valid,
            consumer_binding_valid=consumer_binding_values_valid,
            event_values_valid=event_values_valid,
            next_observation_matches_oak_cache=(
                next_observation_matches_oak_cache
            ),
            update_capacity_available=update_capacity_available,
            post_update_consumer_clock_valid=post_update_consumer_clock_valid,
            learner_update_rejected=~learner_update_valid,
            transaction_applied=transaction_applied,
            curation_priority_override_enabled=(
                learner_update.curation_priority_override_enabled
            ),
            curation_priority_override_applied=(
                learner_update_valid
                & learner_update.curation_priority_override_applied
            ),
            curation_selected_active_worst_slot=jnp.where(
                learner_update_valid,
                learner_update.curation_selected_active_worst_slot,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            curation_selected_promotion_candidate=jnp.where(
                learner_update_valid,
                learner_update.curation_selected_promotion_candidate,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            curation_selected_refresh_candidate=jnp.where(
                learner_update_valid,
                learner_update.curation_selected_refresh_candidate,
                jnp.asarray(-1, dtype=jnp.int32),
            ),
            curation_proposed=curation_proposed,
            safe_curation_boundary=safe_curation_boundary,
            curation_deferred=curation_deferred,
            routing_attempted=routing_attempted,
            input_route_valid=(
                routing_attempted & input_route.diagnostics.valid
            ),
            output_route_valid=(
                routing_attempted & output_route.diagnostics.valid
            ),
            route_states_match=routing_attempted & route_states_match_raw,
            routed_values_finite=(
                routing_attempted & routed_values_finite_raw
            ),
            curation_committed=curation_committed,
            curation_rolled_back=curation_rolled_back,
            postcondition_checked=learner_update_valid,
            postcondition_valid=postcondition_valid,
            postcondition_rolled_back=postcondition_rolled_back,
            semantic_generation_before=state.router_state.generation_count,
            semantic_generation_after=next_state.router_state.generation_count,
            semantic_generation_words_before=state.router_state.generation_words,
            semantic_generation_words_after=next_state.router_state.generation_words,
            observe_words_before=state.observe_words,
            observe_words_after=next_state.observe_words,
        )
        predictions = jnp.where(
            transaction_applied,
            learner_update.predictions,
            jnp.full_like(learner_update.predictions, jnp.nan),
        )
        errors = jnp.where(
            transaction_applied,
            learner_update.errors,
            jnp.full_like(learner_update.errors, jnp.nan),
        )
        metrics = jnp.where(
            transaction_applied,
            learner_update.metrics,
            jnp.zeros_like(learner_update.metrics),
        )
        if next_horde_state is None:
            destination_result: (
                PrototypeFeatureLifecycleResult | PrototypeFeatureLifecycleHordeResult
            ) = PrototypeFeatureLifecycleResult(
                state=next_state,
                oak_state=next_oak_state,
                consumer_binding=next_consumer_binding,
                next_augmented_observation=final_next_augmented,
                predictions=predictions,
                errors=errors,
                metrics=metrics,
                input_route_diagnostics=input_route.diagnostics,
                output_route_diagnostics=output_route.diagnostics,
                diagnostics=diagnostics,
            )
        else:
            destination_result = PrototypeFeatureLifecycleHordeResult(
                state=next_state,
                oak_state=next_oak_state,
                horde_state=next_horde_state,
                consumer_binding=next_consumer_binding,
                next_augmented_observation=final_next_augmented,
                predictions=predictions,
                errors=errors,
                metrics=metrics,
                input_route_diagnostics=input_route.diagnostics,
                output_route_diagnostics=output_route.diagnostics,
                diagnostics=diagnostics,
                horde_diagnostics=PrototypeFeatureLifecycleHordeDiagnostics(
                    horde_state_values_valid=horde_state_values_valid,
                    pre_step_parity_valid=pre_step_parity_valid,
                    post_step_parity_valid=post_step_parity_valid,
                    lifecycle_capacity_capped=lifecycle_capacity_capped,
                ),
            )

        # A successful routed candidate has one independently usable ordinary
        # successor: keep the learner's already-computed pre-curation update,
        # retain every supplied consumer under the source bank, and account
        # the external veto as rollback (never as deferral).  No learner call
        # occurs beyond the single update above.
        external_curation_rollback = curation_committed & transaction_applied
        ordinary_learner_state = jax.lax.cond(
            learner_update_valid,
            lambda _: learner_update.pre_curation_state,
            lambda _: state.learner_state,
            operand=None,
        )
        ordinary_state_candidate = PrototypeFeatureLifecycleState(
            learner_state=ordinary_learner_state,
            router_state=state.router_state,
            observe_count=next_observe_count,
            observe_words=next_observe_words,
            deferred_curation_count=next_deferred_count,
            deferred_curation_words=next_deferred_words,
            committed_curation_count=state.committed_curation_count,
            committed_curation_words=state.committed_curation_words,
            rolled_back_curation_count=jnp.where(
                external_curation_rollback,
                _saturating_int32_increment(state.rolled_back_curation_count),
                state.rolled_back_curation_count,
            ),
            rolled_back_curation_words=jnp.where(
                external_curation_rollback,
                rolled_back_words_incremented,
                state.rolled_back_curation_words,
            ),
        )
        ordinary_next_augmented = self._augment_base_observation(
            ordinary_learner_state,
            safe_next_observation,
        )
        ordinary_postcondition_valid = (
            self.state_valid(ordinary_state_candidate)
            & self._oak_values_valid(oak_state)
            & self.consumer_binding_valid(
                ordinary_state_candidate,
                consumer_binding,
            )
            & _float32_arrays_bit_exact(
                oak_state.stomp_state.base_last_obs,
                ordinary_next_augmented,
            )
            & jnp.all(ordinary_state_candidate.observe_words == oak_state.step_words)
            & jnp.all(
                ordinary_state_candidate.observe_words
                == oak_state.stomp_state.step_words
            )
            & jnp.all(jnp.isfinite(ordinary_next_augmented))
        )
        if horde_state is not None:
            ordinary_postcondition_valid = (
                ordinary_postcondition_valid
                & self.horde_state_valid(horde_state)
                & jnp.all(horde_state.step_words == oak_state.step_words)
            )
        ordinary_postcondition_valid = transaction_applied & ordinary_postcondition_valid
        ordinary_diagnostics = diagnostics.replace(
            transaction_applied=ordinary_postcondition_valid,
            curation_committed=jnp.asarray(False, dtype=jnp.bool_),
            curation_rolled_back=(
                diagnostics.curation_rolled_back | external_curation_rollback
            ),
            postcondition_valid=ordinary_postcondition_valid,
            postcondition_rolled_back=(
                transaction_applied & ~ordinary_postcondition_valid
            ),
            semantic_generation_after=(
                ordinary_state_candidate.router_state.generation_count
            ),
            semantic_generation_words_after=(
                ordinary_state_candidate.router_state.generation_words
            ),
            observe_words_after=ordinary_state_candidate.observe_words,
        )
        if horde_state is None:
            ordinary_candidate: (
                PrototypeFeatureLifecycleResult | PrototypeFeatureLifecycleHordeResult
            ) = PrototypeFeatureLifecycleResult(
                state=ordinary_state_candidate,
                oak_state=oak_state,
                consumer_binding=consumer_binding,
                next_augmented_observation=ordinary_next_augmented,
                predictions=predictions,
                errors=errors,
                metrics=metrics,
                input_route_diagnostics=input_route.diagnostics,
                output_route_diagnostics=output_route.diagnostics,
                diagnostics=ordinary_diagnostics,
            )
        else:
            ordinary_candidate = PrototypeFeatureLifecycleHordeResult(
                state=ordinary_state_candidate,
                oak_state=oak_state,
                horde_state=horde_state,
                consumer_binding=consumer_binding,
                next_augmented_observation=ordinary_next_augmented,
                predictions=predictions,
                errors=errors,
                metrics=metrics,
                input_route_diagnostics=input_route.diagnostics,
                output_route_diagnostics=output_route.diagnostics,
                diagnostics=ordinary_diagnostics,
                horde_diagnostics=PrototypeFeatureLifecycleHordeDiagnostics(
                    horde_state_values_valid=horde_state_values_valid,
                    pre_step_parity_valid=pre_step_parity_valid,
                    post_step_parity_valid=jnp.all(
                        oak_state.step_words == horde_state.step_words
                    ),
                    lifecycle_capacity_capped=lifecycle_capacity_capped,
                ),
            )
        ordinary_result = jax.lax.cond(
            curation_committed,
            lambda _: ordinary_candidate,
            lambda _: destination_result,
            operand=None,
        )
        preparation_valid = transaction_applied & jnp.where(
            curation_committed,
            ordinary_postcondition_valid,
            jnp.asarray(True, dtype=jnp.bool_),
        )
        work = jnp.asarray(1, dtype=jnp.int32)
        if horde_state is None:
            return PrototypeFeatureLifecyclePreparedRoute(
                source_state=state,
                source_oak_state=oak_state,
                source_consumer_binding=consumer_binding,
                event=event,
                curation_priority_override=curation_priority_override,
                ordinary_result=cast(PrototypeFeatureLifecycleResult, ordinary_result),
                destination_result=cast(
                    PrototypeFeatureLifecycleResult,
                    destination_result,
                ),
                internally_valid=preparation_valid,
                preparation_learner_update_evaluations=work,
            )
        return PrototypeFeatureLifecyclePreparedHordeRoute(
            source_state=state,
            source_oak_state=oak_state,
            source_horde_state=horde_state,
            source_consumer_binding=consumer_binding,
            event=event,
            curation_priority_override=curation_priority_override,
            ordinary_result=cast(PrototypeFeatureLifecycleHordeResult, ordinary_result),
            destination_result=cast(
                PrototypeFeatureLifecycleHordeResult,
                destination_result,
            ),
            internally_valid=preparation_valid,
            preparation_learner_update_evaluations=work,
        )


def prototype_feature_lifecycle_lifetime_counter_nbytes() -> int:
    """Return bytes for one telemetry scalar plus one exact word identity."""

    return PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES


def prototype_feature_lifecycle_counter_nbytes() -> int:
    """Return bytes for all four lifecycle-owned logical clocks."""

    return PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES


def measure_prototype_feature_lifecycle_state_nbytes(
    state: PrototypeFeatureLifecycleState,
) -> int:
    """Measure every persistent PyTree leaf in one lifecycle state."""

    if type(state) is not PrototypeFeatureLifecycleState:
        raise TypeError("state must be a PrototypeFeatureLifecycleState")
    return _tree_nbytes(state)


def migrate_legacy_prototype_feature_lifecycle_state(
    lifecycle: PrototypeFeatureLifecycle,
    legacy_state: Any,
) -> PrototypeFeatureLifecycleState:
    """Migrate pre-v2 outer clocks only when int32 history is unambiguous.

    Nested learner and router exact clocks must already be present and must
    authenticate the derived lifecycle identities. Saturated legacy telemetry
    cannot reveal how many post-saturation events occurred and is rejected.
    """

    if type(lifecycle) is not PrototypeFeatureLifecycle:
        raise TypeError("lifecycle must be a PrototypeFeatureLifecycle")
    if isinstance(legacy_state, Mapping):
        fields = dict(legacy_state)
    elif dataclasses.is_dataclass(legacy_state) and not isinstance(legacy_state, type):
        fields = {
            field.name: getattr(legacy_state, field.name)
            for field in dataclasses.fields(legacy_state)
        }
    else:
        raise TypeError("legacy lifecycle state must be a mapping or dataclass")
    exact_word_fields = {
        "observe_words",
        "deferred_curation_words",
        "committed_curation_words",
        "rolled_back_curation_words",
    }
    current_names = {
        field.name
        for field in dataclasses.fields(PrototypeFeatureLifecycleState)  # type: ignore[arg-type]
    }
    legacy_names = current_names - exact_word_fields
    if set(fields) != legacy_names:
        missing = sorted(legacy_names - set(fields))
        extra = sorted(set(fields) - legacy_names)
        raise ValueError(
            "legacy lifecycle field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    counter_pairs = (
        ("observe_count", "observe_words"),
        ("deferred_curation_count", "deferred_curation_words"),
        ("committed_curation_count", "committed_curation_words"),
        ("rolled_back_curation_count", "rolled_back_curation_words"),
    )
    for telemetry_name, words_name in counter_pairs:
        telemetry = jnp.asarray(fields[telemetry_name])
        if telemetry.shape != () or telemetry.dtype != jnp.dtype(jnp.int32):
            raise TypeError(f"legacy lifecycle {telemetry_name} must be scalar int32")
        count = int(telemetry)
        if count < 0:
            raise ValueError(f"negative legacy lifecycle {telemetry_name} indicates wrap")
        if count >= _INT32_MAX:
            raise ValueError(
                f"saturated legacy lifecycle {telemetry_name} is ambiguous"
            )
        fields[words_name] = jnp.asarray((0, count), dtype=jnp.uint32)
    migrated = PrototypeFeatureLifecycleState(**fields)
    if not bool(lifecycle.state_valid(migrated)):
        raise ValueError(
            "legacy lifecycle history does not authenticate its nested exact clocks"
        )
    return migrated


def migrate_legacy_prototype_feature_consumer_binding(
    lifecycle: PrototypeFeatureLifecycle,
    state: PrototypeFeatureLifecycleState,
    legacy_binding: Any,
) -> PrototypeFeatureConsumerBinding:
    """Migrate an unsaturated legacy generation binding against live state."""

    if type(lifecycle) is not PrototypeFeatureLifecycle:
        raise TypeError("lifecycle must be a PrototypeFeatureLifecycle")
    if not bool(lifecycle.state_valid(state)):
        raise ValueError("state must be a valid exact lifecycle state")
    if isinstance(legacy_binding, Mapping):
        fields = dict(legacy_binding)
    elif dataclasses.is_dataclass(legacy_binding) and not isinstance(
        legacy_binding,
        type,
    ):
        fields = {
            field.name: getattr(legacy_binding, field.name)
            for field in dataclasses.fields(legacy_binding)
        }
    else:
        raise TypeError("legacy consumer binding must be a mapping or dataclass")
    if set(fields) != {"semantic_generation", "descriptors"}:
        raise ValueError("legacy consumer binding field manifest is not exact")
    generation = jnp.asarray(fields["semantic_generation"])
    if generation.shape != () or generation.dtype != jnp.dtype(jnp.int32):
        raise TypeError("legacy semantic_generation must be scalar int32")
    generation_value = int(generation)
    if generation_value < 0:
        raise ValueError("negative legacy semantic_generation indicates wrap")
    if generation_value >= _INT32_MAX:
        raise ValueError("saturated legacy semantic_generation is ambiguous")
    migrated = PrototypeFeatureConsumerBinding(
        semantic_generation=generation,
        semantic_generation_words=jnp.asarray(
            (0, generation_value),
            dtype=jnp.uint32,
        ),
        descriptors=fields["descriptors"],
    )
    if not bool(lifecycle.consumer_binding_valid(state, migrated)):
        raise ValueError("legacy consumer binding does not authenticate exact state")
    return migrated


def save_prototype_feature_lifecycle_checkpoint(
    lifecycle: PrototypeFeatureLifecycle,
    state: PrototypeFeatureLifecycleState,
    path: str | Path,
) -> None:
    """Persist only a valid owned lifecycle state and its exact L0 contract.

    This standalone checkpoint does not own or persist the binding, OaK, or
    Horde consumer subtrees.  A shared-composition caller must checkpoint all
    three together; the full Prototype checkpoint is the supported path.
    """

    if type(lifecycle) is not PrototypeFeatureLifecycle:
        raise TypeError("lifecycle must be a PrototypeFeatureLifecycle")
    if not bool(lifecycle.state_valid(state)):
        raise ValueError("prototype feature lifecycle state is invalid")
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA,
            "state_schema": PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA,
            "mechanism_status": PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS,
            "scientific_promotion_allowed": (
                PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED
            ),
            "config": lifecycle.to_config(),
            "resource_budget": lifecycle.resource_budget(state).to_config(),
        },
    )


def load_prototype_feature_lifecycle_checkpoint(
    path: str | Path,
) -> tuple[PrototypeFeatureLifecycle, PrototypeFeatureLifecycleState]:
    """Restore the exact standalone checkpoint structure, resources, and state."""

    metadata = load_checkpoint_metadata(path)
    expected = {
        "schema",
        "state_schema",
        "mechanism_status",
        "scientific_promotion_allowed",
        "config",
        "resource_budget",
    }
    checkpoint_schema = metadata.get("schema")
    if checkpoint_schema == _LEGACY_PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA:
        if set(metadata) != expected - {"state_schema"}:
            raise ValueError(
                "legacy prototype feature lifecycle checkpoint fields are invalid"
            )
        raise ValueError(
            "legacy prototype feature lifecycle checkpoint v1 lacks exact outer "
            "clocks; migrate its decoded state with "
            "migrate_legacy_prototype_feature_lifecycle_state and resave it"
        )
    if set(metadata) != expected:
        raise ValueError("prototype feature lifecycle checkpoint fields are invalid")
    if checkpoint_schema != PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA:
        raise ValueError("prototype feature lifecycle checkpoint schema is unsupported")
    if metadata.get("state_schema") != PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA:
        raise ValueError("prototype feature lifecycle state schema is unsupported")
    if (
        metadata.get("mechanism_status")
        != PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS
    ):
        raise ValueError("prototype feature lifecycle checkpoint is not mechanism-only")
    if metadata.get("scientific_promotion_allowed") is not False:
        raise ValueError("prototype feature lifecycle checkpoint cannot claim promotion")
    raw_config = metadata.get("config")
    if type(raw_config) is not dict:
        raise ValueError("prototype feature lifecycle checkpoint config is invalid")
    lifecycle = PrototypeFeatureLifecycle.from_config(raw_config)
    template = lifecycle.init(jr.key(0))
    restored, restored_metadata = load_checkpoint(template, path)
    if not _exact_json_tree_equal(restored_metadata, metadata):
        raise ValueError("prototype feature lifecycle checkpoint metadata changed between reads")
    state = cast(PrototypeFeatureLifecycleState, restored)
    if not bool(lifecycle.state_valid(state)):
        raise ValueError("prototype feature lifecycle checkpoint state is invalid")
    budget = metadata.get("resource_budget")
    if type(budget) is not dict:
        raise ValueError("prototype feature lifecycle checkpoint resource budget is invalid")
    expected_budget = lifecycle.resource_budget(state).to_config()
    if not _exact_json_tree_equal(budget, expected_budget):
        raise ValueError("prototype feature lifecycle checkpoint resource contract changed")
    return lifecycle, state


__all__ = [
    "PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_DELTA_NBYTES",
    "PROTOTYPE_FEATURE_CONSUMER_BINDING_GENERATION_NBYTES",
    "PROTOTYPE_FEATURE_LIFECYCLE_CHECKPOINT_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_DELTA_NBYTES",
    "PROTOTYPE_FEATURE_LIFECYCLE_COUNTER_NBYTES",
    "PROTOTYPE_FEATURE_LIFECYCLE_HORDE_CONFIG_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_DELTA_NBYTES",
    "PROTOTYPE_FEATURE_LIFECYCLE_LIFETIME_COUNTER_NBYTES",
    "PROTOTYPE_FEATURE_LIFECYCLE_MECHANISM_STATUS",
    "PROTOTYPE_FEATURE_LIFECYCLE_SCIENTIFIC_PROMOTION_ALLOWED",
    "PROTOTYPE_FEATURE_LIFECYCLE_STATE_SCHEMA",
    "PROTOTYPE_FEATURE_LIFECYCLE_TELEMETRY_COUNTER_NBYTES",
    "PrototypeFeatureConsumerBinding",
    "PrototypeFeatureLifecycle",
    "PrototypeFeatureLifecycleAdoptionDiagnostics",
    "PrototypeFeatureLifecycleAdoptionResult",
    "PrototypeFeatureLifecycleConfig",
    "PrototypeFeatureLifecycleDiagnostics",
    "PrototypeFeatureLifecycleEvent",
    "PrototypeFeatureLifecycleExternalReadinessReceipt",
    "PrototypeFeatureLifecycleExternalTransactionResourceBudget",
    "PrototypeFeatureLifecycleHordeAdoptionResult",
    "PrototypeFeatureLifecycleHordeDiagnostics",
    "PrototypeFeatureLifecycleHordeExternalReadinessReceipt",
    "PrototypeFeatureLifecycleHordeResult",
    "PrototypeFeatureLifecyclePreparedHordeRoute",
    "PrototypeFeatureLifecyclePreparedRoute",
    "PrototypeFeatureLifecycleResourceBudget",
    "PrototypeFeatureLifecycleResult",
    "PrototypeFeatureLifecycleState",
    "PrototypePairGradientPullback",
    "load_prototype_feature_lifecycle_checkpoint",
    "measure_prototype_feature_lifecycle_state_nbytes",
    "migrate_legacy_prototype_feature_consumer_binding",
    "migrate_legacy_prototype_feature_lifecycle_state",
    "prototype_feature_lifecycle_counter_nbytes",
    "prototype_feature_lifecycle_lifetime_counter_nbytes",
    "save_prototype_feature_lifecycle_checkpoint",
]
