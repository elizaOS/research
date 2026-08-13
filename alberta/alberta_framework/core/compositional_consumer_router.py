# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Transactional birth-identity routing for exact linear consumers.

This module is the first consumer boundary for
``CompositionalFeatureAdapter``.  It binds one exact linear OaK state and,
optionally, one exact linear ``MultiHeadMLPState`` whose head rows retain their
declared Horde order.  A prepared adapter update is routed in-place by slot
birth identity: unchanged births retain their exact column bits and every
changed birth is scrubbed.  Equal feature descriptors are deliberately
irrelevant to routing.

The complete compositional binding is carried with the caller-owned consumer
state.  Commit recomputes both the adapter proposal and the route, then passes
one ``consumers_ready`` scalar to the adapter's own prepared-update boundary.
Consequently stale, corrupt, non-finite, or misbound transactions leave both
adapter and consumer state unchanged.

An unsafe boundary rejects the whole prepared outer step when a structural
birth change was proposed.  This isolated router does not derive curation
permission itself: live wiring must compute the safe boundary before calling
``prepare_update(..., curation_allowed=...)``.  Passing false consumes the due
curation opportunity without a structural proposal; incorrectly passing true
at an unsafe boundary can still cause a repeated rollback.

This is an isolated development mechanism.  It is not wired into
``PrototypeFeatureLifecycle`` or ``PrototypeAgent`` and grants no scientific
promotion authority.
"""

from __future__ import annotations

import dataclasses
import math
from typing import cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Bool, Int, UInt

from alberta_framework.core.compositional_feature_adapter import (
    CompositionalFeatureAdapter,
    CompositionalFeatureAdapterPreparedUpdate,
    CompositionalFeatureAdapterState,
    CompositionalFeatureBinding,
)
from alberta_framework.core.multi_head_learner import MultiHeadMLPState
from alberta_framework.core.normalizers import _lifetime_counter_valid
from alberta_framework.core.oak import OaKConfig, OaKState
from alberta_framework.core.options import STOMPConfig, SubtaskSpec
from alberta_framework.core.types import LMSState

COMPOSITIONAL_CONSUMER_ROUTER_STATE_SCHEMA = (
    "alberta.compositional-consumer-router.state.v1"
)
COMPOSITIONAL_CONSUMER_ROUTER_MECHANISM_STATUS = "development_mechanism_only"
COMPOSITIONAL_CONSUMER_ROUTER_SCIENTIFIC_PROMOTION_ALLOWED = False

_INT32_MAX = 2**31 - 1
_UINT32_MAX = 2**32 - 1


def _leaf_contract(value: object, template: object) -> bool:
    template_shape = getattr(template, "shape", None)
    template_dtype = getattr(template, "dtype", None)
    if template_shape is not None and template_dtype is not None:
        return bool(
            getattr(value, "shape", None) == template_shape
            and getattr(value, "dtype", None) == template_dtype
        )
    if type(template) is float:
        if type(value) is float:
            return math.isfinite(value)
        return bool(
            getattr(value, "shape", None) == ()
            and jnp.issubdtype(
                getattr(value, "dtype", jnp.dtype(jnp.float32)),
                jnp.floating,
            )
        )
    return type(value) is type(template)


def _tree_contract(value: object, template: object) -> bool:
    if str(jax.tree_util.tree_structure(value)) != str(
        jax.tree_util.tree_structure(template)
    ):
        return False
    return all(
        _leaf_contract(actual, expected)
        for actual, expected in zip(
            jax.tree_util.tree_leaves(value),
            jax.tree_util.tree_leaves(template),
            strict=True,
        )
    )


def _array_bits_equal(left: object, right: object) -> Bool[Array, ""]:
    try:
        left_dtype = getattr(left, "dtype", None)
        right_dtype = getattr(right, "dtype", None)
        if (
            left_dtype is not None
            and jax.dtypes.issubdtype(left_dtype, jax.dtypes.prng_key)
        ) or (
            right_dtype is not None
            and jax.dtypes.issubdtype(right_dtype, jax.dtypes.prng_key)
        ):
            if left_dtype != right_dtype:
                return jnp.asarray(False, dtype=jnp.bool_)
            return jnp.all(
                jr.key_data(cast(Array, left)) == jr.key_data(cast(Array, right))
            )
    except TypeError:
        return jnp.asarray(False, dtype=jnp.bool_)

    left_array = jnp.asarray(left)
    right_array = jnp.asarray(right)
    if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
        return jnp.asarray(False, dtype=jnp.bool_)
    if left_array.dtype == jnp.dtype(jnp.float32):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint32)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint32)
        )
    if left_array.dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint16)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint16)
        )
    if left_array.dtype == jnp.dtype(jnp.float64):
        return jnp.all(
            jax.lax.bitcast_convert_type(left_array, jnp.uint64)
            == jax.lax.bitcast_convert_type(right_array, jnp.uint64)
        )
    return jnp.all(left_array == right_array)


def _tree_bits_equal(left: object, right: object) -> Bool[Array, ""]:
    if str(jax.tree_util.tree_structure(left)) != str(
        jax.tree_util.tree_structure(right)
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(
        jax.tree_util.tree_leaves(left),
        jax.tree_util.tree_leaves(right),
        strict=True,
    ):
        valid &= _array_bits_equal(left_leaf, right_leaf)
    return valid


def _tree_finite(value: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree_util.tree_leaves(value):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            try:
                if jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
                    continue
            except TypeError:
                pass
        array = jnp.asarray(leaf)
        if jnp.issubdtype(array.dtype, jnp.inexact):
            valid &= jnp.all(jnp.isfinite(array))
    return valid


def _tree_nbytes(value: object) -> int:
    total = 0
    for leaf in jax.tree_util.tree_leaves(value):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is not None and dtype is not None:
            count = math.prod(shape) if shape else 1
            total += count * int(dtype.itemsize)
        elif type(leaf) in (float, int):
            total += 8
        elif type(leaf) is bool:
            total += 1
        else:
            raise TypeError(f"unsupported state leaf {type(leaf)!r}")
    return total


def _host_float_adjustment(*states: MultiHeadMLPState) -> int:
    adjustment = 0
    for state in states:
        for value in (state.birth_timestamp, state.uptime_s):
            if type(value) is float:
                continue
            shape = getattr(value, "shape", None)
            dtype = getattr(value, "dtype", None)
            if shape != () or dtype is None or not jnp.issubdtype(dtype, jnp.floating):
                raise TypeError("consumer host-float metadata contract drifted")
            adjustment += 8 - int(dtype.itemsize)
    return adjustment


def _adapter_host_float_adjustment(state: CompositionalFeatureAdapterState) -> int:
    adjustment = 0
    learner = state.learner_state
    for value in (learner.birth_timestamp, learner.uptime_s):
        if type(value) is float:
            continue
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        if shape != () or dtype is None or not jnp.issubdtype(dtype, jnp.floating):
            raise TypeError("adapter host-float metadata contract drifted")
        adjustment += 8 - int(dtype.itemsize)
    return adjustment


def _words_successor(words: Array) -> tuple[Array, Array]:
    maximum = jnp.asarray(_UINT32_MAX, dtype=jnp.uint32)
    available = ~jnp.all(words == maximum)
    low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (low == 0).astype(jnp.uint32)
    candidate = jnp.stack((words[0] + carry, low)).astype(jnp.uint32)
    return jnp.where(available, candidate, words), available


def _masked_zero_last_axis(value: Array, changed: Array) -> Array:
    shape = (1,) * (value.ndim - 1) + (changed.shape[0],)
    return jnp.where(changed.reshape(shape), jnp.zeros_like(value), value)


def _masked_zero_matrix_axes(value: Array, changed: Array) -> Array:
    changed_cells = changed[None, :, None] | changed[None, None, :]
    return jnp.where(changed_cells, jnp.zeros_like(value), value)


def _masked_survivors_exact(
    source: Array,
    candidate: Array,
    changed: Array,
) -> Bool[Array, ""]:
    shape = (1,) * (source.ndim - 1) + (changed.shape[0],)
    survivor_mask = (~changed).reshape(shape)
    source_bits = jax.lax.bitcast_convert_type(source, jnp.uint32)
    candidate_bits = jax.lax.bitcast_convert_type(candidate, jnp.uint32)
    return jnp.all((~survivor_mask) | (source_bits == candidate_bits))


def _masked_changed_zero(
    candidate: Array,
    changed: Array,
) -> Bool[Array, ""]:
    shape = (1,) * (candidate.ndim - 1) + (changed.shape[0],)
    changed_mask = changed.reshape(shape)
    bits = jax.lax.bitcast_convert_type(candidate, jnp.uint32)
    return jnp.all((~changed_mask) | (bits == 0))


@chex.dataclass(frozen=True)
class CompositionalConsumerState:
    """Full bank identity paired with exact caller-owned linear consumers."""

    binding: CompositionalFeatureBinding
    oak_state: OaKState
    horde_state: MultiHeadMLPState | None


@chex.dataclass(frozen=True)
class CompositionalConsumerReadyReceipt:
    """The single consumer verdict supplied to adapter commit."""

    consumers_ready: Bool[Array, ""]
    changed_birth_mask: Bool[Array, " n_features"]
    source_generation_words: UInt[Array, " 2"]
    candidate_generation_words: UInt[Array, " 2"]
    managed_feature_scalars: Int[Array, ""]
    consumer_route_feature_scalar_evaluations: Int[Array, ""]
    cache_representation_calls: Int[Array, ""]
    cache_representation_feature_slot_evaluations: Int[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalConsumerRouteDiagnostics:
    """Source, binding-transition, and exact route checks."""

    source_consumers_valid: Bool[Array, ""]
    post_update_consumers_valid: Bool[Array, ""]
    post_update_clock_parity_valid: Bool[Array, ""]
    source_adapter_valid: Bool[Array, ""]
    candidate_adapter_valid: Bool[Array, ""]
    source_binding_matches: Bool[Array, ""]
    post_update_binding_matches: Bool[Array, ""]
    candidate_binding_matches: Bool[Array, ""]
    proposal_transaction_applied: Bool[Array, ""]
    full_binding_transition_valid: Bool[Array, ""]
    stable_base_unchanged: Bool[Array, ""]
    safe_route_boundary: Bool[Array, ""]
    source_cache_matches: Bool[Array, ""]
    candidate_cache_recomputed: Bool[Array, ""]
    survivor_columns_bit_exact: Bool[Array, ""]
    changed_columns_scrubbed: Bool[Array, ""]
    optimizer_state_authenticated: Bool[Array, ""]
    routed_values_finite: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalConsumerPreparedRoute:
    """Pure source-bound route awaiting atomic adapter/consumer commit."""

    source_state: CompositionalConsumerState
    post_update_state: CompositionalConsumerState
    adapter_proposal: CompositionalFeatureAdapterPreparedUpdate
    candidate_state: CompositionalConsumerState
    receipt: CompositionalConsumerReadyReceipt
    diagnostics: CompositionalConsumerRouteDiagnostics


@chex.dataclass(frozen=True)
class CompositionalConsumerCommitDiagnostics:
    """Commit-time integrity and joint-adoption facts."""

    route_integrity: Bool[Array, ""]
    consumer_source_matches: Bool[Array, ""]
    destination_consumers_valid: Bool[Array, ""]
    consumers_ready: Bool[Array, ""]
    adapter_applied: Bool[Array, ""]
    applied: Bool[Array, ""]
    rejected: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CompositionalConsumerCommitResult:
    """Atomic adapter and consumer state convenient as a scan carry."""

    adapter_state: CompositionalFeatureAdapterState
    consumer_state: CompositionalConsumerState
    diagnostics: CompositionalConsumerCommitDiagnostics


@dataclasses.dataclass(frozen=True)
class CompositionalConsumerResourceBudget:
    """Exact persistent bytes and fixed slot-route work."""

    schema: str
    mechanism_status: str
    scientific_promotion_allowed: bool
    base_feature_dim: int
    n_features: int
    horde_enabled: bool
    horde_heads: int
    binding_persistent_nbytes: int
    oak_persistent_nbytes: int
    horde_persistent_nbytes: int
    total_persistent_state_nbytes: int
    internal_pristine_oak_nbytes: int
    internal_pristine_horde_nbytes: int
    internal_pristine_template_nbytes: int
    managed_feature_scalars: int
    fixed_consumer_route_feature_scalar_evaluations: int
    fixed_cache_representation_calls_per_route: int
    fixed_cache_representation_feature_slot_evaluations_per_route: int
    route_recomputations_per_commit: int
    fixed_prepare_commit_consumer_route_feature_scalar_evaluations: int
    fixed_prepare_commit_cache_representation_calls: int
    fixed_prepare_commit_cache_representation_feature_slot_evaluations: int
    fixed_slot_birth_word_comparisons: int


class CompositionalConsumerRouter:
    """Bind and transactionally route exact linear OaK/Horde consumers."""

    def __init__(
        self,
        adapter: CompositionalFeatureAdapter,
        oak_config: OaKConfig,
        pristine_oak_state: OaKState,
        pristine_horde_state: MultiHeadMLPState | None,
    ) -> None:
        if type(adapter) is not CompositionalFeatureAdapter:
            raise TypeError("adapter must be the exact CompositionalFeatureAdapter class")
        self._adapter = adapter
        self._width = adapter.n_features
        self._base_width = adapter.base_feature_dim
        self._oak_config = oak_config
        self._oak_template = pristine_oak_state
        self._horde_template = pristine_horde_state
        self._has_horde = pristine_horde_state is not None

        self._require_compatible_oak_config(oak_config)
        if not self._oak_static_contract(pristine_oak_state):
            raise ValueError("pristine_oak_state must be an exact linear OaK state")
        if not bool(self._oak_values_valid(pristine_oak_state)):
            raise ValueError("pristine linear OaK state is dynamically invalid")
        if not bool(self._consumer_clocks_zero(pristine_oak_state)):
            raise ValueError("pristine linear OaK state must have zero clocks")
        if self._has_horde:
            assert pristine_horde_state is not None
            if not self._horde_static_contract(pristine_horde_state):
                raise ValueError("pristine_horde_state must be one exact linear Horde")
            if not bool(self._horde_values_valid(pristine_horde_state)):
                raise ValueError("pristine linear Horde state is dynamically invalid")
            if not bool(jnp.all(pristine_horde_state.step_words == 0)):
                raise ValueError("pristine linear Horde state must have zero clocks")

        managed = self._count_managed_feature_scalars(
            pristine_oak_state,
            pristine_horde_state,
        )
        if managed > _INT32_MAX // 3:
            raise ValueError("managed consumer route exceeds exact int32 work accounting")
        self._managed_feature_scalars = managed

    @property
    def adapter(self) -> CompositionalFeatureAdapter:
        """The exact adapter whose proposals this boundary accepts."""

        return self._adapter

    @property
    def oak_config(self) -> OaKConfig:
        """The exact stable-prefix-only OaK semantic attestation."""

        return self._oak_config

    def _require_compatible_oak_config(self, config: OaKConfig) -> None:
        """Attest linear layout and stable-prefix-only option semantics."""

        if type(config) is not OaKConfig:
            raise TypeError("oak_config must be an exact OaKConfig")
        stomp = config.stomp
        if type(stomp) is not STOMPConfig:
            raise TypeError("oak_config.stomp must be an exact STOMPConfig")
        specs = stomp.subtask_specs
        exact_specs = type(specs) is tuple and all(
            type(spec) is SubtaskSpec and type(spec.feature_index) is int
            for spec in specs
        )
        if not (
            exact_specs
            and type(stomp.observation_dim) is int
            and type(stomp.n_primitive_actions) is int
            and type(stomp.base_hidden_sizes) is tuple
            and stomp.observation_dim == self._width
            and stomp.base_hidden_sizes == ()
            and len(specs) > 0
            and all(
                0 <= spec.feature_index < self._base_width
                for spec in specs
            )
        ):
            raise ValueError(
                "OaK config must attest the exact linear OaK layout and bind every "
                "subtask feature to the stable base prefix"
            )

    def _multi_head_linear_contract(
        self,
        state: object,
        template: MultiHeadMLPState,
    ) -> bool:
        if type(state) is not MultiHeadMLPState or not _tree_contract(state, template):
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
            and len(state.head_params.weights) > 0
            and len(state.head_params.weights) == len(state.head_params.biases)
            and len(state.head_params.weights) == len(state.head_optimizer_states)
            and len(state.head_params.weights) == len(state.head_traces)
        ):
            return False
        for weight, bias, optimizer_pair, trace_pair in zip(
            state.head_params.weights,
            state.head_params.biases,
            state.head_optimizer_states,
            state.head_traces,
            strict=True,
        ):
            if not (
                weight.shape == (1, self._width)
                and weight.dtype == jnp.dtype(jnp.float32)
                and bias.shape == (1,)
                and bias.dtype == jnp.dtype(jnp.float32)
                and type(optimizer_pair) is tuple
                and len(optimizer_pair) == 2
                and type(optimizer_pair[0]) is LMSState
                and type(optimizer_pair[1]) is LMSState
                and type(trace_pair) is tuple
                and len(trace_pair) == 2
                and trace_pair[0].shape == weight.shape
                and trace_pair[0].dtype == weight.dtype
                and trace_pair[1].shape == bias.shape
                and trace_pair[1].dtype == bias.dtype
            ):
                return False
        return True

    def _oak_static_contract(self, state: object) -> bool:
        if type(state) is not OaKState or not _tree_contract(state, self._oak_template):
            return False
        stomp = state.stomp_state
        base = stomp.base_learner_state
        if not self._multi_head_linear_contract(
            base,
            self._oak_template.stomp_state.base_learner_state,
        ):
            return False
        template_stomp = self._oak_template.stomp_state
        n_options = self._oak_config.n_options
        n_primitive = self._oak_config.n_primitive_actions
        return all(
            (
                len(base.head_params.weights) == n_options + n_primitive,
                stomp.base_last_obs.shape == (self._width,),
                stomp.option_start_obs.shape == (self._width,),
                stomp.option_policies.q_weights.shape
                == template_stomp.option_policies.q_weights.shape,
                stomp.option_policies.q_weights.shape[-1] == self._width,
                stomp.option_policies.q_weights.shape[:2]
                == (n_options, n_primitive),
                stomp.option_policies.traces.shape
                == stomp.option_policies.q_weights.shape,
                stomp.option_models.next_state_weights.shape
                == template_stomp.option_models.next_state_weights.shape,
                stomp.option_models.next_state_weights.shape[-2:]
                == (self._width, self._width),
                state.execution_counts.shape == (n_options,),
                state.cumulative_pseudo_rewards.shape == (n_options,),
                state.utility_ema.shape == (n_options,),
            )
        )

    def _horde_static_contract(self, state: object) -> bool:
        if not self._has_horde or self._horde_template is None:
            return False
        return self._multi_head_linear_contract(state, self._horde_template)

    def _optimizer_values_valid(self, state: MultiHeadMLPState) -> Bool[Array, ""]:
        valid = jnp.asarray(True, dtype=jnp.bool_)
        for optimizer_pair in state.head_optimizer_states:
            valid &= jnp.isfinite(optimizer_pair[0].step_size)
            valid &= jnp.isfinite(optimizer_pair[1].step_size)
            valid &= optimizer_pair[0].step_size > 0.0
            valid &= optimizer_pair[1].step_size > 0.0
        return valid

    def _multi_head_values_valid(self, state: MultiHeadMLPState) -> Bool[Array, ""]:
        return (
            _tree_finite(state)
            & (jnp.asarray(state.birth_timestamp) >= 0.0)
            & (jnp.asarray(state.uptime_s) >= 0.0)
            & _lifetime_counter_valid(state.step_words, state.step_count)
            & self._optimizer_values_valid(state)
        )

    def _consumer_clocks_zero(self, state: OaKState) -> Bool[Array, ""]:
        stomp = state.stomp_state
        base = stomp.base_learner_state
        return (
            (state.step_count == 0)
            & (stomp.step_count == 0)
            & (base.step_count == 0)
            & jnp.all(state.step_words == 0)
            & jnp.all(stomp.step_words == 0)
            & jnp.all(base.step_words == 0)
        )

    def _oak_values_valid(self, state: OaKState) -> Bool[Array, ""]:
        stomp = state.stomp_state
        base = stomp.base_learner_state
        clocks = (
            _lifetime_counter_valid(state.step_words, state.step_count)
            & _lifetime_counter_valid(stomp.step_words, stomp.step_count)
            & _lifetime_counter_valid(base.step_words, base.step_count)
            & jnp.all(state.step_words == stomp.step_words)
            & jnp.all(state.step_words == base.step_words)
        )
        return (
            _tree_finite(state)
            & clocks
            & self._multi_head_values_valid(base)
            & jnp.all(state.execution_counts >= 0)
            & jnp.all(stomp.option_models.n_completions >= 0)
            & jnp.all(stomp.option_models.n_completions <= state.execution_counts)
        )

    def _horde_values_valid(self, state: MultiHeadMLPState) -> Bool[Array, ""]:
        return self._multi_head_values_valid(state)

    def _bindings_equal(
        self,
        left: CompositionalFeatureBinding,
        right: CompositionalFeatureBinding,
    ) -> Bool[Array, ""]:
        return _tree_bits_equal(left, right)

    def _consumer_clock_matches(
        self,
        state: CompositionalConsumerState,
        adapter_state: CompositionalFeatureAdapterState,
    ) -> Bool[Array, ""]:
        words = adapter_state.learner_state.step_words
        telemetry = adapter_state.learner_state.step_count
        oak = state.oak_state
        stomp = oak.stomp_state
        base = stomp.base_learner_state
        valid = (
            (oak.step_count == telemetry)
            & (stomp.step_count == telemetry)
            & (base.step_count == telemetry)
            & jnp.all(oak.step_words == words)
            & jnp.all(stomp.step_words == words)
            & jnp.all(base.step_words == words)
        )
        if state.horde_state is not None:
            valid &= state.horde_state.step_count == telemetry
            valid &= jnp.all(state.horde_state.step_words == words)
        return valid

    def _consumer_values_and_layout_valid(
        self,
        state: CompositionalConsumerState,
    ) -> Bool[Array, ""]:
        if not self._oak_static_contract(state.oak_state):
            return jnp.asarray(False, dtype=jnp.bool_)
        if self._has_horde:
            if state.horde_state is None or not self._horde_static_contract(
                state.horde_state
            ):
                return jnp.asarray(False, dtype=jnp.bool_)
        elif state.horde_state is not None:
            return jnp.asarray(False, dtype=jnp.bool_)
        valid = self._oak_values_valid(state.oak_state)
        if state.horde_state is not None:
            valid &= self._horde_values_valid(state.horde_state)
            valid &= jnp.all(
                state.horde_state.step_words == state.oak_state.step_words
            )
        return valid

    def _intermediate_state_valid(
        self,
        state: CompositionalConsumerState,
        source_adapter: CompositionalFeatureAdapterState,
        candidate_adapter: CompositionalFeatureAdapterState,
    ) -> Bool[Array, ""]:
        source_next_words, capacity = _words_successor(
            source_adapter.learner_state.step_words
        )
        successor = (
            capacity
            & jnp.all(candidate_adapter.learner_state.step_words == source_next_words)
            & (
                candidate_adapter.learner_state.step_count
                >= source_adapter.learner_state.step_count
            )
        )
        return (
            self._adapter.state_valid(source_adapter)
            & self._adapter.state_valid(candidate_adapter)
            & self._bindings_equal(state.binding, source_adapter.binding)
            & self._consumer_values_and_layout_valid(state)
            & successor
            & self._consumer_clock_matches(state, candidate_adapter)
        )

    def state_valid(
        self,
        state: CompositionalConsumerState,
        adapter_state: CompositionalFeatureAdapterState,
    ) -> Bool[Array, ""]:
        """Validate full binding, finite values, exact layout, and clocks."""

        if type(state) is not CompositionalConsumerState:
            return jnp.asarray(False, dtype=jnp.bool_)
        valid = self._adapter.state_valid(adapter_state)
        valid &= self._bindings_equal(state.binding, adapter_state.binding)
        valid &= self._consumer_values_and_layout_valid(state)
        valid &= self._consumer_clock_matches(state, adapter_state)
        return valid

    def bind_pristine(
        self,
        adapter_state: CompositionalFeatureAdapterState,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState | None,
    ) -> CompositionalConsumerState:
        """Bind exact pristine consumers to one valid genesis adapter bank."""

        if not self._oak_static_contract(oak_state):
            raise ValueError("oak_state does not satisfy the configured linear OaK layout")
        if self._has_horde:
            if horde_state is None or not self._horde_static_contract(horde_state):
                raise ValueError("Horde state does not satisfy the configured layout")
        elif horde_state is not None:
            raise ValueError("Horde state supplied to a router configured without Horde")
        if not bool(self._adapter.state_valid(adapter_state)):
            raise ValueError("cannot bind consumers to an invalid adapter state")
        if not bool(jnp.all(adapter_state.binding.semantic_generation_words == 0)):
            raise ValueError("pristine consumers can bind only at adapter genesis")
        if not bool(jnp.all(adapter_state.learner_state.step_words == 0)):
            raise ValueError("pristine consumers can bind only at zero adapter lifetime")
        if not bool(_tree_bits_equal(oak_state, self._oak_template)):
            raise ValueError("oak_state is not the configured pristine OaK state")
        if self._has_horde:
            assert horde_state is not None and self._horde_template is not None
            if not bool(_tree_bits_equal(horde_state, self._horde_template)):
                raise ValueError("horde_state is not the configured pristine Horde state")
        state = CompositionalConsumerState(
            binding=adapter_state.binding,
            oak_state=oak_state,
            horde_state=horde_state,
        )
        if not bool(self.state_valid(state, adapter_state)):
            raise ValueError("pristine consumer binding is dynamically invalid")
        return state

    def _binding_transition_valid(
        self,
        source: CompositionalFeatureBinding,
        candidate: CompositionalFeatureBinding,
        changed: Array,
        active_change_mask: Array,
    ) -> Bool[Array, ""]:
        descriptor_changed = (
            (source.ops != candidate.ops)
            | (source.parent_a != candidate.parent_a)
            | (source.parent_b != candidate.parent_b)
            | jnp.any(source.theta_bits != candidate.theta_bits, axis=1)
            | (source.depth != candidate.depth)
        )
        any_changed = jnp.any(changed)
        successor, capacity = _words_successor(source.semantic_generation_words)
        generation_valid = jnp.where(
            any_changed,
            capacity
            & jnp.all(candidate.semantic_generation_words == successor),
            jnp.all(
                candidate.semantic_generation_words
                == source.semantic_generation_words
            ),
        )
        birth_values_valid = jnp.all(
            jnp.where(
                changed[:, None],
                candidate.slot_birth_words
                == candidate.semantic_generation_words[None, :],
                candidate.slot_birth_words == source.slot_birth_words,
            )
        )
        unchanged_descriptors = jnp.all((~descriptor_changed) | changed)
        return (
            generation_valid
            & birth_values_valid
            & unchanged_descriptors
            & ~jnp.any(changed[: self._base_width])
            & jnp.all(changed == active_change_mask)
        )

    def _route_multi_head(
        self,
        state: MultiHeadMLPState,
        changed: Array,
    ) -> MultiHeadMLPState:
        weights = tuple(
            _masked_zero_last_axis(weight, changed)
            for weight in state.head_params.weights
        )
        traces = tuple(
            (_masked_zero_last_axis(pair[0], changed), pair[1])
            for pair in state.head_traces
        )
        return cast(
            MultiHeadMLPState,
            state.replace(
                head_params=state.head_params.replace(weights=weights),
                head_traces=traces,
            ),
        )

    def _route_oak(
        self,
        state: OaKState,
        changed: Array,
        candidate_base_cache: Array,
    ) -> OaKState:
        stomp = state.stomp_state
        routed_base = self._route_multi_head(stomp.base_learner_state, changed)
        routed_stomp = stomp.replace(
            base_learner_state=routed_base,
            base_last_obs=candidate_base_cache,
            option_start_obs=_masked_zero_last_axis(stomp.option_start_obs, changed),
            option_policies=stomp.option_policies.replace(
                q_weights=_masked_zero_last_axis(
                    stomp.option_policies.q_weights,
                    changed,
                ),
                traces=_masked_zero_last_axis(
                    stomp.option_policies.traces,
                    changed,
                ),
            ),
            option_models=stomp.option_models.replace(
                next_state_weights=_masked_zero_matrix_axes(
                    stomp.option_models.next_state_weights,
                    changed,
                )
            ),
        )
        return cast(OaKState, state.replace(stomp_state=routed_stomp))

    def _optimizer_authenticated(
        self,
        source: CompositionalConsumerState,
        candidate: CompositionalConsumerState,
    ) -> Bool[Array, ""]:
        source_base = source.oak_state.stomp_state.base_learner_state
        candidate_base = candidate.oak_state.stomp_state.base_learner_state
        valid = _tree_bits_equal(
            source_base.head_optimizer_states,
            candidate_base.head_optimizer_states,
        )
        if source.horde_state is not None and candidate.horde_state is not None:
            valid &= _tree_bits_equal(
                source.horde_state.head_optimizer_states,
                candidate.horde_state.head_optimizer_states,
            )
        return valid

    def _route_checks(
        self,
        source: CompositionalConsumerState,
        candidate: CompositionalConsumerState,
        changed: Array,
    ) -> tuple[Array, Array]:
        survivors = jnp.asarray(True, dtype=jnp.bool_)
        scrubbed = jnp.asarray(True, dtype=jnp.bool_)

        old_stomp = source.oak_state.stomp_state
        new_stomp = candidate.oak_state.stomp_state
        old_base = old_stomp.base_learner_state
        new_base = new_stomp.base_learner_state
        for old, new in zip(
            old_base.head_params.weights,
            new_base.head_params.weights,
            strict=True,
        ):
            survivors &= _masked_survivors_exact(old, new, changed)
            scrubbed &= _masked_changed_zero(new, changed)
        for old, new in zip(old_base.head_traces, new_base.head_traces, strict=True):
            survivors &= _masked_survivors_exact(old[0], new[0], changed)
            scrubbed &= _masked_changed_zero(new[0], changed)
        for old, new in (
            (old_stomp.option_policies.q_weights, new_stomp.option_policies.q_weights),
            (old_stomp.option_policies.traces, new_stomp.option_policies.traces),
            (old_stomp.option_start_obs, new_stomp.option_start_obs),
        ):
            survivors &= _masked_survivors_exact(old, new, changed)
            scrubbed &= _masked_changed_zero(new, changed)

        old_model = old_stomp.option_models.next_state_weights
        new_model = new_stomp.option_models.next_state_weights
        survivor_cells = (~changed)[None, :, None] & (~changed)[None, None, :]
        changed_cells = ~survivor_cells
        old_model_bits = jax.lax.bitcast_convert_type(old_model, jnp.uint32)
        new_model_bits = jax.lax.bitcast_convert_type(new_model, jnp.uint32)
        survivors &= jnp.all((~survivor_cells) | (old_model_bits == new_model_bits))
        scrubbed &= jnp.all((~changed_cells) | (new_model_bits == 0))

        if source.horde_state is not None and candidate.horde_state is not None:
            for old, new in zip(
                source.horde_state.head_params.weights,
                candidate.horde_state.head_params.weights,
                strict=True,
            ):
                survivors &= _masked_survivors_exact(old, new, changed)
                scrubbed &= _masked_changed_zero(new, changed)
            for old, new in zip(
                source.horde_state.head_traces,
                candidate.horde_state.head_traces,
                strict=True,
            ):
                survivors &= _masked_survivors_exact(old[0], new[0], changed)
                scrubbed &= _masked_changed_zero(new[0], changed)
        return survivors, scrubbed

    def prepare_route(
        self,
        source_state: CompositionalConsumerState,
        post_update_state: CompositionalConsumerState,
        proposal: CompositionalFeatureAdapterPreparedUpdate,
    ) -> CompositionalConsumerPreparedRoute:
        """Prepare one stable-source/post-update consumer transaction."""

        if type(source_state) is not CompositionalConsumerState:
            raise TypeError("source_state must be a CompositionalConsumerState")
        if type(post_update_state) is not CompositionalConsumerState:
            raise TypeError("post_update_state must be a CompositionalConsumerState")
        if type(proposal) is not CompositionalFeatureAdapterPreparedUpdate:
            raise TypeError(
                "proposal must be an exact CompositionalFeatureAdapterPreparedUpdate"
            )
        if not self._oak_static_contract(source_state.oak_state) or not (
            self._oak_static_contract(post_update_state.oak_state)
        ):
            raise ValueError("consumer OaK states have an unsafe static contract")
        if self._has_horde:
            if (
                source_state.horde_state is None
                or post_update_state.horde_state is None
                or not self._horde_static_contract(source_state.horde_state)
                or not self._horde_static_contract(post_update_state.horde_state)
            ):
                raise ValueError("consumer Horde states have an unsafe static contract")
        elif source_state.horde_state is not None or post_update_state.horde_state is not None:
            raise ValueError("consumer Horde presence differs from router configuration")

        source_adapter = proposal.source_state
        candidate_adapter = proposal.candidate_state
        changed = jnp.any(
            source_adapter.binding.slot_birth_words
            != candidate_adapter.binding.slot_birth_words,
            axis=1,
        )
        source_cache = self._adapter.representation(
            source_adapter,
            proposal.observation,
        )
        candidate_cache = self._adapter.representation(
            candidate_adapter,
            proposal.observation,
        )
        routed_oak = self._route_oak(
            post_update_state.oak_state,
            changed,
            candidate_cache,
        )
        routed_horde = (
            self._route_multi_head(post_update_state.horde_state, changed)
            if post_update_state.horde_state is not None
            else None
        )
        routed = CompositionalConsumerState(
            binding=candidate_adapter.binding,
            oak_state=routed_oak,
            horde_state=routed_horde,
        )

        source_consumers_valid = self.state_valid(source_state, source_adapter)
        post_update_consumers_valid = self._intermediate_state_valid(
            post_update_state,
            source_adapter,
            candidate_adapter,
        )
        next_words, clock_capacity = _words_successor(
            source_adapter.learner_state.step_words
        )
        post_update_clock_parity_valid = (
            clock_capacity
            & jnp.all(candidate_adapter.learner_state.step_words == next_words)
            & self._consumer_clock_matches(post_update_state, candidate_adapter)
        )
        source_adapter_valid = self._adapter.state_valid(source_adapter)
        candidate_adapter_valid = self._adapter.state_valid(candidate_adapter)
        source_binding_matches = self._bindings_equal(
            source_state.binding,
            source_adapter.binding,
        )
        post_update_binding_matches = self._bindings_equal(
            post_update_state.binding,
            source_adapter.binding,
        )
        candidate_binding_matches = self._bindings_equal(
            routed.binding,
            candidate_adapter.binding,
        )
        full_transition = self._binding_transition_valid(
            source_adapter.binding,
            candidate_adapter.binding,
            changed,
            proposal.diagnostics.active_change_mask,
        )
        stable_base_unchanged = ~jnp.any(changed[: self._base_width])
        stomp = post_update_state.oak_state.stomp_state
        raw_safe_route_boundary = (
            (stomp.executing_option == -1)
            & (stomp.base_last_action >= 0)
            & (
                stomp.base_last_action
                < self._oak_config.n_primitive_actions
            )
        )
        safe_route_boundary = (~jnp.any(changed)) | raw_safe_route_boundary
        source_cache_matches = _array_bits_equal(stomp.base_last_obs, source_cache)
        candidate_cache_recomputed = _array_bits_equal(
            routed.oak_state.stomp_state.base_last_obs,
            candidate_cache,
        )
        survivors, scrubbed = self._route_checks(
            post_update_state,
            routed,
            changed,
        )
        optimizer_authenticated = self._optimizer_authenticated(
            post_update_state,
            routed,
        )
        routed_finite = _tree_finite(routed.oak_state)
        if routed.horde_state is not None:
            routed_finite &= _tree_finite(routed.horde_state)
        candidate_consumers_valid = self.state_valid(routed, candidate_adapter)
        ready = (
            source_consumers_valid
            & post_update_consumers_valid
            & post_update_clock_parity_valid
            & source_adapter_valid
            & candidate_adapter_valid
            & source_binding_matches
            & post_update_binding_matches
            & candidate_binding_matches
            & proposal.diagnostics.transaction_applied
            & full_transition
            & stable_base_unchanged
            & safe_route_boundary
            & source_cache_matches
            & candidate_cache_recomputed
            & survivors
            & scrubbed
            & optimizer_authenticated
            & routed_finite
            & candidate_consumers_valid
        )
        candidate = cast(
            CompositionalConsumerState,
            jax.lax.cond(ready, lambda: routed, lambda: source_state),
        )
        evaluations = 3 * self._managed_feature_scalars
        return CompositionalConsumerPreparedRoute(
            source_state=source_state,
            post_update_state=post_update_state,
            adapter_proposal=proposal,
            candidate_state=candidate,
            receipt=CompositionalConsumerReadyReceipt(
                consumers_ready=ready,
                changed_birth_mask=changed,
                source_generation_words=source_adapter.binding.semantic_generation_words,
                candidate_generation_words=(
                    candidate_adapter.binding.semantic_generation_words
                ),
                managed_feature_scalars=jnp.asarray(
                    self._managed_feature_scalars,
                    dtype=jnp.int32,
                ),
                consumer_route_feature_scalar_evaluations=jnp.asarray(
                    evaluations,
                    dtype=jnp.int32,
                ),
                cache_representation_calls=jnp.asarray(2, dtype=jnp.int32),
                cache_representation_feature_slot_evaluations=jnp.asarray(
                    2 * self._width,
                    dtype=jnp.int32,
                ),
            ),
            diagnostics=CompositionalConsumerRouteDiagnostics(
                source_consumers_valid=source_consumers_valid,
                post_update_consumers_valid=post_update_consumers_valid,
                post_update_clock_parity_valid=post_update_clock_parity_valid,
                source_adapter_valid=source_adapter_valid,
                candidate_adapter_valid=candidate_adapter_valid,
                source_binding_matches=source_binding_matches,
                post_update_binding_matches=post_update_binding_matches,
                candidate_binding_matches=candidate_binding_matches,
                proposal_transaction_applied=proposal.diagnostics.transaction_applied,
                full_binding_transition_valid=full_transition,
                stable_base_unchanged=stable_base_unchanged,
                safe_route_boundary=safe_route_boundary,
                source_cache_matches=source_cache_matches,
                candidate_cache_recomputed=candidate_cache_recomputed,
                survivor_columns_bit_exact=survivors,
                changed_columns_scrubbed=scrubbed,
                optimizer_state_authenticated=optimizer_authenticated,
                routed_values_finite=routed_finite,
            ),
        )

    def commit_prepared_route(
        self,
        destination_adapter_state: CompositionalFeatureAdapterState,
        destination_consumer_state: CompositionalConsumerState,
        prepared: CompositionalConsumerPreparedRoute,
    ) -> CompositionalConsumerCommitResult:
        """Recompute and atomically adopt one adapter/consumer transaction."""

        if type(prepared) is not CompositionalConsumerPreparedRoute:
            raise TypeError("prepared must be an exact CompositionalConsumerPreparedRoute")
        if type(destination_consumer_state) is not CompositionalConsumerState:
            raise TypeError(
                "destination_consumer_state must be a CompositionalConsumerState"
            )
        proposal = prepared.adapter_proposal
        expected_adapter_proposal = self._adapter.prepare_update(
            proposal.source_state,
            proposal.observation,
            proposal.targets,
            context_id=proposal.context_id,
            curation_allowed=proposal.curation_allowed,
        )
        expected_route = self.prepare_route(
            prepared.source_state,
            prepared.post_update_state,
            expected_adapter_proposal,
        )
        route_integrity = _tree_bits_equal(prepared, expected_route)
        consumer_source_matches = _tree_bits_equal(
            destination_consumer_state,
            prepared.source_state,
        )
        destination_consumers_valid = self.state_valid(
            destination_consumer_state,
            destination_adapter_state,
        )
        consumers_ready = (
            route_integrity
            & consumer_source_matches
            & destination_consumers_valid
            & expected_route.receipt.consumers_ready
        )
        adapter_result = self._adapter.commit_prepared_update(
            destination_adapter_state,
            proposal,
            consumers_ready=consumers_ready,
        )
        applied = consumers_ready & adapter_result.diagnostics.applied
        consumer_state = cast(
            CompositionalConsumerState,
            jax.lax.cond(
                applied,
                lambda: expected_route.candidate_state,
                lambda: destination_consumer_state,
            ),
        )
        return CompositionalConsumerCommitResult(
            adapter_state=adapter_result.state,
            consumer_state=consumer_state,
            diagnostics=CompositionalConsumerCommitDiagnostics(
                route_integrity=route_integrity,
                consumer_source_matches=consumer_source_matches,
                destination_consumers_valid=destination_consumers_valid,
                consumers_ready=consumers_ready,
                adapter_applied=adapter_result.diagnostics.applied,
                applied=applied,
                rejected=~applied,
            ),
        )

    def _count_managed_feature_scalars(
        self,
        oak_state: OaKState,
        horde_state: MultiHeadMLPState | None,
    ) -> int:
        stomp = oak_state.stomp_state
        base = stomp.base_learner_state
        total = sum(weight.size for weight in base.head_params.weights)
        total += sum(pair[0].size for pair in base.head_traces)
        total += int(stomp.option_policies.q_weights.size)
        total += int(stomp.option_policies.traces.size)
        total += int(stomp.option_models.next_state_weights.size)
        # ``base_last_obs`` is reconstructed by two adapter representation
        # calls and is disclosed separately from consumer column routing.
        total += int(stomp.option_start_obs.size)
        if horde_state is not None:
            total += sum(weight.size for weight in horde_state.head_params.weights)
            total += sum(pair[0].size for pair in horde_state.head_traces)
        return int(total)

    def _consumer_state_nbytes(self, state: CompositionalConsumerState) -> int:
        multi_heads = [state.oak_state.stomp_state.base_learner_state]
        if state.horde_state is not None:
            multi_heads.append(state.horde_state)
        return _tree_nbytes(state) + _host_float_adjustment(*multi_heads)

    def measure_state_nbytes(self, state: CompositionalConsumerState) -> int:
        """Measure every logical persistent numeric consumer leaf."""

        if type(state) is not CompositionalConsumerState:
            raise TypeError("state must be a CompositionalConsumerState")
        return self._consumer_state_nbytes(state)

    def measure_prepared_route_nbytes(
        self,
        prepared: CompositionalConsumerPreparedRoute,
    ) -> int:
        """Measure logical transient route bytes stably across outer JIT."""

        if type(prepared) is not CompositionalConsumerPreparedRoute:
            raise TypeError("prepared must be a CompositionalConsumerPreparedRoute")
        consumer_multi_heads = [
            prepared.source_state.oak_state.stomp_state.base_learner_state,
            prepared.post_update_state.oak_state.stomp_state.base_learner_state,
            prepared.candidate_state.oak_state.stomp_state.base_learner_state,
        ]
        if prepared.source_state.horde_state is not None:
            consumer_multi_heads.append(prepared.source_state.horde_state)
        if prepared.post_update_state.horde_state is not None:
            consumer_multi_heads.append(prepared.post_update_state.horde_state)
        if prepared.candidate_state.horde_state is not None:
            consumer_multi_heads.append(prepared.candidate_state.horde_state)
        return (
            _tree_nbytes(prepared)
            + _host_float_adjustment(*consumer_multi_heads)
            + _adapter_host_float_adjustment(prepared.adapter_proposal.source_state)
            + _adapter_host_float_adjustment(prepared.adapter_proposal.candidate_state)
        )

    def resource_budget(
        self,
        state: CompositionalConsumerState,
    ) -> CompositionalConsumerResourceBudget:
        """Return exact state bytes and fixed birth-route work."""

        if type(state) is not CompositionalConsumerState:
            raise TypeError("state must be a CompositionalConsumerState")
        binding_bytes = _tree_nbytes(state.binding)
        expected_binding_bytes = 32 * self._width + 44
        if binding_bytes != expected_binding_bytes:
            raise RuntimeError("binding byte formula drifted from the live state tree")
        oak_multi = state.oak_state.stomp_state.base_learner_state
        oak_bytes = _tree_nbytes(state.oak_state) + _host_float_adjustment(oak_multi)
        horde_bytes = 0
        if state.horde_state is not None:
            horde_bytes = _tree_nbytes(state.horde_state) + _host_float_adjustment(
                state.horde_state
            )
        pristine_oak_multi = self._oak_template.stomp_state.base_learner_state
        pristine_oak_bytes = _tree_nbytes(self._oak_template) + _host_float_adjustment(
            pristine_oak_multi
        )
        pristine_horde_bytes = 0
        if self._horde_template is not None:
            pristine_horde_bytes = _tree_nbytes(
                self._horde_template
            ) + _host_float_adjustment(self._horde_template)
        return CompositionalConsumerResourceBudget(
            schema=COMPOSITIONAL_CONSUMER_ROUTER_STATE_SCHEMA,
            mechanism_status=COMPOSITIONAL_CONSUMER_ROUTER_MECHANISM_STATUS,
            scientific_promotion_allowed=False,
            base_feature_dim=self._base_width,
            n_features=self._width,
            horde_enabled=self._has_horde,
            horde_heads=(
                len(self._horde_template.head_params.weights)
                if self._horde_template is not None
                else 0
            ),
            binding_persistent_nbytes=binding_bytes,
            oak_persistent_nbytes=oak_bytes,
            horde_persistent_nbytes=horde_bytes,
            total_persistent_state_nbytes=binding_bytes + oak_bytes + horde_bytes,
            internal_pristine_oak_nbytes=pristine_oak_bytes,
            internal_pristine_horde_nbytes=pristine_horde_bytes,
            internal_pristine_template_nbytes=(
                pristine_oak_bytes + pristine_horde_bytes
            ),
            managed_feature_scalars=self._managed_feature_scalars,
            fixed_consumer_route_feature_scalar_evaluations=(
                3 * self._managed_feature_scalars
            ),
            fixed_cache_representation_calls_per_route=2,
            fixed_cache_representation_feature_slot_evaluations_per_route=(
                2 * self._width
            ),
            route_recomputations_per_commit=1,
            fixed_prepare_commit_consumer_route_feature_scalar_evaluations=(
                6 * self._managed_feature_scalars
            ),
            fixed_prepare_commit_cache_representation_calls=4,
            fixed_prepare_commit_cache_representation_feature_slot_evaluations=(
                4 * self._width
            ),
            fixed_slot_birth_word_comparisons=2 * self._width,
        )


__all__ = [
    "COMPOSITIONAL_CONSUMER_ROUTER_MECHANISM_STATUS",
    "COMPOSITIONAL_CONSUMER_ROUTER_SCIENTIFIC_PROMOTION_ALLOWED",
    "COMPOSITIONAL_CONSUMER_ROUTER_STATE_SCHEMA",
    "CompositionalConsumerCommitDiagnostics",
    "CompositionalConsumerCommitResult",
    "CompositionalConsumerPreparedRoute",
    "CompositionalConsumerReadyReceipt",
    "CompositionalConsumerResourceBudget",
    "CompositionalConsumerRouteDiagnostics",
    "CompositionalConsumerRouter",
    "CompositionalConsumerState",
]
