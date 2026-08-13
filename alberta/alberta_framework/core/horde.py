"""Horde learner: GVF demons sharing a trunk (Sutton et al. 2011).

Wraps ``MultiHeadMLPLearner`` to add:
- Per-demon gamma/lambda via ``HordeSpec``
- TD target computation for temporal demons (gamma > 0)
- GVF metadata and typed update results

Architecture decision: the trunk has no temporal trace decay (gamma=0).
Per-demon gamma/lambda applies only to heads. This avoids the
trace-error coupling problem: ``MultiHeadMLPLearner``'s VJP backward
pass folds per-head errors into the trunk cotangent *before* trace
accumulation, so trunk traces accumulate error-weighted gradients.
With trunk gamma=0, traces reset each step and this is correct.
If trunk gamma*lamda > 0, traces would carry biased error-gradient
products across steps, violating forward-view equivalence (Sutton &
Barto Ch. 12). This also avoids O(n_heads x trunk_params) memory
for per-demon trunk traces.

Reference: Sutton et al. 2011, "Horde: A Scalable Real-time Architecture
for Learning Knowledge from Unsupervised Sensorimotor Interaction"
"""

import dataclasses
import functools
import time
from collections.abc import Mapping
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Float, UInt

from alberta_framework.core.multi_head_learner import (
    MULTI_HEAD_MLP_STATE_SCHEMA,
    AnyOptimizer,
    MultiHeadMLPLearner,
    MultiHeadMLPState,
    migrate_legacy_multi_head_mlp_state,
)
from alberta_framework.core.normalizers import (
    EMANormalizerState,
    Normalizer,
    WelfordNormalizerState,
    _checked_lifetime_words_increment,
    _lifetime_counter_valid,
    _saturating_int32_counter_increment,
)
from alberta_framework.core.optimizers import Bounder
from alberta_framework.core.types import HordeSpec, TraceMode

MIXED_HORDE_STATE_SCHEMA = "alberta.mixed-horde-state.v2"
MIXED_HORDE_LIFETIME_COUNTER_NBYTES = 12
MIXED_HORDE_LIFETIME_COUNTER_DELTA_NBYTES = 8


def _tree_arrays_finite(tree: Any) -> Bool[Array, ""]:
    """Return whether every floating/complex persistent array is finite."""

    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.inexact):
            valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def _transition_source_valid(
    observation: Array,
    cumulants: Array,
    next_observation: Array,
) -> Bool[Array, ""]:
    """Validate one transition while retaining NaN-as-inactive cumulants."""

    observations_valid = jnp.all(jnp.isfinite(observation)) & jnp.all(
        jnp.isfinite(next_observation)
    )
    cumulants_valid = jnp.all(jnp.isfinite(cumulants) | jnp.isnan(cumulants))
    return observations_valid & cumulants_valid

# =============================================================================
# Types
# =============================================================================


@chex.dataclass(frozen=True)
class HordeUpdateResult:
    """Result of a single Horde update step.

    Attributes:
        state: Updated multi-head MLP learner state
        predictions: Predictions from all demons, shape ``(n_demons,)``
        td_errors: TD errors (target - prediction), shape ``(n_demons,)``.
            NaN for inactive demons.
        td_targets: Computed TD targets ``r + gamma * V(s')``,
            shape ``(n_demons,)``. NaN for inactive demons.
        per_demon_metrics: Per-demon metrics, shape ``(n_demons, 3)``.
            Columns: ``[squared_error, raw_error, mean_step_size]``.
            NaN for inactive demons.
        trunk_bounding_metric: Scalar trunk bounding metric
        pre_step_words: Exact child lifetime identity before the transaction.
            ``None`` for Horde implementations without the shared child clock.
        post_step_words: Exact child lifetime identity after the transaction.
            ``None`` for Horde implementations without the shared child clock.
        lifetime_counter_valid: Whether the child and nested counters were
            structurally valid and aligned before the transaction.
        lifetime_capacity_available: Whether every exact child clock had room.
        normalizer_counter_aligned: Whether the configured normalizer clock
            matched the learner clock.
        normalizer_estimator_capacity_available: Whether the configured
            estimator could honestly accept another sample.
        child_counters_aligned: Whether every exact child clock was aligned
            with its wrapper before the transaction.
        source_valid: Whether the transition inputs were finite, except for
            the documented NaN-as-inactive cumulant sentinel.
        candidate_valid: Whether every persistent floating candidate array
            was finite and every exact child clock reached the proposed word.
        state_valid: Whether the supplied persistent state arrays and static
            route structure were valid before candidate construction.
        update_applied: Whether the complete child transaction committed.

        The transaction fields remain optional for compatibility with older
        third-party Horde implementations.  The shared, independent, and mixed
        implementations in this package populate the fields they can support;
        the exact-clock wrappers populate every field.
    """

    state: Any
    predictions: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]
    td_targets: Float[Array, " n_demons"]
    per_demon_metrics: Float[Array, "n_demons 3"]
    trunk_bounding_metric: Float[Array, ""]
    pre_step_words: UInt[Array, " 2"] | None = None
    post_step_words: UInt[Array, " 2"] | None = None
    lifetime_counter_valid: Bool[Array, ""] | None = None
    lifetime_capacity_available: Bool[Array, ""] | None = None
    normalizer_counter_aligned: Bool[Array, ""] | None = None
    normalizer_estimator_capacity_available: Bool[Array, ""] | None = None
    child_counters_aligned: Bool[Array, ""] | None = None
    source_valid: Bool[Array, ""] | None = None
    candidate_valid: Bool[Array, ""] | None = None
    state_valid: Bool[Array, ""] | None = None
    update_applied: Bool[Array, ""] | None = None


@chex.dataclass(frozen=True)
class HordeLearningResult:
    """Result from a Horde scan-based learning loop.

    Attributes:
        state: Final multi-head MLP learner state
        per_demon_metrics: Per-demon metrics over time,
            shape ``(num_steps, n_demons, 3)``
        td_errors: TD errors over time, shape ``(num_steps, n_demons)``
    """

    state: MultiHeadMLPState
    per_demon_metrics: Float[Array, "num_steps n_demons 3"]
    td_errors: Float[Array, "num_steps n_demons"]


@chex.dataclass(frozen=True)
class BatchedHordeResult:
    """Result from batched Horde learning loop.

    Attributes:
        states: Batched multi-head MLP learner states
        per_demon_metrics: Per-demon metrics,
            shape ``(n_seeds, num_steps, n_demons, 3)``
        td_errors: TD errors, shape ``(n_seeds, num_steps, n_demons)``
    """

    states: MultiHeadMLPState
    per_demon_metrics: Float[Array, "n_seeds num_steps n_demons 3"]
    td_errors: Float[Array, "n_seeds num_steps n_demons"]


@chex.dataclass(frozen=True)
class MixedHordeState:
    """State for a mixed shared/independent Horde.

    ``step_count`` is saturating int32 compatibility telemetry.  The exact
    finite lifetime identity is the big-endian uint32 pair ``step_words``.
    Every configured route owns the same exact event identity; a mixed update
    commits only if all route candidates reach the next identity together.
    """

    shared_state: MultiHeadMLPState | None
    independent_state: Any | None
    step_count: Array = None  # type: ignore[assignment]
    step_words: UInt[Array, " 2"] = None  # type: ignore[assignment]
    birth_timestamp: float = 0.0
    uptime_s: float = 0.0


@chex.dataclass(frozen=True)
class MixedHordeLearningResult:
    """Result from a mixed Horde scan-based learning loop."""

    state: MixedHordeState
    per_demon_metrics: Float[Array, "num_steps n_demons 3"]
    td_errors: Float[Array, "num_steps n_demons"]


# =============================================================================
# HordeLearner
# =============================================================================


class HordeLearner:
    """Horde: GVF demons sharing a trunk (Sutton et al. 2011).

    Wraps ``MultiHeadMLPLearner``. Adds:
    - Per-demon gamma/lambda from ``HordeSpec``
    - TD target computation for temporal demons (gamma > 0)
    - GVF metadata

    The trunk uses gamma=0, lamda=0 (no temporal trace decay on shared
    features). Each head uses its own ``gamma * lambda`` product for
    trace decay, set via ``per_head_gamma_lamda`` on the inner learner.

    For all-gamma=0 Hordes (e.g. rlsecd's 5 prediction heads), this
    produces identical results to ``MultiHeadMLPLearner`` since the
    TD target reduces to just the cumulant.

    Single-Step (Daemon) Usage
    --------------------------
    Both ``predict()`` and ``update()`` work with single unbatched
    observations (1D arrays). JIT-compiled automatically.

    Attributes:
        horde_spec: The HordeSpec defining all demons
        n_demons: Number of demons (heads)
    """

    def __init__(
        self,
        horde_spec: HordeSpec,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
        utility_decay: float = 0.99,
    ):
        """Initialize the Horde learner.

        Args:
            horde_spec: Specification of all GVF demons
            hidden_sizes: Tuple of hidden layer sizes (default: two layers of 128)
            optimizer: Optimizer for weight updates. Defaults to LMS(step_size).
            step_size: Base learning rate (used only when optimizer is None)
            bounder: Optional update bounder (e.g. ObGDBounding)
            normalizer: Optional feature normalizer
            sparsity: Fraction of weights zeroed out per neuron (default: 0.9)
            leaky_relu_slope: Negative slope for LeakyReLU (default: 0.01)
            use_layer_norm: Whether to apply parameterless layer normalization
            head_optimizer: Optional separate optimizer for heads
            trace_mode: Eligibility trace mode (ACCUMULATING or REPLACING)
            utility_decay: EMA decay for hidden-unit utility diagnostics.
        """
        self._horde_spec = horde_spec
        self._hidden_sizes = hidden_sizes
        self._step_size = step_size
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm
        self._trace_mode = trace_mode
        self._utility_decay = utility_decay

        # Compute per-head gamma*lambda products
        per_head_gl = tuple(
            float(d.gamma * d.lamda) for d in horde_spec.demons
        )

        self._learner = MultiHeadMLPLearner(
            n_heads=len(horde_spec.demons),
            hidden_sizes=hidden_sizes,
            optimizer=optimizer,
            step_size=step_size,
            bounder=bounder,
            gamma=0.0,  # trunk: no trace decay
            lamda=0.0,
            normalizer=normalizer,
            sparsity=sparsity,
            leaky_relu_slope=leaky_relu_slope,
            use_layer_norm=use_layer_norm,
            head_optimizer=head_optimizer,
            per_head_gamma_lamda=per_head_gl,
            trace_mode=trace_mode,
            utility_decay=utility_decay,
        )

    @property
    def horde_spec(self) -> HordeSpec:
        """The HordeSpec defining all demons."""
        return self._horde_spec

    @property
    def n_demons(self) -> int:
        """Number of demons (heads)."""
        return len(self._horde_spec.demons)

    @property
    def learner(self) -> MultiHeadMLPLearner:
        """The underlying MultiHeadMLPLearner."""
        return self._learner

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration to dict.

        Returns:
            Dict with horde_spec and all MultiHeadMLPLearner constructor args.
        """
        learner_config = self._learner.to_config()
        # Remove fields managed by HordeLearner
        learner_config.pop("type", None)
        learner_config.pop("n_heads", None)
        learner_config.pop("gamma", None)
        learner_config.pop("lamda", None)
        learner_config.pop("per_head_gamma_lamda", None)
        # trace_mode is managed by HordeLearner, already in learner_config

        return {
            "type": "HordeLearner",
            "horde_spec": self._horde_spec.to_config(),
            **learner_config,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HordeLearner":
        """Reconstruct from config dict.

        Args:
            config: Dict as produced by ``to_config()``

        Returns:
            Reconstructed HordeLearner
        """
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        config = dict(config)
        config.pop("type", None)
        state_schema = config.pop("state_schema", MULTI_HEAD_MLP_STATE_SCHEMA)
        if state_schema != MULTI_HEAD_MLP_STATE_SCHEMA:
            raise ValueError(f"unsupported Horde state schema: {state_schema!r}")

        horde_spec = HordeSpec.from_config(config.pop("horde_spec"))
        optimizer = optimizer_from_config(config.pop("optimizer"))
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = (
            normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        )
        head_opt_cfg = config.pop("head_optimizer", None)
        head_optimizer = (
            optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
        )

        trace_mode_str = config.pop("trace_mode", None)
        trace_mode = (
            TraceMode(trace_mode_str) if trace_mode_str is not None else TraceMode.ACCUMULATING
        )

        return cls(
            horde_spec=horde_spec,
            hidden_sizes=tuple(config.pop("hidden_sizes")),
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            **config,
        )

    def init(self, feature_dim: int, key: Array) -> MultiHeadMLPState:
        """Initialize Horde learner state.

        Args:
            feature_dim: Dimension of the input feature vector
            key: JAX random key for weight initialization

        Returns:
            Initial MultiHeadMLPState
        """
        return self._learner.init(feature_dim, key)

    @functools.partial(jax.jit, static_argnums=(0,))
    def predict(self, state: MultiHeadMLPState, observation: Array) -> Array:
        """Compute predictions from all demons.

        Args:
            state: Current learner state
            observation: Input feature vector

        Returns:
            Array of shape ``(n_demons,)`` with one prediction per demon
        """
        return self._learner.predict(state, observation)  # type: ignore[no-any-return]

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
    ) -> HordeUpdateResult:
        """Update Horde given observation, cumulants, and next observation.

        Computes TD targets ``r + gamma * V(s')`` for each demon, then
        delegates to ``MultiHeadMLPLearner.update()``. For gamma=0 demons,
        the target equals the cumulant.

        Args:
            state: Current state
            observation: Input feature vector, shape ``(feature_dim,)``
            cumulants: Per-demon pseudo-rewards, shape ``(n_demons,)``.
                NaN = inactive demon.
            next_observation: Next feature vector, shape ``(feature_dim,)``.
                Used for V(s') bootstrapping. For all-gamma=0 Hordes,
                this is required but doesn't affect results.

        Returns:
            HordeUpdateResult with updated state, predictions, TD errors,
            TD targets, and per-demon metrics
        """
        # 1. Compute V(s') for bootstrapping
        next_preds = self._learner.predict(state, next_observation)

        # 2. TD targets: r + gamma * V(s')
        # For gamma=0 demons: target = cumulant (single-step prediction)
        # NaN cumulants stay NaN (inactive demons)
        gammas = self._horde_spec.gammas
        targets = cumulants + gammas * next_preds

        # 3. Delegate to MultiHeadMLPLearner
        result = self._learner.update(state, observation, targets)

        return HordeUpdateResult(  # type: ignore[call-arg]
            state=result.state,
            predictions=result.predictions,
            td_errors=result.errors,
            td_targets=targets,
            per_demon_metrics=result.per_head_metrics,
            trunk_bounding_metric=result.trunk_bounding_metric,
            pre_step_words=result.pre_step_words,
            post_step_words=result.post_step_words,
            lifetime_counter_valid=result.lifetime_counter_valid,
            lifetime_capacity_available=result.lifetime_capacity_available,
            normalizer_counter_aligned=result.normalizer_counter_aligned,
            normalizer_estimator_capacity_available=(
                result.normalizer_estimator_capacity_available
            ),
            update_applied=result.update_applied,
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def update_with_discounts(
        self,
        state: MultiHeadMLPState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
        discounts: Array,
    ) -> HordeUpdateResult:
        """Update Horde with explicit per-demon transition discounts.

        This is the same TD update as :meth:`update`, except callers supply the
        effective discount vector for this transition. It lets control adapters
        zero the value head at episode boundaries while keeping the Horde's
        fixed GVF metadata and per-head trace decay intact.

        Args:
            state: Current state.
            observation: Input feature vector, shape ``(feature_dim,)``.
            cumulants: Per-demon pseudo-rewards, shape ``(n_demons,)``.
                NaN = inactive demon.
            next_observation: Next feature vector, shape ``(feature_dim,)``.
            discounts: Effective per-demon discounts for this transition,
                shape ``(n_demons,)``.

        Returns:
            HordeUpdateResult with updated state and TD metrics.
        """
        next_preds = self._learner.predict(state, next_observation)
        discounts = jnp.asarray(discounts, dtype=jnp.float32)
        targets = cumulants + discounts * next_preds
        result = self._learner.update(state, observation, targets)

        return HordeUpdateResult(  # type: ignore[call-arg]
            state=result.state,
            predictions=result.predictions,
            td_errors=result.errors,
            td_targets=targets,
            per_demon_metrics=result.per_head_metrics,
            trunk_bounding_metric=result.trunk_bounding_metric,
            pre_step_words=result.pre_step_words,
            post_step_words=result.post_step_words,
            lifetime_counter_valid=result.lifetime_counter_valid,
            lifetime_capacity_available=result.lifetime_capacity_available,
            normalizer_counter_aligned=result.normalizer_counter_aligned,
            normalizer_estimator_capacity_available=(
                result.normalizer_estimator_capacity_available
            ),
            update_applied=result.update_applied,
        )


# =============================================================================
# Mixed Horde
# =============================================================================


class MixedHorde:
    """Route demons to shared or independent Horde implementations.

    Demons with ``gamma * lambda == 0`` use the shared-trunk
    :class:`HordeLearner`; demons with temporal traces use
    ``IndependentDemonHorde`` so nonlinear trunk traces remain forward-view
    correct. Public predictions, targets, and metrics are returned in the
    original demon order.
    """

    def __init__(
        self,
        horde_spec: HordeSpec,
        hidden_sizes: tuple[int, ...] = (128, 128),
        optimizer: AnyOptimizer | None = None,
        step_size: float = 1.0,
        bounder: Bounder | None = None,
        normalizer: (
            Normalizer[EMANormalizerState] | Normalizer[WelfordNormalizerState] | None
        ) = None,
        sparsity: float = 0.9,
        leaky_relu_slope: float = 0.01,
        use_layer_norm: bool = True,
        head_optimizer: AnyOptimizer | None = None,
        trace_mode: TraceMode = TraceMode.ACCUMULATING,
    ):
        from alberta_framework.core.independent_demon_horde import (
            IndependentDemonHorde,
        )

        self._horde_spec = horde_spec
        self._hidden_sizes = hidden_sizes
        self._optimizer = optimizer
        self._step_size = step_size
        self._bounder = bounder
        self._normalizer = normalizer
        self._sparsity = sparsity
        self._leaky_relu_slope = leaky_relu_slope
        self._use_layer_norm = use_layer_norm
        self._head_optimizer = head_optimizer
        self._trace_mode = trace_mode

        self._shared_indices = tuple(
            i for i, d in enumerate(horde_spec.demons) if float(d.gamma * d.lamda) == 0.0
        )
        self._independent_indices = tuple(
            i for i, d in enumerate(horde_spec.demons) if float(d.gamma * d.lamda) != 0.0
        )

        common_kwargs: dict[str, Any] = {
            "hidden_sizes": hidden_sizes,
            "optimizer": optimizer,
            "step_size": step_size,
            "bounder": bounder,
            "normalizer": normalizer,
            "sparsity": sparsity,
            "leaky_relu_slope": leaky_relu_slope,
            "use_layer_norm": use_layer_norm,
            "head_optimizer": head_optimizer,
            "trace_mode": trace_mode,
        }
        self._shared_horde = (
            HordeLearner(
                horde_spec=self._subset_spec(self._shared_indices),
                **common_kwargs,
            )
            if self._shared_indices
            else None
        )
        self._independent_horde = (
            IndependentDemonHorde(
                horde_spec=self._subset_spec(self._independent_indices),
                **common_kwargs,
            )
            if self._independent_indices
            else None
        )

    @property
    def horde_spec(self) -> HordeSpec:
        """The full HordeSpec in original demon order."""
        return self._horde_spec

    @property
    def n_demons(self) -> int:
        """Number of demons."""
        return len(self._horde_spec.demons)

    @property
    def shared_indices(self) -> tuple[int, ...]:
        """Original demon indices routed to the shared Horde."""
        return self._shared_indices

    @property
    def independent_indices(self) -> tuple[int, ...]:
        """Original demon indices routed to independent demons."""
        return self._independent_indices

    @property
    def shared_horde(self) -> HordeLearner | None:
        """Shared-trunk learner, if any demons route there."""
        return self._shared_horde

    @property
    def independent_horde(self) -> Any | None:
        """Independent-demon learner, if any demons route there."""
        return self._independent_horde

    def _subset_spec(self, indices: tuple[int, ...]) -> HordeSpec:
        return HordeSpec(
            demons=tuple(self._horde_spec.demons[i] for i in indices),
            gammas=self._horde_spec.gammas[jnp.asarray(indices, dtype=jnp.int32)],
            lamdas=self._horde_spec.lamdas[jnp.asarray(indices, dtype=jnp.int32)],
        )

    def to_config(self) -> dict[str, Any]:
        """Serialize learner configuration to dict."""
        return {
            "type": "MixedHorde",
            "state_schema": MIXED_HORDE_STATE_SCHEMA,
            "horde_spec": self._horde_spec.to_config(),
            "hidden_sizes": list(self._hidden_sizes),
            "optimizer": (
                self._optimizer.to_config() if self._optimizer is not None else None
            ),
            "bounder": (
                self._bounder.to_config() if self._bounder is not None else None
            ),
            "normalizer": (
                self._normalizer.to_config() if self._normalizer is not None else None
            ),
            "head_optimizer": (
                self._head_optimizer.to_config()
                if self._head_optimizer is not None
                else None
            ),
            "step_size": self._step_size,
            "sparsity": self._sparsity,
            "leaky_relu_slope": self._leaky_relu_slope,
            "use_layer_norm": self._use_layer_norm,
            "trace_mode": self._trace_mode.value,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MixedHorde":
        """Reconstruct from config dict."""
        from alberta_framework.core.normalizers import normalizer_from_config
        from alberta_framework.core.optimizers import (
            bounder_from_config,
            optimizer_from_config,
        )

        config = dict(config)
        config.pop("type", None)
        state_schema = config.pop("state_schema", MIXED_HORDE_STATE_SCHEMA)
        if state_schema != MIXED_HORDE_STATE_SCHEMA:
            raise ValueError(f"Unsupported mixed Horde state schema: {state_schema!r}")
        horde_spec = HordeSpec.from_config(config.pop("horde_spec"))
        opt_cfg = config.pop("optimizer", None)
        optimizer = optimizer_from_config(opt_cfg) if opt_cfg is not None else None
        bounder_cfg = config.pop("bounder", None)
        bounder = bounder_from_config(bounder_cfg) if bounder_cfg is not None else None
        normalizer_cfg = config.pop("normalizer", None)
        normalizer = (
            normalizer_from_config(normalizer_cfg) if normalizer_cfg is not None else None
        )
        head_opt_cfg = config.pop("head_optimizer", None)
        head_optimizer = (
            optimizer_from_config(head_opt_cfg) if head_opt_cfg is not None else None
        )
        trace_mode_str = config.pop("trace_mode", None)
        trace_mode = (
            TraceMode(trace_mode_str) if trace_mode_str is not None else TraceMode.ACCUMULATING
        )
        return cls(
            horde_spec=horde_spec,
            hidden_sizes=tuple(config.pop("hidden_sizes")),
            optimizer=optimizer,
            bounder=bounder,
            normalizer=normalizer,
            head_optimizer=head_optimizer,
            trace_mode=trace_mode,
            **config,
        )

    def init(self, feature_dim: int, key: Array) -> MixedHordeState:
        """Initialize mixed Horde state."""
        if self._shared_horde is not None and self._independent_horde is not None:
            shared_key, independent_key = jax.random.split(key)
        else:
            shared_key = independent_key = key
        shared_state = (
            self._shared_horde.init(feature_dim, shared_key)
            if self._shared_horde is not None
            else None
        )
        independent_state = (
            self._independent_horde.init(feature_dim, independent_key)
            if self._independent_horde is not None
            else None
        )
        return MixedHordeState(
            shared_state=shared_state,
            independent_state=independent_state,
            step_count=jnp.array(0, dtype=jnp.int32),
            step_words=jnp.zeros((2,), dtype=jnp.uint32),
            birth_timestamp=time.time(),
            uptime_s=0.0,
        )

    def predict(self, state: MixedHordeState, observation: Array) -> Array:
        """Compute predictions in original demon order."""
        preds = jnp.full((self.n_demons,), jnp.nan, dtype=jnp.float32)
        if self._shared_horde is not None:
            shared_state = state.shared_state
            shared_preds = self._shared_horde.predict(shared_state, observation)
            preds = preds.at[jnp.asarray(self._shared_indices, dtype=jnp.int32)].set(
                shared_preds
            )
        if self._independent_horde is not None:
            independent_preds = self._independent_horde.predict(
                state.independent_state, observation
            )
            preds = preds.at[
                jnp.asarray(self._independent_indices, dtype=jnp.int32)
            ].set(independent_preds)
        return preds

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: MixedHordeState,
        observation: Array,
        cumulants: Array,
        next_observation: Array,
    ) -> HordeUpdateResult:
        """Atomically update every route and return original demon ordering.

        Shared and independent route candidates are computed from the same
        pre-state, then committed as one transaction.  A malformed/exhausted
        wrapper, a misaligned child clock, an invalid source, a refused child,
        or a non-finite candidate leaves the complete mixed state unchanged.
        """
        if getattr(observation, "shape", None) != getattr(
            next_observation, "shape", None
        ):
            raise ValueError("observation and next_observation shapes must match")
        if getattr(cumulants, "shape", None) != (self.n_demons,):
            raise ValueError(f"cumulants must have shape ({self.n_demons},)")

        outer_valid = _lifetime_counter_valid(state.step_words, state.step_count)
        proposed_words, outer_capacity = _checked_lifetime_words_increment(
            state.step_words
        )
        source_valid = _transition_source_valid(
            observation,
            cumulants,
            next_observation,
        )
        state_valid = _tree_arrays_finite(state)

        predictions = jnp.full((self.n_demons,), jnp.nan, dtype=jnp.float32)
        td_errors = jnp.full((self.n_demons,), jnp.nan, dtype=jnp.float32)
        td_targets = jnp.full((self.n_demons,), jnp.nan, dtype=jnp.float32)
        per_demon_metrics = jnp.full((self.n_demons, 3), jnp.nan, dtype=jnp.float32)
        trunk_bounding_metric = jnp.array(1.0, dtype=jnp.float32)
        candidate_shared_state = state.shared_state
        candidate_independent_state = state.independent_state
        children_pre_aligned = jnp.asarray(True, dtype=jnp.bool_)
        children_post_aligned = jnp.asarray(True, dtype=jnp.bool_)
        child_counter_valid = jnp.asarray(True, dtype=jnp.bool_)
        child_capacity = jnp.asarray(True, dtype=jnp.bool_)
        normalizer_aligned = jnp.asarray(True, dtype=jnp.bool_)
        estimator_capacity = jnp.asarray(True, dtype=jnp.bool_)
        children_applied = jnp.asarray(True, dtype=jnp.bool_)

        if self._shared_horde is not None:
            if state.shared_state is None:
                raise ValueError("mixed Horde shared route state is missing")
            idx = jnp.asarray(self._shared_indices, dtype=jnp.int32)
            shared_state = state.shared_state
            children_pre_aligned = children_pre_aligned & jnp.all(
                shared_state.step_words == state.step_words
            )
            shared_result = self._shared_horde.update(
                shared_state,
                observation,
                cumulants[idx],
                next_observation,
            )
            candidate_shared_state = cast(MultiHeadMLPState, shared_result.state)
            children_post_aligned = children_post_aligned & jnp.all(
                candidate_shared_state.step_words == proposed_words
            )
            child_counter_valid = child_counter_valid & jnp.asarray(
                shared_result.lifetime_counter_valid,
                dtype=jnp.bool_,
            )
            child_capacity = child_capacity & jnp.asarray(
                shared_result.lifetime_capacity_available,
                dtype=jnp.bool_,
            )
            normalizer_aligned = normalizer_aligned & jnp.asarray(
                shared_result.normalizer_counter_aligned,
                dtype=jnp.bool_,
            )
            estimator_capacity = estimator_capacity & jnp.asarray(
                shared_result.normalizer_estimator_capacity_available,
                dtype=jnp.bool_,
            )
            children_applied = children_applied & jnp.asarray(
                shared_result.update_applied,
                dtype=jnp.bool_,
            )
            predictions = predictions.at[idx].set(shared_result.predictions)
            td_errors = td_errors.at[idx].set(shared_result.td_errors)
            td_targets = td_targets.at[idx].set(shared_result.td_targets)
            per_demon_metrics = per_demon_metrics.at[idx].set(
                shared_result.per_demon_metrics
            )
            trunk_bounding_metric = shared_result.trunk_bounding_metric

        if self._independent_horde is not None:
            if state.independent_state is None:
                raise ValueError("mixed Horde independent route state is missing")
            idx = jnp.asarray(self._independent_indices, dtype=jnp.int32)
            independent_state = state.independent_state
            children_pre_aligned = children_pre_aligned & jnp.all(
                independent_state.step_words == state.step_words
            )
            independent_result = self._independent_horde.update(
                independent_state,
                observation,
                cumulants[idx],
                next_observation,
            )
            candidate_independent_state = independent_result.state
            children_post_aligned = children_post_aligned & jnp.all(
                candidate_independent_state.step_words == proposed_words
            )
            child_counter_valid = child_counter_valid & jnp.asarray(
                independent_result.lifetime_counter_valid,
                dtype=jnp.bool_,
            )
            child_capacity = child_capacity & jnp.asarray(
                independent_result.lifetime_capacity_available,
                dtype=jnp.bool_,
            )
            normalizer_aligned = normalizer_aligned & jnp.asarray(
                independent_result.normalizer_counter_aligned,
                dtype=jnp.bool_,
            )
            estimator_capacity = estimator_capacity & jnp.asarray(
                independent_result.normalizer_estimator_capacity_available,
                dtype=jnp.bool_,
            )
            children_applied = children_applied & jnp.asarray(
                independent_result.update_applied,
                dtype=jnp.bool_,
            )
            predictions = predictions.at[idx].set(independent_result.predictions)
            td_errors = td_errors.at[idx].set(independent_result.td_errors)
            td_targets = td_targets.at[idx].set(independent_result.td_targets)
            per_demon_metrics = per_demon_metrics.at[idx].set(
                independent_result.per_demon_metrics
            )

        candidate_state = MixedHordeState(
            shared_state=candidate_shared_state,
            independent_state=candidate_independent_state,
            step_count=_saturating_int32_counter_increment(state.step_count),
            step_words=proposed_words,
            birth_timestamp=state.birth_timestamp,
            uptime_s=state.uptime_s,
        )
        candidate_finite = _tree_arrays_finite(candidate_state)
        active_mask = ~jnp.isnan(cumulants)
        reported_values_valid = (
            jnp.all(jnp.isfinite(predictions))
            & jnp.all((~active_mask) | jnp.isfinite(td_errors))
            & jnp.all((~active_mask) | jnp.isfinite(td_targets))
            & jnp.all(
                (~active_mask[:, None]) | jnp.isfinite(per_demon_metrics)
            )
            & jnp.isfinite(trunk_bounding_metric)
        )
        candidate_valid = (
            children_post_aligned & candidate_finite & reported_values_valid
        )
        update_applied = (
            outer_valid
            & outer_capacity
            & source_valid
            & state_valid
            & children_pre_aligned
            & child_counter_valid
            & child_capacity
            & normalizer_aligned
            & estimator_capacity
            & children_applied
            & candidate_valid
        )
        new_state = jax.lax.cond(
            update_applied,
            lambda _: candidate_state,
            lambda _: state,
            operand=None,
        )
        return HordeUpdateResult(
            state=new_state,
            predictions=predictions,
            td_errors=td_errors,
            td_targets=td_targets,
            per_demon_metrics=per_demon_metrics,
            trunk_bounding_metric=trunk_bounding_metric,
            pre_step_words=state.step_words,
            post_step_words=new_state.step_words,
            lifetime_counter_valid=outer_valid & child_counter_valid,
            lifetime_capacity_available=outer_capacity & child_capacity,
            normalizer_counter_aligned=normalizer_aligned,
            normalizer_estimator_capacity_available=estimator_capacity,
            child_counters_aligned=children_pre_aligned,
            source_valid=source_valid,
            candidate_valid=candidate_valid,
            state_valid=state_valid,
            update_applied=update_applied,
        )


def measure_mixed_horde_state_nbytes(state: MixedHordeState) -> int:
    """Measure persistent JAX-array bytes for one concrete mixed state."""

    def measure(value: Any) -> int:
        if isinstance(value, Array):
            return int(value.size) * int(value.dtype.itemsize)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return sum(
                measure(getattr(value, field.name))
                for field in dataclasses.fields(value)
                if field.name not in {"birth_timestamp", "uptime_s"}
            )
        if isinstance(value, Mapping):
            return sum(measure(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(measure(item) for item in value)
        return 0

    return measure(state)


def mixed_horde_lifetime_counter_nbytes() -> int:
    """Return bytes owned by the mixed wrapper's exact lifetime identity."""

    return MIXED_HORDE_LIFETIME_COUNTER_NBYTES


def _host_state_mapping(state: Any, *, label: str) -> dict[str, Any]:
    """Return an exact shallow host mapping for migration."""

    if isinstance(state, Mapping):
        return dict(state)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return {
            field.name: getattr(state, field.name)
            for field in dataclasses.fields(state)
        }
    raise TypeError(f"legacy {label} state must be a mapping or dataclass")


def _legacy_unsaturated_step(fields: Mapping[str, Any], *, label: str) -> int:
    """Authenticate the only unambiguous region of one legacy int32 clock."""

    step_array = jnp.asarray(fields["step_count"])
    if step_array.shape != () or step_array.dtype != jnp.dtype(jnp.int32):
        raise TypeError(f"legacy {label} step_count must be scalar int32")
    step = int(step_array)
    if step < 0:
        raise ValueError(f"negative legacy {label} step_count indicates wrap")
    if step >= 2**31 - 1:
        raise ValueError(f"saturated legacy {label} step_count is ambiguous")
    return step


def _coerce_or_migrate_multi_head_state(state: Any) -> MultiHeadMLPState:
    fields = _host_state_mapping(state, label="shared Horde child")
    current_names = {
        field.name
        for field in dataclasses.fields(cast(Any, MultiHeadMLPState))
    }
    if set(fields) == current_names:
        return MultiHeadMLPState(**fields)
    return migrate_legacy_multi_head_mlp_state(fields)


def migrate_legacy_mixed_horde_state(legacy_state: Any) -> MixedHordeState:
    """Migrate an exact pre-v2 mixed wrapper without guessing wrapped clocks.

    Only non-negative, unsaturated legacy wrapper clocks are authenticable.
    Every present route must migrate to the same event identity; otherwise the
    historical state could already contain a partial mixed transaction and is
    rejected fail closed.
    """

    from alberta_framework.core.independent_demon_horde import (
        IndependentDemonHordeState,
        migrate_legacy_independent_demon_horde_state,
    )

    fields = _host_state_mapping(legacy_state, label="mixed Horde")
    current_names = {
        field.name for field in dataclasses.fields(cast(Any, MixedHordeState))
    }
    legacy_names = current_names - {"step_words"}
    supplied_names = set(fields)
    if supplied_names != legacy_names:
        missing = sorted(legacy_names - supplied_names)
        extra = sorted(supplied_names - legacy_names)
        raise ValueError(
            "legacy mixed Horde field manifest is not exact; "
            f"missing={missing}, extra={extra}"
        )
    step = _legacy_unsaturated_step(fields, label="mixed Horde")
    expected_words = jnp.asarray((0, step), dtype=jnp.uint32)

    shared = fields["shared_state"]
    independent = fields["independent_state"]
    if shared is None and independent is None:
        raise ValueError("legacy mixed Horde must contain at least one route")
    if shared is not None:
        shared = _coerce_or_migrate_multi_head_state(shared)
        if not bool(jnp.all(shared.step_words == expected_words)):
            raise ValueError("legacy mixed/shared route clocks are not aligned")
        if not bool(_lifetime_counter_valid(shared.step_words, shared.step_count)):
            raise ValueError("migrated shared route lifetime counter is invalid")
        fields["shared_state"] = shared
    if independent is not None:
        independent_fields = _host_state_mapping(
            independent,
            label="independent Horde child",
        )
        independent_current_names = {
            field.name
            for field in dataclasses.fields(cast(Any, IndependentDemonHordeState))
        }
        if set(independent_fields) == independent_current_names:
            independent = IndependentDemonHordeState(**independent_fields)
        else:
            independent = migrate_legacy_independent_demon_horde_state(
                independent_fields
            )
        if not bool(jnp.all(independent.step_words == expected_words)):
            raise ValueError("legacy mixed/independent route clocks are not aligned")
        if not bool(
            _lifetime_counter_valid(
                independent.step_words,
                independent.step_count,
            )
        ):
            raise ValueError("migrated independent route lifetime counter is invalid")
        fields["independent_state"] = independent

    fields["step_words"] = expected_words
    return MixedHordeState(**fields)


# =============================================================================
# Learning Loops
# =============================================================================


def run_horde_learning_loop(
    horde: HordeLearner,
    state: MultiHeadMLPState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
) -> HordeLearningResult:
    """Run Horde learning loop using ``jax.lax.scan``.

    Scans over ``(obs, cumulants, next_obs)`` triples.

    Args:
        horde: Horde learner
        state: Initial learner state
        observations: Input observations, shape ``(num_steps, feature_dim)``
        cumulants: Per-demon cumulants, shape ``(num_steps, n_demons)``.
            NaN = inactive demon for that step.
        next_observations: Next observations, shape ``(num_steps, feature_dim)``

    Returns:
        HordeLearningResult with final state, per-demon metrics, and TD errors
    """

    def step_fn(
        carry: MultiHeadMLPState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[MultiHeadMLPState, tuple[Array, Array]]:
        l_state = carry
        obs, cums, next_obs = inputs
        result = horde.update(l_state, obs, cums, next_obs)
        return result.state, (result.per_demon_metrics, result.td_errors)

    t0 = time.time()
    final_state, (per_demon_metrics, td_errors) = jax.lax.scan(
        step_fn, state, (observations, cumulants, next_observations)
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(uptime_s=final_state.uptime_s + elapsed)  # type: ignore[attr-defined]

    return HordeLearningResult(  # type: ignore[call-arg]
        state=final_state,
        per_demon_metrics=per_demon_metrics,
        td_errors=td_errors,
    )


def run_mixed_horde_learning_loop(
    horde: MixedHorde,
    state: MixedHordeState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
) -> MixedHordeLearningResult:
    """Run a mixed Horde learning loop using ``jax.lax.scan``."""

    def step_fn(
        carry: MixedHordeState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[MixedHordeState, tuple[Array, Array]]:
        obs, cums, next_obs = inputs
        result = horde.update(carry, obs, cums, next_obs)
        return result.state, (result.per_demon_metrics, result.td_errors)

    t0 = time.time()
    final_state, (per_demon_metrics, td_errors) = jax.lax.scan(
        step_fn, state, (observations, cumulants, next_observations)
    )
    elapsed = time.time() - t0
    final_state = final_state.replace(  # type: ignore[attr-defined]
        uptime_s=final_state.uptime_s + elapsed
    )
    return MixedHordeLearningResult(  # type: ignore[call-arg]
        state=final_state,
        per_demon_metrics=per_demon_metrics,
        td_errors=td_errors,
    )


def run_horde_learning_loop_final_state(
    horde: HordeLearner,
    state: MultiHeadMLPState,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
) -> MultiHeadMLPState:
    """Run a Horde scan and return only the final learner state.

    Throughput benchmarks use this helper to avoid materializing the full
    metrics trace when only the final state is needed.
    """

    def step_fn(
        carry: MultiHeadMLPState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[MultiHeadMLPState, None]:
        obs, cums, next_obs = inputs
        result = horde.update(carry, obs, cums, next_obs)
        return result.state, None

    t0 = time.time()
    final_state, _ = jax.lax.scan(
        step_fn,
        state,
        (observations, cumulants, next_observations),
    )
    elapsed = time.time() - t0
    return cast(
        MultiHeadMLPState,
        final_state.replace(uptime_s=final_state.uptime_s + elapsed),  # type: ignore[attr-defined]
    )


def run_horde_learning_loop_batched(
    horde: HordeLearner,
    observations: Array,
    cumulants: Array,
    next_observations: Array,
    keys: Array,
) -> BatchedHordeResult:
    """Run Horde learning loop across seeds using ``jax.vmap``.

    Each seed produces an independently initialized state. All seeds
    share the same observations, cumulants, and next observations.

    Args:
        horde: Horde learner
        observations: Shared observations, shape ``(num_steps, feature_dim)``
        cumulants: Shared cumulants, shape ``(num_steps, n_demons)``
        next_observations: Shared next observations,
            shape ``(num_steps, feature_dim)``
        keys: JAX random keys, shape ``(n_seeds,)`` or ``(n_seeds, 2)``

    Returns:
        BatchedHordeResult with batched states, per-demon metrics, and TD errors
    """
    feature_dim = observations.shape[1]

    def single_run(key: Array) -> tuple[MultiHeadMLPState, Array, Array]:
        init_state = horde.init(feature_dim, key)
        result = run_horde_learning_loop(
            horde, init_state, observations, cumulants, next_observations
        )
        return result.state, result.per_demon_metrics, result.td_errors

    t0 = time.time()
    batched_states, batched_metrics, batched_td_errors = jax.vmap(single_run)(keys)
    elapsed = time.time() - t0
    batched_states = batched_states.replace(  # type: ignore[attr-defined]
        uptime_s=batched_states.uptime_s + elapsed
    )

    return BatchedHordeResult(  # type: ignore[call-arg]
        states=batched_states,
        per_demon_metrics=batched_metrics,
        td_errors=batched_td_errors,
    )
