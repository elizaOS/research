"""Causal prototype features for supervised classification streams (Step 2).

Maintains one bounded, unit-norm prototype per class -- an EMA of normalized
observations of that class -- and exposes softmax-of-cosine-similarity
probabilities as constructed features.  The mechanism targets class-blocked
streams (long single-class blocks), where a learner must track the current
class without overwriting what it learned about earlier classes; the
prototype probabilities give downstream weights an explicit "which class
regime is this" signal that the raw observation does not carry.

The constructor is causal (prototypes are built only from past labelled
observations) and intentionally narrow: :meth:`~PrototypeFeatureConstructor.update`
fires only on non-negative unit-mass simplex targets (one-hot or soft
labels), so dense regression and non-simplex vector targets leave the
prototypes untouched.
"""

import functools

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float


@chex.dataclass(frozen=True)
class PrototypeFeatureState:
    """State for a fixed-budget prototype feature constructor."""

    prototypes: Float[Array, "n_classes feature_dim"]
    counts: Float[Array, " n_classes"]
    step_count: Array


class PrototypeFeatureConstructor:
    """One-prototype-per-class causal feature constructor.

    Args:
        n_classes: Number of simplex target classes/tasks.
        alpha: EMA rate for the observed class prototype; effective memory is
            roughly the last ``1 / alpha`` observations of that class
            (~20 at the 0.05 default).
        temperature: Softmax temperature for cosine-similarity features.
            Cosine similarity lies in ``[-1, 1]``, so logits span
            ``[-1/temperature, 1/temperature]``; the 0.05 default makes the
            output nearly one-hot toward the closest seen prototype.
    """

    def __init__(
        self,
        n_classes: int,
        alpha: float = 0.05,
        temperature: float = 0.05,
    ):
        if n_classes < 2:
            msg = f"n_classes must be >= 2, got {n_classes}"
            raise ValueError(msg)
        if not 0.0 < alpha <= 1.0:
            msg = f"alpha must be in (0, 1], got {alpha}"
            raise ValueError(msg)
        if temperature <= 0.0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self._n_classes = int(n_classes)
        self._alpha = float(alpha)
        self._temperature = float(temperature)

    @property
    def n_classes(self) -> int:
        """Number of prototype classes."""
        return self._n_classes

    def init(self, feature_dim: int) -> PrototypeFeatureState:
        """Return an empty prototype feature state."""
        if feature_dim < 1:
            msg = f"feature_dim must be >= 1, got {feature_dim}"
            raise ValueError(msg)
        return PrototypeFeatureState(  # type: ignore[call-arg]
            prototypes=jnp.zeros((self._n_classes, feature_dim), dtype=jnp.float32),
            counts=jnp.zeros((self._n_classes,), dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    @functools.partial(jax.jit, static_argnums=(0,))
    def features(self, state: PrototypeFeatureState, observation: Array) -> Array:
        """Construct prototype probability features for one observation."""
        obs = observation / (jnp.linalg.norm(observation) + 1e-8)
        prototype_norms = jnp.linalg.norm(state.prototypes, axis=1)
        normalized_prototypes = state.prototypes / (prototype_norms[:, None] + 1e-8)
        cosine = normalized_prototypes @ obs
        seen = state.counts > 0.0
        # Unseen classes get a finite sentinel logit rather than -inf: on a
        # fresh state (all classes unseen) softmax over all-(-inf) logits is
        # NaN, whereas all -20.0 yields the uniform distribution.  Once any
        # class has been seen, exp(-20) ~ 2e-9 makes unseen classes negligible.
        scores = jnp.where(
            seen,
            cosine / jnp.asarray(self._temperature, dtype=jnp.float32),
            -20.0,
        )
        return jax.nn.softmax(scores)

    @functools.partial(jax.jit, static_argnums=(0,))
    def augment(self, state: PrototypeFeatureState, observation: Array) -> Array:
        """Concatenate raw observation and prototype features."""
        return jnp.concatenate([observation, self.features(state, observation)])

    @functools.partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: PrototypeFeatureState,
        observation: Array,
        target: Array,
    ) -> PrototypeFeatureState:
        """Update the observed class prototype when the target is simplex-like.

        The gate accepts targets that are non-negative and sum to one within
        tolerance (one-hot or soft labels); NaN entries mark inactive target
        dimensions and are ignored.  Any other target (dense regression,
        general vectors) leaves prototypes and counts unchanged, though
        ``step_count`` still advances.
        """
        active = ~jnp.isnan(target)
        safe_target = jnp.where(active, target, 0.0)
        target_mass = jnp.sum(jnp.where(active, safe_target, 0.0))
        has_negative = jnp.any(jnp.logical_and(active, safe_target < -1e-6))
        simplex_like = (
            (~has_negative)
            & (target_mass > 1e-8)
            & (jnp.abs(target_mass - 1.0) <= 1e-5)
        )
        label = jnp.argmax(jnp.where(active, safe_target, -jnp.inf)).astype(jnp.int32)
        obs = observation / (jnp.linalg.norm(observation) + 1e-8)
        old = state.prototypes[label]
        alpha = jnp.asarray(self._alpha, dtype=jnp.float32)
        new = (1.0 - alpha) * old + alpha * obs
        new = new / (jnp.linalg.norm(new) + 1e-8)
        prototypes = state.prototypes.at[label].set(
            jnp.where(simplex_like, new, old)
        )
        counts = state.counts.at[label].set(
            state.counts[label] + simplex_like.astype(jnp.float32)
        )
        return PrototypeFeatureState(  # type: ignore[call-arg]
            prototypes=prototypes,
            counts=counts,
            step_count=state.step_count + 1,
        )
