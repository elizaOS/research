# mypy: disable-error-code="call-arg"
"""Stacked linear Horde: thousands of GVF demons as one batched TD update.

The existing Horde implementations (:mod:`~alberta_framework.core.horde`,
:mod:`~alberta_framework.core.independent_demon_horde`) unroll a Python loop
over demons inside ``jit``, so program size — and compile time and memory —
grow linearly with the demon count and throughput collapses well before the
"thousands of demons" regime the Horde architecture calls for (Sutton et al.
2011).  This module takes the other route for the linear case: every demon's
parameters live in one stacked array ``(n_demons, feature_dim)`` and the
entire Horde updates with a handful of batched array operations per step — no
per-demon loop exists anywhere, so program size is *constant* in the demon
count and the demon axis scales like any other array axis (and shards across
devices the same way).

Each demon d learns a linear GVF prediction ``v_d(x) = w_d @ x`` about the
shared behavior stream with its own question functions:

- cumulant ``c_d`` (selected from the observation/target channels or passed
  directly),
- pseudo-termination ``gamma_d``,
- trace decay ``lamda_d``,
- optional per-decision importance ratio ``rho`` for off-policy questions
  (per-decision IS with the ratio composed into the trace,
  ``z_d = rho * (gamma_d * lamda_d * z_d + x)``, matching the repo's
  post-fix off-policy convention).

The update is the textbook TD(lambda) per demon over shared features::

    delta_d = c_d + gamma_d * (w_d @ x') - (w_d @ x)
    z_d     = rho * (gamma_d * lamda_d * z_d + x)
    w_d    += alpha * delta_d * z_d

vectorized across all demons at once.  A ``NaN`` cumulant marks a demon
inactive for that step (its trace still decays; its weights freeze), matching
the NaN-masking convention of the loop-based hordes.
"""

from dataclasses import dataclass
from typing import Any

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

__all__ = [
    "StackedHordeConfig",
    "StackedHordeState",
    "StackedHordeUpdateResult",
    "StackedLinearHorde",
    "nexting_spec",
]


@dataclass(frozen=True)
class StackedHordeConfig:
    """Static configuration for :class:`StackedLinearHorde`.

    Attributes:
        n_demons: Number of GVF demons.
        feature_dim: Shared feature dimension.
        gammas: Per-demon pseudo-termination discounts, length ``n_demons``.
        lamdas: Per-demon trace decays, length ``n_demons``.
        cumulant_indices: Per-demon index into the cumulant-source vector
            handed to :meth:`StackedLinearHorde.update`.
        step_size: Shared TD step-size alpha.
    """

    n_demons: int
    feature_dim: int
    gammas: tuple[float, ...]
    lamdas: tuple[float, ...]
    cumulant_indices: tuple[int, ...]
    step_size: float = 0.05

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.n_demons < 1:
            raise ValueError("n_demons must be positive")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        for name, seq in (
            ("gammas", self.gammas),
            ("lamdas", self.lamdas),
            ("cumulant_indices", self.cumulant_indices),
        ):
            if len(seq) != self.n_demons:
                raise ValueError(f"{name} must have length n_demons={self.n_demons}")
        if any(not 0.0 <= g <= 1.0 for g in self.gammas):
            raise ValueError("every gamma must be in [0, 1]")
        if any(not 0.0 <= la <= 1.0 for la in self.lamdas):
            raise ValueError("every lamda must be in [0, 1]")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive")

    def to_config(self) -> dict[str, Any]:
        """Serialize this config to a dictionary."""
        return {
            "type": "StackedHordeConfig",
            "n_demons": self.n_demons,
            "feature_dim": self.feature_dim,
            "gammas": list(self.gammas),
            "lamdas": list(self.lamdas),
            "cumulant_indices": list(self.cumulant_indices),
            "step_size": self.step_size,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedHordeConfig":
        """Reconstruct a config from :meth:`to_config` output."""
        config = dict(config)
        config.pop("type", None)
        for key in ("gammas", "lamdas", "cumulant_indices"):
            config[key] = tuple(config[key])
        return cls(**config)


def nexting_spec(
    feature_dim: int,
    cumulant_indices: tuple[int, ...],
    gammas: tuple[float, ...] = (0.0, 0.5, 0.9, 0.99),
    lamda: float = 0.7,
    step_size: float = 0.05,
) -> StackedHordeConfig:
    """Build a nexting-style config: every cumulant at every timescale.

    Multi-timescale prediction of many sensor channels at once is the
    canonical Horde workload (Modayil, White & Sutton 2014, "nexting").  The
    returned config has ``len(cumulant_indices) * len(gammas)`` demons,
    ordered cumulant-major.

    Args:
        feature_dim: Shared feature dimension.
        cumulant_indices: Which cumulant channels to predict.
        gammas: Prediction timescales for each channel.
        lamda: Shared trace decay.
        step_size: Shared TD step-size.
    """
    idxs: list[int] = []
    gs: list[float] = []
    for c in cumulant_indices:
        for g in gammas:
            idxs.append(c)
            gs.append(g)
    n = len(idxs)
    return StackedHordeConfig(
        n_demons=n,
        feature_dim=feature_dim,
        gammas=tuple(gs),
        lamdas=(lamda,) * n,
        cumulant_indices=tuple(idxs),
        step_size=step_size,
    )


@chex.dataclass(frozen=True)
class StackedHordeState:
    """State for :class:`StackedLinearHorde`.

    Attributes:
        weights: Stacked demon weights, shape ``(n_demons, feature_dim)``.
        traces: Stacked eligibility traces, same shape.
        step_count: Number of updates applied.
    """

    weights: Float[Array, "n_demons feature_dim"]
    traces: Float[Array, "n_demons feature_dim"]
    step_count: Array


@chex.dataclass(frozen=True)
class StackedHordeUpdateResult:
    """Result of one stacked update.

    Attributes:
        state: Updated horde state.
        predictions: Pre-update predictions ``v_d(x)``, shape ``(n_demons,)``.
        td_errors: Per-demon TD errors (NaN where the demon was inactive).
    """

    state: StackedHordeState
    predictions: Float[Array, " n_demons"]
    td_errors: Float[Array, " n_demons"]


class StackedLinearHorde:
    """Batched linear TD(lambda) over a stacked demon axis (module docstring)."""

    def __init__(self, config: StackedHordeConfig):
        """Initialize the horde around a static config."""
        self._config = config
        self._gammas = jnp.asarray(config.gammas, dtype=jnp.float32)
        self._lamdas = jnp.asarray(config.lamdas, dtype=jnp.float32)
        self._cumulant_idx = jnp.asarray(config.cumulant_indices, dtype=jnp.int32)

    @property
    def config(self) -> StackedHordeConfig:
        """The static configuration."""
        return self._config

    def to_config(self) -> dict[str, Any]:
        """Serialize this horde to a dictionary."""
        return {"type": "StackedLinearHorde", "config": self._config.to_config()}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StackedLinearHorde":
        """Reconstruct a horde from :meth:`to_config` output."""
        return cls(StackedHordeConfig.from_config(dict(config)["config"]))

    def init(self) -> StackedHordeState:
        """Initialize zero weights and traces."""
        cfg = self._config
        shape = (cfg.n_demons, cfg.feature_dim)
        return StackedHordeState(
            weights=jnp.zeros(shape, dtype=jnp.float32),
            traces=jnp.zeros(shape, dtype=jnp.float32),
            step_count=jnp.array(0, dtype=jnp.int32),
        )

    def predict(self, state: StackedHordeState, features: Array) -> Array:
        """All demons' predictions for one feature vector, shape ``(n_demons,)``."""
        return state.weights @ features

    def update(
        self,
        state: StackedHordeState,
        features: Array,
        next_features: Array,
        cumulant_source: Array,
        rho: Array | float = 1.0,
    ) -> StackedHordeUpdateResult:
        """One batched TD(lambda) step for every demon at once.

        Args:
            state: Current horde state.
            features: Feature vector ``x_t``, shape ``(feature_dim,)``.
            next_features: Feature vector ``x_{t+1}``.
            cumulant_source: Vector the demons draw their cumulants from via
                ``config.cumulant_indices`` (e.g. the next observation, or a
                dedicated signal vector).  A ``NaN`` entry deactivates every
                demon reading it for this step: the demon's weights freeze
                and its trace only decays.
            rho: Per-decision importance ratio of the behavior action under
                each demon's target policy.  Scalar (shared) or shape
                ``(n_demons,)``.  On-policy questions use 1.0.

        Returns:
            :class:`StackedHordeUpdateResult` with per-demon predictions and
            TD errors (TD errors are NaN for inactive demons).
        """
        cfg = self._config
        cumulants = cumulant_source[self._cumulant_idx]  # (n_demons,)
        active = jnp.isfinite(cumulants)
        safe_cumulants = jnp.nan_to_num(cumulants)
        rho_vec = jnp.broadcast_to(jnp.asarray(rho, dtype=jnp.float32), (cfg.n_demons,))

        v = state.weights @ features  # (n_demons,)
        v_next = state.weights @ next_features
        td_errors = safe_cumulants + self._gammas * v_next - v

        # Per-decision IS with the ratio composed into the trace:
        # z_d = rho_d * (gamma_d * lamda_d * z_d + x).
        decay = (self._gammas * self._lamdas)[:, None]  # (n_demons, 1)
        accumulated = decay * state.traces + features[None, :]
        # Inactive demons: trace decays but the current gradient is withheld.
        decayed_only = decay * state.traces
        new_traces = rho_vec[:, None] * jnp.where(active[:, None], accumulated, decayed_only)

        masked_delta = jnp.where(active, td_errors, 0.0)
        new_weights = state.weights + cfg.step_size * masked_delta[:, None] * new_traces

        new_state = StackedHordeState(
            weights=new_weights,
            traces=new_traces,
            step_count=state.step_count + 1,
        )
        return StackedHordeUpdateResult(
            state=new_state,
            predictions=v,
            td_errors=jnp.where(active, td_errors, jnp.nan),
        )


def run_stacked_horde_scan(
    horde: StackedLinearHorde,
    state: StackedHordeState,
    features: Float[Array, "num_steps feature_dim"],
    cumulant_sources: Float[Array, "num_steps source_dim"],
    rhos: Float[Array, " num_steps"] | None = None,
) -> tuple[StackedHordeState, Float[Array, "num_steps n_demons"]]:
    """Scan the horde over transition arrays.

    ``features[t]`` and ``features[t+1]`` form each transition; the final
    row only provides a bootstrap target, so ``num_steps - 1`` updates run.

    Args:
        horde: The stacked horde.
        state: Initial state.
        features: Feature vectors per step.
        cumulant_sources: Cumulant-source vectors per transition (row ``t``
            is consumed by the ``t -> t+1`` update).
        rhos: Optional per-transition importance ratios (shared across
            demons); defaults to on-policy 1.0.

    Returns:
        ``(final_state, td_errors)`` with td_errors of shape
        ``(num_steps - 1, n_demons)``.
    """
    num_steps = features.shape[0]
    if rhos is None:
        rhos = jnp.ones((num_steps,), dtype=jnp.float32)

    def step_fn(
        carry: StackedHordeState,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[StackedHordeState, Array]:
        x, x_next, c, rho = inputs
        result = horde.update(carry, x, x_next, c, rho)
        return result.state, result.td_errors

    final_state, td_errors = jax.lax.scan(
        step_fn,
        state,
        (features[:-1], features[1:], cumulant_sources[:-1], rhos[:-1]),
    )
    return final_state, td_errors
