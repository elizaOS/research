"""Source-profiled first-order Utility-based Perturbed Gradient Descent.

This module implements the weight-wise UPGD update from Elsayed and Mahmood,
ICLR 2024, together with the two distinct implementations published in the
authors' repository, as a small JAX PyTree transform.  It is intentionally
separate from ``core.upgd.UPGDLearner``: that historical Alberta learner uses
absolute ``|w g|`` utility and a power gate, and changing it in place would
invalidate existing checkpoints and results.

The protecting update is

``w <- (1 - alpha * weight_decay) * w
       - k * alpha * (gradient + noise) * (1 - scaled_utility)``.

The paper and experiment code use ``k=1``.  The repository README's advertised
implementation uses ``k=2`` while retaining one-alpha decoupled weight decay.
The non-protecting ablation gates only the perturbation.  Utility is the signed
first-order Taylor approximation ``-gradient * weight``, accumulated with an
EMA and bias correction.

The source profiles deliberately keep source discrepancies visible:

``paper_global``
    Corrected global maximum, one-alpha direction, and the paper's global
    time-step clock.
``official_readme_global``
    Uncorrected global maximum and the README's two-alpha gated direction.
``official_experiment_global``
    Uncorrected global maximum and the experiment code's one-alpha direction.
``official_experiment_local``
    Corrected row-local L2 normalization along the final axis with PyTorch's
    ``1e-12`` normalization floor.
``paper_local_literal``
    Appendix E's literal corrected numerator divided by the raw-EMA row norm.
``safe_extended``
    Explicit global or local normalization with numerical guards, masks,
    missing/non-finite-gradient skipping, and active-element clocks.

Source profiles are equation-exact only for finite, all-active gradients.  They
reject masks and Python ``None`` gradients.  Dynamic non-finite values fail
closed because raising from a JIT-compiled update is not portable.

Reference:
    Elsayed, M. & Mahmood, A. R. (2024). Addressing Loss of Plasticity and
    Catastrophic Forgetting in Continual Learning. ICLR 2024.
    https://openreview.net/forum?id=sKPzAXoylB

    Released MIT implementation audited at commit
    ``b75e90ad4b09c28971ac9dbb902a8fd86709b28c``:
    https://github.com/mohmdelsayed/upgd
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import Int

UPGDMode = Literal["protecting", "non_protecting"]
UPGDNormalization = Literal["global", "local"]
UPGDProfile = Literal[
    "paper_global",
    "official_readme_global",
    "official_experiment_global",
    "official_experiment_local",
    "paper_local_literal",
    "safe_extended",
]

_SOURCE_PROFILES = frozenset(
    {
        "paper_global",
        "official_readme_global",
        "official_experiment_global",
        "official_experiment_local",
        "paper_local_literal",
    }
)
_GLOBAL_PROFILES = frozenset(
    {
        "paper_global",
        "official_readme_global",
        "official_experiment_global",
    }
)
_RAW_GLOBAL_PROFILES = frozenset(
    {
        "official_readme_global",
        "official_experiment_global",
    }
)


@dataclass(frozen=True)
class CanonicalUPGDConfig:
    """Configuration for canonical first-order UPGD.

    Args:
        step_size: Base gradient step size ``alpha``.
        utility_decay: EMA decay ``beta`` for signed utility.
        noise_std: Standard deviation ``sigma`` of Gaussian perturbations.
        weight_decay: Decoupled UPGD-W decay coefficient.
        mode: ``"protecting"`` gates gradient and noise;
            ``"non_protecting"`` gates noise only.
        profile: Named equation source.  Profiles select equations, not source
            constructor hyperparameter defaults.
        normalization: Required explicit choice for ``"safe_extended"``.
            Source profiles accept ``None`` or their fixed matching value so
            serialized configurations remain explicit without changing the
            cited equation.
        epsilon: Positive numerical floor used only by ``"safe_extended"``.
    """

    step_size: float = 1e-3
    utility_decay: float = 0.999
    noise_std: float = 1e-3
    weight_decay: float = 0.0
    mode: UPGDMode = "protecting"
    profile: UPGDProfile = "safe_extended"
    normalization: UPGDNormalization | None = "global"
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "step_size",
            "utility_decay",
            "noise_std",
            "weight_decay",
            "epsilon",
        ):
            if isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be numeric, not boolean")
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            raise ValueError("step_size must be finite and positive")
        if not math.isfinite(self.utility_decay) or not 0.0 <= self.utility_decay < 1.0:
            raise ValueError("utility_decay must be finite and in [0, 1)")
        if not math.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ValueError("noise_std must be finite and non-negative")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.mode not in {"protecting", "non_protecting"}:
            raise ValueError("mode must be 'protecting' or 'non_protecting'")
        if self.profile not in _SOURCE_PROFILES | {"safe_extended"}:
            raise ValueError(
                "profile must name a paper, official implementation, or safe_extended profile"
            )
        if self.profile == "safe_extended":
            if self.normalization not in {"global", "local"}:
                raise ValueError("safe_extended requires normalization='global' or 'local'")
        else:
            fixed_normalization = "global" if self.profile in _GLOBAL_PROFILES else "local"
            if self.normalization not in {None, fixed_normalization}:
                raise ValueError(f"{self.profile} fixes normalization={fixed_normalization!r}")
        if self.profile == "official_readme_global" and self.mode != "protecting":
            raise ValueError("official_readme_global only defines protecting UPGD")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

    @property
    def is_source_profile(self) -> bool:
        """Whether this configuration follows a cited source equation."""

        return self.profile in _SOURCE_PROFILES

    @property
    def resolved_normalization(self) -> UPGDNormalization:
        """Return the normalization fixed by the selected profile."""

        if self.profile == "safe_extended":
            if self.normalization is None:  # guarded by __post_init__
                raise AssertionError("safe_extended normalization was not validated")
            return self.normalization
        if self.profile in _GLOBAL_PROFILES:
            return "global"
        return "local"

    @property
    def direction_multiplier(self) -> float:
        """Return the source coefficient on the gated learning direction."""

        return 2.0 if self.profile == "official_readme_global" else 1.0

    @property
    def uses_global_clock(self) -> bool:
        """Whether bias correction uses the global optimizer step."""

        return self.is_source_profile

    def to_config(self) -> dict[str, object]:
        """Return a JSON-serializable configuration."""

        return {
            "type": "CanonicalUPGD",
            "step_size": self.step_size,
            "utility_decay": self.utility_decay,
            "noise_std": self.noise_std,
            "weight_decay": self.weight_decay,
            "mode": self.mode,
            "profile": self.profile,
            "normalization": self.normalization,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_config(cls, config: dict[str, object]) -> CanonicalUPGDConfig:
        """Strictly reconstruct from the complete serialized schema."""

        payload = dict(config)
        expected = {
            "type",
            "step_size",
            "utility_decay",
            "noise_std",
            "weight_decay",
            "mode",
            "profile",
            "normalization",
            "epsilon",
        }
        if set(payload) != expected:
            raise ValueError("config fields do not match the CanonicalUPGD schema")
        type_name = payload.pop("type")
        if type_name != "CanonicalUPGD":
            raise ValueError(f"expected CanonicalUPGD config, got {type_name!r}")
        return cls(**payload)  # type: ignore[arg-type]


@chex.dataclass(frozen=True)
class CanonicalUPGDState:
    """Optimizer state for an arbitrary parameter PyTree."""

    utility_ema: Any
    utility_age: Any
    step: Int[Array, ""]


@chex.dataclass(frozen=True)
class CanonicalUPGDUpdate:
    """Result of one functional UPGD update."""

    params: Any
    state: CanonicalUPGDState
    next_key: Array
    scaled_utility: Any
    corrected_utility: Any
    perturbation: Any
    metrics: dict[str, Array]


def _flatten_with_none(tree: Any) -> tuple[list[Any], jax.tree_util.PyTreeDef]:
    """Flatten a PyTree while treating a missing gradient as a leaf."""

    leaves, structure = jax.tree_util.tree_flatten(
        tree,
        is_leaf=lambda value: value is None,
    )
    return list(leaves), structure


def _local_normalize(
    numerator: Array,
    denominator_utility: Array,
    active: Array,
    epsilon: float | None,
) -> Array:
    """Normalize active entries row-wise along a parameter leaf's final axis."""

    active_denominator = jnp.where(active, denominator_utility, 0.0)
    if numerator.ndim == 0:
        denominator = jnp.abs(active_denominator)
    else:
        denominator = jnp.linalg.norm(
            active_denominator,
            axis=-1,
            keepdims=True,
        )
    if epsilon is not None:
        denominator = jnp.maximum(denominator, epsilon)
    normalized = numerator / denominator
    return jnp.where(active, normalized, 0.0)


def _safe_signed_denominator(value: Array, epsilon: float) -> Array:
    """Floor a scalar magnitude without changing its sign."""

    signed_floor = jnp.where(value < 0.0, -epsilon, epsilon)
    return jnp.where(jnp.abs(value) >= epsilon, value, signed_floor)


class CanonicalUPGD:
    """Functional, JIT-compatible first-order UPGD PyTree transform."""

    def __init__(self, config: CanonicalUPGDConfig | None = None):
        self._config = CanonicalUPGDConfig() if config is None else config

    @property
    def config(self) -> CanonicalUPGDConfig:
        """Return the immutable optimizer configuration."""

        return self._config

    def to_config(self) -> dict[str, object]:
        """Serialize the optimizer configuration."""

        return self._config.to_config()

    @classmethod
    def from_config(cls, config: dict[str, object]) -> CanonicalUPGD:
        """Reconstruct an optimizer from serialized configuration."""

        return cls(CanonicalUPGDConfig.from_config(config))

    def init(self, params: Any) -> CanonicalUPGDState:
        """Initialize signed-utility traces matching ``params``."""

        leaves = jax.tree_util.tree_leaves(params)
        if not leaves:
            raise ValueError("params must contain at least one array leaf")
        for value in leaves:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.floating):
                raise ValueError("UPGD parameters must have floating-point dtype")

        return CanonicalUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree.map(jnp.zeros_like, params),
            utility_age=jax.tree.map(
                lambda value: jnp.zeros(value.shape, dtype=jnp.int32),
                params,
            ),
            step=jnp.array(0, dtype=jnp.int32),
        )

    def update(
        self,
        state: CanonicalUPGDState,
        params: Any,
        gradients: Any,
        key: Array,
        *,
        mask: Any | None = None,
        noise: Any | None = None,
    ) -> CanonicalUPGDUpdate:
        """Apply one source-profiled UPGD-W step.

        With ``safe_extended``, ``mask`` controls where UPGD utility/noise
        gating applies.  Masked-out parameters still receive ordinary gradient
        and decoupled weight-decay updates, allowing callers to protect a trunk
        while training heads and biases with plain SGDW.  A ``None`` gradient
        or a non-finite gradient skips that parameter element and is reported
        in metrics.  Source profiles reject masks and Python ``None`` leaves;
        dynamic non-finite arrays fail closed under eager execution and JIT.

        ``noise`` may supply the already-scaled perturbation PyTree ``xi`` for
        equation-level parity tests.  Normal operation omits it and samples
        ``N(0, noise_std**2)`` from ``key``.
        """

        param_leaves, structure = _flatten_with_none(params)
        grad_leaves, grad_structure = _flatten_with_none(gradients)
        utility_leaves, utility_structure = _flatten_with_none(state.utility_ema)
        age_leaves, age_structure = _flatten_with_none(state.utility_age)
        if not (structure == grad_structure == utility_structure == age_structure):
            raise ValueError("params, gradients, and UPGD state must share a PyTree structure")

        if self._config.is_source_profile and mask is not None:
            raise ValueError("source profiles do not accept masks")
        if self._config.is_source_profile and any(gradient is None for gradient in grad_leaves):
            raise ValueError("source profiles require every gradient leaf")

        if mask is None:
            mask_leaves = [jnp.ones_like(value, dtype=jnp.bool_) for value in param_leaves]
        else:
            mask_leaves, mask_structure = _flatten_with_none(mask)
            if mask_structure != structure:
                raise ValueError("mask must share the parameter PyTree structure")

        if noise is None:
            supplied_noise_leaves: list[Any] = [None] * len(param_leaves)
        else:
            supplied_noise_leaves, noise_structure = _flatten_with_none(noise)
            if noise_structure != structure:
                raise ValueError("noise must share the parameter PyTree structure")

        split_keys = jr.split(key, len(param_leaves) + 1)
        next_key = split_keys[0]
        noise_keys = split_keys[1:]
        beta = self._config.utility_decay
        next_step = state.step + jnp.array(1, dtype=jnp.int32)

        corrected_leaves: list[Array] = []
        new_utility_leaves: list[Array] = []
        new_age_leaves: list[Array] = []
        finite_masks: list[Array] = []
        eligible_masks: list[Array] = []
        active_masks: list[Array] = []
        clean_gradients: list[Array] = []

        for param, gradient, utility, age, mask_leaf in zip(
            param_leaves,
            grad_leaves,
            utility_leaves,
            age_leaves,
            mask_leaves,
            strict=True,
        ):
            eligible = jnp.broadcast_to(jnp.asarray(mask_leaf, dtype=jnp.bool_), param.shape)
            if gradient is None:
                finite = jnp.zeros(param.shape, dtype=jnp.bool_)
                clean_gradient = jnp.zeros_like(param)
            else:
                gradient_array = jnp.asarray(gradient, dtype=param.dtype)
                if gradient_array.shape != param.shape:
                    raise ValueError("every gradient leaf must match its parameter shape")
                finite = jnp.isfinite(param) & jnp.isfinite(gradient_array)
                clean_gradient = jnp.where(finite, gradient_array, 0.0)

            active = eligible & finite
            instantaneous = -clean_gradient * param
            next_utility = jnp.where(
                active,
                beta * utility + (1.0 - beta) * instantaneous,
                utility,
            )
            if self._config.uses_global_clock:
                next_age = jnp.full_like(age, next_step)
                correction_clock = next_step.astype(param.dtype)
            else:
                next_age = age + active.astype(jnp.int32)
                correction_clock = next_age.astype(param.dtype)
            bias_correction = 1.0 - jnp.power(beta, correction_clock)
            corrected = jnp.where(
                next_age > 0,
                next_utility
                / (
                    jnp.maximum(bias_correction, self._config.epsilon)
                    if self._config.profile == "safe_extended"
                    else bias_correction
                ),
                0.0,
            )

            clean_gradients.append(clean_gradient)
            finite_masks.append(finite)
            eligible_masks.append(eligible)
            active_masks.append(active)
            new_utility_leaves.append(next_utility)
            new_age_leaves.append(next_age)
            corrected_leaves.append(corrected)

        if self._config.resolved_normalization == "global":
            maximum = jnp.array(-jnp.inf, dtype=jnp.float32)
            has_active = jnp.array(False)
            reference_leaves = (
                new_utility_leaves
                if self._config.profile in _RAW_GLOBAL_PROFILES
                else corrected_leaves
            )
            for reference, active in zip(
                reference_leaves,
                active_masks,
                strict=True,
            ):
                leaf_maximum = jnp.max(
                    jnp.where(active, reference, -jnp.inf),
                    initial=-jnp.inf,
                )
                maximum = jnp.maximum(maximum, leaf_maximum.astype(jnp.float32))
                has_active = has_active | jnp.any(active)
            global_maximum = jnp.where(has_active, maximum, 0.0)
            denominator = (
                _safe_signed_denominator(
                    global_maximum,
                    self._config.epsilon,
                )
                if self._config.profile == "safe_extended"
                else global_maximum
            )
            normalized_leaves = [
                jnp.where(
                    active,
                    corrected / denominator.astype(corrected.dtype),
                    0.0,
                )
                for corrected, active in zip(
                    corrected_leaves,
                    active_masks,
                    strict=True,
                )
            ]
        else:
            global_maximum = jnp.array(jnp.nan, dtype=jnp.float32)
            if self._config.profile == "paper_local_literal":
                denominator_leaves = new_utility_leaves
                local_epsilon: float | None = None
            elif self._config.profile == "official_experiment_local":
                denominator_leaves = corrected_leaves
                local_epsilon = 1e-12
            else:
                denominator_leaves = corrected_leaves
                local_epsilon = self._config.epsilon
            normalized_leaves = [
                _local_normalize(
                    corrected,
                    denominator_utility,
                    active,
                    local_epsilon,
                )
                for corrected, denominator_utility, active in zip(
                    corrected_leaves,
                    denominator_leaves,
                    active_masks,
                    strict=True,
                )
            ]

        new_param_leaves: list[Array] = []
        gate_leaves: list[Array] = []
        perturbation_leaves: list[Array] = []
        gate_sum = jnp.array(0.0, dtype=jnp.float32)
        utility_sum = jnp.array(0.0, dtype=jnp.float32)
        eligible_count = jnp.array(0.0, dtype=jnp.float32)
        nonfinite_count = jnp.array(0.0, dtype=jnp.float32)

        for (
            param,
            gradient,
            normalized,
            corrected,
            finite,
            eligible,
            active,
            noise_key,
            supplied_noise,
        ) in zip(
            param_leaves,
            clean_gradients,
            normalized_leaves,
            corrected_leaves,
            finite_masks,
            eligible_masks,
            active_masks,
            noise_keys,
            supplied_noise_leaves,
            strict=True,
        ):
            gate = jnp.where(active, jax.nn.sigmoid(normalized), 0.0)
            if supplied_noise is None:
                sampled_noise = (
                    jr.normal(noise_key, param.shape, dtype=param.dtype) * self._config.noise_std
                )
            else:
                sampled_noise = jnp.asarray(supplied_noise, dtype=param.dtype)
                if sampled_noise.shape != param.shape:
                    raise ValueError("every noise leaf must match its parameter shape")
            finite_noise = jnp.where(jnp.isfinite(sampled_noise), sampled_noise, 0.0)
            perturbation = jnp.where(active, finite_noise, 0.0)

            if self._config.mode == "protecting":
                direction = (gradient + perturbation) * (1.0 - gate)
            else:
                direction = gradient + perturbation * (1.0 - gate)

            decayed = param * (1.0 - self._config.step_size * self._config.weight_decay)
            direction_step = self._config.step_size * self._config.direction_multiplier
            candidate = decayed - direction_step * direction
            updated = jnp.where(finite, candidate, param)

            count = jnp.sum(active.astype(jnp.float32))
            gate_sum = gate_sum + jnp.sum(gate.astype(jnp.float32))
            utility_sum = utility_sum + jnp.sum(
                jnp.where(active, jnp.abs(corrected), 0.0).astype(jnp.float32)
            )
            eligible_count = eligible_count + count
            nonfinite_count = nonfinite_count + jnp.sum((~finite).astype(jnp.float32))

            new_param_leaves.append(updated)
            gate_leaves.append(gate)
            perturbation_leaves.append(perturbation)

        count_floor = jnp.maximum(eligible_count, 1.0)
        next_state = CanonicalUPGDState(  # type: ignore[call-arg]
            utility_ema=jax.tree_util.tree_unflatten(structure, new_utility_leaves),
            utility_age=jax.tree_util.tree_unflatten(structure, new_age_leaves),
            step=next_step,
        )
        return CanonicalUPGDUpdate(  # type: ignore[call-arg]
            params=jax.tree_util.tree_unflatten(structure, new_param_leaves),
            state=next_state,
            next_key=next_key,
            scaled_utility=jax.tree_util.tree_unflatten(structure, gate_leaves),
            corrected_utility=jax.tree_util.tree_unflatten(
                structure,
                corrected_leaves,
            ),
            perturbation=jax.tree_util.tree_unflatten(
                structure,
                perturbation_leaves,
            ),
            metrics={
                "mean_scaled_utility": gate_sum / count_floor,
                "mean_absolute_utility": utility_sum / count_floor,
                "global_maximum_utility": global_maximum,
                "eligible_parameter_count": eligible_count,
                "nonfinite_or_missing_count": nonfinite_count,
            },
        )
