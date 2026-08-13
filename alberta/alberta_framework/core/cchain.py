# mypy: disable-error-code="attr-defined,call-arg,no-any-return"
"""Clean-room paper-equation C-CHAIN comparator with bounded transactions.

This module implements an isolated L0 comparator derived from Tang et al.,
ICML 2025, Equation 3 (output churn), Equation 8 (one-half mean squared
output churn), and the Appendix running-loss coefficient ratio.  No source
code from the authors' public repository was used.

Only the Equation 8 objective is claimed exact.  This is an isolated
one-step-lag combined-gradient comparator, not a reproduction of the paper's
complete sequential PPO or DQN algorithms, replay/buffer behavior, optimizer
schedule, environment protocol, or reported efficacy.  It is not integrated
into a default agent path.

The exact regularizer in this module is

``0.5 * mean((f_current(reference) - stop_gradient(f_reference(reference)))**2)``.

Its objective is intentionally generic: callers supply a model callable, a
base-loss callable, float32 parameter PyTrees, and disjoint train/reference
sample identities.  Declarative model/loss binding words and all content tags
are unkeyed, caller-owned post-mint integrity aids.  They do not authenticate
callable identity, data provenance, or external optimizer application.

Proposal and commit are deliberately separated.  A valid proposal performs
exactly one combined ``jax.value_and_grad`` pass after its sample-identity,
state, parameter, and lifetime preflight.  An invalid preflight performs no
autodiff.  Commit performs no autodiff: it validates the proposal, advances
the exact uint32[2] lifetime, moves the proposal's source parameters into the
one-step-lag reference slot, and binds the next expected current parameters to
the caller-supplied applied parameters.  Optimizer application remains wholly
caller-owned and unauthenticated.  The next proposal rejects any substituted
parameter tree.

The paper Appendix specifies a ratio of running mean absolute base loss to
running mean absolute churn loss.  The epsilon denominator, warmup, bounded
trailing ring, and coefficient clamps here are explicit Alberta engineering
controls around that ratio; they are not part of Equation 8.  The coefficient
changes only in a valid commit.  A zero target relative scale disables the
regularizer.

The empirical-NTK helper is diagnostic only.  It does not gate updates,
dispatch actions, produce agent output, establish scientific evidence, or
promote a claim.

Reference:
    Tang, H., Obando-Ceron, J., Castro, P. S., Courville, A., & Berseth, G.
    (2025). Mitigating Plasticity Loss in Continual Reinforcement Learning by
    Reducing Churn. ICML 2025. https://proceedings.mlr.press/v267/tang25g.html
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int, UInt

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)

CCHAIN_CONFIG_SCHEMA = "alberta.cchain.config.v1"
CCHAIN_CHECKPOINT_SCHEMA = "alberta.cchain.checkpoint.v1"
CCHAIN_MECHANISM_STATUS = "l0-development-only-not-assessed"
CCHAIN_EVIDENCE_LEVEL = "L0"
CCHAIN_SCIENTIFIC_PROMOTION_ALLOWED = False
CCHAIN_DISPATCH_AUTHORITY = False
CCHAIN_OUTPUT_AUTHORITY = False
CCHAIN_MODEL_BINDING_AUTHENTICATED = False
CCHAIN_LOSS_BINDING_AUTHENTICATED = False
CCHAIN_EXTERNAL_OPTIMIZER_APPLICATION_AUTHENTICATED = False
CCHAIN_CONTENT_INTEGRITY_SCOPE = "post-mint-unkeyed-integrity-only"
CCHAIN_EXACT_OBJECTIVE_PROFILE = "tang-2025-equation-8-exact-squared-output-churn"
CCHAIN_EQUATION_8_OBJECTIVE_EXACT = True
CCHAIN_COMPARATOR_SCOPE = "isolated-one-step-lag-combined-gradient-comparator"
CCHAIN_FULL_SEQUENTIAL_ALGORITHM_REPRODUCED = False
CCHAIN_EFFICACY_ASSESSED = False
CCHAIN_DEFAULT_AGENT_INTEGRATION = False
CCHAIN_AUTOSCALE_CONTROL_PROFILE = (
    "appendix-absolute-loss-ratio-with-explicit-alberta-window-warmup-epsilon-bounds"
)
CCHAIN_NTK_DIAGNOSTIC_STATUS = "diagnostic-only-not-a-gate-or-evidence"

_PAPER_SOURCE_METADATA: dict[str, object] = {
    "title": "Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn",
    "authors": [
        "Hongyao Tang",
        "Johan Obando-Ceron",
        "Pablo Samuel Castro",
        "Aaron Courville",
        "Glen Berseth",
    ],
    "venue": "ICML 2025",
    "url": "https://proceedings.mlr.press/v267/tang25g.html",
    "equations": [3, 8],
    "appendix_components": ["running mean absolute-loss coefficient ratio", "NTK rank"],
    "implementation_origin": "clean-room-paper-equation",
    "public_repository_source_code_used": False,
}

_UINT32_MAX = 4_294_967_295
_UINT64_MAX = 18_446_744_073_709_551_615
_MAX_AUTO_SCALE_WINDOW = 4_096
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_STATE_TAG_SALT = 0x43535441
_PROPOSAL_TAG_SALT = 0x43505250
_PARAMETER_TAG_SALT = 0x43504152
_REFERENCE_TAG_SALT = 0x43524546
_TRAIN_IDS_TAG_SALT = 0x4354524E
_REFERENCE_IDS_TAG_SALT = 0x43524944
_TAG_OFFSET = 2_166_136_261
_TAG_PRIME = 16_777_619

ModelFn = Callable[[Any, Any], Array]
BaseLossFn = Callable[[Any, Any], Array]


def _strict_words(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must be an exact two-element tuple")
    parsed: list[int] = []
    for word in value:
        if type(word) is not int or not 0 <= word <= _UINT32_MAX:
            raise ValueError(f"{name} words must be strict uint32 integers")
        parsed.append(word)
    result = (parsed[0], parsed[1])
    if result == (0, 0):
        raise ValueError(f"{name} must be nonzero")
    return result


def _strict_nonnegative_int(
    value: object,
    *,
    name: str,
    maximum: int,
) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be a strict integer in [0, {maximum}]")
    return value


def _strict_positive_int(value: object, *, name: str, maximum: int) -> int:
    parsed = _strict_nonnegative_int(value, name=name, maximum=maximum)
    if parsed == 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _finite_float32(
    value: object,
    *,
    name: str,
    minimum: float,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real non-boolean scalar")
    parsed = float(value)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = float(np.float32(parsed))
    if not math.isfinite(parsed) or not math.isfinite(narrowed):
        raise ValueError(f"{name} must remain finite in float32")
    if parsed != 0.0 and abs(narrowed) < _FLOAT32_TINY:
        raise ValueError(f"{name} must not underflow in float32")
    if narrowed < minimum or (strictly_positive and narrowed == 0.0):
        comparator = "positive" if strictly_positive else f">= {minimum}"
        raise ValueError(f"{name} must be {comparator}")
    return narrowed


@dataclasses.dataclass(frozen=True)
class CChainConfig:
    """Immutable equation, binding, autoscale, and lifetime configuration."""

    model_binding_words: tuple[int, int]
    loss_binding_words: tuple[int, int]
    target_relative_scale: float = 0.01
    initial_coefficient: float = 1.0
    auto_scale_epsilon: float = 1.0e-8
    auto_scale_warmup_commits: int = 1
    auto_scale_window: int = 100
    minimum_coefficient: float = 0.0
    maximum_coefficient: float = 1_000_000.0
    max_commits: int = _UINT32_MAX

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_binding_words",
            _strict_words(self.model_binding_words, name="model_binding_words"),
        )
        object.__setattr__(
            self,
            "loss_binding_words",
            _strict_words(self.loss_binding_words, name="loss_binding_words"),
        )
        for name, minimum, positive in (
            ("target_relative_scale", 0.0, False),
            ("initial_coefficient", 0.0, False),
            ("auto_scale_epsilon", _FLOAT32_TINY, True),
            ("minimum_coefficient", 0.0, False),
            ("maximum_coefficient", _FLOAT32_TINY, True),
        ):
            object.__setattr__(
                self,
                name,
                _finite_float32(
                    getattr(self, name),
                    name=name,
                    minimum=minimum,
                    strictly_positive=positive,
                ),
            )
        object.__setattr__(
            self,
            "auto_scale_warmup_commits",
            _strict_nonnegative_int(
                self.auto_scale_warmup_commits,
                name="auto_scale_warmup_commits",
                maximum=_UINT64_MAX,
            ),
        )
        object.__setattr__(
            self,
            "auto_scale_window",
            _strict_positive_int(
                self.auto_scale_window,
                name="auto_scale_window",
                maximum=_MAX_AUTO_SCALE_WINDOW,
            ),
        )
        object.__setattr__(
            self,
            "max_commits",
            _strict_positive_int(
                self.max_commits,
                name="max_commits",
                maximum=_UINT64_MAX,
            ),
        )
        if self.minimum_coefficient > self.maximum_coefficient:
            raise ValueError("minimum_coefficient must not exceed maximum_coefficient")
        if self.target_relative_scale == 0.0:
            if self.initial_coefficient != 0.0:
                raise ValueError(
                    "initial_coefficient must be zero when target_relative_scale is zero"
                )
        elif not self.minimum_coefficient <= self.initial_coefficient <= self.maximum_coefficient:
            raise ValueError("initial_coefficient must lie within coefficient bounds")


@chex.dataclass(frozen=True)
class CChainState:
    """Reference/current bindings, autoscale ring, lifetime, and integrity."""

    reference_params: Any
    expected_current_params: Any
    parameter_signature_words: UInt[Array, " 2"]
    coefficient: Float[Array, ""]
    base_loss_window: Float[Array, " window"]
    churn_loss_window: Float[Array, " window"]
    loss_window_count: Int[Array, ""]
    loss_window_cursor: Int[Array, ""]
    commit_count_words: UInt[Array, " 2"]
    state_integrity_tag: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CChainProposal:
    """One combined-gradient proposal bound to one exact source state."""

    source_params: Any
    gradients: Any
    model_binding_words: UInt[Array, " 2"]
    loss_binding_words: UInt[Array, " 2"]
    source_state_integrity_tag: UInt[Array, " 2"]
    source_commit_count_words: UInt[Array, " 2"]
    destination_commit_count_words: UInt[Array, " 2"]
    source_parameter_content_tag: UInt[Array, " 2"]
    reference_parameter_content_tag: UInt[Array, " 2"]
    train_sample_ids_content_tag: UInt[Array, " 2"]
    reference_sample_ids_content_tag: UInt[Array, " 2"]
    coefficient_used: Float[Array, ""]
    base_loss: Float[Array, ""]
    churn_loss: Float[Array, ""]
    combined_loss: Float[Array, ""]
    sample_identity_preflight_valid: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    valid: Bool[Array, ""]
    proposal_integrity_tag: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CChainProposalDiagnostics:
    state_valid: Bool[Array, ""]
    current_params_match: Bool[Array, ""]
    train_sample_ids_nonzero_unique: Bool[Array, ""]
    reference_sample_ids_nonzero_unique: Bool[Array, ""]
    sample_id_sets_disjoint: Bool[Array, ""]
    sample_identity_preflight_valid: Bool[Array, ""]
    commit_capacity_available: Bool[Array, ""]
    preflight_valid: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    autodiff_pass_count: Int[Array, ""]
    model_binding_authenticated: Bool[Array, ""]
    loss_binding_authenticated: Bool[Array, ""]
    data_provenance_authenticated: Bool[Array, ""]


@chex.dataclass(frozen=True)
class CChainProposalResult:
    proposal: CChainProposal
    diagnostics: CChainProposalDiagnostics


@chex.dataclass(frozen=True)
class CChainCommitDiagnostics:
    state_valid: Bool[Array, ""]
    proposal_integrity_valid: Bool[Array, ""]
    proposal_declared_valid: Bool[Array, ""]
    source_fresh: Bool[Array, ""]
    source_params_match: Bool[Array, ""]
    reference_params_match: Bool[Array, ""]
    binding_words_match: Bool[Array, ""]
    commit_capacity_available: Bool[Array, ""]
    applied_params_finite: Bool[Array, ""]
    external_optimizer_application_authenticated: Bool[Array, ""]
    candidate_state_valid: Bool[Array, ""]
    applied: Bool[Array, ""]
    autodiff_pass_count: Int[Array, ""]
    pre_commit_count_words: UInt[Array, " 2"]
    post_commit_count_words: UInt[Array, " 2"]


@chex.dataclass(frozen=True)
class CChainCommitResult:
    state: CChainState
    diagnostics: CChainCommitDiagnostics


@dataclasses.dataclass(frozen=True)
class CChainResourceBudget:
    persistent_state_scalars: int
    persistent_state_bytes: int
    proposal_scalars: int
    proposal_bytes: int
    parameter_scalars: int
    parameter_bytes: int
    reference_parameter_copies: int
    expected_current_parameter_copies: int
    proposal_source_parameter_copies: int
    proposal_gradient_copies: int
    auto_scale_window: int
    max_commits: int
    valid_proposal_autodiff_passes: int
    rejected_preflight_autodiff_passes: int
    commit_autodiff_passes: int
    external_optimizer_state_owned: int
    dispatch_authority: int
    output_authority: int
    scientific_promotion_allowed: bool
    full_sequential_algorithm_reproduced: bool
    efficacy_assessed: bool
    default_agent_integration: bool
    discarded_functional_state_can_repeat_pure_calls: bool

    def to_config(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@chex.dataclass(frozen=True)
class EmpiricalNTKDiagnostics:
    """Diagnostic-only empirical NTK matrix and paper approximate-rank metrics."""

    gradient_matrix: Float[Array, "n_samples n_parameters"]
    gram_matrix: Float[Array, "n_samples n_samples"]
    singular_values: Float[Array, " n_samples"]
    delta: Float[Array, ""]
    information_fraction: Float[Array, ""]
    approximate_rank: Int[Array, ""]
    off_diagonal_absolute_sum: Float[Array, ""]
    off_diagonal_absolute_mean: Float[Array, ""]
    off_diagonal_rms: Float[Array, ""]
    diagonal_sum: Float[Array, ""]
    diagonal_mean: Float[Array, ""]
    diagonal_rms: Float[Array, ""]
    diagonal_minimum: Float[Array, ""]
    diagonal_maximum: Float[Array, ""]
    input_finite: Bool[Array, ""]
    derived_finite: Bool[Array, ""]
    zero_gradient: Bool[Array, ""]
    valid: Bool[Array, ""]


def _config_settings(config: CChainConfig) -> dict[str, object]:
    return {
        "model_binding_words": list(config.model_binding_words),
        "loss_binding_words": list(config.loss_binding_words),
        "target_relative_scale": config.target_relative_scale,
        "initial_coefficient": config.initial_coefficient,
        "auto_scale_epsilon": config.auto_scale_epsilon,
        "auto_scale_warmup_commits": config.auto_scale_warmup_commits,
        "auto_scale_window": config.auto_scale_window,
        "minimum_coefficient": config.minimum_coefficient,
        "maximum_coefficient": config.maximum_coefficient,
        "max_commits": config.max_commits,
    }


def _paper_source_metadata() -> dict[str, object]:
    return copy.deepcopy(_PAPER_SOURCE_METADATA)


def _config_digest(config: Mapping[str, object]) -> str:
    canonical = json.dumps(dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _config_fingerprint(config: Mapping[str, object]) -> Array:
    digest = bytes.fromhex(_config_digest(config))
    return jnp.asarray(
        (
            int.from_bytes(digest[:4], "big"),
            int.from_bytes(digest[4:8], "big"),
        ),
        dtype=jnp.uint32,
    )


def _array_contract(value: object, *, shape: tuple[int, ...], dtype: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and tuple(cast(Any, value).shape) == shape
        and cast(Any, value).dtype == jnp.dtype(dtype)
    )


def _parameter_tree_static_signature(tree: object) -> tuple[object, tuple[tuple[object, ...], ...]]:
    leaves, structure = jax.tree.flatten(tree)
    signatures: list[tuple[object, ...]] = []
    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            signatures.append(("not-array", type(leaf).__qualname__))
            continue
        signatures.append((tuple(cast(Any, leaf).shape), str(cast(Any, leaf).dtype)))
    return structure, tuple(signatures)


def _validate_parameter_tree(tree: object, *, name: str) -> None:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError(f"{name} must contain at least one parameter leaf")
    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError(f"{name} must be an array-only parameter PyTree")
        if cast(Any, leaf).dtype != jnp.dtype(jnp.float32):
            raise TypeError(f"{name} leaves must have exact float32 dtype")


def _parameter_signature_words(tree: object) -> Array:
    leaves, structure = jax.tree.flatten(tree)
    payload = {
        "tree": str(structure),
        "leaves": [
            {
                "shape": list(cast(Any, leaf).shape),
                "dtype": str(cast(Any, leaf).dtype),
            }
            for leaf in leaves
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).digest()
    return jnp.asarray(
        (
            int.from_bytes(digest[:4], "big"),
            int.from_bytes(digest[4:8], "big"),
        ),
        dtype=jnp.uint32,
    )


def _tree_finite(tree: object) -> Bool[Array, ""]:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for leaf in jax.tree.leaves(tree):
        valid = valid & jnp.all(jnp.isfinite(jnp.asarray(leaf)))
    return valid


def _tree_equal(left: object, right: object) -> Bool[Array, ""]:
    left_leaves, left_structure = jax.tree.flatten(left)
    right_leaves, right_structure = jax.tree.flatten(right)
    if cast(Any, left_structure) != cast(Any, right_structure) or len(left_leaves) != len(
        right_leaves
    ):
        return jnp.asarray(False, dtype=jnp.bool_)
    result = jnp.asarray(True, dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if (
            tuple(cast(Any, left_leaf).shape) != tuple(cast(Any, right_leaf).shape)
            or cast(Any, left_leaf).dtype != cast(Any, right_leaf).dtype
        ):
            return jnp.asarray(False, dtype=jnp.bool_)
        result = result & jnp.array_equal(left_leaf, right_leaf)
    return result


def _float_words(value: Array) -> Array:
    return jax.lax.bitcast_convert_type(value, jnp.uint32)


def _tree_content_words(tree: object) -> Array:
    parts: list[Array] = []
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        if array.dtype == jnp.dtype(jnp.float32):
            words = _float_words(array)
        elif array.dtype in (
            jnp.dtype(jnp.uint32),
            jnp.dtype(jnp.int32),
            jnp.dtype(jnp.bool_),
        ):
            words = array.astype(jnp.uint32)
        else:
            raise TypeError(f"unsupported content-integrity dtype {array.dtype}")
        parts.append(jnp.ravel(words))
    if not parts:
        return jnp.zeros((0,), dtype=jnp.uint32)
    return jnp.concatenate(parts)


def _mix_words(words: Array, *, salt: int) -> UInt[Array, ""]:
    flat = jnp.ravel(words).astype(jnp.uint32)

    def body(index: int, tag: Array) -> Array:
        position = (jnp.asarray(index, dtype=jnp.uint32) + 1) * jnp.asarray(
            0x9E3779B9,
            dtype=jnp.uint32,
        )
        mixed = (tag ^ flat[index] ^ position) * jnp.asarray(
            _TAG_PRIME,
            dtype=jnp.uint32,
        )
        return (mixed << jnp.asarray(13, dtype=jnp.uint32)) | (
            mixed >> jnp.asarray(19, dtype=jnp.uint32)
        )

    result = jax.lax.fori_loop(
        0,
        flat.shape[0],
        body,
        jnp.asarray(_TAG_OFFSET ^ salt, dtype=jnp.uint32),
    )
    return jnp.where(
        result == jnp.asarray(0, dtype=jnp.uint32),
        jnp.asarray(salt, dtype=jnp.uint32),
        result,
    )


def _content_tag(tree: object, *, salt: int) -> UInt[Array, " 2"]:
    words = _tree_content_words(tree)
    return jnp.stack(
        (
            _mix_words(words, salt=salt),
            _mix_words(words, salt=salt ^ 0x9E3779B9),
        )
    ).astype(jnp.uint32)


def _int_to_words(value: int) -> Array:
    return jnp.asarray(
        ((value >> 32) & _UINT32_MAX, value & _UINT32_MAX),
        dtype=jnp.uint32,
    )


def _words_nonzero(words: Array) -> Bool[Array, ""]:
    return jnp.any(words != jnp.asarray(0, dtype=jnp.uint32))


def _words_less_equal(left: Array, right: Array) -> Bool[Array, ""]:
    return (left[0] < right[0]) | ((left[0] == right[0]) & (left[1] <= right[1]))


def _checked_words_increment(words: Array) -> tuple[Array, Array]:
    next_low = words[1] + jnp.asarray(1, dtype=jnp.uint32)
    carry = (next_low == jnp.asarray(0, dtype=jnp.uint32)).astype(jnp.uint32)
    next_high = words[0] + carry
    overflow = (carry != 0) & (next_high == jnp.asarray(0, dtype=jnp.uint32))
    candidate = jnp.stack((next_high, next_low)).astype(jnp.uint32)
    return jnp.where(overflow, words, candidate), ~overflow


def _words_mod(words: Array, modulus: int) -> Int[Array, ""]:
    modulus_words = jnp.asarray(modulus, dtype=jnp.uint32)
    two_to_32_mod = jnp.asarray((1 << 32) % modulus, dtype=jnp.uint32)
    high_mod = words[0] % modulus_words
    low_mod = words[1] % modulus_words

    def multiply_body(_: int, carry: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        accumulated, addend, multiplier = carry
        selected = jnp.where(
            (multiplier & jnp.asarray(1, dtype=jnp.uint32)) != 0,
            (accumulated + addend) % modulus_words,
            accumulated,
        )
        doubled = (addend + addend) % modulus_words
        return selected, doubled, multiplier >> jnp.asarray(1, dtype=jnp.uint32)

    # ``modulus <= INT32_MAX`` means sums of two residues cannot overflow
    # uint32.  Shift-and-add therefore computes the high-word product exactly
    # without requiring JAX x64 mode.
    high_contribution, _, _ = jax.lax.fori_loop(
        0,
        32,
        multiply_body,
        (
            jnp.asarray(0, dtype=jnp.uint32),
            two_to_32_mod,
            high_mod,
        ),
    )
    result = (high_contribution + low_mod) % modulus_words
    return result.astype(jnp.int32)


def _logical_tree_size(tree: object) -> tuple[int, int]:
    scalars = 0
    nbytes = 0
    for leaf in jax.tree.leaves(tree):
        array = jnp.asarray(leaf)
        scalars += int(array.size)
        nbytes += int(array.nbytes)
    return scalars, nbytes


def squared_output_churn(current_output: Array, reference_output: Array) -> Array:
    """Return Tang et al. Equation 8 for one scalar output per sample.

    The leading axis is the reference-sample axis.  A scalar may be represented
    as ``[B]`` or with singleton tail axes such as ``[B, 1]``; genuine
    vector-valued per-sample outputs are outside this exact paper-equation
    surface and fail closed.
    """

    if not hasattr(current_output, "shape") or not hasattr(reference_output, "shape"):
        raise TypeError("current_output and reference_output must be arrays")
    if current_output.dtype != jnp.dtype(jnp.float32):
        raise TypeError("current_output must have exact float32 dtype")
    if reference_output.dtype != jnp.dtype(jnp.float32):
        raise TypeError("reference_output must have exact float32 dtype")
    if current_output.shape != reference_output.shape or current_output.size == 0:
        raise ValueError("current and reference outputs must have one matching nonempty shape")
    if current_output.ndim == 0 or any(size != 1 for size in current_output.shape[1:]):
        raise ValueError("outputs must contain exactly one scalar per reference sample")
    difference = current_output - jax.lax.stop_gradient(reference_output)
    return jnp.asarray(0.5, dtype=jnp.float32) * jnp.mean(jnp.square(difference))


class CChain:
    """Transactional C-CHAIN gradient comparator for generic float32 PyTrees."""

    def __init__(
        self,
        model_fn: ModelFn,
        base_loss_fn: BaseLossFn,
        config: CChainConfig,
    ) -> None:
        if not callable(model_fn):
            raise TypeError("model_fn must be callable")
        if not callable(base_loss_fn):
            raise TypeError("base_loss_fn must be callable")
        if not isinstance(config, CChainConfig):
            raise TypeError("config must be a CChainConfig")
        self._model_fn = model_fn
        self._base_loss_fn = base_loss_fn
        self._config = config
        self._config_fingerprint = _config_fingerprint(self.to_config())
        self._model_binding_words = jnp.asarray(config.model_binding_words, dtype=jnp.uint32)
        self._loss_binding_words = jnp.asarray(config.loss_binding_words, dtype=jnp.uint32)
        self._max_commit_words = _int_to_words(config.max_commits)

    @property
    def config(self) -> CChainConfig:
        return self._config

    def to_config(self) -> dict[str, object]:
        return {
            "schema": CCHAIN_CONFIG_SCHEMA,
            "type": type(self).__name__,
            "mechanism_status": CCHAIN_MECHANISM_STATUS,
            "evidence_level": CCHAIN_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "dispatch_authority": False,
            "output_authority": False,
            "model_binding_authenticated": False,
            "loss_binding_authenticated": False,
            "external_optimizer_application_authenticated": False,
            "content_integrity_scope": CCHAIN_CONTENT_INTEGRITY_SCOPE,
            "equation_8_objective_profile": CCHAIN_EXACT_OBJECTIVE_PROFILE,
            "equation_8_objective_exact": True,
            "comparator_scope": CCHAIN_COMPARATOR_SCOPE,
            "full_sequential_algorithm_reproduced": False,
            "efficacy_assessed": False,
            "default_agent_integration": False,
            "autoscale_control_profile": CCHAIN_AUTOSCALE_CONTROL_PROFILE,
            "autoscale_controls_are_equation_8": False,
            "unrelated_selection_semantics": False,
            "ntk_diagnostic_status": CCHAIN_NTK_DIAGNOSTIC_STATUS,
            "paper_source": _paper_source_metadata(),
            "settings": _config_settings(self._config),
        }

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        model_fn: ModelFn,
        base_loss_fn: BaseLossFn,
    ) -> CChain:
        payload = dict(config)
        expected = {
            "schema",
            "type",
            "mechanism_status",
            "evidence_level",
            "scientific_promotion_allowed",
            "dispatch_authority",
            "output_authority",
            "model_binding_authenticated",
            "loss_binding_authenticated",
            "external_optimizer_application_authenticated",
            "content_integrity_scope",
            "equation_8_objective_profile",
            "equation_8_objective_exact",
            "comparator_scope",
            "full_sequential_algorithm_reproduced",
            "efficacy_assessed",
            "default_agent_integration",
            "autoscale_control_profile",
            "autoscale_controls_are_equation_8",
            "unrelated_selection_semantics",
            "ntk_diagnostic_status",
            "paper_source",
            "settings",
        }
        if set(payload) != expected:
            raise ValueError("C-CHAIN config fields do not match the v1 schema")
        fixed_values: dict[str, object] = {
            "schema": CCHAIN_CONFIG_SCHEMA,
            "type": cls.__name__,
            "mechanism_status": CCHAIN_MECHANISM_STATUS,
            "evidence_level": CCHAIN_EVIDENCE_LEVEL,
            "scientific_promotion_allowed": False,
            "dispatch_authority": False,
            "output_authority": False,
            "model_binding_authenticated": False,
            "loss_binding_authenticated": False,
            "external_optimizer_application_authenticated": False,
            "content_integrity_scope": CCHAIN_CONTENT_INTEGRITY_SCOPE,
            "equation_8_objective_profile": CCHAIN_EXACT_OBJECTIVE_PROFILE,
            "equation_8_objective_exact": True,
            "comparator_scope": CCHAIN_COMPARATOR_SCOPE,
            "full_sequential_algorithm_reproduced": False,
            "efficacy_assessed": False,
            "default_agent_integration": False,
            "autoscale_control_profile": CCHAIN_AUTOSCALE_CONTROL_PROFILE,
            "autoscale_controls_are_equation_8": False,
            "unrelated_selection_semantics": False,
            "ntk_diagnostic_status": CCHAIN_NTK_DIAGNOSTIC_STATUS,
            "paper_source": _paper_source_metadata(),
        }
        for name, expected_value in fixed_values.items():
            if payload.pop(name) != expected_value:
                suffix = " must remain false" if expected_value is False else " is unsupported"
                raise ValueError(f"C-CHAIN config {name}{suffix}")
        settings = payload.pop("settings")
        if not isinstance(settings, Mapping):
            raise ValueError("C-CHAIN config settings are missing")
        settings_payload = dict(settings)
        expected_settings = set(
            _config_settings(CChainConfig(model_binding_words=(1, 1), loss_binding_words=(2, 2)))
        )
        if set(settings_payload) != expected_settings:
            raise ValueError("C-CHAIN settings fields do not match the v1 schema")
        for name in ("model_binding_words", "loss_binding_words"):
            words = settings_payload[name]
            if not isinstance(words, list) or len(words) != 2:
                raise ValueError(f"C-CHAIN {name} must be a two-element list")
            settings_payload[name] = tuple(words)
        restored = cls(
            model_fn,
            base_loss_fn,
            CChainConfig(**cast(dict[str, Any], settings_payload)),
        )
        if restored.to_config() != dict(config):
            raise ValueError("C-CHAIN config is not canonical")
        return restored

    def _state_tag(self, state: CChainState) -> Array:
        payload = cast(
            CChainState,
            cast(Any, state).replace(state_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32)),
        )
        return _content_tag(
            (self._config_fingerprint, payload),
            salt=_STATE_TAG_SALT,
        )

    def _seal_state(self, state: CChainState) -> CChainState:
        return cast(
            CChainState,
            cast(Any, state).replace(state_integrity_tag=self._state_tag(state)),
        )

    def init(self, params: Any) -> CChainState:
        """Initialize equal current/reference copies and a zero lifetime."""

        _validate_parameter_tree(params, name="params")
        if not bool(jax.device_get(_tree_finite(params))):
            raise ValueError("params must be finite")
        window = self._config.auto_scale_window
        coefficient = (
            0.0 if self._config.target_relative_scale == 0.0 else self._config.initial_coefficient
        )
        state = CChainState(
            reference_params=jax.tree.map(lambda value: jnp.array(value), params),
            expected_current_params=jax.tree.map(lambda value: jnp.array(value), params),
            parameter_signature_words=_parameter_signature_words(params),
            coefficient=jnp.asarray(coefficient, dtype=jnp.float32),
            base_loss_window=jnp.zeros((window,), dtype=jnp.float32),
            churn_loss_window=jnp.zeros((window,), dtype=jnp.float32),
            loss_window_count=jnp.asarray(0, dtype=jnp.int32),
            loss_window_cursor=jnp.asarray(0, dtype=jnp.int32),
            commit_count_words=jnp.zeros((2,), dtype=jnp.uint32),
            state_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32),
        )
        sealed = self._seal_state(state)
        if not bool(jax.device_get(self._state_valid(sealed))):
            raise ValueError("failed to construct a valid C-CHAIN state")
        return sealed

    def _state_static_valid(self, state: object) -> bool:
        if not isinstance(state, CChainState):
            return False
        try:
            _validate_parameter_tree(state.reference_params, name="reference_params")
            _validate_parameter_tree(
                state.expected_current_params,
                name="expected_current_params",
            )
        except (TypeError, ValueError):
            return False
        if _parameter_tree_static_signature(
            state.reference_params
        ) != _parameter_tree_static_signature(state.expected_current_params):
            return False
        window = self._config.auto_scale_window
        contracts = (
            (state.parameter_signature_words, (2,), jnp.uint32),
            (state.coefficient, (), jnp.float32),
            (state.base_loss_window, (window,), jnp.float32),
            (state.churn_loss_window, (window,), jnp.float32),
            (state.loss_window_count, (), jnp.int32),
            (state.loss_window_cursor, (), jnp.int32),
            (state.commit_count_words, (2,), jnp.uint32),
            (state.state_integrity_tag, (2,), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype) for value, shape, dtype in contracts
        )

    def _expected_loss_window_count(self, commit_words: Array) -> Array:
        window = self._config.auto_scale_window
        saturated = (commit_words[0] != 0) | (
            commit_words[1] >= jnp.asarray(window, dtype=jnp.uint32)
        )
        return jnp.where(
            saturated,
            jnp.asarray(window, dtype=jnp.int32),
            commit_words[1].astype(jnp.int32),
        )

    def _bounded_window_mean(self, values: Array, count: Array) -> Array:
        def body(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            mean, seen = carry
            active = jnp.asarray(index, dtype=jnp.int32) < count
            next_seen = seen + active.astype(jnp.int32)
            denominator = jnp.maximum(next_seen, jnp.asarray(1, dtype=jnp.int32)).astype(
                jnp.float32
            )
            candidate = mean + (values[index] - mean) / denominator
            return jnp.where(active, candidate, mean), next_seen

        # Windows store finite nonnegative magnitudes, so the incremental mean
        # keeps every intermediate between the observed extrema without a large
        # reduction sum.
        mean, _ = jax.lax.fori_loop(
            0,
            self._config.auto_scale_window,
            body,
            (
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
            ),
        )
        return mean

    def _autoscale_ready(self, commit_words: Array, count: Array) -> Array:
        warmup_words = _int_to_words(self._config.auto_scale_warmup_commits)
        warmup_reached = _words_less_equal(warmup_words, commit_words)
        return (count > 0) & warmup_reached

    def _expected_coefficient(
        self,
        base_window: Array,
        churn_window: Array,
        count: Array,
        commit_words: Array,
    ) -> Array:
        if self._config.target_relative_scale == 0.0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        base_mean = self._bounded_window_mean(base_window, count)
        churn_mean = self._bounded_window_mean(churn_window, count)
        denominator = jnp.maximum(
            churn_mean,
            jnp.asarray(self._config.auto_scale_epsilon, dtype=jnp.float32),
        )
        ratio = (
            jnp.asarray(self._config.target_relative_scale, dtype=jnp.float32)
            * base_mean
            / denominator
        )
        bounded = jnp.clip(
            ratio,
            jnp.asarray(self._config.minimum_coefficient, dtype=jnp.float32),
            jnp.asarray(self._config.maximum_coefficient, dtype=jnp.float32),
        )
        return jnp.where(
            self._autoscale_ready(commit_words, count),
            bounded,
            jnp.asarray(self._config.initial_coefficient, dtype=jnp.float32),
        )

    def _state_valid(self, state: CChainState) -> Bool[Array, ""]:
        expected_count = self._expected_loss_window_count(state.commit_count_words)
        expected_cursor = _words_mod(
            state.commit_count_words,
            self._config.auto_scale_window,
        )
        unused = (
            jnp.arange(self._config.auto_scale_window, dtype=jnp.int32) >= state.loss_window_count
        )
        partial_window = state.loss_window_count < self._config.auto_scale_window
        unused_zero = jnp.all(
            ~unused | ((state.base_loss_window == 0.0) & (state.churn_loss_window == 0.0))
        )
        coefficient_expected = self._expected_coefficient(
            state.base_loss_window,
            state.churn_loss_window,
            state.loss_window_count,
            state.commit_count_words,
        )
        return (
            _tree_finite(state.reference_params)
            & _tree_finite(state.expected_current_params)
            & jnp.array_equal(
                state.parameter_signature_words,
                _parameter_signature_words(state.expected_current_params),
            )
            & jnp.all(jnp.isfinite(state.base_loss_window))
            & jnp.all(jnp.isfinite(state.churn_loss_window))
            & jnp.all(state.base_loss_window >= 0.0)
            & jnp.all(state.churn_loss_window >= 0.0)
            & (state.loss_window_count == expected_count)
            & (state.loss_window_cursor == expected_cursor)
            & (~partial_window | unused_zero)
            & _words_less_equal(state.commit_count_words, self._max_commit_words)
            & jnp.isfinite(state.coefficient)
            & (state.coefficient == coefficient_expected)
            & (state.state_integrity_tag == self._state_tag(state)).all()
        )

    def state_valid(self, state: CChainState) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state has the wrong C-CHAIN static contract")
        return self._state_valid(state)

    def _proposal_tag(self, proposal: CChainProposal) -> Array:
        payload = cast(
            CChainProposal,
            cast(Any, proposal).replace(proposal_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32)),
        )
        return _content_tag(
            (self._config_fingerprint, payload),
            salt=_PROPOSAL_TAG_SALT,
        )

    def _seal_proposal(self, proposal: CChainProposal) -> CChainProposal:
        return cast(
            CChainProposal,
            cast(Any, proposal).replace(proposal_integrity_tag=self._proposal_tag(proposal)),
        )

    def _proposal_static_valid(self, state: CChainState, proposal: object) -> bool:
        if not isinstance(proposal, CChainProposal):
            return False
        signature = _parameter_tree_static_signature(state.expected_current_params)
        if (
            _parameter_tree_static_signature(proposal.source_params) != signature
            or _parameter_tree_static_signature(proposal.gradients) != signature
        ):
            return False
        contracts = (
            (proposal.model_binding_words, (2,), jnp.uint32),
            (proposal.loss_binding_words, (2,), jnp.uint32),
            (proposal.source_state_integrity_tag, (2,), jnp.uint32),
            (proposal.source_commit_count_words, (2,), jnp.uint32),
            (proposal.destination_commit_count_words, (2,), jnp.uint32),
            (proposal.source_parameter_content_tag, (2,), jnp.uint32),
            (proposal.reference_parameter_content_tag, (2,), jnp.uint32),
            (proposal.train_sample_ids_content_tag, (2,), jnp.uint32),
            (proposal.reference_sample_ids_content_tag, (2,), jnp.uint32),
            (proposal.coefficient_used, (), jnp.float32),
            (proposal.base_loss, (), jnp.float32),
            (proposal.churn_loss, (), jnp.float32),
            (proposal.combined_loss, (), jnp.float32),
            (proposal.sample_identity_preflight_valid, (), jnp.bool_),
            (proposal.candidate_finite, (), jnp.bool_),
            (proposal.valid, (), jnp.bool_),
            (proposal.proposal_integrity_tag, (2,), jnp.uint32),
        )
        return all(
            _array_contract(value, shape=shape, dtype=dtype) for value, shape, dtype in contracts
        )

    @staticmethod
    def _require_sample_ids(value: object, *, name: str) -> None:
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be an array")
        array = cast(Any, value)
        if array.dtype != jnp.dtype(jnp.uint32):
            raise TypeError(f"{name} must have exact uint32 dtype")
        if len(array.shape) != 2 or array.shape[0] <= 0 or array.shape[1] != 2:
            raise TypeError(f"{name} must have exact shape [positive_batch, 2]")

    @staticmethod
    def _require_batch_leading_dimension(
        batch: object,
        *,
        name: str,
        expected: int,
    ) -> None:
        leaves = jax.tree.leaves(batch)
        if not leaves:
            raise ValueError(f"{name} must contain at least one batched array leaf")
        observed_batched_leaf = False
        for leaf in leaves:
            if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
                raise TypeError(f"{name} must be an array-only PyTree")
            shape = tuple(cast(Any, leaf).shape)
            if not shape:
                continue
            observed_batched_leaf = True
            if shape[0] != expected:
                raise ValueError(
                    f"{name} non-scalar leaves must share the sample-ID batch dimension"
                )
        if not observed_batched_leaf:
            raise ValueError(f"{name} must contain at least one batched array leaf")

    @staticmethod
    def _sample_ids_nonzero_unique(ids: Array) -> Array:
        nonzero = jnp.all(jnp.any(ids != jnp.asarray(0, dtype=jnp.uint32), axis=1))
        equal = jnp.all(ids[:, None, :] == ids[None, :, :], axis=-1)
        off_diagonal = ~jnp.eye(ids.shape[0], dtype=jnp.bool_)
        unique = ~jnp.any(equal & off_diagonal)
        return nonzero & unique

    @staticmethod
    def _sample_id_sets_disjoint(train_ids: Array, reference_ids: Array) -> Array:
        overlap = jnp.all(
            train_ids[:, None, :] == reference_ids[None, :, :],
            axis=-1,
        )
        return ~jnp.any(overlap)

    def _build_proposal(
        self,
        state: CChainState,
        params: Any,
        gradients: Any,
        *,
        destination_words: Array,
        train_ids_tag: Array,
        reference_ids_tag: Array,
        sample_preflight: Array,
        base_loss: Array,
        churn_loss: Array,
        combined_loss: Array,
        candidate_finite: Array,
        valid: Array,
    ) -> CChainProposal:
        proposal = CChainProposal(
            source_params=params,
            gradients=gradients,
            model_binding_words=self._model_binding_words,
            loss_binding_words=self._loss_binding_words,
            source_state_integrity_tag=state.state_integrity_tag,
            source_commit_count_words=state.commit_count_words,
            destination_commit_count_words=destination_words,
            source_parameter_content_tag=_content_tag(
                params,
                salt=_PARAMETER_TAG_SALT,
            ),
            reference_parameter_content_tag=_content_tag(
                state.reference_params,
                salt=_REFERENCE_TAG_SALT,
            ),
            train_sample_ids_content_tag=train_ids_tag,
            reference_sample_ids_content_tag=reference_ids_tag,
            coefficient_used=state.coefficient,
            base_loss=base_loss,
            churn_loss=churn_loss,
            combined_loss=combined_loss,
            sample_identity_preflight_valid=sample_preflight,
            candidate_finite=candidate_finite,
            valid=valid,
            proposal_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32),
        )
        return self._seal_proposal(proposal)

    def _zero_proposal(
        self,
        state: CChainState,
        params: Any,
        *,
        destination_words: Array,
        train_ids_tag: Array,
        reference_ids_tag: Array,
        sample_preflight: Array,
    ) -> CChainProposal:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return self._build_proposal(
            state,
            params,
            jax.tree.map(jnp.zeros_like, params),
            destination_words=destination_words,
            train_ids_tag=train_ids_tag,
            reference_ids_tag=reference_ids_tag,
            sample_preflight=sample_preflight,
            base_loss=zero,
            churn_loss=zero,
            combined_loss=zero,
            candidate_finite=jnp.asarray(False),
            valid=jnp.asarray(False),
        )

    def _autodiff_proposal(
        self,
        state: CChainState,
        params: Any,
        train_batch: Any,
        reference_batch: Any,
        *,
        reference_batch_size: int,
        destination_words: Array,
        train_ids_tag: Array,
        reference_ids_tag: Array,
        sample_preflight: Array,
    ) -> CChainProposal:
        def combined_objective(candidate_params: Any) -> tuple[Array, tuple[Array, Array, Array]]:
            current_output = self._model_fn(candidate_params, reference_batch)
            reference_output = self._model_fn(state.reference_params, reference_batch)
            if not hasattr(current_output, "shape") or not hasattr(current_output, "dtype"):
                raise TypeError("model_fn must return an array")
            if not hasattr(reference_output, "shape") or not hasattr(reference_output, "dtype"):
                raise TypeError("model_fn must return an array")
            if current_output.dtype != jnp.dtype(jnp.float32):
                raise TypeError("model_fn current output must have exact float32 dtype")
            if reference_output.dtype != jnp.dtype(jnp.float32):
                raise TypeError("model_fn reference output must have exact float32 dtype")
            if (
                current_output.shape != reference_output.shape
                or current_output.ndim == 0
                or current_output.shape[0] != reference_batch_size
                or current_output.size != reference_batch_size
            ):
                raise ValueError(
                    "model_fn outputs must have matching shapes with exactly one "
                    "scalar output per reference sample"
                )
            base_loss = self._base_loss_fn(candidate_params, train_batch)
            if not hasattr(base_loss, "shape") or not hasattr(base_loss, "dtype"):
                raise TypeError("base_loss_fn must return an array scalar")
            if base_loss.shape != () or base_loss.dtype != jnp.dtype(jnp.float32):
                raise TypeError("base_loss_fn must return an exact float32 scalar")
            churn_loss = squared_output_churn(current_output, reference_output)
            combined_loss = base_loss + state.coefficient * churn_loss
            outputs_finite = jnp.all(jnp.isfinite(current_output)) & jnp.all(
                jnp.isfinite(reference_output)
            )
            return combined_loss, (base_loss, churn_loss, outputs_finite)

        (combined_loss, auxiliary), gradients = jax.value_and_grad(
            combined_objective,
            has_aux=True,
        )(params)
        base_loss, churn_loss, outputs_finite = auxiliary
        candidate_finite = (
            outputs_finite
            & jnp.isfinite(base_loss)
            & jnp.isfinite(churn_loss)
            & jnp.isfinite(combined_loss)
            & _tree_finite(gradients)
            & _tree_finite(params)
        )
        return self._build_proposal(
            state,
            params,
            gradients,
            destination_words=destination_words,
            train_ids_tag=train_ids_tag,
            reference_ids_tag=reference_ids_tag,
            sample_preflight=sample_preflight,
            base_loss=base_loss,
            churn_loss=churn_loss,
            combined_loss=combined_loss,
            candidate_finite=candidate_finite,
            valid=candidate_finite,
        )

    def propose(
        self,
        state: CChainState,
        params: Any,
        train_batch: Any,
        reference_batch: Any,
        train_sample_ids: Array,
        reference_sample_ids: Array,
    ) -> CChainProposalResult:
        """Propose one combined gradient after strict disjoint-ID preflight."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong C-CHAIN static contract")
        _validate_parameter_tree(params, name="params")
        if _parameter_tree_static_signature(params) != _parameter_tree_static_signature(
            state.expected_current_params
        ):
            raise TypeError("params do not match the state's parameter PyTree")
        self._require_sample_ids(train_sample_ids, name="train_sample_ids")
        self._require_sample_ids(reference_sample_ids, name="reference_sample_ids")
        self._require_batch_leading_dimension(
            train_batch,
            name="train_batch",
            expected=train_sample_ids.shape[0],
        )
        self._require_batch_leading_dimension(
            reference_batch,
            name="reference_batch",
            expected=reference_sample_ids.shape[0],
        )

        state_valid = self._state_valid(state)
        current_match = _tree_equal(params, state.expected_current_params)
        train_valid = self._sample_ids_nonzero_unique(train_sample_ids)
        reference_valid = self._sample_ids_nonzero_unique(reference_sample_ids)
        disjoint = self._sample_id_sets_disjoint(train_sample_ids, reference_sample_ids)
        sample_preflight = train_valid & reference_valid & disjoint
        destination_words, word_capacity = _checked_words_increment(state.commit_count_words)
        commit_capacity = word_capacity & _words_less_equal(
            destination_words,
            self._max_commit_words,
        )
        preflight = state_valid & current_match & sample_preflight & commit_capacity
        train_ids_tag = _content_tag(train_sample_ids, salt=_TRAIN_IDS_TAG_SALT)
        reference_ids_tag = _content_tag(
            reference_sample_ids,
            salt=_REFERENCE_IDS_TAG_SALT,
        )

        if not isinstance(preflight, jax.core.Tracer) and not bool(jax.device_get(preflight)):
            proposal = self._zero_proposal(
                state,
                params,
                destination_words=destination_words,
                train_ids_tag=train_ids_tag,
                reference_ids_tag=reference_ids_tag,
                sample_preflight=sample_preflight,
            )
            autodiff_count = jnp.asarray(0, dtype=jnp.int32)
        else:
            proposal = cast(
                CChainProposal,
                jax.lax.cond(
                    preflight,
                    lambda: self._autodiff_proposal(
                        state,
                        params,
                        train_batch,
                        reference_batch,
                        reference_batch_size=reference_sample_ids.shape[0],
                        destination_words=destination_words,
                        train_ids_tag=train_ids_tag,
                        reference_ids_tag=reference_ids_tag,
                        sample_preflight=sample_preflight,
                    ),
                    lambda: self._zero_proposal(
                        state,
                        params,
                        destination_words=destination_words,
                        train_ids_tag=train_ids_tag,
                        reference_ids_tag=reference_ids_tag,
                        sample_preflight=sample_preflight,
                    ),
                ),
            )
            autodiff_count = preflight.astype(jnp.int32)

        return CChainProposalResult(
            proposal=proposal,
            diagnostics=CChainProposalDiagnostics(
                state_valid=state_valid,
                current_params_match=current_match,
                train_sample_ids_nonzero_unique=train_valid,
                reference_sample_ids_nonzero_unique=reference_valid,
                sample_id_sets_disjoint=disjoint,
                sample_identity_preflight_valid=sample_preflight,
                commit_capacity_available=commit_capacity,
                preflight_valid=preflight,
                candidate_finite=proposal.candidate_finite,
                autodiff_pass_count=autodiff_count,
                model_binding_authenticated=jnp.asarray(False),
                loss_binding_authenticated=jnp.asarray(False),
                data_provenance_authenticated=jnp.asarray(False),
            ),
        )

    def _proposal_integrity_valid(
        self,
        state: CChainState,
        proposal: CChainProposal,
    ) -> Array:
        expected_destination, capacity = _checked_words_increment(
            proposal.source_commit_count_words
        )
        numeric_finite = (
            _tree_finite(proposal.source_params)
            & _tree_finite(proposal.gradients)
            & jnp.isfinite(proposal.coefficient_used)
            & jnp.isfinite(proposal.base_loss)
            & jnp.isfinite(proposal.churn_loss)
            & jnp.isfinite(proposal.combined_loss)
        )
        return (
            (proposal.proposal_integrity_tag == self._proposal_tag(proposal)).all()
            & (
                proposal.source_parameter_content_tag
                == _content_tag(
                    proposal.source_params,
                    salt=_PARAMETER_TAG_SALT,
                )
            ).all()
            & (
                proposal.reference_parameter_content_tag
                == _content_tag(
                    state.reference_params,
                    salt=_REFERENCE_TAG_SALT,
                )
            ).all()
            & (proposal.model_binding_words == self._model_binding_words).all()
            & (proposal.loss_binding_words == self._loss_binding_words).all()
            & (proposal.destination_commit_count_words == expected_destination).all()
            & capacity
            & _words_less_equal(
                proposal.destination_commit_count_words,
                self._max_commit_words,
            )
            & _words_nonzero(proposal.train_sample_ids_content_tag)
            & _words_nonzero(proposal.reference_sample_ids_content_tag)
            & (proposal.coefficient_used == state.coefficient)
            & (
                proposal.combined_loss
                == proposal.base_loss + proposal.coefficient_used * proposal.churn_loss
            )
            & (proposal.churn_loss >= 0.0)
            & numeric_finite
            & proposal.candidate_finite
            & proposal.sample_identity_preflight_valid
            & proposal.valid
        )

    def proposal_valid(
        self,
        state: CChainState,
        proposal: CChainProposal,
    ) -> Bool[Array, ""]:
        if not self._state_static_valid(state):
            raise TypeError("state has the wrong C-CHAIN static contract")
        if not self._proposal_static_valid(state, proposal):
            return jnp.asarray(False, dtype=jnp.bool_)
        return (
            self._state_valid(state)
            & self._proposal_integrity_valid(state, proposal)
            & (proposal.source_state_integrity_tag == state.state_integrity_tag).all()
            & (proposal.source_commit_count_words == state.commit_count_words).all()
            & _tree_equal(proposal.source_params, state.expected_current_params)
        )

    def commit(
        self,
        state: CChainState,
        proposal: CChainProposal,
        applied_params: Any,
    ) -> CChainCommitResult:
        """Validate and commit without autodiff; optimizer application is external."""

        if not self._state_static_valid(state):
            raise TypeError("state has the wrong C-CHAIN static contract")
        if not self._proposal_static_valid(state, proposal):
            raise TypeError("proposal has the wrong C-CHAIN static contract")
        _validate_parameter_tree(applied_params, name="applied_params")
        if _parameter_tree_static_signature(applied_params) != _parameter_tree_static_signature(
            state.expected_current_params
        ):
            raise TypeError("applied_params do not match the state's parameter PyTree")

        state_valid = self._state_valid(state)
        proposal_integrity = self._proposal_integrity_valid(state, proposal)
        destination_words, word_capacity = _checked_words_increment(state.commit_count_words)
        capacity = word_capacity & _words_less_equal(destination_words, self._max_commit_words)
        source_fresh = (
            (proposal.source_state_integrity_tag == state.state_integrity_tag).all()
            & (proposal.source_commit_count_words == state.commit_count_words).all()
            & (proposal.destination_commit_count_words == destination_words).all()
        )
        source_match = _tree_equal(proposal.source_params, state.expected_current_params)
        reference_match = (
            proposal.reference_parameter_content_tag
            == _content_tag(state.reference_params, salt=_REFERENCE_TAG_SALT)
        ).all()
        binding_match = (proposal.model_binding_words == self._model_binding_words).all() & (
            proposal.loss_binding_words == self._loss_binding_words
        ).all()
        applied_finite = _tree_finite(applied_params)
        preflight = (
            state_valid
            & proposal_integrity
            & proposal.valid
            & source_fresh
            & source_match
            & reference_match
            & binding_match
            & capacity
            & applied_finite
        )

        cursor = state.loss_window_cursor
        base_window = state.base_loss_window.at[cursor].set(jnp.abs(proposal.base_loss))
        churn_window = state.churn_loss_window.at[cursor].set(jnp.abs(proposal.churn_loss))
        window = self._config.auto_scale_window
        next_count = jnp.minimum(
            state.loss_window_count + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(window, dtype=jnp.int32),
        )
        next_cursor = (cursor + jnp.asarray(1, dtype=jnp.int32)) % jnp.asarray(
            window,
            dtype=jnp.int32,
        )
        coefficient = self._expected_coefficient(
            base_window,
            churn_window,
            next_count,
            destination_words,
        )
        candidate = CChainState(
            reference_params=proposal.source_params,
            expected_current_params=applied_params,
            parameter_signature_words=state.parameter_signature_words,
            coefficient=coefficient,
            base_loss_window=base_window,
            churn_loss_window=churn_window,
            loss_window_count=next_count,
            loss_window_cursor=next_cursor,
            commit_count_words=destination_words,
            state_integrity_tag=jnp.zeros((2,), dtype=jnp.uint32),
        )
        candidate = self._seal_state(candidate)
        candidate_valid = self._state_valid(candidate)
        applied = preflight & candidate_valid
        next_state = cast(
            CChainState,
            jax.lax.cond(applied, lambda: candidate, lambda: state),
        )
        return CChainCommitResult(
            state=next_state,
            diagnostics=CChainCommitDiagnostics(
                state_valid=state_valid,
                proposal_integrity_valid=proposal_integrity,
                proposal_declared_valid=proposal.valid,
                source_fresh=source_fresh,
                source_params_match=source_match,
                reference_params_match=reference_match,
                binding_words_match=binding_match,
                commit_capacity_available=capacity,
                applied_params_finite=applied_finite,
                external_optimizer_application_authenticated=jnp.asarray(False),
                candidate_state_valid=candidate_valid,
                applied=applied,
                autodiff_pass_count=jnp.asarray(0, dtype=jnp.int32),
                pre_commit_count_words=state.commit_count_words,
                post_commit_count_words=next_state.commit_count_words,
            ),
        )

    def resource_budget(self, state: CChainState) -> CChainResourceBudget:
        if not self._state_static_valid(state):
            raise TypeError("state has the wrong C-CHAIN static contract")
        if not bool(jax.device_get(self._state_valid(state))):
            raise ValueError("cannot declare resources for an invalid C-CHAIN state")
        zero_proposal = self._zero_proposal(
            state,
            state.expected_current_params,
            destination_words=_checked_words_increment(state.commit_count_words)[0],
            train_ids_tag=jnp.ones((2,), dtype=jnp.uint32),
            reference_ids_tag=jnp.ones((2,), dtype=jnp.uint32),
            sample_preflight=jnp.asarray(False),
        )
        state_scalars, state_bytes = _logical_tree_size(state)
        proposal_scalars, proposal_bytes = _logical_tree_size(zero_proposal)
        parameter_scalars, parameter_bytes = _logical_tree_size(state.expected_current_params)
        return CChainResourceBudget(
            persistent_state_scalars=state_scalars,
            persistent_state_bytes=state_bytes,
            proposal_scalars=proposal_scalars,
            proposal_bytes=proposal_bytes,
            parameter_scalars=parameter_scalars,
            parameter_bytes=parameter_bytes,
            reference_parameter_copies=1,
            expected_current_parameter_copies=1,
            proposal_source_parameter_copies=1,
            proposal_gradient_copies=1,
            auto_scale_window=self._config.auto_scale_window,
            max_commits=self._config.max_commits,
            valid_proposal_autodiff_passes=1,
            rejected_preflight_autodiff_passes=0,
            commit_autodiff_passes=0,
            external_optimizer_state_owned=0,
            dispatch_authority=0,
            output_authority=0,
            scientific_promotion_allowed=False,
            full_sequential_algorithm_reproduced=False,
            efficacy_assessed=False,
            default_agent_integration=False,
            discarded_functional_state_can_repeat_pure_calls=True,
        )


def _gradient_matrix(per_sample_gradients: Any) -> Array:
    leaves = jax.tree.leaves(per_sample_gradients)
    if not leaves:
        raise ValueError("per_sample_gradients must contain at least one leaf")
    sample_count: int | None = None
    flattened: list[Array] = []
    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            raise TypeError("per_sample_gradients must be an array-only PyTree")
        array = cast(Array, leaf)
        if array.dtype != jnp.dtype(jnp.float32):
            raise TypeError("per_sample_gradients leaves must have exact float32 dtype")
        if array.ndim < 1 or array.shape[0] <= 0:
            raise ValueError("every gradient leaf must have a positive leading sample dimension")
        if sample_count is None:
            sample_count = array.shape[0]
        elif array.shape[0] != sample_count:
            raise ValueError("gradient leaves must share one leading sample dimension")
        flattened.append(jnp.reshape(array, (array.shape[0], -1)))
    if sample_count is None:
        raise AssertionError("nonempty gradients lost their sample count")
    matrix = jnp.concatenate(flattened, axis=1)
    if matrix.shape[1] <= 0:
        raise ValueError("per_sample_gradients must contain at least one parameter")
    return matrix


def empirical_ntk_diagnostics(
    per_sample_gradients: Any,
    *,
    delta: float = 0.01,
) -> EmpiricalNTKDiagnostics:
    """Compute a diagnostic empirical NTK Gram matrix and approximate rank.

    The paper rank is the smallest ``k`` whose leading singular values contain
    at least ``1 - delta`` of their total sum.  ``delta=0.01`` is the paper's
    99%-information setting.  Finite all-zero gradients have rank zero.
    Nonfinite input fails closed to zero diagnostic tensors with ``valid=False``.
    """

    parsed_delta = _finite_float32(
        delta,
        name="delta",
        minimum=0.0,
    )
    if parsed_delta >= 1.0:
        raise ValueError("delta must be finite in [0, 1)")
    matrix = _gradient_matrix(per_sample_gradients)
    input_finite = jnp.all(jnp.isfinite(matrix))
    safe_matrix = jnp.where(input_finite, matrix, jnp.zeros_like(matrix))
    raw_gram = safe_matrix @ safe_matrix.T
    gram_finite = jnp.all(jnp.isfinite(raw_gram))
    finite_gram = jnp.where(gram_finite, raw_gram, jnp.zeros_like(raw_gram))
    raw_singular_values = jnp.linalg.svd(finite_gram, compute_uv=False)
    singular_finite = jnp.all(jnp.isfinite(raw_singular_values))
    derived_finite = gram_finite & singular_finite
    valid = input_finite & derived_finite
    gram = jnp.where(valid, finite_gram, jnp.zeros_like(finite_gram))
    singular_values = jnp.where(
        valid,
        raw_singular_values,
        jnp.zeros_like(raw_singular_values),
    )
    total = jnp.sum(singular_values)
    fraction = jnp.asarray(1.0 - parsed_delta, dtype=jnp.float32)
    threshold = fraction * total
    cumulative = jnp.cumsum(singular_values)
    positive_rank = jnp.minimum(
        jnp.sum((cumulative < threshold).astype(jnp.int32)) + 1,
        jnp.asarray(matrix.shape[0], dtype=jnp.int32),
    )
    approximate_rank = jnp.where(
        valid & (total > 0.0),
        positive_rank,
        jnp.asarray(0, dtype=jnp.int32),
    )
    diagonal = jnp.diag(gram)
    off_diagonal_mask = ~jnp.eye(matrix.shape[0], dtype=jnp.bool_)
    off_diagonal = jnp.where(off_diagonal_mask, gram, 0.0)
    off_count = matrix.shape[0] * (matrix.shape[0] - 1)
    safe_off_count = jnp.asarray(max(off_count, 1), dtype=jnp.float32)
    off_absolute_sum = jnp.sum(jnp.abs(off_diagonal))
    off_absolute_mean = jnp.where(
        off_count > 0,
        off_absolute_sum / safe_off_count,
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    off_rms = jnp.where(
        off_count > 0,
        jnp.sqrt(jnp.sum(jnp.square(off_diagonal)) / safe_off_count),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    diagonal_count = jnp.asarray(matrix.shape[0], dtype=jnp.float32)
    diagonal_sum = jnp.sum(diagonal)
    return EmpiricalNTKDiagnostics(
        gradient_matrix=jnp.where(valid, safe_matrix, jnp.zeros_like(safe_matrix)),
        gram_matrix=gram,
        singular_values=singular_values,
        delta=jnp.asarray(parsed_delta, dtype=jnp.float32),
        information_fraction=fraction,
        approximate_rank=approximate_rank,
        off_diagonal_absolute_sum=off_absolute_sum,
        off_diagonal_absolute_mean=off_absolute_mean,
        off_diagonal_rms=off_rms,
        diagonal_sum=diagonal_sum,
        diagonal_mean=diagonal_sum / diagonal_count,
        diagonal_rms=jnp.sqrt(jnp.sum(jnp.square(diagonal)) / diagonal_count),
        diagonal_minimum=jnp.min(diagonal),
        diagonal_maximum=jnp.max(diagonal),
        input_finite=input_finite,
        derived_finite=derived_finite,
        zero_gradient=valid & jnp.all(matrix == 0.0),
        valid=valid,
    )


def save_cchain_checkpoint(
    mechanism: CChain,
    state: CChainState,
    path: str | Path,
) -> None:
    """Save a valid state with strict equation, config, and authority metadata."""

    if not isinstance(mechanism, CChain):
        raise TypeError("mechanism must be a CChain")
    if not bool(jax.device_get(mechanism.state_valid(state))):
        raise ValueError("refusing to save an invalid C-CHAIN state")
    config = mechanism.to_config()
    save_checkpoint(
        state,
        path,
        metadata={
            "schema": CCHAIN_CHECKPOINT_SCHEMA,
            "mechanism_config": config,
            "config_sha256": _config_digest(config),
            "paper_source": _paper_source_metadata(),
            "resource_budget": mechanism.resource_budget(state).to_config(),
            "model_callable_included": False,
            "base_loss_callable_included": False,
            "optimizer_state_included": False,
            "model_binding_authenticated": False,
            "loss_binding_authenticated": False,
            "external_optimizer_application_authenticated": False,
            "full_sequential_algorithm_reproduced": False,
            "efficacy_assessed": False,
            "default_agent_integration": False,
            "dispatch_authority": False,
            "output_authority": False,
            "scientific_promotion_allowed": False,
        },
    )


def load_cchain_checkpoint(
    path: str | Path,
    *,
    params_template: Any,
    model_fn: ModelFn,
    base_loss_fn: BaseLossFn,
) -> tuple[CChain, CChainState]:
    """Restore the sole current schema using a caller-supplied parameter template."""

    metadata = load_checkpoint_metadata(path)
    expected_fields = {
        "schema",
        "mechanism_config",
        "config_sha256",
        "paper_source",
        "resource_budget",
        "model_callable_included",
        "base_loss_callable_included",
        "optimizer_state_included",
        "model_binding_authenticated",
        "loss_binding_authenticated",
        "external_optimizer_application_authenticated",
        "full_sequential_algorithm_reproduced",
        "efficacy_assessed",
        "default_agent_integration",
        "dispatch_authority",
        "output_authority",
        "scientific_promotion_allowed",
    }
    if set(metadata) != expected_fields:
        raise ValueError("C-CHAIN checkpoint metadata fields do not match v1")
    if metadata.get("schema") != CCHAIN_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint is not a C-CHAIN v1 checkpoint")
    config = metadata.get("mechanism_config")
    if not isinstance(config, Mapping):
        raise ValueError("C-CHAIN checkpoint lacks mechanism_config")
    config_dict = dict(config)
    if metadata.get("config_sha256") != _config_digest(config_dict):
        raise ValueError("C-CHAIN checkpoint config digest does not match")
    if metadata.get("paper_source") != _paper_source_metadata():
        raise ValueError("C-CHAIN checkpoint paper source metadata does not match")
    for name in (
        "model_callable_included",
        "base_loss_callable_included",
        "optimizer_state_included",
        "model_binding_authenticated",
        "loss_binding_authenticated",
        "external_optimizer_application_authenticated",
        "full_sequential_algorithm_reproduced",
        "efficacy_assessed",
        "default_agent_integration",
        "dispatch_authority",
        "output_authority",
        "scientific_promotion_allowed",
    ):
        if metadata.get(name) is not False:
            raise ValueError(f"C-CHAIN checkpoint {name} must remain false")
    mechanism = CChain.from_config(
        config_dict,
        model_fn=model_fn,
        base_loss_fn=base_loss_fn,
    )
    template = mechanism.init(params_template)
    if metadata.get("resource_budget") != mechanism.resource_budget(template).to_config():
        raise ValueError("C-CHAIN checkpoint resource budget does not match template")
    restored, second_metadata = load_checkpoint(template, path)
    if second_metadata != metadata:
        raise ValueError("C-CHAIN checkpoint metadata changed between reads")
    state = cast(CChainState, restored)
    if not bool(jax.device_get(mechanism.state_valid(state))):
        raise ValueError("C-CHAIN checkpoint restored an invalid state")
    if mechanism.resource_budget(state).to_config() != metadata["resource_budget"]:
        raise ValueError("C-CHAIN checkpoint restored resource budget does not match")
    return mechanism, state


__all__ = [
    "CCHAIN_AUTOSCALE_CONTROL_PROFILE",
    "CCHAIN_CHECKPOINT_SCHEMA",
    "CCHAIN_CONFIG_SCHEMA",
    "CCHAIN_CONTENT_INTEGRITY_SCOPE",
    "CCHAIN_COMPARATOR_SCOPE",
    "CCHAIN_DEFAULT_AGENT_INTEGRATION",
    "CCHAIN_DISPATCH_AUTHORITY",
    "CCHAIN_EFFICACY_ASSESSED",
    "CCHAIN_EVIDENCE_LEVEL",
    "CCHAIN_EQUATION_8_OBJECTIVE_EXACT",
    "CCHAIN_EXACT_OBJECTIVE_PROFILE",
    "CCHAIN_EXTERNAL_OPTIMIZER_APPLICATION_AUTHENTICATED",
    "CCHAIN_FULL_SEQUENTIAL_ALGORITHM_REPRODUCED",
    "CCHAIN_LOSS_BINDING_AUTHENTICATED",
    "CCHAIN_MECHANISM_STATUS",
    "CCHAIN_MODEL_BINDING_AUTHENTICATED",
    "CCHAIN_NTK_DIAGNOSTIC_STATUS",
    "CCHAIN_OUTPUT_AUTHORITY",
    "CCHAIN_SCIENTIFIC_PROMOTION_ALLOWED",
    "CChain",
    "CChainCommitDiagnostics",
    "CChainCommitResult",
    "CChainConfig",
    "CChainProposal",
    "CChainProposalDiagnostics",
    "CChainProposalResult",
    "CChainResourceBudget",
    "CChainState",
    "EmpiricalNTKDiagnostics",
    "empirical_ntk_diagnostics",
    "load_cchain_checkpoint",
    "save_cchain_checkpoint",
    "squared_output_churn",
]
