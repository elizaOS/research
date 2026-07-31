# mypy: disable-error-code="call-arg"
"""Optimizer-level gradient joy and separate paper-specific actor signals.

:func:`assess_gradient_joy` implements the literal introspective question
"does this candidate gradient/update spark joy?" It checks a candidate
parameter update's first-order effects on independent objective, retention,
and safety probes plus an update-norm trust bound. This is optimizer
control-plane logic, not reward, UX delight, or a generic scalar over the eight
learning-value channels.

Separately, :func:`discrete_delightful_policy_gradient` implements the bounded
discrete-action experiment described in WP5 of
``CONTINUAL_AGENT_IMPLEMENTATION_PLAN.md``. Its paper-specific signal is
``stop_gradient(advantage * -log_probability)``. Callers provide selected
categorical action log-probabilities and detached advantages, then apply the
returned scalar loss only to actor parameters.

Critic, world-model, representation, and safety losses must remain outside the
policy-gradient helper. Kondo compute gating is explicitly unavailable because
masking a loss inside a full autodiff graph does not demonstrate skipped
backward work.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Literal

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

PolicyGradientMode = Literal["ordinary_pg", "delightful_pg"]
GradientCandidateSemantics = Literal["gradient", "update"]
_FLOAT32_MAX = float(np.finfo(np.float32).max)
_FLOAT32_TINY = float(np.finfo(np.float32).tiny)
_FLOAT32_EPSILON = float(np.finfo(np.float32).eps)
_ALIGNMENT_ENDPOINT_TOLERANCE = 4.0 * _FLOAT32_EPSILON


def _validate_float32_config_value(name: str, value: float) -> None:
    """Reject values that become zero or non-finite in float32 computation."""
    value_dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or (
        value_dtype is not None and jnp.issubdtype(value_dtype, jnp.bool_)
    ):
        raise ValueError(f"{name} must be numeric, not boolean")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    magnitude = abs(value)
    if magnitude > _FLOAT32_MAX or (magnitude != 0.0 and magnitude < _FLOAT32_TINY):
        raise ValueError(
            f"{name} must be exactly zero or representable as a finite normal float32"
        )


@chex.dataclass(frozen=True)
class LearningValue:
    """Typed learning-value channels with no implicit aggregate score.

    Every field is supplied by a named producer and retains its own units:

    - ``advantage``: return or reward units from the actor's baseline.
    - ``action_surprisal``: nats from a selected action log-probability.
    - ``delight``: return-nats, ``advantage * action_surprisal``.
    - ``epistemic_surprise``: calibrated reducible-uncertainty units.
    - ``aleatoric_uncertainty``: calibrated outcome-variance units.
    - ``learning_progress``: predictive-loss reduction per model window/version.
    - ``change_probability``: probability in ``[0, 1]`` from a change detector.
    - ``safety_cost``: the safety learner's native cost units.

    The structure deliberately defines no sum, ordering, or universal score.
    Consumers must select the channel appropriate to their decision.
    """

    advantage: Array
    action_surprisal: Array
    delight: Array
    epistemic_surprise: Array
    aleatoric_uncertainty: Array
    learning_progress: Array
    change_probability: Array
    safety_cost: Array


@dataclasses.dataclass(frozen=True)
class GradientJoyConfig:
    """Static contract for a first-order candidate-gradient assessment.

    The candidate is interpreted either as a loss gradient ``g`` whose proposed
    plain-gradient update is ``u = -gradient_step_size * g``, or directly as an
    already formed update ``u``. Callers using momentum, preconditioning,
    clipping, or another optimizer transform must assess the formed update.
    Probe inputs are gradients of quantities to *minimize*, evaluated at the
    same parameters as the candidate.

    A candidate is accepted only when probe independence is caller-attested, all
    evidence is available and finite, the objective probe predicts a strict
    decrease, retention and safety probes remain within their declared
    tolerances, and the update stays inside the norm trust bound. The soft
    weight is the minimum of four named sigmoid factors. Both the raw candidate
    and its tentative soft-weighted update must satisfy the configured
    objective/retention/safety magnitude gates. It is not a sum over
    learning-value channels.

    All numeric values are consumed as float32. Every nonzero value must be a
    finite normal float32, booleans are rejected, and ``max_update_norm`` must
    exceed the norm-resolution floor ``diagnostics_epsilon``. Cosine
    diagnostics are clipped to ``[-1, 1]``; an exact ``1.0`` alignment
    threshold uses a four-machine-epsilon endpoint tolerance for float32
    reduction error.
    """

    candidate_semantics: GradientCandidateSemantics = "gradient"
    gradient_step_size: float = 1.0
    max_update_norm: float = 1.0
    min_objective_decrease: float = 0.0
    max_retention_loss_increase: float = 0.0
    max_safety_cost_increase: float = 0.0
    min_objective_descent_alignment: float = 0.0
    min_retention_descent_alignment: float = 0.0
    min_safety_descent_alignment: float = 0.0
    alignment_temperature: float = 0.1
    norm_temperature: float = 0.1
    diagnostics_epsilon: float = 1.0e-8

    def __post_init__(self) -> None:
        """Validate units, thresholds, and static candidate semantics."""
        if self.candidate_semantics not in ("gradient", "update"):
            raise ValueError("candidate_semantics must be 'gradient' or 'update'")
        positive_finite = {
            "gradient_step_size": self.gradient_step_size,
            "max_update_norm": self.max_update_norm,
            "alignment_temperature": self.alignment_temperature,
            "norm_temperature": self.norm_temperature,
            "diagnostics_epsilon": self.diagnostics_epsilon,
        }
        for name, value in positive_finite.items():
            _validate_float32_config_value(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        finite_thresholds = {
            "min_objective_decrease": self.min_objective_decrease,
            "max_retention_loss_increase": self.max_retention_loss_increase,
            "max_safety_cost_increase": self.max_safety_cost_increase,
        }
        for name, value in finite_thresholds.items():
            _validate_float32_config_value(name, value)
        if self.min_objective_decrease < 0.0:
            raise ValueError("min_objective_decrease must be nonnegative")
        alignments = {
            "min_objective_descent_alignment": (self.min_objective_descent_alignment),
            "min_retention_descent_alignment": (self.min_retention_descent_alignment),
            "min_safety_descent_alignment": self.min_safety_descent_alignment,
        }
        for name, value in alignments.items():
            _validate_float32_config_value(name, value)
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        if self.max_update_norm <= self.diagnostics_epsilon:
            raise ValueError(
                "max_update_norm must be greater than diagnostics_epsilon "
                "so a nonzero trusted update can exist"
            )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "GradientJoyConfig"
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GradientJoyConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class LearningValueAvailability:
    """Named scalar-validity flags for all eight learning-value channels."""

    advantage: Bool[Array, ""]
    action_surprisal: Bool[Array, ""]
    delight: Bool[Array, ""]
    epistemic_surprise: Bool[Array, ""]
    aleatoric_uncertainty: Bool[Array, ""]
    learning_progress: Bool[Array, ""]
    change_probability: Bool[Array, ""]
    safety_cost: Bool[Array, ""]


@chex.dataclass(frozen=True)
class GradientJoyEvidence:
    """Caller-supplied evidence for one candidate update.

    ``objective_probe_gradient`` is accepted under an explicit independence
    contract: callers must obtain it from a validation/probe objective rather
    than reuse the candidate gradient. ``retention_probe_gradient`` measures a
    protected old-skill loss and ``safety_cost_gradient`` measures a separately
    trained safety cost. The implementation cannot infer independence from
    arrays, so ``probe_independence_attested`` must explicitly attest that these
    source requirements hold. ``None``, a false attestation, or a false probe
    availability flag fails closed.

    ``learning_value`` retains the eight typed channels and
    ``learning_value_availability`` declares which producers actually emitted
    them. A channel must be both declared available and scalar-valid. Channels
    are reported separately and are never aggregated into the joy weight.
    """

    objective_probe_gradient: Any | None
    retention_probe_gradient: Any | None
    safety_cost_gradient: Any | None
    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    learning_value: LearningValue
    learning_value_availability: LearningValueAvailability


@chex.dataclass(frozen=True)
class GradientJoyDiagnostics:
    """Named first-order diagnostics; signs follow minimization convention.

    Unqualified update/change fields describe the raw candidate. ``tentative``
    fields describe the soft-weighted candidate before the final hard veto.
    """

    objective_probe_available: Bool[Array, ""]
    retention_probe_available: Bool[Array, ""]
    safety_probe_available: Bool[Array, ""]
    probe_independence_attested: Bool[Array, ""]
    candidate_finite: Bool[Array, ""]
    objective_probe_finite: Bool[Array, ""]
    retention_probe_finite: Bool[Array, ""]
    safety_probe_finite: Bool[Array, ""]
    learning_value_complete: Bool[Array, ""]
    evidence_complete: Bool[Array, ""]
    nonzero_update: Bool[Array, ""]
    tentative_nonzero_update: Bool[Array, ""]
    objective_improves: Bool[Array, ""]
    retention_preserved: Bool[Array, ""]
    safety_preserved: Bool[Array, ""]
    within_trust_region: Bool[Array, ""]
    update_norm: Float[Array, ""]
    objective_probe_norm: Float[Array, ""]
    retention_probe_norm: Float[Array, ""]
    safety_probe_norm: Float[Array, ""]
    predicted_objective_decrease: Float[Array, ""]
    predicted_retention_loss_change: Float[Array, ""]
    predicted_safety_cost_change: Float[Array, ""]
    objective_descent_alignment: Float[Array, ""]
    retention_descent_alignment: Float[Array, ""]
    safety_descent_alignment: Float[Array, ""]
    objective_factor: Float[Array, ""]
    retention_factor: Float[Array, ""]
    safety_factor: Float[Array, ""]
    trust_factor: Float[Array, ""]
    tentative_weight: Float[Array, ""]
    tentative_update_norm: Float[Array, ""]
    predicted_tentative_objective_decrease: Float[Array, ""]
    predicted_tentative_retention_loss_change: Float[Array, ""]
    predicted_tentative_safety_cost_change: Float[Array, ""]


@chex.dataclass(frozen=True)
class GradientJoyAssessment:
    """Detached candidate decision, weakest-link weight, and auditable evidence.

    Acceptance certifies the candidate audit only.  It does not assert that an
    update was representable in a parameter dtype or committed to parameters;
    :class:`GradientJoyApplicationResult` reports that separate boundary.

    ``sparks_joy`` is the literal yes/no spelling of ``accepted``.  It is a
    read-only alias rather than a second stored field, so serialized state and
    JAX PyTree structure have only one source of truth.
    """

    accepted: Bool[Array, ""]
    weight: Float[Array, ""]
    candidate_update: Any
    weighted_update: Any
    learning_value: LearningValue
    channel_availability: LearningValueAvailability
    diagnostics: GradientJoyDiagnostics

    @property
    def sparks_joy(self) -> Bool[Array, ""]:
        """Answer the literal question: does this candidate spark joy?"""
        return self.accepted


@chex.dataclass(frozen=True)
class GradientJoyApplicationResult:
    """Auditable result of atomically attempting an accepted parameter update.

    ``assessment.accepted`` records the formed candidate audit, while
    ``effective_assessment`` separately audits the finite-precision stored
    delta produced by dtype casting and parameter addition. ``applied`` is true
    only when both audits accept, the complete proposed parameter tree is
    finite, and at least one stored parameter value changes. The effective
    assessment is a validation pass: its ``candidate_update`` is the stored
    delta, while its further soft-weighted update is diagnostic and is not
    recursively applied.

    ``proposed_change_nonzero`` reports that comparison independently of the
    three finiteness checks and both audits, so the named fields distinguish a
    safety no-op from an original- or effective-assessment veto.
    """

    parameters: Any
    assessment: GradientJoyAssessment
    effective_assessment: GradientJoyAssessment
    applied: Bool[Array, ""]
    parameters_finite: Bool[Array, ""]
    cast_update_finite: Bool[Array, ""]
    proposed_parameters_finite: Bool[Array, ""]
    proposed_change_nonzero: Bool[Array, ""]


def _detached_float_tree(tree: Any, *, name: str) -> tuple[Any, Any]:
    """Convert a non-empty floating PyTree to detached float32 arrays."""
    leaves, structure = jax.tree_util.tree_flatten(tree)
    if not leaves:
        raise ValueError(f"{name} must be a non-empty PyTree of floating arrays")
    converted: list[Array] = []
    for leaf in leaves:
        array = jnp.asarray(leaf)
        if not jnp.issubdtype(array.dtype, jnp.floating):
            raise ValueError(f"{name} leaves must have floating dtypes")
        converted.append(jax.lax.stop_gradient(jnp.asarray(array, dtype=jnp.float32)))
    return jax.tree_util.tree_unflatten(structure, converted), structure


def _prepare_probe_tree(
    probe: Any | None,
    candidate: Any,
    candidate_structure: Any,
    *,
    name: str,
) -> tuple[Any, Array]:
    """Prepare a probe matching the candidate, or a zero unavailable probe."""
    if probe is None:
        return (
            jax.tree_util.tree_map(jnp.zeros_like, candidate),
            jnp.array(False, dtype=jnp.bool_),
        )
    prepared, structure = _detached_float_tree(probe, name=name)
    if structure != candidate_structure:
        raise ValueError(f"{name} PyTree structure must match candidate")
    candidate_leaves = jax.tree_util.tree_leaves(candidate)
    probe_leaves = jax.tree_util.tree_leaves(prepared)
    for candidate_leaf, probe_leaf in zip(
        candidate_leaves,
        probe_leaves,
        strict=True,
    ):
        if candidate_leaf.shape != probe_leaf.shape:
            raise ValueError(f"{name} leaf shapes must match candidate")
    return prepared, jnp.array(True, dtype=jnp.bool_)


def _tree_dot(left: Any, right: Any) -> Array:
    """Return a scalar inner product for two matching float PyTrees."""
    products = [
        jnp.vdot(left_leaf, right_leaf)
        for left_leaf, right_leaf in zip(
            jax.tree_util.tree_leaves(left),
            jax.tree_util.tree_leaves(right),
            strict=True,
        )
    ]
    return jnp.sum(jnp.stack(products))


def _tree_norm(tree: Any) -> Array:
    """Return the global L2 norm of a float PyTree."""
    return jnp.sqrt(jnp.maximum(_tree_dot(tree, tree), 0.0))


def _tree_is_finite(tree: Any) -> Array:
    """Return whether every element of a PyTree is finite."""
    finite = [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree_util.tree_leaves(tree)]
    return jnp.all(jnp.stack(finite))


def _detached_flag(value: Array, *, name: str) -> Array:
    """Convert an availability input to one detached scalar boolean."""
    flag = jnp.asarray(value)
    if flag.dtype != jnp.bool_:
        raise ValueError(f"{name} must have boolean dtype")
    if flag.shape != ():
        raise ValueError(f"{name} must be scalar")
    return jax.lax.stop_gradient(flag)


def _learning_value_for_gradient_audit(
    learning_value: LearningValue,
    declared_availability: LearningValueAvailability,
) -> tuple[LearningValue, LearningValueAvailability, Array]:
    """Detach, sanitize, and validate eight explicitly available channels."""
    channel_names = (
        "advantage",
        "action_surprisal",
        "delight",
        "epistemic_surprise",
        "aleatoric_uncertainty",
        "learning_progress",
        "change_probability",
        "safety_cost",
    )
    values: dict[str, Array] = {}
    valid: dict[str, Array] = {}
    for name in channel_names:
        raw_value = jnp.asarray(getattr(learning_value, name))
        if jnp.issubdtype(raw_value.dtype, jnp.complexfloating):
            raise ValueError(f"learning_value.{name} must be real-valued")
        if not jnp.issubdtype(raw_value.dtype, jnp.floating):
            raise ValueError(f"learning_value.{name} must have a floating dtype")
        value = jnp.asarray(raw_value, dtype=jnp.float32)
        if value.shape != ():
            raise ValueError(f"learning_value.{name} must be scalar for one gradient audit")
        scalar = jax.lax.stop_gradient(value)
        is_valid = jnp.isfinite(scalar)
        if name in (
            "action_surprisal",
            "epistemic_surprise",
            "aleatoric_uncertainty",
        ):
            is_valid = is_valid & (scalar >= 0.0)
        if name == "change_probability":
            is_valid = is_valid & (scalar >= 0.0) & (scalar <= 1.0)
        declared = _detached_flag(
            getattr(declared_availability, name),
            name=f"learning_value_availability.{name}",
        )
        valid[name] = jax.lax.stop_gradient(declared & is_valid)
        values[name] = jax.lax.stop_gradient(
            jnp.where(is_valid, scalar, jnp.array(0.0, dtype=jnp.float32))
        )

    detached = LearningValue(**values)
    availability = LearningValueAvailability(**valid)
    complete = jnp.all(jnp.stack([valid[name] for name in channel_names]))
    return detached, availability, jax.lax.stop_gradient(complete)


def assess_gradient_joy(
    candidate: Any,
    evidence: GradientJoyEvidence,
    config: GradientJoyConfig | None = None,
) -> GradientJoyAssessment:
    """Assess whether a candidate gradient/update passes a local joy audit.

    Let ``u`` be the proposed parameter update and let ``g_o``, ``g_r``, and
    ``g_s`` be gradients of an independent objective probe, a retention loss,
    and a safety cost. First-order predicted changes are

    ``delta_o = <g_o, u>``, ``delta_r = <g_r, u>``, and
    ``delta_s = <g_s, u>``.

    Negative changes improve their corresponding minimization objectives.
    Candidate alignments and norm produce a tentative weakest-link weight. Hard
    objective, retention, and safety magnitude gates audit both the raw
    candidate and that tentative weighted update. This prevents soft scaling
    from silently violating a required improvement and prevents scaling from
    rescuing a raw harmful candidate. Candidate direction and trust gates remain
    valid because a positive weight preserves direction and only shrinks the
    norm. Missing or invalid evidence and any hard-gate failure zero the final
    weight. No learning-value channel is added to another channel.

    Alignment is exact cosine alignment when the candidate update norm exceeds
    ``diagnostics_epsilon`` and the probe norm and norm product are nonzero and
    representable. It is clipped to ``[-1, 1]``. Unresolved or underflowed norms
    report zero alignment and therefore fail any positive alignment
    requirement. A configured endpoint threshold of exactly ``1.0`` uses a
    four-machine-epsilon float32 comparison tolerance; other thresholds remain
    exact.

    All inputs, diagnostics, the decision, and the returned update are detached.
    This is an optimizer control-plane assessment, not a differentiable
    meta-objective.
    """
    cfg = config or GradientJoyConfig()
    candidate_tree, candidate_structure = _detached_float_tree(
        candidate,
        name="candidate",
    )
    step_size = jnp.asarray(cfg.gradient_step_size, dtype=jnp.float32)
    if cfg.candidate_semantics == "gradient":
        candidate_update = jax.tree_util.tree_map(
            lambda leaf: -step_size * leaf,
            candidate_tree,
        )
    else:
        candidate_update = candidate_tree

    objective_probe, objective_present = _prepare_probe_tree(
        evidence.objective_probe_gradient,
        candidate_update,
        candidate_structure,
        name="objective_probe_gradient",
    )
    retention_probe, retention_present = _prepare_probe_tree(
        evidence.retention_probe_gradient,
        candidate_update,
        candidate_structure,
        name="retention_probe_gradient",
    )
    safety_probe, safety_present = _prepare_probe_tree(
        evidence.safety_cost_gradient,
        candidate_update,
        candidate_structure,
        name="safety_cost_gradient",
    )
    objective_available = (
        _detached_flag(
            evidence.objective_probe_available,
            name="objective_probe_available",
        )
        & objective_present
    )
    retention_available = (
        _detached_flag(
            evidence.retention_probe_available,
            name="retention_probe_available",
        )
        & retention_present
    )
    safety_available = (
        _detached_flag(
            evidence.safety_probe_available,
            name="safety_probe_available",
        )
        & safety_present
    )
    probe_independence_attested = _detached_flag(
        evidence.probe_independence_attested,
        name="probe_independence_attested",
    )
    (
        detached_learning_value,
        channel_availability,
        learning_value_complete,
    ) = _learning_value_for_gradient_audit(
        evidence.learning_value,
        evidence.learning_value_availability,
    )

    candidate_finite = _tree_is_finite(candidate_update)
    objective_probe_finite = _tree_is_finite(objective_probe)
    retention_probe_finite = _tree_is_finite(retention_probe)
    safety_probe_finite = _tree_is_finite(safety_probe)
    evidence_complete = (
        candidate_finite
        & objective_available
        & retention_available
        & safety_available
        & probe_independence_attested
        & objective_probe_finite
        & retention_probe_finite
        & safety_probe_finite
        & learning_value_complete
    )

    update_norm_raw = _tree_norm(candidate_update)
    objective_norm_raw = _tree_norm(objective_probe)
    retention_norm_raw = _tree_norm(retention_probe)
    safety_norm_raw = _tree_norm(safety_probe)
    objective_change_raw = _tree_dot(objective_probe, candidate_update)
    retention_change_raw = _tree_dot(retention_probe, candidate_update)
    safety_change_raw = _tree_dot(safety_probe, candidate_update)

    def _finite_or_zero(value: Array) -> Array:
        return jnp.where(
            jnp.isfinite(value),
            value,
            jnp.array(0.0, dtype=jnp.float32),
        )

    update_norm = _finite_or_zero(update_norm_raw)
    objective_norm = _finite_or_zero(objective_norm_raw)
    retention_norm = _finite_or_zero(retention_norm_raw)
    safety_norm = _finite_or_zero(safety_norm_raw)
    objective_change = _finite_or_zero(objective_change_raw)
    retention_change = _finite_or_zero(retention_change_raw)
    safety_change = _finite_or_zero(safety_change_raw)
    epsilon = jnp.asarray(cfg.diagnostics_epsilon, dtype=jnp.float32)

    def _descent_alignment(
        change: Array,
        probe_norm: Array,
    ) -> Array:
        denominator = update_norm * probe_norm
        norms_are_resolved = (
            (update_norm > epsilon)
            & (probe_norm > 0.0)
            & jnp.isfinite(denominator)
            & (denominator > 0.0)
        )
        return jnp.where(
            norms_are_resolved,
            jnp.clip(-change / denominator, -1.0, 1.0),
            jnp.array(0.0, dtype=jnp.float32),
        )

    def _meets_alignment_threshold(
        alignment: Array,
        configured_threshold: float,
    ) -> Array:
        threshold = jnp.asarray(configured_threshold, dtype=jnp.float32)
        endpoint_tolerance = jnp.asarray(
            _ALIGNMENT_ENDPOINT_TOLERANCE,
            dtype=jnp.float32,
        )
        effective_threshold = jnp.where(
            threshold == jnp.array(1.0, dtype=jnp.float32),
            threshold - endpoint_tolerance,
            threshold,
        )
        return alignment >= effective_threshold

    objective_alignment = _descent_alignment(
        objective_change,
        objective_norm,
    )
    retention_alignment = _descent_alignment(
        retention_change,
        retention_norm,
    )
    safety_alignment = _descent_alignment(safety_change, safety_norm)
    predicted_objective_decrease = -objective_change
    nonzero_update = update_norm > epsilon
    within_trust_region = nonzero_update & (
        update_norm <= jnp.asarray(cfg.max_update_norm, dtype=jnp.float32)
    )

    alignment_temperature = jnp.asarray(
        cfg.alignment_temperature,
        dtype=jnp.float32,
    )
    norm_temperature = jnp.asarray(cfg.norm_temperature, dtype=jnp.float32)
    objective_factor = jax.nn.sigmoid(
        (
            objective_alignment
            - jnp.asarray(
                cfg.min_objective_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    retention_factor = jax.nn.sigmoid(
        (
            retention_alignment
            - jnp.asarray(
                cfg.min_retention_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    safety_factor = jax.nn.sigmoid(
        (
            safety_alignment
            - jnp.asarray(
                cfg.min_safety_descent_alignment,
                dtype=jnp.float32,
            )
        )
        / alignment_temperature
    )
    trust_factor = jax.nn.sigmoid(
        (jnp.asarray(cfg.max_update_norm, dtype=jnp.float32) - update_norm) / norm_temperature
    )
    weakest_link_weight = jnp.min(
        jnp.stack(
            [
                objective_factor,
                retention_factor,
                safety_factor,
                trust_factor,
            ]
        )
    )
    tentative_weight = jax.lax.stop_gradient(weakest_link_weight)
    tentative_update_norm = tentative_weight * update_norm
    tentative_objective_change = tentative_weight * objective_change
    tentative_retention_change = tentative_weight * retention_change
    tentative_safety_change = tentative_weight * safety_change
    predicted_tentative_objective_decrease = -tentative_objective_change
    tentative_nonzero_update = tentative_update_norm > epsilon

    objective_improves = (
        (objective_norm > 0.0)
        & (
            predicted_objective_decrease
            > jnp.asarray(cfg.min_objective_decrease, dtype=jnp.float32)
        )
        & (
            predicted_tentative_objective_decrease
            > jnp.asarray(cfg.min_objective_decrease, dtype=jnp.float32)
        )
        & _meets_alignment_threshold(
            objective_alignment,
            cfg.min_objective_descent_alignment,
        )
    )
    retention_preserved = (
        (retention_change <= jnp.asarray(cfg.max_retention_loss_increase, dtype=jnp.float32))
        & (
            tentative_retention_change
            <= jnp.asarray(cfg.max_retention_loss_increase, dtype=jnp.float32)
        )
    ) & _meets_alignment_threshold(
        retention_alignment,
        cfg.min_retention_descent_alignment,
    )
    safety_preserved = (
        (safety_change <= jnp.asarray(cfg.max_safety_cost_increase, dtype=jnp.float32))
        & (
            tentative_safety_change
            <= jnp.asarray(cfg.max_safety_cost_increase, dtype=jnp.float32)
        )
    ) & _meets_alignment_threshold(
        safety_alignment,
        cfg.min_safety_descent_alignment,
    )
    accepted = (
        evidence_complete
        & objective_improves
        & retention_preserved
        & safety_preserved
        & within_trust_region
        & tentative_nonzero_update
    )
    weight = jax.lax.stop_gradient(
        jnp.where(
            accepted,
            weakest_link_weight,
            jnp.array(0.0, dtype=jnp.float32),
        )
    )
    accepted = jax.lax.stop_gradient(accepted)
    safe_update = jax.tree_util.tree_map(
        lambda leaf: jnp.where(jnp.isfinite(leaf), leaf, jnp.zeros_like(leaf)),
        candidate_update,
    )
    weighted_update = jax.tree_util.tree_map(
        lambda leaf: jax.lax.stop_gradient(weight * leaf),
        safe_update,
    )
    safe_update = jax.tree_util.tree_map(jax.lax.stop_gradient, safe_update)

    diagnostics = GradientJoyDiagnostics(
        objective_probe_available=jax.lax.stop_gradient(objective_available),
        retention_probe_available=jax.lax.stop_gradient(retention_available),
        safety_probe_available=jax.lax.stop_gradient(safety_available),
        probe_independence_attested=jax.lax.stop_gradient(probe_independence_attested),
        candidate_finite=jax.lax.stop_gradient(candidate_finite),
        objective_probe_finite=jax.lax.stop_gradient(objective_probe_finite),
        retention_probe_finite=jax.lax.stop_gradient(retention_probe_finite),
        safety_probe_finite=jax.lax.stop_gradient(safety_probe_finite),
        learning_value_complete=learning_value_complete,
        evidence_complete=jax.lax.stop_gradient(evidence_complete),
        nonzero_update=jax.lax.stop_gradient(nonzero_update),
        tentative_nonzero_update=jax.lax.stop_gradient(tentative_nonzero_update),
        objective_improves=jax.lax.stop_gradient(objective_improves),
        retention_preserved=jax.lax.stop_gradient(retention_preserved),
        safety_preserved=jax.lax.stop_gradient(safety_preserved),
        within_trust_region=jax.lax.stop_gradient(within_trust_region),
        update_norm=jax.lax.stop_gradient(update_norm),
        objective_probe_norm=jax.lax.stop_gradient(objective_norm),
        retention_probe_norm=jax.lax.stop_gradient(retention_norm),
        safety_probe_norm=jax.lax.stop_gradient(safety_norm),
        predicted_objective_decrease=jax.lax.stop_gradient(predicted_objective_decrease),
        predicted_retention_loss_change=jax.lax.stop_gradient(retention_change),
        predicted_safety_cost_change=jax.lax.stop_gradient(safety_change),
        objective_descent_alignment=jax.lax.stop_gradient(objective_alignment),
        retention_descent_alignment=jax.lax.stop_gradient(retention_alignment),
        safety_descent_alignment=jax.lax.stop_gradient(safety_alignment),
        objective_factor=jax.lax.stop_gradient(objective_factor),
        retention_factor=jax.lax.stop_gradient(retention_factor),
        safety_factor=jax.lax.stop_gradient(safety_factor),
        trust_factor=jax.lax.stop_gradient(trust_factor),
        tentative_weight=tentative_weight,
        tentative_update_norm=jax.lax.stop_gradient(tentative_update_norm),
        predicted_tentative_objective_decrease=jax.lax.stop_gradient(
            predicted_tentative_objective_decrease
        ),
        predicted_tentative_retention_loss_change=jax.lax.stop_gradient(
            tentative_retention_change
        ),
        predicted_tentative_safety_cost_change=jax.lax.stop_gradient(
            tentative_safety_change
        ),
    )
    return GradientJoyAssessment(
        accepted=accepted,
        weight=weight,
        candidate_update=safe_update,
        weighted_update=weighted_update,
        learning_value=detached_learning_value,
        channel_availability=channel_availability,
        diagnostics=diagnostics,
    )


def apply_gradient_joy_update(
    parameters: Any,
    candidate: Any,
    evidence: GradientJoyEvidence,
    config: GradientJoyConfig | None = None,
) -> GradientJoyApplicationResult:
    """Assess and atomically apply one joy-weighted parameter update.

    The assessment is created internally so callers cannot substitute a forged
    acceptance decision.  ``parameters`` and the assessed update must be
    non-empty PyTrees with exactly the same structure and leaf shapes; every
    leaf must have a real floating dtype.  These static contract violations
    raise :class:`ValueError` instead of permitting JAX broadcasting.

    Update leaves are cast to their corresponding parameter dtypes before
    addition. The effective stored delta (proposed parameters minus input
    parameters) is then independently re-audited under ``candidate_semantics =
    "update"`` with the same evidence and thresholds. This conservative second
    audit can veto a finite quantized update whose direction, magnitude, or
    trust-bound status no longer matches the accepted float32 candidate.

    The update is applied only when both assessments accept it, every parameter,
    cast update, and proposed parameter is finite, and at least one proposed
    stored value differs from its input. Otherwise every parameter leaf is
    returned unchanged, so a rejection, unsafe cast, overflow, non-finite leaf,
    or update wholly lost to finite-precision addition cannot produce or claim
    a partial update. Existing non-finite parameters are not repaired; they are
    preserved without applying any part of the candidate.

    The typed result keeps the original and effective audit decisions distinct
    from ``applied`` and exposes detached scalar booleans for every application
    check. The helper is compatible with :func:`jax.jit` when the PyTree
    structures and ``config`` are static.
    """
    assessment = assess_gradient_joy(candidate, evidence, config)
    parameter_structure: Any
    update_structure: Any
    parameter_leaves, parameter_structure = jax.tree_util.tree_flatten(parameters)
    update_leaves, update_structure = jax.tree_util.tree_flatten(
        assessment.weighted_update
    )
    if not parameter_leaves:
        raise ValueError("parameters must be a non-empty PyTree of floating arrays")
    if parameter_structure != update_structure:
        raise ValueError("parameter and joy-update PyTree structures must match exactly")

    prepared_parameters: list[Array] = []
    prepared_updates: list[Array] = []
    for position, (parameter_leaf, update_leaf) in enumerate(
        zip(parameter_leaves, update_leaves, strict=True)
    ):
        parameter_array = jnp.asarray(parameter_leaf)
        update_array = jnp.asarray(update_leaf)
        if not jnp.issubdtype(parameter_array.dtype, jnp.floating):
            raise ValueError(
                f"parameter leaf {position} must have a real floating dtype"
            )
        if not jnp.issubdtype(update_array.dtype, jnp.floating):
            raise ValueError(
                f"joy-update leaf {position} must have a real floating dtype"
            )
        if parameter_array.shape != update_array.shape:
            raise ValueError(
                f"parameter and joy-update leaf {position} shapes must match exactly"
            )
        prepared_parameters.append(parameter_array)
        prepared_updates.append(
            jnp.asarray(update_array, dtype=parameter_array.dtype)
        )

    parameter_tree = jax.tree_util.tree_unflatten(
        parameter_structure,
        prepared_parameters,
    )
    update_tree = jax.tree_util.tree_unflatten(
        parameter_structure,
        prepared_updates,
    )
    proposed_parameters = jax.tree_util.tree_map(
        lambda parameter, update: parameter + update,
        parameter_tree,
        update_tree,
    )
    effective_update = jax.tree_util.tree_map(
        lambda parameter, proposed: proposed - parameter,
        parameter_tree,
        proposed_parameters,
    )
    effective_config = dataclasses.replace(
        config or GradientJoyConfig(),
        candidate_semantics="update",
    )
    effective_assessment = assess_gradient_joy(
        effective_update,
        evidence,
        effective_config,
    )
    parameters_finite = jax.lax.stop_gradient(_tree_is_finite(parameter_tree))
    cast_update_finite = jax.lax.stop_gradient(_tree_is_finite(update_tree))
    proposed_parameters_finite = jax.lax.stop_gradient(
        _tree_is_finite(proposed_parameters)
    )
    proposed_change_raw = jnp.any(
        jnp.stack(
            [
                jnp.any(proposed != parameter)
                for parameter, proposed in zip(
                    jax.tree_util.tree_leaves(parameter_tree),
                    jax.tree_util.tree_leaves(proposed_parameters),
                    strict=True,
                )
            ]
        )
    )
    proposed_change_nonzero = jax.lax.stop_gradient(proposed_change_raw)
    applied = jax.lax.stop_gradient(
        assessment.accepted
        & effective_assessment.accepted
        & parameters_finite
        & cast_update_finite
        & proposed_parameters_finite
        & proposed_change_nonzero
    )
    updated_parameters = jax.tree_util.tree_map(
        lambda parameter, proposed: jnp.where(
            applied,
            proposed,
            parameter,
        ),
        parameter_tree,
        proposed_parameters,
    )
    return GradientJoyApplicationResult(
        parameters=updated_parameters,
        assessment=assessment,
        effective_assessment=effective_assessment,
        applied=applied,
        parameters_finite=parameters_finite,
        cast_update_finite=cast_update_finite,
        proposed_parameters_finite=proposed_parameters_finite,
        proposed_change_nonzero=proposed_change_nonzero,
    )


@dataclasses.dataclass(frozen=True)
class DelightfulPolicyGradientConfig:
    """Configuration for the discrete Delightful Policy Gradient experiment.

    Args:
        mode: ``"ordinary_pg"`` applies the matched ordinary policy-gradient
            loss. ``"delightful_pg"`` applies the detached sigmoid delight
            weight.
        temperature: Positive sigmoid temperature in return-nats.
        actor_trace_lambda: Must be exactly zero. A current-sample delight gate
            cannot multiply an eligibility trace containing historical score
            gradients.
        diagnostics_epsilon: Positive denominator floor used only in reported
            diagnostics.
        kondo_enabled: Reserved fail-closed flag. ``True`` is rejected because
            this helper does not skip compiled backward work.

    Nonzero numeric controls must be finite normal float32 values; this prevents
    a positive Python value from becoming zero or infinity inside the JAX loss.
    """

    mode: PolicyGradientMode = "delightful_pg"
    temperature: float = 1.0
    actor_trace_lambda: float = 0.0
    diagnostics_epsilon: float = 1.0e-8
    kondo_enabled: bool = False

    def __post_init__(self) -> None:
        """Validate the experimental contract."""
        if self.mode not in ("ordinary_pg", "delightful_pg"):
            raise ValueError("mode must be 'ordinary_pg' or 'delightful_pg'")
        _validate_float32_config_value("temperature", self.temperature)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        _validate_float32_config_value("actor_trace_lambda", self.actor_trace_lambda)
        if self.actor_trace_lambda != 0.0:
            raise ValueError(
                "Delightful Policy Gradient requires actor_trace_lambda=0; "
                "a current-sample gate cannot reweight historical score gradients"
            )
        _validate_float32_config_value("diagnostics_epsilon", self.diagnostics_epsilon)
        if self.diagnostics_epsilon <= 0.0:
            raise ValueError("diagnostics_epsilon must be positive")
        if self.kondo_enabled:
            raise ValueError(
                "Kondo compute gating is unavailable: this helper computes all "
                "samples and does not skip measured compiled backward work"
            )

    def to_config(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        payload = dataclasses.asdict(self)
        payload["type"] = "DelightfulPolicyGradientConfig"
        return payload

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DelightfulPolicyGradientConfig:
        """Reconstruct from :meth:`to_config` output."""
        payload = dict(config)
        payload.pop("type", None)
        return cls(**payload)


@chex.dataclass(frozen=True)
class DelightfulPolicyGradientDiagnostics:
    """Batch diagnostics for a policy-gradient gate.

    Conditional gate rates are mean soft gate probabilities, not claims of
    skipped gradient computations. Effective sample size is
    ``sum(w)^2 / sum(w^2)``.
    """

    effective_sample_size: Float[Array, ""]
    effective_sample_fraction: Float[Array, ""]
    mean_gate_rate: Float[Array, ""]
    positive_advantage_gate_rate: Float[Array, ""]
    negative_advantage_gate_rate: Float[Array, ""]
    positive_advantage_fraction: Float[Array, ""]
    negative_advantage_fraction: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightfulPolicyGradientResult:
    """Actor loss, typed actor signals, and detached sample coefficients."""

    actor_loss: Float[Array, ""]
    ordinary_actor_loss: Float[Array, ""]
    advantage: Array
    action_surprisal: Array
    delight: Array
    sample_weights: Array
    actor_coefficients: Array
    diagnostics: DelightfulPolicyGradientDiagnostics


@chex.dataclass(frozen=True)
class DelightStratumDiagnostics:
    """Diagnostics for one of common/rare × success/failure."""

    count: Int[Array, ""]
    fraction: Float[Array, ""]
    mean_advantage: Float[Array, ""]
    mean_action_surprisal: Float[Array, ""]
    mean_delight: Float[Array, ""]
    mean_gate_rate: Float[Array, ""]
    mean_gated_advantage: Float[Array, ""]


@chex.dataclass(frozen=True)
class DelightOutcomeStratification:
    """Rare/common success/failure diagnostics for gambling audits."""

    common_success: DelightStratumDiagnostics
    common_failure: DelightStratumDiagnostics
    rare_success: DelightStratumDiagnostics
    rare_failure: DelightStratumDiagnostics
    common_advantage_mean: Float[Array, ""]
    rare_advantage_mean: Float[Array, ""]
    common_gated_advantage_mean: Float[Array, ""]
    rare_gated_advantage_mean: Float[Array, ""]
    common_advantage_variance: Float[Array, ""]
    rare_advantage_variance: Float[Array, ""]


def _masked_mean(values: Array, mask: Array) -> Array:
    """Return a finite conditional mean, or zero when the mask is empty."""
    value_arr = jnp.ravel(jnp.asarray(values, dtype=jnp.float32))
    mask_arr = jnp.ravel(jnp.asarray(mask, dtype=jnp.bool_))
    mask_float = mask_arr.astype(jnp.float32)
    count = jnp.sum(mask_float)
    return jnp.sum(jnp.where(mask_arr, value_arr, 0.0)) / jnp.maximum(count, 1.0)


def discrete_delightful_policy_gradient(
    selected_log_probabilities: Array,
    advantages: Array,
    config: DelightfulPolicyGradientConfig | None = None,
) -> DelightfulPolicyGradientResult:
    """Return a matched ordinary or delightful discrete actor loss.

    Args:
        selected_log_probabilities: Finite log-probabilities of the sampled
            categorical actions under the explicitly chosen actor objective.
            Off-policy callers must perform their declared importance
            correction separately; delight is never a substitute for it.
        advantages: Baseline-derived per-sample advantages. The actor
            coefficient is stopped so this loss cannot train the baseline.
        config: Static experimental configuration. Close over it when using
            :func:`jax.jit`.

    Returns:
        Per-sample actor signals, detached coefficients, scalar losses, and
        diagnostics. The selected loss is exactly
        ``-mean(stop_gradient(weight * advantage) * log_probability)``.
    """
    cfg = config or DelightfulPolicyGradientConfig()
    log_prob = jnp.asarray(selected_log_probabilities, dtype=jnp.float32)
    advantage = jnp.asarray(advantages, dtype=jnp.float32)
    if log_prob.shape != advantage.shape:
        raise ValueError("selected_log_probabilities and advantages must have the same shape")
    if log_prob.size == 0:
        raise ValueError("at least one policy-gradient sample is required")

    action_surprisal = -log_prob
    delight = jax.lax.stop_gradient(advantage * action_surprisal)
    if cfg.mode == "ordinary_pg":
        sample_weights = jnp.ones_like(delight)
    else:
        temperature = jnp.asarray(cfg.temperature, dtype=jnp.float32)
        sample_weights = jax.nn.sigmoid(delight / temperature)
    sample_weights = jax.lax.stop_gradient(sample_weights)
    actor_coefficients = jax.lax.stop_gradient(sample_weights * advantage)
    ordinary_coefficients = jax.lax.stop_gradient(advantage)

    actor_loss = -jnp.mean(actor_coefficients * log_prob)
    ordinary_actor_loss = -jnp.mean(ordinary_coefficients * log_prob)

    flat_weights = jnp.ravel(sample_weights)
    flat_advantages = jnp.ravel(advantage)
    positive = flat_advantages > 0.0
    negative = flat_advantages < 0.0
    n_samples = jnp.asarray(flat_weights.size, dtype=jnp.float32)
    epsilon = jnp.asarray(cfg.diagnostics_epsilon, dtype=jnp.float32)
    weight_sum = jnp.sum(flat_weights)
    effective_sample_size = weight_sum**2 / jnp.maximum(
        jnp.sum(flat_weights**2),
        epsilon,
    )
    diagnostics = DelightfulPolicyGradientDiagnostics(
        effective_sample_size=effective_sample_size,
        effective_sample_fraction=effective_sample_size / jnp.maximum(n_samples, 1.0),
        mean_gate_rate=jnp.mean(flat_weights),
        positive_advantage_gate_rate=_masked_mean(flat_weights, positive),
        negative_advantage_gate_rate=_masked_mean(flat_weights, negative),
        positive_advantage_fraction=jnp.mean(positive.astype(jnp.float32)),
        negative_advantage_fraction=jnp.mean(negative.astype(jnp.float32)),
    )
    return DelightfulPolicyGradientResult(
        actor_loss=actor_loss,
        ordinary_actor_loss=ordinary_actor_loss,
        advantage=advantage,
        action_surprisal=action_surprisal,
        delight=delight,
        sample_weights=sample_weights,
        actor_coefficients=actor_coefficients,
        diagnostics=diagnostics,
    )


def _stratum_diagnostics(
    result: DelightfulPolicyGradientResult,
    mask: Array,
) -> DelightStratumDiagnostics:
    """Summarize one explicit outcome stratum."""
    mask_arr = jnp.ravel(jnp.asarray(mask, dtype=jnp.bool_))
    count = jnp.sum(mask_arr.astype(jnp.int32))
    n_samples = jnp.asarray(jnp.ravel(result.advantage).size, dtype=jnp.float32)
    return DelightStratumDiagnostics(
        count=count,
        fraction=count.astype(jnp.float32) / jnp.maximum(n_samples, 1.0),
        mean_advantage=_masked_mean(result.advantage, mask_arr),
        mean_action_surprisal=_masked_mean(result.action_surprisal, mask_arr),
        mean_delight=_masked_mean(result.delight, mask_arr),
        mean_gate_rate=_masked_mean(result.sample_weights, mask_arr),
        mean_gated_advantage=_masked_mean(result.actor_coefficients, mask_arr),
    )


def stratify_delight_outcomes(
    result: DelightfulPolicyGradientResult,
    rare_mask: Array,
) -> DelightOutcomeStratification:
    """Stratify DG diagnostics into common/rare success/failure groups.

    ``rare_mask`` must be defined by the experiment protocol independently of
    observed reward or advantage—for example, from the behavior action
    probability. Positive advantage denotes success and negative advantage
    denotes failure. Zero-advantage samples are retained in aggregate moments
    but belong to neither success nor failure stratum.
    """
    advantages = jnp.ravel(jnp.asarray(result.advantage, dtype=jnp.float32))
    rare = jnp.ravel(jnp.asarray(rare_mask, dtype=jnp.bool_))
    if rare.shape != advantages.shape:
        raise ValueError("rare_mask must match the flattened advantage shape")
    common = ~rare
    success = advantages > 0.0
    failure = advantages < 0.0

    common_mean = _masked_mean(advantages, common)
    rare_mean = _masked_mean(advantages, rare)
    common_variance = _masked_mean((advantages - common_mean) ** 2, common)
    rare_variance = _masked_mean((advantages - rare_mean) ** 2, rare)
    return DelightOutcomeStratification(
        common_success=_stratum_diagnostics(result, common & success),
        common_failure=_stratum_diagnostics(result, common & failure),
        rare_success=_stratum_diagnostics(result, rare & success),
        rare_failure=_stratum_diagnostics(result, rare & failure),
        common_advantage_mean=common_mean,
        rare_advantage_mean=rare_mean,
        common_gated_advantage_mean=_masked_mean(result.actor_coefficients, common),
        rare_gated_advantage_mean=_masked_mean(result.actor_coefficients, rare),
        common_advantage_variance=common_variance,
        rare_advantage_variance=rare_variance,
    )


__all__ = [
    "DelightOutcomeStratification",
    "DelightStratumDiagnostics",
    "DelightfulPolicyGradientConfig",
    "DelightfulPolicyGradientDiagnostics",
    "DelightfulPolicyGradientResult",
    "GradientCandidateSemantics",
    "GradientJoyApplicationResult",
    "GradientJoyAssessment",
    "GradientJoyConfig",
    "GradientJoyDiagnostics",
    "GradientJoyEvidence",
    "LearningValue",
    "LearningValueAvailability",
    "PolicyGradientMode",
    "apply_gradient_joy_update",
    "assess_gradient_joy",
    "discrete_delightful_policy_gradient",
    "stratify_delight_outcomes",
]
