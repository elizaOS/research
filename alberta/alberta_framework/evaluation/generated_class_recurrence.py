"""Fail-closed substrate for a generated-class recurrence v0 experiment.

This module deliberately stops short of an executable lifecycle experiment.
It fixes a small compositional target grammar, a development-only expression
manifest, a one-head recurrence schedule, capacity-matched control declarations,
exact JAX-state byte accounting, and raw descriptive metrics.  It does not own
a scientific protocol, thresholds, protected targets, evidence authority, or a
promotion path.

The missing causal lifecycle work is explicit.  In particular, deleting a
retired expression must eventually scrub every active/candidate occurrence and
descendant together with all associated heads, traces, utilities, descriptors,
and hidden archives.  A matched sham scrub and a D-never-seen twin are also
required before any runner may be authorized.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.compositional_features import (
    FEATURE_VALUE_CLIP,
    GENERATION_ROBUST_RECURSIVE,
    OP_GATED,
    OP_PRODUCT,
    OP_RAW,
    OP_SUM,
    OP_TANH,
    CompositionalFeatureLearner,
    CompositionalFeatureState,
)

GENERATED_CLASS_RECURRENCE_V0_SCHEMA = (
    "alberta.generated-class-recurrence.development.v0"
)
GENERATED_CLASS_EXPRESSION_MANIFEST_SCHEMA = (
    "alberta.generated-class-expression-manifest.development.v0"
)
GENERATED_CLASS_STATUS = "DEVELOPMENT_SUBSTRATE_EXECUTION_UNAUTHORIZED"

DEVELOPMENT_EXPRESSION_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/development-expressions"
)
PROTECTED_EXPRESSION_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/protected-expressions"
)

FULL_LIFECYCLE = "full_lifecycle"
RANDOM_CURATION = "random_curation"
FROZEN_LIFECYCLE = "frozen_lifecycle"
ZERO_CANDIDATE_HEAD_CARRY = "zero_candidate_head_carry"
FINITE_DEGREE_TWO_ARCHIVE_CEILING = "finite_degree_two_archive_ceiling"

_EXPRESSION_MAX_DEPTH = 3
_INPUT_DIM = 4
_ACTIVE_SLOTS = 14
_CANDIDATE_SLOTS = 8
_N_TASKS = 1
_CONTEXT_ID = 0
_GENERATOR_RESOURCE_CONTEXTS = 1
_GENERATOR_POLICY_COUNT = 4

_PHASE_ORDER = ("A", "B", "A", "D", "A", "C", "A", "D", "A")
_PHASE_LENGTH_NAMESPACE = (
    "alberta/generated-class-recurrence/v0/evaluator/phase-lengths"
)
_PHASE_LENGTH_KEY_DATA = (0x47A2C91D, 0xB80D6E35)
_PHASE_LENGTH_CANDIDATES = (353, 389, 421, 457, 503, 541, 587, 631, 677)
_CURATION_INTERVAL = 32
_CONSERVATIVE_LIFECYCLE_CURATION_LOWER_BOUND = 7
_DEVELOPMENT_CURATION_MARGIN_MULTIPLIER = 4

ExpressionOp = Literal["raw", "sum", "product", "tanh", "gate"]


class GeneratedClassProtocolNotReadyError(RuntimeError):
    """Raised when code tries to execute the intentionally incomplete v0."""


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedExpression:
    """One canonical node in the depth-limited generated target grammar."""

    op: ExpressionOp
    raw_index: int | None = None
    left: GeneratedExpression | None = None
    right: GeneratedExpression | None = None
    theta0: float = 0.0
    theta1: float = 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassTarget:
    """A named root with exact and raw-variable-renaming-invariant identity."""

    name: str
    expression: GeneratedExpression
    whole_tree_digest: str
    alpha_renamed_topology_signature: str
    depth: int
    parameter_free: bool

    @property
    def digest(self) -> str:
        """Backward-compatible short name for the exact whole-tree digest."""

        return self.whole_tree_digest


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassExpressionManifest:
    """A whole-root expression split; subtrees are never treated as examples."""

    schema: str
    namespace: str
    targets: tuple[GeneratedClassTarget, ...]
    split_unit: str
    commutative_canonicalization: str
    alpha_renaming_scope: str
    manifest_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassResourceContract:
    """Fixed learner capacity and independently checkable state-byte budget."""

    active_slots: int
    candidate_slots: int
    input_dim: int
    allocated_max_depth: int
    n_tasks: int
    generator_resource_contexts: int
    generator_policy_count: int
    jax_state_nbytes: int
    jax_state_nbytes_formula: str
    host_timing_metadata_count: int
    host_timing_metadata_included: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassOperationContract:
    """Shared structural work allocation, without a wall-clock claim."""

    learner_updates_per_step: int
    active_feature_evaluations_per_step: int
    candidate_feature_evaluations_per_step: int
    allocated_curation_decision_slots_per_step: int
    task_heads_evaluated_per_step: int
    phase_or_context_branches_per_step: int
    latency_measurement: str
    wall_clock_threshold: float | None
    flop_or_hlo_equivalence_claimed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassReachabilityContract:
    """Finite-support requirements for the critical D lifecycle identity."""

    target_name: str
    target_whole_tree_digest: str
    target_parameter_free: bool
    target_depth: int
    exact_initial_active_occurrences_required: int
    exact_initial_candidate_occurrences_required: int
    required_top_operation: str
    required_top_operation_probability: float
    required_left_parent_digest: str
    required_right_parent_digest: str
    required_parent_choices_have_nonzero_support: bool
    initialization_structure_key_invariant: bool
    no_coefficient_tolerance: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassCurationOpportunityAudit:
    """Structural cadence audit for acquisition, retirement, and reacquisition."""

    curation_interval: int
    conservative_lifecycle_lower_bound: int
    development_margin_multiplier: int
    required_total_opportunities: int
    opportunities_before_first_d: int
    opportunities_in_first_d: int
    opportunities_between_d_phases: int
    opportunities_in_second_d: int
    total_opportunities: int
    every_critical_window_meets_lower_bound: bool
    total_meets_development_margin: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassLifecyclePrerequisites:
    """Causal prerequisites which remain deliberately unsatisfied in v0."""

    causal_shadow_deletion_complete: bool
    matched_sham_scrub_complete: bool
    d_never_seen_twin_complete: bool
    scrub_completeness_proof_complete: bool
    post_scrub_generation_freeze_complete: bool
    fresh_reacquisition_generation_epoch_complete: bool
    fresh_reacquisition_generation_key_namespace_complete: bool
    candidate_identity_refresh_head_zero_complete: bool
    d_retirement_observed: bool
    d_reacquisition_observed: bool
    required_scrub_state: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassRecurrenceV0Protocol:
    """Immutable development declaration for the blocked v0 vertical slice."""

    schema: str
    status: str
    development_only: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    n_tasks: int
    context_id: int
    boundary_signal_exposed: bool
    task_specific_heads: bool
    resets_allowed: bool
    phase_order: tuple[str, ...]
    phase_lengths: tuple[int, ...]
    phase_length_namespace: str
    phase_length_prng_impl: str
    phase_length_key_data: tuple[int, int]
    phase_length_candidates: tuple[int, ...]
    phase_length_manifest_sha256: str
    evaluator_only_fields: tuple[str, ...]
    learner_observation_fields: tuple[str, ...]
    evaluator_label_permutation_trajectory_invariant: bool
    expression_manifest_sha256: str
    expression_split_unit: str
    alpha_topology_disjointness_required: bool
    input_dim: int
    active_slots: int
    candidate_slots: int
    allocated_max_depth: int
    resource_contract: GeneratedClassResourceContract
    operation_contract: GeneratedClassOperationContract
    reachability_contract: GeneratedClassReachabilityContract
    curation_opportunity_audit: GeneratedClassCurationOpportunityAudit
    lifecycle_prerequisites: GeneratedClassLifecyclePrerequisites


@dataclasses.dataclass(frozen=True, slots=True)
class GeneratedClassControl:
    """Capacity-matched control declaration; not an executable arm."""

    name: str
    intervention: str
    resource_contract: GeneratedClassResourceContract
    operation_contract: GeneratedClassOperationContract
    phase_length_manifest_sha256: str
    allocated_max_depth: int
    effective_max_depth: int
    exhaustive_pair_scaffold_ceiling: bool
    evaluator_boundary_dependent: bool
    development_only: bool
    execution_authorized: bool
    evidence_authorized: bool


def _strict_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an exact Python int")
    result = value
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_positive_int(value: object, *, name: str) -> int:
    result = _strict_nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _float32_coefficient(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact Python float")
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = np.float32(value)
    if not bool(np.isfinite(normalized)):
        raise ValueError(f"{name} must be representable as a finite float32")
    return float(normalized)


def _float32_bits(value: float) -> str:
    scalar = np.asarray(value, dtype=np.float32)
    return scalar.view(np.uint32).item().to_bytes(4, "big").hex()


def _expression_depth(expression: GeneratedExpression) -> int:
    if expression.op == "raw":
        return 0
    if expression.left is None or expression.right is None:
        raise ValueError("binary expression is missing a child")
    return 1 + max(
        _expression_depth(expression.left),
        _expression_depth(expression.right),
    )


def _expression_is_parameter_free(expression: GeneratedExpression) -> bool:
    if expression.op == "raw":
        return True
    if expression.left is None or expression.right is None:
        raise ValueError("binary expression is missing a child")
    return (
        expression.op != "tanh"
        and _expression_is_parameter_free(expression.left)
        and _expression_is_parameter_free(expression.right)
    )


def _expression_payload(expression: GeneratedExpression) -> dict[str, object]:
    if expression.op == "raw":
        return {"op": "raw", "raw_index": expression.raw_index}
    if expression.left is None or expression.right is None:
        raise ValueError("binary expression is missing a child")
    payload: dict[str, object] = {
        "op": expression.op,
        "left": _expression_payload(expression.left),
        "right": _expression_payload(expression.right),
    }
    if expression.op == "tanh":
        payload["theta0_float32_bits"] = _float32_bits(expression.theta0)
        payload["theta1_float32_bits"] = _float32_bits(expression.theta1)
    return payload


def _payload_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_expression_bytes(expression: GeneratedExpression) -> bytes:
    _validate_expression(expression)
    return _payload_bytes(_expression_payload(expression))


def _validate_expression(expression: object) -> GeneratedExpression:
    if type(expression) is not GeneratedExpression:
        raise TypeError("expression must be a canonical GeneratedExpression")
    if type(expression.op) is not str or expression.op not in {
        "raw",
        "sum",
        "product",
        "tanh",
        "gate",
    }:
        raise ValueError("expression op is outside the generated-class grammar")

    if expression.op == "raw":
        _strict_nonnegative_int(expression.raw_index, name="raw_index")
        if expression.left is not None or expression.right is not None:
            raise ValueError("raw expression cannot have children")
        if _float32_bits(expression.theta0) != "00000000" or _float32_bits(
            expression.theta1
        ) != "00000000":
            raise ValueError("raw expression cannot carry coefficients")
    else:
        if expression.raw_index is not None:
            raise ValueError("binary expression cannot carry raw_index")
        left = _validate_expression(expression.left)
        right = _validate_expression(expression.right)
        left_bytes = _canonical_expression_bytes(left)
        right_bytes = _canonical_expression_bytes(right)

        if expression.op in {"sum", "product"}:
            if left_bytes > right_bytes:
                raise ValueError("commutative expression children are not canonical")
            if _float32_bits(expression.theta0) != "00000000" or _float32_bits(
                expression.theta1
            ) != "00000000":
                raise ValueError("sum/product expression cannot carry coefficients")
        elif expression.op == "gate":
            if _float32_bits(expression.theta0) != "00000000" or _float32_bits(
                expression.theta1
            ) != "00000000":
                raise ValueError("gate expression cannot carry coefficients")
        else:
            theta0 = _float32_coefficient(expression.theta0, name="theta0")
            theta1 = _float32_coefficient(expression.theta1, name="theta1")
            pair_left = (left_bytes, _float32_bits(theta0))
            pair_right = (right_bytes, _float32_bits(theta1))
            if pair_left > pair_right:
                raise ValueError("tanh child/coefficient pairs are not canonical")

    if _expression_depth(expression) > _EXPRESSION_MAX_DEPTH:
        raise ValueError(f"expression depth exceeds {_EXPRESSION_MAX_DEPTH}")
    return expression


def raw_expression(raw_index: int) -> GeneratedExpression:
    """Build one exact raw-input leaf; bools and NumPy integers are rejected."""

    index = _strict_nonnegative_int(raw_index, name="raw_index")
    return GeneratedExpression(op="raw", raw_index=index)


def _canonical_commutative_children(
    left: GeneratedExpression,
    right: GeneratedExpression,
) -> tuple[GeneratedExpression, GeneratedExpression]:
    left_bytes = _canonical_expression_bytes(left)
    right_bytes = _canonical_expression_bytes(right)
    return (left, right) if left_bytes <= right_bytes else (right, left)


def sum_expression(
    left: GeneratedExpression,
    right: GeneratedExpression,
) -> GeneratedExpression:
    """Build ``clip(left + right)`` with commutative child canonicalization."""

    first, second = _canonical_commutative_children(left, right)
    expression = GeneratedExpression(op="sum", left=first, right=second)
    return _validate_expression(expression)


def product_expression(
    left: GeneratedExpression,
    right: GeneratedExpression,
) -> GeneratedExpression:
    """Build ``clip(left * right)`` with commutative child canonicalization."""

    first, second = _canonical_commutative_children(left, right)
    expression = GeneratedExpression(op="product", left=first, right=second)
    return _validate_expression(expression)


def tanh_expression(
    left: GeneratedExpression,
    right: GeneratedExpression,
    *,
    theta0: float,
    theta1: float,
) -> GeneratedExpression:
    """Build the bias-free ``tanh(theta0*left + theta1*right)`` primitive.

    The two ``(child, coefficient)`` pairs are canonicalized together.  Thus a
    simultaneous child/coefficient swap is identical, while swapping only the
    children changes the expression whenever the coefficients differ.
    """

    first_theta = _float32_coefficient(theta0, name="theta0")
    second_theta = _float32_coefficient(theta1, name="theta1")
    first_pair = (_canonical_expression_bytes(left), _float32_bits(first_theta))
    second_pair = (_canonical_expression_bytes(right), _float32_bits(second_theta))
    if first_pair > second_pair:
        left, right = right, left
        first_theta, second_theta = second_theta, first_theta
    expression = GeneratedExpression(
        op="tanh",
        left=left,
        right=right,
        theta0=first_theta,
        theta1=second_theta,
    )
    return _validate_expression(expression)


def gate_expression(
    left: GeneratedExpression,
    right: GeneratedExpression,
) -> GeneratedExpression:
    """Build the ordered, parameter-free ``left * sigmoid(right)`` primitive."""

    _validate_expression(left)
    _validate_expression(right)
    expression = GeneratedExpression(op="gate", left=left, right=right)
    return _validate_expression(expression)


def expression_digest(expression: GeneratedExpression) -> str:
    """Return the SHA-256 identity of the complete canonical expression tree."""

    return hashlib.sha256(_canonical_expression_bytes(expression)).hexdigest()


def _raw_indices(expression: GeneratedExpression) -> tuple[int, ...]:
    _validate_expression(expression)
    if expression.op == "raw":
        if expression.raw_index is None:
            raise ValueError("validated raw expression has no index")
        return (expression.raw_index,)
    if expression.left is None or expression.right is None:
        raise ValueError("validated binary expression has no children")
    return _raw_indices(expression.left) + _raw_indices(expression.right)


def _rename_raw_indices(
    expression: GeneratedExpression,
    mapping: dict[int, int],
) -> GeneratedExpression:
    if expression.op == "raw":
        if expression.raw_index is None or expression.raw_index not in mapping:
            raise ValueError("raw-index alpha-renaming map is incomplete")
        return raw_expression(mapping[expression.raw_index])
    if expression.left is None or expression.right is None:
        raise ValueError("validated binary expression has no children")
    left = _rename_raw_indices(expression.left, mapping)
    right = _rename_raw_indices(expression.right, mapping)
    if expression.op == "sum":
        return sum_expression(left, right)
    if expression.op == "product":
        return product_expression(left, right)
    if expression.op == "tanh":
        return tanh_expression(
            left,
            right,
            theta0=expression.theta0,
            theta1=expression.theta1,
        )
    if expression.op == "gate":
        return gate_expression(left, right)
    raise ValueError("expression op is outside the generated-class grammar")


def expression_topology_signature(expression: GeneratedExpression) -> str:
    """Hash the full tree modulo a bijective renaming of raw variables.

    The minimum canonical encoding over every bijection to ``0..n-1`` is used.
    This retains repeated-variable patterns, coefficients, ordering of the gate,
    and all intermediate tree grouping, while preventing a raw-index permutation
    from evading a holdout split.
    """

    _validate_expression(expression)
    indices = tuple(sorted(set(_raw_indices(expression))))
    encodings: list[bytes] = []
    for renamed_indices in itertools.permutations(range(len(indices))):
        mapping = dict(zip(indices, renamed_indices, strict=True))
        renamed = _rename_raw_indices(expression, mapping)
        encodings.append(_canonical_expression_bytes(renamed))
    if not encodings:
        raise ValueError("expression has no raw leaves")
    return hashlib.sha256(min(encodings)).hexdigest()


def evaluate_expression(
    expression: GeneratedExpression,
    observation: Array,
) -> Array:
    """Evaluate the exact grammar with the core's float32 per-node clipping."""

    _validate_expression(expression)
    if not isinstance(observation, Array):
        raise TypeError("observation must be a JAX array")
    if observation.ndim != 1:
        raise ValueError("observation must be rank one")
    if observation.dtype != jnp.float32:
        raise TypeError("observation must have dtype float32")

    def evaluate(node: GeneratedExpression) -> Array:
        if node.op == "raw":
            if node.raw_index is None or node.raw_index >= observation.shape[0]:
                raise ValueError("raw expression index is outside the observation")
            value = observation[node.raw_index]
        else:
            if node.left is None or node.right is None:
                raise ValueError("validated binary expression has no children")
            left_value = evaluate(node.left)
            right_value = evaluate(node.right)
            if node.op == "sum":
                value = left_value + right_value
            elif node.op == "product":
                value = left_value * right_value
            elif node.op == "tanh":
                value = jnp.tanh(
                    jnp.asarray(node.theta0, dtype=jnp.float32) * left_value
                    + jnp.asarray(node.theta1, dtype=jnp.float32) * right_value
                )
            elif node.op == "gate":
                value = left_value * jax.nn.sigmoid(right_value)
            else:
                raise ValueError("expression op is outside the generated-class grammar")
        return jnp.clip(
            value,
            jnp.asarray(-FEATURE_VALUE_CLIP, dtype=jnp.float32),
            jnp.asarray(FEATURE_VALUE_CLIP, dtype=jnp.float32),
        )

    return evaluate(expression)


def _validate_development_manifest_namespace(namespace: object) -> str:
    if type(namespace) is not str:
        raise TypeError("expression namespace must be an exact string")
    if namespace == PROTECTED_EXPRESSION_NAMESPACE or namespace.startswith(
        f"{PROTECTED_EXPRESSION_NAMESPACE}/"
    ):
        raise PermissionError("protected expression namespace is evaluator-inaccessible")
    if namespace != DEVELOPMENT_EXPRESSION_NAMESPACE and not namespace.startswith(
        f"{DEVELOPMENT_EXPRESSION_NAMESPACE}/"
    ):
        raise ValueError("only development expression namespaces can be built")
    return namespace


def _manifest_sha256(
    namespace: str,
    targets: tuple[GeneratedClassTarget, ...],
) -> str:
    payload = {
        "schema": GENERATED_CLASS_EXPRESSION_MANIFEST_SCHEMA,
        "namespace": namespace,
        "split_unit": "whole_expression_root",
        "commutative_canonicalization": "sum_product_and_tanh_pair_v1",
        "alpha_renaming_scope": "whole_tree_bijective_raw_variable_renaming_v1",
        "targets": [
            {
                "name": target.name,
                "whole_tree_digest": target.whole_tree_digest,
                "alpha_renamed_topology_signature": (
                    target.alpha_renamed_topology_signature
                ),
                "depth": target.depth,
                "parameter_free": target.parameter_free,
            }
            for target in targets
        ],
    }
    return hashlib.sha256(_payload_bytes(payload)).hexdigest()


def build_development_expression_manifest(
    namespace: str,
    named_expressions: tuple[tuple[str, GeneratedExpression], ...],
) -> GeneratedClassExpressionManifest:
    """Build a hash-bound development manifest; protected names are forbidden."""

    checked_namespace = _validate_development_manifest_namespace(namespace)
    if type(named_expressions) is not tuple or not named_expressions:
        raise TypeError("named_expressions must be a non-empty exact tuple")
    targets: list[GeneratedClassTarget] = []
    for entry in named_expressions:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("each named expression must be an exact (name, expression) tuple")
        name, expression = entry
        if type(name) is not str or not name:
            raise TypeError("target name must be a non-empty exact string")
        checked_expression = _validate_expression(expression)
        targets.append(
            GeneratedClassTarget(
                name=name,
                expression=checked_expression,
                whole_tree_digest=expression_digest(checked_expression),
                alpha_renamed_topology_signature=expression_topology_signature(
                    checked_expression
                ),
                depth=_expression_depth(checked_expression),
                parameter_free=_expression_is_parameter_free(checked_expression),
            )
        )
    frozen_targets = tuple(targets)
    if len({target.name for target in frozen_targets}) != len(frozen_targets):
        raise ValueError("target names must be unique within a manifest")
    if len({target.whole_tree_digest for target in frozen_targets}) != len(
        frozen_targets
    ):
        raise ValueError("exact whole-tree expressions must be unique within a manifest")
    if len(
        {target.alpha_renamed_topology_signature for target in frozen_targets}
    ) != len(frozen_targets):
        raise ValueError("alpha-renamed topologies must be unique within a manifest")
    return GeneratedClassExpressionManifest(
        schema=GENERATED_CLASS_EXPRESSION_MANIFEST_SCHEMA,
        namespace=checked_namespace,
        targets=frozen_targets,
        split_unit="whole_expression_root",
        commutative_canonicalization="sum_product_and_tanh_pair_v1",
        alpha_renaming_scope="whole_tree_bijective_raw_variable_renaming_v1",
        manifest_sha256=_manifest_sha256(checked_namespace, frozen_targets),
    )


def _development_targets() -> tuple[tuple[str, GeneratedExpression], ...]:
    x0, x1, x2 = (raw_expression(index) for index in range(3))
    # Every registered target is in the learner's discrete operation/parent
    # support.  Parameterized tanh remains part of the grammar, but is excluded
    # from this exact-identity manifest because continuous theta sampling would
    # make exact target birth a measure-zero event.
    target_a = sum_expression(
        product_expression(x0, x1),
        x2,
    )
    target_b = product_expression(
        product_expression(x0, x1),
        x2,
    )
    target_c = gate_expression(
        product_expression(x0, x1),
        x2,
    )
    target_d = gate_expression(
        product_expression(x0, x0),
        x1,
    )
    return (("A", target_a), ("B", target_b), ("C", target_c), ("D", target_d))


def derive_expression_manifest(namespace: str) -> GeneratedClassExpressionManifest:
    """Derive the fixed development roots; protected roots are unavailable."""

    if type(namespace) is not str:
        raise TypeError("expression namespace must be an exact string")
    if namespace == PROTECTED_EXPRESSION_NAMESPACE:
        raise PermissionError("protected expression manifest is not development-accessible")
    if namespace != DEVELOPMENT_EXPRESSION_NAMESPACE:
        raise ValueError("unknown generated-class expression namespace")
    return build_development_expression_manifest(namespace, _development_targets())


def _validate_manifest(manifest: object) -> GeneratedClassExpressionManifest:
    if type(manifest) is not GeneratedClassExpressionManifest:
        raise TypeError("manifest must be a GeneratedClassExpressionManifest")
    _validate_development_manifest_namespace(manifest.namespace)
    if manifest.schema != GENERATED_CLASS_EXPRESSION_MANIFEST_SCHEMA:
        raise ValueError("expression manifest schema mismatch")
    if manifest.split_unit != "whole_expression_root":
        raise ValueError("expression manifest must split whole roots")
    if manifest.commutative_canonicalization != "sum_product_and_tanh_pair_v1":
        raise ValueError("expression manifest canonicalization mismatch")
    if manifest.alpha_renaming_scope != (
        "whole_tree_bijective_raw_variable_renaming_v1"
    ):
        raise ValueError("expression manifest alpha-renaming scope mismatch")
    for target in manifest.targets:
        if target.whole_tree_digest != expression_digest(target.expression):
            raise ValueError("expression target exact whole-tree digest mismatch")
        if target.alpha_renamed_topology_signature != expression_topology_signature(
            target.expression
        ):
            raise ValueError("expression target alpha-renamed topology mismatch")
        if target.depth != _expression_depth(target.expression):
            raise ValueError("expression target depth mismatch")
        if target.parameter_free != _expression_is_parameter_free(target.expression):
            raise ValueError("expression target parameter-free declaration mismatch")
    if manifest.manifest_sha256 != _manifest_sha256(manifest.namespace, manifest.targets):
        raise ValueError("expression manifest digest mismatch")
    return manifest


def assert_whole_expression_manifests_disjoint(
    first: GeneratedClassExpressionManifest,
    second: GeneratedClassExpressionManifest,
) -> None:
    """Reject exact-root or alpha-renamed-topology overlap between splits."""

    checked_first = _validate_manifest(first)
    checked_second = _validate_manifest(second)
    exact_overlap = {
        target.whole_tree_digest for target in checked_first.targets
    } & {target.whole_tree_digest for target in checked_second.targets}
    if exact_overlap:
        raise ValueError("whole-expression manifests have exact whole-tree overlap")
    topology_overlap = {
        target.alpha_renamed_topology_signature for target in checked_first.targets
    } & {
        target.alpha_renamed_topology_signature for target in checked_second.targets
    }
    if topology_overlap:
        raise ValueError("whole-expression manifests have alpha-renamed topology overlap")


def compositional_jax_state_nbytes_formula(
    active_slots: int,
    candidate_slots: int,
) -> int:
    """Exact state-array bytes for H=G=1, K=4 and float32/int32 arrays.

    The two host timing floats in ``CompositionalFeatureState`` are excluded.
    """

    active = _strict_positive_int(active_slots, name="active_slots")
    candidates = _strict_nonnegative_int(candidate_slots, name="candidate_slots")
    return 68 * active + 4 * active * candidates + 80 * candidates + 92


def measure_compositional_jax_state_nbytes(
    state: CompositionalFeatureState,
) -> int:
    """Sum concrete JAX leaf bytes independently of the shape formula."""

    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    return sum(
        int(leaf.nbytes)
        for leaf in jax.tree_util.tree_leaves(state)
        if isinstance(leaf, Array)
    )


def _core_binary_expression(
    op: int,
    left: GeneratedExpression,
    right: GeneratedExpression,
    theta: np.ndarray,
) -> GeneratedExpression:
    if op == OP_PRODUCT:
        return product_expression(left, right)
    if op == OP_SUM:
        return sum_expression(left, right)
    if op == OP_TANH:
        return tanh_expression(
            left,
            right,
            theta0=float(theta[0]),
            theta1=float(theta[1]),
        )
    if op == OP_GATED:
        return gate_expression(left, right)
    raise ValueError("compositional state contains an unsupported operation")


def _state_expression_banks(
    state: CompositionalFeatureState,
) -> tuple[tuple[GeneratedExpression, ...], tuple[GeneratedExpression, ...]]:
    if type(state) is not CompositionalFeatureState:
        raise TypeError("state must be an exact CompositionalFeatureState")
    ops = np.asarray(state.ops)
    parent_a = np.asarray(state.parent_a)
    parent_b = np.asarray(state.parent_b)
    theta = np.asarray(state.theta)
    active: list[GeneratedExpression] = []
    for slot in range(ops.shape[0]):
        op = int(ops[slot])
        left_index = int(parent_a[slot])
        right_index = int(parent_b[slot])
        if op == OP_RAW:
            expression = raw_expression(left_index)
        else:
            if not (0 <= left_index < slot and 0 <= right_index < slot):
                raise ValueError("active compositional state is not topologically ordered")
            expression = _core_binary_expression(
                op,
                active[left_index],
                active[right_index],
                theta[slot],
            )
        active.append(expression)

    candidate_ops = np.asarray(state.candidate_ops)
    candidate_parent_a = np.asarray(state.candidate_parent_a)
    candidate_parent_b = np.asarray(state.candidate_parent_b)
    candidate_theta = np.asarray(state.candidate_theta)
    candidates: list[GeneratedExpression] = []
    for slot in range(candidate_ops.shape[0]):
        op = int(candidate_ops[slot])
        left_index = int(candidate_parent_a[slot])
        right_index = int(candidate_parent_b[slot])
        if op == OP_RAW:
            expression = raw_expression(left_index)
        else:
            if not (
                0 <= left_index < len(active) and 0 <= right_index < len(active)
            ):
                raise ValueError("candidate compositional state has an invalid parent")
            expression = _core_binary_expression(
                op,
                active[left_index],
                active[right_index],
                candidate_theta[slot],
            )
        candidates.append(expression)
    return tuple(active), tuple(candidates)


def count_expression_occurrences(
    state: CompositionalFeatureState,
    expression: GeneratedExpression,
) -> tuple[int, int]:
    """Count an exact whole-tree identity in active and candidate banks."""

    wanted_digest = expression_digest(expression)
    active, candidates = _state_expression_banks(state)
    return (
        sum(expression_digest(item) == wanted_digest for item in active),
        sum(expression_digest(item) == wanted_digest for item in candidates),
    )


def _derive_phase_lengths() -> tuple[int, ...]:
    key_data = jnp.asarray(_PHASE_LENGTH_KEY_DATA, dtype=jnp.uint32)
    key = jr.wrap_key_data(key_data, impl="threefry2x32")
    permutation = jr.permutation(
        key,
        jnp.arange(len(_PHASE_LENGTH_CANDIDATES), dtype=jnp.int32),
    )
    return tuple(_PHASE_LENGTH_CANDIDATES[int(index)] for index in permutation)


def _phase_length_manifest_sha256(phase_lengths: tuple[int, ...]) -> str:
    payload = {
        "namespace": _PHASE_LENGTH_NAMESPACE,
        "prng_impl": "threefry2x32",
        "key_data_uint32": list(_PHASE_LENGTH_KEY_DATA),
        "candidate_lengths": list(_PHASE_LENGTH_CANDIDATES),
        "phase_order": list(_PHASE_ORDER),
        "phase_lengths": list(phase_lengths),
        "shared_exactly_across_all_controls": True,
        "evaluator_only": True,
    }
    return hashlib.sha256(_payload_bytes(payload)).hexdigest()


def _curation_opportunities(start: int, stop: int, cadence: int) -> int:
    """Count 1-based cadence events in a zero-based half-open step interval."""

    if not 0 <= start <= stop:
        raise ValueError("curation interval bounds are invalid")
    return stop // cadence - start // cadence


def _build_curation_opportunity_audit(
    phase_lengths: tuple[int, ...],
) -> GeneratedClassCurationOpportunityAudit:
    starts: list[int] = []
    cursor = 0
    for length in phase_lengths:
        starts.append(cursor)
        cursor += length
    d_indices = tuple(index for index, name in enumerate(_PHASE_ORDER) if name == "D")
    if d_indices != (3, 7):
        raise RuntimeError("generated-class recurrence schedule must contain exactly two D phases")
    first_index, second_index = d_indices
    first_start = starts[first_index]
    first_stop = first_start + phase_lengths[first_index]
    second_start = starts[second_index]
    second_stop = second_start + phase_lengths[second_index]
    lower_bound = _CONSERVATIVE_LIFECYCLE_CURATION_LOWER_BOUND
    required_total = lower_bound * _DEVELOPMENT_CURATION_MARGIN_MULTIPLIER
    before_first = _curation_opportunities(0, first_start, _CURATION_INTERVAL)
    in_first = _curation_opportunities(first_start, first_stop, _CURATION_INTERVAL)
    between = _curation_opportunities(first_stop, second_start, _CURATION_INTERVAL)
    in_second = _curation_opportunities(second_start, second_stop, _CURATION_INTERVAL)
    total = _curation_opportunities(0, cursor, _CURATION_INTERVAL)
    critical_windows = (before_first, in_first, between, in_second)
    return GeneratedClassCurationOpportunityAudit(
        curation_interval=_CURATION_INTERVAL,
        conservative_lifecycle_lower_bound=lower_bound,
        development_margin_multiplier=_DEVELOPMENT_CURATION_MARGIN_MULTIPLIER,
        required_total_opportunities=required_total,
        opportunities_before_first_d=before_first,
        opportunities_in_first_d=in_first,
        opportunities_between_d_phases=between,
        opportunities_in_second_d=in_second,
        total_opportunities=total,
        every_critical_window_meets_lower_bound=all(
            opportunities >= lower_bound for opportunities in critical_windows
        ),
        total_meets_development_margin=total >= required_total,
    )


def build_generated_class_recurrence_v0_protocol() -> GeneratedClassRecurrenceV0Protocol:
    """Build the canonical, non-executable v0 development declaration."""

    manifest = derive_expression_manifest(DEVELOPMENT_EXPRESSION_NAMESPACE)
    phase_lengths = _derive_phase_lengths()
    if len(phase_lengths) != len(_PHASE_ORDER) or len(set(phase_lengths)) != len(
        phase_lengths
    ):
        raise RuntimeError("evaluator phase-length manifest is not nonperiodic")
    curation_audit = _build_curation_opportunity_audit(phase_lengths)
    if not curation_audit.every_critical_window_meets_lower_bound:
        raise RuntimeError("a critical D lifecycle window lacks enough curation opportunities")
    if not curation_audit.total_meets_development_margin:
        raise RuntimeError("v0 schedule lacks its declared curation-opportunity margin")
    d_target = next(target for target in manifest.targets if target.name == "D")
    if d_target.expression.left is None or d_target.expression.right is None:
        raise RuntimeError("critical D target must be a binary expression")
    if not all(_expression_is_parameter_free(target.expression) for target in manifest.targets):
        raise RuntimeError("exact generated-class targets must all be parameter-free")
    reachability = GeneratedClassReachabilityContract(
        target_name="D",
        target_whole_tree_digest=d_target.whole_tree_digest,
        target_parameter_free=True,
        target_depth=d_target.depth,
        exact_initial_active_occurrences_required=0,
        exact_initial_candidate_occurrences_required=0,
        required_top_operation="gate",
        required_top_operation_probability=0.25,
        required_left_parent_digest=expression_digest(d_target.expression.left),
        required_right_parent_digest=expression_digest(d_target.expression.right),
        required_parent_choices_have_nonzero_support=True,
        initialization_structure_key_invariant=True,
        no_coefficient_tolerance=True,
    )
    state_nbytes = compositional_jax_state_nbytes_formula(
        _ACTIVE_SLOTS,
        _CANDIDATE_SLOTS,
    )
    resources = GeneratedClassResourceContract(
        active_slots=_ACTIVE_SLOTS,
        candidate_slots=_CANDIDATE_SLOTS,
        input_dim=_INPUT_DIM,
        allocated_max_depth=_EXPRESSION_MAX_DEPTH,
        n_tasks=_N_TASKS,
        generator_resource_contexts=_GENERATOR_RESOURCE_CONTEXTS,
        generator_policy_count=_GENERATOR_POLICY_COUNT,
        jax_state_nbytes=state_nbytes,
        jax_state_nbytes_formula="68*N + 4*N*C + 80*C + 92; H=G=1,K=4",
        host_timing_metadata_count=2,
        host_timing_metadata_included=False,
    )
    operations = GeneratedClassOperationContract(
        learner_updates_per_step=1,
        active_feature_evaluations_per_step=_ACTIVE_SLOTS,
        candidate_feature_evaluations_per_step=_CANDIDATE_SLOTS,
        allocated_curation_decision_slots_per_step=1,
        task_heads_evaluated_per_step=1,
        phase_or_context_branches_per_step=0,
        latency_measurement="structural_only_no_wall_clock_acceptance",
        wall_clock_threshold=None,
        flop_or_hlo_equivalence_claimed=False,
    )
    prerequisites = GeneratedClassLifecyclePrerequisites(
        causal_shadow_deletion_complete=False,
        matched_sham_scrub_complete=False,
        d_never_seen_twin_complete=False,
        scrub_completeness_proof_complete=False,
        post_scrub_generation_freeze_complete=False,
        fresh_reacquisition_generation_epoch_complete=False,
        fresh_reacquisition_generation_key_namespace_complete=False,
        candidate_identity_refresh_head_zero_complete=False,
        d_retirement_observed=False,
        d_reacquisition_observed=False,
        required_scrub_state=(
            "active_occurrences_and_descendants",
            "candidate_occurrences_and_descendants",
            "output_and_candidate_heads",
            "eligibility_and_future_utility_traces",
            "fast_and_slow_utilities",
            "candidate_active_correlations",
            "descriptors_and_selector_state",
            "feature_and_candidate_generator_policy_provenance",
            "all_hidden_exact_expression_archives",
        ),
    )
    return GeneratedClassRecurrenceV0Protocol(
        schema=GENERATED_CLASS_RECURRENCE_V0_SCHEMA,
        status=GENERATED_CLASS_STATUS,
        development_only=True,
        execution_authorized=False,
        evidence_authorized=False,
        scientific_promotion_allowed=False,
        n_tasks=_N_TASKS,
        context_id=_CONTEXT_ID,
        boundary_signal_exposed=False,
        task_specific_heads=False,
        resets_allowed=False,
        phase_order=_PHASE_ORDER,
        phase_lengths=phase_lengths,
        phase_length_namespace=_PHASE_LENGTH_NAMESPACE,
        phase_length_prng_impl="threefry2x32",
        phase_length_key_data=_PHASE_LENGTH_KEY_DATA,
        phase_length_candidates=_PHASE_LENGTH_CANDIDATES,
        phase_length_manifest_sha256=_phase_length_manifest_sha256(phase_lengths),
        evaluator_only_fields=("phase_label", "phase_boundary"),
        learner_observation_fields=("raw_features",),
        evaluator_label_permutation_trajectory_invariant=True,
        expression_manifest_sha256=manifest.manifest_sha256,
        expression_split_unit="whole_expression_root",
        alpha_topology_disjointness_required=True,
        input_dim=_INPUT_DIM,
        active_slots=_ACTIVE_SLOTS,
        candidate_slots=_CANDIDATE_SLOTS,
        allocated_max_depth=_EXPRESSION_MAX_DEPTH,
        resource_contract=resources,
        operation_contract=operations,
        reachability_contract=reachability,
        curation_opportunity_audit=curation_audit,
        lifecycle_prerequisites=prerequisites,
    )


def _validate_protocol(
    protocol: object,
) -> GeneratedClassRecurrenceV0Protocol:
    if type(protocol) is not GeneratedClassRecurrenceV0Protocol:
        raise TypeError("protocol must be an exact GeneratedClassRecurrenceV0Protocol")
    if protocol != build_generated_class_recurrence_v0_protocol():
        raise ValueError("generated-class recurrence v0 protocol is not canonical")
    return protocol


def build_generated_class_v0_controls(
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> tuple[GeneratedClassControl, ...]:
    """Declare five resource-matched arms without authorizing their execution."""

    checked = _validate_protocol(protocol)

    def control(
        name: str,
        intervention: str,
        *,
        effective_max_depth: int = _EXPRESSION_MAX_DEPTH,
        exhaustive_pair_scaffold_ceiling: bool = False,
    ) -> GeneratedClassControl:
        return GeneratedClassControl(
            name=name,
            intervention=intervention,
            resource_contract=checked.resource_contract,
            operation_contract=checked.operation_contract,
            phase_length_manifest_sha256=checked.phase_length_manifest_sha256,
            allocated_max_depth=checked.allocated_max_depth,
            effective_max_depth=effective_max_depth,
            exhaustive_pair_scaffold_ceiling=exhaustive_pair_scaffold_ceiling,
            evaluator_boundary_dependent=False,
            development_only=True,
            execution_authorized=False,
            evidence_authorized=False,
        )

    return (
        control(FULL_LIFECYCLE, "utility_lifecycle_with_candidate_head_carry"),
        control(RANDOM_CURATION, "replace_utility_choice_with_matched_random_choice"),
        control(
            FROZEN_LIFECYCLE,
            "form_identical_curation_proposal_then_mask_structural_write",
        ),
        control(
            ZERO_CANDIDATE_HEAD_CARRY,
            "zero_candidate_head_on_identity_refresh_and_promotion_refresh",
        ),
        control(
            FINITE_DEGREE_TWO_ARCHIVE_CEILING,
            "exhaustive_raw_pair_scaffold_without_recursive_composition",
            effective_max_depth=1,
            exhaustive_pair_scaffold_ceiling=True,
        ),
    )


def build_generated_class_v0_learner(
    control_name: str,
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> CompositionalFeatureLearner:
    """Construct an arm's fixed-shape learner, but do not execute the protocol.

    Random curation, candidate-identity refresh head zeroing, causal scrub, and
    twins require an external lifecycle runner which v0 intentionally does not
    provide.  This builder exists only to make state shape and direct learner
    primitives inspectable.
    """

    checked = _validate_protocol(protocol)
    if type(control_name) is not str:
        raise TypeError("control_name must be an exact string")
    controls = {control.name: control for control in build_generated_class_v0_controls(checked)}
    if control_name not in controls:
        raise ValueError("unknown generated-class recurrence v0 control")
    selected = controls[control_name]
    finite_ceiling = selected.name == FINITE_DEGREE_TWO_ARCHIVE_CEILING
    operation_prior = (
        (0.0, 1.0, 0.0, 0.0, 0.0)
        if finite_ceiling
        else (0.0, 0.25, 0.25, 0.25, 0.25)
    )
    return CompositionalFeatureLearner(
        n_features=checked.active_slots,
        n_tasks=1,
        candidate_count=checked.candidate_slots,
        step_size_output=0.01,
        step_size_theta=0.001,
        utility_decay=0.995,
        replacement_interval=_CURATION_INTERVAL,
        min_feature_age=16,
        candidate_min_age=16,
        promotion_margin=1.0,
        promotion_blend=1.0,
        max_depth=selected.effective_max_depth,
        use_obgd=True,
        generation_strategy=GENERATION_ROBUST_RECURSIVE,
        parent_novelty_weight=0.1,
        parent_depth_prior=0.1 if selected.effective_max_depth > 1 else 0.0,
        retention_depth_bonus=0.05 if selected.effective_max_depth > 1 else 0.0,
        candidate_scoring_mode="energy_novelty",
        candidate_score_trace_decay=0.95,
        candidate_novelty_weight=0.25,
        retention_slow_utility_decay=0.999,
        operation_prior=operation_prior,
        generator_resource_contexts=1,
    )


def require_generated_class_v0_executable(
    protocol: GeneratedClassRecurrenceV0Protocol,
) -> None:
    """Always fail closed until every causal lifecycle prerequisite is landed."""

    checked = _validate_protocol(protocol)
    prerequisites = checked.lifecycle_prerequisites
    missing: list[str] = []
    if not prerequisites.causal_shadow_deletion_complete:
        missing.append("causal shadow deletion and complete descendant-state scrub")
    if not prerequisites.matched_sham_scrub_complete:
        missing.append("matched sham scrub")
    if not prerequisites.d_never_seen_twin_complete:
        missing.append("D-never-seen twin")
    if not prerequisites.scrub_completeness_proof_complete:
        missing.append("scrub-completeness proof")
    if not prerequisites.post_scrub_generation_freeze_complete:
        missing.append("immediate post-scrub generation-frozen occlusion window")
    if not prerequisites.fresh_reacquisition_generation_epoch_complete:
        missing.append("fresh named reacquisition generation epoch")
    if not prerequisites.fresh_reacquisition_generation_key_namespace_complete:
        missing.append("fresh reacquisition generation-key namespace")
    if not prerequisites.candidate_identity_refresh_head_zero_complete:
        missing.append("candidate-identity refresh head-zero control")
    if not prerequisites.d_retirement_observed:
        missing.append("observed D retirement")
    if not prerequisites.d_reacquisition_observed:
        missing.append("observed D reacquisition")
    if not checked.execution_authorized:
        missing.append("explicit execution authority")
    raise GeneratedClassProtocolNotReadyError(
        "generated-class recurrence v0 is not executable; missing: "
        + "; ".join(missing)
    )


def _validate_finite_vector(values: object, *, name: str) -> Array:
    if not isinstance(values, Array):
        raise TypeError(f"{name} must be a JAX array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be rank one")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"{name} must have a floating dtype")
    if not bool(jnp.all(jnp.isfinite(values))):
        raise ValueError(f"{name} must contain only finite values")
    return values


def prequential_squared_loss(predictions: Array, targets: Array) -> Array:
    """Return one raw squared loss per prediction made before its update."""

    checked_predictions = _validate_finite_vector(predictions, name="predictions")
    checked_targets = _validate_finite_vector(targets, name="targets")
    if checked_predictions.shape != checked_targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    return jnp.square(checked_targets - checked_predictions)


def adaptation_window_mse(
    losses: Array,
    *,
    phase_starts: tuple[int, ...],
    phase_lengths: tuple[int, ...],
    window: int,
) -> tuple[float, ...]:
    """Return raw mean prequential loss in each phase's leading window."""

    checked_losses = _validate_finite_vector(losses, name="losses")
    if type(phase_starts) is not tuple or type(phase_lengths) is not tuple:
        raise TypeError("phase starts and lengths must be exact tuples")
    if len(phase_starts) != len(phase_lengths):
        raise ValueError("phase starts and lengths must have equal cardinality")
    checked_window = _strict_positive_int(window, name="window")
    means: list[float] = []
    for index, (start, length) in enumerate(zip(phase_starts, phase_lengths, strict=True)):
        checked_start = _strict_nonnegative_int(start, name=f"phase_starts[{index}]")
        checked_length = _strict_positive_int(length, name=f"phase_lengths[{index}]")
        if checked_window > checked_length:
            raise ValueError("adaptation window cannot exceed its phase length")
        stop = checked_start + checked_window
        if stop > checked_losses.shape[0]:
            raise ValueError("adaptation window extends beyond the loss vector")
        means.append(float(jnp.mean(checked_losses[checked_start:stop])))
    return tuple(means)


def recurrence_savings(
    losses: Array,
    *,
    first_start: int,
    recurrence_start: int,
    window: int,
) -> float:
    """Return absolute MSE saved: first-exposure window minus recurrence window.

    This is a raw signed difference, not a ratio, threshold, pass/fail gate, or
    claim of causal memory.  Positive values mean lower recurrence-window loss.
    """

    checked_losses = _validate_finite_vector(losses, name="losses")
    checked_window = _strict_positive_int(window, name="window")
    checked_first = _strict_nonnegative_int(first_start, name="first_start")
    checked_recurrence = _strict_nonnegative_int(
        recurrence_start,
        name="recurrence_start",
    )
    if checked_first + checked_window > checked_losses.shape[0]:
        raise ValueError("first-exposure window extends beyond the loss vector")
    if checked_recurrence + checked_window > checked_losses.shape[0]:
        raise ValueError("recurrence window extends beyond the loss vector")
    first_mse = float(
        jnp.mean(checked_losses[checked_first : checked_first + checked_window])
    )
    recurrence_mse = float(
        jnp.mean(
            checked_losses[
                checked_recurrence : checked_recurrence + checked_window
            ]
        )
    )
    result = first_mse - recurrence_mse
    if not math.isfinite(result):
        raise ValueError("recurrence savings must be finite")
    return result


__all__ = [
    "DEVELOPMENT_EXPRESSION_NAMESPACE",
    "FINITE_DEGREE_TWO_ARCHIVE_CEILING",
    "FROZEN_LIFECYCLE",
    "FULL_LIFECYCLE",
    "PROTECTED_EXPRESSION_NAMESPACE",
    "RANDOM_CURATION",
    "ZERO_CANDIDATE_HEAD_CARRY",
    "GeneratedClassProtocolNotReadyError",
    "adaptation_window_mse",
    "assert_whole_expression_manifests_disjoint",
    "build_development_expression_manifest",
    "build_generated_class_recurrence_v0_protocol",
    "build_generated_class_v0_controls",
    "build_generated_class_v0_learner",
    "compositional_jax_state_nbytes_formula",
    "count_expression_occurrences",
    "derive_expression_manifest",
    "evaluate_expression",
    "expression_digest",
    "expression_topology_signature",
    "gate_expression",
    "measure_compositional_jax_state_nbytes",
    "prequential_squared_loss",
    "product_expression",
    "raw_expression",
    "recurrence_savings",
    "require_generated_class_v0_executable",
    "sum_expression",
    "tanh_expression",
]
