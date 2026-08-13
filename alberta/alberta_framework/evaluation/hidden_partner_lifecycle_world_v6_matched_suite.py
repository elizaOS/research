"""Scan-free structural matching for a complete v6 development suite.

This module composes already-executed in-memory v6 runs.  Its own matching
logic never constructs keys, initializes a bridge, or executes a transition.
The composed per-run structural validator does reconstruct keys and initialize
an expected bridge state, but it still does not replay transitions.  Neither
layer grants evidence or promotion authority.  The additional checks here
establish only that the complete ordered control panel is structurally matched.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from alberta_framework.core.average_reward import DifferentialSARSAState
from alberta_framework.core.behavior_model import BehaviorModelState
from alberta_framework.core.feature_bank_router import FeatureBankRouterState
from alberta_framework.core.grounded_joint_world_model import GroundedJointWorldModelState
from alberta_framework.core.integrated_hidden_partner import (
    IntegratedGroundedPlannerEvaluation,
    IntegratedHiddenPartnerState,
    IntegratedPlannerEvaluation,
    IntegratedPlannerSelection,
)
from alberta_framework.core.interaction_features import InteractionFeatureState
from alberta_framework.core.joint_partner_world import BoundedJointOutcomeState
from alberta_framework.core.state_builder import OnlineGatedStateBuilderState
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6 import (
    PRIMARY_CONDITION_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_controls import (
    V6_DIAGNOSTIC_ORDER,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_runner import (
    V6_INITIAL_STREAM_BIT_ORDER,
    V6_TRANSITION_STREAM_BIT_ORDER,
    V6DevelopmentRun,
    V6RngRecord,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_scan_plan import (
    MAX_SCAN_STEPS,
    canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes,
    require_v6_control_suite_ready,
    validate_v6_control_suite_readiness,
)
from alberta_framework.evaluation.hidden_partner_lifecycle_world_v6_validator import (
    STRUCTURALLY_VALID_DEVELOPMENT_RUN,
    V6DevelopmentRunValidation,
    validate_hidden_partner_lifecycle_world_v6_development_run,
)
from alberta_framework.evaluation.hidden_partner_world_filter import HiddenPartnerWorldFilterState
from alberta_framework.evaluation.hidden_partner_world_online_bridge import (
    HiddenPartnerWorldOnlineState,
)
from alberta_framework.streams.hidden_partner_world_feedback import (
    HiddenPartnerWorldFeedbackState,
)

HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA = (
    "alberta.hidden-partner-lifecycle-world.matched-suite-development.v1"
)
MATCHED_DEVELOPMENT_SUITE_RECORD = "MATCHED_DEVELOPMENT_SUITE_RECORD"
STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE: Literal[
    "STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE"
] = "STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE"
STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE: Literal[
    "STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE"
] = "STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE"

DEVELOPMENT_ONLY = True
STRUCTURAL_ONLY = True
REPLAY_VERIFIED = False
EXECUTION_AUTHORIZED = False
EVIDENCE_AUTHORIZED = False
SCIENTIFIC_PROMOTION_ALLOWED = False

V6_MATCHED_SUITE_MEMBER_ORDER: tuple[tuple[str, bool], ...] = (
    *((name, True) for name in PRIMARY_CONDITION_ORDER),
    *((name, False) for name in V6_DIAGNOSTIC_ORDER),
)

_TRANSITION_CUE_BIT_NAMES = frozenset(("cue_0_flipped", "cue_1_flipped"))
_INITIAL_CUE_BIT_NAMES = frozenset(("cue_0_positive", "cue_1_positive"))


def _noncue_mask(bit_order: tuple[str, ...], cue_names: frozenset[str]) -> np.uint8:
    if len(bit_order) != 8 or len(set(bit_order)) != 8:
        raise RuntimeError("v6 matched-suite bit order must contain eight unique names")
    if frozenset(bit_order) & cue_names != cue_names:
        raise RuntimeError("v6 matched-suite bit order is missing a reviewed cue bit")
    return np.uint8(
        sum(1 << index for index, name in enumerate(bit_order) if name not in cue_names)
    )


_STREAM_NONCUE_MASK = _noncue_mask(
    V6_TRANSITION_STREAM_BIT_ORDER,
    _TRANSITION_CUE_BIT_NAMES,
)
_INITIAL_NONCUE_MASK = _noncue_mask(
    V6_INITIAL_STREAM_BIT_ORDER,
    _INITIAL_CUE_BIT_NAMES,
)

V6_MATCHED_RNG_FIELD_PARTITION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("suite_key", ("supplied_key_data",)),
    (
        "shared_endpoints_and_counts",
        (
            "initial_world_key_data",
            "final_world_key_data",
            "initial_policy_key_data",
            "final_policy_key_data",
            "initial_interaction_key_data",
            "final_interaction_key_data",
            "world_draw_counts",
            "interaction_key_advance_count",
            "policy_decision_count",
        ),
    ),
    ("initial_stream", ("initial_stream_bits",)),
)
V6_MATCHED_RNG_SHARED_FIELD_ORDER = V6_MATCHED_RNG_FIELD_PARTITION[1][1]
_partitioned_rng_fields = tuple(
    field for _, fields in V6_MATCHED_RNG_FIELD_PARTITION for field in fields
)
_runtime_rng_fields = tuple(
    field.name
    for field in dataclasses.fields(V6RngRecord)  # type: ignore[arg-type]
)
if (
    len(_partitioned_rng_fields) != len(set(_partitioned_rng_fields))
    or len(_partitioned_rng_fields) != len(_runtime_rng_fields)
    or set(_partitioned_rng_fields) != set(_runtime_rng_fields)
):
    raise RuntimeError("every V6RngRecord field must appear in exactly one reviewed partition")

# A positive unmocked composition fixture needs 18 already-executed, strictly
# valid runs.  Constructing it without either full scans or duplicating the
# validator's very large synthetic fixture is not currently available.
V6_MATCHED_REAL_VALIDATOR_POSITIVE_INTEGRATION_BLOCKER = (
    "positive unmocked composition requires 18 pre-executed structurally valid runs"
)

_REVIEWED_INITIAL_DATACLASS_FIELDS: tuple[tuple[type[object], tuple[str, ...]], ...] = (
    (
        HiddenPartnerWorldOnlineState,
        ("world", "agent", "world_filter", "config_token", "action", "valid", "step_count"),
    ),
    (
        HiddenPartnerWorldFeedbackState,
        (
            "signal_key",
            "partner_key",
            "world_key",
            "cue_key",
            "outcome_key",
            "segment_lengths",
            "segment_ends",
            "current_signals",
            "current_cues",
            "world_sign",
            "previous_outcome",
            "previous_partner_action",
            "has_partner_history",
            "step_count",
            "step_words",
        ),
    ),
    (
        IntegratedHiddenPartnerState,
        (
            "state_builder",
            "interaction",
            "behavior",
            "joint_world",
            "grounded_world",
            "control",
            "router",
            "raw_observation",
            "phi",
            "chi",
            "consumer_active_mask",
            "consumer_evidence_streak",
            "consumer_read_idle_steps",
            "current_evaluation",
            "current_q_value_delta",
            "current_selection",
            "step_count",
            "step_words",
        ),
    ),
    (
        OnlineGatedStateBuilderState,
        (
            "parameters",
            "hidden",
            "parameter_sensitivity",
            "step_count",
            "step_words",
            "update_count",
            "update_words",
            "last_gradient_norm",
        ),
    ),
    (
        InteractionFeatureState,
        (
            "key",
            "feature_left",
            "feature_right",
            "output_weights",
            "relevance_probe_weights",
            "relevance_probe_biases",
            "output_biases",
            "utilities",
            "evidence_idle_steps",
            "utility_evidence_streak",
            "active_output_memory_committed",
            "task_activity_ema",
            "ages",
            "candidate_left",
            "candidate_right",
            "candidate_output_weights",
            "candidate_utilities",
            "candidate_ages",
            "candidate_promotion_evidence_streak",
            "candidate_reacquisition_required",
            "feature_second_moments",
            "candidate_second_moments",
            "target_second_moments",
            "feature_parent_a",
            "feature_parent_b",
            "feature_generator",
            "candidate_parent_a",
            "candidate_parent_b",
            "candidate_generator",
            "step_count",
            "step_words",
            "replacement_phase",
            "birth_timestamp",
            "uptime_s",
        ),
    ),
    (
        BehaviorModelState,
        (
            "weights",
            "bias",
            "rng_key",
            "step_count",
            "step_words",
            "nll_ema",
            "accuracy_ema",
            "confidence_ema",
        ),
    ),
    (
        BoundedJointOutcomeState,
        (
            "reward_predictions",
            "outcome_predictions",
            "visit_counts",
            "step_count",
            "step_words",
        ),
    ),
    (
        GroundedJointWorldModelState,
        ("weights", "bias", "update_count", "update_words"),
    ),
    (
        DifferentialSARSAState,
        (
            "q_weights",
            "q_bias",
            "q_trace_weights",
            "q_trace_bias",
            "average_reward",
            "last_observation",
            "last_action",
            "epsilon",
            "rng_key",
            "step_count",
            "step_words",
            "birth_timestamp",
            "uptime_s",
        ),
    ),
    (
        FeatureBankRouterState,
        (
            "descriptors",
            "route_count",
            "generation_count",
            "route_words",
            "generation_words",
        ),
    ),
    (
        IntegratedPlannerEvaluation,
        (
            "predicted_partner_probabilities",
            "partner_probabilities",
            "partner_probabilities_valid",
            "probability_violation",
            "expected_rewards",
            "expected_outcomes",
            "q_values",
            "centered_expected_rewards",
            "model_term",
            "applied_model_term",
            "planner_scores",
            "greedy_action",
            "cell_evaluations",
            "grounded_world",
        ),
    ),
    (
        IntegratedGroundedPlannerEvaluation,
        (
            "table_expected_rewards",
            "grounded_raw_predictions",
            "grounded_reward_cells",
            "grounded_expected_rewards",
            "predictions_valid",
            "planner_applied",
            "cell_evaluations",
        ),
    ),
    (
        IntegratedPlannerSelection,
        (
            "action",
            "noisy_greedy_action",
            "random_action",
            "explored",
            "externally_forced",
            "rng_key_before",
            "rng_key_after",
        ),
    ),
    (HiddenPartnerWorldFilterState, ("posterior_mean", "step_count", "valid")),
)

_reviewed_initial_field_map = dict(_REVIEWED_INITIAL_DATACLASS_FIELDS)
V6_MATCHED_INITIAL_STATE_NODE_COUNT = 14
V6_MATCHED_INITIAL_STATE_LEAF_COUNT = 135
if len(_reviewed_initial_field_map) != len(_REVIEWED_INITIAL_DATACLASS_FIELDS):
    raise RuntimeError("reviewed initial-state dataclass types must be unique")
for _state_type, _reviewed_fields in _REVIEWED_INITIAL_DATACLASS_FIELDS:
    if (
        tuple(
            field.name
            for field in dataclasses.fields(_state_type)  # type: ignore[arg-type]
        )
        != _reviewed_fields
    ):
        raise RuntimeError(
            f"initial-state schema changed for {_state_type.__module__}.{_state_type.__qualname__}"
        )

# The config token binds the intentionally different arm configuration.  The
# v6 runner canonicalizes host birth clocks before they reach this comparison.
V6_MATCHED_INITIAL_ALWAYS_ALLOWED_DIFFERENCE_PATHS: tuple[str, ...] = ("config_token",)
V6_MATCHED_INITIAL_CONTROL_ALLOWED_DIFFERENCE_PATHS: tuple[
    tuple[tuple[str, bool], tuple[str, ...]], ...
] = (
    (
        ("recurrent_memory_masked", True),
        (
            "agent.chi",
            "agent.control.last_observation",
            "agent.current_evaluation.applied_model_term",
            "agent.current_evaluation.centered_expected_rewards",
            "agent.current_evaluation.expected_rewards",
            "agent.current_evaluation.grounded_world.grounded_expected_rewards",
            "agent.current_evaluation.grounded_world.grounded_raw_predictions",
            "agent.current_evaluation.grounded_world.grounded_reward_cells",
            "agent.current_evaluation.model_term",
            "agent.current_evaluation.planner_scores",
        ),
    ),
    (
        ("table_planner", True),
        (
            "agent.control.last_action",
            "agent.current_evaluation.applied_model_term",
            "agent.current_evaluation.centered_expected_rewards",
            "agent.current_evaluation.expected_rewards",
            "agent.current_evaluation.greedy_action",
            "agent.current_evaluation.grounded_world.planner_applied",
            "agent.current_evaluation.model_term",
            "agent.current_evaluation.planner_scores",
            "agent.current_selection.action",
            "agent.current_selection.noisy_greedy_action",
            "action",
        ),
    ),
    (
        ("no_planning", True),
        (
            "agent.control.last_action",
            "agent.current_evaluation.applied_model_term",
            "agent.current_evaluation.greedy_action",
            "agent.current_evaluation.grounded_world.planner_applied",
            "agent.current_evaluation.planner_scores",
            "agent.current_selection.action",
            "agent.current_selection.noisy_greedy_action",
            "action",
        ),
    ),
    (
        ("uniform_action", False),
        (
            "agent.control.last_action",
            "agent.current_selection.action",
            "agent.current_selection.externally_forced",
            "action",
        ),
    ),
    (
        ("equal_cue", False),
        (
            "world.current_cues",
            "agent.state_builder.hidden",
            "agent.state_builder.parameter_sensitivity",
            "agent.raw_observation",
            "agent.phi",
            "agent.chi",
            "agent.control.last_observation",
            "agent.control.last_action",
            "agent.current_evaluation.applied_model_term",
            "agent.current_evaluation.centered_expected_rewards",
            "agent.current_evaluation.expected_rewards",
            "agent.current_evaluation.greedy_action",
            "agent.current_evaluation.grounded_world.grounded_expected_rewards",
            "agent.current_evaluation.grounded_world.grounded_raw_predictions",
            "agent.current_evaluation.grounded_world.grounded_reward_cells",
            "agent.current_evaluation.model_term",
            "agent.current_evaluation.planner_scores",
            "agent.current_selection.action",
            "agent.current_selection.noisy_greedy_action",
            "world_filter.posterior_mean",
            "action",
        ),
    ),
    (
        ("row_bias", False),
        (
            "agent.grounded_world.weights",
            "agent.control.last_action",
            "agent.current_evaluation.applied_model_term",
            "agent.current_evaluation.centered_expected_rewards",
            "agent.current_evaluation.expected_rewards",
            "agent.current_evaluation.greedy_action",
            "agent.current_evaluation.grounded_world.grounded_expected_rewards",
            "agent.current_evaluation.grounded_world.grounded_raw_predictions",
            "agent.current_evaluation.grounded_world.grounded_reward_cells",
            "agent.current_evaluation.model_term",
            "agent.current_evaluation.planner_scores",
            "agent.current_selection.action",
            "agent.current_selection.noisy_greedy_action",
            "action",
        ),
    ),
)
_control_initial_allowlist = dict(V6_MATCHED_INITIAL_CONTROL_ALLOWED_DIFFERENCE_PATHS)
if len(_control_initial_allowlist) != len(
    V6_MATCHED_INITIAL_CONTROL_ALLOWED_DIFFERENCE_PATHS
) or any(key not in V6_MATCHED_SUITE_MEMBER_ORDER for key in _control_initial_allowlist):
    raise RuntimeError("v6 initial-state allowlist contains duplicate or unknown controls")
if any(len(paths) != len(set(paths)) for paths in _control_initial_allowlist.values()):
    raise RuntimeError("v6 initial-state allowlist contains duplicate leaf paths")


MatchedSuiteValidationStatus = Literal[
    "STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE",
    "STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE",
]


@dataclasses.dataclass(frozen=True, slots=True)
class V6MatchedDevelopmentSuite:
    """One unvalidated, authority-free panel of already-executed v6 runs."""

    schema: str
    status: str
    development_only: bool
    structural_only: bool
    replay_verified: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    supplied_key_data: jax.Array
    runs: tuple[V6DevelopmentRun, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class V6MatchedSuiteValidationError:
    """One deterministic path-addressed matched-suite validation error."""

    code: str
    path: str
    message: str


@dataclasses.dataclass(frozen=True, slots=True)
class V6MatchedDevelopmentSuiteValidation:
    """Structural-only verdict for a complete matched development panel."""

    schema: str
    status: MatchedSuiteValidationStatus
    development_only: bool
    structural_only: bool
    replay_verified: bool
    execution_authorized: bool
    evidence_authorized: bool
    scientific_promotion_allowed: bool
    validated_member_count: int
    errors: tuple[V6MatchedSuiteValidationError, ...]


@dataclasses.dataclass
class _ValidationContext:
    errors: list[V6MatchedSuiteValidationError] = dataclasses.field(default_factory=list)

    def add(self, code: str, path: str, message: str) -> None:
        self.errors.append(V6MatchedSuiteValidationError(code=code, path=path, message=message))


def build_v6_matched_development_suite(
    supplied_key_data: jax.Array,
    runs: tuple[V6DevelopmentRun, ...],
) -> V6MatchedDevelopmentSuite:
    """Wrap already-executed runs without validating or executing them."""

    return V6MatchedDevelopmentSuite(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA,
        status=MATCHED_DEVELOPMENT_SUITE_RECORD,
        development_only=DEVELOPMENT_ONLY,
        structural_only=STRUCTURAL_ONLY,
        replay_verified=REPLAY_VERIFIED,
        execution_authorized=EXECUTION_AUTHORIZED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        supplied_key_data=supplied_key_data,
        runs=runs,
    )


def _host_array(
    ctx: _ValidationContext,
    value: object,
    *,
    path: str,
    shape: tuple[int, ...],
    dtype: np.dtype[np.generic],
) -> np.ndarray | None:
    if isinstance(value, jax.core.Tracer):
        ctx.add("TRACER", path, "must not contain a JAX tracer")
        return None
    if isinstance(value, np.ndarray):
        ctx.add("ARRAY_CLASS", path, "must be a concrete JAX array")
        return None
    if not isinstance(value, jax.Array):
        ctx.add("TYPE", path, "must be a concrete JAX array")
        return None
    if value.shape != shape:
        ctx.add("SHAPE", path, f"must have shape {shape}")
        return None
    if value.dtype != dtype:
        ctx.add("DTYPE", path, f"must have dtype {dtype.name}")
        return None
    try:
        return np.asarray(jax.device_get(value))
    except (TypeError, ValueError) as exc:
        ctx.add("CONCRETE", path, f"must be host-materializable: {exc}")
        return None


def _array_bit_equal(left: object, right: object) -> bool:
    """Compare two concrete JAX leaves including float and PRNG bit patterns."""

    if not isinstance(left, jax.Array) or not isinstance(right, jax.Array):
        return False
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    try:
        left_is_key = jnp.issubdtype(left.dtype, jax.dtypes.prng_key)
        right_is_key = jnp.issubdtype(right.dtype, jax.dtypes.prng_key)
        if left_is_key or right_is_key:
            if not left_is_key or not right_is_key:
                return False
            if str(jax.random.key_impl(left)) != str(jax.random.key_impl(right)):
                return False
            left_host = np.asarray(jax.device_get(jax.random.key_data(left)))
            right_host = np.asarray(jax.device_get(jax.random.key_data(right)))
        else:
            left_host = np.asarray(jax.device_get(left))
            right_host = np.asarray(jax.device_get(right))
    except (TypeError, ValueError):
        return False
    return (
        left_host.shape == right_host.shape
        and left_host.dtype == right_host.dtype
        and left_host.tobytes(order="C") == right_host.tobytes(order="C")
    )


def _initial_state_schema_and_leaves(
    state: HiddenPartnerWorldOnlineState,
) -> tuple[tuple[tuple[str, str, tuple[str, ...]], ...], tuple[tuple[str, object], ...]]:
    """Recursively cover the reviewed initial-state schema and every leaf."""

    nodes: list[tuple[str, str, tuple[str, ...]]] = []
    leaves: list[tuple[str, object]] = []

    def walk(value: object, path: str) -> None:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            value_type = type(value)
            reviewed_fields = _reviewed_initial_field_map.get(value_type)
            if reviewed_fields is None:
                raise TypeError(
                    "unreviewed initial-state dataclass at "
                    f"{path or '<root>'}: {value_type.__module__}.{value_type.__qualname__}"
                )
            observed_fields = tuple(field.name for field in dataclasses.fields(value))
            if observed_fields != reviewed_fields:
                raise ValueError(f"reviewed initial-state fields changed at {path or '<root>'}")
            type_name = f"{value_type.__module__}.{value_type.__qualname__}"
            nodes.append((path, type_name, observed_fields))
            for field_name in observed_fields:
                child_path = f"{path}.{field_name}" if path else field_name
                walk(getattr(value, field_name), child_path)
            return
        if type(value) is tuple:
            nodes.append((path, "builtins.tuple", (str(len(value)),)))
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(value, jax.Array) or type(value) in {
            bool,
            bytes,
            float,
            int,
            str,
            type(None),
        }:
            leaves.append((path, value))
            return
        raise TypeError(
            f"unreviewed initial-state leaf at {path}: "
            f"{type(value).__module__}.{type(value).__qualname__}"
        )

    walk(state, "")
    if len(nodes) != V6_MATCHED_INITIAL_STATE_NODE_COUNT:
        raise RuntimeError("reviewed initial-state recursive node count changed")
    if len(leaves) != V6_MATCHED_INITIAL_STATE_LEAF_COUNT:
        raise RuntimeError("reviewed initial-state recursive leaf count changed")
    return tuple(nodes), tuple(leaves)


def _leaf_contract_equal(left: object, right: object) -> bool:
    if isinstance(left, jax.Array) or isinstance(right, jax.Array):
        if not isinstance(left, jax.Array) or not isinstance(right, jax.Array):
            return False
        if left.shape != right.shape or left.dtype != right.dtype:
            return False
        try:
            left_is_key = jnp.issubdtype(left.dtype, jax.dtypes.prng_key)
            right_is_key = jnp.issubdtype(right.dtype, jax.dtypes.prng_key)
            return left_is_key == right_is_key and (
                not left_is_key or str(jax.random.key_impl(left)) == str(jax.random.key_impl(right))
            )
        except (TypeError, ValueError):
            return False
    return type(left) is type(right)


def _leaf_bit_equal(left: object, right: object) -> bool:
    if not _leaf_contract_equal(left, right):
        return False
    if isinstance(left, jax.Array):
        return _array_bit_equal(left, right)
    if type(left) is float:
        left_bits = np.asarray(left, dtype=np.float64).view(np.uint64)
        right_bits = np.asarray(right, dtype=np.float64).view(np.uint64)
        return bool(left_bits == right_bits)
    return bool(left == right)


def _allowed_initial_difference_paths(run: V6DevelopmentRun) -> tuple[str, ...]:
    return (
        *V6_MATCHED_INITIAL_ALWAYS_ALLOWED_DIFFERENCE_PATHS,
        *_control_initial_allowlist.get((run.control_name, run.primary), ()),
    )


def _initial_state_differing_leaf_paths(
    baseline: HiddenPartnerWorldOnlineState,
    member: HiddenPartnerWorldOnlineState,
) -> tuple[str, ...]:
    """Return every bit-different leaf after requiring the exact reviewed schema."""

    baseline_schema, baseline_leaves = _initial_state_schema_and_leaves(baseline)
    member_schema, member_leaves = _initial_state_schema_and_leaves(member)
    if member_schema != baseline_schema:
        raise ValueError("recursive initial-state node schema differs")
    baseline_paths = tuple(path for path, _ in baseline_leaves)
    member_paths = tuple(path for path, _ in member_leaves)
    if member_paths != baseline_paths:
        raise ValueError("recursive initial-state leaf order differs")

    differing: list[str] = []
    for (path, expected), (_, observed) in zip(
        baseline_leaves,
        member_leaves,
        strict=True,
    ):
        if not _leaf_contract_equal(observed, expected):
            raise TypeError(f"initial-state leaf contract differs at {path}")
        if not _leaf_bit_equal(observed, expected):
            differing.append(path)
    return tuple(differing)


def _initial_state_comparison_failures(
    baseline: HiddenPartnerWorldOnlineState,
    run: V6DevelopmentRun,
) -> tuple[tuple[str, str, str], ...]:
    """Return structural/value mismatches outside reviewed control differences."""

    baseline_schema, baseline_leaves = _initial_state_schema_and_leaves(baseline)
    member_schema, member_leaves = _initial_state_schema_and_leaves(run.initial_state)
    if member_schema != baseline_schema:
        return (("INITIAL_SCHEMA", "initial_state", "recursive node schema differs"),)
    baseline_paths = tuple(path for path, _ in baseline_leaves)
    member_paths = tuple(path for path, _ in member_leaves)
    if member_paths != baseline_paths:
        return (("INITIAL_SCHEMA", "initial_state", "recursive leaf order differs"),)

    allowed = _allowed_initial_difference_paths(run)
    failures: list[tuple[str, str, str]] = []
    for (path, expected), (_, observed) in zip(
        baseline_leaves,
        member_leaves,
        strict=True,
    ):
        if not _leaf_contract_equal(observed, expected):
            failures.append(("INITIAL_CONTRACT", path, "leaf type, shape, or dtype differs"))
        elif not _leaf_bit_equal(observed, expected) and path not in allowed:
            failures.append(("INITIAL_SHARED_STATE", path, "leaf differs bit-exactly"))
    return tuple(failures)


def _validate_envelope(ctx: _ValidationContext, suite: V6MatchedDevelopmentSuite) -> None:
    if type(suite.schema) is not str:
        ctx.add("TYPE", "suite.schema", "must be an exact built-in str")
    elif suite.schema != HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA:
        ctx.add("SCHEMA", "suite.schema", "differs from the matched-suite schema")
    if type(suite.status) is not str:
        ctx.add("TYPE", "suite.status", "must be an exact built-in str")
    elif suite.status != MATCHED_DEVELOPMENT_SUITE_RECORD:
        ctx.add("STATUS", "suite.status", "must remain an unvalidated record status")
    if suite.development_only is not True:
        ctx.add("AUTHORITY", "suite.development_only", "must be true")
    if suite.structural_only is not True:
        ctx.add("AUTHORITY", "suite.structural_only", "must be true")
    if suite.replay_verified is not False:
        ctx.add("AUTHORITY", "suite.replay_verified", "must be false")
    for field in (
        "execution_authorized",
        "evidence_authorized",
        "scientific_promotion_allowed",
    ):
        if getattr(suite, field) is not False:
            ctx.add("AUTHORITY", f"suite.{field}", "must be false")


def _validate_per_run_results(
    ctx: _ValidationContext,
    runs: tuple[V6DevelopmentRun, ...],
) -> int:
    validated_count = 0
    for index, run in enumerate(runs):
        path = f"suite.runs[{index}]"
        try:
            result = validate_hidden_partner_lifecycle_world_v6_development_run(run)
        except (AttributeError, IndexError, OSError, TypeError, ValueError, RuntimeError) as exc:
            ctx.add("MEMBER_VALIDATOR", path, f"per-run validator failed closed: {exc}")
            continue
        if type(result) is not V6DevelopmentRunValidation:
            ctx.add("MEMBER_VALIDATOR", path, "returned a noncanonical validation result")
            continue
        if (
            result.status != STRUCTURALLY_VALID_DEVELOPMENT_RUN
            or result.errors
            or result.development_only is not True
            or result.structural_only is not True
            or result.replay_verified is not False
            or result.execution_authorized is not False
            or result.evidence_authorized is not False
            or result.scientific_promotion_allowed is not False
        ):
            ctx.add("MEMBER_INVALID", path, "did not pass strict per-run structural validation")
            continue
        validated_count += 1
    return validated_count


def _validate_order_and_bindings(
    ctx: _ValidationContext,
    runs: tuple[V6DevelopmentRun, ...],
) -> None:
    if len(runs) != len(V6_MATCHED_SUITE_MEMBER_ORDER):
        ctx.add(
            "MEMBER_COUNT",
            "suite.runs",
            f"must contain exactly {len(V6_MATCHED_SUITE_MEMBER_ORDER)} members",
        )
    observed_order = tuple(
        (run.control_name, run.primary) for run in runs if type(run) is V6DevelopmentRun
    )
    if observed_order != V6_MATCHED_SUITE_MEMBER_ORDER:
        ctx.add(
            "MEMBER_ORDER",
            "suite.runs",
            "must contain the exact ordered primary and diagnostic controls",
        )
    try:
        readiness = validate_v6_control_suite_readiness(require_v6_control_suite_ready())
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("LIVE_BINDING", "live.readiness", f"cannot load canonical bindings: {exc}")
        return
    expected_binding_order = tuple(
        (binding.name, binding.family == "primary") for binding in readiness.bindings
    )
    if expected_binding_order != V6_MATCHED_SUITE_MEMBER_ORDER:
        ctx.add("LIVE_BINDING", "live.readiness.bindings", "canonical binding order has drifted")
        return
    for index, (run, binding) in enumerate(zip(runs, readiness.bindings, strict=False)):
        if type(run) is not V6DevelopmentRun:
            continue
        path = f"suite.runs[{index}]"
        if run.control_matrix_sha256 != readiness.control_matrix_sha256:
            ctx.add("CONTROL_MATRIX", f"{path}.control_matrix_sha256", "differs from live matrix")
        if run.control_config_sha256 != binding.control_config_sha256:
            ctx.add("CONTROL_BINDING", f"{path}.control_config_sha256", "differs from live arm")
        if run.bridge_config_sha256 != binding.bridge_config_sha256:
            ctx.add("BRIDGE_BINDING", f"{path}.bridge_config_sha256", "differs from live arm")


def _validate_shared_records(
    ctx: _ValidationContext,
    suite: V6MatchedDevelopmentSuite,
    suite_key_data: np.ndarray,
) -> None:
    runs = suite.runs
    if not runs or type(runs[0]) is not V6DevelopmentRun:
        return
    baseline = runs[0]
    try:
        baseline_plan = canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(baseline.plan)
        _initial_state_schema_and_leaves(baseline.initial_state)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        ctx.add("BASELINE", "suite.runs[0]", f"cannot traverse canonical baseline: {exc}")
        return
    baseline_stream = _host_array(
        ctx,
        baseline.stream_code,
        path="suite.runs[0].stream_code",
        shape=(MAX_SCAN_STEPS,),
        dtype=np.dtype(np.uint8),
    )
    if baseline_stream is not None and baseline_stream.ndim != 1:
        ctx.add("SHAPE", "suite.runs[0].stream_code", "must be rank one")
        baseline_stream = None

    baseline_initial_bits: np.ndarray | None = None
    try:
        baseline_initial_bits = _host_array(
            ctx,
            baseline.rng.initial_stream_bits,
            path="suite.runs[0].rng.initial_stream_bits",
            shape=(),
            dtype=np.dtype(np.uint8),
        )
    except AttributeError as exc:
        ctx.add("RNG_IDENTITY", "suite.runs[0].rng", f"is incomplete: {exc}")

    for index, run in enumerate(runs):
        if type(run) is not V6DevelopmentRun:
            continue
        path = f"suite.runs[{index}]"
        try:
            member_keys = _host_array(
                ctx,
                run.rng.supplied_key_data,
                path=f"{path}.rng.supplied_key_data",
                shape=(2, 2),
                dtype=np.dtype(np.uint32),
            )
            if member_keys is not None and not np.array_equal(member_keys, suite_key_data):
                ctx.add("KEY_DRIFT", f"{path}.rng.supplied_key_data", "differs from suite key")

            if run.source_closure_hashes != baseline.source_closure_hashes:
                ctx.add("SOURCE_IDENTITY", f"{path}.source_closure_hashes", "differs across arms")
            if run.runtime != baseline.runtime:
                ctx.add("RUNTIME_IDENTITY", f"{path}.runtime", "differs across arms")
            if run.control_matrix_sha256 != baseline.control_matrix_sha256:
                ctx.add("CONTROL_MATRIX", f"{path}.control_matrix_sha256", "differs across arms")

            member_plan = canonical_hidden_partner_lifecycle_world_v6_scan_plan_bytes(run.plan)
            if member_plan != baseline_plan:
                ctx.add("PLAN_GEOMETRY", f"{path}.plan", "differs across arms")
            if run.resources != baseline.resources:
                ctx.add("RESOURCE_IDENTITY", f"{path}.resources", "differs across arms")

            for field in V6_MATCHED_RNG_SHARED_FIELD_ORDER:
                if not _array_bit_equal(getattr(run.rng, field), getattr(baseline.rng, field)):
                    ctx.add("RNG_IDENTITY", f"{path}.rng.{field}", "differs across arms")

            for code, initial_path, message in _initial_state_comparison_failures(
                baseline.initial_state,
                run,
            ):
                suffix = f".{initial_path}" if initial_path else ""
                ctx.add(code, f"{path}.initial_state{suffix}", message)

            stream = _host_array(
                ctx,
                run.stream_code,
                path=f"{path}.stream_code",
                shape=(MAX_SCAN_STEPS,),
                dtype=np.dtype(np.uint8),
            )
            is_equal_cue = run.control_name == "equal_cue" and run.primary is False
            if baseline_stream is not None and stream is not None:
                difference = np.bitwise_xor(stream, baseline_stream)
                if is_equal_cue:
                    if bool(np.any(np.bitwise_and(difference, _STREAM_NONCUE_MASK) != 0)):
                        ctx.add(
                            "STREAM_NONCUE_DRIFT",
                            f"{path}.stream_code",
                            "equal-cue may differ only in transition cue bits 5-6",
                        )
                elif bool(np.any(difference != 0)):
                    ctx.add(
                        "STREAM_DRIFT",
                        f"{path}.stream_code",
                        "ordinary arms must share the exact full stream",
                    )

            initial_bits = _host_array(
                ctx,
                run.rng.initial_stream_bits,
                path=f"{path}.rng.initial_stream_bits",
                shape=(),
                dtype=np.dtype(np.uint8),
            )
            if baseline_initial_bits is not None and initial_bits is not None:
                initial_difference = np.bitwise_xor(initial_bits, baseline_initial_bits)
                if is_equal_cue:
                    if bool(np.bitwise_and(initial_difference, _INITIAL_NONCUE_MASK).item() != 0):
                        ctx.add(
                            "INITIAL_STREAM_NONCUE_DRIFT",
                            f"{path}.rng.initial_stream_bits",
                            "equal-cue may differ only in initial cue bits 4-5",
                        )
                elif bool(initial_difference.item() != 0):
                    ctx.add(
                        "INITIAL_STREAM_DRIFT",
                        f"{path}.rng.initial_stream_bits",
                        "ordinary arms must share the exact initial stream byte",
                    )
        except (AttributeError, IndexError, TypeError, ValueError, RuntimeError) as exc:
            ctx.add("MEMBER_STRUCTURE", path, f"matched projection failed closed: {exc}")


def validate_v6_matched_development_suite(
    suite: object,
) -> V6MatchedDevelopmentSuiteValidation:
    """Structurally match a panel; composed member validation initializes but never scans."""

    ctx = _ValidationContext()
    if type(suite) is not V6MatchedDevelopmentSuite:
        ctx.add("TYPE", "suite", "must be an exact V6MatchedDevelopmentSuite")
        return V6MatchedDevelopmentSuiteValidation(
            schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA,
            status=STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE,
            development_only=DEVELOPMENT_ONLY,
            structural_only=STRUCTURAL_ONLY,
            replay_verified=REPLAY_VERIFIED,
            execution_authorized=EXECUTION_AUTHORIZED,
            evidence_authorized=EVIDENCE_AUTHORIZED,
            scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
            validated_member_count=0,
            errors=tuple(ctx.errors),
        )
    exact_suite = suite
    _validate_envelope(ctx, exact_suite)
    if type(exact_suite.runs) is not tuple:
        ctx.add("TYPE", "suite.runs", "must be an exact built-in tuple")
        runs: tuple[V6DevelopmentRun, ...] = ()
    else:
        runs = exact_suite.runs
        for index, run in enumerate(runs):
            if type(run) is not V6DevelopmentRun:
                ctx.add("TYPE", f"suite.runs[{index}]", "must be an exact V6DevelopmentRun")

    key_data = _host_array(
        ctx,
        exact_suite.supplied_key_data,
        path="suite.supplied_key_data",
        shape=(2, 2),
        dtype=np.dtype(np.uint32),
    )
    _validate_order_and_bindings(ctx, runs)
    validated_count = _validate_per_run_results(ctx, runs)
    if key_data is not None:
        _validate_shared_records(ctx, exact_suite, key_data)

    status: MatchedSuiteValidationStatus = (
        STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE
        if not ctx.errors
        else STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE
    )
    return V6MatchedDevelopmentSuiteValidation(
        schema=HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA,
        status=status,
        development_only=DEVELOPMENT_ONLY,
        structural_only=STRUCTURAL_ONLY,
        replay_verified=REPLAY_VERIFIED,
        execution_authorized=EXECUTION_AUTHORIZED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        validated_member_count=validated_count,
        errors=tuple(ctx.errors),
    )


__all__ = [
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "EXECUTION_AUTHORIZED",
    "HIDDEN_PARTNER_LIFECYCLE_WORLD_V6_MATCHED_SUITE_SCHEMA",
    "MATCHED_DEVELOPMENT_SUITE_RECORD",
    "REPLAY_VERIFIED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "STRUCTURALLY_VALID_MATCHED_DEVELOPMENT_SUITE",
    "STRUCTURALLY_INVALID_MATCHED_DEVELOPMENT_SUITE",
    "STRUCTURAL_ONLY",
    "V6_MATCHED_INITIAL_ALWAYS_ALLOWED_DIFFERENCE_PATHS",
    "V6_MATCHED_INITIAL_CONTROL_ALLOWED_DIFFERENCE_PATHS",
    "V6_MATCHED_INITIAL_STATE_LEAF_COUNT",
    "V6_MATCHED_INITIAL_STATE_NODE_COUNT",
    "V6_MATCHED_REAL_VALIDATOR_POSITIVE_INTEGRATION_BLOCKER",
    "V6_MATCHED_RNG_FIELD_PARTITION",
    "V6_MATCHED_RNG_SHARED_FIELD_ORDER",
    "V6_MATCHED_SUITE_MEMBER_ORDER",
    "V6MatchedDevelopmentSuite",
    "V6MatchedDevelopmentSuiteValidation",
    "V6MatchedSuiteValidationError",
    "build_v6_matched_development_suite",
    "validate_v6_matched_development_suite",
]
