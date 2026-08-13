# mypy: disable-error-code="call-arg"
"""Development-only factorized typed-surprise intervention scaffold.

One deterministic online prefix trains an existing :class:`BehaviorModel` and
an existing :class:`GroundedJointWorldModel` without replay, reset, or a
learner-visible task identifier.  The learned prefix states are then copied
into four exactly matched continuations:

* no change;
* a partner-policy mapping change under the original physical law;
* a physical-law change under the original partner mapping; and
* a distractor/noisy-TV change under the original partner and physical laws.

The behavior model predicts the partner action before it is revealed.  The
world model predicts every partner-conditioned cell at that same pre-action
boundary.  Conditional physical error is selected only after the realized
partner action is revealed; the behavior-weighted marginal remains the
pre-action planning quantity.  Both physical metrics explicitly include the
two physical next-observation coordinates, reward, and discount while
excluding the isolated distractor coordinate.  Distractor error is reported
separately.

This module is an in-memory L0 mechanism probe.  It has no output writer,
threshold, expected winner, verdict, artifact authority, evidence claim, or
scientific-promotion path.  Its keyed distractor sequence is deterministic for
reproducibility only; that construction is not a predictability result or a
statistical calibration of aleatoric uncertainty.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.behavior_model import (
    BehaviorModel,
    BehaviorModelConfig,
    BehaviorModelState,
    measure_behavior_model_state_nbytes,
)
from alberta_framework.core.grounded_joint_world_model import (
    GroundedJointWorldModel,
    GroundedJointWorldModelConfig,
    GroundedJointWorldModelState,
    measure_grounded_joint_world_state_nbytes,
)

DEVELOPMENT_SCHEMA: Final = "alberta.typed-surprise-factorization.development.v1"
DEVELOPMENT_ONLY: Final = True
ASSESSMENT_STATUS: Final = "not_assessed"
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
BENCHMARK_EXECUTION_AUTHORITY: Final = False
ARTIFACT_AUTHORITY: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_CLAIMED: Final = False
THRESHOLDS_FROZEN: Final = False
DEVELOPMENT_KEY_FROZEN: Final = False
TASK_IDENTIFIERS_EXPOSED: Final = False
RESETS_EXPOSED: Final = False

BRANCH_NAMES: Final = (
    "no_change",
    "partner_mapping_change",
    "physical_law_change",
    "noisy_tv_distractor_change",
)
METRIC_NAMES: Final = (
    "behavior_nll",
    "behavior_brier",
    "conditional_physical_world_mse",
    "marginal_physical_world_mse",
    "distractor_squared_error",
)
SOURCE_GENERATOR_VERSION: Final = "deterministic-typed-surprise-factorization-v1"
N_PARTNER_ACTIONS: Final = 2
N_FOCAL_ACTIONS: Final = 2
BEHAVIOR_FEATURE_DIM: Final = 2
WORLD_REPRESENTATION_DIM: Final = 2
PHYSICAL_OBSERVATION_DIM: Final = 2
DISTRACTOR_OBSERVATION_INDEX: Final = PHYSICAL_OBSERVATION_DIM
TARGET_OBSERVATION_DIM: Final = PHYSICAL_OBSERVATION_DIM + 1
TARGET_DIM: Final = TARGET_OBSERVATION_DIM + 2
PHYSICAL_TARGET_INDICES: Final = (0, 1, 3, 4)
PASSES_OVER_PREFIX: Final = 1
PASSES_OVER_EACH_CONTINUATION: Final = 1
LOGICAL_PREUPDATE_FLOAT32_SCALARS_PER_STEP: Final = (
    N_PARTNER_ACTIONS
    + N_PARTNER_ACTIONS * TARGET_DIM
    + TARGET_DIM
    + TARGET_DIM
)
LOGICAL_PREUPDATE_WORK_NBYTES_PER_STEP: Final = (
    4 * LOGICAL_PREUPDATE_FLOAT32_SCALARS_PER_STEP
)

_INT32_MAX: Final = 2**31 - 1
_FLOAT32_TINY: Final = float(np.finfo(np.float32).tiny)
_NORMAL_DISCOUNT: Final = np.float32(0.95)
_CHANGED_DISCOUNT: Final = np.float32(0.80)
_LIMITATIONS: Final = (
    "one deterministic binary-action construction is not a population result",
    "the world probe uses one fixed exhaustively covered physical context",
    "the learners receive no branch, phase, regime, or task identifier",
    "conditional physical surprise selects the realized partner action only after reveal",
    "the marginal physical metric is the behavior-weighted pre-action planning quantity",
    "physical metrics exclude the isolated distractor coordinate everywhere",
    "the keyed noisy-TV sequence ensures reproducibility, not unpredictability or calibration",
    "the four branches copy one learned prefix state and are not one continuing deployed life",
    "descriptive branch-minus-control deltas have no threshold or expected winner",
    "logical work bytes exclude compiler buffers, allocator peaks, and update temporaries",
    "no control policy, visual encoder, multi-step planning, or scale claim is tested",
)


def _narrow_positive_float32(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        narrowed = np.float32(value)
    if (
        not np.isfinite(narrowed)
        or narrowed <= np.float32(0.0)
        or float(narrowed) < _FLOAT32_TINY
    ):
        raise ValueError(f"{name} must remain finite and normal after float32 narrowing")
    return float(narrowed)


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseFactorizationConfig:
    """Small deterministic development construction with no evidence role."""

    prefix_steps: int = 32
    continuation_steps: int = 16
    behavior_step_size: float = 0.20
    world_step_size: float = 0.20
    world_initialization_scale: float = 0.01
    development_key: int = 0

    def __post_init__(self) -> None:
        for name in ("prefix_steps", "continuation_steps", "development_key"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an exact integer")
        for name in ("prefix_steps", "continuation_steps"):
            value = getattr(self, name)
            if value < 4 or value % 4 != 0:
                raise ValueError(f"{name} must be a positive multiple of four")
        if self.prefix_steps + self.continuation_steps >= _INT32_MAX:
            raise ValueError("each copied learner clock must fit the exact int32 contract")
        if not 0 <= self.development_key <= 2**32 - 1:
            raise ValueError("development_key must fit one uint32 word")
        for name in (
            "behavior_step_size",
            "world_step_size",
            "world_initialization_scale",
        ):
            object.__setattr__(
                self,
                name,
                _narrow_positive_float32(getattr(self, name), name=name),
            )
        if self.world_step_size > 1.0:
            raise ValueError("world_step_size must not exceed one")


def _configs_exact(
    left: TypedSurpriseFactorizationConfig,
    right: TypedSurpriseFactorizationConfig,
) -> bool:
    for field in dataclasses.fields(TypedSurpriseFactorizationConfig):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if type(left_value) is not type(right_value):
            return False
        if type(left_value) is float:
            if struct.pack(">d", left_value) != struct.pack(">d", right_value):
                return False
        elif left_value != right_value:
            return False
    return True


def _canonical_config_copy(
    config: TypedSurpriseFactorizationConfig,
) -> TypedSurpriseFactorizationConfig:
    """Re-run every config guard and reject noncanonical post-init mutation."""

    if type(config) is not TypedSurpriseFactorizationConfig:
        raise TypeError("config must be a TypedSurpriseFactorizationConfig")
    payload = {
        field.name: getattr(config, field.name)
        for field in dataclasses.fields(TypedSurpriseFactorizationConfig)
    }
    reconstructed = TypedSurpriseFactorizationConfig(**payload)
    if not _configs_exact(config, reconstructed):
        raise ValueError("config fields are not in canonical exact types or float32 form")
    return reconstructed


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurprisePreObservation:
    """Complete learner input available before the partner action is revealed."""

    behavior_features: Array
    world_representation: Array
    focal_action: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseFeedback:
    """Ordinary transition feedback exposed only after pre-action prediction."""

    partner_action: Array
    next_observation: Array
    reward: Array
    discount: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseSegmentSource:
    """Raw source arrays for one uninterrupted prefix or continuation."""

    behavior_features: Array
    world_representations: Array
    focal_actions: Array
    partner_actions: Array
    next_physical_observations: Array
    next_distractors: Array
    rewards: Array
    discounts: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseBranchSource:
    """Evaluator-owned branch name and continuation; name never reaches a learner."""

    name: str
    segment: TypedSurpriseSegmentSource


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseSource:
    """Deterministic common prefix and four matched evaluator continuations."""

    config: TypedSurpriseFactorizationConfig
    prefix: TypedSurpriseSegmentSource
    branches: tuple[
        TypedSurpriseBranchSource,
        TypedSurpriseBranchSource,
        TypedSurpriseBranchSource,
        TypedSurpriseBranchSource,
    ]
    generator_contract_sha256: str
    input_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseTrajectory:
    """Exact predict-before-update arrays for one source segment."""

    behavior_probabilities_pre: Array
    behavior_probabilities_update: Array
    world_predictions_by_partner_pre: Array
    conditional_world_predictions_pre: Array
    conditional_world_predictions_update: Array
    marginal_world_predictions_pre: Array
    world_targets: Array
    behavior_nll: Array
    behavior_brier: Array
    conditional_physical_world_mse: Array
    marginal_physical_world_mse: Array
    distractor_squared_error: Array
    behavior_pre_words: Array
    behavior_post_words: Array
    world_pre_words: Array
    world_post_words: Array
    behavior_update_applied: Array
    world_update_applied: Array
    behavior_prediction_bound: Array
    world_prediction_bound: Array
    selected_joint_action_index: Array
    world_weight_row_change_mask: Array
    world_bias_row_change_mask: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseBranchRun:
    """One continuation from an explicit copy of the learned prefix state."""

    name: str
    initial_behavior_state: BehaviorModelState
    final_behavior_state: BehaviorModelState
    initial_world_state: GroundedJointWorldModelState
    final_world_state: GroundedJointWorldModelState
    trajectory: TypedSurpriseTrajectory


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseSummary:
    """Descriptive float32 means and branch-minus-control differences only."""

    metric_names: tuple[str, str, str, str, str]
    common_prefix_mean: Array
    branch_means: Array
    branch_minus_control: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseResourceSummary:
    """Exact persistent-state, logical-work, and raw-output byte accounting.

    Logical pre-update work counts one two-value behavior simplex, two
    five-value conditional world cells, one five-value target, and one
    five-value marginal.  It excludes update arithmetic, compiler buffers,
    allocator peaks, and temporaries.  Per-branch call and RNG receipts live
    in :class:`TypedSurpriseMatchedBranchAudit`.
    """

    initial_total_state_nbytes: int
    common_prefix_total_state_nbytes: int
    branch_initial_total_state_nbytes: tuple[int, int, int, int]
    branch_final_total_state_nbytes: tuple[int, int, int, int]
    fixed_state_nbytes: bool
    logical_preupdate_float32_scalars_per_step: int
    logical_preupdate_work_nbytes_per_step: int
    prefix_trajectory_nbytes: int
    branch_trajectory_nbytes: tuple[int, int, int, int]
    total_trajectory_nbytes: int
    partner_world_cells_evaluated_per_step: int
    behavior_updates_per_step: int
    world_updates_per_step: int
    replay_capacity: int
    passes_over_prefix: int
    passes_over_each_continuation: int


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseMatchedBranchAudit:
    """Exact matched continuation calls, work, copied state, and RNG receipt."""

    name: str
    transitions: int
    behavior_pre_action_prediction_api_calls: int
    world_conditional_prediction_api_calls: int
    behavior_update_api_calls: int
    world_update_api_calls: int
    behavior_rng_draws: int
    world_rng_draws: int
    copied_learner_states: int
    logical_preupdate_work_nbytes: int
    initial_behavior_rng_key_bytes_hex: str
    final_behavior_rng_key_bytes_hex: str


@dataclasses.dataclass(frozen=True, slots=True)
class TypedSurpriseFactorizationReport:
    """In-memory descriptive report; never an evidence artifact."""

    schema: str
    status: str
    development_only: bool
    assessment_status: str
    scientific_promotion_allowed: bool
    benchmark_execution_authority: bool
    artifact_authority: bool
    output_writes_allowed: bool
    evidence_claimed: bool
    thresholds_frozen: bool
    development_key_frozen: bool
    task_identifiers_exposed: bool
    resets_exposed: bool
    learner_reset_count: int
    descriptive_claims_only: bool
    config: TypedSurpriseFactorizationConfig
    source: TypedSurpriseSource
    initial_behavior_state: BehaviorModelState
    initial_world_state: GroundedJointWorldModelState
    common_prefix_behavior_state: BehaviorModelState
    common_prefix_world_state: GroundedJointWorldModelState
    common_prefix_trajectory: TypedSurpriseTrajectory
    branches: tuple[
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
    ]
    summary: TypedSurpriseSummary
    resource: TypedSurpriseResourceSummary
    branch_audits: tuple[
        TypedSurpriseMatchedBranchAudit,
        TypedSurpriseMatchedBranchAudit,
        TypedSurpriseMatchedBranchAudit,
        TypedSurpriseMatchedBranchAudit,
    ]
    implementation_source_sha256: str
    common_prefix_state_sha256: str
    branch_state_sha256: str
    trajectory_sha256: str
    limitations: tuple[str, ...]


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _implementation_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _array_manifest_sha256(
    arrays: Iterable[tuple[str, Array | np.ndarray]],
    *,
    prefix: str,
) -> str:
    digest = hashlib.sha256(prefix.encode("ascii"))
    for name, value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"dtype": array.dtype.str, "name": name, "shape": list(array.shape)},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_bytes_equal(left: object, right: object) -> bool:
    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _source_contract(config: TypedSurpriseFactorizationConfig) -> dict[str, object]:
    return {
        "version": SOURCE_GENERATOR_VERSION,
        "config": dataclasses.asdict(config),
        "branch_names": list(BRANCH_NAMES),
        "metric_names": list(METRIC_NAMES),
        "behavior_feature_dim": BEHAVIOR_FEATURE_DIM,
        "world_representation_dim": WORLD_REPRESENTATION_DIM,
        "physical_observation_dim": PHYSICAL_OBSERVATION_DIM,
        "distractor_observation_index": DISTRACTOR_OBSERVATION_INDEX,
        "physical_target_indices": list(PHYSICAL_TARGET_INDICES),
        "n_focal_actions": N_FOCAL_ACTIONS,
        "n_partner_actions": N_PARTNER_ACTIONS,
        "public_signal": "alternating-binary-one-hot-behavior-cue-only",
        "world_representation": "fixed-ordinary-physical-context-[1,0]",
        "focal_action": "evaluator-fixed-two-step-block-alternation",
        "prefix_partner_mapping": "public-signal",
        "partner_intervention": "inverse-public-signal",
        "physical_law": "joint-action-only-affine-v1",
        "changed_physical_law": "joint-action-only-affine-v2",
        "stable_distractor_law": "fixed-joint-action-conditioned-affine-v1",
        "noisy_tv_law": "keyed-sha256-rademacher-reproducibility-only",
        "learner_branch_identifiers": False,
        "learner_resets": False,
        "passes_over_prefix": PASSES_OVER_PREFIX,
        "passes_over_each_continuation": PASSES_OVER_EACH_CONTINUATION,
    }


def _segment_arrays(
    segment: TypedSurpriseSegmentSource,
    *,
    prefix: str,
) -> tuple[tuple[str, Array], ...]:
    return tuple(
        (f"{prefix}.{field.name}", getattr(segment, field.name))
        for field in dataclasses.fields(TypedSurpriseSegmentSource)
    )


def _source_arrays(source: TypedSurpriseSource) -> tuple[tuple[str, Array], ...]:
    arrays = list(_segment_arrays(source.prefix, prefix="prefix"))
    for branch in source.branches:
        arrays.extend(_segment_arrays(branch.segment, prefix=f"branch.{branch.name}"))
    return tuple(arrays)


def _base_inputs(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    public_signal = (indices % 2).astype(np.int32)
    behavior_features = np.eye(BEHAVIOR_FEATURE_DIM, dtype=np.float32)[public_signal]
    world_representations = np.zeros(
        (len(indices), WORLD_REPRESENTATION_DIM), dtype=np.float32
    )
    world_representations[:, 0] = np.float32(1.0)
    focal_actions = ((indices // 2) % N_FOCAL_ACTIONS).astype(np.int32)
    return behavior_features, world_representations, focal_actions, public_signal


def _physical_targets(
    indices: np.ndarray,
    focal_actions: np.ndarray,
    partner_actions: np.ndarray,
    *,
    changed: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal_sign = (
        np.float32(2.0) * focal_actions.astype(np.float32) - np.float32(1.0)
    )
    partner_sign = (
        np.float32(2.0) * partner_actions.astype(np.float32) - np.float32(1.0)
    )
    if changed:
        first = (
            -np.float32(0.25) * focal_sign
            - np.float32(0.375) * partner_sign
        ).astype(np.float32)
        second = (
            -np.float32(0.25) * focal_sign + np.float32(0.50) * partner_sign
        ).astype(np.float32)
        rewards = (focal_actions != partner_actions).astype(np.float32)
        discount = _CHANGED_DISCOUNT
    else:
        first = (
            np.float32(0.375) * focal_sign
            - np.float32(0.125) * partner_sign
        ).astype(np.float32)
        second = (
            np.float32(0.50) * focal_sign + np.float32(0.25) * partner_sign
        ).astype(np.float32)
        rewards = (focal_actions == partner_actions).astype(np.float32)
        discount = _NORMAL_DISCOUNT
    observations = np.stack((first, second), axis=1).astype(np.float32)
    discounts = np.full(indices.shape, discount, dtype=np.float32)
    return observations, rewards, discounts


def _stable_distractors(
    focal_actions: np.ndarray,
    partner_actions: np.ndarray,
) -> np.ndarray:
    focal_sign = (
        np.float32(2.0) * focal_actions.astype(np.float32) - np.float32(1.0)
    )
    partner_sign = (
        np.float32(2.0) * partner_actions.astype(np.float32) - np.float32(1.0)
    )
    return cast(
        np.ndarray,
        (
            np.float32(0.0625) * focal_sign
            + np.float32(0.03125) * partner_sign
        ).astype(np.float32),
    )


def _keyed_noisy_tv_distractors(indices: np.ndarray, development_key: int) -> np.ndarray:
    signs = np.fromiter(
        (
            1.0
            if hashlib.sha256(f"{development_key}:{int(step)}".encode("ascii")).digest()[0]
            & 1
            else -1.0
            for step in indices
        ),
        dtype=np.float32,
        count=len(indices),
    )
    return (np.float32(1.25) * signs).astype(np.float32)


def _make_segment(
    indices: np.ndarray,
    *,
    partner_mapping_changed: bool,
    physical_law_changed: bool,
    noisy_tv_changed: bool,
    development_key: int,
) -> TypedSurpriseSegmentSource:
    behavior_features, representations, focal_actions, public_signal = _base_inputs(indices)
    if partner_mapping_changed:
        partner_actions = (1 - public_signal).astype(np.int32)
    else:
        partner_actions = public_signal.copy()
    physical, rewards, discounts = _physical_targets(
        indices,
        focal_actions,
        partner_actions,
        changed=physical_law_changed,
    )
    if noisy_tv_changed:
        distractors = _keyed_noisy_tv_distractors(indices, development_key)
    else:
        distractors = _stable_distractors(focal_actions, partner_actions)
    return TypedSurpriseSegmentSource(
        behavior_features=jnp.asarray(behavior_features, dtype=jnp.float32),
        world_representations=jnp.asarray(representations, dtype=jnp.float32),
        focal_actions=jnp.asarray(focal_actions, dtype=jnp.int32),
        partner_actions=jnp.asarray(partner_actions, dtype=jnp.int32),
        next_physical_observations=jnp.asarray(physical, dtype=jnp.float32),
        next_distractors=jnp.asarray(distractors, dtype=jnp.float32),
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
        discounts=jnp.asarray(discounts, dtype=jnp.float32),
    )


def _build_source_unchecked(config: TypedSurpriseFactorizationConfig) -> TypedSurpriseSource:
    prefix_indices = np.arange(config.prefix_steps, dtype=np.int64)
    continuation_indices = np.arange(
        config.prefix_steps,
        config.prefix_steps + config.continuation_steps,
        dtype=np.int64,
    )
    prefix = _make_segment(
        prefix_indices,
        partner_mapping_changed=False,
        physical_law_changed=False,
        noisy_tv_changed=False,
        development_key=config.development_key,
    )
    interventions = (
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (False, False, True),
    )
    branches = tuple(
        TypedSurpriseBranchSource(
            name=name,
            segment=_make_segment(
                continuation_indices,
                partner_mapping_changed=partner_changed,
                physical_law_changed=physical_changed,
                noisy_tv_changed=noisy_changed,
                development_key=config.development_key,
            ),
        )
        for name, (partner_changed, physical_changed, noisy_changed) in zip(
            BRANCH_NAMES,
            interventions,
            strict=True,
        )
    )
    contract_sha = _canonical_json_sha256(_source_contract(config))
    bare = TypedSurpriseSource(
        config=config,
        prefix=prefix,
        branches=cast(
            tuple[
                TypedSurpriseBranchSource,
                TypedSurpriseBranchSource,
                TypedSurpriseBranchSource,
                TypedSurpriseBranchSource,
            ],
            branches,
        ),
        generator_contract_sha256=contract_sha,
        input_sha256="",
    )
    return dataclasses.replace(
        bare,
        input_sha256=_array_manifest_sha256(
            _source_arrays(bare),
            prefix=f"{SOURCE_GENERATOR_VERSION}:{contract_sha}",
        ),
    )


def build_typed_surprise_source(
    config: TypedSurpriseFactorizationConfig | None = None,
) -> TypedSurpriseSource:
    """Build the exact deterministic prefix and four matched continuations."""

    supplied = TypedSurpriseFactorizationConfig() if config is None else config
    cfg = _canonical_config_copy(supplied)
    return _build_source_unchecked(cfg)


def _segment_contract_errors(
    segment: object,
    *,
    length: int,
    label: str,
) -> tuple[str, ...]:
    if type(segment) is not TypedSurpriseSegmentSource:
        return (f"{label} segment type differs",)
    shapes = {
        "behavior_features": (length, BEHAVIOR_FEATURE_DIM),
        "world_representations": (length, WORLD_REPRESENTATION_DIM),
        "focal_actions": (length,),
        "partner_actions": (length,),
        "next_physical_observations": (length, PHYSICAL_OBSERVATION_DIM),
        "next_distractors": (length,),
        "rewards": (length,),
        "discounts": (length,),
    }
    dtypes = {
        "behavior_features": jnp.dtype(jnp.float32),
        "world_representations": jnp.dtype(jnp.float32),
        "focal_actions": jnp.dtype(jnp.int32),
        "partner_actions": jnp.dtype(jnp.int32),
        "next_physical_observations": jnp.dtype(jnp.float32),
        "next_distractors": jnp.dtype(jnp.float32),
        "rewards": jnp.dtype(jnp.float32),
        "discounts": jnp.dtype(jnp.float32),
    }
    errors: list[str] = []
    for name, shape in shapes.items():
        value = getattr(segment, name)
        if not isinstance(value, Array):
            errors.append(f"{label} {name} is not a JAX array")
            continue
        if value.shape != shape:
            errors.append(f"{label} {name} shape differs")
        if value.dtype != dtypes[name]:
            errors.append(f"{label} {name} dtype differs")
        if value.dtype == jnp.float32 and not bool(np.all(np.isfinite(np.asarray(value)))):
            errors.append(f"{label} {name} contains non-finite values")
    return tuple(errors)


def validate_typed_surprise_source(source: TypedSurpriseSource) -> tuple[str, ...]:
    """Reconstruct the complete source and reject type, order, or byte drift."""

    if type(source) is not TypedSurpriseSource:
        return ("source type differs",)
    if type(source.config) is not TypedSurpriseFactorizationConfig:
        return ("source config type differs",)
    try:
        canonical_config = _canonical_config_copy(source.config)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return ("source config fields are not canonical",)
    errors = list(
        _segment_contract_errors(
            source.prefix,
            length=source.config.prefix_steps,
            label="prefix",
        )
    )
    if (
        type(source.branches) is not tuple
        or len(source.branches) != len(BRANCH_NAMES)
        or any(type(branch) is not TypedSurpriseBranchSource for branch in source.branches)
    ):
        errors.append("source branches type or cardinality differs")
        return tuple(errors)
    names = tuple(branch.name for branch in source.branches)
    if any(type(name) is not str for name in names) or names != BRANCH_NAMES:
        errors.append("source branch names or order differ")
    for branch in source.branches:
        errors.extend(
            _segment_contract_errors(
                branch.segment,
                length=source.config.continuation_steps,
                label=f"branch {branch.name}",
            )
        )
    if errors:
        return tuple(errors)

    expected = _build_source_unchecked(canonical_config)
    if (
        type(source.generator_contract_sha256) is not str
        or source.generator_contract_sha256 != expected.generator_contract_sha256
    ):
        errors.append("source generator contract digest differs")
    actual_digest = _array_manifest_sha256(
        _source_arrays(source),
        prefix=f"{SOURCE_GENERATOR_VERSION}:{expected.generator_contract_sha256}",
    )
    if type(source.input_sha256) is not str or source.input_sha256 != actual_digest:
        errors.append("source input digest does not match its arrays")
    if source.input_sha256 != expected.input_sha256:
        errors.append("source input digest does not reconstruct")
    for (name, actual), (_, reference) in zip(
        _source_arrays(source),
        _source_arrays(expected),
        strict=True,
    ):
        if not _array_bytes_equal(actual, reference):
            errors.append(f"source array {name} does not reconstruct bit-exactly")
    return tuple(errors)


def _models(
    config: TypedSurpriseFactorizationConfig,
) -> tuple[BehaviorModel, GroundedJointWorldModel]:
    behavior = BehaviorModel(
        BehaviorModelConfig(
            n_actions=N_PARTNER_ACTIONS,
            step_size=config.behavior_step_size,
            diagnostic_decay=float(np.float32(0.95)),
        )
    )
    world = GroundedJointWorldModel(
        GroundedJointWorldModelConfig(
            representation_dim=WORLD_REPRESENTATION_DIM,
            target_observation_dim=TARGET_OBSERVATION_DIM,
            n_focal_actions=N_FOCAL_ACTIONS,
            n_partner_actions=N_PARTNER_ACTIONS,
            step_size=config.world_step_size,
            initialization_scale=config.world_initialization_scale,
        )
    )
    return behavior, world


def _copy_behavior_state(state: BehaviorModelState) -> BehaviorModelState:
    return BehaviorModelState(
        weights=jnp.array(state.weights, copy=True),
        bias=jnp.array(state.bias, copy=True),
        rng_key=jr.wrap_key_data(jnp.array(jr.key_data(state.rng_key), copy=True)),
        step_count=jnp.array(state.step_count, copy=True),
        step_words=jnp.array(state.step_words, copy=True),
        nll_ema=jnp.array(state.nll_ema, copy=True),
        accuracy_ema=jnp.array(state.accuracy_ema, copy=True),
        confidence_ema=jnp.array(state.confidence_ema, copy=True),
    )


def _copy_world_state(state: GroundedJointWorldModelState) -> GroundedJointWorldModelState:
    return GroundedJointWorldModelState(
        weights=jnp.array(state.weights, copy=True),
        bias=jnp.array(state.bias, copy=True),
        update_count=jnp.array(state.update_count, copy=True),
        update_words=jnp.array(state.update_words, copy=True),
    )


def _pre_observation_at(
    segment: TypedSurpriseSegmentSource,
    step: int,
) -> TypedSurprisePreObservation:
    return TypedSurprisePreObservation(
        behavior_features=segment.behavior_features[step],
        world_representation=segment.world_representations[step],
        focal_action=segment.focal_actions[step],
    )


def _pre_action_predictions(
    behavior: BehaviorModel,
    behavior_state: BehaviorModelState,
    world: GroundedJointWorldModel,
    world_state: GroundedJointWorldModelState,
    pre: TypedSurprisePreObservation,
) -> tuple[Array, Array, Array]:
    """Freeze behavior belief, all conditional cells, and the marginal."""

    behavior_probabilities = behavior.predict_probabilities(
        behavior_state,
        pre.behavior_features,
    )
    conditional_cells = jnp.stack(
        tuple(
            world.predict(
                world_state,
                pre.world_representation,
                pre.focal_action,
                jnp.asarray(partner_action, dtype=jnp.int32),
            ).raw_predictions
            for partner_action in range(N_PARTNER_ACTIONS)
        )
    )
    marginal_prediction = behavior_probabilities @ conditional_cells
    return behavior_probabilities, conditional_cells, marginal_prediction


def _feedback_at(
    segment: TypedSurpriseSegmentSource,
    step: int,
) -> TypedSurpriseFeedback:
    next_observation = jnp.concatenate(
        (
            segment.next_physical_observations[step],
            segment.next_distractors[step : step + 1],
        )
    )
    return TypedSurpriseFeedback(
        partner_action=segment.partner_actions[step],
        next_observation=next_observation,
        reward=segment.rewards[step],
        discount=segment.discounts[step],
    )


def _physical_projection(value: Array) -> Array:
    return value[jnp.asarray(PHYSICAL_TARGET_INDICES, dtype=jnp.int32)]


def _trajectory_arrays(
    trajectory: TypedSurpriseTrajectory,
    *,
    prefix: str,
) -> tuple[tuple[str, Array], ...]:
    return tuple(
        (f"{prefix}.{field.name}", getattr(trajectory, field.name))
        for field in dataclasses.fields(TypedSurpriseTrajectory)
    )


def _trajectory_nbytes(trajectory: TypedSurpriseTrajectory) -> int:
    return sum(
        int(np.asarray(getattr(trajectory, field.name)).nbytes)
        for field in dataclasses.fields(TypedSurpriseTrajectory)
    )


def _run_segment(
    behavior: BehaviorModel,
    behavior_state: BehaviorModelState,
    world: GroundedJointWorldModel,
    world_state: GroundedJointWorldModelState,
    segment: TypedSurpriseSegmentSource,
) -> tuple[TypedSurpriseTrajectory, BehaviorModelState, GroundedJointWorldModelState]:
    trace_lists: dict[str, list[Array]] = {
        field.name: [] for field in dataclasses.fields(TypedSurpriseTrajectory)
    }
    length = int(segment.focal_actions.shape[0])
    for step in range(length):
        pre = _pre_observation_at(segment, step)

        # Freeze the pre-action belief, every conditional world cell, and the
        # behavior-weighted marginal before feedback can expose the action.
        behavior_probabilities, conditional_cells, marginal_prediction = (
            _pre_action_predictions(
                behavior,
                behavior_state,
                world,
                world_state,
                pre,
            )
        )

        # Only now reveal the partner action and ordinary outcome.  The
        # realized action selects conditional surprise, never the marginal.
        feedback = _feedback_at(segment, step)
        conditional_prediction = conditional_cells[feedback.partner_action]
        behavior_update = behavior.update(
            behavior_state,
            pre.behavior_features,
            feedback.partner_action,
        )
        world_update = world.update(
            world_state,
            pre.world_representation,
            pre.focal_action,
            feedback.partner_action,
            feedback.next_observation,
            feedback.reward,
            feedback.discount,
        )
        behavior_target = jax.nn.one_hot(
            feedback.partner_action,
            N_PARTNER_ACTIONS,
            dtype=jnp.float32,
        )
        behavior_brier = jnp.sum(jnp.square(behavior_probabilities - behavior_target))
        conditional_physical_error = _physical_projection(conditional_prediction) - (
            _physical_projection(world_update.targets)
        )
        marginal_physical_error = _physical_projection(marginal_prediction) - (
            _physical_projection(world_update.targets)
        )
        distractor_error = (
            conditional_prediction[DISTRACTOR_OBSERVATION_INDEX]
            - world_update.targets[DISTRACTOR_OBSERVATION_INDEX]
        )

        trace_lists["behavior_probabilities_pre"].append(behavior_probabilities)
        trace_lists["behavior_probabilities_update"].append(
            behavior_update.probabilities
        )
        trace_lists["world_predictions_by_partner_pre"].append(conditional_cells)
        trace_lists["conditional_world_predictions_pre"].append(
            conditional_prediction
        )
        trace_lists["conditional_world_predictions_update"].append(
            world_update.prediction.raw_predictions
        )
        trace_lists["marginal_world_predictions_pre"].append(marginal_prediction)
        trace_lists["world_targets"].append(world_update.targets)
        trace_lists["behavior_nll"].append(behavior_update.loss)
        trace_lists["behavior_brier"].append(behavior_brier)
        trace_lists["conditional_physical_world_mse"].append(
            jnp.mean(jnp.square(conditional_physical_error))
        )
        trace_lists["marginal_physical_world_mse"].append(
            jnp.mean(jnp.square(marginal_physical_error))
        )
        trace_lists["distractor_squared_error"].append(
            jnp.square(distractor_error)
        )
        trace_lists["behavior_pre_words"].append(behavior_update.pre_step_words)
        trace_lists["behavior_post_words"].append(behavior_update.post_step_words)
        trace_lists["world_pre_words"].append(world_update.pre_update_words)
        trace_lists["world_post_words"].append(world_update.post_update_words)
        trace_lists["behavior_update_applied"].append(behavior_update.update_applied)
        trace_lists["world_update_applied"].append(world_update.update_applied)
        trace_lists["behavior_prediction_bound"].append(
            jnp.array_equal(behavior_probabilities, behavior_update.probabilities)
        )
        trace_lists["world_prediction_bound"].append(
            jnp.array_equal(
                conditional_prediction,
                world_update.prediction.raw_predictions,
            )
        )
        trace_lists["selected_joint_action_index"].append(
            world_update.prediction.joint_action_index
        )
        trace_lists["world_weight_row_change_mask"].append(
            world_update.proposed_weight_row_bit_change_mask
        )
        trace_lists["world_bias_row_change_mask"].append(
            world_update.proposed_bias_row_bit_change_mask
        )
        behavior_state = behavior_update.state
        world_state = world_update.state

    trajectory = TypedSurpriseTrajectory(
        **{name: jnp.stack(values) for name, values in trace_lists.items()}
    )
    return trajectory, behavior_state, world_state


def _metric_matrix(trajectory: TypedSurpriseTrajectory) -> Array:
    return jnp.stack(
        (
            trajectory.behavior_nll,
            trajectory.behavior_brier,
            trajectory.conditional_physical_world_mse,
            trajectory.marginal_physical_world_mse,
            trajectory.distractor_squared_error,
        ),
        axis=1,
    )


def _behavior_state_arrays(
    state: BehaviorModelState,
    *,
    prefix: str,
) -> tuple[tuple[str, Array], ...]:
    return (
        (f"{prefix}.weights", state.weights),
        (f"{prefix}.bias", state.bias),
        (f"{prefix}.rng_key_words", jr.key_data(state.rng_key)),
        (f"{prefix}.step_count", state.step_count),
        (f"{prefix}.step_words", state.step_words),
        (f"{prefix}.nll_ema", state.nll_ema),
        (f"{prefix}.accuracy_ema", state.accuracy_ema),
        (f"{prefix}.confidence_ema", state.confidence_ema),
    )


def _world_state_arrays(
    state: GroundedJointWorldModelState,
    *,
    prefix: str,
) -> tuple[tuple[str, Array], ...]:
    return (
        (f"{prefix}.weights", state.weights),
        (f"{prefix}.bias", state.bias),
        (f"{prefix}.update_count", state.update_count),
        (f"{prefix}.update_words", state.update_words),
    )


def _common_state_arrays(
    behavior_state: BehaviorModelState,
    world_state: GroundedJointWorldModelState,
) -> tuple[tuple[str, Array], ...]:
    return _behavior_state_arrays(behavior_state, prefix="common.behavior") + (
        _world_state_arrays(world_state, prefix="common.world")
    )


def _branch_state_arrays(
    branches: tuple[
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
    ],
) -> tuple[tuple[str, Array], ...]:
    arrays: list[tuple[str, Array]] = []
    for branch in branches:
        arrays.extend(
            _behavior_state_arrays(
                branch.initial_behavior_state,
                prefix=f"{branch.name}.initial_behavior",
            )
        )
        arrays.extend(
            _world_state_arrays(
                branch.initial_world_state,
                prefix=f"{branch.name}.initial_world",
            )
        )
        arrays.extend(
            _behavior_state_arrays(
                branch.final_behavior_state,
                prefix=f"{branch.name}.final_behavior",
            )
        )
        arrays.extend(
            _world_state_arrays(
                branch.final_world_state,
                prefix=f"{branch.name}.final_world",
            )
        )
    return tuple(arrays)


def _all_trajectory_arrays(
    prefix_trajectory: TypedSurpriseTrajectory,
    branches: tuple[
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
    ],
) -> tuple[tuple[str, Array], ...]:
    arrays = list(_trajectory_arrays(prefix_trajectory, prefix="common_prefix"))
    for branch in branches:
        arrays.extend(_trajectory_arrays(branch.trajectory, prefix=f"branch.{branch.name}"))
    return tuple(arrays)


def _state_total_nbytes(
    behavior_state: BehaviorModelState,
    world_state: GroundedJointWorldModelState,
) -> int:
    return measure_behavior_model_state_nbytes(
        behavior_state
    ) + measure_grounded_joint_world_state_nbytes(world_state)


def _key_bytes_hex(key: Array) -> str:
    return np.ascontiguousarray(np.asarray(jr.key_data(key))).tobytes(order="C").hex()


def _branch_audits(
    branches: tuple[
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
        TypedSurpriseBranchRun,
    ],
    *,
    transitions: int,
) -> tuple[
    TypedSurpriseMatchedBranchAudit,
    TypedSurpriseMatchedBranchAudit,
    TypedSurpriseMatchedBranchAudit,
    TypedSurpriseMatchedBranchAudit,
]:
    return cast(
        tuple[
            TypedSurpriseMatchedBranchAudit,
            TypedSurpriseMatchedBranchAudit,
            TypedSurpriseMatchedBranchAudit,
            TypedSurpriseMatchedBranchAudit,
        ],
        tuple(
            TypedSurpriseMatchedBranchAudit(
                name=branch.name,
                transitions=transitions,
                behavior_pre_action_prediction_api_calls=transitions,
                world_conditional_prediction_api_calls=(
                    transitions * N_PARTNER_ACTIONS
                ),
                behavior_update_api_calls=transitions,
                world_update_api_calls=transitions,
                behavior_rng_draws=0,
                world_rng_draws=0,
                copied_learner_states=2,
                logical_preupdate_work_nbytes=(
                    transitions * LOGICAL_PREUPDATE_WORK_NBYTES_PER_STEP
                ),
                initial_behavior_rng_key_bytes_hex=_key_bytes_hex(
                    branch.initial_behavior_state.rng_key
                ),
                final_behavior_rng_key_bytes_hex=_key_bytes_hex(
                    branch.final_behavior_state.rng_key
                ),
            )
            for branch in branches
        ),
    )


def _execute_unchecked(
    config: TypedSurpriseFactorizationConfig,
) -> TypedSurpriseFactorizationReport:
    source = _build_source_unchecked(config)
    behavior, world = _models(config)
    behavior_key, world_key = jr.split(jr.key(config.development_key))
    initial_behavior_state = behavior.init(BEHAVIOR_FEATURE_DIM, behavior_key)
    initial_world_state = world.init(world_key)
    common_trajectory, learned_behavior_state, learned_world_state = _run_segment(
        behavior,
        initial_behavior_state,
        world,
        initial_world_state,
        source.prefix,
    )
    common_behavior_state = _copy_behavior_state(learned_behavior_state)
    common_world_state = _copy_world_state(learned_world_state)

    branch_runs_list: list[TypedSurpriseBranchRun] = []
    for branch_source in source.branches:
        branch_initial_behavior = _copy_behavior_state(common_behavior_state)
        branch_initial_world = _copy_world_state(common_world_state)
        trajectory, final_behavior, final_world = _run_segment(
            behavior,
            branch_initial_behavior,
            world,
            branch_initial_world,
            branch_source.segment,
        )
        branch_runs_list.append(
            TypedSurpriseBranchRun(
                name=branch_source.name,
                initial_behavior_state=branch_initial_behavior,
                final_behavior_state=final_behavior,
                initial_world_state=branch_initial_world,
                final_world_state=final_world,
                trajectory=trajectory,
            )
        )
    branches = cast(
        tuple[
            TypedSurpriseBranchRun,
            TypedSurpriseBranchRun,
            TypedSurpriseBranchRun,
            TypedSurpriseBranchRun,
        ],
        tuple(branch_runs_list),
    )

    branch_means = jnp.stack(
        tuple(jnp.mean(_metric_matrix(branch.trajectory), axis=0) for branch in branches)
    ).astype(jnp.float32)
    summary = TypedSurpriseSummary(
        metric_names=METRIC_NAMES,
        common_prefix_mean=jnp.mean(_metric_matrix(common_trajectory), axis=0).astype(
            jnp.float32
        ),
        branch_means=branch_means,
        branch_minus_control=(branch_means - branch_means[0]).astype(jnp.float32),
    )

    initial_nbytes = _state_total_nbytes(initial_behavior_state, initial_world_state)
    common_nbytes = _state_total_nbytes(common_behavior_state, common_world_state)
    branch_initial_nbytes = tuple(
        _state_total_nbytes(branch.initial_behavior_state, branch.initial_world_state)
        for branch in branches
    )
    branch_final_nbytes = tuple(
        _state_total_nbytes(branch.final_behavior_state, branch.final_world_state)
        for branch in branches
    )
    branch_trajectory_nbytes = tuple(
        _trajectory_nbytes(branch.trajectory) for branch in branches
    )
    resource = TypedSurpriseResourceSummary(
        initial_total_state_nbytes=initial_nbytes,
        common_prefix_total_state_nbytes=common_nbytes,
        branch_initial_total_state_nbytes=cast(
            tuple[int, int, int, int], branch_initial_nbytes
        ),
        branch_final_total_state_nbytes=cast(tuple[int, int, int, int], branch_final_nbytes),
        fixed_state_nbytes=(
            initial_nbytes == common_nbytes
            and all(value == common_nbytes for value in branch_initial_nbytes)
            and all(value == common_nbytes for value in branch_final_nbytes)
        ),
        logical_preupdate_float32_scalars_per_step=(
            LOGICAL_PREUPDATE_FLOAT32_SCALARS_PER_STEP
        ),
        logical_preupdate_work_nbytes_per_step=(
            LOGICAL_PREUPDATE_WORK_NBYTES_PER_STEP
        ),
        prefix_trajectory_nbytes=_trajectory_nbytes(common_trajectory),
        branch_trajectory_nbytes=cast(
            tuple[int, int, int, int], branch_trajectory_nbytes
        ),
        total_trajectory_nbytes=(
            _trajectory_nbytes(common_trajectory) + sum(branch_trajectory_nbytes)
        ),
        partner_world_cells_evaluated_per_step=N_PARTNER_ACTIONS,
        behavior_updates_per_step=1,
        world_updates_per_step=1,
        replay_capacity=(
            behavior.resource_budget(BEHAVIOR_FEATURE_DIM).replay_capacity
            + world.resource_budget.replay_capacity
        ),
        passes_over_prefix=PASSES_OVER_PREFIX,
        passes_over_each_continuation=PASSES_OVER_EACH_CONTINUATION,
    )
    branch_audits = _branch_audits(
        branches,
        transitions=config.continuation_steps,
    )
    common_state_sha = _array_manifest_sha256(
        _common_state_arrays(common_behavior_state, common_world_state),
        prefix=f"{DEVELOPMENT_SCHEMA}:{source.input_sha256}:common-state",
    )
    branch_state_sha = _array_manifest_sha256(
        _branch_state_arrays(branches),
        prefix=f"{DEVELOPMENT_SCHEMA}:{source.input_sha256}:branch-states",
    )
    trajectory_sha = _array_manifest_sha256(
        _all_trajectory_arrays(common_trajectory, branches),
        prefix=f"{DEVELOPMENT_SCHEMA}:{source.input_sha256}:trajectories",
    )
    return TypedSurpriseFactorizationReport(
        schema=DEVELOPMENT_SCHEMA,
        status="development_only_descriptive_not_assessed",
        development_only=DEVELOPMENT_ONLY,
        assessment_status=ASSESSMENT_STATUS,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        benchmark_execution_authority=BENCHMARK_EXECUTION_AUTHORITY,
        artifact_authority=ARTIFACT_AUTHORITY,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        evidence_claimed=EVIDENCE_CLAIMED,
        thresholds_frozen=THRESHOLDS_FROZEN,
        development_key_frozen=DEVELOPMENT_KEY_FROZEN,
        task_identifiers_exposed=TASK_IDENTIFIERS_EXPOSED,
        resets_exposed=RESETS_EXPOSED,
        learner_reset_count=0,
        descriptive_claims_only=True,
        config=config,
        source=source,
        initial_behavior_state=initial_behavior_state,
        initial_world_state=initial_world_state,
        common_prefix_behavior_state=common_behavior_state,
        common_prefix_world_state=common_world_state,
        common_prefix_trajectory=common_trajectory,
        branches=branches,
        summary=summary,
        resource=resource,
        branch_audits=branch_audits,
        implementation_source_sha256=_implementation_source_sha256(),
        common_prefix_state_sha256=common_state_sha,
        branch_state_sha256=branch_state_sha,
        trajectory_sha256=trajectory_sha,
        limitations=_LIMITATIONS,
    )


def run_typed_surprise_factorization_development(
    config: TypedSurpriseFactorizationConfig | None = None,
) -> TypedSurpriseFactorizationReport:
    """Run the in-memory common prefix and four nonpromoting continuations."""

    supplied = TypedSurpriseFactorizationConfig() if config is None else config
    cfg = _canonical_config_copy(supplied)
    return _execute_unchecked(cfg)


def _tree_exact(left: object, right: object) -> bool:
    left_leaves, left_tree = jax.tree.flatten(left)
    right_leaves, right_tree = jax.tree.flatten(right)
    if cast(object, left_tree) != cast(object, right_tree) or len(left_leaves) != len(
        right_leaves
    ):
        return False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if isinstance(left_leaf, Array) and isinstance(right_leaf, Array):
            left_value = left_leaf
            right_value = right_leaf
            left_is_key = jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                left_value.dtype, jax.dtypes.prng_key
            )
            right_is_key = jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                right_value.dtype, jax.dtypes.prng_key
            )
            if left_is_key != right_is_key:
                return False
            if left_is_key:
                left_value = jr.key_data(left_value)
                right_value = jr.key_data(right_value)
            if not _array_bytes_equal(left_value, right_value):
                return False
        elif type(left_leaf) is not type(right_leaf) or left_leaf != right_leaf:
            return False
    return True


def _trajectory_contract_errors(
    trajectory: object,
    *,
    length: int,
    label: str,
) -> tuple[str, ...]:
    if type(trajectory) is not TypedSurpriseTrajectory:
        return (f"{label} trajectory type differs",)
    shapes = {
        "behavior_probabilities_pre": (length, N_PARTNER_ACTIONS),
        "behavior_probabilities_update": (length, N_PARTNER_ACTIONS),
        "world_predictions_by_partner_pre": (length, N_PARTNER_ACTIONS, TARGET_DIM),
        "conditional_world_predictions_pre": (length, TARGET_DIM),
        "conditional_world_predictions_update": (length, TARGET_DIM),
        "marginal_world_predictions_pre": (length, TARGET_DIM),
        "world_targets": (length, TARGET_DIM),
        "behavior_nll": (length,),
        "behavior_brier": (length,),
        "conditional_physical_world_mse": (length,),
        "marginal_physical_world_mse": (length,),
        "distractor_squared_error": (length,),
        "behavior_pre_words": (length, 2),
        "behavior_post_words": (length, 2),
        "world_pre_words": (length, 2),
        "world_post_words": (length, 2),
        "behavior_update_applied": (length,),
        "world_update_applied": (length,),
        "behavior_prediction_bound": (length,),
        "world_prediction_bound": (length,),
        "selected_joint_action_index": (length,),
        "world_weight_row_change_mask": (length, N_FOCAL_ACTIONS * N_PARTNER_ACTIONS),
        "world_bias_row_change_mask": (length, N_FOCAL_ACTIONS * N_PARTNER_ACTIONS),
    }
    float_fields = {
        "behavior_probabilities_pre",
        "behavior_probabilities_update",
        "world_predictions_by_partner_pre",
        "conditional_world_predictions_pre",
        "conditional_world_predictions_update",
        "marginal_world_predictions_pre",
        "world_targets",
        "behavior_nll",
        "behavior_brier",
        "conditional_physical_world_mse",
        "marginal_physical_world_mse",
        "distractor_squared_error",
    }
    uint_fields = {
        "behavior_pre_words",
        "behavior_post_words",
        "world_pre_words",
        "world_post_words",
    }
    bool_fields = {
        "behavior_update_applied",
        "world_update_applied",
        "behavior_prediction_bound",
        "world_prediction_bound",
        "world_weight_row_change_mask",
        "world_bias_row_change_mask",
    }
    errors: list[str] = []
    for name, shape in shapes.items():
        value = getattr(trajectory, name)
        if not isinstance(value, Array):
            errors.append(f"{label} trajectory {name} is not a JAX array")
            continue
        expected_dtype = (
            jnp.dtype(jnp.float32)
            if name in float_fields
            else jnp.dtype(jnp.uint32)
            if name in uint_fields
            else jnp.dtype(jnp.bool_)
            if name in bool_fields
            else jnp.dtype(jnp.int32)
        )
        if value.shape != shape:
            errors.append(f"{label} trajectory {name} shape differs")
        if value.dtype != expected_dtype:
            errors.append(f"{label} trajectory {name} dtype differs")
        if name in float_fields and not bool(np.all(np.isfinite(np.asarray(value)))):
            errors.append(f"{label} trajectory {name} contains non-finite values")
    return tuple(errors)


def _strict_scalar_tree_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        return all(
            _strict_scalar_tree_equal(
                getattr(left, field.name),
                getattr(right, field.name),
            )
            for field in dataclasses.fields(left)
        )
    if type(left) is tuple:
        right_tuple = cast(tuple[object, ...], right)
        return len(left) == len(right_tuple) and all(
            _strict_scalar_tree_equal(a, b)
            for a, b in zip(left, right_tuple, strict=True)
        )
    return bool(left == right)


def _state_is_finite(state: BehaviorModelState | GroundedJointWorldModelState) -> bool:
    for leaf in jax.tree.leaves(state):
        if isinstance(leaf, Array) and jnp.issubdtype(leaf.dtype, jnp.floating):
            if not bool(np.all(np.isfinite(np.asarray(leaf)))):
                return False
    return True


def validate_typed_surprise_factorization_report(
    report: TypedSurpriseFactorizationReport,
) -> tuple[str, ...]:
    """Validate deterministic raw execution, resources, and non-authority."""

    if type(report) is not TypedSurpriseFactorizationReport:
        return ("report type differs",)
    errors: list[str] = []
    flag_contract = (
        type(report.schema) is str and report.schema == DEVELOPMENT_SCHEMA,
        type(report.status) is str
        and report.status == "development_only_descriptive_not_assessed",
        report.development_only is True,
        type(report.assessment_status) is str
        and report.assessment_status == ASSESSMENT_STATUS,
        report.scientific_promotion_allowed is False,
        report.benchmark_execution_authority is False,
        report.artifact_authority is False,
        report.output_writes_allowed is False,
        report.evidence_claimed is False,
        report.thresholds_frozen is False,
        report.development_key_frozen is False,
        report.task_identifiers_exposed is False,
        report.resets_exposed is False,
        type(report.learner_reset_count) is int and report.learner_reset_count == 0,
        report.descriptive_claims_only is True,
    )
    if not all(flag_contract):
        errors.append("development-only non-authority contract changed")
    if (
        type(report.limitations) is not tuple
        or any(type(item) is not str for item in report.limitations)
        or report.limitations != _LIMITATIONS
    ):
        errors.append("report limitations differ")
    config_valid = type(report.config) is TypedSurpriseFactorizationConfig
    if not config_valid:
        errors.append("report config type differs")
        canonical_report_config = None
    else:
        try:
            canonical_report_config = _canonical_config_copy(report.config)
        except (AttributeError, TypeError, ValueError, OverflowError):
            errors.append("report config fields are not canonical")
            canonical_report_config = None
            config_valid = False
    source_valid = type(report.source) is TypedSurpriseSource
    if not source_valid:
        errors.append("report source type differs")
    elif type(report.source.config) is not TypedSurpriseFactorizationConfig:
        errors.append("report source config type differs")
        source_valid = False
    elif not config_valid or not _configs_exact(report.config, report.source.config):
        errors.append("report config is not exactly bound to source config")
    if source_valid:
        errors.extend(validate_typed_surprise_source(report.source))
    if not config_valid or not source_valid:
        return tuple(errors)

    state_contracts = (
        (report.initial_behavior_state, BehaviorModelState, "initial behavior state"),
        (report.initial_world_state, GroundedJointWorldModelState, "initial world state"),
        (
            report.common_prefix_behavior_state,
            BehaviorModelState,
            "common-prefix behavior state",
        ),
        (
            report.common_prefix_world_state,
            GroundedJointWorldModelState,
            "common-prefix world state",
        ),
    )
    nested_valid = True
    for value, expected_type, label in state_contracts:
        if type(value) is not expected_type:
            errors.append(f"{label} type differs")
            nested_valid = False
    prefix_errors = _trajectory_contract_errors(
        report.common_prefix_trajectory,
        length=report.config.prefix_steps,
        label="common-prefix",
    )
    errors.extend(prefix_errors)
    nested_valid = nested_valid and not prefix_errors
    if (
        type(report.branches) is not tuple
        or len(report.branches) != len(BRANCH_NAMES)
        or any(type(branch) is not TypedSurpriseBranchRun for branch in report.branches)
    ):
        errors.append("report branches type or cardinality differs")
        nested_valid = False
    else:
        names = tuple(branch.name for branch in report.branches)
        if any(type(name) is not str for name in names) or names != BRANCH_NAMES:
            errors.append("report branch names or order differ")
            nested_valid = False
        for branch in report.branches:
            for value, expected_type, label in (
                (
                    branch.initial_behavior_state,
                    BehaviorModelState,
                    f"branch {branch.name} initial behavior state",
                ),
                (
                    branch.final_behavior_state,
                    BehaviorModelState,
                    f"branch {branch.name} final behavior state",
                ),
                (
                    branch.initial_world_state,
                    GroundedJointWorldModelState,
                    f"branch {branch.name} initial world state",
                ),
                (
                    branch.final_world_state,
                    GroundedJointWorldModelState,
                    f"branch {branch.name} final world state",
                ),
            ):
                if type(value) is not expected_type:
                    errors.append(f"{label} type differs")
                    nested_valid = False
            branch_errors = _trajectory_contract_errors(
                branch.trajectory,
                length=report.config.continuation_steps,
                label=f"branch {branch.name}",
            )
            errors.extend(branch_errors)
            nested_valid = nested_valid and not branch_errors
    if type(report.summary) is not TypedSurpriseSummary:
        errors.append("report summary type differs")
        nested_valid = False
    else:
        if (
            type(report.summary.metric_names) is not tuple
            or report.summary.metric_names != METRIC_NAMES
        ):
            errors.append("summary metric names differ")
            nested_valid = False
        for name, shape in (
            ("common_prefix_mean", (len(METRIC_NAMES),)),
            ("branch_means", (len(BRANCH_NAMES), len(METRIC_NAMES))),
            (
                "branch_minus_control",
                (len(BRANCH_NAMES), len(METRIC_NAMES)),
            ),
        ):
            value = getattr(report.summary, name)
            if (
                not isinstance(value, Array)
                or value.shape != shape
                or value.dtype != jnp.float32
            ):
                errors.append(f"summary {name} array contract differs")
                nested_valid = False
            elif not bool(np.all(np.isfinite(np.asarray(value)))):
                errors.append(f"summary {name} contains non-finite values")
    if type(report.resource) is not TypedSurpriseResourceSummary:
        errors.append("report resource type differs")
        nested_valid = False
    if (
        type(report.branch_audits) is not tuple
        or len(report.branch_audits) != len(BRANCH_NAMES)
        or any(
            type(item) is not TypedSurpriseMatchedBranchAudit
            for item in report.branch_audits
        )
    ):
        errors.append("report branch audits type or cardinality differs")
        nested_valid = False
    elif (
        any(type(item.name) is not str for item in report.branch_audits)
        or tuple(item.name for item in report.branch_audits) != BRANCH_NAMES
    ):
        errors.append("report branch audit names or order differ")
        nested_valid = False
    if not nested_valid:
        return tuple(errors)

    assert canonical_report_config is not None
    expected = _execute_unchecked(canonical_report_config)
    trajectory_pairs = (
        (report.common_prefix_trajectory, expected.common_prefix_trajectory, "common-prefix"),
    ) + tuple(
        (actual.trajectory, reference.trajectory, f"branch {actual.name}")
        for actual, reference in zip(report.branches, expected.branches, strict=True)
    )
    for actual, reference, label in trajectory_pairs:
        for (name, actual_array), (_, reference_array) in zip(
            _trajectory_arrays(actual, prefix=label),
            _trajectory_arrays(reference, prefix=label),
            strict=True,
        ):
            if not _array_bytes_equal(actual_array, reference_array):
                errors.append(f"{name} differs from deterministic execution")

    top_state_pairs = (
        (report.initial_behavior_state, expected.initial_behavior_state, "initial behavior"),
        (report.initial_world_state, expected.initial_world_state, "initial world"),
        (
            report.common_prefix_behavior_state,
            expected.common_prefix_behavior_state,
            "common-prefix behavior",
        ),
        (
            report.common_prefix_world_state,
            expected.common_prefix_world_state,
            "common-prefix world",
        ),
    )
    for actual_state, reference_state, label in top_state_pairs:
        if not _tree_exact(actual_state, reference_state):
            errors.append(f"{label} state differs from deterministic execution")
        if not _state_is_finite(actual_state):
            errors.append(f"{label} state contains non-finite values")
    for actual_branch, expected_branch in zip(
        report.branches, expected.branches, strict=True
    ):
        branch_state_pairs = (
            (
                actual_branch.initial_behavior_state,
                expected_branch.initial_behavior_state,
                "initial behavior",
            ),
            (
                actual_branch.final_behavior_state,
                expected_branch.final_behavior_state,
                "final behavior",
            ),
            (
                actual_branch.initial_world_state,
                expected_branch.initial_world_state,
                "initial world",
            ),
            (
                actual_branch.final_world_state,
                expected_branch.final_world_state,
                "final world",
            ),
        )
        for actual_state, reference_state, label in branch_state_pairs:
            if not _tree_exact(actual_state, reference_state):
                errors.append(
                    f"branch {actual_branch.name} {label} state differs from "
                    "deterministic execution"
                )
            if not _state_is_finite(actual_state):
                errors.append(f"branch {actual_branch.name} {label} state is non-finite")
        if not _tree_exact(
            actual_branch.initial_behavior_state,
            report.common_prefix_behavior_state,
        ) or not _tree_exact(
            actual_branch.initial_world_state,
            report.common_prefix_world_state,
        ):
            errors.append(f"branch {actual_branch.name} does not start from copied prefix state")

    for name in ("common_prefix_mean", "branch_means", "branch_minus_control"):
        if not _array_bytes_equal(
            getattr(report.summary, name), getattr(expected.summary, name)
        ):
            errors.append(f"summary {name} differs from deterministic execution")
    if not _strict_scalar_tree_equal(report.resource, expected.resource):
        errors.append("resource accounting differs")
    if not _strict_scalar_tree_equal(report.branch_audits, expected.branch_audits):
        errors.append("matched branch call, work, or RNG audit differs")
    elif any(
        audit.initial_behavior_rng_key_bytes_hex
        != audit.final_behavior_rng_key_bytes_hex
        for audit in report.branch_audits
    ):
        errors.append("a branch advanced behavior RNG despite a zero-draw receipt")

    implementation_sha = _implementation_source_sha256()
    if (
        type(report.implementation_source_sha256) is not str
        or report.implementation_source_sha256 != implementation_sha
        or report.implementation_source_sha256
        != expected.implementation_source_sha256
    ):
        errors.append("implementation source digest differs")

    expected_common_sha = _array_manifest_sha256(
        _common_state_arrays(
            report.common_prefix_behavior_state,
            report.common_prefix_world_state,
        ),
        prefix=f"{DEVELOPMENT_SCHEMA}:{report.source.input_sha256}:common-state",
    )
    if (
        type(report.common_prefix_state_sha256) is not str
        or report.common_prefix_state_sha256 != expected_common_sha
        or report.common_prefix_state_sha256 != expected.common_prefix_state_sha256
    ):
        errors.append("common-prefix state digest differs")
    expected_branch_sha = _array_manifest_sha256(
        _branch_state_arrays(report.branches),
        prefix=f"{DEVELOPMENT_SCHEMA}:{report.source.input_sha256}:branch-states",
    )
    if (
        type(report.branch_state_sha256) is not str
        or report.branch_state_sha256 != expected_branch_sha
        or report.branch_state_sha256 != expected.branch_state_sha256
    ):
        errors.append("branch state digest differs")
    expected_trajectory_sha = _array_manifest_sha256(
        _all_trajectory_arrays(report.common_prefix_trajectory, report.branches),
        prefix=f"{DEVELOPMENT_SCHEMA}:{report.source.input_sha256}:trajectories",
    )
    if (
        type(report.trajectory_sha256) is not str
        or report.trajectory_sha256 != expected_trajectory_sha
        or report.trajectory_sha256 != expected.trajectory_sha256
    ):
        errors.append("trajectory digest differs")
    return tuple(errors)


__all__ = [
    "ARTIFACT_AUTHORITY",
    "ASSESSMENT_STATUS",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "BRANCH_NAMES",
    "DEVELOPMENT_KEY_FROZEN",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "DISTRACTOR_OBSERVATION_INDEX",
    "EVIDENCE_CLAIMED",
    "LOGICAL_PREUPDATE_FLOAT32_SCALARS_PER_STEP",
    "LOGICAL_PREUPDATE_WORK_NBYTES_PER_STEP",
    "METRIC_NAMES",
    "OUTPUT_WRITES_ALLOWED",
    "PHYSICAL_TARGET_INDICES",
    "RESETS_EXPOSED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TASK_IDENTIFIERS_EXPOSED",
    "THRESHOLDS_FROZEN",
    "TypedSurpriseBranchRun",
    "TypedSurpriseBranchSource",
    "TypedSurpriseFactorizationConfig",
    "TypedSurpriseFactorizationReport",
    "TypedSurpriseFeedback",
    "TypedSurpriseMatchedBranchAudit",
    "TypedSurprisePreObservation",
    "TypedSurpriseResourceSummary",
    "TypedSurpriseSegmentSource",
    "TypedSurpriseSource",
    "TypedSurpriseSummary",
    "TypedSurpriseTrajectory",
    "build_typed_surprise_source",
    "run_typed_surprise_factorization_development",
    "validate_typed_surprise_factorization_report",
    "validate_typed_surprise_source",
]
