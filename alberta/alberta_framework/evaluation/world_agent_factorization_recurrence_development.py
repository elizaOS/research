"""Development-only online world/agent factorization recurrence probe.

This module feeds one uninterrupted deterministic ``A -> B -> A`` hidden-
partner life to two existing, causally separate learners:

* :class:`BehaviorModel` predicts the partner action from the ordinary public
  cue before that action is revealed;
* :class:`GroundedJointWorldModel` predicts fixed next-observation, reward, and
  continuation targets conditional on the evaluator-fixed focal action and the
  subsequently observed partner action.

The hidden partner mapping changes in the middle phase and recurs without a
learner-visible phase id, reset, or task label.  The physical target law does
not change.  Every transition is consumed exactly once and neither learner
owns replay storage.  Before the action is revealed, the evaluator also forms
an action-marginal from the complete entry-state conditional table under the
behavior-model belief; the revealed action cannot enter that mixture.  Its
error against the realized target is reported separately from conditional
world error because it includes partner-action uncertainty.

This is an in-memory L0 mechanism scaffold.  It has no output writer,
threshold, verdict, benchmark authority, artifact authority, evidence claim,
or scientific-promotion path.  Metrics are descriptive and ``not_assessed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable
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

DEVELOPMENT_SCHEMA: Final = "alberta.world-agent-factorization-recurrence.development.v1"
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

PHASE_NAMES: Final = ("A_initial", "B_interference", "A_recurrence")
SOURCE_GENERATOR_VERSION: Final = "deterministic-hidden-partner-aba-v1"
N_PARTNER_ACTIONS: Final = 2
N_FOCAL_ACTIONS: Final = 2
BEHAVIOR_FEATURE_DIM: Final = 2
WORLD_REPRESENTATION_DIM: Final = 2
TARGET_OBSERVATION_DIM: Final = 2
TARGET_DIM: Final = TARGET_OBSERVATION_DIM + 2
PASSES_OVER_SOURCE: Final = 1

_INT32_MAX: Final = 2**31 - 1
_LIMITATIONS: Final = (
    "one deterministic analytic life is not a population or robustness result",
    "the hidden partner uses one binary public cue and a scripted recurring mapping",
    "the grounded world is linear and the physical target law does not change",
    "the marginal error is against a realized target and includes partner-action uncertainty",
    "logical work bytes exclude compiler buffers, allocator peaks, and update temporaries",
    "descriptive recurrence deltas have no acceptance thresholds",
    "no control policy, visual encoder, multi-step planning, or scale claim is tested",
)


@dataclasses.dataclass(frozen=True, slots=True)
class WorldAgentFactorizationRecurrenceConfig:
    """Small deterministic development construction with no evidence role."""

    phase_steps: int = 32
    summary_window: int = 8
    behavior_step_size: float = 0.20
    world_step_size: float = 0.20
    world_initialization_scale: float = 0.01
    development_key: int = 0

    def __post_init__(self) -> None:
        for name in ("phase_steps", "summary_window", "development_key"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.phase_steps < 4 or self.phase_steps % 4 != 0:
            raise ValueError("phase_steps must be a positive multiple of four")
        if not 1 <= self.summary_window <= self.phase_steps:
            raise ValueError("summary_window must lie in [1, phase_steps]")
        if self.total_steps >= _INT32_MAX:
            raise ValueError("the complete life must fit the exact model counter contract")
        if not 0 <= self.development_key <= 2**32 - 1:
            raise ValueError("development_key must fit one uint32 word")
        for name in (
            "behavior_step_size",
            "world_step_size",
            "world_initialization_scale",
        ):
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be an exact built-in float")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value_float32 = np.float32(value)
            if not np.isfinite(value_float32) or value_float32 <= np.float32(0.0):
                raise ValueError(f"{name} must remain finite and positive in float32")
        if self.world_step_size > 1.0:
            raise ValueError("world_step_size must not exceed one")

    @property
    def total_steps(self) -> int:
        """Return the uninterrupted A/B/A transition count."""

        return 3 * self.phase_steps


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationPreObservation:
    """Complete learner input available before the partner acts."""

    behavior_features: Array
    world_representation: Array
    focal_action: Array


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationFeedback:
    """Ordinary feedback revealed only after behavior prediction."""

    partner_action: Array
    next_observation: Array
    reward: Array
    discount: Array


@dataclasses.dataclass(frozen=True, slots=True)
class WorldAgentFactorizationSource:
    """Evaluator-owned deterministic source; phase ids never enter a learner call."""

    config: WorldAgentFactorizationRecurrenceConfig
    behavior_features: Array
    world_representations: Array
    focal_actions: Array
    partner_actions: Array
    next_observations: Array
    rewards: Array
    discounts: Array
    evaluator_phase_ids: Array
    generator_contract_sha256: str
    input_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationMetrics:
    """Four independently constructed prequential metric channels."""

    behavior_nll: float
    behavior_brier: float
    conditional_world_mse: float
    marginal_world_mse: float


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationPhaseSummary:
    """Descriptive mean over one evaluator-owned phase."""

    name: str
    start: int
    stop: int
    metrics: FactorizationMetrics


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationRecurrenceSummary:
    """Windowed A-recurrence accounting with no threshold or verdict."""

    initial_a_reference: FactorizationMetrics
    recurrence_entry: FactorizationMetrics
    recurrence_late: FactorizationMetrics
    entry_forgetting: FactorizationMetrics
    within_recurrence_recovery: FactorizationMetrics
    residual_forgetting: FactorizationMetrics


@dataclasses.dataclass(frozen=True, slots=True)
class FactorizationTrajectory:
    """Raw predict-before-update measurements from the single pass."""

    behavior_probabilities_pre: Array
    behavior_probabilities_update: Array
    world_predictions_by_partner_pre: Array
    conditional_world_predictions_pre: Array
    conditional_world_predictions_update: Array
    marginal_world_predictions_pre: Array
    world_targets: Array
    behavior_nll: Array
    behavior_brier: Array
    conditional_world_mse: Array
    marginal_world_mse: Array
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
class FactorizationResourceSummary:
    """Exact state/output bytes and a narrowly defined logical work projection.

    ``logical_preupdate_work_nbytes_per_step`` counts only the float32 values
    named by the evaluator before model commits: one partner simplex, every
    partner-conditioned raw world prediction, one realized target vector, and
    one marginal prediction vector.  It deliberately does not claim physical
    allocator, compiler-buffer, or arithmetic-operation accounting.
    """

    behavior_initial_state_nbytes: int
    behavior_final_state_nbytes: int
    world_initial_state_nbytes: int
    world_final_state_nbytes: int
    initial_total_state_nbytes: int
    final_total_state_nbytes: int
    fixed_state_nbytes: bool
    logical_preupdate_float32_scalars_per_step: int
    logical_preupdate_work_nbytes_per_step: int
    trajectory_nbytes: int
    partner_world_cells_evaluated_per_step: int
    behavior_updates_per_step: int
    world_updates_per_step: int
    replay_capacity: int
    passes_over_source: int


@dataclasses.dataclass(frozen=True, slots=True)
class WorldAgentFactorizationRecurrenceReport:
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
    config: WorldAgentFactorizationRecurrenceConfig
    source: WorldAgentFactorizationSource
    initial_behavior_state: BehaviorModelState
    final_behavior_state: BehaviorModelState
    initial_world_state: GroundedJointWorldModelState
    final_world_state: GroundedJointWorldModelState
    trajectory: FactorizationTrajectory
    phase_summaries: tuple[
        FactorizationPhaseSummary,
        FactorizationPhaseSummary,
        FactorizationPhaseSummary,
    ]
    recurrence: FactorizationRecurrenceSummary
    resource: FactorizationResourceSummary
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


def _array_manifest_sha256(
    arrays: Iterable[tuple[str, Array | np.ndarray]],
    *,
    prefix: str,
) -> str:
    digest = hashlib.sha256(prefix.encode("ascii"))
    for name, value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"name": name, "shape": list(array.shape), "dtype": array.dtype.str},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_bytes_equal(left: object, right: object) -> bool:
    """Compare array shape, dtype, and payload bytes without numeric coercion."""

    left_array = np.ascontiguousarray(np.asarray(left))
    right_array = np.ascontiguousarray(np.asarray(right))
    return (
        left_array.shape == right_array.shape
        and left_array.dtype == right_array.dtype
        and left_array.tobytes(order="C") == right_array.tobytes(order="C")
    )


def _source_contract(config: WorldAgentFactorizationRecurrenceConfig) -> dict[str, object]:
    return {
        "version": SOURCE_GENERATOR_VERSION,
        "phase_names": list(PHASE_NAMES),
        "phase_steps": config.phase_steps,
        "summary_window": config.summary_window,
        "behavior_feature_dim": BEHAVIOR_FEATURE_DIM,
        "world_representation_dim": WORLD_REPRESENTATION_DIM,
        "target_observation_dim": TARGET_OBSERVATION_DIM,
        "n_focal_actions": N_FOCAL_ACTIONS,
        "n_partner_actions": N_PARTNER_ACTIONS,
        "public_signal": "alternating-binary-one-hot",
        "focal_action": "evaluator-fixed-two-step-block-alternation",
        "partner_mapping": "A-signal;B-inverse-signal;A-signal",
        "world_target": "phase-invariant-fixed-joint-action-conditioned-affine-law",
        "discount": 0.95,
        "passes_over_source": PASSES_OVER_SOURCE,
        "learner_phase_identifiers": False,
        "learner_resets": False,
    }


def _source_arrays(
    source: WorldAgentFactorizationSource,
) -> tuple[tuple[str, Array], ...]:
    return (
        ("behavior_features", source.behavior_features),
        ("world_representations", source.world_representations),
        ("focal_actions", source.focal_actions),
        ("partner_actions", source.partner_actions),
        ("next_observations", source.next_observations),
        ("rewards", source.rewards),
        ("discounts", source.discounts),
        ("evaluator_phase_ids", source.evaluator_phase_ids),
    )


def build_world_agent_factorization_source(
    config: WorldAgentFactorizationRecurrenceConfig | None = None,
) -> WorldAgentFactorizationSource:
    """Construct the exact deterministic single-pass A/B/A source."""

    cfg = config or WorldAgentFactorizationRecurrenceConfig()
    steps = np.arange(cfg.total_steps, dtype=np.int64)
    phase_ids = (steps // cfg.phase_steps).astype(np.int32)
    public_signal = (steps % 2).astype(np.int32)
    behavior_features = np.eye(BEHAVIOR_FEATURE_DIM, dtype=np.float32)[public_signal]
    world_representations = behavior_features.copy()
    focal_actions = ((steps // 2) % N_FOCAL_ACTIONS).astype(np.int32)
    a_mapping = public_signal
    b_mapping = 1 - public_signal
    partner_actions = np.where(phase_ids == 1, b_mapping, a_mapping).astype(np.int32)

    next_signal_sign = (2.0 * ((steps + 1) % 2) - 1.0).astype(np.float32)
    focal_sign = (2.0 * focal_actions - 1.0).astype(np.float32)
    partner_sign = (2.0 * partner_actions - 1.0).astype(np.float32)
    interaction_coordinate = (0.5 * focal_sign + 0.25 * partner_sign).astype(np.float32)
    next_observations = np.stack(
        (next_signal_sign, interaction_coordinate),
        axis=1,
    ).astype(np.float32)
    rewards = (focal_actions == partner_actions).astype(np.float32)
    discounts = np.full((cfg.total_steps,), np.float32(0.95), dtype=np.float32)

    contract_digest = _canonical_json_sha256(_source_contract(cfg))
    bare = WorldAgentFactorizationSource(
        config=cfg,
        behavior_features=jnp.asarray(behavior_features, dtype=jnp.float32),
        world_representations=jnp.asarray(world_representations, dtype=jnp.float32),
        focal_actions=jnp.asarray(focal_actions, dtype=jnp.int32),
        partner_actions=jnp.asarray(partner_actions, dtype=jnp.int32),
        next_observations=jnp.asarray(next_observations, dtype=jnp.float32),
        rewards=jnp.asarray(rewards, dtype=jnp.float32),
        discounts=jnp.asarray(discounts, dtype=jnp.float32),
        evaluator_phase_ids=jnp.asarray(phase_ids, dtype=jnp.int32),
        generator_contract_sha256=contract_digest,
        input_sha256="",
    )
    return dataclasses.replace(
        bare,
        input_sha256=_array_manifest_sha256(
            _source_arrays(bare),
            prefix=f"{SOURCE_GENERATOR_VERSION}:{contract_digest}",
        ),
    )


def validate_world_agent_factorization_source(
    source: WorldAgentFactorizationSource,
) -> tuple[str, ...]:
    """Reconstruct the deterministic source and reject shape/content drift."""

    errors: list[str] = []
    if type(source) is not WorldAgentFactorizationSource:
        return ("source type differs",)
    if type(source.config) is not WorldAgentFactorizationRecurrenceConfig:
        return ("source config type differs",)
    expected = build_world_agent_factorization_source(source.config)
    if source.generator_contract_sha256 != expected.generator_contract_sha256:
        errors.append("source generator contract digest differs")
    actual_digest = _array_manifest_sha256(
        _source_arrays(source),
        prefix=f"{SOURCE_GENERATOR_VERSION}:{source.generator_contract_sha256}",
    )
    if source.input_sha256 != actual_digest:
        errors.append("source input digest does not match its arrays")
    if source.input_sha256 != expected.input_sha256:
        errors.append("source input digest does not reconstruct")
    for (name, actual), (_, reference) in zip(
        _source_arrays(source),
        _source_arrays(expected),
        strict=True,
    ):
        actual_host = np.asarray(actual)
        reference_host = np.asarray(reference)
        if actual_host.shape != reference_host.shape:
            errors.append(f"{name} shape differs")
        elif actual_host.dtype != reference_host.dtype:
            errors.append(f"{name} dtype differs")
        elif not _array_bytes_equal(actual_host, reference_host):
            errors.append(f"{name} does not reconstruct bit-exactly")
    return tuple(errors)


def _pre_observation_at(
    source: WorldAgentFactorizationSource,
    step: int,
) -> FactorizationPreObservation:
    return FactorizationPreObservation(
        behavior_features=source.behavior_features[step],
        world_representation=source.world_representations[step],
        focal_action=source.focal_actions[step],
    )


def _pre_action_predictions(
    behavior: BehaviorModel,
    behavior_state: BehaviorModelState,
    world: GroundedJointWorldModel,
    world_state: GroundedJointWorldModelState,
    pre: FactorizationPreObservation,
) -> tuple[Array, Array, Array]:
    """Freeze behavior belief, all world cells, and their marginal before feedback."""

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
    source: WorldAgentFactorizationSource,
    step: int,
) -> FactorizationFeedback:
    return FactorizationFeedback(
        partner_action=source.partner_actions[step],
        next_observation=source.next_observations[step],
        reward=source.rewards[step],
        discount=source.discounts[step],
    )


def _trajectory_arrays(
    trajectory: FactorizationTrajectory,
) -> tuple[tuple[str, Array], ...]:
    return tuple(
        (field.name, getattr(trajectory, field.name))
        for field in dataclasses.fields(trajectory)
    )


def _trajectory_nbytes(trajectory: FactorizationTrajectory) -> int:
    return sum(int(value.nbytes) for _, value in _trajectory_arrays(trajectory))


def _mean_metrics(
    trajectory: FactorizationTrajectory,
    start: int,
    stop: int,
) -> FactorizationMetrics:
    return FactorizationMetrics(
        behavior_nll=float(jnp.mean(trajectory.behavior_nll[start:stop])),
        behavior_brier=float(jnp.mean(trajectory.behavior_brier[start:stop])),
        conditional_world_mse=float(
            jnp.mean(trajectory.conditional_world_mse[start:stop])
        ),
        marginal_world_mse=float(jnp.mean(trajectory.marginal_world_mse[start:stop])),
    )


def _metrics_difference(
    left: FactorizationMetrics,
    right: FactorizationMetrics,
) -> FactorizationMetrics:
    return FactorizationMetrics(
        behavior_nll=left.behavior_nll - right.behavior_nll,
        behavior_brier=left.behavior_brier - right.behavior_brier,
        conditional_world_mse=left.conditional_world_mse - right.conditional_world_mse,
        marginal_world_mse=left.marginal_world_mse - right.marginal_world_mse,
    )


def _summaries(
    config: WorldAgentFactorizationRecurrenceConfig,
    trajectory: FactorizationTrajectory,
) -> tuple[
    tuple[
        FactorizationPhaseSummary,
        FactorizationPhaseSummary,
        FactorizationPhaseSummary,
    ],
    FactorizationRecurrenceSummary,
]:
    phase_summaries = tuple(
        FactorizationPhaseSummary(
            name=name,
            start=index * config.phase_steps,
            stop=(index + 1) * config.phase_steps,
            metrics=_mean_metrics(
                trajectory,
                index * config.phase_steps,
                (index + 1) * config.phase_steps,
            ),
        )
        for index, name in enumerate(PHASE_NAMES)
    )
    initial_reference = _mean_metrics(
        trajectory,
        config.phase_steps - config.summary_window,
        config.phase_steps,
    )
    recurrence_start = 2 * config.phase_steps
    recurrence_entry = _mean_metrics(
        trajectory,
        recurrence_start,
        recurrence_start + config.summary_window,
    )
    recurrence_late = _mean_metrics(
        trajectory,
        config.total_steps - config.summary_window,
        config.total_steps,
    )
    recurrence = FactorizationRecurrenceSummary(
        initial_a_reference=initial_reference,
        recurrence_entry=recurrence_entry,
        recurrence_late=recurrence_late,
        entry_forgetting=_metrics_difference(recurrence_entry, initial_reference),
        within_recurrence_recovery=_metrics_difference(recurrence_entry, recurrence_late),
        residual_forgetting=_metrics_difference(recurrence_late, initial_reference),
    )
    return (
        (
            phase_summaries[0],
            phase_summaries[1],
            phase_summaries[2],
        ),
        recurrence,
    )


def run_world_agent_factorization_recurrence_development(
    config: WorldAgentFactorizationRecurrenceConfig | None = None,
) -> WorldAgentFactorizationRecurrenceReport:
    """Run one in-memory, single-pass, predict-before-update development life."""

    cfg = config or WorldAgentFactorizationRecurrenceConfig()
    source = build_world_agent_factorization_source(cfg)
    source_errors = validate_world_agent_factorization_source(source)
    if source_errors:
        raise RuntimeError(f"source reconstruction failed: {source_errors!r}")

    behavior = BehaviorModel(
        BehaviorModelConfig(
            n_actions=N_PARTNER_ACTIONS,
            step_size=cfg.behavior_step_size,
            diagnostic_decay=0.95,
        )
    )
    world = GroundedJointWorldModel(
        GroundedJointWorldModelConfig(
            representation_dim=WORLD_REPRESENTATION_DIM,
            target_observation_dim=TARGET_OBSERVATION_DIM,
            n_focal_actions=N_FOCAL_ACTIONS,
            n_partner_actions=N_PARTNER_ACTIONS,
            step_size=cfg.world_step_size,
            initialization_scale=cfg.world_initialization_scale,
        )
    )
    behavior_key, world_key = jr.split(jr.key(cfg.development_key))
    initial_behavior_state = behavior.init(BEHAVIOR_FEATURE_DIM, behavior_key)
    initial_world_state = world.init(world_key)
    behavior_state = initial_behavior_state
    world_state = initial_world_state

    trace_lists: dict[str, list[Array]] = {
        field.name: [] for field in dataclasses.fields(FactorizationTrajectory)
    }
    for step in range(cfg.total_steps):
        pre = _pre_observation_at(source, step)

        # Causal boundary 1: freeze every pre-action prediction before any
        # feedback object can expose the realized partner action.
        behavior_probabilities, conditional_cells, marginal_prediction = (
            _pre_action_predictions(
                behavior,
                behavior_state,
                world,
                world_state,
                pre,
            )
        )

        # Causal boundary 2: now reveal/select/score the actual joint action,
        # then select both learner successors.
        feedback = _feedback_at(source, step)
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
        brier = jnp.sum(jnp.square(behavior_probabilities - behavior_target))
        marginal_error = marginal_prediction - world_update.targets

        trace_lists["behavior_probabilities_pre"].append(behavior_probabilities)
        trace_lists["behavior_probabilities_update"].append(behavior_update.probabilities)
        trace_lists["world_predictions_by_partner_pre"].append(conditional_cells)
        trace_lists["conditional_world_predictions_pre"].append(conditional_prediction)
        trace_lists["conditional_world_predictions_update"].append(
            world_update.prediction.raw_predictions
        )
        trace_lists["marginal_world_predictions_pre"].append(marginal_prediction)
        trace_lists["world_targets"].append(world_update.targets)
        trace_lists["behavior_nll"].append(behavior_update.loss)
        trace_lists["behavior_brier"].append(brier)
        trace_lists["conditional_world_mse"].append(jnp.mean(jnp.square(world_update.errors)))
        trace_lists["marginal_world_mse"].append(jnp.mean(jnp.square(marginal_error)))
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

    trajectory = FactorizationTrajectory(
        **{
            name: jnp.stack(values)
            for name, values in trace_lists.items()
        }
    )
    phase_summaries, recurrence = _summaries(cfg, trajectory)
    behavior_initial_nbytes = measure_behavior_model_state_nbytes(initial_behavior_state)
    behavior_final_nbytes = measure_behavior_model_state_nbytes(behavior_state)
    world_initial_nbytes = measure_grounded_joint_world_state_nbytes(initial_world_state)
    world_final_nbytes = measure_grounded_joint_world_state_nbytes(world_state)
    logical_float32_scalars = (
        N_PARTNER_ACTIONS
        + N_PARTNER_ACTIONS * TARGET_DIM
        + TARGET_DIM
        + TARGET_DIM
    )
    resource = FactorizationResourceSummary(
        behavior_initial_state_nbytes=behavior_initial_nbytes,
        behavior_final_state_nbytes=behavior_final_nbytes,
        world_initial_state_nbytes=world_initial_nbytes,
        world_final_state_nbytes=world_final_nbytes,
        initial_total_state_nbytes=behavior_initial_nbytes + world_initial_nbytes,
        final_total_state_nbytes=behavior_final_nbytes + world_final_nbytes,
        fixed_state_nbytes=(
            behavior_initial_nbytes == behavior_final_nbytes
            and world_initial_nbytes == world_final_nbytes
        ),
        logical_preupdate_float32_scalars_per_step=logical_float32_scalars,
        logical_preupdate_work_nbytes_per_step=4 * logical_float32_scalars,
        trajectory_nbytes=_trajectory_nbytes(trajectory),
        partner_world_cells_evaluated_per_step=N_PARTNER_ACTIONS,
        behavior_updates_per_step=1,
        world_updates_per_step=1,
        replay_capacity=(
            behavior.resource_budget(BEHAVIOR_FEATURE_DIM).replay_capacity
            + world.resource_budget.replay_capacity
        ),
        passes_over_source=PASSES_OVER_SOURCE,
    )
    trajectory_sha256 = _array_manifest_sha256(
        _trajectory_arrays(trajectory),
        prefix=f"{DEVELOPMENT_SCHEMA}:{source.input_sha256}",
    )
    return WorldAgentFactorizationRecurrenceReport(
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
        config=cfg,
        source=source,
        initial_behavior_state=initial_behavior_state,
        final_behavior_state=behavior_state,
        initial_world_state=initial_world_state,
        final_world_state=world_state,
        trajectory=trajectory,
        phase_summaries=phase_summaries,
        recurrence=recurrence,
        resource=resource,
        trajectory_sha256=trajectory_sha256,
        limitations=_LIMITATIONS,
    )


def _tree_exact(left: object, right: object) -> bool:
    """Compare deterministic learner pytrees without numeric tolerance."""

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
                left_value.dtype,
                jax.dtypes.prng_key,
            )
            right_is_key = jax.dtypes.issubdtype(  # type: ignore[attr-defined]
                right_value.dtype,
                jax.dtypes.prng_key,
            )
            if left_is_key != right_is_key:
                return False
            if left_is_key:
                left_value = jr.key_data(left_value)
                right_value = jr.key_data(right_value)
            left_host = np.asarray(left_value)
            right_host = np.asarray(right_value)
            if (
                left_host.shape != right_host.shape
                or left_host.dtype != right_host.dtype
                or not _array_bytes_equal(left_host, right_host)
            ):
                return False
        elif type(left_leaf) is not type(right_leaf) or left_leaf != right_leaf:
            return False
    return True


def _metrics_equal(
    left: FactorizationMetrics,
    right: FactorizationMetrics,
) -> bool:
    return (
        type(left) is FactorizationMetrics
        and type(right) is FactorizationMetrics
        and all(
            getattr(left, field.name) == getattr(right, field.name)
            for field in dataclasses.fields(FactorizationMetrics)
        )
    )


def validate_world_agent_factorization_recurrence_report(
    report: WorldAgentFactorizationRecurrenceReport,
) -> tuple[str, ...]:
    """Validate structural, causal, resource, and non-authority contracts only."""

    if type(report) is not WorldAgentFactorizationRecurrenceReport:
        return ("report type differs",)
    source_errors = validate_world_agent_factorization_source(report.source)
    errors = list(source_errors)
    expected_flags = (
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
    if not all(expected_flags):
        errors.append("development-only non-authority contract changed")
    report_config_valid = type(report.config) is WorldAgentFactorizationRecurrenceConfig
    if not report_config_valid:
        errors.append("report config type differs")
    source_type_valid = type(report.source) is WorldAgentFactorizationSource
    source_config_valid = (
        source_type_valid
        and type(report.source.config) is WorldAgentFactorizationRecurrenceConfig
    )
    configs_exactly_bound = (
        report_config_valid
        and source_config_valid
        and report.config == report.source.config
    )
    if not configs_exactly_bound:
        errors.append("report config is not exactly bound to the source config")
    if type(report.limitations) is not tuple or report.limitations != _LIMITATIONS:
        errors.append("report limitations differ")
    if source_errors or not report_config_valid or not source_config_valid:
        return tuple(errors)

    nested_contract_valid = True
    exact_nested_types = (
        (report.trajectory, FactorizationTrajectory, "report trajectory type differs"),
        (report.resource, FactorizationResourceSummary, "report resource type differs"),
        (
            report.recurrence,
            FactorizationRecurrenceSummary,
            "report recurrence type differs",
        ),
        (
            report.initial_behavior_state,
            BehaviorModelState,
            "initial behavior state type differs",
        ),
        (
            report.final_behavior_state,
            BehaviorModelState,
            "final behavior state type differs",
        ),
        (
            report.initial_world_state,
            GroundedJointWorldModelState,
            "initial world state type differs",
        ),
        (
            report.final_world_state,
            GroundedJointWorldModelState,
            "final world state type differs",
        ),
    )
    for value, expected_type, message in exact_nested_types:
        if type(value) is not expected_type:
            errors.append(message)
            nested_contract_valid = False
    if (
        type(report.phase_summaries) is not tuple
        or len(report.phase_summaries) != len(PHASE_NAMES)
        or any(type(item) is not FactorizationPhaseSummary for item in report.phase_summaries)
    ):
        errors.append("report phase summaries type differs")
        nested_contract_valid = False
    if not nested_contract_valid:
        return tuple(errors)

    total = report.config.total_steps
    expected_shapes = {
        "behavior_probabilities_pre": (total, N_PARTNER_ACTIONS),
        "behavior_probabilities_update": (total, N_PARTNER_ACTIONS),
        "world_predictions_by_partner_pre": (total, N_PARTNER_ACTIONS, TARGET_DIM),
        "conditional_world_predictions_pre": (total, TARGET_DIM),
        "conditional_world_predictions_update": (total, TARGET_DIM),
        "marginal_world_predictions_pre": (total, TARGET_DIM),
        "world_targets": (total, TARGET_DIM),
        "behavior_nll": (total,),
        "behavior_brier": (total,),
        "conditional_world_mse": (total,),
        "marginal_world_mse": (total,),
        "behavior_pre_words": (total, 2),
        "behavior_post_words": (total, 2),
        "world_pre_words": (total, 2),
        "world_post_words": (total, 2),
        "behavior_update_applied": (total,),
        "world_update_applied": (total,),
        "behavior_prediction_bound": (total,),
        "world_prediction_bound": (total,),
        "selected_joint_action_index": (total,),
        "world_weight_row_change_mask": (total, N_FOCAL_ACTIONS * N_PARTNER_ACTIONS),
        "world_bias_row_change_mask": (total, N_FOCAL_ACTIONS * N_PARTNER_ACTIONS),
    }
    expected_dtypes = {
        "behavior_probabilities_pre": jnp.dtype(jnp.float32),
        "behavior_probabilities_update": jnp.dtype(jnp.float32),
        "world_predictions_by_partner_pre": jnp.dtype(jnp.float32),
        "conditional_world_predictions_pre": jnp.dtype(jnp.float32),
        "conditional_world_predictions_update": jnp.dtype(jnp.float32),
        "marginal_world_predictions_pre": jnp.dtype(jnp.float32),
        "world_targets": jnp.dtype(jnp.float32),
        "behavior_nll": jnp.dtype(jnp.float32),
        "behavior_brier": jnp.dtype(jnp.float32),
        "conditional_world_mse": jnp.dtype(jnp.float32),
        "marginal_world_mse": jnp.dtype(jnp.float32),
        "behavior_pre_words": jnp.dtype(jnp.uint32),
        "behavior_post_words": jnp.dtype(jnp.uint32),
        "world_pre_words": jnp.dtype(jnp.uint32),
        "world_post_words": jnp.dtype(jnp.uint32),
        "behavior_update_applied": jnp.dtype(jnp.bool_),
        "world_update_applied": jnp.dtype(jnp.bool_),
        "behavior_prediction_bound": jnp.dtype(jnp.bool_),
        "world_prediction_bound": jnp.dtype(jnp.bool_),
        "selected_joint_action_index": jnp.dtype(jnp.int32),
        "world_weight_row_change_mask": jnp.dtype(jnp.bool_),
        "world_bias_row_change_mask": jnp.dtype(jnp.bool_),
    }
    trajectory_contract_valid = True
    for name, expected_shape in expected_shapes.items():
        value = getattr(report.trajectory, name)
        if value.shape != expected_shape:
            errors.append(f"trajectory {name} shape differs")
            trajectory_contract_valid = False
        if value.dtype != expected_dtypes[name]:
            errors.append(f"trajectory {name} dtype differs")
            trajectory_contract_valid = False
    if not trajectory_contract_valid:
        return tuple(errors)

    trajectory = report.trajectory
    # The public runner does not call this report validator.  Replaying it here
    # is therefore nonrecursive and binds every raw output and learner state to
    # the same deterministic kernels that produced the submitted report.
    expected_report = run_world_agent_factorization_recurrence_development(report.config)
    for (name, actual_array), (_, expected_array) in zip(
        _trajectory_arrays(trajectory),
        _trajectory_arrays(expected_report.trajectory),
        strict=True,
    ):
        if not _array_bytes_equal(actual_array, expected_array):
            errors.append(f"trajectory {name} differs from deterministic execution")
    state_contracts = (
        (
            report.initial_behavior_state,
            expected_report.initial_behavior_state,
            "initial behavior state",
        ),
        (
            report.final_behavior_state,
            expected_report.final_behavior_state,
            "final behavior state",
        ),
        (
            report.initial_world_state,
            expected_report.initial_world_state,
            "initial world state",
        ),
        (
            report.final_world_state,
            expected_report.final_world_state,
            "final world state",
        ),
    )
    for actual_state, expected_state, label in state_contracts:
        if not _tree_exact(actual_state, expected_state):
            errors.append(f"{label} differs from deterministic execution")

    behavior_pre = np.asarray(trajectory.behavior_probabilities_pre)
    behavior_update = np.asarray(trajectory.behavior_probabilities_update)
    cells = np.asarray(trajectory.world_predictions_by_partner_pre)
    partner_actions = np.asarray(report.source.partner_actions)
    rows = np.arange(total)
    conditional = cells[rows, partner_actions]
    targets = np.asarray(trajectory.world_targets)
    marginal = np.asarray(
        jnp.stack(
            tuple(
                trajectory.behavior_probabilities_pre[step]
                @ trajectory.world_predictions_by_partner_pre[step]
                for step in range(total)
            )
        )
    )
    exact_numeric_contracts = (
        (behavior_pre, behavior_update, "behavior prediction/update binding"),
        (
            conditional,
            np.asarray(trajectory.conditional_world_predictions_pre),
            "selected conditional prediction binding",
        ),
        (
            conditional,
            np.asarray(trajectory.conditional_world_predictions_update),
            "world prediction/update binding",
        ),
        (
            marginal,
            np.asarray(trajectory.marginal_world_predictions_pre),
            "marginal prediction construction",
        ),
    )
    for expected_numeric, actual_numeric, label in exact_numeric_contracts:
        if not np.array_equal(expected_numeric, actual_numeric):
            errors.append(f"{label} differs")
    if not np.all(np.isfinite(behavior_pre)) or not np.all(np.isfinite(behavior_update)):
        errors.append("behavior probabilities are non-finite")
    if not np.all(np.isfinite(np.asarray(trajectory.behavior_nll))):
        errors.append("behavior metrics are non-finite")
    if not np.all(np.isfinite(np.asarray(trajectory.behavior_brier))):
        errors.append("behavior Brier is non-finite")
    if not np.all(np.isfinite(cells)):
        errors.append("world conditional cells are non-finite")
    if not np.all(np.isfinite(targets)):
        errors.append("world targets are non-finite")
    if not np.all(np.isfinite(np.asarray(trajectory.conditional_world_mse))):
        errors.append("world metrics are non-finite")
    if not np.all(np.isfinite(np.asarray(trajectory.marginal_world_mse))):
        errors.append("marginal world MSE is non-finite")

    expected_pre_words = np.stack(
        (
            np.zeros((total,), dtype=np.uint32),
            np.arange(total, dtype=np.uint32),
        ),
        axis=1,
    )
    expected_post_words = expected_pre_words.copy()
    expected_post_words[:, 1] += np.uint32(1)
    for name in ("behavior_pre_words", "world_pre_words"):
        if not np.array_equal(np.asarray(getattr(trajectory, name)), expected_pre_words):
            errors.append(f"{name} does not show one uninterrupted clock")
    for name in ("behavior_post_words", "world_post_words"):
        if not np.array_equal(np.asarray(getattr(trajectory, name)), expected_post_words):
            errors.append(f"{name} does not show one uninterrupted clock")
    for name in (
        "behavior_update_applied",
        "world_update_applied",
        "behavior_prediction_bound",
        "world_prediction_bound",
    ):
        if not bool(np.all(np.asarray(getattr(trajectory, name)))):
            errors.append(f"{name} contains a failed causal event")

    selected = np.asarray(trajectory.selected_joint_action_index)
    expected_selected = (
        np.asarray(report.source.focal_actions) * N_PARTNER_ACTIONS + partner_actions
    )
    if not np.array_equal(selected, expected_selected):
        errors.append("selected world row does not match the observed joint action")
    selected_mask = np.eye(N_FOCAL_ACTIONS * N_PARTNER_ACTIONS, dtype=np.bool_)[selected]
    for name in ("world_weight_row_change_mask", "world_bias_row_change_mask"):
        changed = np.asarray(getattr(trajectory, name))
        if bool(np.any(changed & ~selected_mask)):
            errors.append(f"{name} changed a nonexecuted joint-action row")

    resource = report.resource
    expected_behavior_initial = measure_behavior_model_state_nbytes(
        report.initial_behavior_state
    )
    expected_behavior_final = measure_behavior_model_state_nbytes(report.final_behavior_state)
    expected_world_initial = measure_grounded_joint_world_state_nbytes(report.initial_world_state)
    expected_world_final = measure_grounded_joint_world_state_nbytes(report.final_world_state)
    if (
        resource.behavior_initial_state_nbytes != expected_behavior_initial
        or resource.behavior_final_state_nbytes != expected_behavior_final
        or resource.world_initial_state_nbytes != expected_world_initial
        or resource.world_final_state_nbytes != expected_world_final
        or resource.initial_total_state_nbytes
        != expected_behavior_initial + expected_world_initial
        or resource.final_total_state_nbytes != expected_behavior_final + expected_world_final
        or resource.fixed_state_nbytes is not True
    ):
        errors.append("persistent state byte accounting differs")
    expected_logical_scalars = (
        N_PARTNER_ACTIONS
        + N_PARTNER_ACTIONS * TARGET_DIM
        + TARGET_DIM
        + TARGET_DIM
    )
    if (
        resource.logical_preupdate_float32_scalars_per_step != expected_logical_scalars
        or resource.logical_preupdate_work_nbytes_per_step != 4 * expected_logical_scalars
        or resource.trajectory_nbytes != _trajectory_nbytes(trajectory)
        or resource.partner_world_cells_evaluated_per_step != N_PARTNER_ACTIONS
        or resource.behavior_updates_per_step != 1
        or resource.world_updates_per_step != 1
        or resource.replay_capacity != 0
        or resource.passes_over_source != PASSES_OVER_SOURCE
    ):
        errors.append("fixed logical work or output byte accounting differs")

    expected_phases, expected_recurrence = _summaries(report.config, trajectory)
    if tuple(summary.name for summary in report.phase_summaries) != PHASE_NAMES:
        errors.append("phase summary names differ")
    for phase_summary, expected_phase in zip(
        report.phase_summaries,
        expected_phases,
        strict=True,
    ):
        if (
            phase_summary.start != expected_phase.start
            or phase_summary.stop != expected_phase.stop
            or not _metrics_equal(phase_summary.metrics, expected_phase.metrics)
        ):
            errors.append(f"phase summary {phase_summary.name} differs")
    for field in dataclasses.fields(FactorizationRecurrenceSummary):
        if not _metrics_equal(
            getattr(report.recurrence, field.name),
            getattr(expected_recurrence, field.name),
        ):
            errors.append(f"recurrence summary {field.name} differs")

    expected_trajectory_digest = _array_manifest_sha256(
        _trajectory_arrays(trajectory),
        prefix=f"{DEVELOPMENT_SCHEMA}:{report.source.input_sha256}",
    )
    if report.trajectory_sha256 != expected_trajectory_digest:
        errors.append("trajectory digest differs")
    return tuple(errors)


__all__ = [
    "ARTIFACT_AUTHORITY",
    "ASSESSMENT_STATUS",
    "BENCHMARK_EXECUTION_AUTHORITY",
    "DEVELOPMENT_KEY_FROZEN",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "EVIDENCE_CLAIMED",
    "FactorizationFeedback",
    "FactorizationMetrics",
    "FactorizationPhaseSummary",
    "FactorizationPreObservation",
    "FactorizationRecurrenceSummary",
    "FactorizationResourceSummary",
    "FactorizationTrajectory",
    "OUTPUT_WRITES_ALLOWED",
    "PHASE_NAMES",
    "RESETS_EXPOSED",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "TASK_IDENTIFIERS_EXPOSED",
    "THRESHOLDS_FROZEN",
    "WorldAgentFactorizationRecurrenceConfig",
    "WorldAgentFactorizationRecurrenceReport",
    "WorldAgentFactorizationSource",
    "build_world_agent_factorization_source",
    "run_world_agent_factorization_recurrence_development",
    "validate_world_agent_factorization_recurrence_report",
    "validate_world_agent_factorization_source",
]
