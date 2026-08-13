"""Development-only recurring latent-world-model instrumentation.

This module runs one uninterrupted deterministic ``A -> B -> A`` vector-
dynamics stream through three matched :class:`LatentWorldModel` arms:

* a fixed encoder;
* a prediction-trained encoder whose collapse gate is disabled; and
* the same prediction-trained encoder with the existing collapse gate active.

The learner receives only observation, action, reward, discount, and the next
observation.  Phase labels and resets are never presented.  The evaluator
records raw prequential errors, descriptive recurrence deltas, fixed persistent
resource measurements, encoder update/gate rates, SIGReg diagnostics, and
matched physical-versus-nuisance perturbation surprise.  It deliberately has
no artifact writer, threshold, aggregate verdict, seed-search surface, or
scientific-promotion path.  Every outcome remains descriptive and
``not_assessed``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Final, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.latent_world_model import (
    LatentWorldModel,
    LatentWorldModelConfig,
    LatentWorldModelState,
    measure_latent_world_model_state_nbytes,
)
from alberta_framework.core.sigreg import (
    SIGRegConfig,
    sample_sigreg_directions,
    sigreg_diagnostics,
)

DEVELOPMENT_SCHEMA: Final = "alberta.latent-world-model-recurrence.development.v1"
DEVELOPMENT_ONLY: Final = True
ASSESSMENT_STATUS: Final = "not_assessed"
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
THRESHOLDS_FROZEN: Final = False
DEVELOPMENT_KEYS_FROZEN: Final = False
EVIDENCE_CLAIMED: Final = False
TASK_IDENTIFIERS_EXPOSED: Final = False
RESETS_EXPOSED: Final = False

FIXED_ENCODER: Final = "fixed_encoder"
TRAINABLE_PREDICTION_ONLY: Final = "trainable_prediction_only"
TRAINABLE_COLLAPSE_GATED: Final = "trainable_collapse_gated"
ArmName = Literal[
    "fixed_encoder",
    "trainable_prediction_only",
    "trainable_collapse_gated",
]
ARM_ORDER: Final[tuple[ArmName, ArmName, ArmName]] = (
    FIXED_ENCODER,
    TRAINABLE_PREDICTION_ONLY,
    TRAINABLE_COLLAPSE_GATED,
)

PHASE_NAMES: Final = ("A_initial", "B_interference", "A_recurrence")
SOURCE_GENERATOR_VERSION: Final = "recurring-vector-dynamics-v2"
OBSERVATION_DIM: Final = 4
N_ACTIONS: Final = 2
PHYSICAL_DIM: Final = 2
NUISANCE_DIM: Final = 2

_A_BASE_ROTATION: Final = 0.38
_B_BASE_ROTATION: Final = -0.57
_ACTION_ROTATION_OFFSETS: Final = (-0.09, 0.09)
_ACTION_REWARD_BIASES: Final = (-0.08, 0.08)
_A_REWARD_VECTOR: Final = (0.70, -0.30)
_B_REWARD_VECTOR: Final = (-0.40, 0.60)
_A_DISCOUNT: Final = 0.97
_B_DISCOUNT: Final = 0.84
_INITIAL_PHYSICAL_STATE: Final = (0.65, -0.20)
_STYLE_SCALE: Final = 0.35
_STYLE_FREQUENCIES: Final = (0.19, 0.13)
_PERTURBATION_FREQUENCY: Final = 0.37

_LIMITATIONS: Final = (
    "one deterministic analytic stream is not a population or robustness claim",
    "recurrence deltas are descriptive window means without acceptance thresholds",
    "persistent-state equality does not imply equal encoder-gradient compute",
    "two legacy learner timing scalars may materialize under scan; resources "
    "are compared arm-wise at matched points",
    "SIGReg is diagnostic only and is not optimized by any arm",
    "physical-versus-nuisance surprise depends on one random-feature initialization",
    "the low-dimensional unit-circle stream does not establish image, control, or scale efficacy",
)


@dataclasses.dataclass(frozen=True, slots=True)
class RecurringVectorDynamicsProbeConfig:
    """Small deterministic development protocol; no field is an evidence gate."""

    phase_steps: int = 64
    summary_window: int = 16
    latent_dim: int = 6
    hidden_sizes: tuple[int, ...] = (16,)
    predictor_step_size: float = 0.03
    encoder_step_size: float = 0.01
    max_encoder_update: float = 0.05
    min_latent_std: float = 0.05
    collapse_gate_threshold: float = 0.25
    perturbation_scale: float = 0.20
    sigreg_projections: int = 16
    development_key: int = 0

    def __post_init__(self) -> None:
        integer_fields = {
            "phase_steps": self.phase_steps,
            "summary_window": self.summary_window,
            "latent_dim": self.latent_dim,
            "sigreg_projections": self.sigreg_projections,
            "development_key": self.development_key,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.phase_steps < 2:
            raise ValueError("phase_steps must be at least two")
        if not 1 <= self.summary_window <= self.phase_steps:
            raise ValueError("summary_window must be in [1, phase_steps]")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.sigreg_projections <= 0:
            raise ValueError("sigreg_projections must be positive")
        if not 0 <= self.development_key <= 2**32 - 1:
            raise ValueError("development_key must fit one uint32 word")
        if not isinstance(self.hidden_sizes, tuple) or any(
            type(width) is not int or width <= 0 for width in self.hidden_sizes
        ):
            raise ValueError("hidden_sizes must be a tuple of positive integers")
        positive_floats: dict[str, float] = {
            "predictor_step_size": self.predictor_step_size,
            "encoder_step_size": self.encoder_step_size,
            "max_encoder_update": self.max_encoder_update,
            "min_latent_std": self.min_latent_std,
            "perturbation_scale": self.perturbation_scale,
        }
        for float_name, float_value in positive_floats.items():
            if isinstance(float_value, bool) or not isinstance(float_value, (int, float)):
                raise TypeError(f"{float_name} must be a finite real number")
            if not math.isfinite(float(float_value)) or float(float_value) <= 0.0:
                raise ValueError(f"{float_name} must be finite and positive")
        threshold = self.collapse_gate_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("collapse_gate_threshold must be a finite real number")
        if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) < 1.0:
            raise ValueError("collapse_gate_threshold must be finite and in [0, 1)")

    @property
    def total_steps(self) -> int:
        """Total uninterrupted transition count."""

        return 3 * self.phase_steps


@dataclasses.dataclass(frozen=True, slots=True)
class RecurringVectorDynamicsSource:
    """Exact source arrays shared by every arm."""

    config: RecurringVectorDynamicsProbeConfig
    observations: Array
    actions: Array
    rewards: Array
    discounts: Array
    next_observations: Array
    physical_violation_next_observations: Array
    nuisance_perturbation_next_observations: Array
    perturbation_l2: Array
    generator_contract_sha256: str
    input_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ErrorSummary:
    """Mean prequential squared errors for one interval."""

    latent: float
    reward: float
    discount: float
    joint: float


@dataclasses.dataclass(frozen=True, slots=True)
class SIGRegPhaseSummary:
    """Host-readable SIGReg diagnostics for one phase embedding sequence."""

    loss: float
    latent_mean_abs: float
    latent_std_mean: float
    latent_std_min: float
    projected_mean_abs: float
    projected_std_mean: float


@dataclasses.dataclass(frozen=True, slots=True)
class SurpriseSeparationSummary:
    """Descriptive matched-perturbation surprise means."""

    nominal: float
    physical_violation: float
    nuisance_perturbation: float
    physical_excess: float
    nuisance_excess: float
    physical_minus_nuisance: float
    perturbation_l2: float


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseSummary:
    """Descriptive metrics for one externally indexed phase."""

    name: str
    start: int
    stop: int
    prequential_error: ErrorSummary
    sigreg: SIGRegPhaseSummary
    surprise: SurpriseSeparationSummary


@dataclasses.dataclass(frozen=True, slots=True)
class RecurrenceSummary:
    """Windowed A-recurrence accounting without a threshold or verdict.

    ``entry_forgetting`` is A-recurrence entry minus the late initial-A
    reference. ``within_recurrence_recovery`` is recurrence entry minus late
    recurrence. ``residual_forgetting`` is late recurrence minus late initial A.
    Positive and negative values are both retained as observations.
    """

    initial_a_reference: ErrorSummary
    recurrence_entry: ErrorSummary
    recurrence_late: ErrorSummary
    entry_forgetting: ErrorSummary
    within_recurrence_recovery: ErrorSummary
    residual_forgetting: ErrorSummary


@dataclasses.dataclass(frozen=True, slots=True)
class PersistentResourceSummary:
    """Persistent array resources only; runtime work is intentionally excluded."""

    initial_state_nbytes: int
    final_state_nbytes: int
    initial_array_elements: int
    final_array_elements: int
    predictor_input_dim: int
    prediction_heads: int


@dataclasses.dataclass(frozen=True, slots=True)
class PrequentialTrajectory:
    """Raw online measurements, all computed before each transition update."""

    latent_errors: Array
    reward_errors: Array
    discount_errors: Array
    joint_errors: Array
    target_next_embeddings: Array
    collapse_scores: Array
    latent_std_means: Array
    nominal_surprises: Array
    physical_violation_surprises: Array
    nuisance_perturbation_surprises: Array
    encoder_updates: Array
    encoder_gates: Array
    world_updates: Array


@dataclasses.dataclass(frozen=True, slots=True)
class ArmReport:
    """One matched arm's raw and descriptive development measurements."""

    name: ArmName
    encoder_learning: bool
    collapse_gate_threshold: float
    source_input_sha256: str
    initial_state_sha256: str
    sigreg_directions_sha256: str
    nonintervention_config_sha256: str
    resource: PersistentResourceSummary
    trajectory: PrequentialTrajectory
    phase_summaries: tuple[PhaseSummary, ...]
    recurrence: RecurrenceSummary
    overall_surprise: SurpriseSeparationSummary
    world_update_rate: float
    encoder_update_rate: float
    encoder_gate_rate: float


@dataclasses.dataclass(frozen=True, slots=True)
class CommonRandomnessBinding:
    """Digests showing that every arm consumed one source/state/direction draw."""

    source_input_sha256: str
    initial_state_sha256: str
    sigreg_directions_sha256: str
    generator_contract_sha256: str
    source_reconstruction_valid: bool
    common_randomness_preserved: bool


@dataclasses.dataclass(frozen=True, slots=True)
class LatentWorldModelRecurrenceDevelopmentReport:
    """In-memory development report; it is not an evidence artifact."""

    schema: str
    status: str
    development_only: bool
    assessment_status: str
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    thresholds_frozen: bool
    development_keys_frozen: bool
    evidence_claimed: bool
    task_identifiers_exposed: bool
    resets_exposed: bool
    descriptive_claims_only: bool
    config: RecurringVectorDynamicsProbeConfig
    source: RecurringVectorDynamicsSource
    binding: CommonRandomnessBinding
    arms: tuple[ArmReport, ...]
    fixed_persistent_resources_equal: bool
    matched_nonintervention_config: bool
    limitations: tuple[str, ...]


def _canonical_json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_manifest_sha256(
    arrays: Iterable[tuple[str, Array | np.ndarray]],
    *,
    prefix: str,
) -> str:
    digest = hashlib.sha256(prefix.encode("utf-8"))
    for name, value in arrays:
        array = np.ascontiguousarray(np.asarray(value))
        header = json.dumps(
            {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _source_contract_payload(
    config: RecurringVectorDynamicsProbeConfig,
) -> dict[str, object]:
    return {
        "version": SOURCE_GENERATOR_VERSION,
        "phase_steps": config.phase_steps,
        "observation_layout": ["physical_0", "physical_1", "style_0", "style_1"],
        "a_base_rotation_radians": _A_BASE_ROTATION,
        "b_base_rotation_radians": _B_BASE_ROTATION,
        "action_rotation_offsets_radians": _ACTION_ROTATION_OFFSETS,
        "action_reward_biases": _ACTION_REWARD_BIASES,
        "a_reward_vector": _A_REWARD_VECTOR,
        "b_reward_vector": _B_REWARD_VECTOR,
        "a_discount": _A_DISCOUNT,
        "b_discount": _B_DISCOUNT,
        "initial_physical_state": _INITIAL_PHYSICAL_STATE,
        "style_scale": _STYLE_SCALE,
        "style_frequencies": _STYLE_FREQUENCIES,
        "perturbation_scale": config.perturbation_scale,
        "perturbation_frequency": _PERTURBATION_FREQUENCY,
        "phase_order": PHASE_NAMES,
        "task_identifiers_exposed": False,
        "resets_exposed": False,
    }


def _source_arrays(source: RecurringVectorDynamicsSource) -> tuple[tuple[str, Array], ...]:
    return (
        ("observations", source.observations),
        ("actions", source.actions),
        ("rewards", source.rewards),
        ("discounts", source.discounts),
        ("next_observations", source.next_observations),
        (
            "physical_violation_next_observations",
            source.physical_violation_next_observations,
        ),
        (
            "nuisance_perturbation_next_observations",
            source.nuisance_perturbation_next_observations,
        ),
        ("perturbation_l2", source.perturbation_l2),
    )


def _style(step: int) -> np.ndarray:
    return np.asarray(
        (
            _STYLE_SCALE * math.sin(_STYLE_FREQUENCIES[0] * step + 0.1),
            _STYLE_SCALE * math.cos(_STYLE_FREQUENCIES[1] * step - 0.2),
        ),
        dtype=np.float32,
    )


def build_recurring_vector_dynamics_source(
    config: RecurringVectorDynamicsProbeConfig | None = None,
) -> RecurringVectorDynamicsSource:
    """Construct the exact deterministic A/B/A source shared by every arm."""

    cfg = config or RecurringVectorDynamicsProbeConfig()
    total = cfg.total_steps
    observations = np.empty((total, OBSERVATION_DIM), dtype=np.float32)
    actions = np.empty((total,), dtype=np.int32)
    rewards = np.empty((total,), dtype=np.float32)
    discounts = np.empty((total,), dtype=np.float32)
    next_observations = np.empty_like(observations)
    physical_violations = np.empty_like(observations)
    nuisance_perturbations = np.empty_like(observations)
    perturbation_l2 = np.empty((total,), dtype=np.float32)

    physical = np.asarray(_INITIAL_PHYSICAL_STATE, dtype=np.float32)
    physical /= np.linalg.norm(physical)
    current = np.concatenate((physical, _style(0))).astype(np.float32)
    for step in range(total):
        phase = step // cfg.phase_steps
        in_a = phase != 1
        reward_vector = np.asarray(
            _A_REWARD_VECTOR if in_a else _B_REWARD_VECTOR,
            dtype=np.float32,
        )
        action = step % N_ACTIONS
        angle = (
            (_A_BASE_ROTATION if in_a else _B_BASE_ROTATION)
            + _ACTION_ROTATION_OFFSETS[action]
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)),
            dtype=np.float32,
        )
        next_physical = (rotation @ physical).astype(np.float32)
        next_observation = np.concatenate((next_physical, _style(step + 1))).astype(np.float32)
        reward = reward_vector @ physical + np.float32(_ACTION_REWARD_BIASES[action])

        observations[step] = current
        actions[step] = action
        rewards[step] = np.float32(reward)
        discounts[step] = np.float32(_A_DISCOUNT if in_a else _B_DISCOUNT)
        next_observations[step] = next_observation

        angle = _PERTURBATION_FREQUENCY * (step + 1)
        perturbation = np.asarray(
            (
                cfg.perturbation_scale * math.cos(angle),
                cfg.perturbation_scale * math.sin(angle),
            ),
            dtype=np.float32,
        )
        physical_probe = next_observation.copy()
        nuisance_probe = next_observation.copy()
        physical_probe[:PHYSICAL_DIM] += perturbation
        nuisance_probe[PHYSICAL_DIM:] += perturbation
        physical_violations[step] = physical_probe
        nuisance_perturbations[step] = nuisance_probe
        perturbation_l2[step] = np.float32(np.linalg.norm(perturbation))

        physical = next_physical
        current = next_observation

    contract_sha256 = _canonical_json_sha256(_source_contract_payload(cfg))
    partial = RecurringVectorDynamicsSource(
        config=cfg,
        observations=jnp.asarray(observations),
        actions=jnp.asarray(actions),
        rewards=jnp.asarray(rewards),
        discounts=jnp.asarray(discounts),
        next_observations=jnp.asarray(next_observations),
        physical_violation_next_observations=jnp.asarray(physical_violations),
        nuisance_perturbation_next_observations=jnp.asarray(nuisance_perturbations),
        perturbation_l2=jnp.asarray(perturbation_l2),
        generator_contract_sha256=contract_sha256,
        input_sha256="",
    )
    input_sha256 = _array_manifest_sha256(
        _source_arrays(partial),
        prefix=contract_sha256,
    )
    return dataclasses.replace(partial, input_sha256=input_sha256)


def validate_recurring_vector_dynamics_source(
    source: RecurringVectorDynamicsSource,
) -> tuple[str, ...]:
    """Reconstruct and bit-compare an in-memory source; this is not evidence validation."""

    errors: list[str] = []
    expected = build_recurring_vector_dynamics_source(source.config)
    if source.generator_contract_sha256 != expected.generator_contract_sha256:
        errors.append("generator contract digest does not reconstruct")
    measured_digest = _array_manifest_sha256(
        _source_arrays(source),
        prefix=source.generator_contract_sha256,
    )
    if source.input_sha256 != measured_digest:
        errors.append("source input digest does not match its arrays")
    if source.input_sha256 != expected.input_sha256:
        errors.append("source input digest does not reconstruct")
    for (name, actual), (_, reconstructed) in zip(
        _source_arrays(source),
        _source_arrays(expected),
        strict=True,
    ):
        if not np.array_equal(np.asarray(actual), np.asarray(reconstructed)):
            errors.append(f"{name} does not reconstruct bit-exactly")

    observations = np.asarray(source.observations)
    next_observations = np.asarray(source.next_observations)
    if observations.shape != (source.config.total_steps, OBSERVATION_DIM):
        errors.append("observation shape is inconsistent with the protocol")
    elif not np.array_equal(observations[1:], next_observations[:-1]):
        errors.append("source contains a reset or discontinuity")

    physical_delta = np.asarray(source.physical_violation_next_observations) - next_observations
    nuisance_delta = np.asarray(source.nuisance_perturbation_next_observations) - next_observations
    physical_norm = np.linalg.norm(physical_delta[:, :PHYSICAL_DIM], axis=1)
    nuisance_norm = np.linalg.norm(nuisance_delta[:, PHYSICAL_DIM:], axis=1)
    if not np.allclose(physical_norm, nuisance_norm, rtol=1.0e-6, atol=1.0e-7):
        errors.append("physical and nuisance perturbation magnitudes are not matched")
    if not np.allclose(
        physical_norm,
        np.asarray(source.perturbation_l2),
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        errors.append("physical perturbation magnitude does not match its source record")
    if np.any(physical_delta[:, PHYSICAL_DIM:] != 0.0):
        errors.append("physical probe changes nuisance channels")
    if np.any(nuisance_delta[:, :PHYSICAL_DIM] != 0.0):
        errors.append("nuisance probe changes physical channels")
    return tuple(errors)


def _arm_model_config(
    config: RecurringVectorDynamicsProbeConfig,
    arm: ArmName,
) -> LatentWorldModelConfig:
    encoder_learning = arm != FIXED_ENCODER
    gate_threshold = 1.0 if arm == TRAINABLE_PREDICTION_ONLY else config.collapse_gate_threshold
    return LatentWorldModelConfig(
        observation_dim=OBSERVATION_DIM,
        n_actions=N_ACTIONS,
        latent_dim=config.latent_dim,
        gamma=0.99,
        hidden_sizes=config.hidden_sizes,
        step_size=config.predictor_step_size,
        sparsity=0.0,
        collapse_decay=0.95,
        min_latent_std=config.min_latent_std,
        encoder_learning=encoder_learning,
        encoder_step_size=config.encoder_step_size,
        max_encoder_update=config.max_encoder_update,
        encoder_collapse_gate_threshold=gate_threshold,
    )


def _state_array_digest(state: LatentWorldModelState) -> str:
    arrays = tuple(
        (f"persistent_leaf_{index}", leaf)
        for index, leaf in enumerate(jax.tree.leaves(state))
        if isinstance(leaf, Array)
    )
    return _array_manifest_sha256(arrays, prefix="latent-world-model-state-arrays-v1")


def _state_array_elements(state: LatentWorldModelState) -> int:
    return sum(int(leaf.size) for leaf in jax.tree.leaves(state) if isinstance(leaf, Array))


def _nonintervention_config_digest(model: LatentWorldModel) -> str:
    payload: dict[str, object] = model.config.to_config()
    payload.pop("encoder_learning")
    payload.pop("encoder_collapse_gate_threshold")
    return _canonical_json_sha256(payload)


def _run_prequential_trajectory(
    model: LatentWorldModel,
    initial_state: LatentWorldModelState,
    source: RecurringVectorDynamicsSource,
) -> tuple[LatentWorldModelState, PrequentialTrajectory]:
    def scan_step(
        state: LatentWorldModelState,
        inputs: tuple[Array, ...],
    ) -> tuple[LatentWorldModelState, tuple[Array, ...]]:
        (
            observation,
            action,
            reward,
            discount,
            next_observation,
            physical_violation,
            nuisance_perturbation,
        ) = inputs
        result = model.update(
            state,
            observation,
            action,
            reward,
            discount,
            next_observation,
        )
        physical_target = model.encode(state, physical_violation)
        nuisance_target = model.encode(state, nuisance_perturbation)
        physical_surprise = jnp.mean((result.prediction.next_latent - physical_target) ** 2)
        nuisance_surprise = jnp.mean((result.prediction.next_latent - nuisance_target) ** 2)
        latent_error = result.surprise
        reward_error = result.reward_error**2
        discount_error = result.discount_error**2
        joint_error = latent_error + reward_error + discount_error
        return result.state, (
            latent_error,
            reward_error,
            discount_error,
            joint_error,
            result.target_next_latent,
            result.collapse_score,
            result.latent_std_mean,
            result.surprise,
            physical_surprise,
            nuisance_surprise,
            result.encoder_update_applied,
            result.encoder_collapse_gated,
            result.update_applied,
        )

    final_state, outputs = jax.lax.scan(
        scan_step,
        initial_state,
        (
            source.observations,
            source.actions,
            source.rewards,
            source.discounts,
            source.next_observations,
            source.physical_violation_next_observations,
            source.nuisance_perturbation_next_observations,
        ),
    )
    return final_state, PrequentialTrajectory(*outputs)


def _mean_error(trajectory: PrequentialTrajectory, start: int, stop: int) -> ErrorSummary:
    return ErrorSummary(
        latent=float(jnp.mean(trajectory.latent_errors[start:stop])),
        reward=float(jnp.mean(trajectory.reward_errors[start:stop])),
        discount=float(jnp.mean(trajectory.discount_errors[start:stop])),
        joint=float(jnp.mean(trajectory.joint_errors[start:stop])),
    )


def _error_difference(left: ErrorSummary, right: ErrorSummary) -> ErrorSummary:
    return ErrorSummary(
        latent=left.latent - right.latent,
        reward=left.reward - right.reward,
        discount=left.discount - right.discount,
        joint=left.joint - right.joint,
    )


def _surprise_summary(
    trajectory: PrequentialTrajectory,
    perturbation_l2: Array,
    start: int,
    stop: int,
) -> SurpriseSeparationSummary:
    nominal = float(jnp.mean(trajectory.nominal_surprises[start:stop]))
    physical = float(jnp.mean(trajectory.physical_violation_surprises[start:stop]))
    nuisance = float(jnp.mean(trajectory.nuisance_perturbation_surprises[start:stop]))
    return SurpriseSeparationSummary(
        nominal=nominal,
        physical_violation=physical,
        nuisance_perturbation=nuisance,
        physical_excess=physical - nominal,
        nuisance_excess=nuisance - nominal,
        physical_minus_nuisance=physical - nuisance,
        perturbation_l2=float(jnp.mean(perturbation_l2[start:stop])),
    )


def _sigreg_summary(
    embeddings: Array,
    directions: Array,
    config: SIGRegConfig,
) -> SIGRegPhaseSummary:
    diagnostics = sigreg_diagnostics(embeddings, directions, config)
    return SIGRegPhaseSummary(
        loss=float(diagnostics.loss),
        latent_mean_abs=float(diagnostics.latent_mean_abs),
        latent_std_mean=float(diagnostics.latent_std_mean),
        latent_std_min=float(diagnostics.latent_std_min),
        projected_mean_abs=float(diagnostics.projected_mean_abs),
        projected_std_mean=float(diagnostics.projected_std_mean),
    )


def _summarize_arm(
    *,
    name: ArmName,
    model: LatentWorldModel,
    initial_state: LatentWorldModelState,
    final_state: LatentWorldModelState,
    source: RecurringVectorDynamicsSource,
    trajectory: PrequentialTrajectory,
    directions: Array,
    directions_sha256: str,
) -> ArmReport:
    phase_steps = source.config.phase_steps
    sigreg_config = SIGRegConfig(n_projections=source.config.sigreg_projections)
    phase_summaries = tuple(
        PhaseSummary(
            name=phase_name,
            start=index * phase_steps,
            stop=(index + 1) * phase_steps,
            prequential_error=_mean_error(
                trajectory,
                index * phase_steps,
                (index + 1) * phase_steps,
            ),
            sigreg=_sigreg_summary(
                trajectory.target_next_embeddings[index * phase_steps : (index + 1) * phase_steps],
                directions,
                sigreg_config,
            ),
            surprise=_surprise_summary(
                trajectory,
                source.perturbation_l2,
                index * phase_steps,
                (index + 1) * phase_steps,
            ),
        )
        for index, phase_name in enumerate(PHASE_NAMES)
    )

    window = source.config.summary_window
    initial_reference = _mean_error(
        trajectory,
        phase_steps - window,
        phase_steps,
    )
    recurrence_entry = _mean_error(
        trajectory,
        2 * phase_steps,
        2 * phase_steps + window,
    )
    recurrence_late = _mean_error(
        trajectory,
        3 * phase_steps - window,
        3 * phase_steps,
    )
    recurrence = RecurrenceSummary(
        initial_a_reference=initial_reference,
        recurrence_entry=recurrence_entry,
        recurrence_late=recurrence_late,
        entry_forgetting=_error_difference(recurrence_entry, initial_reference),
        within_recurrence_recovery=_error_difference(recurrence_entry, recurrence_late),
        residual_forgetting=_error_difference(recurrence_late, initial_reference),
    )

    resource = PersistentResourceSummary(
        initial_state_nbytes=measure_latent_world_model_state_nbytes(initial_state),
        final_state_nbytes=measure_latent_world_model_state_nbytes(final_state),
        initial_array_elements=_state_array_elements(initial_state),
        final_array_elements=_state_array_elements(final_state),
        predictor_input_dim=model.input_dim,
        prediction_heads=model.n_heads,
    )
    return ArmReport(
        name=name,
        encoder_learning=model.config.encoder_learning,
        collapse_gate_threshold=model.config.encoder_collapse_gate_threshold,
        source_input_sha256=source.input_sha256,
        initial_state_sha256=_state_array_digest(initial_state),
        sigreg_directions_sha256=directions_sha256,
        nonintervention_config_sha256=_nonintervention_config_digest(model),
        resource=resource,
        trajectory=trajectory,
        phase_summaries=phase_summaries,
        recurrence=recurrence,
        overall_surprise=_surprise_summary(
            trajectory,
            source.perturbation_l2,
            0,
            source.config.total_steps,
        ),
        world_update_rate=float(jnp.mean(trajectory.world_updates.astype(jnp.float32))),
        encoder_update_rate=float(jnp.mean(trajectory.encoder_updates.astype(jnp.float32))),
        encoder_gate_rate=float(jnp.mean(trajectory.encoder_gates.astype(jnp.float32))),
    )


def run_latent_world_model_recurrence_development(
    config: RecurringVectorDynamicsProbeConfig | None = None,
) -> LatentWorldModelRecurrenceDevelopmentReport:
    """Run the in-memory matched probe and return descriptive measurements."""

    cfg = config or RecurringVectorDynamicsProbeConfig()
    source = build_recurring_vector_dynamics_source(cfg)
    source_errors = validate_recurring_vector_dynamics_source(source)
    if source_errors:
        raise RuntimeError(f"source reconstruction failed: {source_errors!r}")

    models = tuple(LatentWorldModel(_arm_model_config(cfg, arm)) for arm in ARM_ORDER)
    base_key = jr.key(cfg.development_key)
    initial_key, direction_key = jr.split(base_key)
    initial_state = models[0].init(initial_key)
    sigreg_config = SIGRegConfig(n_projections=cfg.sigreg_projections)
    directions = sample_sigreg_directions(direction_key, cfg.latent_dim, sigreg_config)
    directions_sha256 = _array_manifest_sha256(
        (("sigreg_directions", directions),),
        prefix="latent-world-model-recurrence-sigreg-v1",
    )

    arm_reports: list[ArmReport] = []
    for name, model in zip(ARM_ORDER, models, strict=True):
        final_state, trajectory = _run_prequential_trajectory(
            model,
            initial_state,
            source,
        )
        arm_reports.append(
            _summarize_arm(
                name=name,
                model=model,
                initial_state=initial_state,
                final_state=final_state,
                source=source,
                trajectory=trajectory,
                directions=directions,
                directions_sha256=directions_sha256,
            )
        )

    arms = tuple(arm_reports)
    resource_equal = all(arm.resource == arms[0].resource for arm in arms[1:])
    config_equal = all(
        arm.nonintervention_config_sha256 == arms[0].nonintervention_config_sha256
        for arm in arms[1:]
    )
    common_randomness = all(
        arm.source_input_sha256 == source.input_sha256
        and arm.initial_state_sha256 == arms[0].initial_state_sha256
        and arm.sigreg_directions_sha256 == directions_sha256
        for arm in arms
    )
    return LatentWorldModelRecurrenceDevelopmentReport(
        schema=DEVELOPMENT_SCHEMA,
        status="development_only_descriptive_not_assessed",
        development_only=DEVELOPMENT_ONLY,
        assessment_status=ASSESSMENT_STATUS,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        thresholds_frozen=THRESHOLDS_FROZEN,
        development_keys_frozen=DEVELOPMENT_KEYS_FROZEN,
        evidence_claimed=EVIDENCE_CLAIMED,
        task_identifiers_exposed=TASK_IDENTIFIERS_EXPOSED,
        resets_exposed=RESETS_EXPOSED,
        descriptive_claims_only=True,
        config=cfg,
        source=source,
        binding=CommonRandomnessBinding(
            source_input_sha256=source.input_sha256,
            initial_state_sha256=arms[0].initial_state_sha256,
            sigreg_directions_sha256=directions_sha256,
            generator_contract_sha256=source.generator_contract_sha256,
            source_reconstruction_valid=True,
            common_randomness_preserved=common_randomness,
        ),
        arms=arms,
        fixed_persistent_resources_equal=resource_equal,
        matched_nonintervention_config=config_equal,
        limitations=_LIMITATIONS,
    )


def validate_latent_world_model_recurrence_development_report(
    report: LatentWorldModelRecurrenceDevelopmentReport,
) -> tuple[str, ...]:
    """Check local reconstruction and nonpromotion contracts, never efficacy."""

    errors = list(validate_recurring_vector_dynamics_source(report.source))
    expected_contract = (
        report.schema == DEVELOPMENT_SCHEMA,
        report.status == "development_only_descriptive_not_assessed",
        report.development_only is True,
        report.assessment_status == ASSESSMENT_STATUS,
        report.scientific_promotion_allowed is False,
        report.output_writes_allowed is False,
        report.thresholds_frozen is False,
        report.development_keys_frozen is False,
        report.evidence_claimed is False,
        report.task_identifiers_exposed is False,
        report.resets_exposed is False,
        report.descriptive_claims_only is True,
    )
    if not all(expected_contract):
        errors.append("development-only nonpromotion contract changed")
    if report.config != report.source.config:
        errors.append("report config is not bound to the source config")
    if tuple(arm.name for arm in report.arms) != ARM_ORDER:
        errors.append("arm order or membership changed")
    if not report.arms:
        errors.append("report contains no arms")
        return tuple(errors)

    expected_models = tuple(
        LatentWorldModel(_arm_model_config(report.config, arm)) for arm in ARM_ORDER
    )
    initial_key, direction_key = jr.split(jr.key(report.config.development_key))
    expected_initial_state = expected_models[0].init(initial_key)
    expected_initial_digest = _state_array_digest(expected_initial_state)
    expected_sigreg_config = SIGRegConfig(
        n_projections=report.config.sigreg_projections
    )
    expected_directions = sample_sigreg_directions(
        direction_key,
        report.config.latent_dim,
        expected_sigreg_config,
    )
    expected_directions_digest = _array_manifest_sha256(
        (("sigreg_directions", expected_directions),),
        prefix="latent-world-model-recurrence-sigreg-v1",
    )

    resources_equal = all(arm.resource == report.arms[0].resource for arm in report.arms[1:])
    if report.fixed_persistent_resources_equal != resources_equal or not resources_equal:
        errors.append("persistent resource equality is false or inconsistent")
    config_equal = all(
        arm.nonintervention_config_sha256 == report.arms[0].nonintervention_config_sha256
        for arm in report.arms[1:]
    )
    if report.matched_nonintervention_config != config_equal or not config_equal:
        errors.append("nonintervention configs are not matched")

    binding = report.binding
    if binding.source_input_sha256 != report.source.input_sha256:
        errors.append("binding source digest does not match report source")
    if binding.generator_contract_sha256 != report.source.generator_contract_sha256:
        errors.append("binding generator digest does not match report source")
    common_randomness = all(
        arm.source_input_sha256 == binding.source_input_sha256
        and arm.initial_state_sha256 == binding.initial_state_sha256
        and arm.sigreg_directions_sha256 == binding.sigreg_directions_sha256
        for arm in report.arms
    )
    if binding.common_randomness_preserved != common_randomness or not common_randomness:
        errors.append("common-randomness binding is false or inconsistent")
    if binding.source_reconstruction_valid is not True:
        errors.append("source reconstruction is not recorded as valid")
    if binding.initial_state_sha256 != expected_initial_digest:
        errors.append("initial-state binding does not reconstruct")
    if binding.sigreg_directions_sha256 != expected_directions_digest:
        errors.append("SIGReg-direction binding does not reconstruct")

    total = report.config.total_steps
    scalar_fields = (
        "latent_errors",
        "reward_errors",
        "discount_errors",
        "joint_errors",
        "collapse_scores",
        "latent_std_means",
        "nominal_surprises",
        "physical_violation_surprises",
        "nuisance_perturbation_surprises",
    )
    boolean_fields = ("encoder_updates", "encoder_gates", "world_updates")
    for arm, model in zip(report.arms, expected_models, strict=True):
        if arm.encoder_learning is not model.config.encoder_learning:
            errors.append(f"{arm.name} encoder-learning label is inconsistent")
        if arm.collapse_gate_threshold != model.config.encoder_collapse_gate_threshold:
            errors.append(f"{arm.name} collapse-gate threshold is inconsistent")
        if arm.nonintervention_config_sha256 != _nonintervention_config_digest(model):
            errors.append(f"{arm.name} nonintervention config does not reconstruct")
        if arm.initial_state_sha256 != expected_initial_digest:
            errors.append(f"{arm.name} initial state does not reconstruct")
        if arm.sigreg_directions_sha256 != expected_directions_digest:
            errors.append(f"{arm.name} SIGReg directions do not reconstruct")

        for field in scalar_fields:
            value = np.asarray(getattr(arm.trajectory, field))
            if value.shape != (total,):
                errors.append(f"{arm.name} {field} has the wrong shape")
            if not np.all(np.isfinite(value)):
                errors.append(f"{arm.name} {field} contains non-finite values")
        embeddings = np.asarray(arm.trajectory.target_next_embeddings)
        if embeddings.shape != (total, report.config.latent_dim):
            errors.append(f"{arm.name} target embeddings have the wrong shape")
        if not np.all(np.isfinite(embeddings)):
            errors.append(f"{arm.name} target embeddings contain non-finite values")
        for field in boolean_fields:
            value = np.asarray(getattr(arm.trajectory, field))
            if value.shape != (total,) or value.dtype != np.dtype(np.bool_):
                errors.append(f"{arm.name} {field} must be a bool trajectory")

        measured_world_rate = float(
            np.mean(np.asarray(arm.trajectory.world_updates, dtype=np.float32))
        )
        measured_encoder_rate = float(
            np.mean(np.asarray(arm.trajectory.encoder_updates, dtype=np.float32))
        )
        measured_gate_rate = float(
            np.mean(np.asarray(arm.trajectory.encoder_gates, dtype=np.float32))
        )
        if not math.isclose(
            arm.world_update_rate, measured_world_rate, rel_tol=1.0e-7, abs_tol=1.0e-8
        ):
            errors.append(f"{arm.name} world update rate does not reconstruct")
        if not math.isclose(
            arm.encoder_update_rate,
            measured_encoder_rate,
            rel_tol=1.0e-7,
            abs_tol=1.0e-8,
        ):
            errors.append(f"{arm.name} encoder update rate does not reconstruct")
        if not math.isclose(
            arm.encoder_gate_rate, measured_gate_rate, rel_tol=1.0e-7, abs_tol=1.0e-8
        ):
            errors.append(f"{arm.name} encoder gate rate does not reconstruct")
        if not 0.0 <= arm.world_update_rate <= 1.0:
            errors.append(f"{arm.name} world update rate is invalid")
        if not 0.0 <= arm.encoder_update_rate <= 1.0:
            errors.append(f"{arm.name} encoder update rate is invalid")
        if not 0.0 <= arm.encoder_gate_rate <= 1.0:
            errors.append(f"{arm.name} encoder gate rate is invalid")
        if tuple(phase.name for phase in arm.phase_summaries) != PHASE_NAMES:
            errors.append(f"{arm.name} phase summaries are incomplete")
        expected_bounds = tuple(
            (index * report.config.phase_steps, (index + 1) * report.config.phase_steps)
            for index in range(len(PHASE_NAMES))
        )
        actual_bounds = tuple(
            (phase.start, phase.stop) for phase in arm.phase_summaries
        )
        if actual_bounds != expected_bounds:
            errors.append(f"{arm.name} phase bounds are inconsistent")
    return tuple(errors)
