# mypy: disable-error-code="call-arg"
"""Tiny development-only online SIGReg objective comparison.

This evaluator asks one deliberately narrow question left open by
``latent_world_model_recurrence_development``: what changes when SIGReg is an
*optimized encoder objective*, rather than a descriptive diagnostic?  Three
matched linear latent-world-model arms consume one uninterrupted deterministic
``A -> B -> A`` stream exactly once:

* ``prediction_only`` learns a one-step action-conditioned latent predictor;
* ``sigreg_inert`` additionally evaluates the SIGReg loss and encoder gradient
  but discards that gradient; and
* ``sigreg_routed`` evaluates the identical regularizer work and routes the
  bounded gradient to the post-transition encoder state.

Every prediction is made before its next observation is supplied.  Predictor
learning uses the current transition once.  SIGReg necessarily needs a sample
set, so its explicitly accounted context is a fixed-size FIFO of observations
already seen; it re-embeds those observations for the auxiliary distribution
loss but never replays their actions, transitions, prediction targets, or
outcomes.  Phase identities are evaluator-only indices and never enter the
learner step.

A deterministic ridge readout is fit once, from the first full causal context,
then frozen.  Its raw physical- and nuisance-channel errors expose collapse and
representation drift independently of the moving latent target.  The readout
is diagnostic only and no value or gradient from it reaches the learner.

This is an in-memory L0/development mechanism probe.  It has no artifact
writer, evidence root, frozen key, efficacy threshold, verdict, or promotion
path.  All reported differences remain descriptive and ``not_assessed``.
Logical transient byte counts are exact for the named float32 tensors and are
not claims about compiler fusion, allocator peaks, wall time, or FLOPs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Final, Literal, NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.sigreg import sliced_sigreg_loss

DEVELOPMENT_SCHEMA: Final = "alberta.online-sigreg-objective.development.v1"
EVIDENCE_LEVEL: Final = "L0"
DEVELOPMENT_ONLY: Final = True
ASSESSMENT_STATUS: Final = "not_assessed"
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_CLAIMED: Final = False
THRESHOLDS_FROZEN: Final = False
DEVELOPMENT_KEYS_FROZEN: Final = False
PHASE_IDENTIFIERS_EXPOSED: Final = False
TRANSITION_REPLAYED: Final = False

PREDICTION_ONLY: Final = "prediction_only"
SIGREG_INERT: Final = "sigreg_inert"
SIGREG_ROUTED: Final = "sigreg_routed"
ArmName = Literal["prediction_only", "sigreg_inert", "sigreg_routed"]
ARM_ORDER: Final[tuple[ArmName, ArmName, ArmName]] = (
    PREDICTION_ONLY,
    SIGREG_INERT,
    SIGREG_ROUTED,
)

PHASE_NAMES: Final = ("A_initial", "B_interference", "A_recurrence")
SOURCE_VERSION: Final = "online-sigreg-recurring-dynamics-v1"
OBSERVATION_DIM: Final = 4
PHYSICAL_DIM: Final = 2
NUISANCE_DIM: Final = 2
N_ACTIONS: Final = 2

_A_BASE_ROTATION: Final = 0.31
_B_BASE_ROTATION: Final = -0.49
_ACTION_ROTATION_OFFSETS: Final = (-0.07, 0.07)
_INITIAL_PHYSICAL_STATE: Final = (0.8, -0.6)
_NUISANCE_SCALE: Final = 0.72

_LIMITATIONS: Final = (
    "one deterministic low-dimensional stream is not a robustness or scaling result",
    "the encoder and predictor are linear-tanh mechanisms, not the visual LeWorldModel stack",
    "SIGReg uses a bounded past-only observation context and therefore re-embeds old "
    "observations even though transitions and targets are never replayed",
    "one frozen prefix-only ridge readout diagnoses retention but is not a universal probe",
    "logical transient bytes describe named tensors, not compiler or allocator peak memory",
    "report-output bytes count exact JAX array payloads, not Python object/container overhead",
    "algorithmic gradient-count matching does not establish equal wall time or energy",
    "all recurrence, collapse, and feature-quality differences are descriptive without gates",
    "same-context post-update SIGReg loss is an in-sample intervention sanity metric, not "
    "generalization or efficacy",
    "there is no action selection, reward, control objective, or prediction-to-decision utility",
)


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineSIGRegDevelopmentConfig:
    """Small deterministic configuration with no evidential thresholds."""

    phase_steps: int = 24
    summary_window: int = 6
    context_size: int = 8
    latent_dim: int = 2
    sigreg_projections: int = 6
    prediction_step_size: float = 0.025
    sigreg_step_size: float = 0.015
    max_parameter_update: float = 0.04
    probe_ridge: float = 0.01
    min_latent_std: float = 0.05
    development_key: int = 0

    def __post_init__(self) -> None:
        integer_fields = {
            "phase_steps": self.phase_steps,
            "summary_window": self.summary_window,
            "context_size": self.context_size,
            "latent_dim": self.latent_dim,
            "sigreg_projections": self.sigreg_projections,
            "development_key": self.development_key,
        }
        for name, value in integer_fields.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an exact integer")
        if self.phase_steps < 4:
            raise ValueError("phase_steps must be at least four")
        if not 1 <= self.summary_window <= self.phase_steps:
            raise ValueError("summary_window must be in [1, phase_steps]")
        if not 3 <= self.context_size <= self.phase_steps:
            raise ValueError("context_size must be in [3, phase_steps]")
        if self.phase_steps - self.summary_window < self.context_size:
            raise ValueError(
                "the late initial-A window must be disjoint from the probe-fit prefix"
            )
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.sigreg_projections <= 0:
            raise ValueError("sigreg_projections must be positive")
        if not 0 <= self.development_key <= 2**32 - 1:
            raise ValueError("development_key must fit one uint32 word")

        positive_floats = {
            "prediction_step_size": self.prediction_step_size,
            "sigreg_step_size": self.sigreg_step_size,
            "max_parameter_update": self.max_parameter_update,
            "probe_ridge": self.probe_ridge,
        }
        for float_name, float_value in positive_floats.items():
            _require_exact_finite_float(float_name, float_value)
            if float_value <= 0.0:
                raise ValueError(f"{float_name} must be positive")
        ridge_floor = float(np.finfo(np.float32).eps) * self.context_size
        if self.probe_ridge < ridge_floor:
            raise ValueError(
                "probe_ridge must be at least context_size times float32 epsilon"
            )
        _require_exact_finite_float("min_latent_std", self.min_latent_std)
        if self.min_latent_std < 0.0:
            raise ValueError("min_latent_std must be non-negative")

    @property
    def total_steps(self) -> int:
        """Number of transitions in the uninterrupted A/B/A life."""

        return 3 * self.phase_steps

    @property
    def first_sigreg_transition(self) -> int:
        """First zero-based transition whose post-outcome context is full."""

        return self.context_size - 2

    @property
    def eligible_sigreg_steps(self) -> int:
        """Exact number of full-context post-outcome regularizer events."""

        return self.total_steps - self.first_sigreg_transition


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineSIGRegSource:
    """Exact source arrays; no phase-identity array exists."""

    config: OnlineSIGRegDevelopmentConfig
    observations: Array
    actions: Array
    next_observations: Array
    generator_contract_sha256: str
    input_sha256: str


@chex.dataclass(frozen=True)
class OnlineSIGRegState:
    """All persistent learner and causal diagnostic arrays."""

    encoder_matrix: Array
    encoder_bias: Array
    predictor_matrix: Array
    predictor_bias: Array
    observation_context: Array
    context_count: Array
    context_cursor: Array
    sigreg_directions: Array
    frozen_probe_matrix: Array
    frozen_probe_ready: Array
    step_count: Array


@dataclasses.dataclass(frozen=True, slots=True)
class TimingOwnership:
    """Human-readable ownership of every event-time operation."""

    pre_outcome_prediction_inputs: tuple[str, ...]
    outcome_revealed_after_prediction: str
    predictor_update_inputs: tuple[str, ...]
    regularizer_context: str
    regularizer_update_effective: str
    frozen_probe_fit: str
    frozen_probe_measurement: str
    evaluator_only_segmentation: str


@dataclasses.dataclass(frozen=True, slots=True)
class PersistentResourceSummary:
    """Exact byte counts over state and retained report JAX array payloads."""

    accounting_scope: str
    initial_total_nbytes: int
    final_total_nbytes: int
    learner_parameter_nbytes: int
    observation_context_nbytes: int
    sigreg_direction_nbytes: int
    frozen_probe_nbytes: int
    scalar_control_nbytes: int
    exact_component_sum_matches_total: bool
    trajectory_output_nbytes: int
    shared_source_input_nbytes: int
    retained_report_array_nbytes_including_shared_source: int


@dataclasses.dataclass(frozen=True, slots=True)
class LogicalTransientTensor:
    """One exact source-level tensor payload, not a device-memory claim."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes_per_evaluation: int
    evaluations: int
    total_named_payload_nbytes: int
    timing: str


@dataclasses.dataclass(frozen=True, slots=True)
class WorkSummary:
    """Exact semantic operation counts for one arm."""

    transitions_consumed_once: int
    transition_replays: int
    prediction_gradient_evaluations: int
    sigreg_objective_gradient_evaluations: int
    sigreg_gradient_candidates: int
    sigreg_gradients_routed: int
    sigreg_gradients_discarded: int
    sigreg_gradients_rejected_nonfinite: int
    post_update_sigreg_diagnostics: int
    past_observations_reembedded_for_sigreg_objective: int
    frozen_probe_fits: int
    logical_transient_tensors: tuple[LogicalTransientTensor, ...]
    logical_transient_payload_nbytes: int
    allocator_peak_assessed: bool
    wall_clock_matched: bool


@dataclasses.dataclass(frozen=True, slots=True)
class PrequentialTrajectory:
    """Raw ordered metrics and update receipts."""

    latent_prediction_mse: Array
    physical_probe_mse: Array
    nuisance_probe_mse: Array
    probe_available: Array
    sigreg_objective_loss: Array
    sigreg_gradient_norm: Array
    sigreg_candidate_update_norm: Array
    post_update_sigreg_loss: Array
    post_update_latent_std_min: Array
    post_update_collapsed_fraction: Array
    sigreg_context_available: Array
    prediction_update_applied: Array
    sigreg_gradient_evaluated: Array
    sigreg_gradient_routed: Array
    sigreg_gradient_discarded: Array
    sigreg_gradient_rejected_nonfinite: Array
    pre_step_count: Array
    post_step_count: Array


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Descriptive means for one externally indexed phase."""

    name: str
    start: int
    stop: int
    latent_prediction_mse: float
    physical_probe_mse: float
    nuisance_probe_mse: float
    post_update_sigreg_loss: float
    post_update_latent_std_min: float
    post_update_collapsed_fraction: float
    probe_measurements: int
    sigreg_measurements: int


@dataclasses.dataclass(frozen=True, slots=True)
class WindowRecurrence:
    """Three A windows and signed descriptive differences."""

    initial_a_late: float
    recurrence_a_entry: float
    recurrence_a_late: float
    entry_minus_initial: float
    entry_minus_late: float
    late_minus_initial: float


@dataclasses.dataclass(frozen=True, slots=True)
class RecurrenceSummary:
    """Recurrence/retention views with no gate or pass/fail direction."""

    latent_prediction_mse: WindowRecurrence
    physical_probe_mse: WindowRecurrence
    nuisance_probe_mse: WindowRecurrence
    latent_std_min: WindowRecurrence
    collapsed_fraction: WindowRecurrence


@dataclasses.dataclass(frozen=True, slots=True)
class ArmReport:
    """One arm's matched state, raw trajectory, resources, and summaries."""

    name: ArmName
    sigreg_objective_enabled: bool
    sigreg_gradient_routed_by_design: bool
    source_input_sha256: str
    initial_state_sha256: str
    frozen_probe_sha256: str
    final_learner_sha256: str
    resource: PersistentResourceSummary
    work: WorkSummary
    trajectory: PrequentialTrajectory
    phase_metrics: tuple[PhaseMetrics, ...]
    recurrence: RecurrenceSummary


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineSIGRegDevelopmentReport:
    """In-memory L0 report; this is not an evidence artifact."""

    schema: str
    evidence_level: str
    status: str
    development_only: bool
    assessment_status: str
    scientific_promotion_allowed: bool
    output_writes_allowed: bool
    evidence_claimed: bool
    thresholds_frozen: bool
    development_keys_frozen: bool
    phase_identifiers_exposed: bool
    transition_replayed: bool
    descriptive_claims_only: bool
    config: OnlineSIGRegDevelopmentConfig
    source: OnlineSIGRegSource
    timing: TimingOwnership
    arms: tuple[ArmReport, ...]
    common_initial_state: bool
    persistent_resources_equal: bool
    routed_and_inert_objective_compute_matched: bool
    inert_and_prediction_only_learners_identical: bool
    limitations: tuple[str, ...]


class _StepOutput(NamedTuple):
    latent_prediction_mse: Array
    physical_probe_mse: Array
    nuisance_probe_mse: Array
    probe_available: Array
    sigreg_objective_loss: Array
    sigreg_gradient_norm: Array
    sigreg_candidate_update_norm: Array
    post_update_sigreg_loss: Array
    post_update_latent_std_min: Array
    post_update_collapsed_fraction: Array
    sigreg_context_available: Array
    prediction_update_applied: Array
    sigreg_gradient_evaluated: Array
    sigreg_gradient_routed: Array
    sigreg_gradient_discarded: Array
    sigreg_gradient_rejected_nonfinite: Array
    pre_step_count: Array
    post_step_count: Array


@dataclasses.dataclass(frozen=True, slots=True)
class _LearnerConfig:
    """Phase-free configuration visible to the online transition kernel."""

    context_size: int
    prediction_step_size: float
    sigreg_step_size: float
    max_parameter_update: float
    probe_ridge: float
    min_latent_std: float


def _learner_config(config: OnlineSIGRegDevelopmentConfig) -> _LearnerConfig:
    return _LearnerConfig(
        context_size=config.context_size,
        prediction_step_size=config.prediction_step_size,
        sigreg_step_size=config.sigreg_step_size,
        max_parameter_update=config.max_parameter_update,
        probe_ridge=config.probe_ridge,
        min_latent_std=config.min_latent_std,
    )


def _require_exact_finite_float(name: str, value: object) -> None:
    if type(value) is not float:
        raise TypeError(f"{name} must be an exact built-in float")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore"):
        float32_value = np.float32(value)
    if not bool(np.isfinite(float32_value)) or (
        value != 0.0 and float32_value == np.float32(0.0)
    ):
        raise ValueError(f"{name} must remain finite and nonzero in float32")


def _config_payload(config: OnlineSIGRegDevelopmentConfig) -> dict[str, object]:
    return {
        "schema": "alberta.online-sigreg-objective.config.v1",
        "phase_steps": config.phase_steps,
        "summary_window": config.summary_window,
        "context_size": config.context_size,
        "latent_dim": config.latent_dim,
        "sigreg_projections": config.sigreg_projections,
        "prediction_step_size": config.prediction_step_size,
        "sigreg_step_size": config.sigreg_step_size,
        "max_parameter_update": config.max_parameter_update,
        "probe_ridge": config.probe_ridge,
        "min_latent_std": config.min_latent_std,
        "development_key": config.development_key,
    }


def _config_sha256(config: OnlineSIGRegDevelopmentConfig) -> str:
    return _canonical_json_sha256(_config_payload(config))


def _validate_config_instance(config: object) -> tuple[str, ...]:
    if type(config) is not OnlineSIGRegDevelopmentConfig:
        return ("config must be an exact OnlineSIGRegDevelopmentConfig",)
    try:
        payload = _config_payload(config)
        OnlineSIGRegDevelopmentConfig(
            phase_steps=cast(int, payload["phase_steps"]),
            summary_window=cast(int, payload["summary_window"]),
            context_size=cast(int, payload["context_size"]),
            latent_dim=cast(int, payload["latent_dim"]),
            sigreg_projections=cast(int, payload["sigreg_projections"]),
            prediction_step_size=cast(float, payload["prediction_step_size"]),
            sigreg_step_size=cast(float, payload["sigreg_step_size"]),
            max_parameter_update=cast(float, payload["max_parameter_update"]),
            probe_ridge=cast(float, payload["probe_ridge"]),
            min_latent_std=cast(float, payload["min_latent_std"]),
            development_key=cast(int, payload["development_key"]),
        )
        _config_sha256(config)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return (f"config is malformed: {type(exc).__name__}",)
    return ()


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


def _exact_value_equal(actual: object, expected: object) -> bool:
    """Recursively compare types and values, including exact array bytes."""

    if type(actual) is not type(expected):
        return False
    if isinstance(actual, Array) and isinstance(expected, Array):
        return _arrays_bit_equal(actual, expected)
    if dataclasses.is_dataclass(actual) and not isinstance(actual, type):
        actual_fields = dataclasses.fields(actual)
        expected_fields = dataclasses.fields(expected)  # type: ignore[arg-type]
        if tuple(field.name for field in actual_fields) != tuple(
            field.name for field in expected_fields
        ):
            return False
        return all(
            _exact_value_equal(
                getattr(actual, field.name),
                getattr(expected, field.name),
            )
            for field in actual_fields
        )
    if isinstance(actual, tuple) and isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _exact_value_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(actual, float) and isinstance(expected, float):
        return actual.hex() == expected.hex()
    return bool(actual == expected)


def _arrays_bit_equal(
    actual: Array | np.ndarray,
    expected: Array | np.ndarray,
) -> bool:
    actual_array = np.ascontiguousarray(np.asarray(actual))
    expected_array = np.ascontiguousarray(np.asarray(expected))
    return (
        actual_array.shape == expected_array.shape
        and actual_array.dtype == expected_array.dtype
        and actual_array.tobytes(order="C") == expected_array.tobytes(order="C")
    )


def _source_contract(config: OnlineSIGRegDevelopmentConfig) -> dict[str, object]:
    return {
        "version": SOURCE_VERSION,
        "phase_steps": config.phase_steps,
        "phase_order": PHASE_NAMES,
        "observation_layout": (
            "physical_0",
            "physical_1",
            "nuisance_0",
            "nuisance_1",
        ),
        "a_base_rotation": _A_BASE_ROTATION,
        "b_base_rotation": _B_BASE_ROTATION,
        "action_rotation_offsets": _ACTION_ROTATION_OFFSETS,
        "initial_physical_state": _INITIAL_PHYSICAL_STATE,
        "nuisance_scale": _NUISANCE_SCALE,
        "phase_identifiers_exposed": False,
        "resets_exposed": False,
    }


def _nuisance(index: int) -> np.ndarray:
    """Deterministic bounded nuisance with no simple one-step linear rule."""

    position = float(index + 1)
    return np.asarray(
        (
            _NUISANCE_SCALE * math.sin(0.37 * position**2 + 1.11 * position + 0.2),
            _NUISANCE_SCALE * math.cos(0.29 * position**2 + 1.73 * position - 0.4),
        ),
        dtype=np.float32,
    )


def build_online_sigreg_source(
    config: OnlineSIGRegDevelopmentConfig | None = None,
) -> OnlineSIGRegSource:
    """Build one continuous deterministic A/B/A source."""

    if config is None:
        cfg = OnlineSIGRegDevelopmentConfig()
    elif type(config) is OnlineSIGRegDevelopmentConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact OnlineSIGRegDevelopmentConfig or None")
    observations = np.empty((cfg.total_steps + 1, OBSERVATION_DIM), dtype=np.float32)
    actions = np.arange(cfg.total_steps, dtype=np.int32) % N_ACTIONS
    physical = np.asarray(_INITIAL_PHYSICAL_STATE, dtype=np.float32)
    physical /= np.linalg.norm(physical)
    observations[0] = np.concatenate((physical, _nuisance(0)))

    for step in range(cfg.total_steps):
        phase_index = step // cfg.phase_steps
        in_a = phase_index != 1
        angle = (
            (_A_BASE_ROTATION if in_a else _B_BASE_ROTATION)
            + _ACTION_ROTATION_OFFSETS[int(actions[step])]
        )
        cosine = math.cos(angle)
        sine = math.sin(angle)
        rotation = np.asarray(
            ((cosine, -sine), (sine, cosine)),
            dtype=np.float32,
        )
        physical = (rotation @ physical).astype(np.float32)
        observations[step + 1] = np.concatenate((physical, _nuisance(step + 1)))

    contract_sha256 = _canonical_json_sha256(_source_contract(cfg))
    input_sha256 = _array_manifest_sha256(
        (
            ("observations", observations[:-1]),
            ("actions", actions),
            ("next_observations", observations[1:]),
        ),
        prefix=contract_sha256,
    )
    return OnlineSIGRegSource(
        config=cfg,
        observations=jnp.asarray(observations[:-1]),
        actions=jnp.asarray(actions),
        next_observations=jnp.asarray(observations[1:]),
        generator_contract_sha256=contract_sha256,
        input_sha256=input_sha256,
    )


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_online_sigreg_source(source: object) -> tuple[str, ...]:
    """Reconstruct and bit-check the in-memory source."""

    if type(source) is not OnlineSIGRegSource:
        return ("source must be an exact OnlineSIGRegSource",)
    errors: list[str] = []
    config_errors = _validate_config_instance(source.config)
    if config_errors:
        return tuple(f"source {error}" for error in config_errors)
    expected = build_online_sigreg_source(source.config)
    arrays = (
        ("observations", source.observations),
        ("actions", source.actions),
        ("next_observations", source.next_observations),
    )
    expected_contracts = {
        "observations": (
            (source.config.total_steps, OBSERVATION_DIM),
            jnp.dtype(jnp.float32),
        ),
        "actions": ((source.config.total_steps,), jnp.dtype(jnp.int32)),
        "next_observations": (
            (source.config.total_steps, OBSERVATION_DIM),
            jnp.dtype(jnp.float32),
        ),
    }
    arrays_valid = True
    for name, value in arrays:
        expected_shape, expected_dtype = expected_contracts[name]
        if not isinstance(value, Array):
            errors.append(f"{name} must be a JAX Array")
            arrays_valid = False
            continue
        if value.shape != expected_shape:
            errors.append(f"{name} has the wrong shape")
            arrays_valid = False
        if value.dtype != expected_dtype:
            errors.append(f"{name} has the wrong dtype")
            arrays_valid = False
        if not bool(jnp.all(jnp.isfinite(value))):
            errors.append(f"{name} contains non-finite values")
            arrays_valid = False
    digests_valid = True
    if not _is_sha256(source.generator_contract_sha256):
        errors.append("source generator contract digest must be lowercase sha256")
        digests_valid = False
    if not _is_sha256(source.input_sha256):
        errors.append("source input digest must be lowercase sha256")
        digests_valid = False
    if not arrays_valid or not digests_valid:
        return tuple(errors)

    measured = _array_manifest_sha256(
        arrays,
        prefix=source.generator_contract_sha256,
    )
    if source.generator_contract_sha256 != expected.generator_contract_sha256:
        errors.append("source generator contract does not reconstruct")
    if source.input_sha256 != measured:
        errors.append("source digest does not match its arrays")
    if source.input_sha256 != expected.input_sha256:
        errors.append("source digest does not reconstruct")
    for (name, actual), (_, reconstructed) in zip(
        arrays,
        (
            ("observations", expected.observations),
            ("actions", expected.actions),
            ("next_observations", expected.next_observations),
        ),
        strict=True,
    ):
        if not _arrays_bit_equal(actual, reconstructed):
            errors.append(f"{name} does not reconstruct bit-exactly")

    if not _arrays_bit_equal(
        source.observations[1:],
        source.next_observations[:-1],
    ):
        errors.append("source has a reset or discontinuity")
    return tuple(errors)


def _encode(encoder_matrix: Array, encoder_bias: Array, observation: Array) -> Array:
    return jnp.tanh(jnp.asarray(observation, dtype=jnp.float32) @ encoder_matrix + encoder_bias)


def _predict_next_latent(
    state: OnlineSIGRegState,
    observation: Array,
    action: Array,
) -> Array:
    """Pre-outcome prediction; no next observation is accepted by this API."""

    latent = _encode(state.encoder_matrix, state.encoder_bias, observation)
    features = jnp.concatenate(
        (
            latent,
            jax.nn.one_hot(action, N_ACTIONS, dtype=jnp.float32),
        )
    )
    return latent + features @ state.predictor_matrix + state.predictor_bias


def _prediction_objective(
    encoder_matrix: Array,
    encoder_bias: Array,
    predictor_matrix: Array,
    predictor_bias: Array,
    observation: Array,
    action: Array,
    next_observation: Array,
) -> Array:
    latent = _encode(encoder_matrix, encoder_bias, observation)
    target = jax.lax.stop_gradient(
        _encode(encoder_matrix, encoder_bias, next_observation)
    )
    features = jnp.concatenate(
        (latent, jax.nn.one_hot(action, N_ACTIONS, dtype=jnp.float32))
    )
    prediction = latent + features @ predictor_matrix + predictor_bias
    return jnp.mean((prediction - target) ** 2)


def _sigreg_objective(
    encoder_matrix: Array,
    encoder_bias: Array,
    observation_context: Array,
    directions: Array,
) -> Array:
    embeddings = jnp.tanh(observation_context @ encoder_matrix + encoder_bias)
    return sliced_sigreg_loss(embeddings, directions)


def _bounded_candidate(parameter: Array, gradient: Array, step_size: float, bound: float) -> Array:
    delta = jnp.clip(
        -jnp.asarray(step_size, dtype=jnp.float32) * gradient,
        -jnp.asarray(bound, dtype=jnp.float32),
        jnp.asarray(bound, dtype=jnp.float32),
    )
    return parameter + delta


def _all_finite(values: tuple[Array, ...]) -> Array:
    valid = jnp.asarray(True, dtype=jnp.bool_)
    for value in values:
        valid = valid & jnp.all(jnp.isfinite(value))
    return valid


def _finite_l2_norm(*values: Array) -> tuple[Array, Array]:
    """Return a zeroed float32 L2 diagnostic and whether its reduction stayed finite."""

    squared_sum = jnp.asarray(0.0, dtype=jnp.float32)
    for value in values:
        squared_sum = squared_sum + jnp.sum(jnp.square(value))
    raw_norm = jnp.sqrt(squared_sum)
    finite = jnp.isfinite(raw_norm)
    return jnp.where(finite, raw_norm, jnp.asarray(0.0, dtype=jnp.float32)), finite


def _fit_frozen_probe(
    encoder_matrix: Array,
    encoder_bias: Array,
    observation_context: Array,
    ridge: float,
) -> Array:
    embeddings = jnp.tanh(observation_context @ encoder_matrix + encoder_bias)
    design = jnp.concatenate(
        (embeddings, jnp.ones((embeddings.shape[0], 1), dtype=jnp.float32)),
        axis=1,
    )
    identity = jnp.eye(design.shape[1], dtype=jnp.float32)
    gram = design.T @ design + jnp.asarray(ridge, dtype=jnp.float32) * identity
    return cast(Array, jnp.linalg.solve(gram, design.T @ observation_context))


def _append_context(
    context: Array,
    cursor: Array,
    count: Array,
    observation: Array,
) -> tuple[Array, Array, Array]:
    size = context.shape[0]
    updated = context.at[cursor].set(observation)
    next_cursor = jnp.mod(cursor + jnp.asarray(1, dtype=jnp.int32), size)
    next_count = jnp.minimum(count + jnp.asarray(1, dtype=jnp.int32), size)
    return updated, next_cursor, next_count


def _online_step(
    state: OnlineSIGRegState,
    inputs: tuple[Array, Array, Array],
    *,
    learner_config: _LearnerConfig,
    objective_enabled: bool,
    route_sigreg_gradient: bool,
) -> tuple[OnlineSIGRegState, _StepOutput]:
    """One causal transition; static arm flags alter only SIGReg routing/work."""

    observation, action, next_observation = inputs
    pre_step_count = state.step_count

    prediction = _predict_next_latent(state, observation, action)
    target = _encode(state.encoder_matrix, state.encoder_bias, next_observation)
    latent_prediction_mse = jnp.mean((prediction - target) ** 2)

    probe_features = jnp.concatenate(
        (
            _encode(state.encoder_matrix, state.encoder_bias, observation),
            jnp.ones((1,), dtype=jnp.float32),
        )
    )
    raw_probe_prediction = probe_features @ state.frozen_probe_matrix
    physical_probe_mse = jnp.where(
        state.frozen_probe_ready,
        jnp.mean((raw_probe_prediction[:PHYSICAL_DIM] - observation[:PHYSICAL_DIM]) ** 2),
        jnp.asarray(0.0, dtype=jnp.float32),
    )
    nuisance_probe_mse = jnp.where(
        state.frozen_probe_ready,
        jnp.mean((raw_probe_prediction[PHYSICAL_DIM:] - observation[PHYSICAL_DIM:]) ** 2),
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    _, prediction_gradients = jax.value_and_grad(
        _prediction_objective,
        argnums=(0, 1, 2, 3),
    )(
        state.encoder_matrix,
        state.encoder_bias,
        state.predictor_matrix,
        state.predictor_bias,
        observation,
        action,
        next_observation,
    )
    pred_matrix_grad, pred_bias_grad, predictor_matrix_grad, predictor_bias_grad = (
        prediction_gradients
    )
    prediction_candidates = (
        _bounded_candidate(
            state.encoder_matrix,
            pred_matrix_grad,
            learner_config.prediction_step_size,
            learner_config.max_parameter_update,
        ),
        _bounded_candidate(
            state.encoder_bias,
            pred_bias_grad,
            learner_config.prediction_step_size,
            learner_config.max_parameter_update,
        ),
        _bounded_candidate(
            state.predictor_matrix,
            predictor_matrix_grad,
            learner_config.prediction_step_size,
            learner_config.max_parameter_update,
        ),
        _bounded_candidate(
            state.predictor_bias,
            predictor_bias_grad,
            learner_config.prediction_step_size,
            learner_config.max_parameter_update,
        ),
    )
    prediction_update_valid = _all_finite((*prediction_gradients, *prediction_candidates))
    (
        post_prediction_encoder_matrix,
        post_prediction_encoder_bias,
        post_prediction_predictor_matrix,
        post_prediction_predictor_bias,
    ) = tuple(
        jnp.where(prediction_update_valid, candidate, current)
        for candidate, current in zip(
            prediction_candidates,
            (
                state.encoder_matrix,
                state.encoder_bias,
                state.predictor_matrix,
                state.predictor_bias,
            ),
            strict=True,
        )
    )

    context, context_cursor, context_count = _append_context(
        state.observation_context,
        state.context_cursor,
        state.context_count,
        next_observation,
    )
    context_available = context_count == learner_config.context_size
    fit_probe_now = context_available & ~state.frozen_probe_ready
    fitted_probe = jax.lax.cond(
        fit_probe_now,
        lambda: _fit_frozen_probe(
            post_prediction_encoder_matrix,
            post_prediction_encoder_bias,
            context,
            learner_config.probe_ridge,
        ),
        lambda: state.frozen_probe_matrix,
    )
    frozen_probe_matrix = jnp.where(
        fit_probe_now,
        fitted_probe,
        state.frozen_probe_matrix,
    )
    frozen_probe_ready = state.frozen_probe_ready | fit_probe_now

    if objective_enabled:

        def evaluate_sigreg() -> tuple[
            Array,
            tuple[Array, Array],
            Array,
            Array,
            Array,
        ]:
            loss, gradients = jax.value_and_grad(
                _sigreg_objective,
                argnums=(0, 1),
            )(
                post_prediction_encoder_matrix,
                post_prediction_encoder_bias,
                context,
                state.sigreg_directions,
            )
            matrix_gradient, bias_gradient = gradients
            candidates = (
                _bounded_candidate(
                    post_prediction_encoder_matrix,
                    matrix_gradient,
                    learner_config.sigreg_step_size,
                    learner_config.max_parameter_update,
                ),
                _bounded_candidate(
                    post_prediction_encoder_bias,
                    bias_gradient,
                    learner_config.sigreg_step_size,
                    learner_config.max_parameter_update,
                ),
            )
            gradient_norm, gradient_norm_finite = _finite_l2_norm(
                matrix_gradient,
                bias_gradient,
            )
            candidate_update_norm, candidate_norm_finite = _finite_l2_norm(
                candidates[0] - post_prediction_encoder_matrix,
                candidates[1] - post_prediction_encoder_bias,
            )
            finite = (
                jnp.isfinite(loss)
                & _all_finite((*gradients, *candidates))
                & gradient_norm_finite
                & candidate_norm_finite
            )
            gradient_norm = jnp.where(finite, gradient_norm, jnp.float32(0.0))
            candidate_update_norm = jnp.where(
                finite,
                candidate_update_norm,
                jnp.float32(0.0),
            )
            return (
                loss,
                candidates,
                finite,
                gradient_norm,
                candidate_update_norm,
            )

        (
            objective_loss,
            sigreg_candidates,
            sigreg_gradient_finite,
            sigreg_gradient_norm,
            sigreg_candidate_update_norm,
        ) = jax.lax.cond(
            context_available,
            evaluate_sigreg,
            lambda: (
                jnp.asarray(0.0, dtype=jnp.float32),
                (
                    post_prediction_encoder_matrix,
                    post_prediction_encoder_bias,
                ),
                jnp.asarray(True, dtype=jnp.bool_),
                jnp.asarray(0.0, dtype=jnp.float32),
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )
    else:
        objective_loss = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_candidates = (
            post_prediction_encoder_matrix,
            post_prediction_encoder_bias,
        )
        sigreg_gradient_finite = jnp.asarray(True, dtype=jnp.bool_)
        sigreg_gradient_norm = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_candidate_update_norm = jnp.asarray(0.0, dtype=jnp.float32)
    gradient_evaluated = context_available & objective_enabled
    gradient_routed = (
        gradient_evaluated & route_sigreg_gradient & sigreg_gradient_finite
    )
    gradient_discarded = (
        gradient_evaluated
        & (not route_sigreg_gradient)
        & sigreg_gradient_finite
    )
    gradient_rejected_nonfinite = gradient_evaluated & ~sigreg_gradient_finite
    final_encoder_matrix = jnp.where(
        gradient_routed,
        sigreg_candidates[0],
        post_prediction_encoder_matrix,
    )
    final_encoder_bias = jnp.where(
        gradient_routed,
        sigreg_candidates[1],
        post_prediction_encoder_bias,
    )

    def post_update_diagnostics() -> tuple[Array, Array, Array]:
        embeddings = jnp.tanh(context @ final_encoder_matrix + final_encoder_bias)
        latent_std = jnp.std(embeddings, axis=0)
        collapse_fraction = jnp.mean(
            (
                latent_std
                < jnp.asarray(learner_config.min_latent_std, dtype=jnp.float32)
            ).astype(jnp.float32)
        )
        return (
            sliced_sigreg_loss(embeddings, state.sigreg_directions),
            jnp.min(latent_std),
            collapse_fraction,
        )

    post_sigreg_loss, latent_std_min, collapsed_fraction = jax.lax.cond(
        context_available,
        post_update_diagnostics,
        lambda: (
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )

    post_step_count = state.step_count + jnp.asarray(1, dtype=jnp.int32)
    next_state = OnlineSIGRegState(
        encoder_matrix=final_encoder_matrix,
        encoder_bias=final_encoder_bias,
        predictor_matrix=post_prediction_predictor_matrix,
        predictor_bias=post_prediction_predictor_bias,
        observation_context=context,
        context_count=context_count,
        context_cursor=context_cursor,
        sigreg_directions=state.sigreg_directions,
        frozen_probe_matrix=frozen_probe_matrix,
        frozen_probe_ready=frozen_probe_ready,
        step_count=post_step_count,
    )
    return next_state, _StepOutput(
        latent_prediction_mse=latent_prediction_mse,
        physical_probe_mse=physical_probe_mse,
        nuisance_probe_mse=nuisance_probe_mse,
        probe_available=state.frozen_probe_ready,
        sigreg_objective_loss=objective_loss,
        sigreg_gradient_norm=sigreg_gradient_norm,
        sigreg_candidate_update_norm=sigreg_candidate_update_norm,
        post_update_sigreg_loss=post_sigreg_loss,
        post_update_latent_std_min=latent_std_min,
        post_update_collapsed_fraction=collapsed_fraction,
        sigreg_context_available=context_available,
        prediction_update_applied=prediction_update_valid,
        sigreg_gradient_evaluated=gradient_evaluated,
        sigreg_gradient_routed=gradient_routed,
        sigreg_gradient_discarded=gradient_discarded,
        sigreg_gradient_rejected_nonfinite=gradient_rejected_nonfinite,
        pre_step_count=pre_step_count,
        post_step_count=post_step_count,
    )


def _initial_state(
    config: OnlineSIGRegDevelopmentConfig,
    first_observation: Array,
) -> OnlineSIGRegState:
    encoder_key, direction_key = jr.split(jr.key(config.development_key))
    encoder_matrix = (
        jr.normal(
            encoder_key,
            (OBSERVATION_DIM, config.latent_dim),
            dtype=jnp.float32,
        )
        / jnp.sqrt(jnp.asarray(OBSERVATION_DIM, dtype=jnp.float32))
    )
    direction_samples = jr.normal(
        direction_key,
        (config.sigreg_projections, config.latent_dim),
        dtype=jnp.float32,
    )
    direction_norms = jnp.linalg.norm(direction_samples, axis=1, keepdims=True)
    directions = direction_samples / jnp.maximum(
        direction_norms,
        jnp.asarray(1.0e-8, dtype=jnp.float32),
    )
    context = jnp.zeros((config.context_size, OBSERVATION_DIM), dtype=jnp.float32)
    context = context.at[0].set(first_observation)
    return OnlineSIGRegState(
        encoder_matrix=encoder_matrix,
        encoder_bias=jnp.zeros((config.latent_dim,), dtype=jnp.float32),
        predictor_matrix=jnp.zeros(
            (config.latent_dim + N_ACTIONS, config.latent_dim),
            dtype=jnp.float32,
        ),
        predictor_bias=jnp.zeros((config.latent_dim,), dtype=jnp.float32),
        observation_context=context,
        context_count=jnp.asarray(1, dtype=jnp.int32),
        context_cursor=jnp.asarray(1 % config.context_size, dtype=jnp.int32),
        sigreg_directions=directions,
        frozen_probe_matrix=jnp.zeros(
            (config.latent_dim + 1, OBSERVATION_DIM),
            dtype=jnp.float32,
        ),
        frozen_probe_ready=jnp.asarray(False, dtype=jnp.bool_),
        step_count=jnp.asarray(0, dtype=jnp.int32),
    )


def _state_arrays(state: OnlineSIGRegState) -> tuple[tuple[str, Array], ...]:
    return tuple(
        (field.name, cast(Array, getattr(state, field.name)))
        for field in dataclasses.fields(OnlineSIGRegState)  # type: ignore[arg-type]
    )


def _state_sha256(state: OnlineSIGRegState, *, learner_only: bool = False) -> str:
    names = {
        "encoder_matrix",
        "encoder_bias",
        "predictor_matrix",
        "predictor_bias",
    }
    arrays = tuple(
        (name, value)
        for name, value in _state_arrays(state)
        if not learner_only or name in names
    )
    prefix = "online-sigreg-learner-v1" if learner_only else "online-sigreg-state-v1"
    return _array_manifest_sha256(arrays, prefix=prefix)


def _array_nbytes(array: Array) -> int:
    return int(array.size) * int(array.dtype.itemsize)


def _measure_resources(
    initial_state: OnlineSIGRegState,
    final_state: OnlineSIGRegState,
    trajectory: PrequentialTrajectory,
    source: OnlineSIGRegSource,
) -> PersistentResourceSummary:
    learner = sum(
        _array_nbytes(value)
        for name, value in _state_arrays(initial_state)
        if name
        in {"encoder_matrix", "encoder_bias", "predictor_matrix", "predictor_bias"}
    )
    context = _array_nbytes(initial_state.observation_context)
    directions = _array_nbytes(initial_state.sigreg_directions)
    probe = _array_nbytes(initial_state.frozen_probe_matrix)
    controls = sum(
        _array_nbytes(value)
        for name, value in _state_arrays(initial_state)
        if name in {"context_count", "context_cursor", "frozen_probe_ready", "step_count"}
    )
    initial_total = sum(_array_nbytes(value) for _, value in _state_arrays(initial_state))
    final_total = sum(_array_nbytes(value) for _, value in _state_arrays(final_state))
    component_sum = learner + context + directions + probe + controls
    trajectory_output = sum(
        _array_nbytes(cast(Array, getattr(trajectory, field.name)))
        for field in dataclasses.fields(PrequentialTrajectory)
    )
    source_input = sum(
        _array_nbytes(value)
        for value in (
            source.observations,
            source.actions,
            source.next_observations,
        )
    )
    return PersistentResourceSummary(
        accounting_scope=(
            "exact JAX array payload bytes; excludes Python objects, compiler buffers, "
            "allocator peaks, and shared-runtime overhead"
        ),
        initial_total_nbytes=initial_total,
        final_total_nbytes=final_total,
        learner_parameter_nbytes=learner,
        observation_context_nbytes=context,
        sigreg_direction_nbytes=directions,
        frozen_probe_nbytes=probe,
        scalar_control_nbytes=controls,
        exact_component_sum_matches_total=(
            component_sum == initial_total == final_total
        ),
        trajectory_output_nbytes=trajectory_output,
        shared_source_input_nbytes=source_input,
        retained_report_array_nbytes_including_shared_source=(
            trajectory_output + source_input
        ),
    )


def _logical_tensor(
    name: str,
    shape: tuple[int, ...],
    evaluations: int,
    timing: str,
) -> LogicalTransientTensor:
    elements = math.prod(shape)
    nbytes = 4 * elements
    return LogicalTransientTensor(
        name=name,
        shape=shape,
        dtype="float32",
        nbytes_per_evaluation=nbytes,
        evaluations=evaluations,
        total_named_payload_nbytes=nbytes * evaluations,
        timing=timing,
    )


def _work_summary(
    config: OnlineSIGRegDevelopmentConfig,
    trajectory: PrequentialTrajectory,
    *,
    objective_enabled: bool,
) -> WorkSummary:
    total = config.total_steps
    objective_evaluations = int(
        np.sum(np.asarray(trajectory.sigreg_gradient_evaluated, dtype=np.int32))
    )
    routed = int(
        np.sum(np.asarray(trajectory.sigreg_gradient_routed, dtype=np.int32))
    )
    discarded = int(
        np.sum(np.asarray(trajectory.sigreg_gradient_discarded, dtype=np.int32))
    )
    rejected_nonfinite = int(
        np.sum(
            np.asarray(
                trajectory.sigreg_gradient_rejected_nonfinite,
                dtype=np.int32,
            )
        )
    )
    post_diagnostics = int(
        np.sum(np.asarray(trajectory.sigreg_context_available, dtype=np.int32))
    )
    parameter_elements = (
        OBSERVATION_DIM * config.latent_dim
        + config.latent_dim
        + (config.latent_dim + N_ACTIONS) * config.latent_dim
        + config.latent_dim
    )
    encoder_elements = OBSERVATION_DIM * config.latent_dim + config.latent_dim
    tensors = [
        _logical_tensor(
            "prediction_parameter_gradient",
            (parameter_elements,),
            total,
            "post_outcome_prediction_update",
        ),
        _logical_tensor(
            "post_update_sigreg_embeddings",
            (config.context_size, config.latent_dim),
            post_diagnostics,
            "post_update_diagnostic",
        ),
        _logical_tensor(
            "post_update_sigreg_projections",
            (config.context_size, config.sigreg_projections),
            post_diagnostics,
            "post_update_diagnostic",
        ),
        _logical_tensor(
            "post_update_sigreg_pairwise_differences",
            (config.sigreg_projections, config.context_size, config.context_size),
            post_diagnostics,
            "post_update_diagnostic",
        ),
        _logical_tensor(
            "post_update_sigreg_pairwise_kernel",
            (config.sigreg_projections, config.context_size, config.context_size),
            post_diagnostics,
            "post_update_diagnostic",
        ),
        _logical_tensor(
            "frozen_probe_design",
            (config.context_size, config.latent_dim + 1),
            1,
            "one_time_post_outcome_probe_fit",
        ),
        _logical_tensor(
            "frozen_probe_gram",
            (config.latent_dim + 1, config.latent_dim + 1),
            1,
            "one_time_post_outcome_probe_fit",
        ),
        _logical_tensor(
            "frozen_probe_rhs",
            (config.latent_dim + 1, OBSERVATION_DIM),
            1,
            "one_time_post_outcome_probe_fit",
        ),
    ]
    if objective_enabled:
        tensors.extend(
            (
                _logical_tensor(
                    "sigreg_objective_embeddings",
                    (config.context_size, config.latent_dim),
                    objective_evaluations,
                    "post_outcome_pre_route_objective",
                ),
                _logical_tensor(
                    "sigreg_objective_projections",
                    (config.context_size, config.sigreg_projections),
                    objective_evaluations,
                    "post_outcome_pre_route_objective",
                ),
                _logical_tensor(
                    "sigreg_objective_pairwise_differences",
                    (
                        config.sigreg_projections,
                        config.context_size,
                        config.context_size,
                    ),
                    objective_evaluations,
                    "post_outcome_pre_route_objective",
                ),
                _logical_tensor(
                    "sigreg_objective_pairwise_kernel",
                    (
                        config.sigreg_projections,
                        config.context_size,
                        config.context_size,
                    ),
                    objective_evaluations,
                    "post_outcome_pre_route_objective",
                ),
                _logical_tensor(
                    "sigreg_encoder_gradient",
                    (encoder_elements,),
                    objective_evaluations,
                    "post_outcome_pre_route_objective",
                ),
                _logical_tensor(
                    "sigreg_encoder_candidate",
                    (encoder_elements,),
                    objective_evaluations,
                    "post_outcome_pre_route_candidate",
                ),
            )
        )
    frozen_tensors = tuple(tensors)
    return WorkSummary(
        transitions_consumed_once=total,
        transition_replays=0,
        prediction_gradient_evaluations=total,
        sigreg_objective_gradient_evaluations=objective_evaluations,
        sigreg_gradient_candidates=objective_evaluations,
        sigreg_gradients_routed=routed,
        sigreg_gradients_discarded=discarded,
        sigreg_gradients_rejected_nonfinite=rejected_nonfinite,
        post_update_sigreg_diagnostics=post_diagnostics,
        past_observations_reembedded_for_sigreg_objective=(
            config.context_size * objective_evaluations
        ),
        frozen_probe_fits=1,
        logical_transient_tensors=frozen_tensors,
        logical_transient_payload_nbytes=sum(
            tensor.total_named_payload_nbytes for tensor in frozen_tensors
        ),
        allocator_peak_assessed=False,
        wall_clock_matched=False,
    )


def _masked_mean(values: Array, mask: Array, start: int, stop: int) -> float:
    selected_values = np.asarray(values[start:stop], dtype=np.float64)
    selected_mask = np.asarray(mask[start:stop], dtype=np.bool_)
    if not np.any(selected_mask):
        raise RuntimeError("a configured summary window has no causal measurements")
    return float(np.mean(selected_values[selected_mask]))


def _phase_metrics(
    trajectory: PrequentialTrajectory,
    config: OnlineSIGRegDevelopmentConfig,
) -> tuple[PhaseMetrics, ...]:
    summaries: list[PhaseMetrics] = []
    for index, name in enumerate(PHASE_NAMES):
        start = index * config.phase_steps
        stop = (index + 1) * config.phase_steps
        probe_mask = trajectory.probe_available
        sigreg_mask = trajectory.sigreg_context_available
        summaries.append(
            PhaseMetrics(
                name=name,
                start=start,
                stop=stop,
                latent_prediction_mse=float(
                    np.mean(np.asarray(trajectory.latent_prediction_mse[start:stop]))
                ),
                physical_probe_mse=_masked_mean(
                    trajectory.physical_probe_mse,
                    probe_mask,
                    start,
                    stop,
                ),
                nuisance_probe_mse=_masked_mean(
                    trajectory.nuisance_probe_mse,
                    probe_mask,
                    start,
                    stop,
                ),
                post_update_sigreg_loss=_masked_mean(
                    trajectory.post_update_sigreg_loss,
                    sigreg_mask,
                    start,
                    stop,
                ),
                post_update_latent_std_min=_masked_mean(
                    trajectory.post_update_latent_std_min,
                    sigreg_mask,
                    start,
                    stop,
                ),
                post_update_collapsed_fraction=_masked_mean(
                    trajectory.post_update_collapsed_fraction,
                    sigreg_mask,
                    start,
                    stop,
                ),
                probe_measurements=int(
                    np.sum(np.asarray(probe_mask[start:stop], dtype=np.int32))
                ),
                sigreg_measurements=int(
                    np.sum(np.asarray(sigreg_mask[start:stop], dtype=np.int32))
                ),
            )
        )
    return tuple(summaries)


def _recurrence_windows(
    values: Array,
    mask: Array,
    config: OnlineSIGRegDevelopmentConfig,
) -> WindowRecurrence:
    window = config.summary_window
    initial = _masked_mean(
        values,
        mask,
        config.phase_steps - window,
        config.phase_steps,
    )
    entry = _masked_mean(
        values,
        mask,
        2 * config.phase_steps,
        2 * config.phase_steps + window,
    )
    late = _masked_mean(
        values,
        mask,
        3 * config.phase_steps - window,
        3 * config.phase_steps,
    )
    return WindowRecurrence(
        initial_a_late=initial,
        recurrence_a_entry=entry,
        recurrence_a_late=late,
        entry_minus_initial=entry - initial,
        entry_minus_late=entry - late,
        late_minus_initial=late - initial,
    )


def _recurrence_summary(
    trajectory: PrequentialTrajectory,
    config: OnlineSIGRegDevelopmentConfig,
) -> RecurrenceSummary:
    all_valid = jnp.ones((config.total_steps,), dtype=jnp.bool_)
    return RecurrenceSummary(
        latent_prediction_mse=_recurrence_windows(
            trajectory.latent_prediction_mse,
            all_valid,
            config,
        ),
        physical_probe_mse=_recurrence_windows(
            trajectory.physical_probe_mse,
            trajectory.probe_available,
            config,
        ),
        nuisance_probe_mse=_recurrence_windows(
            trajectory.nuisance_probe_mse,
            trajectory.probe_available,
            config,
        ),
        latent_std_min=_recurrence_windows(
            trajectory.post_update_latent_std_min,
            trajectory.sigreg_context_available,
            config,
        ),
        collapsed_fraction=_recurrence_windows(
            trajectory.post_update_collapsed_fraction,
            trajectory.sigreg_context_available,
            config,
        ),
    )


_LEARNER_TRAJECTORY_FIELDS: Final = (
    "latent_prediction_mse",
    "physical_probe_mse",
    "nuisance_probe_mse",
    "probe_available",
    "post_update_sigreg_loss",
    "post_update_latent_std_min",
    "post_update_collapsed_fraction",
    "sigreg_context_available",
    "prediction_update_applied",
    "pre_step_count",
    "post_step_count",
)


def _learner_trajectories_equal(
    left: PrequentialTrajectory,
    right: PrequentialTrajectory,
) -> bool:
    return all(
        _exact_value_equal(getattr(left, field), getattr(right, field))
        for field in _LEARNER_TRAJECTORY_FIELDS
    )


def _run_arm(
    name: ArmName,
    config: OnlineSIGRegDevelopmentConfig,
    source: OnlineSIGRegSource,
    initial_state: OnlineSIGRegState,
) -> tuple[ArmReport, OnlineSIGRegState]:
    objective_enabled = name != PREDICTION_ONLY
    route_sigreg_gradient = name == SIGREG_ROUTED
    learner_config = _learner_config(config)

    def scan_step(
        state: OnlineSIGRegState,
        inputs: tuple[Array, Array, Array],
    ) -> tuple[OnlineSIGRegState, _StepOutput]:
        return _online_step(
            state,
            inputs,
            learner_config=learner_config,
            objective_enabled=objective_enabled,
            route_sigreg_gradient=route_sigreg_gradient,
        )

    final_state, outputs = jax.lax.scan(
        scan_step,
        initial_state,
        (source.observations, source.actions, source.next_observations),
    )
    trajectory = PrequentialTrajectory(*outputs)
    frozen_probe_sha256 = _array_manifest_sha256(
        (("frozen_probe_matrix", final_state.frozen_probe_matrix),),
        prefix="online-sigreg-frozen-prefix-probe-v1",
    )
    report = ArmReport(
        name=name,
        sigreg_objective_enabled=objective_enabled,
        sigreg_gradient_routed_by_design=route_sigreg_gradient,
        source_input_sha256=source.input_sha256,
        initial_state_sha256=_state_sha256(initial_state),
        frozen_probe_sha256=frozen_probe_sha256,
        final_learner_sha256=_state_sha256(final_state, learner_only=True),
        resource=_measure_resources(
            initial_state,
            final_state,
            trajectory,
            source,
        ),
        work=_work_summary(
            config,
            trajectory,
            objective_enabled=objective_enabled,
        ),
        trajectory=trajectory,
        phase_metrics=_phase_metrics(trajectory, config),
        recurrence=_recurrence_summary(trajectory, config),
    )
    return report, final_state


def _timing_contract() -> TimingOwnership:
    return TimingOwnership(
        pre_outcome_prediction_inputs=("observation_t", "action_t", "pre_update_state_t"),
        outcome_revealed_after_prediction="next_observation_t is scored only after prediction",
        predictor_update_inputs=("observation_t", "action_t", "next_observation_t"),
        regularizer_context=(
            "fixed FIFO of raw observations through next_observation_t; no actions, transitions, "
            "targets, rewards, phase IDs, or future observations"
        ),
        regularizer_update_effective=(
            "routed gradient changes transition t's post-update state and same-context "
            "SIGReg diagnostic; it first changes pre-outcome prediction/probe at event t+1"
        ),
        frozen_probe_fit=(
            "fit once after the first full context, before the first routed SIGReg update"
        ),
        frozen_probe_measurement=(
            "pre-outcome current observation; probe value and gradient never reach the learner"
        ),
        evaluator_only_segmentation="A/B/A boundaries are applied only after the life",
    )


def _build_deterministic_report(
    cfg: OnlineSIGRegDevelopmentConfig,
) -> OnlineSIGRegDevelopmentReport:
    """Build one report without calling the public report validator."""

    source = build_online_sigreg_source(cfg)
    source_errors = validate_online_sigreg_source(source)
    if source_errors:
        raise RuntimeError(f"source reconstruction failed: {source_errors!r}")
    initial_state = _initial_state(cfg, source.observations[0])

    reports: list[ArmReport] = []
    final_states: dict[ArmName, OnlineSIGRegState] = {}
    for name in ARM_ORDER:
        arm_report, final_state = _run_arm(name, cfg, source, initial_state)
        reports.append(arm_report)
        final_states[name] = final_state
    arms = tuple(reports)

    common_initial = all(
        arm.initial_state_sha256 == arms[0].initial_state_sha256 for arm in arms
    )
    resources_equal = all(arm.resource == arms[0].resource for arm in arms[1:])
    active_work = arms[2].work
    inert_work = arms[1].work
    objective_compute_matched = (
        active_work.sigreg_objective_gradient_evaluations
        == inert_work.sigreg_objective_gradient_evaluations
        and active_work.sigreg_gradient_candidates == inert_work.sigreg_gradient_candidates
        and active_work.past_observations_reembedded_for_sigreg_objective
        == inert_work.past_observations_reembedded_for_sigreg_objective
        and tuple(
            (tensor.name, tensor.shape, tensor.evaluations)
            for tensor in active_work.logical_transient_tensors
        )
        == tuple(
            (tensor.name, tensor.shape, tensor.evaluations)
            for tensor in inert_work.logical_transient_tensors
        )
    )
    prediction_state = final_states[PREDICTION_ONLY]
    inert_state = final_states[SIGREG_INERT]
    inert_matches_prediction = (
        arms[0].final_learner_sha256 == arms[1].final_learner_sha256
        and _learner_trajectories_equal(
            arms[0].trajectory,
            arms[1].trajectory,
        )
        and all(
            _arrays_bit_equal(left, right)
            for left, right in zip(
                (
                    prediction_state.encoder_matrix,
                    prediction_state.encoder_bias,
                    prediction_state.predictor_matrix,
                    prediction_state.predictor_bias,
                ),
                (
                    inert_state.encoder_matrix,
                    inert_state.encoder_bias,
                    inert_state.predictor_matrix,
                    inert_state.predictor_bias,
                ),
                strict=True,
            )
        )
    )
    return OnlineSIGRegDevelopmentReport(
        schema=DEVELOPMENT_SCHEMA,
        evidence_level=EVIDENCE_LEVEL,
        status="development_only_descriptive_not_assessed",
        development_only=DEVELOPMENT_ONLY,
        assessment_status=ASSESSMENT_STATUS,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        output_writes_allowed=OUTPUT_WRITES_ALLOWED,
        evidence_claimed=EVIDENCE_CLAIMED,
        thresholds_frozen=THRESHOLDS_FROZEN,
        development_keys_frozen=DEVELOPMENT_KEYS_FROZEN,
        phase_identifiers_exposed=PHASE_IDENTIFIERS_EXPOSED,
        transition_replayed=TRANSITION_REPLAYED,
        descriptive_claims_only=True,
        config=cfg,
        source=source,
        timing=_timing_contract(),
        arms=arms,
        common_initial_state=common_initial,
        persistent_resources_equal=resources_equal,
        routed_and_inert_objective_compute_matched=objective_compute_matched,
        inert_and_prediction_only_learners_identical=inert_matches_prediction,
        limitations=_LIMITATIONS,
    )


def run_online_sigreg_objective_development(
    config: OnlineSIGRegDevelopmentConfig | None = None,
) -> OnlineSIGRegDevelopmentReport:
    """Run the three-arm in-memory L0 comparison."""

    if config is None:
        cfg = OnlineSIGRegDevelopmentConfig()
    elif type(config) is OnlineSIGRegDevelopmentConfig:
        cfg = config
    else:
        raise TypeError("config must be an exact OnlineSIGRegDevelopmentConfig or None")
    return _build_deterministic_report(cfg)


def validate_online_sigreg_development_report(report: object) -> tuple[str, ...]:
    """Reconstruct every report value and validate nonauthority, never efficacy."""

    if type(report) is not OnlineSIGRegDevelopmentReport:
        return ("report must be an exact OnlineSIGRegDevelopmentReport",)
    errors: list[str] = []
    report_config_errors = _validate_config_instance(report.config)
    if report_config_errors:
        return tuple(f"report {error}" for error in report_config_errors)
    if type(report.source) is not OnlineSIGRegSource:
        return ("report source must be an exact OnlineSIGRegSource",)
    source_config_errors = _validate_config_instance(report.source.config)
    if source_config_errors:
        return tuple(f"report source {error}" for error in source_config_errors)
    errors.extend(validate_online_sigreg_source(report.source))

    if type(report.arms) is not tuple or len(report.arms) != len(ARM_ORDER):
        errors.append("arms must be the exact three-arm tuple")
        return tuple(errors)
    if any(type(arm) is not ArmReport for arm in report.arms):
        errors.append("every arm must be an exact ArmReport")
        return tuple(errors)
    if type(report.timing) is not TimingOwnership:
        errors.append("timing must be an exact TimingOwnership")
    if type(report.limitations) is not tuple or any(
        type(limitation) is not str for limitation in report.limitations
    ):
        errors.append("limitations must be an exact tuple of strings")

    if _config_sha256(report.config) != _config_sha256(report.source.config):
        errors.append("report config is not canonically bound to source config")

    expected = _build_deterministic_report(report.config)
    authority_fields = (
        "schema",
        "evidence_level",
        "status",
        "development_only",
        "assessment_status",
        "scientific_promotion_allowed",
        "output_writes_allowed",
        "evidence_claimed",
        "thresholds_frozen",
        "development_keys_frozen",
        "phase_identifiers_exposed",
        "transition_replayed",
        "descriptive_claims_only",
    )
    if any(
        not _exact_value_equal(
            getattr(report, field),
            getattr(expected, field),
        )
        for field in authority_fields
    ):
        errors.append("L0 development nonauthority contract changed")
    if not _exact_value_equal(report.config, expected.config):
        errors.append("report config does not exactly reconstruct")
    if not _exact_value_equal(report.source, expected.source):
        errors.append("report source does not exactly reconstruct")
    if not _exact_value_equal(report.timing, expected.timing):
        errors.append("timing ownership does not exactly reconstruct")
    if not _exact_value_equal(report.limitations, expected.limitations):
        errors.append("limitations do not exactly reconstruct")

    if tuple(arm.name for arm in report.arms) != ARM_ORDER:
        errors.append("arm order or membership changed")
    identity_fields = (
        "name",
        "sigreg_objective_enabled",
        "sigreg_gradient_routed_by_design",
        "source_input_sha256",
        "initial_state_sha256",
        "frozen_probe_sha256",
        "final_learner_sha256",
    )
    for arm, expected_arm in zip(report.arms, expected.arms, strict=True):
        if any(
            not _exact_value_equal(
                getattr(arm, field),
                getattr(expected_arm, field),
            )
            for field in identity_fields
        ):
            errors.append(f"{expected_arm.name} identity or hash binding changed")
        if not _exact_value_equal(arm.resource, expected_arm.resource):
            errors.append(f"{expected_arm.name} resource accounting does not reconstruct")
        if not _exact_value_equal(arm.work, expected_arm.work):
            errors.append(f"{expected_arm.name} work accounting does not reconstruct")
        if not _exact_value_equal(arm.trajectory, expected_arm.trajectory):
            errors.append(f"{expected_arm.name} raw trajectory does not reconstruct")
        if not _exact_value_equal(arm.phase_metrics, expected_arm.phase_metrics):
            errors.append(f"{expected_arm.name} phase metrics do not reconstruct")
        if not _exact_value_equal(arm.recurrence, expected_arm.recurrence):
            errors.append(f"{expected_arm.name} recurrence metrics do not reconstruct")

    aggregate_fields = (
        "common_initial_state",
        "persistent_resources_equal",
        "routed_and_inert_objective_compute_matched",
        "inert_and_prediction_only_learners_identical",
    )
    for field in aggregate_fields:
        if not _exact_value_equal(getattr(report, field), getattr(expected, field)):
            errors.append(f"{field} does not reconstruct")

    prediction_arm, inert_arm, _ = report.arms
    inert_identity = (
        prediction_arm.final_learner_sha256 == inert_arm.final_learner_sha256
        and _learner_trajectories_equal(
            prediction_arm.trajectory,
            inert_arm.trajectory,
        )
    )
    if (
        report.inert_and_prediction_only_learners_identical is not True
        or not inert_identity
    ):
        errors.append("inert and prediction-only identity does not hold")
    return tuple(errors)


__all__ = [
    "ARM_ORDER",
    "ASSESSMENT_STATUS",
    "DEVELOPMENT_KEYS_FROZEN",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SCHEMA",
    "EVIDENCE_CLAIMED",
    "EVIDENCE_LEVEL",
    "OUTPUT_WRITES_ALLOWED",
    "PHASE_IDENTIFIERS_EXPOSED",
    "PREDICTION_ONLY",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SIGREG_INERT",
    "SIGREG_ROUTED",
    "THRESHOLDS_FROZEN",
    "TRANSITION_REPLAYED",
    "ArmReport",
    "LogicalTransientTensor",
    "OnlineSIGRegDevelopmentConfig",
    "OnlineSIGRegDevelopmentReport",
    "OnlineSIGRegSource",
    "PersistentResourceSummary",
    "PrequentialTrajectory",
    "RecurrenceSummary",
    "TimingOwnership",
    "WindowRecurrence",
    "WorkSummary",
    "build_online_sigreg_source",
    "run_online_sigreg_objective_development",
    "validate_online_sigreg_development_report",
    "validate_online_sigreg_source",
]
