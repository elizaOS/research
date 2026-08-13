"""Development-only optimization-centric plasticity diagnostics.

This evaluator owns one uninterrupted ``A/B/A`` regression stream and runs
two matched copies of a bounded nonlinear learner.  The copies share an exact
initial snapshot, observations, scalar targets, update opportunities, and
diagnostic perturbation directions.  Their only declared difference is an
initialization-centred L2 parameter constraint in the second condition.

At each task switch the report retains bit-exact old-task and incoming-task
gradients, fixed-radius local loss probes, a separate hidden-activation
dormancy measurement, and the learner snapshot on which every measurement was
made.  The zero-gradient/local-neighbourhood flag is a fixed descriptive rule
with a declared numeric floor; it is not a calibrated threshold, an efficacy
verdict, or an evidence gate.  Dormancy is reported alongside that rule and is
never one of its inputs.

Reports are in-memory L0 development records with strict config, executed
source, and observable runtime identity.  They are reconstructable by exact
causal replay and are always ``not_assessed``.  This module has no output
writer, acceptance rule, evidence authority, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

CONFIG_SCHEMA: Final = "alberta.optimization-centric-plasticity.config.v1"
SNAPSHOT_SCHEMA: Final = "alberta.optimization-centric-plasticity.snapshot.v1"
REPORT_SCHEMA: Final = "alberta.optimization-centric-plasticity.report.v1"
PROTOCOL_ID: Final = "alberta.optimization-centric-plasticity.protocol.v1"
DEVELOPMENT_STATUS: Final = "DEVELOPMENT_ONLY_DESCRIPTIVE_NO_EVIDENCE_OR_PROMOTION"
ASSESSMENT_STATUS: Final = "not_assessed"

UNCONSTRAINED: Final = "unconstrained"
L2_CONSTRAINED: Final = "l2_constrained"

OUTPUT_WRITES_AUTHORIZED: Final = False
THRESHOLDS_AUTHORIZED: Final = False
EVIDENCE_AUTHORIZED: Final = False
PROMOTION_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False

_SEED: Final = 6201
_PHASE_ORDER: Final = ("A", "B", "A")
_UPDATES_PER_PHASE: Final = 4
_INPUT_DIM: Final = 2
_HIDDEN_DIM: Final = 3
_PARAMETER_COUNT: Final = _INPUT_DIM * _HIDDEN_DIM + 2 * _HIDDEN_DIM + 1
_LEARNING_RATE: Final = 0.20
_CONSTRAINT_RADIUS: Final = 0.10
_ZERO_GRADIENT_DESCRIPTIVE_FLOOR: Final = 1.0e-8
_ALIGNMENT_DESCRIPTIVE_FLOOR: Final = 1.0e-12
_LOCAL_PROBE_RADIUS: Final = 0.025
_LOCAL_LOSS_DESCRIPTIVE_TOLERANCE: Final = 1.0e-7
_DORMANT_ACTIVATION_DESCRIPTIVE_CUTOFF: Final = 0.05
_LOCAL_DIRECTION_COUNT: Final = 2

_INITIALIZATION_DOMAIN: Final = 0x494E4954  # INIT
_DIRECTION_DOMAIN: Final = 0x44495253  # DIRS
_UINT32_MAX: Final = 2**32 - 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS: Final = (
    Path("alberta_framework/evaluation/optimization_centric_plasticity_development.py"),
    Path("pyproject.toml"),
)

_PROBE_OBSERVATIONS: Final = (
    (-1.0, -1.0),
    (-1.0, 1.0),
    (1.0, -1.0),
    (1.0, 1.0),
)
_TARGETS_A: Final = (1.0, -1.0, -1.0, 1.0)
_TARGETS_B: Final = (-1.0, -1.0, 1.0, 1.0)
_TARGET_MANIFEST: Final = (
    ("A", _TARGETS_A),
    ("B", _TARGETS_B),
)

type ConstraintMode = Literal["unconstrained", "l2_constrained"]
type GradientAlignment = Literal["aligned", "conflicting", "orthogonal", "undefined"]


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _dataclass_payload(value: object) -> object:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance")
    return dataclasses.asdict(value)


def _dataclass_digest(value: object) -> str:
    return _digest(_dataclass_payload(value))


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise ValueError(f"{name} must be an exact bounded integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _float32_bits(values: object) -> tuple[int, ...]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1))
    return tuple(int(word) for word in array.view(np.uint32))


def _scalar_float32_bits(value: object) -> int:
    words = _float32_bits(value)
    if len(words) != 1:
        raise ValueError("value must contain exactly one float32 scalar")
    return words[0]


def _bits_to_float32(words: tuple[int, ...]) -> np.ndarray:
    for index, word in enumerate(words):
        _exact_int(word, name=f"float32_bits[{index}]", maximum=_UINT32_MAX)
    return np.asarray(words, dtype=np.uint32).view(np.float32).copy()


def _scalar_from_float32_bits(word: int) -> float:
    return float(_bits_to_float32((word,))[0])


def _key_words(key: Array) -> tuple[int, int]:
    raw = np.asarray(jr.key_data(key), dtype=np.uint32).reshape(-1)
    if raw.shape != (2,):
        raise RuntimeError("typed PRNG key must contain exactly two uint32 words")
    return int(raw[0]), int(raw[1])


def _targets(regime_id: str) -> tuple[float, ...]:
    if regime_id == "A":
        return _TARGETS_A
    if regime_id == "B":
        return _TARGETS_B
    raise ValueError("unknown evaluator regime")


_TARGET_MANIFEST_SHA256: Final = _digest(
    {
        "observations": _PROBE_OBSERVATIONS,
        "targets": _TARGET_MANIFEST,
        "learner_visible_regime_metadata": False,
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizationCentricPlasticityDevelopmentConfig:
    """Fully frozen, finite descriptive protocol configuration."""

    seed: int = _SEED
    phase_order: tuple[str, str, str] = _PHASE_ORDER
    updates_per_phase: int = _UPDATES_PER_PHASE
    input_dim: int = _INPUT_DIM
    hidden_dim: int = _HIDDEN_DIM
    learning_rate: float = _LEARNING_RATE
    constraint_radius: float = _CONSTRAINT_RADIUS
    zero_gradient_descriptive_floor: float = _ZERO_GRADIENT_DESCRIPTIVE_FLOOR
    alignment_descriptive_floor: float = _ALIGNMENT_DESCRIPTIVE_FLOOR
    local_probe_radius: float = _LOCAL_PROBE_RADIUS
    local_loss_descriptive_tolerance: float = _LOCAL_LOSS_DESCRIPTIVE_TOLERANCE
    dormant_activation_descriptive_cutoff: float = (
        _DORMANT_ACTIVATION_DESCRIPTIVE_CUTOFF
    )
    local_direction_count: int = _LOCAL_DIRECTION_COUNT
    target_manifest_sha256: str = _TARGET_MANIFEST_SHA256
    dtype: str = "float32"
    architecture: str = "two_input_three_tanh_hidden_one_tanh_output_regressor"
    loss: str = "mean_half_squared_error"
    protocol_id: str = PROTOCOL_ID
    schema: str = CONFIG_SCHEMA

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "seed": _SEED,
            "phase_order": _PHASE_ORDER,
            "updates_per_phase": _UPDATES_PER_PHASE,
            "input_dim": _INPUT_DIM,
            "hidden_dim": _HIDDEN_DIM,
            "learning_rate": _LEARNING_RATE,
            "constraint_radius": _CONSTRAINT_RADIUS,
            "zero_gradient_descriptive_floor": _ZERO_GRADIENT_DESCRIPTIVE_FLOOR,
            "alignment_descriptive_floor": _ALIGNMENT_DESCRIPTIVE_FLOOR,
            "local_probe_radius": _LOCAL_PROBE_RADIUS,
            "local_loss_descriptive_tolerance": _LOCAL_LOSS_DESCRIPTIVE_TOLERANCE,
            "dormant_activation_descriptive_cutoff": (
                _DORMANT_ACTIVATION_DESCRIPTIVE_CUTOFF
            ),
            "local_direction_count": _LOCAL_DIRECTION_COUNT,
            "target_manifest_sha256": _TARGET_MANIFEST_SHA256,
            "dtype": "float32",
            "architecture": "two_input_three_tanh_hidden_one_tanh_output_regressor",
            "loss": "mean_half_squared_error",
            "protocol_id": PROTOCOL_ID,
            "schema": CONFIG_SCHEMA,
        }
        for name, expected_value in expected.items():
            actual = getattr(self, name)
            if type(actual) is not type(expected_value) or actual != expected_value:
                raise ValueError(f"{name} is frozen at {expected_value!r}")

    @property
    def parameter_count(self) -> int:
        return self.input_dim * self.hidden_dim + 2 * self.hidden_dim + 1

    @property
    def phase_count(self) -> int:
        return len(self.phase_order)

    @property
    def switch_count(self) -> int:
        return len(self.phase_order) - 1

    @property
    def protocol_step_count(self) -> int:
        return self.phase_count * self.updates_per_phase


@dataclasses.dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    path: str
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    identity_scope: str
    python_implementation: str
    python_version: str
    python_compiler: str
    operating_system: str
    operating_system_release: str
    machine: str
    byteorder: str
    jax_version: str
    jaxlib_version: str
    numpy_version: str
    default_backend: str
    device_count: int
    local_device_count: int
    device_platforms: tuple[str, ...]
    device_kinds: tuple[str, ...]
    jax_enable_x64: bool
    jax_default_prng_impl: str
    jax_threefry_partitionable: bool
    jax_disable_jit: bool


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenLearnerSnapshot:
    """Immutable bit-exact learner parameter receipt."""

    schema: str
    config_sha256: str
    revision: int
    parameter_count: int
    parameter_float32_bits: tuple[int, ...]
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ProtocolStep:
    step_index: int
    phase_index: int
    phase_step: int
    evaluator_regime_id: str
    phase_boundary: bool
    observation_float32_bits: tuple[int, int]
    target_float32_bits: int


@dataclasses.dataclass(frozen=True, slots=True)
class SwitchProbeManifest:
    switch_index: int
    old_evaluator_regime_id: str
    incoming_evaluator_regime_id: str
    direction_key_words_uint32: tuple[int, int]
    direction_float32_bits: tuple[tuple[int, ...], ...]
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class GradientMeasurement:
    loss_float32_bits: int
    gradient_float32_bits: tuple[int, ...]
    l2_norm: float
    max_absolute_component: float
    nonzero_component_count: int

    @property
    def loss(self) -> float:
        return _scalar_from_float32_bits(self.loss_float32_bits)

    def to_float32(self) -> tuple[float, ...]:
        return tuple(float(value) for value in _bits_to_float32(self.gradient_float32_bits))


@dataclasses.dataclass(frozen=True, slots=True)
class LocalPerturbationMeasurement:
    direction_index: int
    direction_float32_bits: tuple[int, ...]
    radius: float
    central_loss_float32_bits: int
    plus_loss_float32_bits: int
    minus_loss_float32_bits: int
    improving_direction_found: bool


@dataclasses.dataclass(frozen=True, slots=True)
class DormancyMeasurement:
    mean_absolute_activation_float32_bits: tuple[int, ...]
    dormant_mask: tuple[bool, ...]
    dormant_fraction: float
    descriptive_cutoff: float
    used_as_zero_gradient_proxy: bool
    used_in_local_optimum_rule: bool


@dataclasses.dataclass(frozen=True, slots=True)
class SwitchDiagnostic:
    switch_index: int
    old_evaluator_regime_id: str
    incoming_evaluator_regime_id: str
    snapshot: FrozenLearnerSnapshot
    old_task_gradient: GradientMeasurement
    incoming_task_gradient: GradientMeasurement
    gradient_dot_product: float
    gradient_cosine: float | None
    gradient_alignment: GradientAlignment
    incoming_zero_gradient: bool
    local_perturbations: tuple[LocalPerturbationMeasurement, ...]
    local_optimum_trapped: bool
    dormancy: DormancyMeasurement


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterChangeMeasurement:
    start_snapshot_sha256: str
    end_snapshot_sha256: str
    delta_float32_bits: tuple[int, ...]
    l2_displacement: float
    relative_l2_displacement: float
    max_absolute_displacement: float
    changed_coordinate_count: int
    bitwise_churn_fraction: float
    sign_flip_count: int
    sign_flip_fraction: float

    def to_float32(self) -> tuple[float, ...]:
        return tuple(float(value) for value in _bits_to_float32(self.delta_float32_bits))


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingUpdateReceipt:
    global_step: int
    phase_index: int
    phase_step: int
    evaluator_regime_id: str
    observation_float32_bits: tuple[int, int]
    target_float32_bits: int
    pre_snapshot: FrozenLearnerSnapshot
    gradient: GradientMeasurement
    proposal_parameter_float32_bits: tuple[int, ...]
    constraint_attempted: bool
    constraint_applied: bool
    proposal_l2_from_initial: float
    applied_l2_from_initial: float
    post_snapshot: FrozenLearnerSnapshot


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseReport:
    phase_index: int
    evaluator_regime_id: str
    start_snapshot: FrozenLearnerSnapshot
    updates: tuple[TrainingUpdateReceipt, ...]
    end_snapshot: FrozenLearnerSnapshot
    parameter_change: ParameterChangeMeasurement


@dataclasses.dataclass(frozen=True, slots=True)
class ConditionReport:
    name: str
    constraint_mode: ConstraintMode
    initial_snapshot: FrozenLearnerSnapshot
    phases: tuple[PhaseReport, ...]
    switch_diagnostics: tuple[SwitchDiagnostic, ...]
    final_snapshot: FrozenLearnerSnapshot
    training_stream_sha256: str
    update_opportunities: int
    realized_updates: int
    projection_attempts: int
    projection_applications: int


@dataclasses.dataclass(frozen=True, slots=True)
class MatchedInterventionAudit:
    same_initial_snapshot: bool
    same_ordered_training_stream: bool
    same_learning_rate: bool
    equal_update_opportunities: bool
    only_declared_difference: str
    unconstrained_projection_applications: int
    constrained_projection_applications: int
    efficacy_assessed: bool


@dataclasses.dataclass(frozen=True, slots=True)
class ResourceAccounting:
    accounting_scope: str
    condition_count: int
    phase_count: int
    switch_diagnostic_count: int
    training_update_opportunities: int
    realized_training_updates: int
    training_gradient_evaluations: int
    diagnostic_gradient_evaluations: int
    local_perturbation_loss_evaluations: int
    dormant_activation_batch_evaluations: int
    training_examples: int
    diagnostic_gradient_probe_examples: int
    local_perturbation_probe_examples: int
    dormant_activation_probe_examples: int
    parameter_projection_attempts: int
    parameter_projection_applications: int
    parameter_scalar_gradient_values: int
    parameter_scalar_proposals: int
    raw_gradient_vectors_retained: int
    raw_parameter_delta_vectors_retained: int
    retained_snapshot_references: int
    learner_snapshot_freezes: int
    config_bound_snapshot_verification_calls: int
    typed_key_receipts: int
    random_float32_draws: int
    measured_wall_clock_samples: int
    wall_clock_threshold: float | None
    output_write_calls: int
    artifact_bytes_written: int


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizationCentricPlasticityDevelopmentReport:
    schema: str
    status: str
    assessment_status: str
    development_only: bool
    output_writes_authorized: bool
    thresholds_authorized: bool
    evidence_authorized: bool
    promotion_authorized: bool
    scientific_promotion_allowed: bool
    config: OptimizationCentricPlasticityDevelopmentConfig
    config_sha256: str
    source_manifest: tuple[SourceFileIdentity, ...]
    source_manifest_sha256: str
    runtime_identity: RuntimeIdentity
    runtime_identity_sha256: str
    learner_visible_fields: tuple[str, ...]
    evaluator_only_fields: tuple[str, ...]
    initialization_key_words_uint32: tuple[int, int]
    protocol: tuple[ProtocolStep, ...]
    protocol_sha256: str
    switch_probe_manifests: tuple[SwitchProbeManifest, ...]
    initial_snapshot: FrozenLearnerSnapshot
    conditions: tuple[ConditionReport, ...]
    matched_intervention: MatchedInterventionAudit
    resources: ResourceAccounting
    reconstruction_mode: str
    limitations: tuple[str, ...]
    report_sha256: str

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


def optimization_centric_plasticity_source_manifest(
    root: Path = _REPO_ROOT,
) -> tuple[SourceFileIdentity, ...]:
    """Hash the complete declared executed-source surface of this lane."""

    return tuple(
        SourceFileIdentity(
            path=path.as_posix(),
            sha256=hashlib.sha256((root / path).read_bytes()).hexdigest(),
        )
        for path in _SOURCE_PATHS
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def optimization_centric_plasticity_runtime_identity() -> RuntimeIdentity:
    """Return observable non-secret Python, JAX, and device identity."""

    devices = tuple(jax.devices())
    return RuntimeIdentity(
        identity_scope=(
            "observable Python/JAX/device identity; exact source-bound causal replay "
            "is authoritative"
        ),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        python_compiler=platform.python_compiler(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        machine=platform.machine(),
        byteorder=sys.byteorder,
        jax_version=str(jax.__version__),
        jaxlib_version=_package_version("jaxlib"),
        numpy_version=str(np.__version__),
        default_backend=jax.default_backend(),
        device_count=len(devices),
        local_device_count=int(jax.local_device_count()),
        device_platforms=tuple(str(device.platform) for device in devices),
        device_kinds=tuple(str(device.device_kind) for device in devices),
        jax_enable_x64=bool(jax.config.jax_enable_x64),
        jax_default_prng_impl=str(jax.config.jax_default_prng_impl),
        jax_threefry_partitionable=bool(jax.config.jax_threefry_partitionable),
        jax_disable_jit=bool(jax.config.jax_disable_jit),
    )


def _config_sha256(config: OptimizationCentricPlasticityDevelopmentConfig) -> str:
    return _dataclass_digest(config)


def _snapshot_sha256(snapshot: FrozenLearnerSnapshot) -> str:
    payload = dataclasses.asdict(snapshot)
    payload["sha256"] = ""
    return _digest(payload)


def freeze_learner_snapshot(
    parameters: object,
    *,
    revision: int,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> FrozenLearnerSnapshot:
    """Copy parameters into an immutable IEEE-754 bit receipt and verify it."""

    revision = _exact_int(revision, name="snapshot revision", maximum=_UINT32_MAX)
    array = np.asarray(parameters, dtype=np.float32)
    if array.shape != (config.parameter_count,) or not np.all(np.isfinite(array)):
        raise ValueError("snapshot parameters must be one finite canonical vector")
    bare = FrozenLearnerSnapshot(
        schema=SNAPSHOT_SCHEMA,
        config_sha256=_config_sha256(config),
        revision=revision,
        parameter_count=config.parameter_count,
        parameter_float32_bits=_float32_bits(array),
        sha256="",
    )
    snapshot = dataclasses.replace(bare, sha256=_snapshot_sha256(bare))
    verify_learner_snapshot(snapshot, config)
    return snapshot


def verify_learner_snapshot(
    snapshot: FrozenLearnerSnapshot,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> None:
    """Fail closed unless a snapshot is canonical, finite, and config-bound."""

    if type(snapshot) is not FrozenLearnerSnapshot:
        raise ValueError("snapshot type is invalid")
    if snapshot.schema != SNAPSHOT_SCHEMA:
        raise ValueError("snapshot schema is invalid")
    if snapshot.config_sha256 != _config_sha256(config):
        raise ValueError("snapshot config binding is invalid")
    _exact_int(snapshot.revision, name="snapshot revision", maximum=_UINT32_MAX)
    if snapshot.parameter_count != config.parameter_count:
        raise ValueError("snapshot parameter count is invalid")
    if len(snapshot.parameter_float32_bits) != config.parameter_count:
        raise ValueError("snapshot parameter payload length is invalid")
    parameters = _bits_to_float32(snapshot.parameter_float32_bits)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("snapshot parameters must be finite")
    _validate_sha256(snapshot.sha256, name="snapshot sha256")
    if snapshot.sha256 != _snapshot_sha256(snapshot):
        raise ValueError("snapshot integrity digest differs")


def thaw_learner_snapshot(snapshot: FrozenLearnerSnapshot) -> Array:
    """Return a fresh JAX array; changing the result cannot change the receipt."""

    if type(snapshot) is not FrozenLearnerSnapshot:
        raise ValueError("snapshot type is invalid")
    if snapshot.schema != SNAPSHOT_SCHEMA:
        raise ValueError("snapshot schema is invalid")
    _exact_int(snapshot.revision, name="snapshot revision", maximum=_UINT32_MAX)
    _exact_int(snapshot.parameter_count, name="snapshot parameter count", minimum=1)
    if len(snapshot.parameter_float32_bits) != snapshot.parameter_count:
        raise ValueError("snapshot parameter payload length is invalid")
    _validate_sha256(snapshot.config_sha256, name="snapshot config sha256")
    _validate_sha256(snapshot.sha256, name="snapshot sha256")
    if snapshot.sha256 != _snapshot_sha256(snapshot):
        raise ValueError("snapshot integrity digest differs")
    parameters = _bits_to_float32(snapshot.parameter_float32_bits)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("snapshot parameters must be finite")
    return jnp.asarray(parameters, dtype=jnp.float32)


def _unpack_parameters(
    parameters: Array,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> tuple[Array, Array, Array, Array]:
    cursor = 0
    first_count = config.input_dim * config.hidden_dim
    input_weights = parameters[cursor : cursor + first_count].reshape(
        (config.input_dim, config.hidden_dim)
    )
    cursor += first_count
    hidden_bias = parameters[cursor : cursor + config.hidden_dim]
    cursor += config.hidden_dim
    output_weights = parameters[cursor : cursor + config.hidden_dim]
    cursor += config.hidden_dim
    output_bias = parameters[cursor]
    return input_weights, hidden_bias, output_weights, output_bias


def _predict_and_hidden(
    parameters: Array,
    observations: Array,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> tuple[Array, Array]:
    input_weights, hidden_bias, output_weights, output_bias = _unpack_parameters(
        parameters,
        config,
    )
    hidden = jnp.tanh(observations @ input_weights + hidden_bias)
    predictions = jnp.tanh(hidden @ output_weights + output_bias)
    return predictions, hidden


def _loss(
    parameters: Array,
    observations: Array,
    targets: Array,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> Array:
    predictions, _ = _predict_and_hidden(parameters, observations, config)
    return jnp.mean(0.5 * jnp.square(predictions - targets))


def _gradient_measurement(
    snapshot: FrozenLearnerSnapshot,
    observations: Array,
    targets: Array,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> GradientMeasurement:
    verify_learner_snapshot(snapshot, config)
    loss_value, gradient = jax.value_and_grad(_loss)(
        thaw_learner_snapshot(snapshot),
        observations,
        targets,
        config,
    )
    gradient_array = np.asarray(gradient, dtype=np.float32)
    gradient64 = gradient_array.astype(np.float64)
    return GradientMeasurement(
        loss_float32_bits=_scalar_float32_bits(loss_value),
        gradient_float32_bits=_float32_bits(gradient_array),
        l2_norm=float(np.linalg.norm(gradient64)),
        max_absolute_component=float(np.max(np.abs(gradient64))),
        nonzero_component_count=int(np.count_nonzero(gradient_array)),
    )


def _probe_arrays(regime_id: str) -> tuple[Array, Array]:
    return (
        jnp.asarray(_PROBE_OBSERVATIONS, dtype=jnp.float32),
        jnp.asarray(_targets(regime_id), dtype=jnp.float32),
    )


def _build_protocol(
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> tuple[ProtocolStep, ...]:
    records: list[ProtocolStep] = []
    for phase_index, regime_id in enumerate(config.phase_order):
        targets = _targets(regime_id)
        for phase_step in range(config.updates_per_phase):
            sample_index = phase_step % len(_PROBE_OBSERVATIONS)
            records.append(
                ProtocolStep(
                    step_index=len(records),
                    phase_index=phase_index,
                    phase_step=phase_step,
                    evaluator_regime_id=regime_id,
                    phase_boundary=phase_step == 0,
                    observation_float32_bits=cast(
                        tuple[int, int],
                        _float32_bits(_PROBE_OBSERVATIONS[sample_index]),
                    ),
                    target_float32_bits=_scalar_float32_bits(targets[sample_index]),
                )
            )
    return tuple(records)


def _build_switch_probe_manifests(
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> tuple[SwitchProbeManifest, ...]:
    root_key = jr.fold_in(jr.key(config.seed), _DIRECTION_DOMAIN)
    manifests: list[SwitchProbeManifest] = []
    for switch_index in range(config.switch_count):
        key = jr.fold_in(root_key, switch_index)
        raw_directions = jr.normal(
            key,
            (config.local_direction_count, config.parameter_count),
            dtype=jnp.float32,
        )
        norms = jnp.linalg.norm(raw_directions, axis=1, keepdims=True)
        directions = raw_directions / jnp.maximum(norms, jnp.asarray(1.0e-12))
        bare = SwitchProbeManifest(
            switch_index=switch_index,
            old_evaluator_regime_id=config.phase_order[switch_index],
            incoming_evaluator_regime_id=config.phase_order[switch_index + 1],
            direction_key_words_uint32=_key_words(key),
            direction_float32_bits=tuple(
                _float32_bits(directions[index])
                for index in range(config.local_direction_count)
            ),
            sha256="",
        )
        manifests.append(dataclasses.replace(bare, sha256=_dataclass_digest(bare)))
    return tuple(manifests)


def _initial_snapshot(
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> tuple[tuple[int, int], FrozenLearnerSnapshot]:
    key = jr.fold_in(jr.key(config.seed), _INITIALIZATION_DOMAIN)
    parameters = 0.18 * jr.normal(
        key,
        (config.parameter_count,),
        dtype=jnp.float32,
    )
    return _key_words(key), freeze_learner_snapshot(parameters, revision=0, config=config)


def _validate_gradient_measurement(
    measurement: GradientMeasurement,
    config: OptimizationCentricPlasticityDevelopmentConfig,
    *,
    name: str,
) -> None:
    if type(measurement) is not GradientMeasurement:
        raise ValueError(f"{name} gradient measurement type is invalid")
    _exact_int(
        measurement.loss_float32_bits,
        name=f"{name}.loss_float32_bits",
        maximum=_UINT32_MAX,
    )
    loss = _scalar_from_float32_bits(measurement.loss_float32_bits)
    if not math.isfinite(loss):
        raise ValueError(f"{name} loss must be finite")
    if len(measurement.gradient_float32_bits) != config.parameter_count:
        raise ValueError(f"{name} gradient payload length is invalid")
    gradient = _bits_to_float32(measurement.gradient_float32_bits)
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"{name} gradient must be finite")
    gradient64 = gradient.astype(np.float64)
    expected_norm = float(np.linalg.norm(gradient64))
    expected_max = float(np.max(np.abs(gradient64)))
    expected_nonzero = int(np.count_nonzero(gradient))
    if (
        measurement.l2_norm != expected_norm
        or measurement.max_absolute_component != expected_max
        or measurement.nonzero_component_count != expected_nonzero
    ):
        raise ValueError(f"{name} gradient derived measurements differ")


def _measure_dormancy(
    snapshot: FrozenLearnerSnapshot,
    incoming_regime_id: str,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> DormancyMeasurement:
    observations, _ = _probe_arrays(incoming_regime_id)
    _, hidden = _predict_and_hidden(thaw_learner_snapshot(snapshot), observations, config)
    mean_absolute = np.asarray(jnp.mean(jnp.abs(hidden), axis=0), dtype=np.float32)
    dormant_mask = tuple(
        bool(value <= config.dormant_activation_descriptive_cutoff)
        for value in mean_absolute
    )
    return DormancyMeasurement(
        mean_absolute_activation_float32_bits=_float32_bits(mean_absolute),
        dormant_mask=dormant_mask,
        dormant_fraction=float(sum(dormant_mask) / config.hidden_dim),
        descriptive_cutoff=config.dormant_activation_descriptive_cutoff,
        used_as_zero_gradient_proxy=False,
        used_in_local_optimum_rule=False,
    )


def _measure_switch(
    snapshot: FrozenLearnerSnapshot,
    manifest: SwitchProbeManifest,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> SwitchDiagnostic:
    old_observations, old_targets = _probe_arrays(manifest.old_evaluator_regime_id)
    incoming_observations, incoming_targets = _probe_arrays(
        manifest.incoming_evaluator_regime_id
    )
    old_gradient = _gradient_measurement(
        snapshot,
        old_observations,
        old_targets,
        config,
    )
    incoming_gradient = _gradient_measurement(
        snapshot,
        incoming_observations,
        incoming_targets,
        config,
    )
    old_vector = _bits_to_float32(old_gradient.gradient_float32_bits).astype(np.float64)
    incoming_vector = _bits_to_float32(
        incoming_gradient.gradient_float32_bits
    ).astype(np.float64)
    dot = float(np.dot(old_vector, incoming_vector))
    denominator = old_gradient.l2_norm * incoming_gradient.l2_norm
    if denominator == 0.0:
        cosine: float | None = None
        alignment: GradientAlignment = "undefined"
    else:
        cosine = dot / denominator
        if dot > config.alignment_descriptive_floor:
            alignment = "aligned"
        elif dot < -config.alignment_descriptive_floor:
            alignment = "conflicting"
        else:
            alignment = "orthogonal"

    parameters = thaw_learner_snapshot(snapshot)
    central_loss_bits = incoming_gradient.loss_float32_bits
    central_loss = incoming_gradient.loss
    local_measurements: list[LocalPerturbationMeasurement] = []
    for direction_index, direction_bits in enumerate(
        manifest.direction_float32_bits
    ):
        direction = jnp.asarray(_bits_to_float32(direction_bits), dtype=jnp.float32)
        displacement = jnp.asarray(config.local_probe_radius, dtype=jnp.float32) * direction
        plus_loss = _loss(
            parameters + displacement,
            incoming_observations,
            incoming_targets,
            config,
        )
        minus_loss = _loss(
            parameters - displacement,
            incoming_observations,
            incoming_targets,
            config,
        )
        plus_bits = _scalar_float32_bits(plus_loss)
        minus_bits = _scalar_float32_bits(minus_loss)
        improving = min(
            _scalar_from_float32_bits(plus_bits),
            _scalar_from_float32_bits(minus_bits),
        ) < central_loss - config.local_loss_descriptive_tolerance
        local_measurements.append(
            LocalPerturbationMeasurement(
                direction_index=direction_index,
                direction_float32_bits=direction_bits,
                radius=config.local_probe_radius,
                central_loss_float32_bits=central_loss_bits,
                plus_loss_float32_bits=plus_bits,
                minus_loss_float32_bits=minus_bits,
                improving_direction_found=improving,
            )
        )

    incoming_zero_gradient = (
        incoming_gradient.l2_norm <= config.zero_gradient_descriptive_floor
    )
    local_optimum_trapped = incoming_zero_gradient and all(
        not measurement.improving_direction_found
        for measurement in local_measurements
    )
    return SwitchDiagnostic(
        switch_index=manifest.switch_index,
        old_evaluator_regime_id=manifest.old_evaluator_regime_id,
        incoming_evaluator_regime_id=manifest.incoming_evaluator_regime_id,
        snapshot=snapshot,
        old_task_gradient=old_gradient,
        incoming_task_gradient=incoming_gradient,
        gradient_dot_product=dot,
        gradient_cosine=cosine,
        gradient_alignment=alignment,
        incoming_zero_gradient=incoming_zero_gradient,
        local_perturbations=tuple(local_measurements),
        local_optimum_trapped=local_optimum_trapped,
        dormancy=_measure_dormancy(
            snapshot,
            manifest.incoming_evaluator_regime_id,
            config,
        ),
    )


def diagnose_frozen_learner_switch(
    snapshot: FrozenLearnerSnapshot,
    *,
    switch_index: int,
    config: OptimizationCentricPlasticityDevelopmentConfig | None = None,
) -> SwitchDiagnostic:
    """Measure one frozen evaluator-owned switch without accepting external targets."""

    resolved = (
        OptimizationCentricPlasticityDevelopmentConfig()
        if config is None
        else config
    )
    if type(resolved) is not OptimizationCentricPlasticityDevelopmentConfig:
        raise TypeError("config type is invalid")
    switch_index = _exact_int(
        switch_index,
        name="switch_index",
        maximum=resolved.switch_count - 1,
    )
    verify_learner_snapshot(snapshot, resolved)
    manifest = _build_switch_probe_manifests(resolved)[switch_index]
    return _measure_switch(snapshot, manifest, resolved)


def _parameter_change(
    start: FrozenLearnerSnapshot,
    end: FrozenLearnerSnapshot,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> ParameterChangeMeasurement:
    verify_learner_snapshot(start, config)
    verify_learner_snapshot(end, config)
    before = _bits_to_float32(start.parameter_float32_bits)
    after = _bits_to_float32(end.parameter_float32_bits)
    delta = np.asarray(after - before, dtype=np.float32)
    delta64 = delta.astype(np.float64)
    before64 = before.astype(np.float64)
    changed_count = sum(
        left != right
        for left, right in zip(
            start.parameter_float32_bits,
            end.parameter_float32_bits,
            strict=True,
        )
    )
    sign_flips = int(
        np.count_nonzero(
            ((before < 0.0) & (after >= 0.0))
            | ((before >= 0.0) & (after < 0.0))
        )
    )
    l2 = float(np.linalg.norm(delta64))
    return ParameterChangeMeasurement(
        start_snapshot_sha256=start.sha256,
        end_snapshot_sha256=end.sha256,
        delta_float32_bits=_float32_bits(delta),
        l2_displacement=l2,
        relative_l2_displacement=l2 / max(float(np.linalg.norm(before64)), 1.0e-12),
        max_absolute_displacement=float(np.max(np.abs(delta64))),
        changed_coordinate_count=changed_count,
        bitwise_churn_fraction=float(changed_count / config.parameter_count),
        sign_flip_count=sign_flips,
        sign_flip_fraction=float(sign_flips / config.parameter_count),
    )


def _training_update(
    snapshot: FrozenLearnerSnapshot,
    initial_snapshot: FrozenLearnerSnapshot,
    protocol_step: ProtocolStep,
    constraint_mode: ConstraintMode,
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> TrainingUpdateReceipt:
    verify_learner_snapshot(snapshot, config)
    verify_learner_snapshot(initial_snapshot, config)
    observation = jnp.asarray(
        _bits_to_float32(protocol_step.observation_float32_bits),
        dtype=jnp.float32,
    ).reshape((1, config.input_dim))
    target = jnp.asarray(
        (_scalar_from_float32_bits(protocol_step.target_float32_bits),),
        dtype=jnp.float32,
    )
    gradient = _gradient_measurement(snapshot, observation, target, config)
    gradient_array = jnp.asarray(
        _bits_to_float32(gradient.gradient_float32_bits),
        dtype=jnp.float32,
    )
    parameters = thaw_learner_snapshot(snapshot)
    proposal = parameters - jnp.asarray(config.learning_rate, dtype=jnp.float32) * gradient_array
    initial_parameters = thaw_learner_snapshot(initial_snapshot)
    proposal_delta = proposal - initial_parameters
    proposal_l2 = float(
        np.linalg.norm(np.asarray(proposal_delta, dtype=np.float32).astype(np.float64))
    )
    constraint_attempted = constraint_mode == L2_CONSTRAINED
    constraint_applied = constraint_attempted and proposal_l2 > config.constraint_radius
    applied = proposal
    if constraint_applied:
        jax_norm = jnp.linalg.norm(proposal_delta)
        scale = jnp.asarray(config.constraint_radius, dtype=jnp.float32) / jax_norm
        applied = initial_parameters + proposal_delta * scale
    applied_delta = applied - initial_parameters
    applied_l2 = float(
        np.linalg.norm(np.asarray(applied_delta, dtype=np.float32).astype(np.float64))
    )
    post = freeze_learner_snapshot(
        applied,
        revision=snapshot.revision + 1,
        config=config,
    )
    return TrainingUpdateReceipt(
        global_step=protocol_step.step_index,
        phase_index=protocol_step.phase_index,
        phase_step=protocol_step.phase_step,
        evaluator_regime_id=protocol_step.evaluator_regime_id,
        observation_float32_bits=protocol_step.observation_float32_bits,
        target_float32_bits=protocol_step.target_float32_bits,
        pre_snapshot=snapshot,
        gradient=gradient,
        proposal_parameter_float32_bits=_float32_bits(proposal),
        constraint_attempted=constraint_attempted,
        constraint_applied=constraint_applied,
        proposal_l2_from_initial=proposal_l2,
        applied_l2_from_initial=applied_l2,
        post_snapshot=post,
    )


def _run_condition(
    *,
    name: str,
    constraint_mode: ConstraintMode,
    initial_snapshot: FrozenLearnerSnapshot,
    protocol: tuple[ProtocolStep, ...],
    switch_manifests: tuple[SwitchProbeManifest, ...],
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> ConditionReport:
    current = initial_snapshot
    phases: list[PhaseReport] = []
    diagnostics: list[SwitchDiagnostic] = []
    projection_attempts = 0
    projection_applications = 0
    for phase_index, regime_id in enumerate(config.phase_order):
        if phase_index > 0:
            diagnostics.append(
                _measure_switch(current, switch_manifests[phase_index - 1], config)
            )
        phase_start = current
        updates: list[TrainingUpdateReceipt] = []
        first_step = phase_index * config.updates_per_phase
        phase_protocol = protocol[first_step : first_step + config.updates_per_phase]
        for protocol_step in phase_protocol:
            receipt = _training_update(
                current,
                initial_snapshot,
                protocol_step,
                constraint_mode,
                config,
            )
            updates.append(receipt)
            projection_attempts += int(receipt.constraint_attempted)
            projection_applications += int(receipt.constraint_applied)
            current = receipt.post_snapshot
        phases.append(
            PhaseReport(
                phase_index=phase_index,
                evaluator_regime_id=regime_id,
                start_snapshot=phase_start,
                updates=tuple(updates),
                end_snapshot=current,
                parameter_change=_parameter_change(phase_start, current, config),
            )
        )
    return ConditionReport(
        name=name,
        constraint_mode=constraint_mode,
        initial_snapshot=initial_snapshot,
        phases=tuple(phases),
        switch_diagnostics=tuple(diagnostics),
        final_snapshot=current,
        training_stream_sha256=_digest(
            [dataclasses.asdict(step) for step in protocol]
        ),
        update_opportunities=config.protocol_step_count,
        realized_updates=sum(len(phase.updates) for phase in phases),
        projection_attempts=projection_attempts,
        projection_applications=projection_applications,
    )


def _resource_accounting(
    conditions: tuple[ConditionReport, ...],
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> ResourceAccounting:
    condition_count = len(conditions)
    phase_count = condition_count * config.phase_count
    diagnostic_count = condition_count * config.switch_count
    update_count = condition_count * config.protocol_step_count
    diagnostic_gradient_evaluations = diagnostic_count * 2
    local_loss_evaluations = (
        diagnostic_count * config.local_direction_count * 2
    )
    probe_example_count = len(_PROBE_OBSERVATIONS)
    projection_attempts = sum(condition.projection_attempts for condition in conditions)
    projection_applications = sum(
        condition.projection_applications for condition in conditions
    )
    # Snapshot references are counted where they are retained in the report,
    # including duplicate immutable receipts.  This is a logical payload count,
    # not an allocator-residency claim.
    snapshot_references = (
        1
        + condition_count * 2
        + phase_count * 2
        + update_count * 2
        + diagnostic_count
    )
    return ResourceAccounting(
        accounting_scope=(
            "one causal evaluator execution before report validation/reconstruction; exact "
            "logical calls and values, not FLOPs, allocator residency, or wall-clock work"
        ),
        condition_count=condition_count,
        phase_count=phase_count,
        switch_diagnostic_count=diagnostic_count,
        training_update_opportunities=update_count,
        realized_training_updates=sum(
            condition.realized_updates for condition in conditions
        ),
        training_gradient_evaluations=update_count,
        diagnostic_gradient_evaluations=diagnostic_gradient_evaluations,
        local_perturbation_loss_evaluations=local_loss_evaluations,
        dormant_activation_batch_evaluations=diagnostic_count,
        training_examples=update_count,
        diagnostic_gradient_probe_examples=(
            diagnostic_gradient_evaluations * probe_example_count
        ),
        local_perturbation_probe_examples=(
            local_loss_evaluations * probe_example_count
        ),
        dormant_activation_probe_examples=diagnostic_count * probe_example_count,
        parameter_projection_attempts=projection_attempts,
        parameter_projection_applications=projection_applications,
        parameter_scalar_gradient_values=(
            (update_count + diagnostic_gradient_evaluations) * config.parameter_count
        ),
        parameter_scalar_proposals=update_count * config.parameter_count,
        raw_gradient_vectors_retained=update_count + diagnostic_gradient_evaluations,
        raw_parameter_delta_vectors_retained=phase_count,
        retained_snapshot_references=snapshot_references,
        learner_snapshot_freezes=1 + update_count,
        config_bound_snapshot_verification_calls=(
            1 + 4 * update_count + 2 * phase_count + 2 * diagnostic_count
        ),
        typed_key_receipts=1 + config.switch_count,
        random_float32_draws=(
            config.parameter_count
            + config.switch_count
            * config.local_direction_count
            * config.parameter_count
        ),
        measured_wall_clock_samples=0,
        wall_clock_threshold=None,
        output_write_calls=0,
        artifact_bytes_written=0,
    )


_LIMITATIONS: Final = (
    "L0 development mechanism on one consumed synthetic seed; no promotion inference is permitted",
    "the A/B/A labels, scalar targets, boundaries, and probe directions are evaluator-owned",
    "the learner receives only numeric observations and scalar targets and is never reset "
    "at a switch",
    "the zero-gradient floor and local probe rule are fixed descriptive instrumentation, "
    "not calibrated gates",
    "two sampled perturbation directions cannot establish a true local optimum in the full "
    "parameter space",
    "dormant hidden activation is a separate descriptive measurement and is not a gradient proxy",
    "the initialization-centred L2 intervention is a mechanism comparator, not an efficacy claim",
    "logical counters do not measure FLOPs, allocator residency, wall-clock latency, or energy",
    "the declared source manifest is an integrity binding, not authenticity or hardware "
    "attestation",
)


def _report_sha256(report: OptimizationCentricPlasticityDevelopmentReport) -> str:
    payload = dataclasses.asdict(report)
    payload["report_sha256"] = ""
    return _digest(payload)


def _execute(
    config: OptimizationCentricPlasticityDevelopmentConfig,
) -> OptimizationCentricPlasticityDevelopmentReport:
    source_before = optimization_centric_plasticity_source_manifest()
    runtime = optimization_centric_plasticity_runtime_identity()
    config_sha256 = _config_sha256(config)
    protocol = _build_protocol(config)
    switch_manifests = _build_switch_probe_manifests(config)
    initialization_key_words, initial_snapshot = _initial_snapshot(config)
    conditions = (
        _run_condition(
            name=UNCONSTRAINED,
            constraint_mode=UNCONSTRAINED,
            initial_snapshot=initial_snapshot,
            protocol=protocol,
            switch_manifests=switch_manifests,
            config=config,
        ),
        _run_condition(
            name=L2_CONSTRAINED,
            constraint_mode=L2_CONSTRAINED,
            initial_snapshot=initial_snapshot,
            protocol=protocol,
            switch_manifests=switch_manifests,
            config=config,
        ),
    )
    source_after = optimization_centric_plasticity_source_manifest()
    if source_after != source_before:
        raise RuntimeError("declared source changed during development evaluation")
    matched = MatchedInterventionAudit(
        same_initial_snapshot=(
            conditions[0].initial_snapshot.sha256
            == conditions[1].initial_snapshot.sha256
            == initial_snapshot.sha256
        ),
        same_ordered_training_stream=(
            conditions[0].training_stream_sha256
            == conditions[1].training_stream_sha256
        ),
        same_learning_rate=True,
        equal_update_opportunities=(
            conditions[0].update_opportunities
            == conditions[1].update_opportunities
            == config.protocol_step_count
        ),
        only_declared_difference="l2_parameter_constraint",
        unconstrained_projection_applications=conditions[0].projection_applications,
        constrained_projection_applications=conditions[1].projection_applications,
        efficacy_assessed=False,
    )
    bare = OptimizationCentricPlasticityDevelopmentReport(
        schema=REPORT_SCHEMA,
        status=DEVELOPMENT_STATUS,
        assessment_status=ASSESSMENT_STATUS,
        development_only=True,
        output_writes_authorized=OUTPUT_WRITES_AUTHORIZED,
        thresholds_authorized=THRESHOLDS_AUTHORIZED,
        evidence_authorized=EVIDENCE_AUTHORIZED,
        promotion_authorized=PROMOTION_AUTHORIZED,
        scientific_promotion_allowed=SCIENTIFIC_PROMOTION_ALLOWED,
        config=config,
        config_sha256=config_sha256,
        source_manifest=source_before,
        source_manifest_sha256=_digest(
            [dataclasses.asdict(item) for item in source_before]
        ),
        runtime_identity=runtime,
        runtime_identity_sha256=_dataclass_digest(runtime),
        learner_visible_fields=("observation", "target"),
        evaluator_only_fields=(
            "evaluator_regime_id",
            "phase_index",
            "phase_boundary",
        ),
        initialization_key_words_uint32=initialization_key_words,
        protocol=protocol,
        protocol_sha256=_digest([dataclasses.asdict(step) for step in protocol]),
        switch_probe_manifests=switch_manifests,
        initial_snapshot=initial_snapshot,
        conditions=conditions,
        matched_intervention=matched,
        resources=_resource_accounting(conditions, config),
        reconstruction_mode=(
            "deterministic exact causal replay from frozen config plus bound source/runtime; "
            "all raw stream, gradient, perturbation, activation, and parameter receipts retained"
        ),
        limitations=_LIMITATIONS,
        report_sha256="",
    )
    return dataclasses.replace(bare, report_sha256=_report_sha256(bare))


def _validate_parameter_change(
    measurement: ParameterChangeMeasurement,
    start: FrozenLearnerSnapshot,
    end: FrozenLearnerSnapshot,
    config: OptimizationCentricPlasticityDevelopmentConfig,
    *,
    name: str,
) -> None:
    if type(measurement) is not ParameterChangeMeasurement:
        raise ValueError(f"{name} parameter change type is invalid")
    if (
        measurement.start_snapshot_sha256 != start.sha256
        or measurement.end_snapshot_sha256 != end.sha256
    ):
        raise ValueError(f"{name} parameter change snapshot binding differs")
    before = _bits_to_float32(start.parameter_float32_bits)
    after = _bits_to_float32(end.parameter_float32_bits)
    expected_delta = np.asarray(after - before, dtype=np.float32)
    if measurement.delta_float32_bits != _float32_bits(expected_delta):
        raise ValueError(f"{name} raw parameter displacement differs")
    delta64 = expected_delta.astype(np.float64)
    before64 = before.astype(np.float64)
    changed = sum(
        left != right
        for left, right in zip(
            start.parameter_float32_bits,
            end.parameter_float32_bits,
            strict=True,
        )
    )
    sign_flips = int(
        np.count_nonzero(
            ((before < 0.0) & (after >= 0.0))
            | ((before >= 0.0) & (after < 0.0))
        )
    )
    l2 = float(np.linalg.norm(delta64))
    expected = (
        l2,
        l2 / max(float(np.linalg.norm(before64)), 1.0e-12),
        float(np.max(np.abs(delta64))),
        changed,
        float(changed / config.parameter_count),
        sign_flips,
        float(sign_flips / config.parameter_count),
    )
    actual = (
        measurement.l2_displacement,
        measurement.relative_l2_displacement,
        measurement.max_absolute_displacement,
        measurement.changed_coordinate_count,
        measurement.bitwise_churn_fraction,
        measurement.sign_flip_count,
        measurement.sign_flip_fraction,
    )
    if actual != expected:
        raise ValueError(f"{name} parameter displacement/churn metrics differ")


def _validate_switch_diagnostic(
    diagnostic: SwitchDiagnostic,
    manifest: SwitchProbeManifest,
    config: OptimizationCentricPlasticityDevelopmentConfig,
    *,
    name: str,
) -> None:
    if type(diagnostic) is not SwitchDiagnostic:
        raise ValueError(f"{name} switch diagnostic type is invalid")
    verify_learner_snapshot(diagnostic.snapshot, config)
    if (
        diagnostic.switch_index != manifest.switch_index
        or diagnostic.old_evaluator_regime_id != manifest.old_evaluator_regime_id
        or diagnostic.incoming_evaluator_regime_id
        != manifest.incoming_evaluator_regime_id
    ):
        raise ValueError(f"{name} switch identity differs")
    _validate_gradient_measurement(
        diagnostic.old_task_gradient,
        config,
        name=f"{name}.old_task",
    )
    _validate_gradient_measurement(
        diagnostic.incoming_task_gradient,
        config,
        name=f"{name}.incoming_task",
    )
    old = _bits_to_float32(
        diagnostic.old_task_gradient.gradient_float32_bits
    ).astype(np.float64)
    incoming = _bits_to_float32(
        diagnostic.incoming_task_gradient.gradient_float32_bits
    ).astype(np.float64)
    dot = float(np.dot(old, incoming))
    denominator = (
        diagnostic.old_task_gradient.l2_norm
        * diagnostic.incoming_task_gradient.l2_norm
    )
    if denominator == 0.0:
        cosine: float | None = None
        alignment: GradientAlignment = "undefined"
    else:
        cosine = dot / denominator
        if dot > config.alignment_descriptive_floor:
            alignment = "aligned"
        elif dot < -config.alignment_descriptive_floor:
            alignment = "conflicting"
        else:
            alignment = "orthogonal"
    if (
        diagnostic.gradient_dot_product != dot
        or diagnostic.gradient_cosine != cosine
        or diagnostic.gradient_alignment != alignment
    ):
        raise ValueError(f"{name} gradient alignment metrics differ")
    zero_gradient = (
        diagnostic.incoming_task_gradient.l2_norm
        <= config.zero_gradient_descriptive_floor
    )
    if diagnostic.incoming_zero_gradient != zero_gradient:
        raise ValueError(f"{name} zero-gradient descriptive rule differs")
    if len(diagnostic.local_perturbations) != config.local_direction_count:
        raise ValueError(f"{name} local perturbation count differs")
    for index, (measurement, direction_bits) in enumerate(
        zip(
            diagnostic.local_perturbations,
            manifest.direction_float32_bits,
            strict=True,
        )
    ):
        if (
            type(measurement) is not LocalPerturbationMeasurement
            or measurement.direction_index != index
            or measurement.direction_float32_bits != direction_bits
            or measurement.radius != config.local_probe_radius
            or measurement.central_loss_float32_bits
            != diagnostic.incoming_task_gradient.loss_float32_bits
        ):
            raise ValueError(f"{name} local perturbation identity differs")
        plus = _scalar_from_float32_bits(measurement.plus_loss_float32_bits)
        minus = _scalar_from_float32_bits(measurement.minus_loss_float32_bits)
        central = diagnostic.incoming_task_gradient.loss
        if not all(math.isfinite(value) for value in (plus, minus, central)):
            raise ValueError(f"{name} local perturbation losses must be finite")
        improving = min(plus, minus) < (
            central - config.local_loss_descriptive_tolerance
        )
        if measurement.improving_direction_found != improving:
            raise ValueError(f"{name} local perturbation rule differs")
    expected_trapped = zero_gradient and all(
        not measurement.improving_direction_found
        for measurement in diagnostic.local_perturbations
    )
    if diagnostic.local_optimum_trapped != expected_trapped:
        raise ValueError(f"{name} local-optimum descriptive rule differs")

    dormancy = diagnostic.dormancy
    if type(dormancy) is not DormancyMeasurement:
        raise ValueError(f"{name} dormancy type is invalid")
    if dormancy.used_as_zero_gradient_proxy or dormancy.used_in_local_optimum_rule:
        raise ValueError(f"{name} dormancy must remain separate from gradient rules")
    if (
        dormancy.descriptive_cutoff
        != config.dormant_activation_descriptive_cutoff
        or len(dormancy.mean_absolute_activation_float32_bits) != config.hidden_dim
        or len(dormancy.dormant_mask) != config.hidden_dim
    ):
        raise ValueError(f"{name} dormancy shape/config differs")
    activations = _bits_to_float32(
        dormancy.mean_absolute_activation_float32_bits
    )
    if not np.all(np.isfinite(activations)) or np.any(activations < 0.0):
        raise ValueError(f"{name} dormant activations must be finite and nonnegative")
    expected_mask = tuple(
        bool(value <= config.dormant_activation_descriptive_cutoff)
        for value in activations
    )
    if (
        dormancy.dormant_mask != expected_mask
        or dormancy.dormant_fraction != sum(expected_mask) / config.hidden_dim
    ):
        raise ValueError(f"{name} dormancy metrics differ")


def _validate_structural_report(
    report: OptimizationCentricPlasticityDevelopmentReport,
) -> None:
    if type(report) is not OptimizationCentricPlasticityDevelopmentReport:
        raise ValueError("report type is invalid")
    if (
        report.schema != REPORT_SCHEMA
        or report.status != DEVELOPMENT_STATUS
        or report.assessment_status != ASSESSMENT_STATUS
        or not report.development_only
    ):
        raise ValueError("report development identity differs")
    if (
        report.output_writes_authorized
        or report.thresholds_authorized
        or report.evidence_authorized
        or report.promotion_authorized
        or report.scientific_promotion_allowed
    ):
        raise ValueError("report authority must remain fail-closed")
    if type(report.config) is not OptimizationCentricPlasticityDevelopmentConfig:
        raise ValueError("report config type is invalid")
    expected_config = OptimizationCentricPlasticityDevelopmentConfig()
    if report.config != expected_config:
        raise ValueError("report config is not the frozen protocol config")
    expected_config_sha256 = _config_sha256(report.config)
    if report.config_sha256 != expected_config_sha256:
        raise ValueError("report config identity differs")

    current_source = optimization_centric_plasticity_source_manifest()
    if report.source_manifest != current_source:
        raise ValueError("report source manifest differs from the executed source")
    for index, item in enumerate(report.source_manifest):
        if type(item) is not SourceFileIdentity:
            raise ValueError(f"report source manifest item {index} type is invalid")
        _validate_sha256(item.sha256, name=f"source_manifest[{index}].sha256")
    expected_source_sha256 = _digest(
        [dataclasses.asdict(item) for item in report.source_manifest]
    )
    if report.source_manifest_sha256 != expected_source_sha256:
        raise ValueError("report source manifest digest differs")

    current_runtime = optimization_centric_plasticity_runtime_identity()
    if report.runtime_identity != current_runtime:
        raise ValueError("report runtime identity differs from the current runtime")
    if report.runtime_identity_sha256 != _dataclass_digest(report.runtime_identity):
        raise ValueError("report runtime identity digest differs")
    if report.learner_visible_fields != ("observation", "target") or (
        report.evaluator_only_fields
        != ("evaluator_regime_id", "phase_index", "phase_boundary")
    ):
        raise ValueError("report learner/evaluator field boundary differs")

    protocol = _build_protocol(report.config)
    if report.protocol != protocol:
        raise ValueError("report evaluator-owned protocol differs")
    expected_protocol_sha256 = _digest(
        [dataclasses.asdict(step) for step in protocol]
    )
    if report.protocol_sha256 != expected_protocol_sha256:
        raise ValueError("report protocol digest differs")
    manifests = _build_switch_probe_manifests(report.config)
    if report.switch_probe_manifests != manifests:
        raise ValueError("report switch probe manifest differs")
    _, expected_initial = _initial_snapshot(report.config)
    if report.initial_snapshot != expected_initial:
        raise ValueError("report initial learner snapshot differs")
    verify_learner_snapshot(report.initial_snapshot, report.config)

    if tuple(condition.name for condition in report.conditions) != (
        UNCONSTRAINED,
        L2_CONSTRAINED,
    ) or tuple(condition.constraint_mode for condition in report.conditions) != (
        UNCONSTRAINED,
        L2_CONSTRAINED,
    ):
        raise ValueError("report matched condition order differs")
    stream_sha256 = expected_protocol_sha256
    for condition_index, condition in enumerate(report.conditions):
        location = f"conditions[{condition_index}]"
        if type(condition) is not ConditionReport:
            raise ValueError(f"{location} type is invalid")
        if (
            condition.initial_snapshot != report.initial_snapshot
            or condition.training_stream_sha256 != stream_sha256
            or condition.update_opportunities != report.config.protocol_step_count
            or condition.realized_updates != report.config.protocol_step_count
            or len(condition.phases) != report.config.phase_count
            or len(condition.switch_diagnostics) != report.config.switch_count
        ):
            raise ValueError(f"{location} matched stream/resources differ")
        expected_attempts = (
            report.config.protocol_step_count
            if condition.constraint_mode == L2_CONSTRAINED
            else 0
        )
        if condition.projection_attempts != expected_attempts:
            raise ValueError(f"{location} projection attempts differ")
        current = condition.initial_snapshot
        observed_applications = 0
        for phase_index, phase in enumerate(condition.phases):
            phase_location = f"{location}.phases[{phase_index}]"
            expected_regime = report.config.phase_order[phase_index]
            if (
                type(phase) is not PhaseReport
                or phase.phase_index != phase_index
                or phase.evaluator_regime_id != expected_regime
                or phase.start_snapshot != current
                or len(phase.updates) != report.config.updates_per_phase
            ):
                raise ValueError(f"{phase_location} identity/chain differs")
            verify_learner_snapshot(phase.start_snapshot, report.config)
            first_step = phase_index * report.config.updates_per_phase
            for phase_step, update in enumerate(phase.updates):
                update_location = f"{phase_location}.updates[{phase_step}]"
                protocol_step = protocol[first_step + phase_step]
                if type(update) is not TrainingUpdateReceipt:
                    raise ValueError(f"{update_location} type is invalid")
                if (
                    update.global_step != protocol_step.step_index
                    or update.phase_index != protocol_step.phase_index
                    or update.phase_step != protocol_step.phase_step
                    or update.evaluator_regime_id
                    != protocol_step.evaluator_regime_id
                    or update.observation_float32_bits
                    != protocol_step.observation_float32_bits
                    or update.target_float32_bits != protocol_step.target_float32_bits
                    or update.pre_snapshot != current
                    or update.post_snapshot.revision != current.revision + 1
                ):
                    raise ValueError(f"{update_location} causal receipt differs")
                verify_learner_snapshot(update.pre_snapshot, report.config)
                verify_learner_snapshot(update.post_snapshot, report.config)
                _validate_gradient_measurement(
                    update.gradient,
                    report.config,
                    name=f"{update_location}.gradient",
                )
                if len(update.proposal_parameter_float32_bits) != report.config.parameter_count:
                    raise ValueError(f"{update_location} proposal length differs")
                proposal = _bits_to_float32(update.proposal_parameter_float32_bits)
                initial = _bits_to_float32(
                    condition.initial_snapshot.parameter_float32_bits
                )
                post = _bits_to_float32(update.post_snapshot.parameter_float32_bits)
                if not np.all(np.isfinite(proposal)):
                    raise ValueError(f"{update_location} proposal must be finite")
                proposal_l2 = float(
                    np.linalg.norm((proposal - initial).astype(np.float64))
                )
                applied_l2 = float(
                    np.linalg.norm((post - initial).astype(np.float64))
                )
                attempted = condition.constraint_mode == L2_CONSTRAINED
                applied = attempted and proposal_l2 > report.config.constraint_radius
                if (
                    update.constraint_attempted != attempted
                    or update.constraint_applied != applied
                    or update.proposal_l2_from_initial != proposal_l2
                    or update.applied_l2_from_initial != applied_l2
                ):
                    raise ValueError(f"{update_location} constraint receipt differs")
                if not attempted and update.post_snapshot.parameter_float32_bits != (
                    update.proposal_parameter_float32_bits
                ):
                    raise ValueError(f"{update_location} unconstrained proposal changed")
                if attempted and applied_l2 > report.config.constraint_radius + 1.0e-6:
                    raise ValueError(f"{update_location} constrained result exceeds radius")
                observed_applications += int(applied)
                current = update.post_snapshot
            if phase.end_snapshot != current:
                raise ValueError(f"{phase_location} end snapshot differs")
            _validate_parameter_change(
                phase.parameter_change,
                phase.start_snapshot,
                phase.end_snapshot,
                report.config,
                name=phase_location,
            )
        if condition.final_snapshot != current:
            raise ValueError(f"{location} final snapshot differs")
        if condition.projection_applications != observed_applications:
            raise ValueError(f"{location} projection application count differs")
        for diagnostic_index, diagnostic in enumerate(
            condition.switch_diagnostics
        ):
            expected_snapshot = condition.phases[diagnostic_index].end_snapshot
            if diagnostic.snapshot != expected_snapshot:
                raise ValueError(f"{location} diagnostic snapshot differs")
            _validate_switch_diagnostic(
                diagnostic,
                manifests[diagnostic_index],
                report.config,
                name=f"{location}.switch_diagnostics[{diagnostic_index}]",
            )

    unconstrained, constrained = report.conditions
    expected_matched = MatchedInterventionAudit(
        same_initial_snapshot=True,
        same_ordered_training_stream=True,
        same_learning_rate=True,
        equal_update_opportunities=True,
        only_declared_difference="l2_parameter_constraint",
        unconstrained_projection_applications=unconstrained.projection_applications,
        constrained_projection_applications=constrained.projection_applications,
        efficacy_assessed=False,
    )
    if report.matched_intervention != expected_matched:
        raise ValueError("report matched intervention audit differs")
    expected_resources = _resource_accounting(report.conditions, report.config)
    if report.resources != expected_resources:
        raise ValueError("report exact resource accounting differs")
    if (
        report.resources.output_write_calls != 0
        or report.resources.artifact_bytes_written != 0
        or report.resources.measured_wall_clock_samples != 0
        or report.resources.wall_clock_threshold is not None
    ):
        raise ValueError("report forbidden resource/threshold claim differs")
    if report.limitations != _LIMITATIONS:
        raise ValueError("report limitations differ")
    if not report.reconstruction_mode.startswith("deterministic exact causal replay"):
        raise ValueError("report reconstruction mode differs")
    _validate_sha256(report.report_sha256, name="report sha256")
    if report.report_sha256 != _report_sha256(report):
        raise ValueError("report integrity digest differs")


def reconstruct_optimization_centric_plasticity_development(
    report: OptimizationCentricPlasticityDevelopmentReport,
) -> OptimizationCentricPlasticityDevelopmentReport:
    """Re-execute the complete bound stream and require byte-level equality."""

    _validate_structural_report(report)
    reconstructed = _execute(report.config)
    if reconstructed != report:
        raise ValueError("deterministic report reconstruction differs")
    return reconstructed


def validate_optimization_centric_plasticity_development(
    report: OptimizationCentricPlasticityDevelopmentReport,
    *,
    reconstruct: bool = True,
) -> None:
    """Validate identity, raw receipts, derived metrics, and optional full replay."""

    if type(reconstruct) is not bool:
        raise TypeError("reconstruct must be boolean")
    _validate_structural_report(report)
    if reconstruct:
        reconstructed = _execute(report.config)
        if reconstructed != report:
            raise ValueError("deterministic report reconstruction differs")


def run_optimization_centric_plasticity_development(
    config: OptimizationCentricPlasticityDevelopmentConfig | None = None,
) -> OptimizationCentricPlasticityDevelopmentReport:
    """Run the bounded in-memory diagnostic; no artifact is written."""

    resolved = (
        OptimizationCentricPlasticityDevelopmentConfig()
        if config is None
        else config
    )
    if type(resolved) is not OptimizationCentricPlasticityDevelopmentConfig:
        raise TypeError("config type is invalid")
    report = _execute(resolved)
    _validate_structural_report(report)
    return report


__all__ = [
    "ASSESSMENT_STATUS",
    "CONFIG_SCHEMA",
    "DEVELOPMENT_STATUS",
    "DormancyMeasurement",
    "FrozenLearnerSnapshot",
    "GradientMeasurement",
    "L2_CONSTRAINED",
    "OptimizationCentricPlasticityDevelopmentConfig",
    "OptimizationCentricPlasticityDevelopmentReport",
    "REPORT_SCHEMA",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SNAPSHOT_SCHEMA",
    "UNCONSTRAINED",
    "diagnose_frozen_learner_switch",
    "freeze_learner_snapshot",
    "optimization_centric_plasticity_runtime_identity",
    "optimization_centric_plasticity_source_manifest",
    "reconstruct_optimization_centric_plasticity_development",
    "run_optimization_centric_plasticity_development",
    "thaw_learner_snapshot",
    "validate_optimization_centric_plasticity_development",
    "verify_learner_snapshot",
]
