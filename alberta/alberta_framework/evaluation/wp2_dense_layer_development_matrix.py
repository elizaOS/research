"""Frozen, nonwriting WP2 dense-layer development matrix.

This module runs five small nonlinear learners on one source-frozen A/B/A
stream: SGD, Adam, Adam plus the isolated dense-layer spectral objective,
AdamO on the hidden weight matrix, and Adam plus the isolated CPR hidden-layer
transform.  Architecture, stream geometry, typed initialization/data keys, and
all optimizer/mechanism coefficients are constants in this source.  The
learner sees only observations and targets; phase labels and boundaries remain
evaluator-only.

The in-memory report retains raw prequential and fixed-probe traces, descriptive
switch recovery and forgetting, phase parameter displacement/churn, hidden-unit
dormancy and effective-rank traces, exact logical state bytes, mechanism-specific
work, and digest-bound checkpoints.  Validation can reproduce the complete
matrix by deterministic causal replay.  There is intentionally no writer.

This lane is always ``not_assessed``.  AdamO performs rectangular Gram work and
CPR performs per-example utility and scheduled reset work, so resources are
explicitly *not* called matched.  No tuning, winner, default, efficacy, evidence,
scientific claim, promotion, or WP2-exit authority follows from execution.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Final, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.adam_o import AdamO, AdamOConfig, AdamOState
from alberta_framework.core.calibrated_partial_resets import (
    CalibratedPartialResets,
    CalibratedPartialResetsConfig,
    CalibratedPartialResetsParameters,
    CalibratedPartialResetsState,
)
from alberta_framework.core.spectral_regularization import (
    SpectralRegularizationConfig,
    SpectralRegularizationState,
    SpectralRegularizer,
)

WP2_DENSE_LAYER_CONFIG_SCHEMA: Final = "alberta.wp2-dense-layer-development.config.v1"
WP2_DENSE_LAYER_CHECKPOINT_SCHEMA: Final = "alberta.wp2-dense-layer-development.checkpoint.v1"
WP2_DENSE_LAYER_REPORT_SCHEMA: Final = "alberta.wp2-dense-layer-development.report.v1"
WP2_DENSE_LAYER_PROTOCOL_ID: Final = "alberta.wp2-dense-layer-aba-frozen.v1"
WP2_DENSE_LAYER_DEVELOPMENT_STATUS: Final = "DEVELOPMENT_ONLY_NONWRITING_NO_EVIDENCE_OR_PROMOTION"
WP2_DENSE_LAYER_ASSESSMENT_STATUS: Final = "not_assessed"
WP2_DENSE_LAYER_RESOURCE_COMPARABILITY: Final = "not_assessed"

SGD_ARM: Final = "sgd"
ADAM_ARM: Final = "adam"
SPECTRAL_ARM: Final = "adam_plus_spectral_regularization"
ADAMO_ARM: Final = "adamo_hidden_matrix"
CPR_ARM: Final = "adam_plus_cpr"
WP2_DENSE_LAYER_ARM_ORDER: Final = (
    SGD_ARM,
    ADAM_ARM,
    SPECTRAL_ARM,
    ADAMO_ARM,
    CPR_ARM,
)

_PHASE_ORDER: Final = ("A", "B", "A")
_UPDATES_PER_PHASE: Final = 8
_INPUT_DIM: Final = 2
_HIDDEN_DIM: Final = 4
_PROBE_COUNT: Final = 16
_PARAMETER_COUNT: Final = _INPUT_DIM * _HIDDEN_DIM + 2 * _HIDDEN_DIM + 1
_INITIALIZATION_SEED: Final = 12_020
_DATA_SEED: Final = 12_021
_SGD_LEARNING_RATE: Final = 0.08
_ADAM_LEARNING_RATE: Final = 0.025
_ADAM_BETA1: Final = 0.9
_ADAM_BETA2: Final = 0.99
_ADAM_EPSILON: Final = 1.0e-8
_SPECTRAL_COEFFICIENT: Final = 1.0e-3
_SPECTRAL_EXPONENT: Final = 2
_SPECTRAL_POWER_ITERATIONS: Final = 1
_ADAMO_ORTHOGONALITY_STRENGTH: Final = 1.0e-3
_CPR_REPLACEMENT_RATE: Final = 0.10
_CPR_SHARPNESS: Final = 8.0
_CPR_UTILITY_DECAY: Final = 0.90
_CPR_UPDATE_FREQUENCY: Final = 4
_DORMANT_CUTOFF: Final = 0.05
_RECOVERY_GAP_FRACTION: Final = 0.50
_UINT32_MAX: Final = 2**32 - 1
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_PATHS: Final = (
    Path("alberta_framework/evaluation/wp2_dense_layer_development_matrix.py"),
    Path("alberta_framework/core/spectral_regularization.py"),
    Path("alberta_framework/core/adam_o.py"),
    Path("alberta_framework/core/calibrated_partial_resets.py"),
    Path("pyproject.toml"),
)
_RESOURCE_NONCOMPARABILITY_REASON: Final = (
    "not_assessed: AdamO adds rectangular Gram construction/gradient work and CPR "
    "adds per-example utility plus scheduled reset draws/work; logical resources are "
    "reported by arm but are not called matched"
)
_LIMITATIONS: Final = (
    "one consumed source-frozen small nonlinear A/B/A diagnostic only",
    "single hidden dense layer; no convolutional, agent, or control integration",
    "descriptive recovery, forgetting, dormancy, and rank diagnostics are not gates",
    "AdamO Gram and CPR reset work make resource comparability not_assessed",
    "no writer, tuning, winner, default, efficacy, evidence, promotion, or WP2-exit path",
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _dataclass_digest(value: object, *, blank_field: str | None = None) -> str:
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance")
    payload = dataclasses.asdict(value)
    if blank_field is not None:
        payload[blank_field] = ""
    return _digest(payload)


def _require_digest(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _float32_bits(values: object) -> tuple[int, ...]:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(-1))
    return tuple(int(word) for word in array.view(np.uint32))


def _bits_to_float32(words: tuple[int, ...]) -> np.ndarray:
    if any(type(word) is not int or not 0 <= word <= _UINT32_MAX for word in words):
        raise ValueError("float32 bit receipt contains a non-uint32 word")
    return np.asarray(words, dtype=np.uint32).view(np.float32).copy()


def _key_words(key: Array) -> tuple[int, int]:
    raw = np.asarray(jr.key_data(key), dtype=np.uint32).reshape(-1)
    if raw.shape != (2,):
        raise RuntimeError("typed PRNG key must contain two uint32 words")
    return int(raw[0]), int(raw[1])


def _uint64_words(value: int) -> tuple[int, int]:
    if type(value) is not int or not 0 <= value <= 2**64 - 1:
        raise ValueError("logical clock must be uint64")
    return (value >> 32) & _UINT32_MAX, value & _UINT32_MAX


@dataclasses.dataclass(frozen=True, slots=True)
class WP2DenseLayerDevelopmentConfig:
    """The exact source-frozen development schedule; fields are not tunable."""

    schema: str = WP2_DENSE_LAYER_CONFIG_SCHEMA
    protocol_id: str = WP2_DENSE_LAYER_PROTOCOL_ID
    phase_order: tuple[str, str, str] = _PHASE_ORDER
    updates_per_phase: int = _UPDATES_PER_PHASE
    input_dim: int = _INPUT_DIM
    hidden_dim: int = _HIDDEN_DIM
    probe_count: int = _PROBE_COUNT
    parameter_count: int = _PARAMETER_COUNT
    initialization_seed: int = _INITIALIZATION_SEED
    data_seed: int = _DATA_SEED
    sgd_learning_rate: float = _SGD_LEARNING_RATE
    adam_learning_rate: float = _ADAM_LEARNING_RATE
    adam_beta1: float = _ADAM_BETA1
    adam_beta2: float = _ADAM_BETA2
    adam_epsilon: float = _ADAM_EPSILON
    spectral_coefficient: float = _SPECTRAL_COEFFICIENT
    spectral_exponent: int = _SPECTRAL_EXPONENT
    spectral_power_iterations: int = _SPECTRAL_POWER_ITERATIONS
    adamo_orthogonality_strength: float = _ADAMO_ORTHOGONALITY_STRENGTH
    cpr_replacement_rate: float = _CPR_REPLACEMENT_RATE
    cpr_sharpness: float = _CPR_SHARPNESS
    cpr_utility_decay: float = _CPR_UTILITY_DECAY
    cpr_update_frequency: int = _CPR_UPDATE_FREQUENCY
    dormant_descriptive_cutoff: float = _DORMANT_CUTOFF
    recovery_gap_fraction: float = _RECOVERY_GAP_FRACTION

    def __post_init__(self) -> None:
        expected: dict[str, object] = {
            "schema": WP2_DENSE_LAYER_CONFIG_SCHEMA,
            "protocol_id": WP2_DENSE_LAYER_PROTOCOL_ID,
            "phase_order": _PHASE_ORDER,
            "updates_per_phase": _UPDATES_PER_PHASE,
            "input_dim": _INPUT_DIM,
            "hidden_dim": _HIDDEN_DIM,
            "probe_count": _PROBE_COUNT,
            "parameter_count": _PARAMETER_COUNT,
            "initialization_seed": _INITIALIZATION_SEED,
            "data_seed": _DATA_SEED,
            "sgd_learning_rate": _SGD_LEARNING_RATE,
            "adam_learning_rate": _ADAM_LEARNING_RATE,
            "adam_beta1": _ADAM_BETA1,
            "adam_beta2": _ADAM_BETA2,
            "adam_epsilon": _ADAM_EPSILON,
            "spectral_coefficient": _SPECTRAL_COEFFICIENT,
            "spectral_exponent": _SPECTRAL_EXPONENT,
            "spectral_power_iterations": _SPECTRAL_POWER_ITERATIONS,
            "adamo_orthogonality_strength": _ADAMO_ORTHOGONALITY_STRENGTH,
            "cpr_replacement_rate": _CPR_REPLACEMENT_RATE,
            "cpr_sharpness": _CPR_SHARPNESS,
            "cpr_utility_decay": _CPR_UTILITY_DECAY,
            "cpr_update_frequency": _CPR_UPDATE_FREQUENCY,
            "dormant_descriptive_cutoff": _DORMANT_CUTOFF,
            "recovery_gap_fraction": _RECOVERY_GAP_FRACTION,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"{name} is source-frozen and cannot be changed")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class SourceFileIdentity:
    path: str
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class DenseLayerCheckpoint:
    schema: str
    arm_name: str
    config_sha256: str
    step: int
    step_words_uint32: tuple[int, int]
    parameter_count: int
    parameter_float32_bits: tuple[int, ...]
    optimizer_state_float32_bits: tuple[int, ...]
    mechanism_state_float32_bits: tuple[int, ...]
    mechanism_state_uint32_words: tuple[int, ...]
    sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class PrequentialReceipt:
    global_step: int
    phase_index: int
    phase_step: int
    regime_id: str
    observation_float32_bits: tuple[int, int]
    target_float32_bits: int
    prediction_float32_bits: int
    squared_error: float


@dataclasses.dataclass(frozen=True, slots=True)
class ProbeReceipt:
    completed_updates: int
    a_mse: float
    b_mse: float


@dataclasses.dataclass(frozen=True, slots=True)
class RepresentationReceipt:
    completed_updates: int
    mean_absolute_activation_float32_bits: tuple[int, ...]
    dormant_mask: tuple[bool, ...]
    dormant_fraction: float
    singular_values_float32_bits: tuple[int, ...]
    effective_rank: float


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterChangeReceipt:
    delta_float32_bits: tuple[int, ...]
    l2_displacement: float
    relative_l2_displacement: float
    max_absolute_displacement: float
    changed_coordinate_count: int
    bitwise_churn_fraction: float
    sign_flip_count: int
    sign_flip_fraction: float


@dataclasses.dataclass(frozen=True, slots=True)
class PhaseMetrics:
    phase_index: int
    regime_id: str
    prequential_mse: float
    entry_current_task_mse: float
    exit_current_task_mse: float
    old_task_id: str | None
    entry_old_task_mse: float | None
    exit_old_task_mse: float | None
    old_task_forgetting: float | None
    parameter_change: ParameterChangeReceipt


@dataclasses.dataclass(frozen=True, slots=True)
class SwitchMetrics:
    switch_index: int
    old_regime_id: str
    incoming_regime_id: str
    entry_incoming_mse: float
    best_incoming_mse: float
    exit_incoming_mse: float
    half_gap_recovery_target: float
    half_gap_recovery_steps: int
    entry_old_mse: float
    exit_old_mse: float
    old_task_forgetting: float
    recurring_a_reference_mse: float | None
    recurring_a_entry_forgetting: float | None


@dataclasses.dataclass(frozen=True, slots=True)
class ArmWorkReceipt:
    task_gradient_evaluations: int
    sgd_parameter_update_calls: int
    adam_moment_updates: int
    spectral_evaluations: int
    spectral_power_matvecs: int
    spectral_backward_evaluations: int
    adamo_gram_gradient_evaluations: int
    adamo_gram_matrix_elements: int
    cpr_per_example_gradient_evaluations: int
    cpr_reset_events: int
    cpr_initialization_draws: int
    output_write_calls: int
    artifact_bytes_written: int


@dataclasses.dataclass(frozen=True, slots=True)
class ArmStateBytes:
    parameter_bytes: int
    base_optimizer_state_bytes: int
    mechanism_state_bytes: int
    protocol_clock_bytes: int
    total_persistent_bytes: int


@dataclasses.dataclass(frozen=True, slots=True)
class ArmDevelopmentReport:
    name: str
    mechanism_status: str
    training_stream_sha256: str
    initial_checkpoint: DenseLayerCheckpoint
    final_checkpoint: DenseLayerCheckpoint
    prequential_trace: tuple[PrequentialReceipt, ...]
    probe_trace: tuple[ProbeReceipt, ...]
    representation_trace: tuple[RepresentationReceipt, ...]
    phase_metrics: tuple[PhaseMetrics, ...]
    switch_metrics: tuple[SwitchMetrics, ...]
    recurring_a_entry_forgetting: float
    work: ArmWorkReceipt
    state_bytes: ArmStateBytes


@dataclasses.dataclass(frozen=True, slots=True)
class MatrixResourceReceipt:
    arm_count: int
    total_update_opportunities: int
    total_task_gradient_evaluations: int
    total_persistent_bytes_across_isolated_arms: int
    output_write_calls: int
    artifact_bytes_written: int
    resources_matched: bool
    comparability_assessment: str


@dataclasses.dataclass(frozen=True, slots=True)
class WP2DenseLayerDevelopmentReport:
    schema: str
    status: str
    assessment_status: str
    resource_comparability: str
    resources_matched: bool
    resource_noncomparability_reason: str
    development_only: bool
    output_writes_authorized: bool
    tuning_authorized: bool
    winner_selection_authorized: bool
    default_selection_authorized: bool
    efficacy_claim_authorized: bool
    evidence_authorized: bool
    promotion_authorized: bool
    scientific_promotion_allowed: bool
    config: WP2DenseLayerDevelopmentConfig
    config_sha256: str
    source_manifest: tuple[SourceFileIdentity, ...]
    source_manifest_sha256: str
    initialization_key_words_uint32: tuple[int, int]
    data_key_words_uint32: tuple[int, int]
    protocol_sha256: str
    arms: tuple[ArmDevelopmentReport, ...]
    resources: MatrixResourceReceipt
    replay_verified: bool
    reconstruction_mode: str
    limitations: tuple[str, ...]
    report_sha256: str

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], dataclasses.asdict(self))


def _source_manifest() -> tuple[SourceFileIdentity, ...]:
    return tuple(
        SourceFileIdentity(
            path=path.as_posix(),
            sha256=hashlib.sha256((_REPO_ROOT / path).read_bytes()).hexdigest(),
        )
        for path in _SOURCE_PATHS
    )


def _config_sha256(config: WP2DenseLayerDevelopmentConfig) -> str:
    return _dataclass_digest(config)


def _checkpoint_sha256(checkpoint: DenseLayerCheckpoint) -> str:
    return _dataclass_digest(checkpoint, blank_field="sha256")


def _report_sha256(report: WP2DenseLayerDevelopmentReport) -> str:
    return _dataclass_digest(report, blank_field="report_sha256")


def _pack_parameters(
    incoming_weight: Array,
    hidden_bias: Array,
    outgoing_weight: Array,
    output_bias: Array,
) -> Array:
    return jnp.concatenate(
        (
            incoming_weight.reshape(-1),
            hidden_bias,
            outgoing_weight.reshape(-1),
            output_bias.reshape(1),
        )
    ).astype(jnp.float32)


def _unpack_parameters(parameters: Array) -> tuple[Array, Array, Array, Array]:
    incoming_count = _INPUT_DIM * _HIDDEN_DIM
    incoming = parameters[:incoming_count].reshape((_INPUT_DIM, _HIDDEN_DIM))
    hidden_bias = parameters[incoming_count : incoming_count + _HIDDEN_DIM]
    output_start = incoming_count + _HIDDEN_DIM
    outgoing = parameters[output_start : output_start + _HIDDEN_DIM].reshape((_HIDDEN_DIM, 1))
    output_bias = parameters[-1]
    return incoming, hidden_bias, outgoing, output_bias


def _predict_and_hidden(parameters: Array, observations: Array) -> tuple[Array, Array]:
    incoming, hidden_bias, outgoing, output_bias = _unpack_parameters(parameters)
    hidden = jnp.tanh(observations @ incoming + hidden_bias)
    prediction = jnp.tanh(hidden @ outgoing + output_bias).reshape(-1)
    return prediction, hidden


def _single_loss(parameters: Array, observation: Array, target: Array) -> Array:
    prediction, _ = _predict_and_hidden(parameters, observation[jnp.newaxis, :])
    return cast(Array, jnp.float32(0.5) * jnp.square(prediction[0] - target))


def _mse(parameters: Array, observations: Array, targets: Array) -> float:
    prediction, _ = _predict_and_hidden(parameters, observations)
    return float(jnp.mean(jnp.square(prediction - targets)))


def _target(regime: str, observations: Array) -> Array:
    first = observations[:, 0]
    second = observations[:, 1]
    if regime == "A":
        raw = 1.25 * first * second + 0.35 * first - 0.20 * second
    elif regime == "B":
        raw = -1.10 * first * second - 0.25 * first + 0.45 * second
    else:
        raise ValueError("unknown evaluator regime")
    return jnp.tanh(raw).astype(jnp.float32)


def _initial_parameters(key: Array) -> Array:
    incoming_key, hidden_bias_key, outgoing_key, output_bias_key = jr.split(key, 4)
    incoming = jr.uniform(
        incoming_key,
        (_INPUT_DIM, _HIDDEN_DIM),
        dtype=jnp.float32,
        minval=-math.sqrt(6.0 / _INPUT_DIM),
        maxval=math.sqrt(6.0 / _INPUT_DIM),
    )
    hidden_bias = jr.uniform(
        hidden_bias_key,
        (_HIDDEN_DIM,),
        dtype=jnp.float32,
        minval=-0.05,
        maxval=0.05,
    )
    outgoing = jr.uniform(
        outgoing_key,
        (_HIDDEN_DIM, 1),
        dtype=jnp.float32,
        minval=-math.sqrt(6.0 / _HIDDEN_DIM),
        maxval=math.sqrt(6.0 / _HIDDEN_DIM),
    )
    output_bias = jr.uniform(
        output_bias_key,
        (),
        dtype=jnp.float32,
        minval=-0.05,
        maxval=0.05,
    )
    return _pack_parameters(incoming, hidden_bias, outgoing, output_bias)


def _stream() -> tuple[Array, Array, dict[str, Array], str]:
    training_key, probe_key = jr.split(jr.key(_DATA_SEED))
    training_observations = jr.uniform(
        training_key,
        (_UPDATES_PER_PHASE, _INPUT_DIM),
        dtype=jnp.float32,
        minval=-1.0,
        maxval=1.0,
    )
    probe_observations = jr.uniform(
        probe_key,
        (_PROBE_COUNT, _INPUT_DIM),
        dtype=jnp.float32,
        minval=-1.0,
        maxval=1.0,
    )
    targets = {regime: _target(regime, training_observations) for regime in ("A", "B")}
    manifest = []
    for phase_index, regime in enumerate(_PHASE_ORDER):
        for phase_step in range(_UPDATES_PER_PHASE):
            manifest.append(
                {
                    "phase_index": phase_index,
                    "phase_step": phase_step,
                    "regime_id": regime,
                    "observation_float32_bits": _float32_bits(training_observations[phase_step]),
                    "target_float32_bits": _float32_bits(targets[regime][phase_step])[0],
                }
            )
    return training_observations, probe_observations, targets, _digest(manifest)


def _probe_receipt(
    parameters: Array,
    probe_observations: Array,
    completed_updates: int,
) -> ProbeReceipt:
    return ProbeReceipt(
        completed_updates=completed_updates,
        a_mse=_mse(parameters, probe_observations, _target("A", probe_observations)),
        b_mse=_mse(parameters, probe_observations, _target("B", probe_observations)),
    )


def _representation_receipt(
    parameters: Array,
    probe_observations: Array,
    completed_updates: int,
) -> RepresentationReceipt:
    _, hidden = _predict_and_hidden(parameters, probe_observations)
    mean_absolute = jnp.mean(jnp.abs(hidden), axis=0)
    dormant = mean_absolute < jnp.float32(_DORMANT_CUTOFF)
    centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
    singular = jnp.linalg.svd(centered, compute_uv=False)
    total = jnp.sum(singular)
    probabilities = jnp.where(total > 0.0, singular / total, jnp.zeros_like(singular))
    entropy = -jnp.sum(jnp.where(probabilities > 0.0, probabilities * jnp.log(probabilities), 0.0))
    effective_rank = jnp.where(total > 0.0, jnp.exp(entropy), 0.0)
    return RepresentationReceipt(
        completed_updates=completed_updates,
        mean_absolute_activation_float32_bits=_float32_bits(mean_absolute),
        dormant_mask=tuple(bool(value) for value in np.asarray(dormant)),
        dormant_fraction=float(jnp.mean(dormant.astype(jnp.float32))),
        singular_values_float32_bits=_float32_bits(singular),
        effective_rank=float(effective_rank),
    )


def _parameter_change(start: Array, end: Array) -> ParameterChangeReceipt:
    start_array = np.asarray(start, dtype=np.float32)
    end_array = np.asarray(end, dtype=np.float32)
    delta = np.asarray(end_array - start_array, dtype=np.float32)
    changed = start_array.view(np.uint32) != end_array.view(np.uint32)
    sign_flip = (np.signbit(start_array) != np.signbit(end_array)) & (
        (start_array != 0.0) | (end_array != 0.0)
    )
    start_norm = float(np.linalg.norm(start_array.astype(np.float64)))
    displacement = float(np.linalg.norm(delta.astype(np.float64)))
    return ParameterChangeReceipt(
        delta_float32_bits=_float32_bits(delta),
        l2_displacement=displacement,
        relative_l2_displacement=displacement / max(start_norm, np.finfo(np.float64).tiny),
        max_absolute_displacement=float(np.max(np.abs(delta))),
        changed_coordinate_count=int(np.count_nonzero(changed)),
        bitwise_churn_fraction=float(np.mean(changed)),
        sign_flip_count=int(np.count_nonzero(sign_flip)),
        sign_flip_fraction=float(np.mean(sign_flip)),
    )


def _adam_update(
    parameters: Array,
    gradient: Array,
    first_moment: Array,
    second_moment: Array,
    step: int,
    mask: Array,
) -> tuple[Array, Array, Array]:
    beta1 = jnp.float32(_ADAM_BETA1)
    beta2 = jnp.float32(_ADAM_BETA2)
    candidate_first = beta1 * first_moment + (1.0 - beta1) * gradient
    candidate_second = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
    first = jnp.where(mask, candidate_first, first_moment)
    second = jnp.where(mask, candidate_second, second_moment)
    next_step = jnp.float32(step + 1)
    first_hat = first / (1.0 - beta1**next_step)
    second_hat = second / (1.0 - beta2**next_step)
    delta = (
        jnp.float32(_ADAM_LEARNING_RATE)
        * first_hat
        / (jnp.sqrt(second_hat) + jnp.float32(_ADAM_EPSILON))
    )
    return parameters - jnp.where(mask, delta, 0.0), first, second


@dataclasses.dataclass
class _ArmRuntime:
    parameters: Array
    first_moment: Array
    second_moment: Array
    spectral_state: SpectralRegularizationState | None
    adamo_state: AdamOState | None
    cpr_state: CalibratedPartialResetsState | None
    step: int = 0
    cpr_reset_events: int = 0


def _mechanisms() -> tuple[SpectralRegularizer, AdamO, CalibratedPartialResets]:
    maximum_updates = len(_PHASE_ORDER) * _UPDATES_PER_PHASE
    return (
        SpectralRegularizer(
            SpectralRegularizationConfig(
                output_dim=_HIDDEN_DIM,
                input_dim=_INPUT_DIM,
                coefficient=_SPECTRAL_COEFFICIENT,
                exponent=_SPECTRAL_EXPONENT,
                power_iterations=_SPECTRAL_POWER_ITERATIONS,
                maximum_updates=maximum_updates,
            )
        ),
        AdamO(
            AdamOConfig(
                rows=_HIDDEN_DIM,
                columns=_INPUT_DIM,
                learning_rate=_ADAM_LEARNING_RATE,
                beta1=_ADAM_BETA1,
                beta2=_ADAM_BETA2,
                epsilon=_ADAM_EPSILON,
                orthogonality_strength=_ADAMO_ORTHOGONALITY_STRENGTH,
                isometry_step_size=_ADAM_LEARNING_RATE,
                maximum_updates=maximum_updates,
            )
        ),
        CalibratedPartialResets(
            CalibratedPartialResetsConfig(
                input_dim=_INPUT_DIM,
                unit_count=_HIDDEN_DIM,
                output_dim=1,
                replacement_rate=_CPR_REPLACEMENT_RATE,
                sharpness=_CPR_SHARPNESS,
                utility_decay=_CPR_UTILITY_DECAY,
                update_frequency=_CPR_UPDATE_FREQUENCY,
                initialization_scale=1.0,
                maximum_updates=maximum_updates,
            )
        ),
    )


def _checkpoint(
    arm_name: str,
    runtime: _ArmRuntime,
    config: WP2DenseLayerDevelopmentConfig,
) -> DenseLayerCheckpoint:
    optimizer_bits: tuple[int, ...] = ()
    if arm_name != SGD_ARM:
        optimizer_bits = _float32_bits(
            jnp.concatenate((runtime.first_moment, runtime.second_moment))
        )
    mechanism_float: tuple[int, ...] = ()
    mechanism_words: tuple[int, ...] = ()
    if runtime.spectral_state is not None:
        mechanism_float = _float32_bits(runtime.spectral_state.right_probe)
        mechanism_words = (
            *_key_words(runtime.spectral_state.rng_key),
            *(int(value) for value in np.asarray(runtime.spectral_state.update_count_words)),
        )
    elif runtime.adamo_state is not None:
        mechanism_float = _float32_bits(
            jnp.concatenate(
                (
                    runtime.adamo_state.first_moment.reshape(-1),
                    runtime.adamo_state.second_moment.reshape(-1),
                )
            )
        )
        mechanism_words = tuple(
            int(value) for value in np.asarray(runtime.adamo_state.update_count_words)
        )
    elif runtime.cpr_state is not None:
        mechanism_float = _float32_bits(runtime.cpr_state.utility)
        mechanism_words = (
            *_key_words(runtime.cpr_state.rng_key),
            *(int(value) for value in np.asarray(runtime.cpr_state.update_count_words)),
            *(int(value) for value in np.asarray(runtime.cpr_state.reset_event_count_words)),
        )
    bare = DenseLayerCheckpoint(
        schema=WP2_DENSE_LAYER_CHECKPOINT_SCHEMA,
        arm_name=arm_name,
        config_sha256=_config_sha256(config),
        step=runtime.step,
        step_words_uint32=_uint64_words(runtime.step),
        parameter_count=_PARAMETER_COUNT,
        parameter_float32_bits=_float32_bits(runtime.parameters),
        optimizer_state_float32_bits=optimizer_bits,
        mechanism_state_float32_bits=mechanism_float,
        mechanism_state_uint32_words=mechanism_words,
        sha256="",
    )
    return dataclasses.replace(bare, sha256=_checkpoint_sha256(bare))


def _validate_checkpoint(
    checkpoint: DenseLayerCheckpoint,
    config: WP2DenseLayerDevelopmentConfig,
) -> None:
    if type(checkpoint) is not DenseLayerCheckpoint:
        raise ValueError("checkpoint type is invalid")
    if checkpoint.schema != WP2_DENSE_LAYER_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema is invalid")
    if checkpoint.arm_name not in WP2_DENSE_LAYER_ARM_ORDER:
        raise ValueError("checkpoint arm is invalid")
    if checkpoint.config_sha256 != _config_sha256(config):
        raise ValueError("checkpoint config binding differs")
    if checkpoint.step_words_uint32 != _uint64_words(checkpoint.step):
        raise ValueError("checkpoint clock differs")
    if (
        checkpoint.parameter_count != _PARAMETER_COUNT
        or len(checkpoint.parameter_float32_bits) != _PARAMETER_COUNT
    ):
        raise ValueError("checkpoint parameter shape differs")
    for payload in (
        checkpoint.parameter_float32_bits,
        checkpoint.optimizer_state_float32_bits,
        checkpoint.mechanism_state_float32_bits,
    ):
        if not np.all(np.isfinite(_bits_to_float32(payload))):
            raise ValueError("checkpoint float payload is nonfinite")
    if any(
        type(word) is not int or not 0 <= word <= _UINT32_MAX
        for word in checkpoint.mechanism_state_uint32_words
    ):
        raise ValueError("checkpoint mechanism words are invalid")
    _require_digest(checkpoint.sha256, name="checkpoint sha256")
    if checkpoint.sha256 != _checkpoint_sha256(checkpoint):
        raise ValueError("checkpoint integrity digest differs")


def restore_wp2_dense_layer_checkpoint_parameters(
    checkpoint: DenseLayerCheckpoint,
    config: WP2DenseLayerDevelopmentConfig,
) -> np.ndarray:
    """Restore a fresh parameter vector from one in-memory checkpoint receipt."""

    if type(config) is not WP2DenseLayerDevelopmentConfig:
        raise TypeError("config type is invalid")
    _validate_checkpoint(checkpoint, config)
    return _bits_to_float32(checkpoint.parameter_float32_bits)


def _state_bytes(
    arm_name: str,
    spectral: SpectralRegularizer,
    adamo: AdamO,
    cpr: CalibratedPartialResets,
) -> ArmStateBytes:
    parameter_bytes = _PARAMETER_COUNT * 4
    base_optimizer = 0 if arm_name == SGD_ARM else 2 * _PARAMETER_COUNT * 4
    mechanism = 0
    if arm_name == SPECTRAL_ARM:
        mechanism = spectral.resource_declaration().persistent_bytes
    elif arm_name == ADAMO_ARM:
        mechanism = adamo.resource_declaration().persistent_bytes
    elif arm_name == CPR_ARM:
        mechanism = cpr.resource_declaration().persistent_bytes
    clock = 8
    return ArmStateBytes(
        parameter_bytes=parameter_bytes,
        base_optimizer_state_bytes=base_optimizer,
        mechanism_state_bytes=mechanism,
        protocol_clock_bytes=clock,
        total_persistent_bytes=parameter_bytes + base_optimizer + mechanism + clock,
    )


def _run_arm(
    arm_name: str,
    initial_parameters: Array,
    training_observations: Array,
    probe_observations: Array,
    targets: dict[str, Array],
    training_stream_sha256: str,
    config: WP2DenseLayerDevelopmentConfig,
) -> ArmDevelopmentReport:
    spectral, adamo, cpr = _mechanisms()
    zeros = jnp.zeros((_PARAMETER_COUNT,), dtype=jnp.float32)
    initial_key = jr.key(_INITIALIZATION_SEED)
    runtime = _ArmRuntime(
        parameters=jnp.array(initial_parameters, copy=True),
        first_moment=zeros,
        second_moment=zeros,
        spectral_state=(
            spectral.init(jr.fold_in(initial_key, 0x53504543)) if arm_name == SPECTRAL_ARM else None
        ),
        adamo_state=adamo.init() if arm_name == ADAMO_ARM else None,
        cpr_state=(cpr.init(jr.fold_in(initial_key, 0x43505231)) if arm_name == CPR_ARM else None),
    )
    initial_checkpoint = _checkpoint(arm_name, runtime, config)
    prequential: list[PrequentialReceipt] = []
    probes = [_probe_receipt(runtime.parameters, probe_observations, 0)]
    representations = [_representation_receipt(runtime.parameters, probe_observations, 0)]
    phases: list[PhaseMetrics] = []
    switches: list[SwitchMetrics] = []
    end_a1_mse: float | None = None
    incoming_count = _INPUT_DIM * _HIDDEN_DIM
    full_mask = jnp.ones((_PARAMETER_COUNT,), dtype=jnp.bool_)
    auxiliary_mask = full_mask.at[:incoming_count].set(False)

    for phase_index, regime in enumerate(_PHASE_ORDER):
        phase_start = jnp.array(runtime.parameters, copy=True)
        entry_probe = probes[-1]
        phase_losses: list[float] = []
        phase_post_incoming: list[float] = []
        old_regime = None if phase_index == 0 else ("A" if regime == "B" else "B")
        entry_current = entry_probe.a_mse if regime == "A" else entry_probe.b_mse
        entry_old = (
            None
            if old_regime is None
            else entry_probe.a_mse
            if old_regime == "A"
            else entry_probe.b_mse
        )

        for phase_step in range(_UPDATES_PER_PHASE):
            observation = training_observations[phase_step]
            target = targets[regime][phase_step]
            prediction, _ = _predict_and_hidden(runtime.parameters, observation[jnp.newaxis, :])
            squared_error = float(jnp.square(prediction[0] - target))
            phase_losses.append(squared_error)
            prequential.append(
                PrequentialReceipt(
                    global_step=runtime.step,
                    phase_index=phase_index,
                    phase_step=phase_step,
                    regime_id=regime,
                    observation_float32_bits=cast(tuple[int, int], _float32_bits(observation)),
                    target_float32_bits=_float32_bits(target)[0],
                    prediction_float32_bits=_float32_bits(prediction[0])[0],
                    squared_error=squared_error,
                )
            )
            gradient = jax.grad(_single_loss)(runtime.parameters, observation, target)

            if arm_name == SGD_ARM:
                runtime.parameters = runtime.parameters - jnp.float32(_SGD_LEARNING_RATE) * gradient
            elif arm_name == SPECTRAL_ARM:
                incoming, hidden_bias, _, _ = _unpack_parameters(runtime.parameters)
                if runtime.spectral_state is None:
                    raise RuntimeError("spectral state is missing")
                spectral_result = spectral.evaluate(runtime.spectral_state, incoming.T, hidden_bias)
                if not bool(spectral_result.accepted):
                    raise RuntimeError("source-frozen spectral update rejected")
                regularized_gradient = gradient.at[:incoming_count].add(
                    spectral_result.weight_gradient.T.reshape(-1)
                )
                regularized_gradient = regularized_gradient.at[
                    incoming_count : incoming_count + _HIDDEN_DIM
                ].add(spectral_result.bias_gradient)
                (
                    runtime.parameters,
                    runtime.first_moment,
                    runtime.second_moment,
                ) = _adam_update(
                    runtime.parameters,
                    regularized_gradient,
                    runtime.first_moment,
                    runtime.second_moment,
                    runtime.step,
                    full_mask,
                )
                runtime.spectral_state = spectral_result.state
            elif arm_name == ADAMO_ARM:
                if runtime.adamo_state is None:
                    raise RuntimeError("AdamO state is missing")
                source_incoming, _, _, _ = _unpack_parameters(runtime.parameters)
                base_parameters, runtime.first_moment, runtime.second_moment = _adam_update(
                    runtime.parameters,
                    gradient,
                    runtime.first_moment,
                    runtime.second_moment,
                    runtime.step,
                    auxiliary_mask,
                )
                adamo_result = adamo.update(
                    runtime.adamo_state,
                    source_incoming.T,
                    gradient[:incoming_count].reshape((_INPUT_DIM, _HIDDEN_DIM)).T,
                )
                if not bool(adamo_result.accepted):
                    raise RuntimeError("source-frozen AdamO update rejected")
                _, hidden_bias, outgoing, output_bias = _unpack_parameters(base_parameters)
                runtime.parameters = _pack_parameters(
                    (source_incoming.T - adamo_result.parameter_delta).T,
                    hidden_bias,
                    outgoing,
                    output_bias,
                )
                runtime.adamo_state = adamo_result.state
            else:
                (
                    runtime.parameters,
                    runtime.first_moment,
                    runtime.second_moment,
                ) = _adam_update(
                    runtime.parameters,
                    gradient,
                    runtime.first_moment,
                    runtime.second_moment,
                    runtime.step,
                    full_mask,
                )
                if arm_name == CPR_ARM:
                    if runtime.cpr_state is None:
                        raise RuntimeError("CPR state is missing")
                    incoming, hidden_bias, outgoing, output_bias = _unpack_parameters(
                        runtime.parameters
                    )
                    cpr_result = cpr.update_after_optimizer(
                        runtime.cpr_state,
                        CalibratedPartialResetsParameters(  # type: ignore[call-arg]
                            incoming_weight=incoming,
                            outgoing_weight=outgoing,
                        ),
                        gradient[:incoming_count].reshape((1, _INPUT_DIM, _HIDDEN_DIM)),
                    )
                    if not bool(cpr_result.accepted):
                        raise RuntimeError("source-frozen CPR update rejected")
                    runtime.parameters = _pack_parameters(
                        cpr_result.parameters.incoming_weight,
                        hidden_bias,
                        cpr_result.parameters.outgoing_weight,
                        output_bias,
                    )
                    runtime.cpr_state = cpr_result.state
                    runtime.cpr_reset_events += int(bool(cpr_result.reset_applied))

            if not bool(jnp.all(jnp.isfinite(runtime.parameters))):
                raise RuntimeError("source-frozen update produced nonfinite parameters")
            runtime.step += 1
            post_probe = _probe_receipt(runtime.parameters, probe_observations, runtime.step)
            probes.append(post_probe)
            representations.append(
                _representation_receipt(runtime.parameters, probe_observations, runtime.step)
            )
            phase_post_incoming.append(post_probe.a_mse if regime == "A" else post_probe.b_mse)

        exit_probe = probes[-1]
        exit_current = exit_probe.a_mse if regime == "A" else exit_probe.b_mse
        exit_old = (
            None
            if old_regime is None
            else exit_probe.a_mse
            if old_regime == "A"
            else exit_probe.b_mse
        )
        forgetting = None if entry_old is None or exit_old is None else exit_old - entry_old
        phases.append(
            PhaseMetrics(
                phase_index=phase_index,
                regime_id=regime,
                prequential_mse=float(np.mean(np.asarray(phase_losses, dtype=np.float64))),
                entry_current_task_mse=entry_current,
                exit_current_task_mse=exit_current,
                old_task_id=old_regime,
                entry_old_task_mse=entry_old,
                exit_old_task_mse=exit_old,
                old_task_forgetting=forgetting,
                parameter_change=_parameter_change(phase_start, runtime.parameters),
            )
        )
        if phase_index == 0:
            end_a1_mse = exit_current
        else:
            best = min(entry_current, *phase_post_incoming)
            target_mse = entry_current - _RECOVERY_GAP_FRACTION * (entry_current - best)
            recovery_steps = 0
            if best < entry_current:
                recovery_steps = next(
                    index
                    for index, value in enumerate(phase_post_incoming, start=1)
                    if value <= target_mse
                )
            recurrence_reference = end_a1_mse if regime == "A" else None
            recurrence_forgetting = (
                entry_current - recurrence_reference if recurrence_reference is not None else None
            )
            switches.append(
                SwitchMetrics(
                    switch_index=phase_index - 1,
                    old_regime_id=cast(str, old_regime),
                    incoming_regime_id=regime,
                    entry_incoming_mse=entry_current,
                    best_incoming_mse=best,
                    exit_incoming_mse=exit_current,
                    half_gap_recovery_target=target_mse,
                    half_gap_recovery_steps=recovery_steps,
                    entry_old_mse=cast(float, entry_old),
                    exit_old_mse=cast(float, exit_old),
                    old_task_forgetting=cast(float, forgetting),
                    recurring_a_reference_mse=recurrence_reference,
                    recurring_a_entry_forgetting=recurrence_forgetting,
                )
            )

    if len(switches) != 2 or switches[1].recurring_a_entry_forgetting is None:
        raise RuntimeError("A/B/A recurrence diagnostics were not completed")
    total_updates = len(_PHASE_ORDER) * _UPDATES_PER_PHASE
    spectral_resource = spectral.resource_declaration()
    adamo_resource = adamo.resource_declaration()
    cpr_resource = cpr.resource_declaration()
    work = ArmWorkReceipt(
        task_gradient_evaluations=total_updates,
        sgd_parameter_update_calls=total_updates if arm_name == SGD_ARM else 0,
        adam_moment_updates=0 if arm_name == SGD_ARM else total_updates,
        spectral_evaluations=total_updates if arm_name == SPECTRAL_ARM else 0,
        spectral_power_matvecs=(
            total_updates * spectral_resource.power_matvecs_per_evaluation
            if arm_name == SPECTRAL_ARM
            else 0
        ),
        spectral_backward_evaluations=(
            total_updates * spectral_resource.backward_evaluations_per_update
            if arm_name == SPECTRAL_ARM
            else 0
        ),
        adamo_gram_gradient_evaluations=(total_updates if arm_name == ADAMO_ARM else 0),
        adamo_gram_matrix_elements=(
            total_updates * adamo_resource.gram_matrix_elements if arm_name == ADAMO_ARM else 0
        ),
        cpr_per_example_gradient_evaluations=(total_updates if arm_name == CPR_ARM else 0),
        cpr_reset_events=runtime.cpr_reset_events if arm_name == CPR_ARM else 0,
        cpr_initialization_draws=(
            runtime.cpr_reset_events * cpr_resource.initialization_draws_per_reset_event
            if arm_name == CPR_ARM
            else 0
        ),
        output_write_calls=0,
        artifact_bytes_written=0,
    )
    status = {
        SGD_ARM: "baseline-sgd-development-only-not-assessed",
        ADAM_ARM: "baseline-adam-development-only-not-assessed",
        SPECTRAL_ARM: "isolated-spectral-dense-layer-development-only-not-assessed",
        ADAMO_ARM: "isolated-adamo-dense-layer-development-only-not-assessed",
        CPR_ARM: "isolated-cpr-dense-layer-development-only-not-assessed",
    }[arm_name]
    return ArmDevelopmentReport(
        name=arm_name,
        mechanism_status=status,
        training_stream_sha256=training_stream_sha256,
        initial_checkpoint=initial_checkpoint,
        final_checkpoint=_checkpoint(arm_name, runtime, config),
        prequential_trace=tuple(prequential),
        probe_trace=tuple(probes),
        representation_trace=tuple(representations),
        phase_metrics=tuple(phases),
        switch_metrics=tuple(switches),
        recurring_a_entry_forgetting=switches[1].recurring_a_entry_forgetting,
        work=work,
        state_bytes=_state_bytes(arm_name, spectral, adamo, cpr),
    )


def _execute() -> WP2DenseLayerDevelopmentReport:
    config = WP2DenseLayerDevelopmentConfig()
    initialization_key = jr.key(_INITIALIZATION_SEED)
    initial_parameters = _initial_parameters(initialization_key)
    training, probes, targets, stream_sha256 = _stream()
    arms = tuple(
        _run_arm(
            arm_name,
            initial_parameters,
            training,
            probes,
            targets,
            stream_sha256,
            config,
        )
        for arm_name in WP2_DENSE_LAYER_ARM_ORDER
    )
    source_manifest = _source_manifest()
    resources = MatrixResourceReceipt(
        arm_count=len(arms),
        total_update_opportunities=len(arms) * len(_PHASE_ORDER) * _UPDATES_PER_PHASE,
        total_task_gradient_evaluations=sum(arm.work.task_gradient_evaluations for arm in arms),
        total_persistent_bytes_across_isolated_arms=sum(
            arm.state_bytes.total_persistent_bytes for arm in arms
        ),
        output_write_calls=0,
        artifact_bytes_written=0,
        resources_matched=False,
        comparability_assessment=WP2_DENSE_LAYER_RESOURCE_COMPARABILITY,
    )
    bare = WP2DenseLayerDevelopmentReport(
        schema=WP2_DENSE_LAYER_REPORT_SCHEMA,
        status=WP2_DENSE_LAYER_DEVELOPMENT_STATUS,
        assessment_status=WP2_DENSE_LAYER_ASSESSMENT_STATUS,
        resource_comparability=WP2_DENSE_LAYER_RESOURCE_COMPARABILITY,
        resources_matched=False,
        resource_noncomparability_reason=_RESOURCE_NONCOMPARABILITY_REASON,
        development_only=True,
        output_writes_authorized=False,
        tuning_authorized=False,
        winner_selection_authorized=False,
        default_selection_authorized=False,
        efficacy_claim_authorized=False,
        evidence_authorized=False,
        promotion_authorized=False,
        scientific_promotion_allowed=False,
        config=config,
        config_sha256=_config_sha256(config),
        source_manifest=source_manifest,
        source_manifest_sha256=_digest([dataclasses.asdict(item) for item in source_manifest]),
        initialization_key_words_uint32=_key_words(initialization_key),
        data_key_words_uint32=_key_words(jr.key(_DATA_SEED)),
        protocol_sha256=stream_sha256,
        arms=arms,
        resources=resources,
        replay_verified=True,
        reconstruction_mode=("deterministic exact source/config/key-bound in-memory causal replay"),
        limitations=_LIMITATIONS,
        report_sha256="",
    )
    return dataclasses.replace(bare, report_sha256=_report_sha256(bare))


def _validate_structural(report: WP2DenseLayerDevelopmentReport) -> None:
    if type(report) is not WP2DenseLayerDevelopmentReport:
        raise TypeError("report type is invalid")
    canonical_config = WP2DenseLayerDevelopmentConfig()
    fixed = {
        "schema": WP2_DENSE_LAYER_REPORT_SCHEMA,
        "status": WP2_DENSE_LAYER_DEVELOPMENT_STATUS,
        "assessment_status": WP2_DENSE_LAYER_ASSESSMENT_STATUS,
        "resource_comparability": WP2_DENSE_LAYER_RESOURCE_COMPARABILITY,
        "resources_matched": False,
        "resource_noncomparability_reason": _RESOURCE_NONCOMPARABILITY_REASON,
        "development_only": True,
        "output_writes_authorized": False,
        "tuning_authorized": False,
        "winner_selection_authorized": False,
        "default_selection_authorized": False,
        "efficacy_claim_authorized": False,
        "evidence_authorized": False,
        "promotion_authorized": False,
        "scientific_promotion_allowed": False,
        "replay_verified": True,
    }
    for name, expected in fixed.items():
        if getattr(report, name) != expected:
            raise ValueError(f"report {name} differs")
    if report.config != canonical_config or report.config_sha256 != _config_sha256(
        canonical_config
    ):
        raise ValueError("report frozen config differs")
    current_source = _source_manifest()
    if report.source_manifest != current_source or report.source_manifest_sha256 != _digest(
        [dataclasses.asdict(item) for item in current_source]
    ):
        raise ValueError("report source manifest differs")
    if report.initialization_key_words_uint32 != _key_words(
        jr.key(_INITIALIZATION_SEED)
    ) or report.data_key_words_uint32 != _key_words(jr.key(_DATA_SEED)):
        raise ValueError("report frozen key receipt differs")
    _, _, _, stream_digest = _stream()
    if report.protocol_sha256 != stream_digest:
        raise ValueError("report protocol digest differs")
    if tuple(arm.name for arm in report.arms) != WP2_DENSE_LAYER_ARM_ORDER:
        raise ValueError("report arm order differs")
    expected_updates = len(_PHASE_ORDER) * _UPDATES_PER_PHASE
    for arm in report.arms:
        if arm.training_stream_sha256 != report.protocol_sha256:
            raise ValueError("report arm stream differs")
        _validate_checkpoint(arm.initial_checkpoint, canonical_config)
        _validate_checkpoint(arm.final_checkpoint, canonical_config)
        if (
            arm.initial_checkpoint.step != 0
            or arm.final_checkpoint.step != expected_updates
            or len(arm.prequential_trace) != expected_updates
            or len(arm.probe_trace) != expected_updates + 1
            or len(arm.representation_trace) != expected_updates + 1
            or len(arm.phase_metrics) != len(_PHASE_ORDER)
            or len(arm.switch_metrics) != len(_PHASE_ORDER) - 1
        ):
            raise ValueError("report arm trace/checkpoint geometry differs")
        if arm.work.output_write_calls != 0 or arm.work.artifact_bytes_written != 0:
            raise ValueError("report arm contains forbidden writes")
        state = arm.state_bytes
        if state.total_persistent_bytes != (
            state.parameter_bytes
            + state.base_optimizer_state_bytes
            + state.mechanism_state_bytes
            + state.protocol_clock_bytes
        ):
            raise ValueError("report arm state byte arithmetic differs")
    if report.resources != MatrixResourceReceipt(
        arm_count=len(report.arms),
        total_update_opportunities=len(report.arms) * expected_updates,
        total_task_gradient_evaluations=sum(
            arm.work.task_gradient_evaluations for arm in report.arms
        ),
        total_persistent_bytes_across_isolated_arms=sum(
            arm.state_bytes.total_persistent_bytes for arm in report.arms
        ),
        output_write_calls=0,
        artifact_bytes_written=0,
        resources_matched=False,
        comparability_assessment=WP2_DENSE_LAYER_RESOURCE_COMPARABILITY,
    ):
        raise ValueError("report resource receipt differs")
    if report.limitations != _LIMITATIONS or not report.reconstruction_mode.startswith(
        "deterministic exact"
    ):
        raise ValueError("report limitations/reconstruction mode differs")
    _require_digest(report.report_sha256, name="report sha256")
    if report.report_sha256 != _report_sha256(report):
        raise ValueError("report integrity digest differs")


def validate_wp2_dense_layer_development_matrix(
    report: WP2DenseLayerDevelopmentReport,
    *,
    reconstruct: bool = True,
) -> None:
    """Fail closed on authority, checkpoint, trace, resource, source, or replay drift."""

    if type(reconstruct) is not bool:
        raise TypeError("reconstruct must be boolean")
    _validate_structural(report)
    if reconstruct and _execute() != report:
        raise ValueError("deterministic report replay differs")


def reconstruct_wp2_dense_layer_development_matrix(
    report: WP2DenseLayerDevelopmentReport,
) -> WP2DenseLayerDevelopmentReport:
    """Re-execute the frozen matrix and require complete immutable equality."""

    _validate_structural(report)
    replay = _execute()
    if replay != report:
        raise ValueError("deterministic report replay differs")
    return replay


def run_wp2_dense_layer_development_matrix() -> WP2DenseLayerDevelopmentReport:
    """Run and replay the bounded matrix in memory; no artifact is written."""

    report = _execute()
    _validate_structural(report)
    replay = _execute()
    if replay != report:
        raise RuntimeError("source-frozen development matrix did not replay exactly")
    return report


__all__ = [
    "ADAM_ARM",
    "ADAMO_ARM",
    "CPR_ARM",
    "SGD_ARM",
    "SPECTRAL_ARM",
    "WP2_DENSE_LAYER_ARM_ORDER",
    "WP2_DENSE_LAYER_ASSESSMENT_STATUS",
    "WP2_DENSE_LAYER_CHECKPOINT_SCHEMA",
    "WP2_DENSE_LAYER_CONFIG_SCHEMA",
    "WP2_DENSE_LAYER_DEVELOPMENT_STATUS",
    "WP2_DENSE_LAYER_PROTOCOL_ID",
    "WP2_DENSE_LAYER_REPORT_SCHEMA",
    "WP2_DENSE_LAYER_RESOURCE_COMPARABILITY",
    "ArmDevelopmentReport",
    "ArmStateBytes",
    "ArmWorkReceipt",
    "DenseLayerCheckpoint",
    "MatrixResourceReceipt",
    "ParameterChangeReceipt",
    "PhaseMetrics",
    "PrequentialReceipt",
    "ProbeReceipt",
    "RepresentationReceipt",
    "SourceFileIdentity",
    "SwitchMetrics",
    "WP2DenseLayerDevelopmentConfig",
    "WP2DenseLayerDevelopmentReport",
    "reconstruct_wp2_dense_layer_development_matrix",
    "restore_wp2_dense_layer_checkpoint_parameters",
    "run_wp2_dense_layer_development_matrix",
    "validate_wp2_dense_layer_development_matrix",
]
