# mypy: disable-error-code="attr-defined,call-arg"
"""Strict nonpromoting matched-stream world-model retention diagnostics.

This development harness compares three already-defined mechanisms on one
evaluator-owned sequence of raw ``(observation, action, outcome)`` events:

* :class:`ShallowRidgeWorldModel`;
* a plain :class:`WorldModelEnsemble`; and
* :class:`ModelReplayRehearsal` using the same ensemble child configuration.

The stream is an uninterrupted A/B/A recurrence with interleaved stable and
aleatoric noisy-TV transitions — the classic irreducible-noise trap for
prediction-driven learners (Schmidhuber 1991, "Curious model-building control
systems"; Burda et al. 2018, "Large-Scale Study of Curiosity-Driven
Learning").  Phase, regime, and noisy-TV labels exist only in the evaluator
trace.  Learners receive no task identifier or boundary.  A unique provenance
integer lets the evaluator audit replay composition; it is not used by the
model, priority equation, or representation filter.

All errors are computed independently from decoded predictions and common raw
grounded ``[next_observation, reward, continuation]`` targets.  Internal model
losses are not substituted for that common score.  The report is descriptive,
development-only, and not assessed: it has no acceptance threshold, winner,
promotion path, or scientific-evidence claim.

Reports bind the implementation source, each raw stream, each learner trace,
and the complete canonical payload with SHA-256.  Validation reconstructs the
fixed protocol, reruns every learner, and compares every trace and derived
equation.  Atomic save validates before replacing the destination.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.dual_replay import DualReplayConfig, ReplayEntries
from alberta_framework.core.learning_signals import LearningSignalEstimatorConfig
from alberta_framework.core.model_replay_rehearsal import (
    ModelReplayRehearsal,
    ModelReplayRehearsalConfig,
    ModelReplayRehearsalState,
    RealModelReplayEvent,
)
from alberta_framework.core.shallow_ridge_world_model import (
    ShallowRidgeWorldModel,
    ShallowRidgeWorldModelConfig,
)
from alberta_framework.core.world_model import ActionConditionedWorldModelConfig
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleConfig,
    WorldModelEnsembleState,
)

WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA = "alberta.world-model-retention.development.v1"
DEVELOPMENT_ONLY = True
SCIENTIFIC_PROMOTION_ALLOWED = False
ASSESSMENT_STATUS = "not_assessed"

LearnerName = Literal[
    "shallow_ridge_world_model",
    "plain_world_model_ensemble",
    "model_replay_rehearsal",
]

LEARNER_ORDER: tuple[LearnerName, ...] = (
    "shallow_ridge_world_model",
    "plain_world_model_ensemble",
    "model_replay_rehearsal",
)
# The protocol is deliberately tiny (three 6-step phases, two seeds): the
# report is descriptive only, and strict validation reruns every learner on
# every load, so the whole harness must stay cheap enough to replay
# constantly.  The windows below are sized to fit inside one 6-step phase.
REGIME_SCHEDULE = ("A", "B", "A")
DEVELOPMENT_SEEDS = (7, 29)
PHASE_LENGTH = 6
ADAPTATION_WINDOW = 3
FINAL_WINDOW = 3
RECOVERY_WINDOW = 2
CALIBRATION_BIN_LOWER_EDGES = (0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)

_CLAIM_SCOPE = "development-only matched-stream world-model retention diagnostics"
_INTERPRETATION = (
    "descriptive mechanism diagnostics only; no threshold, ordering, or result can promote "
    "a scientific claim"
)
_UNAVAILABLE_SINGLE_MODEL = "single shallow model has no ensemble disagreement"
_UNAVAILABLE_NO_REPLAY = "learner owns no replay memory or replay-priority mechanism"
_UINT32_MAX = 2**32 - 1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict_json_equal(actual: object, expected: object) -> bool:
    """Compare JSON trees without Python's bool/int or tuple/list coercions."""
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return type(actual) is float and math.isfinite(actual) and actual == expected
    if isinstance(expected, str):
        return type(actual) is str and actual == expected
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(actual, expected, strict=True)
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[key], expected[key]) for key in expected)
        )
    return False


def _json_tree_is_canonical(value: object) -> bool:
    if value is None or isinstance(value, bool | str):
        return True
    if type(value) is int:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_tree_is_canonical(item) for item in cast(list[object], value))
    if isinstance(value, Mapping):
        return all(
            type(key) is str and _json_tree_is_canonical(item)
            for key, item in value.items()
        )
    return False


def _ensemble_config() -> WorldModelEnsembleConfig:
    model = ActionConditionedWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        gamma=1.0,
        observation_scale=(1.0, 1.0),
        reward_scale=1.0,
        predict_delta=False,
        hidden_sizes=(),
        step_size=0.025,
        sparsity=0.0,
        use_layer_norm=False,
        utility_decay=0.9,
        error_decay=0.8,
        observation_clip_margin=0.5,
        max_delta_scale=10.0,
        include_action_interactions=True,
    )
    signals = LearningSignalEstimatorConfig(
        ensemble_size=2,
        target_dim=4,
        variance_floor=1.0e-6,
        fast_loss_decay=0.5,
        slow_loss_decay=0.9,
        progress_warmup_steps=2,
        change_calibration_steps=4,
        change_z_threshold=3.0,
        change_temperature=0.5,
        change_decay=0.8,
        calibration_scale_floor=0.25,
        max_normalized_residual=1.0e6,
        max_input_magnitude=100.0,
        max_predicted_variance=10_000.0,
        max_observed_loss=10_000.0,
    )
    return WorldModelEnsembleConfig(
        model=model,
        signal_estimator=signals,
        ensemble_size=2,
        bootstrap_probability=0.75,
        residual_variance_decay=0.8,
        residual_variance_warmup_steps=2,
        residual_variance_floor=1.0e-6,
    )


def _rehearsal_config() -> ModelReplayRehearsalConfig:
    return ModelReplayRehearsalConfig(
        ensemble=_ensemble_config(),
        replay=DualReplayConfig(
            total_capacity=8,
            short_term_capacity=4,
            observation_dim=2,
            action_dim=2,
            short_term_sample_size=1,
            long_term_sample_size=1,
            long_term_policy="calibrated",
            max_representation_lag=0,
            surprise_scale=1.0,
            coverage_scale=1.0,
            progress_scale=1.0,
            surprise_weight=1.0,
            coverage_weight=1.0,
            progress_weight=1.0,
            calibrated_priority_threshold=0.05,
            calibrated_replacement_margin=0.0,
            aleatoric_control="downweight",
            max_aleatoric_uncertainty=1.0,
            aleatoric_downweight_scale=0.25,
        ),
        action_encoding="one_hot",
    )


def _shallow_config() -> ShallowRidgeWorldModelConfig:
    return ShallowRidgeWorldModelConfig(
        observation_dim=2,
        n_actions=2,
        ridge=1.0,
        max_updates=len(REGIME_SCHEDULE) * PHASE_LENGTH,
        max_input_magnitude=100.0,
        max_statistic_magnitude=1_000_000.0,
        max_parameter_magnitude=100_000.0,
        max_prediction_magnitude=100_000.0,
    )


@dataclasses.dataclass(frozen=True)
class WorldModelRetentionDevelopmentConfig:
    """One immutable development protocol; there are no tunable report gates."""

    @property
    def seeds(self) -> tuple[int, ...]:
        return DEVELOPMENT_SEEDS

    @property
    def phase_length(self) -> int:
        return PHASE_LENGTH

    @property
    def total_steps(self) -> int:
        return len(REGIME_SCHEDULE) * PHASE_LENGTH

    def to_config(self) -> dict[str, object]:
        """Return the exact versioned development protocol."""
        return {
            "schema_version": WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA,
            "development_only": DEVELOPMENT_ONLY,
            "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
            "assessment_status": ASSESSMENT_STATUS,
            "claim_scope": _CLAIM_SCOPE,
            "interpretation": _INTERPRETATION,
            "seeds": list(DEVELOPMENT_SEEDS),
            "regime_schedule": list(REGIME_SCHEDULE),
            "phase_length": PHASE_LENGTH,
            "total_steps": self.total_steps,
            "adaptation_window": ADAPTATION_WINDOW,
            "final_window": FINAL_WINDOW,
            "recovery_window": RECOVERY_WINDOW,
            "calibration_bin_lower_edges": list(CALIBRATION_BIN_LOWER_EDGES),
            "noisy_tv_schedule": "((step + seed) mod 3) == 1",
            "action_schedule": "((5 * step + seed) mod 7) >= 3",
            "prediction_order": "predict_before_update",
            "common_grounded_target_order": [
                "next_observation[0]",
                "next_observation[1]",
                "reward",
                "continuation",
            ],
            "calibration_disagreement_surface": "decoded_grounded_member_outputs",
            "calibration_squared_error_surface": "decoded_grounded_mean_prediction",
            "learner_input_fields": [
                "observation",
                "action",
                "next_observation",
                "reward",
                "continuation",
            ],
            "evaluator_only_fields": [
                "step",
                "phase_index",
                "regime_id",
                "noisy_tv",
                "environment_rng_words",
            ],
            "composer_bookkeeping_fields": {
                "representation_version": "constant_zero",
                "provenance_id": "unique_event_index_for_audit_only",
                "source_id": "constant_zero",
                "safety_cost": "unavailable",
            },
            "learners": list(LEARNER_ORDER),
            "shallow_ridge_config": _shallow_config().to_config(),
            "ensemble_config": _ensemble_config().to_config(),
            "rehearsal_config": _rehearsal_config().to_config(),
            "thresholds": None,
        }

    @classmethod
    def from_config(
        cls,
        payload: Mapping[str, object],
    ) -> WorldModelRetentionDevelopmentConfig:
        """Accept only the exact immutable development configuration."""
        expected = cls().to_config()
        if not _strict_json_equal(payload, expected):
            raise ValueError("world-model retention protocol is noncanonical or was changed")
        return cls()


def _float32(value: object) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float32))


def _int(value: object) -> int:
    return int(np.asarray(jax.device_get(value), dtype=np.int64))


def _bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _float32_list(value: object) -> list[float]:
    array = np.asarray(jax.device_get(value), dtype=np.float32)
    return cast(list[float], array.astype(float).tolist())


def _bool_list(value: object) -> list[bool]:
    return cast(list[bool], np.asarray(jax.device_get(value), dtype=np.bool_).tolist())


def _int_list(value: object) -> list[int]:
    return cast(list[int], np.asarray(jax.device_get(value), dtype=np.int32).tolist())


def _uint32_list(value: object) -> list[int]:
    array = np.asarray(jax.device_get(value), dtype=np.uint32)
    return [int(item) for item in array]


def _predictive_state_sha256(state: object) -> str:
    """Hash deterministic predictive state while excluding runtime-only clocks."""
    digest = hashlib.sha256()
    paths_and_leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    for path, leaf in paths_and_leaves:
        attribute_names = {
            getattr(component, "name", None)
            for component in path
            if getattr(component, "name", None) is not None
        }
        if attribute_names & {"birth_timestamp", "uptime_s"}:
            continue
        digest.update(str(path).encode("utf-8"))
        dtype = getattr(leaf, "dtype", None)
        materialized = (
            jr.key_data(leaf)
            if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
            else leaf
        )
        array = np.asarray(jax.device_get(materialized))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _generate_stream(seed: int) -> dict[str, object]:
    total = len(REGIME_SCHEDULE) * PHASE_LENGTH
    observation = np.asarray(
        [np.float32(0.15 + 0.01 * (seed % 5)), np.float32(-0.2)],
        dtype=np.float32,
    )
    steps: list[int] = []
    phases: list[int] = []
    regimes: list[str] = []
    noisy_flags: list[bool] = []
    observations: list[list[float]] = []
    actions: list[int] = []
    next_observations: list[list[float]] = []
    rewards: list[float] = []
    continuations: list[float] = []
    rng_words: list[list[int]] = []
    environment_root = jr.fold_in(jr.key(seed), 0x57524C44)

    for step in range(total):
        phase = step // PHASE_LENGTH
        regime = REGIME_SCHEDULE[phase]
        action = int(((5 * step + seed) % 7) >= 3)
        noisy_tv = (step + seed) % 3 == 1
        outcome_key = jr.fold_in(environment_root, step)
        draw = np.float32(jax.device_get(jr.normal(outcome_key, (), dtype=jnp.float32)))

        # Dynamics design: both regimes are contracting affine maps on the
        # signal channel (|slope| < 1 keeps the stream bounded), and regime B
        # flips the slope sign so its dynamics conflict with regime A rather
        # than extend them.  On noisy-TV steps the distractor channel is
        # i.i.d. N(0, 1.35^2) — no state or action dependence, hence
        # irreducibly unpredictable, with a standard deviation exceeding the
        # signal's magnitude; on other steps it is a predictable contracting
        # map coupled to the signal.
        signal = observation[0]
        distractor = observation[1]
        if regime == "A":
            next_signal = np.float32(0.55 * signal + (0.30 if action else -0.16))
            reward = np.float32(0.75 * signal + (0.20 if action else -0.08))
            continuation = np.float32(0.98)
        else:
            next_signal = np.float32(-0.45 * signal + (-0.26 if action else 0.24))
            reward = np.float32(-0.65 * signal + (-0.18 if action else 0.22))
            continuation = np.float32(0.86)
        if noisy_tv:
            next_distractor = np.float32(1.35 * draw)
        else:
            next_distractor = np.float32(
                0.35 * distractor + 0.20 * signal + (0.08 if action else -0.05)
            )
        next_observation = np.asarray([next_signal, next_distractor], dtype=np.float32)

        steps.append(step)
        phases.append(phase)
        regimes.append(regime)
        noisy_flags.append(noisy_tv)
        observations.append(observation.astype(float).tolist())
        actions.append(action)
        next_observations.append(next_observation.astype(float).tolist())
        rewards.append(float(reward))
        continuations.append(float(continuation))
        rng_words.append(_uint32_list(jr.key_data(outcome_key)))
        observation = next_observation

    payload: dict[str, object] = {
        "seed": seed,
        "step": steps,
        "phase_index": phases,
        "regime_id": regimes,
        "noisy_tv": noisy_flags,
        "observation": observations,
        "action": actions,
        "next_observation": next_observations,
        "reward": rewards,
        "continuation": continuations,
        "environment_rng_words": rng_words,
        "labels_visible_to_learners": False,
    }
    return payload


def _stream_arrays(stream: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {
        "observation": np.asarray(stream["observation"], dtype=np.float32),
        "action": np.asarray(stream["action"], dtype=np.int32),
        "next_observation": np.asarray(stream["next_observation"], dtype=np.float32),
        "reward": np.asarray(stream["reward"], dtype=np.float32),
        "continuation": np.asarray(stream["continuation"], dtype=np.float32),
        "phase_index": np.asarray(stream["phase_index"], dtype=np.int32),
        "noisy_tv": np.asarray(stream["noisy_tv"], dtype=np.bool_),
    }


def _empty_trace() -> dict[str, object]:
    return {
        "predicted_next_observation": [],
        "predicted_reward": [],
        "predicted_continuation": [],
        "next_observation_error": [],
        "next_observation_squared_error": [],
        "next_observation_mse": [],
        "reward_error": [],
        "reward_squared_error": [],
        "continuation_error": [],
        "continuation_squared_error": [],
        "aggregate_prequential_loss": [],
        "update_applied": [],
        "ensemble_disagreement": [],
        "per_head_epistemic_variance": [],
        "signal_values": [],
        "signal_availability": [],
        "real_member_updates_applied": [],
        "replay_sample_valid": [],
        "replay_sample_provenance_ids": [],
        "replay_member_updates_applied": [],
    }


def _append_prediction(
    trace: dict[str, object],
    *,
    next_prediction: object,
    reward_prediction: object,
    continuation_prediction: object,
    next_target: np.ndarray,
    reward_target: np.float32,
    continuation_target: np.float32,
) -> None:
    predicted_next = np.asarray(jax.device_get(next_prediction), dtype=np.float32)
    predicted_reward = np.float32(jax.device_get(reward_prediction))
    predicted_continuation = np.float32(jax.device_get(continuation_prediction))
    next_error = predicted_next - next_target
    reward_error = np.float32(predicted_reward - reward_target)
    continuation_error = np.float32(predicted_continuation - continuation_target)
    next_squared = np.square(next_error, dtype=np.float32)
    reward_squared = np.float32(reward_error * reward_error)
    continuation_squared = np.float32(continuation_error * continuation_error)
    grounded_squared = np.concatenate(
        (
            next_squared,
            np.asarray([reward_squared, continuation_squared], dtype=np.float32),
        )
    )
    cast(list[object], trace["predicted_next_observation"]).append(
        predicted_next.astype(float).tolist()
    )
    cast(list[object], trace["predicted_reward"]).append(float(predicted_reward))
    cast(list[object], trace["predicted_continuation"]).append(float(predicted_continuation))
    cast(list[object], trace["next_observation_error"]).append(next_error.astype(float).tolist())
    cast(list[object], trace["next_observation_squared_error"]).append(
        next_squared.astype(float).tolist()
    )
    cast(list[object], trace["next_observation_mse"]).append(
        float(np.mean(next_squared, dtype=np.float32))
    )
    cast(list[object], trace["reward_error"]).append(float(reward_error))
    cast(list[object], trace["reward_squared_error"]).append(float(reward_squared))
    cast(list[object], trace["continuation_error"]).append(float(continuation_error))
    cast(list[object], trace["continuation_squared_error"]).append(
        float(continuation_squared)
    )
    cast(list[object], trace["aggregate_prequential_loss"]).append(
        float(np.mean(grounded_squared, dtype=np.float32))
    )


def _signal_payload(signals: object) -> tuple[dict[str, float], dict[str, bool]]:
    values = {
        "epistemic_disagreement": _float32(getattr(signals, "epistemic_disagreement")),
        "epistemic_surprise": _float32(getattr(signals, "epistemic_surprise")),
        "aleatoric_uncertainty": _float32(getattr(signals, "aleatoric_uncertainty")),
        "learning_progress": _float32(getattr(signals, "learning_progress")),
    }
    source = getattr(signals, "availability")
    availability = {
        "epistemic": _bool(source.epistemic),
        "aleatoric": _bool(source.aleatoric),
        "learning_progress": _bool(source.learning_progress),
    }
    return values, availability


def _grounded_disagreement(prediction: object) -> tuple[float, list[float]]:
    """Population variance across decoded member outputs in common target units."""
    grounded_members = jnp.concatenate(
        (
            getattr(prediction, "member_next_observations"),
            jnp.reshape(getattr(prediction, "member_rewards"), (-1, 1)),
            jnp.reshape(getattr(prediction, "member_discounts"), (-1, 1)),
        ),
        axis=1,
    )
    per_head = jnp.var(grounded_members, axis=0)
    return _float32(jnp.mean(per_head)), _float32_list(per_head)


def _finalize_unavailable_fields(trace: dict[str, object], learner: LearnerName) -> None:
    if learner == "shallow_ridge_world_model":
        trace["ensemble_disagreement"] = None
        trace["per_head_epistemic_variance"] = None
        trace["signal_values"] = None
        trace["signal_availability"] = None
        trace["real_member_updates_applied"] = None
    if learner != "model_replay_rehearsal":
        trace["replay_sample_valid"] = None
        trace["replay_sample_provenance_ids"] = None
        trace["replay_member_updates_applied"] = None


def _run_shallow(
    model: ShallowRidgeWorldModel,
    stream: Mapping[str, object],
) -> tuple[dict[str, object], object, str]:
    arrays = _stream_arrays(stream)
    state = model.init()
    initial_sha = _predictive_state_sha256(state)
    trace = _empty_trace()
    for step in range(arrays["action"].size):
        result = model.update(
            state,
            jnp.asarray(arrays["observation"][step], dtype=jnp.float32),
            jnp.asarray(arrays["action"][step], dtype=jnp.int32),
            jnp.asarray(arrays["next_observation"][step], dtype=jnp.float32),
            jnp.asarray(arrays["reward"][step], dtype=jnp.float32),
            jnp.asarray(arrays["continuation"][step], dtype=jnp.float32),
        )
        _append_prediction(
            trace,
            next_prediction=result.prediction.next_observation,
            reward_prediction=result.prediction.reward,
            continuation_prediction=result.prediction.continuation,
            next_target=arrays["next_observation"][step],
            reward_target=arrays["reward"][step],
            continuation_target=arrays["continuation"][step],
        )
        cast(list[object], trace["update_applied"]).append(_bool(result.diagnostics.applied))
        state = result.state
    _finalize_unavailable_fields(trace, "shallow_ridge_world_model")
    return trace, state, initial_sha


def _run_plain_ensemble(
    ensemble: WorldModelEnsemble,
    stream: Mapping[str, object],
    root_key: Array,
) -> tuple[dict[str, object], object, str]:
    arrays = _stream_arrays(stream)
    ensemble_key, _ = jr.split(root_key)
    state = ensemble.init(ensemble_key)
    initial_sha = _predictive_state_sha256(state)
    trace = _empty_trace()
    for step in range(arrays["action"].size):
        result = ensemble.update(
            state,
            jnp.asarray(arrays["observation"][step], dtype=jnp.float32),
            jnp.asarray(arrays["action"][step], dtype=jnp.int32),
            jnp.asarray(arrays["reward"][step], dtype=jnp.float32),
            jnp.asarray(arrays["continuation"][step], dtype=jnp.float32),
            jnp.asarray(arrays["next_observation"][step], dtype=jnp.float32),
        )
        prediction = result.prediction
        _append_prediction(
            trace,
            next_prediction=prediction.mean_next_observation,
            reward_prediction=prediction.mean_reward,
            continuation_prediction=prediction.mean_discount,
            next_target=arrays["next_observation"][step],
            reward_target=arrays["reward"][step],
            continuation_target=arrays["continuation"][step],
        )
        cast(list[object], trace["update_applied"]).append(_bool(result.diagnostics.applied))
        disagreement, per_head = _grounded_disagreement(prediction)
        cast(list[object], trace["ensemble_disagreement"]).append(
            disagreement
        )
        cast(list[object], trace["per_head_epistemic_variance"]).append(
            per_head
        )
        values, availability = _signal_payload(result.signals)
        cast(list[object], trace["signal_values"]).append(values)
        cast(list[object], trace["signal_availability"]).append(availability)
        cast(list[object], trace["real_member_updates_applied"]).append(
            _bool_list(result.member_updates_applied)
        )
        state = result.state
    _finalize_unavailable_fields(trace, "plain_world_model_ensemble")
    return trace, state, initial_sha


def _run_rehearsal(
    composer: ModelReplayRehearsal,
    stream: Mapping[str, object],
    root_key: Array,
) -> tuple[dict[str, object], object, str]:
    arrays = _stream_arrays(stream)
    state = composer.init(root_key)
    initial_sha = _predictive_state_sha256(state.ensemble_state)
    trace = _empty_trace()
    for step in range(arrays["action"].size):
        event = RealModelReplayEvent(
            observation=jnp.asarray(arrays["observation"][step], dtype=jnp.float32),
            action=jnp.asarray(arrays["action"][step], dtype=jnp.int32),
            reward=jnp.asarray(arrays["reward"][step], dtype=jnp.float32),
            discount=jnp.asarray(arrays["continuation"][step], dtype=jnp.float32),
            terminated=jnp.asarray(False, dtype=jnp.bool_),
            truncated=jnp.asarray(False, dtype=jnp.bool_),
            next_observation=jnp.asarray(arrays["next_observation"][step], dtype=jnp.float32),
            representation_version=jnp.asarray(0, dtype=jnp.int32),
            provenance_id=jnp.asarray(step, dtype=jnp.int32),
            source_id=jnp.asarray(0, dtype=jnp.int32),
            safety_cost=jnp.asarray(0.0, dtype=jnp.float32),
            safety_cost_available=jnp.asarray(False, dtype=jnp.bool_),
            valid=jnp.asarray(True, dtype=jnp.bool_),
        )
        result = composer.step(state, event)
        prediction = result.real_prediction
        _append_prediction(
            trace,
            next_prediction=prediction.mean_next_observation,
            reward_prediction=prediction.mean_reward,
            continuation_prediction=prediction.mean_discount,
            next_target=arrays["next_observation"][step],
            reward_target=arrays["reward"][step],
            continuation_target=arrays["continuation"][step],
        )
        cast(list[object], trace["update_applied"]).append(
            _bool(result.diagnostics.transaction_applied)
        )
        disagreement, per_head = _grounded_disagreement(prediction)
        cast(list[object], trace["ensemble_disagreement"]).append(
            disagreement
        )
        cast(list[object], trace["per_head_epistemic_variance"]).append(
            per_head
        )
        values, availability = _signal_payload(result.real_signals)
        cast(list[object], trace["signal_values"]).append(values)
        cast(list[object], trace["signal_availability"]).append(availability)
        cast(list[object], trace["real_member_updates_applied"]).append(
            _bool_list(result.state.ensemble_state.last_bootstrap_mask)
        )
        cast(list[object], trace["replay_sample_valid"]).append(
            _bool_list(result.trace.sample_valid)
        )
        cast(list[object], trace["replay_sample_provenance_ids"]).append(
            _int_list(result.trace.provenance_ids)
        )
        cast(list[object], trace["replay_member_updates_applied"]).append(
            [
                _bool_list(row)
                for row in np.asarray(
                    jax.device_get(result.trace.member_updates_applied), dtype=np.bool_
                )
            ]
        )
        state = result.state
    return trace, state, initial_sha


def _mean_or_none(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values, dtype=np.float64))


def _rolling_means(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(
        [
            np.mean(values[start : start + window], dtype=np.float64)
            for start in range(0, values.size - window + 1)
        ],
        dtype=np.float64,
    )


def _metrics(trace: Mapping[str, object]) -> dict[str, object]:
    aggregate = np.asarray(trace["aggregate_prequential_loss"], dtype=np.float64)
    next_mse = np.asarray(trace["next_observation_mse"], dtype=np.float64)
    reward_mse = np.asarray(trace["reward_squared_error"], dtype=np.float64)
    continuation_mse = np.asarray(trace["continuation_squared_error"], dtype=np.float64)
    phase_slices = [
        slice(index * PHASE_LENGTH, (index + 1) * PHASE_LENGTH)
        for index in range(len(REGIME_SCHEDULE))
    ]
    phase_metrics = []
    for index, phase_slice in enumerate(phase_slices):
        phase_metrics.append(
            {
                "phase_index": index,
                "regime_id": REGIME_SCHEDULE[index],
                "mean_aggregate_prequential_loss": float(np.mean(aggregate[phase_slice])),
                "mean_next_observation_mse": float(np.mean(next_mse[phase_slice])),
                "mean_reward_squared_error": float(np.mean(reward_mse[phase_slice])),
                "mean_continuation_squared_error": float(
                    np.mean(continuation_mse[phase_slice])
                ),
            }
        )
    adaptation = []
    for phase_index in (1, 2):
        start = phase_index * PHASE_LENGTH
        window = slice(start, start + ADAPTATION_WINDOW)
        adaptation.append(
            {
                "phase_index": phase_index,
                "regime_id": REGIME_SCHEDULE[phase_index],
                "window_steps": ADAPTATION_WINDOW,
                "aggregate_loss_auc": float(np.sum(aggregate[window], dtype=np.float64)),
                "next_observation_mse_auc": float(np.sum(next_mse[window], dtype=np.float64)),
                "reward_squared_error_auc": float(np.sum(reward_mse[window], dtype=np.float64)),
                "continuation_squared_error_auc": float(
                    np.sum(continuation_mse[window], dtype=np.float64)
                ),
            }
        )

    initial_a = aggregate[phase_slices[0]]
    recurrent_a = aggregate[phase_slices[2]]
    initial_final = float(np.mean(initial_a[-FINAL_WINDOW:], dtype=np.float64))
    recurrent_initial = float(np.mean(recurrent_a[:RECOVERY_WINDOW], dtype=np.float64))
    recurrent_final = float(np.mean(recurrent_a[-FINAL_WINDOW:], dtype=np.float64))
    recurrent_rolling = _rolling_means(recurrent_a, RECOVERY_WINDOW)
    recovery_candidates = np.flatnonzero(recurrent_rolling <= initial_final)
    recovery_step = None if recovery_candidates.size == 0 else int(recovery_candidates[0])
    initial_windows = _rolling_means(initial_a, FINAL_WINDOW)
    best_initial = float(np.min(initial_windows))
    return {
        "raw_grounded_error_summary": {
            "mean_next_observation_mse": float(np.mean(next_mse)),
            "mean_reward_squared_error": float(np.mean(reward_mse)),
            "mean_continuation_squared_error": float(np.mean(continuation_mse)),
        },
        "aggregate_prequential_loss": {
            "sum": float(np.sum(aggregate, dtype=np.float64)),
            "mean": float(np.mean(aggregate, dtype=np.float64)),
        },
        "phase_metrics": phase_metrics,
        "post_change_adaptation_auc": adaptation,
        "recurrence_and_recovery": {
            "initial_a_final_window_loss": initial_final,
            "recurrent_a_initial_window_loss": recurrent_initial,
            "recurrent_a_final_window_loss": recurrent_final,
            "recurrence_initial_cost_delta": recurrent_initial - initial_final,
            "recurrence_final_retention_delta": recurrent_final - initial_final,
            "recovery_window": RECOVERY_WINDOW,
            "recovery_condition": (
                "first recurrent-A rolling mean no greater than initial-A final-window mean"
            ),
            "recovery_step": recovery_step,
        },
        "best_to_final_forgetting": {
            "window": FINAL_WINDOW,
            "best_initial_a_window_loss": best_initial,
            "recurrent_a_final_window_loss": recurrent_final,
            "signed_final_minus_best_loss": recurrent_final - best_initial,
        },
    }


def _calibration(trace: Mapping[str, object], learner: LearnerName) -> dict[str, object]:
    if learner == "shallow_ridge_world_model":
        return {
            "available": False,
            "reason": _UNAVAILABLE_SINGLE_MODEL,
            "bin_lower_edges": None,
            "bins": None,
        }
    disagreement = np.asarray(trace["ensemble_disagreement"], dtype=np.float64)
    squared_error = np.asarray(trace["aggregate_prequential_loss"], dtype=np.float64)
    bins: list[dict[str, object]] = []
    edges = CALIBRATION_BIN_LOWER_EDGES
    for index, lower in enumerate(edges):
        upper = edges[index + 1] if index + 1 < len(edges) else None
        mask = disagreement >= lower
        if upper is not None:
            mask &= disagreement < upper
        selected_disagreement = disagreement[mask]
        selected_error = squared_error[mask]
        bins.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "count": int(np.sum(mask)),
                "mean_disagreement": _mean_or_none(selected_disagreement),
                "mean_squared_error": _mean_or_none(selected_error),
                "signed_error_minus_disagreement": (
                    None
                    if selected_error.size == 0
                    else float(np.mean(selected_error) - np.mean(selected_disagreement))
                ),
            }
        )
    return {
        "available": True,
        "reason": None,
        "bin_lower_edges": list(edges),
        "bins": bins,
    }


def _signal_lane_summary(
    trace: Mapping[str, object],
    noisy: np.ndarray,
) -> dict[str, object]:
    availability = cast(Sequence[Mapping[str, bool]], trace["signal_availability"])
    values = cast(Sequence[Mapping[str, float]], trace["signal_values"])

    def summarize(field: str, availability_field: str, mask: np.ndarray) -> dict[str, object]:
        selected = [
            values[index][field]
            for index in range(len(values))
            if bool(mask[index]) and availability[index][availability_field]
        ]
        return {
            "available_count": len(selected),
            "mean_when_available": (
                None if not selected else float(np.mean(np.asarray(selected, dtype=np.float64)))
            ),
        }

    return {
        "aleatoric_uncertainty": {
            "stable": summarize("aleatoric_uncertainty", "aleatoric", ~noisy),
            "noisy_tv": summarize("aleatoric_uncertainty", "aleatoric", noisy),
        },
        "epistemic_surprise": {
            "stable": summarize("epistemic_surprise", "epistemic", ~noisy),
            "noisy_tv": summarize("epistemic_surprise", "epistemic", noisy),
        },
    }


def _entry_payload(
    entries: ReplayEntries,
    noisy: np.ndarray,
) -> list[dict[str, object]]:
    valid = np.asarray(jax.device_get(entries.valid), dtype=np.bool_)
    provenance = np.asarray(jax.device_get(entries.provenance_ids), dtype=np.int32)
    priorities = np.asarray(jax.device_get(entries.insertion_priorities), dtype=np.float32)
    priority_available = np.asarray(
        jax.device_get(entries.insertion_priority_available), dtype=np.bool_
    )
    result: list[dict[str, object]] = []
    for slot in np.flatnonzero(valid):
        event_id = int(provenance[slot])
        result.append(
            {
                "slot": int(slot),
                "provenance_id": event_id,
                "noisy_tv": bool(noisy[event_id]),
                "insertion_priority": (
                    float(priorities[slot]) if bool(priority_available[slot]) else None
                ),
                "insertion_priority_available": bool(priority_available[slot]),
            }
        )
    return result


def _composition_counts(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    noisy_count = sum(item["noisy_tv"] is True for item in entries)
    stable_count = len(entries) - noisy_count
    priorities_by_lane: dict[str, list[float]] = {"stable": [], "noisy_tv": []}
    for item in entries:
        priority = item["insertion_priority"]
        if priority is not None:
            lane = "noisy_tv" if item["noisy_tv"] is True else "stable"
            priorities_by_lane[lane].append(cast(float, priority))
    return {
        "active_count": len(entries),
        "stable_count": stable_count,
        "noisy_tv_count": noisy_count,
        "noisy_tv_fraction": None if not entries else noisy_count / len(entries),
        "mean_available_insertion_priority": {
            lane: (
                None
                if not lane_values
                else float(np.mean(np.asarray(lane_values, dtype=np.float64)))
            )
            for lane, lane_values in priorities_by_lane.items()
        },
    }


def _replay_diagnostic(
    trace: Mapping[str, object],
    state: object,
    noisy: np.ndarray,
    learner: LearnerName,
) -> dict[str, object]:
    if learner != "model_replay_rehearsal":
        return {
            "available": False,
            "reason": _UNAVAILABLE_NO_REPLAY,
            "priority_policy": None,
            "sampled_composition": None,
            "final_memory_composition": None,
        }
    composer_state = state
    replay_state = getattr(composer_state, "replay_state")
    short_entries = _entry_payload(replay_state.short_term, noisy)
    long_entries = _entry_payload(replay_state.long_term, noisy)
    valid = np.asarray(trace["replay_sample_valid"], dtype=np.bool_)
    provenance = np.asarray(trace["replay_sample_provenance_ids"], dtype=np.int32)
    short_quota = _rehearsal_config().replay.short_term_sample_size
    sampled: dict[str, dict[str, int]] = {
        "short_term": {"valid": 0, "stable": 0, "noisy_tv": 0, "padding": 0},
        "long_term": {"valid": 0, "stable": 0, "noisy_tv": 0, "padding": 0},
    }
    for step in range(valid.shape[0]):
        for position in range(valid.shape[1]):
            stratum = "short_term" if position < short_quota else "long_term"
            if not valid[step, position]:
                sampled[stratum]["padding"] += 1
                continue
            sampled[stratum]["valid"] += 1
            event_id = int(provenance[step, position])
            lane = "noisy_tv" if bool(noisy[event_id]) else "stable"
            sampled[stratum][lane] += 1
    return {
        "available": True,
        "reason": None,
        "priority_policy": {
            "policy": "calibrated",
            "aleatoric_control": "downweight",
            "raw_prediction_error_used": False,
            "candidate_labels_used": False,
        },
        "sampled_composition": sampled,
        "final_memory_composition": {
            "short_term": {
                "entries": short_entries,
                "counts": _composition_counts(short_entries),
            },
            "long_term": {
                "entries": long_entries,
                "counts": _composition_counts(long_entries),
            },
        },
    }


def _noisy_tv_diagnostic(
    trace: Mapping[str, object],
    state: object,
    stream: Mapping[str, object],
    learner: LearnerName,
) -> dict[str, object]:
    noisy = np.asarray(stream["noisy_tv"], dtype=np.bool_)
    losses = np.asarray(trace["aggregate_prequential_loss"], dtype=np.float64)
    error_summary = {
        "stable_count": int(np.sum(~noisy)),
        "noisy_tv_count": int(np.sum(noisy)),
        "stable_mean_aggregate_loss": float(np.mean(losses[~noisy])),
        "noisy_tv_mean_aggregate_loss": float(np.mean(losses[noisy])),
    }
    signal_summary: dict[str, object]
    if learner == "shallow_ridge_world_model":
        signal_summary = {
            "available": False,
            "reason": "shallow model exposes no typed uncertainty signals",
            "by_lane": None,
        }
    else:
        signal_summary = {
            "available": True,
            "reason": None,
            "by_lane": _signal_lane_summary(trace, noisy),
        }
    return {
        "evaluator_labels_visible_to_learner": False,
        "raw_error_by_lane": error_summary,
        "typed_signal_by_lane": signal_summary,
        "replay_prioritization_and_composition": _replay_diagnostic(
            trace, state, noisy, learner
        ),
    }


def _operation_and_resource_accounting(
    learner: LearnerName,
    model: object,
    state: object,
    trace: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    steps = len(cast(Sequence[object], trace["update_applied"]))
    commits = sum(cast(Sequence[bool], trace["update_applied"]))
    if learner == "shallow_ridge_world_model":
        shallow = cast(ShallowRidgeWorldModel, model)
        shallow_state = state
        operations = {
            "logical_scope": "algorithm events and committed updates; excludes JAX compiler work",
            "real_prediction_events": steps,
            "real_update_attempts": steps,
            "real_update_commits": commits,
            "real_member_update_candidates": steps,
            "real_member_update_commits": _int(getattr(shallow_state, "update_count")),
            "replay_quota_positions": None,
            "replay_available_positions": None,
            "replay_padding_positions": None,
            "replay_member_update_candidates": None,
            "replay_member_update_commits": None,
        }
        resources = {
            "exact_budget": shallow.resource_budget.to_dict(),
            "final_counters": {
                "update_count": _int(getattr(shallow_state, "update_count")),
                "action_counts": _int_list(getattr(shallow_state, "action_counts")),
            },
        }
        return operations, resources
    if learner == "plain_world_model_ensemble":
        ensemble = cast(WorldModelEnsemble, model)
        ensemble_state = cast(WorldModelEnsembleState, state)
        member_counts = _int_list(getattr(ensemble_state, "member_update_counts"))
        replay_counts = _int_list(getattr(ensemble_state, "replay_member_update_counts"))
        operations = {
            "logical_scope": "algorithm events and committed updates; excludes JAX compiler work",
            "real_prediction_events": steps,
            "real_update_attempts": steps,
            "real_update_commits": commits,
            "real_member_update_candidates": steps * ensemble.config.ensemble_size,
            "real_member_update_commits": sum(member_counts),
            "replay_quota_positions": None,
            "replay_available_positions": None,
            "replay_padding_positions": None,
            "replay_member_update_candidates": None,
            "replay_member_update_commits": None,
        }
        resources = {
            "exact_budget": ensemble.resource_budget(ensemble_state).to_config(),
            "final_counters": {
                "event_count": _int(getattr(ensemble_state, "event_count")),
                "member_update_counts": member_counts,
                "replay_event_count": _int(getattr(ensemble_state, "replay_event_count")),
                "replay_member_update_counts": replay_counts,
            },
        }
        return operations, resources
    composer = cast(ModelReplayRehearsal, model)
    composer_state = cast(ModelReplayRehearsalState, state)
    ensemble_state = getattr(composer_state, "ensemble_state")
    member_counts = _int_list(ensemble_state.member_update_counts)
    replay_member_counts = _int_list(ensemble_state.replay_member_update_counts)
    available = _int(getattr(composer_state, "rehearsal_applied_count"))
    padding = _int(getattr(composer_state, "rehearsal_padding_count"))
    quota_positions = _int(getattr(composer_state, "rehearsal_attempt_count"))
    operations = {
        "logical_scope": "algorithm events and committed updates; excludes JAX compiler work",
        "real_prediction_events": steps,
        "real_update_attempts": _int(getattr(composer_state, "real_attempt_count")),
        "real_update_commits": _int(getattr(composer_state, "accepted_real_event_count")),
        "real_member_update_candidates": steps * composer.config.ensemble.ensemble_size,
        "real_member_update_commits": sum(member_counts),
        "replay_quota_positions": quota_positions,
        "replay_available_positions": available,
        "replay_padding_positions": padding,
        "replay_member_update_candidates": (
            available * composer.config.ensemble.ensemble_size
        ),
        "replay_member_update_commits": sum(replay_member_counts),
    }
    replay_state = getattr(composer_state, "replay_state")
    replay_accounting = composer.replay.accounting(replay_state)
    replay_accounting_fields = (
        "total_capacity",
        "short_term_capacity",
        "long_term_capacity",
        "active_entries",
        "short_term_entries",
        "long_term_entries",
        "slot_bytes",
        "persistent_bytes",
        "write_attempts",
        "accepted_transitions",
        "rejected_transitions",
        "long_term_candidates",
        "long_term_writes",
        "short_term_evictions",
        "long_term_evictions",
        "long_term_rejections",
        "samples",
    )
    resources = {
        "exact_budget": composer.resource_budget(composer_state).to_config(),
        "final_counters": {
            "real_attempt_count": _int(composer_state.real_attempt_count),
            "accepted_real_event_count": _int(composer_state.accepted_real_event_count),
            "rejected_real_event_count": _int(composer_state.rejected_real_event_count),
            "rehearsal_attempt_count": quota_positions,
            "rehearsal_applied_count": available,
            "rehearsal_padding_count": padding,
            "ensemble_event_count": _int(ensemble_state.event_count),
            "ensemble_member_update_counts": member_counts,
            "ensemble_replay_event_count": _int(ensemble_state.replay_event_count),
            "ensemble_replay_member_update_counts": replay_member_counts,
            "dual_replay_accounting": {
                name: _int(getattr(replay_accounting, name))
                for name in replay_accounting_fields
            },
        },
    }
    return operations, resources


def _run_record(
    learner: LearnerName,
    model: object,
    stream: Mapping[str, object],
    root_key: Array,
    stream_sha256: str,
) -> tuple[dict[str, object], object]:
    if learner == "shallow_ridge_world_model":
        trace, state, initial_sha = _run_shallow(cast(ShallowRidgeWorldModel, model), stream)
        construction = cast(ShallowRidgeWorldModel, model).to_config()
    elif learner == "plain_world_model_ensemble":
        trace, state, initial_sha = _run_plain_ensemble(
            cast(WorldModelEnsemble, model), stream, root_key
        )
        construction = cast(WorldModelEnsemble, model).to_config()
    else:
        trace, state, initial_sha = _run_rehearsal(
            cast(ModelReplayRehearsal, model), stream, root_key
        )
        construction = cast(ModelReplayRehearsal, model).to_config()
    operations, resources = _operation_and_resource_accounting(
        learner, model, state, trace
    )
    record: dict[str, object] = {
        "learner": learner,
        "seed": stream["seed"],
        "common_stream_sha256": stream_sha256,
        "construction": construction,
        "initial_predictive_state_sha256": initial_sha,
        "initial_predictive_state_sha256_excludes": ["birth_timestamp", "uptime_s"],
        "prediction_order": "predict_before_update",
        "common_grounded_targets": True,
        "trace": trace,
        "trace_sha256": _canonical_sha256(trace),
        "metrics": _metrics(trace),
        "ensemble_disagreement_calibration": _calibration(trace, learner),
        "noisy_tv_diagnostic": _noisy_tv_diagnostic(trace, state, stream, learner),
        "operations": operations,
        "resources": resources,
        "assessment": {
            "status": ASSESSMENT_STATUS,
            "thresholds": None,
            "passed": None,
            "reason": "development diagnostics do not define an acceptance decision",
        },
    }
    return record, state


def _comparisons(runs: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for seed in DEVELOPMENT_SEEDS:
        seed_runs = {
            cast(LearnerName, run["learner"]): run for run in runs if run["seed"] == seed
        }
        plain = seed_runs["plain_world_model_ensemble"]
        replay = seed_runs["model_replay_rehearsal"]
        plain_trace = cast(Mapping[str, object], plain["trace"])
        replay_trace = cast(Mapping[str, object], replay["trace"])

        def mean_loss(name: LearnerName) -> float:
            metrics = cast(Mapping[str, object], seed_runs[name]["metrics"])
            aggregate = cast(Mapping[str, float], metrics["aggregate_prequential_loss"])
            return aggregate["mean"]

        plain_loss = mean_loss("plain_world_model_ensemble")
        comparisons.append(
            {
                "seed": seed,
                "common_raw_stream": len(
                    {run["common_stream_sha256"] for run in seed_runs.values()}
                )
                == 1,
                "plain_and_rehearsal_initial_ensemble_state_matched": (
                    plain["initial_predictive_state_sha256"]
                    == replay["initial_predictive_state_sha256"]
                ),
                "plain_and_rehearsal_real_bootstrap_masks_matched": (
                    plain_trace["real_member_updates_applied"]
                    == replay_trace["real_member_updates_applied"]
                ),
                "mean_aggregate_prequential_loss": {
                    learner: mean_loss(learner) for learner in LEARNER_ORDER
                },
                "signed_mean_loss_delta_vs_plain": {
                    "shallow_ridge_world_model": (
                        mean_loss("shallow_ridge_world_model") - plain_loss
                    ),
                    "model_replay_rehearsal": (
                        mean_loss("model_replay_rehearsal") - plain_loss
                    ),
                },
                "realized_resource_parity": False,
                "resource_parity_claim": None,
                "assessment": ASSESSMENT_STATUS,
                "winner": None,
            }
        )
    return comparisons


def _implementation_binding() -> dict[str, str]:
    source = Path(__file__).resolve()
    return {
        "path": "alberta_framework/benchmarks/world_model_retention_development.py",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _build_report(config: WorldModelRetentionDevelopmentConfig) -> dict[str, object]:
    shallow = ShallowRidgeWorldModel(_shallow_config())
    plain = WorldModelEnsemble(_ensemble_config())
    rehearsal = ModelReplayRehearsal(_rehearsal_config())
    models: dict[LearnerName, object] = {
        "shallow_ridge_world_model": shallow,
        "plain_world_model_ensemble": plain,
        "model_replay_rehearsal": rehearsal,
    }
    streams: list[dict[str, object]] = []
    runs: list[dict[str, object]] = []
    for seed in config.seeds:
        stream = _generate_stream(seed)
        stream_sha = _canonical_sha256(stream)
        streams.append({"stream": stream, "stream_sha256": stream_sha})
        root_key = jr.fold_in(jr.key(seed), 0x4D4F444C)
        for learner in LEARNER_ORDER:
            record, _ = _run_record(
                learner,
                models[learner],
                stream,
                root_key,
                stream_sha,
            )
            runs.append(record)
    body: dict[str, object] = {
        "schema_version": WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA,
        "development_only": DEVELOPMENT_ONLY,
        "scientific_promotion_allowed": SCIENTIFIC_PROMOTION_ALLOWED,
        "assessment_status": ASSESSMENT_STATUS,
        "claim_scope": _CLAIM_SCOPE,
        "interpretation": _INTERPRETATION,
        "implementation": _implementation_binding(),
        "protocol": config.to_config(),
        "matched_streams": streams,
        "runs": runs,
        "comparisons": _comparisons(runs),
        "comparability": {
            "common_raw_grounded_targets": True,
            "realized_resource_parity": False,
            "resource_parity_claim": None,
            "unavoidable_algorithm_and_state_differences": [
                "shallow ridge retains exact sufficient statistics and has no RNG or ensemble",
                "shallow ridge performs bounded PSD/eigvalsh state validation on each update",
                (
                    "ensemble runtime birth_timestamp/uptime_s metadata differs by construction; "
                    "the matched initialization SHA excludes only those nonpredictive clocks"
                ),
                "plain ensemble uses stochastic bootstrap masks and learned residual proxies",
                "rehearsal adds fixed-capacity dual replay and replay-only member updates",
                (
                    "rehearsal owns additional persistent state and executes additional "
                    "update positions"
                ),
            ],
            "accounting_scope": (
                "exact logical persistent resources and update events exposed by each mechanism; "
                "excludes FLOPs, compiler workspace, device allocation, energy, and wall clock"
            ),
        },
        "promotion": {
            "allowed": False,
            "evidence_level": None,
            "thresholds": None,
            "decision": None,
        },
    }
    body["report_digest"] = {
        "algorithm": "sha256",
        "scope": "$ excluding $.report_digest",
        "sha256": _canonical_sha256(body),
    }
    return body


@functools.lru_cache(maxsize=1)
def _expected_report(
    config: WorldModelRetentionDevelopmentConfig,
) -> dict[str, object]:
    return _build_report(config)


@dataclasses.dataclass(frozen=True)
class WorldModelRetentionDevelopmentValidation:
    """Fail-closed reconstruction result for one development report."""

    valid: bool
    errors: tuple[str, ...]


def validate_world_model_retention_development_report(
    report: Mapping[str, object],
) -> WorldModelRetentionDevelopmentValidation:
    """Rebuild every stream, learner trace, metric, resource count, and digest."""
    errors: list[str] = []
    if not isinstance(report, Mapping):
        return WorldModelRetentionDevelopmentValidation(False, ("report must be an object",))
    if not _json_tree_is_canonical(report):
        errors.append("report contains noncanonical JSON scalar/container types or non-finite data")
    if report.get("schema_version") != WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA:
        errors.append("world-model retention report schema is unsupported")
    if report.get("development_only") is not True:
        errors.append("report must remain development-only")
    if report.get("scientific_promotion_allowed") is not False:
        errors.append("report must forbid scientific promotion")
    if report.get("assessment_status") != ASSESSMENT_STATUS:
        errors.append("report must remain not assessed")
    implementation = report.get("implementation")
    if not isinstance(implementation, Mapping) or not _strict_json_equal(
        implementation, _implementation_binding()
    ):
        errors.append("implementation SHA-256 binding is invalid")

    protocol_payload = report.get("protocol")
    config: WorldModelRetentionDevelopmentConfig | None = None
    if not isinstance(protocol_payload, Mapping):
        errors.append("protocol must be an object")
    else:
        try:
            config = WorldModelRetentionDevelopmentConfig.from_config(protocol_payload)
        except (TypeError, ValueError) as exc:
            errors.append(f"protocol is invalid: {exc}")

    digest = report.get("report_digest")
    if not isinstance(digest, Mapping) or set(digest) != {"algorithm", "scope", "sha256"}:
        errors.append("report_digest fields do not match the v1 schema")
    else:
        unsigned = dict(report)
        unsigned.pop("report_digest", None)
        try:
            expected_digest = _canonical_sha256(unsigned)
        except (TypeError, ValueError, OverflowError):
            expected_digest = None
        if (
            expected_digest is None
            or digest.get("algorithm") != "sha256"
            or digest.get("scope") != "$ excluding $.report_digest"
            or digest.get("sha256") != expected_digest
        ):
            errors.append("report digest does not bind the complete payload")

    if config is not None:
        expected = _expected_report(config)
        try:
            reconstructs = _strict_json_equal(report, expected)
        except (TypeError, ValueError, OverflowError):
            reconstructs = False
        if not reconstructs:
            errors.append(
                "report does not exactly reconstruct every fixed stream, trace, equation, "
                "availability field, or resource count"
            )
    return WorldModelRetentionDevelopmentValidation(not errors, tuple(errors))


def run_world_model_retention_development(
    config: WorldModelRetentionDevelopmentConfig | None = None,
) -> dict[str, object]:
    """Run the one fixed nonpromoting matched-stream development protocol."""
    protocol = config or WorldModelRetentionDevelopmentConfig()
    if not isinstance(protocol, WorldModelRetentionDevelopmentConfig):
        raise TypeError("config must be a WorldModelRetentionDevelopmentConfig")
    # A JSON round-trip prevents callers from mutating the cached reconstruction.
    report = cast(dict[str, object], json.loads(_canonical_bytes(_expected_report(protocol))))
    validation = validate_world_model_retention_development_report(report)
    if not validation.valid:
        raise RuntimeError(
            "generated report failed self-validation: " + "; ".join(validation.errors)
        )
    return report


def world_model_retention_development_report_json(
    report: Mapping[str, object],
) -> str:
    """Return canonical pretty JSON only for a strictly valid report."""
    validation = validate_world_model_retention_development_report(report)
    if not validation.valid:
        raise ValueError("invalid world-model retention report: " + "; ".join(validation.errors))
    return json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def atomic_save_world_model_retention_development_report(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Validate, fsync, and atomically replace one report destination."""
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError("report destination parent directory does not exist")
    encoded = world_model_retention_development_report_json(report).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = [
    "ADAPTATION_WINDOW",
    "ASSESSMENT_STATUS",
    "CALIBRATION_BIN_LOWER_EDGES",
    "DEVELOPMENT_ONLY",
    "DEVELOPMENT_SEEDS",
    "FINAL_WINDOW",
    "LEARNER_ORDER",
    "PHASE_LENGTH",
    "RECOVERY_WINDOW",
    "REGIME_SCHEDULE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "WORLD_MODEL_RETENTION_DEVELOPMENT_SCHEMA",
    "WorldModelRetentionDevelopmentConfig",
    "WorldModelRetentionDevelopmentValidation",
    "atomic_save_world_model_retention_development_report",
    "run_world_model_retention_development",
    "validate_world_model_retention_development_report",
    "world_model_retention_development_report_json",
]
