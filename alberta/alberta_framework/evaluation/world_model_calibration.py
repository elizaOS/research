# mypy: disable-error-code="attr-defined,call-arg"
"""Strict development-only diagnostics for frozen world-model snapshots.

The evaluator owns held-out action-conditioned transition probes and only
calls read-only ``predict`` methods.  It never updates a learner, infers a
regime identifier, sets a performance threshold, or promotes a claim.  Raw
per-case predictions and targets are retained so every descriptive aggregate
can be reconstructed independently.

Ensemble disagreement and the residual-variance EMA are reported separately.
The latter remains explicitly non-probabilistic: no likelihood is defined by
the current world model, so probabilistic calibration, interval coverage, and
proper scoring-rule claims are unavailable.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from alberta_framework.core.checkpoints import (
    load_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint,
)
from alberta_framework.core.world_model import (
    ActionConditionedWorldModel,
    ActionConditionedWorldModelState,
)
from alberta_framework.core.world_model_ensemble import (
    WorldModelEnsemble,
    WorldModelEnsembleState,
)

WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA = "alberta.world-model-calibration.config.v1"
WORLD_MODEL_CALIBRATION_PROBE_SCHEMA = "alberta.world-model-calibration.probes.v1"
WORLD_MODEL_CALIBRATION_REPORT_SCHEMA = "alberta.world-model-calibration.report.v1"
WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA = (
    "alberta.world-model-calibration.snapshot.v1"
)
DEVELOPMENT_STATUS = "development-only-not-assessed"
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    Path("alberta_framework/core/initializers.py"),
    Path("alberta_framework/core/learning_signals.py"),
    Path("alberta_framework/core/multi_head_learner.py"),
    Path("alberta_framework/core/normalizers.py"),
    Path("alberta_framework/core/optimizers.py"),
    Path("alberta_framework/core/types.py"),
    Path("alberta_framework/core/world_model.py"),
    Path("alberta_framework/core/world_model_ensemble.py"),
    Path("alberta_framework/evaluation/world_model_calibration.py"),
)

ModelKind = Literal["ensemble", "single"]
EpistemicBinning = Literal["equal_count", "frozen_edges"]
DistributionPartition = Literal["in_distribution", "ood"]
FrozenModel = WorldModelEnsemble | ActionConditionedWorldModel
FrozenState = WorldModelEnsembleState | ActionConditionedWorldModelState

_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_REGIME_TOKENS = ("regime", "latent", "oracle")
_LIMITATIONS = (
    "development diagnostics only; assessment status is not-assessed",
    "held-out probes supplied here do not establish external validity",
    "ensemble disagreement is descriptive and is not a calibrated probability",
    "residual variance is a non-probabilistic EMA proxy because no likelihood exists",
    "no performance thresholds, scientific promotion, or comparative claim are made",
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def world_model_calibration_source_snapshot(
    root: Path = REPO_ROOT,
) -> dict[str, str]:
    """Hash every source file that can affect a diagnostic report."""
    return {relative.as_posix(): _file_sha256(root / relative) for relative in SOURCE_PATHS}


def _validate_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase canonical identifier")
    if any(token in value for token in _REGIME_TOKENS):
        raise ValueError(f"{name} must not encode a regime, latent, or oracle identifier")
    return value


def _finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_tuple(name: str, values: object, *, nonempty: bool = True) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{name} must be a tuple")
    if nonempty and not values:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_finite_real(f"{name}[{index}]", value) for index, value in enumerate(values))


def _strict_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclasses.dataclass(frozen=True)
class WorldModelCalibrationConfig:
    """Frozen descriptive binning, coverage, and resource bounds."""

    epistemic_binning: EpistemicBinning = "equal_count"
    epistemic_bin_count: int = 5
    epistemic_bin_edges: tuple[float, ...] = ()
    minimum_descriptive_bin_count: int = 2
    coverage_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    state_norm_edges: tuple[float, ...] = (1.0, 2.0)
    action_region_by_action: tuple[int, ...] = ()
    max_one_step_cases: int = 1024
    max_rollout_probes: int = 0
    max_rollout_horizon: int = 0

    def __post_init__(self) -> None:
        if self.epistemic_binning not in {"equal_count", "frozen_edges"}:
            raise ValueError("epistemic_binning must be equal_count or frozen_edges")
        _strict_positive_int("epistemic_bin_count", self.epistemic_bin_count)
        _strict_positive_int(
            "minimum_descriptive_bin_count",
            self.minimum_descriptive_bin_count,
        )
        edges = _finite_tuple(
            "epistemic_bin_edges",
            self.epistemic_bin_edges,
            nonempty=False,
        )
        if self.epistemic_binning == "equal_count" and edges:
            raise ValueError("equal_count binning requires empty epistemic_bin_edges")
        if self.epistemic_binning == "frozen_edges" and not edges:
            raise ValueError("frozen_edges binning requires epistemic_bin_edges")
        if any(left >= right for left, right in zip(edges, edges[1:], strict=False)):
            raise ValueError("epistemic_bin_edges must be strictly increasing")
        coverages = _finite_tuple("coverage_fractions", self.coverage_fractions)
        if (
            any(value <= 0.0 or value > 1.0 for value in coverages)
            or any(
                left >= right
                for left, right in zip(coverages, coverages[1:], strict=False)
            )
            or coverages[-1] != 1.0
        ):
            raise ValueError("coverage_fractions must increase in (0, 1] and end at 1")
        state_edges = _finite_tuple(
            "state_norm_edges",
            self.state_norm_edges,
            nonempty=False,
        )
        if any(value <= 0.0 for value in state_edges) or any(
            left >= right for left, right in zip(state_edges, state_edges[1:], strict=False)
        ):
            raise ValueError("state_norm_edges must be positive and strictly increasing")
        if not isinstance(self.action_region_by_action, tuple) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.action_region_by_action
        ):
            raise ValueError("action_region_by_action must contain non-negative integers")
        _strict_positive_int("max_one_step_cases", self.max_one_step_cases)
        for name in ("max_rollout_probes", "max_rollout_horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (self.max_rollout_probes == 0) != (self.max_rollout_horizon == 0):
            raise ValueError("rollout probe count and horizon must be enabled together")

    @property
    def effective_epistemic_bin_count(self) -> int:
        return (
            self.epistemic_bin_count
            if self.epistemic_binning == "equal_count"
            else len(self.epistemic_bin_edges) + 1
        )

    def to_config(self) -> dict[str, object]:
        return {
            "schema": WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA,
            "type": "WorldModelCalibrationConfig",
            "development_status": DEVELOPMENT_STATUS,
            "scientific_promotion_allowed": False,
            "performance_thresholds_applied": False,
            "epistemic_binning": self.epistemic_binning,
            "epistemic_bin_count": self.epistemic_bin_count,
            "epistemic_bin_edges": list(self.epistemic_bin_edges),
            "minimum_descriptive_bin_count": self.minimum_descriptive_bin_count,
            "coverage_fractions": list(self.coverage_fractions),
            "state_norm_edges": list(self.state_norm_edges),
            "action_region_by_action": list(self.action_region_by_action),
            "max_one_step_cases": self.max_one_step_cases,
            "max_rollout_probes": self.max_rollout_probes,
            "max_rollout_horizon": self.max_rollout_horizon,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> WorldModelCalibrationConfig:
        expected = set(cls().to_config())
        if set(payload) != expected:
            raise ValueError("world-model calibration config fields do not match v1")
        if payload.get("schema") != WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA:
            raise ValueError("unsupported world-model calibration config schema")
        if payload.get("type") != "WorldModelCalibrationConfig":
            raise ValueError("unexpected world-model calibration config type")
        if payload.get("development_status") != DEVELOPMENT_STATUS:
            raise ValueError("world-model calibration config must remain development-only")
        if payload.get("scientific_promotion_allowed") is not False:
            raise ValueError("world-model calibration config must forbid promotion")
        if payload.get("performance_thresholds_applied") is not False:
            raise ValueError("world-model calibration config cannot apply thresholds")

        def sequence(name: str) -> list[object]:
            value = payload.get(name)
            if not isinstance(value, list):
                raise ValueError(f"{name} must be a JSON array")
            return value

        return cls(
            epistemic_binning=cast(EpistemicBinning, payload.get("epistemic_binning")),
            epistemic_bin_count=cast(int, payload.get("epistemic_bin_count")),
            epistemic_bin_edges=tuple(
                cast(float, value) for value in sequence("epistemic_bin_edges")
            ),
            minimum_descriptive_bin_count=cast(
                int,
                payload.get("minimum_descriptive_bin_count"),
            ),
            coverage_fractions=tuple(
                cast(float, value) for value in sequence("coverage_fractions")
            ),
            state_norm_edges=tuple(
                cast(float, value) for value in sequence("state_norm_edges")
            ),
            action_region_by_action=tuple(
                cast(int, value) for value in sequence("action_region_by_action")
            ),
            max_one_step_cases=cast(int, payload.get("max_one_step_cases")),
            max_rollout_probes=cast(int, payload.get("max_rollout_probes")),
            max_rollout_horizon=cast(int, payload.get("max_rollout_horizon")),
        )


@dataclasses.dataclass(frozen=True)
class WorldModelCalibrationCase:
    """One evaluator-owned held-out transition with no regime identifier."""

    case_id: str
    observation: tuple[float, ...]
    action: int
    next_observation_target: tuple[float, ...]
    reward_target: float
    continuation_target: float
    partition: DistributionPartition

    def __post_init__(self) -> None:
        _validate_identifier("case_id", self.case_id)
        observation = _finite_tuple("observation", self.observation)
        target = _finite_tuple("next_observation_target", self.next_observation_target)
        if len(observation) != len(target):
            raise ValueError("observation and next_observation_target dimensions must match")
        if isinstance(self.action, bool) or not isinstance(self.action, int) or self.action < 0:
            raise ValueError("action must be a non-negative integer")
        _finite_real("reward_target", self.reward_target)
        continuation = _finite_real("continuation_target", self.continuation_target)
        if continuation < 0.0 or continuation > 1.0:
            raise ValueError("continuation_target must be in [0, 1]")
        if self.partition not in {"in_distribution", "ood"}:
            raise ValueError("partition must be in_distribution or ood")

    def to_config(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "observation": list(self.observation),
            "action": self.action,
            "next_observation_target": list(self.next_observation_target),
            "reward_target": self.reward_target,
            "continuation_target": self.continuation_target,
            "partition": self.partition,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> WorldModelCalibrationCase:
        expected = {
            "case_id",
            "observation",
            "action",
            "next_observation_target",
            "reward_target",
            "continuation_target",
            "partition",
        }
        if set(payload) != expected:
            raise ValueError("calibration case fields do not match v1")
        observation = payload.get("observation")
        target = payload.get("next_observation_target")
        if not isinstance(observation, list) or not isinstance(target, list):
            raise ValueError("case observations must be JSON arrays")
        return cls(
            case_id=cast(str, payload.get("case_id")),
            observation=tuple(cast(float, value) for value in observation),
            action=cast(int, payload.get("action")),
            next_observation_target=tuple(cast(float, value) for value in target),
            reward_target=cast(float, payload.get("reward_target")),
            continuation_target=cast(float, payload.get("continuation_target")),
            partition=cast(DistributionPartition, payload.get("partition")),
        )


@dataclasses.dataclass(frozen=True)
class WorldModelOpenLoopProbe:
    """Optional bounded rollout probe with explicit reconstruction authority."""

    probe_id: str
    initial_observation: tuple[float, ...]
    actions: tuple[int, ...]
    target_next_observations: tuple[tuple[float, ...], ...] = ()
    target_rewards: tuple[float, ...] = ()
    target_continuations: tuple[float, ...] = ()
    grounded_targets_available: bool = False
    exact_reconstruction_available: bool = False

    def __post_init__(self) -> None:
        _validate_identifier("probe_id", self.probe_id)
        observation = _finite_tuple("initial_observation", self.initial_observation)
        if not isinstance(self.actions, tuple) or not self.actions:
            raise ValueError("rollout actions must be a non-empty tuple")
        if any(
            isinstance(action, bool) or not isinstance(action, int) or action < 0
            for action in self.actions
        ):
            raise ValueError("rollout actions must be non-negative integers")
        if not isinstance(self.grounded_targets_available, bool) or not isinstance(
            self.exact_reconstruction_available,
            bool,
        ):
            raise ValueError("rollout availability flags must be boolean")
        if not isinstance(self.target_next_observations, tuple) or not isinstance(
            self.target_rewards,
            tuple,
        ) or not isinstance(self.target_continuations, tuple):
            raise ValueError("rollout target sequences must be tuples")
        for index, target in enumerate(self.target_next_observations):
            if len(_finite_tuple(f"target_next_observations[{index}]", target)) != len(
                observation
            ):
                raise ValueError("rollout target observation dimension mismatch")
        for index, reward in enumerate(self.target_rewards):
            _finite_real(f"target_rewards[{index}]", reward)
        for index, continuation in enumerate(self.target_continuations):
            value = _finite_real(f"target_continuations[{index}]", continuation)
            if value < 0.0 or value > 1.0:
                raise ValueError("rollout continuation targets must be in [0, 1]")
        if self.grounded_targets_available and self.exact_reconstruction_available:
            horizon = len(self.actions)
            if (
                len(self.target_next_observations) != horizon
                or len(self.target_rewards) != horizon
                or len(self.target_continuations) != horizon
            ):
                raise ValueError("grounded exact rollout targets must match the action horizon")

    def to_config(self) -> dict[str, object]:
        return {
            "probe_id": self.probe_id,
            "initial_observation": list(self.initial_observation),
            "actions": list(self.actions),
            "target_next_observations": [list(value) for value in self.target_next_observations],
            "target_rewards": list(self.target_rewards),
            "target_continuations": list(self.target_continuations),
            "grounded_targets_available": self.grounded_targets_available,
            "exact_reconstruction_available": self.exact_reconstruction_available,
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> WorldModelOpenLoopProbe:
        expected = {
            "probe_id",
            "initial_observation",
            "actions",
            "target_next_observations",
            "target_rewards",
            "target_continuations",
            "grounded_targets_available",
            "exact_reconstruction_available",
        }
        if set(payload) != expected:
            raise ValueError("open-loop probe fields do not match v1")
        initial = payload.get("initial_observation")
        actions = payload.get("actions")
        observations = payload.get("target_next_observations")
        rewards = payload.get("target_rewards")
        continuations = payload.get("target_continuations")
        sequences = (initial, actions, observations, rewards, continuations)
        if not all(isinstance(value, list) for value in sequences):
            raise ValueError("open-loop probe sequences must be JSON arrays")
        observation_rows = cast(list[object], observations)
        if any(not isinstance(row, list) for row in observation_rows):
            raise ValueError("target_next_observations must contain arrays")
        return cls(
            probe_id=cast(str, payload.get("probe_id")),
            initial_observation=tuple(cast(float, value) for value in cast(list[object], initial)),
            actions=tuple(cast(int, value) for value in cast(list[object], actions)),
            target_next_observations=tuple(
                tuple(cast(float, value) for value in cast(list[object], row))
                for row in observation_rows
            ),
            target_rewards=tuple(cast(float, value) for value in cast(list[object], rewards)),
            target_continuations=tuple(
                cast(float, value) for value in cast(list[object], continuations)
            ),
            grounded_targets_available=cast(bool, payload.get("grounded_targets_available")),
            exact_reconstruction_available=cast(
                bool,
                payload.get("exact_reconstruction_available"),
            ),
        )


@dataclasses.dataclass(frozen=True)
class WorldModelCalibrationProbeSet:
    """Evaluator-owned held-out cases, never learner or regime inputs."""

    probe_set_id: str
    cases: tuple[WorldModelCalibrationCase, ...]
    open_loop_probes: tuple[WorldModelOpenLoopProbe, ...] = ()
    ownership: str = "evaluator-owned-held-out"
    learner_use: str = "never"
    regime_identifiers_available: bool = False

    def __post_init__(self) -> None:
        _validate_identifier("probe_set_id", self.probe_set_id)
        if not isinstance(self.cases, tuple) or not self.cases:
            raise ValueError("probe set must contain held-out cases")
        if not isinstance(self.open_loop_probes, tuple):
            raise ValueError("open_loop_probes must be a tuple")
        identifiers = [case.case_id for case in self.cases] + [
            probe.probe_id for probe in self.open_loop_probes
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("probe identifiers must be unique")
        if self.ownership != "evaluator-owned-held-out" or self.learner_use != "never":
            raise ValueError("probe ownership and learner-use contracts are fixed")
        if self.regime_identifiers_available is not False:
            raise ValueError("regime identifiers must be unavailable")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": WORLD_MODEL_CALIBRATION_PROBE_SCHEMA,
            "type": "WorldModelCalibrationProbeSet",
            "development_status": DEVELOPMENT_STATUS,
            "ownership": self.ownership,
            "learner_use": self.learner_use,
            "regime_identifiers_available": self.regime_identifiers_available,
            "probe_set_id": self.probe_set_id,
            "cases": [case.to_config() for case in self.cases],
            "open_loop_probes": [probe.to_config() for probe in self.open_loop_probes],
        }

    @classmethod
    def from_config(cls, payload: Mapping[str, object]) -> WorldModelCalibrationProbeSet:
        expected = set(
            cls(
                probe_set_id="template",
                cases=(
                    WorldModelCalibrationCase(
                        case_id="case",
                        observation=(0.0,),
                        action=0,
                        next_observation_target=(0.0,),
                        reward_target=0.0,
                        continuation_target=0.0,
                        partition="in_distribution",
                    ),
                ),
            ).to_config()
        )
        if set(payload) != expected:
            raise ValueError("world-model probe-set fields do not match v1")
        if payload.get("schema") != WORLD_MODEL_CALIBRATION_PROBE_SCHEMA:
            raise ValueError("unsupported world-model probe-set schema")
        if payload.get("type") != "WorldModelCalibrationProbeSet":
            raise ValueError("unexpected world-model probe-set type")
        if payload.get("development_status") != DEVELOPMENT_STATUS:
            raise ValueError("probe set must remain development-only")
        cases = payload.get("cases")
        rollouts = payload.get("open_loop_probes")
        if not isinstance(cases, list) or not isinstance(rollouts, list):
            raise ValueError("probe cases and rollouts must be arrays")
        if any(not isinstance(value, Mapping) for value in (*cases, *rollouts)):
            raise ValueError("probe records must be objects")
        return cls(
            probe_set_id=cast(str, payload.get("probe_set_id")),
            cases=tuple(
                WorldModelCalibrationCase.from_config(cast(Mapping[str, object], value))
                for value in cases
            ),
            open_loop_probes=tuple(
                WorldModelOpenLoopProbe.from_config(cast(Mapping[str, object], value))
                for value in rollouts
            ),
            ownership=cast(str, payload.get("ownership")),
            learner_use=cast(str, payload.get("learner_use")),
            regime_identifiers_available=cast(
                bool,
                payload.get("regime_identifiers_available"),
            ),
        )


@dataclasses.dataclass(frozen=True)
class WorldModelCalibrationValidation:
    """Fail-closed report validation outcome without an assessment verdict."""

    valid: bool
    assessment_status: str
    errors: tuple[str, ...]


def _materialize_key(value: object) -> object:
    dtype = getattr(value, "dtype", None)
    if dtype is not None and jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key):
        return jr.key_data(cast(Array, value))
    return value


def _state_tree_payload(state: FrozenState) -> dict[str, object]:
    leaves, structure = jax.tree.flatten(state)
    payload: list[dict[str, object]] = []
    for index, leaf in enumerate(leaves):
        array = np.asarray(jax.device_get(jnp.asarray(_materialize_key(leaf))))
        contiguous = np.ascontiguousarray(array)
        payload.append(
            {
                "index": index,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "data_base64": base64.b64encode(contiguous.tobytes(order="C")).decode(
                    "ascii"
                ),
            }
        )
    return {"tree_structure": str(structure), "leaves": payload}


def _logical_tree_size(state: FrozenState) -> tuple[int, int]:
    arrays = [
        np.asarray(jax.device_get(jnp.asarray(_materialize_key(leaf))))
        for leaf in jax.tree.leaves(state)
    ]
    return (
        sum(int(array.size) for array in arrays),
        sum(int(array.nbytes) for array in arrays),
    )


def frozen_world_model_state_sha256(state: FrozenState) -> str:
    """Hash complete shape, dtype, tree structure, key data, and state bytes."""
    return _canonical_sha256(_state_tree_payload(state))


def _model_kind(model: FrozenModel) -> ModelKind:
    if isinstance(model, WorldModelEnsemble):
        return "ensemble"
    if isinstance(model, ActionConditionedWorldModel):
        return "single"
    raise TypeError("model must be WorldModelEnsemble or ActionConditionedWorldModel")


def _model_config(model: FrozenModel) -> dict[str, Any]:
    return model.to_config()


def _model_dimensions(model: FrozenModel) -> tuple[int, int]:
    if isinstance(model, WorldModelEnsemble):
        return model.config.model.observation_dim, model.config.model.n_actions
    return model.config.observation_dim, model.config.n_actions


def _snapshot_descriptor(model: FrozenModel, state: FrozenState) -> dict[str, object]:
    kind = _model_kind(model)
    if kind == "ensemble":
        if not isinstance(state, WorldModelEnsembleState):
            raise TypeError("ensemble model requires WorldModelEnsembleState")
        ensemble = cast(WorldModelEnsemble, model)
        if not bool(jax.device_get(ensemble.state_valid(state))):
            raise ValueError("frozen ensemble state is invalid")
        budget = ensemble.resource_budget(state)
        scalars = budget.persistent_state_scalars
        state_bytes = budget.persistent_state_bytes
        ensemble_size: int | None = ensemble.config.ensemble_size
        real_event_count = int(jax.device_get(state.event_count))
        residual_warmup: int | None = ensemble.config.residual_variance_warmup_steps
    else:
        if not isinstance(state, ActionConditionedWorldModelState):
            raise TypeError("single model requires ActionConditionedWorldModelState")
        scalars, state_bytes = _logical_tree_size(state)
        ensemble_size = None
        real_event_count = int(jax.device_get(state.step_count))
        residual_warmup = None
    model_config = _model_config(model)
    descriptor: dict[str, object] = {
        "model_kind": kind,
        "model_config": model_config,
        "model_config_sha256": _canonical_sha256(model_config),
        "state_sha256": frozen_world_model_state_sha256(state),
        "state_logical_scalars": scalars,
        "state_bytes": state_bytes,
        "ensemble_size_available": ensemble_size is not None,
        "ensemble_size": ensemble_size,
        "real_event_count": real_event_count,
        "residual_proxy_warmup_steps_available": residual_warmup is not None,
        "residual_proxy_warmup_steps": residual_warmup,
    }
    return descriptor


def frozen_world_model_snapshot_sha256(model: FrozenModel, state: FrozenState) -> str:
    """Hash the canonical model construction, complete state hash, and resources."""
    return _canonical_sha256(_snapshot_descriptor(model, state))


def _model_from_snapshot_config(
    kind: object,
    config: Mapping[str, object],
) -> FrozenModel:
    mutable = cast(dict[str, Any], dict(config))
    if kind == "ensemble":
        model: FrozenModel = WorldModelEnsemble.from_config(mutable)
    elif kind == "single":
        model = ActionConditionedWorldModel.from_config(mutable)
    else:
        raise ValueError("snapshot model_kind must be ensemble or single")
    if model.to_config() != config:
        raise ValueError("snapshot model config is not canonical")
    return model


def save_world_model_calibration_snapshot_checkpoint(
    model: FrozenModel,
    state: FrozenState,
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Persist a strict frozen snapshot checkpoint without learner mutation."""
    destination = Path(path).expanduser()
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite snapshot checkpoint: {destination}")
    descriptor = _snapshot_descriptor(model, state)
    sources = world_model_calibration_source_snapshot(root)
    save_checkpoint(
        state,
        destination,
        metadata={
            "schema": WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA,
            "development_status": DEVELOPMENT_STATUS,
            "scientific_promotion_allowed": False,
            "snapshot": descriptor,
            "snapshot_sha256": _canonical_sha256(descriptor),
            "source_sha256": sources,
            "source_manifest_sha256": _canonical_sha256(sources),
        },
    )


def load_world_model_calibration_snapshot_checkpoint(
    path: str | Path,
    *,
    template_key: Array | None = None,
    root: Path = REPO_ROOT,
) -> tuple[FrozenModel, FrozenState]:
    """Restore and verify a strict frozen ensemble or single-model snapshot."""
    metadata = load_checkpoint_metadata(path)
    expected_top = {
        "schema",
        "development_status",
        "scientific_promotion_allowed",
        "snapshot",
        "snapshot_sha256",
        "source_sha256",
        "source_manifest_sha256",
    }
    if set(metadata) != expected_top:
        raise ValueError("snapshot checkpoint metadata fields do not match v1")
    if metadata.get("schema") != WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported world-model calibration snapshot checkpoint")
    if metadata.get("development_status") != DEVELOPMENT_STATUS:
        raise ValueError("snapshot checkpoint must remain development-only")
    if metadata.get("scientific_promotion_allowed") is not False:
        raise ValueError("snapshot checkpoint must forbid promotion")
    sources = world_model_calibration_source_snapshot(root)
    if metadata.get("source_sha256") != sources or metadata.get(
        "source_manifest_sha256"
    ) != _canonical_sha256(sources):
        raise ValueError("snapshot checkpoint source hashes do not match")
    snapshot = metadata.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot checkpoint is missing snapshot descriptor")
    if metadata.get("snapshot_sha256") != _canonical_sha256(snapshot):
        raise ValueError("snapshot checkpoint descriptor digest does not match")
    model_config = snapshot.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("snapshot descriptor is missing model_config")
    if snapshot.get("model_config_sha256") != _canonical_sha256(model_config):
        raise ValueError("snapshot model config digest does not match")
    model = _model_from_snapshot_config(snapshot.get("model_kind"), model_config)
    key = jr.key(0) if template_key is None else template_key
    template: FrozenState
    if isinstance(model, WorldModelEnsemble):
        template = model.init(key)
    else:
        template = model.init(key)
    restored, restored_metadata = load_checkpoint(template, path)
    if restored_metadata != metadata:
        raise ValueError("snapshot checkpoint metadata changed between reads")
    state = cast(FrozenState, restored)
    reconstructed = _snapshot_descriptor(model, state)
    if reconstructed != snapshot or _canonical_sha256(reconstructed) != metadata.get(
        "snapshot_sha256"
    ):
        raise ValueError("restored snapshot state, resources, or hash do not match")
    return model, state


def _python_float(value: object) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _python_vector(value: object) -> list[float]:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    return [float(item) for item in array.reshape((-1,))]


def _python_matrix(value: object) -> list[list[float]]:
    array = np.asarray(jax.device_get(value), dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("prediction matrix must be two-dimensional")
    return [[float(item) for item in row] for row in array]


def _effective_action_regions(
    config: WorldModelCalibrationConfig,
    n_actions: int,
) -> tuple[int, ...]:
    if config.action_region_by_action:
        if len(config.action_region_by_action) != n_actions:
            raise ValueError("action_region_by_action length must match model n_actions")
        return config.action_region_by_action
    return tuple(range(n_actions))


def _validate_probe_dimensions(
    probes: WorldModelCalibrationProbeSet,
    *,
    observation_dim: int,
    n_actions: int,
    config: WorldModelCalibrationConfig,
) -> None:
    if len(probes.cases) > config.max_one_step_cases:
        raise ValueError("held-out case count exceeds max_one_step_cases")
    for case in probes.cases:
        if len(case.observation) != observation_dim:
            raise ValueError(f"case {case.case_id} observation dimension does not match model")
        if case.action >= n_actions:
            raise ValueError(f"case {case.case_id} action is outside model action space")
    if len(probes.open_loop_probes) > config.max_rollout_probes:
        raise ValueError("open-loop probe count exceeds max_rollout_probes")
    for probe in probes.open_loop_probes:
        if len(probe.initial_observation) != observation_dim:
            raise ValueError(f"probe {probe.probe_id} observation dimension does not match model")
        if len(probe.actions) > config.max_rollout_horizon:
            raise ValueError(f"probe {probe.probe_id} exceeds max_rollout_horizon")
        if any(action >= n_actions for action in probe.actions):
            raise ValueError(f"probe {probe.probe_id} action is outside model action space")


def _predict_one_case(
    model: FrozenModel,
    state: FrozenState,
    case: WorldModelCalibrationCase,
    config: WorldModelCalibrationConfig,
    action_regions: tuple[int, ...],
) -> dict[str, object]:
    observation = jnp.asarray(case.observation, dtype=jnp.float32)
    action = jnp.asarray(case.action, dtype=jnp.int32)
    reward = jnp.asarray(case.reward_target, dtype=jnp.float32)
    continuation = jnp.asarray(case.continuation_target, dtype=jnp.float32)
    next_observation = jnp.asarray(case.next_observation_target, dtype=jnp.float32)
    state_region = int(
        np.digitize(
            np.asarray([np.linalg.norm(np.asarray(case.observation, dtype=np.float64))]),
            np.asarray(config.state_norm_edges, dtype=np.float64),
        )[0]
    )
    action_region = action_regions[case.action]

    if isinstance(model, WorldModelEnsemble):
        if not isinstance(state, WorldModelEnsembleState):
            raise TypeError("ensemble model requires ensemble state")
        prediction = model.predict(state, observation, action)
        if not bool(jax.device_get(prediction.valid)):
            raise ValueError(f"case {case.case_id} produced an invalid ensemble prediction")
        raw_targets = model.member_model.targets(
            observation,
            reward,
            continuation,
            next_observation,
        )
        member_next = np.asarray(
            jax.device_get(prediction.member_next_observations),
            dtype=np.float64,
        )
        member_reward = np.asarray(jax.device_get(prediction.member_rewards), dtype=np.float64)
        member_continuation = np.asarray(
            jax.device_get(prediction.member_discounts),
            dtype=np.float64,
        )
        decoded = np.concatenate(
            (member_next, member_reward[:, None], member_continuation[:, None]),
            axis=1,
        )
        decoded_variances = np.var(decoded, axis=0)
        members: dict[str, object] = {
            "available": True,
            "count": model.config.ensemble_size,
            "next_observations": _python_matrix(prediction.member_next_observations),
            "rewards": _python_vector(prediction.member_rewards),
            "continuations": _python_vector(prediction.member_discounts),
            "raw_predictions": _python_matrix(prediction.member_raw_predictions),
        }
        epistemic: dict[str, object] = {
            "available": True,
            "decoded_per_head_variance": [float(value) for value in decoded_variances],
            "decoded_mean_disagreement": float(np.mean(decoded_variances)),
            "ensemble_raw_disagreement": _python_float(
                prediction.epistemic_disagreement
            ),
        }
        residual_proxy: dict[str, object] = {
            "available": True,
            "ready": bool(jax.device_get(prediction.residual_proxy_ready)),
            "member_raw_head_variances": _python_matrix(
                prediction.residual_variances
            ),
            "probabilistic_calibration_available": False,
            "interpretation": (
                "non-probabilistic residual-variance EMA proxy; no valid likelihood"
            ),
        }
        mean_raw = prediction.mean_raw_prediction
        mean_next = prediction.mean_next_observation
        mean_reward = prediction.mean_reward
        mean_continuation = prediction.mean_discount
    elif isinstance(model, ActionConditionedWorldModel):
        if not isinstance(state, ActionConditionedWorldModelState):
            raise TypeError("single model requires single-model state")
        prediction = model.predict(state, observation, action)
        raw_targets = model.targets(
            observation,
            reward,
            continuation,
            next_observation,
        )
        members = {
            "available": False,
            "count": None,
            "next_observations": None,
            "rewards": None,
            "continuations": None,
            "raw_predictions": None,
        }
        epistemic = {
            "available": False,
            "decoded_per_head_variance": None,
            "decoded_mean_disagreement": None,
            "ensemble_raw_disagreement": None,
        }
        residual_proxy = {
            "available": False,
            "ready": False,
            "member_raw_head_variances": None,
            "probabilistic_calibration_available": False,
            "interpretation": "unavailable: model snapshot has no ensemble residual proxy",
        }
        mean_raw = prediction.raw_predictions
        mean_next = prediction.next_observation
        mean_reward = prediction.reward
        mean_continuation = prediction.discount
    else:
        raise TypeError("unsupported frozen model")

    return {
        "case_id": case.case_id,
        "observation": list(case.observation),
        "action": case.action,
        "partition": case.partition,
        "state_region": state_region,
        "action_region": action_region,
        "targets": {
            "next_observation": list(case.next_observation_target),
            "reward": case.reward_target,
            "continuation": case.continuation_target,
            "raw": _python_vector(raw_targets),
        },
        "mean_predictions": {
            "next_observation": _python_vector(mean_next),
            "reward": _python_float(mean_reward),
            "continuation": _python_float(mean_continuation),
            "raw": _python_vector(mean_raw),
        },
        "members": members,
        "epistemic": epistemic,
        "residual_variance_proxy": residual_proxy,
    }


def _rollout_applicability(
    config: WorldModelCalibrationConfig,
    probes: WorldModelCalibrationProbeSet,
) -> tuple[bool, str]:
    if config.max_rollout_probes == 0:
        return False, "unavailable: bounded open-loop probes are disabled by config"
    if not probes.open_loop_probes:
        return False, "unavailable: no open-loop probes were supplied"
    if any(
        not probe.grounded_targets_available
        or not probe.exact_reconstruction_available
        for probe in probes.open_loop_probes
    ):
        return (
            False,
            "unavailable: grounded targets and exact action-sequence reconstruction are required",
        )
    return True, "available: every rollout has grounded targets and exact reconstruction"


def _predict_mean(
    model: FrozenModel,
    state: FrozenState,
    observation: Array,
    action: Array,
) -> tuple[Array, Array, Array]:
    if isinstance(model, WorldModelEnsemble):
        if not isinstance(state, WorldModelEnsembleState):
            raise TypeError("ensemble model requires ensemble state")
        prediction = model.predict(state, observation, action)
        if not bool(jax.device_get(prediction.valid)):
            raise ValueError("open-loop ensemble prediction is invalid")
        return (
            prediction.mean_next_observation,
            prediction.mean_reward,
            prediction.mean_discount,
        )
    if not isinstance(state, ActionConditionedWorldModelState):
        raise TypeError("single model requires single-model state")
    prediction = model.predict(state, observation, action)
    return prediction.next_observation, prediction.reward, prediction.discount


def _evaluate_open_loop_trace(
    model: FrozenModel,
    state: FrozenState,
    config: WorldModelCalibrationConfig,
    probes: WorldModelCalibrationProbeSet,
) -> tuple[dict[str, object], int]:
    applicable, reason = _rollout_applicability(config, probes)
    if not applicable:
        return {"available": False, "reason": reason, "probes": []}, 0
    records: list[dict[str, object]] = []
    calls = 0
    for probe in probes.open_loop_probes:
        observation = jnp.asarray(probe.initial_observation, dtype=jnp.float32)
        steps: list[dict[str, object]] = []
        for index, action_value in enumerate(probe.actions):
            action = jnp.asarray(action_value, dtype=jnp.int32)
            next_prediction, reward_prediction, continuation_prediction = _predict_mean(
                model,
                state,
                observation,
                action,
            )
            calls += 1
            steps.append(
                {
                    "step": index,
                    "action": action_value,
                    "input_observation": _python_vector(observation),
                    "mean_predictions": {
                        "next_observation": _python_vector(next_prediction),
                        "reward": _python_float(reward_prediction),
                        "continuation": _python_float(continuation_prediction),
                    },
                    "grounded_targets": {
                        "next_observation": list(probe.target_next_observations[index]),
                        "reward": probe.target_rewards[index],
                        "continuation": probe.target_continuations[index],
                    },
                }
            )
            observation = next_prediction
        records.append(
            {
                "probe_id": probe.probe_id,
                "initial_observation": list(probe.initial_observation),
                "steps": steps,
            }
        )
    return {"available": True, "reason": reason, "probes": records}, calls


def _evaluate_raw_trace(
    model: FrozenModel,
    state: FrozenState,
    config: WorldModelCalibrationConfig,
    probes: WorldModelCalibrationProbeSet,
) -> tuple[dict[str, object], int, int]:
    observation_dim, n_actions = _model_dimensions(model)
    _validate_probe_dimensions(
        probes,
        observation_dim=observation_dim,
        n_actions=n_actions,
        config=config,
    )
    action_regions = _effective_action_regions(config, n_actions)
    cases = [
        _predict_one_case(model, state, case, config, action_regions)
        for case in probes.cases
    ]
    rollout, rollout_calls = _evaluate_open_loop_trace(model, state, config, probes)
    return {"cases": cases, "open_loop": rollout}, len(cases), rollout_calls


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _numeric_vector(value: object, name: str, *, size: int | None = None) -> np.ndarray:
    values = _list(value, name)
    array = np.asarray(
        [_number(item, f"{name}[{index}]") for index, item in enumerate(values)],
        dtype=np.float64,
    )
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have length {size}")
    return array


def _numeric_matrix(
    value: object,
    name: str,
    *,
    columns: int | None = None,
) -> np.ndarray:
    rows = _list(value, name)
    parsed = [
        _numeric_vector(row, f"{name}[{index}]", size=columns)
        for index, row in enumerate(rows)
    ]
    if not parsed:
        return np.zeros((0, 0 if columns is None else columns), dtype=np.float64)
    width = parsed[0].shape[0]
    if any(row.shape != (width,) for row in parsed):
        raise ValueError(f"{name} rows must have equal length")
    return np.stack(parsed)


def _case_primitives(
    record: object,
    *,
    observation_dim: int,
) -> dict[str, object]:
    case = _mapping(record, "raw case")
    expected = {
        "case_id",
        "observation",
        "action",
        "partition",
        "state_region",
        "action_region",
        "targets",
        "mean_predictions",
        "members",
        "epistemic",
        "residual_variance_proxy",
    }
    if set(case) != expected:
        raise ValueError("raw case fields do not match v1")
    _validate_identifier("raw case_id", case.get("case_id"))
    observation = _numeric_vector(
        case.get("observation"),
        "raw observation",
        size=observation_dim,
    )
    action = case.get("action")
    state_region = case.get("state_region")
    action_region = case.get("action_region")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (action, state_region, action_region)
    ):
        raise ValueError("raw action and region indices must be non-negative integers")
    partition = case.get("partition")
    if partition not in {"in_distribution", "ood"}:
        raise ValueError("raw partition is invalid")
    targets = _mapping(case.get("targets"), "raw targets")
    means = _mapping(case.get("mean_predictions"), "raw mean_predictions")
    expected_prediction = {"next_observation", "reward", "continuation", "raw"}
    if set(targets) != expected_prediction or set(means) != expected_prediction:
        raise ValueError("target or mean-prediction fields do not match v1")
    target_next = _numeric_vector(
        targets.get("next_observation"),
        "target next_observation",
        size=observation_dim,
    )
    mean_next = _numeric_vector(
        means.get("next_observation"),
        "mean next_observation",
        size=observation_dim,
    )
    target_reward = _number(targets.get("reward"), "target reward")
    mean_reward = _number(means.get("reward"), "mean reward")
    target_continuation = _number(targets.get("continuation"), "target continuation")
    mean_continuation = _number(means.get("continuation"), "mean continuation")
    raw_target = _numeric_vector(targets.get("raw"), "raw target")
    raw_mean = _numeric_vector(means.get("raw"), "raw mean")
    if raw_target.shape != raw_mean.shape or raw_target.shape != (observation_dim + 2,):
        raise ValueError("raw target and prediction head dimensions are invalid")

    members = _mapping(case.get("members"), "raw members")
    expected_members = {
        "available",
        "count",
        "next_observations",
        "rewards",
        "continuations",
        "raw_predictions",
    }
    if set(members) != expected_members or not isinstance(members.get("available"), bool):
        raise ValueError("raw member fields are invalid")
    member_available = cast(bool, members["available"])
    member_raw: np.ndarray | None = None
    if member_available:
        count = members.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("available member predictions require count >= 2")
        member_next = _numeric_matrix(
            members.get("next_observations"),
            "member next_observations",
            columns=observation_dim,
        )
        member_rewards = _numeric_vector(members.get("rewards"), "member rewards", size=count)
        member_continuations = _numeric_vector(
            members.get("continuations"),
            "member continuations",
            size=count,
        )
        member_raw = _numeric_matrix(
            members.get("raw_predictions"),
            "member raw_predictions",
            columns=observation_dim + 2,
        )
        if member_next.shape[0] != count or member_raw.shape[0] != count:
            raise ValueError("member prediction rows do not match member count")
        if not np.allclose(np.mean(member_next, axis=0), mean_next, rtol=0.0, atol=1e-6):
            raise ValueError("mean next-observation prediction does not reconstruct from members")
        if not math.isclose(
            float(np.mean(member_rewards)), mean_reward, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError("mean reward prediction does not reconstruct from members")
        if not math.isclose(
            float(np.mean(member_continuations)),
            mean_continuation,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("mean continuation prediction does not reconstruct from members")
        if not np.allclose(np.mean(member_raw, axis=0), raw_mean, rtol=0.0, atol=1e-6):
            raise ValueError("mean raw prediction does not reconstruct from members")
        decoded = np.concatenate(
            (member_next, member_rewards[:, None], member_continuations[:, None]),
            axis=1,
        )
        decoded_variance = np.var(decoded, axis=0)
    else:
        if members.get("count") is not None or any(
            members.get(name) is not None
            for name in (
                "next_observations",
                "rewards",
                "continuations",
                "raw_predictions",
            )
        ):
            raise ValueError("unavailable member fields must be null")
        decoded_variance = None

    epistemic = _mapping(case.get("epistemic"), "raw epistemic")
    expected_epistemic = {
        "available",
        "decoded_per_head_variance",
        "decoded_mean_disagreement",
        "ensemble_raw_disagreement",
    }
    if set(epistemic) != expected_epistemic or epistemic.get("available") is not member_available:
        raise ValueError("epistemic availability must exactly match member availability")
    disagreement: float | None = None
    if member_available:
        recorded_variance = _numeric_vector(
            epistemic.get("decoded_per_head_variance"),
            "decoded epistemic variance",
            size=observation_dim + 2,
        )
        if decoded_variance is None or not np.allclose(
            recorded_variance,
            decoded_variance,
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("decoded epistemic variance does not reconstruct")
        disagreement = _number(
            epistemic.get("decoded_mean_disagreement"),
            "decoded mean disagreement",
        )
        if not math.isclose(
            disagreement,
            float(np.mean(recorded_variance)),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ValueError("decoded mean disagreement does not reconstruct")
        raw_disagreement = _number(
            epistemic.get("ensemble_raw_disagreement"),
            "raw disagreement",
        )
        if member_raw is None or not math.isclose(
            raw_disagreement,
            float(np.mean(np.var(member_raw, axis=0))),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("raw ensemble disagreement does not reconstruct from members")
    elif any(epistemic.get(name) is not None for name in expected_epistemic - {"available"}):
        raise ValueError("unavailable epistemic fields must be null")

    residual = _mapping(case.get("residual_variance_proxy"), "residual proxy")
    expected_residual = {
        "available",
        "ready",
        "member_raw_head_variances",
        "probabilistic_calibration_available",
        "interpretation",
    }
    if set(residual) != expected_residual:
        raise ValueError("residual proxy fields do not match v1")
    if residual.get("probabilistic_calibration_available") is not False:
        raise ValueError("probabilistic calibration must remain unavailable")
    residual_available = residual.get("available")
    residual_ready = residual.get("ready")
    if not isinstance(residual_available, bool) or not isinstance(residual_ready, bool):
        raise ValueError("residual proxy availability fields must be boolean")
    residual_values: np.ndarray | None = None
    if residual_available:
        if not member_available or member_raw is None:
            raise ValueError("residual proxy requires member raw predictions")
        residual_values = _numeric_matrix(
            residual.get("member_raw_head_variances"),
            "residual proxy matrix",
            columns=observation_dim + 2,
        )
        if residual_values.shape != member_raw.shape:
            raise ValueError("residual proxy shape must match member raw predictions")
        if np.any(residual_values < 0.0):
            raise ValueError("residual proxy values must be non-negative")
    elif residual.get("member_raw_head_variances") is not None or residual_ready:
        raise ValueError("unavailable residual proxy fields must be null/not-ready")

    decoded_target = np.concatenate(
        (target_next, np.asarray([target_reward, target_continuation]))
    )
    decoded_mean = np.concatenate(
        (mean_next, np.asarray([mean_reward, mean_continuation]))
    )
    squared_error = np.square(decoded_mean - decoded_target)
    return {
        "case_id": case["case_id"],
        "observation": observation,
        "action": cast(int, action),
        "partition": partition,
        "state_region": cast(int, state_region),
        "action_region": cast(int, action_region),
        "squared_error": squared_error,
        "realized_mean_squared_error": float(np.mean(squared_error)),
        "disagreement": disagreement,
        "member_raw": member_raw,
        "raw_target": raw_target,
        "residual_available": residual_available,
        "residual_ready": residual_ready,
        "residual_values": residual_values,
    }


def _descriptive_group(
    primitives: Sequence[dict[str, object]],
    indices: Sequence[int],
    *,
    observation_dim: int,
    minimum_count: int,
) -> dict[str, object]:
    count = len(indices)
    sparse = count < minimum_count
    if count == 0:
        return {
            "count": 0,
            "sparse": True,
            "descriptive_applicable": False,
            "next_observation_mse_by_dimension": None,
            "reward_mse": None,
            "continuation_mse": None,
            "all_head_mean_squared_error": None,
        }
    errors = np.stack(
        [cast(np.ndarray, primitives[index]["squared_error"]) for index in indices]
    )
    return {
        "count": count,
        "sparse": sparse,
        "descriptive_applicable": not sparse,
        "next_observation_mse_by_dimension": [
            float(value) for value in np.mean(errors[:, :observation_dim], axis=0)
        ],
        "reward_mse": float(np.mean(errors[:, observation_dim])),
        "continuation_mse": float(np.mean(errors[:, observation_dim + 1])),
        "all_head_mean_squared_error": float(np.mean(errors)),
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and values[order[end]] == values[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool) -> dict[str, object]:
    name = "spearman_rank" if rank else "pearson"
    if left.shape[0] < 2:
        return {"name": name, "available": False, "value": None, "reason": "fewer than two cases"}
    x = _average_ranks(left) if rank else left
    y = _average_ranks(right) if rank else right
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"name": name, "available": False, "value": None, "reason": "constant input"}
    return {
        "name": name,
        "available": True,
        "value": float(np.corrcoef(x, y)[0, 1]),
        "reason": "available",
    }


def _epistemic_summary(
    primitives: Sequence[dict[str, object]],
    config: WorldModelCalibrationConfig,
) -> dict[str, object]:
    available = all(item["disagreement"] is not None for item in primitives)
    if not available:
        return {
            "available": False,
            "reason": "unavailable: snapshot has no member predictions",
            "realized_error_definition": "mean squared error across decoded heads",
            "binning": None,
            "bins": [],
            "correlations": [],
            "coverage_risk_curve": [],
        }
    disagreement = np.asarray(
        [cast(float, item["disagreement"]) for item in primitives],
        dtype=np.float64,
    )
    errors = np.asarray(
        [cast(float, item["realized_mean_squared_error"]) for item in primitives],
        dtype=np.float64,
    )
    n_bins = config.effective_epistemic_bin_count
    if config.epistemic_binning == "equal_count":
        order = np.argsort(disagreement, kind="mergesort")
        assignments = np.zeros(disagreement.shape[0], dtype=np.int64)
        for rank, index in enumerate(order):
            assignments[index] = min((rank * n_bins) // disagreement.shape[0], n_bins - 1)
        binning: dict[str, object] = {
            "method": "equal_count",
            "frozen_bin_count": n_bins,
            "frozen_edges": None,
            "tie_policy": "stable-case-order",
        }
    else:
        edges = np.asarray(config.epistemic_bin_edges, dtype=np.float64)
        assignments = np.digitize(disagreement, edges)
        binning = {
            "method": "frozen_edges",
            "frozen_bin_count": n_bins,
            "frozen_edges": list(config.epistemic_bin_edges),
            "tie_policy": "left-closed-right-open",
        }
    bins: list[dict[str, object]] = []
    for bin_index in range(n_bins):
        selected = np.flatnonzero(assignments == bin_index)
        count = int(selected.shape[0])
        bins.append(
            {
                "bin": bin_index,
                "count": count,
                "sparse": count < config.minimum_descriptive_bin_count,
                "descriptive_applicable": count >= config.minimum_descriptive_bin_count,
                "mean_epistemic_disagreement": (
                    float(np.mean(disagreement[selected])) if count else None
                ),
                "mean_realized_squared_error": (
                    float(np.mean(errors[selected])) if count else None
                ),
            }
        )
    order = np.argsort(disagreement, kind="mergesort")
    coverage: list[dict[str, object]] = []
    for fraction in config.coverage_fractions:
        count = min(
            disagreement.shape[0],
            max(1, int(math.ceil(fraction * disagreement.shape[0]))),
        )
        selected = order[:count]
        coverage.append(
            {
                "requested_coverage": fraction,
                "selected_count": count,
                "realized_coverage": count / disagreement.shape[0],
                "mean_realized_squared_error": float(np.mean(errors[selected])),
            }
        )
    return {
        "available": True,
        "reason": "available: member dispersion and realized errors are present",
        "realized_error_definition": "mean squared error across decoded heads",
        "binning": binning,
        "bins": bins,
        "correlations": [
            _correlation(disagreement, errors, rank=False),
            _correlation(disagreement, errors, rank=True),
        ],
        "coverage_risk_curve": coverage,
    }


def _residual_proxy_summary(
    primitives: Sequence[dict[str, object]],
    *,
    observation_dim: int,
) -> dict[str, object]:
    if not all(cast(bool, item["residual_available"]) for item in primitives):
        return {
            "available": False,
            "ready": False,
            "descriptive_applicable": False,
            "probabilistic_calibration_available": False,
            "interpretation": "unavailable: snapshot has no ensemble residual proxy",
            "raw_head_diagnostics": [],
        }
    ready = all(cast(bool, item["residual_ready"]) for item in primitives)
    proxy = np.stack(
        [cast(np.ndarray, item["residual_values"]) for item in primitives]
    )
    member_raw = np.stack([cast(np.ndarray, item["member_raw"]) for item in primitives])
    raw_target = np.stack([cast(np.ndarray, item["raw_target"]) for item in primitives])
    realized = np.square(member_raw - raw_target[:, None, :])
    names = [f"next_observation_raw_{index}" for index in range(observation_dim)] + [
        "reward_raw",
        "continuation_raw",
    ]
    diagnostics: list[dict[str, object]] = []
    for index, name in enumerate(names):
        mean_proxy = float(np.mean(proxy[:, :, index]))
        mean_realized = float(np.mean(realized[:, :, index]))
        diagnostics.append(
            {
                "head": name,
                "mean_residual_variance_proxy": mean_proxy,
                "mean_realized_member_squared_residual": mean_realized,
                "proxy_to_realized_ratio": (
                    mean_proxy / mean_realized if mean_realized > 0.0 else None
                ),
            }
        )
    return {
        "available": True,
        "ready": ready,
        "descriptive_applicable": ready,
        "probabilistic_calibration_available": False,
        "interpretation": (
            "non-probabilistic residual-variance EMA proxy; no likelihood, interval, "
            "or probabilistic calibration interpretation is available"
        ),
        "raw_head_diagnostics": diagnostics,
    }


def _open_loop_summary(raw_open_loop: object, *, observation_dim: int) -> dict[str, object]:
    rollout = _mapping(raw_open_loop, "raw open_loop")
    if set(rollout) != {"available", "reason", "probes"}:
        raise ValueError("raw open_loop fields do not match v1")
    if not isinstance(rollout.get("available"), bool) or not isinstance(
        rollout.get("reason"), str
    ):
        raise ValueError("open-loop availability and reason are invalid")
    probes = _list(rollout.get("probes"), "raw open_loop.probes")
    if not cast(bool, rollout["available"]):
        if probes:
            raise ValueError("unavailable open-loop trace must contain no probes")
        return {
            "available": False,
            "reason": rollout["reason"],
            "probe_count": 0,
            "prediction_call_count": 0,
            "next_observation_mse_by_dimension": None,
            "reward_mse": None,
            "continuation_mse": None,
        }
    squared: list[np.ndarray] = []
    calls = 0
    for probe_index, probe_value in enumerate(probes):
        probe = _mapping(probe_value, f"open_loop.probes[{probe_index}]")
        if set(probe) != {"probe_id", "initial_observation", "steps"}:
            raise ValueError("open-loop probe trace fields are invalid")
        _validate_identifier("open-loop probe_id", probe.get("probe_id"))
        _numeric_vector(
            probe.get("initial_observation"),
            "open-loop initial observation",
            size=observation_dim,
        )
        steps = _list(probe.get("steps"), "open-loop steps")
        for step_index, step_value in enumerate(steps):
            step = _mapping(step_value, f"open-loop step {step_index}")
            if set(step) != {
                "step",
                "action",
                "input_observation",
                "mean_predictions",
                "grounded_targets",
            }:
                raise ValueError("open-loop step fields are invalid")
            if step.get("step") != step_index:
                raise ValueError("open-loop step indices are noncanonical")
            _numeric_vector(
                step.get("input_observation"),
                "open-loop input observation",
                size=observation_dim,
            )
            prediction = _mapping(step.get("mean_predictions"), "open-loop prediction")
            target = _mapping(step.get("grounded_targets"), "open-loop target")
            expected = {"next_observation", "reward", "continuation"}
            if set(prediction) != expected or set(target) != expected:
                raise ValueError("open-loop prediction/target fields are invalid")
            predicted_vector = np.concatenate(
                (
                    _numeric_vector(
                        prediction.get("next_observation"),
                        "open-loop predicted observation",
                        size=observation_dim,
                    ),
                    np.asarray(
                        [
                            _number(prediction.get("reward"), "open-loop predicted reward"),
                            _number(
                                prediction.get("continuation"),
                                "open-loop predicted continuation",
                            ),
                        ]
                    ),
                )
            )
            target_vector = np.concatenate(
                (
                    _numeric_vector(
                        target.get("next_observation"),
                        "open-loop target observation",
                        size=observation_dim,
                    ),
                    np.asarray(
                        [
                            _number(target.get("reward"), "open-loop target reward"),
                            _number(
                                target.get("continuation"),
                                "open-loop target continuation",
                            ),
                        ]
                    ),
                )
            )
            squared.append(np.square(predicted_vector - target_vector))
            calls += 1
    if not squared:
        raise ValueError("available open-loop trace must contain prediction steps")
    errors = np.stack(squared)
    return {
        "available": True,
        "reason": rollout["reason"],
        "probe_count": len(probes),
        "prediction_call_count": calls,
        "next_observation_mse_by_dimension": [
            float(value) for value in np.mean(errors[:, :observation_dim], axis=0)
        ],
        "reward_mse": float(np.mean(errors[:, observation_dim])),
        "continuation_mse": float(np.mean(errors[:, observation_dim + 1])),
    }


def reconstruct_world_model_calibration_summary(
    raw_trace: Mapping[str, object],
    config: WorldModelCalibrationConfig,
    *,
    observation_dim: int,
    action_regions: tuple[int, ...],
) -> dict[str, object]:
    """Reconstruct every descriptive aggregate from primitive raw trace fields."""
    if set(raw_trace) != {"cases", "open_loop"}:
        raise ValueError("raw trace fields do not match v1")
    case_values = _list(raw_trace.get("cases"), "raw cases")
    if not case_values:
        raise ValueError("raw trace must contain cases")
    primitives = [
        _case_primitives(value, observation_dim=observation_dim)
        for value in case_values
    ]
    all_indices = list(range(len(primitives)))
    overall = _descriptive_group(
        primitives,
        all_indices,
        observation_dim=observation_dim,
        minimum_count=config.minimum_descriptive_bin_count,
    )
    partitions = []
    for name in ("in_distribution", "ood"):
        indices = [
            index for index, item in enumerate(primitives) if item["partition"] == name
        ]
        partitions.append(
            {
                "partition": name,
                **_descriptive_group(
                    primitives,
                    indices,
                    observation_dim=observation_dim,
                    minimum_count=config.minimum_descriptive_bin_count,
                ),
            }
        )
    state_regions = []
    for region in range(len(config.state_norm_edges) + 1):
        indices = [
            index for index, item in enumerate(primitives) if item["state_region"] == region
        ]
        state_regions.append(
            {
                "state_region": region,
                **_descriptive_group(
                    primitives,
                    indices,
                    observation_dim=observation_dim,
                    minimum_count=config.minimum_descriptive_bin_count,
                ),
            }
        )
    action_region_metrics = []
    for region in sorted(set(action_regions)):
        indices = [
            index for index, item in enumerate(primitives) if item["action_region"] == region
        ]
        action_region_metrics.append(
            {
                "action_region": region,
                **_descriptive_group(
                    primitives,
                    indices,
                    observation_dim=observation_dim,
                    minimum_count=config.minimum_descriptive_bin_count,
                ),
            }
        )
    member_available = all(item["member_raw"] is not None for item in primitives)
    residual_summary = _residual_proxy_summary(
        primitives,
        observation_dim=observation_dim,
    )
    return {
        "assessment_status": "not-assessed",
        "thresholds_applied": False,
        "overall_head_squared_error": overall,
        "partition_metrics": partitions,
        "state_region_metrics": state_regions,
        "action_region_metrics": action_region_metrics,
        "epistemic_diagnostics": _epistemic_summary(primitives, config),
        "residual_variance_proxy_diagnostics": residual_summary,
        "open_loop_diagnostics": _open_loop_summary(
            raw_trace.get("open_loop"),
            observation_dim=observation_dim,
        ),
        "applicability": {
            "member_predictions_available": member_available,
            "epistemic_diagnostics_available": member_available,
            "residual_proxy_available": residual_summary["available"],
            "residual_proxy_warmup_complete": residual_summary["ready"],
            "probabilistic_calibration_available": False,
        },
    }


def _resource_accounting(
    snapshot: Mapping[str, object],
    *,
    one_step_calls: int,
    rollout_calls: int,
) -> dict[str, object]:
    ensemble_size = snapshot.get("ensemble_size")
    multiplier = ensemble_size if isinstance(ensemble_size, int) else 1
    total_calls = one_step_calls + rollout_calls
    return {
        "snapshot_state_logical_scalars": snapshot.get("state_logical_scalars"),
        "snapshot_state_bytes": snapshot.get("state_bytes"),
        "one_step_predict_calls": one_step_calls,
        "open_loop_predict_calls": rollout_calls,
        "total_predict_api_calls": total_calls,
        "underlying_member_predict_calls": total_calls * multiplier,
        "learner_update_calls": 0,
        "model_update_calls": 0,
        "regime_identifier_reads": 0,
        "persistent_evaluator_state_bytes": 0,
    }


def build_world_model_calibration_report(
    model: FrozenModel,
    state: FrozenState,
    config: WorldModelCalibrationConfig,
    probes: WorldModelCalibrationProbeSet,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Evaluate a frozen snapshot and return a strict reconstructable report."""
    descriptor_before = _snapshot_descriptor(model, state)
    state_hash_before = frozen_world_model_state_sha256(state)
    raw_trace, one_step_calls, rollout_calls = _evaluate_raw_trace(
        model,
        state,
        config,
        probes,
    )
    state_hash_after = frozen_world_model_state_sha256(state)
    descriptor_after = _snapshot_descriptor(model, state)
    if state_hash_after != state_hash_before or descriptor_after != descriptor_before:
        raise RuntimeError("world-model calibration evaluation mutated the frozen snapshot")
    observation_dim, n_actions = _model_dimensions(model)
    action_regions = _effective_action_regions(config, n_actions)
    summary = reconstruct_world_model_calibration_summary(
        raw_trace,
        config,
        observation_dim=observation_dim,
        action_regions=action_regions,
    )
    sources = world_model_calibration_source_snapshot(root)
    config_payload = config.to_config()
    probe_payload = probes.to_config()
    resources = _resource_accounting(
        descriptor_before,
        one_step_calls=one_step_calls,
        rollout_calls=rollout_calls,
    )
    hashes = {
        "config_sha256": _canonical_sha256(config_payload),
        "source_manifest_sha256": _canonical_sha256(sources),
        "snapshot_sha256": _canonical_sha256(descriptor_before),
        "probe_set_sha256": _canonical_sha256(probe_payload),
        "raw_trace_sha256": _canonical_sha256(raw_trace),
        "summary_sha256": _canonical_sha256(summary),
        "resource_accounting_sha256": _canonical_sha256(resources),
    }
    payload: dict[str, object] = {
        "development_only": True,
        "assessment_status": "not-assessed",
        "scientific_promotion_allowed": False,
        "calibration_claimed": False,
        "performance_thresholds_applied": False,
        "config": config_payload,
        "source_sha256": sources,
        "snapshot": descriptor_before,
        "probe_set": probe_payload,
        "raw_trace": raw_trace,
        "summary": summary,
        "resource_accounting": resources,
        "hashes": hashes,
        "limitations": list(_LIMITATIONS),
    }
    return {
        "schema": WORLD_MODEL_CALIBRATION_REPORT_SCHEMA,
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }


def _validate_probe_trace_binding(
    raw_trace: Mapping[str, object],
    probes: WorldModelCalibrationProbeSet,
    config: WorldModelCalibrationConfig,
    model: FrozenModel,
    *,
    observation_dim: int,
    n_actions: int,
) -> None:
    case_records = _list(raw_trace.get("cases"), "raw cases")
    if len(case_records) != len(probes.cases):
        raise ValueError("raw case count does not match probe set")
    action_regions = _effective_action_regions(config, n_actions)
    for index, (record_value, probe) in enumerate(
        zip(case_records, probes.cases, strict=True)
    ):
        record = _mapping(record_value, f"raw case {index}")
        if record.get("case_id") != probe.case_id:
            raise ValueError(f"raw case {index} identifier does not match probe")
        if record.get("observation") != list(probe.observation):
            raise ValueError(f"raw case {index} observation does not match probe")
        if record.get("action") != probe.action or record.get("partition") != probe.partition:
            raise ValueError(f"raw case {index} action/partition does not match probe")
        targets = _mapping(record.get("targets"), f"raw case {index} targets")
        if (
            targets.get("next_observation") != list(probe.next_observation_target)
            or targets.get("reward") != probe.reward_target
            or targets.get("continuation") != probe.continuation_target
        ):
            raise ValueError(f"raw case {index} decoded targets do not match probe")
        target_model = model.member_model if isinstance(model, WorldModelEnsemble) else model
        expected_raw_target = _python_vector(
            target_model.targets(
                jnp.asarray(probe.observation, dtype=jnp.float32),
                jnp.asarray(probe.reward_target, dtype=jnp.float32),
                jnp.asarray(probe.continuation_target, dtype=jnp.float32),
                jnp.asarray(probe.next_observation_target, dtype=jnp.float32),
            )
        )
        if targets.get("raw") != expected_raw_target:
            raise ValueError(f"raw case {index} normalized targets do not reconstruct")
        expected_state_region = int(
            np.digitize(
                np.asarray(
                    [np.linalg.norm(np.asarray(probe.observation, dtype=np.float64))]
                ),
                np.asarray(config.state_norm_edges, dtype=np.float64),
            )[0]
        )
        if (
            record.get("state_region") != expected_state_region
            or record.get("action_region") != action_regions[probe.action]
        ):
            raise ValueError(f"raw case {index} region bins do not reconstruct")
    _validate_probe_dimensions(
        probes,
        observation_dim=observation_dim,
        n_actions=n_actions,
        config=config,
    )


def validate_world_model_calibration_report(
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
    model: FrozenModel | None = None,
    state: FrozenState | None = None,
    probes: WorldModelCalibrationProbeSet | None = None,
) -> WorldModelCalibrationValidation:
    """Fail closed on schema, hashes, reconstruction, sources, or snapshot replay."""
    errors: list[str] = []
    if set(report) != {"schema", "payload", "payload_sha256"}:
        errors.append("report top-level fields do not match v1")
    if report.get("schema") != WORLD_MODEL_CALIBRATION_REPORT_SCHEMA:
        errors.append("world-model calibration report schema is unsupported")
    payload_value = report.get("payload")
    if not isinstance(payload_value, Mapping):
        errors.append("report payload must be an object")
        payload: Mapping[str, object] = {}
    else:
        payload = cast(Mapping[str, object], payload_value)
    expected_payload = {
        "development_only",
        "assessment_status",
        "scientific_promotion_allowed",
        "calibration_claimed",
        "performance_thresholds_applied",
        "config",
        "source_sha256",
        "snapshot",
        "probe_set",
        "raw_trace",
        "summary",
        "resource_accounting",
        "hashes",
        "limitations",
    }
    if set(payload) != expected_payload:
        errors.append("report payload fields do not match v1")
    if payload.get("development_only") is not True:
        errors.append("report must remain development-only")
    if payload.get("assessment_status") != "not-assessed":
        errors.append("report assessment status must remain not-assessed")
    if payload.get("scientific_promotion_allowed") is not False:
        errors.append("report must forbid scientific promotion")
    if payload.get("calibration_claimed") is not False:
        errors.append("report cannot claim calibration")
    if payload.get("performance_thresholds_applied") is not False:
        errors.append("report cannot apply performance thresholds")
    if payload.get("limitations") != list(_LIMITATIONS):
        errors.append("report limitations are not the exact v1 text")
    try:
        if report.get("payload_sha256") != _canonical_sha256(payload):
            errors.append("report payload digest does not match")
    except (TypeError, ValueError) as error:
        errors.append(f"report payload is not canonical JSON: {error}")

    config: WorldModelCalibrationConfig | None = None
    config_value = payload.get("config")
    if not isinstance(config_value, Mapping):
        errors.append("report config must be an object")
    else:
        try:
            config = WorldModelCalibrationConfig.from_config(config_value)
        except (TypeError, ValueError) as error:
            errors.append(f"report config is invalid: {error}")
        else:
            if config.to_config() != config_value:
                errors.append("report config is noncanonical")

    source_value = payload.get("source_sha256")
    try:
        current_sources = world_model_calibration_source_snapshot(root)
    except OSError as error:
        errors.append(f"cannot hash calibration evaluator sources: {error}")
        current_sources = {}
    if source_value != current_sources:
        errors.append("report source hashes do not match current pinned sources")

    snapshot_value = payload.get("snapshot")
    report_model: FrozenModel | None = None
    observation_dim: int | None = None
    n_actions: int | None = None
    if not isinstance(snapshot_value, Mapping):
        errors.append("report snapshot descriptor must be an object")
        snapshot: Mapping[str, object] = {}
    else:
        snapshot = cast(Mapping[str, object], snapshot_value)
        expected_snapshot = {
            "model_kind",
            "model_config",
            "model_config_sha256",
            "state_sha256",
            "state_logical_scalars",
            "state_bytes",
            "ensemble_size_available",
            "ensemble_size",
            "real_event_count",
            "residual_proxy_warmup_steps_available",
            "residual_proxy_warmup_steps",
        }
        if set(snapshot) != expected_snapshot:
            errors.append("snapshot descriptor fields do not match v1")
        model_config = snapshot.get("model_config")
        if not isinstance(model_config, Mapping):
            errors.append("snapshot model_config must be an object")
        else:
            try:
                model_config_digest = _canonical_sha256(model_config)
            except (TypeError, ValueError) as error:
                errors.append(f"snapshot model_config is not canonical JSON: {error}")
                model_config_digest = None
            if snapshot.get("model_config_sha256") != model_config_digest:
                errors.append("snapshot model config digest does not match")
            try:
                report_model = _model_from_snapshot_config(
                    snapshot.get("model_kind"),
                    model_config,
                )
            except (TypeError, ValueError) as error:
                errors.append(f"snapshot model config is invalid: {error}")
            else:
                observation_dim, n_actions = _model_dimensions(report_model)
        for field in ("state_logical_scalars", "state_bytes", "real_event_count"):
            value = snapshot.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                errors.append(f"snapshot {field} must be a non-negative integer")

    probe_value = payload.get("probe_set")
    reconstructed_probes: WorldModelCalibrationProbeSet | None = None
    if not isinstance(probe_value, Mapping):
        errors.append("report probe_set must be an object")
    else:
        try:
            reconstructed_probes = WorldModelCalibrationProbeSet.from_config(probe_value)
        except (TypeError, ValueError) as error:
            errors.append(f"report probe_set is invalid: {error}")
        else:
            if reconstructed_probes.to_config() != probe_value:
                errors.append("report probe_set is noncanonical")

    raw_value = payload.get("raw_trace")
    summary_value = payload.get("summary")
    resources_value = payload.get("resource_accounting")
    reconstructed_summary: dict[str, object] | None = None
    if (
        isinstance(raw_value, Mapping)
        and config is not None
        and observation_dim is not None
        and n_actions is not None
        and report_model is not None
        and reconstructed_probes is not None
    ):
        try:
            action_regions = _effective_action_regions(config, n_actions)
            _validate_probe_trace_binding(
                raw_value,
                reconstructed_probes,
                config,
                report_model,
                observation_dim=observation_dim,
                n_actions=n_actions,
            )
            reconstructed_summary = reconstruct_world_model_calibration_summary(
                raw_value,
                config,
                observation_dim=observation_dim,
                action_regions=action_regions,
            )
        except (TypeError, ValueError) as error:
            errors.append(f"raw trace cannot be reconstructed: {error}")
        else:
            if _canonical_json_bytes(reconstructed_summary) != _canonical_json_bytes(
                summary_value
            ):
                errors.append("summary does not reconstruct exactly from raw trace")
            open_loop = cast(
                Mapping[str, object],
                reconstructed_summary["open_loop_diagnostics"],
            )
            expected_resources = _resource_accounting(
                snapshot,
                one_step_calls=len(reconstructed_probes.cases),
                rollout_calls=cast(int, open_loop["prediction_call_count"]),
            )
            if expected_resources != resources_value:
                errors.append("resource accounting does not reconstruct exactly")
    else:
        if not isinstance(raw_value, Mapping):
            errors.append("raw_trace must be an object")
        if not isinstance(summary_value, Mapping):
            errors.append("summary must be an object")
        if not isinstance(resources_value, Mapping):
            errors.append("resource_accounting must be an object")

    hashes_value = payload.get("hashes")
    expected_hash_fields = {
        "config_sha256",
        "source_manifest_sha256",
        "snapshot_sha256",
        "probe_set_sha256",
        "raw_trace_sha256",
        "summary_sha256",
        "resource_accounting_sha256",
    }
    if not isinstance(hashes_value, Mapping) or set(hashes_value) != expected_hash_fields:
        errors.append("report hash fields do not match v1")
    else:
        try:
            expected_hashes = {
                "config_sha256": _canonical_sha256(config_value),
                "source_manifest_sha256": _canonical_sha256(source_value),
                "snapshot_sha256": _canonical_sha256(snapshot_value),
                "probe_set_sha256": _canonical_sha256(probe_value),
                "raw_trace_sha256": _canonical_sha256(raw_value),
                "summary_sha256": _canonical_sha256(summary_value),
                "resource_accounting_sha256": _canonical_sha256(resources_value),
            }
        except (TypeError, ValueError) as error:
            errors.append(f"report components are not canonical JSON: {error}")
        else:
            if dict(hashes_value) != expected_hashes:
                errors.append("one or more canonical component hashes do not match")

    optional_values = (model, state, probes)
    if any(value is not None for value in optional_values) and not all(
        value is not None for value in optional_values
    ):
        errors.append("model, state, and probes must be supplied together for replay validation")
    elif model is not None and state is not None and probes is not None and config is not None:
        state_hash_before = frozen_world_model_state_sha256(state)
        try:
            live_snapshot = _snapshot_descriptor(model, state)
            live_raw, live_one_step_calls, live_rollout_calls = _evaluate_raw_trace(
                model,
                state,
                config,
                probes,
            )
        except (TypeError, ValueError) as error:
            errors.append(f"snapshot replay validation failed: {error}")
        else:
            if frozen_world_model_state_sha256(state) != state_hash_before:
                errors.append("snapshot replay validation mutated the supplied state")
            if live_snapshot != snapshot_value:
                errors.append("supplied snapshot does not match report snapshot hash/resources")
            if probes.to_config() != probe_value:
                errors.append("supplied probes do not match report probe hash")
            if _canonical_json_bytes(live_raw) != _canonical_json_bytes(raw_value):
                errors.append("raw predictions do not replay exactly from supplied snapshot")
            if resources_value != _resource_accounting(
                live_snapshot,
                one_step_calls=live_one_step_calls,
                rollout_calls=live_rollout_calls,
            ):
                errors.append("supplied snapshot replay resource accounting differs")

    return WorldModelCalibrationValidation(
        valid=not errors,
        assessment_status="not-assessed",
        errors=tuple(errors),
    )


def canonical_world_model_calibration_report_bytes(
    report: Mapping[str, object],
) -> bytes:
    """Return the only accepted on-disk report encoding."""
    return _canonical_json_bytes(report) + b"\n"


def _strict_json_object(data: bytes) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    parsed = json.loads(
        data.decode("utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(parsed, dict):
        raise ValueError("world-model calibration report must be a JSON object")
    return parsed


def load_world_model_calibration_report(
    path: str | Path,
    *,
    root: Path = REPO_ROOT,
) -> dict[str, object]:
    """Load only exact canonical JSON and require full structural validity."""
    data = Path(path).read_bytes()
    report = _strict_json_object(data)
    if data != canonical_world_model_calibration_report_bytes(report):
        raise ValueError("world-model calibration report is not exact canonical JSON")
    validation = validate_world_model_calibration_report(report, root=root)
    if not validation.valid:
        raise ValueError("invalid world-model calibration report: " + "; ".join(validation.errors))
    return report


def save_world_model_calibration_report(
    path: str | Path,
    report: Mapping[str, object],
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Validate and atomically create one canonical report without overwrite."""
    validation = validate_world_model_calibration_report(report, root=root)
    if not validation.valid:
        raise ValueError(
            "refusing to save invalid calibration report: "
            + "; ".join(validation.errors)
        )
    expanded = Path(path).expanduser()
    destination = expanded.resolve()
    if os.path.lexists(expanded) or os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite calibration report: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_world_model_calibration_report_bytes(report))
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite concurrently created calibration report: {destination}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "DEVELOPMENT_STATUS",
    "REPO_ROOT",
    "SOURCE_PATHS",
    "WORLD_MODEL_CALIBRATION_CHECKPOINT_SCHEMA",
    "WORLD_MODEL_CALIBRATION_CONFIG_SCHEMA",
    "WORLD_MODEL_CALIBRATION_PROBE_SCHEMA",
    "WORLD_MODEL_CALIBRATION_REPORT_SCHEMA",
    "WorldModelCalibrationCase",
    "WorldModelCalibrationConfig",
    "WorldModelCalibrationProbeSet",
    "WorldModelCalibrationValidation",
    "WorldModelOpenLoopProbe",
    "build_world_model_calibration_report",
    "canonical_world_model_calibration_report_bytes",
    "frozen_world_model_snapshot_sha256",
    "frozen_world_model_state_sha256",
    "load_world_model_calibration_report",
    "load_world_model_calibration_snapshot_checkpoint",
    "reconstruct_world_model_calibration_summary",
    "save_world_model_calibration_report",
    "save_world_model_calibration_snapshot_checkpoint",
    "validate_world_model_calibration_report",
    "world_model_calibration_source_snapshot",
]
