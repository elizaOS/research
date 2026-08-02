"""Privileged continuing-control references kept outside matched conditions.

This development-only companion executes three deliberately privileged control
references on independent environment copies.  Regime identifiers are consumed
only by suite-owned routing and evaluator callbacks; they are never arguments to
``ContinuingControlLearner`` methods.  Every environment action is armed before
its outcome and bound to an exact wrapper-owned decision identifier.

The resulting reports are descriptive ``not-assessed`` artifacts.  Privileged
references are never inserted into the candidate/baseline condition list and
cannot promote evidence or establish matched resource parity.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn, Protocol, TypeGuard, cast, runtime_checkable

import numpy as np

import alberta_framework.evaluation.continual_control_evaluator as control_core
from alberta_framework.evaluation.continual_control_evaluator import (
    ContinuingControlBudget,
    ContinuingControlEnvironment,
    ContinuingControlLearner,
    ContinuingControlProtocol,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlLearnerUpdate,
    ControlProbe,
    ControlResourceUsage,
    ControlTransition,
)

REFERENCE_SUITE_SCHEMA = "alberta.privileged_control_reference_suite.v1"
REFERENCE_REPORT_SCHEMA = "alberta.privileged_control_reference_report.v1"
REFERENCE_CHECKPOINT_SCHEMA = "alberta.privileged_control_reference_checkpoint.v1"
REFERENCE_RUN_CONFIG_SCHEMA = "alberta.privileged_control_reference_run_config.v1"
REFERENCE_EXTRA_BUDGET_SCHEMA = "alberta.privileged_control_reference_extra_budget.v1"
STATIONARY_STREAM_SCHEMA = "alberta.stationary_control_reference_stream.v1"
STATIONARY_EXAMPLE_SCHEMA = "alberta.stationary_control_reference_example.v1"
ORACLE_SOURCE_SCHEMA = "alberta.frozen_exact_control_oracle_action_scores.v1"

RETAINED_FRESH_PER_REGIME_ROLE = "retained_fresh_once_per_regime_identity"
STATIONARY_MULTITASK_ROLE = "stationary_multitask"
ORACLE_ACTION_DATA_ROLE = "exact_frozen_oracle_action_data_upper"
EXACT_ORACLE_SCORE_SEMANTICS = "exact_frozen_counterfactual_outcome_by_action"
EXACT_ORACLE_CALLBACK_TEMPORAL_CONTRACT = (
    "exact frozen counterfactual outcome scores returned before outcome"
)
REFERENCE_ROLES = (
    RETAINED_FRESH_PER_REGIME_ROLE,
    STATIONARY_MULTITASK_ROLE,
    ORACLE_ACTION_DATA_ROLE,
)

ACCEPTANCE_STATUS = "not-assessed"
REPORT_INTERPRETATION = (
    "Development-only privileged continual-control references. These are descriptive "
    "context bounds, not matched baselines, thresholds, scientific evidence, or an "
    "Alberta Plan completion claim."
)
REFERENCE_LIMITATIONS = (
    "development-only; the configured seed is consumed and is not a promotion seed",
    "all three references receive privileges unavailable to ordinary matched conditions",
    "the retained fresh-per-regime-identity reference initializes once per evaluator regime "
    "identity and reuses that trained state on recurrence; it is not fresh per segment or reset "
    "at each regime change",
    "retained fresh-per-regime-identity recurrence can be unavailable when its learner state "
    "cannot own the recurrence observation without an undisclosed reset",
    "stationary-multitask training consumes the frozen additional stream and budget reported",
    "the exact-oracle role accepts only a source declaring exact frozen counterfactual outcomes "
    "for every action, not stochastic expected scores; only the selected score can be checked "
    "against the realized outcome",
    "callback configuration cannot audit hidden external side effects",
    "held-out probes score one selected action rather than a frozen-policy rollout",
    "no thresholds, hypothesis tests, evidence promotion, or default-mechanism selection",
)

FRESH_UNAVAILABLE_REASON_PREFIX = (
    "retained fresh-per-regime-identity state cannot own recurrence observation for evaluator "
    "regime "
)
STATIONARY_UNAVAILABLE_REASON = (
    "stationary-multitask pretrained state cannot own the common protocol initial observation"
)

type DecisionId = tuple[int, int, int, int]


def _is_finite_number(value: object) -> TypeGuard[int | float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def _finite_float(value: object, *, name: str) -> float:
    if not _is_finite_number(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _nonnegative_float(value: object, *, name: str) -> float:
    resolved = _finite_float(value, name=name)
    if resolved < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return resolved


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    resolved = _nonnegative_int(value, name=name)
    if resolved == 0:
        raise ValueError(f"{name} must be positive")
    return resolved


def _uint32(value: object, *, name: str) -> int:
    resolved = _nonnegative_int(value, name=name)
    if resolved > np.iinfo(np.uint32).max:
        raise ValueError(f"{name} must fit in uint32")
    return resolved


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _versioned_identifier(value: object, *, name: str) -> str:
    resolved = _string(value, name=name)
    if ".v" not in resolved:
        raise ValueError(f"{name} must be a versioned identifier")
    return resolved


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _exact_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(mapping) != expected:
        raise ValueError(f"{name} keys are invalid")


def _observation(value: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError(f"{name} must be a non-empty immutable tuple")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _observation_from_json(value: object, *, name: str) -> tuple[float, ...]:
    raw = _list(value, name=name)
    if not raw:
        raise ValueError(f"{name} must be non-empty")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(raw))


def _decision_id(value: object, *, name: str) -> DecisionId:
    if not isinstance(value, tuple) or len(value) != 4:
        raise ValueError(f"{name} must contain four uint32 words")
    return cast(
        DecisionId,
        tuple(_uint32(item, name=f"{name}[{index}]") for index, item in enumerate(value)),
    )


def _decision_id_from_json(value: object, *, name: str) -> DecisionId:
    raw = _list(value, name=name)
    if len(raw) != 4:
        raise ValueError(f"{name} must contain four uint32 words")
    return cast(
        DecisionId,
        tuple(_uint32(item, name=f"{name}[{index}]") for index, item in enumerate(raw)),
    )


def _lifecycle_id(value: object, *, name: str) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{name} must contain two uint32 words")
    return (
        _uint32(value[0], name=f"{name}[0]"),
        _uint32(value[1], name=f"{name}[1]"),
    )


def _generation_decision_id(lifecycle_id: tuple[int, int], generation: int) -> DecisionId:
    resolved = _nonnegative_int(generation, name="decision generation")
    if resolved >= 1 << 64:
        raise ValueError("decision generation is exhausted")
    return (
        lifecycle_id[0],
        lifecycle_id[1],
        (resolved >> 32) & np.iinfo(np.uint32).max,
        resolved & np.iinfo(np.uint32).max,
    )


def _json_clone(value: object) -> object:
    return json.loads(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    resolved = _string(value, name=name)
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return resolved


def _source_core_hashes() -> dict[str, str]:
    paths = {
        "alberta_framework/evaluation/continual_control_reference_suite.py": Path(
            __file__
        ).resolve(),
        "alberta_framework/evaluation/continual_control_evaluator.py": Path(
            control_core.__file__
        ).resolve(),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _assert_explicit_seed_fields(value: object, *, seed: int, name: str) -> int:
    found = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} config keys must be strings")
            if key in {"seed", "campaign_seed", "factory_seed"}:
                if _uint32(item, name=f"{name}.{key}") != seed:
                    raise ValueError(f"{name}.{key} does not match the suite seed")
                found += 1
            found += _assert_explicit_seed_fields(item, seed=seed, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found += _assert_explicit_seed_fields(item, seed=seed, name=f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} contains a non-finite float")
    elif value is not None and not isinstance(value, bool | int | float | str):
        raise ValueError(f"{name} contains a non-JSON value")
    return found


@dataclasses.dataclass(frozen=True)
class StationaryReferenceExample:
    """One frozen evaluator-owned extra-stream outcome table."""

    reference_regime_id: str
    observation: tuple[float, ...]
    action_scores: tuple[float, ...]
    discount: float
    terminated: bool
    truncated: bool
    bootstrap_observation: tuple[float, ...]
    reset_observation: tuple[float, ...] | None
    schema_version: str = STATIONARY_EXAMPLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATIONARY_EXAMPLE_SCHEMA:
            raise ValueError("stationary example schema_version is invalid")
        _string(self.reference_regime_id, name="reference_regime_id")
        source = _observation(self.observation, name="stationary example observation")
        if not isinstance(self.action_scores, tuple) or not self.action_scores:
            raise ValueError("stationary example action_scores must be a non-empty tuple")
        for index, score in enumerate(self.action_scores):
            _finite_float(score, name=f"stationary example action_scores[{index}]")
        discount = _finite_float(self.discount, name="stationary example discount")
        if not 0.0 <= discount <= 1.0:
            raise ValueError("stationary example discount must lie in [0, 1]")
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise ValueError("stationary example boundary flags must be boolean")
        bootstrap = _observation(
            self.bootstrap_observation,
            name="stationary example bootstrap_observation",
        )
        if len(source) != len(bootstrap):
            raise ValueError("stationary example observation dimensions must match")
        boundary = self.terminated or self.truncated
        if (discount == 0.0) != self.terminated:
            raise ValueError("zero stationary discount must exactly identify termination")
        if boundary:
            if self.reset_observation is None:
                raise ValueError("stationary boundary requires reset_observation")
            reset = _observation(
                self.reset_observation,
                name="stationary example reset_observation",
            )
            if len(reset) != len(source):
                raise ValueError("stationary reset observation dimension must match")
        elif self.reset_observation is not None:
            raise ValueError("stationary reset_observation must be None away from a boundary")

    @property
    def next_decision_observation(self) -> tuple[float, ...]:
        return (
            self.bootstrap_observation if self.reset_observation is None else self.reset_observation
        )

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(cls, value: object) -> StationaryReferenceExample:
        mapping = _mapping(value, name="stationary example")
        _exact_keys(
            mapping,
            {field.name for field in dataclasses.fields(cls)},
            name="stationary example",
        )
        reset_raw = mapping["reset_observation"]
        example = cls(
            reference_regime_id=_string(
                mapping["reference_regime_id"],
                name="stationary example.reference_regime_id",
            ),
            observation=_observation_from_json(
                mapping["observation"],
                name="stationary example.observation",
            ),
            action_scores=tuple(
                _finite_float(item, name="stationary example.action_scores")
                for item in _list(
                    mapping["action_scores"],
                    name="stationary example.action_scores",
                )
            ),
            discount=_finite_float(
                mapping["discount"],
                name="stationary example.discount",
            ),
            terminated=_boolean(
                mapping["terminated"],
                name="stationary example.terminated",
            ),
            truncated=_boolean(
                mapping["truncated"],
                name="stationary example.truncated",
            ),
            bootstrap_observation=_observation_from_json(
                mapping["bootstrap_observation"],
                name="stationary example.bootstrap_observation",
            ),
            reset_observation=(
                None
                if reset_raw is None
                else _observation_from_json(
                    reset_raw,
                    name="stationary example.reset_observation",
                )
            ),
            schema_version=_string(
                mapping["schema_version"],
                name="stationary example.schema_version",
            ),
        )
        if _canonical_json(example.to_config()) != _canonical_json(mapping):
            raise ValueError("stationary example is noncanonical")
        return example


@dataclasses.dataclass(frozen=True)
class FrozenStationaryReferenceStream:
    """Frozen stationary-multitask extra stream with evaluator-only labels."""

    stream_id: str
    seed: int
    examples: tuple[StationaryReferenceExample, ...]
    schema_version: str = STATIONARY_STREAM_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STATIONARY_STREAM_SCHEMA:
            raise ValueError("stationary stream schema_version is invalid")
        _versioned_identifier(self.stream_id, name="stream_id")
        _uint32(self.seed, name="stationary stream seed")
        if not isinstance(self.examples, tuple) or len(self.examples) < 2:
            raise ValueError("stationary stream requires at least two immutable examples")
        if any(not isinstance(example, StationaryReferenceExample) for example in self.examples):
            raise TypeError("stationary stream examples have an invalid type")
        for index, (left, right) in enumerate(zip(self.examples, self.examples[1:], strict=False)):
            if left.reference_regime_id == right.reference_regime_id:
                raise ValueError("stationary stream regime labels must be interleaved")
            if left.next_decision_observation != right.observation:
                raise ValueError(f"stationary stream is discontinuous after example {index}")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(cls, value: object) -> FrozenStationaryReferenceStream:
        mapping = _mapping(value, name="stationary stream")
        _exact_keys(
            mapping,
            {"stream_id", "seed", "examples", "schema_version"},
            name="stationary stream",
        )
        stream = cls(
            stream_id=_versioned_identifier(
                mapping["stream_id"],
                name="stationary stream.stream_id",
            ),
            seed=_uint32(mapping["seed"], name="stationary stream.seed"),
            examples=tuple(
                StationaryReferenceExample.from_config(item)
                for item in _list(mapping["examples"], name="stationary stream.examples")
            ),
            schema_version=_string(
                mapping["schema_version"],
                name="stationary stream.schema_version",
            ),
        )
        if _canonical_json(stream.to_config()) != _canonical_json(mapping):
            raise ValueError("stationary stream is noncanonical")
        return stream


@dataclasses.dataclass(frozen=True)
class PrivilegedReferenceExtraDataBudget:
    stationary_transition_limit: int
    stationary_decision_call_limit: int
    stationary_update_call_limit: int
    stationary_backward_call_limit: int
    stationary_reward_table_scalar_limit: int
    oracle_callback_limit: int
    oracle_action_score_scalar_limit: int
    oracle_probe_action_score_scalar_limit: int
    schema_version: str = REFERENCE_EXTRA_BUDGET_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_EXTRA_BUDGET_SCHEMA:
            raise ValueError("reference extra-data budget schema_version is invalid")
        for field in dataclasses.fields(self):
            if field.name != "schema_version":
                _nonnegative_int(getattr(self, field.name), name=field.name)

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(cls, value: object) -> PrivilegedReferenceExtraDataBudget:
        mapping = _mapping(value, name="reference extra-data budget")
        _exact_keys(
            mapping,
            {field.name for field in dataclasses.fields(cls)},
            name="reference extra-data budget",
        )
        budget = cls(
            stationary_transition_limit=_nonnegative_int(
                mapping["stationary_transition_limit"],
                name="stationary_transition_limit",
            ),
            stationary_decision_call_limit=_nonnegative_int(
                mapping["stationary_decision_call_limit"],
                name="stationary_decision_call_limit",
            ),
            stationary_update_call_limit=_nonnegative_int(
                mapping["stationary_update_call_limit"],
                name="stationary_update_call_limit",
            ),
            stationary_backward_call_limit=_nonnegative_int(
                mapping["stationary_backward_call_limit"],
                name="stationary_backward_call_limit",
            ),
            stationary_reward_table_scalar_limit=_nonnegative_int(
                mapping["stationary_reward_table_scalar_limit"],
                name="stationary_reward_table_scalar_limit",
            ),
            oracle_callback_limit=_nonnegative_int(
                mapping["oracle_callback_limit"],
                name="oracle_callback_limit",
            ),
            oracle_action_score_scalar_limit=_nonnegative_int(
                mapping["oracle_action_score_scalar_limit"],
                name="oracle_action_score_scalar_limit",
            ),
            oracle_probe_action_score_scalar_limit=_nonnegative_int(
                mapping["oracle_probe_action_score_scalar_limit"],
                name="oracle_probe_action_score_scalar_limit",
            ),
            schema_version=_string(mapping["schema_version"], name="budget.schema_version"),
        )
        if _canonical_json(budget.to_config()) != _canonical_json(mapping):
            raise ValueError("reference extra-data budget is noncanonical")
        return budget


@dataclasses.dataclass(frozen=True)
class PrivilegedReferenceRunConfig:
    suite_id: str
    seed: int
    fresh_lifecycle_id: tuple[int, int]
    stationary_lifecycle_id: tuple[int, int]
    oracle_lifecycle_id: tuple[int, int]
    extra_data_budget: PrivilegedReferenceExtraDataBudget
    schema_version: str = REFERENCE_RUN_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REFERENCE_RUN_CONFIG_SCHEMA:
            raise ValueError("reference run config schema_version is invalid")
        _versioned_identifier(self.suite_id, name="suite_id")
        _uint32(self.seed, name="suite seed")
        lifecycle_ids = (
            _lifecycle_id(self.fresh_lifecycle_id, name="fresh_lifecycle_id"),
            _lifecycle_id(self.stationary_lifecycle_id, name="stationary_lifecycle_id"),
            _lifecycle_id(self.oracle_lifecycle_id, name="oracle_lifecycle_id"),
        )
        if len(set(lifecycle_ids)) != len(lifecycle_ids):
            raise ValueError("reference wrapper lifecycle IDs must be unique")
        if not isinstance(self.extra_data_budget, PrivilegedReferenceExtraDataBudget):
            raise TypeError("extra_data_budget has an invalid type")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(cls, value: object) -> PrivilegedReferenceRunConfig:
        mapping = _mapping(value, name="reference run config")
        _exact_keys(
            mapping,
            {
                "suite_id",
                "seed",
                "fresh_lifecycle_id",
                "stationary_lifecycle_id",
                "oracle_lifecycle_id",
                "extra_data_budget",
                "schema_version",
            },
            name="reference run config",
        )

        def lifecycle(field: str) -> tuple[int, int]:
            raw = _list(mapping[field], name=f"reference run config.{field}")
            if len(raw) != 2:
                raise ValueError(f"reference run config.{field} must have two words")
            return (
                _uint32(raw[0], name=f"reference run config.{field}[0]"),
                _uint32(raw[1], name=f"reference run config.{field}[1]"),
            )

        config = cls(
            suite_id=_versioned_identifier(
                mapping["suite_id"],
                name="reference run config.suite_id",
            ),
            seed=_uint32(mapping["seed"], name="reference run config.seed"),
            fresh_lifecycle_id=lifecycle("fresh_lifecycle_id"),
            stationary_lifecycle_id=lifecycle("stationary_lifecycle_id"),
            oracle_lifecycle_id=lifecycle("oracle_lifecycle_id"),
            extra_data_budget=PrivilegedReferenceExtraDataBudget.from_config(
                mapping["extra_data_budget"]
            ),
            schema_version=_string(
                mapping["schema_version"],
                name="reference run config.schema_version",
            ),
        )
        if _canonical_json(config.to_config()) != _canonical_json(mapping):
            raise ValueError("reference run config is noncanonical")
        return config


class ReferenceEnvironmentFactory(Protocol):
    def __call__(self, seed: int) -> ContinuingControlEnvironment: ...


class RetainedRegimeIdentityLearnerFactory(Protocol):
    def __call__(self, seed: int, evaluator_regime_id: str) -> ContinuingControlLearner: ...


class StationaryReferenceLearnerFactory(Protocol):
    def __call__(self, seed: int) -> ContinuingControlLearner: ...


@runtime_checkable
class FrozenExactOracleOutcomeSource(Protocol):
    def to_config(self) -> dict[str, object]: ...

    def action_scores(
        self,
        observation: tuple[float, ...],
        *,
        evaluator_regime_id: str,
        step: int,
    ) -> tuple[float, ...]: ...


class ExactOracleOutcomeSourceFactory(Protocol):
    def __call__(self, seed: int) -> FrozenExactOracleOutcomeSource: ...


@dataclasses.dataclass(frozen=True)
class PrivilegedReferenceValidation:
    valid: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _PendingLearningDecision:
    evaluator_regime_id: str
    outer: ControlDecision
    inner: ControlDecision


@dataclasses.dataclass(frozen=True)
class _ReferenceActionRecord:
    evaluator_regime_id: str
    transition: ControlTransition
    inner_decision_id: DecisionId | None
    oracle_action_scores: tuple[float, ...] | None


@dataclasses.dataclass(frozen=True)
class _ExtraTrainingRecord:
    reference_regime_id: str
    transition: ControlTransition
    backward_call_count: int | None


@dataclasses.dataclass(frozen=True)
class _ReferenceTrace:
    actions: tuple[_ReferenceActionRecord, ...]
    evaluator_matrix: tuple[tuple[float, ...], ...]
    backward_call_counts: tuple[int | None, ...]


@dataclasses.dataclass(frozen=True)
class _FreshReferenceState:
    available: bool
    unavailable_reason: str | None
    environment_state: Any
    current_observation: tuple[float, ...]
    learner_states: tuple[Any | None, ...]
    pending: _PendingLearningDecision | None
    trace: _ReferenceTrace
    learner_initialization_count: int
    ephemeral_probe_initialization_count: int


@dataclasses.dataclass(frozen=True)
class _StationaryReferenceState:
    available: bool
    unavailable_reason: str | None
    environment_state: Any
    current_observation: tuple[float, ...]
    learner_state: Any
    pending: _PendingLearningDecision | None
    trace: _ReferenceTrace
    extra_training_trace: tuple[_ExtraTrainingRecord, ...]


@dataclasses.dataclass(frozen=True)
class _OracleReferenceState:
    available: bool
    unavailable_reason: str | None
    environment_state: Any
    current_observation: tuple[float, ...]
    pending: ControlDecision | None
    pending_action_scores: tuple[float, ...] | None
    trace: _ReferenceTrace


@dataclasses.dataclass(frozen=True)
class PrivilegedReferenceRunState:
    """Checkpointable state after complete transactions for every active role."""

    step: int
    fresh_per_regime: _FreshReferenceState
    stationary_multitask: _StationaryReferenceState
    oracle_action_data_upper: _OracleReferenceState


def _decision_to_config(decision: ControlDecision) -> dict[str, object]:
    return decision.to_config()


def _decision_from_config(value: object, *, name: str) -> ControlDecision:
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {"observation", "action", "decision_id", "armed"},
        name=name,
    )
    decision = ControlDecision(
        observation=_observation_from_json(
            mapping["observation"],
            name=f"{name}.observation",
        ),
        action=_nonnegative_int(mapping["action"], name=f"{name}.action"),
        decision_id=_decision_id_from_json(
            mapping["decision_id"],
            name=f"{name}.decision_id",
        ),
        armed=_boolean(mapping["armed"], name=f"{name}.armed"),
    )
    if _canonical_json(decision.to_config()) != _canonical_json(mapping):
        raise ValueError(f"{name} is noncanonical")
    return decision


def _transition_to_config(transition: ControlTransition) -> dict[str, object]:
    return {
        "observation": list(transition.observation),
        "action": transition.action,
        "decision_id": list(transition.decision_id),
        "reward": transition.reward,
        "discount": transition.discount,
        "terminated": transition.terminated,
        "truncated": transition.truncated,
        "bootstrap_observation": list(transition.bootstrap_observation),
        "reset_observation": (
            None if transition.reset_observation is None else list(transition.reset_observation)
        ),
        "safety_violation": transition.safety_violation,
        "intervention": transition.intervention,
        "near_miss": transition.near_miss,
        "safety_cost": transition.safety_cost,
        "near_miss_cost": transition.near_miss_cost,
    }


def _transition_from_config(value: object, *, name: str) -> ControlTransition:
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {
            "observation",
            "action",
            "decision_id",
            "reward",
            "discount",
            "terminated",
            "truncated",
            "bootstrap_observation",
            "reset_observation",
            "safety_violation",
            "intervention",
            "near_miss",
            "safety_cost",
            "near_miss_cost",
        },
        name=name,
    )
    reset_raw = mapping["reset_observation"]
    transition = ControlTransition(
        observation=_observation_from_json(
            mapping["observation"],
            name=f"{name}.observation",
        ),
        action=_nonnegative_int(mapping["action"], name=f"{name}.action"),
        decision_id=_decision_id_from_json(
            mapping["decision_id"],
            name=f"{name}.decision_id",
        ),
        reward=_finite_float(mapping["reward"], name=f"{name}.reward"),
        discount=_finite_float(mapping["discount"], name=f"{name}.discount"),
        terminated=_boolean(mapping["terminated"], name=f"{name}.terminated"),
        truncated=_boolean(mapping["truncated"], name=f"{name}.truncated"),
        bootstrap_observation=_observation_from_json(
            mapping["bootstrap_observation"],
            name=f"{name}.bootstrap_observation",
        ),
        reset_observation=(
            None
            if reset_raw is None
            else _observation_from_json(
                reset_raw,
                name=f"{name}.reset_observation",
            )
        ),
        safety_violation=_boolean(
            mapping["safety_violation"],
            name=f"{name}.safety_violation",
        ),
        intervention=_boolean(
            mapping["intervention"],
            name=f"{name}.intervention",
        ),
        near_miss=_boolean(mapping["near_miss"], name=f"{name}.near_miss"),
        safety_cost=_nonnegative_float(
            mapping["safety_cost"],
            name=f"{name}.safety_cost",
        ),
        near_miss_cost=_nonnegative_float(
            mapping["near_miss_cost"],
            name=f"{name}.near_miss_cost",
        ),
    )
    if _canonical_json(_transition_to_config(transition)) != _canonical_json(mapping):
        raise ValueError(f"{name} is noncanonical")
    return transition


def _pending_to_config(pending: _PendingLearningDecision | None) -> object:
    if pending is None:
        return None
    return {
        "evaluator_regime_id": pending.evaluator_regime_id,
        "outer": _decision_to_config(pending.outer),
        "inner": _decision_to_config(pending.inner),
    }


def _pending_from_config(value: object, *, name: str) -> _PendingLearningDecision | None:
    if value is None:
        return None
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {"evaluator_regime_id", "outer", "inner"},
        name=name,
    )
    return _PendingLearningDecision(
        evaluator_regime_id=_string(
            mapping["evaluator_regime_id"],
            name=f"{name}.evaluator_regime_id",
        ),
        outer=_decision_from_config(mapping["outer"], name=f"{name}.outer"),
        inner=_decision_from_config(mapping["inner"], name=f"{name}.inner"),
    )


def _action_record_to_config(record: _ReferenceActionRecord) -> dict[str, object]:
    inner_available = record.inner_decision_id is not None
    return {
        "evaluator_regime_id": record.evaluator_regime_id,
        "transition": _transition_to_config(record.transition),
        "inner_decision_id": (
            None if record.inner_decision_id is None else list(record.inner_decision_id)
        ),
        "inner_decision_observation": (
            list(record.transition.observation) if inner_available else None
        ),
        "inner_decision_action": record.transition.action if inner_available else None,
        "oracle_action_scores": (
            None if record.oracle_action_scores is None else list(record.oracle_action_scores)
        ),
        "decision_selected_before_outcome": True,
        "transition_ownership_verified": True,
    }


def _action_record_from_config(
    value: object,
    *,
    name: str,
) -> _ReferenceActionRecord:
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {
            "evaluator_regime_id",
            "transition",
            "inner_decision_id",
            "inner_decision_observation",
            "inner_decision_action",
            "oracle_action_scores",
            "decision_selected_before_outcome",
            "transition_ownership_verified",
        },
        name=name,
    )
    if not _boolean(
        mapping["decision_selected_before_outcome"],
        name=f"{name}.decision_selected_before_outcome",
    ):
        raise ValueError(f"{name} must preserve predict-before-outcome")
    if not _boolean(
        mapping["transition_ownership_verified"],
        name=f"{name}.transition_ownership_verified",
    ):
        raise ValueError(f"{name} must preserve transition ownership")
    inner_raw = mapping["inner_decision_id"]
    inner_observation_raw = mapping["inner_decision_observation"]
    inner_action_raw = mapping["inner_decision_action"]
    oracle_raw = mapping["oracle_action_scores"]
    if inner_raw is None:
        if inner_observation_raw is not None or inner_action_raw is not None:
            raise ValueError(f"{name} inner decision ownership must be wholly unavailable")
    else:
        inner_observation = _observation_from_json(
            inner_observation_raw,
            name=f"{name}.inner_decision_observation",
        )
        inner_action = _nonnegative_int(
            inner_action_raw,
            name=f"{name}.inner_decision_action",
        )
        transition_mapping = _mapping(
            mapping["transition"],
            name=f"{name}.transition",
        )
        if (
            inner_observation
            != _observation_from_json(
                transition_mapping["observation"],
                name=f"{name}.transition.observation",
            )
            or inner_action != transition_mapping["action"]
        ):
            raise ValueError(f"{name} wrapper and inner decision ownership disagree")
    record = _ReferenceActionRecord(
        evaluator_regime_id=_string(
            mapping["evaluator_regime_id"],
            name=f"{name}.evaluator_regime_id",
        ),
        transition=_transition_from_config(
            mapping["transition"],
            name=f"{name}.transition",
        ),
        inner_decision_id=(
            None
            if inner_raw is None
            else _decision_id_from_json(inner_raw, name=f"{name}.inner_decision_id")
        ),
        oracle_action_scores=(
            None
            if oracle_raw is None
            else tuple(
                _finite_float(item, name=f"{name}.oracle_action_scores")
                for item in _list(oracle_raw, name=f"{name}.oracle_action_scores")
            )
        ),
    )
    if _canonical_json(_action_record_to_config(record)) != _canonical_json(mapping):
        raise ValueError(f"{name} is noncanonical")
    return record


def _extra_record_to_config(record: _ExtraTrainingRecord) -> dict[str, object]:
    return {
        "reference_regime_id": record.reference_regime_id,
        "transition": _transition_to_config(record.transition),
        "backward_call_count": record.backward_call_count,
        "decision_selected_before_outcome": True,
        "transition_ownership_verified": True,
    }


def _extra_record_from_config(value: object, *, name: str) -> _ExtraTrainingRecord:
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {
            "reference_regime_id",
            "transition",
            "backward_call_count",
            "decision_selected_before_outcome",
            "transition_ownership_verified",
        },
        name=name,
    )
    if not _boolean(
        mapping["decision_selected_before_outcome"],
        name=f"{name}.decision_selected_before_outcome",
    ) or not _boolean(
        mapping["transition_ownership_verified"],
        name=f"{name}.transition_ownership_verified",
    ):
        raise ValueError(f"{name} ownership or temporal ordering is invalid")
    backward_raw = mapping["backward_call_count"]
    record = _ExtraTrainingRecord(
        reference_regime_id=_string(
            mapping["reference_regime_id"],
            name=f"{name}.reference_regime_id",
        ),
        transition=_transition_from_config(
            mapping["transition"],
            name=f"{name}.transition",
        ),
        backward_call_count=(
            None
            if backward_raw is None
            else _nonnegative_int(backward_raw, name=f"{name}.backward_call_count")
        ),
    )
    if _canonical_json(_extra_record_to_config(record)) != _canonical_json(mapping):
        raise ValueError(f"{name} is noncanonical")
    return record


def _trace_to_config(trace: _ReferenceTrace) -> dict[str, object]:
    return {
        "actions": [_action_record_to_config(record) for record in trace.actions],
        "evaluator_matrix": [list(row) for row in trace.evaluator_matrix],
        "backward_call_counts": list(trace.backward_call_counts),
    }


def _trace_from_config(value: object, *, name: str) -> _ReferenceTrace:
    mapping = _mapping(value, name=name)
    _exact_keys(
        mapping,
        {"actions", "evaluator_matrix", "backward_call_counts"},
        name=name,
    )
    backward: list[int | None] = []
    for item in _list(
        mapping["backward_call_counts"],
        name=f"{name}.backward_call_counts",
    ):
        backward.append(
            None if item is None else _nonnegative_int(item, name=f"{name}.backward_call_counts")
        )
    trace = _ReferenceTrace(
        actions=tuple(
            _action_record_from_config(item, name=f"{name}.actions[{index}]")
            for index, item in enumerate(_list(mapping["actions"], name=f"{name}.actions"))
        ),
        evaluator_matrix=tuple(
            tuple(
                _finite_float(item, name=f"{name}.evaluator_matrix")
                for item in _list(row, name=f"{name}.evaluator_matrix")
            )
            for row in _list(
                mapping["evaluator_matrix"],
                name=f"{name}.evaluator_matrix",
            )
        ),
        backward_call_counts=tuple(backward),
    )
    if _canonical_json(_trace_to_config(trace)) != _canonical_json(mapping):
        raise ValueError(f"{name} is noncanonical")
    return trace


def _translated_transition(
    transition: ControlTransition,
    inner: ControlDecision,
) -> ControlTransition:
    if (
        transition.observation != inner.observation
        or transition.action != inner.action
        or not inner.armed
    ):
        raise ValueError("wrapper and inner decision ownership disagree")
    return dataclasses.replace(transition, decision_id=inner.decision_id)


def _unavailable_metric_applicability(
    protocol: ContinuingControlProtocol,
    reason: str,
) -> dict[str, object]:
    base = control_core._control_metric_applicability(protocol)
    return {
        metric: {
            "applicable": False,
            "unavailable_reason": reason,
            "common_protocol_applicability": _json_clone(applicability),
        }
        for metric, applicability in base.items()
    }


class PrivilegedContinualControlReferenceSuite:
    """Run privileged controls without adding a learner to matched conditions."""

    def __init__(
        self,
        *,
        config: PrivilegedReferenceRunConfig,
        protocol: ContinuingControlProtocol,
        common_evaluation_budget: ContinuingControlBudget,
        environment_factory: ReferenceEnvironmentFactory,
        probes: Mapping[str, Sequence[ControlProbe]],
        fresh_learner_factory: RetainedRegimeIdentityLearnerFactory,
        stationary_learner_factory: StationaryReferenceLearnerFactory,
        stationary_stream: FrozenStationaryReferenceStream,
        oracle_source_factory: ExactOracleOutcomeSourceFactory,
    ):
        if not isinstance(config, PrivilegedReferenceRunConfig):
            raise TypeError("config must be a PrivilegedReferenceRunConfig")
        if not isinstance(protocol, ContinuingControlProtocol):
            raise TypeError("protocol must be a ContinuingControlProtocol")
        if not isinstance(common_evaluation_budget, ContinuingControlBudget):
            raise TypeError("common_evaluation_budget must be a ContinuingControlBudget")
        for factory_name, factory in (
            ("environment_factory", environment_factory),
            ("fresh_learner_factory", fresh_learner_factory),
            ("stationary_learner_factory", stationary_learner_factory),
            ("oracle_source_factory", oracle_source_factory),
        ):
            if not callable(factory):
                raise TypeError(f"{factory_name} must be callable")
        if not isinstance(stationary_stream, FrozenStationaryReferenceStream):
            raise TypeError("stationary_stream has an invalid type")

        self._config = PrivilegedReferenceRunConfig.from_config(config.to_config())
        self._protocol, self._protocol_config = control_core._parse_protocol_config(
            _json_clone(dataclasses.asdict(protocol))
        )
        self._budget, self._budget_config = control_core._parse_budget_config(
            _json_clone(dataclasses.asdict(common_evaluation_budget))
        )
        if self._budget.transition_limit != len(self._protocol.regime_schedule):
            raise ValueError("common transition limit must exactly match the protocol")
        transition_count = len(self._protocol.regime_schedule)
        for field_name in (
            "decision_call_limit",
            "environment_call_limit",
            "update_call_limit",
            "stored_decision_id_limit",
        ):
            if getattr(self._budget, field_name) < transition_count:
                raise ValueError(f"common {field_name} is too small")

        self._probes = {regime_id: tuple(values) for regime_id, values in probes.items()}
        if set(self._probes) != set(self._protocol.evaluator_regime_ids):
            raise ValueError("probe keys must exactly match evaluator_regime_ids")
        if any(not values for values in self._probes.values()):
            raise ValueError("each evaluator regime requires at least one probe")
        for regime_id, values in self._probes.items():
            for index, probe in enumerate(values):
                if not isinstance(probe, ControlProbe):
                    raise TypeError(f"probes.{regime_id}[{index}] must be a ControlProbe")

        seed = self._config.seed
        environments = tuple(environment_factory(seed) for _ in REFERENCE_ROLES)
        if any(not isinstance(env, ContinuingControlEnvironment) for env in environments):
            raise TypeError("environment_factory must return ContinuingControlEnvironment")
        if len({id(env) for env in environments}) != len(environments):
            raise ValueError("reference roles require independent environment instances")
        self._environments = cast(
            tuple[
                ContinuingControlEnvironment,
                ContinuingControlEnvironment,
                ContinuingControlEnvironment,
            ],
            environments,
        )
        environment_configs = tuple(
            self._json_object(env.to_config(), name=f"{role} environment config")
            for role, env in zip(REFERENCE_ROLES, self._environments, strict=True)
        )
        if len({_canonical_json(value) for value in environment_configs}) != 1:
            raise ValueError("all reference environment configs must be identical")
        if (
            _assert_explicit_seed_fields(
                environment_configs[0],
                seed=seed,
                name="environment",
            )
            == 0
        ):
            raise ValueError("environment config must explicitly bind the suite seed")
        self._environment_config = environment_configs[0]

        initial_payloads: list[object] = []
        initial_observations: list[tuple[float, ...]] = []
        for role, environment in zip(REFERENCE_ROLES, self._environments, strict=True):
            first = environment.init()
            first_payload = self._environment_state_payload(environment, first)
            second_payload = self._environment_state_payload(environment, environment.init())
            if first_payload != second_payload:
                raise ValueError(f"{role} environment init is not deterministic")
            observation = self._environment_observation(environment, first)
            initial_payloads.append(first_payload)
            initial_observations.append(observation)
        if len(set(initial_observations)) != 1:
            raise ValueError("independent reference environments have different initial states")
        self._initial_environment_payloads = tuple(initial_payloads)
        self._initial_observation = initial_observations[0]

        n_actions = _positive_int(self._environments[0].n_actions, name="environment.n_actions")
        if any(environment.n_actions != n_actions for environment in self._environments):
            raise ValueError("reference environments disagree on action count")
        for values in self._probes.values():
            for probe in values:
                if len(probe.action_scores) != n_actions:
                    raise ValueError("probe action-score width must match environment actions")
        self._n_actions = n_actions

        self._fresh_learners = tuple(
            fresh_learner_factory(seed, regime_id)
            for regime_id in self._protocol.evaluator_regime_ids
        )
        self._stationary_learner = stationary_learner_factory(seed)
        all_learners = (*self._fresh_learners, self._stationary_learner)
        if any(not isinstance(learner, ContinuingControlLearner) for learner in all_learners):
            raise TypeError("reference learner factories must return ContinuingControlLearner")
        if len({id(learner) for learner in all_learners}) != len(all_learners):
            raise ValueError("fresh and stationary references require independent learners")
        if any(learner.n_actions != n_actions for learner in all_learners):
            raise ValueError("reference learner action counts must match the environment")
        self._fresh_learner_configs = tuple(
            self._json_object(
                learner.to_config(),
                name=f"fresh learner {regime_id} config",
            )
            for regime_id, learner in zip(
                self._protocol.evaluator_regime_ids,
                self._fresh_learners,
                strict=True,
            )
        )
        self._stationary_learner_config = self._json_object(
            self._stationary_learner.to_config(),
            name="stationary learner config",
        )
        for index, learner_config in enumerate(
            (*self._fresh_learner_configs, self._stationary_learner_config)
        ):
            _string(learner_config.get("type"), name=f"reference learner[{index}].type")
            _versioned_identifier(
                learner_config.get("schema_version"),
                name=f"reference learner[{index}].schema_version",
            )
            _string(learner_config.get("name"), name=f"reference learner[{index}].name")
            _assert_explicit_seed_fields(
                learner_config,
                seed=seed,
                name=f"reference learner[{index}]",
            )

        self._stationary_stream = FrozenStationaryReferenceStream.from_config(
            stationary_stream.to_config()
        )
        if self._stationary_stream.seed != seed:
            raise ValueError("stationary stream seed does not match suite seed")
        if set(example.reference_regime_id for example in self._stationary_stream.examples) != set(
            self._protocol.evaluator_regime_ids
        ):
            raise ValueError("stationary stream regime IDs must match the protocol exactly")
        for example in self._stationary_stream.examples:
            if len(example.action_scores) != n_actions:
                raise ValueError("stationary stream action-score width is invalid")
        if len(self._stationary_stream.examples[0].observation) != len(self._initial_observation):
            raise ValueError("stationary stream observation dimension is incompatible")

        oracle_source = oracle_source_factory(seed)
        if not isinstance(oracle_source, FrozenExactOracleOutcomeSource):
            raise TypeError("oracle_source_factory must return FrozenExactOracleOutcomeSource")
        if isinstance(oracle_source, ContinuingControlLearner):
            raise TypeError("oracle source must remain outside ContinuingControlLearner")
        self._oracle_source = oracle_source
        self._oracle_source_config = self._json_object(
            oracle_source.to_config(),
            name="oracle source config",
        )
        _string(self._oracle_source_config.get("type"), name="oracle source.type")
        if self._oracle_source_config.get("schema_version") != ORACLE_SOURCE_SCHEMA:
            raise ValueError("oracle source schema_version is invalid")
        if (
            _string(
                self._oracle_source_config.get("score_semantics"),
                name="oracle source.score_semantics",
            )
            != EXACT_ORACLE_SCORE_SEMANTICS
        ):
            raise ValueError(
                "oracle source must declare exact frozen counterfactual outcome semantics; "
                "stochastic expected scores are not accepted by this role"
            )
        if (
            _assert_explicit_seed_fields(
                self._oracle_source_config,
                seed=seed,
                name="oracle source",
            )
            == 0
        ):
            raise ValueError("oracle source config must explicitly bind the suite seed")

        probe_examples = sum(len(values) for values in self._probes.values())
        maximum_probe_calls = len(self._protocol.checkpoint_steps) * probe_examples
        if self._budget.probe_call_limit < maximum_probe_calls:
            raise ValueError("common probe-call budget is too small")
        self._probe_examples_per_checkpoint = probe_examples
        self._probe_action_score_scalars_per_checkpoint = sum(
            len(probe.action_scores) for values in self._probes.values() for probe in values
        )
        self._probe_config = {
            regime_id: [probe.to_config() for probe in self._probes[regime_id]]
            for regime_id in self._protocol.evaluator_regime_ids
        }
        self._validate_exact_extra_data_budget()

    @staticmethod
    def _json_object(value: object, *, name: str) -> dict[str, object]:
        cloned = _json_clone(value)
        if not isinstance(cloned, dict) or any(not isinstance(key, str) for key in cloned):
            raise ValueError(f"{name} must be a JSON object")
        return cast(dict[str, object], cloned)

    def _validate_exact_extra_data_budget(self) -> None:
        budget = self._config.extra_data_budget
        example_count = len(self._stationary_stream.examples)
        reward_scalars = sum(
            len(example.action_scores) for example in self._stationary_stream.examples
        )
        exact = {
            "stationary_transition_limit": example_count,
            "stationary_decision_call_limit": example_count,
            "stationary_update_call_limit": example_count,
            "stationary_reward_table_scalar_limit": reward_scalars,
            "oracle_callback_limit": len(self._protocol.regime_schedule),
            "oracle_action_score_scalar_limit": (
                len(self._protocol.regime_schedule) * self._n_actions
            ),
            "oracle_probe_action_score_scalar_limit": (
                len(self._protocol.checkpoint_steps)
                * self._probe_action_score_scalars_per_checkpoint
            ),
        }
        for field_name, expected in exact.items():
            if getattr(budget, field_name) != expected:
                raise ValueError(f"{field_name} must exactly equal {expected}")

    @staticmethod
    def _environment_state_payload(
        environment: ContinuingControlEnvironment,
        state: Any,
    ) -> object:
        payload = _json_clone(environment.state_to_config(state))
        if _json_clone(environment.state_to_config(state)) != payload:
            raise ValueError("environment state serialization is not idempotent")
        restored = environment.state_from_config(_json_clone(payload))
        if _json_clone(environment.state_to_config(restored)) != payload:
            raise ValueError("environment state serialization is not canonical")
        return payload

    def _environment_observation(
        self,
        environment: ContinuingControlEnvironment,
        state: Any,
    ) -> tuple[float, ...]:
        payload = self._environment_state_payload(environment, state)
        working = environment.state_from_config(_json_clone(payload))
        observation = _observation(
            environment.observation(working),
            name="reference environment observation",
        )
        if self._environment_state_payload(environment, working) != payload:
            raise ValueError("environment observation query mutated its state")
        return observation

    @staticmethod
    def _learner_state_payload(learner: ContinuingControlLearner, state: Any) -> object:
        payload = _json_clone(learner.state_to_config(state))
        if _json_clone(learner.state_to_config(state)) != payload:
            raise ValueError(f"{learner.name}: state serialization is not idempotent")
        restored = learner.state_from_config(_json_clone(payload))
        if _json_clone(learner.state_to_config(restored)) != payload:
            raise ValueError(f"{learner.name}: state serialization is not canonical")
        return payload

    def _assert_environment_configs_stable(self) -> None:
        for environment in self._environments:
            if (
                self._json_object(
                    environment.to_config(),
                    name="environment config",
                )
                != self._environment_config
            ):
                raise ValueError("reference environment config changed during execution")

    def _assert_learner_config_stable(
        self,
        learner: ContinuingControlLearner,
        expected: Mapping[str, object],
    ) -> None:
        if self._json_object(learner.to_config(), name="learner config") != expected:
            raise ValueError(f"{learner.name}: config changed during reference execution")

    def _learner_state_valid(
        self,
        learner: ContinuingControlLearner,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        valid = learner.state_valid_for_observation(working, observation)
        if not isinstance(valid, bool):
            raise TypeError(f"{learner.name}: state validity must be boolean")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: state validity query mutated state")
        return valid

    def _initialize_learner(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        observation: tuple[float, ...],
    ) -> Any:
        self._assert_learner_config_stable(learner, expected_config)
        first = learner.init(observation)
        first_payload = self._learner_state_payload(learner, first)
        second_payload = self._learner_state_payload(learner, learner.init(observation))
        if first_payload != second_payload:
            raise ValueError(f"{learner.name}: init is not deterministic")
        if not self._learner_state_valid(learner, first, observation):
            raise ValueError(f"{learner.name}: initialized state does not own observation")
        self._assert_learner_config_stable(learner, expected_config)
        return learner.state_from_config(_json_clone(first_payload))

    def _inner_decision(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        state: Any,
        observation: tuple[float, ...],
    ) -> ControlDecision:
        self._assert_learner_config_stable(learner, expected_config)
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        raw = learner.decide(working, observation)
        if not isinstance(raw, ControlDecision):
            raise TypeError(f"{learner.name}: decide must return ControlDecision")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: decide mutated state")
        if not raw.armed or raw.observation != observation:
            raise ValueError(f"{learner.name}: decision ownership is invalid")
        if raw.action >= self._n_actions:
            raise ValueError(f"{learner.name}: decision action is out of range")
        self._assert_learner_config_stable(learner, expected_config)
        return raw

    def _pending_learning_decision(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        state: Any,
        observation: tuple[float, ...],
        *,
        evaluator_regime_id: str,
        lifecycle_id: tuple[int, int],
        generation: int,
    ) -> _PendingLearningDecision:
        inner = self._inner_decision(learner, expected_config, state, observation)
        outer = ControlDecision(
            observation=observation,
            action=inner.action,
            decision_id=_generation_decision_id(lifecycle_id, generation),
        )
        return _PendingLearningDecision(evaluator_regime_id, outer, inner)

    def _learner_update(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        state: Any,
        transition: ControlTransition,
    ) -> tuple[Any, int | None]:
        self._assert_learner_config_stable(learner, expected_config)
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        raw = learner.update(working, transition)
        if not isinstance(raw, ControlLearnerUpdate):
            raise TypeError(f"{learner.name}: update must return ControlLearnerUpdate")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError(f"{learner.name}: update mutated its source state")
        if not raw.applied:
            raise ValueError(f"{learner.name}: rejected an owned reference transition")
        maximum = learner.max_backward_calls_per_update
        if maximum is None:
            if raw.backward_call_count is not None:
                raise ValueError("learner reported backward calls without a declared bound")
        else:
            if raw.backward_call_count is None:
                raise ValueError("learner backward-call count is unexpectedly unavailable")
            if raw.backward_call_count > _nonnegative_int(
                maximum,
                name="max_backward_calls_per_update",
            ):
                raise ValueError("learner backward-call bound exceeded")
        next_payload = self._learner_state_payload(learner, raw.state)
        next_state = learner.state_from_config(_json_clone(next_payload))
        if not self._learner_state_valid(
            learner,
            next_state,
            transition.next_decision_observation,
        ):
            raise ValueError(f"{learner.name}: updated state does not own next observation")
        self._assert_learner_config_stable(learner, expected_config)
        return next_state, raw.backward_call_count

    def _environment_step(
        self,
        environment: ContinuingControlEnvironment,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> tuple[Any, ControlTransition]:
        self._assert_environment_configs_stable()
        payload = self._environment_state_payload(environment, state)
        working = environment.state_from_config(_json_clone(payload))
        raw = environment.step(working, decision, evaluator_regime_id)
        if not isinstance(raw, ControlEnvironmentUpdate):
            raise TypeError("environment.step must return ControlEnvironmentUpdate")
        if self._environment_state_payload(environment, working) != payload:
            raise ValueError("environment.step mutated its source state")
        transition = raw.transition
        if not isinstance(transition, ControlTransition):
            raise TypeError("environment update transition has an invalid type")
        if (
            transition.observation != decision.observation
            or transition.action != decision.action
            or transition.decision_id != decision.decision_id
        ):
            raise ValueError("environment transition does not own dispatched decision")
        next_payload = self._environment_state_payload(environment, raw.state)
        next_state = environment.state_from_config(_json_clone(next_payload))
        if self._environment_observation(environment, next_state) != (
            transition.next_decision_observation
        ):
            raise ValueError("environment next state and transition disagree")
        return next_state, transition

    def _probe_action(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        state: Any,
        probe: ControlProbe,
    ) -> int:
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        action = _nonnegative_int(
            learner.probe_action(working, probe.observation),
            name="held-out probe action",
        )
        if action >= self._n_actions:
            raise ValueError("held-out probe action is out of range")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError("held-out probe mutated snapshot state")
        self._assert_learner_config_stable(learner, expected_config)
        return action

    def _resource_payload(
        self,
        learner: ContinuingControlLearner,
        expected_config: Mapping[str, object],
        state: Any,
    ) -> dict[str, object]:
        payload = self._learner_state_payload(learner, state)
        working = learner.state_from_config(_json_clone(payload))
        usage = learner.resource_usage(working)
        if not isinstance(usage, ControlResourceUsage):
            raise TypeError("learner resource_usage must return ControlResourceUsage")
        if self._learner_state_payload(learner, working) != payload:
            raise ValueError("resource accounting mutated learner state")
        self._assert_learner_config_stable(learner, expected_config)
        return dataclasses.asdict(usage)

    def _oracle_scores(
        self,
        observation: tuple[float, ...],
        *,
        evaluator_regime_id: str,
        step: int,
    ) -> tuple[float, ...]:
        if (
            self._json_object(
                self._oracle_source.to_config(),
                name="oracle source config",
            )
            != self._oracle_source_config
        ):
            raise ValueError("oracle source config changed during execution")
        raw = self._oracle_source.action_scores(
            observation,
            evaluator_regime_id=evaluator_regime_id,
            step=step,
        )
        if not isinstance(raw, tuple) or len(raw) != self._n_actions:
            raise ValueError("oracle action scores have an invalid width or container")
        scores = tuple(
            _finite_float(value, name=f"oracle action_scores[{index}]")
            for index, value in enumerate(raw)
        )
        if (
            self._json_object(
                self._oracle_source.to_config(),
                name="oracle source config",
            )
            != self._oracle_source_config
        ):
            raise ValueError("oracle callback mutated its frozen config")
        return scores

    def to_config(self) -> dict[str, object]:
        """Return a strict identity with no ordinary candidate/baseline conditions."""
        self._assert_environment_configs_stable()
        for learner, expected in zip(
            self._fresh_learners,
            self._fresh_learner_configs,
            strict=True,
        ):
            self._assert_learner_config_stable(learner, expected)
        self._assert_learner_config_stable(
            self._stationary_learner,
            self._stationary_learner_config,
        )
        if (
            self._json_object(
                self._oracle_source.to_config(),
                name="oracle source config",
            )
            != self._oracle_source_config
        ):
            raise ValueError("oracle source config changed")
        return {
            "type": "PrivilegedContinualControlReferenceSuite",
            "schema_version": REFERENCE_SUITE_SCHEMA,
            "report_schema_version": REFERENCE_REPORT_SCHEMA,
            "checkpoint_schema_version": REFERENCE_CHECKPOINT_SCHEMA,
            "run_config": self._config.to_config(),
            "protocol": _json_clone(self._protocol_config),
            "common_evaluation_budget": _json_clone(self._budget_config),
            "environment": {
                "factory_seed": self._config.seed,
                "config": _json_clone(self._environment_config),
                "config_sha256": _digest(self._environment_config),
                "independent_copy_count": 3,
            },
            RETAINED_FRESH_PER_REGIME_ROLE: {
                "factory_seed": self._config.seed,
                "lifecycle_id": list(self._config.fresh_lifecycle_id),
                "learner_configs": [
                    {
                        "evaluator_regime_id": regime_id,
                        "config": _json_clone(learner_config),
                        "config_sha256": _digest(learner_config),
                    }
                    for regime_id, learner_config in zip(
                        self._protocol.evaluator_regime_ids,
                        self._fresh_learner_configs,
                        strict=True,
                    )
                ],
                "regime_selection_owner": "privileged suite wrapper only",
                "initialization_scope": "once per evaluator regime identity",
                "recurrence_policy": "retain and reuse that regime-identity learner state",
                "fresh_per_segment_or_change": False,
            },
            "stationary_multitask": {
                "factory_seed": self._config.seed,
                "lifecycle_id": list(self._config.stationary_lifecycle_id),
                "learner_config": _json_clone(self._stationary_learner_config),
                "learner_config_sha256": _digest(self._stationary_learner_config),
                "stream": self._stationary_stream.to_config(),
                "stream_sha256": _digest(self._stationary_stream.to_config()),
            },
            ORACLE_ACTION_DATA_ROLE: {
                "factory_seed": self._config.seed,
                "lifecycle_id": list(self._config.oracle_lifecycle_id),
                "source_config": _json_clone(self._oracle_source_config),
                "source_config_sha256": _digest(self._oracle_source_config),
                "score_semantics": EXACT_ORACLE_SCORE_SEMANTICS,
                "callback_temporal_contract": EXACT_ORACLE_CALLBACK_TEMPORAL_CONTRACT,
                "selected_score_realized_reward_equality_required": True,
                "stochastic_expected_score_source_supported": False,
            },
            "probe_sha256": _digest(self._probe_config),
            "probe_examples_per_checkpoint": self._probe_examples_per_checkpoint,
            "probe_action_score_scalars_per_checkpoint": (
                self._probe_action_score_scalars_per_checkpoint
            ),
            "reference_roles": list(REFERENCE_ROLES),
            "ordinary_conditions_included": False,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "accepted_scientific_evidence": False,
        }

    @staticmethod
    def _empty_trace() -> _ReferenceTrace:
        return _ReferenceTrace(actions=(), evaluator_matrix=(), backward_call_counts=())

    def _pretrain_stationary(self) -> tuple[Any, tuple[_ExtraTrainingRecord, ...]]:
        learner = self._stationary_learner
        expected = self._stationary_learner_config
        state = self._initialize_learner(
            learner,
            expected,
            self._stationary_stream.examples[0].observation,
        )
        records: list[_ExtraTrainingRecord] = []
        used_ids: set[DecisionId] = set()
        for index, example in enumerate(self._stationary_stream.examples):
            if not self._learner_state_valid(learner, state, example.observation):
                raise ValueError(f"stationary learner cannot own frozen stream example {index}")
            decision = self._inner_decision(
                learner,
                expected,
                state,
                example.observation,
            )
            if decision.decision_id in used_ids:
                raise ValueError("stationary extra stream reused a decision identifier")
            used_ids.add(decision.decision_id)
            transition = ControlTransition(
                observation=decision.observation,
                action=decision.action,
                decision_id=decision.decision_id,
                reward=example.action_scores[decision.action],
                discount=example.discount,
                terminated=example.terminated,
                truncated=example.truncated,
                bootstrap_observation=example.bootstrap_observation,
                reset_observation=example.reset_observation,
            )
            state, backward = self._learner_update(
                learner,
                expected,
                state,
                transition,
            )
            records.append(
                _ExtraTrainingRecord(
                    reference_regime_id=example.reference_regime_id,
                    transition=transition,
                    backward_call_count=backward,
                )
            )
        known_backward = [
            record.backward_call_count
            for record in records
            if record.backward_call_count is not None
        ]
        if sum(known_backward) > (self._config.extra_data_budget.stationary_backward_call_limit):
            raise ValueError("stationary extra-stream backward-call budget exceeded")
        return state, tuple(records)

    def _fresh_probe(
        self,
        states: tuple[Any | None, ...],
    ) -> tuple[tuple[float, ...], int]:
        row: list[float] = []
        ephemeral_initializations = 0
        for regime_index, regime_id in enumerate(self._protocol.evaluator_regime_ids):
            learner = self._fresh_learners[regime_index]
            expected = self._fresh_learner_configs[regime_index]
            live_state = states[regime_index]
            probes = self._probes[regime_id]
            if live_state is None:
                live_state = self._initialize_learner(
                    learner,
                    expected,
                    probes[0].observation,
                )
                ephemeral_initializations += 1
            scores = [
                probe.action_scores[self._probe_action(learner, expected, live_state, probe)]
                for probe in probes
            ]
            row.append(sum(scores) / len(scores))
        return tuple(row), ephemeral_initializations

    def _stationary_probe(self, state: Any) -> tuple[float, ...]:
        row: list[float] = []
        for regime_id in self._protocol.evaluator_regime_ids:
            scores = [
                probe.action_scores[
                    self._probe_action(
                        self._stationary_learner,
                        self._stationary_learner_config,
                        state,
                        probe,
                    )
                ]
                for probe in self._probes[regime_id]
            ]
            row.append(sum(scores) / len(scores))
        return tuple(row)

    def _oracle_probe(self) -> tuple[float, ...]:
        return tuple(
            sum(max(probe.action_scores) for probe in self._probes[regime_id])
            / len(self._probes[regime_id])
            for regime_id in self._protocol.evaluator_regime_ids
        )

    def init(self) -> PrivilegedReferenceRunState:
        """Initialize all references and arm their first pre-outcome decisions."""
        self.to_config()
        first_regime = self._protocol.regime_schedule[0]
        first_regime_index = self._protocol.evaluator_regime_ids.index(first_regime)

        fresh_environment = self._environments[0]
        fresh_environment_state = fresh_environment.state_from_config(
            _json_clone(self._initial_environment_payloads[0])
        )
        fresh_learner = self._fresh_learners[first_regime_index]
        fresh_config = self._fresh_learner_configs[first_regime_index]
        first_fresh_state = self._initialize_learner(
            fresh_learner,
            fresh_config,
            self._initial_observation,
        )
        fresh_states: list[Any | None] = [None] * len(self._fresh_learners)
        fresh_states[first_regime_index] = first_fresh_state
        fresh_pending = self._pending_learning_decision(
            fresh_learner,
            fresh_config,
            first_fresh_state,
            self._initial_observation,
            evaluator_regime_id=first_regime,
            lifecycle_id=self._config.fresh_lifecycle_id,
            generation=0,
        )
        fresh = _FreshReferenceState(
            available=True,
            unavailable_reason=None,
            environment_state=fresh_environment_state,
            current_observation=self._initial_observation,
            learner_states=tuple(fresh_states),
            pending=fresh_pending,
            trace=self._empty_trace(),
            learner_initialization_count=1,
            ephemeral_probe_initialization_count=0,
        )

        stationary_environment = self._environments[1]
        stationary_environment_state = stationary_environment.state_from_config(
            _json_clone(self._initial_environment_payloads[1])
        )
        stationary_learner_state, extra_trace = self._pretrain_stationary()
        stationary_available = self._learner_state_valid(
            self._stationary_learner,
            stationary_learner_state,
            self._initial_observation,
        )
        stationary_pending = (
            self._pending_learning_decision(
                self._stationary_learner,
                self._stationary_learner_config,
                stationary_learner_state,
                self._initial_observation,
                evaluator_regime_id=first_regime,
                lifecycle_id=self._config.stationary_lifecycle_id,
                generation=0,
            )
            if stationary_available
            else None
        )
        stationary = _StationaryReferenceState(
            available=stationary_available,
            unavailable_reason=(None if stationary_available else STATIONARY_UNAVAILABLE_REASON),
            environment_state=stationary_environment_state,
            current_observation=self._initial_observation,
            learner_state=stationary_learner_state,
            pending=stationary_pending,
            trace=self._empty_trace(),
            extra_training_trace=extra_trace,
        )

        oracle_environment = self._environments[2]
        oracle_environment_state = oracle_environment.state_from_config(
            _json_clone(self._initial_environment_payloads[2])
        )
        oracle_scores = self._oracle_scores(
            self._initial_observation,
            evaluator_regime_id=first_regime,
            step=0,
        )
        oracle_action = max(
            range(self._n_actions),
            key=lambda action: (oracle_scores[action], -action),
        )
        oracle_pending = ControlDecision(
            observation=self._initial_observation,
            action=oracle_action,
            decision_id=_generation_decision_id(self._config.oracle_lifecycle_id, 0),
        )
        oracle = _OracleReferenceState(
            available=True,
            unavailable_reason=None,
            environment_state=oracle_environment_state,
            current_observation=self._initial_observation,
            pending=oracle_pending,
            pending_action_scores=oracle_scores,
            trace=self._empty_trace(),
        )
        state = PrivilegedReferenceRunState(0, fresh, stationary, oracle)
        self._validate_state(state)
        return state

    def _advance_fresh(
        self,
        state: _FreshReferenceState,
        *,
        completed_step: int,
        regime_id: str,
    ) -> _FreshReferenceState:
        if not state.available:
            return state
        pending = state.pending
        if pending is None or pending.evaluator_regime_id != regime_id:
            raise ValueError("fresh reference pending regime is invalid")
        regime_index = self._protocol.evaluator_regime_ids.index(regime_id)
        learner = self._fresh_learners[regime_index]
        learner_config = self._fresh_learner_configs[regime_index]
        learner_state = state.learner_states[regime_index]
        if learner_state is None:
            raise ValueError("fresh selected learner state is uninitialized")
        environment_state, transition = self._environment_step(
            self._environments[0],
            state.environment_state,
            pending.outer,
            regime_id,
        )
        translated = _translated_transition(transition, pending.inner)
        updated, backward = self._learner_update(
            learner,
            learner_config,
            learner_state,
            translated,
        )
        states = list(state.learner_states)
        states[regime_index] = updated
        record = _ReferenceActionRecord(
            evaluator_regime_id=regime_id,
            transition=transition,
            inner_decision_id=pending.inner.decision_id,
            oracle_action_scores=None,
        )
        matrix = state.trace.evaluator_matrix
        ephemeral = state.ephemeral_probe_initialization_count
        if completed_step in self._protocol.checkpoint_steps:
            row, new_ephemeral = self._fresh_probe(tuple(states))
            matrix = (*matrix, row)
            ephemeral += new_ephemeral
        trace = _ReferenceTrace(
            actions=(*state.trace.actions, record),
            evaluator_matrix=matrix,
            backward_call_counts=(*state.trace.backward_call_counts, backward),
        )
        pending_next: _PendingLearningDecision | None = None
        available = True
        unavailable_reason: str | None = None
        initialization_count = state.learner_initialization_count
        if completed_step < len(self._protocol.regime_schedule):
            next_regime = self._protocol.regime_schedule[completed_step]
            next_index = self._protocol.evaluator_regime_ids.index(next_regime)
            next_learner = self._fresh_learners[next_index]
            next_config = self._fresh_learner_configs[next_index]
            next_state = states[next_index]
            if next_state is None:
                next_state = self._initialize_learner(
                    next_learner,
                    next_config,
                    transition.next_decision_observation,
                )
                states[next_index] = next_state
                initialization_count += 1
            elif not self._learner_state_valid(
                next_learner,
                next_state,
                transition.next_decision_observation,
            ):
                available = False
                unavailable_reason = FRESH_UNAVAILABLE_REASON_PREFIX + repr(next_regime)
            if available:
                pending_next = self._pending_learning_decision(
                    next_learner,
                    next_config,
                    next_state,
                    transition.next_decision_observation,
                    evaluator_regime_id=next_regime,
                    lifecycle_id=self._config.fresh_lifecycle_id,
                    generation=completed_step,
                )
        return _FreshReferenceState(
            available=available,
            unavailable_reason=unavailable_reason,
            environment_state=environment_state,
            current_observation=transition.next_decision_observation,
            learner_states=tuple(states),
            pending=pending_next,
            trace=trace,
            learner_initialization_count=initialization_count,
            ephemeral_probe_initialization_count=ephemeral,
        )

    def _advance_stationary(
        self,
        state: _StationaryReferenceState,
        *,
        completed_step: int,
        regime_id: str,
    ) -> _StationaryReferenceState:
        if not state.available:
            return state
        pending = state.pending
        if pending is None or pending.evaluator_regime_id != regime_id:
            raise ValueError("stationary reference pending regime is invalid")
        environment_state, transition = self._environment_step(
            self._environments[1],
            state.environment_state,
            pending.outer,
            regime_id,
        )
        translated = _translated_transition(transition, pending.inner)
        learner_state, backward = self._learner_update(
            self._stationary_learner,
            self._stationary_learner_config,
            state.learner_state,
            translated,
        )
        record = _ReferenceActionRecord(
            evaluator_regime_id=regime_id,
            transition=transition,
            inner_decision_id=pending.inner.decision_id,
            oracle_action_scores=None,
        )
        matrix = state.trace.evaluator_matrix
        if completed_step in self._protocol.checkpoint_steps:
            matrix = (*matrix, self._stationary_probe(learner_state))
        trace = _ReferenceTrace(
            actions=(*state.trace.actions, record),
            evaluator_matrix=matrix,
            backward_call_counts=(*state.trace.backward_call_counts, backward),
        )
        pending_next: _PendingLearningDecision | None = None
        if completed_step < len(self._protocol.regime_schedule):
            next_regime = self._protocol.regime_schedule[completed_step]
            pending_next = self._pending_learning_decision(
                self._stationary_learner,
                self._stationary_learner_config,
                learner_state,
                transition.next_decision_observation,
                evaluator_regime_id=next_regime,
                lifecycle_id=self._config.stationary_lifecycle_id,
                generation=completed_step,
            )
        return _StationaryReferenceState(
            available=True,
            unavailable_reason=None,
            environment_state=environment_state,
            current_observation=transition.next_decision_observation,
            learner_state=learner_state,
            pending=pending_next,
            trace=trace,
            extra_training_trace=state.extra_training_trace,
        )

    def _advance_oracle(
        self,
        state: _OracleReferenceState,
        *,
        completed_step: int,
        regime_id: str,
    ) -> _OracleReferenceState:
        pending = state.pending
        scores = state.pending_action_scores
        if pending is None or scores is None:
            raise ValueError("oracle reference has no pending pre-outcome action scores")
        environment_state, transition = self._environment_step(
            self._environments[2],
            state.environment_state,
            pending,
            regime_id,
        )
        if transition.reward != scores[pending.action]:
            raise ValueError(
                "exact oracle selected-action score does not match the realized outcome"
            )
        record = _ReferenceActionRecord(
            evaluator_regime_id=regime_id,
            transition=transition,
            inner_decision_id=None,
            oracle_action_scores=scores,
        )
        matrix = state.trace.evaluator_matrix
        if completed_step in self._protocol.checkpoint_steps:
            matrix = (*matrix, self._oracle_probe())
        trace = _ReferenceTrace(
            actions=(*state.trace.actions, record),
            evaluator_matrix=matrix,
            backward_call_counts=(),
        )
        pending_next: ControlDecision | None = None
        next_scores: tuple[float, ...] | None = None
        if completed_step < len(self._protocol.regime_schedule):
            next_regime = self._protocol.regime_schedule[completed_step]
            next_scores = self._oracle_scores(
                transition.next_decision_observation,
                evaluator_regime_id=next_regime,
                step=completed_step,
            )
            action = max(
                range(self._n_actions),
                key=lambda index: (next_scores[index], -index),
            )
            pending_next = ControlDecision(
                observation=transition.next_decision_observation,
                action=action,
                decision_id=_generation_decision_id(
                    self._config.oracle_lifecycle_id,
                    completed_step,
                ),
            )
        return _OracleReferenceState(
            available=True,
            unavailable_reason=None,
            environment_state=environment_state,
            current_observation=transition.next_decision_observation,
            pending=pending_next,
            pending_action_scores=next_scores,
            trace=trace,
        )

    def advance(
        self,
        state: PrivilegedReferenceRunState,
        *,
        steps: int = 1,
    ) -> PrivilegedReferenceRunState:
        """Advance complete predict/environment/update transactions."""
        self._validate_state(state)
        requested = _positive_int(steps, name="steps")
        if state.step + requested > len(self._protocol.regime_schedule):
            raise ValueError("advance would exceed the bounded reference schedule")
        current = state
        for _ in range(requested):
            completed_step = current.step + 1
            regime_id = self._protocol.regime_schedule[current.step]
            current = PrivilegedReferenceRunState(
                step=completed_step,
                fresh_per_regime=self._advance_fresh(
                    current.fresh_per_regime,
                    completed_step=completed_step,
                    regime_id=regime_id,
                ),
                stationary_multitask=self._advance_stationary(
                    current.stationary_multitask,
                    completed_step=completed_step,
                    regime_id=regime_id,
                ),
                oracle_action_data_upper=self._advance_oracle(
                    current.oracle_action_data_upper,
                    completed_step=completed_step,
                    regime_id=regime_id,
                ),
            )
            self._validate_state(current)
        return current

    def _validate_trace(
        self,
        trace: _ReferenceTrace,
        *,
        role: str,
        lifecycle_id: tuple[int, int],
        suite_step: int,
        available: bool,
    ) -> None:
        if not isinstance(trace, _ReferenceTrace):
            raise TypeError(f"{role} trace has an invalid type")
        processed = len(trace.actions)
        if available and processed != suite_step:
            raise ValueError(f"{role} available trace does not match suite step")
        if not available and processed > suite_step:
            raise ValueError(f"{role} unavailable trace exceeds suite step")
        expected_rows = sum(
            checkpoint <= processed for checkpoint in self._protocol.checkpoint_steps
        )
        if len(trace.evaluator_matrix) != expected_rows:
            raise ValueError(f"{role} held-out row count is inconsistent")
        for row in trace.evaluator_matrix:
            if len(row) != len(self._protocol.evaluator_regime_ids):
                raise ValueError(f"{role} held-out row width is invalid")
            for value in row:
                _finite_float(value, name=f"{role} held-out score")
        expected_backward = 0 if role == ORACLE_ACTION_DATA_ROLE else processed
        if len(trace.backward_call_counts) != expected_backward:
            raise ValueError(f"{role} backward-call trace length is inconsistent")
        known_backward = [value for value in trace.backward_call_counts if value is not None]
        if sum(known_backward) > self._budget.backward_call_limit:
            raise ValueError(f"{role} evaluation backward-call budget exceeded")

        used_ids: set[DecisionId] = set()
        previous_next: tuple[float, ...] | None = None
        for index, record in enumerate(trace.actions):
            if not isinstance(record, _ReferenceActionRecord):
                raise TypeError(f"{role} action record has an invalid type")
            expected_regime = self._protocol.regime_schedule[index]
            if record.evaluator_regime_id != expected_regime:
                raise ValueError(f"{role} action trace regime order is invalid")
            transition = record.transition
            if transition.decision_id != _generation_decision_id(lifecycle_id, index):
                raise ValueError(f"{role} wrapper decision identifier is invalid")
            if transition.decision_id in used_ids:
                raise ValueError(f"{role} reused a wrapper decision identifier")
            used_ids.add(transition.decision_id)
            if previous_next is not None and transition.observation != previous_next:
                raise ValueError(f"{role} transition trace is observationally discontinuous")
            previous_next = transition.next_decision_observation
            if role == ORACLE_ACTION_DATA_ROLE:
                if record.inner_decision_id is not None:
                    raise ValueError("oracle trace cannot claim an inner learner decision")
                scores = record.oracle_action_scores
                if scores is None or len(scores) != self._n_actions:
                    raise ValueError("oracle trace must retain its pre-outcome action scores")
                expected_action = max(
                    range(self._n_actions),
                    key=lambda action: (scores[action], -action),
                )
                if transition.action != expected_action:
                    raise ValueError("oracle action is not the frozen score argmax")
                if transition.reward != scores[transition.action]:
                    raise ValueError("exact oracle selected score and realized reward disagree")
            else:
                if record.inner_decision_id is None:
                    raise ValueError(f"{role} trace is missing inner decision ownership")
                if record.oracle_action_scores is not None:
                    raise ValueError(f"{role} trace cannot contain oracle action scores")
        if processed:
            first = trace.actions[0].transition.observation
            if first != self._initial_observation:
                raise ValueError(f"{role} trace does not start from the common observation")
        if processed > self._budget.environment_call_limit:
            raise ValueError(f"{role} environment-call budget exceeded")
        if processed > self._budget.decision_call_limit:
            raise ValueError(f"{role} decision-call budget exceeded")
        if role != ORACLE_ACTION_DATA_ROLE and processed > self._budget.update_call_limit:
            raise ValueError(f"{role} update-call budget exceeded")
        probe_calls = len(trace.evaluator_matrix) * self._probe_examples_per_checkpoint
        if probe_calls > self._budget.probe_call_limit:
            raise ValueError(f"{role} probe-call budget exceeded")
        if processed > self._budget.stored_decision_id_limit:
            raise ValueError(f"{role} stored-decision-ID budget exceeded")

    def _validate_pending_learning(
        self,
        pending: _PendingLearningDecision | None,
        *,
        role: str,
        lifecycle_id: tuple[int, int],
        suite_step: int,
        available: bool,
        current_observation: tuple[float, ...],
    ) -> None:
        expected = available and suite_step < len(self._protocol.regime_schedule)
        if (pending is not None) != expected:
            raise ValueError(f"{role} pending-decision availability is inconsistent")
        if pending is None:
            return
        regime_id = self._protocol.regime_schedule[suite_step]
        if pending.evaluator_regime_id != regime_id:
            raise ValueError(f"{role} pending evaluator regime is invalid")
        if (
            not pending.outer.armed
            or not pending.inner.armed
            or pending.outer.observation != current_observation
            or pending.inner.observation != current_observation
            or pending.outer.action != pending.inner.action
            or pending.outer.action >= self._n_actions
            or pending.outer.decision_id != _generation_decision_id(lifecycle_id, suite_step)
        ):
            raise ValueError(f"{role} pending wrapper/inner ownership is invalid")

    def _validate_extra_training_trace(
        self,
        trace: tuple[_ExtraTrainingRecord, ...],
    ) -> None:
        examples = self._stationary_stream.examples
        if len(trace) != len(examples):
            raise ValueError("stationary extra training trace is incomplete")
        used_ids: set[DecisionId] = set()
        for index, (record, example) in enumerate(zip(trace, examples, strict=True)):
            if record.reference_regime_id != example.reference_regime_id:
                raise ValueError("stationary extra trace regime identity is invalid")
            transition = record.transition
            if transition.decision_id in used_ids:
                raise ValueError("stationary extra trace reused a decision identifier")
            used_ids.add(transition.decision_id)
            expected_static = (
                transition.observation == example.observation
                and transition.reward == example.action_scores[transition.action]
                and transition.discount == example.discount
                and transition.terminated == example.terminated
                and transition.truncated == example.truncated
                and transition.bootstrap_observation == example.bootstrap_observation
                and transition.reset_observation == example.reset_observation
            )
            if transition.action >= self._n_actions or not expected_static:
                raise ValueError(f"stationary extra trace example {index} is invalid")
        known_backward = [
            record.backward_call_count for record in trace if record.backward_call_count is not None
        ]
        if sum(known_backward) > (self._config.extra_data_budget.stationary_backward_call_limit):
            raise ValueError("stationary extra trace exceeds backward-call budget")

    def _validate_state(self, state: PrivilegedReferenceRunState) -> None:
        self.to_config()
        if not isinstance(state, PrivilegedReferenceRunState):
            raise TypeError("state must be a PrivilegedReferenceRunState")
        step = _nonnegative_int(state.step, name="state.step")
        if step > len(self._protocol.regime_schedule):
            raise ValueError("state.step exceeds the reference schedule")

        fresh = state.fresh_per_regime
        if not isinstance(fresh, _FreshReferenceState):
            raise TypeError("fresh reference state has an invalid type")
        if fresh.available != (fresh.unavailable_reason is None):
            raise ValueError("fresh reference availability is inconsistent")
        if not fresh.available:
            reason = cast(str, fresh.unavailable_reason)
            if not reason.startswith(FRESH_UNAVAILABLE_REASON_PREFIX):
                raise ValueError("fresh reference unavailability reason is invalid")
        if len(fresh.learner_states) != len(self._fresh_learners):
            raise ValueError("fresh learner-state count is invalid")
        initialized_count = 0
        for learner, expected, learner_state in zip(
            self._fresh_learners,
            self._fresh_learner_configs,
            fresh.learner_states,
            strict=True,
        ):
            self._assert_learner_config_stable(learner, expected)
            if learner_state is not None:
                self._learner_state_payload(learner, learner_state)
                initialized_count += 1
        if fresh.learner_initialization_count != initialized_count:
            raise ValueError("fresh persistent learner initialization count is invalid")
        expected_ephemeral = sum(
            len(
                set(self._protocol.evaluator_regime_ids)
                - set(self._protocol.regime_schedule[:checkpoint])
            )
            for checkpoint in self._protocol.checkpoint_steps
            if checkpoint <= len(fresh.trace.actions)
        )
        if fresh.ephemeral_probe_initialization_count != expected_ephemeral:
            raise ValueError("fresh ephemeral probe initialization count is invalid")
        _observation(fresh.current_observation, name="fresh current_observation")
        if (
            self._environment_observation(
                self._environments[0],
                fresh.environment_state,
            )
            != fresh.current_observation
        ):
            raise ValueError("fresh environment observation cache is inconsistent")
        self._validate_trace(
            fresh.trace,
            role=RETAINED_FRESH_PER_REGIME_ROLE,
            lifecycle_id=self._config.fresh_lifecycle_id,
            suite_step=step,
            available=fresh.available,
        )
        self._validate_pending_learning(
            fresh.pending,
            role=RETAINED_FRESH_PER_REGIME_ROLE,
            lifecycle_id=self._config.fresh_lifecycle_id,
            suite_step=step,
            available=fresh.available,
            current_observation=fresh.current_observation,
        )
        if fresh.pending is not None:
            index = self._protocol.evaluator_regime_ids.index(fresh.pending.evaluator_regime_id)
            learner_state = fresh.learner_states[index]
            if learner_state is None or not self._learner_state_valid(
                self._fresh_learners[index],
                learner_state,
                fresh.current_observation,
            ):
                raise ValueError("fresh pending learner state does not own observation")

        stationary = state.stationary_multitask
        if not isinstance(stationary, _StationaryReferenceState):
            raise TypeError("stationary reference state has an invalid type")
        if stationary.available != (stationary.unavailable_reason is None):
            raise ValueError("stationary availability is inconsistent")
        if not stationary.available and (
            stationary.unavailable_reason != STATIONARY_UNAVAILABLE_REASON
            or stationary.trace.actions
        ):
            raise ValueError("stationary unavailability state is invalid")
        _observation(
            stationary.current_observation,
            name="stationary current_observation",
        )
        if (
            self._environment_observation(
                self._environments[1],
                stationary.environment_state,
            )
            != stationary.current_observation
        ):
            raise ValueError("stationary environment observation cache is inconsistent")
        self._learner_state_payload(
            self._stationary_learner,
            stationary.learner_state,
        )
        self._validate_extra_training_trace(stationary.extra_training_trace)
        self._validate_trace(
            stationary.trace,
            role=STATIONARY_MULTITASK_ROLE,
            lifecycle_id=self._config.stationary_lifecycle_id,
            suite_step=step,
            available=stationary.available,
        )
        self._validate_pending_learning(
            stationary.pending,
            role=STATIONARY_MULTITASK_ROLE,
            lifecycle_id=self._config.stationary_lifecycle_id,
            suite_step=step,
            available=stationary.available,
            current_observation=stationary.current_observation,
        )
        if stationary.available and not self._learner_state_valid(
            self._stationary_learner,
            stationary.learner_state,
            stationary.current_observation,
        ):
            raise ValueError("stationary learner state does not own current observation")

        oracle = state.oracle_action_data_upper
        if not isinstance(oracle, _OracleReferenceState):
            raise TypeError("oracle reference state has an invalid type")
        if not oracle.available or oracle.unavailable_reason is not None:
            raise ValueError("oracle action-data reference must remain available")
        _observation(oracle.current_observation, name="oracle current_observation")
        if (
            self._environment_observation(
                self._environments[2],
                oracle.environment_state,
            )
            != oracle.current_observation
        ):
            raise ValueError("oracle environment observation cache is inconsistent")
        self._validate_trace(
            oracle.trace,
            role=ORACLE_ACTION_DATA_ROLE,
            lifecycle_id=self._config.oracle_lifecycle_id,
            suite_step=step,
            available=True,
        )
        pending_expected = step < len(self._protocol.regime_schedule)
        if (oracle.pending is not None) != pending_expected or (
            (oracle.pending_action_scores is not None) != pending_expected
        ):
            raise ValueError("oracle pending score/decision state is inconsistent")
        if oracle.pending is not None:
            scores = cast(tuple[float, ...], oracle.pending_action_scores)
            if len(scores) != self._n_actions:
                raise ValueError("oracle pending action-score width is invalid")
            action = max(
                range(self._n_actions),
                key=lambda index: (scores[index], -index),
            )
            if (
                oracle.pending.observation != oracle.current_observation
                or oracle.pending.action != action
                or oracle.pending.decision_id
                != _generation_decision_id(self._config.oracle_lifecycle_id, step)
            ):
                raise ValueError("oracle pending decision is invalid")

    def _state_payload(self, state: PrivilegedReferenceRunState) -> dict[str, object]:
        self._validate_state(state)
        fresh = state.fresh_per_regime
        stationary = state.stationary_multitask
        oracle = state.oracle_action_data_upper
        return {
            "step": state.step,
            RETAINED_FRESH_PER_REGIME_ROLE: {
                "available": fresh.available,
                "unavailable_reason": fresh.unavailable_reason,
                "environment_state": self._environment_state_payload(
                    self._environments[0],
                    fresh.environment_state,
                ),
                "current_observation": list(fresh.current_observation),
                "learner_states": [
                    {
                        "evaluator_regime_id": regime_id,
                        "state": (
                            None
                            if learner_state is None
                            else self._learner_state_payload(learner, learner_state)
                        ),
                    }
                    for regime_id, learner, learner_state in zip(
                        self._protocol.evaluator_regime_ids,
                        self._fresh_learners,
                        fresh.learner_states,
                        strict=True,
                    )
                ],
                "pending": _pending_to_config(fresh.pending),
                "trace": _trace_to_config(fresh.trace),
                "learner_initialization_count": fresh.learner_initialization_count,
                "ephemeral_probe_initialization_count": (
                    fresh.ephemeral_probe_initialization_count
                ),
            },
            STATIONARY_MULTITASK_ROLE: {
                "available": stationary.available,
                "unavailable_reason": stationary.unavailable_reason,
                "environment_state": self._environment_state_payload(
                    self._environments[1],
                    stationary.environment_state,
                ),
                "current_observation": list(stationary.current_observation),
                "learner_state": self._learner_state_payload(
                    self._stationary_learner,
                    stationary.learner_state,
                ),
                "pending": _pending_to_config(stationary.pending),
                "trace": _trace_to_config(stationary.trace),
                "extra_training_trace": [
                    _extra_record_to_config(record) for record in stationary.extra_training_trace
                ],
            },
            ORACLE_ACTION_DATA_ROLE: {
                "available": oracle.available,
                "unavailable_reason": oracle.unavailable_reason,
                "environment_state": self._environment_state_payload(
                    self._environments[2],
                    oracle.environment_state,
                ),
                "current_observation": list(oracle.current_observation),
                "pending": (
                    None if oracle.pending is None else _decision_to_config(oracle.pending)
                ),
                "pending_action_scores": (
                    None
                    if oracle.pending_action_scores is None
                    else list(oracle.pending_action_scores)
                ),
                "trace": _trace_to_config(oracle.trace),
            },
        }

    @staticmethod
    def _privilege_disclosure(role: str) -> dict[str, object]:
        if role == RETAINED_FRESH_PER_REGIME_ROLE:
            return {
                "evaluator_regime_id_visible_to_privileged_wrapper": True,
                "evaluator_regime_id_visible_to_inner_learner": False,
                "independent_persistent_learner_state_per_regime": True,
                "initialization_policy": "lazy once at first evaluator-regime exposure",
                "fresh_per_segment_or_regime_change": False,
                "same_identity_state_retained_and_reused_on_recurrence": True,
                "determinism_verification_init_per_logical_initialization": 1,
                "recurrence_reset_allowed": False,
                "ephemeral_untrained_probe_initialization": True,
                "additional_training_transitions": 0,
            }
        if role == STATIONARY_MULTITASK_ROLE:
            return {
                "evaluator_regime_id_visible_to_inner_learner": False,
                "reference_stream_regime_label_visible_to_evaluator_only": True,
                "single_persistent_learner_state": True,
                "pretraining_before_common_evaluation": True,
                "learner_init_calls_include_one_determinism_verification": True,
                "frozen_interleaved_extra_stream": True,
                "additional_training_budget_reported_exactly": True,
            }
        if role == ORACLE_ACTION_DATA_ROLE:
            return {
                "implements_continuing_control_learner": False,
                "evaluator_regime_id_visible_to_frozen_callback": True,
                "score_semantics": EXACT_ORACLE_SCORE_SEMANTICS,
                "exact_frozen_counterfactual_outcomes_visible_before_each_outcome": True,
                "stochastic_expected_action_scores_accepted": False,
                "selected_exact_score_checked_against_realized_outcome": True,
                "unselected_exact_scores_trusted_not_outcome_audited": True,
                "held_out_probe_scores_visible_for_action_selection": True,
                "learner_update_or_training_state": False,
            }
        raise ValueError("unknown privileged reference role")

    @staticmethod
    def _comparability_disclosure(role: str) -> dict[str, object]:
        reasons = {
            RETAINED_FRESH_PER_REGIME_ROLE: (
                "multiple privileged regime-selected learner states, each initialized once and "
                "retained across recurrences of its identity, exceed the ordinary single-learner "
                "condition contract; this is not a reset-per-segment comparator"
            ),
            STATIONARY_MULTITASK_ROLE: (
                "frozen additional training data is outside the common evaluation budget"
            ),
            ORACLE_ACTION_DATA_ROLE: (
                "exact frozen pre-outcome counterfactual outcomes for every action are unavailable "
                "to ordinary conditions; stochastic expected-score references are a distinct, "
                "unsupported comparator"
            ),
        }
        return {
            "included_in_ordinary_conditions": False,
            "eligible_as_matched_baseline": False,
            "common_v2_longitudinal_metrics_used_when_available": True,
            "matched_realized_compute_or_memory_claimed": False,
            "genuinely_comparable_under_common_condition_contract": False,
            "reason": reasons[role],
        }

    def _evaluation_usage(self, role: str, trace: _ReferenceTrace) -> dict[str, object]:
        processed = len(trace.actions)
        known = [value for value in trace.backward_call_counts if value is not None]
        backward_available = len(known) == len(trace.backward_call_counts)
        return {
            "common_ceiling_budget": _json_clone(self._budget_config),
            "processed_transitions": processed,
            "decision_calls": processed,
            "environment_calls": processed,
            "update_calls": 0 if role == ORACLE_ACTION_DATA_ROLE else processed,
            "probe_calls": (len(trace.evaluator_matrix) * self._probe_examples_per_checkpoint),
            "backward_call_count_available": backward_available,
            "backward_calls": sum(known) if backward_available else None,
            "stored_wrapper_decision_ids": processed,
            "predict_before_outcome_count": processed,
            "ownership_verified_count": processed,
            "shared_ceiling_is_realized_parity": False,
        }

    @staticmethod
    def _safety_summary(trace: _ReferenceTrace) -> dict[str, object]:
        transitions = [record.transition for record in trace.actions]
        return {
            "safety_violations": sum(item.safety_violation for item in transitions),
            "interventions": sum(item.intervention for item in transitions),
            "near_misses": sum(item.near_miss for item in transitions),
            "cumulative_safety_cost": sum(item.safety_cost for item in transitions),
            "cumulative_near_miss_cost": sum(item.near_miss_cost for item in transitions),
            "maximum_step_safety_cost": max(
                (item.safety_cost for item in transitions),
                default=0.0,
            ),
        }

    def _fresh_resource_usage(self, state: _FreshReferenceState) -> dict[str, object]:
        payloads = [
            self._resource_payload(learner, config, learner_state)
            for learner, config, learner_state in zip(
                self._fresh_learners,
                self._fresh_learner_configs,
                state.learner_states,
                strict=True,
            )
            if learner_state is not None
        ]
        parameter_values = [payload["trainable_parameter_count"] for payload in payloads]
        parameters_available = all(value is not None for value in parameter_values)
        return {
            "persistent_state_bytes": sum(
                cast(int, payload["persistent_state_bytes"]) for payload in payloads
            ),
            "state_scalar_count": sum(
                cast(int, payload["state_scalar_count"]) for payload in payloads
            ),
            "trainable_parameter_count": (
                sum(cast(int, value) for value in parameter_values)
                if parameters_available
                else None
            ),
            "measurement_method": (
                "sum of exact logical usage reported by every initialized per-regime learner"
            ),
            "retained_learner_state_count": len(payloads),
            "common_single_state_ceiling_comparable": False,
        }

    def _stationary_resource_usage(
        self,
        state: _StationaryReferenceState,
    ) -> dict[str, object]:
        payload = self._resource_payload(
            self._stationary_learner,
            self._stationary_learner_config,
            state.learner_state,
        )
        return {
            **payload,
            "retained_learner_state_count": 1,
            "common_single_state_ceiling_comparable": False,
        }

    def _additional_data_usage(
        self,
        role: str,
        state: PrivilegedReferenceRunState,
    ) -> dict[str, object]:
        if role == RETAINED_FRESH_PER_REGIME_ROLE:
            fresh = state.fresh_per_regime
            return {
                "training_transitions": 0,
                "persistent_learner_initializations": (fresh.learner_initialization_count),
                "persistent_learner_init_api_calls": (2 * fresh.learner_initialization_count),
                "ephemeral_probe_initializations": (fresh.ephemeral_probe_initialization_count),
                "ephemeral_probe_init_api_calls": (2 * fresh.ephemeral_probe_initialization_count),
                "recurrence_resets": 0,
                "regime_selector_calls": len(fresh.trace.actions),
            }
        if role == STATIONARY_MULTITASK_ROLE:
            trace = state.stationary_multitask.extra_training_trace
            known = [
                record.backward_call_count
                for record in trace
                if record.backward_call_count is not None
            ]
            return {
                "declared_exact_budget": self._config.extra_data_budget.to_config(),
                "pretraining_completed": True,
                "learner_init_api_calls": 2,
                "training_transitions": len(trace),
                "decision_calls": len(trace),
                "update_calls": len(trace),
                "backward_call_count_available": len(known) == len(trace),
                "backward_calls": sum(known) if len(known) == len(trace) else None,
                "reward_table_scalars_available_to_evaluator": sum(
                    len(example.action_scores) for example in self._stationary_stream.examples
                ),
                "selected_reward_scalars_revealed_to_learner": len(trace),
                "evaluator_only_regime_labels": len(trace),
            }
        if role == ORACLE_ACTION_DATA_ROLE:
            oracle = state.oracle_action_data_upper
            return {
                "declared_exact_budget": self._config.extra_data_budget.to_config(),
                "environment_action_score_callbacks": len(oracle.trace.actions),
                "environment_action_score_scalars": (len(oracle.trace.actions) * self._n_actions),
                "probe_action_score_scalars_used_for_selection": (
                    len(oracle.trace.evaluator_matrix)
                    * self._probe_action_score_scalars_per_checkpoint
                ),
                "learner_training_transitions": 0,
            }
        raise ValueError("unknown privileged reference role")

    def _role_report(
        self,
        role: str,
        state: PrivilegedReferenceRunState,
    ) -> dict[str, object]:
        if role == RETAINED_FRESH_PER_REGIME_ROLE:
            fresh_state = state.fresh_per_regime
            trace = fresh_state.trace
            resource = self._fresh_resource_usage(fresh_state)
            extra_trace: object = None
            available = fresh_state.available
            reason = fresh_state.unavailable_reason
        elif role == STATIONARY_MULTITASK_ROLE:
            stationary_state = state.stationary_multitask
            trace = stationary_state.trace
            resource = self._stationary_resource_usage(stationary_state)
            extra_trace = [
                _extra_record_to_config(record) for record in stationary_state.extra_training_trace
            ]
            available = stationary_state.available
            reason = stationary_state.unavailable_reason
        elif role == ORACLE_ACTION_DATA_ROLE:
            oracle_state = state.oracle_action_data_upper
            trace = oracle_state.trace
            resource = {
                "persistent_state_bytes": 0,
                "state_scalar_count": 0,
                "trainable_parameter_count": 0,
                "measurement_method": "no learner or trainable state",
                "retained_learner_state_count": 0,
                "common_single_state_ceiling_comparable": False,
            }
            extra_trace = None
            available = oracle_state.available
            reason = oracle_state.unavailable_reason
        else:
            raise ValueError("unknown privileged reference role")

        if available:
            rewards = tuple(record.transition.reward for record in trace.actions)
            computed_metrics = control_core._control_metrics(
                rewards,
                trace.evaluator_matrix,
                self._protocol,
            )
            metrics: dict[str, object] | None = computed_metrics
            applicability = _json_clone(computed_metrics["metric_applicability"])
        else:
            metrics = None
            applicability = _unavailable_metric_applicability(
                self._protocol,
                cast(str, reason),
            )
        return {
            "role": role,
            "seed": self._config.seed,
            "available": available,
            "unavailable_reason": reason,
            "privilege_disclosure": self._privilege_disclosure(role),
            "comparability_disclosure": self._comparability_disclosure(role),
            "trace": _trace_to_config(trace),
            "metrics": metrics,
            "metric_applicability": applicability,
            "realized_evaluation_usage": self._evaluation_usage(role, trace),
            "additional_data_usage": self._additional_data_usage(role, state),
            "additional_data_trace": extra_trace,
            "resource_usage": resource,
            "safety": self._safety_summary(trace),
        }

    def build_report(self, state: PrivilegedReferenceRunState) -> dict[str, object]:
        """Build a strict report only after the bounded common schedule ends."""
        self._validate_state(state)
        if state.step != len(self._protocol.regime_schedule):
            raise ValueError("reference report requires a complete common schedule")
        suite_config = self.to_config()
        roles = [self._role_report(role, state) for role in REFERENCE_ROLES]
        report = {
            "schema_version": REFERENCE_REPORT_SCHEMA,
            "acceptance_status": ACCEPTANCE_STATUS,
            "development_only": True,
            "scientific_promotion_allowed": False,
            "accepted_scientific_evidence": False,
            "interpretation": REPORT_INTERPRETATION,
            "metric_definitions": dict(control_core.CONTROL_METRIC_DEFINITIONS),
            "suite_config": suite_config,
            "suite_config_sha256": _digest(suite_config),
            "source_core_sha256": _source_core_hashes(),
            "reference_roles": roles,
            "reference_roles_sha256": _digest(roles),
            "ordinary_conditions_included": False,
            "claim_thresholds_included": False,
            "limitations": list(REFERENCE_LIMITATIONS),
        }
        canonical = cast(dict[str, object], _json_clone(report))
        reconstructed = _reconstruct_reference_report(
            canonical,
            verify_current_source=True,
        )
        if _canonical_json(canonical) != _canonical_json(reconstructed):
            raise RuntimeError("internal privileged reference report is noncanonical")
        return canonical

    def save_checkpoint(
        self,
        state: PrivilegedReferenceRunState,
        path: str | Path,
    ) -> None:
        """Atomically save state only between complete reference transactions."""
        state_payload = self._state_payload(state)
        suite_config = self.to_config()
        payload = {
            "schema_version": REFERENCE_CHECKPOINT_SCHEMA,
            "suite_config": suite_config,
            "suite_config_sha256": _digest(suite_config),
            "source_core_sha256": _source_core_hashes(),
            "state": state_payload,
            "state_sha256": _digest(state_payload),
        }
        _atomic_write_json(payload, path)

    def _state_from_payload(self, value: object) -> PrivilegedReferenceRunState:
        mapping = _mapping(value, name="reference checkpoint state")
        _exact_keys(
            mapping,
            {
                "step",
                RETAINED_FRESH_PER_REGIME_ROLE,
                STATIONARY_MULTITASK_ROLE,
                ORACLE_ACTION_DATA_ROLE,
            },
            name="reference checkpoint state",
        )
        step = _nonnegative_int(mapping["step"], name="reference state.step")

        fresh_raw = _mapping(mapping[RETAINED_FRESH_PER_REGIME_ROLE], name="fresh state")
        _exact_keys(
            fresh_raw,
            {
                "available",
                "unavailable_reason",
                "environment_state",
                "current_observation",
                "learner_states",
                "pending",
                "trace",
                "learner_initialization_count",
                "ephemeral_probe_initialization_count",
            },
            name="fresh state",
        )
        fresh_available = _boolean(fresh_raw["available"], name="fresh.available")
        fresh_reason_raw = fresh_raw["unavailable_reason"]
        fresh_reason = (
            None
            if fresh_reason_raw is None
            else _string(fresh_reason_raw, name="fresh.unavailable_reason")
        )
        raw_learner_states = _list(
            fresh_raw["learner_states"],
            name="fresh.learner_states",
        )
        if len(raw_learner_states) != len(self._fresh_learners):
            raise ValueError("fresh checkpoint learner-state count is invalid")
        fresh_states: list[Any | None] = []
        for index, (raw_state, regime_id, learner) in enumerate(
            zip(
                raw_learner_states,
                self._protocol.evaluator_regime_ids,
                self._fresh_learners,
                strict=True,
            )
        ):
            location = f"fresh.learner_states[{index}]"
            state_mapping = _mapping(raw_state, name=location)
            _exact_keys(
                state_mapping,
                {"evaluator_regime_id", "state"},
                name=location,
            )
            if state_mapping["evaluator_regime_id"] != regime_id:
                raise ValueError("fresh checkpoint regime order is invalid")
            state_payload = state_mapping["state"]
            fresh_states.append(
                None
                if state_payload is None
                else learner.state_from_config(_json_clone(state_payload))
            )
        fresh = _FreshReferenceState(
            available=fresh_available,
            unavailable_reason=fresh_reason,
            environment_state=self._environments[0].state_from_config(
                _json_clone(fresh_raw["environment_state"])
            ),
            current_observation=_observation_from_json(
                fresh_raw["current_observation"],
                name="fresh.current_observation",
            ),
            learner_states=tuple(fresh_states),
            pending=_pending_from_config(fresh_raw["pending"], name="fresh.pending"),
            trace=_trace_from_config(fresh_raw["trace"], name="fresh.trace"),
            learner_initialization_count=_nonnegative_int(
                fresh_raw["learner_initialization_count"],
                name="fresh.learner_initialization_count",
            ),
            ephemeral_probe_initialization_count=_nonnegative_int(
                fresh_raw["ephemeral_probe_initialization_count"],
                name="fresh.ephemeral_probe_initialization_count",
            ),
        )

        stationary_raw = _mapping(
            mapping[STATIONARY_MULTITASK_ROLE],
            name="stationary state",
        )
        _exact_keys(
            stationary_raw,
            {
                "available",
                "unavailable_reason",
                "environment_state",
                "current_observation",
                "learner_state",
                "pending",
                "trace",
                "extra_training_trace",
            },
            name="stationary state",
        )
        stationary_reason_raw = stationary_raw["unavailable_reason"]
        stationary = _StationaryReferenceState(
            available=_boolean(
                stationary_raw["available"],
                name="stationary.available",
            ),
            unavailable_reason=(
                None
                if stationary_reason_raw is None
                else _string(
                    stationary_reason_raw,
                    name="stationary.unavailable_reason",
                )
            ),
            environment_state=self._environments[1].state_from_config(
                _json_clone(stationary_raw["environment_state"])
            ),
            current_observation=_observation_from_json(
                stationary_raw["current_observation"],
                name="stationary.current_observation",
            ),
            learner_state=self._stationary_learner.state_from_config(
                _json_clone(stationary_raw["learner_state"])
            ),
            pending=_pending_from_config(
                stationary_raw["pending"],
                name="stationary.pending",
            ),
            trace=_trace_from_config(
                stationary_raw["trace"],
                name="stationary.trace",
            ),
            extra_training_trace=tuple(
                _extra_record_from_config(
                    item,
                    name=f"stationary.extra_training_trace[{index}]",
                )
                for index, item in enumerate(
                    _list(
                        stationary_raw["extra_training_trace"],
                        name="stationary.extra_training_trace",
                    )
                )
            ),
        )

        oracle_raw = _mapping(
            mapping[ORACLE_ACTION_DATA_ROLE],
            name="oracle state",
        )
        _exact_keys(
            oracle_raw,
            {
                "available",
                "unavailable_reason",
                "environment_state",
                "current_observation",
                "pending",
                "pending_action_scores",
                "trace",
            },
            name="oracle state",
        )
        oracle_reason_raw = oracle_raw["unavailable_reason"]
        oracle_pending_raw = oracle_raw["pending"]
        oracle_scores_raw = oracle_raw["pending_action_scores"]
        oracle = _OracleReferenceState(
            available=_boolean(oracle_raw["available"], name="oracle.available"),
            unavailable_reason=(
                None
                if oracle_reason_raw is None
                else _string(oracle_reason_raw, name="oracle.unavailable_reason")
            ),
            environment_state=self._environments[2].state_from_config(
                _json_clone(oracle_raw["environment_state"])
            ),
            current_observation=_observation_from_json(
                oracle_raw["current_observation"],
                name="oracle.current_observation",
            ),
            pending=(
                None
                if oracle_pending_raw is None
                else _decision_from_config(oracle_pending_raw, name="oracle.pending")
            ),
            pending_action_scores=(
                None
                if oracle_scores_raw is None
                else tuple(
                    _finite_float(item, name="oracle.pending_action_scores")
                    for item in _list(
                        oracle_scores_raw,
                        name="oracle.pending_action_scores",
                    )
                )
            ),
            trace=_trace_from_config(oracle_raw["trace"], name="oracle.trace"),
        )
        state = PrivilegedReferenceRunState(step, fresh, stationary, oracle)
        self._validate_state(state)
        canonical = self._state_payload(state)
        if _canonical_json(canonical) != _canonical_json(mapping):
            raise ValueError("reference checkpoint state is noncanonical")
        return state

    def load_checkpoint(self, path: str | Path) -> PrivilegedReferenceRunState:
        """Load and reconstruct a checkpoint bound to this exact suite."""
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_nonstandard_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
        mapping = _mapping(raw, name="reference checkpoint")
        _exact_keys(
            mapping,
            {
                "schema_version",
                "suite_config",
                "suite_config_sha256",
                "source_core_sha256",
                "state",
                "state_sha256",
            },
            name="reference checkpoint",
        )
        if mapping["schema_version"] != REFERENCE_CHECKPOINT_SCHEMA:
            raise ValueError("reference checkpoint schema_version is invalid")
        suite_config = self.to_config()
        if _canonical_json(mapping["suite_config"]) != _canonical_json(suite_config):
            raise ValueError("reference checkpoint suite config does not match")
        if _sha256(
            mapping["suite_config_sha256"],
            name="checkpoint.suite_config_sha256",
        ) != _digest(suite_config):
            raise ValueError("reference checkpoint suite config digest does not match")
        source_hashes = {
            path_name: _sha256(value, name=f"checkpoint.source_core_sha256.{path_name}")
            for path_name, value in _mapping(
                mapping["source_core_sha256"],
                name="checkpoint.source_core_sha256",
            ).items()
        }
        if source_hashes != _source_core_hashes():
            raise ValueError("reference checkpoint source-core hashes do not match")
        if _sha256(
            mapping["state_sha256"],
            name="checkpoint.state_sha256",
        ) != _digest(mapping["state"]):
            raise ValueError("reference checkpoint state digest does not match")
        return self._state_from_payload(mapping["state"])


def _parse_suite_config(
    value: object,
) -> tuple[
    dict[str, object],
    PrivilegedReferenceRunConfig,
    ContinuingControlProtocol,
    ContinuingControlBudget,
    FrozenStationaryReferenceStream,
    int,
    int,
    int,
]:
    mapping = _mapping(value, name="reference suite config")
    expected_keys = {
        "type",
        "schema_version",
        "report_schema_version",
        "checkpoint_schema_version",
        "run_config",
        "protocol",
        "common_evaluation_budget",
        "environment",
        RETAINED_FRESH_PER_REGIME_ROLE,
        "stationary_multitask",
        ORACLE_ACTION_DATA_ROLE,
        "probe_sha256",
        "probe_examples_per_checkpoint",
        "probe_action_score_scalars_per_checkpoint",
        "reference_roles",
        "ordinary_conditions_included",
        "development_only",
        "scientific_promotion_allowed",
        "accepted_scientific_evidence",
    }
    _exact_keys(mapping, expected_keys, name="reference suite config")
    if mapping["type"] != "PrivilegedContinualControlReferenceSuite":
        raise ValueError("reference suite config type is invalid")
    if mapping["schema_version"] != REFERENCE_SUITE_SCHEMA:
        raise ValueError("reference suite schema_version is invalid")
    if mapping["report_schema_version"] != REFERENCE_REPORT_SCHEMA:
        raise ValueError("reference report schema identity is invalid")
    if mapping["checkpoint_schema_version"] != REFERENCE_CHECKPOINT_SCHEMA:
        raise ValueError("reference checkpoint schema identity is invalid")
    run_config = PrivilegedReferenceRunConfig.from_config(mapping["run_config"])
    protocol, protocol_config = control_core._parse_protocol_config(mapping["protocol"])
    budget, budget_config = control_core._parse_budget_config(mapping["common_evaluation_budget"])
    if budget.transition_limit != len(protocol.regime_schedule):
        raise ValueError("reference common budget transition limit is invalid")

    environment = _mapping(mapping["environment"], name="reference environment identity")
    _exact_keys(
        environment,
        {"factory_seed", "config", "config_sha256", "independent_copy_count"},
        name="reference environment identity",
    )
    if _uint32(environment["factory_seed"], name="environment.factory_seed") != (run_config.seed):
        raise ValueError("environment factory seed does not match suite seed")
    environment_config = _mapping(
        environment["config"],
        name="reference environment config",
    )
    _string(environment_config.get("type"), name="environment config.type")
    _versioned_identifier(
        environment_config.get("schema_version"),
        name="environment config.schema_version",
    )
    if (
        _assert_explicit_seed_fields(
            environment_config,
            seed=run_config.seed,
            name="environment config",
        )
        == 0
    ):
        raise ValueError("environment config does not explicitly bind suite seed")
    if _sha256(
        environment["config_sha256"],
        name="environment.config_sha256",
    ) != _digest(environment_config):
        raise ValueError("environment config digest does not match")
    if (
        _positive_int(
            environment["independent_copy_count"],
            name="environment.independent_copy_count",
        )
        != 3
    ):
        raise ValueError("reference suite must declare three independent environments")

    fresh = _mapping(
        mapping[RETAINED_FRESH_PER_REGIME_ROLE],
        name="fresh reference identity",
    )
    _exact_keys(
        fresh,
        {
            "factory_seed",
            "lifecycle_id",
            "learner_configs",
            "regime_selection_owner",
            "initialization_scope",
            "recurrence_policy",
            "fresh_per_segment_or_change",
        },
        name="fresh reference identity",
    )
    if _uint32(fresh["factory_seed"], name="fresh.factory_seed") != run_config.seed:
        raise ValueError("fresh factory seed does not match suite seed")
    if (
        _decision_id_from_lifecycle_json(
            fresh["lifecycle_id"],
            name="fresh.lifecycle_id",
        )
        != run_config.fresh_lifecycle_id
    ):
        raise ValueError("fresh lifecycle identity does not match run config")
    raw_fresh_configs = _list(fresh["learner_configs"], name="fresh.learner_configs")
    if len(raw_fresh_configs) != len(protocol.evaluator_regime_ids):
        raise ValueError("fresh learner config count does not match protocol regimes")
    parsed_fresh_configs: list[dict[str, object]] = []
    for index, (raw, regime_id) in enumerate(
        zip(raw_fresh_configs, protocol.evaluator_regime_ids, strict=True)
    ):
        location = f"fresh.learner_configs[{index}]"
        identity = _mapping(raw, name=location)
        _exact_keys(
            identity,
            {"evaluator_regime_id", "config", "config_sha256"},
            name=location,
        )
        if identity["evaluator_regime_id"] != regime_id:
            raise ValueError("fresh learner config regime order is invalid")
        learner_config = cast(
            dict[str, object],
            _json_clone(_mapping(identity["config"], name=f"{location}.config")),
        )
        _string(learner_config.get("type"), name=f"{location}.config.type")
        _versioned_identifier(
            learner_config.get("schema_version"),
            name=f"{location}.config.schema_version",
        )
        _string(learner_config.get("name"), name=f"{location}.config.name")
        _assert_explicit_seed_fields(
            learner_config,
            seed=run_config.seed,
            name=f"{location}.config",
        )
        if _sha256(
            identity["config_sha256"],
            name=f"{location}.config_sha256",
        ) != _digest(learner_config):
            raise ValueError("fresh learner config digest does not match")
        parsed_fresh_configs.append(learner_config)
    if fresh["regime_selection_owner"] != "privileged suite wrapper only":
        raise ValueError("fresh regime selector ownership is invalid")
    if fresh["initialization_scope"] != "once per evaluator regime identity":
        raise ValueError("fresh initialization scope is invalid")
    if fresh["recurrence_policy"] != "retain and reuse that regime-identity learner state":
        raise ValueError("fresh recurrence policy is invalid")
    if _boolean(
        fresh["fresh_per_segment_or_change"],
        name="fresh.fresh_per_segment_or_change",
    ):
        raise ValueError("retained fresh reference cannot claim reset-per-segment semantics")

    stationary = _mapping(
        mapping["stationary_multitask"],
        name="stationary reference identity",
    )
    _exact_keys(
        stationary,
        {
            "factory_seed",
            "lifecycle_id",
            "learner_config",
            "learner_config_sha256",
            "stream",
            "stream_sha256",
        },
        name="stationary reference identity",
    )
    if _uint32(stationary["factory_seed"], name="stationary.factory_seed") != (run_config.seed):
        raise ValueError("stationary factory seed does not match suite seed")
    if (
        _decision_id_from_lifecycle_json(
            stationary["lifecycle_id"],
            name="stationary.lifecycle_id",
        )
        != run_config.stationary_lifecycle_id
    ):
        raise ValueError("stationary lifecycle identity does not match run config")
    stationary_learner_config = _mapping(
        stationary["learner_config"],
        name="stationary.learner_config",
    )
    _string(stationary_learner_config.get("type"), name="stationary learner.type")
    _versioned_identifier(
        stationary_learner_config.get("schema_version"),
        name="stationary learner.schema_version",
    )
    _string(stationary_learner_config.get("name"), name="stationary learner.name")
    _assert_explicit_seed_fields(
        stationary_learner_config,
        seed=run_config.seed,
        name="stationary learner",
    )
    if _sha256(
        stationary["learner_config_sha256"],
        name="stationary.learner_config_sha256",
    ) != _digest(stationary_learner_config):
        raise ValueError("stationary learner config digest does not match")
    stream = FrozenStationaryReferenceStream.from_config(stationary["stream"])
    if stream.seed != run_config.seed:
        raise ValueError("stationary stream seed does not match suite seed")
    if set(example.reference_regime_id for example in stream.examples) != set(
        protocol.evaluator_regime_ids
    ):
        raise ValueError("stationary stream regimes do not match protocol")
    if _sha256(
        stationary["stream_sha256"],
        name="stationary.stream_sha256",
    ) != _digest(stream.to_config()):
        raise ValueError("stationary stream digest does not match")

    oracle = _mapping(
        mapping[ORACLE_ACTION_DATA_ROLE],
        name="oracle reference identity",
    )
    _exact_keys(
        oracle,
        {
            "factory_seed",
            "lifecycle_id",
            "source_config",
            "source_config_sha256",
            "score_semantics",
            "callback_temporal_contract",
            "selected_score_realized_reward_equality_required",
            "stochastic_expected_score_source_supported",
        },
        name="oracle reference identity",
    )
    if _uint32(oracle["factory_seed"], name="oracle.factory_seed") != run_config.seed:
        raise ValueError("oracle factory seed does not match suite seed")
    if (
        _decision_id_from_lifecycle_json(
            oracle["lifecycle_id"],
            name="oracle.lifecycle_id",
        )
        != run_config.oracle_lifecycle_id
    ):
        raise ValueError("oracle lifecycle identity does not match run config")
    oracle_config = _mapping(oracle["source_config"], name="oracle.source_config")
    _string(oracle_config.get("type"), name="oracle source.type")
    if oracle_config.get("schema_version") != ORACLE_SOURCE_SCHEMA:
        raise ValueError("oracle source schema_version is invalid")
    if (
        _string(
            oracle_config.get("score_semantics"),
            name="oracle source.score_semantics",
        )
        != EXACT_ORACLE_SCORE_SEMANTICS
    ):
        raise ValueError("oracle source must declare exact frozen counterfactual outcome semantics")
    if (
        _assert_explicit_seed_fields(
            oracle_config,
            seed=run_config.seed,
            name="oracle source",
        )
        == 0
    ):
        raise ValueError("oracle source does not explicitly bind suite seed")
    if _sha256(
        oracle["source_config_sha256"],
        name="oracle.source_config_sha256",
    ) != _digest(oracle_config):
        raise ValueError("oracle source config digest does not match")
    if oracle["score_semantics"] != EXACT_ORACLE_SCORE_SEMANTICS:
        raise ValueError("oracle suite score semantics are invalid")
    if oracle["callback_temporal_contract"] != EXACT_ORACLE_CALLBACK_TEMPORAL_CONTRACT:
        raise ValueError("oracle callback temporal contract is invalid")
    if not _boolean(
        oracle["selected_score_realized_reward_equality_required"],
        name="oracle.selected_score_realized_reward_equality_required",
    ):
        raise ValueError("exact oracle selected-score equality check must remain required")
    if _boolean(
        oracle["stochastic_expected_score_source_supported"],
        name="oracle.stochastic_expected_score_source_supported",
    ):
        raise ValueError("stochastic expected scores cannot be relabeled as exact oracle data")

    _sha256(mapping["probe_sha256"], name="reference suite probe_sha256")
    probe_examples = _positive_int(
        mapping["probe_examples_per_checkpoint"],
        name="probe_examples_per_checkpoint",
    )
    probe_score_scalars = _positive_int(
        mapping["probe_action_score_scalars_per_checkpoint"],
        name="probe_action_score_scalars_per_checkpoint",
    )
    if probe_examples * len(protocol.checkpoint_steps) > budget.probe_call_limit:
        raise ValueError("reference probe calls exceed common budget")
    n_actions = len(stream.examples[0].action_scores)
    if any(len(example.action_scores) != n_actions for example in stream.examples):
        raise ValueError("stationary action-score widths are inconsistent")
    extra = run_config.extra_data_budget
    exact_extra = {
        "stationary_transition_limit": len(stream.examples),
        "stationary_decision_call_limit": len(stream.examples),
        "stationary_update_call_limit": len(stream.examples),
        "stationary_reward_table_scalar_limit": sum(
            len(example.action_scores) for example in stream.examples
        ),
        "oracle_callback_limit": len(protocol.regime_schedule),
        "oracle_action_score_scalar_limit": len(protocol.regime_schedule) * n_actions,
        "oracle_probe_action_score_scalar_limit": (
            len(protocol.checkpoint_steps) * probe_score_scalars
        ),
    }
    for field_name, expected in exact_extra.items():
        if getattr(extra, field_name) != expected:
            raise ValueError(f"suite config {field_name} is not exact")

    if _list(mapping["reference_roles"], name="reference_roles") != list(REFERENCE_ROLES):
        raise ValueError("reference role identity or order is invalid")
    if _boolean(
        mapping["ordinary_conditions_included"],
        name="ordinary_conditions_included",
    ):
        raise ValueError("privileged references cannot enter ordinary conditions")
    if not _boolean(mapping["development_only"], name="development_only"):
        raise ValueError("reference suite must remain development-only")
    if _boolean(
        mapping["scientific_promotion_allowed"],
        name="scientific_promotion_allowed",
    ) or _boolean(
        mapping["accepted_scientific_evidence"],
        name="accepted_scientific_evidence",
    ):
        raise ValueError("reference suite cannot claim scientific promotion or acceptance")

    canonical = cast(dict[str, object], _json_clone(mapping))
    if canonical["protocol"] != protocol_config:
        raise ValueError("reference protocol config is noncanonical")
    if canonical["common_evaluation_budget"] != budget_config:
        raise ValueError("reference common budget config is noncanonical")
    return (
        canonical,
        run_config,
        protocol,
        budget,
        stream,
        n_actions,
        probe_examples,
        probe_score_scalars,
    )


def _decision_id_from_lifecycle_json(value: object, *, name: str) -> tuple[int, int]:
    raw = _list(value, name=name)
    if len(raw) != 2:
        raise ValueError(f"{name} must contain two uint32 words")
    return (
        _uint32(raw[0], name=f"{name}[0]"),
        _uint32(raw[1], name=f"{name}[1]"),
    )


def _validate_report_trace(
    trace: _ReferenceTrace,
    *,
    role: str,
    lifecycle_id: tuple[int, int],
    available: bool,
    protocol: ContinuingControlProtocol,
    budget: ContinuingControlBudget,
    n_actions: int,
    probe_examples: int,
) -> None:
    processed = len(trace.actions)
    total = len(protocol.regime_schedule)
    if available and processed != total:
        raise ValueError(f"{role} available report trace is incomplete")
    if not available and processed >= total:
        raise ValueError(f"{role} unavailable report trace must be a strict prefix")
    expected_rows = sum(checkpoint <= processed for checkpoint in protocol.checkpoint_steps)
    if len(trace.evaluator_matrix) != expected_rows:
        raise ValueError(f"{role} report held-out row count is invalid")
    for row in trace.evaluator_matrix:
        if len(row) != len(protocol.evaluator_regime_ids):
            raise ValueError(f"{role} report held-out row width is invalid")
    expected_backward = 0 if role == ORACLE_ACTION_DATA_ROLE else processed
    if len(trace.backward_call_counts) != expected_backward:
        raise ValueError(f"{role} report backward trace is invalid")
    known_backward = [value for value in trace.backward_call_counts if value is not None]
    if sum(known_backward) > budget.backward_call_limit:
        raise ValueError(f"{role} report backward-call budget is exceeded")
    previous_next: tuple[float, ...] | None = None
    used_ids: set[DecisionId] = set()
    for index, record in enumerate(trace.actions):
        if record.evaluator_regime_id != protocol.regime_schedule[index]:
            raise ValueError(f"{role} report regime schedule is invalid")
        transition = record.transition
        if transition.decision_id != _generation_decision_id(lifecycle_id, index):
            raise ValueError(f"{role} report wrapper decision ID is invalid")
        if transition.decision_id in used_ids:
            raise ValueError(f"{role} report reuses a wrapper decision ID")
        used_ids.add(transition.decision_id)
        if transition.action >= n_actions:
            raise ValueError(f"{role} report action is out of range")
        if previous_next is not None and transition.observation != previous_next:
            raise ValueError(f"{role} report transition trace is discontinuous")
        previous_next = transition.next_decision_observation
        if role == ORACLE_ACTION_DATA_ROLE:
            scores = record.oracle_action_scores
            if record.inner_decision_id is not None or scores is None:
                raise ValueError("oracle report ownership fields are invalid")
            if len(scores) != n_actions:
                raise ValueError("oracle report action-score width is invalid")
            action = max(range(n_actions), key=lambda item: (scores[item], -item))
            if transition.action != action or transition.reward != scores[action]:
                raise ValueError("oracle report action scores do not reconstruct")
        elif record.inner_decision_id is None or record.oracle_action_scores is not None:
            raise ValueError(f"{role} report learner ownership fields are invalid")
    if processed > budget.transition_limit:
        raise ValueError(f"{role} report transition budget exceeded")
    if processed > budget.decision_call_limit or processed > budget.environment_call_limit:
        raise ValueError(f"{role} report evaluation call budget exceeded")
    if role != ORACLE_ACTION_DATA_ROLE and processed > budget.update_call_limit:
        raise ValueError(f"{role} report update budget exceeded")
    if len(trace.evaluator_matrix) * probe_examples > budget.probe_call_limit:
        raise ValueError(f"{role} report probe budget exceeded")
    if processed > budget.stored_decision_id_limit:
        raise ValueError(f"{role} report decision-ID storage budget exceeded")


def _expected_safety(trace: _ReferenceTrace) -> dict[str, object]:
    transitions = [record.transition for record in trace.actions]
    return {
        "safety_violations": sum(item.safety_violation for item in transitions),
        "interventions": sum(item.intervention for item in transitions),
        "near_misses": sum(item.near_miss for item in transitions),
        "cumulative_safety_cost": sum(item.safety_cost for item in transitions),
        "cumulative_near_miss_cost": sum(item.near_miss_cost for item in transitions),
        "maximum_step_safety_cost": max(
            (item.safety_cost for item in transitions),
            default=0.0,
        ),
    }


def _parse_resource_usage(value: object, *, role: str) -> dict[str, object]:
    mapping = _mapping(value, name=f"{role}.resource_usage")
    _exact_keys(
        mapping,
        {
            "persistent_state_bytes",
            "state_scalar_count",
            "trainable_parameter_count",
            "measurement_method",
            "retained_learner_state_count",
            "common_single_state_ceiling_comparable",
        },
        name=f"{role}.resource_usage",
    )
    raw_parameters = mapping["trainable_parameter_count"]
    result: dict[str, object] = {
        "persistent_state_bytes": _nonnegative_int(
            mapping["persistent_state_bytes"],
            name=f"{role}.resource_usage.persistent_state_bytes",
        ),
        "state_scalar_count": _nonnegative_int(
            mapping["state_scalar_count"],
            name=f"{role}.resource_usage.state_scalar_count",
        ),
        "trainable_parameter_count": (
            None
            if raw_parameters is None
            else _nonnegative_int(
                raw_parameters,
                name=f"{role}.resource_usage.trainable_parameter_count",
            )
        ),
        "measurement_method": _string(
            mapping["measurement_method"],
            name=f"{role}.resource_usage.measurement_method",
        ),
        "retained_learner_state_count": _nonnegative_int(
            mapping["retained_learner_state_count"],
            name=f"{role}.resource_usage.retained_learner_state_count",
        ),
        "common_single_state_ceiling_comparable": _boolean(
            mapping["common_single_state_ceiling_comparable"],
            name=f"{role}.resource_usage.common_single_state_ceiling_comparable",
        ),
    }
    if result["common_single_state_ceiling_comparable"] is not False:
        raise ValueError("privileged resource usage cannot claim common comparability")
    if role == ORACLE_ACTION_DATA_ROLE and result != {
        "persistent_state_bytes": 0,
        "state_scalar_count": 0,
        "trainable_parameter_count": 0,
        "measurement_method": "no learner or trainable state",
        "retained_learner_state_count": 0,
        "common_single_state_ceiling_comparable": False,
    }:
        raise ValueError("oracle resource usage is invalid")
    return result


def _expected_evaluation_usage(
    role: str,
    trace: _ReferenceTrace,
    *,
    budget_config: Mapping[str, object],
    probe_examples: int,
) -> dict[str, object]:
    processed = len(trace.actions)
    known = [value for value in trace.backward_call_counts if value is not None]
    available = len(known) == len(trace.backward_call_counts)
    return {
        "common_ceiling_budget": _json_clone(budget_config),
        "processed_transitions": processed,
        "decision_calls": processed,
        "environment_calls": processed,
        "update_calls": 0 if role == ORACLE_ACTION_DATA_ROLE else processed,
        "probe_calls": len(trace.evaluator_matrix) * probe_examples,
        "backward_call_count_available": available,
        "backward_calls": sum(known) if available else None,
        "stored_wrapper_decision_ids": processed,
        "predict_before_outcome_count": processed,
        "ownership_verified_count": processed,
        "shared_ceiling_is_realized_parity": False,
    }


def _parse_stationary_extra_trace(
    value: object,
    *,
    stream: FrozenStationaryReferenceStream,
    n_actions: int,
    backward_limit: int,
) -> list[dict[str, object]]:
    raw = _list(value, name="stationary additional_data_trace")
    if len(raw) != len(stream.examples):
        raise ValueError("stationary additional-data trace is incomplete")
    records = [
        _extra_record_from_config(item, name=f"stationary additional_data_trace[{index}]")
        for index, item in enumerate(raw)
    ]
    used_ids: set[DecisionId] = set()
    for index, (record, example) in enumerate(zip(records, stream.examples, strict=True)):
        transition = record.transition
        if record.reference_regime_id != example.reference_regime_id:
            raise ValueError("stationary additional-data regime order is invalid")
        if transition.decision_id in used_ids:
            raise ValueError("stationary additional-data decision ID is duplicated")
        used_ids.add(transition.decision_id)
        if (
            transition.action >= n_actions
            or transition.observation != example.observation
            or transition.reward != example.action_scores[transition.action]
            or transition.discount != example.discount
            or transition.terminated != example.terminated
            or transition.truncated != example.truncated
            or transition.bootstrap_observation != example.bootstrap_observation
            or transition.reset_observation != example.reset_observation
        ):
            raise ValueError(f"stationary additional-data transition {index} is invalid")
    known = [
        record.backward_call_count for record in records if record.backward_call_count is not None
    ]
    if sum(known) > backward_limit:
        raise ValueError("stationary additional-data backward-call limit is exceeded")
    return [_extra_record_to_config(record) for record in records]


def _expected_additional_usage(
    role: str,
    trace: _ReferenceTrace,
    *,
    protocol: ContinuingControlProtocol,
    run_config: PrivilegedReferenceRunConfig,
    stream: FrozenStationaryReferenceStream,
    n_actions: int,
    probe_score_scalars: int,
    stationary_extra_trace: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    if role == RETAINED_FRESH_PER_REGIME_ROLE:
        processed = len(trace.actions)
        initialized_prefix = min(processed + 1, len(protocol.regime_schedule))
        initialized = len(set(protocol.regime_schedule[:initialized_prefix]))
        ephemeral = sum(
            len(set(protocol.evaluator_regime_ids) - set(protocol.regime_schedule[:checkpoint]))
            for checkpoint in protocol.checkpoint_steps
            if checkpoint <= processed
        )
        return {
            "training_transitions": 0,
            "persistent_learner_initializations": initialized,
            "persistent_learner_init_api_calls": 2 * initialized,
            "ephemeral_probe_initializations": ephemeral,
            "ephemeral_probe_init_api_calls": 2 * ephemeral,
            "recurrence_resets": 0,
            "regime_selector_calls": processed,
        }
    if role == STATIONARY_MULTITASK_ROLE:
        if stationary_extra_trace is None:
            raise ValueError("stationary additional-data trace is required")
        backward_values = [record["backward_call_count"] for record in stationary_extra_trace]
        known = [cast(int, value) for value in backward_values if value is not None]
        available = len(known) == len(backward_values)
        return {
            "declared_exact_budget": run_config.extra_data_budget.to_config(),
            "pretraining_completed": True,
            "learner_init_api_calls": 2,
            "training_transitions": len(stream.examples),
            "decision_calls": len(stream.examples),
            "update_calls": len(stream.examples),
            "backward_call_count_available": available,
            "backward_calls": sum(known) if available else None,
            "reward_table_scalars_available_to_evaluator": sum(
                len(example.action_scores) for example in stream.examples
            ),
            "selected_reward_scalars_revealed_to_learner": len(stream.examples),
            "evaluator_only_regime_labels": len(stream.examples),
        }
    if role == ORACLE_ACTION_DATA_ROLE:
        return {
            "declared_exact_budget": run_config.extra_data_budget.to_config(),
            "environment_action_score_callbacks": len(trace.actions),
            "environment_action_score_scalars": len(trace.actions) * n_actions,
            "probe_action_score_scalars_used_for_selection": (
                len(trace.evaluator_matrix) * probe_score_scalars
            ),
            "learner_training_transitions": 0,
        }
    raise ValueError("unknown privileged reference role")


def _reconstruct_role_report(
    value: object,
    *,
    role: str,
    suite_config: Mapping[str, object],
    run_config: PrivilegedReferenceRunConfig,
    protocol: ContinuingControlProtocol,
    budget: ContinuingControlBudget,
    stream: FrozenStationaryReferenceStream,
    n_actions: int,
    probe_examples: int,
    probe_score_scalars: int,
) -> dict[str, object]:
    mapping = _mapping(value, name=f"reference role {role}")
    _exact_keys(
        mapping,
        {
            "role",
            "seed",
            "available",
            "unavailable_reason",
            "privilege_disclosure",
            "comparability_disclosure",
            "trace",
            "metrics",
            "metric_applicability",
            "realized_evaluation_usage",
            "additional_data_usage",
            "additional_data_trace",
            "resource_usage",
            "safety",
        },
        name=f"reference role {role}",
    )
    if mapping["role"] != role:
        raise ValueError("reference role identity or order is invalid")
    if _uint32(mapping["seed"], name=f"{role}.seed") != run_config.seed:
        raise ValueError(f"{role} seed does not match suite seed")
    available = _boolean(mapping["available"], name=f"{role}.available")
    reason_raw = mapping["unavailable_reason"]
    reason = None if reason_raw is None else _string(reason_raw, name=f"{role}.unavailable_reason")
    if available != (reason is None):
        raise ValueError(f"{role} availability and reason disagree")
    if role == ORACLE_ACTION_DATA_ROLE and not available:
        raise ValueError("oracle action-data reference cannot be unavailable")

    trace = _trace_from_config(mapping["trace"], name=f"{role}.trace")
    lifecycle = {
        RETAINED_FRESH_PER_REGIME_ROLE: run_config.fresh_lifecycle_id,
        STATIONARY_MULTITASK_ROLE: run_config.stationary_lifecycle_id,
        ORACLE_ACTION_DATA_ROLE: run_config.oracle_lifecycle_id,
    }[role]
    _validate_report_trace(
        trace,
        role=role,
        lifecycle_id=lifecycle,
        available=available,
        protocol=protocol,
        budget=budget,
        n_actions=n_actions,
        probe_examples=probe_examples,
    )
    if not available:
        processed = len(trace.actions)
        if role == RETAINED_FRESH_PER_REGIME_ROLE:
            expected_reason = FRESH_UNAVAILABLE_REASON_PREFIX + repr(
                protocol.regime_schedule[processed]
            )
            if reason != expected_reason:
                raise ValueError("fresh reference unavailability reason is invalid")
        elif role == STATIONARY_MULTITASK_ROLE:
            if reason != STATIONARY_UNAVAILABLE_REASON or processed != 0:
                raise ValueError("stationary reference unavailability is invalid")

    expected_privileges = PrivilegedContinualControlReferenceSuite._privilege_disclosure(role)
    if _canonical_json(mapping["privilege_disclosure"]) != _canonical_json(expected_privileges):
        raise ValueError(f"{role} privilege disclosure is invalid")
    expected_comparability = PrivilegedContinualControlReferenceSuite._comparability_disclosure(
        role
    )
    if _canonical_json(mapping["comparability_disclosure"]) != _canonical_json(
        expected_comparability
    ):
        raise ValueError(f"{role} comparability disclosure is invalid")

    if available:
        metrics = control_core._control_metrics(
            tuple(record.transition.reward for record in trace.actions),
            trace.evaluator_matrix,
            protocol,
        )
        applicability = _json_clone(metrics["metric_applicability"])
        if _canonical_json(mapping["metrics"]) != _canonical_json(metrics):
            raise ValueError(f"{role} longitudinal metrics do not reconstruct")
    else:
        metrics = None
        applicability = _unavailable_metric_applicability(
            protocol,
            cast(str, reason),
        )
        if mapping["metrics"] is not None:
            raise ValueError(f"{role} unavailable metrics must remain unavailable")
    if _canonical_json(mapping["metric_applicability"]) != _canonical_json(applicability):
        raise ValueError(f"{role} metric applicability does not reconstruct")

    budget_config = _mapping(
        suite_config["common_evaluation_budget"],
        name="suite common budget",
    )
    expected_evaluation = _expected_evaluation_usage(
        role,
        trace,
        budget_config=budget_config,
        probe_examples=probe_examples,
    )
    if _canonical_json(mapping["realized_evaluation_usage"]) != _canonical_json(
        expected_evaluation
    ):
        raise ValueError(f"{role} realized evaluation usage does not reconstruct")

    extra_trace_raw = mapping["additional_data_trace"]
    parsed_extra_trace: list[dict[str, object]] | None = None
    if role == STATIONARY_MULTITASK_ROLE:
        parsed_extra_trace = _parse_stationary_extra_trace(
            extra_trace_raw,
            stream=stream,
            n_actions=n_actions,
            backward_limit=run_config.extra_data_budget.stationary_backward_call_limit,
        )
    elif extra_trace_raw is not None:
        raise ValueError(f"{role} cannot contain a stationary additional-data trace")
    expected_additional = _expected_additional_usage(
        role,
        trace,
        protocol=protocol,
        run_config=run_config,
        stream=stream,
        n_actions=n_actions,
        probe_score_scalars=probe_score_scalars,
        stationary_extra_trace=parsed_extra_trace,
    )
    if _canonical_json(mapping["additional_data_usage"]) != _canonical_json(expected_additional):
        raise ValueError(f"{role} additional-data usage does not reconstruct")

    resource = _parse_resource_usage(mapping["resource_usage"], role=role)
    if (
        role == RETAINED_FRESH_PER_REGIME_ROLE
        and resource["retained_learner_state_count"]
        != (expected_additional["persistent_learner_initializations"])
    ):
        raise ValueError("fresh resource state count does not match initialization count")
    if role == STATIONARY_MULTITASK_ROLE and resource["retained_learner_state_count"] != 1:
        raise ValueError("stationary resource state count must be one")
    safety = _expected_safety(trace)
    if _canonical_json(mapping["safety"]) != _canonical_json(safety):
        raise ValueError(f"{role} safety summary does not reconstruct")

    reconstructed = {
        "role": role,
        "seed": run_config.seed,
        "available": available,
        "unavailable_reason": reason,
        "privilege_disclosure": expected_privileges,
        "comparability_disclosure": expected_comparability,
        "trace": _trace_to_config(trace),
        "metrics": metrics,
        "metric_applicability": applicability,
        "realized_evaluation_usage": expected_evaluation,
        "additional_data_usage": expected_additional,
        "additional_data_trace": parsed_extra_trace,
        "resource_usage": resource,
        "safety": safety,
    }
    if _canonical_json(reconstructed) != _canonical_json(mapping):
        raise ValueError(f"{role} report contains noncanonical numeric or structural data")
    return cast(dict[str, object], _json_clone(reconstructed))


def _reconstruct_reference_report(
    report: Mapping[str, object],
    *,
    verify_current_source: bool,
) -> dict[str, object]:
    _exact_keys(
        report,
        {
            "schema_version",
            "acceptance_status",
            "development_only",
            "scientific_promotion_allowed",
            "accepted_scientific_evidence",
            "interpretation",
            "metric_definitions",
            "suite_config",
            "suite_config_sha256",
            "source_core_sha256",
            "reference_roles",
            "reference_roles_sha256",
            "ordinary_conditions_included",
            "claim_thresholds_included",
            "limitations",
        },
        name="privileged reference report",
    )
    if report["schema_version"] != REFERENCE_REPORT_SCHEMA:
        raise ValueError("privileged reference report schema_version is invalid")
    if report["acceptance_status"] != ACCEPTANCE_STATUS:
        raise ValueError("privileged reference report must remain not-assessed")
    if not _boolean(report["development_only"], name="report.development_only"):
        raise ValueError("privileged reference report must remain development-only")
    if _boolean(
        report["scientific_promotion_allowed"],
        name="report.scientific_promotion_allowed",
    ) or _boolean(
        report["accepted_scientific_evidence"],
        name="report.accepted_scientific_evidence",
    ):
        raise ValueError("privileged reference report cannot claim promotion or evidence")
    if report["interpretation"] != REPORT_INTERPRETATION:
        raise ValueError("privileged reference report interpretation is invalid")
    if _canonical_json(report["metric_definitions"]) != _canonical_json(
        dict(control_core.CONTROL_METRIC_DEFINITIONS)
    ):
        raise ValueError("privileged reference metric definitions are invalid")

    (
        suite_config,
        run_config,
        protocol,
        budget,
        stream,
        n_actions,
        probe_examples,
        probe_score_scalars,
    ) = _parse_suite_config(report["suite_config"])
    if _sha256(
        report["suite_config_sha256"],
        name="report.suite_config_sha256",
    ) != _digest(suite_config):
        raise ValueError("privileged reference suite config digest does not match")

    raw_source_hashes = _mapping(
        report["source_core_sha256"],
        name="report.source_core_sha256",
    )
    source_hashes = {
        path_name: _sha256(value, name=f"source_core_sha256.{path_name}")
        for path_name, value in raw_source_hashes.items()
    }
    current_hashes = _source_core_hashes()
    if set(source_hashes) != set(current_hashes):
        raise ValueError("privileged reference source-core hash paths are invalid")
    if verify_current_source and source_hashes != current_hashes:
        raise ValueError("privileged reference source-core hashes do not match")

    raw_roles = _list(report["reference_roles"], name="report.reference_roles")
    if len(raw_roles) != len(REFERENCE_ROLES):
        raise ValueError("privileged reference report role count is invalid")
    roles = [
        _reconstruct_role_report(
            raw,
            role=role,
            suite_config=suite_config,
            run_config=run_config,
            protocol=protocol,
            budget=budget,
            stream=stream,
            n_actions=n_actions,
            probe_examples=probe_examples,
            probe_score_scalars=probe_score_scalars,
        )
        for raw, role in zip(raw_roles, REFERENCE_ROLES, strict=True)
    ]
    if _sha256(
        report["reference_roles_sha256"],
        name="report.reference_roles_sha256",
    ) != _digest(roles):
        raise ValueError("privileged reference role digest does not match")
    if _boolean(
        report["ordinary_conditions_included"],
        name="report.ordinary_conditions_included",
    ):
        raise ValueError("privileged references cannot enter ordinary conditions")
    if _boolean(
        report["claim_thresholds_included"],
        name="report.claim_thresholds_included",
    ):
        raise ValueError("privileged reference report cannot include claim thresholds")
    limitations = [
        _string(value, name="report limitations")
        for value in _list(report["limitations"], name="report.limitations")
    ]
    if limitations != list(REFERENCE_LIMITATIONS):
        raise ValueError("privileged reference limitations are invalid or incomplete")

    reconstructed = {
        "schema_version": REFERENCE_REPORT_SCHEMA,
        "acceptance_status": ACCEPTANCE_STATUS,
        "development_only": True,
        "scientific_promotion_allowed": False,
        "accepted_scientific_evidence": False,
        "interpretation": REPORT_INTERPRETATION,
        "metric_definitions": dict(control_core.CONTROL_METRIC_DEFINITIONS),
        "suite_config": suite_config,
        "suite_config_sha256": _digest(suite_config),
        "source_core_sha256": source_hashes,
        "reference_roles": roles,
        "reference_roles_sha256": _digest(roles),
        "ordinary_conditions_included": False,
        "claim_thresholds_included": False,
        "limitations": limitations,
    }
    if _canonical_json(reconstructed) != _canonical_json(report):
        raise ValueError("privileged reference report is noncanonical")
    return cast(dict[str, object], _json_clone(reconstructed))


def validate_privileged_control_reference_report(
    report: Mapping[str, object],
    *,
    verify_current_source: bool = True,
) -> PrivilegedReferenceValidation:
    """Strictly reconstruct identities, traces, metrics, privileges, and budgets."""
    try:
        _reconstruct_reference_report(
            report,
            verify_current_source=verify_current_source,
        )
    except (KeyError, IndexError, TypeError, ValueError) as error:
        return PrivilegedReferenceValidation(False, (str(error),))
    return PrivilegedReferenceValidation(True, ())


def privileged_control_reference_report_json(report: Mapping[str, object]) -> str:
    validation = validate_privileged_control_reference_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid privileged continual-control reference report: " + "; ".join(validation.errors)
        )
    return (
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_json(payload: Mapping[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_privileged_control_reference_report(
    report: Mapping[str, object],
    path: str | Path,
) -> None:
    """Atomically save only a strictly reconstructing reference report."""
    validation = validate_privileged_control_reference_report(report)
    if not validation.valid:
        raise ValueError(
            "invalid privileged continual-control reference report: " + "; ".join(validation.errors)
        )
    _atomic_write_json(report, path)


def load_privileged_control_reference_report(
    path: str | Path,
    *,
    verify_current_source: bool = True,
) -> dict[str, object]:
    """Load strict JSON and reconstruct the complete reference report."""
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    return _reconstruct_reference_report(
        _mapping(raw, name="privileged reference report"),
        verify_current_source=verify_current_source,
    )


def _reject_nonstandard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON numeric constant {value!r}")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


__all__ = [
    "ACCEPTANCE_STATUS",
    "EXACT_ORACLE_CALLBACK_TEMPORAL_CONTRACT",
    "EXACT_ORACLE_SCORE_SEMANTICS",
    "RETAINED_FRESH_PER_REGIME_ROLE",
    "ExactOracleOutcomeSourceFactory",
    "FrozenExactOracleOutcomeSource",
    "FrozenStationaryReferenceStream",
    "ORACLE_ACTION_DATA_ROLE",
    "ORACLE_SOURCE_SCHEMA",
    "PrivilegedContinualControlReferenceSuite",
    "PrivilegedReferenceExtraDataBudget",
    "PrivilegedReferenceRunConfig",
    "PrivilegedReferenceRunState",
    "PrivilegedReferenceValidation",
    "REFERENCE_CHECKPOINT_SCHEMA",
    "REFERENCE_EXTRA_BUDGET_SCHEMA",
    "REFERENCE_REPORT_SCHEMA",
    "REFERENCE_ROLES",
    "REFERENCE_RUN_CONFIG_SCHEMA",
    "REFERENCE_SUITE_SCHEMA",
    "ReferenceEnvironmentFactory",
    "STATIONARY_EXAMPLE_SCHEMA",
    "STATIONARY_MULTITASK_ROLE",
    "STATIONARY_STREAM_SCHEMA",
    "StationaryReferenceExample",
    "StationaryReferenceLearnerFactory",
    "RetainedRegimeIdentityLearnerFactory",
    "load_privileged_control_reference_report",
    "privileged_control_reference_report_json",
    "save_privileged_control_reference_report",
    "validate_privileged_control_reference_report",
]
