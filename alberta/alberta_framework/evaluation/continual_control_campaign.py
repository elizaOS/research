"""Strict paired multi-seed campaigns for continual-control development runs.

Every condition and the environment are explicitly bound to the same campaign
seed through the wrappers in this module.  A campaign executes one complete
:class:`~alberta_framework.evaluation.continual_control_evaluator.ContinualControlEvaluator`
per declared unique seed, retains every raw report, and computes paired
candidate-minus-baseline differences using shared stratified bootstrap draws.

Artifacts are descriptive and always ``not-assessed``.  They carry no
acceptance threshold and cannot promote evidence or a default mechanism.
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
from typing import Any, NoReturn, Protocol, TypeGuard, cast

import numpy as np

import alberta_framework.evaluation.continual_control_evaluator as control_core
from alberta_framework.evaluation.continual_control_evaluator import (
    ContinualControlEvaluator,
    ContinuingControlEnvironment,
    ContinuingControlLearner,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlLearnerUpdate,
    ControlResourceUsage,
    ControlTransition,
    validate_continual_control_report,
)

CAMPAIGN_REPORT_SCHEMA = "alberta.continual_control_campaign_report.v1"
CAMPAIGN_CONFIG_SCHEMA = "alberta.continual_control_campaign_config.v1"
BOOTSTRAP_CONFIG_SCHEMA = "alberta.stratified_paired_bootstrap.v1"
SEEDED_LEARNER_SCHEMA = "alberta.campaign_seeded_control_learner.v1"
SEEDED_ENVIRONMENT_SCHEMA = "alberta.campaign_seeded_control_environment.v1"
ACCEPTANCE_STATUS = "not-assessed"
BOOTSTRAP_METHOD = "sha256-counter-stratified-paired-mean-percentile-type7.v1"
CAMPAIGN_INTERPRETATION = (
    "Development-only paired multi-seed continual-control summary. Confidence intervals "
    "are descriptive bootstrap intervals, not acceptance tests or evidence promotion."
)
CAMPAIGN_LIMITATIONS = (
    "development-only; declared seeds are consumed and are not promotion seeds",
    "bootstrap intervals do not replace preregistration or an untouched held-out protocol",
    "condition seed wrappers bind configuration but cannot audit hidden external side effects",
    "held-out control metrics remain one-action probe scores rather than rollout returns",
    "shared ceilings do not establish equal realized compute or memory",
    "no multiplicity correction, hypothesis threshold, power analysis, or evidence decision",
)

type Direction = str
type ScalarValue = float | None


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


def _strict_float(value: object, *, name: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON float")
    return value


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


def _versioned_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or ".v" not in value:
        raise ValueError(f"{name} must be a versioned identifier")
    return value


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


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


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    resolved = _string(value, name=name)
    if len(resolved) != 64 or any(character not in "0123456789abcdef" for character in resolved):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return resolved


def _source_core_hashes() -> dict[str, str]:
    campaign_path = Path(__file__).resolve()
    control_path = Path(control_core.__file__).resolve()
    return {
        "alberta_framework/evaluation/continual_control_campaign.py": _file_digest(campaign_path),
        "alberta_framework/evaluation/continual_control_evaluator.py": _file_digest(control_path),
    }


@dataclasses.dataclass(frozen=True)
class CampaignSeedStratum:
    """A frozen, ordered partition of declared campaign seeds."""

    name: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("stratum name must be non-empty")
        if not isinstance(self.seeds, tuple) or not self.seeds:
            raise ValueError("stratum seeds must be a non-empty immutable tuple")
        for index, seed in enumerate(self.seeds):
            _uint32(seed, name=f"stratum.seeds[{index}]")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("stratum seeds must be unique")


@dataclasses.dataclass(frozen=True)
class StratifiedBootstrapConfig:
    """Frozen paired bootstrap RNG, strata, interval, and replicate count."""

    bootstrap_seed: int
    resamples: int
    confidence_level: float
    strata: tuple[CampaignSeedStratum, ...]
    schema_version: str = BOOTSTRAP_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BOOTSTRAP_CONFIG_SCHEMA:
            raise ValueError("bootstrap schema_version is invalid")
        _uint32(self.bootstrap_seed, name="bootstrap_seed")
        _positive_int(self.resamples, name="resamples")
        confidence = _strict_float(self.confidence_level, name="confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one")
        if not isinstance(self.strata, tuple) or not self.strata:
            raise ValueError("bootstrap strata must be a non-empty immutable tuple")
        if any(not isinstance(stratum, CampaignSeedStratum) for stratum in self.strata):
            raise TypeError("bootstrap strata must contain CampaignSeedStratum values")
        if len({stratum.name for stratum in self.strata}) != len(self.strata):
            raise ValueError("bootstrap stratum names must be unique")


@dataclasses.dataclass(frozen=True)
class PairedControlCampaignConfig:
    """Immutable campaign seed universe and deterministic bootstrap protocol."""

    campaign_id: str
    declared_seeds: tuple[int, ...]
    bootstrap: StratifiedBootstrapConfig
    schema_version: str = CAMPAIGN_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != CAMPAIGN_CONFIG_SCHEMA:
            raise ValueError("campaign config schema_version is invalid")
        _versioned_identifier(self.campaign_id, name="campaign_id")
        if not isinstance(self.declared_seeds, tuple) or len(self.declared_seeds) < 2:
            raise ValueError(
                "a multi-seed campaign requires an immutable tuple of at least two seeds"
            )
        if not isinstance(self.bootstrap, StratifiedBootstrapConfig):
            raise TypeError("bootstrap must be a StratifiedBootstrapConfig")
        for index, seed in enumerate(self.declared_seeds):
            _uint32(seed, name=f"declared_seeds[{index}]")
        if len(set(self.declared_seeds)) != len(self.declared_seeds):
            raise ValueError("declared campaign seeds must be unique")
        flattened = tuple(seed for stratum in self.bootstrap.strata for seed in stratum.seeds)
        if len(flattened) != len(set(flattened)):
            raise ValueError("bootstrap strata must not overlap")
        if set(flattened) != set(self.declared_seeds):
            raise ValueError("bootstrap strata must partition declared_seeds exactly")
        for stratum in self.bootstrap.strata:
            expected_order = tuple(
                seed for seed in self.declared_seeds if seed in set(stratum.seeds)
            )
            if stratum.seeds != expected_order:
                raise ValueError("stratum seed order must follow declared_seeds order")

    def to_config(self) -> dict[str, object]:
        return cast(dict[str, object], _json_clone(dataclasses.asdict(self)))

    @classmethod
    def from_config(cls, value: object) -> PairedControlCampaignConfig:
        mapping = _mapping(value, name="campaign_config")
        _exact_keys(
            mapping,
            {"campaign_id", "declared_seeds", "bootstrap", "schema_version"},
            name="campaign_config",
        )
        raw_bootstrap = _mapping(mapping["bootstrap"], name="campaign_config.bootstrap")
        _exact_keys(
            raw_bootstrap,
            {
                "bootstrap_seed",
                "resamples",
                "confidence_level",
                "strata",
                "schema_version",
            },
            name="campaign_config.bootstrap",
        )
        raw_strata = _list(raw_bootstrap["strata"], name="campaign_config.bootstrap.strata")
        strata: list[CampaignSeedStratum] = []
        for index, raw in enumerate(raw_strata):
            location = f"campaign_config.bootstrap.strata[{index}]"
            stratum = _mapping(raw, name=location)
            _exact_keys(stratum, {"name", "seeds"}, name=location)
            strata.append(
                CampaignSeedStratum(
                    name=_string(stratum["name"], name=f"{location}.name"),
                    seeds=tuple(
                        _uint32(seed, name=f"{location}.seeds[{seed_index}]")
                        for seed_index, seed in enumerate(
                            _list(stratum["seeds"], name=f"{location}.seeds")
                        )
                    ),
                )
            )
        config = cls(
            campaign_id=_versioned_identifier(
                mapping["campaign_id"],
                name="campaign_config.campaign_id",
            ),
            declared_seeds=tuple(
                _uint32(seed, name=f"campaign_config.declared_seeds[{index}]")
                for index, seed in enumerate(
                    _list(mapping["declared_seeds"], name="campaign_config.declared_seeds")
                )
            ),
            bootstrap=StratifiedBootstrapConfig(
                bootstrap_seed=_uint32(
                    raw_bootstrap["bootstrap_seed"],
                    name="campaign_config.bootstrap.bootstrap_seed",
                ),
                resamples=_positive_int(
                    raw_bootstrap["resamples"],
                    name="campaign_config.bootstrap.resamples",
                ),
                confidence_level=_strict_float(
                    raw_bootstrap["confidence_level"],
                    name="campaign_config.bootstrap.confidence_level",
                ),
                strata=tuple(strata),
                schema_version=_string(
                    raw_bootstrap["schema_version"],
                    name="campaign_config.bootstrap.schema_version",
                ),
            ),
            schema_version=_string(
                mapping["schema_version"],
                name="campaign_config.schema_version",
            ),
        )
        if _canonical_json(config.to_config()) != _canonical_json(mapping):
            raise ValueError("campaign_config is not canonical")
        return config


class CampaignSeededControlLearner:
    """Explicitly bind a learner instance to one campaign seed."""

    def __init__(self, learner: ContinuingControlLearner, *, campaign_seed: int):
        if not isinstance(learner, ContinuingControlLearner):
            raise TypeError("learner must implement ContinuingControlLearner")
        self._learner = learner
        self._campaign_seed = _uint32(campaign_seed, name="campaign_seed")

    @property
    def name(self) -> str:
        return self._learner.name

    @property
    def n_actions(self) -> int:
        return self._learner.n_actions

    @property
    def max_backward_calls_per_update(self) -> int | None:
        return self._learner.max_backward_calls_per_update

    def to_config(self) -> dict[str, object]:
        return {
            "type": "CampaignSeededControlLearner",
            "schema_version": SEEDED_LEARNER_SCHEMA,
            "name": self.name,
            "campaign_seed": self._campaign_seed,
            "learner": _json_clone(self._learner.to_config()),
        }

    def init(self, initial_observation: tuple[float, ...]) -> Any:
        return self._learner.init(initial_observation)

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        return self._learner.state_valid_for_observation(state, observation)

    def decide(self, state: Any, observation: tuple[float, ...]) -> ControlDecision:
        return self._learner.decide(state, observation)

    def update(self, state: Any, transition: ControlTransition) -> ControlLearnerUpdate:
        return self._learner.update(state, transition)

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        return self._learner.probe_action(state, observation)

    def state_to_config(self, state: Any) -> object:
        return self._learner.state_to_config(state)

    def state_from_config(self, payload: object) -> Any:
        return self._learner.state_from_config(payload)

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        return self._learner.resource_usage(state)

    def diagnostics(self, state: Any) -> Mapping[str, bool | int | float | str] | None:
        return self._learner.diagnostics(state)


class CampaignSeededControlEnvironment:
    """Explicitly bind a functional environment instance to one campaign seed."""

    def __init__(
        self,
        environment: ContinuingControlEnvironment,
        *,
        campaign_seed: int,
    ):
        if not isinstance(environment, ContinuingControlEnvironment):
            raise TypeError("environment must implement ContinuingControlEnvironment")
        self._environment = environment
        self._campaign_seed = _uint32(campaign_seed, name="campaign_seed")

    @property
    def n_actions(self) -> int:
        return self._environment.n_actions

    def to_config(self) -> dict[str, object]:
        return {
            "type": "CampaignSeededControlEnvironment",
            "schema_version": SEEDED_ENVIRONMENT_SCHEMA,
            "campaign_seed": self._campaign_seed,
            "environment": _json_clone(self._environment.to_config()),
        }

    def init(self) -> Any:
        return self._environment.init()

    def observation(self, state: Any) -> tuple[float, ...]:
        return self._environment.observation(state)

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate:
        return self._environment.step(state, decision, evaluator_regime_id)

    def state_to_config(self, state: Any) -> object:
        return self._environment.state_to_config(state)

    def state_from_config(self, payload: object) -> Any:
        return self._environment.state_from_config(payload)


class ControlEvaluatorFactory(Protocol):
    def __call__(self, seed: int) -> ContinualControlEvaluator: ...


@dataclasses.dataclass(frozen=True)
class CampaignValidation:
    valid: bool
    errors: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _ScalarMetric:
    name: str
    direction: Direction
    value: ScalarValue


def _condition_records(report: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _mapping(value, name=f"report.conditions[{index}]")
        for index, value in enumerate(_list(report["conditions"], name="report.conditions"))
    )


def _optional_scalar(value: object, *, name: str) -> ScalarValue:
    if value is None:
        return None
    return _finite_float(value, name=name)


def _nested(mapping: Mapping[str, object], *path: str) -> object:
    value: object = mapping
    for index, key in enumerate(path):
        value = _mapping(value, name=".".join(path[:index]) or "metrics")[key]
    return value


def _extract_scalar_metrics(
    condition: Mapping[str, object],
    *,
    higher_is_better: bool,
    regime_ids: tuple[str, ...],
) -> tuple[_ScalarMetric, ...]:
    metrics = _mapping(condition["metrics"], name="condition.metrics")
    return_direction = "higher" if higher_is_better else "lower"
    result: list[_ScalarMetric] = []

    def add(name: str, value: object, direction: Direction) -> None:
        if direction not in {"higher", "lower"}:
            raise ValueError("scalar metric direction is invalid")
        result.append(
            _ScalarMetric(
                name=name,
                direction=direction,
                value=_optional_scalar(value, name=f"metrics.{name}"),
            )
        )

    add("prequential_return", metrics["prequential_return"], return_direction)
    add("lifetime_return", metrics["lifetime_return"], return_direction)
    add(
        "adaptation_auc.mean_normalized_auc",
        _nested(metrics, "adaptation_auc", "mean_normalized_auc"),
        return_direction,
    )
    add(
        "recovery.recovery_rate_over_assessable",
        _nested(metrics, "recovery", "recovery_rate_over_assessable"),
        "higher",
    )
    add(
        "recovery.mean_recovery_steps_over_recovered",
        _nested(metrics, "recovery", "mean_recovery_steps_over_recovered"),
        "lower",
    )
    add(
        "recovery.maximum_recovery_steps",
        _nested(metrics, "recovery", "maximum_recovery_steps"),
        "lower",
    )
    add("mean_final_performance", metrics["mean_final_performance"], return_direction)
    add(
        "forgetting.mean_over_available",
        _nested(metrics, "forgetting", "mean_over_available"),
        "lower",
    )
    add(
        "forgetting.maximum_over_available",
        _nested(metrics, "forgetting", "maximum_over_available"),
        "lower",
    )
    add(
        "backward_transfer.mean_over_available",
        _nested(metrics, "backward_transfer", "mean_over_available"),
        "higher",
    )
    add(
        "forward_transfer.mean_over_available",
        _nested(metrics, "forward_transfer", "mean_over_available"),
        "higher",
    )
    add("stability.mean_gap", _nested(metrics, "stability", "mean_gap"), "lower")
    add("stability.maximum_gap", _nested(metrics, "stability", "maximum_gap"), "lower")
    add(
        "worst_window.mean_return",
        _nested(metrics, "worst_window", "mean_return"),
        return_direction,
    )
    for regime_id in regime_ids:
        add(
            f"per_regime_online_mean_return/{regime_id}",
            _nested(metrics, "per_regime_online_mean_return", regime_id),
            return_direction,
        )
        add(
            f"per_regime_final_performance/{regime_id}",
            _nested(metrics, "per_regime_final_performance", regime_id),
            return_direction,
        )
        add(
            f"forgetting.per_regime/{regime_id}",
            _nested(metrics, "forgetting", "per_regime", regime_id),
            "lower",
        )
        add(
            f"backward_transfer.per_regime/{regime_id}",
            _nested(metrics, "backward_transfer", "per_regime", regime_id),
            "higher",
        )
        add(
            f"forward_transfer.per_regime/{regime_id}",
            _nested(metrics, "forward_transfer", "per_regime", regime_id),
            "higher",
        )
    adaptation = _mapping(metrics["adaptation_auc"], name="metrics.adaptation_auc")
    for record in _list(adaptation["segments"], name="metrics.adaptation_auc.segments"):
        segment = _mapping(record, name="adaptation segment")
        segment_index = _nonnegative_int(segment["segment_index"], name="segment_index")
        add(
            f"adaptation_auc.segment/{segment_index}",
            segment["normalized_auc"],
            return_direction,
        )
    recovery = _mapping(metrics["recovery"], name="metrics.recovery")
    for record in _list(recovery["events"], name="metrics.recovery.events"):
        event = _mapping(record, name="recovery event")
        segment_index = _nonnegative_int(event["segment_index"], name="segment_index")
        add(
            f"recovery.steps.segment/{segment_index}",
            event["recovery_steps"],
            "lower",
        )
    stability = _mapping(metrics["stability"], name="metrics.stability")
    for record in _list(stability["events"], name="metrics.stability.events"):
        event = _mapping(record, name="stability event")
        segment_index = _nonnegative_int(event["segment_index"], name="segment_index")
        add(f"stability.gap.segment/{segment_index}", event["gap"], "lower")
    if len({metric.name for metric in result}) != len(result):
        raise ValueError("scalar metric names must be unique")
    return tuple(result)


def _normalize_seed_config(value: object, *, seed: int) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("config keys must be strings")
            if key in {"seed", "campaign_seed"}:
                if _uint32(item, name=f"config.{key}") != seed:
                    raise ValueError(f"config.{key} does not match declared campaign seed")
                normalized[key] = "<campaign-seed>"
            else:
                normalized[key] = _normalize_seed_config(item, seed=seed)
        return normalized
    if isinstance(value, list):
        return [_normalize_seed_config(item, seed=seed) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("condition config contains a non-finite float")
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise ValueError("condition config contains a non-JSON value")


def _seed_bound_invariants(
    evaluator_config: Mapping[str, object],
    *,
    seed: int,
) -> dict[str, object]:
    _exact_keys(
        evaluator_config,
        {
            "type",
            "schema_version",
            "report_schema_version",
            "run_id",
            "protocol",
            "environment",
            "conditions",
            "budget",
            "probe_sha256",
            "probe_examples_per_checkpoint",
            "latency_measurement_method",
            "development_only",
            "accepted_scientific_evidence",
        },
        name="evaluator config",
    )
    evaluator_type = _string(evaluator_config["type"], name="evaluator.type")
    evaluator_schema = _versioned_identifier(
        evaluator_config["schema_version"],
        name="evaluator.schema_version",
    )
    report_schema = _versioned_identifier(
        evaluator_config["report_schema_version"],
        name="evaluator.report_schema_version",
    )
    if not _boolean(evaluator_config["development_only"], name="development_only"):
        raise ValueError("campaign evaluator must be development-only")
    if _boolean(
        evaluator_config["accepted_scientific_evidence"],
        name="accepted_scientific_evidence",
    ):
        raise ValueError("campaign evaluator cannot claim accepted scientific evidence")
    environment = _mapping(evaluator_config["environment"], name="environment config")
    _exact_keys(
        environment,
        {"type", "schema_version", "campaign_seed", "environment"},
        name="environment seed wrapper",
    )
    if environment.get("type") != "CampaignSeededControlEnvironment":
        raise ValueError("campaign environment must use CampaignSeededControlEnvironment")
    if environment.get("schema_version") != SEEDED_ENVIRONMENT_SCHEMA:
        raise ValueError("campaign environment seed-wrapper schema is invalid")
    if _uint32(environment.get("campaign_seed"), name="environment.campaign_seed") != seed:
        raise ValueError("environment campaign_seed does not match declared seed")
    inner_environment = _mapping(
        environment["environment"],
        name="inner environment config",
    )
    _string(inner_environment.get("type"), name="inner environment.type")
    _versioned_identifier(
        inner_environment.get("schema_version"),
        name="inner environment.schema_version",
    )
    normalized_environment = _normalize_seed_config(environment, seed=seed)

    condition_identities: list[dict[str, object]] = []
    raw_conditions = _list(evaluator_config["conditions"], name="evaluator conditions")
    for index, raw in enumerate(raw_conditions):
        condition = _mapping(raw, name=f"evaluator conditions[{index}]")
        _exact_keys(condition, {"role", "learner"}, name=f"condition[{index}]")
        expected_role = "candidate" if index == 0 else "baseline"
        if condition["role"] != expected_role:
            raise ValueError("campaign condition roles or order are invalid")
        learner = _mapping(condition["learner"], name=f"condition[{index}].learner")
        _exact_keys(
            learner,
            {"type", "schema_version", "name", "campaign_seed", "learner"},
            name=f"condition[{index}].learner",
        )
        if learner.get("type") != "CampaignSeededControlLearner":
            raise ValueError("every campaign condition must use CampaignSeededControlLearner")
        if learner.get("schema_version") != SEEDED_LEARNER_SCHEMA:
            raise ValueError("campaign learner seed-wrapper schema is invalid")
        if _uint32(learner.get("campaign_seed"), name="learner.campaign_seed") != seed:
            raise ValueError("learner campaign_seed does not match declared seed")
        inner = _mapping(learner["learner"], name=f"condition[{index}].inner learner")
        learner_name = _string(learner["name"], name=f"condition[{index}].learner.name")
        learner_type = _string(inner.get("type"), name=f"condition[{index}].inner.type")
        learner_schema = _versioned_identifier(
            inner.get("schema_version"),
            name=f"condition[{index}].inner.schema_version",
        )
        condition_identities.append(
            {
                "role": expected_role,
                "name": learner_name,
                "learner_type": learner_type,
                "learner_schema_version": learner_schema,
                "normalized_learner_config": _normalize_seed_config(
                    learner,
                    seed=seed,
                ),
            }
        )
    condition_names = [identity["name"] for identity in condition_identities]
    if len(condition_names) < 3 or len(set(condition_names)) != len(condition_names):
        raise ValueError("campaign requires one candidate and at least two unique baselines")
    return {
        "evaluator_type": evaluator_type,
        "evaluator_schema_version": evaluator_schema,
        "report_schema_version": report_schema,
        "protocol": _json_clone(evaluator_config["protocol"]),
        "budget": _json_clone(evaluator_config["budget"]),
        "probe_sha256": _sha256(
            evaluator_config["probe_sha256"],
            name="evaluator.probe_sha256",
        ),
        "probe_examples_per_checkpoint": _positive_int(
            evaluator_config["probe_examples_per_checkpoint"],
            name="evaluator.probe_examples_per_checkpoint",
        ),
        "latency_measurement_method": _string(
            evaluator_config["latency_measurement_method"],
            name="evaluator.latency_measurement_method",
        ),
        "development_only": True,
        "accepted_scientific_evidence": False,
        "environment_identity": normalized_environment,
        "condition_identities": condition_identities,
    }


def _condition_scalar_payload(
    report: Mapping[str, object],
) -> tuple[list[dict[str, object]], str]:
    protocol = _mapping(report["protocol"], name="report.protocol")
    higher = _boolean(protocol["higher_is_better"], name="protocol.higher_is_better")
    regime_ids = tuple(
        _string(value, name="protocol.evaluator_regime_ids")
        for value in _list(
            protocol["evaluator_regime_ids"],
            name="protocol.evaluator_regime_ids",
        )
    )
    records = _condition_records(report)
    applicability_values: list[object] = []
    payload: list[dict[str, object]] = []
    expected_names: tuple[str, ...] | None = None
    expected_directions: tuple[str, ...] | None = None
    for condition in records:
        metrics = _mapping(condition["metrics"], name="condition.metrics")
        applicability_values.append(_json_clone(metrics["metric_applicability"]))
        scalars = _extract_scalar_metrics(
            condition,
            higher_is_better=higher,
            regime_ids=regime_ids,
        )
        names = tuple(metric.name for metric in scalars)
        directions = tuple(metric.direction for metric in scalars)
        if expected_names is None:
            expected_names = names
            expected_directions = directions
        elif names != expected_names or directions != expected_directions:
            raise ValueError("condition scalar metric identity or direction mismatch")
        payload.append(
            {
                "condition": condition["name"],
                "metrics": [
                    {
                        "name": metric.name,
                        "direction": metric.direction,
                        "value": metric.value,
                    }
                    for metric in scalars
                ],
            }
        )
    first_applicability = _canonical_json(applicability_values[0])
    if any(_canonical_json(value) != first_applicability for value in applicability_values[1:]):
        raise ValueError("metric applicability mismatch across report conditions")
    return payload, hashlib.sha256(first_applicability.encode("utf-8")).hexdigest()


def _hash_index(
    bootstrap_seed: int,
    *,
    replicate: int,
    stratum: str,
    draw: int,
    size: int,
) -> int:
    payload = _canonical_json(
        {
            "bootstrap_seed": bootstrap_seed,
            "replicate": replicate,
            "stratum": stratum,
            "draw": draw,
        }
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % size


def _type7_quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("bootstrap quantile requires values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _bootstrap_interval(
    differences: Mapping[int, float],
    config: PairedControlCampaignConfig,
) -> dict[str, object]:
    replicate_means: list[float] = []
    for replicate in range(config.bootstrap.resamples):
        sampled: list[float] = []
        for stratum in config.bootstrap.strata:
            for draw in range(len(stratum.seeds)):
                index = _hash_index(
                    config.bootstrap.bootstrap_seed,
                    replicate=replicate,
                    stratum=stratum.name,
                    draw=draw,
                    size=len(stratum.seeds),
                )
                sampled.append(differences[stratum.seeds[index]])
        replicate_means.append(sum(sampled) / len(sampled))
    alpha = (1.0 - config.bootstrap.confidence_level) / 2.0
    return {
        "lower": _type7_quantile(replicate_means, alpha),
        "upper": _type7_quantile(replicate_means, 1.0 - alpha),
        "confidence_level": config.bootstrap.confidence_level,
        "resamples": config.bootstrap.resamples,
        "method": BOOTSTRAP_METHOD,
    }


def _scalar_maps(seed_record: Mapping[str, object]) -> dict[str, dict[str, _ScalarMetric]]:
    result: dict[str, dict[str, _ScalarMetric]] = {}
    conditions = _list(seed_record["scalar_metrics"], name="seed_record.scalar_metrics")
    for index, raw_condition in enumerate(conditions):
        condition = _mapping(raw_condition, name=f"scalar_metrics[{index}]")
        _exact_keys(condition, {"condition", "metrics"}, name=f"scalar_metrics[{index}]")
        condition_name = _string(condition["condition"], name="scalar condition name")
        if condition_name in result:
            raise ValueError("scalar condition names must be unique")
        metrics: dict[str, _ScalarMetric] = {}
        for metric_index, raw_metric in enumerate(
            _list(condition["metrics"], name="scalar condition metrics")
        ):
            metric = _mapping(raw_metric, name=f"scalar metric[{metric_index}]")
            _exact_keys(
                metric,
                {"name", "direction", "value"},
                name=f"scalar metric[{metric_index}]",
            )
            name = _string(metric["name"], name="scalar metric name")
            direction = _string(metric["direction"], name="scalar metric direction")
            if direction not in {"higher", "lower"}:
                raise ValueError("scalar metric direction is invalid")
            if name in metrics:
                raise ValueError("scalar metric names must be unique")
            value = _optional_scalar(metric["value"], name=f"scalar metric {name}")
            metrics[name] = _ScalarMetric(name, direction, value)
        result[condition_name] = metrics
    return result


def _paired_comparisons(
    seed_records: Sequence[Mapping[str, object]],
    config: PairedControlCampaignConfig,
    condition_identities: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observed_seeds = tuple(
        _uint32(record["seed"], name="seed_record.seed") for record in seed_records
    )
    if observed_seeds != config.declared_seeds:
        raise ValueError("paired seed records must exactly follow declared_seeds")
    applicability_hashes = tuple(
        _sha256(
            record["metric_applicability_sha256"],
            name="seed_record.metric_applicability_sha256",
        )
        for record in seed_records
    )
    if len(set(applicability_hashes)) != 1:
        raise ValueError("metric applicability must be identical across campaign seeds")
    if not condition_identities:
        raise ValueError("paired comparisons require condition identities")
    candidate_name = _string(condition_identities[0]["name"], name="candidate name")
    baseline_names = tuple(
        _string(identity["name"], name="baseline name") for identity in condition_identities[1:]
    )
    maps_by_seed = {
        _uint32(record["seed"], name="seed_record.seed"): _scalar_maps(record)
        for record in seed_records
    }
    expected_conditions = (candidate_name, *baseline_names)
    for seed, scalar_map in maps_by_seed.items():
        if tuple(scalar_map) != expected_conditions:
            raise ValueError(f"seed {seed} scalar condition identity or order mismatch")
    first_map = maps_by_seed[config.declared_seeds[0]][candidate_name]
    comparisons: list[dict[str, object]] = []
    for baseline_name in baseline_names:
        metric_records: list[dict[str, object]] = []
        for metric_name, template in first_map.items():
            pairs: list[dict[str, object]] = []
            differences: dict[int, float] = {}
            unavailable_count = 0
            for seed in config.declared_seeds:
                if set(maps_by_seed[seed][candidate_name]) != set(first_map):
                    raise ValueError("candidate scalar metric identity mismatch across seeds")
                if set(maps_by_seed[seed][baseline_name]) != set(first_map):
                    raise ValueError("baseline scalar metric identity mismatch across seeds")
                candidate_metric = maps_by_seed[seed][candidate_name][metric_name]
                baseline_metric = maps_by_seed[seed][baseline_name][metric_name]
                if (
                    candidate_metric.direction != template.direction
                    or baseline_metric.direction != template.direction
                ):
                    raise ValueError("paired scalar metric direction mismatch")
                candidate_value = candidate_metric.value
                baseline_value = baseline_metric.value
                difference: float | None = None
                if candidate_value is not None and baseline_value is not None:
                    difference = _finite_float(
                        (
                            candidate_value - baseline_value
                            if template.direction == "higher"
                            else baseline_value - candidate_value
                        ),
                        name="direction-normalized paired difference",
                    )
                    differences[seed] = difference
                else:
                    unavailable_count += 1
                pairs.append(
                    {
                        "seed": seed,
                        "candidate": candidate_value,
                        "baseline": baseline_value,
                        "direction_normalized_difference": difference,
                    }
                )
            available = unavailable_count == 0
            if available:
                mean_difference: float | None = _finite_float(
                    sum(differences.values()) / len(differences),
                    name="mean direction-normalized paired difference",
                )
                interval: dict[str, object] | None = _bootstrap_interval(
                    differences,
                    config,
                )
                reason: str | None = None
            else:
                mean_difference = None
                interval = None
                all_missing = all(
                    pair["candidate"] is None and pair["baseline"] is None for pair in pairs
                )
                reason = (
                    "metric is explicitly unavailable for candidate and baseline in every seed"
                    if all_missing
                    else "one or more paired seed outcomes are unavailable"
                )
            metric_records.append(
                {
                    "name": metric_name,
                    "direction": template.direction,
                    "available": available,
                    "unavailable_reason": reason,
                    "per_seed": pairs,
                    "mean_direction_normalized_difference": mean_difference,
                    "confidence_interval": interval,
                }
            )
        comparisons.append(
            {
                "baseline": baseline_name,
                "metrics": metric_records,
            }
        )
    return comparisons


def _seed_record(
    *,
    seed: int,
    report: Mapping[str, object],
) -> dict[str, object]:
    scalar_metrics, applicability_hash = _condition_scalar_payload(report)
    return {
        "seed": seed,
        "report_sha256": _digest(report),
        "metric_applicability_sha256": applicability_hash,
        "scalar_metrics": scalar_metrics,
        "report": _json_clone(report),
    }


class PairedContinualControlCampaign:
    """Execute and summarize one complete paired evaluator per declared seed."""

    def __init__(
        self,
        config: PairedControlCampaignConfig,
        evaluator_factory: ControlEvaluatorFactory,
    ):
        if not isinstance(config, PairedControlCampaignConfig):
            raise TypeError("config must be a PairedControlCampaignConfig")
        if not callable(evaluator_factory):
            raise TypeError("evaluator_factory must be callable")
        self._config = PairedControlCampaignConfig.from_config(config.to_config())
        self._factory = evaluator_factory

    @property
    def config(self) -> PairedControlCampaignConfig:
        return self._config

    def run(self) -> dict[str, object]:
        seed_records: list[dict[str, object]] = []
        invariant: dict[str, object] | None = None
        for seed in self._config.declared_seeds:
            evaluator = self._factory(seed)
            if not isinstance(evaluator, ContinualControlEvaluator):
                raise TypeError("evaluator_factory must return ContinualControlEvaluator")
            evaluator_config = evaluator.to_config()
            observed_invariant = _seed_bound_invariants(evaluator_config, seed=seed)
            if invariant is None:
                invariant = observed_invariant
            elif _canonical_json(observed_invariant) != _canonical_json(invariant):
                raise ValueError(
                    "protocol, budget, environment, or condition identity differs across seeds"
                )
            transition_count = _positive_int(
                _mapping(evaluator_config["budget"], name="evaluator budget")["transition_limit"],
                name="transition_limit",
            )
            initial = evaluator.init()
            completed = evaluator.advance(initial, steps=transition_count)
            if completed.step != transition_count:
                raise ValueError("campaign evaluator returned a partial run")
            report = evaluator.build_report(completed)
            validation = validate_continual_control_report(
                report,
                expected_evaluator_config=evaluator_config,
            )
            if not validation.valid:
                raise ValueError(
                    "campaign evaluator report is invalid: " + "; ".join(validation.errors)
                )
            seed_records.append(_seed_record(seed=seed, report=report))
        if invariant is None:
            raise RuntimeError("campaign has no declared seeds")
        condition_identities = cast(
            list[Mapping[str, object]],
            invariant["condition_identities"],
        )
        artifact = {
            "schema_version": CAMPAIGN_REPORT_SCHEMA,
            "acceptance_status": ACCEPTANCE_STATUS,
            "development_only": True,
            "interpretation": CAMPAIGN_INTERPRETATION,
            "campaign_config": self._config.to_config(),
            "campaign_config_sha256": _digest(self._config.to_config()),
            "source_core_sha256": _source_core_hashes(),
            "invariant_evaluator_identity": invariant,
            "invariant_evaluator_identity_sha256": _digest(invariant),
            "seed_runs": seed_records,
            "paired_comparisons": _paired_comparisons(
                seed_records,
                self._config,
                condition_identities,
            ),
            "limitations": list(CAMPAIGN_LIMITATIONS),
            "accepted_scientific_evidence": False,
        }
        canonical = cast(dict[str, object], _json_clone(artifact))
        reconstructed = _reconstruct_campaign_report(
            canonical,
            verify_current_source=True,
        )
        if _canonical_json(canonical) != _canonical_json(reconstructed):
            raise RuntimeError("internal campaign report construction is noncanonical")
        return canonical


def _parse_seed_record(
    value: object,
    *,
    expected_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    record = _mapping(value, name=f"seed_runs[{expected_seed}]")
    _exact_keys(
        record,
        {
            "seed",
            "report_sha256",
            "metric_applicability_sha256",
            "scalar_metrics",
            "report",
        },
        name=f"seed_runs[{expected_seed}]",
    )
    if _uint32(record["seed"], name="seed_run.seed") != expected_seed:
        raise ValueError("seed run order or identity does not match declared_seeds")
    report = _mapping(record["report"], name="seed_run.report")
    validation = validate_continual_control_report(report)
    if not validation.valid:
        raise ValueError(
            "seed run contains an invalid control report: " + "; ".join(validation.errors)
        )
    expected = _seed_record(seed=expected_seed, report=report)
    if _canonical_json(expected) != _canonical_json(record):
        raise ValueError("seed run scalars, applicability, or report digest do not reconstruct")
    _sha256(record["report_sha256"], name="seed_run.report_sha256")
    _sha256(
        record["metric_applicability_sha256"],
        name="seed_run.metric_applicability_sha256",
    )
    return expected, cast(dict[str, object], _json_clone(report))


def _reconstruct_campaign_report(
    artifact: Mapping[str, object],
    *,
    verify_current_source: bool,
) -> dict[str, object]:
    _exact_keys(
        artifact,
        {
            "schema_version",
            "acceptance_status",
            "development_only",
            "interpretation",
            "campaign_config",
            "campaign_config_sha256",
            "source_core_sha256",
            "invariant_evaluator_identity",
            "invariant_evaluator_identity_sha256",
            "seed_runs",
            "paired_comparisons",
            "limitations",
            "accepted_scientific_evidence",
        },
        name="campaign report",
    )
    if artifact["schema_version"] != CAMPAIGN_REPORT_SCHEMA:
        raise ValueError("campaign report schema_version is invalid")
    if artifact["acceptance_status"] != ACCEPTANCE_STATUS:
        raise ValueError("campaign acceptance_status must remain not-assessed")
    if not _boolean(artifact["development_only"], name="development_only"):
        raise ValueError("campaign must remain development-only")
    if artifact["interpretation"] != CAMPAIGN_INTERPRETATION:
        raise ValueError("campaign interpretation is invalid")
    config = PairedControlCampaignConfig.from_config(artifact["campaign_config"])
    config_payload = config.to_config()
    if _sha256(
        artifact["campaign_config_sha256"],
        name="campaign_config_sha256",
    ) != _digest(config_payload):
        raise ValueError("campaign_config_sha256 does not match campaign_config")
    source_hashes = _mapping(artifact["source_core_sha256"], name="source_core_sha256")
    parsed_source_hashes = {
        path: _sha256(value, name=f"source_core_sha256.{path}")
        for path, value in source_hashes.items()
    }
    current_source_hashes = _source_core_hashes()
    if verify_current_source and parsed_source_hashes != current_source_hashes:
        raise ValueError("campaign source-core hashes do not match current source")
    if set(parsed_source_hashes) != set(current_source_hashes):
        raise ValueError("campaign source-core hash paths are invalid")

    raw_seed_runs = _list(artifact["seed_runs"], name="seed_runs")
    if len(raw_seed_runs) != len(config.declared_seeds):
        raise ValueError("campaign has a missing or extra seed run")
    seed_records: list[dict[str, object]] = []
    invariant: dict[str, object] | None = None
    for raw, seed in zip(raw_seed_runs, config.declared_seeds, strict=True):
        seed_record, report = _parse_seed_record(raw, expected_seed=seed)
        evaluator_config = _mapping(report["evaluator_config"], name="evaluator_config")
        observed_invariant = _seed_bound_invariants(evaluator_config, seed=seed)
        if invariant is None:
            invariant = observed_invariant
        elif _canonical_json(observed_invariant) != _canonical_json(invariant):
            raise ValueError(
                "protocol, budget, environment, or condition identity differs across seeds"
            )
        seed_records.append(seed_record)
    if invariant is None:
        raise ValueError("campaign has no seed runs")
    raw_invariant = _mapping(
        artifact["invariant_evaluator_identity"],
        name="invariant_evaluator_identity",
    )
    if _canonical_json(raw_invariant) != _canonical_json(invariant):
        raise ValueError("invariant evaluator identity does not reconstruct")
    if _sha256(
        artifact["invariant_evaluator_identity_sha256"],
        name="invariant_evaluator_identity_sha256",
    ) != _digest(invariant):
        raise ValueError("invariant evaluator identity digest does not match")
    condition_identities = cast(
        list[Mapping[str, object]],
        invariant["condition_identities"],
    )
    paired = _paired_comparisons(seed_records, config, condition_identities)
    if _canonical_json(artifact["paired_comparisons"]) != _canonical_json(paired):
        raise ValueError("paired comparisons or bootstrap intervals do not reconstruct")
    limitations = [
        _string(value, name="campaign limitations")
        for value in _list(artifact["limitations"], name="limitations")
    ]
    if limitations != list(CAMPAIGN_LIMITATIONS):
        raise ValueError("campaign limitations are invalid or incomplete")
    if (
        _boolean(
            artifact["accepted_scientific_evidence"],
            name="accepted_scientific_evidence",
        )
        is not False
    ):
        raise ValueError("campaign cannot claim accepted scientific evidence")
    reconstructed = {
        "schema_version": CAMPAIGN_REPORT_SCHEMA,
        "acceptance_status": ACCEPTANCE_STATUS,
        "development_only": True,
        "interpretation": CAMPAIGN_INTERPRETATION,
        "campaign_config": config_payload,
        "campaign_config_sha256": _digest(config_payload),
        "source_core_sha256": parsed_source_hashes,
        "invariant_evaluator_identity": invariant,
        "invariant_evaluator_identity_sha256": _digest(invariant),
        "seed_runs": seed_records,
        "paired_comparisons": paired,
        "limitations": limitations,
        "accepted_scientific_evidence": False,
    }
    if _canonical_json(reconstructed) != _canonical_json(artifact):
        raise ValueError("campaign report contains a noncanonical numeric or structural payload")
    return cast(dict[str, object], _json_clone(reconstructed))


def validate_continual_control_campaign(
    artifact: Mapping[str, object],
    *,
    verify_current_source: bool = True,
) -> CampaignValidation:
    """Strictly reconstruct a campaign, every raw report, and every paired CI."""
    try:
        _reconstruct_campaign_report(
            artifact,
            verify_current_source=verify_current_source,
        )
    except (KeyError, TypeError, ValueError) as error:
        return CampaignValidation(False, (str(error),))
    return CampaignValidation(True, ())


def continual_control_campaign_json(artifact: Mapping[str, object]) -> str:
    validation = validate_continual_control_campaign(artifact)
    if not validation.valid:
        raise ValueError("invalid continual-control campaign: " + "; ".join(validation.errors))
    return (
        json.dumps(
            artifact,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def save_continual_control_campaign(
    artifact: Mapping[str, object],
    path: str | Path,
) -> None:
    """Atomically save one strictly valid campaign artifact."""
    payload = continual_control_campaign_json(artifact)
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_continual_control_campaign(
    path: str | Path,
    *,
    verify_current_source: bool = True,
) -> dict[str, object]:
    """Load strict JSON and reconstruct every raw report and paired statistic."""
    raw = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    return _reconstruct_campaign_report(
        _mapping(raw, name="campaign report"),
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
    "BOOTSTRAP_CONFIG_SCHEMA",
    "BOOTSTRAP_METHOD",
    "CAMPAIGN_CONFIG_SCHEMA",
    "CAMPAIGN_INTERPRETATION",
    "CAMPAIGN_LIMITATIONS",
    "CAMPAIGN_REPORT_SCHEMA",
    "CampaignSeedStratum",
    "CampaignSeededControlEnvironment",
    "CampaignSeededControlLearner",
    "CampaignValidation",
    "ControlEvaluatorFactory",
    "PairedContinualControlCampaign",
    "PairedControlCampaignConfig",
    "SEEDED_ENVIRONMENT_SCHEMA",
    "SEEDED_LEARNER_SCHEMA",
    "StratifiedBootstrapConfig",
    "continual_control_campaign_json",
    "load_continual_control_campaign",
    "save_continual_control_campaign",
    "validate_continual_control_campaign",
]
