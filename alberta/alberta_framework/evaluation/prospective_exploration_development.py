# mypy: disable-error-code="arg-type,attr-defined,call-arg,no-any-return,type-var"
"""Closed-loop WP5 exploration comparison on consumed development data.

Six comparator arms each own an independent stochastic-trap environment, a
causal online estimator, a :class:`ProspectiveExploration` selector state, and a
caller-owned hard-shield state.  Only evaluator-owned exogenous noise is
paired.  Each estimator derives expected improvement, ensemble disagreement,
information gain, and learning progress from that arm's executed observation
history; no latent progress target, counterfactual reward, or oracle score is
provided.

``ProspectiveExploration`` receives the same four candidates and a permissive
all-true internal mask in every arm.  It therefore ranks candidates but does
not decide actual admissibility in this lane.  The selected candidate crosses
a separate state-dependent hard shield, and only the shield's owned action
receipt can advance the environment.  The resulting observation is then
bound to the exact pre-update estimator revision.

This is a tiny, consumed-data L0 development lane.  Its traces, checkpoint,
and report exist in memory only and are always ``not_assessed``.  Descriptive
returns and action counts select no winner and establish no threshold,
efficacy, safety, deployment, evidence, or promotion claim.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from alberta_framework.core.causal_exploration_estimator import (
    CAUSAL_EXPLORATION_ASSESSMENT_STATUS,
    COLLECT_ACTION,
    INVEST_ACTION,
    N_EXPLORATION_ACTIONS,
    NOISY_TV_ACTION,
    CallerOwnedHardShieldConfig,
    CallerOwnedHardShieldState,
    CausalExplorationEstimator,
    CausalExplorationEstimatorConfig,
    CausalExplorationEstimatorState,
    ExplorationExogenousEvent,
    RankedExplorationDecision,
    StochasticTrapEnvironmentConfig,
    StochasticTrapEnvironmentState,
    apply_caller_owned_hard_shield,
    caller_owned_hard_shield_state_valid,
    initial_caller_owned_hard_shield,
    initial_stochastic_trap_environment,
    measure_causal_exploration_core_resources,
    stochastic_trap_environment_state_valid,
    stochastic_trap_environment_step,
    stochastic_trap_observation,
    stochastic_trap_safety_mask,
)
from alberta_framework.core.prospective_exploration import (
    PROSPECTIVE_EXPLORATION_MODES,
    ExplorationCandidateBatch,
    ProspectiveExploration,
    ProspectiveExplorationConfig,
    ProspectiveExplorationState,
)

PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA = (
    "alberta.prospective-exploration-development.config.v2"
)
PROSPECTIVE_EXPLORATION_PROTOCOL_SCHEMA = (
    "alberta.prospective-exploration-development.protocol.v2"
)
PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA = (
    "alberta.prospective-exploration-development.checkpoint.v2"
)
PROSPECTIVE_EXPLORATION_REPORT_SCHEMA = (
    "alberta.prospective-exploration-development.report.v2"
)

DEVELOPMENT_STATUS = "not_assessed"
ASSESSMENT_STATUS = "not_assessed"
OUTPUT_WRITES = False
ARTIFACT_WRITER_AVAILABLE = False
ACTION_DISPATCH_AUTHORITY = False
PHYSICAL_SAFETY_CLAIM = False
DEPLOYMENT_AUTHORITY = False
EVIDENCE_CLAIMED = False
PROMOTION_AUTHORITY = False
SCIENTIFIC_PROMOTION_ALLOWED = False

FIXED_SEED = 5_821
FIXED_SELECTOR_SEED = 103
FIXED_HORIZON = 8
FIXED_CHECKPOINT_SPLIT = 3
FIXED_DELAYED_INVESTMENTS = 3
FIXED_ENSEMBLE_SIZE = 3
FIXED_EPSILON = 0.25
FIXED_METRIC_CAP = 10.0
FIXED_CANDIDATE_BUDGET = N_EXPLORATION_ACTIONS

ModeName = Literal[
    "expected_improvement_surprisal",
    "random",
    "epsilon_greedy",
    "ensemble_disagreement",
    "information_gain",
    "learning_progress",
]
MODE_ORDER: tuple[ModeName, ...] = PROSPECTIVE_EXPLORATION_MODES

_UINT32_MAX = 2**32 - 1
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
_MAX_REPORT_BYTES = 32 * 1024 * 1024

_LIMITATIONS = (
    "finite consumed development data only; all outcomes remain not_assessed",
    "descriptive comparator returns and action counts are not acceptance thresholds",
    "the tabular-action linear TD ensemble is a bounded estimator, not exact sequential VOI",
    "the synthetic noisy-TV channel and delayed opportunity are not a deployment environment",
    "paired exogenous noise does not pair endogenous trajectories or learned estimates",
    "logical operation counts and array bytes do not measure wall-clock, energy, or "
    "allocator peaks",
    "the hard simulation shield is not a physical-safety certificate",
    "canonical hashes detect accidental changes but provide no authentication",
    "exact replay covers the declared Python lane, not a transitive build or hardware attestation",
    "no result grants policy, dispatch, safety, deployment, evidence, or promotion authority",
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


def _strict_json_equal(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None
    if type(expected) in {bool, int, float, str}:
        return type(actual) is type(expected) and actual == expected
    if type(expected) is list:
        return (
            type(actual) is list
            and len(cast(list[object], actual)) == len(cast(list[object], expected))
            and all(
                _strict_json_equal(left, right)
                for left, right in zip(
                    cast(list[object], actual),
                    cast(list[object], expected),
                    strict=True,
                )
            )
        )
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_json_equal(actual[name], expected[name]) for name in expected)
        )
    return False


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an object with exact string keys")
    return cast(Mapping[str, object], value)


def _exact_int(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int = _UINT32_MAX,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an exact integer in [{minimum}, {maximum}]")
    return value


def _exact_float(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be an exact finite float")
    return value


def _as_list(value: object) -> list[object]:
    materialized = np.asarray(jax.device_get(value))
    result = materialized.tolist()
    if type(result) is not list:
        raise TypeError("array payload must have positive rank")
    return cast(list[object], result)


def _as_float(value: object) -> float:
    result = float(np.asarray(jax.device_get(value)))
    if not math.isfinite(result):
        raise ValueError("trace scalar must be finite")
    return result


def _as_int(value: object) -> int:
    return int(np.asarray(jax.device_get(value)))


def _as_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value)))


def _owner_digest(mode_index: int, role: int) -> tuple[int, ...]:
    """Create a readable, collision-free development owner identity."""

    if not 0 <= mode_index <= len(MODE_ORDER):
        raise ValueError("mode_index is outside the development arm range")
    if not 1 <= role <= 32:
        raise ValueError("role is outside the development owner range")
    prefix = 0xCA500000 + 0x100 * mode_index + role
    return (prefix, role, mode_index + 1, 0xE71A, 0x5821, 0, 0, 1)


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveExplorationDevelopmentConfig:
    """Frozen cheap consumed-data protocol; it exposes no outcome threshold."""

    seed: int = FIXED_SEED
    selector_seed: int = FIXED_SELECTOR_SEED
    horizon: int = FIXED_HORIZON
    checkpoint_split: int = FIXED_CHECKPOINT_SPLIT
    delayed_investments_required: int = FIXED_DELAYED_INVESTMENTS
    ensemble_size: int = FIXED_ENSEMBLE_SIZE
    epsilon: float = FIXED_EPSILON
    metric_cap: float = FIXED_METRIC_CAP

    def __post_init__(self) -> None:
        frozen = {
            "seed": FIXED_SEED,
            "selector_seed": FIXED_SELECTOR_SEED,
            "horizon": FIXED_HORIZON,
            "checkpoint_split": FIXED_CHECKPOINT_SPLIT,
            "delayed_investments_required": FIXED_DELAYED_INVESTMENTS,
            "ensemble_size": FIXED_ENSEMBLE_SIZE,
        }
        for name, expected in frozen.items():
            if type(getattr(self, name)) is not int or getattr(self, name) != expected:
                raise ValueError(f"{name} is frozen at {expected}")
        frozen_floats = {"epsilon": FIXED_EPSILON, "metric_cap": FIXED_METRIC_CAP}
        for name, float_expected in frozen_floats.items():
            if _exact_float(getattr(self, name), name=name) != float_expected:
                raise ValueError(f"{name} is frozen at {float_expected}")
        if not 0 < self.checkpoint_split < self.horizon:
            raise ValueError("checkpoint_split must be internal to the horizon")

    def to_config(self) -> dict[str, object]:
        return {
            "schema": PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA,
            "seed": self.seed,
            "selector_seed": self.selector_seed,
            "horizon": self.horizon,
            "checkpoint_split": self.checkpoint_split,
            "delayed_investments_required": self.delayed_investments_required,
            "ensemble_size": self.ensemble_size,
            "epsilon": self.epsilon,
            "metric_cap": self.metric_cap,
            "n_actions": N_EXPLORATION_ACTIONS,
            "candidate_budget": FIXED_CANDIDATE_BUDGET,
            "modes": list(MODE_ORDER),
            "assessment_status": ASSESSMENT_STATUS,
            "development_data_consumed": True,
            "untouched_held_out_data": False,
            "thresholds": [],
            "output_path": None,
            "output_writes": False,
            "artifact_writer_available": False,
            "action_dispatch_authority": False,
            "physical_safety_claim": False,
            "deployment_authority": False,
            "evidence_claimed": False,
            "promotion_authority": False,
            "scientific_promotion_allowed": False,
        }

    @classmethod
    def from_config(
        cls,
        value: Mapping[str, object],
    ) -> ProspectiveExplorationDevelopmentConfig:
        raw = _mapping(value, name="config")
        expected = cls().to_config()
        if set(raw) != set(expected):
            raise ValueError("prospective exploration config fields differ")
        constructor = {
            name: raw[name]
            for name in (
                "seed",
                "selector_seed",
                "horizon",
                "checkpoint_split",
                "delayed_investments_required",
                "ensemble_size",
                "epsilon",
                "metric_cap",
            )
        }
        result = cls(**constructor)
        if not _strict_json_equal(raw, result.to_config()):
            raise ValueError("prospective exploration fixed config fields differ")
        return result


def prospective_exploration_protocol(
    config: ProspectiveExplorationDevelopmentConfig,
) -> dict[str, object]:
    """Describe the nonpromoting comparison before inspecting its outcomes."""

    if type(config) is not ProspectiveExplorationDevelopmentConfig:
        raise TypeError("config must be an exact ProspectiveExplorationDevelopmentConfig")
    return {
        "schema": PROSPECTIVE_EXPLORATION_PROTOCOL_SCHEMA,
        "config": config.to_config(),
        "environment": {
            "kind": "continuing stochastic-trap/noisy-TV plus delayed-benefit world",
            "actions": ["stabilize", "noisy_tv", "invest", "collect"],
            "delayed_investments_required": config.delayed_investments_required,
            "noisy_tv_channel_is_exogenous_and_unpredictable": True,
            "noisy_tv_resets_delayed_progress": True,
            "invest_has_immediate_cost": True,
            "collect_requires_prior_investments": True,
            "independent_environment_state_per_comparator_arm": True,
            "paired_scope": "evaluator-owned exogenous noise only",
        },
        "score_estimation": {
            "method": "online action-conditioned linear TD ensemble",
            "inputs": [
                "current observation",
                "executed action",
                "observed reward",
                "next observation",
            ],
            "expected_improvement": "Gaussian ensemble proxy against current predicted best",
            "ensemble_disagreement": "standard deviation of learned head predictions",
            "information_gain": "feature-power over observation-updated action precision",
            "learning_progress": "positive recent TD-error reduction from fast versus slow traces",
            "long_horizon_mechanism": "discounted TD bootstrap from observed next state",
            "counterfactual_reward_input": False,
            "latent_progress_target_input": False,
            "oracle_score_input": False,
        },
        "ranking_and_safety_boundary": {
            "selector": "ProspectiveExploration",
            "selector_internal_mask": "fixed all true for ranking only",
            "selector_owns_actual_action_admissibility": False,
            "actual_admissibility_owner": "caller-owned hard shield",
            "shield_applied_after_ranking_before_environment_execution": True,
            "certified_simulation_fallback_action": "stabilize",
            "environment_accepts_only_shield_action_receipt": True,
        },
        "comparators": list(MODE_ORDER),
        "matching": {
            "candidate_slots_per_decision": FIXED_CANDIDATE_BUDGET,
            "selector_logical_uniform_draws_per_decision": FIXED_CANDIDATE_BUDGET + 1,
            "estimator_update_opportunities_per_event": 1,
            "environment_step_opportunities_per_event": 1,
            "same_selector_key_and_draw_schedule": True,
            "same_estimator_initialization_key_and_draw_count": True,
            "same_candidate_actions_and_order": True,
            "same_paired_exogenous_schedule": True,
        },
        "assessment_status": ASSESSMENT_STATUS,
        "thresholds": [],
        "winner_selection": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "output_path": None,
        "artifact_writer_available": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
    }


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveExplorationArmState:
    """One independent comparator trajectory and its causal raw record chain."""

    mode: ModeName
    environment: StochasticTrapEnvironmentState
    estimator: CausalExplorationEstimatorState
    selector: ProspectiveExplorationState
    shield: CallerOwnedHardShieldState
    cumulative_reward: float
    records_json: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveExplorationRunState:
    """Full host-orchestrated composite state for in-memory checkpoints."""

    event_index: int
    arms: tuple[ProspectiveExplorationArmState, ...]
    integrity_sha256: str


@dataclasses.dataclass(frozen=True, slots=True)
class ProspectiveExplorationValidationReceipt:
    """Strict deterministic report-validation receipt without promotion authority."""

    valid: bool
    assessment_status: str
    exact_causal_replay: bool
    checkpoint_resume_exact: bool
    raw_trace_reconstructable: bool
    matched_budgets: bool
    independent_environment_owners: bool
    output_written: bool
    promotion_authority: bool


class ProspectiveExplorationDevelopmentEvaluator:
    """Host-only orchestrator for the six-arm closed-loop development lane."""

    def __init__(self, config: ProspectiveExplorationDevelopmentConfig) -> None:
        if type(config) is not ProspectiveExplorationDevelopmentConfig:
            raise TypeError("config must be an exact ProspectiveExplorationDevelopmentConfig")
        self.config = config

    @staticmethod
    def _mode_index(mode: ModeName) -> int:
        return MODE_ORDER.index(mode) + 1

    def _estimator_config(self, mode: ModeName) -> CausalExplorationEstimatorConfig:
        index = self._mode_index(mode)
        return CausalExplorationEstimatorConfig(
            ensemble_size=self.config.ensemble_size,
            discount=0.95,
            step_size=0.18,
            prior_scale=0.04,
            fast_error_rate=0.4,
            slow_error_rate=0.08,
            weight_clip=10.0,
            metric_cap=self.config.metric_cap,
            estimator_owner_digest=_owner_digest(index, 1),
            action_owner_digest=_owner_digest(index, 2),
            decision_owner_digest=_owner_digest(index, 3),
            environment_owner_digest=_owner_digest(index, 4),
        )

    def _environment_config(self, mode: ModeName) -> StochasticTrapEnvironmentConfig:
        index = self._mode_index(mode)
        return StochasticTrapEnvironmentConfig(
            delayed_investments_required=self.config.delayed_investments_required,
            stabilize_reward=0.08,
            invest_cost=-0.04,
            collect_reward=0.9,
            observation_noise_scale=5.0,
            reward_noise_scale=0.01,
            schedule_owner_digest=_owner_digest(0, 1),
            environment_owner_digest=_owner_digest(index, 4),
            estimator_owner_digest=_owner_digest(index, 1),
            action_owner_digest=_owner_digest(index, 2),
            decision_owner_digest=_owner_digest(index, 3),
            shield_owner_digest=_owner_digest(index, 5),
        )

    def _shield_config(self, mode: ModeName) -> CallerOwnedHardShieldConfig:
        index = self._mode_index(mode)
        return CallerOwnedHardShieldConfig(
            estimator_owner_digest=_owner_digest(index, 1),
            action_owner_digest=_owner_digest(index, 2),
            decision_owner_digest=_owner_digest(index, 3),
            shield_owner_digest=_owner_digest(index, 5),
        )

    def _selector_config(self, mode: ModeName) -> ProspectiveExplorationConfig:
        index = self._mode_index(mode)
        return ProspectiveExplorationConfig(
            n_actions=N_EXPLORATION_ACTIONS,
            candidate_budget=FIXED_CANDIDATE_BUDGET,
            mode=mode,
            epsilon=self.config.epsilon if mode == "epsilon_greedy" else 0.0,
            host_surprisal_cap=6.0,
            max_expected_improvement=self.config.metric_cap,
            max_ensemble_disagreement=self.config.metric_cap,
            max_information_gain=self.config.metric_cap,
            max_learning_progress=self.config.metric_cap,
            source_owner_digest=_owner_digest(index, 6),
            host_policy_owner_digest=_owner_digest(index, 7),
            candidate_owner_digest=_owner_digest(index, 8),
            score_owner_digest=_owner_digest(index, 9),
            safety_owner_digest=_owner_digest(index, 10),
        )

    def common_event(self, event_index: int) -> ExplorationExogenousEvent:
        """Materialize one paired evaluator-owned event from typed Threefry keys."""

        _exact_int(
            event_index,
            name="event_index",
            maximum=self.config.horizon - 1,
        )
        root = jr.key(self.config.seed, impl="threefry2x32")
        event_key = jr.fold_in(root, np.uint32(event_index + 1))
        stable_key, reward_key, tv_key = jr.split(event_key, 3)
        return ExplorationExogenousEvent(
            source_event_words=jnp.asarray((0, event_index + 1), dtype=jnp.uint32),
            stable_noise=jr.normal(stable_key, (), dtype=jnp.float32),
            reward_noise=jr.normal(reward_key, (), dtype=jnp.float32),
            noisy_tv_noise=jr.normal(tv_key, (), dtype=jnp.float32),
            schedule_owner_digest=jnp.asarray(_owner_digest(0, 1), dtype=jnp.uint32),
        )

    def _initial_arm(self, mode: ModeName) -> ProspectiveExplorationArmState:
        estimator_config = self._estimator_config(mode)
        estimator = CausalExplorationEstimator(estimator_config)
        shared_estimator_key = jr.key(self.config.seed + 101, impl="threefry2x32")
        selector = ProspectiveExploration(self._selector_config(mode))
        shared_selector_key = jr.key(self.config.selector_seed, impl="threefry2x32")
        return ProspectiveExplorationArmState(
            mode=mode,
            environment=initial_stochastic_trap_environment(
                self._environment_config(mode)
            ),
            estimator=estimator.init(shared_estimator_key),
            selector=selector.init(shared_selector_key),
            shield=initial_caller_owned_hard_shield(self._shield_config(mode)),
            cumulative_reward=0.0,
            records_json=(),
        )

    def initial_state(self) -> ProspectiveExplorationRunState:
        state = ProspectiveExplorationRunState(
            event_index=0,
            arms=tuple(self._initial_arm(mode) for mode in MODE_ORDER),
            integrity_sha256="",
        )
        return self._seal_state(state)

    @staticmethod
    def _estimator_payload(state: CausalExplorationEstimatorState) -> dict[str, object]:
        return {
            "weights": _as_list(state.weights),
            "action_counts": _as_list(state.action_counts),
            "action_precision": _as_list(state.action_precision),
            "fast_absolute_td_error": _as_list(state.fast_absolute_td_error),
            "slow_absolute_td_error": _as_list(state.slow_absolute_td_error),
            "revision_words": _as_list(state.revision_words),
            "last_source_event_words": _as_list(state.last_source_event_words),
            "last_decision_words": _as_list(state.last_decision_words),
            "rng_key_words": _as_list(jr.key_data(state.rng_key)),
            "estimator_owner_digest": _as_list(state.estimator_owner_digest),
        }

    @staticmethod
    def _environment_payload(state: StochasticTrapEnvironmentState) -> dict[str, object]:
        return {
            "delayed_progress": _as_int(state.delayed_progress),
            "previous_reward": _as_float(state.previous_reward),
            "noisy_tv_channel": _as_float(state.noisy_tv_channel),
            "stable_signal": _as_float(state.stable_signal),
            "event_words": _as_list(state.event_words),
            "last_decision_words": _as_list(state.last_decision_words),
            "last_estimator_revision_words": _as_list(
                state.last_estimator_revision_words
            ),
            "environment_owner_digest": _as_list(state.environment_owner_digest),
        }

    @staticmethod
    def _selector_payload(state: ProspectiveExplorationState) -> dict[str, object]:
        return {
            "rng_key_words": _as_list(jr.key_data(state.rng_key)),
            "decision_words": _as_list(state.decision_words),
            "last_source_event_words": _as_list(state.last_source_event_words),
            "last_host_policy_revision_words": _as_list(
                state.last_host_policy_revision_words
            ),
            "last_candidate_revision_words": _as_list(
                state.last_candidate_revision_words
            ),
            "last_score_revision_words": _as_list(state.last_score_revision_words),
            "last_safety_revision_words": _as_list(state.last_safety_revision_words),
        }

    @staticmethod
    def _shield_payload(state: CallerOwnedHardShieldState) -> dict[str, object]:
        return {
            "revision_words": _as_list(state.revision_words),
            "last_source_event_words": _as_list(state.last_source_event_words),
            "last_decision_words": _as_list(state.last_decision_words),
            "last_estimator_revision_words": _as_list(
                state.last_estimator_revision_words
            ),
            "shield_owner_digest": _as_list(state.shield_owner_digest),
        }

    def _state_body(self, state: ProspectiveExplorationRunState) -> dict[str, object]:
        return {
            "event_index": state.event_index,
            "arms": [
                {
                    "mode": arm.mode,
                    "environment": self._environment_payload(arm.environment),
                    "estimator": self._estimator_payload(arm.estimator),
                    "selector": self._selector_payload(arm.selector),
                    "shield": self._shield_payload(arm.shield),
                    "cumulative_reward": arm.cumulative_reward,
                    "records_json": list(arm.records_json),
                }
                for arm in state.arms
            ],
        }

    def _seal_state(
        self,
        state: ProspectiveExplorationRunState,
    ) -> ProspectiveExplorationRunState:
        return dataclasses.replace(
            state,
            integrity_sha256=_canonical_sha256(self._state_body(state)),
        )

    def _arm_records_valid(
        self,
        arm: ProspectiveExplorationArmState,
        event_index: int,
    ) -> bool:
        """Validate the canonical per-arm hash chain and reconstruct its return."""

        parent = "0" * 64
        reconstructed_return = 0.0
        owner_index = self._mode_index(arm.mode)
        for index, encoded in enumerate(arm.records_json):
            if type(encoded) is not str:
                return False
            try:
                decoded = json.loads(encoded)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            if type(decoded) is not dict:
                return False
            record = cast(dict[str, object], decoded)
            if encoded != _canonical_json_bytes(record).decode("utf-8"):
                return False
            required = {
                "schema",
                "mode",
                "event_index",
                "source_event_words",
                "environment_owner_digest",
                "causal_estimates",
                "ranking",
                "caller_owned_hard_shield",
                "observed_transition",
                "environment_post",
                "estimator_update",
                "physical_action_dispatched",
                "causal_parent_sha256",
                "record_sha256",
            }
            if not required <= set(record):
                return False
            body = {name: record[name] for name in record if name != "record_sha256"}
            if record["record_sha256"] != _canonical_sha256(body):
                return False
            if record["causal_parent_sha256"] != parent:
                return False
            record_sha256 = record["record_sha256"]
            if type(record_sha256) is not str:
                return False
            parent = record_sha256
            expected_words = [0, index + 1]
            previous_words = [0, index]
            if (
                record["schema"]
                != "alberta.prospective-exploration-development.trace.v2"
                or record["mode"] != arm.mode
                or record["event_index"] != index
                or record["source_event_words"] != expected_words
                or record["environment_owner_digest"]
                != list(_owner_digest(owner_index, 4))
                or record["physical_action_dispatched"] is not False
            ):
                return False
            estimate = _mapping(record["causal_estimates"], name="causal estimates")
            ranking = _mapping(record["ranking"], name="ranking")
            shield = _mapping(
                record["caller_owned_hard_shield"], name="hard shield"
            )
            transition = _mapping(
                record["observed_transition"], name="observed transition"
            )
            environment_post = _mapping(
                record["environment_post"], name="environment post"
            )
            update = _mapping(record["estimator_update"], name="estimator update")
            if (
                estimate.get("estimator_revision_words") != previous_words
                or estimate.get("estimator_owner_digest")
                != list(_owner_digest(owner_index, 1))
                or estimate.get("causal_online_estimate") is not True
                or estimate.get("oracle_input_used") is not False
                or estimate.get("derived_from_this_arm_history_only") is not True
                or ranking.get("pre_decision_words") != previous_words
                or ranking.get("post_decision_words") != expected_words
                or ranking.get("permissive_internal_mask") != [True, True, True, True]
                or ranking.get("owns_actual_admissibility") is not False
                or shield.get("applied_after_ranking") is not True
                or shield.get("applied_before_environment") is not True
                or shield.get("action_available") is not True
                or shield.get("action_owner_digest")
                != list(_owner_digest(owner_index, 2))
                or shield.get("decision_owner_digest")
                != list(_owner_digest(owner_index, 3))
                or shield.get("shield_owner_digest")
                != list(_owner_digest(owner_index, 5))
                or transition.get("action") != shield.get("executable_action")
                or transition.get("decision_words") != expected_words
                or transition.get("estimator_pre_revision_words") != previous_words
                or transition.get("environment_applied") is not True
                or environment_post.get("event_words") != expected_words
                or update.get("called") is not True
                or update.get("applied") is not True
                or update.get("pre_revision_words") != previous_words
                or update.get("post_revision_words") != expected_words
            ):
                return False
            reward = transition.get("reward")
            if type(reward) is not float or not math.isfinite(reward):
                return False
            reconstructed_return = float(reconstructed_return + reward)
        return (
            len(arm.records_json) == event_index
            and type(arm.cumulative_reward) is float
            and math.isfinite(arm.cumulative_reward)
            and arm.cumulative_reward == reconstructed_return
        )

    def _validate_state_structure(self, state: ProspectiveExplorationRunState) -> bool:
        """Validate types, component invariants, clocks, records, and integrity."""

        try:
            if type(state) is not ProspectiveExplorationRunState:
                return False
            if type(state.event_index) is not int or not (
                0 <= state.event_index <= self.config.horizon
            ):
                return False
            if type(state.arms) is not tuple or len(state.arms) != len(MODE_ORDER):
                return False
            if type(state.integrity_sha256) is not str or len(state.integrity_sha256) != 64:
                return False
            current_words = [0, state.event_index]
            previous_words = [0, max(0, state.event_index - 1)]
            for expected_mode, arm in zip(MODE_ORDER, state.arms, strict=True):
                if type(arm) is not ProspectiveExplorationArmState or arm.mode != expected_mode:
                    return False
                if type(arm.records_json) is not tuple:
                    return False
                estimator = CausalExplorationEstimator(
                    self._estimator_config(expected_mode)
                )
                environment_config = self._environment_config(expected_mode)
                shield_config = self._shield_config(expected_mode)
                selector = ProspectiveExploration(self._selector_config(expected_mode))
                if not all(
                    (
                        _as_bool(estimator.state_valid(arm.estimator)),
                        _as_bool(
                            stochastic_trap_environment_state_valid(
                                environment_config,
                                arm.environment,
                            )
                        ),
                        _as_bool(
                            caller_owned_hard_shield_state_valid(
                                shield_config,
                                arm.shield,
                            )
                        ),
                        _as_bool(selector.state_valid(arm.selector)),
                    )
                ):
                    return False
                if (
                    _as_list(arm.estimator.revision_words) != current_words
                    or _as_list(arm.environment.event_words) != current_words
                    or _as_list(arm.selector.decision_words) != current_words
                    or _as_list(arm.shield.revision_words) != current_words
                    or _as_list(arm.selector.last_source_event_words) != current_words
                    or _as_list(arm.selector.last_candidate_revision_words)
                    != current_words
                    or _as_list(arm.selector.last_safety_revision_words) != current_words
                    or _as_list(arm.environment.last_estimator_revision_words)
                    != previous_words
                    or _as_list(arm.shield.last_estimator_revision_words)
                    != previous_words
                    or _as_list(arm.selector.last_host_policy_revision_words)
                    != previous_words
                    or _as_list(arm.selector.last_score_revision_words)
                    != previous_words
                ):
                    return False
                if not self._arm_records_valid(arm, state.event_index):
                    return False
            return state.integrity_sha256 == _canonical_sha256(self._state_body(state))
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def validate_state(self, state: ProspectiveExplorationRunState) -> bool:
        """Accept only the exact causal prefix, even if a tamper is re-sealed."""

        if not self._validate_state_structure(state):
            return False
        expected = self._reconstruct_unchecked(state.event_index)
        return (
            state.integrity_sha256 == expected.integrity_sha256
            and _strict_json_equal(self._state_body(state), self._state_body(expected))
        )

    def _candidate_batch(
        self,
        arm: ProspectiveExplorationArmState,
        event: ExplorationExogenousEvent,
    ) -> tuple[ExplorationCandidateBatch, object]:
        mode = arm.mode
        estimator = CausalExplorationEstimator(self._estimator_config(mode))
        observation = stochastic_trap_observation(
            self._environment_config(mode), arm.environment
        )
        estimates = estimator.estimate(
            arm.estimator,
            observation,
            event.source_event_words,
        )
        index = self._mode_index(mode)
        identities = jnp.stack(
            (
                jnp.full(
                    (N_EXPLORATION_ACTIONS,),
                    event.source_event_words[1],
                    dtype=jnp.uint32,
                ),
                jnp.arange(1, N_EXPLORATION_ACTIONS + 1, dtype=jnp.uint32),
            ),
            axis=1,
        )
        # The selector is a ranker here.  Actual admissibility is intentionally
        # withheld until the separate caller-owned hard-shield transaction.
        permissive_ranking_mask = jnp.ones(
            (N_EXPLORATION_ACTIONS,), dtype=jnp.bool_
        )
        batch = ExplorationCandidateBatch(
            candidate_actions=estimates.candidate_actions,
            candidate_identity_words=identities,
            candidate_valid=jnp.ones((N_EXPLORATION_ACTIONS,), dtype=jnp.bool_),
            host_policy=estimates.host_policy,
            host_action=estimates.host_action,
            expected_improvement=estimates.expected_improvement,
            ensemble_disagreement=estimates.ensemble_disagreement,
            information_gain=estimates.information_gain,
            learning_progress=estimates.learning_progress,
            candidate_safety_allowed=permissive_ranking_mask,
            host_action_safety_allowed=jnp.asarray(True, dtype=jnp.bool_),
            source_event_words=event.source_event_words,
            candidate_source_event_words=event.source_event_words,
            score_source_event_words=event.source_event_words,
            host_policy_source_event_words=event.source_event_words,
            safety_source_event_words=event.source_event_words,
            host_policy_revision_words=estimates.estimator_revision_words,
            candidate_revision_words=event.source_event_words,
            score_revision_words=estimates.estimator_revision_words,
            safety_revision_words=event.source_event_words,
            source_owner_digest=jnp.asarray(_owner_digest(index, 6), dtype=jnp.uint32),
            host_policy_owner_digest=jnp.asarray(
                _owner_digest(index, 7), dtype=jnp.uint32
            ),
            candidate_owner_digest=jnp.asarray(
                _owner_digest(index, 8), dtype=jnp.uint32
            ),
            score_owner_digest=jnp.asarray(_owner_digest(index, 9), dtype=jnp.uint32),
            safety_owner_digest=jnp.asarray(_owner_digest(index, 10), dtype=jnp.uint32),
            causal_pre_decision_attested=estimates.causal_online_estimate,
        )
        return batch, estimates

    def _step_arm(
        self,
        arm: ProspectiveExplorationArmState,
        event: ExplorationExogenousEvent,
    ) -> ProspectiveExplorationArmState:
        mode = arm.mode
        environment_config = self._environment_config(mode)
        estimator_config = self._estimator_config(mode)
        estimator = CausalExplorationEstimator(estimator_config)
        selector = ProspectiveExploration(self._selector_config(mode))
        batch, estimates_raw = self._candidate_batch(arm, event)
        estimates = cast(Any, estimates_raw)
        selection = selector.select(arm.selector, batch)
        ranked = RankedExplorationDecision(
            selected_action=selection.selected_candidate_action,
            host_action=batch.host_action,
            ranking_applied=selection.decision_applied,
            source_event_words=event.source_event_words,
            pre_decision_words=selection.pre_decision_words,
            post_decision_words=selection.post_decision_words,
            estimator_revision_words=estimates.estimator_revision_words,
            estimator_owner_digest=estimates.estimator_owner_digest,
            decision_owner_digest=jnp.asarray(
                estimator_config.decision_owner_digest,
                dtype=jnp.uint32,
            ),
        )
        actual_safety_mask = stochastic_trap_safety_mask(
            environment_config,
            arm.environment,
        )
        shield = apply_caller_owned_hard_shield(
            self._shield_config(mode),
            arm.shield,
            ranked,
            actual_safety_mask,
        )
        environment = stochastic_trap_environment_step(
            environment_config,
            arm.environment,
            event,
            shield.decision,
        )
        update = estimator.update(arm.estimator, environment.transition)
        if not all(
            (
                _as_bool(selection.decision_applied),
                _as_bool(shield.applied),
                _as_bool(shield.decision.action_available),
                _as_bool(environment.applied),
                _as_bool(update.applied),
            )
        ):
            raise RuntimeError("closed-loop exploration transaction failed closed")

        parent = "0" * 64
        if arm.records_json:
            parent = cast(str, json.loads(arm.records_json[-1])["record_sha256"])
        record: dict[str, object] = {
            "schema": "alberta.prospective-exploration-development.trace.v2",
            "mode": mode,
            "event_index": len(arm.records_json),
            "source_event_words": _as_list(event.source_event_words),
            "schedule_owner_digest": _as_list(event.schedule_owner_digest),
            "exogenous_observation_noise": {
                "stable_noise": _as_float(event.stable_noise),
                "reward_noise": _as_float(event.reward_noise),
                "noisy_tv_noise": _as_float(event.noisy_tv_noise),
                "paired_scope": "exogenous noise only",
            },
            "environment_owner_digest": _as_list(
                arm.environment.environment_owner_digest
            ),
            "environment_pre": {
                "delayed_progress": _as_int(arm.environment.delayed_progress),
                "observation": _as_list(environment.observation),
                "event_words": _as_list(arm.environment.event_words),
            },
            "candidate_actions": _as_list(batch.candidate_actions),
            "candidate_identity_words": _as_list(batch.candidate_identity_words),
            "candidate_budget": FIXED_CANDIDATE_BUDGET,
            "host_policy": _as_list(batch.host_policy),
            "host_action": _as_int(batch.host_action),
            "causal_estimates": {
                "expected_improvement": _as_list(batch.expected_improvement),
                "ensemble_disagreement": _as_list(batch.ensemble_disagreement),
                "information_gain": _as_list(batch.information_gain),
                "learning_progress": _as_list(batch.learning_progress),
                "estimator_revision_words": _as_list(
                    estimates.estimator_revision_words
                ),
                "estimator_owner_digest": _as_list(estimates.estimator_owner_digest),
                "causal_online_estimate": _as_bool(estimates.causal_online_estimate),
                "oracle_input_used": _as_bool(estimates.oracle_input_used),
                "derived_from_this_arm_history_only": True,
            },
            "ranking": {
                "selector": "ProspectiveExploration",
                "mode": mode,
                "permissive_internal_mask": _as_list(batch.candidate_safety_allowed),
                "owns_actual_admissibility": False,
                "pre_decision_words": _as_list(selection.pre_decision_words),
                "post_decision_words": _as_list(selection.post_decision_words),
                "selected_index": _as_int(selection.selected_index),
                "selected_action": _as_int(selection.selected_candidate_action),
                "selected_expected_improvement_surprisal_score": _as_float(
                    selection.selected_expected_improvement_surprisal_score
                ),
                "logical_uniform_draws": FIXED_CANDIDATE_BUDGET + 1,
            },
            "caller_owned_hard_shield": {
                "actual_safety_mask": _as_list(actual_safety_mask),
                "shield_owner_digest": _as_list(shield.state.shield_owner_digest),
                "applied_after_ranking": True,
                "applied_before_environment": True,
                "selected_action_allowed": _as_bool(shield.selected_action_allowed),
                "selected_action_executed": _as_bool(shield.selected_action_executed),
                "fallback_used": _as_bool(shield.fallback_used),
                "executable_action": _as_int(shield.decision.action),
                "action_available": _as_bool(shield.decision.action_available),
                "action_owner_digest": _as_list(shield.decision.action_owner_digest),
                "decision_owner_digest": _as_list(
                    shield.decision.decision_owner_digest
                ),
                "shield_revision_words": _as_list(shield.state.revision_words),
            },
            "observed_transition": {
                "action": _as_int(environment.transition.action),
                "reward": _as_float(environment.reward),
                "next_observation": _as_list(environment.next_observation),
                "decision_words": _as_list(environment.transition.decision_words),
                "estimator_pre_revision_words": _as_list(
                    environment.transition.estimator_revision_words
                ),
                "environment_applied": _as_bool(environment.applied),
                "noisy_tv_observed": _as_bool(environment.noisy_tv_observed),
                "delayed_investment_applied": _as_bool(
                    environment.delayed_investment_applied
                ),
                "delayed_collection_applied": _as_bool(
                    environment.delayed_collection_applied
                ),
            },
            "environment_post": {
                "delayed_progress": _as_int(environment.state.delayed_progress),
                "event_words": _as_list(environment.state.event_words),
            },
            "estimator_update": {
                "called": True,
                "applied": _as_bool(update.applied),
                "pre_revision_words": _as_list(update.pre_revision_words),
                "post_revision_words": _as_list(update.post_revision_words),
                "mean_td_error": _as_float(update.mean_td_error),
                "mean_absolute_td_error": _as_float(update.mean_absolute_td_error),
            },
            "physical_action_dispatched": False,
            "causal_parent_sha256": parent,
        }
        record["record_sha256"] = _canonical_sha256(record)
        encoded = _canonical_json_bytes(record).decode("utf-8")
        reward = _as_float(environment.reward)
        return ProspectiveExplorationArmState(
            mode=mode,
            environment=environment.state,
            estimator=update.state,
            selector=selection.state,
            shield=shield.state,
            cumulative_reward=float(arm.cumulative_reward + reward),
            records_json=(*arm.records_json, encoded),
        )

    def _advance_validated(
        self,
        state: ProspectiveExplorationRunState,
    ) -> ProspectiveExplorationRunState:
        if state.event_index >= self.config.horizon:
            raise ValueError("development horizon is exhausted")
        event = self.common_event(state.event_index)
        next_state = ProspectiveExplorationRunState(
            event_index=state.event_index + 1,
            arms=tuple(self._step_arm(arm, event) for arm in state.arms),
            integrity_sha256="",
        )
        return self._seal_state(next_state)

    def step(
        self,
        state: ProspectiveExplorationRunState,
    ) -> ProspectiveExplorationRunState:
        if not self.validate_state(state):
            raise ValueError("run state fails its exact causal integrity contract")
        return self._advance_validated(state)

    def run_to_end(
        self,
        state: ProspectiveExplorationRunState | None = None,
    ) -> ProspectiveExplorationRunState:
        current = self.initial_state() if state is None else state
        if not self.validate_state(current):
            raise ValueError("initial run state fails its integrity contract")
        while current.event_index < self.config.horizon:
            current = self._advance_validated(current)
        return current

    def _reconstruct_unchecked(
        self,
        event_count: int,
    ) -> ProspectiveExplorationRunState:
        _exact_int(event_count, name="event_count", maximum=self.config.horizon)
        return _cached_exact_prefix(self.config, event_count)

    def _reconstruct(self, event_count: int) -> ProspectiveExplorationRunState:
        state = self._reconstruct_unchecked(event_count)
        if not self._validate_state_structure(state):
            raise AssertionError("internally reconstructed state is invalid")
        return state

    def checkpoint_payload(
        self,
        state: ProspectiveExplorationRunState,
    ) -> dict[str, object]:
        """Return a full in-memory checkpoint with no filesystem writer."""

        if not self.validate_state(state):
            raise ValueError("cannot checkpoint an invalid exploration state")
        body: dict[str, object] = {
            "schema": PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA,
            "config": self.config.to_config(),
            "event_index": state.event_index,
            "full_composite_state": self._state_body(state),
            "state_integrity_sha256": state.integrity_sha256,
            "in_memory_only": True,
            "output_path": None,
            "artifact_writer_available": False,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
        }
        body["checkpoint_sha256"] = _canonical_sha256(body)
        if len(_canonical_json_bytes(body)) > _MAX_CHECKPOINT_BYTES:
            raise ValueError("checkpoint exceeds the bounded in-memory size")
        return body

    def restore_checkpoint(
        self,
        value: Mapping[str, object],
    ) -> ProspectiveExplorationRunState:
        """Restore only an exact causal prefix reconstructed independently."""

        payload = dict(_mapping(value, name="checkpoint"))
        expected_fields = {
            "schema",
            "config",
            "event_index",
            "full_composite_state",
            "state_integrity_sha256",
            "in_memory_only",
            "output_path",
            "artifact_writer_available",
            "assessment_status",
            "promotion_authority",
            "checkpoint_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("checkpoint fields differ")
        supplied_sha = payload.pop("checkpoint_sha256")
        if type(supplied_sha) is not str or supplied_sha != _canonical_sha256(payload):
            raise ValueError("checkpoint digest differs")
        if payload["schema"] != PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint schema differs")
        if not _strict_json_equal(payload["config"], self.config.to_config()):
            raise ValueError("checkpoint config differs")
        fixed = {
            "in_memory_only": True,
            "output_path": None,
            "artifact_writer_available": False,
            "assessment_status": ASSESSMENT_STATUS,
            "promotion_authority": False,
        }
        for name, expected in fixed.items():
            if not _strict_json_equal(payload[name], expected):
                raise ValueError(f"checkpoint fixed field {name} differs")
        event_index = _exact_int(
            payload["event_index"],
            name="checkpoint event_index",
            maximum=self.config.horizon,
        )
        expected_state = self._reconstruct(event_index)
        if not _strict_json_equal(
            payload["full_composite_state"], self._state_body(expected_state)
        ):
            raise ValueError("checkpoint is not the exact causal prefix")
        if payload["state_integrity_sha256"] != expected_state.integrity_sha256:
            raise ValueError("checkpoint state integrity differs")
        return expected_state

    @staticmethod
    def records(state: ProspectiveExplorationRunState) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for event_index in range(state.event_index):
            for arm in state.arms:
                raw = json.loads(arm.records_json[event_index])
                if type(raw) is not dict:
                    raise ValueError("raw record must decode to an object")
                records.append(cast(dict[str, object], raw))
        return records

    def resource_report(
        self,
        state: ProspectiveExplorationRunState,
    ) -> dict[str, object]:
        """Report exact array bytes and actual fixed logical opportunities."""

        if not self.validate_state(state):
            raise ValueError("resource state fails integrity")
        per_arm: dict[str, object] = {}
        logical_signatures: list[tuple[int, ...]] = []
        for arm in state.arms:
            estimator = CausalExplorationEstimator(self._estimator_config(arm.mode))
            estimator_budget = estimator.resource_budget(arm.estimator)
            selector = ProspectiveExploration(self._selector_config(arm.mode))
            selector_budget = selector.resource_budget(arm.selector)
            core = measure_causal_exploration_core_resources(
                arm.estimator,
                arm.environment,
                arm.shield,
            )
            logical = {
                "candidate_slots": state.event_index * FIXED_CANDIDATE_BUDGET,
                "candidate_generation_random_draws": 0,
                "selector_uniform_draws": (
                    state.event_index * selector_budget.logical_uniform_draws_per_decision
                ),
                "estimator_initialization_normal_draws": (
                    estimator_budget.initialization_normal_draws
                ),
                "estimator_score_scalars": (
                    state.event_index * estimator_budget.candidate_scores_per_decision
                ),
                "estimator_update_opportunities": state.event_index,
                "estimator_updates_applied": state.event_index,
                "estimator_update_random_draws": 0,
                "estimator_update_parameter_opportunities": (
                    state.event_index
                    * estimator_budget.update_parameter_opportunities_per_transition
                ),
                "hard_shield_calls": state.event_index,
                "environment_step_opportunities": state.event_index,
                "environment_steps_applied": state.event_index,
            }
            signature = tuple(logical[name] for name in sorted(logical))
            logical_signatures.append(signature)
            per_arm[arm.mode] = {
                "persistent_array_bytes": {
                    "causal_core": dataclasses.asdict(core),
                    "selector_state_nbytes": selector_budget.total_state_nbytes,
                    "total": core.total_state_nbytes + selector_budget.total_state_nbytes,
                },
                "logical": logical,
                "record_json_logical_bytes": sum(
                    len(raw.encode("utf-8")) for raw in arm.records_json
                ),
                "temporary_scope": estimator_budget.temporary_scope,
            }
        return {
            "scope": (
                "all persistent JAX array leaves plus canonical raw-record logical bytes; "
                "excludes Python object overhead, compiler/XLA workspaces, and allocator peaks"
            ),
            "per_arm": per_arm,
            "logical_opportunities_matched": len(set(logical_signatures)) == 1,
            "common_paired_exogenous_normal_draws": 3 * state.event_index,
            "common_schedule_generated_once": True,
            "wall_clock_measured": False,
            "energy_measured": False,
        }


@functools.lru_cache(maxsize=FIXED_HORIZON + 1)
def _cached_exact_prefix(
    config: ProspectiveExplorationDevelopmentConfig,
    event_count: int,
) -> ProspectiveExplorationRunState:
    """Memoize only internally generated immutable prefixes, never caller state."""

    _exact_int(event_count, name="event_count", maximum=config.horizon)
    evaluator = ProspectiveExplorationDevelopmentEvaluator(config)
    current = evaluator.initial_state()
    while current.event_index < event_count:
        current = evaluator._advance_validated(current)
    if not evaluator._validate_state_structure(current):
        raise AssertionError("cached internally generated prefix is invalid")
    return current


def _summaries_from_records(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconstruct descriptive summaries exclusively from raw record fields."""

    summaries: dict[str, object] = {}
    for mode in MODE_ORDER:
        arm_records = [record for record in records if record.get("mode") == mode]
        rewards: list[float] = []
        selected: Counter[int] = Counter()
        executed: Counter[int] = Counter()
        fallbacks = 0
        for index, record in enumerate(arm_records):
            if record.get("event_index") != index:
                raise ValueError("raw record event order differs")
            ranking = _mapping(record.get("ranking"), name="ranking")
            shield = _mapping(
                record.get("caller_owned_hard_shield"), name="hard shield"
            )
            transition = _mapping(
                record.get("observed_transition"), name="observed transition"
            )
            selected[_exact_int(ranking["selected_action"], name="selected action")] += 1
            executed[_exact_int(transition["action"], name="executed action")] += 1
            rewards.append(_exact_float(transition["reward"], name="reward"))
            fallbacks += int(shield["fallback_used"] is True)
        summaries[mode] = {
            "event_count": len(arm_records),
            "cumulative_reward_descriptive_only": float(sum(rewards)),
            "selected_action_counts": {
                str(action): selected[action] for action in range(N_EXPLORATION_ACTIONS)
            },
            "executed_action_counts": {
                str(action): executed[action] for action in range(N_EXPLORATION_ACTIONS)
            },
            "noisy_tv_executions": executed[NOISY_TV_ACTION],
            "delayed_investment_executions": executed[INVEST_ACTION],
            "delayed_collection_executions": executed[COLLECT_ACTION],
            "hard_shield_fallbacks": fallbacks,
            "assessment_status": ASSESSMENT_STATUS,
        }
    return summaries


def _trace_diagnostics(
    evaluator: ProspectiveExplorationDevelopmentEvaluator,
    state: ProspectiveExplorationRunState,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    owners = [tuple(_as_list(arm.environment.environment_owner_digest)) for arm in state.arms]
    causal = True
    no_oracle = True
    shield_boundary = True
    chain_heads = {mode: "0" * 64 for mode in MODE_ORDER}
    for record in records:
        mode = cast(ModeName, record["mode"])
        body = {name: record[name] for name in record if name != "record_sha256"}
        causal = causal and record["record_sha256"] == _canonical_sha256(body)
        causal = causal and record["causal_parent_sha256"] == chain_heads[mode]
        chain_heads[mode] = cast(str, record["record_sha256"])
        estimate = _mapping(record["causal_estimates"], name="causal estimates")
        ranking = _mapping(record["ranking"], name="ranking")
        shield = _mapping(record["caller_owned_hard_shield"], name="hard shield")
        transition = _mapping(record["observed_transition"], name="transition")
        update = _mapping(record["estimator_update"], name="estimator update")
        no_oracle = no_oracle and estimate["oracle_input_used"] is False
        no_oracle = no_oracle and estimate["causal_online_estimate"] is True
        causal = causal and (
            estimate["estimator_revision_words"]
            == transition["estimator_pre_revision_words"]
            == update["pre_revision_words"]
        )
        causal = causal and ranking["post_decision_words"] == transition["decision_words"]
        shield_boundary = shield_boundary and ranking["owns_actual_admissibility"] is False
        shield_boundary = shield_boundary and all(
            cast(list[bool], ranking["permissive_internal_mask"])
        )
        shield_boundary = shield_boundary and shield["applied_after_ranking"] is True
        shield_boundary = shield_boundary and shield["applied_before_environment"] is True
    resources = evaluator.resource_report(state)
    return {
        "all_six_comparators_executed": {record["mode"] for record in records}
        == set(MODE_ORDER),
        "independent_environment_owner_per_arm": len(set(owners)) == len(MODE_ORDER),
        "environment_state_shared_between_arms": False,
        "paired_exogenous_noise_only": True,
        "each_score_derived_from_own_executed_history": causal and no_oracle,
        "oracle_score_input_used": not no_oracle,
        "exact_action_decision_estimator_revision_chain": causal,
        "selector_internal_mask_permissive": shield_boundary,
        "caller_hard_shield_owns_actual_admissibility": shield_boundary,
        "logical_candidate_rng_update_budgets_matched": resources[
            "logical_opportunities_matched"
        ],
        "raw_trace_reconstructs_summaries": True,
        "physical_action_dispatch_count": 0,
        "outcome_assessed": False,
    }


def build_prospective_exploration_development_report() -> dict[str, object]:
    """Build the deterministic in-memory report without writing an artifact."""

    config = ProspectiveExplorationDevelopmentConfig()
    evaluator = ProspectiveExplorationDevelopmentEvaluator(config)
    final_state = evaluator._reconstruct(config.horizon)
    prefix = evaluator._reconstruct(config.checkpoint_split)
    checkpoint = evaluator.checkpoint_payload(prefix)
    restored = evaluator.restore_checkpoint(json.loads(_canonical_json_bytes(checkpoint)))
    resumed = evaluator.run_to_end(restored)
    checkpoint_resume_exact = (
        evaluator._state_body(resumed) == evaluator._state_body(final_state)
        and resumed.integrity_sha256 == final_state.integrity_sha256
    )
    records = evaluator.records(final_state)
    summaries = _summaries_from_records(records)
    report: dict[str, object] = {
        "schema": PROSPECTIVE_EXPLORATION_REPORT_SCHEMA,
        "protocol": prospective_exploration_protocol(config),
        "config": config.to_config(),
        "assessment_status": ASSESSMENT_STATUS,
        "development_status": DEVELOPMENT_STATUS,
        "development_data_consumed": True,
        "untouched_held_out_data": False,
        "thresholds": [],
        "winner_selected": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "physical_safety_certificate": False,
        "output_path": None,
        "output_written": False,
        "artifact_writer_available": False,
        "action_dispatch_authority": False,
        "deployment_authority": False,
        "evidence_claimed": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "records": records,
        "records_sha256": _canonical_sha256(records),
        "summaries": summaries,
        "summaries_reconstructed_from_raw_records": True,
        "resources": evaluator.resource_report(final_state),
        "checkpoint": {
            "schema": checkpoint["schema"],
            "split": config.checkpoint_split,
            "in_memory_only": True,
            "logical_bytes": len(_canonical_json_bytes(checkpoint)),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "resume_exact": checkpoint_resume_exact,
            "artifact_path": None,
        },
        "diagnostics": _trace_diagnostics(evaluator, final_state, records),
        "limitations": list(_LIMITATIONS),
    }
    report["report_sha256"] = _canonical_sha256(report)
    if len(_canonical_json_bytes(report)) > _MAX_REPORT_BYTES:
        raise ValueError("development report exceeds its bounded in-memory size")
    return report


def validate_prospective_exploration_development_report(
    value: Mapping[str, object],
) -> ProspectiveExplorationValidationReceipt:
    """Fail closed unless every raw record exactly replays from the protocol."""

    report = dict(_mapping(value, name="report"))
    if "report_sha256" not in report:
        raise ValueError("report digest is missing")
    supplied_sha = report.pop("report_sha256")
    if type(supplied_sha) is not str or supplied_sha != _canonical_sha256(report):
        raise ValueError("report digest differs")
    if report.get("schema") != PROSPECTIVE_EXPLORATION_REPORT_SCHEMA:
        raise ValueError("report schema differs")
    expected_fields = {
        "schema",
        "protocol",
        "config",
        "assessment_status",
        "development_status",
        "development_data_consumed",
        "untouched_held_out_data",
        "thresholds",
        "winner_selected",
        "efficacy_claimed",
        "safety_claimed",
        "physical_safety_certificate",
        "output_path",
        "output_written",
        "artifact_writer_available",
        "action_dispatch_authority",
        "deployment_authority",
        "evidence_claimed",
        "promotion_authority",
        "scientific_promotion_allowed",
        "records",
        "records_sha256",
        "summaries",
        "summaries_reconstructed_from_raw_records",
        "resources",
        "checkpoint",
        "diagnostics",
        "limitations",
    }
    if set(report) != expected_fields:
        raise ValueError("report fields differ")
    fixed = {
        "assessment_status": ASSESSMENT_STATUS,
        "development_status": DEVELOPMENT_STATUS,
        "development_data_consumed": True,
        "untouched_held_out_data": False,
        "thresholds": [],
        "winner_selected": False,
        "efficacy_claimed": False,
        "safety_claimed": False,
        "physical_safety_certificate": False,
        "output_path": None,
        "output_written": False,
        "artifact_writer_available": False,
        "action_dispatch_authority": False,
        "deployment_authority": False,
        "evidence_claimed": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
        "summaries_reconstructed_from_raw_records": True,
    }
    for name, expected in fixed.items():
        if not _strict_json_equal(report.get(name), expected):
            raise ValueError(f"report fixed field {name} differs")
    records_raw = report.get("records")
    if type(records_raw) is not list:
        raise ValueError("report records must be an array")
    records = [
        _mapping(record, name=f"records[{index}]")
        for index, record in enumerate(cast(list[object], records_raw))
    ]
    if report.get("records_sha256") != _canonical_sha256(records_raw):
        raise ValueError("raw records digest differs")
    reconstructed_summaries = _summaries_from_records(records)
    if not _strict_json_equal(report.get("summaries"), reconstructed_summaries):
        raise ValueError("summaries do not reconstruct from raw records")
    config_payload = _mapping(report["config"], name="config")
    config = ProspectiveExplorationDevelopmentConfig.from_config(config_payload)
    if not _strict_json_equal(report["protocol"], prospective_exploration_protocol(config)):
        raise ValueError("report protocol differs")
    evaluator = ProspectiveExplorationDevelopmentEvaluator(config)
    current = evaluator._reconstruct(config.horizon)
    checkpoint_prefix = evaluator._reconstruct(config.checkpoint_split)
    expected_records = evaluator.records(current)
    if not _strict_json_equal(records_raw, expected_records):
        raise ValueError("report fails exact causal replay")
    if not _strict_json_equal(report["resources"], evaluator.resource_report(current)):
        raise ValueError("report resource reconstruction differs")
    if not _strict_json_equal(
        report["diagnostics"],
        _trace_diagnostics(evaluator, current, expected_records),
    ):
        raise ValueError("report diagnostics reconstruction differs")
    if not _strict_json_equal(report["limitations"], list(_LIMITATIONS)):
        raise ValueError("report limitations differ")
    checkpoint_payload = evaluator.checkpoint_payload(checkpoint_prefix)
    resumed = evaluator.run_to_end(checkpoint_prefix)
    resume_exact = (
        evaluator._state_body(resumed) == evaluator._state_body(current)
        and resumed.integrity_sha256 == current.integrity_sha256
    )
    expected_checkpoint = {
        "schema": checkpoint_payload["schema"],
        "split": config.checkpoint_split,
        "in_memory_only": True,
        "logical_bytes": len(_canonical_json_bytes(checkpoint_payload)),
        "checkpoint_sha256": checkpoint_payload["checkpoint_sha256"],
        "resume_exact": resume_exact,
        "artifact_path": None,
    }
    if not _strict_json_equal(report["checkpoint"], expected_checkpoint):
        raise ValueError("report checkpoint reconstruction differs")
    diagnostics = _mapping(report["diagnostics"], name="diagnostics")
    checkpoint = _mapping(report["checkpoint"], name="checkpoint")
    resources = _mapping(report["resources"], name="resources")
    return ProspectiveExplorationValidationReceipt(
        valid=True,
        assessment_status=ASSESSMENT_STATUS,
        exact_causal_replay=True,
        checkpoint_resume_exact=checkpoint["resume_exact"] is True,
        raw_trace_reconstructable=True,
        matched_budgets=resources["logical_opportunities_matched"] is True,
        independent_environment_owners=(
            diagnostics["independent_environment_owner_per_arm"] is True
        ),
        output_written=False,
        promotion_authority=False,
    )


assert CAUSAL_EXPLORATION_ASSESSMENT_STATUS == ASSESSMENT_STATUS


__all__ = [
    "ACTION_DISPATCH_AUTHORITY",
    "ARTIFACT_WRITER_AVAILABLE",
    "ASSESSMENT_STATUS",
    "DEPLOYMENT_AUTHORITY",
    "DEVELOPMENT_STATUS",
    "EVIDENCE_CLAIMED",
    "FIXED_CHECKPOINT_SPLIT",
    "FIXED_HORIZON",
    "MODE_ORDER",
    "OUTPUT_WRITES",
    "PHYSICAL_SAFETY_CLAIM",
    "PROMOTION_AUTHORITY",
    "PROSPECTIVE_EXPLORATION_CHECKPOINT_SCHEMA",
    "PROSPECTIVE_EXPLORATION_CONFIG_SCHEMA",
    "PROSPECTIVE_EXPLORATION_PROTOCOL_SCHEMA",
    "PROSPECTIVE_EXPLORATION_REPORT_SCHEMA",
    "ProspectiveExplorationArmState",
    "ProspectiveExplorationDevelopmentConfig",
    "ProspectiveExplorationDevelopmentEvaluator",
    "ProspectiveExplorationRunState",
    "ProspectiveExplorationValidationReceipt",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "build_prospective_exploration_development_report",
    "prospective_exploration_protocol",
    "validate_prospective_exploration_development_report",
]
