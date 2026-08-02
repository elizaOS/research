from __future__ import annotations

import copy
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.evaluation.continual_control_evaluator import (
    ContinuingControlBudget,
    ContinuingControlLearner,
    ContinuingControlProtocol,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlLearnerUpdate,
    ControlProbe,
    ControlResourceUsage,
    ControlTransition,
)
from alberta_framework.evaluation.continual_control_reference_suite import (
    EXACT_ORACLE_SCORE_SEMANTICS,
    ORACLE_ACTION_DATA_ROLE,
    ORACLE_SOURCE_SCHEMA,
    REFERENCE_ROLES,
    RETAINED_FRESH_PER_REGIME_ROLE,
    STATIONARY_MULTITASK_ROLE,
    FrozenStationaryReferenceStream,
    PrivilegedContinualControlReferenceSuite,
    PrivilegedReferenceExtraDataBudget,
    PrivilegedReferenceRunConfig,
    StationaryReferenceExample,
    load_privileged_control_reference_report,
    save_privileged_control_reference_report,
    validate_privileged_control_reference_report,
)

SCHEDULE = ("A", "A", "B", "B", "A", "A")
SEED = 29


def _decision_id(lifecycle: tuple[int, int], generation: int) -> tuple[int, int, int, int]:
    return (
        lifecycle[0],
        lifecycle[1],
        generation >> 32,
        generation & ((1 << 32) - 1),
    )


def _scores(seed: int, step: int, regime_id: str) -> tuple[float, float]:
    jitter = ((seed * 13 + step * 7) % 5) * 0.01
    if regime_id == "A":
        return (1.0 + jitter, -1.0 + jitter)
    return (-1.0 + jitter, 1.0 + jitter)


@dataclass(frozen=True)
class FlexibleState:
    generation: int


class FlexibleCueLearner:
    """A test learner whose state remains valid across privileged routing gaps."""

    def __init__(
        self,
        *,
        seed: int,
        name: str,
        lifecycle_id: tuple[int, int],
    ) -> None:
        self._seed = seed
        self._name = name
        self._lifecycle_id = lifecycle_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_actions(self) -> int:
        return 2

    @property
    def max_backward_calls_per_update(self) -> int:
        return 0

    def to_config(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "schema_version": "tests.flexible_cue_reference_learner.v1",
            "name": self._name,
            "seed": self._seed,
            "lifecycle_id": list(self._lifecycle_id),
        }

    def init(self, initial_observation: tuple[float, ...]) -> Any:
        if len(initial_observation) != 1:
            raise ValueError("test learner requires one observation scalar")
        return FlexibleState(0)

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        return isinstance(state, FlexibleState) and len(observation) == 1

    def decide(self, state: Any, observation: tuple[float, ...]) -> ControlDecision:
        resolved = cast(FlexibleState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("flexible state is invalid")
        return ControlDecision(
            observation=observation,
            action=int(observation[0] >= 0.5),
            decision_id=_decision_id(self._lifecycle_id, resolved.generation),
        )

    def update(self, state: Any, transition: ControlTransition) -> ControlLearnerUpdate:
        resolved = cast(FlexibleState, state)
        expected = self.decide(resolved, transition.observation)
        valid = (
            transition.action == expected.action and transition.decision_id == expected.decision_id
        )
        return ControlLearnerUpdate(
            FlexibleState(resolved.generation + int(valid)),
            valid,
            0,
        )

    def probe_action(self, state: Any, observation: tuple[float, ...]) -> int:
        del state
        return int(observation[0] >= 0.5)

    def state_to_config(self, state: Any) -> object:
        return {"generation": cast(FlexibleState, state).generation}

    def state_from_config(self, payload: object) -> Any:
        if not isinstance(payload, Mapping) or set(payload) != {"generation"}:
            raise ValueError("flexible state payload is invalid")
        generation = payload["generation"]
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("flexible generation must be an integer")
        return FlexibleState(generation)

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        self.state_to_config(state)
        return ControlResourceUsage(
            persistent_state_bytes=8,
            state_scalar_count=1,
            trainable_parameter_count=0,
            measurement_method="test exact uint64 state accounting",
        )

    def diagnostics(self, state: Any) -> Mapping[str, bool | int | float | str]:
        return {"generation": cast(FlexibleState, state).generation}


@dataclass(frozen=True)
class BoundState:
    observation: tuple[float, ...]
    generation: int


class ObservationBoundCueLearner(FlexibleCueLearner):
    def init(self, initial_observation: tuple[float, ...]) -> BoundState:
        return BoundState(initial_observation, 0)

    def state_valid_for_observation(
        self,
        state: Any,
        observation: tuple[float, ...],
    ) -> bool:
        return isinstance(state, BoundState) and state.observation == observation

    def decide(self, state: Any, observation: tuple[float, ...]) -> ControlDecision:
        resolved = cast(BoundState, state)
        if not self.state_valid_for_observation(resolved, observation):
            raise ValueError("bound state does not own observation")
        return ControlDecision(
            observation=observation,
            action=int(observation[0] >= 0.5),
            decision_id=_decision_id(self._lifecycle_id, resolved.generation),
        )

    def update(self, state: Any, transition: ControlTransition) -> ControlLearnerUpdate:
        resolved = cast(BoundState, state)
        expected = self.decide(resolved, transition.observation)
        valid = (
            transition.action == expected.action and transition.decision_id == expected.decision_id
        )
        return ControlLearnerUpdate(
            BoundState(
                transition.next_decision_observation,
                resolved.generation + int(valid),
            ),
            valid,
            0,
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(BoundState, state)
        return {
            "observation": list(resolved.observation),
            "generation": resolved.generation,
        }

    def state_from_config(self, payload: object) -> BoundState:
        if not isinstance(payload, Mapping) or set(payload) != {
            "observation",
            "generation",
        }:
            raise ValueError("bound state payload is invalid")
        observation = payload["observation"]
        generation = payload["generation"]
        if not isinstance(observation, list):
            raise ValueError("bound observation must be an array")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise ValueError("bound generation must be an integer")
        return BoundState(tuple(float(value) for value in observation), generation)

    def resource_usage(self, state: Any) -> ControlResourceUsage:
        self.state_to_config(state)
        return ControlResourceUsage(
            persistent_state_bytes=16,
            state_scalar_count=2,
            trainable_parameter_count=0,
            measurement_method="test exact observation-bound state accounting",
        )


@dataclass(frozen=True)
class EnvironmentState:
    step: int
    observation: tuple[float, ...]


class SeededReferenceEnvironment:
    def __init__(
        self,
        *,
        seed: int,
        copy_index: int,
        oracle_prepared_steps: set[int],
    ) -> None:
        self._seed = seed
        self._copy_index = copy_index
        self._oracle_prepared_steps = oracle_prepared_steps

    @property
    def n_actions(self) -> int:
        return 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": "SeededReferenceEnvironment",
            "schema_version": "tests.seeded_reference_environment.v1",
            "seed": self._seed,
            "schedule": list(SCHEDULE),
        }

    @staticmethod
    def _observation(step: int) -> tuple[float, ...]:
        regime = SCHEDULE[min(step, len(SCHEDULE) - 1)]
        return (0.0 if regime == "A" else 1.0,)

    def init(self) -> EnvironmentState:
        return EnvironmentState(0, self._observation(0))

    def observation(self, state: Any) -> tuple[float, ...]:
        return cast(EnvironmentState, state).observation

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate:
        resolved = cast(EnvironmentState, state)
        if self._copy_index == 2 and resolved.step not in self._oracle_prepared_steps:
            raise ValueError("oracle outcome requested before frozen action scores")
        if evaluator_regime_id != SCHEDULE[resolved.step]:
            raise ValueError("environment regime schedule mismatch")
        if decision.observation != resolved.observation:
            raise ValueError("environment received stale observation")
        score_values = _scores(self._seed, resolved.step, evaluator_regime_id)
        completed = resolved.step + 1
        terminated = completed == len(SCHEDULE)
        next_observation = self._observation(0) if terminated else self._observation(completed)
        return ControlEnvironmentUpdate(
            state=EnvironmentState(completed, next_observation),
            transition=ControlTransition(
                observation=decision.observation,
                action=decision.action,
                decision_id=decision.decision_id,
                reward=score_values[decision.action],
                discount=0.0 if terminated else 0.9,
                terminated=terminated,
                truncated=False,
                bootstrap_observation=next_observation,
                reset_observation=next_observation if terminated else None,
            ),
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(EnvironmentState, state)
        return {"step": resolved.step, "observation": list(resolved.observation)}

    def state_from_config(self, payload: object) -> EnvironmentState:
        if not isinstance(payload, Mapping) or set(payload) != {"step", "observation"}:
            raise ValueError("environment state payload is invalid")
        step = payload["step"]
        observation = payload["observation"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("environment step must be an integer")
        if not isinstance(observation, list):
            raise ValueError("environment observation must be an array")
        return EnvironmentState(step, tuple(float(value) for value in observation))


class FrozenTestOracle:
    def __init__(
        self,
        *,
        seed: int,
        prepared_steps: set[int],
        score_offset: float = 0.0,
        score_semantics: str = EXACT_ORACLE_SCORE_SEMANTICS,
    ) -> None:
        self._seed = seed
        self._prepared_steps = prepared_steps
        self._score_offset = score_offset
        self._score_semantics = score_semantics

    def to_config(self) -> dict[str, object]:
        return {
            "type": "FrozenTestOracle",
            "schema_version": ORACLE_SOURCE_SCHEMA,
            "seed": self._seed,
            "score_source_id": "tests.reference-score-source.v1",
            "score_offset": self._score_offset,
            "score_semantics": self._score_semantics,
        }

    def action_scores(
        self,
        observation: tuple[float, ...],
        *,
        evaluator_regime_id: str,
        step: int,
    ) -> tuple[float, ...]:
        del observation
        self._prepared_steps.add(step)
        return tuple(
            score + self._score_offset for score in _scores(self._seed, step, evaluator_regime_id)
        )


def _protocol() -> ContinuingControlProtocol:
    return ContinuingControlProtocol(
        protocol_id="tests.privileged-control-references.v2",
        higher_is_better=True,
        regime_schedule=SCHEDULE,
        evaluator_regime_ids=("A", "B"),
        checkpoint_steps=(2, 4, 6),
        first_exposure_checkpoint={"A": 0, "B": 1},
        forward_transfer_reference={"A": 0.5, "B": 0.5},
        recovery_thresholds={"A": 0.5, "B": 0.5},
        stability_references={"A": 1.0, "B": 1.0},
        recovery_window=1,
        worst_window_size=2,
        operation_latency_deadline_ms=1.0,
    )


def _common_budget() -> ContinuingControlBudget:
    return ContinuingControlBudget(
        transition_limit=6,
        decision_call_limit=6,
        environment_call_limit=6,
        update_call_limit=6,
        probe_call_limit=6,
        backward_call_limit=0,
        persistent_state_bytes_limit=1_000,
        state_scalar_count_limit=100,
        trainable_parameter_count_limit=None,
        stored_decision_id_limit=6,
    )


def _stream(*, seed: int = SEED) -> FrozenStationaryReferenceStream:
    regimes = ("A", "B", "A", "B")
    examples: list[StationaryReferenceExample] = []
    for index, regime in enumerate(regimes):
        next_regime = regimes[index + 1] if index + 1 < len(regimes) else "A"
        examples.append(
            StationaryReferenceExample(
                reference_regime_id=regime,
                observation=(0.0 if regime == "A" else 1.0,),
                action_scores=_scores(seed, index, regime),
                discount=0.9,
                terminated=False,
                truncated=False,
                bootstrap_observation=(0.0 if next_regime == "A" else 1.0,),
                reset_observation=None,
            )
        )
    return FrozenStationaryReferenceStream(
        stream_id="tests.stationary-reference-stream.v1",
        seed=seed,
        examples=tuple(examples),
    )


def _run_config(*, seed: int = SEED) -> PrivilegedReferenceRunConfig:
    return PrivilegedReferenceRunConfig(
        suite_id="tests.privileged-reference-suite.v1",
        seed=seed,
        fresh_lifecycle_id=(501, 1),
        stationary_lifecycle_id=(502, 1),
        oracle_lifecycle_id=(503, 1),
        extra_data_budget=PrivilegedReferenceExtraDataBudget(
            stationary_transition_limit=4,
            stationary_decision_call_limit=4,
            stationary_update_call_limit=4,
            stationary_backward_call_limit=0,
            stationary_reward_table_scalar_limit=8,
            oracle_callback_limit=6,
            oracle_action_score_scalar_limit=12,
            oracle_probe_action_score_scalar_limit=12,
        ),
    )


def _suite(
    *,
    fresh_bound: bool = False,
    config: PrivilegedReferenceRunConfig | None = None,
    environment_seed_offset: int = 0,
    oracle_score_offset: float = 0.0,
    oracle_score_semantics: str = EXACT_ORACLE_SCORE_SEMANTICS,
) -> tuple[PrivilegedContinualControlReferenceSuite, list[object], set[int]]:
    environment_instances: list[object] = []
    prepared_steps: set[int] = set()

    def environment_factory(seed: int) -> SeededReferenceEnvironment:
        environment = SeededReferenceEnvironment(
            seed=seed + environment_seed_offset,
            copy_index=len(environment_instances),
            oracle_prepared_steps=prepared_steps,
        )
        environment_instances.append(environment)
        return environment

    def fresh_factory(
        seed: int,
        evaluator_regime_id: str,
    ) -> ContinuingControlLearner:
        learner_type = ObservationBoundCueLearner if fresh_bound else FlexibleCueLearner
        lifecycle = (601, 1) if evaluator_regime_id == "A" else (602, 1)
        return learner_type(
            seed=seed,
            name=f"fresh_{evaluator_regime_id}",
            lifecycle_id=lifecycle,
        )

    def stationary_factory(seed: int) -> FlexibleCueLearner:
        return FlexibleCueLearner(
            seed=seed,
            name="stationary",
            lifecycle_id=(603, 1),
        )

    suite = PrivilegedContinualControlReferenceSuite(
        config=_run_config() if config is None else config,
        protocol=_protocol(),
        common_evaluation_budget=_common_budget(),
        environment_factory=environment_factory,
        probes={
            "A": (ControlProbe((0.0,), (1.0, 0.0)),),
            "B": (ControlProbe((1.0,), (0.0, 1.0)),),
        },
        fresh_learner_factory=fresh_factory,
        stationary_learner_factory=stationary_factory,
        stationary_stream=_stream(seed=(config.seed if config is not None else SEED)),
        oracle_source_factory=lambda seed: FrozenTestOracle(
            seed=seed,
            prepared_steps=prepared_steps,
            score_offset=oracle_score_offset,
            score_semantics=oracle_score_semantics,
        ),
    )
    return suite, environment_instances, prepared_steps


def _role(report: Mapping[str, object], role: str) -> Mapping[str, object]:
    roles = cast(list[Mapping[str, object]], report["reference_roles"])
    return next(record for record in roles if record["role"] == role)


@pytest.fixture(scope="module")
def reference_report() -> dict[str, object]:
    suite, _, _ = _suite()
    return suite.build_report(suite.advance(suite.init(), steps=6))


@pytest.mark.unit
def test_privileged_references_stay_outside_continuing_learner_conditions() -> None:
    suite, environments, prepared = _suite()
    config = suite.to_config()
    assert "conditions" not in config
    assert config["ordinary_conditions_included"] is False
    assert config["reference_roles"] == list(REFERENCE_ROLES)
    assert RETAINED_FRESH_PER_REGIME_ROLE == "retained_fresh_once_per_regime_identity"
    fresh_identity = cast(Mapping[str, object], config[RETAINED_FRESH_PER_REGIME_ROLE])
    assert fresh_identity["initialization_scope"] == "once per evaluator regime identity"
    assert fresh_identity["fresh_per_segment_or_change"] is False
    oracle_identity = cast(Mapping[str, object], config[ORACLE_ACTION_DATA_ROLE])
    assert oracle_identity["score_semantics"] == EXACT_ORACLE_SCORE_SEMANTICS
    assert oracle_identity["stochastic_expected_score_source_supported"] is False
    assert len(environments) == 3
    assert len({id(environment) for environment in environments}) == 3

    state = suite.advance(suite.init(), steps=6)
    report = suite.build_report(state)
    assert prepared == set(range(6))
    assert validate_privileged_control_reference_report(report).valid
    assert report["acceptance_status"] == "not-assessed"
    assert report["scientific_promotion_allowed"] is False
    assert report["claim_thresholds_included"] is False

    for method_name in ("init", "decide", "update", "probe_action"):
        parameters = inspect.signature(getattr(ContinuingControlLearner, method_name)).parameters
        assert all("regime" not in name for name in parameters)


@pytest.mark.unit
def test_reference_roles_disclose_privileges_budgets_and_v2_metrics(
    reference_report: dict[str, object],
) -> None:
    fresh = _role(reference_report, RETAINED_FRESH_PER_REGIME_ROLE)
    stationary = _role(reference_report, STATIONARY_MULTITASK_ROLE)
    oracle = _role(reference_report, ORACLE_ACTION_DATA_ROLE)
    for record in (fresh, stationary, oracle):
        assert record["available"] is True
        assert record["metrics"] is not None
        comparison = cast(Mapping[str, object], record["comparability_disclosure"])
        assert comparison["included_in_ordinary_conditions"] is False
        assert comparison["eligible_as_matched_baseline"] is False

    fresh_extra = cast(Mapping[str, object], fresh["additional_data_usage"])
    assert fresh_extra["persistent_learner_initializations"] == 2
    assert fresh_extra["recurrence_resets"] == 0
    assert fresh_extra["training_transitions"] == 0
    fresh_privileges = cast(Mapping[str, object], fresh["privilege_disclosure"])
    assert fresh_privileges["fresh_per_segment_or_regime_change"] is False
    assert fresh_privileges["same_identity_state_retained_and_reused_on_recurrence"] is True

    stationary_extra = cast(Mapping[str, object], stationary["additional_data_usage"])
    assert stationary_extra["training_transitions"] == 4
    assert stationary_extra["reward_table_scalars_available_to_evaluator"] == 8
    assert stationary_extra["selected_reward_scalars_revealed_to_learner"] == 4
    assert len(cast(list[object], stationary["additional_data_trace"])) == 4

    oracle_extra = cast(Mapping[str, object], oracle["additional_data_usage"])
    assert oracle_extra["environment_action_score_callbacks"] == 6
    assert oracle_extra["environment_action_score_scalars"] == 12
    assert oracle_extra["probe_action_score_scalars_used_for_selection"] == 12
    oracle_trace = cast(Mapping[str, object], oracle["trace"])
    actions = cast(list[Mapping[str, object]], oracle_trace["actions"])
    assert all(action["oracle_action_scores"] is not None for action in actions)
    assert all(action["decision_selected_before_outcome"] is True for action in actions)
    oracle_privileges = cast(Mapping[str, object], oracle["privilege_disclosure"])
    assert oracle_privileges["score_semantics"] == EXACT_ORACLE_SCORE_SEMANTICS
    assert oracle_privileges["stochastic_expected_action_scores_accepted"] is False


@pytest.mark.unit
def test_fresh_recurrence_incompatibility_is_explicitly_unavailable() -> None:
    suite, _, _ = _suite(fresh_bound=True)
    report = suite.build_report(suite.advance(suite.init(), steps=6))
    fresh = _role(report, RETAINED_FRESH_PER_REGIME_ROLE)
    assert fresh["available"] is False
    assert str(fresh["unavailable_reason"]).startswith(
        "retained fresh-per-regime-identity state cannot own recurrence observation"
    )
    assert fresh["metrics"] is None
    applicability = cast(Mapping[str, Mapping[str, object]], fresh["metric_applicability"])
    assert all(value["applicable"] is False for value in applicability.values())
    assert _role(report, STATIONARY_MULTITASK_ROLE)["available"] is True
    assert _role(report, ORACLE_ACTION_DATA_ROLE)["available"] is True
    assert validate_privileged_control_reference_report(report).valid


@pytest.mark.unit
def test_checkpoint_resume_and_atomic_report_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite, _, _ = _suite()
    partial = suite.advance(suite.init(), steps=3)
    checkpoint = tmp_path / "reference-checkpoint.json"
    suite.save_checkpoint(partial, checkpoint)
    restored = suite.load_checkpoint(checkpoint)
    resumed_report = suite.build_report(suite.advance(restored, steps=3))

    uninterrupted_suite, _, _ = _suite()
    uninterrupted_report = uninterrupted_suite.build_report(
        uninterrupted_suite.advance(uninterrupted_suite.init(), steps=6)
    )
    assert resumed_report == uninterrupted_report

    report_path = tmp_path / "reference-report.json"
    save_privileged_control_reference_report(resumed_report, report_path)
    original = report_path.read_text(encoding="utf-8")
    assert load_privileged_control_reference_report(report_path) == resumed_report

    def fail_replace(source: str | Path, target: str | Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(
        "alberta_framework.evaluation.continual_control_reference_suite.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="injected"):
        save_privileged_control_reference_report(resumed_report, report_path)
    assert report_path.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".reference-report.json.*.tmp")) == []

    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_privileged_control_reference_report(duplicate_path)

    nonfinite_path = tmp_path / "nonfinite.json"
    nonfinite_path.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard"):
        load_privileged_control_reference_report(nonfinite_path)


@pytest.mark.unit
def test_report_validator_fails_closed_on_tamper(
    reference_report: dict[str, object],
) -> None:
    def invalid(payload: dict[str, object]) -> None:
        assert not validate_privileged_control_reference_report(payload).valid

    seed_tamper = copy.deepcopy(reference_report)
    suite_config = cast(dict[str, object], seed_tamper["suite_config"])
    run_config = cast(dict[str, object], suite_config["run_config"])
    run_config["seed"] = 30
    invalid(seed_tamper)

    regime_tamper = copy.deepcopy(reference_report)
    fresh = cast(list[dict[str, object]], regime_tamper["reference_roles"])[0]
    trace = cast(dict[str, object], fresh["trace"])
    actions = cast(list[dict[str, object]], trace["actions"])
    actions[0]["evaluator_regime_id"] = "B"
    invalid(regime_tamper)

    data_budget_tamper = copy.deepcopy(reference_report)
    roles = cast(list[dict[str, object]], data_budget_tamper["reference_roles"])
    stationary_usage = cast(dict[str, object], roles[1]["additional_data_usage"])
    stationary_usage["training_transitions"] = 3
    invalid(data_budget_tamper)

    metric_tamper = copy.deepcopy(reference_report)
    oracle = cast(list[dict[str, object]], metric_tamper["reference_roles"])[2]
    metrics = cast(dict[str, object], oracle["metrics"])
    metrics["lifetime_return"] = -999.0
    invalid(metric_tamper)

    source_tamper = copy.deepcopy(reference_report)
    sources = cast(dict[str, object], source_tamper["source_core_sha256"])
    sources["alberta_framework/evaluation/continual_control_reference_suite.py"] = "0" * 64
    invalid(source_tamper)

    condition_injection = copy.deepcopy(reference_report)
    injected_config = cast(dict[str, object], condition_injection["suite_config"])
    injected_config["conditions"] = []
    invalid(condition_injection)

    fresh_semantics_tamper = copy.deepcopy(reference_report)
    fresh_suite = cast(dict[str, object], fresh_semantics_tamper["suite_config"])
    fresh_identity = cast(
        dict[str, object],
        fresh_suite[RETAINED_FRESH_PER_REGIME_ROLE],
    )
    fresh_identity["fresh_per_segment_or_change"] = True
    invalid(fresh_semantics_tamper)

    oracle_semantics_tamper = copy.deepcopy(reference_report)
    oracle_record = cast(list[dict[str, object]], oracle_semantics_tamper["reference_roles"])[2]
    oracle_privileges = cast(dict[str, object], oracle_record["privilege_disclosure"])
    oracle_privileges["stochastic_expected_action_scores_accepted"] = True
    invalid(oracle_semantics_tamper)

    numeric_tamper = copy.deepcopy(reference_report)
    numeric_config = cast(dict[str, object], numeric_tamper["suite_config"])
    numeric_run = cast(dict[str, object], numeric_config["run_config"])
    numeric_run["seed"] = 29.0
    invalid(numeric_tamper)


@pytest.mark.unit
def test_seed_and_exact_extra_data_budget_mismatches_fail_closed() -> None:
    with pytest.raises(ValueError, match="suite seed"):
        _suite(environment_seed_offset=1)

    wrong_budget = PrivilegedReferenceRunConfig(
        suite_id="tests.wrong-budget.v1",
        seed=SEED,
        fresh_lifecycle_id=(701, 1),
        stationary_lifecycle_id=(702, 1),
        oracle_lifecycle_id=(703, 1),
        extra_data_budget=PrivilegedReferenceExtraDataBudget(
            stationary_transition_limit=3,
            stationary_decision_call_limit=4,
            stationary_update_call_limit=4,
            stationary_backward_call_limit=0,
            stationary_reward_table_scalar_limit=8,
            oracle_callback_limit=6,
            oracle_action_score_scalar_limit=12,
            oracle_probe_action_score_scalar_limit=12,
        ),
    )
    with pytest.raises(ValueError, match="stationary_transition_limit"):
        _suite(config=wrong_budget)

    bad_oracle_suite, _, _ = _suite(oracle_score_offset=0.5)
    with pytest.raises(ValueError, match="realized outcome"):
        bad_oracle_suite.advance(bad_oracle_suite.init(), steps=1)

    with pytest.raises(ValueError, match="stochastic expected scores"):
        _suite(oracle_score_semantics="stochastic_expected_action_score")


@pytest.mark.unit
def test_checkpoint_rejects_state_and_source_tamper(tmp_path: Path) -> None:
    suite, _, _ = _suite()
    checkpoint = tmp_path / "checkpoint.json"
    suite.save_checkpoint(suite.advance(suite.init(), steps=2), checkpoint)
    payload = cast(
        dict[str, object],
        json.loads(checkpoint.read_text(encoding="utf-8")),
    )
    state = cast(dict[str, object], payload["state"])
    state["step"] = 3
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        suite.load_checkpoint(checkpoint)

    suite.save_checkpoint(suite.advance(suite.init(), steps=2), checkpoint)
    source_payload = cast(
        dict[str, object],
        json.loads(checkpoint.read_text(encoding="utf-8")),
    )
    hashes = cast(dict[str, object], source_payload["source_core_sha256"])
    hashes["alberta_framework/evaluation/continual_control_reference_suite.py"] = "0" * 64
    checkpoint.write_text(json.dumps(source_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source-core"):
        suite.load_checkpoint(checkpoint)
