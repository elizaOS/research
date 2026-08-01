from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.evaluation.continual_control_campaign import (
    ACCEPTANCE_STATUS,
    BOOTSTRAP_METHOD,
    CampaignSeededControlEnvironment,
    CampaignSeededControlLearner,
    CampaignSeedStratum,
    PairedContinualControlCampaign,
    PairedControlCampaignConfig,
    StratifiedBootstrapConfig,
    load_continual_control_campaign,
    save_continual_control_campaign,
    validate_continual_control_campaign,
)
from alberta_framework.evaluation.continual_control_evaluator import (
    ContinualControlEvaluator,
    ContinualControlRunState,
    ContinuingControlBudget,
    ContinuingControlProtocol,
    ControlDecision,
    ControlEnvironmentUpdate,
    ControlProbe,
    ControlTransition,
    FrozenActionControlBaseline,
    RunningRewardBanditControlBaseline,
)

SCHEDULE = ("A", "A", "B", "B", "A", "A")
SEEDS = (11, 12, 13, 14)


class ConstantDurationClock:
    """Make every recorded operation take exactly one microsecond."""

    def __init__(self) -> None:
        self._now = 0

    def __call__(self) -> int:
        now = self._now
        self._now += 1_000
        return now


@dataclass(frozen=True)
class SeededEnvironmentState:
    step: int
    observation: tuple[float, ...]


class SeededSwitchEnvironment:
    """Functional switched environment with deterministic seed-dependent rewards."""

    def __init__(self, *, seed: int, schedule: tuple[str, ...] = SCHEDULE) -> None:
        self._seed = seed
        self._schedule = schedule

    @property
    def n_actions(self) -> int:
        return 2

    def to_config(self) -> dict[str, object]:
        return {
            "type": "SeededSwitchEnvironment",
            "schema_version": "tests.seeded_switch_environment.v1",
            "seed": self._seed,
            "schedule": list(self._schedule),
        }

    def _cue(self, step: int) -> tuple[float, ...]:
        regime = self._schedule[min(step, len(self._schedule) - 1)]
        return (0.0 if regime == "A" else 1.0,)

    def init(self) -> SeededEnvironmentState:
        return SeededEnvironmentState(0, self._cue(0))

    def observation(self, state: Any) -> tuple[float, ...]:
        return cast(SeededEnvironmentState, state).observation

    def step(
        self,
        state: Any,
        decision: ControlDecision,
        evaluator_regime_id: str,
    ) -> ControlEnvironmentUpdate:
        resolved = cast(SeededEnvironmentState, state)
        if evaluator_regime_id != self._schedule[resolved.step]:
            raise ValueError("evaluator and environment schedules differ")
        if decision.observation != resolved.observation:
            raise ValueError("environment received a stale observation")
        expected_action = 0 if evaluator_regime_id == "A" else 1
        base_reward = 1.0 if decision.action == expected_action else -1.0
        jitter_code = (self._seed * 17 + resolved.step * 7 + decision.action * 11) % 9
        reward = base_reward + (jitter_code - 4) * 0.01
        completed = resolved.step + 1
        terminated = completed == len(self._schedule)
        next_observation = self._cue(0) if terminated else self._cue(completed)
        return ControlEnvironmentUpdate(
            state=SeededEnvironmentState(completed, next_observation),
            transition=ControlTransition(
                observation=decision.observation,
                action=decision.action,
                decision_id=decision.decision_id,
                reward=reward,
                discount=0.0 if terminated else 0.9,
                terminated=terminated,
                truncated=False,
                bootstrap_observation=next_observation,
                reset_observation=next_observation if terminated else None,
            ),
        )

    def state_to_config(self, state: Any) -> object:
        resolved = cast(SeededEnvironmentState, state)
        return {
            "step": resolved.step,
            "observation": list(resolved.observation),
        }

    def state_from_config(self, payload: object) -> SeededEnvironmentState:
        if not isinstance(payload, Mapping) or set(payload) != {"step", "observation"}:
            raise ValueError("seeded environment state is invalid")
        step = payload["step"]
        observation = payload["observation"]
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("seeded environment step must be an integer")
        if not isinstance(observation, list):
            raise ValueError("seeded environment observation must be an array")
        return SeededEnvironmentState(
            step,
            tuple(float(value) for value in observation),
        )


class PartialEvaluator(ContinualControlEvaluator):
    def advance(
        self,
        state: ContinualControlRunState,
        *,
        steps: int = 1,
    ) -> ContinualControlRunState:
        return super().advance(state, steps=steps - 1)


def _protocol(*, schedule: tuple[str, ...] = SCHEDULE) -> ContinuingControlProtocol:
    return ContinuingControlProtocol(
        protocol_id="tests.paired-control-campaign.v1",
        higher_is_better=True,
        regime_schedule=schedule,
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


def _budget() -> ContinuingControlBudget:
    return ContinuingControlBudget(
        transition_limit=6,
        decision_call_limit=6,
        environment_call_limit=6,
        update_call_limit=6,
        probe_call_limit=6,
        backward_call_limit=0,
        persistent_state_bytes_limit=10_000,
        state_scalar_count_limit=1_000,
        trainable_parameter_count_limit=None,
        stored_decision_id_limit=6,
    )


def _campaign_config() -> PairedControlCampaignConfig:
    return PairedControlCampaignConfig(
        campaign_id="tests.paired-control-development.v1",
        declared_seeds=SEEDS,
        bootstrap=StratifiedBootstrapConfig(
            bootstrap_seed=77,
            resamples=128,
            confidence_level=0.9,
            strata=(
                CampaignSeedStratum("lower", (11, 12)),
                CampaignSeedStratum("upper", (13, 14)),
            ),
        ),
    )


def _evaluator(
    seed: int,
    *,
    reverse_baselines: bool = False,
    mismatched_condition_seed: bool = False,
    evaluator_type: type[ContinualControlEvaluator] = ContinualControlEvaluator,
) -> ContinualControlEvaluator:
    environment = CampaignSeededControlEnvironment(
        SeededSwitchEnvironment(seed=seed),
        campaign_seed=seed,
    )
    candidate = CampaignSeededControlLearner(
        FrozenActionControlBaseline(
            n_actions=2,
            action=0,
            name="candidate",
            lifecycle_id=(101, 1),
        ),
        campaign_seed=seed,
    )
    baselines = [
        CampaignSeededControlLearner(
            FrozenActionControlBaseline(
                n_actions=2,
                action=1,
                name="frozen_one",
                lifecycle_id=(102, 1),
            ),
            campaign_seed=seed + int(mismatched_condition_seed),
        ),
        CampaignSeededControlLearner(
            RunningRewardBanditControlBaseline(
                n_actions=2,
                name="running_bandit",
                lifecycle_id=(103, 1),
            ),
            campaign_seed=seed,
        ),
    ]
    if reverse_baselines:
        baselines.reverse()
    return evaluator_type(
        run_id=f"tests.paired-control-seed-{seed}.v1",
        protocol=_protocol(),
        environment=environment,
        probes={
            "A": (ControlProbe((0.0,), (1.0, 0.0)),),
            "B": (ControlProbe((1.0,), (0.0, 1.0)),),
        },
        candidate=candidate,
        baselines=baselines,
        budget=_budget(),
        clock_ns=ConstantDurationClock(),
        latency_measurement_method="deterministic test clock: 1000 ns per operation",
    )


def _comparative_metric(
    artifact: Mapping[str, object],
    *,
    baseline: str,
    metric_name: str,
) -> Mapping[str, object]:
    comparisons = cast(list[Mapping[str, object]], artifact["paired_comparisons"])
    comparison = next(value for value in comparisons if value["baseline"] == baseline)
    metrics = cast(list[Mapping[str, object]], comparison["metrics"])
    return next(value for value in metrics if value["name"] == metric_name)


@pytest.fixture(scope="module")
def campaign_artifact() -> dict[str, object]:
    return PairedContinualControlCampaign(_campaign_config(), _evaluator).run()


@pytest.mark.unit
def test_campaign_config_rejects_invalid_seed_universe_and_noncanonical_numeric() -> None:
    with pytest.raises(ValueError, match="unique"):
        PairedControlCampaignConfig(
            campaign_id="tests.duplicate.v1",
            declared_seeds=(1, 1),
            bootstrap=StratifiedBootstrapConfig(
                bootstrap_seed=3,
                resamples=8,
                confidence_level=0.9,
                strata=(CampaignSeedStratum("all", (1,)),),
            ),
        )
    with pytest.raises(ValueError, match="partition"):
        PairedControlCampaignConfig(
            campaign_id="tests.missing.v1",
            declared_seeds=(1, 2),
            bootstrap=StratifiedBootstrapConfig(
                bootstrap_seed=3,
                resamples=8,
                confidence_level=0.9,
                strata=(CampaignSeedStratum("partial", (1,)),),
            ),
        )
    with pytest.raises(ValueError, match="order"):
        PairedControlCampaignConfig(
            campaign_id="tests.reordered.v1",
            declared_seeds=(1, 2),
            bootstrap=StratifiedBootstrapConfig(
                bootstrap_seed=3,
                resamples=8,
                confidence_level=0.9,
                strata=(CampaignSeedStratum("all", (2, 1)),),
            ),
        )
    with pytest.raises(ValueError, match="JSON float"):
        StratifiedBootstrapConfig(
            bootstrap_seed=3,
            resamples=8,
            confidence_level=1,
            strata=(CampaignSeedStratum("all", (1, 2)),),
        )
    with pytest.raises(ValueError, match="immutable tuple"):
        CampaignSeedStratum("mutable", [1, 2])  # type: ignore[arg-type]


@pytest.mark.unit
def test_campaign_runs_one_factory_per_seed_and_builds_deterministic_pairs() -> None:
    factory_calls: list[int] = []

    def tracked_factory(seed: int) -> ContinualControlEvaluator:
        factory_calls.append(seed)
        return _evaluator(seed)

    first = PairedContinualControlCampaign(_campaign_config(), tracked_factory).run()
    second = PairedContinualControlCampaign(_campaign_config(), _evaluator).run()
    assert factory_calls == list(SEEDS)
    assert first == second
    assert first["acceptance_status"] == ACCEPTANCE_STATUS
    assert first["development_only"] is True
    assert first["accepted_scientific_evidence"] is False
    assert validate_continual_control_campaign(first).valid

    seed_runs = cast(list[Mapping[str, object]], first["seed_runs"])
    assert [record["seed"] for record in seed_runs] == list(SEEDS)
    assert len({record["metric_applicability_sha256"] for record in seed_runs}) == 1
    for seed, seed_record in zip(SEEDS, seed_runs, strict=True):
        report = cast(Mapping[str, object], seed_record["report"])
        evaluator_config = cast(Mapping[str, object], report["evaluator_config"])
        protocol = cast(Mapping[str, object], evaluator_config["protocol"])
        assert protocol["regime_schedule"] == list(SCHEDULE)
        environment = cast(Mapping[str, object], evaluator_config["environment"])
        assert environment["campaign_seed"] == seed
        conditions = cast(list[Mapping[str, object]], evaluator_config["conditions"])
        assert [condition["role"] for condition in conditions] == [
            "candidate",
            "baseline",
            "baseline",
        ]
        for condition in conditions:
            learner = cast(Mapping[str, object], condition["learner"])
            assert learner["campaign_seed"] == seed
    source_hashes = cast(Mapping[str, str], first["source_core_sha256"])
    assert set(source_hashes) == {
        "alberta_framework/evaluation/continual_control_campaign.py",
        "alberta_framework/evaluation/continual_control_evaluator.py",
    }

    lifetime = _comparative_metric(
        first,
        baseline="frozen_one",
        metric_name="lifetime_return",
    )
    assert lifetime["direction"] == "higher"
    assert lifetime["available"] is True
    pairs = cast(list[Mapping[str, object]], lifetime["per_seed"])
    for pair in pairs:
        candidate = cast(float, pair["candidate"])
        baseline = cast(float, pair["baseline"])
        assert pair["direction_normalized_difference"] == pytest.approx(candidate - baseline)
    assert lifetime["mean_direction_normalized_difference"] == pytest.approx(
        sum(cast(float, pair["direction_normalized_difference"]) for pair in pairs) / len(pairs)
    )
    interval = cast(Mapping[str, object], lifetime["confidence_interval"])
    assert interval["method"] == BOOTSTRAP_METHOD
    assert interval["resamples"] == 128
    assert interval["confidence_level"] == 0.9
    assert cast(float, interval["lower"]) <= cast(float, interval["upper"])

    stability = _comparative_metric(
        first,
        baseline="frozen_one",
        metric_name="stability.mean_gap",
    )
    assert stability["direction"] == "lower"
    stability_pairs = cast(list[Mapping[str, object]], stability["per_seed"])
    for pair in stability_pairs:
        assert pair["direction_normalized_difference"] == pytest.approx(
            cast(float, pair["baseline"]) - cast(float, pair["candidate"])
        )

    unavailable = _comparative_metric(
        first,
        baseline="frozen_one",
        metric_name="forward_transfer.per_regime/A",
    )
    assert unavailable["available"] is False
    assert unavailable["confidence_interval"] is None
    assert unavailable["mean_direction_normalized_difference"] is None
    assert all(
        pair["direction_normalized_difference"] is None
        for pair in cast(list[Mapping[str, object]], unavailable["per_seed"])
    )


@pytest.mark.unit
def test_campaign_rejects_cross_seed_reorder_seed_mismatch_and_partial_run() -> None:
    with pytest.raises(ValueError, match="identity differs"):
        PairedContinualControlCampaign(
            _campaign_config(),
            lambda seed: _evaluator(seed, reverse_baselines=seed == 12),
        ).run()
    with pytest.raises(ValueError, match="campaign_seed"):
        PairedContinualControlCampaign(
            _campaign_config(),
            lambda seed: _evaluator(seed, mismatched_condition_seed=seed == 12),
        ).run()
    with pytest.raises(ValueError, match="partial run"):
        PairedContinualControlCampaign(
            _campaign_config(),
            lambda seed: _evaluator(seed, evaluator_type=PartialEvaluator),
        ).run()


@pytest.mark.unit
def test_campaign_validator_rejects_tamper_and_noncanonical_numeric(
    campaign_artifact: dict[str, object],
) -> None:
    def invalid(mutated: dict[str, object]) -> None:
        assert not validate_continual_control_campaign(mutated).valid

    missing = copy.deepcopy(campaign_artifact)
    cast(list[object], missing["seed_runs"]).pop()
    invalid(missing)

    duplicate = copy.deepcopy(campaign_artifact)
    duplicate_runs = cast(list[object], duplicate["seed_runs"])
    duplicate_runs[1] = copy.deepcopy(duplicate_runs[0])
    invalid(duplicate)

    reordered = copy.deepcopy(campaign_artifact)
    reordered_runs = cast(list[object], reordered["seed_runs"])
    reordered_runs[0], reordered_runs[1] = reordered_runs[1], reordered_runs[0]
    invalid(reordered)

    scalar_tamper = copy.deepcopy(campaign_artifact)
    scalar_runs = cast(list[dict[str, object]], scalar_tamper["seed_runs"])
    scalar_conditions = cast(list[dict[str, object]], scalar_runs[0]["scalar_metrics"])
    scalar_metrics = cast(list[dict[str, object]], scalar_conditions[0]["metrics"])
    scalar_metrics[0]["value"] = 123.0
    invalid(scalar_tamper)

    interval_tamper = copy.deepcopy(campaign_artifact)
    comparisons = cast(list[dict[str, object]], interval_tamper["paired_comparisons"])
    comparison_metrics = cast(list[dict[str, object]], comparisons[0]["metrics"])
    interval = cast(dict[str, object], comparison_metrics[0]["confidence_interval"])
    interval["lower"] = -999.0
    invalid(interval_tamper)

    applicability_tamper = copy.deepcopy(campaign_artifact)
    applicability_runs = cast(list[dict[str, object]], applicability_tamper["seed_runs"])
    report = cast(dict[str, object], applicability_runs[0]["report"])
    conditions = cast(list[dict[str, object]], report["conditions"])
    metrics = cast(dict[str, object], conditions[1]["metrics"])
    applicability = cast(dict[str, object], metrics["metric_applicability"])
    applicability["lifetime_return"] = {"applicable": False}
    invalid(applicability_tamper)

    source_tamper = copy.deepcopy(campaign_artifact)
    source_hashes = cast(dict[str, object], source_tamper["source_core_sha256"])
    source_hashes["alberta_framework/evaluation/continual_control_campaign.py"] = "0" * 64
    invalid(source_tamper)

    numeric_tamper = copy.deepcopy(campaign_artifact)
    config = cast(dict[str, object], numeric_tamper["campaign_config"])
    seeds = cast(list[object], config["declared_seeds"])
    seeds[0] = 11.0
    invalid(numeric_tamper)


@pytest.mark.unit
def test_campaign_atomic_save_strict_load_and_failure_cleanup(
    campaign_artifact: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "campaign.json"
    save_continual_control_campaign(campaign_artifact, destination)
    original = destination.read_text(encoding="utf-8")
    assert load_continual_control_campaign(destination) == campaign_artifact

    def fail_replace(source: str | Path, target: str | Path) -> None:
        del source, target
        raise OSError("injected replace failure")

    monkeypatch.setattr(
        "alberta_framework.evaluation.continual_control_campaign.os.replace",
        fail_replace,
    )
    with pytest.raises(OSError, match="injected"):
        save_continual_control_campaign(campaign_artifact, destination)
    assert destination.read_text(encoding="utf-8") == original
    assert list(tmp_path.glob(".campaign.json.*.tmp")) == []

    duplicate_json = tmp_path / "duplicate.json"
    duplicate_json.write_text('{"x": 1, "x": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_continual_control_campaign(duplicate_json)

    nonfinite_json = tmp_path / "nonfinite.json"
    nonfinite_json.write_text('{"x": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-standard"):
        load_continual_control_campaign(nonfinite_json)
