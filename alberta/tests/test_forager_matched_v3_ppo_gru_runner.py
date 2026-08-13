"""Cheap adversarial checks for the matched-v3 PPO-GRU full driver contract."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru_runner as runner


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _resign(receipt: dict[str, Any]) -> bytes:
    body = copy.deepcopy(receipt)
    body.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    return _canonical(receipt)


_RUNTIME_IDENTITY = _canonical(
    {
        "schema_version": "alberta.synthetic_runtime.v1",
        "classification": "synthetic_engineering_non_authorizing",
        "components": {"backend": "fake", "foragax": "fake", "jax": "fake"},
        "claims": {
            "execution_ready": False,
            "execution_authorized": False,
            "scientific_promotion_allowed": False,
            "performance_claim_allowed": False,
            "universal_sota_claim_allowed": False,
            "authority_granted": False,
        },
    }
)


def _production_runtime_identity_bytes() -> bytes:
    identity = bridge.MatchedV3ForagaxRuntimeIdentity(
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        default_prng_impl="threefry2x32",
        threefry_partitionable=True,
        jax_enable_x64=False,
        backend="cpu",
        foragax_version="0.55.0",
        foragax_install_tree_sha256=bridge.FORAGAX_INSTALL_TREE_SHA256,
        foragax_package_root="/synthetic/foragax",
        runtime_qualified=False,
    )
    return _canonical(runner._bridge_runtime_identity_dict(identity))


_TINY_GEOMETRY = runner.PPOGRURunnerGeometry(
    horizon=16,
    rollout_steps=8,
    segment_steps=2,
    update_epochs=4,
)


def _observation(index: int) -> jnp.ndarray:
    value = np.zeros((9, 9, 3), dtype=np.float32)
    value[index % 9, (index // 9) % 9, index % 3] = 1.0
    return jnp.asarray(value)


@dataclass(frozen=True)
class _FakeBridgeState:
    environment_seed: int
    observation: jnp.ndarray
    reset_count: int
    step_count: int

    @property
    def environment_key_use_count(self) -> int:
        return self.reset_count + self.step_count


@dataclass(frozen=True)
class _FakeTransition:
    state: _FakeBridgeState
    action: int
    reward: int
    done: bool = False
    truncated: bool = False
    info_validated: bool = True


@dataclass(frozen=True)
class _FakeTrainState:
    optimizer_updates: int


class _Harness:
    def __init__(
        self,
        *,
        initialization_agent_draws: int = 1,
        initialization_environment_draws: int = 0,
        update_delta: int = 1,
        stale_bridge_state: bool = False,
        reverse_segment_payloads: bool = False,
        missing_segment: bool = False,
        nonfinite_field: str | None = None,
        bridge_failure: bool = False,
        replace_model: bool = False,
    ) -> None:
        self.initialization_agent_draws = initialization_agent_draws
        self.initialization_environment_draws = initialization_environment_draws
        self.update_delta = update_delta
        self.stale_bridge_state = stale_bridge_state
        self.reverse_segment_payloads = reverse_segment_payloads
        self.missing_segment = missing_segment
        self.nonfinite_field = nonfinite_field
        self.bridge_failure = bridge_failure
        self.replace_model = replace_model
        self.bridge_initialize_calls = 0
        self.bridge_step_calls = 0
        self.runtime_parse_calls = 0
        self.rollout_traces: list[runner.PPOGRURolloutTrace] = []
        self.orders: list[tuple[int, ...]] = []
        self.update_indices: list[int] = []
        self.learning_rates: list[float] = []

    def initialize_bridge(self, environment_seed: int) -> _FakeBridgeState:
        self.bridge_initialize_calls += 1
        return _FakeBridgeState(environment_seed, _observation(0), 1, 0)

    def step_bridge(self, state: _FakeBridgeState, action: int) -> _FakeTransition:
        self.bridge_step_calls += 1
        if self.bridge_failure:
            raise RuntimeError("single synthetic failure")
        reward_support = (1, 0, -1, 30)
        if self.nonfinite_field == "reward":
            reward: Any = float("nan")
        else:
            reward = reward_support[state.step_count % len(reward_support)]
        next_state = (
            state
            if self.stale_bridge_state
            else _FakeBridgeState(
                state.environment_seed,
                _observation(state.step_count + 1),
                1,
                state.step_count + 1,
            )
        )
        return _FakeTransition(next_state, action, reward)

    def parse_runtime_identity(self, raw: bytes) -> dict[str, Any]:
        self.runtime_parse_calls += 1
        return cast(dict[str, Any], json.loads(raw.decode("ascii")))

    def initialize_training(
        self,
        config: ppo_gru.PPOGRUConfig,
        rng_state: ppo_gru.PPOGRURNGState,
    ) -> tuple[runner.PPOGRUTrainingHandle, ppo_gru.PPOGRURNGState]:
        del config
        current = rng_state
        for _ in range(self.initialization_environment_draws):
            current, _ = ppo_gru.next_ppo_gru_environment_key(current)
        for _ in range(self.initialization_agent_draws):
            current, _ = ppo_gru.next_ppo_gru_agent_key(current)
        return (
            runner.PPOGRUTrainingHandle(
                model="synthetic-model",
                state=_FakeTrainState(optimizer_updates=0),
            ),
            current,
        )

    def evaluate_step(
        self,
        training: runner.PPOGRUTrainingHandle,
        carry: jnp.ndarray,
        observation: jnp.ndarray,
        reset_before: bool,
    ) -> runner.PPOGRUStepEvaluation:
        del training, observation, reset_before
        outgoing = carry + jnp.asarray(0.001, dtype=jnp.float32)
        logits = jnp.asarray([0.25, -0.5, 0.75, 0.0], dtype=jnp.float32)
        value = jnp.asarray(float(np.asarray(carry)[0]), dtype=jnp.float32)
        if self.nonfinite_field == "carry":
            outgoing = outgoing.at[0].set(jnp.nan)
        elif self.nonfinite_field == "logits":
            logits = logits.at[0].set(jnp.inf)
        elif self.nonfinite_field == "value":
            value = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return runner.PPOGRUStepEvaluation(
            outgoing_carry=outgoing,
            logits=logits,
            value=value,
        )

    def validate_rollout(
        self,
        training: runner.PPOGRUTrainingHandle,
        trace: runner.PPOGRURolloutTrace,
        config: ppo_gru.PPOGRUConfig,
    ) -> runner.PPOGRURolloutTrace:
        del training, config
        self.rollout_traces.append(trace)
        return trace

    def build_segments(
        self,
        validated: Any,
        order: tuple[int, ...],
        geometry: runner.PPOGRURunnerGeometry,
    ) -> tuple[runner.PPOGRURunnerSegment, ...]:
        del validated
        self.orders.append(order)
        selected_order = tuple(reversed(order)) if self.reverse_segment_payloads else order
        values = tuple(
            runner.PPOGRURunnerSegment(
                segment_id=segment_id,
                time_indices=tuple(
                    range(
                        segment_id * geometry.segment_steps,
                        (segment_id + 1) * geometry.segment_steps,
                    )
                ),
                payload=segment_id,
            )
            for segment_id in selected_order
        )
        return values[:-1] if self.missing_segment else values

    def update_segment(
        self,
        training: runner.PPOGRUTrainingHandle,
        payload: Any,
        expected_update_index: int,
        expected_learning_rate: float,
    ) -> runner.PPOGRUTrainingHandle:
        del payload
        state = training.state
        assert type(state) is _FakeTrainState
        self.update_indices.append(expected_update_index)
        self.learning_rates.append(expected_learning_rate)
        return runner.PPOGRUTrainingHandle(
            model=(object() if self.replace_model else training.model),
            state=_FakeTrainState(
                optimizer_updates=state.optimizer_updates + self.update_delta
            ),
        )

    def optimizer_update_count(self, training: runner.PPOGRUTrainingHandle) -> int:
        state = training.state
        assert type(state) is _FakeTrainState
        return state.optimizer_updates

    def dependencies(self) -> runner.PPOGRURunnerDependencies:
        return runner.PPOGRURunnerDependencies(
            classification="synthetic_engineering_only",
            initialize_bridge=self.initialize_bridge,
            step_bridge=self.step_bridge,
            parse_runtime_identity=self.parse_runtime_identity,
            initialize_training=self.initialize_training,
            evaluate_step=self.evaluate_step,
            validate_rollout=self.validate_rollout,
            build_segments=self.build_segments,
            update_segment=self.update_segment,
            optimizer_update_count=self.optimizer_update_count,
        )


def _run(harness: _Harness) -> runner.PPOGRURunnerOutcome:
    return runner.run_ppo_gru_engineering_driver(
        environment_seed=17,
        agent_seed=29,
        runtime_identity_bytes=_RUNTIME_IDENTITY,
        geometry=_TINY_GEOMETRY,
        dependencies=harness.dependencies(),
    )


@pytest.mark.unit
def test_descriptor_binds_exact_core_bridge_and_closed_production_schedule() -> None:
    raw = runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
    descriptor = runner.parse_matched_v3_ppo_gru_runner_descriptor(raw)
    assert hashlib.sha256(raw).hexdigest() == runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256
    assert runner.PPO_GRU_RUNNER_DESCRIPTOR_SHA256 == (
        "e9cfa6785ef48783224f548fa17db0f8291ee1a47ef29f098692c31beb5f00b2"
    )
    assert descriptor["production_geometry"] == {
        "horizon": 499_712,
        "rollout_steps": 512,
        "segment_steps": 128,
        "segments_per_rollout": 4,
        "update_epochs": 4,
        "rollout_count": 976,
        "optimizer_updates_per_rollout": 16,
        "optimizer_update_count": 15_616,
    }
    core = descriptor["dependencies"]["ppo_gru_core"]
    assert core["configuration_sha256"] == ppo_gru.PPO_GRU_CONFIGURATION_SHA256
    assert core["source_descriptor_sha256"] == ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
    assert core["implementation_source_sha256"] == hashlib.sha256(
        Path(ppo_gru.__file__).read_bytes()
    ).hexdigest()
    shared_bridge = descriptor["dependencies"]["foragax_bridge"]
    assert shared_bridge["descriptor_sha256"] == bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
    assert shared_bridge["implementation_source_sha256"] == hashlib.sha256(
        Path(bridge.__file__).read_bytes()
    ).hexdigest()
    assert descriptor["rng_ownership"] == {
        "implementation": "threefry2x32",
        "environment_owner": "shared_foragax_bridge",
        "ppo_environment_key_consumption": 0,
        "agent_owner": "ppo_gru_core_rng_state",
        "parameter_initialization_draws": 1,
        "action_draws": 499_712,
        "action_draws_per_interaction": 1,
        "permutation_draws": 3_904,
        "permutation_draws_per_epoch": 1,
        "total_agent_draws": 503_617,
    }
    assert descriptor["seed_provenance"] == {
        "classification": "caller_supplied_unverified",
        "upstream_receipt_bound": False,
        "protected_seed_status": "unverified",
    }
    assert descriptor["receipts"][
        "production_emission_requires_process_local_completion_capability"
    ] is True
    assert descriptor["receipts"][
        "persisted_parser_is_structural_not_execution_attestation"
    ] is True
    assert descriptor["trajectory"]["raw_reward_trace_retained"] is True
    assert descriptor["trajectory"]["raw_reward_trace_encoding"] == (
        "signed_int8_twos_complement"
    )
    assert descriptor["trajectory"]["score_reduction"] == "exact_int64_sum"
    assert descriptor["trajectory"]["score_scaling"] == "none"
    assert descriptor["claims"] == {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }


@pytest.mark.unit
def test_descriptor_is_exact_canonical_detached_and_rejects_mutation() -> None:
    raw = runner.canonical_matched_v3_ppo_gru_runner_descriptor_bytes()
    detached = runner.matched_v3_ppo_gru_runner_descriptor()
    detached["claims"]["execution_authorized"] = True
    assert runner.matched_v3_ppo_gru_runner_descriptor()["claims"][
        "execution_authorized"
    ] is False
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="canonical"):
        runner.parse_matched_v3_ppo_gru_runner_descriptor(b" " + raw)
    mutation = json.loads(raw)
    mutation["claims"]["execution_authorized"] = True
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="frozen"):
        runner.parse_matched_v3_ppo_gru_runner_descriptor(_canonical(mutation))


@pytest.mark.unit
def test_tiny_driver_closes_all_counters_linear_schedule_and_raw_score() -> None:
    harness = _Harness()
    outcome = _run(harness)
    assert outcome.classification == "synthetic_engineering_complete"
    assert outcome.environment_interactions == 16
    assert outcome.rollout_count == 2
    assert outcome.optimizer_update_count == 32
    assert outcome.parameter_initialization_draw_count == 1
    assert outcome.action_draw_count == 16
    assert outcome.permutation_draw_count == 8
    assert outcome.total_agent_draw_count == 25
    assert outcome.ppo_environment_draw_count == 0
    assert outcome.bridge_reset_count == 1
    assert outcome.bridge_step_count == 16
    assert outcome.bridge_environment_key_use_count == 17
    assert outcome.raw_reward_trace == bytes([1, 0, 255, 30] * 4)
    assert outcome.raw_reward_trace_sha256 == hashlib.sha256(
        outcome.raw_reward_trace
    ).hexdigest()
    assert int(np.sum(np.frombuffer(outcome.raw_reward_trace, dtype=np.int8))) == 120
    assert outcome.raw_cumulative_score == 120
    assert outcome.production_horizon_complete is False
    assert harness.bridge_initialize_calls == 1
    assert harness.bridge_step_calls == 16
    assert harness.runtime_parse_calls == 2
    assert len(harness.rollout_traces) == 2
    assert len(harness.orders) == 8
    assert all(sorted(order) == [0, 1, 2, 3] for order in harness.orders)
    assert harness.update_indices == list(range(32))
    np.testing.assert_allclose(harness.learning_rates[:16], 0.00025)
    np.testing.assert_allclose(harness.learning_rates[16:], 0.000125)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("bytes_without_digest", "digest drifted"),
        ("membership", "outside exact Forager support"),
        ("score", "exact int64 reward-trace sum"),
        ("mutable", "immutable exact bytes"),
    ],
)
def test_raw_reward_trace_tampering_membership_and_score_fail_closed(
    mutation: str,
    message: str,
) -> None:
    outcome = _run(_Harness())
    changed = bytearray(outcome.raw_reward_trace)
    changed[0] = 2 if mutation == "membership" else 0
    if mutation == "bytes_without_digest":
        tampered = dataclasses.replace(outcome, raw_reward_trace=bytes(changed))
    elif mutation == "membership":
        raw = bytes(changed)
        tampered = dataclasses.replace(
            outcome,
            raw_reward_trace=raw,
            raw_reward_trace_sha256=hashlib.sha256(raw).hexdigest(),
        )
    elif mutation == "score":
        tampered = dataclasses.replace(
            outcome,
            raw_cumulative_score=outcome.raw_cumulative_score + 1,
        )
    else:
        tampered = dataclasses.replace(
            outcome,
            raw_reward_trace=cast(bytes, changed),
        )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
        runner.canonical_ppo_gru_engineering_receipt_bytes(tampered)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("carry", "outgoing carry.*finite"),
        ("logits", "logits must be finite"),
        ("value", "value must be finite"),
        ("reward", "reward is outside"),
    ],
)
def test_nonfinite_driver_values_fail_closed(field: str, message: str) -> None:
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
        _run(_Harness(nonfinite_field=field))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("harness", "message"),
    [
        (_Harness(initialization_agent_draws=2), "agent draw count"),
        (_Harness(initialization_agent_draws=0), "agent draw count"),
        (_Harness(initialization_environment_draws=1), "environment key"),
        (_Harness(update_delta=2), "optimizer update count"),
        (_Harness(update_delta=0), "optimizer update count"),
        (_Harness(stale_bridge_state=True), "stale/reused state"),
        (_Harness(reverse_segment_payloads=True), "payload order"),
        (_Harness(missing_segment=True), "exactly four"),
        (_Harness(replace_model=True), "replaced the bound model"),
    ],
)
def test_off_by_one_double_consumption_stale_state_and_segment_order_fail_closed(
    harness: _Harness, message: str
) -> None:
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
        _run(harness)


@pytest.mark.unit
def test_failed_bridge_state_is_never_retried() -> None:
    harness = _Harness(bridge_failure=True)
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="without retry"):
        _run(harness)
    assert harness.bridge_step_calls == 1


def _validate_trace(
    trace: runner.PPOGRURolloutTrace,
    expected: runner.PPOGRURolloutTrace,
) -> None:
    runner.validate_ppo_gru_runner_rollout_trace(
        trace,
        _TINY_GEOMETRY,
        hidden_size=128,
        expected_rollout_index=0,
        expected_initial_carry=expected.initial_carry,
        expected_initial_observation=expected.steps[0].observation,
        expected_action_keys=tuple(step.action_key for step in expected.steps),
        expected_bootstrap_value=expected.bootstrap_value,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("carry", "incoming carry differs"),
        ("reset", "reset boundary"),
        ("bootstrap", "bootstrap value differs"),
        ("action", "action differs"),
        ("logprob", "old log probability differs"),
        ("key", "action key differs"),
        ("done", "continuing nonterminal"),
        ("observation_link", "observation differs"),
    ],
)
def test_rollout_trace_tampering_fails_before_optimizer_update(
    mutation: str, message: str
) -> None:
    harness = _Harness()
    _run(harness)
    expected = harness.rollout_traces[0]
    steps = list(expected.steps)
    if mutation == "carry":
        steps[1] = dataclasses.replace(
            steps[1], incoming_carry=steps[1].incoming_carry.at[0].add(1.0)
        )
    elif mutation == "reset":
        steps[1] = dataclasses.replace(steps[1], reset_before=True)
    elif mutation == "bootstrap":
        tampered = dataclasses.replace(
            expected, bootstrap_value=expected.bootstrap_value + jnp.float32(0.5)
        )
        with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
            _validate_trace(tampered, expected)
        return
    elif mutation == "action":
        steps[0] = dataclasses.replace(steps[0], action=(steps[0].action + 1) % 4)
    elif mutation == "logprob":
        steps[0] = dataclasses.replace(
            steps[0], old_log_prob=steps[0].old_log_prob + jnp.float32(0.25)
        )
    elif mutation == "key":
        steps[0] = dataclasses.replace(steps[0], action_key=steps[1].action_key)
    elif mutation == "done":
        steps[0] = dataclasses.replace(steps[0], transition_done=True)
    else:
        steps[1] = dataclasses.replace(steps[1], observation=_observation(99))
    tampered = dataclasses.replace(expected, steps=tuple(steps))
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
        _validate_trace(tampered, expected)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("logits", "replay logits"),
        ("terminal_outgoing", "replay outgoing carries"),
        ("bootstrap_carry", "final/bootstrap carry"),
    ],
)
def test_production_model_replay_binds_full_logits_and_terminal_carry(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    harness = _Harness()
    _run(harness)
    expected = harness.rollout_traces[0]
    replay = ppo_gru.PPOGRUSequenceEvaluation(
        final_carry=expected.bootstrap_carry,
        incoming_carries=jnp.stack([step.incoming_carry for step in expected.steps]),
        outgoing_carries=jnp.stack([step.outgoing_carry for step in expected.steps]),
        logits=jnp.stack([step.logits for step in expected.steps]),
        values=jnp.stack([step.old_value for step in expected.steps]),
    )
    monkeypatch.setattr(
        ppo_gru,
        "evaluate_ppo_gru_sequence",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        ppo_gru,
        "validate_ppo_gru_rollout",
        lambda *args, **kwargs: (
            jnp.zeros((len(expected.steps),), dtype=jnp.float32),
            jnp.zeros((len(expected.steps),), dtype=jnp.float32),
        ),
    )
    training = runner.PPOGRUTrainingHandle(
        model=ppo_gru.PPOGRUActorCritic(hidden_size=128, num_actions=4),
        state=ppo_gru.PPOGRUTrainState(
            variables={},
            optimizer_state=(),
            optimizer_updates=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    runner._production_validate_rollout(
        training,
        expected,
        ppo_gru.matched_v3_ppo_gru_configuration(),
    )

    steps = list(expected.steps)
    tampered = expected
    if mutation == "logits":
        steps[0] = dataclasses.replace(
            steps[0],
            logits=steps[0].logits.at[0].add(jnp.float32(0.5)),
        )
        tampered = dataclasses.replace(expected, steps=tuple(steps))
    elif mutation == "terminal_outgoing":
        changed = steps[-1].outgoing_carry.at[0].add(jnp.float32(1.0))
        steps[-1] = dataclasses.replace(steps[-1], outgoing_carry=changed)
        tampered = dataclasses.replace(
            expected,
            steps=tuple(steps),
            bootstrap_carry=changed,
        )
    else:
        tampered = dataclasses.replace(
            expected,
            bootstrap_carry=expected.bootstrap_carry.at[0].add(jnp.float32(1.0)),
        )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match=message):
        runner._production_validate_rollout(
            training,
            tampered,
            ppo_gru.matched_v3_ppo_gru_configuration(),
        )


@pytest.mark.unit
def test_engineering_receipt_is_canonical_strict_bound_and_non_authorizing() -> None:
    outcome = _run(_Harness())
    raw = runner.canonical_ppo_gru_engineering_receipt_bytes(outcome)
    parsed = runner.parse_ppo_gru_engineering_receipt(
        raw, expected_receipt_sha256=hashlib.sha256(raw).hexdigest()
    )
    assert raw == _canonical(parsed)
    assert parsed["raw_cumulative_score"] == 120
    assert parsed["raw_reward_trace"] == {
        "encoding": "signed_int8_twos_complement",
        "length": 16,
        "sha256": outcome.raw_reward_trace_sha256,
        "score_reduction": "exact_int64_sum",
        "score_scaling": "none",
    }
    assert parsed["production_horizon_complete"] is False
    assert parsed["accounting"]["ppo_environment_draw_count"] == 0
    assert parsed["claims"]["authority_granted"] is False
    assert parsed["seeds"]["provenance"] == "caller_supplied_unverified"
    assert parsed["seeds"]["upstream_receipt_bound"] is False
    assert parsed["seeds"]["protected_seed_status"] == "unverified"
    assert parsed["runtime_identity_sha256"] == hashlib.sha256(
        _RUNTIME_IDENTITY
    ).hexdigest()
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="partial horizon"):
        runner.canonical_ppo_gru_result_receipt_bytes(outcome)


@pytest.mark.unit
def test_exact_result_receipt_contract_is_strict_but_still_non_authorizing() -> None:
    engineering = _run(_Harness())
    geometry = runner.MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY
    runtime_identity_bytes = _production_runtime_identity_bytes()
    production_rewards = bytes(geometry.horizon)
    contract_only = dataclasses.replace(
        engineering,
        classification="production_runtime_unqualified_complete",
        geometry=geometry,
        runtime_identity_bytes=runtime_identity_bytes,
        runtime_identity_sha256=hashlib.sha256(runtime_identity_bytes).hexdigest(),
        environment_interactions=geometry.horizon,
        rollout_count=geometry.rollout_count,
        optimizer_update_count=geometry.optimizer_update_count,
        action_draw_count=geometry.action_draw_count,
        permutation_draw_count=geometry.permutation_draw_count,
        total_agent_draw_count=geometry.total_agent_draw_count,
        bridge_step_count=geometry.horizon,
        bridge_environment_key_use_count=1 + geometry.horizon,
        raw_reward_trace=production_rewards,
        raw_reward_trace_sha256=hashlib.sha256(production_rewards).hexdigest(),
        raw_cumulative_score=0,
        production_horizon_complete=True,
    )
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRURunnerError,
        match="completion capability",
    ):
        runner.canonical_ppo_gru_result_receipt_bytes(contract_only)
    unregistered = dataclasses.replace(
        contract_only,
        _production_capability=runner._ProductionOutcomeCapability(),
    )
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRURunnerError,
        match="exact registered completed-run object",
    ):
        runner.canonical_ppo_gru_result_receipt_bytes(unregistered)

    structural = json.loads(runner.canonical_ppo_gru_engineering_receipt_bytes(engineering))
    structural["schema_version"] = runner.PPO_GRU_RESULT_RECEIPT_SCHEMA_VERSION
    structural["classification"] = "production_runtime_unqualified_non_authorizing"
    structural["geometry"] = geometry.to_dict()
    structural["accounting"] = {
        "environment_interactions": geometry.horizon,
        "rollout_count": geometry.rollout_count,
        "optimizer_update_count": geometry.optimizer_update_count,
        "parameter_initialization_draw_count": 1,
        "action_draw_count": geometry.action_draw_count,
        "permutation_draw_count": geometry.permutation_draw_count,
        "total_agent_draw_count": geometry.total_agent_draw_count,
        "ppo_environment_draw_count": 0,
        "bridge_reset_count": 1,
        "bridge_step_count": geometry.horizon,
        "bridge_environment_key_use_count": 1 + geometry.horizon,
    }
    structural["raw_cumulative_score"] = 0
    structural["raw_reward_trace"] = {
        "encoding": "signed_int8_twos_complement",
        "length": geometry.horizon,
        "sha256": hashlib.sha256(production_rewards).hexdigest(),
        "score_reduction": "exact_int64_sum",
        "score_scaling": "none",
    }
    structural["production_horizon_complete"] = True
    structural["runtime_identity"] = json.loads(runtime_identity_bytes)
    structural["runtime_identity_sha256"] = hashlib.sha256(
        runtime_identity_bytes
    ).hexdigest()
    raw = _resign(structural)
    parsed = runner.parse_ppo_gru_result_receipt(raw)
    assert parsed["production_horizon_complete"] is True
    assert parsed["accounting"]["environment_interactions"] == 499_712
    assert parsed["accounting"]["optimizer_update_count"] == 15_616
    assert parsed["claims"]["execution_authorized"] is False
    assert any("do not independently attest execution" in item for item in parsed["limitations"])
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="schema"):
        runner.parse_ppo_gru_engineering_receipt(raw)

    invented_authority = json.loads(raw)
    invented_authority["runtime_identity"]["ready"] = True
    invented_runtime_raw = _canonical(invented_authority["runtime_identity"])
    invented_authority["runtime_identity_sha256"] = hashlib.sha256(
        invented_runtime_raw
    ).hexdigest()
    with pytest.raises(
        runner.ForagerMatchedV3PPOGRURunnerError,
        match="production field membership",
    ):
        runner.parse_ppo_gru_result_receipt(_resign(invented_authority))


@pytest.mark.unit
def test_receipt_duplicate_noncanonical_digest_mutation_and_authority_fail_closed() -> None:
    outcome = _run(_Harness())
    raw = runner.canonical_ppo_gru_engineering_receipt_bytes(outcome)
    duplicate = raw.replace(
        b'{"accounting":', b'{"schema_version":"duplicate","accounting":', 1
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="duplicate"):
        runner.parse_ppo_gru_engineering_receipt(duplicate)
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="canonical"):
        runner.parse_ppo_gru_engineering_receipt(b" " + raw)
    digest_mutation = json.loads(raw)
    digest_mutation["raw_cumulative_score"] += 1
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="body digest"):
        runner.parse_ppo_gru_engineering_receipt(_canonical(digest_mutation))
    authority = json.loads(raw)
    authority["claims"]["execution_authorized"] = True
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="claims drifted"):
        runner.parse_ppo_gru_engineering_receipt(_resign(authority))
    counter = json.loads(raw)
    counter["accounting"]["action_draw_count"] -= 1
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="accounting"):
        runner.parse_ppo_gru_engineering_receipt(_resign(counter))
    boolean_counter = json.loads(raw)
    boolean_counter["accounting"]["parameter_initialization_draw_count"] = True
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="accounting"):
        runner.parse_ppo_gru_engineering_receipt(_resign(boolean_counter))
    rollout_counter = json.loads(raw)
    rollout_counter["accounting"]["rollout_count"] += 1
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="accounting"):
        runner.parse_ppo_gru_engineering_receipt(_resign(rollout_counter))
    boolean_geometry = json.loads(raw)
    boolean_geometry["geometry"]["segments_per_rollout"] = True
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="exact integers"):
        runner.parse_ppo_gru_engineering_receipt(_resign(boolean_geometry))
    seed_provenance = json.loads(raw)
    seed_provenance["seeds"]["provenance"] = "verified"
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="seed provenance"):
        runner.parse_ppo_gru_engineering_receipt(_resign(seed_provenance))
    reward_trace = json.loads(raw)
    reward_trace["raw_reward_trace"]["length"] -= 1
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="trace metadata"):
        runner.parse_ppo_gru_engineering_receipt(_resign(reward_trace))
    score = json.loads(raw)
    score["raw_cumulative_score"] = 30 * _TINY_GEOMETRY.horizon + 1
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="support bounds"):
        runner.parse_ppo_gru_engineering_receipt(_resign(score))
    limitations = json.loads(raw)
    limitations["limitations"][0] = "changed"
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="limitations"):
        runner.parse_ppo_gru_engineering_receipt(_resign(limitations))
    runtime = json.loads(raw)
    runtime["runtime_identity"]["classification"] = "changed"
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="runtime identity"):
        runner.parse_ppo_gru_engineering_receipt(_resign(runtime))
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="artifact digest"):
        runner.parse_ppo_gru_engineering_receipt(
            raw, expected_receipt_sha256="0" * 64
        )

    original_claims = runner._non_authorizing_claims
    try:
        runner._non_authorizing_claims = lambda: {
            **original_claims(),
            "execution_authorized": True,
        }
        with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="cannot grant"):
            runner.canonical_ppo_gru_engineering_receipt_bytes(outcome)
    finally:
        runner._non_authorizing_claims = original_claims


@pytest.mark.unit
def test_partial_or_forged_complete_outcomes_are_rejected() -> None:
    outcome = _run(_Harness())
    forged_complete = dataclasses.replace(outcome, production_horizon_complete=True)
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="mislabeled"):
        runner.canonical_ppo_gru_engineering_receipt_bytes(forged_complete)
    forged_geometry = dataclasses.replace(
        outcome,
        classification="production_runtime_unqualified_complete",
        geometry=runner.MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY,
        production_horizon_complete=True,
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="accounting"):
        runner.canonical_ppo_gru_result_receipt_bytes(forged_geometry)


@pytest.mark.unit
def test_runtime_identity_authority_and_drift_fail_closed() -> None:
    authority = json.loads(_RUNTIME_IDENTITY)
    authority["claims"]["execution_authorized"] = True
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="cannot grant"):
        runner.run_ppo_gru_engineering_driver(
            environment_seed=17,
            agent_seed=29,
            runtime_identity_bytes=_canonical(authority),
            geometry=_TINY_GEOMETRY,
            dependencies=_Harness().dependencies(),
        )

    harness = _Harness()

    def drifting_parser(raw: bytes) -> dict[str, Any]:
        parsed = cast(dict[str, Any], json.loads(raw))
        if harness.runtime_parse_calls:
            parsed["classification"] = "drifted"
        harness.runtime_parse_calls += 1
        return parsed

    dependencies = dataclasses.replace(
        harness.dependencies(), parse_runtime_identity=drifting_parser
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="changed"):
        runner.run_ppo_gru_engineering_driver(
            environment_seed=17,
            agent_seed=29,
            runtime_identity_bytes=_RUNTIME_IDENTITY,
            geometry=_TINY_GEOMETRY,
            dependencies=dependencies,
        )


@pytest.mark.unit
def test_production_runtime_adapter_reuses_one_opened_bridge_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = bridge.MatchedV3ForagaxRuntimeIdentity(
        jax_version="0.11.0",
        jaxlib_version="0.11.0",
        default_prng_impl="threefry2x32",
        threefry_partitionable=True,
        jax_enable_x64=False,
        backend="cpu",
        foragax_version="0.55.0",
        foragax_install_tree_sha256=bridge.FORAGAX_INSTALL_TREE_SHA256,
        foragax_package_root="/synthetic/foragax",
        runtime_qualified=False,
    )
    fake_runtime = bridge.MatchedV3ForagaxRuntime(
        runtime_identity=identity,
        _environment=object(),
        _params=object(),
        _capability=bridge._RuntimeCapability(),
    )
    calls = 0

    def open_once() -> bridge.MatchedV3ForagaxRuntime:
        nonlocal calls
        calls += 1
        return fake_runtime

    monkeypatch.setattr(bridge, "open_matched_v3_foragax_runtime", open_once)
    opened = runner.open_matched_v3_ppo_gru_runner_runtime()
    assert calls == 1
    assert opened.bridge_runtime is fake_runtime
    assert getattr(opened.dependencies.initialize_bridge, "__self__", None) is fake_runtime
    runtime_identity = json.loads(opened.runtime_identity_bytes)
    assert runtime_identity["schema_version"] == (
        runner.PPO_GRU_RUNTIME_IDENTITY_SCHEMA_VERSION
    )
    assert runtime_identity["runtime"]["runtime_qualified"] is False
    assert runtime_identity["bridge_descriptor_sha256"] == (
        bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
    )
    assert runner.production_ppo_gru_runner_dependencies(opened) is opened.dependencies

    replaced = dataclasses.replace(opened)
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="exact registered"):
        runner.production_ppo_gru_runner_dependencies(replaced)

    injected = dataclasses.replace(opened, dependencies=_Harness().dependencies())
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="exact registered"):
        runner.run_matched_v3_ppo_gru_production(
            environment_seed=17,
            agent_seed=29,
            runtime=injected,
        )

    constructed = runner.PPOGRUProductionRuntime(
        bridge_runtime=opened.bridge_runtime,
        runtime_identity_bytes=opened.runtime_identity_bytes,
        dependencies=opened.dependencies,
        _capability=runner._ProductionRuntimeCapability(),
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="exact registered"):
        runner.production_ppo_gru_runner_dependencies(constructed)

    original_evaluate = opened.dependencies.evaluate_step
    try:
        object.__setattr__(opened.dependencies, "evaluate_step", lambda *args: None)
        with pytest.raises(
            runner.ForagerMatchedV3PPOGRURunnerError,
            match="callable evaluate_step",
        ):
            runner.production_ppo_gru_runner_dependencies(opened)
    finally:
        object.__setattr__(opened.dependencies, "evaluate_step", original_evaluate)
    assert runner.production_ppo_gru_runner_dependencies(opened) is opened.dependencies


@pytest.mark.unit
def test_engineering_surface_cannot_impersonate_production_or_use_production_deps() -> None:
    harness = _Harness()
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="impersonate"):
        runner.run_ppo_gru_engineering_driver(
            environment_seed=17,
            agent_seed=29,
            runtime_identity_bytes=_RUNTIME_IDENTITY,
            geometry=runner.MATCHED_V3_PPO_GRU_PRODUCTION_GEOMETRY,
            dependencies=harness.dependencies(),
        )
    dependencies = dataclasses.replace(
        harness.dependencies(), classification="production_adapter_runtime_unqualified"
    )
    with pytest.raises(runner.ForagerMatchedV3PPOGRURunnerError, match="synthetic"):
        runner.run_ppo_gru_engineering_driver(
            environment_seed=17,
            agent_seed=29,
            runtime_identity_bytes=_RUNTIME_IDENTITY,
            geometry=_TINY_GEOMETRY,
            dependencies=dependencies,
        )


@pytest.mark.unit
def test_source_has_no_filesystem_write_console_or_protected_seed_literal() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "write_text(" not in source
    assert "write_bytes(" not in source
    assert "open(" not in source
    assert "print(" not in source
    assert "console." not in source
    assert "outputs/" not in source
