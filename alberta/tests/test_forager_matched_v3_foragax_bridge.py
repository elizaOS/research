"""Fail-closed synthetic tests for the matched-v3 Foragax bridge."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib
import importlib
import json
import threading
from collections.abc import Callable
from importlib import metadata as importlib_metadata
from types import SimpleNamespace
from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from alberta_framework.benchmarks import forager, forager_rng_parity
from alberta_framework.benchmarks import forager_matched_v3_foragax_bridge as bridge
from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru

pytestmark = pytest.mark.unit


@dataclasses.dataclass(frozen=True)
class _FakeParams:
    max_steps_in_episode: int | None = None


@dataclasses.dataclass(frozen=True)
class _FakeSpace:
    shape: tuple[int, ...] | None = None
    n: int | None = None


def _observation(*, channel: int = 0) -> jnp.ndarray:
    value = np.zeros((9, 9, 3), dtype=np.float32)
    value[0, 0, channel] = 1.0
    return jnp.asarray(value)


def _environment_state(time: int) -> dict[str, Any]:
    return {"time": jnp.asarray(time, dtype=jnp.int32)}


def _info() -> dict[str, Any]:
    return {
        "discount": jnp.asarray(1.0, dtype=jnp.float32),
        "temperatures": jnp.zeros((4,), dtype=jnp.float32),
        "biome_id": jnp.asarray(-1, dtype=jnp.int16),
        "object_collected_id": jnp.asarray(-1, dtype=jnp.int32),
        "current_biome_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "max_biome_mean": jnp.asarray(1.0, dtype=jnp.float32),
        "biome_regret": jnp.asarray(1.0, dtype=jnp.float32),
        "biome_rank": jnp.asarray(2, dtype=jnp.int32),
        "rewards": jnp.zeros((9, 9), dtype=jnp.float16),
    }


class _FakeForagax:
    name = "ForagaxTwoBiomeLarge-v1"
    observation_type = "color"
    aperture_size = (9, 9)
    num_actions = 4

    def __init__(self) -> None:
        self.default_params = _FakeParams()
        self.reset_keys: list[tuple[int, int]] = []
        self.step_keys: list[tuple[int, int]] = []
        self.actions: list[tuple[np.dtype[Any], int]] = []
        self.reset_result: object | None = None
        self.step_result: object | None = None
        self.reward = 0

    def action_space(self, params: object) -> _FakeSpace:
        assert params is self.default_params
        return _FakeSpace(n=4)

    def observation_space(self, params: object) -> _FakeSpace:
        assert params is self.default_params
        return _FakeSpace(shape=(9, 9, 3))

    def reset(self, key: Any, params: object) -> object:
        assert params is self.default_params
        self.reset_keys.append(
            cast(tuple[int, int], tuple(int(item) for item in np.asarray(jr.key_data(key))))
        )
        if self.reset_result is not None:
            return self.reset_result
        return _observation(), _environment_state(0)

    def step(self, key: Any, state: object, action: Any, params: object) -> object:
        assert params is self.default_params
        self.step_keys.append(
            cast(tuple[int, int], tuple(int(item) for item in np.asarray(jr.key_data(key))))
        )
        action_array = np.asarray(action)
        self.actions.append((action_array.dtype, int(action_array)))
        if isinstance(self.step_result, BaseException):
            raise self.step_result
        if self.step_result is not None:
            return self.step_result
        time = int(np.asarray(state["time"])) + 1  # type: ignore[index]
        return (
            _observation(channel=time % 3),
            _environment_state(time),
            jnp.asarray(self.reward, dtype=jnp.float32),
            jnp.asarray(False, dtype=jnp.bool_),
            _info(),
        )


class _BlockingForagax(_FakeForagax):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def step(self, key: Any, state: object, action: Any, params: object) -> object:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("synthetic blocking step timed out")
        return super().step(key, state, action, params)


def _runtime_identity(
    **changes: object,
) -> bridge.MatchedV3ForagaxRuntimeIdentity:
    values: dict[str, object] = {
        "jax_version": "0.11.0",
        "jaxlib_version": "0.11.0",
        "default_prng_impl": "threefry2x32",
        "threefry_partitionable": True,
        "jax_enable_x64": False,
        "backend": "cpu",
        "foragax_version": "0.55.0",
        "foragax_install_tree_sha256": (
            "3d79040c87a0d91d4b084da0f661b08e5c23be3769914655afd3017f693a6eca"
        ),
        "foragax_package_root": "/synthetic/site-packages/foragax",
        "runtime_qualified": False,
    }
    values.update(changes)
    return bridge.MatchedV3ForagaxRuntimeIdentity(**values)  # type: ignore[arg-type]


def _factory_for(
    environment: _FakeForagax,
) -> tuple[Callable[..., _FakeForagax], list[tuple[str, dict[str, object]]]]:
    calls: list[tuple[str, dict[str, object]]] = []

    def make(environment_id: str, **kwargs: object) -> _FakeForagax:
        calls.append((environment_id, dict(kwargs)))
        return environment

    return make, calls


def _open_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    environment: _FakeForagax | None = None,
) -> tuple[
    bridge.MatchedV3ForagaxRuntime,
    _FakeForagax,
    list[tuple[str, dict[str, object]]],
]:
    fake = _FakeForagax() if environment is None else environment
    make, calls = _factory_for(fake)
    monkeypatch.setattr(bridge, "_validated_runtime_identity", _runtime_identity)
    monkeypatch.setattr(bridge, "_load_registry_make", lambda identity: make)
    runtime = bridge.open_matched_v3_foragax_runtime()
    return runtime, fake, calls


def _open_fake(
    monkeypatch: pytest.MonkeyPatch,
    environment: _FakeForagax | None = None,
    *,
    seed: object = 17,
) -> tuple[
    bridge.MatchedV3ForagaxBridgeState,
    _FakeForagax,
    list[tuple[str, dict[str, object]]],
]:
    runtime, fake, calls = _open_fake_runtime(monkeypatch, environment)
    state = bridge.initialize_matched_v3_foragax_bridge(seed, runtime=runtime)
    return state, fake, calls


def test_descriptor_is_literal_frozen_detached_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_runtime_open() -> object:
        raise AssertionError("descriptor access attempted to inspect the runtime")

    monkeypatch.setattr(bridge, "_validated_runtime_identity", forbidden_runtime_open)
    raw = bridge.canonical_matched_v3_foragax_bridge_descriptor_bytes()
    descriptor = bridge.matched_v3_foragax_bridge_descriptor()

    assert json.loads(raw) == descriptor
    assert hashlib.sha256(raw).hexdigest() == bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256
    assert bridge.FORAGAX_BRIDGE_DESCRIPTOR_SHA256 == (
        "1bf4f43bdf759a650e2f2662f8d5c86eb35d12eeb3a8399a3b5566b7bf8e45ab"
    )
    assert bridge.parse_matched_v3_foragax_bridge_descriptor(raw) == descriptor
    assert descriptor["source"]["source_review_complete"] is False
    assert descriptor["source"]["source_closure_bound"] is False
    assert descriptor["runtime"] == {
        "backend_observed_at_open": True,
        "backend_qualified": False,
        "convenience_api_opens_one_runtime": True,
        "default_prng_impl": "threefry2x32",
        "environment_capabilities_checked_once_at_open": True,
        "foragax_install_tree_checked_at_open": True,
        "jax_enable_x64": False,
        "jax_required_version": "0.11.0",
        "jax_threefry_partitionable": True,
        "jaxlib_required_version": "0.11.0",
        "per_step_host_api_jitted": False,
        "real_foragax_api_inspected": True,
        "reusable_runtime_handle": True,
        "runtime_parity_executed": False,
        "runtime_qualified": False,
    }
    assert descriptor["rng"]["identity"] == "dedicated_environment_split_chain_v1"
    assert descriptor["rng"]["root"] == (
        "jax.random.key(environment_seed,impl=threefry2x32)"
    )
    assert descriptor["rng"]["public_contract_root"] == "jax.random.key(seed)"
    assert descriptor["claims"] == {
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
    assert any("compiled chunk kernel" in item for item in descriptor["limitations"])
    descriptor["claims"]["execution_ready"] = True
    assert bridge.matched_v3_foragax_bridge_descriptor()["claims"][
        "execution_ready"
    ] is False


def test_descriptor_parser_rejects_mutation_duplicate_noncanonical_and_aliases() -> None:
    raw = bridge.canonical_matched_v3_foragax_bridge_descriptor_bytes()
    descriptor = bridge.matched_v3_foragax_bridge_descriptor()

    mutation = json.loads(raw)
    mutation["runtime"]["runtime_qualified"] = True
    changed = json.dumps(mutation, sort_keys=True, separators=(",", ":")).encode()
    for invalid in (
        changed,
        b" " + raw,
        raw[:-1] + b',"status":"implemented_unqualified"}',
    ):
        with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError):
            bridge.parse_matched_v3_foragax_bridge_descriptor(invalid)

    shared: list[object] = []
    descriptor["alias_a"] = shared
    descriptor["alias_b"] = shared
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="alias"):
        bridge.parse_matched_v3_foragax_bridge_descriptor(descriptor)


def test_runtime_is_validated_once_reused_and_constructs_the_exact_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, environment, calls = _open_fake_runtime(monkeypatch)
    first = runtime.initialize(17)
    second = bridge.initialize_matched_v3_foragax_bridge(19, runtime=runtime)

    assert calls == [
        (
            "ForagaxTwoBiomeLarge-v1",
            {
                "aperture_size": 9,
                "observation_type": "color",
                "random_shift_max_steps": 0,
                "reward_delay": 0,
            },
        )
    ]
    assert runtime.runtime_identity == _runtime_identity()
    assert first.environment_seed == 17
    assert second.environment_seed == 19
    assert first.reset_count == second.reset_count == 1
    assert first.step_count == second.step_count == 0
    assert len(environment.reset_keys) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("jax_version", "0.11.1"),
        ("jaxlib_version", "0.10.0"),
        ("default_prng_impl", "rbg"),
        ("threefry_partitionable", False),
        ("jax_enable_x64", True),
        ("foragax_version", "0.54.0"),
        ("foragax_install_tree_sha256", "0" * 64),
        ("runtime_qualified", True),
    ],
)
def test_runtime_identity_drift_fails_before_registry_make(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    observed = dataclasses.asdict(_runtime_identity())
    observed[field] = value
    monkeypatch.setattr(bridge, "_observe_runtime_identity", lambda: observed)
    monkeypatch.setattr(
        bridge,
        "_load_registry_make",
        lambda identity: (_ for _ in ()).throw(AssertionError("make reached")),
    )
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="runtime"):
        bridge.open_matched_v3_foragax_runtime()


def test_direct_split_schedule_matches_cross_module_contracts_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, environment, _ = _open_fake_runtime(monkeypatch)
    state = runtime.initialize(17)
    for action in (0, 3, 1):
        state = bridge.step_matched_v3_foragax_bridge(state, action).state

    config = forager_rng_parity.FixedActionProbeConfig(seed=17, actions=(0, 3, 1))
    reset_frame, transition_frames = forager_rng_parity.expected_key_schedule(config)
    assert environment.reset_keys == [reset_frame.environment_key]
    assert environment.step_keys == [frame.environment_key for frame in transition_frames]

    ppo_state = ppo_gru.initialize_ppo_gru_rng_state(17, 991)
    ppo_keys: list[tuple[int, int]] = []
    for _ in range(4):
        ppo_state, key = ppo_gru.next_ppo_gru_environment_key(ppo_state)
        ppo_keys.append(
            cast(tuple[int, int], tuple(int(item) for item in np.asarray(jr.key_data(key))))
        )
    assert ppo_keys == environment.reset_keys + environment.step_keys
    assert forager.forager_rng_contract()["identity"] == (
        bridge.MATCHED_V3_ENVIRONMENT_RNG_SCHEDULE
    )


def test_agent_draws_cannot_change_the_environment_key_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_runtime, first_environment, _ = _open_fake_runtime(monkeypatch)
    first = first_runtime.initialize(29)
    first = bridge.step_matched_v3_foragax_bridge(first, 1).state
    first = bridge.step_matched_v3_foragax_bridge(first, 2).state

    agent_key = jr.key(991, impl="threefry2x32")
    for _ in range(100):
        agent_key, _ = jr.split(agent_key)

    second_environment = _FakeForagax()
    second_make, _ = _factory_for(second_environment)
    monkeypatch.setattr(bridge, "_load_registry_make", lambda identity: second_make)
    second_runtime = bridge.open_matched_v3_foragax_runtime()
    second = second_runtime.initialize(29)
    second = bridge.step_matched_v3_foragax_bridge(second, 1).state
    second = bridge.step_matched_v3_foragax_bridge(second, 2).state

    assert first_environment.reset_keys == second_environment.reset_keys
    assert first_environment.step_keys == second_environment.step_keys
    assert np.array_equal(np.asarray(first.observation), np.asarray(second.observation))
    assert "agent" not in bridge.initialize_matched_v3_foragax_bridge.__annotations__


@pytest.mark.parametrize("seed", [True, -1, 2**31, 1.0, np.int32(7)])
def test_environment_seed_rejects_type_and_range_aliases(
    monkeypatch: pytest.MonkeyPatch, seed: object
) -> None:
    runtime, environment, _ = _open_fake_runtime(monkeypatch)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="uint31"):
        runtime.initialize(seed)
    assert not environment.reset_keys


@pytest.mark.parametrize(
    "action",
    [True, -1, 4, 1.0, np.asarray([1], dtype=np.int32), np.asarray(1, dtype=np.uint32)],
)
def test_action_validation_fails_before_state_consumption(
    monkeypatch: pytest.MonkeyPatch, action: object
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="action"):
        bridge.step_matched_v3_foragax_bridge(state, action)
    assert not environment.step_keys
    bridge.step_matched_v3_foragax_bridge(state, 0)


@pytest.mark.parametrize("kind", ["shape", "dtype", "fraction", "multi_hot", "nan"])
def test_reset_observation_validation_is_exact_and_zero_hot_is_allowed(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    environment = _FakeForagax()
    value = np.zeros((9, 9, 3), dtype=np.float32)
    if kind == "shape":
        value = np.zeros((9, 9, 4), dtype=np.float32)
    elif kind == "dtype":
        value = np.zeros((9, 9, 3), dtype=np.float64)
    elif kind == "fraction":
        value[0, 0, 0] = 0.5
    elif kind == "multi_hot":
        value[0, 0, :2] = 1.0
    else:
        value[0, 0, 0] = np.nan
    environment.reset_result = (value, _environment_state(0))
    runtime, _, _ = _open_fake_runtime(monkeypatch, environment)

    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="observation"):
        runtime.initialize(17)


@pytest.mark.parametrize("reward", [-2, 2, 30.5, np.float64(1), np.int32(1), np.nan])
def test_raw_reward_validation_is_exact_and_failure_poisons_trajectory(
    monkeypatch: pytest.MonkeyPatch, reward: object
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    environment.step_result = (
        _observation(),
        _environment_state(1),
        reward,
        jnp.asarray(False, dtype=jnp.bool_),
        _info(),
    )
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="reward"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="poison"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    assert len(environment.step_keys) == 1


@pytest.mark.parametrize("done", [True, 0, 1, np.int32(0), np.asarray([False])])
def test_done_must_be_an_exact_false_scalar_and_never_auto_resets(
    monkeypatch: pytest.MonkeyPatch, done: object
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    environment.step_result = (
        _observation(),
        _environment_state(1),
        jnp.asarray(0, dtype=jnp.float32),
        done,
        _info(),
    )
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="done"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    assert len(environment.reset_keys) == 1


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "discount", "temperatures", "regret", "rank", "rewards"],
)
def test_info_contract_is_exact_and_never_exposed_to_the_adapter(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    info = _info()
    if mutation == "missing":
        info.pop("discount")
    elif mutation == "extra":
        info["terminated"] = False
    elif mutation == "discount":
        info["discount"] = jnp.asarray(0.0, dtype=jnp.float32)
    elif mutation == "temperatures":
        info["temperatures"] = jnp.ones((4,), dtype=jnp.float32)
    elif mutation == "regret":
        info["biome_regret"] = jnp.asarray(-1.0, dtype=jnp.float32)
    elif mutation == "rank":
        info["biome_rank"] = jnp.asarray(4, dtype=jnp.int32)
    else:
        info["rewards"] = jnp.full((9, 9), 2.0, dtype=jnp.float16)
    environment.step_result = (
        _observation(),
        _environment_state(1),
        jnp.asarray(0, dtype=jnp.float32),
        jnp.asarray(False, dtype=jnp.bool_),
        info,
    )

    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="info"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    assert len(environment.step_keys) == 1


def test_successful_transition_has_exact_accounting_and_hides_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    transition = bridge.step_matched_v3_foragax_bridge(state, np.int32(3))

    assert transition.state.reset_count == 1
    assert transition.state.step_count == 1
    assert transition.state.environment_key_use_count == 2
    assert transition.action == 3
    assert transition.reward == 0
    assert transition.done is False
    assert transition.truncated is False
    assert transition.info_validated is True
    assert not hasattr(transition, "info")
    assert environment.actions == [(np.dtype(np.int32), 3)]


def test_reset_step_arity_and_environment_time_progression_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeForagax()
    environment.reset_result = (_observation(),)
    runtime, _, _ = _open_fake_runtime(monkeypatch, environment)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="reset"):
        runtime.initialize(17)

    environment.reset_result = None
    state = runtime.initialize(17)
    environment.step_result = (_observation(), _environment_state(1))
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="step"):
        bridge.step_matched_v3_foragax_bridge(state, 0)

    state = runtime.initialize(19)
    environment.step_result = (
        _observation(),
        _environment_state(2),
        jnp.asarray(0, dtype=jnp.float32),
        jnp.asarray(False, dtype=jnp.bool_),
        _info(),
    )
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="time"):
        bridge.step_matched_v3_foragax_bridge(state, 0)


def test_stale_original_and_exact_fork_cannot_both_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    fork = dataclasses.replace(state)
    advanced = bridge.step_matched_v3_foragax_bridge(fork, 0).state

    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="stale"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="stale"):
        bridge.step_matched_v3_foragax_bridge(fork, 0)
    bridge.step_matched_v3_foragax_bridge(advanced, 0)
    assert len(environment.step_keys) == 2


@pytest.mark.parametrize("field", ["key", "environment_state", "observation", "count"])
def test_registry_rejects_valid_looking_state_substitutions_before_environment_use(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    if field == "key":
        forged = dataclasses.replace(
            state, _environment_key=jr.key(99, impl="threefry2x32")
        )
    elif field == "environment_state":
        forged = dataclasses.replace(state, _environment_state=_environment_state(0))
    elif field == "observation":
        forged = dataclasses.replace(state, observation=state.observation + 0.0)
    else:
        forged = dataclasses.replace(state, step_count=1)

    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="registry"):
        bridge.step_matched_v3_foragax_bridge(forged, 0)
    assert not environment.step_keys
    bridge.step_matched_v3_foragax_bridge(state, 0)


def test_environment_exception_consumes_state_and_poisons_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    environment.step_result = RuntimeError("synthetic environment failure")

    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="step failed"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="poison"):
        bridge.step_matched_v3_foragax_bridge(state, 0)
    assert len(environment.step_keys) == 1


def test_atomic_state_consumption_rejects_concurrent_double_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _BlockingForagax()
    state, _, _ = _open_fake(monkeypatch, environment)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(bridge.step_matched_v3_foragax_bridge, state, 0)
        assert environment.entered.wait(timeout=5)
        with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="stale"):
            bridge.step_matched_v3_foragax_bridge(state, 0)
        environment.release.set()
        transition = future.result(timeout=5)

    assert transition.state.step_count == 1
    assert len(environment.step_keys) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "ForagaxTwoBiomeLarge-v2"),
        ("observation_type", "rgb"),
        ("aperture_size", (7, 7)),
        ("num_actions", 5),
    ],
)
def test_environment_capability_drift_fails_before_reset(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    environment = _FakeForagax()
    setattr(environment, field, value)
    make, _ = _factory_for(environment)
    monkeypatch.setattr(bridge, "_validated_runtime_identity", _runtime_identity)
    monkeypatch.setattr(bridge, "_load_registry_make", lambda identity: make)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="capability"):
        bridge.open_matched_v3_foragax_runtime()
    assert not environment.reset_keys


def test_default_params_and_spaces_must_describe_one_continuing_exact_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _FakeForagax()
    environment.default_params = _FakeParams(max_steps_in_episode=10)
    make, _ = _factory_for(environment)
    monkeypatch.setattr(bridge, "_validated_runtime_identity", _runtime_identity)
    monkeypatch.setattr(bridge, "_load_registry_make", lambda identity: make)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="continuing"):
        bridge.open_matched_v3_foragax_runtime()

    environment = _FakeForagax()
    environment.observation_space = lambda params: _FakeSpace(shape=(9, 9, 4))  # type: ignore[method-assign]
    make, _ = _factory_for(environment)
    monkeypatch.setattr(bridge, "_load_registry_make", lambda identity: make)
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="capability"):
        bridge.open_matched_v3_foragax_runtime()


def test_lazy_registry_loader_requires_exact_distribution_origin_and_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _runtime_identity()
    monkeypatch.setattr(importlib_metadata, "version", lambda name: "0.54.0")
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="0.55.0"):
        bridge._load_registry_make(identity)

    monkeypatch.setattr(importlib_metadata, "version", lambda name: "0.55.0")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(
            __name__="foragax.registry",
            __file__="/synthetic/site-packages/foragax/registry.py",
            make=None,
        ),
    )
    with pytest.raises(bridge.ForagerMatchedV3ForagaxBridgeError, match="callable"):
        bridge._load_registry_make(identity)


def test_valid_transition_preserves_each_exact_raw_reward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, environment, _ = _open_fake(monkeypatch)
    for reward in (-1, 0, 1, 30):
        environment.reward = reward
        transition = bridge.step_matched_v3_foragax_bridge(state, 2)
        assert transition.reward == reward
        assert type(transition.reward) is int
        state = transition.state
    assert state.step_count == 4
    assert state.reset_count == 1
    assert state.environment_key_use_count == 5
