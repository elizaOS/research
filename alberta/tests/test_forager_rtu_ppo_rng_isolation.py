from __future__ import annotations

import copy
import hashlib
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_rtu_ppo_rng_isolation as isolation


def _upstream_fixture() -> bytes:
    return b"""\
from typing import Any
import jax

PERIOD = 182500

class _Struct:
    @staticmethod
    def field(*, pytree_node):
        return None

struct = _Struct()

class GymnaxEnvState:
    to_render: bool = struct.field(pytree_node=True)
    cond_render: Any = struct.field(pytree_node=False)
    env_step: Any = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    env_state: Any = struct.field(pytree_node=True)

    @classmethod
    def create(cls, *, env_step, env_params, env_state, **kwargs):
        return cls()

def agent_step(last_obs, train_state, rng, hstate):
    return 0, 0, 0, 0

def env_step(runner_state, _):
    train_state, gymnax_state, rng, hstate = runner_state
    rng, _rng = jax.random.split(rng)
    action, log_prob, value, last_hidden = agent_step(
        None, train_state, _rng, hstate
    )
    # STEP ENV
    obs, env_state, reward, done, info = gymnax_state.env_step(
        _rng, gymnax_state.env_state, action.squeeze(), gymnax_state.env_params
    )
    gymnax_state = GymnaxEnvState.create(
        to_render=gymnax_state.to_render,
        cond_render=gymnax_state.cond_render,
        env_step=gymnax_state.env_step,
        env_params=gymnax_state.env_params,
        env_state=env_state,
    )
    runner_state = (
        train_state,
        gymnax_state,
        rng,
        last_hidden,
    )
    return runner_state, (obs, reward, done, info)

def experiment(rng, config):
    env = config.env
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env.default_params)
    gymnax_state = GymnaxEnvState.create(
        to_render=False,
        cond_render=config.render,
        env_step=env.step,
        env_params=env.default_params,
        env_state=env_state,
    )
    action_dim = 4

    def experiment_step(carry, iteration_idx):
        gymnax_state = carry
        gymnax_state = GymnaxEnvState.create(
            to_render=False,
            cond_render=gymnax_state.cond_render,
            env_step=gymnax_state.env_step,
            env_params=gymnax_state.env_params,
            env_state=gymnax_state.env_state,
        )

        env_step_state = (
            gymnax_state,
            iteration_idx,
        )
        return env_step_state

    return obs, gymnax_state, action_dim, experiment_step
"""


def _allow_fixture(
    monkeypatch: pytest.MonkeyPatch,
    source: bytes,
) -> None:
    monkeypatch.setattr(
        isolation,
        "UPSTREAM_RTU_PPO_SOURCE_SHA256",
        hashlib.sha256(source).hexdigest(),
    )
    derived = source
    for replacement in isolation._REPLACEMENTS:
        derived = derived.replace(replacement.before, replacement.after, 1)
    monkeypatch.setattr(
        isolation,
        "EXPECTED_PATCHED_RTU_PPO_SHA256",
        hashlib.sha256(derived).hexdigest(),
    )


def test_frozen_upstream_and_derived_hashes() -> None:
    assert isolation.UPSTREAM_RTU_PPO_SOURCE_SHA256 == (
        "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
    )
    assert isolation.EXPECTED_PATCHED_RTU_PPO_SHA256 == (
        "c47f3e087cb01722e824efc1d62c2e5880e75a2d937ae8fc122af24ce8967f2d"
    )
    assert isolation.UPSTREAM_SOURCE_COMMIT == (
        "9710f60fa30da5badc451ad7ce3ff296d5070830"
    )


def test_exact_derivation_separates_agent_and_environment_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)

    first = isolation.derive_isolated_rtu_ppo_source(source)
    second = isolation.derive_isolated_rtu_ppo_source(source)

    assert first.source == second.source
    assert first.source_sha256 == hashlib.sha256(first.source).hexdigest()
    assert first.descriptor == second.descriptor
    assert first.source.count(b"environment_rng: Any") == 1
    assert first.source.count(b"environment_rng=") == 3
    assert first.source.count(b"environment_step_rng") == 2
    assert first.source.count(b"ISOLATED_AGENT_RNG_NAMESPACE") == 2
    assert b"gymnax_state.env_step(\n        _rng," not in first.source
    assert first.descriptor["environment_rng"]["schedule"] == (
        "dedicated_environment_split_chain_v1"
    )
    assert first.descriptor["environment_rng"][
        "action_key_shared_with_environment"
    ] is False
    assert first.descriptor["agent_rng"]["namespace"] == 0xA63E7C11
    assert len(first.descriptor["replacement_records"]) == 7
    assert len(first.descriptor["payload_sha256"]) == 64


def test_descriptor_is_recursively_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    descriptor = isolation.derive_isolated_rtu_ppo_source(source).descriptor

    with pytest.raises(TypeError):
        descriptor["agent_rng"]["namespace"] = 0
    with pytest.raises(AttributeError):
        descriptor["replacement_records"].append({})
    with pytest.raises(TypeError):
        descriptor["replacement_records"][0]["replacement_id"] = "tampered"


def test_wrong_source_identity_and_nonbytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    with pytest.raises(TypeError, match="source must be bytes"):
        isolation.derive_isolated_rtu_ppo_source("source")  # type: ignore[arg-type]
    with pytest.raises(isolation.RTUPPORngIsolationError, match="SHA-256 differs"):
        isolation.derive_isolated_rtu_ppo_source(source)

    changed = source.replace(b"PERIOD = 182500", b"PERIOD = 182501")
    _allow_fixture(monkeypatch, changed)
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="declare_isolated_agent_namespace.*matched 0",
    ):
        isolation.derive_isolated_rtu_ppo_source(changed)


def test_duplicate_or_missing_replacement_anchor_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    duplicated = source.replace(
        b"PERIOD = 182500\n",
        b"PERIOD = 182500\nPERIOD = 182500\n",
    )
    _allow_fixture(monkeypatch, duplicated)
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="declare_isolated_agent_namespace.*matched 2",
    ):
        isolation.derive_isolated_rtu_ppo_source(duplicated)

    missing = source.replace(
        b"    env_state: Any = struct.field(pytree_node=True)\n",
        b"    env_state: object = struct.field(pytree_node=True)\n",
    )
    _allow_fixture(monkeypatch, missing)
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="carry_environment_rng.*matched 0",
    ):
        isolation.derive_isolated_rtu_ppo_source(missing)


def test_ast_validation_rejects_recombined_action_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    derived = isolation.derive_isolated_rtu_ppo_source(source).source
    tampered = derived.replace(
        b"        environment_step_rng,\n",
        b"        _rng,\n",
        1,
    )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="does not consume environment_step_rng",
    ):
        isolation._validate_derived_ast(tampered)


def test_ast_validation_rejects_missing_environment_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    derived = isolation.derive_isolated_rtu_ppo_source(source).source
    tampered = derived.replace(
        b"        environment_rng=gymnax_state.environment_rng,\n",
        b"",
        1,
    )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="every GymnaxEnvState.create",
    ):
        isolation._validate_derived_ast(tampered)


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            b"    environment_rng, environment_step_rng = jax.random.split(\n",
            b"    discarded_rng, environment_step_rng = jax.random.split(\n",
            "split and retain",
        ),
        (
            b"    environment_rng, reset_rng = jax.random.split(environment_rng)\n",
            b"    discarded_rng, reset_rng = jax.random.split(environment_rng)\n",
            "root one environment chain",
        ),
        (
            b"    rng = jax.random.fold_in(rng, ISOLATED_AGENT_RNG_NAMESPACE)\n",
            b"    discarded_rng = jax.random.fold_in(rng, ISOLATED_AGENT_RNG_NAMESPACE)\n",
            "root one environment chain",
        ),
        (
            b"        environment_rng=environment_rng,\n",
            b"        environment_rng=_rng,\n",
            "post-split environment RNG",
        ),
    ],
)
def test_ast_validation_rejects_discarded_or_wrong_rng_carries(
    monkeypatch: pytest.MonkeyPatch,
    before: bytes,
    after: bytes,
    message: str,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    derived = isolation.derive_isolated_rtu_ppo_source(source).source
    tampered = derived.replace(before, after, 1)
    assert tampered != derived
    with pytest.raises(isolation.RTUPPORngIsolationError, match=message):
        isolation._validate_derived_ast(tampered)


def test_validation_rederives_source_and_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    result = isolation.derive_isolated_rtu_ppo_source(source)
    descriptor = cast(
        dict[str, Any],
        copy.deepcopy(isolation._plain_json(result.descriptor)),
    )

    isolation.validate_isolated_rtu_ppo_source(
        source,
        result.source,
        descriptor,
    )

    tampered_source = result.source.replace(
        b"ISOLATED_AGENT_RNG_NAMESPACE = 0xA63E7C11",
        b"ISOLATED_AGENT_RNG_NAMESPACE = 0xA63E7C12",
    )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="source bytes differ",
    ):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            tampered_source,
            descriptor,
        )

    tampered_descriptor = copy.deepcopy(descriptor)
    tampered_descriptor["agent_rng"]["namespace"] += 1
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="descriptor differs",
    ):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            tampered_descriptor,
        )


def test_validation_rejects_non_detached_or_wrong_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    result = isolation.derive_isolated_rtu_ppo_source(source)
    with pytest.raises(TypeError, match="upstream_source"):
        isolation.validate_isolated_rtu_ppo_source(
            cast(Any, "source"),
            result.source,
            {},
        )
    with pytest.raises(TypeError, match="derived_source"):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            cast(Any, "derived"),
            {},
        )
    with pytest.raises(TypeError, match="descriptor"):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            cast(Any, []),
        )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="detached canonical object",
    ):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            result.descriptor,
        )
