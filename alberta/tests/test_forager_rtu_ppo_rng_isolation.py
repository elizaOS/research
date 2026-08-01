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

def main():
    rngs = []
    for seed in [0]:
        rng = jax.random.PRNGKey(seed)
        rngs.append(rng)
    return rngs
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
    patch = isolation._unified_diff(source, derived)
    monkeypatch.setattr(
        isolation,
        "EXPECTED_UNIFIED_DIFF_SHA256",
        hashlib.sha256(patch).hexdigest(),
    )


def test_frozen_upstream_and_derived_hashes() -> None:
    assert isolation.UPSTREAM_RTU_PPO_SOURCE_SHA256 == (
        "e75a6762690832067a24a649559a55e0aa89abba005d600f090b1bf284b3fc24"
    )
    assert isolation.EXPECTED_PATCHED_RTU_PPO_SHA256 == (
        "70bbdd0943d82570c1dc0d28494cf93f9c1b208ef67b3a547585fe5897cdf409"
    )
    assert isolation.EXPECTED_UNIFIED_DIFF_SHA256 == (
        "46ac3d6c1ae5740bee97fea23abf002ffb161ab4b1b35c041b24b717645e076f"
    )
    assert isolation.UPSTREAM_SOURCE_COMMIT == (
        "9710f60fa30da5badc451ad7ce3ff296d5070830"
    )
    assert isolation.UPSTREAM_SOURCE_TREE_GIT_SHA1 == (
        "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
    )
    assert isolation.UPSTREAM_RTU_PPO_BLOB_GIT_SHA1 == (
        "63bdc359079ef14b0de1e5964ed49b02c62b3e59"
    )
    assert isolation.UPSTREAM_SOURCE_ARCHIVE_SHA256 == (
        "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
    )


def test_exact_derivation_separates_agent_and_environment_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)

    first = isolation.derive_isolated_rtu_ppo_source(source)
    second = isolation.derive_isolated_rtu_ppo_source(source)

    assert first.source == second.source
    assert first.upstream_source_sha256 == hashlib.sha256(source).hexdigest()
    assert first.source_sha256 == hashlib.sha256(first.source).hexdigest()
    assert first.patch == second.patch
    assert first.patch_sha256 == hashlib.sha256(first.patch).hexdigest()
    assert first.patch.startswith(b"--- a/src/rtu_ppo.py\n+++ b/src/rtu_ppo.py\n")
    assert first.descriptor_sha256 == hashlib.sha256(
        isolation._canonical_json(dict(first.descriptor))
    ).hexdigest()
    assert first.descriptor == second.descriptor
    assert first.source.count(b"environment_rng: Any") == 1
    assert first.source.count(b"environment_rng=") == 3
    assert first.source.count(b"environment_step_rng") == 2
    assert first.source.count(b"ISOLATED_AGENT_RNG_NAMESPACE") == 2
    assert first.source.count(
        b'jax.random.key(seed, impl="threefry2x32")'
    ) == 1
    assert b"jax.random.PRNGKey(seed)" not in first.source
    assert b"gymnax_state.env_step(\n        _rng," not in first.source
    assert first.descriptor["environment_rng"]["schedule"] == (
        "dedicated_environment_split_chain_v1"
    )
    assert first.descriptor["environment_rng"]["schedule_sha256"] == (
        "51d811e6fccd2b015b1703f22775f880089bbca3fc8938421ad3e18526882cb0"
    )
    assert first.descriptor["environment_rng"][
        "action_key_shared_with_environment"
    ] is False
    assert first.descriptor["agent_rng"]["namespace"] == 0xA63E7C11
    assert first.descriptor["agent_rng"]["identity"] == "isolated_agent_rng_v1"
    assert first.descriptor["agent_rng"]["environment_key_shared"] is False
    assert first.descriptor["patch_sha256"] == first.patch_sha256
    assert first.descriptor["upstream"]["tree_git_sha1"] == (
        isolation.UPSTREAM_SOURCE_TREE_GIT_SHA1
    )
    assert len(first.descriptor["replacement_records"]) == 8
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
        result.patch,
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
            result.patch,
            descriptor,
        )

    tampered_patch = result.patch.replace(
        b"ISOLATED_AGENT_RNG_NAMESPACE",
        b"TAMPERED_AGENT_RNG_NAMESPACE",
        1,
    )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="patch bytes differ",
    ):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            tampered_patch,
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
            result.patch,
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
            result.patch,
            {},
        )
    with pytest.raises(TypeError, match="derived_source"):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            cast(Any, "derived"),
            result.patch,
            {},
        )
    with pytest.raises(TypeError, match="patch"):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            cast(Any, "patch"),
            {},
        )
    with pytest.raises(TypeError, match="descriptor"):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            result.patch,
            cast(Any, []),
        )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="detached canonical object",
    ):
        isolation.validate_isolated_rtu_ppo_source(
            source,
            result.source,
            result.patch,
            result.descriptor,
        )


@pytest.mark.parametrize(
    ("before", "after", "message"),
    [
        (
            b'jax.random.key(seed, impl="threefry2x32")',
            b'jax.random.key(seed, impl="rbg")',
            "environment root must be",
        ),
        (
            b'jax.random.key(seed, impl="threefry2x32")',
            b"jax.random.PRNGKey(seed)",
            "exactly one explicit environment seed root",
        ),
        (
            b"jax.random.fold_in(rng, ISOLATED_AGENT_RNG_NAMESPACE)",
            b"jax.random.fold_in(environment_rng, ISOLATED_AGENT_RNG_NAMESPACE)",
            "root one environment chain",
        ),
    ],
)
def test_ast_validation_rejects_seed_root_or_agent_root_drift(
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


def test_public_runtime_probe_is_fixed_nonpromoting_and_agent_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)

    first = isolation.run_public_rng_isolation_probe(source)
    second = isolation.run_public_rng_isolation_probe(source)

    assert first == second
    assert first["seed"] == 0
    assert first["transition_count"] == 4
    assert first["prng_impl"] == "threefry2x32"
    assert first["promotion_authorized"] is False
    assert first["evidence_boundary"] == (
        "public_seed_key_schedule_only_no_environment_or_training"
    )
    assert first["environment_trace_sha256"] == (
        isolation.PUBLIC_PROBE_EXPECTED_ENVIRONMENT_TRACE_SHA256
    )
    checks = first["agent_consumption_checks"]
    assert tuple(check["agent_split_count"] for check in checks) == (0, 1, 7, 32)
    assert {
        check["environment_trace_sha256"] for check in checks
    } == {first["environment_trace_sha256"]}
    detached = cast(dict[str, Any], isolation._plain_json(first))
    payload_sha256 = detached.pop("payload_sha256")
    assert payload_sha256 == hashlib.sha256(
        isolation._canonical_json(detached)
    ).hexdigest()


def test_public_runtime_probe_rejects_frozen_schedule_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _upstream_fixture()
    _allow_fixture(monkeypatch, source)
    monkeypatch.setattr(
        isolation,
        "PUBLIC_PROBE_EXPECTED_ENVIRONMENT_TRACE_SHA256",
        "0" * 64,
    )
    with pytest.raises(
        isolation.RTUPPORngIsolationError,
        match="public environment key trace differs",
    ):
        isolation.run_public_rng_isolation_probe(source)
