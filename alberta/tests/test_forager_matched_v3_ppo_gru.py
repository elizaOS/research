"""Cheap contract and core tests for the POBAX-derived matched-v3 PPO-GRU arm."""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from alberta_framework.benchmarks import forager_matched_v3_ppo_gru as ppo_gru

_POBAX_ROOT = Path("/tmp/pobax-v3-source")
_EXPECTED_CONFIGURATION_SHA256 = (
    "07e897431bf8925ddde95b2fc155c7ae4566a3bc42e8407579b9b816e6afdf70"
)
_EXPECTED_SOURCE_DESCRIPTOR_SHA256 = (
    "64f9568f56f76152f3c6bf4d99a076663ac3d2d60408e1eaa63b8bdffec8d4ca"
)
_EXPECTED_SOURCE_HASHES = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "pobax/algos/ppo.py": (
        "0c82725027e6022d48847bca45a87e6f8d9b54d720bbb844f053d4b8448ce153"
    ),
    "pobax/config.py": (
        "38bb46c93734c8882ab7ad7bdfbee9d64bb21db04231ccd15b9ec2a6eb02034c"
    ),
    "pobax/models/actor_critic.py": (
        "bb707481b32eefc1219adbc38abd527c3c600cf8941ae963bf6b6540c9b2158f"
    ),
    "pobax/models/discrete.py": (
        "ad7ac11a03b49f7ea53fcf11b0b97cc7697f57447f4661a22fb235a6ab90885c"
    ),
    "pobax/models/__init__.py": (
        "c4434b0b1eba13c227cdf479380f5347aa57aba4d2f78a12112c056cdada323a"
    ),
    "pobax/models/network.py": (
        "b3ea151f6a7f9000dd1b529cbcc262c150b767c66664399008aa89283a2e520a"
    ),
    "pobax/models/value.py": (
        "e875e7ef951aba37ea4648328442aaece0fc3415de580c6b5115843eb32366bd"
    ),
    "pyproject.toml": (
        "4f02e96a5d8471f9637ec36dc9536398183f49fb28fa07c5b7f371ffcdbe81d5"
    ),
    "requirements.txt": (
        "8d8a36a4428d481b15c47b9ed1aec573c3dc2472af746be611e9a17dae40a17c"
    ),
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pinned_sources() -> dict[str, bytes]:
    missing = [path for path in _EXPECTED_SOURCE_HASHES if not (_POBAX_ROOT / path).is_file()]
    if missing:
        pytest.skip(f"pinned POBAX checkout unavailable: {missing!r}")
    return {path: (_POBAX_ROOT / path).read_bytes() for path in _EXPECTED_SOURCE_HASHES}


@pytest.mark.integration
def test_pobax_source_identity_license_and_relevant_files_are_exact() -> None:
    assert ppo_gru.POBAX_CANONICAL_URL == "https://github.com/taodav/pobax"
    assert ppo_gru.POBAX_COMMIT_GIT_SHA1 == (
        "a5e1d62d14e4efe783885b9d4f19cffa2a568eec"
    )
    assert ppo_gru.POBAX_TREE_GIT_SHA1 == "d67cf5c209f2e7de9ce517d4bc72a2741ccaf6a6"
    assert ppo_gru.POBAX_ARCHIVE_SHA256 == (
        "f354028549d79a1b3f1ee67deaa46454a0be60d9346764e5aed9e8ab93768ad9"
    )
    assert ppo_gru.POBAX_ARCHIVE_SIZE_BYTES == 1_699_840
    assert ppo_gru.POBAX_LICENSE == "Apache-2.0"
    assert dict(ppo_gru.REQUIRED_POBAX_SOURCE_SHA256_BY_PATH) == _EXPECTED_SOURCE_HASHES

    verified = ppo_gru.verify_pobax_source_files(_pinned_sources())
    assert verified["status"] == "verified_pinned_source_files"
    assert verified["commit_git_sha1"] == ppo_gru.POBAX_COMMIT_GIT_SHA1
    assert verified["source_sha256_by_path"] == _EXPECTED_SOURCE_HASHES
    assert verified["license"] == "Apache-2.0"


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["missing", "extra", "bytes", "mapping_alias"])
def test_pobax_source_verifier_fails_closed_without_external_checkout(
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources: Any = {
        "LICENSE": b"synthetic license",
        "pobax/algos/ppo.py": b"synthetic ppo",
    }
    monkeypatch.setattr(
        ppo_gru,
        "REQUIRED_POBAX_SOURCE_SHA256_BY_PATH",
        MappingProxyType(
            {path: hashlib.sha256(raw).hexdigest() for path, raw in sources.items()}
        ),
    )
    monkeypatch.setattr(
        ppo_gru,
        "REQUIRED_POBAX_SOURCE_SIZE_BYTES_BY_PATH",
        MappingProxyType({path: len(raw) for path, raw in sources.items()}),
    )
    if mutation == "missing":
        sources.pop("pobax/algos/ppo.py")
    elif mutation == "extra":
        sources["extra.py"] = b""
    elif mutation == "bytes":
        sources["pobax/algos/ppo.py"] += b"\n# drift\n"
    else:
        class MappingAlias(dict[str, bytes]):
            pass

        sources = MappingAlias(sources)

    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="source"):
        ppo_gru.verify_pobax_source_files(sources)


@pytest.mark.unit
def test_exact_task_configuration_and_accounting_close_arithmetically() -> None:
    config = ppo_gru.matched_v3_ppo_gru_configuration()
    payload = config.to_dict()
    accounting = payload["accounting"]

    assert payload["candidate_id"] == "adapted_ppo_gru"
    assert payload["status"] == "implemented_unqualified"
    assert payload["task"] == {
        "environment_id": "ForagaxTwoBiomeLarge-v1",
        "continuing": True,
        "observation_type": "color",
        "observation_shape": [9, 9, 3],
        "aperture_size": 9,
        "num_actions": 4,
        "action_distribution": "categorical",
        "reward_range": [-1.0, 30.0],
        "reward_preprocessing": "identity_no_scaling",
        "public_trajectory_count": 1,
        "parallel_environments": 1,
        "parallel_reward_aggregation": False,
    }
    assert config.horizon == 499_712
    assert config.rollout_steps == 512
    assert config.rollout_count == 976
    assert config.segment_steps == 128
    assert config.segments_per_rollout == 4
    assert config.update_epochs == 4
    assert config.optimizer_updates_per_rollout == 16
    assert config.optimizer_update_count == 15_616
    assert config.loss_transition_evaluations == 1_998_848
    assert config.rollout_steps * config.rollout_count == config.horizon
    assert config.segment_steps * config.segments_per_rollout == config.rollout_steps
    assert accounting == {
        "environment_interactions": 499_712,
        "rollout_count": 976,
        "transitions_per_rollout": 512,
        "segments_per_rollout": 4,
        "transitions_per_segment": 128,
        "update_epochs": 4,
        "optimizer_updates_per_rollout": 16,
        "optimizer_update_count": 15_616,
        "loss_transition_evaluations": 1_998_848,
        "agent_parameter_initialization_draws": 1,
        "agent_action_sampling_draws": 499_712,
        "agent_segment_permutation_draws": 3_904,
        "total_agent_subkey_draws": 503_617,
        "reward_aggregation": "none_single_continuing_trajectory",
    }
    assert payload["seed_contract"] == {
        "environment_seed": "required_exact_uint31_root",
        "agent_seed": "required_exact_uint31_root",
        "prng_implementation": "threefry2x32",
        "roots_are_logically_separate_consumption_chains": True,
        "equal_numeric_values_allowed": True,
        "equal_numeric_values_correlate_key_streams": True,
        "statistical_independence_claimed": False,
        "environment_draws_from_agent_root": False,
        "agent_draws_from_environment_root": False,
    }
    assert payload["optimization"]["gae_recursion"] == (
        "independent_128_step_segments_bootstrapped_at_next_behavior_value"
    )
    assert payload["rollout_and_segmentation"]["gae_recursion_steps"] == 128
    assert payload["rollout_and_segmentation"]["segment_order"] == (
        "required_agent_rng_permutation_once_per_epoch"
    )
    assert payload["architecture"]["categorical_sampling_mode"] == "low"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("horizon", 499_711),
        ("num_envs", 2),
        ("num_actions", 5),
        ("observation_shape", (9, 9)),
        ("rollout_steps", 511),
        ("segment_steps", 129),
        ("update_epochs", 0),
        ("hidden_size", True),
        ("gamma", 1.1),
        ("max_grad_norm", 0.0),
    ],
)
def test_invalid_exact_task_or_optimizer_configuration_is_rejected(
    field: str, value: object
) -> None:
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError):
        dataclasses.replace(
            ppo_gru.matched_v3_ppo_gru_configuration(),
            **cast(dict[str, Any], {field: value}),
        )


@pytest.mark.unit
def test_canonical_configuration_and_source_descriptor_are_strict_and_non_authorizing() -> None:
    config_bytes = ppo_gru.canonical_matched_v3_ppo_gru_configuration_bytes()
    descriptor = ppo_gru.matched_v3_ppo_gru_source_descriptor()
    descriptor_bytes = ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes()

    assert config_bytes == _canonical(ppo_gru.matched_v3_ppo_gru_configuration().to_dict())
    assert hashlib.sha256(config_bytes).hexdigest() == ppo_gru.PPO_GRU_CONFIGURATION_SHA256
    assert descriptor_bytes == _canonical(descriptor)
    assert hashlib.sha256(descriptor_bytes).hexdigest() == (
        ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256
    )
    assert ppo_gru.PPO_GRU_CONFIGURATION_SHA256 == _EXPECTED_CONFIGURATION_SHA256
    assert ppo_gru.PPO_GRU_SOURCE_DESCRIPTOR_SHA256 == (
        _EXPECTED_SOURCE_DESCRIPTOR_SHA256
    )
    assert descriptor["status"] == "implemented_unqualified"
    assert descriptor["relationship"] == "derived_exact_task_adapter"
    assert descriptor["claims"] == {
        "configuration_complete": True,
        "core_implementation_complete": True,
        "validated_epoch_driver_complete": False,
        "full_forager_runner_complete": False,
        "execution_ready": False,
        "execution_authorized": False,
        "scientific_promotion_allowed": False,
        "performance_claim_allowed": False,
        "universal_sota_claim_allowed": False,
        "authority_granted": False,
    }
    assert descriptor["runner_blockers"] == [
        "qualified_foragax_environment_bridge_missing",
        "full_horizon_compilation_and_memory_profile_unqualified",
        "environment_trace_and_rng_parity_unqualified",
        "validated_epoch_driver_unimplemented",
        "artifact_writer_and_execution_receipt_unimplemented",
    ]
    assert descriptor["upstream"]["upstream_review_anchors_bound"] is True
    assert descriptor["upstream"]["source_closure_bound"] is False
    assert ppo_gru.parse_matched_v3_ppo_gru_configuration(config_bytes).to_dict() == (
        ppo_gru.matched_v3_ppo_gru_configuration().to_dict()
    )
    assert ppo_gru.parse_matched_v3_ppo_gru_source_descriptor(descriptor_bytes) == (
        descriptor
    )

    mutated = copy.deepcopy(descriptor)
    mutated["claims"]["execution_authorized"] = True
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="descriptor"):
        ppo_gru.parse_matched_v3_ppo_gru_source_descriptor(_canonical(mutated))
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="canonical"):
        ppo_gru.parse_matched_v3_ppo_gru_configuration(
            json.dumps(
                ppo_gru.matched_v3_ppo_gru_configuration().to_dict(), indent=2
            ).encode("utf-8")
        )


@pytest.mark.unit
def test_descriptor_decodes_from_canonical_bytes_despite_private_cache_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes()
    expected = json.loads(canonical)
    mutated = copy.deepcopy(getattr(ppo_gru, "_SOURCE_DESCRIPTOR"))
    mutated["claims"]["execution_authorized"] = True
    monkeypatch.setattr(ppo_gru, "_SOURCE_DESCRIPTOR", mutated)

    assert ppo_gru.matched_v3_ppo_gru_source_descriptor() == expected
    assert ppo_gru.parse_matched_v3_ppo_gru_source_descriptor(canonical) == expected
    assert ppo_gru.canonical_matched_v3_ppo_gru_source_descriptor_bytes() == canonical


@pytest.mark.unit
@pytest.mark.parametrize("seed", [-1, 2**31, True, 1.0, "1", None])
def test_seed_roots_reject_non_uint31_aliases(seed: object) -> None:
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="uint31"):
        ppo_gru.validate_ppo_gru_seed_pair(seed, 0)


@pytest.mark.unit
def test_environment_and_agent_rng_chains_are_logically_separate_and_pinned() -> None:
    equal = ppo_gru.validate_ppo_gru_seed_pair(17, 17)
    distinct = ppo_gru.validate_ppo_gru_seed_pair(17, 29)
    maximum = ppo_gru.validate_ppo_gru_seed_pair(2**31 - 1, 2**31 - 1)
    assert equal.environment_seed == equal.agent_seed == 17
    assert distinct.environment_seed == 17
    assert distinct.agent_seed == 29
    assert maximum.environment_seed == maximum.agent_seed == 2**31 - 1

    correlated = ppo_gru.initialize_ppo_gru_rng_state(17, 17)
    assert ppo_gru.validate_ppo_gru_rng_state(correlated) == (0, 0)
    assert str(jax.random.key_impl(correlated.environment_key)) == "threefry2x32"
    np.testing.assert_array_equal(
        jax.random.key_data(correlated.environment_key),
        jax.random.key_data(correlated.agent_key),
    )

    baseline = ppo_gru.initialize_ppo_gru_rng_state(17, 29)
    consumed = baseline
    for _ in range(31):
        consumed, _ = ppo_gru.next_ppo_gru_agent_key(consumed)
    baseline_environment_keys: list[np.ndarray] = []
    consumed_environment_keys: list[np.ndarray] = []
    for _ in range(8):
        baseline, key = ppo_gru.next_ppo_gru_environment_key(baseline)
        baseline_environment_keys.append(np.asarray(jax.random.key_data(key)))
        consumed, key = ppo_gru.next_ppo_gru_environment_key(consumed)
        consumed_environment_keys.append(np.asarray(jax.random.key_data(key)))
    np.testing.assert_array_equal(baseline_environment_keys, consumed_environment_keys)
    assert int(consumed.agent_draw_count) == 31
    assert int(consumed.environment_draw_count) == 8


@pytest.mark.unit
@pytest.mark.slow
def test_train_initialization_and_epoch_permutations_share_one_agent_key_owner() -> None:
    config, _, _ = _small_model()
    rng = ppo_gru.initialize_ppo_gru_rng_state(3, 5)
    root = rng.agent_key
    expected_after_initialization, initialization_key = jax.random.split(root)
    train_state, rng = ppo_gru.initialize_ppo_gru_train_state(config, rng_state=rng)
    assert int(rng.agent_draw_count) == 1
    np.testing.assert_array_equal(
        jax.random.key_data(rng.agent_key),
        jax.random.key_data(expected_after_initialization),
    )

    rng, action_key = ppo_gru.next_ppo_gru_agent_key(rng)
    assert not np.array_equal(
        np.asarray(jax.random.key_data(action_key)),
        np.asarray(jax.random.key_data(initialization_key)),
    )
    before_permutation = int(rng.agent_draw_count)
    rng, order = jax.jit(ppo_gru.next_ppo_gru_segment_order)(rng)
    assert int(rng.agent_draw_count) == before_permutation + 1
    assert sorted(np.asarray(order).tolist()) == [0, 1, 2, 3]
    assert int(train_state.optimizer_updates) == 0

    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="first"):
        ppo_gru.initialize_ppo_gru_train_state(config, rng_state=rng)


@pytest.mark.unit
def test_native_jax_categorical_sampling_log_prob_and_entropy() -> None:
    logits = jnp.asarray([[0.0, 1.0, -1.0, 0.5], [1.0, 0.0, 0.0, -2.0]])
    actions = jnp.asarray([1, 0], dtype=jnp.int32)
    log_prob = ppo_gru.categorical_log_prob(logits, actions)
    entropy = ppo_gru.categorical_entropy(logits)

    expected_log_probs = jax.nn.log_softmax(logits)
    np.testing.assert_allclose(
        log_prob,
        jnp.take_along_axis(expected_log_probs, actions[:, None], axis=-1).squeeze(-1),
    )
    np.testing.assert_allclose(
        entropy,
        -jnp.sum(jnp.exp(expected_log_probs) * expected_log_probs, axis=-1),
    )
    first = ppo_gru.sample_categorical_action(
        jax.random.key(5, impl="threefry2x32"), logits
    )
    second = ppo_gru.sample_categorical_action(
        jax.random.key(5, impl="threefry2x32"), logits
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2,)
    assert np.all((np.asarray(first) >= 0) & (np.asarray(first) < 4))
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="0..3"):
        ppo_gru.categorical_log_prob(logits[:1], jnp.asarray([-1], dtype=jnp.int32))


@functools.lru_cache(maxsize=1)
def _small_model() -> tuple[ppo_gru.PPOGRUConfig, ppo_gru.PPOGRUActorCritic, Any]:
    config = dataclasses.replace(
        ppo_gru.matched_v3_ppo_gru_configuration(), hidden_size=2
    )
    model = ppo_gru.PPOGRUActorCritic(
        hidden_size=config.hidden_size,
        num_actions=config.num_actions,
    )
    variables = model.init(
        jax.random.key(11, impl="threefry2x32"),
        jnp.zeros((config.hidden_size,), dtype=jnp.float32),
        jnp.zeros(config.observation_shape, dtype=jnp.float32),
        jnp.asarray(False),
    )
    return config, model, variables


@pytest.mark.unit
@pytest.mark.slow
def test_gru_carry_is_preserved_and_reset_exactly_at_done_boundaries() -> None:
    config, model, variables = _small_model()
    obs = jnp.linspace(0.0, 1.0, math.prod(config.observation_shape)).reshape(
        config.observation_shape
    )
    zero = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
    nonzero = jnp.arange(config.hidden_size, dtype=jnp.float32) + 1.0

    reset_zero = model.apply(variables, zero, obs, jnp.asarray(True))
    reset_nonzero = model.apply(variables, nonzero, obs, jnp.asarray(True))
    for left, right in zip(reset_zero, reset_nonzero, strict=True):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)
    continued_zero = model.apply(variables, zero, obs, jnp.asarray(False))
    continued_nonzero = model.apply(variables, nonzero, obs, jnp.asarray(False))
    assert not np.allclose(continued_zero[0], continued_nonzero[0])

    observations = jnp.stack((obs * 0.5, obs * 0.25, obs, obs * 0.75))
    resets = jnp.asarray([False, False, True, False])
    combined = ppo_gru.evaluate_ppo_gru_sequence(
        model, variables, nonzero, observations, resets
    )
    suffix = ppo_gru.evaluate_ppo_gru_sequence(
        model,
        variables,
        zero,
        observations[2:],
        jnp.asarray([True, False]),
    )
    np.testing.assert_allclose(combined.logits[2:], suffix.logits, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(combined.values[2:], suffix.values, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(combined.final_carry, suffix.final_carry, rtol=0.0, atol=0.0)


@pytest.mark.unit
def test_gae_uses_transition_done_boundary_and_bootstrap_value() -> None:
    advantages, targets = ppo_gru.calculate_gae(
        rewards=jnp.asarray([1.0, 1.0]),
        values=jnp.asarray([0.5, 0.25]),
        transition_dones=jnp.asarray([False, True]),
        bootstrap_value=jnp.asarray(7.0),
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, jnp.asarray([1.5, 0.75]))
    np.testing.assert_allclose(targets, jnp.asarray([2.0, 1.0]))
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="boolean"):
        ppo_gru.calculate_gae(
            rewards=jnp.asarray([1.0], dtype=jnp.float32),
            values=jnp.asarray([0.0], dtype=jnp.float32),
            transition_dones=jnp.asarray([2], dtype=jnp.int32),
            bootstrap_value=jnp.asarray(0.0, dtype=jnp.float32),
            gamma=1.0,
            gae_lambda=1.0,
        )


@pytest.mark.unit
def test_segmented_gae_preserves_upstream_128_step_lane_recursion_semantics() -> None:
    rollout = ppo_gru.PPOGRURollout(
        initial_carry=jnp.zeros((2,), dtype=jnp.float32),
        observations=jnp.zeros((4, 9, 9, 3), dtype=jnp.float32),
        reset_before=jnp.zeros((4,), dtype=jnp.bool_),
        actions=jnp.zeros((4,), dtype=jnp.int32),
        rewards=jnp.ones((4,), dtype=jnp.float32),
        transition_dones=jnp.zeros((4,), dtype=jnp.bool_),
        old_log_probs=jnp.zeros((4,), dtype=jnp.float32),
        old_values=jnp.zeros((4,), dtype=jnp.float32),
        incoming_carries=jnp.zeros((4, 2), dtype=jnp.float32),
        bootstrap_observation=jnp.zeros((9, 9, 3), dtype=jnp.float32),
        bootstrap_value=jnp.asarray(10.0),
    )
    advantages, targets = ppo_gru.calculate_segmented_gae(
        rollout,
        segment_steps=2,
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(advantages, jnp.asarray([2.0, 1.0, 12.0, 11.0]))
    np.testing.assert_allclose(targets, advantages)

    unsegmented, _ = ppo_gru.calculate_gae(
        rewards=rollout.rewards,
        values=rollout.old_values,
        transition_dones=rollout.transition_dones,
        bootstrap_value=rollout.bootstrap_value,
        gamma=1.0,
        gae_lambda=1.0,
    )
    np.testing.assert_allclose(unsegmented, jnp.asarray([14.0, 13.0, 12.0, 11.0]))


def _tiny_rollout() -> ppo_gru.PPOGRURollout:
    steps = 8
    observations = jnp.zeros((steps, 9, 9, 3), dtype=jnp.float32)
    observations = observations.at[jnp.arange(steps), 0, 0, jnp.arange(steps) % 3].set(
        1.0
    )
    carries = jnp.arange(steps * 3, dtype=jnp.float32).reshape(steps, 3)
    return ppo_gru.PPOGRURollout(
        initial_carry=carries[0],
        observations=observations,
        reset_before=jnp.asarray([False, False, True, False, False, False, True, False]),
        actions=jnp.arange(steps, dtype=jnp.int32) % 4,
        rewards=jnp.arange(steps, dtype=jnp.float32),
        transition_dones=jnp.asarray(
            [False, True, False, False, False, True, False, False]
        ),
        old_log_probs=-jnp.arange(steps, dtype=jnp.float32),
        old_values=jnp.arange(steps, dtype=jnp.float32) / 2,
        incoming_carries=carries,
        bootstrap_observation=jnp.zeros((9, 9, 3), dtype=jnp.float32),
        bootstrap_value=jnp.asarray(0.25),
    )


@pytest.mark.unit
def test_time_segment_minibatches_preserve_sequence_and_bind_initial_carry() -> None:
    rollout = _tiny_rollout()
    advantages = jnp.arange(8, dtype=jnp.float32) + 10
    targets = jnp.arange(8, dtype=jnp.float32) + 20
    order = (2, 0, 3, 1)
    segments = ppo_gru.build_ppo_gru_sequence_segments(
        rollout,
        advantages,
        targets,
        segment_steps=2,
        segment_order=order,
    )

    np.testing.assert_array_equal(segments.segment_ids, jnp.asarray(order))
    np.testing.assert_array_equal(
        segments.time_indices,
        jnp.asarray([[4, 5], [0, 1], [6, 7], [2, 3]]),
    )
    np.testing.assert_array_equal(
        segments.initial_carries,
        rollout.incoming_carries[jnp.asarray([4, 0, 6, 2])],
    )
    np.testing.assert_array_equal(
        segments.observations,
        rollout.observations.reshape((4, 2, 9, 9, 3))[jnp.asarray(order)],
    )
    assert np.all(np.diff(np.asarray(segments.time_indices), axis=1) == 1)
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="permutation"):
        ppo_gru.build_ppo_gru_sequence_segments(
            rollout,
            advantages,
            targets,
            segment_steps=2,
            segment_order=jnp.asarray([0, 0, 2, 3], dtype=jnp.int32),
        )


@functools.lru_cache(maxsize=1)
def _valid_exact_rollout() -> tuple[
    ppo_gru.PPOGRUConfig,
    ppo_gru.PPOGRUActorCritic,
    Any,
    ppo_gru.PPOGRURollout,
]:
    config, model, variables = _small_model()
    steps = config.rollout_steps
    time = jnp.arange(steps, dtype=jnp.int32)
    observations = jnp.zeros((steps, *config.observation_shape), dtype=jnp.float32)
    observations = observations.at[time, 4, 4, time % 3].set(1.0)
    bootstrap_observation = jnp.zeros(config.observation_shape, dtype=jnp.float32)
    bootstrap_observation = bootstrap_observation.at[4, 4, 0].set(1.0)
    transition_dones = jnp.zeros((steps,), dtype=jnp.bool_).at[127].set(True)
    reset_before = jnp.concatenate(
        (jnp.asarray([False]), transition_dones[:-1]), axis=0
    )
    initial_carry = jnp.asarray([0.25, -0.5], dtype=jnp.float32)
    evaluation = ppo_gru.evaluate_ppo_gru_sequence(
        model,
        variables,
        initial_carry,
        observations,
        reset_before,
    )
    actions = time % 4
    old_log_probs = ppo_gru.categorical_log_prob(evaluation.logits, actions)
    _, _, bootstrap_value = cast(
        tuple[Any, Any, Any],
        model.apply(
            variables,
            evaluation.final_carry,
            bootstrap_observation,
            transition_dones[-1],
        ),
    )
    rollout = ppo_gru.PPOGRURollout(
        initial_carry=initial_carry,
        observations=observations,
        reset_before=reset_before,
        actions=actions,
        rewards=jnp.zeros((steps,), dtype=jnp.float32),
        transition_dones=transition_dones,
        old_log_probs=old_log_probs,
        old_values=evaluation.values,
        incoming_carries=evaluation.incoming_carries,
        bootstrap_observation=bootstrap_observation,
        bootstrap_value=bootstrap_value,
    )
    return config, model, variables, rollout


@pytest.mark.unit
@pytest.mark.slow
def test_exact_rollout_replay_gae_and_segment_rows_are_bound() -> None:
    config, model, variables, rollout = _valid_exact_rollout()
    order = jnp.asarray([2, 0, 3, 1], dtype=jnp.int32)
    segments = ppo_gru.build_validated_ppo_gru_sequence_segments(
        model,
        variables,
        rollout,
        config,
        expected_initial_carry=rollout.initial_carry,
        expected_initial_reset=False,
        segment_order=order,
    )
    ppo_gru.validate_ppo_gru_sequence_segments(
        segments,
        model,
        variables,
        rollout,
        config,
        expected_initial_carry=rollout.initial_carry,
        expected_initial_reset=False,
    )
    batch = ppo_gru.ppo_gru_loss_batch_from_segment(segments, 0, config)
    np.testing.assert_array_equal(batch.initial_carry, segments.initial_carries[0])
    np.testing.assert_array_equal(batch.advantages, segments.advantages[0])

    duplicate_order = jnp.zeros((4,), dtype=jnp.int32)
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="permutation"):
        ppo_gru.build_validated_ppo_gru_sequence_segments(
            model,
            variables,
            rollout,
            config,
            expected_initial_carry=rollout.initial_carry,
            expected_initial_reset=False,
            segment_order=duplicate_order,
        )


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.parametrize(
    "field", ["time_indices", "initial_carries", "observations", "advantages"]
)
def test_segment_validator_rejects_timestep_or_carry_forgery(field: str) -> None:
    config, model, variables, rollout = _valid_exact_rollout()
    segments = ppo_gru.build_validated_ppo_gru_sequence_segments(
        model,
        variables,
        rollout,
        config,
        expected_initial_carry=rollout.initial_carry,
        expected_initial_reset=False,
        segment_order=(0, 1, 2, 3),
    )
    if field == "time_indices":
        forged = dataclasses.replace(
            segments,
            time_indices=segments.time_indices.at[0, :2].set(jnp.asarray([1, 0])),
        )
    elif field == "initial_carries":
        forged = dataclasses.replace(
            segments,
            initial_carries=segments.initial_carries.at[0, 0].add(1),
        )
    elif field == "observations":
        forged = dataclasses.replace(
            segments,
            observations=segments.observations.at[0, 0].set(segments.observations[0, 1]),
        )
    else:
        forged = dataclasses.replace(
            segments,
            advantages=segments.advantages.at[0, 0].add(1.0),
        )
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="segment"):
        ppo_gru.validate_ppo_gru_sequence_segments(
            forged,
            model,
            variables,
            rollout,
            config,
            expected_initial_carry=rollout.initial_carry,
            expected_initial_reset=False,
        )


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.parametrize(
    "mutation",
    [
        "action_range",
        "initial_reset",
        "shifted_reset",
        "initial_carry",
        "bootstrap_value",
        "observation",
        "reward_range",
        "nonfinite_value",
    ],
)
def test_rollout_validator_rejects_behavior_or_task_forgery(mutation: str) -> None:
    config, model, variables, rollout = _valid_exact_rollout()
    if mutation == "action_range":
        forged = dataclasses.replace(
            rollout, actions=rollout.actions.at[0].set(jnp.int32(-1))
        )
    elif mutation == "initial_reset":
        forged = dataclasses.replace(
            rollout, reset_before=rollout.reset_before.at[0].set(True)
        )
    elif mutation == "shifted_reset":
        forged = dataclasses.replace(
            rollout, reset_before=rollout.reset_before.at[1].set(True)
        )
    elif mutation == "initial_carry":
        forged = dataclasses.replace(
            rollout, initial_carry=rollout.initial_carry.at[0].add(1.0)
        )
    elif mutation == "bootstrap_value":
        forged = dataclasses.replace(
            rollout, bootstrap_value=rollout.bootstrap_value + 1.0
        )
    elif mutation == "observation":
        forged = dataclasses.replace(
            rollout, observations=rollout.observations.at[0, 0, 0, 0].set(0.5)
        )
    elif mutation == "reward_range":
        forged = dataclasses.replace(
            rollout, rewards=rollout.rewards.at[0].set(31.0)
        )
    else:
        forged = dataclasses.replace(
            rollout, old_values=rollout.old_values.at[0].set(jnp.nan)
        )
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError):
        ppo_gru.validate_ppo_gru_rollout(
            model,
            variables,
            forged,
            config,
            expected_initial_carry=rollout.initial_carry,
            expected_initial_reset=False,
        )


@pytest.mark.unit
@pytest.mark.slow
def test_segmented_gae_core_is_jittable_and_matches_strict_rollout_validation() -> None:
    config, model, variables, rollout = _valid_exact_rollout()
    expected, expected_targets = ppo_gru.validate_ppo_gru_rollout(
        model,
        variables,
        rollout,
        config,
        expected_initial_carry=rollout.initial_carry,
        expected_initial_reset=False,
    )
    compiled = jax.jit(
        lambda value: ppo_gru.calculate_segmented_gae_core(
            value,
            segment_steps=config.segment_steps,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
    )
    actual, actual_targets = compiled(rollout)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_targets, expected_targets)


@pytest.mark.unit
def test_clipped_policy_value_entropy_loss_matches_manual_arithmetic() -> None:
    logits = jnp.asarray(
        [[math.log(9.0), 0.0, -100.0, -100.0], [0.0, math.log(9.0), -100.0, -100.0]]
    )
    actions = jnp.asarray([0, 0], dtype=jnp.int32)
    old_log_probs = jnp.log(jnp.asarray([0.5, 0.5]))
    advantages = jnp.asarray([1.0, -1.0])
    old_values = jnp.asarray([0.0, 0.0])
    new_values = jnp.asarray([1.0, -1.0])
    targets = jnp.asarray([2.0, -2.0])
    result = ppo_gru.ppo_clipped_loss_from_predictions(
        logits=logits,
        values=new_values,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=old_values,
        normalized_advantages=advantages,
        targets=targets,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
    )

    new_log_probs = ppo_gru.categorical_log_prob(logits, actions)
    ratio = jnp.exp(new_log_probs - old_log_probs)
    expected_policy = -jnp.mean(
        jnp.minimum(
            ratio * advantages,
            jnp.clip(ratio, 0.8, 1.2) * advantages,
        )
    )
    clipped_values = old_values + jnp.clip(new_values - old_values, -0.2, 0.2)
    expected_value = jnp.mean(
        jnp.maximum((new_values - targets) ** 2, (clipped_values - targets) ** 2)
    )
    expected_entropy = jnp.mean(ppo_gru.categorical_entropy(logits))
    expected_total = expected_policy + 0.5 * expected_value - 0.01 * expected_entropy
    np.testing.assert_allclose(result.policy_loss, expected_policy)
    np.testing.assert_allclose(result.value_loss, expected_value)
    np.testing.assert_allclose(result.entropy, expected_entropy)
    np.testing.assert_allclose(result.total_loss, expected_total)
    assert float(result.clip_fraction) == pytest.approx(1.0)


@pytest.mark.unit
def test_global_gradient_clipping_bounds_the_whole_tree_norm() -> None:
    gradients = {"a": jnp.asarray([3.0, 4.0]), "b": jnp.asarray([0.0])}
    clipped, before, after = ppo_gru.clip_ppo_gru_gradients(gradients, 1.0)
    assert float(before) == pytest.approx(5.0)
    assert float(after) == pytest.approx(1.0)
    np.testing.assert_allclose(clipped["a"], jnp.asarray([0.6, 0.8]))


@pytest.mark.unit
@pytest.mark.slow
def test_tiny_deterministic_segment_update_changes_parameters_once() -> None:
    config, model, variables = _small_model()
    observations = jnp.zeros(
        (config.segment_steps, *config.observation_shape), dtype=jnp.float32
    )
    reset_before = jnp.zeros((config.segment_steps,), dtype=jnp.bool_).at[64].set(True)
    initial_carry = jnp.zeros((config.hidden_size,), dtype=jnp.float32)
    evaluation = ppo_gru.evaluate_ppo_gru_sequence(
        model, variables, initial_carry, observations, reset_before
    )
    actions = jnp.arange(config.segment_steps, dtype=jnp.int32) % 4
    old_log_probs = ppo_gru.categorical_log_prob(evaluation.logits, actions)
    advantages = jnp.tile(
        jnp.asarray([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32),
        config.segment_steps // 4,
    )
    target_offsets = jnp.tile(
        jnp.asarray([0.5, -0.25, 0.25, -0.5], dtype=jnp.float32),
        config.segment_steps // 4,
    )
    batch = ppo_gru.PPOGRULossBatch(
        initial_carry=initial_carry,
        observations=observations,
        reset_before=reset_before,
        actions=actions,
        old_log_probs=old_log_probs,
        old_values=evaluation.values,
        advantages=advantages,
        targets=evaluation.values + target_offsets,
    )
    rng = ppo_gru.initialize_ppo_gru_rng_state(37, 37)
    state, _ = ppo_gru.initialize_ppo_gru_train_state(config, rng_state=rng)
    state = dataclasses.replace(state, variables=variables)
    state = ppo_gru.reset_ppo_gru_optimizer_for_variables(state, config)
    first = ppo_gru.ppo_gru_update(model, state, batch, config)
    second = ppo_gru.ppo_gru_update(model, state, batch, config)
    traced = jax.jit(
        lambda current: ppo_gru.ppo_gru_update_core(model, current, batch, config)
    )(state)

    assert int(first.state.optimizer_updates) == 1
    assert float(first.gradient_norm_after_clip) <= config.max_grad_norm + 1e-6
    np.testing.assert_allclose(first.loss.total_loss, second.loss.total_loss)
    np.testing.assert_allclose(first.loss.total_loss, traced.loss.total_loss)
    leaves_before = jax.tree.leaves(state.variables)
    leaves_after = jax.tree.leaves(first.state.variables)
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(leaves_before, leaves_after, strict=True)
    )

    forged_counter = dataclasses.replace(
        state, optimizer_updates=jnp.asarray(0.5, dtype=jnp.float32)
    )
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="int32"):
        ppo_gru.validate_ppo_gru_update_counter(forged_counter, config)


@pytest.mark.unit
def test_runner_gate_and_module_source_remain_honestly_blocked_and_distrax_free() -> None:
    with pytest.raises(ppo_gru.ForagerMatchedV3PPOGRUError, match="runner is not ready"):
        ppo_gru.assert_matched_v3_ppo_gru_runner_ready()
    source = Path(ppo_gru.__file__).read_text(encoding="utf-8")
    assert "import distrax" not in source
    assert "from distrax" not in source
    assert "print(" not in source
