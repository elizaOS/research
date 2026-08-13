# mypy: disable-error-code="arg-type,method-assign"
"""CI-cheap contracts for the paired embodied development benchmark shell.

The real four-attempt realization is intentionally outside pytest; run
``alberta-prototype-embodied-paired-development`` explicitly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework.benchmarks.prototype_embodied_paired_development as paired_module
from alberta_framework.benchmarks.prototype_embodied_paired_development import (
    ASSESSMENT_STATUS,
    NO_ACTION_SENTINEL,
    PERMITTED_STOMP_CONFIG_DIFFERENCES,
    UNAVAILABLE_REWARD_SENTINEL,
    PrototypeEmbodiedPairedDevelopmentBenchmark,
    PrototypeEmbodiedPairedDevelopmentConfig,
    PrototypeEmbodiedPairedRunState,
    _array_payload,
    _assessment_payload,
    _canonical_sha256,
    _normalized_harness_config,
    _normalized_trapezoid,
    _strict_json_equal,
    _tree_bits_equal,
    _tree_sha256,
    _validate_clean_v1_initial_causal_surface,
    _validate_fixed_v1_arm_record,
)

pytestmark = pytest.mark.unit


def test_public_module_exports_and_installed_script_entry_are_wired() -> None:
    expected = {
        "build_prototype_embodied_paired_development_benchmark",
        "main",
    }
    assert expected <= set(paired_module.__all__)
    assert all(callable(getattr(paired_module, name)) for name in expected)
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"][
        "alberta-prototype-embodied-paired-development"
    ] == "alberta_framework.benchmarks.prototype_embodied_paired_development:main"


def _nested_harness_config(*, control: bool, sixth_difference: bool = False) -> dict[str, Any]:
    stomp = {
        "base_step_size": 0.0 if control else 0.1,
        "base_avg_reward_step_size": 0.0 if control else 0.01,
        "option_step_size": 0.0 if control else 0.2,
        "option_avg_reward_step_size": 0.0 if control else 0.02,
        "option_model_step_size": 0.0 if control else 0.3,
        "epsilon_base": 0.11 if sixth_difference else 0.1,
    }
    return {
        "adapter": {
            "semantic": {
                "composition": {
                    "prototype": {
                        "oak": {"stomp": stomp},
                    }
                }
            }
        }
    }


def _run_state(*, adaptive: float, integrity: str = "integrity") -> Any:
    return PrototypeEmbodiedPairedRunState(
        attempt_index=1,
        adaptive_stomp=jnp.asarray((adaptive,), dtype=jnp.float32),
        zero_stomp_step_size_control=jnp.asarray((2.0,), dtype=jnp.float32),
        chain_heads=("a" * 64, "b" * 64),
        records_json=("{}",),
        binding_sha256="c" * 64,
        integrity_sha256=integrity,
    )


def _benchmark_shell() -> PrototypeEmbodiedPairedDevelopmentBenchmark:
    benchmark = object.__new__(PrototypeEmbodiedPairedDevelopmentBenchmark)
    benchmark.config = PrototypeEmbodiedPairedDevelopmentConfig()
    benchmark._binding = {
        "source_manifest": {"schema": "selected-source", "files": []},
        "runtime_identity": {"schema": "selected-runtime", "backend": "test"},
    }
    benchmark._binding_sha256 = "c" * 64
    benchmark.common_schedule_payload = lambda: [
        {"attempt_index": index} for index in range(4)
    ]
    return benchmark


def test_tree_bits_equal_is_dtype_shape_and_host_byte_exact() -> None:
    positive_zero = jnp.asarray(0.0, dtype=jnp.float32)
    negative_zero = jnp.asarray(-0.0, dtype=jnp.float32)
    assert not _tree_bits_equal(positive_zero, negative_zero)
    assert _tree_sha256(positive_zero) != _tree_sha256(negative_zero)
    assert not _tree_bits_equal(
        jnp.asarray((1.0,), dtype=jnp.float32),
        jnp.asarray((1,), dtype=jnp.int32),
    )
    assert not _tree_bits_equal(
        jnp.asarray((1.0,), dtype=jnp.float32),
        jnp.asarray(((1.0,),), dtype=jnp.float32),
    )
    assert _tree_bits_equal(jr.key(7), jr.key(7))
    assert not _tree_bits_equal(jr.key(7), jr.key(8))

    rbg = jr.key(7, impl="rbg")
    unsafe_rbg = jr.key(7, impl="unsafe_rbg")
    assert jnp.array_equal(jr.key_data(rbg), jr.key_data(unsafe_rbg))
    assert not _tree_bits_equal(rbg, unsafe_rbg)
    assert _tree_sha256(rbg) != _tree_sha256(unsafe_rbg)
    assert not _tree_bits_equal(rbg, jr.key_data(rbg))
    payload = _array_payload(rbg)
    assert payload["typed_prng_key"] is True
    assert payload["prng_impl"] == "rbg"
    assert payload["logical_dtype"] == "key<rbg>"


def test_strict_json_float_equality_distinguishes_signed_zero() -> None:
    assert not _strict_json_equal(0.0, -0.0)
    assert not _strict_json_equal(-0.0, 0.0)
    assert _strict_json_equal(0.0, 0.0)


def test_v1_config_is_fixed_to_four_attempts_and_one_disconnect() -> None:
    config = PrototypeEmbodiedPairedDevelopmentConfig()
    assert config.to_config()["fixed_v1_protocol"] is True
    assert config.to_config()["expected_committed_plant_transitions_per_arm"] == 3
    assert config.to_config()["expected_no_action_attempts_per_arm"] == 1
    with pytest.raises(ValueError, match="exact int 4"):
        PrototypeEmbodiedPairedDevelopmentConfig(attempts=5)
    with pytest.raises(ValueError, match=r"exactly \(1,\)"):
        PrototypeEmbodiedPairedDevelopmentConfig(bridge_disconnect_attempts=(2,))


def test_assessment_surface_makes_no_winner_adaptation_or_delight_claim() -> None:
    assert _assessment_payload() == {
        "status": ASSESSMENT_STATUS,
        "winner": None,
        "verdict": None,
        "thresholds": [],
        "performance_claimed": False,
        "adaptation_efficacy_claimed": False,
        "safety_claimed": False,
        "delight_assessed": False,
        "kondo_gate_forward_admission_intent_assessed": False,
        "kondo_sparse_actor_backward_inclusion_assessed": False,
        "historical_gradientjoy_compatibility_names_assessed": False,
        "evidence_authority": False,
        "promotion_authority": False,
        "scientific_promotion_allowed": False,
    }


def test_only_the_five_declared_stomp_config_values_normalize_away() -> None:
    assert len(PERMITTED_STOMP_CONFIG_DIFFERENCES) == 5
    adaptive = _nested_harness_config(control=False)
    control = _nested_harness_config(control=True)
    assert _normalized_harness_config(adaptive) == _normalized_harness_config(control)
    sixth = _nested_harness_config(control=True, sixth_difference=True)
    assert _normalized_harness_config(adaptive) != _normalized_harness_config(sixth)


def test_fixed_v1_outcome_and_availability_sentinels_fail_closed() -> None:
    connected = {
        "action_available": True,
        "reward_available": True,
        "no_action": False,
        "executed_action": 0,
        "reward": 0.25,
        "plant": {"committed": True},
    }
    disconnected = {
        "action_available": False,
        "reward_available": False,
        "no_action": True,
        "executed_action": NO_ACTION_SENTINEL,
        "reward": UNAVAILABLE_REWARD_SENTINEL,
        "plant": {"committed": False},
    }
    _validate_fixed_v1_arm_record(
        "adaptive_stomp", connected, expected_connected=True
    )
    _validate_fixed_v1_arm_record(
        "zero_stomp_step_size_control", disconnected, expected_connected=False
    )
    for bad in (
        {**disconnected, "reward": 1.0},
        {**disconnected, "reward": -0.0},
        {**disconnected, "reward": 0},
        {**disconnected, "executed_action": -1.0},
    ):
        with pytest.raises(ValueError, match="sentinels"):
            _validate_fixed_v1_arm_record(
                "adaptive_stomp", bad, expected_connected=False
            )


def test_clean_initial_causal_surface_and_remaining_capacity_fail_closed() -> None:
    clean: dict[str, Any] = {
        "pending_available": (False, False),
        "commit_available": (False, False),
        "plant_counts": (0, 0),
        "prototype_steps": ([0, 0], [0, 0]),
        "oak_steps": ([0, 0], [0, 0]),
        "adapter_receipt_clocks": ([0, 0], [0, 0]),
        "adapter_settled": (False, False),
        "maximum_plant_transitions": (4, 4),
        "connected_opportunities": 3,
    }
    assert _validate_clean_v1_initial_causal_surface(**clean) == (4, 4)

    for override, message in (
        ({"pending_available": (True, False)}, "pending receipts"),
        ({"commit_available": (False, True)}, "prior commit records"),
        ({"oak_steps": ([0, 1], [0, 0])}, "clocks and ledgers"),
        ({"maximum_plant_transitions": (2, 4)}, "remaining plant capacity"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate_clean_v1_initial_causal_surface(**{**clean, **override})


def test_normalized_lifetime_auc_is_exact_and_descriptive() -> None:
    rewards = [1.0, 3.0, 2.0]
    assert _normalized_trapezoid(rewards) == 2.25
    cumulative_over_attempts = [0.0, 1.0, 1.0, 4.0, 6.0]
    assert _normalized_trapezoid(cumulative_over_attempts) == 2.25
    assert _normalized_trapezoid([]) is None
    assert _normalized_trapezoid([1.0]) is None


@pytest.mark.parametrize("drift", ["source", "runtime"])
def test_live_selected_source_or_runtime_drift_refuses_state(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    benchmark = _benchmark_shell()
    monkeypatch.setattr(
        paired_module,
        "prototype_embodied_paired_source_manifest",
        lambda: copy.deepcopy(benchmark._binding["source_manifest"]),
    )
    monkeypatch.setattr(
        paired_module,
        "prototype_embodied_paired_runtime_identity",
        lambda: copy.deepcopy(benchmark._binding["runtime_identity"]),
    )
    assert benchmark._live_source_runtime_matches_binding()
    if drift == "source":
        monkeypatch.setattr(
            paired_module,
            "prototype_embodied_paired_source_manifest",
            lambda: {"schema": "selected-source", "files": ["drift"]},
        )
    else:
        monkeypatch.setattr(
            paired_module,
            "prototype_embodied_paired_runtime_identity",
            lambda: {"schema": "selected-runtime", "backend": "drift"},
        )
    assert not benchmark._live_source_runtime_matches_binding()
    assert not benchmark.validate_state(_run_state(adaptive=1.0))


def test_resealed_report_content_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_shell()
    expected_body: dict[str, Any] = {
        "schema": "synthetic-report",
        "assessment": _assessment_payload(),
        "outputs": {"writes": False, "path": None},
        "signed_zero_witness": 0.0,
    }
    expected = {**expected_body, "report_sha256": _canonical_sha256(expected_body)}
    monkeypatch.setattr(benchmark, "run", lambda: object())
    monkeypatch.setattr(benchmark, "report", lambda _state: copy.deepcopy(expected))
    assert benchmark.validate_report(expected).valid

    tampered_body = copy.deepcopy(expected_body)
    tampered_assessment = dict(tampered_body["assessment"])
    tampered_assessment["winner"] = "adaptive_stomp"
    tampered_body["assessment"] = tampered_assessment
    resealed = {
        **tampered_body,
        "report_sha256": _canonical_sha256(tampered_body),
    }
    assert not benchmark.validate_report(resealed).valid

    signed_zero_body = copy.deepcopy(expected_body)
    signed_zero_body["signed_zero_witness"] = -0.0
    signed_zero_resealed = {
        **signed_zero_body,
        "report_sha256": _canonical_sha256(signed_zero_body),
    }
    assert not benchmark.validate_report(signed_zero_resealed).valid


def test_causally_valid_looking_resealed_checkpoint_prefix_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_shell()
    reconstructed = _run_state(adaptive=1.0)
    tampered = dataclasses.replace(reconstructed, adaptive_stomp=jnp.asarray((9.0,)))
    monkeypatch.setattr(
        benchmark,
        "_state_structurally_valid",
        lambda _state: True,
    )
    monkeypatch.setattr(
        benchmark,
        "_reconstruct_prefix",
        lambda _attempt_index: reconstructed,
    )
    checkpoint = benchmark.checkpoint_payload(reconstructed)
    resealed = dict(checkpoint)
    resealed["state"] = tampered
    resealed["state_sha256"] = _canonical_sha256(benchmark._state_body(tampered))
    body = {
        key: resealed[key]
        for key in resealed
        if key not in {"state", "checkpoint_sha256"}
    }
    resealed["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="state is invalid"):
        benchmark.restore_checkpoint(resealed)


def test_resealed_arbitrary_record_and_stale_harness_prefix_fails_causal_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_shell()
    reconstructed = _run_state(adaptive=1.0)
    monkeypatch.setattr(benchmark, "_state_structurally_valid", lambda _state: True)
    monkeypatch.setattr(
        benchmark,
        "_reconstruct_prefix",
        lambda _attempt_index: reconstructed,
    )
    for records_json in (
        ('{"attempt_index":0,"arms":{}}',),
        ('{"executed_action":-1.0}',),
        ('{"reward":-0.0}',),
    ):
        attacker = benchmark._seal_state(
            dataclasses.replace(
                reconstructed,
                adaptive_stomp=jnp.asarray((9.0,), dtype=jnp.float32),
                records_json=records_json,
                chain_heads=("d" * 64, "e" * 64),
                integrity_sha256="",
            )
        )
        assert not benchmark.validate_state(attacker)


def test_resealed_signed_zero_checkpoint_state_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = _benchmark_shell()
    reconstructed = _run_state(adaptive=0.0)
    monkeypatch.setattr(benchmark, "_state_structurally_valid", lambda _state: True)
    monkeypatch.setattr(
        benchmark,
        "_reconstruct_prefix",
        lambda _attempt_index: reconstructed,
    )
    checkpoint = benchmark.checkpoint_payload(reconstructed)
    tampered = benchmark._seal_state(
        dataclasses.replace(
            reconstructed,
            adaptive_stomp=jnp.asarray((-0.0,), dtype=jnp.float32),
            integrity_sha256="",
        )
    )
    resealed = dict(checkpoint)
    resealed["state"] = tampered
    resealed["state_sha256"] = _canonical_sha256(benchmark._state_body(tampered))
    body = {
        key: resealed[key]
        for key in resealed
        if key not in {"state", "checkpoint_sha256"}
    }
    resealed["checkpoint_sha256"] = _canonical_sha256(body)
    with pytest.raises(ValueError, match="state is invalid"):
        benchmark.restore_checkpoint(resealed)


def test_installed_cli_emits_only_the_in_memory_report(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[int] = []
    expected = {
        "schema": "synthetic-cli-report",
        "assessment": _assessment_payload(),
        "outputs": {"writes": False, "path": None, "artifact_created": False},
    }

    class FakeBenchmark:
        def run(self) -> str:
            return "synthetic-state"

        def report(self, state: object) -> dict[str, Any]:
            assert state == "synthetic-state"
            return copy.deepcopy(expected)

        def validate_report(self, report: object) -> SimpleNamespace:
            assert report == expected
            return SimpleNamespace(valid=True)

    def build(*, development_key: int) -> FakeBenchmark:
        seen.append(development_key)
        return FakeBenchmark()

    monkeypatch.setattr(
        paired_module,
        "build_prototype_embodied_paired_development_benchmark",
        build,
    )
    assert paired_module.main(["--development-key", "23"]) == 0
    assert seen == [23]
    assert json.loads(capsys.readouterr().out) == expected
