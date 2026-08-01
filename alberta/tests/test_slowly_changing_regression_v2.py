"""Strict-contract tests for the nonpromoting SCR v2 development lane."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import alberta_framework.benchmarks.slowly_changing_regression_v2 as scr_v2_module
from alberta_framework.benchmarks.slowly_changing_regression import (
    SCRLearnerParams,
    SlowlyChangingRegressionConfig,
)
from alberta_framework.benchmarks.slowly_changing_regression_v2 import (
    LOCAL_CBP_METHOD,
    LOCAL_UPGD_METHOD,
    PUBLICATION_BP_METHOD,
    SCR_V2_ARTIFACT_SCHEMA,
    SCR_V2_METHOD_IDS,
    SCR_V2_PLAN_SCHEMA,
    SCR_V2_SHARD_RESERVATION_SCHEMA,
    SCR_V2_SHARD_SCHEMA,
    PublicationBPState,
    SCRV2ValidationError,
    _build_runtime_manifest,
    _build_source_manifest,
    build_scr_v2_run_plan,
    build_scr_v2_run_spec,
    init_publication_bp,
    merge_scr_v2_shards,
    publication_bp_predict,
    publication_bp_update,
    run_scr_v2_seed,
    run_scr_v2_shard,
    strict_scr_json_loads,
    validate_scr_v2_artifact,
    validate_scr_v2_run_plan,
    validate_scr_v2_shard,
    write_scr_v2_run_plan,
)

TINY = SlowlyChangingRegressionConfig(
    num_bits=8,
    num_flipping_bits=4,
    flip_period=50,
    target_hidden_units=20,
    num_examples=200,
)
PARAMS = SCRLearnerParams(hidden_units=5, step_size=0.01)


def _write_json(path: Path, value: object, *, immutable: bool = True) -> Path:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    if immutable:
        path.chmod(0o444)
    return path


def _contains_key(value: object, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _refresh_plan_issuance_command(plan: dict[str, Any]) -> None:
    plan["issuance_command"] = scr_v2_module._build_command_provenance(
        scr_v2_module._canonical_plan_semantic_argv(
            plan["run_spec_sha256"],
            plan["source_manifest_sha256"],
            plan["runtime_manifest_sha256"],
        ),
        invocation_origin="direct_api",
        process_argv=None,
    )


def _retarget_shard_command(payload: dict[str, Any], path: Path) -> None:
    canonical = payload["execution"]["command"]["canonical_semantic_argv"]
    canonical[-1] = str(path.absolute())
    payload["execution"]["command"]["canonical_semantic_argv_sha256"] = (
        scr_v2_module._sha256_json(canonical)
    )


def _copy_immutable(source: Path, destination: Path) -> Path:
    destination.write_bytes(source.read_bytes())
    destination.chmod(0o444)
    return destination


def _bind_shard_reservation(
    payload: dict[str, Any],
    plan_path: Path,
    shard_path: Path,
) -> Path:
    plan_raw = plan_path.read_bytes()
    method_id = payload["method_id"]
    seed_id = payload["seed_id"]
    reservation_path = scr_v2_module._shard_reservation_path(
        plan_path,
        plan_raw,
        method_id,
        seed_id,
    )
    reservation = scr_v2_module._build_shard_reservation(
        plan_path=plan_path,
        plan_raw=plan_raw,
        method_id=method_id,
        seed_id=seed_id,
        output=shard_path,
        prescribed_command=payload["execution"]["command"],
    )
    reservation_raw = scr_v2_module._canonical_json_bytes(reservation)
    reservation_path.parent.mkdir(parents=True, exist_ok=True)
    reservation_path.write_bytes(reservation_raw)
    reservation_path.chmod(0o444)
    payload["reservation_binding"] = {
        "path": str(reservation_path.absolute()),
        "byte_size": len(reservation_raw),
        "sha256": hashlib.sha256(reservation_raw).hexdigest(),
    }
    return reservation_path


def _copy_bound_bundle(
    bundle: dict[str, Any],
    root: Path,
    prefix: str,
) -> tuple[Path, tuple[Path, ...]]:
    bundle_root = root / f"{prefix}-bundle"
    bundle_root.mkdir()
    plan_path = _copy_immutable(
        Path(bundle["plan"]),
        bundle_root / "plan.json",
    )
    shard_paths: list[Path] = []
    for index, original in enumerate(bundle["shards"]):
        payload = json.loads(Path(original).read_text())
        shard_path = bundle_root / f"shard-{index}.json"
        _retarget_shard_command(payload, shard_path)
        _bind_shard_reservation(payload, plan_path, shard_path)
        _write_json(shard_path, payload)
        shard_paths.append(shard_path)
    return plan_path, tuple(shard_paths)


def _capture_stable_source_manifest() -> dict[str, Any]:
    for _ in range(32):
        try:
            return _build_source_manifest()
        except SCRV2ValidationError:
            continue
    pytest.fail("could not capture one internally stable source manifest for unit tests")


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    # Production validation always rebuilds the live fail-closed closure.  The
    # unit fixture freezes one internally coherent capture so unrelated agents
    # editing transitively imported modules cannot invalidate a bundle halfway
    # through this test module.
    captured_source = _capture_stable_source_manifest()
    captured_runtime = _build_runtime_manifest()
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        scr_v2_module,
        "_build_source_manifest",
        lambda: copy.deepcopy(captured_source),
    )
    patcher.setattr(
        scr_v2_module,
        "_build_runtime_manifest",
        lambda: copy.deepcopy(captured_runtime),
    )
    root = tmp_path_factory.mktemp("scr_v2_bundle")
    try:
        plan = write_scr_v2_run_plan(
            root / "run_plan.v2.json",
            TINY,
            PARAMS,
            SCR_V2_METHOD_IDS,
            (0,),
            50,
            created_unix=10,
        )
        shards = tuple(
            run_scr_v2_shard(plan, method, 0)
            for method in SCR_V2_METHOD_IDS
        )
        artifact = merge_scr_v2_shards(
            plan,
            shards,
            root / "artifact.v2.json",
            created_unix=20,
        )
        yield {"root": root, "plan": plan, "shards": shards, "artifact": artifact}
    finally:
        patcher.undo()


class TestPublicationBP:
    def test_kaiming_uniform_shapes_bounds_and_zero_biases(self) -> None:
        state = init_publication_bp(20, 5, jr.key(7))
        assert state.hidden_weights.shape == (5, 20)
        assert state.output_weights.shape == (5,)
        assert float(jnp.max(jnp.abs(state.hidden_weights))) <= math.sqrt(6.0 / 20)
        assert float(jnp.max(jnp.abs(state.output_weights))) <= math.sqrt(3.0 / 5)
        assert jnp.array_equal(state.hidden_bias, jnp.zeros(5))
        assert float(state.output_bias) == 0.0

    def test_update_matches_independent_true_mse_gradient_fixture(self) -> None:
        state = PublicationBPState(  # type: ignore[call-arg]
            hidden_weights=jnp.asarray([[0.5, -0.2], [0.1, 0.3]], dtype=jnp.float32),
            hidden_bias=jnp.asarray([0.05, -0.1], dtype=jnp.float32),
            output_weights=jnp.asarray([0.4, -0.7], dtype=jnp.float32),
            output_bias=jnp.asarray(0.2, dtype=jnp.float32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        observation = jnp.asarray([0.6, -0.4], dtype=jnp.float32)
        target = jnp.asarray(0.9, dtype=jnp.float32)
        alpha = 0.03

        def loss_fn(
            hidden_weights: jax.Array,
            hidden_bias: jax.Array,
            output_weights: jax.Array,
            output_bias: jax.Array,
        ) -> jax.Array:
            hidden = jax.nn.relu(hidden_weights @ observation + hidden_bias)
            prediction = jnp.dot(output_weights, hidden) + output_bias
            return jnp.mean((prediction - target) ** 2)

        gradients = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(
            state.hidden_weights,
            state.hidden_bias,
            state.output_weights,
            state.output_bias,
        )
        updated, squared_error = publication_bp_update(state, observation, target, alpha)
        expected = tuple(
            parameter - alpha * gradient
            for parameter, gradient in zip(
                (
                    state.hidden_weights,
                    state.hidden_bias,
                    state.output_weights,
                    state.output_bias,
                ),
                gradients,
                strict=True,
            )
        )
        for actual, wanted in zip(
            (
                updated.hidden_weights,
                updated.hidden_bias,
                updated.output_weights,
                updated.output_bias,
            ),
            expected,
            strict=True,
        ):
            assert jnp.allclose(actual, wanted, rtol=1e-6, atol=1e-7)
        residual = publication_bp_predict(state, observation) - target
        assert jnp.allclose(squared_error, residual**2)
        # Frozen scalar-output PyTorch MSE/SGD fixture for the same tensors.
        assert jnp.allclose(
            updated.hidden_weights,
            jnp.asarray([[0.5076032, -0.2050688], [0.1, 0.3]], dtype=jnp.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            updated.hidden_bias,
            jnp.asarray([0.062672, -0.1], dtype=jnp.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        assert jnp.allclose(
            updated.output_weights,
            jnp.asarray([0.4136224, -0.7], dtype=jnp.float32),
            rtol=1e-6,
            atol=1e-7,
        )
        assert float(updated.output_bias) == pytest.approx(0.23168, rel=1e-6)

    def test_true_mse_output_bias_step_has_factor_two(self) -> None:
        state = PublicationBPState(  # type: ignore[call-arg]
            hidden_weights=jnp.ones((1, 1), dtype=jnp.float32),
            hidden_bias=jnp.zeros((1,), dtype=jnp.float32),
            output_weights=jnp.ones((1,), dtype=jnp.float32),
            output_bias=jnp.asarray(0.0, dtype=jnp.float32),
            step_count=jnp.asarray(0, dtype=jnp.int32),
        )
        updated, _ = publication_bp_update(
            state,
            jnp.asarray([1.0], dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            0.1,
        )
        assert float(updated.output_bias) == pytest.approx(-0.2)

    def test_selected_bp_seed_runner_is_deterministic_and_finite(self) -> None:
        first = run_scr_v2_seed(PUBLICATION_BP_METHOD, TINY, PARAMS, 4, 50)
        second = run_scr_v2_seed(PUBLICATION_BP_METHOD, TINY, PARAMS, 4, 50)
        assert jnp.array_equal(first, second)
        assert first.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(first)))


class TestClosedRunSpec:
    def test_full_defaults_match_only_scoped_selected_arm(self) -> None:
        spec = build_scr_v2_run_spec(
            SlowlyChangingRegressionConfig(),
            SCRLearnerParams(),
            SCR_V2_METHOD_IDS,
            (0, 1),
            40_000,
        )
        assert spec["selected_configuration_match"] == {
            "scope": "nature_task_shape_and_pinned_source_relu_sgd_0p01_arm_only",
            "task_shape": "match",
            "selected_publication_bp_arm": "match",
        }
        assert spec["planned_shard_count"] == 6
        assert spec["data_contract"] == {
            "kind": "deterministic_online_synthetic_stream",
            "external_dataset_required": False,
            "generator": "scr_example",
            "environment_identity_binds": [
                "task_configuration",
                "target_network",
                "slow_bit_schedule",
                "fast_bit_prng_key",
            ],
            "same_seed_environment_shared_across_methods": True,
        }
        assert [item["id"] for item in spec["deviations"]] == [
            "article_vs_pinned_source_scale",
            "implementation_and_rng",
            "data_materialization_and_reuse",
            "slow_bit_transition_semantics",
            "target_affine_bias_encoding",
            "publication_sweep_scope",
            "generated_horizon_boundary",
            "extension_implementations",
            "numeric_execution",
        ]

    def test_tiny_config_is_derived_as_mismatch(self) -> None:
        spec = build_scr_v2_run_spec(TINY, PARAMS, SCR_V2_METHOD_IDS, (0,), 50)
        assert spec["selected_configuration_match"]["task_shape"] == "mismatch"
        assert spec["selected_configuration_match"]["selected_publication_bp_arm"] == "match"

    def test_methods_have_distinct_closed_roles(self) -> None:
        spec = build_scr_v2_run_spec(TINY, PARAMS, SCR_V2_METHOD_IDS, (0,), 50)
        assert spec["methods"] == [
            {
                "method_id": PUBLICATION_BP_METHOD,
                "role": "selected_publication_backprop_comparator_arm",
            },
            {
                "method_id": LOCAL_CBP_METHOD,
                "role": "local_continual_backprop_extension",
            },
            {
                "method_id": LOCAL_UPGD_METHOD,
                "role": "local_upgd_extension",
            },
        ]

    @pytest.mark.parametrize("bad_seeds", [(True,), (0, 0), (1, 0), (-1,)])
    def test_seed_schedule_rejects_bool_duplicate_unsorted_or_negative(
        self, bad_seeds: tuple[int, ...]
    ) -> None:
        with pytest.raises(SCRV2ValidationError):
            build_scr_v2_run_spec(TINY, PARAMS, SCR_V2_METHOD_IDS, bad_seeds, 50)

    def test_float_fields_and_int32_loop_bounds_are_strict(self) -> None:
        integer_beta = SlowlyChangingRegressionConfig(
            num_bits=8,
            num_flipping_bits=4,
            flip_period=50,
            target_hidden_units=20,
            ltu_beta=1,  # type: ignore[arg-type]
            num_examples=200,
        )
        with pytest.raises(SCRV2ValidationError, match="ltu_beta"):
            build_scr_v2_run_spec(integer_beta, PARAMS, SCR_V2_METHOD_IDS, (0,), 50)

        integer_step = SCRLearnerParams(step_size=1)  # type: ignore[arg-type]
        with pytest.raises(SCRV2ValidationError, match="step_size"):
            build_scr_v2_run_spec(TINY, integer_step, SCR_V2_METHOD_IDS, (0,), 50)

        too_long = SlowlyChangingRegressionConfig(
            num_bits=8,
            num_flipping_bits=4,
            flip_period=50,
            target_hidden_units=20,
            num_examples=2**31,
        )
        with pytest.raises(SCRV2ValidationError, match="int32"):
            build_scr_v2_run_spec(too_long, PARAMS, SCR_V2_METHOD_IDS, (0,), 50)

    def test_run_plan_recursively_omits_protocol_exact_marker(self) -> None:
        plan = build_scr_v2_run_plan(
            TINY,
            PARAMS,
            SCR_V2_METHOD_IDS,
            (0,),
            50,
            created_unix=1,
        )
        assert not _contains_key(plan, "is_protocol_exact")
        assert plan["scientific_promotion_allowed"] is False
        assert (
            plan["execution_envelope"]["kind"]
            == "self_issued_development_manifest_without_external_chronology"
        )
        assert plan["execution_envelope"]["external_chronology_attestation_present"] is False
        assert (
            plan["execution_envelope"]["timestamp_semantics"]
            == "self_reported_diagnostic_only_not_external_chronology"
        )
        assert plan["issuance_command"]["invocation_origin"] == "direct_api"
        assert plan["issuance_command"]["self_reported_process_argv"] is None
        assert plan["issuance_command"]["canonical_semantic_argv"][0] == "plan"

    def test_source_manifest_covers_static_transitive_local_imports(self) -> None:
        manifest = _capture_stable_source_manifest()
        assert manifest["scope"] == "static_transitive_local_python_imports_plus_lockfiles"
        paths = {entry["path"] for entry in manifest["files"]}
        assert "alberta_framework/core/baseline_optimizers.py" in paths
        assert "pyproject.toml" in paths
        assert "uv.lock" in paths

    def test_runtime_manifest_binds_dependency_and_device_details(self) -> None:
        runtime = _build_runtime_manifest()
        assert runtime["chex"]
        assert runtime["jaxtyping"]
        assert runtime["jax_devices"]
        assert runtime["python_executable"]["byte_size"] > 0
        assert len(runtime["python_executable"]["sha256"]) == 64
        assert set(runtime["distribution_content"]) == {
            "absl-py",
            "aiofiles",
            "chex",
            "cloudpickle",
            "etils",
            "humanize",
            "jax",
            "jaxlib",
            "ml-dtypes",
            "msgpack",
            "numpy",
            "opt-einsum",
            "orbax-checkpoint",
            "prometheus-client",
            "protobuf",
            "psutil",
            "pygments",
            "pyyaml",
            "scipy",
            "simplejson",
            "tensorstore",
            "toolz",
            "typing-extensions",
            "uvloop",
            "wadler-lindig",
            "jaxtyping",
        }
        assert runtime["distribution_content"]["jax"]["status"] == "content_hashed"
        assert (
            runtime["distribution_content_scope"]
            == "explicit_clean_import_observed_plus_required_dependency_set"
        )
        assert runtime["unbound_runtime_scope"] == [
            "system_shared_libraries_loaded_by_python_or_extension_modules",
            "device_drivers_and_firmware",
            "dynamically_loaded_code_outside_distribution_file_manifests",
        ]
        assert "jax_default_matmul_precision" in runtime
        assert type(runtime["jax_config"]["jax_random_seed_offset"]) is int
        assert "jax_disable_jit" in runtime["jax_config"]

    def test_runtime_discovery_failure_is_wrapped_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_discovery() -> dict[str, Any]:
            raise RuntimeError("device discovery exploded")

        monkeypatch.setattr(scr_v2_module, "_discover_runtime_manifest", broken_discovery)
        with pytest.raises(SCRV2ValidationError, match="runtime discovery failed closed"):
            _build_runtime_manifest()

    def test_transitive_distribution_content_drift_breaks_current_binding(
        self,
        tiny_bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        changed_runtime = copy.deepcopy(plan["runtime_manifest"])
        changed_runtime["distribution_content"]["ml-dtypes"]["sha256"] = "0" * 64
        monkeypatch.setattr(
            scr_v2_module,
            "_build_runtime_manifest",
            lambda: copy.deepcopy(changed_runtime),
        )
        report = validate_scr_v2_run_plan(tiny_bundle["plan"])
        assert not report.valid
        assert "current runtime differs" in report.errors[0]


class TestStrictJSON:
    @pytest.mark.parametrize(
        "payload",
        [
            '{"a": 1, "a": 2}',
            '{"a": NaN}',
            '{"a": Infinity}',
            '{"a": -Infinity}',
        ],
    )
    def test_duplicate_keys_and_nonfinite_constants_are_rejected(self, payload: str) -> None:
        with pytest.raises(SCRV2ValidationError):
            strict_scr_json_loads(payload)

    def test_nested_finite_json_is_accepted(self) -> None:
        assert strict_scr_json_loads('{"a": [{"b": 1.5}]}') == {"a": [{"b": 1.5}]}

    def test_excessive_json_nesting_is_a_closed_validation_error(self) -> None:
        payload = "[" * 2_000 + "0" + "]" * 2_000
        with pytest.raises(SCRV2ValidationError):
            strict_scr_json_loads(payload)


class TestPlanAndShardContracts:
    def test_future_plan_timestamp_is_rejected_before_runtime_discovery(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def must_not_discover() -> dict[str, Any]:
            raise AssertionError("runtime discovery must not start for a future timestamp")

        monkeypatch.setattr(scr_v2_module, "_build_runtime_manifest", must_not_discover)
        with pytest.raises(SCRV2ValidationError, match="cannot be in the future"):
            build_scr_v2_run_plan(
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
                created_unix=int(time.time()) + 60,
            )

    def test_future_timestamp_policy_allows_only_five_seconds_of_clock_skew(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(scr_v2_module.time, "time", lambda: 1_000.0)
        assert scr_v2_module._require_not_future_unix(1_005, "timestamp") == 1_005
        with pytest.raises(SCRV2ValidationError, match="5-second clock-skew tolerance"):
            scr_v2_module._require_not_future_unix(1_006, "timestamp")

    def test_validator_rejects_fabricated_future_plan_timestamp(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        plan["created_unix"] = int(time.time()) + 60
        path = _write_json(tmp_path / "future-plan.json", plan)
        report = validate_scr_v2_run_plan(path)
        assert not report.valid
        assert "cannot be in the future" in report.errors[0]

    def test_plan_output_and_provenance_preflight_precede_source_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        occupied = tmp_path / "occupied-plan.json"
        occupied.write_text("occupied")

        def must_not_discover() -> dict[str, Any]:
            raise AssertionError("source discovery must not start after failed preflight")

        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", must_not_discover)
        with pytest.raises(FileExistsError, match="overwrite"):
            write_scr_v2_run_plan(
                occupied,
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
            )
        with pytest.raises(SCRV2ValidationError, match="invalid invocation origin"):
            write_scr_v2_run_plan(
                tmp_path / "bad-origin.json",
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
                invocation_origin="invented",
            )

    def test_plan_is_namespaced_valid_and_permanently_nonpromoting(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        report = validate_scr_v2_run_plan(tiny_bundle["plan"])
        assert plan["schema"] == SCR_V2_PLAN_SCHEMA
        assert report.valid
        assert not report.scientific_promotion_allowed

    def test_plan_write_refuses_overwrite(self, tmp_path: Path) -> None:
        path = write_scr_v2_run_plan(
            tmp_path / "plan.json",
            TINY,
            PARAMS,
            (PUBLICATION_BP_METHOD,),
            (0,),
            50,
            created_unix=1,
        )
        before = path.read_bytes()
        with pytest.raises(FileExistsError, match="overwrite"):
            write_scr_v2_run_plan(
                path,
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
                created_unix=2,
            )
        assert path.read_bytes() == before

    def test_outputs_are_published_without_write_bits(self, tmp_path: Path) -> None:
        path = write_scr_v2_run_plan(
            tmp_path / "readonly-plan.json",
            TINY,
            PARAMS,
            (PUBLICATION_BP_METHOD,),
            (0,),
            50,
            created_unix=1,
        )
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0

    def test_new_ancestor_entries_and_final_directory_are_fsynced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synced_modes: list[int] = []
        real_fsync = os.fsync

        def tracking_fsync(file_descriptor: int) -> None:
            synced_modes.append(os.fstat(file_descriptor).st_mode)
            real_fsync(file_descriptor)

        monkeypatch.setattr(os, "fsync", tracking_fsync)
        output = scr_v2_module._atomic_write_new(
            tmp_path / "new-parent" / "new-child" / "value.json",
            b"{}\n",
        )
        assert output.read_bytes() == b"{}\n"
        assert sum(stat.S_ISDIR(mode) for mode in synced_modes) >= 5

    def test_atomic_publication_rejects_temporary_name_replacement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_link = os.link

        def replacing_link(
            source: str,
            destination: str,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            assert src_dir_fd is not None
            os.unlink(source, dir_fd=src_dir_fd)
            attacker_fd = os.open(
                source,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
                dir_fd=src_dir_fd,
            )
            os.write(attacker_fd, b"attacker bytes\n")
            os.close(attacker_fd)
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(os, "link", replacing_link)
        output = tmp_path / "swapped.json"
        with pytest.raises(SCRV2ValidationError, match="descriptor-held"):
            scr_v2_module._atomic_write_new(output, b"trusted bytes\n")
        assert output.read_bytes() == b"attacker bytes\n"
        substituted_temporary_names = tuple(tmp_path.glob(".*.tmp"))
        assert len(substituted_temporary_names) == 1
        assert substituted_temporary_names[0].read_bytes() == b"attacker bytes\n"

    def test_atomic_publication_rejects_extra_hard_link(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_link = os.link
        attacker_link = tmp_path / "attacker-link"

        def adding_link(
            source: str,
            destination: str,
            *,
            src_dir_fd: int | None = None,
            dst_dir_fd: int | None = None,
            follow_symlinks: bool = True,
        ) -> None:
            real_link(
                source,
                attacker_link.name,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )
            real_link(
                source,
                destination,
                src_dir_fd=src_dir_fd,
                dst_dir_fd=dst_dir_fd,
                follow_symlinks=follow_symlinks,
            )

        monkeypatch.setattr(os, "link", adding_link)
        output = tmp_path / "extra-link.json"
        with pytest.raises(SCRV2ValidationError, match="link count"):
            scr_v2_module._atomic_write_new(output, b"trusted bytes\n")
        assert not output.exists()
        assert attacker_link.read_bytes() == b"trusted bytes\n"
        attacker_link.unlink()

    def test_atomic_publication_removes_target_when_readback_differs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "bad-readback.json"
        real_read = scr_v2_module._read_regular_bytes

        def corrupt_readback(path: Path, *, require_immutable: bool) -> bytes:
            raw = real_read(path, require_immutable=require_immutable)
            if Path(path) == output and require_immutable:
                return raw + b"corruption"
            return raw

        monkeypatch.setattr(scr_v2_module, "_read_regular_bytes", corrupt_readback)
        with pytest.raises(SCRV2ValidationError, match="output bytes differ"):
            scr_v2_module._atomic_write_new(output, b"trusted bytes\n")
        assert not output.exists()

    def test_atomic_publication_preserves_unknown_target_replaced_after_readback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        output = tmp_path / "post-read-replacement.json"
        real_read = scr_v2_module._read_regular_bytes
        replaced = False

        def replace_after_readback(path: Path, *, require_immutable: bool) -> bytes:
            nonlocal replaced
            raw = real_read(path, require_immutable=require_immutable)
            if Path(path) == output and require_immutable and not replaced:
                output.unlink()
                output.write_bytes(b"unknown concurrent publication\n")
                output.chmod(0o444)
                replaced = True
            return raw

        monkeypatch.setattr(scr_v2_module, "_read_regular_bytes", replace_after_readback)
        with pytest.raises(SCRV2ValidationError, match="changed after final readback"):
            scr_v2_module._atomic_write_new(output, b"trusted bytes\n")
        assert output.read_bytes() == b"unknown concurrent publication\n"

    def test_atomic_publication_rejects_renamed_ancestor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parent = tmp_path / "live"
        parent.mkdir()
        moved = tmp_path / "moved"
        real_link = os.link
        changed = False

        def moving_link(*args: Any, **kwargs: Any) -> None:
            nonlocal changed
            if not changed:
                parent.rename(moved)
                parent.mkdir()
                changed = True
            real_link(*args, **kwargs)

        monkeypatch.setattr(os, "link", moving_link)
        output = parent / "value.json"
        with pytest.raises(SCRV2ValidationError, match="ancestor directory changed"):
            scr_v2_module._atomic_write_new(output, b"{}\n")
        assert not output.exists()
        assert not (moved / "value.json").exists()

    def test_plan_write_rejects_final_and_ancestor_symlinks(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()

        final_target = outside / "escaped-plan.json"
        final_link = tmp_path / "final-link.json"
        final_link.symlink_to(final_target)
        with pytest.raises((OSError, SCRV2ValidationError)):
            write_scr_v2_run_plan(
                final_link,
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
                created_unix=1,
            )
        assert not final_target.exists()

        ancestor_link = tmp_path / "ancestor-link"
        ancestor_link.symlink_to(outside, target_is_directory=True)
        with pytest.raises((OSError, SCRV2ValidationError)):
            write_scr_v2_run_plan(
                ancestor_link / "escaped-plan.json",
                TINY,
                PARAMS,
                (PUBLICATION_BP_METHOD,),
                (0,),
                50,
                created_unix=1,
            )
        assert not (outside / "escaped-plan.json").exists()

    def test_plan_validation_rejects_symlink_hardlink_and_writable_copy(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        original = Path(tiny_bundle["plan"])

        symlink = tmp_path / "plan-symlink.json"
        symlink.symlink_to(original)
        assert not validate_scr_v2_run_plan(symlink).valid

        hardlink = tmp_path / "plan-hardlink.json"
        os.link(original, hardlink)
        try:
            assert not validate_scr_v2_run_plan(hardlink).valid
        finally:
            hardlink.unlink()
        assert validate_scr_v2_run_plan(original).valid

        writable = tmp_path / "plan-writable.json"
        writable.write_bytes(original.read_bytes())
        writable.chmod(0o644)
        assert not validate_scr_v2_run_plan(writable).valid

    def test_boolean_alias_cannot_replace_derived_integer(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        plan["run_spec"]["planned_seed_count"] = True
        canonical = json.dumps(
            plan["run_spec"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        plan["run_spec_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = _write_json(tmp_path / "bool-count.json", plan)
        path.chmod(0o444)
        report = validate_scr_v2_run_plan(path, verify_current_bindings=False)
        assert not report.valid
        assert "derived closed specification" in report.errors[0]

    def test_invalid_task_value_returns_report_instead_of_escaping(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        plan["run_spec"]["task"]["num_bits"] = 0
        canonical = json.dumps(
            plan["run_spec"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        plan["run_spec_sha256"] = hashlib.sha256(canonical).hexdigest()
        path = _write_json(tmp_path / "invalid-task.json", plan)
        path.chmod(0o444)
        report = validate_scr_v2_run_plan(path, verify_current_bindings=False)
        assert not report.valid
        assert "num_bits" in report.errors[0]

    def test_plan_rejects_nested_extra_and_tampered_derived_match(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        extra = copy.deepcopy(plan)
        extra["run_spec"]["task"]["unexpected"] = 1
        extra_path = _write_json(tmp_path / "extra.json", extra)
        extra_report = validate_scr_v2_run_plan(
            extra_path, verify_current_bindings=False
        )
        assert not extra_report.valid
        assert "run_spec.task keys differ" in extra_report.errors[0]

        tampered = copy.deepcopy(plan)
        tampered["run_spec"]["selected_configuration_match"]["task_shape"] = "match"
        canonical = json.dumps(
            tampered["run_spec"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        tampered["run_spec_sha256"] = hashlib.sha256(canonical).hexdigest()
        tampered_path = _write_json(tmp_path / "tampered.json", tampered)
        tampered_report = validate_scr_v2_run_plan(
            tampered_path, verify_current_bindings=False
        )
        assert not tampered_report.valid
        assert "derived closed specification" in tampered_report.errors[0]

    def test_plan_requires_one_canonical_byte_encoding(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        path = tmp_path / "compact-plan.json"
        path.write_text(json.dumps(plan, separators=(",", ":"), sort_keys=True))
        path.chmod(0o444)
        report = validate_scr_v2_run_plan(path, verify_current_bindings=False)
        assert not report.valid
        assert "canonical v2 JSON encoding" in report.errors[0]

    def test_plan_validation_final_reread_detects_late_replacement(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        path = tmp_path / "late-plan.json"
        path.write_bytes(Path(tiny_bundle["plan"]).read_bytes())
        path.chmod(0o444)
        real_reread = scr_v2_module._require_exact_reread
        replaced = False

        def replace_before_final_reread(
            reread_path: Path,
            expected: bytes,
            context: str,
        ) -> None:
            nonlocal replaced
            if context == "run plan" and not replaced:
                changed = json.loads(expected)
                changed["created_unix"] += 1
                path.chmod(0o644)
                path.write_bytes(scr_v2_module._canonical_json_bytes(changed))
                path.chmod(0o444)
                replaced = True
            real_reread(reread_path, expected, context)

        monkeypatch.setattr(
            scr_v2_module,
            "_require_exact_reread",
            replace_before_final_reread,
        )
        report = validate_scr_v2_run_plan(path)
        assert not report.valid
        assert "run plan bytes changed" in report.errors[0]

    def test_plan_validation_rechecks_current_source_after_final_reread(
        self,
        tiny_bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline = json.loads(Path(tiny_bundle["plan"]).read_text())["source_manifest"]
        final_reread_completed = False
        real_reread = scr_v2_module._require_exact_reread

        def tracking_reread(path: Path, expected: bytes, context: str) -> None:
            nonlocal final_reread_completed
            real_reread(path, expected, context)
            if context == "run plan":
                final_reread_completed = True

        def source_after_reread() -> dict[str, Any]:
            manifest = copy.deepcopy(baseline)
            if final_reread_completed:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_require_exact_reread", tracking_reread)
        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", source_after_reread)
        report = validate_scr_v2_run_plan(tiny_bundle["plan"])
        assert not report.valid
        assert "current source bytes differ" in report.errors[0]

    @pytest.mark.parametrize("validator_kind", ["plan", "shard", "artifact"])
    def test_public_validators_wrap_unexpected_exceptions(
        self,
        validator_kind: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def explode(*args: object, **kwargs: object) -> Any:
            raise RuntimeError("injected unexpected validator failure")

        if validator_kind == "artifact":
            monkeypatch.setattr(scr_v2_module, "_read_strict_json", explode)
            report = validate_scr_v2_artifact(Path("unused-artifact.json"))
        else:
            monkeypatch.setattr(scr_v2_module, "_read_validated_plan", explode)
            if validator_kind == "plan":
                report = validate_scr_v2_run_plan(Path("unused-plan.json"))
            else:
                report = validate_scr_v2_shard(
                    Path("unused-shard.json"),
                    Path("unused-plan.json"),
                )
        assert not report.valid
        assert (
            report.errors
            == ("unexpected validation failure (RuntimeError): "
                "injected unexpected validator failure",)
        )

    def test_plan_current_source_binding_fails_closed_after_digest_tamper(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        plan["source_manifest"]["files"][0]["sha256"] = "0" * 64
        canonical = json.dumps(
            plan["source_manifest"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        plan["source_manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
        _refresh_plan_issuance_command(plan)
        path = _write_json(tmp_path / "source-tamper.json", plan)
        report = validate_scr_v2_run_plan(path)
        assert not report.valid
        assert "current source bytes differ" in report.errors[0]

    @pytest.mark.parametrize(
        "runtime_member",
        ["python_executable", "distribution_content"],
    )
    def test_plan_current_runtime_byte_binding_fails_closed_after_digest_tamper(
        self,
        runtime_member: str,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        if runtime_member == "python_executable":
            plan["runtime_manifest"]["python_executable"]["sha256"] = "0" * 64
        else:
            plan["runtime_manifest"]["distribution_content"]["jax"]["sha256"] = "0" * 64
        plan["runtime_manifest_sha256"] = scr_v2_module._sha256_json(
            plan["runtime_manifest"]
        )
        _refresh_plan_issuance_command(plan)
        path = _write_json(tmp_path / f"runtime-{runtime_member}-tamper.json", plan)
        report = validate_scr_v2_run_plan(path)
        assert not report.valid
        assert "current runtime differs" in report.errors[0]

    def test_derived_plan_command_rejects_digest_identity_tampering(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        canonical = plan["issuance_command"]["canonical_semantic_argv"]
        canonical[2] = "0" * 64
        plan["issuance_command"]["canonical_semantic_argv_sha256"] = (
            scr_v2_module._sha256_json(canonical)
        )
        path = _write_json(tmp_path / "plan-command-tamper.json", plan)
        report = validate_scr_v2_run_plan(path, verify_current_bindings=False)
        assert not report.valid
        assert "derived command identity" in report.errors[0]

    def test_each_shard_has_one_scalar_method_seed_identity(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        for path, method in zip(tiny_bundle["shards"], SCR_V2_METHOD_IDS, strict=True):
            shard = json.loads(Path(path).read_text())
            assert shard["schema"] == SCR_V2_SHARD_SCHEMA
            assert shard["method_id"] == method
            assert shard["seed_id"] == 0
            assert "seeds" not in shard
            assert shard["execution"]["command"]["invocation_origin"] == "direct_api"
            assert shard["execution"]["command"]["self_reported_process_argv"] is None
            report = validate_scr_v2_shard(path, tiny_bundle["plan"])
            assert report.valid
            assert report.structurally_valid
            assert report.computational_replay_performed
            assert not report.scientific_promotion_allowed

    def test_shard_validation_rejects_symlink_hardlink_and_writable_copy(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        original = Path(tiny_bundle["shards"][0])

        symlink = tmp_path / "shard-symlink.json"
        symlink.symlink_to(original)
        assert not validate_scr_v2_shard(symlink, tiny_bundle["plan"]).valid

        hardlink = tmp_path / "shard-hardlink.json"
        os.link(original, hardlink)
        try:
            assert not validate_scr_v2_shard(hardlink, tiny_bundle["plan"]).valid
        finally:
            hardlink.unlink()

        writable = tmp_path / "shard-writable.json"
        writable.write_bytes(original.read_bytes())
        writable.chmod(0o644)
        assert not validate_scr_v2_shard(writable, tiny_bundle["plan"]).valid

    def test_one_bin_shard_rejects_boolean_num_bins(self, tmp_path: Path) -> None:
        one_bin = SlowlyChangingRegressionConfig(
            num_bits=8,
            num_flipping_bits=4,
            flip_period=50,
            target_hidden_units=20,
            num_examples=50,
        )
        plan = write_scr_v2_run_plan(
            tmp_path / "one-bin-plan.json",
            one_bin,
            PARAMS,
            (PUBLICATION_BP_METHOD,),
            (0,),
            50,
            created_unix=1,
        )
        shard = run_scr_v2_shard(plan, PUBLICATION_BP_METHOD, 0)
        payload = json.loads(shard.read_text())
        payload["measurements"]["num_bins"] = True
        tampered = _write_json(tmp_path / "bool-num-bins.json", payload)
        tampered.chmod(0o444)
        assert not validate_scr_v2_shard(tampered, plan).valid

    def test_huge_integer_returns_invalid_report_instead_of_escaping(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = json.loads(Path(tiny_bundle["shards"][0]).read_text())
        payload["execution"]["duration_seconds"] = 10**400
        path = _write_json(tmp_path / "huge-duration.json", payload)
        path.chmod(0o444)
        report = validate_scr_v2_shard(path, tiny_bundle["plan"])
        assert not report.valid

    def test_future_shard_timestamp_is_rejected(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = json.loads(Path(tiny_bundle["shards"][0]).read_text())
        path = tmp_path / "future-shard.json"
        payload["execution"]["finished_unix"] = int(time.time()) + 60
        _retarget_shard_command(payload, path)
        _write_json(path, payload)
        report = validate_scr_v2_shard(path, tiny_bundle["plan"])
        assert not report.valid
        assert "cannot be in the future" in report.errors[0]

    @pytest.mark.parametrize("method", SCR_V2_METHOD_IDS)
    def test_shard_measurements_replay_exactly_for_every_method(
        self, tiny_bundle: dict[str, Any], method: str
    ) -> None:
        index = SCR_V2_METHOD_IDS.index(method)
        report = validate_scr_v2_shard(
            tiny_bundle["shards"][index],
            tiny_bundle["plan"],
            replay_measurements=True,
        )
        assert report.valid

    def test_shard_validation_final_reread_detects_post_replay_replacement(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = _copy_immutable(
            Path(tiny_bundle["plan"]),
            tmp_path / "late-replaced-plan.json",
        )
        payload = json.loads(Path(tiny_bundle["shards"][0]).read_text())
        path = tmp_path / "late-replaced-shard.json"
        _retarget_shard_command(payload, path)
        _bind_shard_reservation(payload, plan_path, path)
        _write_json(path, payload)

        def replace_after_replay(*args: object, **kwargs: object) -> None:
            changed = copy.deepcopy(payload)
            changed["execution"]["duration_seconds"] += 1.0
            path.chmod(0o644)
            path.write_bytes(scr_v2_module._canonical_json_bytes(changed))
            path.chmod(0o444)

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            replace_after_replay,
        )
        report = validate_scr_v2_shard(path, plan_path)
        assert not report.valid
        assert "validated shard bytes changed" in report.errors[0]

    def test_structural_only_shard_report_is_explicitly_nonvalid(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        report = validate_scr_v2_shard(
            tiny_bundle["shards"][0],
            tiny_bundle["plan"],
            replay_measurements=False,
        )
        assert not report.valid
        assert report.structurally_valid
        assert not report.computational_replay_performed
        assert "exact computational replay was not performed" in report.errors[0]

    def test_derived_worker_command_rejects_identity_tampering(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        payload = json.loads(Path(tiny_bundle["shards"][0]).read_text())
        path = tmp_path / "worker-command-tamper.json"
        canonical = payload["execution"]["command"]["canonical_semantic_argv"]
        canonical[6] = "999"
        canonical[-1] = str(path.absolute())
        payload["execution"]["command"]["canonical_semantic_argv_sha256"] = (
            scr_v2_module._sha256_json(canonical)
        )
        _write_json(path, payload)
        report = validate_scr_v2_shard(path, tiny_bundle["plan"])
        assert not report.valid
        assert "derived command identity" in report.errors[0]

    def test_all_methods_bind_the_same_environment_for_a_seed(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        identities = {
            json.loads(Path(path).read_text())["environment_identity"]["sha256"]
            for path in tiny_bundle["shards"]
        }
        assert len(identities) == 1

    def test_unplanned_method_or_seed_is_rejected_before_execution(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        with pytest.raises(SCRV2ValidationError, match="not planned"):
            run_scr_v2_shard(
                tiny_bundle["plan"],
                PUBLICATION_BP_METHOD,
                999,
                tmp_path / "never.json",
            )

    def test_shard_provenance_is_prevalidated_before_plan_or_worker(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def must_not_read_plan(*args: object, **kwargs: object) -> tuple[bytes, dict[str, Any]]:
            raise AssertionError("plan/runtime discovery must not start")

        monkeypatch.setattr(scr_v2_module, "_read_validated_plan", must_not_read_plan)
        with pytest.raises(SCRV2ValidationError, match="invalid invocation origin"):
            run_scr_v2_shard(
                tiny_bundle["plan"],
                PUBLICATION_BP_METHOD,
                0,
                tmp_path / "bad-origin-shard.json",
                invocation_origin="invented",
            )
        with pytest.raises(SCRV2ValidationError, match="nonempty string array"):
            run_scr_v2_shard(
                tiny_bundle["plan"],
                PUBLICATION_BP_METHOD,
                0,
                tmp_path / "empty-argv-shard.json",
                invocation_origin="cli",
                process_argv=(),
            )

    def test_occupied_shard_path_is_rejected_before_execution(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        occupied = tmp_path / "occupied.json"
        occupied.write_text("do not replace")

        def must_not_execute(*args: object, **kwargs: object) -> jax.Array:
            raise AssertionError("shard execution must not start for an occupied destination")

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_seed", must_not_execute)
        with pytest.raises(FileExistsError, match="overwrite"):
            run_scr_v2_shard(
                tiny_bundle["plan"],
                PUBLICATION_BP_METHOD,
                0,
                occupied,
            )
        assert occupied.read_text() == "do not replace"

    def test_persistent_reservation_survives_failed_worker_and_blocks_rerun(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "failed-worker-plan.json"
        plan_path.write_bytes(Path(tiny_bundle["plan"]).read_bytes())
        plan_path.chmod(0o444)
        output = tmp_path / "failed-worker.json"

        def fail_worker(*args: object, **kwargs: object) -> jax.Array:
            raise RuntimeError("worker failed after seed consumption")

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_seed", fail_worker)
        with pytest.raises(RuntimeError, match="seed consumption"):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                output,
            )
        plan_raw = plan_path.read_bytes()
        reservation = scr_v2_module._shard_reservation_path(
            plan_path,
            plan_raw,
            PUBLICATION_BP_METHOD,
            0,
        )
        assert not output.exists()
        assert reservation.is_file()
        assert stat.S_IMODE(reservation.stat().st_mode) == 0o444
        payload = json.loads(reservation.read_text())
        assert payload["schema"] == SCR_V2_SHARD_RESERVATION_SCHEMA
        assert payload["state"] == "execution_started_development_seed_irrevocably_consumed"
        assert payload["external_chronology_attestation_present"] is False

        executed = False

        def must_not_rerun(*args: object, **kwargs: object) -> jax.Array:
            nonlocal executed
            executed = True
            return jnp.zeros((4,), dtype=jnp.float32)

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_seed", must_not_rerun)
        alternate_output = tmp_path / "alternate-output-must-not-run.json"
        with pytest.raises(FileExistsError, match="overwrite"):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                alternate_output,
            )
        assert not executed
        assert not alternate_output.exists()

    def test_distinct_plan_digests_have_distinct_reservation_namespaces(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        del tiny_bundle
        first_plan = write_scr_v2_run_plan(
            tmp_path / "first-plan.json",
            TINY,
            PARAMS,
            (PUBLICATION_BP_METHOD,),
            (0,),
            50,
            created_unix=101,
        )
        second_plan = write_scr_v2_run_plan(
            tmp_path / "second-plan.json",
            TINY,
            PARAMS,
            (PUBLICATION_BP_METHOD,),
            (0,),
            50,
            created_unix=102,
        )
        monkeypatch.setattr(
            scr_v2_module,
            "run_scr_v2_seed",
            lambda *args, **kwargs: jnp.zeros((4,), dtype=jnp.float32),
        )
        first_output = run_scr_v2_shard(
            first_plan,
            PUBLICATION_BP_METHOD,
            0,
            tmp_path / "first-output.json",
        )
        second_output = run_scr_v2_shard(
            second_plan,
            PUBLICATION_BP_METHOD,
            0,
            tmp_path / "second-output.json",
        )
        first_reservation = scr_v2_module._shard_reservation_path(
            first_plan,
            first_plan.read_bytes(),
            PUBLICATION_BP_METHOD,
            0,
        )
        second_reservation = scr_v2_module._shard_reservation_path(
            second_plan,
            second_plan.read_bytes(),
            PUBLICATION_BP_METHOD,
            0,
        )
        assert first_output.is_file() and second_output.is_file()
        assert first_reservation != second_reservation
        assert first_reservation.is_file() and second_reservation.is_file()

    def test_successful_shards_retain_bound_reservations(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        plan_path = Path(tiny_bundle["plan"])
        plan_raw = plan_path.read_bytes()
        for shard_path in tiny_bundle["shards"]:
            shard = json.loads(Path(shard_path).read_text())
            reservation_path = scr_v2_module._shard_reservation_path(
                plan_path,
                plan_raw,
                shard["method_id"],
                shard["seed_id"],
            )
            reservation = json.loads(reservation_path.read_text())
            assert reservation["method_id"] == shard["method_id"]
            assert reservation["seed_id"] == shard["seed_id"]
            assert reservation["target_locator"] == str(Path(shard_path).absolute())
            assert reservation["prescribed_command"] == shard["execution"]["command"]
            assert shard["reservation_binding"] == {
                "path": str(reservation_path.absolute()),
                "byte_size": len(reservation_path.read_bytes()),
                "sha256": hashlib.sha256(reservation_path.read_bytes()).hexdigest(),
            }

    def test_public_shard_validation_requires_the_exact_bound_reservation(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        plan_path, shard_paths = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "missing-reservation-validation",
        )
        shard = json.loads(shard_paths[0].read_text())
        reservation_path = Path(shard["reservation_binding"]["path"])
        reservation_path.unlink()
        report = validate_scr_v2_shard(shard_paths[0], plan_path)
        assert not report.valid
        assert "No such file or directory" in report.errors[0]

    def test_shard_validation_rechecks_current_source_after_reservation_reread(
        self,
        tiny_bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        baseline = json.loads(Path(tiny_bundle["plan"]).read_text())["source_manifest"]
        reservation_reread_completed = False
        real_reread = scr_v2_module._require_exact_reread

        def tracking_reread(path: Path, expected: bytes, context: str) -> None:
            nonlocal reservation_reread_completed
            real_reread(path, expected, context)
            if context == "validated shard reservation":
                reservation_reread_completed = True

        def source_after_reread() -> dict[str, Any]:
            manifest = copy.deepcopy(baseline)
            if reservation_reread_completed:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_require_exact_reread", tracking_reread)
        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", source_after_reread)
        report = validate_scr_v2_shard(
            tiny_bundle["shards"][0],
            tiny_bundle["plan"],
        )
        assert not report.valid
        assert "final shard validation after final rereads" in report.errors[0]

    def test_late_external_plan_replacement_blocks_shard_publication(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "replaceable-plan.json"
        original = Path(tiny_bundle["plan"]).read_bytes()
        plan_path.write_bytes(original)
        plan_path.chmod(0o444)
        output = tmp_path / "must-not-publish.json"

        def replacing_worker(*args: object, **kwargs: object) -> jax.Array:
            payload = json.loads(original)
            payload["created_unix"] += 1
            plan_path.chmod(0o644)
            plan_path.write_bytes(scr_v2_module._canonical_json_bytes(payload))
            plan_path.chmod(0o444)
            return jnp.zeros((4,), dtype=jnp.float32)

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_seed", replacing_worker)
        with pytest.raises(SCRV2ValidationError, match="external run plan bytes changed"):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                output,
            )
        assert not output.exists()
        reservation_path = scr_v2_module._shard_reservation_path(
            plan_path,
            original,
            PUBLICATION_BP_METHOD,
            0,
        )
        assert reservation_path.is_file()

    def test_reservation_replacement_during_worker_blocks_shard_publication(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = _copy_immutable(
            Path(tiny_bundle["plan"]),
            tmp_path / "reservation-race-plan.json",
        )
        output = tmp_path / "reservation-race.json"
        reservation_path = scr_v2_module._shard_reservation_path(
            plan_path,
            plan_path.read_bytes(),
            PUBLICATION_BP_METHOD,
            0,
        )

        def replacing_worker(*args: object, **kwargs: object) -> jax.Array:
            reservation = json.loads(reservation_path.read_text())
            reservation["state"] = "attacker_replaced_marker"
            reservation_path.chmod(0o644)
            reservation_path.write_bytes(
                scr_v2_module._canonical_json_bytes(reservation)
            )
            reservation_path.chmod(0o444)
            return jnp.zeros((4,), dtype=jnp.float32)

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_seed", replacing_worker)
        with pytest.raises(
            SCRV2ValidationError,
            match="persistent shard reservation bytes changed",
        ):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                output,
            )
        assert not output.exists()

    def test_source_drift_during_shard_does_not_publish_poisoned_output(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = _copy_immutable(
            Path(tiny_bundle["plan"]),
            tmp_path / "source-drift-plan.json",
        )
        baseline = json.loads(Path(tiny_bundle["plan"]).read_text())["source_manifest"]
        calls = 0

        def drifting_manifest() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            manifest = copy.deepcopy(baseline)
            if calls >= 2:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", drifting_manifest)
        output = tmp_path / "must-not-exist.json"
        with pytest.raises(SCRV2ValidationError, match="while the shard was executing"):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                output,
            )
        assert not output.exists()

    def test_source_drift_after_payload_construction_blocks_final_shard_publish(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = _copy_immutable(
            Path(tiny_bundle["plan"]),
            tmp_path / "late-source-drift-plan.json",
        )
        baseline = json.loads(Path(tiny_bundle["plan"]).read_text())["source_manifest"]
        reservation_reread_completed = False
        real_reread = scr_v2_module._require_exact_reread

        def tracking_reread(path: Path, expected: bytes, context: str) -> None:
            nonlocal reservation_reread_completed
            real_reread(path, expected, context)
            if context == "persistent shard reservation":
                reservation_reread_completed = True

        def late_drift() -> dict[str, Any]:
            manifest = copy.deepcopy(baseline)
            if reservation_reread_completed:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_require_exact_reread", tracking_reread)
        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", late_drift)
        output = tmp_path / "late-drift-must-not-exist.json"
        with pytest.raises(SCRV2ValidationError, match="final shard publication"):
            run_scr_v2_shard(
                plan_path,
                PUBLICATION_BP_METHOD,
                0,
                output,
            )
        assert reservation_reread_completed
        assert not output.exists()

    def test_v1_shard_is_rejected(self, tiny_bundle: dict[str, Any], tmp_path: Path) -> None:
        path = _write_json(
            tmp_path / "v1.json",
            {"schema": "slowly_changing_regression.replication.v1"},
        )
        assert not validate_scr_v2_shard(path, tiny_bundle["plan"]).valid


class TestArtifactContract:
    def test_artifact_reconstructs_and_remains_nonpromoting(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        report = validate_scr_v2_artifact(tiny_bundle["artifact"])
        assert artifact["schema"] == SCR_V2_ARTIFACT_SCHEMA
        assert report.valid
        assert report.structurally_valid
        assert report.computational_replay_performed
        assert not report.scientific_promotion_allowed
        assert artifact["scientific_promotion_allowed"] is False
        assert artifact["computational_integrity"] == {
            "kind": "exact_deterministic_replay",
            "merge_exact_replay_performed": True,
            "replay_scope": "all_planned_method_seed_shards",
            "replayed_shard_count": 3,
            "trusted_external_receipt": False,
        }
        assert artifact["merge_command"]["invocation_origin"] == "direct_api"
        assert artifact["external_plan"]["path"] == str(Path(tiny_bundle["plan"]).absolute())
        assert artifact["external_plan"]["sha256"] == artifact["plan_binding"]["sha256"]
        assert artifact["interpretation"]["post_hoc_thresholds_used"] is False
        assert artifact["interpretation"]["sota_claim_allowed"] is False
        assert artifact["observed_coverage"] == {
            "method_ids": list(SCR_V2_METHOD_IDS),
            "seed_ids_by_method": {method: [0] for method in SCR_V2_METHOD_IDS},
            "paired_seed_ids": [0],
            "shard_count": 3,
        }

    def test_structural_only_artifact_report_is_explicitly_nonvalid(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        report = validate_scr_v2_artifact(
            tiny_bundle["artifact"], replay_measurements=False
        )
        assert not report.valid
        assert report.structurally_valid
        assert not report.computational_replay_performed
        assert "exact computational replay was not performed" in report.errors[0]

    def test_disabling_current_bindings_cannot_return_valid(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        plan_report = validate_scr_v2_run_plan(
            tiny_bundle["plan"], verify_current_bindings=False
        )
        assert not plan_report.valid
        assert plan_report.structurally_valid

        shard_report = validate_scr_v2_shard(
            tiny_bundle["shards"][0],
            tiny_bundle["plan"],
            verify_current_bindings=False,
        )
        assert not shard_report.valid
        assert shard_report.computational_replay_performed

        artifact_report = validate_scr_v2_artifact(
            tiny_bundle["artifact"], verify_current_bindings=False
        )
        assert not artifact_report.valid
        assert artifact_report.computational_replay_performed
        assert "bindings were not verified" in artifact_report.errors[0]

    def test_artifact_validation_rejects_symlink_hardlink_and_writable_copy(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        original = Path(tiny_bundle["artifact"])

        symlink = tmp_path / "artifact-symlink.json"
        symlink.symlink_to(original)
        assert not validate_scr_v2_artifact(symlink).valid

        hardlink = tmp_path / "artifact-hardlink.json"
        os.link(original, hardlink)
        try:
            assert not validate_scr_v2_artifact(hardlink).valid
        finally:
            hardlink.unlink()

        writable = tmp_path / "artifact-writable.json"
        writable.write_bytes(original.read_bytes())
        writable.chmod(0o644)
        assert not validate_scr_v2_artifact(writable).valid

    def test_artifact_rejects_referenced_shard_hardlink(
        self, tiny_bundle: dict[str, Any], tmp_path: Path
    ) -> None:
        shard = Path(tiny_bundle["shards"][0])
        hardlink = tmp_path / "referenced-shard-hardlink.json"
        os.link(shard, hardlink)
        try:
            report = validate_scr_v2_artifact(tiny_bundle["artifact"])
            assert not report.valid
            assert "exactly one hard link" in report.errors[0]
        finally:
            hardlink.unlink()

    def test_artifact_rejects_referenced_shard_symlink(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        shard = Path(tiny_bundle["shards"][0])
        backup = shard.with_suffix(".immutable-backup")
        shard.rename(backup)
        shard.symlink_to(backup.name)
        try:
            assert not validate_scr_v2_artifact(tiny_bundle["artifact"]).valid
        finally:
            shard.unlink()
            backup.rename(shard)

    def test_result_contains_descriptions_not_pass_fail_checks(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        assert len(artifact["results"]) == 3
        for result in artifact["results"]:
            summary = result["descriptive_summary"]
            assert set(summary) == {
                "bin_mean",
                "bin_population_std",
                "bin_population_std_over_sqrt_seed_count",
                "first_bin_mean",
                "last_bin_mean",
                "last_over_first",
                "last_over_first_defined",
                "whole_run_mean",
            }
            assert summary["bin_population_std_over_sqrt_seed_count"] == [0.0] * 4
        assert not _contains_key(artifact, "bin_standard_error")
        assert not _contains_key(artifact, "all_pass")
        assert not _contains_key(artifact, "threshold")

    @pytest.mark.parametrize("field", ["sha256", "byte_size", "environment_sha256"])
    def test_manifest_tampering_fails_closed(self, tiny_bundle: dict[str, Any], field: str) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        if field == "byte_size":
            artifact["shard_manifest"][0][field] += 1
        else:
            artifact["shard_manifest"][0][field] = "0" * 64
        path = _write_json(Path(tiny_bundle["root"]) / f"manifest-{field}.json", artifact)
        assert not validate_scr_v2_artifact(path).valid

    def test_result_tampering_fails_reconstruction(self, tiny_bundle: dict[str, Any]) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        artifact["results"][0]["descriptive_summary"]["whole_run_mean"] += 0.01
        path = _write_json(Path(tiny_bundle["root"]) / "result-tamper.json", artifact)
        assert not validate_scr_v2_artifact(path).valid

    def test_derived_merge_command_rejects_shard_identity_tampering(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        path = Path(tiny_bundle["root"]) / "merge-command-tamper.json"
        canonical = artifact["merge_command"]["canonical_semantic_argv"]
        canonical[4] = "0" * 64
        canonical[-1] = str(path.absolute())
        artifact["merge_command"]["canonical_semantic_argv_sha256"] = (
            scr_v2_module._sha256_json(canonical)
        )
        _write_json(path, artifact)
        report = validate_scr_v2_artifact(path)
        assert not report.valid
        assert "derived command identity" in report.errors[0]

    def test_promotion_boolean_cannot_be_enabled(self, tiny_bundle: dict[str, Any]) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        artifact["scientific_promotion_allowed"] = True
        path = _write_json(Path(tiny_bundle["root"]) / "promotion-tamper.json", artifact)
        report = validate_scr_v2_artifact(path)
        assert not report.valid
        assert not report.scientific_promotion_allowed

    def test_future_artifact_timestamp_is_rejected(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        artifact = json.loads(Path(tiny_bundle["artifact"]).read_text())
        path = Path(tiny_bundle["root"]) / "future-artifact.json"
        artifact["created_unix"] = int(time.time()) + 60
        canonical = artifact["merge_command"]["canonical_semantic_argv"]
        canonical[-1] = str(path.absolute())
        artifact["merge_command"]["canonical_semantic_argv_sha256"] = (
            scr_v2_module._sha256_json(canonical)
        )
        _write_json(path, artifact)
        report = validate_scr_v2_artifact(path)
        assert not report.valid
        assert "cannot be in the future" in report.errors[0]

    def test_duplicate_key_and_nonfinite_artifacts_fail_before_schema_validation(
        self, tmp_path: Path
    ) -> None:
        duplicate = tmp_path / "duplicate.json"
        duplicate.write_text(
            '{"schema":"alberta.slowly_changing_regression.artifact.v2",'
            '"schema":"alberta.slowly_changing_regression.artifact.v2"}'
        )
        nonfinite = tmp_path / "nonfinite.json"
        nonfinite.write_text(
            '{"schema":"alberta.slowly_changing_regression.artifact.v2","created_unix":NaN}'
        )
        duplicate.chmod(0o444)
        nonfinite.chmod(0o444)
        assert not validate_scr_v2_artifact(duplicate).valid
        assert not validate_scr_v2_artifact(nonfinite).valid

    def test_v1_artifact_is_rejected_not_retrofitted(self, tmp_path: Path) -> None:
        path = _write_json(
            tmp_path / "replication.v1.json",
            {"schema": "slowly_changing_regression.replication.v1"},
        )
        report = validate_scr_v2_artifact(path, verify_current_bindings=False)
        assert not report.valid
        assert "artifact keys differ" in report.errors[0]

    def test_merge_rejects_missing_duplicate_or_unplanned_coverage(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        shards = tiny_bundle["shards"]
        root = Path(tiny_bundle["root"])
        with pytest.raises(SCRV2ValidationError, match="count"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                shards[:-1],
                root / "missing.json",
            )
        with pytest.raises(SCRV2ValidationError, match="duplicate"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                (shards[0], shards[0], shards[2]),
                root / "duplicate.json",
            )

    def test_merge_preflights_output_provenance_timestamp_and_locators_before_replay(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        occupied = tmp_path / "occupied-artifact.json"
        occupied.write_text("occupied")
        real_read_plan = scr_v2_module._read_validated_plan

        def must_not_read_plan(*args: object, **kwargs: object) -> tuple[bytes, dict[str, Any]]:
            raise AssertionError("plan validation and replay must not start")

        monkeypatch.setattr(scr_v2_module, "_read_validated_plan", must_not_read_plan)
        with pytest.raises(FileExistsError, match="overwrite"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                tiny_bundle["shards"],
                occupied,
            )
        with pytest.raises(SCRV2ValidationError, match="invalid invocation origin"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                tiny_bundle["shards"],
                tmp_path / "bad-origin.json",
                invocation_origin="invented",
            )
        with pytest.raises(SCRV2ValidationError, match="cannot be in the future"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                tiny_bundle["shards"],
                tmp_path / "future.json",
                created_unix=int(time.time()) + 60,
            )
        monkeypatch.setattr(scr_v2_module, "_read_validated_plan", real_read_plan)
        with pytest.raises(SCRV2ValidationError, match="inside artifact directory"):
            merge_scr_v2_shards(
                tiny_bundle["plan"],
                tuple(tmp_path / f"outside-{index}.json" for index in range(3)),
                tmp_path / "inside" / "artifact.json",
            )

    def test_merge_exact_replay_rejects_fabricated_finite_curve(
        self, tiny_bundle: dict[str, Any]
    ) -> None:
        root = Path(tiny_bundle["root"])
        plan_path, copied_shards = _copy_bound_bundle(
            tiny_bundle,
            root,
            "fabricated-finite",
        )
        tampered = copied_shards[0]
        payload = json.loads(tampered.read_text())
        payload["measurements"]["bin_mean_squared_error"] = [0.0] * 4
        tampered.chmod(0o644)
        _write_json(tampered, payload)
        output = root / "fabricated-must-not-merge.json"
        with pytest.raises(SCRV2ValidationError, match="deterministic replay"):
            merge_scr_v2_shards(plan_path, copied_shards, output)
        assert not output.exists()

    def test_merge_requires_every_shards_bound_reservation_before_replay(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path, shard_paths = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "merge-missing-reservation",
        )
        first_shard = json.loads(shard_paths[0].read_text())
        Path(first_shard["reservation_binding"]["path"]).unlink()
        replay_started = False

        def must_not_replay(*args: object, **kwargs: object) -> None:
            nonlocal replay_started
            replay_started = True

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            must_not_replay,
        )
        output = tmp_path / "missing-reservation-must-not-merge.json"
        with pytest.raises(OSError):
            merge_scr_v2_shards(plan_path, shard_paths, output)
        assert not replay_started
        assert not output.exists()

    def test_merge_final_reread_detects_shard_replacement_after_all_replays(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path, copied_shards = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "copied",
        )
        replay_calls = 0

        def mutate_after_last_replay(*args: object, **kwargs: object) -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == len(copied_shards):
                path = copied_shards[0]
                changed = json.loads(path.read_text())
                changed["execution"]["duration_seconds"] += 1.0
                path.chmod(0o644)
                path.write_bytes(scr_v2_module._canonical_json_bytes(changed))
                path.chmod(0o444)

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            mutate_after_last_replay,
        )
        output = tmp_path / "late-shard-must-not-merge.json"
        with pytest.raises(SCRV2ValidationError, match="merge input shard.*bytes changed"):
            merge_scr_v2_shards(plan_path, copied_shards, output)
        assert replay_calls == len(copied_shards)
        assert not output.exists()

    def test_merge_final_reread_detects_external_plan_replacement_after_replay(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path, copied_shards = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "plan-race",
        )
        original_plan = plan_path.read_bytes()
        replay_calls = 0

        def mutate_plan_after_last_replay(*args: object, **kwargs: object) -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == len(copied_shards):
                changed = json.loads(original_plan)
                changed["created_unix"] += 1
                plan_path.chmod(0o644)
                plan_path.write_bytes(scr_v2_module._canonical_json_bytes(changed))
                plan_path.chmod(0o444)

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            mutate_plan_after_last_replay,
        )
        output = tmp_path / "late-plan-must-not-merge.json"
        with pytest.raises(SCRV2ValidationError, match="external run plan bytes changed"):
            merge_scr_v2_shards(plan_path, tuple(copied_shards), output)
        assert replay_calls == len(copied_shards)
        assert not output.exists()

    def test_late_source_drift_blocks_final_merge_publication(
        self, tiny_bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        baseline = json.loads(Path(tiny_bundle["plan"]).read_text())["source_manifest"]
        reservation_rereads = 0
        all_final_rereads_completed = False
        real_reread = scr_v2_module._require_exact_reread

        def tracking_reread(path: Path, expected: bytes, context: str) -> None:
            nonlocal reservation_rereads, all_final_rereads_completed
            real_reread(path, expected, context)
            if context.startswith("merge input shard reservation"):
                reservation_rereads += 1
                all_final_rereads_completed = reservation_rereads == len(
                    tiny_bundle["shards"]
                )

        def late_drift() -> dict[str, Any]:
            manifest = copy.deepcopy(baseline)
            if all_final_rereads_completed:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_require_exact_reread", tracking_reread)
        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", late_drift)
        output = Path(tiny_bundle["root"]) / "late-merge-must-not-exist.json"
        with pytest.raises(SCRV2ValidationError, match="final artifact publication"):
            merge_scr_v2_shards(tiny_bundle["plan"], tiny_bundle["shards"], output)
        assert all_final_rereads_completed
        assert not output.exists()

    def test_late_source_and_runtime_drift_fail_final_artifact_validation(
        self,
        tiny_bundle: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = json.loads(Path(tiny_bundle["plan"]).read_text())
        baseline_source = plan["source_manifest"]
        reservation_rereads = 0
        all_final_rereads_completed = False
        real_reread = scr_v2_module._require_exact_reread

        def tracking_reread(path: Path, expected: bytes, context: str) -> None:
            nonlocal reservation_rereads, all_final_rereads_completed
            real_reread(path, expected, context)
            if context.startswith("artifact input shard reservation"):
                reservation_rereads += 1
                all_final_rereads_completed = reservation_rereads >= len(
                    tiny_bundle["shards"]
                )

        def late_source_drift() -> dict[str, Any]:
            manifest = copy.deepcopy(baseline_source)
            if all_final_rereads_completed:
                manifest["files"][0]["sha256"] = "0" * 64
            return manifest

        monkeypatch.setattr(scr_v2_module, "_require_exact_reread", tracking_reread)
        monkeypatch.setattr(scr_v2_module, "_build_source_manifest", late_source_drift)
        source_report = validate_scr_v2_artifact(tiny_bundle["artifact"])
        assert not source_report.valid
        assert "final artifact validation" in source_report.errors[0]

        monkeypatch.setattr(
            scr_v2_module,
            "_build_source_manifest",
            lambda: copy.deepcopy(baseline_source),
        )
        baseline_runtime = plan["runtime_manifest"]

        def late_runtime_drift() -> dict[str, Any]:
            runtime = copy.deepcopy(baseline_runtime)
            if all_final_rereads_completed:
                runtime["python"] = "0.0.0-drift"
            return runtime

        monkeypatch.setattr(scr_v2_module, "_build_runtime_manifest", late_runtime_drift)
        runtime_report = validate_scr_v2_artifact(tiny_bundle["artifact"])
        assert not runtime_report.valid
        assert "final artifact validation" in runtime_report.errors[0]

    @pytest.mark.parametrize("replacement_target", ["artifact", "external_plan", "shard"])
    def test_artifact_final_rereads_detect_post_replay_replacement(
        self,
        replacement_target: str,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path, shard_paths = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "artifact-race",
        )
        artifact_path = merge_scr_v2_shards(
            plan_path,
            tuple(shard_paths),
            tmp_path / "artifact-race.json",
            created_unix=30,
        )
        replay_calls = 0

        def replace_after_last_replay(*args: object, **kwargs: object) -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls != len(shard_paths):
                return
            if replacement_target == "artifact":
                target = artifact_path
                changed = json.loads(target.read_text())
                changed["created_unix"] += 1
            elif replacement_target == "external_plan":
                target = plan_path
                changed = json.loads(target.read_text())
                changed["created_unix"] += 1
            else:
                target = shard_paths[0]
                changed = json.loads(target.read_text())
                changed["execution"]["duration_seconds"] += 1.0
            target.chmod(0o644)
            target.write_bytes(scr_v2_module._canonical_json_bytes(changed))
            target.chmod(0o444)

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            replace_after_last_replay,
        )
        report = validate_scr_v2_artifact(artifact_path)
        assert not report.valid
        assert replay_calls == len(shard_paths)
        expected = {
            "artifact": "artifact bytes changed",
            "external_plan": "external run plan bytes changed",
            "shard": "artifact input shard",
        }[replacement_target]
        assert expected in report.errors[0]

    def test_artifact_final_reread_detects_reservation_replacement_after_replay(
        self,
        tiny_bundle: dict[str, Any],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path, shard_paths = _copy_bound_bundle(
            tiny_bundle,
            tmp_path,
            "artifact-reservation-race",
        )
        artifact_path = merge_scr_v2_shards(
            plan_path,
            shard_paths,
            tmp_path / "artifact-reservation-race.json",
            created_unix=30,
        )
        first_shard = json.loads(shard_paths[0].read_text())
        reservation_path = Path(first_shard["reservation_binding"]["path"])
        replay_calls = 0

        def replace_after_last_replay(*args: object, **kwargs: object) -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == len(shard_paths):
                reservation = json.loads(reservation_path.read_text())
                reservation["state"] = "concurrently_replaced_marker"
                reservation_path.chmod(0o644)
                reservation_path.write_bytes(
                    scr_v2_module._canonical_json_bytes(reservation)
                )
                reservation_path.chmod(0o444)

        monkeypatch.setattr(
            scr_v2_module,
            "_validate_replayed_measurements",
            replace_after_last_replay,
        )
        report = validate_scr_v2_artifact(artifact_path)
        assert not report.valid
        assert replay_calls == len(shard_paths)
        assert "artifact input shard reservation" in report.errors[0]


class TestCLIContracts:
    def test_cli_wraps_unexpected_failures_as_closed_protocol_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explode(*args: object, **kwargs: object) -> Path:
            raise RuntimeError("injected CLI failure")

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_shard", explode)
        with pytest.raises(
            SCRV2ValidationError,
            match=r"SCR v2 command failed closed \(RuntimeError\): injected CLI failure",
        ):
            scr_v2_module.main(
                (
                    "run-shard",
                    "--plan",
                    str(tmp_path / "plan.json"),
                    "--method",
                    PUBLICATION_BP_METHOD,
                    "--seed-id",
                    "0",
                    "--output",
                    str(tmp_path / "shard.json"),
                )
            )

    def test_plan_and_shard_cli_label_raw_argv_as_self_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: list[tuple[str, dict[str, Any]]] = []

        def fake_write(*args: Any, **kwargs: Any) -> Path:
            captured.append(("plan", kwargs))
            return Path(args[0])

        def fake_shard(*args: Any, **kwargs: Any) -> Path:
            captured.append(("shard", kwargs))
            return Path(args[3])

        monkeypatch.setattr(scr_v2_module, "write_scr_v2_run_plan", fake_write)
        plan_argv = (
            "plan",
            "--output",
            str(tmp_path / "plan.json"),
            "--runs",
            "1",
            "--examples",
            "2",
            "--bin-size",
            "1",
            "--flip-period",
            "1",
            "--num-bits",
            "2",
            "--num-flipping-bits",
            "1",
            "--target-hidden-units",
            "2",
            "--hidden-units",
            "1",
            "--methods",
            "bp",
        )
        scr_v2_module.main(plan_argv)
        assert captured[-1][1]["invocation_origin"] == "cli"
        assert captured[-1][1]["process_argv"] == plan_argv

        monkeypatch.setattr(scr_v2_module, "run_scr_v2_shard", fake_shard)
        shard_argv = (
            "run-shard",
            "--plan",
            str(tmp_path / "plan.json"),
            "--method",
            PUBLICATION_BP_METHOD,
            "--seed-id",
            "0",
            "--output",
            str(tmp_path / "custom.json"),
        )
        scr_v2_module.main(shard_argv)
        assert captured[-1][1]["invocation_origin"] == "cli"
        assert captured[-1][1]["process_argv"] == shard_argv

    def test_merge_cli_discovers_custom_json_and_accepts_explicit_shards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_paths: list[tuple[Path, ...]] = []
        captured_kwargs: list[dict[str, Any]] = []

        def fake_merge(*args: Any, **kwargs: Any) -> Path:
            captured_paths.append(tuple(args[1]))
            captured_kwargs.append(kwargs)
            return Path(args[2])

        monkeypatch.setattr(scr_v2_module, "merge_scr_v2_shards", fake_merge)
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        custom = shards_dir / "custom-output.json"
        canonical = shards_dir / "seed-0000.json"
        custom.write_text("{}")
        canonical.write_text("{}")
        directory_argv = (
            "merge",
            "--plan",
            str(tmp_path / "plan.json"),
            "--shards-dir",
            str(shards_dir),
            "--output",
            str(tmp_path / "artifact.json"),
        )
        scr_v2_module.main(directory_argv)
        assert captured_paths[-1] == tuple(sorted((custom, canonical)))
        assert captured_kwargs[-1]["invocation_origin"] == "cli"
        assert captured_kwargs[-1]["process_argv"] == directory_argv

        explicit_argv = (
            "merge",
            "--plan",
            str(tmp_path / "plan.json"),
            "--shard",
            str(custom),
            "--shard",
            str(canonical),
            "--output",
            str(tmp_path / "explicit-artifact.json"),
        )
        scr_v2_module.main(explicit_argv)
        assert captured_paths[-1] == (custom, canonical)
        assert captured_kwargs[-1]["process_argv"] == explicit_argv

    def test_structural_only_cli_cannot_exit_as_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replay_arguments: list[bool] = []

        def fake_validate(path: Path, *, replay_measurements: bool) -> Any:
            replay_arguments.append(replay_measurements)
            if replay_measurements:
                return scr_v2_module.SCRV2ValidationReport(
                    True,
                    False,
                    (),
                    structurally_valid=True,
                    computational_replay_performed=True,
                )
            return scr_v2_module.SCRV2ValidationReport(
                False,
                False,
                ("structural-only is nonvalid",),
                structurally_valid=True,
                computational_replay_performed=False,
            )

        monkeypatch.setattr(scr_v2_module, "validate_scr_v2_artifact", fake_validate)
        with pytest.raises(SCRV2ValidationError, match="structural-only is nonvalid"):
            scr_v2_module.main(
                (
                    "validate",
                    "--artifact",
                    str(tmp_path / "artifact.json"),
                    "--structural-only",
                )
            )
        result = scr_v2_module.main(
            ("validate", "--artifact", str(tmp_path / "artifact.json"))
        )
        assert result == tmp_path / "artifact.json"
        assert replay_arguments == [False, True]


def test_fixture_has_no_accidental_generator_leak(
    tiny_bundle: dict[str, Any],
) -> None:
    """Keep the module fixture concrete; shard validation must read each file once."""

    assert not isinstance(tiny_bundle["shards"], Iterator)
    assert all(Path(path).is_file() for path in tiny_bundle["shards"])
