from __future__ import annotations

import copy
import hashlib
import json

import pytest

from alberta_framework.benchmarks.runtime_profile import (
    ENVIRONMENT_RNG_SCHEDULE_SCHEMA_VERSION,
    ENVIRONMENT_RUNTIME_PROFILE_SCHEMA_VERSION,
    EnvironmentRngSchedule,
    EnvironmentRuntimeIdentity,
    environment_rng_schedule_sha256,
    environment_runtime_profile_sha256,
    validate_environment_runtime_identity,
    validate_environment_runtime_profile,
)


def _matched_gpu_profile() -> dict[str, object]:
    distribution_versions = {
        "continual-foragax": "0.55.0",
        "imageio-ffmpeg": "0.6.0",
        "jax": "0.9.0.1",
        "jax-cuda12-pjrt": "0.9.0.1",
        "jax-cuda12-plugin": "0.9.0.1",
        "jaxlib": "0.9.0.1",
        "numpy": "2.3.1",
        "pyexputils": "0.1.2",
        "pyfixedreps": "0.1.2",
        "replaytables": "0.1.2",
    }
    scientific_package_records = {
        name: {
            "record_sha256": f"{position + 2:x}" * 64,
            "version": version,
        }
        for position, (name, version) in enumerate(
            distribution_versions.items()
        )
    }
    cuda_wheel_library_paths = ["/opt/cuda-wheels"]
    cuda_wheel_library_profile_sha256 = hashlib.sha256(
        json.dumps(
            {
                "paths": cuda_wheel_library_paths,
                "schema_version": "alberta.cuda_wheel_library_profile.v1",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    driver_tree_sha256 = "5" * 64
    libcuda_sha256 = "6" * 64
    gpu_user_library_bundle_sha256 = hashlib.sha256(
        json.dumps(
            {
                "cuda_wheel_library_profile_sha256": (
                    cuda_wheel_library_profile_sha256
                ),
                "driver_user_library_tree_sha256": driver_tree_sha256,
                "libcuda_sha256": libcuda_sha256,
                "schema_version": "alberta.gpu_user_library_bundle.v1",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    image_id = "sha256:" + "e" * 64
    determinism_qualification = {
        "artifact_sha256": "7" * 64,
        "backend": "gpu",
        "config_sha256": "8" * 64,
        "effective_seed": 1_000_000,
        "environment_profile_sha256": "9" * 64,
        "evidence_envelope_sha256": "a" * 64,
        "executor_kind": "oci",
        "image_id": image_id,
        "member_payloads_sha256": "b" * 64,
        "repeat_count": 2,
        "rewards_sha256": "c" * 64,
        "runtime_profile_id": "foragax-current-gpu-a",
        "schema_version": "alberta.oci_determinism_qualification.v2",
        "seed_class": "open_development",
        "source_archive_sha256": "d" * 64,
        "state": "sealed_oci_two_run_exact",
        "steps": 10_000,
        "workload_identity_sha256": "e" * 64,
    }
    determinism_qualification_sha256 = hashlib.sha256(
        json.dumps(
            determinism_qualification,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return {
        "schema_version": ENVIRONMENT_RUNTIME_PROFILE_SCHEMA_VERSION,
        "bundled_executables": {
            "imageio-ffmpeg": {
                "distribution": "imageio-ffmpeg",
                "mode": 0o555,
                "record_sha256": scientific_package_records[
                    "imageio-ffmpeg"
                ]["record_sha256"],
                "relative_path": (
                    "imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
                ),
                "sha256": "f" * 64,
                "version": "0.6.0",
            }
        },
        "python": {
            "build": ["main", "Jul 1 2026"],
            "cache_tag": "cpython-312",
            "compiler": "GCC 13.3.0",
            "executable_sha256": "1" * 64,
            "hash_seed": "0",
            "implementation": "CPython",
            "platform": "Linux-6.8-x86_64-with-glibc2.39",
            "runtime_version": "3.12.10 (main, Jul 1 2026)",
            "soabi": "cpython-312-x86_64-linux-gnu",
            "version": "3.12.10",
        },
        "scientific_packages": sorted(
            [
                "continual-foragax==0.55.0",
                "imageio-ffmpeg==0.6.0",
                "jax==0.9.0.1",
                "jax-cuda12-pjrt==0.9.0.1",
                "jax-cuda12-plugin==0.9.0.1",
                "jaxlib==0.9.0.1",
                "numpy==2.3.1",
            ]
        ),
        "scientific_package_records": scientific_package_records,
        "foragax": {
            "distribution": "continual-foragax",
            "install_tree_hash_scheme": "relative-path+size+bytes-v1",
            "install_tree_sha256": "b" * 64,
            "version": "0.55.0",
        },
        "jax": {
            "backend": "gpu",
            "config": {
                "jax_compilation_cache_dir": "/run/alberta/jax-cache",
                "jax_default_matmul_precision": None,
                "jax_default_prng_impl": "threefry2x32",
                "jax_enable_compilation_cache": False,
                "jax_enable_x64": False,
                "jax_numpy_dtype_promotion": "standard",
                "jax_platforms": None,
                "jax_threefry_partitionable": True,
            },
            "devices": [
                {
                    "device_kind": "NVIDIA GeForce RTX 4090",
                    "id": 0,
                    "platform": "gpu",
                    "process_index": 0,
                }
            ],
            "version": "0.9.0.1",
        },
        "dependency_contract": {
            "cuda_wheel_library_profile_sha256": (
                cuda_wheel_library_profile_sha256
            ),
            "cuda_wheel_library_paths": cuda_wheel_library_paths,
            "dependency_lock_sha256": "c" * 64,
            "determinism_qualification": determinism_qualification,
            "determinism_qualification_sha256": (
                determinism_qualification_sha256
            ),
            "driver_user_library_hash_scheme": (
                "canonical-entry-json+mode+size+bytes-v1"
            ),
            "driver_user_library_paths": ["/opt/nvidia-driver"],
            "driver_user_library_tree_sha256": driver_tree_sha256,
            "executor_kind": "oci",
            "gpu_user_library_bundle_sha256": (
                gpu_user_library_bundle_sha256
            ),
            "image_id": image_id,
            "image_reference_digest": "sha256:" + "f" * 64,
            "libcuda_sha256": libcuda_sha256,
            "native_runtime_inventory_sha256": "3" * 64,
            "native_runtime_inventory_hash_scheme": (
                "canonical-entry-json+mode+size+bytes-v1"
            ),
            "native_runtime_inventory_root": "/opt/venv",
            "runtime_binary_sha256": "1" * 64,
            "sbom_sha256": "4" * 64,
            "scientific_runtime_class": (
                "matched_current_foragax_0_55_cuda12"
            ),
        },
        "import_shadow_contract": {
            "base_sys_path_contract": [
                {
                    "device": 1,
                    "exists": True,
                    "inode": 2,
                    "is_dir": True,
                    "path": "/usr/lib/python3.12",
                    "resolved_path": "/usr/lib/python3.12",
                    "writable": False,
                }
            ],
            "cwd": "/opt/continual-foragax-agents",
            "cwd_matches_source_root": True,
            "cwd_writable": False,
            "python_flags": {
                "dont_write_bytecode": 1,
                "isolated": 1,
                "no_user_site": 1,
                "safe_path": True,
            },
            "pythonhome": "",
            "pythonpath": "",
            "scratch_directories": {
                "CUDA_CACHE_PATH": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/cuda-cache",
                    "writable": True,
                },
                "HOME": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/home",
                    "writable": True,
                },
                "JAX_COMPILATION_CACHE_DIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/jax-cache",
                    "writable": True,
                },
                "MPLCONFIGDIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/matplotlib",
                    "writable": True,
                },
                "TMPDIR": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/tmp",
                    "writable": True,
                },
                "XDG_CACHE_HOME": {
                    "is_dir": True,
                    "is_mount": True,
                    "mode": 0o700,
                    "path": "/run/alberta/cache",
                    "writable": True,
                },
            },
            "tmp_root_writable": False,
            "tmp_src_entries": [],
            "tmp_src_exists": True,
            "tmp_src_is_mount": True,
            "tmp_src_mode": 0o555,
            "tmp_src_writable": False,
            "trusted_source_device": 1,
            "trusted_source_inode": 10,
            "trusted_source_path": "/opt/continual-foragax-agents/src",
            "trusted_source_path_in_base_sys_path": False,
            "trusted_source_path_is_dir": True,
            "trusted_source_path_writable": False,
            "workload_sys_path_contract": {
                "cwd_append_path": "/opt/continual-foragax-agents",
                "launcher_mode": "isolated-runpy-prepend-v1",
                "ordered_prefix": [
                    {
                        "empty": True,
                        "path": "/tmp/src",
                        "writable": False,
                    },
                    {
                        "empty": False,
                        "path": "/opt/continual-foragax-agents/src",
                        "writable": False,
                    },
                ],
                (
                    "trusted_source_preceded_only_by_empty_read_only_tmp_src"
                ): True,
            },
        },
        "gpu_host_runtime": {
            "device_identities": [
                {
                    "device_index": 0,
                    "device_path": "/dev/nvidia0",
                    "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                    "pci_bus_id": "0000:01:00.0",
                }
            ],
            "device_paths": [
                "/dev/nvidia0",
                "/dev/nvidiactl",
                "/dev/nvidia-uvm",
            ],
            "kernel_driver_version": "595.71.05",
            "libcuda_sha256": libcuda_sha256,
        },
        "container_environment": sorted(
            [
                "CUBLAS_WORKSPACE_CONFIG=:4096:8",
                "CUDA_CACHE_DISABLE=1",
                "CUDA_CACHE_MAXSIZE=268435456",
                "CUDA_CACHE_PATH=/run/alberta/cuda-cache",
                "CUDA_VISIBLE_DEVICES=0",
                "HOME=/run/alberta/home",
                "JAX_COMPILATION_CACHE_DIR=/run/alberta/jax-cache",
                "JAX_ENABLE_COMPILATION_CACHE=false",
                "LANG=C.UTF-8",
                "LC_ALL=C.UTF-8",
                "LD_LIBRARY_PATH=/opt/cuda-wheels:/opt/nvidia-driver",
                "MPLCONFIGDIR=/run/alberta/matplotlib",
                "NVIDIA_VISIBLE_DEVICES=void",
                "PYTHONHASHSEED=0",
                "PYTHONHOME=",
                "PYTHONNOUSERSITE=1",
                "PYTHONPATH=",
                "PYTHONDONTWRITEBYTECODE=1",
                "PYTHONUTF8=1",
                "TMPDIR=/run/alberta/tmp",
                "TZ=UTC",
                "XDG_CACHE_HOME=/run/alberta/cache",
                (
                    "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false "
                    "--xla_gpu_deterministic_ops=true"
                ),
                "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            ]
        ),
    }


def test_runtime_profile_hash_and_identity_are_structurally_verified() -> None:
    profile = _matched_gpu_profile()
    normalized = validate_environment_runtime_profile(profile)
    assert normalized == profile
    assert normalized is not profile

    profile_sha256 = environment_runtime_profile_sha256(profile)
    schedule: EnvironmentRngSchedule = "dedicated_environment_split_chain_v1"
    schedule_sha256 = environment_rng_schedule_sha256(schedule)
    identity = validate_environment_runtime_identity(
        runtime_profile_id="foragax-current-gpu-a",
        runtime_profile=profile,
        environment_runtime_profile_sha256=profile_sha256,
        environment_rng_schedule=schedule,
        environment_rng_schedule_digest=schedule_sha256,
    )
    assert identity == EnvironmentRuntimeIdentity(
        runtime_profile_id="foragax-current-gpu-a",
        environment_runtime_profile_sha256=profile_sha256,
        environment_rng_schedule=schedule,
        environment_rng_schedule_sha256=schedule_sha256,
    )
    assert (
        ENVIRONMENT_RNG_SCHEDULE_SCHEMA_VERSION
        == "alberta.environment_rng_schedule.v1"
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("python", "hash_seed"), "random"),
        (
            ("bundled_executables", "imageio-ffmpeg", "mode"),
            0o444,
        ),
        (
            ("scientific_packages", 2),
            "jax-cuda13-pjrt==0.9.0.1",
        ),
        (
            (
                "scientific_package_records",
                "jax-cuda12-plugin",
                "version",
            ),
            "0.9.1",
        ),
        (
            ("dependency_contract", "scientific_runtime_class"),
            "head_diagnostics_unpaired",
        ),
        (("import_shadow_contract", "tmp_src_writable"), True),
        (
            ("container_environment", 0),
            "CUBLAS_WORKSPACE_CONFIG=:16:8",
        ),
        (
            ("container_environment", -1),
            "XLA_FLAGS=--xla_gpu_enable_triton_gemm=false",
        ),
    ],
)
def test_runtime_profile_rejects_identity_drift(
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    profile = copy.deepcopy(_matched_gpu_profile())
    target: object = profile
    for component in path[:-1]:
        target = target[component]  # type: ignore[index]
    target[path[-1]] = replacement  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_environment_runtime_profile(profile)


def test_runtime_profile_rejects_duplicate_environment_variable() -> None:
    profile = _matched_gpu_profile()
    environment = profile["container_environment"]
    assert isinstance(environment, list)
    environment.append("PYTHONHASHSEED=1")
    environment.sort()
    with pytest.raises(ValueError, match="defines a variable twice"):
        validate_environment_runtime_profile(profile)


def test_runtime_identity_rejects_unverified_hashes_and_schedule() -> None:
    profile = _matched_gpu_profile()
    schedule = "dedicated_environment_split_chain_v1"
    with pytest.raises(ValueError, match="profile digest does not verify"):
        validate_environment_runtime_identity(
            runtime_profile_id="foragax-current-gpu-a",
            runtime_profile=profile,
            environment_runtime_profile_sha256="0" * 64,
            environment_rng_schedule=schedule,
            environment_rng_schedule_digest=environment_rng_schedule_sha256(
                schedule
            ),
        )
    with pytest.raises(ValueError, match="unsupported"):
        environment_rng_schedule_sha256("unknown_schedule")


def test_runtime_identity_rejects_profile_id_relabeling() -> None:
    profile = _matched_gpu_profile()
    schedule = "dedicated_environment_split_chain_v1"
    with pytest.raises(ValueError, match="does not match the determinism"):
        validate_environment_runtime_identity(
            runtime_profile_id="unrelated-runtime-profile",
            runtime_profile=profile,
            environment_runtime_profile_sha256=(
                environment_runtime_profile_sha256(profile)
            ),
            environment_rng_schedule=schedule,
            environment_rng_schedule_digest=environment_rng_schedule_sha256(
                schedule
            ),
        )


@pytest.mark.parametrize("device_id", [-1, 0])
def test_runtime_profile_rejects_invalid_or_duplicate_jax_device_ids(
    device_id: int,
) -> None:
    profile = _matched_gpu_profile()
    jax = profile["jax"]
    assert isinstance(jax, dict)
    devices = jax["devices"]
    assert isinstance(devices, list)
    if device_id < 0:
        device = devices[0]
        assert isinstance(device, dict)
        device["id"] = device_id
    else:
        devices.append(copy.deepcopy(devices[0]))
    with pytest.raises(ValueError, match="JAX device identity"):
        validate_environment_runtime_profile(profile)


def test_runtime_profile_relates_logical_jax_devices_to_visible_gpus() -> None:
    profile = _matched_gpu_profile()
    jax = profile["jax"]
    assert isinstance(jax, dict)
    devices = jax["devices"]
    assert isinstance(devices, list)
    device = devices[0]
    assert isinstance(device, dict)
    device["id"] = 1
    with pytest.raises(ValueError, match="do not match visible GPU"):
        validate_environment_runtime_profile(profile)


def test_gpu_qualified_runtime_profile_rejects_cpu_relabeling() -> None:
    profile = _matched_gpu_profile()
    jax = profile["jax"]
    assert isinstance(jax, dict)
    jax["backend"] = "cpu"
    devices = jax["devices"]
    assert isinstance(devices, list)
    device = devices[0]
    assert isinstance(device, dict)
    device["platform"] = "cpu"
    profile["gpu_host_runtime"] = None
    environment = profile["container_environment"]
    assert isinstance(environment, list)
    environment.extend(("JAX_PLATFORM_NAME=cpu", "JAX_PLATFORMS=cpu"))
    environment.sort()
    with pytest.raises(ValueError, match="GPU-qualified runtime profile"):
        validate_environment_runtime_profile(profile)


def test_runtime_profile_rejects_extra_or_reversioned_packages() -> None:
    extra = _matched_gpu_profile()
    extra_packages = extra["scientific_packages"]
    assert isinstance(extra_packages, list)
    extra_packages.append("unrecorded-runtime-hook==1.0")
    extra_packages.sort()
    with pytest.raises(ValueError, match="scientific packages differ"):
        validate_environment_runtime_profile(extra)

    reversioned = _matched_gpu_profile()
    packages = reversioned["scientific_packages"]
    assert isinstance(packages, list)
    packages.remove("numpy==2.3.1")
    packages.append("numpy==999.0")
    packages.sort()
    with pytest.raises(ValueError, match="scientific packages differ"):
        validate_environment_runtime_profile(reversioned)


def test_runtime_profile_rejects_extra_distribution_record() -> None:
    profile = _matched_gpu_profile()
    records = profile["scientific_package_records"]
    assert isinstance(records, dict)
    records["unrecorded-runtime-hook"] = {
        "record_sha256": "0" * 64,
        "version": "1.0",
    }
    with pytest.raises(ValueError, match="RECORD set differs"):
        validate_environment_runtime_profile(profile)


@pytest.mark.parametrize(
    "distribution",
    [
        "imageio-ffmpeg",
        "numpy",
        "pyexputils",
        "pyfixedreps",
        "replaytables",
    ],
)
def test_runtime_profile_rejects_non_jax_record_version_drift(
    distribution: str,
) -> None:
    profile = _matched_gpu_profile()
    records = profile["scientific_package_records"]
    assert isinstance(records, dict)
    record = records[distribution]
    assert isinstance(record, dict)
    record["version"] = "999.0"
    with pytest.raises(ValueError, match="RECORD version differs"):
        validate_environment_runtime_profile(profile)
