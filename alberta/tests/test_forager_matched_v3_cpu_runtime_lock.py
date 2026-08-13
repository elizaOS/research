"""Synthetic tests for the pure matched-v3 CPU runtime lock contract."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_runtime_lock as lock_module

_DESCRIPTOR_SHA256 = "31d4c5a101f441bc082bdaf9250050f7950440271e6360854d5faa9fcd7ff34a"
_UPSTREAM_COMMIT = "9710f60fa30da5badc451ad7ce3ff296d5070830"
_UPSTREAM_TREE = "a5ad878ac4be0567c43dfd9177471c4b5a910bfa"
_UPSTREAM_ARCHIVE_SHA256 = "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
_UPSTREAM_LOCK_SHA256 = "46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd"
_UPSTREAM_PYPROJECT_SHA256 = "297500b39833ac8210240dd248f93a4f6a3dab4572f11185accecaca8ffed417"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _set_body(value: dict[str, Any], field: str) -> None:
    body = dict(value)
    body.pop(field, None)
    value[field] = hashlib.sha256(_canonical(body)).hexdigest()


def _inventory(value: Any, label: str) -> str:
    return hashlib.sha256(_canonical({label: value})).hexdigest()


def _expanded_tags(python: str, abi: str, platform: str) -> list[str]:
    return sorted(
        f"{python_tag}-{abi_tag}-{platform_tag}"
        for python_tag in python.split(".")
        for abi_tag in abi.split(".")
        for platform_tag in platform.split(".")
    )


def _requirement(
    raw: str,
    name: str,
    *,
    marker: str | None = None,
    active: bool = True,
    selected_version: str | None = None,
) -> dict[str, Any]:
    return {
        "raw": raw,
        "name": name,
        "marker": marker,
        "active": active,
        "selected_version": selected_version,
    }


def _wheel(
    name: str,
    version: str,
    *,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
    requires_python: str,
    provides_extra: list[str] | None = None,
    requires_dist: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tags = _expanded_tags(python_tag, abi_tag, platform_tag)
    filename = f"{name.replace('-', '_')}-{version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
    wheel_sha256 = _sha(f"wheel:{name}:{version}:{filename}")
    dist_info = f"{name.replace('-', '_')}-{version}.dist-info"
    wheel = {
        "filename": filename,
        "source_url": f"https://files.pythonhosted.org/packages/aa/bb/{filename}",
        "cas_key": f"sha256/{wheel_sha256[:2]}/{wheel_sha256}/{filename}",
        "size_bytes": 1_000 + len(name),
        "sha256": wheel_sha256,
        "tags": list(tags),
        "metadata": {
            "path": f"{dist_info}/METADATA",
            "size_bytes": 200 + len(name),
            "sha256": _sha(f"METADATA:{name}"),
            "metadata_version": "2.4",
            "name": name,
            "version": version,
            "requires_python": requires_python,
            "provides_extra": [] if provides_extra is None else provides_extra,
            "requires_dist": [] if requires_dist is None else requires_dist,
        },
        "wheel": {
            "path": f"{dist_info}/WHEEL",
            "size_bytes": 80 + len(name),
            "sha256": _sha(f"WHEEL:{name}"),
            "wheel_version": "1.0",
            "generator": "synthetic-test 1",
            "root_is_purelib": abi_tag == "none" and platform_tag == "any",
            "tags": list(tags),
        },
        "record": {
            "path": f"{dist_info}/RECORD",
            "size_bytes": 300 + len(name),
            "sha256": _sha(f"RECORD:{name}"),
            "entry_count": 4,
            "entries_sha256": _sha(f"RECORD-entries:{name}"),
        },
        "wheel_body_sha256": "0" * 64,
    }
    _set_body(wheel, "wheel_body_sha256")
    return wheel


def _package(
    name: str,
    version: str,
    *,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
    requires_python: str,
    direct: bool = True,
    selected_extras: list[str] | None = None,
    provides_extra: list[str] | None = None,
    requires_dist: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    package = {
        "name": name,
        "version": version,
        "direct": direct,
        "selected_extras": [] if selected_extras is None else selected_extras,
        "installation_kind": "wheel",
        "build_required": False,
        "wheels": [
            _wheel(
                name,
                version,
                python_tag=python_tag,
                abi_tag=abi_tag,
                platform_tag=platform_tag,
                requires_python=requires_python,
                provides_extra=provides_extra,
                requires_dist=requires_dist,
            )
        ],
        "package_body_sha256": "0" * 64,
    }
    _set_body(package, "package_body_sha256")
    return package


def _refresh(lock: dict[str, Any]) -> dict[str, Any]:
    resolution = lock["resolution"]
    resolution["marker_environment_sha256"] = hashlib.sha256(
        _canonical(resolution["marker_environment"])
    ).hexdigest()

    overlay = lock["overlay_delta"]
    overlay["direct_requirements_sha256"] = hashlib.sha256(
        _canonical({"direct_requirements": resolution["direct_requirements"]})
    ).hexdigest()
    for operation in overlay["operations"]:
        _set_body(operation, "operation_body_sha256")
    overlay["operation_count"] = len(overlay["operations"])
    overlay["operations_sha256"] = hashlib.sha256(
        _canonical({"operations": overlay["operations"]})
    ).hexdigest()
    _set_body(overlay, "overlay_body_sha256")

    solver = lock["solver_provenance"]
    solver["marker_environment_sha256"] = resolution["marker_environment_sha256"]
    solver["argv_sha256"] = hashlib.sha256(_canonical({"argv": solver["argv"]})).hexdigest()
    solver["environment_sha256"] = hashlib.sha256(
        _canonical({"environment": solver["environment"]})
    ).hexdigest()

    for package in lock["packages"]:
        for wheel in package["wheels"]:
            _set_body(wheel, "wheel_body_sha256")
        _set_body(package, "package_body_sha256")

    packages = lock["packages"]
    wheels = [package["wheels"][0] for package in packages if package["wheels"]]
    dependencies = [
        {"from": package["name"], **dependency}
        for package in packages
        if package["wheels"]
        for dependency in package["wheels"][0]["metadata"]["requires_dist"]
    ]
    distribution_inventory = [
        {
            "name": package["name"],
            "version": package["version"],
            "direct": package["direct"],
            "package_body_sha256": package["package_body_sha256"],
        }
        for package in packages
    ]
    wheel_inventory = [
        {
            "name": package["name"],
            "filename": wheel["filename"],
            "size_bytes": wheel["size_bytes"],
            "sha256": wheel["sha256"],
            "cas_key": wheel["cas_key"],
            "wheel_body_sha256": wheel["wheel_body_sha256"],
        }
        for package in packages
        if package["wheels"]
        for wheel in package["wheels"][:1]
    ]
    closure = lock["closure"]
    closure.update(
        {
            "distribution_count": len(packages),
            "wheel_count": len(wheels),
            "total_wheel_bytes": sum(wheel["size_bytes"] for wheel in wheels),
            "total_metadata_bytes": sum(wheel["metadata"]["size_bytes"] for wheel in wheels),
            "total_wheel_file_bytes": sum(wheel["wheel"]["size_bytes"] for wheel in wheels),
            "total_record_bytes": sum(wheel["record"]["size_bytes"] for wheel in wheels),
            "requires_dist_count": len(dependencies),
            "active_dependency_count": sum(
                1 for dependency in dependencies if dependency["active"] is True
            ),
            "distribution_inventory_sha256": _inventory(distribution_inventory, "distributions"),
            "wheel_inventory_sha256": _inventory(wheel_inventory, "wheels"),
            "dependency_inventory_sha256": _inventory(dependencies, "dependencies"),
            "packages_body_sha256": _inventory(packages, "packages"),
        }
    )
    _set_body(closure, "closure_body_sha256")

    manifest = lock["wheelhouse"]["manifest"]
    manifest["entry_count"] = closure["wheel_count"]
    manifest["total_bytes"] = closure["total_wheel_bytes"]
    manifest["inventory_sha256"] = closure["wheel_inventory_sha256"]
    archive = lock["wheelhouse"]["archive"]
    archive["manifest_sha256"] = manifest["sha256"]
    archive["manifest_body_sha256"] = manifest["body_sha256"]
    _set_body(lock, "lock_body_sha256")
    return lock


def _valid_lock() -> dict[str, Any]:
    optional_cuda = _requirement(
        'nvidia-cuda-runtime-cu12>=12; extra == "cuda12"',
        "nvidia-cuda-runtime-cu12",
        marker='extra == "cuda12"',
        active=False,
        selected_version=None,
    )
    packages = [
        _package(
            "continual-foragax",
            "0.55.0",
            python_tag="py3",
            abi_tag="none",
            platform_tag="any",
            requires_python=">=3.10",
        ),
        _package(
            "jax",
            "0.11.0",
            python_tag="py2.py3",
            abi_tag="none",
            platform_tag="any",
            requires_python="~=3.12.0",
            provides_extra=["cuda12"],
            requires_dist=[optional_cuda],
        ),
        _package(
            "jaxlib",
            "0.11.0",
            python_tag="cp312",
            abi_tag="cp312",
            platform_tag="manylinux_2_28_x86_64",
            requires_python="==3.12.*",
        ),
    ]
    direct_requirements = [
        "continual-foragax==0.55.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
    ]
    operation = {
        "op": "replace",
        "path": "/pyproject/project/dependencies",
        "expected": ["jax==0.9.0.1", "jaxlib==0.9.0.1"],
        "replacement": list(direct_requirements),
        "operation_body_sha256": "0" * 64,
    }
    lock = {
        "schema_version": lock_module.CPU_RUNTIME_LOCK_SCHEMA_VERSION,
        "status": "future_content_lock_unexecuted_non_authorizing",
        "classification": lock_module.CPU_RUNTIME_LOCK_CLASSIFICATION,
        "target": {
            "implementation": "CPython",
            "python_version": "3.12.3",
            "python_tag": "cp312",
            "abi_tag": "cp312",
            "os": "linux",
            "architecture": "x86_64",
            "platform": "linux-amd64",
            "libc_family": "glibc",
            "libc_version": "2.28",
            "cpu_only": True,
        },
        "upstream": {
            "repository_id": "continual-foragax-agents",
            "repository_url": "https://github.com/steventango/continual-foragax-agents",
            "commit_git_sha1": _UPSTREAM_COMMIT,
            "tree_git_sha1": _UPSTREAM_TREE,
            "archive": {
                "size_bytes": 314_961_920,
                "sha256": _UPSTREAM_ARCHIVE_SHA256,
            },
            "pyproject": {
                "path": "pyproject.toml",
                "size_bytes": 1_927,
                "sha256": _UPSTREAM_PYPROJECT_SHA256,
            },
            "lock": {
                "path": "uv.lock",
                "size_bytes": 200_000,
                "sha256": _UPSTREAM_LOCK_SHA256,
            },
            "root_project_distribution": "continual-foragax-agents",
            "root_project_installed": False,
        },
        "overlay_delta": {
            "schema_version": lock_module.CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION,
            "base_pyproject_sha256": _UPSTREAM_PYPROJECT_SHA256,
            "base_lock_sha256": _UPSTREAM_LOCK_SHA256,
            "delta_format": "canonical_json_operations_v1",
            "operations": [operation],
            "operation_count": 0,
            "operations_sha256": "0" * 64,
            "direct_requirements_sha256": "0" * 64,
            "source_builds_allowed": False,
            "overlay_body_sha256": "0" * 64,
        },
        "solver_provenance": {
            "informational_only": True,
            "argv": ["uv", "lock", "--python", "3.12.3"],
            "argv_sha256": "0" * 64,
            "environment": [
                "LANG=C.UTF-8",
                "UV_INDEX_URL=https://pypi.org/simple",
            ],
            "environment_sha256": "0" * 64,
            "interpreter_implementation": "CPython",
            "interpreter_version": "3.12.3",
            "interpreter_binary_sha256": _sha("python-binary"),
            "solver": "uv",
            "solver_version": "0.8.0",
            "solver_binary_sha256": _sha("uv-binary"),
            "marker_environment_sha256": "0" * 64,
            "index_url": "https://pypi.org/simple",
            "index_capture_timestamp_utc": "2026-08-02T12:34:56Z",
            "resolution_input_sha256": _sha("resolution-input"),
            "resolution_report_size_bytes": 9_000,
            "resolution_report_sha256": _sha("resolution-report"),
            "trusted_for_acceptance": False,
        },
        "resolution": {
            "selected_extras": [],
            "marker_environment": {
                "implementation_name": "cpython",
                "implementation_version": "3.12.3",
                "os_name": "posix",
                "platform_machine": "x86_64",
                "platform_python_implementation": "CPython",
                "platform_release": "6.8.0-synthetic",
                "platform_system": "Linux",
                "platform_version": "#1 SMP synthetic",
                "python_full_version": "3.12.3",
                "python_version": "3.12",
                "sys_platform": "linux",
            },
            "marker_environment_sha256": "0" * 64,
            "direct_requirements": direct_requirements,
        },
        "packages": packages,
        "closure": {
            "distribution_count": 0,
            "wheel_count": 0,
            "total_wheel_bytes": 0,
            "total_metadata_bytes": 0,
            "total_wheel_file_bytes": 0,
            "total_record_bytes": 0,
            "requires_dist_count": 0,
            "active_dependency_count": 0,
            "distribution_inventory_sha256": "0" * 64,
            "wheel_inventory_sha256": "0" * 64,
            "dependency_inventory_sha256": "0" * 64,
            "packages_body_sha256": "0" * 64,
            "closure_body_sha256": "0" * 64,
        },
        "wheelhouse": {
            "schema_version": lock_module.CPU_RUNTIME_WHEELHOUSE_MANIFEST_SCHEMA_VERSION,
            "cas_layout": "sha256/first-two/full-digest/wheel-filename",
            "manifest": {
                "filename": "wheelhouse.cas-manifest.v1.json",
                "size_bytes": 2_000,
                "sha256": _sha("wheelhouse-manifest"),
                "body_sha256": _sha("wheelhouse-manifest-body"),
                "entry_count": 0,
                "total_bytes": 0,
                "inventory_sha256": "0" * 64,
            },
            "archive": {
                "filename": "wheelhouse.v1.tar",
                "format": "ustar",
                "size_bytes": 50_000,
                "sha256": _sha("wheelhouse-archive"),
                "manifest_sha256": "0" * 64,
                "manifest_body_sha256": "0" * 64,
            },
            "networkless_install_required": True,
        },
        "claims": lock_module.cpu_runtime_lock_descriptor()["claims"],
        "limitations": lock_module.cpu_runtime_lock_descriptor()["limitations"],
        "lock_body_sha256": "0" * 64,
    }
    return _refresh(lock)


def _set_wheel_tag(
    lock: dict[str, Any],
    package_index: int,
    *,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
) -> None:
    package = lock["packages"][package_index]
    wheel = package["wheels"][0]
    filename = (
        f"{package['name'].replace('-', '_')}-{package['version']}-"
        f"{python_tag}-{abi_tag}-{platform_tag}.whl"
    )
    tags = _expanded_tags(python_tag, abi_tag, platform_tag)
    wheel["filename"] = filename
    wheel["source_url"] = f"https://files.pythonhosted.org/packages/aa/bb/{filename}"
    wheel["cas_key"] = f"sha256/{wheel['sha256'][:2]}/{wheel['sha256']}/{filename}"
    wheel["tags"] = list(tags)
    wheel["wheel"]["tags"] = list(tags)
    wheel["wheel"]["root_is_purelib"] = abi_tag == "none" and platform_tag == "any"


def _assert_invalid(value: dict[str, Any]) -> None:
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.validate_cpu_runtime_lock(value)


def test_descriptor_identity_detachment_and_exact_parser() -> None:
    raw = lock_module.canonical_cpu_runtime_lock_descriptor_bytes()
    assert hashlib.sha256(raw).hexdigest() == _DESCRIPTOR_SHA256
    assert lock_module.cpu_runtime_lock_descriptor_sha256() == _DESCRIPTOR_SHA256
    first = lock_module.cpu_runtime_lock_descriptor()
    second = lock_module.parse_cpu_runtime_lock_descriptor(raw)
    first["claims"]["production_lock_exists"] = True
    assert second["claims"]["production_lock_exists"] is False
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.parse_cpu_runtime_lock_descriptor(raw.replace(b"schema_only", b"schema-only"))


def test_module_import_surface_is_pure_and_nonexecuting() -> None:
    source_path = Path(lock_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imported_roots <= {
        "__future__",
        "collections",
        "hashlib",
        "hmac",
        "itertools",
        "json",
        "re",
        "typing",
        "urllib",
    }
    forbidden_calls = {"open", "exec", "eval", "compile", "__import__"}
    assert (
        not {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        & forbidden_calls
    )
    public = set(lock_module.__all__)
    assert not any(
        token in name.lower()
        for name in public
        for token in ("build", "capture", "download", "execute", "install", "issue", "write")
    )


def test_valid_generic_lock_roundtrip_is_detached() -> None:
    value = _valid_lock()
    raw = lock_module.canonical_cpu_runtime_lock_bytes(value)
    digest = hashlib.sha256(raw).hexdigest()
    parsed = lock_module.parse_cpu_runtime_lock(raw, expected_file_sha256=digest)
    parsed["target"]["python_version"] = "3.12.4"
    assert value["target"]["python_version"] == "3.12.3"
    assert lock_module.cpu_runtime_lock_sha256(value) == digest
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.parse_cpu_runtime_lock(raw, expected_file_sha256=_sha("wrong-file"))


def test_strict_json_duplicate_noncanonical_and_float_rejection() -> None:
    raw = lock_module.canonical_cpu_runtime_lock_bytes(_valid_lock())
    duplicate = raw.replace(b'{"claims":', b'{"claims":{},"claims":', 1)
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.parse_cpu_runtime_lock(
            duplicate,
            expected_file_sha256=hashlib.sha256(duplicate).hexdigest(),
        )
    noncanonical = raw[:-1] + b" \n"
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.parse_cpu_runtime_lock(
            noncanonical,
            expected_file_sha256=hashlib.sha256(noncanonical).hexdigest(),
        )
    value = _valid_lock()
    value["target"]["unknown"] = 1
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][0]["wheels"][0]["size_bytes"] = 1.0
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][0]["wheels"][0]["size_bytes"] = True
    _assert_invalid(value)


def test_mapping_alias_and_noncanonical_distribution_name_rejection() -> None:
    value = _valid_lock()
    value["limitations"] = value["claims"]
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][1]["name"] = "JAX"
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("target", "implementation", "PyPy"),
        ("target", "python_version", "3.12"),
        ("target", "abi_tag", "cp311"),
        ("target", "architecture", "aarch64"),
        ("target", "platform", "linux-arm64"),
        ("target", "libc_family", "musl"),
        ("upstream", "commit_git_sha1", "1" * 40),
        ("upstream", "tree_git_sha1", "2" * 40),
        ("upstream", "root_project_installed", True),
    ],
)
def test_target_and_upstream_exactness(section: str, field: str, replacement: object) -> None:
    value = _valid_lock()
    value[section][field] = replacement
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize("identity", ["archive", "pyproject", "lock"])
def test_upstream_file_identity_exactness(identity: str) -> None:
    value = _valid_lock()
    value["upstream"][identity]["sha256"] = _sha(f"wrong-{identity}")
    if identity == "pyproject":
        value["overlay_delta"]["base_pyproject_sha256"] = value["upstream"][identity]["sha256"]
    _refresh(value)
    _assert_invalid(value)


def test_complete_marker_environment_and_hash_binding() -> None:
    value = _valid_lock()
    value["resolution"]["marker_environment"].pop("platform_release")
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["resolution"]["marker_environment_sha256"] = _sha("stale-marker")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)
    value = _valid_lock()
    value["resolution"]["marker_environment"]["platform_release"] = "6.9.1-exact"
    _refresh(value)
    assert (
        lock_module.validate_cpu_runtime_lock(value)["resolution"]["marker_environment_sha256"]
        == hashlib.sha256(_canonical(value["resolution"]["marker_environment"])).hexdigest()
    )


def test_overlay_exact_operations_and_all_hashes() -> None:
    value = _valid_lock()
    value["overlay_delta"]["operations"][0]["path"] = "/outside/value"
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["overlay_delta"]["operations"][0]["replacement"] = None
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    second = {
        "op": "remove",
        "path": "/lock/package/source",
        "expected": "registry",
        "replacement": None,
        "operation_body_sha256": "0" * 64,
    }
    value["overlay_delta"]["operations"].append(second)
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["overlay_delta"]["operations"][0]["operation_body_sha256"] = _sha("stale")
    _set_body(value["overlay_delta"], "overlay_body_sha256")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)
    value = _valid_lock()
    value["overlay_delta"]["operations_sha256"] = _sha("stale-operations")
    _set_body(value["overlay_delta"], "overlay_body_sha256")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)
    value = _valid_lock()
    value["overlay_delta"]["direct_requirements_sha256"] = _sha("stale-direct")
    _set_body(value["overlay_delta"], "overlay_body_sha256")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("interpreter_implementation", "PyPy"),
        ("interpreter_version", "3.12.4"),
        ("interpreter_binary_sha256", "0" * 64),
        ("solver_binary_sha256", "0" * 64),
        ("index_capture_timestamp_utc", "2026-08-02T12:34:56+00:00"),
        ("resolution_input_sha256", "0" * 64),
        ("trusted_for_acceptance", True),
    ],
)
def test_solver_exact_identity_fields(field: str, replacement: object) -> None:
    value = _valid_lock()
    value["solver_provenance"][field] = replacement
    _refresh(value)
    _assert_invalid(value)


def test_solver_argv_and_environment_content_hashes() -> None:
    value = _valid_lock()
    value["solver_provenance"]["argv"].append("--offline")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)
    value = _valid_lock()
    value["solver_provenance"]["environment"].append("LANG=C.UTF-8")
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["solver_provenance"]["environment_sha256"] = _sha("stale-environment")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("python_tag", "abi_tag", "platform_tag"),
    [
        ("cp36", "abi3", "manylinux1_x86_64"),
        ("cp37", "abi3", "manylinux2010_x86_64"),
        ("cp310", "abi3", "manylinux2014_x86_64"),
        ("cp310", "abi3", "manylinux_2_17_x86_64"),
    ],
)
def test_legacy_stable_abi_and_manylinux_tags_are_compatible(
    python_tag: str, abi_tag: str, platform_tag: str
) -> None:
    value = _valid_lock()
    _set_wheel_tag(
        value,
        2,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
    )
    _refresh(value)
    lock_module.validate_cpu_runtime_lock(value)


@pytest.mark.parametrize(
    ("python_tag", "abi_tag", "platform_tag"),
    [
        ("cp311", "cp311", "manylinux_2_17_x86_64"),
        ("cp312", "cp312", "musllinux_1_2_x86_64"),
        ("cp312", "cp312", "manylinux_2_28_aarch64"),
        ("cp312", "cp312", "manylinux_2_31_x86_64"),
    ],
)
def test_wrong_python_abi_platform_or_libc_wheel_tags_fail(
    python_tag: str, abi_tag: str, platform_tag: str
) -> None:
    value = _valid_lock()
    _set_wheel_tag(
        value,
        2,
        python_tag=python_tag,
        abi_tag=abi_tag,
        platform_tag=platform_tag,
    )
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_url", "http://files.pythonhosted.org/packages/aa/bb/x.whl"),
        ("source_url", "https://example.com/packages/aa/bb/x.whl"),
        ("source_url", "https://files.pythonhosted.org/packages/aa/bb/x.whl?mutable=1"),
        ("cas_key", "sha256/not-the-wheel"),
        ("filename", "source.tar.gz"),
    ],
)
def test_wheel_filename_url_and_cas_are_exact(field: str, replacement: str) -> None:
    value = _valid_lock()
    value["packages"][0]["wheels"][0][field] = replacement
    _refresh(value)
    _assert_invalid(value)


def test_exact_mixed_case_wheel_and_dist_info_paths_preserve_canonical_identity() -> None:
    value = _valid_lock()
    wheel = value["packages"][0]["wheels"][0]
    filename = "Continual_Foragax-0.55.0-py3-none-any.whl"
    dist_info = "Continual_Foragax-0.55.0.dist-info"
    wheel["filename"] = filename
    wheel["source_url"] = f"https://files.pythonhosted.org/packages/aa/bb/{filename}"
    wheel["cas_key"] = f"sha256/{wheel['sha256'][:2]}/{wheel['sha256']}/{filename}"
    wheel["metadata"]["path"] = f"{dist_info}/METADATA"
    wheel["wheel"]["path"] = f"{dist_info}/WHEEL"
    wheel["record"]["path"] = f"{dist_info}/RECORD"
    _refresh(value)

    lock_module.validate_cpu_runtime_lock(value)

    wheel["metadata"]["path"] = "Different_Project-0.55.0.dist-info/METADATA"
    _refresh(value)
    _assert_invalid(value)


def test_duplicate_and_multiple_wheel_rejection() -> None:
    value = _valid_lock()
    value["packages"][1]["wheels"] = []
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][1]["wheels"].append(copy.deepcopy(value["packages"][1]["wheels"][0]))
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][1]["wheels"][0]["sha256"] = value["packages"][0]["wheels"][0]["sha256"]
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    duplicate = copy.deepcopy(value["packages"][1])
    value["packages"].insert(2, duplicate)
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    wheel = value["packages"][0]["wheels"][0]
    wheel["tags"].append(wheel["tags"][0])
    wheel["wheel"]["tags"].append(wheel["wheel"]["tags"][0])
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][1]["wheels"][0]["filename"] = value["packages"][0]["wheels"][0]["filename"]
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    "replacement",
    [
        "continual-foragax==0.55.1",
        "jax==0.11.1",
        "jaxlib==0.11.1",
    ],
)
def test_mandatory_direct_versions_are_exact(replacement: str) -> None:
    value = _valid_lock()
    name = replacement.partition("==")[0]
    value["resolution"]["direct_requirements"] = [
        item
        for item in value["resolution"]["direct_requirements"]
        if not item.startswith(f"{name}==")
    ] + [replacement]
    value["resolution"]["direct_requirements"].sort()
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("requires_python", "valid"),
    [
        ("~=3.12.0", True),
        ("~=3.12", True),
        (">=3.10,<3.13", True),
        ("~=3.11.0", False),
        ("~=3", False),
        (">=3.13", False),
    ],
)
def test_requires_python_compatible_release_semantics(requires_python: str, valid: bool) -> None:
    value = _valid_lock()
    value["packages"][1]["wheels"][0]["metadata"]["requires_python"] = requires_python
    _refresh(value)
    if valid:
        lock_module.validate_cpu_runtime_lock(value)
    else:
        _assert_invalid(value)


def test_absent_optional_requires_python_is_preserved_as_null() -> None:
    value = _valid_lock()
    value["packages"][1]["wheels"][0]["metadata"]["requires_python"] = None
    _refresh(value)

    lock_module.validate_cpu_runtime_lock(value)


def test_declared_selected_extra_and_inactive_accelerator_metadata() -> None:
    value = _valid_lock()
    lock_module.validate_cpu_runtime_lock(value)
    value["packages"][1]["selected_extras"] = ["cpu"]
    value["resolution"]["selected_extras"] = ["cpu"]
    _refresh(value)
    _assert_invalid(value)
    value["packages"][1]["wheels"][0]["metadata"]["provides_extra"].append("cpu")
    value["packages"][1]["wheels"][0]["metadata"]["provides_extra"].sort()
    _refresh(value)
    lock_module.validate_cpu_runtime_lock(value)


def test_inactive_accelerator_marker_extra_must_be_declared_by_owner() -> None:
    value = _valid_lock()
    value["packages"][1]["wheels"][0]["metadata"]["provides_extra"] = []
    _refresh(value)
    _assert_invalid(value)


def test_raw_requires_dist_extra_order_is_not_semantic() -> None:
    value = _valid_lock()
    requirements = value["packages"][1]["wheels"][0]["metadata"]["requires_dist"]
    requirements.append(
        _requirement(
            'etils[dev,all] ; extra == "docs"',
            "etils",
            marker='extra == "docs"',
            active=False,
            selected_version=None,
        )
    )
    requirements.sort(key=lambda item: item["raw"])
    _refresh(value)

    lock_module.validate_cpu_runtime_lock(value)


def test_inactive_accelerator_marker_must_prove_extra_guard_across_boolean_structure() -> None:
    value = _valid_lock()
    dependency = value["packages"][1]["wheels"][0]["metadata"]["requires_dist"][0]
    dependency["raw"] = (
        'nvidia-cuda-runtime-cu12>=12; sys_platform == "linux" and extra == "cuda12"'
    )
    dependency["marker"] = 'sys_platform == "linux" and extra == "cuda12"'
    _refresh(value)
    lock_module.validate_cpu_runtime_lock(value)

    dependency["raw"] = 'nvidia-cuda-runtime-cu12>=12; extra == "cuda12" or sys_platform == "linux"'
    dependency["marker"] = 'extra == "cuda12" or sys_platform == "linux"'
    _refresh(value)
    _assert_invalid(value)


def test_active_direct_and_unguarded_accelerator_edges_fail() -> None:
    value = _valid_lock()
    dependency = value["packages"][1]["wheels"][0]["metadata"]["requires_dist"][0]
    dependency["active"] = True
    dependency["selected_version"] = "12.0"
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    dependency = value["packages"][1]["wheels"][0]["metadata"]["requires_dist"][0]
    dependency["raw"] = "nvidia-cuda-runtime-cu12>=12"
    dependency["marker"] = None
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["resolution"]["direct_requirements"].append("nvidia-cuda-runtime-cu12==12.0")
    value["resolution"]["direct_requirements"].sort()
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    "raw",
    [
        "example @ https://example.com/example.whl",
        "example @ git+https://example.com/repository.git",
        "example @ file:///var/empty/example.whl",
        "example @ ../example",
        "example>=1 trailing-garbage",
        "jax>=1:2",
        "jax==1**",
        "jax~=1!2",
        "jax<=1.2.*",
    ],
)
def test_requires_dist_url_vcs_path_and_trailing_syntax_fail(raw: str) -> None:
    value = _valid_lock()
    dependency_name = "jax" if raw.startswith("jax") else "example"
    selected_version = "0.11.0" if dependency_name == "jax" else "1.0"
    value["packages"][0]["wheels"][0]["metadata"]["requires_dist"] = [
        _requirement(raw, dependency_name, selected_version=selected_version)
    ]
    _refresh(value)
    _assert_invalid(value)


def test_selected_distribution_versions_must_be_valid_pep440() -> None:
    value = _valid_lock()
    invalid = _package(
        "example",
        "1..0",
        python_tag="py3",
        abi_tag="none",
        platform_tag="any",
        requires_python=">=3.12",
    )
    value["packages"].append(invalid)
    value["packages"].sort(key=lambda package: package["name"])
    value["resolution"]["direct_requirements"].append("example==1..0")
    value["resolution"]["direct_requirements"].sort()
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    "marker",
    [
        '(python_version) == "3.12"',
        'python_version == ("3.12")',
    ],
)
def test_requires_dist_marker_parentheses_must_wrap_complete_expressions(marker: str) -> None:
    value = _valid_lock()
    value["packages"][0]["wheels"][0]["metadata"]["requires_dist"] = [
        _requirement(
            f"jax>=0.11; {marker}",
            "jax",
            marker=marker,
            selected_version="0.11.0",
        )
    ]
    _refresh(value)
    _assert_invalid(value)


@pytest.mark.parametrize(
    "marker",
    [
        '(python_version == "3.12")',
        '(python_version == "3.12" and sys_platform == "linux") or extra == "cpu"',
        'python_version == "3.12" or (sys_platform == "linux" and os_name == "posix")',
    ],
)
def test_requires_dist_marker_accepts_complete_nested_expressions(marker: str) -> None:
    value = _valid_lock()
    value["packages"][0]["wheels"][0]["metadata"]["requires_dist"] = [
        _requirement(
            f"jax>=0.11; {marker}",
            "jax",
            marker=marker,
            selected_version="0.11.0",
        )
    ]
    _refresh(value)
    lock_module.validate_cpu_runtime_lock(value)


def test_active_requested_extra_must_be_selected_on_the_target_distribution() -> None:
    value = _valid_lock()
    value["packages"][0]["selected_extras"] = ["cpu"]
    value["packages"][0]["wheels"][0]["metadata"]["provides_extra"] = ["cpu"]
    value["packages"][1]["wheels"][0]["metadata"]["provides_extra"].append("cpu")
    value["packages"][1]["wheels"][0]["metadata"]["provides_extra"].sort()
    value["packages"][0]["wheels"][0]["metadata"]["requires_dist"] = [
        _requirement("jax[cpu]>=0.11", "jax", selected_version="0.11.0")
    ]
    value["resolution"]["selected_extras"] = ["cpu"]
    _refresh(value)
    _assert_invalid(value)

    value["packages"][0]["selected_extras"] = []
    value["packages"][1]["selected_extras"] = ["cpu"]
    _refresh(value)
    lock_module.validate_cpu_runtime_lock(value)


def test_root_project_sdist_and_source_build_rejection() -> None:
    value = _valid_lock()
    value["packages"][0]["name"] = "continual-foragax-agents"
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["packages"][0]["installation_kind"] = "sdist"
    value["packages"][0]["build_required"] = True
    _refresh(value)
    _assert_invalid(value)


def test_unreachable_and_missing_direct_distributions_fail() -> None:
    value = _valid_lock()
    orphan = _package(
        "orphan",
        "1.0",
        python_tag="py3",
        abi_tag="none",
        platform_tag="any",
        requires_python=">=3.12",
        direct=False,
    )
    value["packages"].append(orphan)
    _refresh(value)
    _assert_invalid(value)
    value = _valid_lock()
    value["resolution"]["direct_requirements"].append("missing==1.0")
    value["resolution"]["direct_requirements"].sort()
    _refresh(value)
    _assert_invalid(value)


def test_active_edge_makes_transitive_distribution_reachable() -> None:
    value = _valid_lock()
    orphan = _package(
        "orphan",
        "1.0",
        python_tag="py3",
        abi_tag="none",
        platform_tag="any",
        requires_python=">=3.12",
        direct=False,
    )
    value["packages"].append(orphan)
    value["packages"][0]["wheels"][0]["metadata"]["requires_dist"] = [
        _requirement("orphan==1.0", "orphan", selected_version="1.0")
    ]
    _refresh(value)
    assert lock_module.validate_cpu_runtime_lock(value)["closure"]["distribution_count"] == 4


def test_wheel_root_is_purelib_is_an_exact_install_scheme_declaration() -> None:
    value = _valid_lock()
    value["packages"][0]["wheels"][0]["wheel"]["root_is_purelib"] = False
    _refresh(value)
    parsed = lock_module.validate_cpu_runtime_lock(value)
    assert parsed["packages"][0]["wheels"][0]["wheel"]["root_is_purelib"] is False
    value = _valid_lock()
    value["packages"][2]["wheels"][0]["wheel"]["root_is_purelib"] = True
    _refresh(value)
    parsed = lock_module.validate_cpu_runtime_lock(value)
    assert parsed["packages"][2]["wheels"][0]["wheel"]["root_is_purelib"] is True


@pytest.mark.parametrize(
    "field",
    [
        "distribution_count",
        "total_wheel_bytes",
        "distribution_inventory_sha256",
        "wheel_inventory_sha256",
        "dependency_inventory_sha256",
        "packages_body_sha256",
        "closure_body_sha256",
    ],
)
def test_closure_count_inventory_and_body_hashes(field: str) -> None:
    value = _valid_lock()
    closure = value["closure"]
    closure[field] = closure[field] + 1 if type(closure[field]) is int else _sha(field)
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("manifest", "sha256"),
        ("manifest", "body_sha256"),
        ("manifest", "entry_count"),
        ("manifest", "total_bytes"),
        ("manifest", "inventory_sha256"),
        ("archive", "sha256"),
        ("archive", "manifest_sha256"),
        ("archive", "manifest_body_sha256"),
    ],
)
def test_wheelhouse_manifest_and_archive_identities(section: str, field: str) -> None:
    value = _valid_lock()
    item = value["wheelhouse"][section]
    if section == "archive" and field == "sha256":
        item[field] = "0" * 64
    else:
        item[field] = item[field] + 1 if type(item[field]) is int else _sha(f"wrong-{field}")
    _set_body(value, "lock_body_sha256")
    _assert_invalid(value)


def test_production_count_is_separate_and_claims_stay_false() -> None:
    assert lock_module.PRODUCTION_DISTRIBUTION_COUNT == 104
    value = _valid_lock()
    assert lock_module.validate_cpu_runtime_lock(value)["closure"]["distribution_count"] == 3
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.validate_production_cpu_runtime_lock(value)
    with pytest.raises(lock_module.ForagerMatchedV3CpuRuntimeLockError):
        lock_module.parse_cpu_runtime_lock(
            lock_module.canonical_cpu_runtime_lock_bytes(value),
            expected_file_sha256=lock_module.cpu_runtime_lock_sha256(value),
            production=True,
        )
    assert all(claim is False for claim in value["claims"].values())
    value["claims"]["execution_authorized"] = True
    _refresh(value)
    _assert_invalid(value)
