"""Focused tests for the pure matched-v3 CPU runtime-lock issuer."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_runtime_lock as runtime_lock
from alberta_framework.benchmarks import forager_matched_v3_cpu_runtime_lock_issuer as issuer
from alberta_framework.benchmarks import forager_matched_v3_cpu_wheelhouse as wheelhouse

pytestmark = pytest.mark.unit

_UPSTREAM_ARCHIVE_SHA256 = "1f6976de38f34a697c947891de26ad3373b294195fe82094e9d1d5b8ddfd43b6"
_UPSTREAM_LOCK_SHA256 = "46c2990caf152b84bcb3ac39de5173304cdbf5edd61a68f3d0000b843dabbacd"
_UPSTREAM_PYPROJECT_SHA256 = "297500b39833ac8210240dd248f93a4f6a3dab4572f11185accecaca8ffed417"


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


def _compact(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha(value: bytes | str) -> str:
    raw = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _set_body(value: dict[str, Any], field: str) -> None:
    body = copy.deepcopy(value)
    body.pop(field, None)
    value[field] = _sha(_canonical(body))


def _marker_environment() -> dict[str, str]:
    return {
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
    }


def _target() -> dict[str, Any]:
    return {
        "abi": "cp312",
        "compatible_tags": [
            "cp312-cp312-manylinux_2_28_x86_64",
            "py3-none-any",
        ],
        "implementation": "CPython",
        "libc": {"family": "glibc", "version": "2.28"},
        "marker_environment": _marker_environment(),
        "oci_platform": "linux/amd64",
        "platform": "linux_x86_64",
        "python_version": "3.12.3",
    }


def _requirement(
    raw: str,
    name: str,
    *,
    extras: list[str] | None = None,
    marker: str | None = None,
    specifier: str = "",
) -> dict[str, Any]:
    return {
        "extras": [] if extras is None else extras,
        "marker": marker,
        "name": name,
        "raw": raw,
        "specifier": specifier,
    }


def _receipt_package(
    name: str,
    version: str,
    *,
    platform_tag: str = "any",
    selected_requirement: dict[str, Any] | None = None,
) -> dict[str, Any]:
    escaped_name = name.replace("-", "_")
    if platform_tag == "any":
        python_tag, abi_tag = "py3", "none"
    else:
        python_tag, abi_tag = "cp312", "cp312"
    tag = f"{python_tag}-{abi_tag}-{platform_tag}"
    filename = f"{escaped_name}-{version}-{tag}.whl"
    wheel_sha = _sha(f"wheel:{name}:{version}")
    dist_info = f"{escaped_name}-{version}.dist-info"
    requirements = [] if selected_requirement is None else [selected_requirement]
    provided = ["feature"] if name == "continual-foragax" else []
    metadata = {
        "metadata_version": "2.4",
        "name": name,
        "path": f"{dist_info}/METADATA",
        "provides_extra": provided,
        "requires_dist": requirements,
        "requires_python": ">=3.12,<3.13",
        "sha256": _sha(f"METADATA:{name}"),
        "size_bytes": 200 + len(name),
        "version": version,
    }
    wheel_file = {
        "build": None,
        "generator": "synthetic-test 1",
        "path": f"{dist_info}/WHEEL",
        "root_is_purelib": platform_tag == "any",
        "sha256": _sha(f"WHEEL:{name}"),
        "size_bytes": 80 + len(name),
        "tags": [tag],
        "wheel_version": "1.0",
    }
    record = {
        "entries_sha256": _sha(f"RECORD-entries:{name}"),
        "entry_count": 4,
        "path": f"{dist_info}/RECORD",
        "sha256": _sha(f"RECORD:{name}"),
        "size_bytes": 300 + len(name),
    }
    return {
        "dist_info_directory": dist_info,
        "filename": filename,
        "metadata": metadata,
        "name": name,
        "record": record,
        "sha256": wheel_sha,
        "size_bytes": 1_000 + len(name),
        "tags": [tag],
        "version": version,
        "wheel": wheel_file,
    }


def _root_record(raw: str, name: str, version: str, extras: list[str]) -> dict[str, Any]:
    return _requirement(
        raw,
        name,
        extras=extras,
        marker=None,
        specifier=f"=={version}",
    )


def _fixture_values() -> dict[str, Any]:
    dependency = _requirement(
        "jax==0.11.0",
        "jax",
        marker=None,
        specifier="==0.11.0",
    )
    packages = [
        _receipt_package(
            "continual-foragax",
            "0.55.0",
            selected_requirement=dependency,
        ),
        _receipt_package("jax", "0.11.0"),
        _receipt_package(
            "jaxlib",
            "0.11.0",
            platform_tag="manylinux_2_28_x86_64",
        ),
    ]
    roots = [
        "continual-foragax[feature]==0.55.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
    ]
    root_records = [
        _root_record(roots[0], "continual-foragax", "0.55.0", ["feature"]),
        _root_record(roots[1], "jax", "0.11.0", []),
        _root_record(roots[2], "jaxlib", "0.11.0", []),
    ]
    wheels = [
        {
            "filename": package["filename"],
            "origin_url": (
                "https://files.pythonhosted.org/packages/aa/bb/"
                + "c" * 60
                + "/"
                + package["filename"]
            ),
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
        }
        for package in sorted(packages, key=lambda item: item["filename"])
    ]
    manifest_raw = _canonical({"synthetic": "capture-manifest"})
    receipt_raw = _canonical({"synthetic": "wheelhouse-receipt"})
    manifest = {
        "manifest_body_sha256": _sha("capture-manifest-body"),
        "root_requirements": roots,
        "target": _target(),
        "wheels": wheels,
    }
    members = [
        {
            "archive_name": f"{package['sha256']}.whl",
            "filename": package["filename"],
            "mode": "0444",
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
        }
        for package in sorted(packages, key=lambda item: item["sha256"])
    ]
    archive_size = sum(
        512 + member["size_bytes"] + (-member["size_bytes"]) % 512 for member in members
    )
    archive_size += 2 * 512
    archive_size += (-archive_size) % 10_240
    receipt = {
        "archive": {
            "inventory_sha256": _sha(_compact(members)),
            "members": members,
            "sha256": _sha("wheelhouse-archive"),
            "size_bytes": archive_size,
        },
        "capture_manifest": {
            "body_sha256": manifest["manifest_body_sha256"],
            "full_file_sha256": _sha(manifest_raw),
        },
        "closure": {
            "activated_extras": {
                "continual-foragax": ["feature"],
                "jax": [],
                "jaxlib": [],
            },
            "edges": [
                {
                    "active_contexts": ["", "feature"],
                    "requirement": copy.deepcopy(dependency),
                    "source": "continual-foragax",
                    "target": "jax",
                }
            ],
            "root_requirements": root_records,
        },
        "packages": packages,
        "receipt_body_sha256": _sha("wheelhouse-receipt-body"),
        "root_requirements": roots,
        "target": copy.deepcopy(manifest["target"]),
    }
    return {
        "manifest": manifest,
        "manifest_raw": manifest_raw,
        "receipt": receipt,
        "receipt_raw": receipt_raw,
    }


def _root_pin_inventory(receipt: dict[str, Any]) -> dict[str, Any]:
    packages = {package["name"]: package for package in receipt["packages"]}
    pins = [
        {
            "name": root["name"],
            "requirement": root["raw"],
            "selected_extras": root["extras"],
            "version": packages[root["name"]]["version"],
        }
        for root in receipt["closure"]["root_requirements"]
    ]
    inventory = {
        "inventory_sha256": _sha(_canonical({"pins": pins})),
        "pin_count": len(pins),
        "pins": pins,
    }
    return inventory


def _envelope(values: dict[str, Any]) -> tuple[bytes, str]:
    manifest = values["manifest"]
    receipt = values["receipt"]
    direct_requirements = [
        "continual-foragax==0.55.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
    ]
    overlay_requirements = [
        "continual-foragax[feature]>=0.50.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
    ]
    operation = {
        "expected": ["jax==0.9.0.1", "jaxlib==0.9.0.1"],
        "op": "replace",
        "operation_body_sha256": "0" * 64,
        "path": "/pyproject/project/dependencies",
        "replacement": overlay_requirements,
    }
    _set_body(operation, "operation_body_sha256")
    overlay = {
        "base_lock_sha256": _UPSTREAM_LOCK_SHA256,
        "base_pyproject_sha256": _UPSTREAM_PYPROJECT_SHA256,
        "delta_format": "canonical_json_operations_v1",
        "direct_requirements_sha256": _sha(
            _canonical({"direct_requirements": direct_requirements})
        ),
        "operation_count": 1,
        "operations": [operation],
        "operations_sha256": _sha(_canonical({"operations": [operation]})),
        "overlay_body_sha256": "0" * 64,
        "schema_version": runtime_lock.CPU_RUNTIME_LOCK_OVERLAY_SCHEMA_VERSION,
        "source_builds_allowed": False,
    }
    _set_body(overlay, "overlay_body_sha256")
    marker_sha = _sha(_canonical(_marker_environment()))
    resolution_lock_sha = _sha("final-union-uv-lock")
    solver = {
        "argv": ["uv", "lock", "--python", "3.12.3"],
        "argv_sha256": "0" * 64,
        "environment": ["LANG=C.UTF-8", "UV_INDEX_URL=https://pypi.org/simple"],
        "environment_sha256": "0" * 64,
        "index_capture_timestamp_utc": "2026-08-02T12:34:56Z",
        "index_url": "https://pypi.org/simple",
        "informational_only": True,
        "interpreter_binary_sha256": _sha("python-binary"),
        "interpreter_implementation": "CPython",
        "interpreter_version": "3.12.3",
        "marker_environment_sha256": marker_sha,
        "resolution_input_sha256": resolution_lock_sha,
        "resolution_report_sha256": _sha("resolution-report"),
        "resolution_report_size_bytes": 9_000,
        "solver": "uv",
        "solver_binary_sha256": _sha("uv-binary"),
        "solver_version": "0.8.0",
        "trusted_for_acceptance": False,
    }
    solver["argv_sha256"] = _sha(_canonical({"argv": solver["argv"]}))
    solver["environment_sha256"] = _sha(_canonical({"environment": solver["environment"]}))
    selected_wheels = [
        {
            "filename": package["filename"],
            "name": package["name"],
            "sha256": package["sha256"],
            "size_bytes": package["size_bytes"],
            "source_url": next(
                wheel["origin_url"]
                for wheel in manifest["wheels"]
                if wheel["filename"] == package["filename"]
            ),
            "version": package["version"],
        }
        for package in receipt["packages"]
    ]
    resolution_inventory = {
        "lock_format": "uv_lock_toml",
        "lock_sha256": resolution_lock_sha,
        "lock_size_bytes": 112_074,
        "selected_wheel_count": len(selected_wheels),
        "selected_wheel_inventory_sha256": _sha(_canonical({"selected_wheels": selected_wheels})),
        "selected_wheels": selected_wheels,
    }
    descriptor = runtime_lock.cpu_runtime_lock_descriptor()
    envelope = {
        "claims": copy.deepcopy(descriptor["claims"]),
        "classification": issuer.CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_CLASSIFICATION,
        "envelope_body_sha256": "0" * 64,
        "input_bindings": {
            "capture_manifest_body_sha256": manifest["manifest_body_sha256"],
            "capture_manifest_sha256": _sha(values["manifest_raw"]),
            "wheelhouse_archive_inventory_sha256": receipt["archive"]["inventory_sha256"],
            "wheelhouse_archive_sha256": receipt["archive"]["sha256"],
            "wheelhouse_archive_size_bytes": receipt["archive"]["size_bytes"],
            "wheelhouse_receipt_body_sha256": receipt["receipt_body_sha256"],
            "wheelhouse_receipt_sha256": _sha(values["receipt_raw"]),
        },
        "limitations": [
            *copy.deepcopy(descriptor["limitations"]),
            issuer.CPU_RUNTIME_LOCK_ISSUANCE_RANGE_LIMITATION,
        ],
        "overlay_delta": overlay,
        "resolution_inventory": resolution_inventory,
        "root_pin_inventory": _root_pin_inventory(receipt),
        "schema_version": issuer.CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_SCHEMA_VERSION,
        "solver_provenance": solver,
        "status": issuer.CPU_RUNTIME_LOCK_ISSUANCE_ENVELOPE_STATUS,
        "upstream": {
            "archive": {
                "sha256": _UPSTREAM_ARCHIVE_SHA256,
                "size_bytes": 314_961_920,
            },
            "commit_git_sha1": "9710f60fa30da5badc451ad7ce3ff296d5070830",
            "lock": {
                "path": "uv.lock",
                "sha256": _UPSTREAM_LOCK_SHA256,
                "size_bytes": 200_000,
            },
            "pyproject": {
                "path": "pyproject.toml",
                "sha256": _UPSTREAM_PYPROJECT_SHA256,
                "size_bytes": 1_927,
            },
            "repository_id": "continual-foragax-agents",
            "repository_url": "https://github.com/steventango/continual-foragax-agents",
            "root_project_distribution": "continual-foragax-agents",
            "root_project_installed": False,
            "tree_git_sha1": "a5ad878ac4be0567c43dfd9177471c4b5a910bfa",
        },
    }
    _set_body(envelope, "envelope_body_sha256")
    raw = _canonical(envelope)
    return raw, _sha(raw)


def _patch_parsers(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any],
) -> list[tuple[str, bytes, str]]:
    calls: list[tuple[str, bytes, str]] = []

    def parse_manifest(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        calls.append(("manifest", raw, expected_file_sha256))
        return copy.deepcopy(values["manifest"])

    def parse_receipt(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        calls.append(("receipt", raw, expected_file_sha256))
        return copy.deepcopy(values["receipt"])

    monkeypatch.setattr(
        wheelhouse,
        "parse_cpu_wheel_capture_manifest",
        parse_manifest,
    )
    monkeypatch.setattr(
        wheelhouse,
        "parse_cpu_wheelhouse_receipt",
        parse_receipt,
    )
    return calls


def _issue(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any] | None = None,
) -> issuer.CpuRuntimeLockIssuanceArtifacts:
    exact = _fixture_values() if values is None else values
    _patch_parsers(monkeypatch, exact)
    envelope_raw, envelope_sha = _envelope(exact)
    return issuer.issue_matched_v3_cpu_runtime_lock(
        capture_manifest_raw=exact["manifest_raw"],
        expected_capture_manifest_sha256=_sha(exact["manifest_raw"]),
        wheelhouse_receipt_raw=exact["receipt_raw"],
        expected_wheelhouse_receipt_sha256=_sha(exact["receipt_raw"]),
        issuance_envelope_raw=envelope_raw,
        expected_issuance_envelope_sha256=envelope_sha,
    )


def test_issuer_maps_exact_content_into_lock_and_cas_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    calls = _patch_parsers(monkeypatch, values)
    envelope_raw, envelope_sha = _envelope(values)
    kwargs = {
        "capture_manifest_raw": values["manifest_raw"],
        "expected_capture_manifest_sha256": _sha(values["manifest_raw"]),
        "wheelhouse_receipt_raw": values["receipt_raw"],
        "expected_wheelhouse_receipt_sha256": _sha(values["receipt_raw"]),
        "issuance_envelope_raw": envelope_raw,
        "expected_issuance_envelope_sha256": envelope_sha,
    }
    first = issuer.issue_matched_v3_cpu_runtime_lock(**kwargs)
    second = issuer.issue_matched_v3_cpu_runtime_lock(**kwargs)

    assert first == second
    assert calls[:2] == [
        ("manifest", values["manifest_raw"], _sha(values["manifest_raw"])),
        ("receipt", values["receipt_raw"], _sha(values["receipt_raw"])),
    ]
    lock = runtime_lock.parse_cpu_runtime_lock(
        first.runtime_lock_bytes,
        expected_file_sha256=first.runtime_lock_sha256,
    )
    assert lock["resolution"]["direct_requirements"] == [
        "continual-foragax==0.55.0",
        "jax==0.11.0",
        "jaxlib==0.11.0",
    ]
    assert lock["packages"][0]["selected_extras"] == ["feature"]
    requirement = lock["packages"][0]["wheels"][0]["metadata"]["requires_dist"][0]
    assert requirement == {
        "active": True,
        "marker": None,
        "name": "jax",
        "raw": "jax==0.11.0",
        "selected_version": "0.11.0",
    }
    assert all(value is False for value in lock["claims"].values())

    cas = issuer.parse_cpu_runtime_wheelhouse_cas_manifest(
        first.cas_manifest_bytes,
        expected_file_sha256=first.cas_manifest_sha256,
    )
    assert cas["entry_count"] == 3
    assert cas["wheel_inventory_sha256"] == lock["closure"]["wheel_inventory_sha256"]
    by_name = {entry["name"]: entry for entry in cas["entries"]}
    continual = by_name["continual-foragax"]
    assert continual["archive_name"] == f"{continual['sha256']}.whl"
    assert continual["cas_key"].endswith("/" + continual["filename"])
    assert all(value is False for value in cas["claims"].values())
    assert lock["wheelhouse"]["manifest"]["sha256"] == first.cas_manifest_sha256
    assert first.root_pin_count == 3


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda values: values["receipt"]["capture_manifest"].update(
                {"body_sha256": _sha("different-body")}
            ),
            "capture-manifest body",
        ),
        (
            lambda values: values["receipt"].update({"root_requirements": ["jax==0.11.0"]}),
            "root requirements",
        ),
        (
            lambda values: values["receipt"]["target"].update({"platform": "other"}),
            "target",
        ),
        (
            lambda values: values["manifest"]["wheels"][0].update({"size_bytes": 99}),
            "wheel identity",
        ),
    ],
)
def test_issuer_rejects_manifest_receipt_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    match: str,
) -> None:
    values = _fixture_values()
    mutate(values)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match=match):
        _issue(monkeypatch, values)


def test_issuer_rejects_duplicate_source_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _fixture_values()
    wheels = values["manifest"]["wheels"]
    wheels[1]["origin_url"] = wheels[0]["origin_url"]
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="source URL"):
        _issue(monkeypatch, values)


def test_issuer_preserves_case_sensitive_dist_info_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    package = values["receipt"]["packages"][0]
    dist_info = "Continual_Foragax-0.55.0.dist-info"
    package["dist_info_directory"] = dist_info
    package["metadata"]["path"] = f"{dist_info}/METADATA"
    package["wheel"]["path"] = f"{dist_info}/WHEEL"
    package["record"]["path"] = f"{dist_info}/RECORD"
    artifacts = _issue(monkeypatch, values)
    lock = runtime_lock.parse_cpu_runtime_lock(
        artifacts.runtime_lock_bytes,
        expected_file_sha256=artifacts.runtime_lock_sha256,
    )
    selected = lock["packages"][0]["wheels"][0]
    assert selected["metadata"]["path"] == f"{dist_info}/METADATA"
    assert selected["wheel"]["path"] == f"{dist_info}/WHEEL"
    assert selected["record"]["path"] == f"{dist_info}/RECORD"


def test_issuer_preserves_null_requires_python(monkeypatch: pytest.MonkeyPatch) -> None:
    values = _fixture_values()
    values["receipt"]["packages"][1]["metadata"]["requires_python"] = None
    artifacts = _issue(monkeypatch, values)
    lock = runtime_lock.parse_cpu_runtime_lock(
        artifacts.runtime_lock_bytes,
        expected_file_sha256=artifacts.runtime_lock_sha256,
    )
    assert lock["packages"][1]["wheels"][0]["metadata"]["requires_python"] is None


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda values: values["receipt"]["closure"].update({"edges": []}),
            "unconditional",
        ),
        (
            lambda values: values["receipt"]["closure"]["activated_extras"].update(
                {"continual-foragax": []}
            ),
            "active contexts|activated extras",
        ),
        (
            lambda values: values["receipt"]["closure"]["edges"][0].update(
                {"active_contexts": ["feature"]}
            ),
            "unconditional contexts",
        ),
    ],
)
def test_issuer_rejects_nonreconstructible_marker_or_extra_activity(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    match: str,
) -> None:
    values = _fixture_values()
    mutate(values)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match=match):
        _issue(monkeypatch, values)


def test_issuer_rejects_raw_structured_marker_laundering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    raw = "jax==0.11.0; python_version >= '3.12'"
    structured_marker = 'python_version < "3.12"'
    metadata_requirement = values["receipt"]["packages"][0]["metadata"]["requires_dist"][0]
    edge_requirement = values["receipt"]["closure"]["edges"][0]["requirement"]
    for requirement in (metadata_requirement, edge_requirement):
        requirement["raw"] = raw
        requirement["marker"] = structured_marker
    with pytest.raises(
        issuer.ForagerMatchedV3CpuRuntimeLockIssuerError,
        match="raw marker differs",
    ):
        _issue(monkeypatch, values)


def test_issuer_accepts_safe_marker_quote_and_redundant_group_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    raw = "jax==0.11.0; (python_version >= '3.12') and extra == 'feature'"
    structured_marker = 'python_version >= "3.12" and extra == "feature"'
    metadata_requirement = values["receipt"]["packages"][0]["metadata"]["requires_dist"][0]
    edge_requirement = values["receipt"]["closure"]["edges"][0]["requirement"]
    for requirement in (metadata_requirement, edge_requirement):
        requirement["raw"] = raw
        requirement["marker"] = structured_marker
    values["receipt"]["closure"]["edges"][0]["active_contexts"] = ["feature"]
    artifacts = _issue(monkeypatch, values)
    lock = runtime_lock.parse_cpu_runtime_lock(
        artifacts.runtime_lock_bytes,
        expected_file_sha256=artifacts.runtime_lock_sha256,
    )
    requirement = lock["packages"][0]["wheels"][0]["metadata"]["requires_dist"][0]
    assert requirement["marker"] == "(python_version >= '3.12') and extra == 'feature'"
    assert requirement["active"] is True


def test_issuer_rejects_raw_structured_root_laundering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    raw = "continual-foragax[feature]>=0; python_version < '3.0'"
    values["manifest"]["root_requirements"][0] = raw
    values["receipt"]["root_requirements"][0] = raw
    values["receipt"]["closure"]["root_requirements"][0]["raw"] = raw
    with pytest.raises(
        issuer.ForagerMatchedV3CpuRuntimeLockIssuerError,
        match="specifier differs|structured marker|canonical structured pin",
    ):
        _issue(monkeypatch, values)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda values: values["receipt"]["closure"]["root_requirements"][0].update(
                {"specifier": ">=0.55.0"}
            ),
            "specifier differs|exactly pin",
        ),
        (
            lambda values: values["receipt"]["archive"].update({"members": []}),
            "archive mapping",
        ),
    ],
)
def test_issuer_rejects_nonreconstructible_root_or_archive_fact(
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    match: str,
) -> None:
    values = _fixture_values()
    mutate(values)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match=match):
        _issue(monkeypatch, values)


def test_envelope_must_exactly_bind_inputs_and_root_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture_values()
    _patch_parsers(monkeypatch, values)
    envelope_raw, _ = _envelope(values)
    envelope = json.loads(envelope_raw)
    envelope["input_bindings"]["wheelhouse_receipt_sha256"] = _sha("other-receipt")
    _set_body(envelope, "envelope_body_sha256")
    changed_raw = _canonical(envelope)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="receipt"):
        issuer.issue_matched_v3_cpu_runtime_lock(
            capture_manifest_raw=values["manifest_raw"],
            expected_capture_manifest_sha256=_sha(values["manifest_raw"]),
            wheelhouse_receipt_raw=values["receipt_raw"],
            expected_wheelhouse_receipt_sha256=_sha(values["receipt_raw"]),
            issuance_envelope_raw=changed_raw,
            expected_issuance_envelope_sha256=_sha(changed_raw),
        )

    envelope_raw, _ = _envelope(values)
    envelope = json.loads(envelope_raw)
    envelope["root_pin_inventory"]["pins"][0]["selected_extras"] = []
    pins = envelope["root_pin_inventory"]["pins"]
    envelope["root_pin_inventory"]["inventory_sha256"] = _sha(_canonical({"pins": pins}))
    _set_body(envelope, "envelope_body_sha256")
    changed_raw = _canonical(envelope)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="root-pin"):
        issuer.issue_matched_v3_cpu_runtime_lock(
            capture_manifest_raw=values["manifest_raw"],
            expected_capture_manifest_sha256=_sha(values["manifest_raw"]),
            wheelhouse_receipt_raw=values["receipt_raw"],
            expected_wheelhouse_receipt_sha256=_sha(values["receipt_raw"]),
            issuance_envelope_raw=changed_raw,
            expected_issuance_envelope_sha256=_sha(changed_raw),
        )


@pytest.mark.parametrize("tamper", ["source_url", "archive_size"])
def test_cas_parser_rejects_nonreconstructible_source_or_archive(
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    artifacts = _issue(monkeypatch)
    manifest = json.loads(artifacts.cas_manifest_bytes)
    if tamper == "source_url":
        manifest["entries"][0]["source_url"] = "not-a-url"
        manifest["entry_inventory_sha256"] = _sha(_canonical({"entries": manifest["entries"]}))
    else:
        manifest["source_archive"]["size_bytes"] = 1
    _set_body(manifest, "manifest_body_sha256")
    raw = _canonical(manifest)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError):
        issuer.parse_cpu_runtime_wheelhouse_cas_manifest(
            raw,
            expected_file_sha256=_sha(raw),
        )


@pytest.mark.parametrize("tamper", ["upstream", "overlay", "solver"])
def test_envelope_parser_rejects_policy_substitution(tamper: str) -> None:
    values = _fixture_values()
    raw, _ = _envelope(values)
    envelope = json.loads(raw)
    if tamper == "upstream":
        envelope["upstream"] = {}
    elif tamper == "overlay":
        envelope["overlay_delta"]["source_builds_allowed"] = True
        _set_body(envelope["overlay_delta"], "overlay_body_sha256")
    else:
        envelope["solver_provenance"]["trusted_for_acceptance"] = True
    _set_body(envelope, "envelope_body_sha256")
    changed = _canonical(envelope)
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError):
        issuer.parse_cpu_runtime_lock_issuance_envelope(
            changed,
            expected_file_sha256=_sha(changed),
        )


def test_production_gate_calls_frozen_validator_and_requires_exact_shape_and_root_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _issue(monkeypatch)
    generic_lock = runtime_lock.parse_cpu_runtime_lock(
        artifacts.runtime_lock_bytes,
        expected_file_sha256=artifacts.runtime_lock_sha256,
    )
    generic_cas = issuer.parse_cpu_runtime_wheelhouse_cas_manifest(
        artifacts.cas_manifest_bytes,
        expected_file_sha256=artifacts.cas_manifest_sha256,
    )
    expected_pin_sha = _sha("production-root-pin-inventory")
    expected_selected_sha = _sha("production-selected-wheel-inventory")
    expected_lock_sha = _sha("production-final-union-lock")
    expected_lock_size = 112_074
    artifacts = dataclasses.replace(
        artifacts,
        root_pin_count=issuer.PRODUCTION_ROOT_PIN_COUNT,
        root_pin_inventory_sha256=expected_pin_sha,
    )
    monkeypatch.setattr(
        issuer,
        "issue_matched_v3_cpu_runtime_lock",
        lambda **kwargs: artifacts,
    )
    production = {
        "packages": [{"name": f"package-{index}"} for index in range(104)],
        "target": {
            "libc_family": "glibc",
            "libc_version": "2.28",
            "platform": "linux-amd64",
            "python_version": "3.12.3",
        },
        "wheelhouse": copy.deepcopy(generic_lock["wheelhouse"]),
    }
    production["wheelhouse"]["manifest"].update(
        {
            "entry_count": 104,
            "size_bytes": len(artifacts.cas_manifest_bytes),
            "total_bytes": generic_cas["total_bytes"],
        }
    )
    calls: list[dict[str, Any]] = []

    def validate(value: Any) -> dict[str, Any]:
        calls.append(value)
        return copy.deepcopy(production)

    monkeypatch.setattr(
        runtime_lock,
        "validate_production_cpu_runtime_lock",
        validate,
    )
    monkeypatch.setattr(
        runtime_lock,
        "parse_cpu_runtime_lock",
        lambda raw, *, expected_file_sha256: copy.deepcopy(production),
    )
    monkeypatch.setattr(
        issuer,
        "parse_cpu_runtime_lock_issuance_envelope",
        lambda raw, *, expected_file_sha256: {
            "input_bindings": {
                "capture_manifest_sha256": artifacts.capture_manifest_sha256,
                "wheelhouse_archive_inventory_sha256": generic_cas["source_archive"][
                    "inventory_sha256"
                ],
                "wheelhouse_archive_sha256": generic_cas["source_archive"]["sha256"],
                "wheelhouse_archive_size_bytes": generic_cas["source_archive"]["size_bytes"],
                "wheelhouse_receipt_body_sha256": generic_cas["source_receipt"]["body_sha256"],
                "wheelhouse_receipt_sha256": artifacts.wheelhouse_receipt_sha256,
            },
            "resolution_inventory": {
                "lock_sha256": expected_lock_sha,
                "lock_size_bytes": expected_lock_size,
                "selected_wheel_count": 104,
                "selected_wheel_inventory_sha256": expected_selected_sha,
            },
            "root_pin_inventory": {
                "inventory_sha256": expected_pin_sha,
                "pin_count": issuer.PRODUCTION_ROOT_PIN_COUNT,
            },
        },
    )
    monkeypatch.setattr(
        issuer,
        "parse_cpu_runtime_wheelhouse_cas_manifest",
        lambda raw, *, expected_file_sha256: {
            "entry_count": 104,
            "manifest_body_sha256": generic_cas["manifest_body_sha256"],
            "source_archive": copy.deepcopy(generic_cas["source_archive"]),
            "source_receipt": copy.deepcopy(generic_cas["source_receipt"]),
            "total_bytes": generic_cas["total_bytes"],
            "wheel_inventory_sha256": generic_cas["wheel_inventory_sha256"],
        },
    )
    assert (
        issuer.validate_production_cpu_runtime_lock_issuance(
            artifacts,
            expected_root_pin_inventory_sha256=expected_pin_sha,
            expected_selected_wheel_inventory_sha256=expected_selected_sha,
            expected_resolution_lock_sha256=expected_lock_sha,
            expected_resolution_lock_size_bytes=expected_lock_size,
        )
        == artifacts
    )
    assert calls == [production]

    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="root-pin"):
        issuer.validate_production_cpu_runtime_lock_issuance(
            artifacts,
            expected_root_pin_inventory_sha256=_sha("wrong-pin-inventory"),
            expected_selected_wheel_inventory_sha256=expected_selected_sha,
            expected_resolution_lock_sha256=expected_lock_sha,
            expected_resolution_lock_size_bytes=expected_lock_size,
        )
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="target"):
        issuer.validate_production_cpu_runtime_lock_issuance(
            artifacts,
            expected_root_pin_inventory_sha256=expected_pin_sha,
            expected_selected_wheel_inventory_sha256=_sha("wrong-selected-inventory"),
            expected_resolution_lock_sha256=expected_lock_sha,
            expected_resolution_lock_size_bytes=expected_lock_size,
        )
    production["packages"].pop()
    with pytest.raises(issuer.ForagerMatchedV3CpuRuntimeLockIssuerError, match="104"):
        issuer.validate_production_cpu_runtime_lock_issuance(
            artifacts,
            expected_root_pin_inventory_sha256=expected_pin_sha,
            expected_selected_wheel_inventory_sha256=expected_selected_sha,
            expected_resolution_lock_sha256=expected_lock_sha,
            expected_resolution_lock_size_bytes=expected_lock_size,
        )


def test_production_gate_reissues_exact_retained_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _issue(monkeypatch)
    substituted = dataclasses.replace(
        artifacts,
        root_pin_count=issuer.PRODUCTION_ROOT_PIN_COUNT,
    )
    with pytest.raises(
        issuer.ForagerMatchedV3CpuRuntimeLockIssuerError,
        match="pure-content reissuance",
    ):
        issuer.validate_production_cpu_runtime_lock_issuance(
            substituted,
            expected_root_pin_inventory_sha256=artifacts.root_pin_inventory_sha256,
            expected_selected_wheel_inventory_sha256=_sha("selected"),
            expected_resolution_lock_sha256=_sha("lock"),
            expected_resolution_lock_size_bytes=1,
        )


def test_module_has_no_filesystem_network_or_wheel_reader_surface() -> None:
    source_path = Path(issuer.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    forbidden_imports = {"os", "pathlib", "socket", "subprocess", "tarfile", "zipfile"}
    imported = {
        alias.name.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module.partition(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert imported.isdisjoint(forbidden_imports)
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"open", "exec", "eval", "compile"}
        for node in ast.walk(tree)
    )
