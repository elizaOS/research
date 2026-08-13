"""Adversarial tests for the sealed matched-v3 OCI build context."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import pickle
import stat
import sys
from typing import Any

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_oci_build_context as context

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


def _sha(raw: bytes | str) -> str:
    payload = raw.encode("ascii") if isinstance(raw, str) else raw
    return hashlib.sha256(payload).hexdigest()


def _fixture() -> tuple[bytes, str, dict[str, Any], dict[str, bytes]]:
    inputs = {
        "inputs/external-foragax-source.v1.tar": b"external archive bytes",
        "inputs/local-alberta-source.v1.tar": b"local archive bytes",
        "inputs/wheelhouse.v1.tar": b"wheelhouse archive bytes",
    }
    generated = {
        "Dockerfile": (
            "FROM --platform=linux/amd64 " + context.plan_contract.BASE_IMAGE_REFERENCE + "\n"
        ),
        "generated/archive-checksums.txt": "archive checksums\n",
        "generated/elizaos-forager-sources.pth": "/source/external\n/source/local\n",
        "generated/materialize-wheelhouse.py": "raise SystemExit(0)\n",
        "generated/requirements.lock": "example==1 --hash=sha256:" + "b" * 64 + "\n",
        "generated/runtime-inventory.json": "{}\n",
        "generated/verify-sources.py": "raise SystemExit(0)\n",
        "generated/verify-runtime.py": "raise SystemExit(0)\n",
        "generated/wheel-map.json": "{}\n",
    }
    generated_records = [
        {
            "content": generated[path],
            "path": path,
            "sha256": _sha(generated[path]),
            "size_bytes": len(generated[path].encode("ascii")),
        }
        for path in sorted(generated, key=str.encode)
    ]
    context_records = [
        {
            "path": path,
            "role": {
                "inputs/external-foragax-source.v1.tar": "external_foragax_source",
                "inputs/local-alberta-source.v1.tar": "local_alberta_source",
                "inputs/wheelhouse.v1.tar": "wheelhouse_archive",
            }[path],
            "sha256": _sha(raw),
            "size_bytes": len(raw),
        }
        for path, raw in inputs.items()
    ]
    context_records.sort(key=lambda item: item["path"].encode("ascii"))
    external_sha = _sha(inputs["inputs/external-foragax-source.v1.tar"])
    local_sha = _sha(inputs["inputs/local-alberta-source.v1.tar"])
    wheelhouse_sha = _sha(inputs["inputs/wheelhouse.v1.tar"])
    plan = {
        "base_image": {
            "manifest_digest": context.plan_contract.BASE_IMAGE_MANIFEST_DIGEST,
            "platform": context.plan_contract.BASE_IMAGE_PLATFORM,
            "reference": context.plan_contract.BASE_IMAGE_REFERENCE,
        },
        "bindings": {
            "runtime_lock": {"sha256": "c" * 64},
            "sources": [
                {"archive_sha256": external_sha, "role": "external_foragax"},
                {"archive_sha256": local_sha, "role": "local_alberta"},
            ],
            "wheelhouse": {
                "archive_sha256": wheelhouse_sha,
                "cas_manifest_sha256": "d" * 64,
            },
        },
        "build": {
            "command": [
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
            ],
            "context_inputs": context_records,
            "generated_files": generated_records,
        },
        "execution_toolchain": context._execution_toolchain(),
        "plan_body_sha256": "e" * 64,
        "schema_version": "alberta.forager_matched_v3.cpu_oci_build_plan.v1",
    }
    raw = _canonical({"fixture": "opaque plan bytes"})
    return raw, _sha(raw), plan, inputs


def _patch_parser(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    digest: str,
    plan: dict[str, Any],
) -> None:
    def parse(candidate: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert candidate == raw
        assert expected_file_sha256 == digest
        return copy.deepcopy(plan)

    monkeypatch.setattr(context.plan_contract, "parse_cpu_oci_build_plan", parse)


def _rehash_receipt(value: dict[str, Any]) -> tuple[bytes, str]:
    value["execution_projection_sha256"] = context._execution_projection_sha256(value)
    body = copy.deepcopy(value)
    body.pop("receipt_body_sha256", None)
    value["receipt_body_sha256"] = _sha(context._canonical_json(body))
    raw = context._canonical_json(value)
    return raw, _sha(raw)


def _retain(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[context.RetainedMatchedV3CpuOciBuildContext, Any]:
    raw, digest, plan, inputs = _fixture()
    _patch_parser(monkeypatch, raw, digest, plan)
    manager = context.retain_matched_v3_cpu_oci_build_context(
        plan_bytes=raw,
        expected_plan_sha256=digest,
        wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"],
        external_foragax_source_archive_bytes=(inputs["inputs/external-foragax-source.v1.tar"]),
        local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
    )
    return manager.__enter__(), manager


def test_context_is_exact_canonical_sealed_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, digest, plan, inputs = _fixture()
    _patch_parser(monkeypatch, raw, digest, plan)

    identities: list[tuple[str, bytes]] = []
    for _ in range(2):
        with context.retain_matched_v3_cpu_oci_build_context(
            plan_bytes=raw,
            expected_plan_sha256=digest,
            wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"],
            external_foragax_source_archive_bytes=(inputs["inputs/external-foragax-source.v1.tar"]),
            local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
        ) as retained:
            receipt = retained.reverify()
            archive = retained.read_context_bytes()
            identities.append((retained.archive_sha256, archive))
            assert retained.plan_sha256 == digest
            assert retained.member_count == 12
            assert receipt["archive"]["member_count"] == 12
            expected_paths = [
                "Dockerfile",
                "generated/archive-checksums.txt",
                "generated/elizaos-forager-sources.pth",
                "generated/materialize-wheelhouse.py",
                "generated/requirements.lock",
                "generated/runtime-inventory.json",
                "generated/verify-sources.py",
                "generated/verify-runtime.py",
                "generated/wheel-map.json",
                "inputs/external-foragax-source.v1.tar",
                "inputs/local-alberta-source.v1.tar",
                "inputs/wheelhouse.v1.tar",
            ]
            assert [item["path"] for item in receipt["members"]] == sorted(
                expected_paths,
                key=str.encode,
            )
            assert all(item["mode"] == "0444" for item in receipt["members"])
            metadata = os.stat(retained.proc_fd_path)
            assert stat.S_ISREG(metadata.st_mode)
            assert stat.S_IMODE(metadata.st_mode) == 0o400
            assert metadata.st_nlink == 0
            descriptor = retained.duplicate_readonly_descriptor()
            try:
                seals = fcntl.fcntl(descriptor, fcntl.F_GET_SEALS)
                required = sum(
                    getattr(fcntl, name)
                    for name in ("F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL")
                )
                assert seals & required == required
                assert os.read(descriptor, len(archive) + 1) == archive
            finally:
                os.close(descriptor)
    assert identities[0] == identities[1]


def test_context_releases_large_input_payloads_before_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, digest, plan, inputs = _fixture()
    _patch_parser(monkeypatch, raw, digest, plan)
    wheelhouse = inputs["inputs/wheelhouse.v1.tar"]
    baseline_references = sys.getrefcount(wheelhouse)
    manager = context.retain_matched_v3_cpu_oci_build_context(
        plan_bytes=raw,
        expected_plan_sha256=digest,
        wheelhouse_archive_bytes=wheelhouse,
        external_foragax_source_archive_bytes=(inputs["inputs/external-foragax-source.v1.tar"]),
        local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
    )
    assert sys.getrefcount(wheelhouse) > baseline_references

    with manager as retained:
        assert retained.reverify()["archive"]["member_count"] == 12
        assert sys.getrefcount(wheelhouse) == baseline_references
        generator_locals = manager.gen.gi_frame.f_locals
        assert {
            "created_receipt",
            "external_foragax_source_archive_bytes",
            "local_alberta_source_archive_bytes",
            "member_records",
            "members",
            "plan",
            "plan_bytes",
            "wheelhouse_archive_bytes",
        }.isdisjoint(generator_locals)


def test_context_fd_factories_close_untransferred_descriptors_on_baseexception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_descriptors: list[int] = []
    real_memfd_create = context.os.memfd_create

    def capture_memfd(*args: Any, **kwargs: Any) -> int:
        descriptor = real_memfd_create(*args, **kwargs)
        created_descriptors.append(descriptor)
        return descriptor

    with monkeypatch.context() as isolated:
        isolated.setattr(context.os, "memfd_create", capture_memfd)
        isolated.setattr(
            context.os,
            "get_inheritable",
            lambda _descriptor: (_ for _ in ()).throw(
                KeyboardInterrupt("synthetic create validation interruption")
            ),
        )
        with pytest.raises(KeyboardInterrupt, match="create validation"):
            context._create_memfd()
    assert len(created_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(created_descriptors[0])

    writable = context._create_memfd()
    os.write(writable, b"x")
    reopened_descriptors: list[int] = []
    real_open = context.os.open

    def capture_reopen(path: Any, *args: Any, **kwargs: Any) -> int:
        descriptor = real_open(path, *args, **kwargs)
        if str(path).startswith("/proc/self/fd/"):
            reopened_descriptors.append(descriptor)
        return descriptor

    try:
        with monkeypatch.context() as isolated:
            isolated.setattr(context.os, "open", capture_reopen)
            isolated.setattr(
                context.os,
                "get_inheritable",
                lambda descriptor: (
                    (_ for _ in ()).throw(
                        KeyboardInterrupt("synthetic reopen validation interruption")
                    )
                    if descriptor in reopened_descriptors
                    else os.get_inheritable(descriptor)
                ),
            )
            with pytest.raises(KeyboardInterrupt, match="reopen validation"):
                context._seal_and_reopen(writable, expected_size=1)
        assert len(reopened_descriptors) == 1
        with pytest.raises(OSError):
            os.fstat(reopened_descriptors[0])
    finally:
        os.close(writable)


def test_context_cross_checks_every_raw_input_before_creating_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, digest, plan, inputs = _fixture()
    _patch_parser(monkeypatch, raw, digest, plan)
    with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="identity differs"):
        context.retain_matched_v3_cpu_oci_build_context(
            plan_bytes=raw,
            expected_plan_sha256=digest,
            wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"] + b"substitution",
            external_foragax_source_archive_bytes=(inputs["inputs/external-foragax-source.v1.tar"]),
            local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
        ).__enter__()


@pytest.mark.parametrize("mutation", ["extra", "missing", "duplicate"])
def test_context_requires_the_exact_twelve_planned_paths(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    raw, digest, plan, inputs = _fixture()
    if mutation == "extra":
        plan["build"]["generated_files"].append(
            {
                "content": "extra\n",
                "path": "generated/extra.txt",
                "sha256": _sha("extra\n"),
                "size_bytes": 6,
            }
        )
    elif mutation == "missing":
        plan["build"]["generated_files"].pop()
    else:
        plan["build"]["generated_files"].append(copy.deepcopy(plan["build"]["generated_files"][0]))
    _patch_parser(monkeypatch, raw, digest, plan)
    with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="exact planned"):
        context.retain_matched_v3_cpu_oci_build_context(
            plan_bytes=raw,
            expected_plan_sha256=digest,
            wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"],
            external_foragax_source_archive_bytes=(inputs["inputs/external-foragax-source.v1.tar"]),
            local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
        ).__enter__()


def test_context_receipt_and_capability_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(retained)
        receipt = bytearray(retained.receipt_bytes)
        receipt[-2] ^= 1
        with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError):
            context.parse_matched_v3_cpu_oci_build_context_receipt(
                bytes(receipt), expected_receipt_sha256=retained.receipt_sha256
            )
    finally:
        manager.__exit__(None, None, None)
    assert retained.closed
    with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="closed"):
        retained.read_context_bytes()


@pytest.mark.parametrize(
    ("label", "member_path"),
    [
        (
            "io.elizaos.alberta.forager-matched-v3.external-source-sha256",
            "inputs/external-foragax-source.v1.tar",
        ),
        (
            "io.elizaos.alberta.forager-matched-v3.local-source-sha256",
            "inputs/local-alberta-source.v1.tar",
        ),
        (
            "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256",
            "inputs/wheelhouse.v1.tar",
        ),
    ],
)
def test_context_receipt_cross_binds_payload_labels_to_member_bytes(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    member_path: str,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        expected_member = next(item for item in receipt["members"] if item["path"] == member_path)
        assert receipt["expected_image_labels"][label] == expected_member["sha256"]
        receipt["expected_image_labels"][label] = "f" * 64
        if receipt["expected_image_labels"][label] == expected_member["sha256"]:
            receipt["expected_image_labels"][label] = "e" * 64
        raw, digest = _rehash_receipt(receipt)
        with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="member"):
            context.parse_matched_v3_cpu_oci_build_context_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("field", ["docker_sha256", "path", "environment"])
def test_context_receipt_rejects_resealed_execution_toolchain_substitution(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        if field == "docker_sha256":
            receipt["execution_toolchain"]["docker_cli"]["sha256"] = "f" * 64
        elif field == "path":
            receipt["execution_toolchain"]["buildx_plugin"]["path"] = "docker"
        else:
            receipt["execution_toolchain"]["environment"]["fixed"]["PATH"] = "/tmp"
        raw, digest = _rehash_receipt(receipt)
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="execution toolchain|environment",
        ):
            context.parse_matched_v3_cpu_oci_build_context_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_context_receipt_rejects_fully_resealed_alternate_base_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        alternate_digest = "sha256:" + "e" * 64
        receipt["base_image"]["manifest_digest"] = alternate_digest
        receipt["base_image"]["reference"] = "docker.io/library/python@" + alternate_digest
        receipt["expected_image_labels"]["io.elizaos.alberta.forager-matched-v3.base-manifest"] = (
            alternate_digest
        )
        raw, digest = _rehash_receipt(receipt)
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="base digest reference or platform",
        ):
            context.parse_matched_v3_cpu_oci_build_context_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_context_receipt_rejects_fully_resealed_alternate_plan_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        receipt["plan"]["schema_version"] = "attacker.schema.v999"
        raw, digest = _rehash_receipt(receipt)
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="plan schema differs",
        ):
            context.parse_matched_v3_cpu_oci_build_context_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_retained_context_reverify_cross_binds_stored_plan_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        receipt["plan"]["full_file_sha256"] = "f" * 64
        if receipt["plan"]["full_file_sha256"] == retained.plan_sha256:
            receipt["plan"]["full_file_sha256"] = "e" * 64
        raw, digest = _rehash_receipt(receipt)
        retained._receipt_bytes = raw
        retained._receipt_sha256 = digest
        with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="plan identity"):
            retained.reverify()
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize(
    "label",
    [
        "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256",
        "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256",
    ],
)
def test_retained_context_reverify_cross_binds_stored_label_inventory(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        receipt["expected_image_labels"][label] = "f" * 64
        raw, digest = _rehash_receipt(receipt)
        retained._receipt_bytes = raw
        retained._receipt_sha256 = digest
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="execution projection|label inventory",
        ):
            retained.reverify()
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("mutation", ["reference", "platform_and_command", "routing"])
def test_retained_context_reverify_cross_binds_complete_execution_projection(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        if mutation == "reference":
            receipt["base_image"]["reference"] = "docker.io/library/python@sha256:" + "f" * 64
        elif mutation == "platform_and_command":
            receipt["base_image"]["platform"] = "linux/arm64"
            command = receipt["build_command_template"]
            command[command.index("--platform=linux/amd64")] = "--platform=linux/arm64"
        else:
            receipt["build_command_template"][1] = "--host=tcp://remote.invalid:2376"
        raw, digest = _rehash_receipt(receipt)
        retained._receipt_bytes = raw
        retained._receipt_sha256 = digest
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="execution projection|base digest reference|build command",
        ):
            retained.reverify()
    finally:
        manager.__exit__(None, None, None)


def test_plan_parser_receives_both_exact_caller_pins(monkeypatch: pytest.MonkeyPatch) -> None:
    raw, digest, plan, inputs = _fixture()
    _patch_parser(monkeypatch, raw, digest, plan)
    with context.retain_matched_v3_cpu_oci_build_context(
        plan_bytes=raw,
        expected_plan_sha256=digest,
        wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"],
        external_foragax_source_archive_bytes=inputs["inputs/external-foragax-source.v1.tar"],
        local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
    ) as retained:
        assert retained.plan_sha256 == digest


def test_context_ustar_headers_order_payload_padding_and_tail_are_independently_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    try:
        receipt = retained.receipt()
        raw = retained.read_context_bytes()
        offset = 0
        observed_paths: list[str] = []
        for member in receipt["members"]:
            header = raw[offset : offset + 512]
            assert len(header) == 512
            name = header[0:100].split(b"\0", 1)[0].decode("ascii")
            prefix = header[345:500].split(b"\0", 1)[0].decode("ascii")
            path = f"{prefix}/{name}" if prefix else name
            observed_paths.append(path)
            assert path == member["path"]
            assert header[100:108] == b"0000444\0"
            assert header[108:116] == b"0000000\0"
            assert header[116:124] == b"0000000\0"
            assert header[124:136] == f"{member['size_bytes']:011o}".encode("ascii") + b"\0"
            assert header[136:148] == b"00000000000\0"
            assert header[156:157] == b"0"
            assert header[257:263] == b"ustar\0"
            assert header[263:265] == b"00"
            checksum_header = bytearray(header)
            supplied_checksum = int(header[148:154], 8)
            checksum_header[148:156] = b"        "
            assert supplied_checksum == sum(checksum_header)
            offset += 512
            size = member["size_bytes"]
            payload = raw[offset : offset + size]
            assert _sha(payload) == member["sha256"]
            offset += size
            padding = (-size) % 512
            assert raw[offset : offset + padding] == bytes(padding)
            offset += padding
        assert observed_paths == sorted(observed_paths, key=str.encode)
        assert raw[offset : offset + 1024] == bytes(1024)
        offset += 1024
        assert raw[offset:] == bytes(len(raw) - offset)
        assert len(raw) % 10_240 == 0
    finally:
        manager.__exit__(None, None, None)


def test_context_capability_rejects_pid_and_descriptor_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    descriptor = int(retained.proc_fd_path.rsplit("/", 1)[1])
    real_getpid = os.getpid
    try:
        with monkeypatch.context() as isolated:
            isolated.setattr(context.os, "getpid", lambda: real_getpid() + 1)
            with pytest.raises(context.ForagerMatchedV3CpuOciBuildContextError, match="PID"):
                retained.reverify()
        assert retained.closed
        with pytest.raises(OSError):
            os.fstat(descriptor)
    finally:
        manager.__exit__(None, None, None)

    retained, manager = _retain(monkeypatch)
    descriptor = int(retained.proc_fd_path.rsplit("/", 1)[1])
    replacement = os.open("/dev/null", os.O_RDONLY)
    try:
        os.dup2(replacement, descriptor)
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="identity|inaccessible",
        ):
            retained.reverify()
        assert retained.closed
    finally:
        os.close(replacement)
        os.close(descriptor)
        manager.__exit__(None, None, None)


def test_context_metadata_failure_closes_still_owned_descriptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager = _retain(monkeypatch)
    descriptor = int(retained.proc_fd_path.rsplit("/", 1)[1])
    with monkeypatch.context() as isolated:
        isolated.setattr(
            context.fcntl,
            "fcntl",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("synthetic fcntl failure")),
        )
        with pytest.raises(
            context.ForagerMatchedV3CpuOciBuildContextError,
            match="inaccessible",
        ):
            retained.reverify()
    assert retained.closed
    with pytest.raises(OSError):
        os.fstat(descriptor)
    manager.__exit__(None, None, None)


def test_no_context_publication_or_extraction_api_exists() -> None:
    public = set(context.__all__)
    assert not any("publish" in name or "extract" in name for name in public)
    assert "retain_matched_v3_cpu_oci_build_context" in public
    assert not hasattr(context.RetainedMatchedV3CpuOciBuildContext, "write_to")
