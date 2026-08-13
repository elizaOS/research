"""No-Docker tests for the explicit matched-v3 CPU OCI build executor."""

from __future__ import annotations

import copy
import hashlib
import inspect as inspect_module
import json
import os
import pickle
import signal
import stat
import sys
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import forager_matched_v3_cpu_oci_build_context as context
from alberta_framework.benchmarks import forager_matched_v3_cpu_oci_build_executor as executor

pytestmark = pytest.mark.unit


@pytest.fixture
def python_executable_descriptor() -> Iterator[int]:
    descriptor = os.open(
        sys.executable,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("ascii")
        + b"\n"
    )


def _sha(raw: bytes | str) -> str:
    payload = raw.encode("ascii") if isinstance(raw, str) else raw
    return hashlib.sha256(payload).hexdigest()


def _rehash_execution_receipt(value: dict[str, Any]) -> tuple[bytes, str]:
    body = copy.deepcopy(value)
    body.pop("receipt_body_sha256", None)
    value["receipt_body_sha256"] = _sha(executor._canonical_json(body))
    raw = executor._canonical_json(value)
    return raw, _sha(raw)


def _context_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[context.RetainedMatchedV3CpuOciBuildContext, Any, dict[str, Any]]:
    @dataclass(frozen=True)
    class SyntheticToolchain:
        docker_descriptor: int
        buildx_descriptor: int

        def descriptor_for(self, argv: tuple[str, ...]) -> int:
            if argv[0] == executor._DOCKER_CLI_PATH:
                return self.docker_descriptor
            if argv[0] == executor._BUILDX_PLUGIN_PATH:
                return self.buildx_descriptor
            raise AssertionError(f"unexpected synthetic toolchain argv: {argv!r}")

        def reverify(self, *, image_state_uncertain: bool) -> dict[str, Any]:
            del image_state_uncertain
            contract = executor._execution_toolchain_contract()
            return {
                key: copy.deepcopy(value)
                for key, value in contract.items()
                if key in {"buildx_plugin", "docker_cli", "docker_dynamic_runtime"}
            }

    @contextmanager
    def retain_synthetic_toolchain() -> Iterator[SyntheticToolchain]:
        docker_descriptor = os.open(
            sys.executable,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        buildx_descriptor = os.open(
            sys.executable,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            yield SyntheticToolchain(docker_descriptor, buildx_descriptor)
        finally:
            os.close(buildx_descriptor)
            os.close(docker_descriptor)

    monkeypatch.setattr(executor, "_retain_execution_toolchain", retain_synthetic_toolchain)
    inputs = {
        "inputs/external-foragax-source.v1.tar": b"external",
        "inputs/local-alberta-source.v1.tar": b"local",
        "inputs/wheelhouse.v1.tar": b"wheels",
    }
    generated = {
        "Dockerfile": "FROM scratch\n",
        "generated/archive-checksums.txt": "checksums\n",
        "generated/elizaos-forager-sources.pth": "sources\n",
        "generated/materialize-wheelhouse.py": "pass\n",
        "generated/requirements.lock": "requirements\n",
        "generated/runtime-inventory.json": "{}\n",
        "generated/verify-sources.py": "pass\n",
        "generated/verify-runtime.py": "pass\n",
        "generated/wheel-map.json": "{}\n",
    }
    external_sha = _sha(inputs["inputs/external-foragax-source.v1.tar"])
    local_sha = _sha(inputs["inputs/local-alberta-source.v1.tar"])
    wheel_sha = _sha(inputs["inputs/wheelhouse.v1.tar"])
    plan = {
        "base_image": {
            "manifest_digest": context.plan_contract.BASE_IMAGE_MANIFEST_DIGEST,
            "platform": context.plan_contract.BASE_IMAGE_PLATFORM,
            "reference": context.plan_contract.BASE_IMAGE_REFERENCE,
        },
        "bindings": {
            "runtime_lock": {"sha256": "b" * 64},
            "sources": [
                {"archive_sha256": external_sha, "role": "external_foragax"},
                {"archive_sha256": local_sha, "role": "local_alberta"},
            ],
            "wheelhouse": {
                "archive_sha256": wheel_sha,
                "cas_manifest_sha256": "c" * 64,
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
            "context_inputs": [
                {
                    "path": path,
                    "role": {
                        "inputs/external-foragax-source.v1.tar": "external_foragax_source",
                        "inputs/local-alberta-source.v1.tar": "local_alberta_source",
                        "inputs/wheelhouse.v1.tar": "wheelhouse_archive",
                    }[path],
                    "sha256": _sha(payload),
                    "size_bytes": len(payload),
                }
                for path, payload in sorted(inputs.items())
            ],
            "generated_files": [
                {
                    "content": generated[path],
                    "path": path,
                    "sha256": _sha(generated[path]),
                    "size_bytes": len(generated[path].encode("ascii")),
                }
                for path in sorted(generated, key=str.encode)
            ],
        },
        "execution_toolchain": executor._execution_toolchain_contract(),
        "plan_body_sha256": "d" * 64,
        "schema_version": "alberta.forager_matched_v3.cpu_oci_build_plan.v1",
    }
    plan_raw = _canonical({"opaque": "plan"})
    plan_sha = _sha(plan_raw)

    def parse(raw: bytes, *, expected_file_sha256: str) -> dict[str, Any]:
        assert raw == plan_raw
        assert expected_file_sha256 == plan_sha
        return copy.deepcopy(plan)

    monkeypatch.setattr(context.plan_contract, "parse_cpu_oci_build_plan", parse)
    manager = context.retain_matched_v3_cpu_oci_build_context(
        plan_bytes=plan_raw,
        expected_plan_sha256=plan_sha,
        wheelhouse_archive_bytes=inputs["inputs/wheelhouse.v1.tar"],
        external_foragax_source_archive_bytes=inputs["inputs/external-foragax-source.v1.tar"],
        local_alberta_source_archive_bytes=inputs["inputs/local-alberta-source.v1.tar"],
    )
    return manager.__enter__(), manager, plan


class FakeRunner:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[tuple[tuple[str, ...], int | None]] = []
        self.environments: list[dict[str, str]] = []
        self.image_id = "sha256:" + "f" * 64
        self.iid_payload = self.image_id.encode("ascii")
        self.base_returncode = 0
        self.base_platform = ("linux", "amd64")
        self.base_repository_digest = self.plan["base_image"]["reference"]
        self.build_returncode = 0
        self.build_timed_out = False
        self.build_output_limit_exceeded = False
        self.omit_iidfile = False
        self.consume_complete_context = True
        self.inspect_returncode = 0
        self.raise_same_class_on_built_inspect = False
        self.built_id = self.image_id
        self.built_platform = ("linux", "amd64")
        self.built_labels = self._expected_labels()
        self.context_sha256: str | None = None
        self.iidfile_path: Path | None = None
        self.after_build: Any = None
        self.builder_preflight_records = [self._builder_record()]
        self.builder_postflight_records = copy.deepcopy(self.builder_preflight_records)
        self.builder_preflight_raw: bytes | None = None
        self.builder_postflight_raw: bytes | None = None
        self.builder_query_count = 0

    def _builder_record(self) -> dict[str, Any]:
        return {
            "Current": True,
            "Driver": "docker",
            "Dynamic": False,
            "LastActivity": "2026-07-31T12:00:04Z",
            "Name": "default",
            "Nodes": [
                {
                    "Endpoint": "default",
                    "GCPolicy": [{"All": False, "KeepBytes": 1}],
                    "IDs": ["bounded-worker-id"],
                    "Labels": {"org.mobyproject.buildkit.worker.executor": "oci"},
                    "Name": "default",
                    "Platforms": ["linux/amd64", "linux/amd64/v2", "linux/amd64/v3"],
                    "Status": "running",
                    "Version": "v0.31.1",
                }
            ],
        }

    def _expected_labels(self) -> dict[str, str]:
        bindings = self.plan["bindings"]
        sources = {item["role"]: item for item in bindings["sources"]}
        return {
            "io.elizaos.alberta.forager-matched-v3.base-manifest": self.plan["base_image"][
                "manifest_digest"
            ],
            "io.elizaos.alberta.forager-matched-v3.cas-manifest-sha256": bindings["wheelhouse"][
                "cas_manifest_sha256"
            ],
            "io.elizaos.alberta.forager-matched-v3.external-source-sha256": sources[
                "external_foragax"
            ]["archive_sha256"],
            "io.elizaos.alberta.forager-matched-v3.local-source-sha256": sources["local_alberta"][
                "archive_sha256"
            ],
            "io.elizaos.alberta.forager-matched-v3.runtime-lock-sha256": bindings["runtime_lock"][
                "sha256"
            ],
            "io.elizaos.alberta.forager-matched-v3.wheelhouse-sha256": bindings["wheelhouse"][
                "archive_sha256"
            ],
        }

    def __call__(
        self,
        argv: tuple[str, ...],
        *,
        environment: Mapping[str, str],
        executable_descriptor: int,
        inherited_descriptors: tuple[int, ...],
        working_directory: str,
        stdin_descriptor: int | None,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
    ) -> executor.BoundedProcessResult:
        del timeout_seconds, stdout_limit_bytes, stderr_limit_bytes
        assert stat.S_ISREG(os.fstat(executable_descriptor).st_mode)
        assert len(inherited_descriptors) == 1
        assert stat.S_ISDIR(os.fstat(inherited_descriptors[0]).st_mode)
        assert working_directory == environment["TMPDIR"]
        self.environments.append(dict(environment))
        self.calls.append((argv, stdin_descriptor))
        if argv == executor._builder_ls_command():
            records = (
                self.builder_preflight_records
                if self.builder_query_count == 0
                else self.builder_postflight_records
            )
            raw = (
                self.builder_preflight_raw
                if self.builder_query_count == 0
                else self.builder_postflight_raw
            )
            self.builder_query_count += 1
            return executor.BoundedProcessResult(
                0,
                raw if raw is not None else b"".join(_canonical(record) for record in records),
                b"",
            )
        if argv == executor._inspect_command(self.plan["base_image"]["reference"]):
            base = {
                "Architecture": self.base_platform[1],
                "Config": {"Labels": {}},
                "Id": "sha256:" + "1" * 64,
                "Os": self.base_platform[0],
                "RepoDigests": [self.base_repository_digest],
            }
            return executor.BoundedProcessResult(self.base_returncode, _canonical(base), b"")
        if argv[:3] == tuple(self.plan["build"]["command"][:3]):
            assert stdin_descriptor is not None
            digest = hashlib.sha256()
            if self.consume_complete_context:
                while block := os.read(stdin_descriptor, 1024 * 1024):
                    digest.update(block)
            else:
                digest.update(os.read(stdin_descriptor, 1))
            self.context_sha256 = digest.hexdigest()
            iid_argument = next(item for item in argv if item.startswith("--iidfile="))
            self.iidfile_path = Path(iid_argument.split("=", 1)[1])
            if not self.omit_iidfile:
                self.iidfile_path.write_bytes(self.iid_payload)
            if self.after_build is not None:
                self.after_build()
            return executor.BoundedProcessResult(
                self.build_returncode,
                b"bounded build output",
                b"bounded build diagnostics",
                timed_out=self.build_timed_out,
                output_limit_exceeded=self.build_output_limit_exceeded,
            )
        assert argv == executor._inspect_command(self.image_id)
        if self.raise_same_class_on_built_inspect:
            raise executor.ForagerMatchedV3CpuOciBuildExecutorError("synthetic inspect error")
        built = {
            "Architecture": self.built_platform[1],
            "Config": {
                "Labels": self.built_labels,
                "User": "65532:65532",
                "WorkingDir": "/work",
            },
            "Id": self.built_id,
            "Os": self.built_platform[0],
            "RepoDigests": [],
        }
        return executor.BoundedProcessResult(self.inspect_returncode, _canonical(built), b"")


def test_executor_streams_only_context_and_returns_exact_nonauthorizing_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
            timeout_seconds=123,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        assert result.image_id == runner.image_id
        assert receipt["observation"]["container_image_built"] is True
        assert receipt["observation"]["daemon_egress_isolation_attested"] is False
        assert receipt["observation"]["builder_locality_postflight_matched"] is True
        assert receipt["observation"]["isolated_cli_environment_used"] is True
        assert receipt["observation"]["pinned_executable_descriptors_used"] is True
        assert (
            receipt["execution_toolchain"]["contract"] == executor._execution_toolchain_contract()
        )
        assert receipt["builder_locality"]["daemon_host"] == "unix:///var/run/docker.sock"
        assert (
            receipt["builder_locality"]["preflight"]["projection"]
            == receipt["builder_locality"]["postflight"]["projection"]
        )
        assert all(value is False for value in receipt["claims"].values())
        assert runner.context_sha256 == retained.archive_sha256
        assert len(runner.calls) == 5
        assert [argv for argv, _stdin in runner.calls].count(executor._builder_ls_command()) == 2
        assert all(
            argv[0] in {executor._DOCKER_CLI_PATH, executor._BUILDX_PLUGIN_PATH}
            for argv, _ in runner.calls
        )
        assert all(environment["PATH"] == "/usr/bin:/bin" for environment in runner.environments)
        assert all(
            environment["DOCKER_HOST"] == "unix:///var/run/docker.sock"
            for environment in runner.environments
        )
        assert all(
            environment["DOCKER_CONFIG"].startswith("/proc/self/fd/")
            for environment in runner.environments
        )
        assert all("--bootstrap" not in argv for argv, _stdin in runner.calls)
        build_argv, build_stdin = next(
            call for call in runner.calls if call[0][:3] == tuple(plan["build"]["command"][:3])
        )
        iid_argument = next(item for item in build_argv if item.startswith("--iidfile="))
        assert list(build_argv) == [*plan["build"]["command"][:-1], iid_argument, "-"]
        assert build_argv[-1] == "-"
        assert any(item.startswith("--iidfile=") for item in build_argv)
        assert build_stdin is not None
        assert all(stdin is None for argv, stdin in runner.calls if argv != build_argv)
        assert runner.iidfile_path is not None
        assert not runner.iidfile_path.parent.exists()
    finally:
        manager.__exit__(None, None, None)


def test_executor_requires_explicit_opt_in_before_any_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="explicit"):
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=None,
            )
        assert runner.calls == []
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize(
    "variable",
    [
        "BUILDKIT_HOST",
        "BUILDX_BUILDER",
        "BUILDX_CONFIG",
        "DOCKER_API_VERSION",
        "DOCKER_CLI_PLUGIN_EXTRA_DIRS",
        "DOCKER_CONFIG",
        "DOCKER_CONTEXT",
        "DOCKER_HOST",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
    ],
)
def test_executor_rejects_ambient_routing_before_any_docker_command(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    monkeypatch.setenv(variable, "attacker-controlled")
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="routing variables",
        ) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert caught.value.image_state_uncertain is False
        assert runner.calls == []
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda runner: runner.builder_preflight_records[0].__setitem__("Driver", "remote"),
        lambda runner: runner.builder_preflight_records[0].__setitem__(
            "Driver", "docker-container"
        ),
        lambda runner: runner.builder_preflight_records[0].__setitem__("Dynamic", True),
        lambda runner: runner.builder_preflight_records[0]["Nodes"].append(
            copy.deepcopy(runner.builder_preflight_records[0]["Nodes"][0])
        ),
        lambda runner: runner.builder_preflight_records.append(
            copy.deepcopy(runner.builder_preflight_records[0])
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Endpoint", "ssh://remote.invalid"
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Status", "stopped"
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Platforms", ["linux/arm64"]
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "ProxyConfig", {"HTTP_PROXY": "http://remote.invalid"}
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "DriverOpts", {"network": "host"}
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Flags", ["--allow-insecure-entitlement=network.host"]
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Files", {"buildkitd.toml": "attacker-controlled"}
        ),
        lambda runner: runner.builder_preflight_records[0]["Nodes"][0].__setitem__(
            "Err", "remote failure"
        ),
    ],
    ids=[
        "remote-driver",
        "docker-container-driver",
        "dynamic",
        "multi-node",
        "duplicate-default",
        "wrong-endpoint",
        "not-running",
        "wrong-platform",
        "proxy-config",
        "driver-options",
        "buildkit-flags",
        "buildkit-files",
        "node-error",
    ],
)
def test_executor_rejects_nonlocal_default_builder_before_build(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    mutation(runner)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert caught.value.image_state_uncertain is False
        assert not any(
            argv[:5] == tuple(plan["build"]["command"][:5]) for argv, _stdin in runner.calls
        )
    finally:
        manager.__exit__(None, None, None)


def test_executor_rejects_duplicate_key_builder_jsonl_before_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    runner.builder_preflight_raw = b'{"Name":"default","Name":"default"}\n'
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert caught.value.image_state_uncertain is False
        assert len(runner.calls) == 1
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("drift", ["endpoint", "platforms"])
def test_executor_rejects_postbuild_builder_locality_drift_as_uncertain(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    node = runner.builder_postflight_records[0]["Nodes"][0]
    if drift == "endpoint":
        node["Endpoint"] = "tcp://remote.invalid:1234"
    else:
        node["Platforms"].append("linux/arm64")
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert caught.value.image_state_uncertain is True
        assert "image state is uncertain" in str(caught.value)
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda runner: setattr(runner, "base_returncode", 1), "base image preflight"),
        (lambda runner: setattr(runner, "omit_iidfile", True), "iidfile"),
        (lambda runner: setattr(runner, "iid_payload", b"sha256:" + b"f" * 64 + b"\n"), "iidfile"),
        (lambda runner: setattr(runner, "build_timed_out", True), "timed out"),
        (lambda runner: setattr(runner, "build_returncode", 1), "Docker Buildx build"),
        (
            lambda runner: setattr(runner, "build_output_limit_exceeded", True),
            "output bound",
        ),
        (lambda runner: setattr(runner, "built_id", "sha256:" + "e" * 64), "image ID"),
        (lambda runner: setattr(runner, "built_platform", ("linux", "arm64")), "platform"),
        (lambda runner: runner.built_labels.pop(next(iter(runner.built_labels))), "labels"),
        (lambda runner: runner.built_labels.update({"unexpected": "label"}), "labels"),
        (lambda runner: setattr(runner, "inspect_returncode", 1), "built image inspect"),
        (
            lambda runner: setattr(runner, "raise_same_class_on_built_inspect", True),
            "synthetic inspect error",
        ),
        (
            lambda runner: setattr(runner, "consume_complete_context", False),
            "through EOF",
        ),
        (lambda runner: setattr(runner, "base_platform", ("linux", "arm64")), "platform"),
        (
            lambda runner: setattr(
                runner,
                "base_repository_digest",
                "docker.io/library/python@sha256:" + "e" * 64,
            ),
            "repository digest",
        ),
    ],
)
def test_executor_fails_closed_on_process_or_result_drift(
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    mutation(runner)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match=message,
        ) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        build_started = any(
            call[0][:5] == tuple(plan["build"]["command"][:5]) for call in runner.calls
        )
        assert caught.value.image_state_uncertain is build_started
        if runner.iidfile_path is not None:
            assert not runner.iidfile_path.parent.exists()
    finally:
        manager.__exit__(None, None, None)


def test_executor_receipt_tamper_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        tampered = bytearray(result.receipt_bytes)
        tampered[-2] ^= 1
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                bytes(tampered), expected_receipt_sha256=result.receipt_sha256
            )
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("attack", ["contract", "preflight", "environment"])
def test_execution_receipt_rejects_resealed_toolchain_substitution(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        if attack == "contract":
            receipt["execution_toolchain"]["contract"]["docker_cli"]["sha256"] = "f" * 64
        elif attack == "preflight":
            receipt["execution_toolchain"]["preflight"]["buildx_plugin"]["path"] = "docker"
        else:
            receipt["execution_toolchain"]["contract"]["environment"]["fixed"]["PATH"] = "/tmp"
        raw, digest = _rehash_execution_receipt(receipt)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="toolchain|environment|context",
        ):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_executor_snapshots_context_identity_before_image_creating_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    runner.after_build = retained.close
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        assert receipt["observation"]["container_image_built"] is True
        assert retained.closed
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("attack", ["base_command", "repository_digest", "coordinated"])
def test_execution_receipt_rejects_resealed_context_projection_attacks(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        alternate_digest = "sha256:" + "e" * 64
        alternate_reference = "docker.io/library/python@" + alternate_digest
        if attack == "base_command":
            receipt["base_preflight"]["command"][-1] = alternate_reference
        elif attack == "repository_digest":
            receipt["base_preflight"]["matched_repository_digest"] = "arbitrary"
        else:
            receipt["base_preflight"]["command"][-1] = alternate_reference
            receipt["base_preflight"]["matched_repository_digest"] = alternate_reference
            labels = receipt["image_inspect"]["expected_labels"]
            labels["io.elizaos.alberta.forager-matched-v3.base-manifest"] = alternate_digest
            receipt["context"]["expected_image_labels_sha256"] = _sha(
                executor._canonical_json({"expected_image_labels": labels})
            )
        raw, digest = _rehash_execution_receipt(receipt)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="context|base|repository|projection",
        ):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_execution_receipt_rejects_fully_resealed_alternate_base_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        embedded = json.loads(receipt["context"]["canonical_receipt"])
        alternate_digest = "sha256:" + "e" * 64
        alternate_reference = "docker.io/library/python@" + alternate_digest
        embedded["base_image"]["manifest_digest"] = alternate_digest
        embedded["base_image"]["reference"] = alternate_reference
        embedded["expected_image_labels"]["io.elizaos.alberta.forager-matched-v3.base-manifest"] = (
            alternate_digest
        )
        embedded["execution_projection_sha256"] = context._execution_projection_sha256(embedded)
        embedded_body = copy.deepcopy(embedded)
        embedded_body.pop("receipt_body_sha256")
        embedded["receipt_body_sha256"] = _sha(context._canonical_json(embedded_body))
        embedded_raw = context._canonical_json(embedded)
        receipt["context"]["canonical_receipt"] = embedded_raw.decode("ascii")
        receipt["context"]["canonical_receipt_size_bytes"] = len(embedded_raw)
        receipt["context"]["receipt_sha256"] = _sha(embedded_raw)
        receipt["context"]["execution_projection_sha256"] = embedded["execution_projection_sha256"]
        receipt["context"]["expected_image_labels_sha256"] = _sha(
            executor._canonical_json({"expected_image_labels": embedded["expected_image_labels"]})
        )
        receipt["base_preflight"]["command"][-1] = alternate_reference
        receipt["base_preflight"]["matched_repository_digest"] = alternate_reference
        receipt["image_inspect"]["expected_labels"][
            "io.elizaos.alberta.forager-matched-v3.base-manifest"
        ] = alternate_digest
        raw, digest = _rehash_execution_receipt(receipt)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="context|base",
        ):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_execution_receipt_rejects_fully_resealed_alternate_plan_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        embedded = json.loads(receipt["context"]["canonical_receipt"])
        embedded["plan"]["schema_version"] = "attacker.schema.v999"
        embedded["execution_projection_sha256"] = context._execution_projection_sha256(embedded)
        embedded_body = copy.deepcopy(embedded)
        embedded_body.pop("receipt_body_sha256")
        embedded["receipt_body_sha256"] = _sha(context._canonical_json(embedded_body))
        embedded_raw = context._canonical_json(embedded)
        receipt["context"]["canonical_receipt"] = embedded_raw.decode("ascii")
        receipt["context"]["canonical_receipt_size_bytes"] = len(embedded_raw)
        receipt["context"]["receipt_sha256"] = _sha(embedded_raw)
        receipt["context"]["execution_projection_sha256"] = embedded["execution_projection_sha256"]
        raw, digest = _rehash_execution_receipt(receipt)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="embedded canonical context receipt",
        ):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


@pytest.mark.parametrize("attack", ["command", "remote_projection", "postflight_drift"])
def test_execution_receipt_rejects_resealed_builder_locality_attacks(
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        result = executor.execute_matched_v3_cpu_oci_build(
            context_capability=retained,
            authorization=authorization,
        )
        receipt = executor.parse_matched_v3_cpu_oci_build_execution_receipt(
            result.receipt_bytes,
            expected_receipt_sha256=result.receipt_sha256,
        )
        locality = receipt["builder_locality"]
        if attack == "command":
            locality["preflight"]["command"][1] = "--host=tcp://remote.invalid:2376"
        elif attack == "remote_projection":
            for phase in ("preflight", "postflight"):
                record = locality[phase]
                record["projection"]["driver"] = "remote"
                projection_raw = executor._canonical_json(record["projection"])
                record["projection_sha256"] = _sha(projection_raw)
                record["projection_size_bytes"] = len(projection_raw)
        else:
            record = locality["postflight"]
            record["projection"]["node"]["platforms"].append("linux/arm64")
            projection_raw = executor._canonical_json(record["projection"])
            record["projection_sha256"] = _sha(projection_raw)
            record["projection_size_bytes"] = len(projection_raw)
        raw, digest = _rehash_execution_receipt(receipt)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="builder locality",
        ):
            executor.parse_matched_v3_cpu_oci_build_execution_receipt(
                raw,
                expected_receipt_sha256=digest,
            )
    finally:
        manager.__exit__(None, None, None)


def test_executor_surface_has_no_tag_publish_pull_or_prune_operation() -> None:
    public = set(executor.__all__)
    assert not any(
        token in name for name in public for token in ("tag", "publish", "pull", "prune")
    )
    parameters = inspect_module.signature(executor.execute_matched_v3_cpu_oci_build).parameters
    assert "process_runner" not in parameters


def test_authorization_is_context_bound_pid_bound_nonserializable_and_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    second, second_manager, _ = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    try:
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(authorization)
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="different"):
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=second,
                authorization=authorization,
            )
        assert authorization.consumed
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="consumed"):
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert runner.calls == []
        pid_authorization = executor.authorize_matched_v3_cpu_oci_build(
            context_capability=retained,
            exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
        )
        real_getpid = os.getpid
        with monkeypatch.context() as isolated:
            isolated.setattr(executor.os, "getpid", lambda: real_getpid() + 1)
            with pytest.raises(
                executor.ForagerMatchedV3CpuOciBuildExecutorError,
                match="PID change",
            ):
                executor.execute_matched_v3_cpu_oci_build(
                    context_capability=retained,
                    authorization=pid_authorization,
                )
    finally:
        second_manager.__exit__(None, None, None)
        manager.__exit__(None, None, None)


def test_authorization_is_atomically_single_use_across_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, _plan = _context_fixture(monkeypatch)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    barrier = threading.Barrier(3)
    results: list[str] = []
    result_lock = threading.Lock()

    def consume() -> None:
        barrier.wait()
        try:
            authorization._consume(retained)
        except executor.ForagerMatchedV3CpuOciBuildExecutorError as exc:
            result = str(exc)
        else:
            result = "success"
        with result_lock:
            results.append(result)

    threads = [threading.Thread(target=consume) for _ in range(2)]
    try:
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=5)
        assert all(not thread.is_alive() for thread in threads)
        assert results.count("success") == 1
        assert len(results) == 2
        assert any("already consumed" in result for result in results)
    finally:
        manager.__exit__(None, None, None)


def test_authorization_rejects_inexact_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, _plan = _context_fixture(monkeypatch)
    try:
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="differs"):
            executor.authorize_matched_v3_cpu_oci_build(
                context_capability=retained,
                exact_acknowledgement="yes",
            )
    finally:
        manager.__exit__(None, None, None)


def test_authorization_rejects_replacement_after_forced_object_id_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, _plan = _context_fixture(monkeypatch)
    monkeypatch.setattr(executor, "id", lambda _value: 1, raising=False)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    manager.__exit__(None, None, None)
    del retained

    replacement, replacement_manager, replacement_plan = _context_fixture(monkeypatch)
    runner = FakeRunner(replacement_plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    try:
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="different"):
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=replacement,
                authorization=authorization,
            )
        assert authorization.consumed
        assert authorization._context_capability is None
        assert runner.calls == []
    finally:
        replacement_manager.__exit__(None, None, None)


def test_iidfile_rejects_symlink_hardlink_and_directory_drift() -> None:
    with executor._exclusive_iidfile() as private:
        private.file_path.symlink_to("/dev/null")
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="metadata"):
            executor._stable_iidfile(private)
    with executor._exclusive_iidfile() as private:
        source = private.directory_path / "source.id"
        source.write_bytes(("sha256:" + "f" * 64).encode("ascii"))
        os.link(source, private.file_path)
        with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError, match="metadata"):
            executor._stable_iidfile(private)
        source.unlink()
    with executor._exclusive_iidfile() as private:
        moved = private.directory_path.with_name(private.directory_path.name + "-moved")
        os.rename(private.directory_path, moved)
        private.directory_path.mkdir(mode=0o700)
        private.file_path.write_bytes(("sha256:" + "f" * 64).encode("ascii"))
        try:
            with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError):
                executor._stable_iidfile(private)
        finally:
            private.file_path.unlink()
            private.directory_path.rmdir()
            os.rename(moved, private.directory_path)


def test_iidfile_early_setup_failure_removes_owned_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_mkdtemp = executor.tempfile.mkdtemp
    created_path: Path | None = None

    def capture_mkdtemp(*args: Any, **kwargs: Any) -> str:
        nonlocal created_path
        value = real_mkdtemp(*args, **kwargs)
        created_path = Path(value)
        return value

    def fail_chmod(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("synthetic early iidfile setup failure")

    monkeypatch.setattr(executor.tempfile, "mkdtemp", capture_mkdtemp)
    monkeypatch.setattr(executor.os, "chmod", fail_chmod)
    try:
        with pytest.raises(PermissionError, match="synthetic early"):
            with executor._exclusive_iidfile():
                raise AssertionError("unreachable")
        assert created_path is not None
        assert not created_path.exists()
    finally:
        if created_path is not None and created_path.exists():
            created_path.rmdir()


def test_low_level_eof_and_iidfile_os_errors_are_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = os.open("/dev/null", os.O_RDONLY)
    os.close(descriptor)
    with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError) as eof_error:
        executor._require_context_streamed_to_eof(descriptor, expected_size=1)
    assert eof_error.value.image_state_uncertain is True
    with executor._exclusive_iidfile() as private:
        with monkeypatch.context() as isolated:
            isolated.setattr(
                executor.os,
                "stat",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("synthetic")),
            )
            with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError) as iid_error:
                executor._stable_iidfile(private)
            assert iid_error.value.image_state_uncertain is True


def test_stable_iidfile_close_failure_never_masks_primary_read_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with executor._exclusive_iidfile() as private:
        private.file_path.write_bytes(("sha256:" + "f" * 64).encode("ascii"))
        real_open = executor.os.open
        real_read = executor.os.read
        real_close = executor.os.close
        stable_descriptors: list[int] = []

        def open_descriptor(path: Any, *args: Any, **kwargs: Any) -> int:
            descriptor = real_open(path, *args, **kwargs)
            if path == executor._IID_FILENAME:
                stable_descriptors.append(descriptor)
            return descriptor

        def fail_read(descriptor: int, _size: int) -> bytes:
            if descriptor in stable_descriptors:
                raise PermissionError("synthetic stable-read failure")
            return real_read(descriptor, _size)

        def fail_close(descriptor: int) -> None:
            real_close(descriptor)
            if descriptor in stable_descriptors:
                raise OSError("synthetic stable-descriptor close failure")

        with monkeypatch.context() as isolated:
            isolated.setattr(executor.os, "open", open_descriptor)
            isolated.setattr(executor.os, "read", fail_read)
            isolated.setattr(executor.os, "close", fail_close)
            with pytest.raises(
                executor.ForagerMatchedV3CpuOciBuildExecutorError,
                match="stable read failed",
            ) as caught:
                executor._stable_iidfile(private)
        assert caught.value.image_state_uncertain is True
        assert any("descriptor cleanup also failed" in note for note in caught.value.__notes__)


def test_postbuild_context_descriptor_cleanup_failure_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    duplicated: list[int] = []
    real_duplicate = context.RetainedMatchedV3CpuOciBuildContext.duplicate_readonly_descriptor
    real_close = os.close
    injected = False

    def duplicate(candidate: context.RetainedMatchedV3CpuOciBuildContext) -> int:
        descriptor = real_duplicate(candidate)
        duplicated.append(descriptor)
        return descriptor

    def close(descriptor: int) -> None:
        nonlocal injected
        if duplicated and descriptor == duplicated[-1] and not injected:
            injected = True
            real_close(descriptor)
            raise OSError("synthetic context descriptor close failure")
        real_close(descriptor)

    monkeypatch.setattr(
        context.RetainedMatchedV3CpuOciBuildContext,
        "duplicate_readonly_descriptor",
        duplicate,
    )
    monkeypatch.setattr(executor.os, "close", close)
    try:
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="descriptor cleanup failed",
        ) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert injected
        assert caught.value.image_state_uncertain is True
    finally:
        manager.__exit__(None, None, None)


def test_successful_build_iidfile_cleanup_failure_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    real_unlink = executor.os.unlink
    injected = False

    def unlink(path: str, *args: Any, **kwargs: Any) -> None:
        nonlocal injected
        real_unlink(path, *args, **kwargs)
        if path == executor._IID_FILENAME and not injected:
            injected = True
            raise PermissionError("synthetic iidfile unlink failure")

    monkeypatch.setattr(executor.os, "unlink", unlink)
    try:
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="iidfile cleanup failed",
        ) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert injected
        assert caught.value.image_state_uncertain is True
    finally:
        manager.__exit__(None, None, None)


def test_iidfile_cleanup_failure_never_masks_primary_uncertain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained, manager, plan = _context_fixture(monkeypatch)
    runner = FakeRunner(plan)
    runner.omit_iidfile = True
    monkeypatch.setattr(executor, "_default_process_runner", runner)
    authorization = executor.authorize_matched_v3_cpu_oci_build(
        context_capability=retained,
        exact_acknowledgement=executor.CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT,
    )
    real_unlink = executor.os.unlink
    injected = False

    def unlink(path: str, *args: Any, **kwargs: Any) -> None:
        nonlocal injected
        if path == executor._IID_FILENAME and not injected:
            injected = True
            raise PermissionError("synthetic iidfile cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(executor.os, "unlink", unlink)
    try:
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="iidfile is missing",
        ) as caught:
            executor.execute_matched_v3_cpu_oci_build(
                context_capability=retained,
                authorization=authorization,
            )
        assert injected
        assert caught.value.image_state_uncertain is True
        assert any("cleanup also failed" in note for note in caught.value.__notes__)
    finally:
        manager.__exit__(None, None, None)


def test_default_bounded_runner_handles_local_success_output_limit_and_descendants(
    python_executable_descriptor: int,
) -> None:
    success = executor._default_process_runner(
        (sys.executable, "-c", "import sys;print('out');print('err',file=sys.stderr)"),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=5,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )
    assert success.returncode == 0
    assert success.stdout == b"out\n"
    assert success.stderr == b"err\n"
    overflow = executor._default_process_runner(
        (sys.executable, "-c", "import sys;sys.stdout.write('x'*8192)"),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=5,
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert overflow.output_limit_exceeded is True
    assert len(overflow.stdout) == 32
    descendant = executor._default_process_runner(
        (
            sys.executable,
            "-c",
            (
                "import subprocess,sys;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
                "print(p.pid,flush=True)"
            ),
        ),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=1,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )
    assert descendant.timed_out is True
    child_pid = int(descendant.stdout.strip())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_default_runner_executes_the_open_descriptor_not_argv_or_path(
    python_executable_descriptor: int,
) -> None:
    result = executor._default_process_runner(
        ("docker-from-attacker-path", "-c", "print('descriptor-bound')"),
        environment={"PATH": "/tmp/attacker-first"},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=5,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )
    assert result.returncode == 0
    assert result.stdout == b"descriptor-bound\n"


def test_anchored_group_signals_before_proc_inspection_and_kills_after_scan_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Process:
        pid = 456_789

    events: list[tuple[str, int]] = []
    clock = [0.0]

    def monotonic() -> float:
        clock[0] += 0.2
        return clock[0]

    def killpg(_process_group: int, signal_number: int) -> None:
        events.append(("signal", signal_number))

    def inspect_members(_process_group: int, *, leader_pid: int) -> tuple[int, ...]:
        events.append(("inspect", leader_pid))
        raise executor.ForagerMatchedV3CpuOciBuildExecutorError("synthetic proc denial")

    monkeypatch.setattr(executor.time, "monotonic", monotonic)
    monkeypatch.setattr(executor.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(executor.os, "killpg", killpg)
    monkeypatch.setattr(executor, "_live_process_group_member_pids", inspect_members)
    with pytest.raises(
        executor.ForagerMatchedV3CpuOciBuildExecutorError,
        match="synthetic proc denial",
    ):
        executor._terminate_anchored_process_group(cast(Any, Process()))
    assert events[0] == ("signal", signal.SIGTERM)
    assert ("signal", signal.SIGKILL) in events
    assert events.index(("signal", signal.SIGKILL)) > next(
        index for index, event in enumerate(events) if event[0] == "inspect"
    )


def test_private_cli_state_is_descriptor_anchored_exact_and_removed() -> None:
    directory_path: Path
    descriptor: int
    with executor._exclusive_cli_state() as state:
        directory_path = state.directory_path
        descriptor = state.directory_descriptor
        expected_keys = set(executor._CLI_FIXED_ENVIRONMENT) | set(
            executor._CLI_PRIVATE_DIRECTORIES
        )
        assert set(state.environment) == expected_keys
        descriptor_root = f"/proc/self/fd/{descriptor}"
        for key, relative in executor._CLI_PRIVATE_DIRECTORIES.items():
            assert state.environment[key] == f"{descriptor_root}/{relative}"
        assert state.working_directory == state.environment["TMPDIR"]
        os.symlink("/", "escape", dir_fd=descriptor)
        state.reverify(image_state_uncertain=False)
    assert not directory_path.exists()
    with pytest.raises(OSError):
        os.fstat(descriptor)


@pytest.mark.parametrize("manager_name", ["cli", "iidfile"])
@pytest.mark.parametrize("replacement", [False, True])
def test_private_executor_directory_swap_or_vanish_fails_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manager_name: str,
    replacement: bool,
) -> None:
    created_paths: list[Path] = []
    real_mkdtemp = executor.tempfile.mkdtemp

    def make_private_directory(*, prefix: str) -> str:
        path = Path(real_mkdtemp(prefix=prefix, dir=tmp_path))
        created_paths.append(path)
        return str(path)

    monkeypatch.setattr(executor.tempfile, "mkdtemp", make_private_directory)
    manager = (
        executor._exclusive_cli_state() if manager_name == "cli" else executor._exclusive_iidfile()
    )
    moved_path: Path | None = None
    original_path: Path | None = None
    try:
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="cleanup failed",
        ):
            with manager as private:
                original_path = private.directory_path
                moved_path = original_path.with_name(original_path.name + ".moved")
                original_path.rename(moved_path)
                if replacement:
                    original_path.mkdir(mode=0o700)
        assert moved_path is not None and moved_path.is_dir()
        if replacement:
            assert original_path is not None and original_path.is_dir()
    finally:
        for path in (original_path, moved_path):
            if path is not None and path.exists():
                path.rmdir()
    assert len(created_paths) == 1


def test_retained_toolchain_verifies_content_drift_and_closes_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    docker_path = tmp_path / "docker"
    buildx_path = tmp_path / "docker-buildx"
    loader_path = tmp_path / "loader"
    libc_path = tmp_path / "libc.so.6"
    cache_path = tmp_path / "ld.so.cache"
    docker_path.write_bytes(b"synthetic docker cli")
    buildx_path.write_bytes(b"synthetic buildx plugin")
    loader_path.write_bytes(b"synthetic loader")
    libc_path.write_bytes(b"synthetic libc")
    cache_path.write_bytes(b"synthetic loader cache")
    for path in (docker_path, buildx_path, loader_path, libc_path):
        path.chmod(0o755)
    cache_path.chmod(0o644)
    original_contract = executor._execution_toolchain_contract()

    def executable_record(path: Path, *, mode: str = "0755") -> dict[str, Any]:
        raw = path.read_bytes()
        return {
            "gid": os.getegid(),
            "mode": mode,
            "path": str(path),
            "sha256": _sha(raw),
            "size_bytes": len(raw),
            "uid": os.geteuid(),
        }

    dynamic_runtime = copy.deepcopy(original_contract["docker_dynamic_runtime"])
    dynamic_runtime["regular_files"] = {
        "ld_so_cache": executable_record(cache_path, mode="0644"),
        "libc": executable_record(libc_path),
        "loader": executable_record(loader_path),
    }
    contract = {
        **original_contract,
        "buildx_plugin": executable_record(buildx_path),
        "docker_cli": executable_record(docker_path),
        "docker_dynamic_runtime": dynamic_runtime,
    }
    monkeypatch.setattr(executor, "_execution_toolchain_contract", lambda: contract)
    monkeypatch.setattr(
        executor,
        "_require_secure_tool_path_ancestors",
        lambda _path, *, image_state_uncertain: None,
    )
    monkeypatch.setattr(
        executor,
        "_reverify_dynamic_runtime_routes",
        lambda *, image_state_uncertain: {
            "ld_so_preload": copy.deepcopy(dynamic_runtime["ld_so_preload"]),
            "symlinks": copy.deepcopy(dynamic_runtime["symlinks"]),
        },
    )
    descriptors: tuple[int, ...]
    with executor._retain_execution_toolchain() as retained:
        descriptors = (
            retained.docker_cli.descriptor,
            retained.buildx_plugin.descriptor,
            retained.docker_loader.descriptor,
            retained.docker_libc.descriptor,
            retained.docker_ld_so_cache.descriptor,
        )
        assert retained.reverify(image_state_uncertain=False) == {
            "buildx_plugin": contract["buildx_plugin"],
            "docker_cli": contract["docker_cli"],
            "docker_dynamic_runtime": dynamic_runtime,
        }
        docker_path.rename(tmp_path / "docker.original")
        docker_path.write_bytes(b"substituted docker cli")
        docker_path.chmod(0o755)
        with pytest.raises(
            executor.ForagerMatchedV3CpuOciBuildExecutorError,
            match="identity differs",
        ) as caught:
            retained.reverify(image_state_uncertain=True)
        assert caught.value.image_state_uncertain is True
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_dynamic_runtime_routes_require_exact_symlinks_and_absent_preload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_link = tmp_path / "lib"
    second_link = tmp_path / "loader"
    preload_path = tmp_path / "ld.so.preload"
    first_link.symlink_to("usr/lib")
    second_link.symlink_to("../lib/loader")
    dynamic_runtime: dict[str, Any] = {
        "ld_so_preload": {
            "path": str(preload_path),
            "required_state": "absent",
        },
        "regular_files": {},
        "symlinks": [
            {
                "gid": os.getegid(),
                "path": str(first_link),
                "target": "usr/lib",
                "uid": os.geteuid(),
            },
            {
                "gid": os.getegid(),
                "path": str(second_link),
                "target": "../lib/loader",
                "uid": os.geteuid(),
            },
        ],
    }
    contract = executor._execution_toolchain_contract()
    contract["docker_dynamic_runtime"] = dynamic_runtime
    monkeypatch.setattr(executor, "_execution_toolchain_contract", lambda: contract)
    monkeypatch.setattr(
        executor,
        "_require_secure_tool_path_ancestors",
        lambda _path, *, image_state_uncertain: None,
    )

    assert executor._reverify_dynamic_runtime_routes(image_state_uncertain=False) == {
        "ld_so_preload": dynamic_runtime["ld_so_preload"],
        "symlinks": dynamic_runtime["symlinks"],
    }

    first_link.unlink()
    first_link.symlink_to("attacker/lib")
    with pytest.raises(
        executor.ForagerMatchedV3CpuOciBuildExecutorError,
        match="symlink identity differs",
    ) as caught:
        executor._reverify_dynamic_runtime_routes(image_state_uncertain=True)
    assert caught.value.image_state_uncertain is True

    first_link.unlink()
    first_link.symlink_to("usr/lib")
    preload_path.write_text("attacker\n", encoding="ascii")
    with pytest.raises(
        executor.ForagerMatchedV3CpuOciBuildExecutorError,
        match="preload file must remain absent",
    ):
        executor._reverify_dynamic_runtime_routes(image_state_uncertain=False)


def test_default_runner_removes_pipe_detached_same_group_descendant_after_success(
    python_executable_descriptor: int,
) -> None:
    result = executor._default_process_runner(
        (
            sys.executable,
            "-c",
            (
                "import subprocess,sys;"
                "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
                "stderr=subprocess.DEVNULL);"
                "print(p.pid,flush=True)"
            ),
        ),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=5,
        stdout_limit_bytes=1024,
        stderr_limit_bytes=1024,
    )
    assert result.returncode == 0
    assert result.timed_out is False
    child_pid = int(result.stdout.strip())
    deadline = time.monotonic() + 2.0
    while Path(f"/proc/{child_pid}").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not Path(f"/proc/{child_pid}").exists()


def test_default_runner_honors_command_deadline_after_both_output_pipes_close(
    python_executable_descriptor: int,
) -> None:
    success = executor._default_process_runner(
        (
            sys.executable,
            "-c",
            "import os,time;os.close(1);os.close(2);time.sleep(0.1)",
        ),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=2,
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert success.returncode == 0
    assert success.timed_out is False
    timeout = executor._default_process_runner(
        (
            sys.executable,
            "-c",
            "import os,time;os.close(1);os.close(2);time.sleep(30)",
        ),
        environment={},
        executable_descriptor=python_executable_descriptor,
        inherited_descriptors=(),
        working_directory="/",
        stdin_descriptor=None,
        timeout_seconds=1,
        stdout_limit_bytes=32,
        stderr_limit_bytes=32,
    )
    assert timeout.timed_out is True


@pytest.mark.parametrize("failure", ["missing_pipe", "selector_setup", "selector_loop"])
def test_default_runner_setup_and_cleanup_failures_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    python_executable_descriptor: int,
) -> None:
    class FakePipe:
        closed = False

        def fileno(self) -> int:
            return 123

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self) -> None:
            self.pid = 456_789
            self.stdout = None if failure == "missing_pipe" else FakePipe()
            self.stderr = FakePipe()
            self.wait_timeouts: list[float | None] = []

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            self.wait_timeouts.append(timeout)
            return -9

    class FailingSelector:
        def register(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def get_map(self) -> dict[str, Any]:
            raise OSError("synthetic selector loop failure")

        def close(self) -> None:
            return None

    process = FakeProcess()
    terminated: list[int] = []
    anchored: list[int] = []
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(executor.os, "set_blocking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(executor.os, "waitid", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        executor,
        "_terminate_process_group",
        lambda candidate: terminated.append(candidate.pid),
    )
    monkeypatch.setattr(
        executor,
        "_terminate_anchored_process_group",
        lambda candidate: anchored.append(candidate.pid),
    )
    if failure == "selector_setup":
        monkeypatch.setattr(
            executor.selectors,
            "DefaultSelector",
            lambda: (_ for _ in ()).throw(OSError("synthetic selector setup failure")),
        )
    elif failure == "selector_loop":
        monkeypatch.setattr(executor.selectors, "DefaultSelector", FailingSelector)
    with pytest.raises(executor.ForagerMatchedV3CpuOciBuildExecutorError):
        executor._default_process_runner(
            ("harmless-fake-command",),
            environment={},
            executable_descriptor=python_executable_descriptor,
            inherited_descriptors=(),
            working_directory="/",
            stdin_descriptor=None,
            timeout_seconds=1,
            stdout_limit_bytes=32,
            stderr_limit_bytes=32,
        )
    assert terminated == [process.pid]
    assert anchored == [process.pid]
    assert process.wait_timeouts
    assert all(timeout is not None for timeout in process.wait_timeouts)
