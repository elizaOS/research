"""No-container tests for the matched-v3 CPU OCI engineering smoke."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import stat
import tempfile
import threading
import tomllib
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_publication as build_publication,
)
from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_engineering_smoke as smoke,
)

pytestmark = pytest.mark.unit


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_hash_and_image_identities_reject_zero_sentinels() -> None:
    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError):
        smoke._require_sha256("0" * 64, label="test hash")
    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError):
        smoke._require_image_id("sha256:" + "0" * 64, label="test image")


def _published_build(tmp_path: Path, *, image_digit: str = "d") -> Any:
    return build_publication.PublishedMatchedV3CpuOciBuild(
        intent_directory=tmp_path / "build" / "intents" / "sha256" / ("a" * 64),
        success_directory=tmp_path / "build" / "successes" / "sha256" / ("b" * 64),
        context_receipt_sha256="a" * 64,
        execution_receipt_sha256="b" * 64,
        publication_receipt_sha256="c" * 64,
        image_id="sha256:" + image_digit * 64,
    )


def _toolchain_record() -> dict[str, Any]:
    return {
        "contract": {"fixture": "retained-toolchain"},
        "contract_sha256": "e" * 64,
        "docker_cli": {
            "gid": 0,
            "mode": "0755",
            "path": "/usr/bin/docker",
            "sha256": "f" * 64,
            "size_bytes": 45_355_843,
            "uid": 0,
        },
    }


def _daemon_info(*, drifted: bool = False) -> dict[str, Any]:
    return {
        "Architecture": "x86_64",
        "CgroupDriver": "systemd",
        "CgroupVersion": "2",
        "DockerRootDir": "/var/lib/docker",
        "Driver": "overlayfs",
        "ID": ("drifted-daemon-id" if drifted else "fixture-daemon-id"),
        "KernelVersion": "6.8.0-fixture",
        "MemTotal": 32_963_682_304,
        "NCPU": 24,
        "Name": "fixture-daemon",
        "OSType": "linux",
        "OperatingSystem": "Ubuntu 24.04.4 LTS",
        "SecurityOptions": ["name=seccomp,profile=builtin", "name=cgroupns"],
        "ServerVersion": "29.6.1",
    }


def _daemon_projection() -> dict[str, Any]:
    return {
        "architecture": "x86_64",
        "cgroup_driver": "systemd",
        "cgroup_version": "2",
        "cpu_count": 24,
        "daemon_id": "fixture-daemon-id",
        "docker_root_directory": "/var/lib/docker",
        "kernel_version": "6.8.0-fixture",
        "memory_bytes": 32_963_682_304,
        "name": "fixture-daemon",
        "operating_system": "Ubuntu 24.04.4 LTS",
        "operating_system_type": "linux",
        "security_options": ["name=seccomp,profile=builtin", "name=cgroupns"],
        "server_version": "29.6.1",
        "storage_driver": "overlayfs",
    }


def _fixture_observations() -> list[smoke._ProbeObservation]:
    return [
        smoke._ProbeObservation(
            probe_id=probe.probe_id,
            container_name=(
                f"alberta-matched-v3-smoke-{probe.name_component}-{index:032x}"
            ),
            container_id=f"{index:064x}",
            postrun_inspect_object_sha256=f"{index + 2:064x}",
            prestart_inspect_object_sha256=f"{index + 4:064x}",
            returncode=0,
            stdout=probe.expected_stdout,
            stderr=probe.expected_stderr,
            cleanup_state="force_removed_by_id_with_all_absence_proofs",
        )
        for index, probe in enumerate(smoke._PROBES, start=1)
    ]


def _success_payload_fixture() -> dict[str, Any]:
    return smoke._success_payload(
        intent_sha256="1" * 64,
        build_record=smoke._build_record(_published_build(Path("/tmp/fixture"))),
        toolchain_record=_toolchain_record(),
        daemon_projection=_daemon_projection(),
        observations=_fixture_observations(),
    )


def _permissive_canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


class _FakeDocker:
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.invocations: list[
            tuple[tuple[str, ...], int, int, int, bool]
        ] = []
        self.containers: dict[str, dict[str, Any]] = {}
        self.next_id = 1
        self.daemon_info_calls = 0
        self.daemon_drift_after_info_calls: int | None = None
        self.daemon_drift_on_info_calls: set[int] = set()
        self.daemon_loss_after_info_calls: int | None = None
        self.wrong_probe_output = False
        self.cleanup_retains = False
        self.inspect_drift = False
        self.postrun_oom = False
        self.create_returncode = 0
        self.create_stderr = b""
        self.raise_after_create: BaseException | None = None
        self.create_stdout_override: bytes | None = None
        self.cidfile_override: bytes | None = None
        self.start_returncode = 0
        self.start_stderr = b""
        self.raise_on_start: BaseException | None = None

    def _result(
        self,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> smoke._ProcessResult:
        return smoke._ProcessResult(returncode, stdout, stderr)

    def _container_for_target(self, target: str) -> tuple[str, dict[str, Any]] | None:
        if target in self.containers:
            return target, self.containers[target]
        for container_id, record in self.containers.items():
            if record["name"] == target:
                return container_id, record
        return None

    def _inspection(self, container_id: str) -> bytes:
        record = self.containers[container_id]
        exited = record["state"] == "exited"
        host = {
            "AutoRemove": False,
            "Binds": None,
            "CapDrop": ["ALL"],
            "CgroupnsMode": "private",
            "DeviceRequests": [],
            "Devices": [],
            "IpcMode": "private",
            "Memory": smoke._MEMORY_BYTES,
            "MemorySwap": smoke._MEMORY_BYTES,
            "NanoCpus": smoke._NANO_CPUS,
            "NetworkMode": "bridge" if self.inspect_drift else "none",
            "PidMode": "",
            "PidsLimit": smoke._PIDS_LIMIT,
            "PortBindings": {},
            "Privileged": False,
            "PublishAllPorts": False,
            "ReadonlyRootfs": True,
            "RestartPolicy": {"MaximumRetryCount": 0, "Name": "no"},
            "SecurityOpt": ["no-new-privileges"],
            "Tmpfs": {
                "/run/alberta": smoke._TMPFS_SPEC.split(":", 1)[1],
            },
        }
        value = {
            "Config": {
                "AttachStderr": True,
                "AttachStdin": False,
                "AttachStdout": True,
                "Cmd": list(record["argv"]),
                "Entrypoint": None,
                "Env": [
                    *(f"{key}={value}" for key, value in smoke._REQUIRED_IMAGE_ENVIRONMENT.items()),
                    *smoke._CONTAINER_ENVIRONMENT,
                ],
                "Healthcheck": {"Test": ["NONE"]},
                "Image": record["image_id"],
                "OpenStdin": False,
                "Tty": False,
                "User": "65532:65532",
                "WorkingDir": "/work",
            },
            "Args": list(record["argv"][1:]),
            "HostConfig": host,
            "Id": container_id,
            "Image": record["image_id"],
            "Mounts": [],
            "Name": f"/{record['name']}",
            "Path": record["argv"][0],
            "RestartCount": 0,
            "State": {
                "Dead": False,
                "Error": "",
                "ExitCode": 0,
                "OOMKilled": self.postrun_oom if exited else False,
                "Paused": False,
                "Restarting": False,
                "Running": False,
                "Status": "exited" if exited else "created",
            },
        }
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"

    def invoke(
        self,
        argv: tuple[str, ...],
        *,
        timeout_seconds: int,
        stdout_limit_bytes: int,
        stderr_limit_bytes: int,
        container_state_uncertain: bool,
    ) -> smoke._ProcessResult:
        assert type(timeout_seconds) is int and timeout_seconds > 0
        assert type(stdout_limit_bytes) is int and stdout_limit_bytes > 0
        assert type(stderr_limit_bytes) is int and stderr_limit_bytes > 0
        assert type(container_state_uncertain) is bool
        self.commands.append(argv)
        self.invocations.append(
            (
                argv,
                timeout_seconds,
                stdout_limit_bytes,
                stderr_limit_bytes,
                container_state_uncertain,
            )
        )
        assert argv[:2] == ("/usr/bin/docker", f"--host={smoke._DOCKER_HOST}")
        if argv[2:] == ("info", "--format={{json .}}"):
            self.daemon_info_calls += 1
            if (
                self.daemon_loss_after_info_calls is not None
                and self.daemon_info_calls > self.daemon_loss_after_info_calls
            ):
                return self._result(1, stderr=b"injected daemon loss\n")
            drifted = (
                self.daemon_info_calls in self.daemon_drift_on_info_calls
                or (
                    self.daemon_drift_after_info_calls is not None
                    and self.daemon_info_calls > self.daemon_drift_after_info_calls
                )
            )
            raw = json.dumps(
                _daemon_info(drifted=drifted),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii") + b"\n"
            return self._result(stdout=raw)
        operation = argv[2:4]
        if operation == ("container", "ls"):
            assert argv[4:7] == ("--all", "--quiet", "--no-trunc")
            assert len(argv) == 8
            filter_value = argv[-1].removeprefix("--filter=")
            matches: list[str] = []
            for container_id, record in self.containers.items():
                if filter_value == f"id={container_id}" or filter_value == (
                    f"name=^/{record['name']}$"
                ):
                    matches.append(container_id)
            payload = b"".join(f"{item}\n".encode("ascii") for item in matches)
            return self._result(stdout=payload)
        if operation == ("container", "create"):
            name = next(item.removeprefix("--name=") for item in argv if item.startswith("--name="))
            cidfile = Path(
                next(
                    item.removeprefix("--cidfile=")
                    for item in argv
                    if item.startswith("--cidfile=")
                )
            )
            image_index = next(
                index
                for index, item in enumerate(argv)
                if item.startswith("sha256:")
            )
            probe_argv = tuple(argv[image_index + 1 :])
            if probe_argv not in {probe.argv for probe in smoke._PROBES}:
                raise AssertionError(f"unexpected engineering-smoke probe argv: {probe_argv!r}")
            container_id = f"{self.next_id:064x}"
            self.next_id += 1
            cidfile.write_bytes(
                self.cidfile_override
                if self.cidfile_override is not None
                else f"{container_id}\n".encode("ascii")
            )
            self.containers[container_id] = {
                "argv": probe_argv,
                "image_id": argv[image_index],
                "name": name,
                "state": "created",
            }
            if self.raise_after_create is not None:
                raise self.raise_after_create
            stdout = (
                self.create_stdout_override
                if self.create_stdout_override is not None
                else f"{container_id}\n".encode("ascii")
            )
            return self._result(
                self.create_returncode,
                stdout=stdout,
                stderr=self.create_stderr,
            )
        if operation == ("container", "inspect"):
            assert argv[4] == "--format={{json .}}"
            assert len(argv) == 6
            container_id = argv[-1]
            if container_id not in self.containers:
                return self._result(1, stderr=b"absent\n")
            return self._result(stdout=self._inspection(container_id))
        if operation == ("container", "start"):
            assert argv[4] == "--attach"
            assert len(argv) == 6
            container_id = argv[-1]
            record = self.containers[container_id]
            if record["argv"] not in {probe.argv for probe in smoke._PROBES}:
                raise AssertionError(
                    f"unexpected retained engineering-smoke probe argv: {record['argv']!r}"
                )
            if self.raise_on_start is not None:
                raise self.raise_on_start
            record["state"] = "exited"
            output = next(
                probe.expected_stdout
                for probe in smoke._PROBES
                if record["argv"] == probe.argv
            )
            if self.wrong_probe_output:
                output = b"unexpected\n"
            return self._result(
                self.start_returncode,
                stdout=output,
                stderr=self.start_stderr,
            )
        if operation == ("container", "rm"):
            assert argv[4] == "--force"
            assert len(argv) == 6
            target = argv[-1]
            found = self._container_for_target(target)
            if self.cleanup_retains:
                return self._result(1, stderr=b"injected cleanup failure\n")
            if found is None:
                return self._result(1, stderr=b"absent\n")
            container_id, _record = found
            del self.containers[container_id]
            return self._result(stdout=f"{container_id}\n".encode("ascii"))
        raise AssertionError(f"unexpected Docker command: {argv!r}")


def _install_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeDocker,
    *,
    replay_mismatch: bool = False,
) -> tuple[smoke.MatchedV3CpuOciEngineeringSmokeRequest, Any, list[Any]]:
    artifact_root = tmp_path / "artifacts"
    build_root = tmp_path / "build"
    publication_root = tmp_path / "smoke"
    for root in (artifact_root, build_root, publication_root):
        root.mkdir(mode=0o700)
    published = _published_build(tmp_path)
    validation_results: list[Any] = []

    def validate(*_args: Any, **_kwargs: Any) -> Any:
        if replay_mismatch and validation_results:
            result = _published_build(tmp_path, image_digit="9")
        else:
            result = published
        validation_results.append(result)
        return result

    monkeypatch.setattr(
        build_publication,
        "validate_published_matched_v3_cpu_oci_build",
        validate,
    )
    monkeypatch.setattr(
        smoke,
        "_bound_toolchain_record",
        lambda *_args, **_kwargs: copy.deepcopy(_toolchain_record()),
    )

    @contextmanager
    def retain(_expected: Any) -> Iterator[smoke._RuntimeBinding]:
        private = tmp_path / "private-cli"
        private.mkdir(mode=0o700, exist_ok=True)
        yield smoke._RuntimeBinding(
            docker_path="/usr/bin/docker",
            cli_working_directory=private.as_posix(),
            toolchain_record=_toolchain_record(),
            invoke=fake.invoke,
        )

    monkeypatch.setattr(smoke, "_retain_runtime_binding", retain)
    request = smoke.MatchedV3CpuOciEngineeringSmokeRequest(
        artifact_root=artifact_root,
        build_publication_root=build_root,
        publication_root=publication_root,
        expected_build_context_receipt_sha256=published.context_receipt_sha256,
        expected_build_execution_receipt_sha256=published.execution_receipt_sha256,
        exact_acknowledgement=smoke.ENGINEERING_SMOKE_ACKNOWLEDGEMENT,
        timeout_seconds=60,
    )
    return request, published, validation_results


def _operations(fake: _FakeDocker, operation: str) -> list[tuple[str, ...]]:
    return [
        command
        for command in fake.commands
        if len(command) > 3 and command[2] == "container" and command[3] == operation
    ]


def test_success_runs_exactly_two_create_first_lifecycles_and_replays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, published_build, validation_results = _install_harness(
        tmp_path,
        monkeypatch,
        fake,
    )

    published = smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert len(_operations(fake, "create")) == 2
    assert len(_operations(fake, "start")) == 2
    assert len(_operations(fake, "inspect")) == 4
    assert len(_operations(fake, "rm")) == 2
    assert len(_operations(fake, "ls")) == 6
    assert fake.daemon_info_calls == 14
    assert not fake.containers
    assert len(validation_results) == 3
    for command, timeout, stdout_limit, stderr_limit, uncertain in fake.invocations:
        operation = command[2:4]
        assert type(uncertain) is bool
        if command[2:] == ("info", "--format={{json .}}") or operation in {
            ("container", "ls"),
            ("container", "rm"),
        }:
            assert timeout == smoke._CLEANUP_TIMEOUT_SECONDS
            assert stdout_limit == smoke._MAX_CONTROL_OUTPUT_BYTES
            assert stderr_limit == smoke._MAX_CONTROL_OUTPUT_BYTES
        elif operation in {("container", "create"), ("container", "inspect")}:
            assert timeout == request.timeout_seconds
            assert stdout_limit == smoke._MAX_CONTROL_OUTPUT_BYTES
            assert stderr_limit == smoke._MAX_CONTROL_OUTPUT_BYTES
            assert uncertain is True
        elif operation == ("container", "start"):
            assert timeout == request.timeout_seconds
            assert stdout_limit == smoke._MAX_PROCESS_OUTPUT_BYTES
            assert stderr_limit == smoke._MAX_PROCESS_OUTPUT_BYTES
            assert uncertain is True
        else:
            raise AssertionError(f"unclassified fake Docker invocation: {command!r}")
    assert published.image_id == published_build.image_id
    assert published.success_directory.is_dir()
    assert stat.S_IMODE(published.success_directory.stat().st_mode) == 0o500
    receipt_path = published.success_directory / smoke._SUCCESS_FILENAME
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o400
    assert receipt_path.stat().st_nlink == 1
    receipt = json.loads(receipt_path.read_bytes())
    assert receipt["classification"] == smoke._classification()
    assert receipt["claims"] == smoke._claims()
    assert receipt["daemon_projection"] == _daemon_projection()
    assert [item["probe_id"] for item in receipt["observations"]] == [
        "python_version",
        "runtime_verifier",
    ]
    assert receipt["container_count_created"] == 2
    assert receipt["container_count_started"] == 2
    assert all(
        observation["cleanup"]
        == {
            "exact_id_absent": True,
            "exact_name_absent": True,
            "state": "force_removed_by_id_with_all_absence_proofs",
        }
        for observation in receipt["observations"]
    )

    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_receipt_sha256=published.receipt_sha256,
    )

    assert replayed == published
    assert len(validation_results) == 4


def test_create_commands_freeze_sandbox_and_do_not_enable_auto_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    for command in _operations(fake, "create"):
        assert "--pull=never" in command
        assert "--platform=linux/amd64" in command
        assert "--network=none" in command
        assert "--read-only" in command
        assert "--cap-drop=ALL" in command
        assert "--security-opt=no-new-privileges" in command
        assert "--user=65532:65532" in command
        assert "--workdir=/work" in command
        assert "--cpus=2.0" in command
        assert "--memory=4g" in command
        assert "--memory-swap=4g" in command
        assert "--pids-limit=256" in command
        assert f"--tmpfs={smoke._TMPFS_SPEC}" in command
        assert "--rm" not in command
        assert not any(item.startswith("--pid=") for item in command)
    for command in _operations(fake, "inspect"):
        assert "--type=container" not in command
        assert command[4] == "--format={{json .}}"
    forbidden = {"exec", "run", "push", "pull", "prune", "tag"}
    assert not any(
        len(command) > 3 and command[2] == "container" and command[3] in forbidden
        for command in fake.commands
    )
    assert all("image" not in command[2:4] for command in fake.commands)


def test_fake_boundary_rejects_every_nonfrozen_probe_tail(tmp_path: Path) -> None:
    fake = _FakeDocker()
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    binding = smoke._RuntimeBinding(
        docker_path="/usr/bin/docker",
        cli_working_directory=private.as_posix(),
        toolchain_record=_toolchain_record(),
        invoke=fake.invoke,
    )
    injected = smoke._Probe(
        probe_id="python_version",
        name_component="python-version",
        argv=("/usr/local/bin/python", "-c", "print('not frozen')"),
        expected_stdout=b"",
    )
    command = smoke._create_command(
        binding,
        image_id="sha256:" + "d" * 64,
        probe=injected,
        container_name="alberta-matched-v3-smoke-python-version-" + "1" * 32,
        cidfile=private / "container.cid",
    )

    with pytest.raises(AssertionError, match="unexpected engineering-smoke probe argv"):
        fake.invoke(
            command,
            timeout_seconds=60,
            stdout_limit_bytes=4096,
            stderr_limit_bytes=4096,
            container_state_uncertain=True,
        )

    assert not fake.containers


def test_cleanup_proves_full_id_and_exact_name_even_after_successful_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    filters = [command[-1] for command in _operations(fake, "ls")]
    id_filters = [item for item in filters if item.startswith("--filter=id=")]
    name_filters = [item for item in filters if item.startswith("--filter=name=^/")]
    assert len(id_filters) == 2
    assert len(name_filters) == 4
    assert all(len(item.removeprefix("--filter=id=")) == 64 for item in id_filters)
    assert all(item.endswith("$") for item in name_filters)


def test_wrong_acknowledgement_fails_before_any_publication_or_docker_action(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / name for name in ("artifacts", "build", "smoke")]
    for root in roots:
        root.mkdir(mode=0o700)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="acknowledgement differs",
    ):
        smoke.MatchedV3CpuOciEngineeringSmokeRequest(
            artifact_root=roots[0],
            build_publication_root=roots[1],
            publication_root=roots[2],
            expected_build_context_receipt_sha256="a" * 64,
            expected_build_execution_receipt_sha256="b" * 64,
            exact_acknowledgement="yes",
        )
    assert not any(roots[2].iterdir())


def test_request_rejects_overlapping_publication_and_build_roots(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    build_root = tmp_path / "build"
    publication_root = build_root / "smoke"
    artifact_root.mkdir(mode=0o700)
    publication_root.mkdir(mode=0o700, parents=True)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="pairwise nonoverlapping",
    ):
        smoke.MatchedV3CpuOciEngineeringSmokeRequest(
            artifact_root=artifact_root,
            build_publication_root=build_root,
            publication_root=publication_root,
            expected_build_context_receipt_sha256="a" * 64,
            expected_build_execution_receipt_sha256="b" * 64,
            exact_acknowledgement=smoke.ENGINEERING_SMOKE_ACKNOWLEDGEMENT,
        )


@pytest.mark.parametrize("failure_receipt", [False, True], ids=["success", "failure"])
def test_fresh_validators_reject_overlapping_roots_before_publication_read(
    tmp_path: Path,
    failure_receipt: bool,
) -> None:
    shared_root = tmp_path / "shared"
    artifact_root = tmp_path / "artifacts"
    shared_root.mkdir(mode=0o700)
    artifact_root.mkdir(mode=0o700)

    validator = (
        smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure
        if failure_receipt
        else smoke.validate_published_matched_v3_cpu_oci_engineering_smoke
    )
    receipt_keyword = (
        {"expected_failure_receipt_sha256": "a" * 64}
        if failure_receipt
        else {"expected_receipt_sha256": "a" * 64}
    )
    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="pairwise nonoverlapping",
    ):
        validator(
            shared_root,
            build_publication_root=shared_root,
            artifact_root=artifact_root,
            **receipt_keyword,
        )


def test_public_api_and_cli_accept_no_image_runner_or_arbitrary_probe() -> None:
    request_parameters = inspect.signature(
        smoke.MatchedV3CpuOciEngineeringSmokeRequest
    ).parameters
    execute_parameters = inspect.signature(
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke
    ).parameters
    forbidden = {"image", "image_id", "runner", "runtime", "probe", "command"}
    assert forbidden.isdisjoint(request_parameters)
    assert set(execute_parameters) == {"request"}
    parser = smoke._argument_parser()
    subparser_action = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    )
    execute = cast(Any, subparser_action).choices["execute"]
    destinations = {action.dest for action in execute._actions}
    assert forbidden.isdisjoint(destinations)
    assert (
        "MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError"
        in smoke.__all__
    )
    assert (
        "MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError"
        in smoke.__all__
    )
    assert (
        "MatchedV3CpuOciEngineeringSmokeFailurePublicationUncertainError"
        in smoke.__all__
    )


def test_same_intent_refuses_automatic_retry_before_another_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)
    command_count = len(fake.commands)

    with pytest.raises(
        smoke.MatchedV3CpuOciEngineeringSmokeIntentExistsError,
        match="refusing automatic retry",
    ):
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert len(fake.commands) == command_count


def test_post_intent_runtime_handoff_interrupt_publishes_addressed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    def interrupt_handoff(_toolchain_record: Any) -> Any:
        raise KeyboardInterrupt("injected post-intent handoff interrupt")

    monkeypatch.setattr(smoke, "_retain_runtime_binding", interrupt_handoff)

    with pytest.raises(KeyboardInterrupt) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert not fake.commands
    intents = list((request.publication_root / "intents" / "sha256").iterdir())
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(intents) == 1
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["intent_sha256"] == intents[0].name
    assert receipt["phase"] == "runtime_binding"
    assert receipt["container_state_uncertain"] is False
    assert getattr(caught.value, "failure_committed", None) is True
    assert getattr(caught.value, "failure_receipt_sha256", None) == failures[0].name


def test_concurrent_intent_publication_has_one_atomic_winner(tmp_path: Path) -> None:
    publication_root = tmp_path / "publication"
    publication_root.mkdir(mode=0o700)
    raw = b'{"fixture":"concurrent-intent"}\n'
    address = _sha(raw)
    barrier = threading.Barrier(2)

    def publish() -> Path:
        barrier.wait(timeout=5)
        return smoke._publish_files(
            publication_root,
            category="intents",
            address=address,
            files={smoke._INTENT_FILENAME: raw},
            intent=True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(publish) for _ in range(2)]
    outcomes: list[Path] = []
    failures: list[BaseException] = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except BaseException as exc:
            failures.append(exc)

    assert outcomes == [publication_root / "intents" / "sha256" / address]
    assert len(failures) == 1
    assert isinstance(
        failures[0],
        smoke.MatchedV3CpuOciEngineeringSmokeIntentExistsError,
    )
    assert stat.S_IMODE(outcomes[0].stat().st_mode) == 0o500
    assert stat.S_IMODE((outcomes[0] / smoke._INTENT_FILENAME).stat().st_mode) == 0o400


@pytest.mark.parametrize("committed", [False, True], ids=["precommit", "postcommit"])
def test_intent_cas_fault_preserves_commit_state_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    committed: bool,
) -> None:
    publication_root = tmp_path / "publication"
    publication_root.mkdir(mode=0o700)
    raw = b'{"fixture":"intent-commit-state"}\n'
    address = _sha(raw)

    def fail_publication(
        _root: Any,
        *,
        category: str,
        address: str,
        files: Any,
        intent: bool,
        commit_state: Any,
    ) -> Path:
        assert category == "intents"
        assert address == _sha(raw)
        assert files == {smoke._INTENT_FILENAME: raw}
        assert intent is True
        assert commit_state is not None
        if committed:
            commit_state.committed = True
        raise build_publication.ForagerMatchedV3CpuOciBuildPublicationError(
            "injected publication fault"
        )

    monkeypatch.setattr(build_publication, "_publish_files", fail_publication)

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke._publish_files(
            publication_root,
            category="intents",
            address=address,
            files={smoke._INTENT_FILENAME: raw},
            intent=True,
        )

    if committed:
        assert type(caught.value) is (
            smoke.MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError
        )
        assert caught.value.intent_sha256 == address
        assert caught.value.intent_committed is True
    else:
        assert type(caught.value) is smoke.ForagerMatchedV3CpuOciEngineeringSmokeError
        assert not hasattr(caught.value, "intent_committed")
    assert caught.value.container_state_uncertain is False


def test_success_cas_fault_preserves_unknown_commit_state_and_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication_root = tmp_path / "publication"
    publication_root.mkdir(mode=0o700)
    raw = b'{"fixture":"success-commit-state"}\n'
    address = _sha(raw)

    def fail_publication(
        _root: Any,
        *,
        category: str,
        address: str,
        files: Any,
        intent: bool,
        commit_state: Any,
    ) -> Path:
        assert category == "successes"
        assert address == _sha(raw)
        assert files == {smoke._SUCCESS_FILENAME: raw}
        assert intent is False
        assert commit_state is None
        raise build_publication.MatchedV3CpuOciBuildPublicationStateUncertainError(
            "injected uncertain success publication"
        )

    monkeypatch.setattr(build_publication, "_publish_files", fail_publication)

    with pytest.raises(
        smoke.MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError
    ) as caught:
        smoke._publish_files(
            publication_root,
            category="successes",
            address=address,
            files={smoke._SUCCESS_FILENAME: raw},
        )

    assert caught.value.receipt_sha256 == address
    assert caught.value.success_committed is None
    assert caught.value.success_publication_state_uncertain is True
    assert caught.value.container_state_uncertain is False


def test_committed_success_readback_failure_retains_only_success_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    def fail_readback(*_args: Any, **_kwargs: Any) -> Any:
        raise smoke.ForagerMatchedV3CpuOciEngineeringSmokeError(
            "injected final success readback fault"
        )

    monkeypatch.setattr(
        smoke,
        "validate_published_matched_v3_cpu_oci_engineering_smoke",
        fail_readback,
    )

    with pytest.raises(
        smoke.MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError,
        match="success committed but final readback failed",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    successes = list((request.publication_root / "successes" / "sha256").iterdir())
    assert len(successes) == 1
    assert caught.value.receipt_sha256 == successes[0].name
    assert caught.value.success_committed is True
    assert caught.value.intent_committed is True
    assert caught.value.intent_sha256 == next(
        (request.publication_root / "intents" / "sha256").iterdir()
    ).name
    assert not list((request.publication_root / "failures" / "sha256").iterdir())
    projection = smoke._cli_error(caught.value)
    assert projection["receipt_sha256"] == successes[0].name
    assert projection["success_committed"] is True
    assert projection["success_publication_state_uncertain"] is True
    assert projection["intent_committed"] is True
    assert projection["intent_sha256"] == caught.value.intent_sha256


def test_interrupt_after_success_commit_cannot_publish_contradictory_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    original_publish = smoke._publish_files

    def interrupt_after_success(
        publication_root: Path,
        *,
        category: str,
        address: str,
        files: Any,
        intent: bool = False,
        intent_commit_state: Any = None,
    ) -> Path:
        published = original_publish(
            publication_root,
            category=category,
            address=address,
            files=files,
            intent=intent,
            intent_commit_state=intent_commit_state,
        )
        if category == "successes":
            raise KeyboardInterrupt("injected success handoff interrupt")
        return published

    monkeypatch.setattr(smoke, "_publish_files", interrupt_after_success)

    with pytest.raises(
        smoke.MatchedV3CpuOciEngineeringSmokeSuccessPublicationUncertainError
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    successes = list((request.publication_root / "successes" / "sha256").iterdir())
    assert len(successes) == 1
    assert caught.value.receipt_sha256 == successes[0].name
    assert caught.value.success_committed is True
    assert caught.value.intent_committed is True
    assert not list((request.publication_root / "failures" / "sha256").iterdir())


def test_probe_mismatch_cleans_up_and_publishes_known_absent_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="probe result differs",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is False
    assert not fake.containers
    assert not list((request.publication_root / "successes" / "sha256").iterdir())
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is False
    assert receipt["retry_authorized"] is False
    assert receipt["classification"] == smoke._classification()
    assert receipt["lifecycle"]["create_invoked"] is True
    assert receipt["lifecycle"]["cleanup"] == {
        "all_candidate_ids_absent": True,
        "exact_name_absent": True,
        "proven_absent_ids": [receipt["lifecycle"]["authoritative_container_id"]],
        "resolved_container_id": receipt["lifecycle"]["authoritative_container_id"],
        "state": "force_removed_by_id_with_all_absence_proofs",
    }
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.receipt_sha256 == failures[0].name
    assert replayed.intent_sha256 == receipt["intent_sha256"]
    assert replayed.phase == "probe_python_version"
    assert replayed.container_state_uncertain is False
    projection = smoke._cli_error(caught.value)
    assert projection["intent_committed"] is True
    assert projection["intent_sha256"] == receipt["intent_sha256"]
    assert projection["failure_committed"] is True
    assert projection["failure_receipt_sha256"] == failures[0].name
    assert projection["failure_full_lineage_validated"] is True


def test_ambiguous_failure_cas_retains_unknown_state_and_exact_recovery_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    original_publish = build_publication._publish_files

    def ambiguous_failure_publish(
        root: Any,
        *,
        category: str,
        address: str,
        files: Any,
        intent: bool = False,
        commit_state: Any = None,
    ) -> Path:
        published = original_publish(
            root,
            category=category,
            address=address,
            files=files,
            intent=intent,
            commit_state=commit_state,
        )
        if category == "failures":
            raise build_publication.MatchedV3CpuOciBuildPublicationStateUncertainError(
                "injected failure CAS ambiguity"
            )
        return published

    monkeypatch.setattr(
        build_publication,
        "_publish_files",
        ambiguous_failure_publish,
    )

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    projection = smoke._cli_error(caught.value)
    assert projection["intent_committed"] is True
    assert projection["intent_sha256"] == next(
        (request.publication_root / "intents" / "sha256").iterdir()
    ).name
    assert projection["failure_receipt_sha256"] == failures[0].name
    assert projection["failure_committed"] is None
    assert projection["failure_publication_state_uncertain"] is True
    assert projection["failure_full_lineage_validated"] is None
    assert any(
        "fresh-validate this exact address" in note for note in caught.value.__notes__
    )


def test_valid_create_id_disagreement_never_removes_uncorroborated_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    actual_id = f"{1:064x}"
    unrelated_id = "f" * 64
    fake.containers[unrelated_id] = {
        "argv": smoke._PROBES[0].argv,
        "image_id": "sha256:" + "e" * 64,
        "name": "unrelated-container",
        "state": "created",
    }
    fake.create_stdout_override = f"{unrelated_id}\n".encode("ascii")
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="cidfile and create output differ",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert getattr(caught.value, "container_state_uncertain", None) is True
    assert set(fake.containers) == {unrelated_id}
    assert fake.containers[unrelated_id]["name"] == "unrelated-container"
    remove_targets = [command[-1] for command in _operations(fake, "rm")]
    assert remove_targets == [actual_id]
    assert unrelated_id not in remove_targets
    assert [command[-1] for command in _operations(fake, "inspect")] == [actual_id]
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    lifecycle = receipt["lifecycle"]
    assert lifecycle["authoritative_container_id"] is None
    assert lifecycle["candidate_container_ids"] == [actual_id, unrelated_id]
    assert lifecycle["create_invoked"] is True
    assert lifecycle["cleanup"] == {
        "all_candidate_ids_absent": False,
        "exact_name_absent": False,
        "proven_absent_ids": [actual_id],
        "resolved_container_id": actual_id,
        "state": "proving_all_absence_routes",
    }


def test_cleanup_failure_forbids_success_and_records_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    fake.cleanup_retains = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["lifecycle"]["create_invoked"] is True
    assert receipt["lifecycle"]["cleanup"]["resolved_container_id"] == (
        receipt["lifecycle"]["authoritative_container_id"]
    )
    assert receipt["lifecycle"]["cleanup"]["state"] == "proving_all_absence_routes"
    assert fake.containers


def test_precreate_daemon_loss_records_replayable_no_create_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.daemon_loss_after_info_calls = 1
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="Docker daemon identity projection failed",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is False
    assert not _operations(fake, "create")
    assert not _operations(fake, "rm")
    assert not fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["phase"] == "probe_python_version"
    assert receipt["daemon_projection"] == _daemon_projection()
    assert receipt["lifecycle"] == {
        "authoritative_container_id": None,
        "candidate_container_ids": [],
        "cleanup": {
            "all_candidate_ids_absent": True,
            "exact_name_absent": False,
            "proven_absent_ids": [],
            "resolved_container_id": None,
            "state": "not_attempted",
        },
        "container_name": receipt["lifecycle"]["container_name"],
        "create_invoked": False,
        "probe_id": "python_version",
        "uncertainty_latched": False,
    }
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is False


def test_transient_postcreate_daemon_flip_latches_uncertainty_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.daemon_drift_on_info_calls = {3}
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="Docker daemon identity changed",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    assert not fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["lifecycle"]["uncertainty_latched"] is True
    assert receipt["lifecycle"]["cleanup"]["state"] == (
        "force_removed_by_id_with_all_absence_proofs"
    )
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


def test_indeterminate_create_completion_remains_uncertain_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.raise_after_create = smoke.ForagerMatchedV3CpuOciEngineeringSmokeError(
        "injected indeterminate create completion",
        container_state_uncertain=True,
    )
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="indeterminate create completion",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    assert not fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["lifecycle"]["uncertainty_latched"] is True
    assert receipt["lifecycle"]["cleanup"]["state"] == (
        "force_removed_by_id_with_all_absence_proofs"
    )
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


def test_temporary_directory_exit_cannot_mask_uncertain_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    fake.cleanup_retains = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    real_temporary_directory = tempfile.TemporaryDirectory

    class ExitFailingTemporaryDirectory:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner: tempfile.TemporaryDirectory[str] = real_temporary_directory(
                *args,
                **kwargs,
            )

        def __enter__(self) -> str:
            return self._inner.__enter__()

        def __exit__(self, *exc_info: Any) -> None:
            self._inner.__exit__(*exc_info)
            raise OSError("injected temporary-directory exit failure")

    monkeypatch.setattr(
        tempfile,
        "TemporaryDirectory",
        ExitFailingTemporaryDirectory,
    )

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="outer lifecycle boundary",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    assert fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["lifecycle"]["cleanup"]["state"] == "proving_all_absence_routes"
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


def test_failure_publication_rebuilds_lifecycle_from_raw_attempt_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    fake.cleanup_retains = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    original_lifecycle_record = smoke._lifecycle_record
    calls = 0

    def fail_first_final_serialization(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise MemoryError("injected lifecycle serialization failure")
        return original_lifecycle_record(**kwargs)

    monkeypatch.setattr(smoke, "_lifecycle_record", fail_first_final_serialization)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="outer lifecycle boundary",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    assert fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["lifecycle"]["create_invoked"] is True
    assert receipt["lifecycle"]["cleanup"]["state"] == "proving_all_absence_routes"
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


@pytest.mark.parametrize(
    ("daemon_fault", "message"),
    [
        ("drift", "Docker daemon identity changed"),
        ("loss", "Docker daemon identity projection failed"),
    ],
)
def test_daemon_discontinuity_refuses_cleanup_targeting_and_records_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    daemon_fault: str,
    message: str,
) -> None:
    fake = _FakeDocker()
    if daemon_fault == "drift":
        fake.daemon_drift_after_info_calls = 5
    else:
        fake.daemon_loss_after_info_calls = 5
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match=message,
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value.container_state_uncertain is True
    assert len(_operations(fake, "create")) == 1
    assert not _operations(fake, "rm")
    assert set(fake.containers) == {f"{1:064x}"}
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    assert receipt["daemon_projection"] == _daemon_projection()
    lifecycle = receipt["lifecycle"]
    assert lifecycle["authoritative_container_id"] == f"{1:064x}"
    assert lifecycle["candidate_container_ids"] == [f"{1:064x}"]
    assert lifecycle["create_invoked"] is True
    assert lifecycle["cleanup"] == {
        "all_candidate_ids_absent": False,
        "exact_name_absent": False,
        "proven_absent_ids": [],
        "resolved_container_id": None,
        "state": "daemon_precleanup_revalidation",
    }
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


@pytest.mark.parametrize(
    "fault",
    [
        "create_nonzero",
        "create_stderr",
        "create_output_malformed",
        "cidfile_malformed",
        "start_nonzero",
        "start_stderr",
        "bounded_start_failure",
    ],
)
def test_process_failure_matrix_always_removes_and_proves_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    fake = _FakeDocker()
    if fault == "create_nonzero":
        fake.create_returncode = 125
    elif fault == "create_stderr":
        fake.create_stderr = b"injected create stderr\n"
    elif fault == "create_output_malformed":
        fake.create_stdout_override = b"not-a-container-id\n"
    elif fault == "cidfile_malformed":
        fake.cidfile_override = b"partial"
    elif fault == "start_nonzero":
        fake.start_returncode = 7
    elif fault == "start_stderr":
        fake.start_stderr = b"unexpected stderr\n"
    else:
        fake.raise_on_start = smoke.ForagerMatchedV3CpuOciEngineeringSmokeError(
            "injected bounded process timeout or overflow",
            container_state_uncertain=True,
        )
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    expected_uncertainty = fault in {"create_nonzero", "create_stderr"}
    assert caught.value.container_state_uncertain is expected_uncertainty
    assert not fake.containers
    assert not list((request.publication_root / "successes" / "sha256").iterdir())
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is expected_uncertainty
    assert receipt["lifecycle"]["uncertainty_latched"] is expected_uncertainty
    filters = [command[-1] for command in _operations(fake, "ls")]
    assert any(item.startswith("--filter=id=") for item in filters)
    assert any(item.startswith("--filter=name=^/") for item in filters)


def test_base_exception_interrupt_still_removes_container_before_propagating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.raise_on_start = KeyboardInterrupt("injected interruption")
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert not fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is False


def test_base_exception_with_failed_cleanup_records_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.raise_on_start = KeyboardInterrupt("injected interruption")
    fake.cleanup_retains = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(KeyboardInterrupt, match="injected interruption") as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert getattr(caught.value, "container_state_uncertain", None) is True
    assert fake.containers
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["container_state_uncertain"] is True
    replayed = smoke.validate_published_matched_v3_cpu_oci_engineering_smoke_failure(
        request.publication_root,
        build_publication_root=request.build_publication_root,
        artifact_root=request.artifact_root,
        expected_failure_receipt_sha256=failures[0].name,
    )
    assert replayed.container_state_uncertain is True


@pytest.mark.parametrize("failure_kind", ["prestart_drift", "postrun_oom"])
def test_inspection_drift_forbids_success_and_uses_only_corroborated_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    fake = _FakeDocker()
    fake.inspect_drift = failure_kind == "prestart_drift"
    fake.postrun_oom = failure_kind == "postrun_oom"
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    if failure_kind == "prestart_drift":
        assert caught.value.container_state_uncertain is True
        assert fake.containers
        assert not _operations(fake, "rm")
    else:
        assert caught.value.container_state_uncertain is False
        assert not fake.containers
        assert len(_operations(fake, "rm")) == 1
    assert not list((request.publication_root / "successes" / "sha256").iterdir())


def test_final_build_revalidation_mismatch_forbids_success_after_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, validation_results = _install_harness(
        tmp_path,
        monkeypatch,
        fake,
        replay_mismatch=True,
    )

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="post-smoke build publication replay differs",
    ) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert len(validation_results) == 3
    assert len(_operations(fake, "create")) == 2
    assert not fake.containers
    assert not list((request.publication_root / "successes" / "sha256").iterdir())
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    assert caught.value.failure_committed is True
    assert caught.value.failure_receipt_sha256 == failures[0].name
    assert caught.value.failure_full_lineage_validated is False
    assert any(
        "durable engineering-smoke failure receipt:" in note
        for note in caught.value.__notes__
    )
    assert any("full-lineage replay also failed" in note for note in caught.value.__notes__)


def test_final_readback_rejects_toctou_success_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    published = smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)
    original_read = smoke._read_published_file
    success_reads = 0

    def racing_read(
        publication_root: Path,
        *,
        category: str,
        address: str,
        filename: str,
    ) -> bytes:
        nonlocal success_reads
        raw = original_read(
            publication_root,
            category=category,
            address=address,
            filename=filename,
        )
        if category == "successes":
            success_reads += 1
            if success_reads == 2:
                return raw + b"injected-after-validation"
        return raw

    monkeypatch.setattr(smoke, "_read_published_file", racing_read)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="changed during final replay",
    ):
        smoke.validate_published_matched_v3_cpu_oci_engineering_smoke(
            request.publication_root,
            build_publication_root=request.build_publication_root,
            artifact_root=request.artifact_root,
            expected_receipt_sha256=published.receipt_sha256,
        )

    assert success_reads == 2


def test_success_receipt_parser_rejects_false_authority_claim() -> None:
    value = _success_payload_fixture()
    value["classification"]["qualification"] = True
    body = dict(value)
    body.pop("receipt_body_sha256")
    value["receipt_body_sha256"] = _sha(smoke._canonical_json(body))
    raw = smoke._canonical_json(value)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="classification differs",
    ):
        smoke._validate_success_receipt(raw, expected_sha256=_sha(raw))


def test_success_receipt_parser_rejects_duplicate_json_keys() -> None:
    valid_raw = smoke._canonical_json(_success_payload_fixture())
    raw = valid_raw.replace(b"{", b'{"status":"injected-duplicate",', 1)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="duplicate key",
    ):
        smoke._validate_success_receipt(raw, expected_sha256=_sha(raw))


def test_success_receipt_parser_rejects_coherently_rehashed_float() -> None:
    value = _success_payload_fixture()
    value["container_count_created"] = 2.0
    body = dict(value)
    body.pop("receipt_body_sha256")
    value["receipt_body_sha256"] = _sha(_permissive_canonical_json(body))
    raw = _permissive_canonical_json(value)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="floating-point value",
    ):
        smoke._validate_success_receipt(raw, expected_sha256=_sha(raw))


def test_success_receipt_parser_rejects_bool_for_integer_returncode() -> None:
    value = _success_payload_fixture()
    observations = cast(list[dict[str, Any]], value["observations"])
    observed = cast(dict[str, Any], observations[0]["observed"])
    observed["returncode"] = False
    body = dict(value)
    body.pop("receipt_body_sha256")
    value["receipt_body_sha256"] = _sha(smoke._canonical_json(body))
    raw = smoke._canonical_json(value)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="differs from its frozen probe",
    ):
        smoke._validate_success_receipt(raw, expected_sha256=_sha(raw))


def test_failure_receipt_parser_rejects_retry_authority() -> None:
    build = _published_build(Path("/tmp/fixture"))
    value = smoke._failure_payload(
        intent_sha256="1" * 64,
        build_record=smoke._build_record(build),
        phase="runtime_binding",
        error=smoke.ForagerMatchedV3CpuOciEngineeringSmokeError("fixture failure"),
        daemon_projection=None,
        lifecycle=None,
    )
    value["retry_authorized"] = True
    raw = smoke._canonical_json(value)

    with pytest.raises(
        smoke.ForagerMatchedV3CpuOciEngineeringSmokeError,
        match="classification differs",
    ):
        smoke._validate_failure_receipt(raw, expected_sha256=_sha(raw))


def test_failure_payload_truncates_multibyte_error_on_a_valid_utf8_boundary() -> None:
    build = _published_build(Path("/tmp/fixture"))
    value = smoke._failure_payload(
        intent_sha256="1" * 64,
        build_record=smoke._build_record(build),
        phase="runtime_binding",
        error=RuntimeError("€" * 2731),
        daemon_projection=None,
        lifecycle=None,
    )
    message = value["error"]["message"]
    assert type(message) is str
    assert 1 <= len(message.encode("utf-8")) <= smoke._MAX_ERROR_MESSAGE_BYTES
    raw = smoke._canonical_json(value)
    assert smoke._validate_failure_receipt(raw, expected_sha256=_sha(raw)) == value


def test_failure_local_readback_fault_retains_committed_address_without_durable_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeDocker()
    fake.wrong_probe_output = True
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)
    original_read = smoke._read_published_file

    def corrupt_failure_read(
        publication_root: Path,
        *,
        category: str,
        address: str,
        filename: str,
    ) -> bytes:
        raw = original_read(
            publication_root,
            category=category,
            address=address,
            filename=filename,
        )
        return raw + b"injected-readback-corruption" if category == "failures" else raw

    monkeypatch.setattr(smoke, "_read_published_file", corrupt_failure_read)

    with pytest.raises(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    assert caught.value.failure_committed is True
    assert caught.value.failure_receipt_sha256 == failures[0].name
    assert caught.value.failure_full_lineage_validated is None
    assert any("full lineage replay pending" in note for note in caught.value.__notes__)
    assert not any(
        note.startswith("durable engineering-smoke failure receipt:")
        for note in caught.value.__notes__
    )


def test_hostile_exception_rendering_and_attributes_cannot_mask_primary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HostileSmokeError(smoke.ForagerMatchedV3CpuOciEngineeringSmokeError):
        def __init__(self) -> None:
            object.__setattr__(self, "armed", False)
            super().__init__("hidden hostile failure", container_state_uncertain=True)
            object.__setattr__(self, "armed", True)

        def __setattr__(self, name: str, value: Any) -> None:
            if (
                object.__getattribute__(self, "armed")
                and name == "container_state_uncertain"
            ):
                raise RuntimeError("hostile setattr")
            super().__setattr__(name, value)

        def __str__(self) -> str:
            raise RuntimeError("hostile str")

        def add_note(self, note: str) -> None:
            raise RuntimeError(f"hostile note: {note}")

    fake = _FakeDocker()
    primary = HostileSmokeError()
    fake.raise_on_start = primary
    request, _published, _validation_results = _install_harness(tmp_path, monkeypatch, fake)

    with pytest.raises(HostileSmokeError) as caught:
        smoke.execute_and_publish_matched_v3_cpu_oci_engineering_smoke(request)

    assert caught.value is primary
    assert object.__getattribute__(primary, "container_state_uncertain") is False
    failures = list((request.publication_root / "failures" / "sha256").iterdir())
    assert len(failures) == 1
    receipt = json.loads((failures[0] / smoke._FAILURE_FILENAME).read_bytes())
    assert receipt["error"] == {
        "message": "HostileSmokeError",
        "type": "HostileSmokeError",
    }


def test_cli_error_retains_actionable_intent_commit_state() -> None:
    intent_sha = "1" * 64
    error = smoke.MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
        "committed intent replay failed",
        intent_sha256=intent_sha,
    )

    projection = smoke._cli_error(error)

    assert projection["intent_committed"] is True
    assert projection["intent_sha256"] == intent_sha
    assert projection["container_state_uncertain"] is False
    assert len(projection["error"]["message"].encode("utf-8")) <= (
        smoke._MAX_ERROR_MESSAGE_BYTES
    )


def test_cli_error_preserves_unknown_intent_commit_state() -> None:
    error = smoke.MatchedV3CpuOciEngineeringSmokeIntentPublicationUncertainError(
        "intent visibility is unknown",
        intent_sha256="2" * 64,
        intent_committed=None,
    )

    projection = smoke._cli_error(error)

    assert projection["intent_committed"] is None
    assert projection["intent_sha256"] == "2" * 64


def test_cli_error_falls_back_when_exception_class_name_lookup_is_hostile() -> None:
    class HostileNameMeta(type):
        def __getattribute__(cls, name: str) -> Any:
            if name == "__name__":
                raise RuntimeError("hostile class name")
            return super().__getattribute__(name)

    class HostileNameError(RuntimeError, metaclass=HostileNameMeta):
        pass

    projection = smoke._cli_error(HostileNameError("bounded message"))

    assert projection["error"] == {
        "message": "bounded message",
        "type": "UnclassifiedError",
    }


def test_console_script_is_registered() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["alberta-forager-matched-v3-cpu-oci-engineering-smoke"] == (
        "alberta_framework.benchmarks.forager_matched_v3_cpu_oci_engineering_smoke:main"
    )
