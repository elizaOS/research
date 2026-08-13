"""Focused no-Docker tests for durable matched-v3 CPU OCI build publication."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import threading
import tomllib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from alberta_framework.benchmarks import (
    forager_matched_v3_cpu_oci_build_publication as publication,
)

pytestmark = pytest.mark.unit


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def test_image_identity_rejects_zero_sentinel() -> None:
    with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError):
        publication._require_image_id("sha256:" + "0" * 64, label="test image")


def _make_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    core = repository / "alberta_framework" / "core"
    core.mkdir(parents=True, mode=0o700)
    files = {
        "pyproject.toml": b"[project]\nname = 'publication-fixture'\n",
        "uv.lock": b"version = 1\n",
        "FORAGER_BENCHMARK.md": b"# Fixture protocol\n",
        "alberta_framework/__init__.py": b'"""fixture"""\n',
        "alberta_framework/core/value.py": b"VALUE = 7\n",
    }
    for relative, raw in files.items():
        target = repository / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return repository


class _ExecutorError(RuntimeError):
    def __init__(self, message: str, *, image_state_uncertain: bool) -> None:
        super().__init__(message)
        self.image_state_uncertain = image_state_uncertain


@dataclass
class _Harness:
    repository: Path
    artifact_root: Path
    publication_root: Path
    snapshot: Any
    request: publication.MatchedV3CpuOciBuildPublicationRequest
    context_sha256: str
    plan_sha256: str
    image_id: str
    events: list[str]
    loader_calls: int = 0
    authorization_calls: int = 0
    executor_calls: int = 0
    execution_error: BaseException | None = None
    context_exit_error: BaseException | None = None
    after_authorize: Any | None = None
    execution_succeeded: bool = False

    @property
    def intent_directory(self) -> Path:
        return self.publication_root / "intents" / "sha256" / self.context_sha256

    def failure_digest_from(self, error: BaseException) -> str:
        prefix = "durable matched-v3 OCI failure receipt: sha256:"
        matches = [
            note.removeprefix(prefix)
            for note in getattr(error, "__notes__", ())
            if type(note) is str and note.startswith(prefix)
        ]
        assert len(matches) == 1
        return matches[0]


def _install_harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    repository = _make_repository(tmp_path)
    snapshot = publication.local_snapshot_contract.measure_matched_v3_local_source_snapshot(
        repository_root=repository
    )
    artifact_root = tmp_path / "artifacts"
    publication_root = tmp_path / "publications"
    artifact_root.mkdir(mode=0o700)
    publication_root.mkdir(mode=0o700)
    request = publication.MatchedV3CpuOciBuildPublicationRequest(
        repository_root=repository,
        artifact_root=artifact_root,
        publication_root=publication_root,
        expected_snapshot_manifest_bytes=snapshot.canonical_manifest_bytes,
        expected_snapshot_manifest_sha256=snapshot.full_sha256,
        expected_snapshot_tree_sha256=snapshot.tree_sha256,
        exact_acknowledgement=publication._EXECUTION_ACKNOWLEDGEMENT,
        timeout_seconds=60,
    )
    plan_bytes = b'{"fixture":"plan"}\n'
    plan_sha = _sha(plan_bytes)
    context_receipt_bytes = b'{"fixture":"context"}\n'
    context_sha = _sha(context_receipt_bytes)
    image_id = "sha256:" + "d" * 64
    harness = _Harness(
        repository=repository,
        artifact_root=artifact_root,
        publication_root=publication_root,
        snapshot=snapshot,
        request=request,
        context_sha256=context_sha,
        plan_sha256=plan_sha,
        image_id=image_id,
        events=[],
    )

    production = publication._LoadedProductionInputs(
        issuance_artifacts=cast(Any, object()),
        wheelhouse_archive_bytes=b"fixture-wheelhouse",
        external_source_archive_bytes=b"fixture-external-source",
        external_source_receipt_bytes=b"fixture-external-receipt",
    )
    monkeypatch.setattr(publication, "_load_production_inputs", lambda _root: production)

    local_archive = b"fixture-local-source-archive"
    local_receipt = b'{"fixture":"local-receipt"}\n'

    class FakeLocalBundle:
        archive_sha256 = _sha(local_archive)
        archive_size_bytes = len(local_archive)
        receipt_bytes = local_receipt
        receipt_sha256 = _sha(local_receipt)

        def read_archive_bytes(self) -> bytes:
            return local_archive

        def receipt(self) -> dict[str, Any]:
            return {"archive": {"member_count": 1}}

        def close(self) -> None:
            return None

    @contextmanager
    def retain_local_bundle(**_kwargs: Any) -> Iterator[FakeLocalBundle]:
        yield FakeLocalBundle()

    monkeypatch.setattr(
        publication.local_bundle_contract,
        "retain_matched_v3_local_source_bundle",
        retain_local_bundle,
    )
    monkeypatch.setattr(
        publication.local_bundle_contract,
        "parse_matched_v3_local_source_bundle_receipt",
        lambda _raw, **_kwargs: {"archive": {"member_count": 1}},
    )

    plan_artifacts = publication.plan_contract.CpuOciBuildPlanArtifacts(
        plan_bytes=plan_bytes,
        plan_sha256=plan_sha,
    )
    monkeypatch.setattr(
        publication.plan_contract,
        "build_matched_v3_cpu_oci_build_plan",
        lambda **_kwargs: plan_artifacts,
    )

    class FakeContext:
        receipt_bytes = context_receipt_bytes
        receipt_sha256 = context_sha
        archive_sha256 = "a" * 64
        archive_size_bytes = 10_240
        execution_projection_sha256 = "b" * 64
        plan_sha256 = plan_sha

    @contextmanager
    def retain_context(**_kwargs: Any) -> Iterator[FakeContext]:
        yield FakeContext()
        if harness.context_exit_error is not None and harness.execution_succeeded:
            raise harness.context_exit_error

    monkeypatch.setattr(
        publication.context_contract,
        "retain_matched_v3_cpu_oci_build_context",
        retain_context,
    )

    original_verify = publication.local_snapshot_contract.verify_matched_v3_local_source_snapshot

    def verify_snapshot(**kwargs: Any) -> Any:
        harness.events.append("verify_snapshot")
        return original_verify(**kwargs)

    monkeypatch.setattr(
        publication.local_snapshot_contract,
        "verify_matched_v3_local_source_snapshot",
        verify_snapshot,
    )

    def validate_intent(
        directory: int,
        *,
        expected_context_receipt_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, bytes]]:
        intent_raw = publication._read_unpinned_file_at(
            directory,
            publication._INTENT_FILENAME,
            maximum_size_bytes=publication._MAX_JSON_BYTES,
        )
        intent = publication.parse_matched_v3_cpu_oci_build_intent(
            intent_raw,
            expected_file_sha256=_sha(intent_raw),
        )
        context = cast(Mapping[str, Any], intent["context"])
        assert context["receipt_sha256"] == expected_context_receipt_sha256
        files = publication._replay_directory_fd(
            directory,
            publication._intent_expected_files(intent, intent_raw),
        )
        return intent, files

    monkeypatch.setattr(publication, "_validate_intent_directory_fd", validate_intent)

    class FakeExecutor:
        CPU_OCI_BUILD_EXECUTION_ACKNOWLEDGEMENT = publication._EXECUTION_ACKNOWLEDGEMENT

        @staticmethod
        def parse_matched_v3_cpu_oci_build_execution_receipt(
            raw: bytes,
            *,
            expected_receipt_sha256: str,
        ) -> dict[str, Any]:
            assert _sha(raw) == expected_receipt_sha256
            value = json.loads(raw)
            assert type(value) is dict
            return cast(dict[str, Any], value)

        @staticmethod
        def authorize_matched_v3_cpu_oci_build(**_kwargs: Any) -> object:
            harness.events.append("authorize")
            harness.authorization_calls += 1
            assert harness.intent_directory.is_dir()
            assert set(path.name for path in harness.intent_directory.iterdir()) == {
                publication._CONTEXT_RECEIPT_FILENAME,
                publication._INTENT_FILENAME,
                publication._LOCAL_ARCHIVE_FILENAME,
                publication._LOCAL_RECEIPT_FILENAME,
                publication._PLAN_FILENAME,
                publication._SNAPSHOT_FILENAME,
            }
            if harness.after_authorize is not None:
                harness.after_authorize()
            return object()

        @staticmethod
        def execute_matched_v3_cpu_oci_build(
            *,
            context_capability: Any,
            authorization: object,
            timeout_seconds: int,
        ) -> Any:
            del authorization
            harness.events.append("execute")
            harness.executor_calls += 1
            if harness.execution_error is not None:
                raise harness.execution_error
            receipt = {
                "build": {
                    "image_id": image_id,
                    "timeout_seconds": timeout_seconds,
                },
                "context": {
                    "canonical_receipt": context_capability.receipt_bytes.decode("ascii"),
                    "plan_sha256": context_capability.plan_sha256,
                    "receipt_sha256": context_capability.receipt_sha256,
                },
            }
            receipt_bytes = publication._canonical_json(receipt)
            harness.execution_succeeded = True
            return SimpleNamespace(
                image_id=image_id,
                receipt_bytes=receipt_bytes,
                receipt_sha256=_sha(receipt_bytes),
            )

    fake_executor = FakeExecutor()

    def load_executor() -> FakeExecutor:
        harness.events.append("load_executor")
        harness.loader_calls += 1
        return fake_executor

    monkeypatch.setattr(publication, "_load_executor_contract", load_executor)
    return harness


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    return _install_harness(tmp_path, monkeypatch)


def test_measure_mode_is_new_only_and_never_loads_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = _make_repository(tmp_path)
    output = tmp_path / "snapshot.v1.json"

    def forbidden_loader() -> Any:
        raise AssertionError("measurement loaded execution authority")

    monkeypatch.setattr(publication, "_load_executor_contract", forbidden_loader)
    status = publication.main(
        [
            "measure",
            "--repository-root",
            str(repository),
            "--snapshot-manifest-output",
            str(output),
        ]
    )
    captured = capsys.readouterr()
    record = json.loads(captured.out)

    assert status == 0
    assert captured.err == ""
    assert stat.S_IMODE(output.stat().st_mode) == 0o400
    assert _sha(output.read_bytes()) == record["snapshot_manifest_sha256"]
    assert record["snapshot_tree_sha256"] != record["snapshot_manifest_sha256"]
    assert all(value is False for value in record["claims"].values())

    assert (
        publication.main(
            [
                "measure",
                "--repository-root",
                str(repository),
                "--snapshot-manifest-output",
                str(output),
            ]
        )
        == 2
    )
    duplicate_error = json.loads(capsys.readouterr().err)
    assert duplicate_error["retry_authorized"] is False


def test_console_entry_is_registered_to_separate_publication_module() -> None:
    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["alberta-forager-matched-v3-cpu-oci-build"] == (
        "alberta_framework.benchmarks.forager_matched_v3_cpu_oci_build_publication:main"
    )


def test_request_rejects_tree_pin_that_differs_from_exact_manifest(
    tmp_path: Path,
) -> None:
    repository = _make_repository(tmp_path)
    snapshot = publication.local_snapshot_contract.measure_matched_v3_local_source_snapshot(
        repository_root=repository
    )
    artifact_root = tmp_path / "artifacts"
    publication_root = tmp_path / "publications"
    artifact_root.mkdir(mode=0o700)
    publication_root.mkdir(mode=0o700)

    with pytest.raises(
        publication.ForagerMatchedV3CpuOciBuildPublicationError,
        match="tree pin differs",
    ):
        publication.MatchedV3CpuOciBuildPublicationRequest(
            repository_root=repository,
            artifact_root=artifact_root,
            publication_root=publication_root,
            expected_snapshot_manifest_bytes=snapshot.canonical_manifest_bytes,
            expected_snapshot_manifest_sha256=snapshot.full_sha256,
            expected_snapshot_tree_sha256="9" * 64,
            exact_acknowledgement=publication._EXECUTION_ACKNOWLEDGEMENT,
            timeout_seconds=60,
        )


def test_success_commits_complete_intent_before_one_executor_and_freshly_validates(
    harness: _Harness,
) -> None:
    result = publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.authorization_calls == 1
    assert harness.executor_calls == 1
    assert harness.loader_calls == 1
    assert harness.events[:3] == [
        "verify_snapshot",
        "load_executor",
        "verify_snapshot",
    ]
    assert harness.events.index("authorize") < harness.events.index("execute")
    assert result.context_receipt_sha256 == harness.context_sha256
    assert result.image_id == harness.image_id
    assert set(path.name for path in result.intent_directory.iterdir()) == {
        publication._CONTEXT_RECEIPT_FILENAME,
        publication._INTENT_FILENAME,
        publication._LOCAL_ARCHIVE_FILENAME,
        publication._LOCAL_RECEIPT_FILENAME,
        publication._PLAN_FILENAME,
        publication._SNAPSHOT_FILENAME,
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400 for path in result.intent_directory.iterdir()
    )
    assert stat.S_IMODE(result.intent_directory.stat().st_mode) == 0o500
    assert not any(
        "wheelhouse" in path.name or "external" in path.name
        for path in result.intent_directory.iterdir()
    )

    replayed = publication.validate_published_matched_v3_cpu_oci_build(
        harness.publication_root,
        artifact_root=harness.artifact_root,
        expected_context_receipt_sha256=result.context_receipt_sha256,
        expected_execution_receipt_sha256=result.execution_receipt_sha256,
    )
    assert replayed == result


def test_executor_loader_is_retained_across_final_verify_and_post_intent_work(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_loader = publication._load_executor_contract
    calls = 0

    def once_only_loader() -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("executor loader was consulted after final verification")
        return original_loader()

    monkeypatch.setattr(publication, "_load_executor_contract", once_only_loader)
    publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
    assert calls == 1


def test_duplicate_context_is_a_locatable_no_retry_fence_before_authorization(
    harness: _Harness,
) -> None:
    publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
    authorization_calls = harness.authorization_calls
    executor_calls = harness.executor_calls

    with pytest.raises(publication.MatchedV3CpuOciBuildIntentExistsError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert captured.value.context_receipt_sha256 == harness.context_sha256
    assert captured.value.image_state_uncertain is True
    assert harness.authorization_calls == authorization_calls
    assert harness.executor_calls == executor_calls


def test_duplicate_context_cleanup_failure_cannot_mask_no_retry_fence(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
    executor_calls = harness.executor_calls

    def failed_cleanup(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("injected duplicate staging cleanup failure")

    monkeypatch.setattr(publication, "_cleanup_staging", failed_cleanup)
    with pytest.raises(publication.MatchedV3CpuOciBuildIntentExistsError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert captured.value.context_receipt_sha256 == harness.context_sha256
    assert any("cleanup" in note for note in captured.value.__notes__)
    assert harness.executor_calls == executor_calls


def test_incomplete_existing_intent_is_not_resumed_or_repaired(harness: _Harness) -> None:
    address = harness.intent_directory
    (harness.publication_root / "intents").mkdir(mode=0o700)
    address.parent.mkdir(mode=0o700)
    address.mkdir(mode=0o700)
    incomplete = address / publication._INTENT_FILENAME
    incomplete.write_bytes(b'{"incomplete":true}\n')
    incomplete.chmod(0o400)
    address.chmod(0o500)

    with pytest.raises(publication.MatchedV3CpuOciBuildIntentExistsError):
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert incomplete.read_bytes() == b'{"incomplete":true}\n'


def test_concurrent_same_context_has_exactly_one_executor(harness: _Harness) -> None:
    barrier = threading.Barrier(3)
    results: list[Any] = []
    failures: list[BaseException] = []

    def run() -> None:
        barrier.wait()
        try:
            results.append(
                publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
            )
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], publication.MatchedV3CpuOciBuildIntentExistsError)
    assert harness.authorization_calls == 1
    assert harness.executor_calls == 1


@pytest.mark.parametrize(
    ("error", "expected_phase", "expected_uncertain"),
    [
        (
            _ExecutorError("known pre-start", image_state_uncertain=False),
            "executor_failed_pre_start",
            False,
        ),
        (
            _ExecutorError("known uncertain", image_state_uncertain=True),
            "executor_failed_uncertain",
            True,
        ),
        (RuntimeError("generic after entry"), "executor_failed_uncertain", True),
    ],
)
def test_executor_failures_publish_exact_phase_and_cross_linked_intent(
    harness: _Harness,
    error: BaseException,
    expected_phase: str,
    expected_uncertain: bool,
) -> None:
    harness.execution_error = error

    with pytest.raises(type(error)) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    failure_sha = harness.failure_digest_from(captured.value)
    replayed = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=failure_sha,
    )
    assert replayed.phase == expected_phase
    assert replayed.image_state_uncertain is expected_uncertain
    assert set(path.name for path in replayed.directory.iterdir()) == {
        publication._FAILURE_FILENAME
    }


def _execute_argv(harness: _Harness, manifest_path: Path) -> list[str]:
    return [
        "execute",
        "--repository-root",
        str(harness.repository),
        "--artifact-root",
        str(harness.artifact_root),
        "--publication-root",
        str(harness.publication_root),
        "--snapshot-manifest",
        str(manifest_path),
        "--snapshot-manifest-sha256",
        harness.snapshot.full_sha256,
        "--snapshot-tree-sha256",
        harness.snapshot.tree_sha256,
        "--exact-acknowledgement",
        publication._EXECUTION_ACKNOWLEDGEMENT,
        "--timeout-seconds",
        "60",
    ]


def _write_request_manifest(harness: _Harness, path: Path) -> None:
    path.write_bytes(harness.snapshot.canonical_manifest_bytes)
    path.chmod(0o400)


def test_cli_preserves_generic_post_entry_uncertainty_and_failure_address(
    harness: _Harness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = harness.artifact_root / "request-snapshot.v1.json"
    _write_request_manifest(harness, manifest)
    harness.execution_error = RuntimeError("generic executor failure")

    status = publication.main(_execute_argv(harness, manifest))
    captured = capsys.readouterr()
    error = json.loads(captured.err)

    assert status == 2
    assert captured.out == ""
    assert error["phase"] == "executor_failed_uncertain"
    assert error["image_state_uncertain"] is True
    assert error["durable_failure_receipt_sha256"] is not None
    assert error["retry_authorized"] is False
    replayed = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=error["durable_failure_receipt_sha256"],
    )
    assert replayed.phase == "executor_failed_uncertain"


def test_cli_duplicate_reports_exact_context_without_retry_authority(
    harness: _Harness,
    capsys: pytest.CaptureFixture[str],
) -> None:
    publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
    manifest = harness.artifact_root / "request-snapshot.v1.json"
    _write_request_manifest(harness, manifest)
    executor_calls = harness.executor_calls

    assert publication.main(_execute_argv(harness, manifest)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["context_receipt_sha256"] == harness.context_sha256
    assert error["durable_failure_receipt_sha256"] is None
    assert error["image_state_uncertain"] is True
    assert error["retry_authorized"] is False
    assert harness.executor_calls == executor_calls


def test_execute_rejects_snapshot_drift_before_intent_or_executor(harness: _Harness) -> None:
    (harness.repository / "alberta_framework" / "core" / "value.py").write_text(
        "VALUE = 8\n",
        encoding="ascii",
    )

    with pytest.raises(Exception, match="differs from caller-carried expectations"):
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert not harness.intent_directory.exists()


def test_success_then_context_cleanup_error_is_rich_and_self_contained(
    harness: _Harness,
) -> None:
    harness.context_exit_error = OSError("injected context cleanup error")

    with pytest.raises(
        publication.MatchedV3CpuOciBuildSuccessPublicationUncertainError
    ) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    error = captured.value
    assert error.context_receipt_sha256 == harness.context_sha256
    assert error.image_id == harness.image_id
    assert error.image_state_uncertain is True
    failure_sha = harness.failure_digest_from(error)
    replayed = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=failure_sha,
    )
    assert replayed.phase == "success_publication_failed_after_build"
    assert set(path.name for path in replayed.directory.iterdir()) == {
        publication._EXECUTION_RECEIPT_FILENAME,
        publication._FAILURE_FILENAME,
    }
    assert any((harness.publication_root / "successes" / "sha256").iterdir())


def test_root_exit_after_success_is_rich_and_attempts_durable_failure(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open_root = publication._open_root

    @contextmanager
    def root_exit_failure(
        path: Path,
        *,
        label: str,
        mutable: bool,
    ) -> Iterator[Any]:
        with original_open_root(path, label=label, mutable=mutable) as opened:
            yield opened
            if label == "build publication root":
                raise OSError("injected publication-root exit failure")

    monkeypatch.setattr(publication, "_open_root", root_exit_failure)
    with pytest.raises(
        publication.MatchedV3CpuOciBuildSuccessPublicationUncertainError
    ) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    error = captured.value
    assert error.context_receipt_sha256 == harness.context_sha256
    assert error.image_id == harness.image_id
    failure_sha = harness.failure_digest_from(error)
    monkeypatch.setattr(publication, "_open_root", original_open_root)
    failure = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=failure_sha,
    )
    assert failure.phase == "success_publication_failed_after_build"


def test_success_commit_then_publisher_error_keeps_rich_exact_identities(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish_files = publication._publish_files

    def fail_after_success_commit(*args: Any, **kwargs: Any) -> Path:
        destination = original_publish_files(*args, **kwargs)
        if kwargs.get("category") == "successes":
            raise OSError("injected post-success-commit persistence failure")
        return destination

    monkeypatch.setattr(publication, "_publish_files", fail_after_success_commit)
    with pytest.raises(
        publication.MatchedV3CpuOciBuildSuccessPublicationUncertainError
    ) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert captured.value.context_receipt_sha256 == harness.context_sha256
    assert captured.value.image_id == harness.image_id
    assert captured.value.execution_receipt_sha256 != "0" * 64
    failure_sha = harness.failure_digest_from(captured.value)
    failure = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=failure_sha,
    )
    assert failure.phase == "success_publication_failed_after_build"


def test_post_intent_failure_validator_rejects_wrong_intent_cross_link(
    harness: _Harness,
) -> None:
    publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
    with publication._open_root(
        harness.publication_root,
        label="test publication root",
        mutable=True,
    ) as root:
        wrong = publication._publish_failure(
            root,
            phase="authorization_failed_pre_start",
            error=RuntimeError("synthetic cross-link failure"),
            context_receipt_sha256=harness.context_sha256,
            intent_sha256="e" * 64,
            plan_sha256=harness.plan_sha256,
            authorization_created=False,
            executor_invoked=False,
            build_succeeded=False,
            image_state_uncertain=False,
        )

    with pytest.raises(
        publication.ForagerMatchedV3CpuOciBuildPublicationError,
        match="differs from its durable intent",
    ):
        publication.validate_published_matched_v3_cpu_oci_build_failure(
            harness.publication_root,
            expected_failure_receipt_sha256=wrong.receipt_sha256,
        )


def test_cleanup_error_aggregation_preserves_primary_and_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def failed_close(descriptor: int) -> None:
        calls.append(descriptor)
        raise OSError(f"close {descriptor}")

    monkeypatch.setattr(publication.os, "close", failed_close)
    primary = RuntimeError("primary")
    publication._close_descriptors(
        ((101, "first"), (102, "second"), (101, "duplicate")),
        primary=primary,
    )
    assert calls == [101, 102]
    assert len(primary.__notes__) == 2
    assert all("cleanup also failed" in note for note in primary.__notes__)


def test_public_validators_reject_relative_root_and_path_subclasses(tmp_path: Path) -> None:
    digest = "f" * 64

    class DerivedPath(type(Path())):
        pass

    for invalid in (Path("relative"), Path("/"), DerivedPath(str(tmp_path))):
        with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError):
            publication.validate_published_matched_v3_cpu_oci_build_failure(
                invalid,
                expected_failure_receipt_sha256=digest,
            )


def test_intent_address_swap_during_executor_cannot_return_clean_success(
    harness: _Harness,
) -> None:
    displaced = harness.intent_directory.with_name("displaced-intent")

    def displace_intent() -> None:
        harness.intent_directory.rename(displaced)
        harness.intent_directory.mkdir(mode=0o500)

    harness.after_authorize = displace_intent
    with pytest.raises(
        publication.MatchedV3CpuOciBuildSuccessPublicationUncertainError
    ) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert captured.value.context_receipt_sha256 == harness.context_sha256
    assert captured.value.image_id == harness.image_id
    assert harness.executor_calls == 1
    notes = list(captured.value.__notes__)
    assert notes.count("matched-v3 OCI failure phase: success_publication_failed_after_build") == 1
    assert notes.count("matched-v3 OCI image state uncertain: true") == 1
    assert notes.count(publication._INDETERMINATE_INTENT_DEFERRED_NOTE) == 1
    assert not any(note.startswith("durable matched-v3 OCI failure receipt:") for note in notes)
    assert not any((harness.publication_root / "failures" / "sha256").iterdir())
    error_record = publication._cli_error_record(captured.value)
    assert error_record["phase"] == "success_publication_failed_after_build"
    assert error_record["image_state_uncertain"] is True
    assert error_record["durable_failure_receipt_sha256"] is None
    assert error_record["retry_authorized"] is False
    assert any((harness.publication_root / "successes" / "sha256").iterdir())


def _prepared_publication_root(tmp_path: Path) -> Path:
    root = tmp_path / "publication-root"
    root.mkdir(mode=0o700)
    with publication._open_root(root, label="test publication root", mutable=True) as opened:
        publication._prepare_layout(opened)
    return root


def test_intent_move_then_rename_error_latches_visible_postintent_phase(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rename = publication._rename_new_only

    def move_then_raise(parent: int, source: str, target: str) -> None:
        original_rename(parent, source, target)
        if target == harness.context_sha256:
            raise OSError(errno.EIO, "injected post-syscall rename failure")

    monkeypatch.setattr(publication, "_rename_new_only", move_then_raise)
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.intent_directory.is_dir()
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert "matched-v3 OCI failure phase: authorization_failed_pre_start" in (
        captured.value.__notes__
    )
    failure = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=harness.failure_digest_from(captured.value),
    )
    assert failure.phase == "authorization_failed_pre_start"


def test_raise_after_real_intent_return_uses_latched_postintent_phase(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedAfterIntent(BaseException):
        pass

    original_publish_intent = publication._publish_intent

    def publish_then_raise(*args: Any, **kwargs: Any) -> Path:
        original_publish_intent(*args, **kwargs)
        assert kwargs["commit_state"].committed is True
        raise InjectedAfterIntent("injected after real intent publication return")

    monkeypatch.setattr(publication, "_publish_intent", publish_then_raise)
    with pytest.raises(InjectedAfterIntent) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.intent_directory.is_dir()
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert publication._INDETERMINATE_INTENT_DEFERRED_NOTE not in captured.value.__notes__
    failure = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=harness.failure_digest_from(captured.value),
    )
    assert failure.phase == "authorization_failed_pre_start"


def test_postintent_namespace_swap_defers_unlinked_failure_before_authorization(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rename = publication._rename_new_only
    swapped = False

    def rename_then_swap_namespace(parent: int, source: str, target: str) -> None:
        nonlocal swapped
        original_rename(parent, source, target)
        if target != harness.context_sha256:
            return
        category_fd = os.open("..", publication._directory_flags(), dir_fd=parent)
        try:
            os.rename(
                "sha256",
                "sha256-displaced",
                src_dir_fd=category_fd,
                dst_dir_fd=category_fd,
            )
            os.mkdir("sha256", 0o700, dir_fd=category_fd)
        finally:
            os.close(category_fd)
        swapped = True

    monkeypatch.setattr(publication, "_rename_new_only", rename_then_swap_namespace)
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert swapped is True
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert not harness.intent_directory.exists()
    assert (
        harness.publication_root / "intents" / "sha256-displaced" / harness.context_sha256
    ).is_dir()
    assert not any((harness.publication_root / "failures" / "sha256").iterdir())
    notes = list(captured.value.__notes__)
    assert notes.count("matched-v3 OCI failure phase: intent_publication_uncertain_pre_start") == 1
    assert notes.count("matched-v3 OCI image state uncertain: false") == 1
    assert notes.count(publication._INDETERMINATE_INTENT_DEFERRED_NOTE) == 1
    assert not any(note.startswith("durable matched-v3 OCI failure receipt:") for note in notes)
    error_record = publication._cli_error_record(captured.value)
    assert error_record["durable_failure_receipt_sha256"] is None
    assert error_record["phase"] == "intent_publication_uncertain_pre_start"
    assert error_record["retry_authorized"] is False


def test_postintent_address_displacement_defers_unlinked_failure_before_authorization(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InjectedAfterIntent(BaseException):
        pass

    original_publish_intent = publication._publish_intent
    displaced = harness.intent_directory.with_name("displaced-intent-address")

    def publish_displace_then_raise(*args: Any, **kwargs: Any) -> Path:
        destination = original_publish_intent(*args, **kwargs)
        assert kwargs["commit_state"].committed is True
        destination.rename(displaced)
        raise InjectedAfterIntent("injected after canonical intent address displacement")

    monkeypatch.setattr(publication, "_publish_intent", publish_displace_then_raise)
    with pytest.raises(InjectedAfterIntent) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert not harness.intent_directory.exists()
    assert displaced.is_dir()
    assert not any((harness.publication_root / "failures" / "sha256").iterdir())
    notes = list(captured.value.__notes__)
    assert notes.count("matched-v3 OCI failure phase: intent_publication_uncertain_pre_start") == 1
    assert notes.count("matched-v3 OCI image state uncertain: false") == 1
    assert notes.count(publication._INDETERMINATE_INTENT_DEFERRED_NOTE) == 1
    assert not any(note.startswith("durable matched-v3 OCI failure receipt:") for note in notes)
    error_record = publication._cli_error_record(captured.value)
    assert error_record["durable_failure_receipt_sha256"] is None
    assert error_record["phase"] == "intent_publication_uncertain_pre_start"
    assert error_record["retry_authorized"] is False


def _install_indeterminate_intent_fault(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    *,
    intent_visible: bool,
) -> tuple[Any, Any]:
    original_rename = publication._rename_new_only
    original_reconcile = publication._reconcile_staging_rename

    def indeterminate_rename(parent: int, source: str, target: str) -> None:
        if target == harness.context_sha256:
            if intent_visible:
                original_rename(parent, source, target)
            raise OSError(errno.EIO, "injected indeterminate intent rename")
        original_rename(parent, source, target)

    def indeterminate_reconcile(
        parent: int,
        source: str,
        target: str,
        staging_fd: int,
    ) -> str:
        if target == harness.context_sha256:
            raise OSError(errno.EIO, "injected indeterminate intent reconciliation")
        return original_reconcile(parent, source, target, staging_fd)

    monkeypatch.setattr(publication, "_rename_new_only", indeterminate_rename)
    monkeypatch.setattr(publication, "_reconcile_staging_rename", indeterminate_reconcile)
    return original_rename, original_reconcile


@pytest.mark.parametrize("intent_visible", [False, True])
def test_indeterminate_intent_defers_failure_without_bogus_links(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    intent_visible: bool,
) -> None:
    original_rename, original_reconcile = _install_indeterminate_intent_fault(
        harness,
        monkeypatch,
        intent_visible=intent_visible,
    )
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    notes = list(captured.value.__notes__)
    assert notes.count("matched-v3 OCI failure phase: intent_publication_uncertain_pre_start") == 1
    assert notes.count("matched-v3 OCI image state uncertain: false") == 1
    assert notes.count(publication._INDETERMINATE_INTENT_DEFERRED_NOTE) == 1
    assert not any(note.startswith("durable matched-v3 OCI failure receipt:") for note in notes)
    error_record = publication._cli_error_record(captured.value)
    assert error_record["durable_failure_receipt_sha256"] is None
    assert error_record["phase"] == "intent_publication_uncertain_pre_start"
    assert error_record["retry_authorized"] is False
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert not any((harness.publication_root / "failures" / "sha256").iterdir())

    intent_namespace = harness.publication_root / "intents" / "sha256"
    staging = [path for path in intent_namespace.iterdir() if path.name.startswith("staging-")]
    assert harness.intent_directory.is_dir() is intent_visible
    assert len(staging) == (0 if intent_visible else 1)
    if staging:
        assert stat.S_IMODE(staging[0].stat().st_mode) == 0o500
    if intent_visible:
        monkeypatch.setattr(publication, "_rename_new_only", original_rename)
        monkeypatch.setattr(publication, "_reconcile_staging_rename", original_reconcile)
        with pytest.raises(publication.MatchedV3CpuOciBuildIntentExistsError):
            publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)
        assert harness.authorization_calls == 0
        assert harness.executor_calls == 0


@pytest.mark.parametrize("intent_visible", [False, True])
def test_cli_indeterminate_intent_has_no_failure_address_or_retry_authority(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    intent_visible: bool,
) -> None:
    manifest = harness.artifact_root / "request-snapshot.v1.json"
    _write_request_manifest(harness, manifest)
    _install_indeterminate_intent_fault(
        harness,
        monkeypatch,
        intent_visible=intent_visible,
    )

    assert publication.main(_execute_argv(harness, manifest)) == 2
    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert captured.out == ""
    assert error["durable_failure_receipt_sha256"] is None
    assert error["phase"] == "intent_publication_uncertain_pre_start"
    assert error["image_state_uncertain"] is False
    assert error["retry_authorized"] is False
    assert not any((harness.publication_root / "failures" / "sha256").iterdir())
    assert harness.intent_directory.is_dir() is intent_visible
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0


def test_true_eexist_remains_exact_intent_fence_and_cleans_staging(
    tmp_path: Path,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    address = "4" * 64
    namespace = root_path / "intents" / "sha256"
    existing = namespace / address
    existing.mkdir(mode=0o500)

    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        with pytest.raises(publication.MatchedV3CpuOciBuildIntentExistsError) as captured:
            publication._publish_files(
                opened,
                category="intents",
                address=address,
                files={"intent.v1.json": b"{}\n"},
                intent=True,
            )

    assert captured.value.context_receipt_sha256 == address
    assert set(path.name for path in namespace.iterdir()) == {address}


@pytest.mark.parametrize(
    "noncommit_mode",
    ["rename_error", "rename_return", "reconcile_error", "reconcile_uncertain"],
)
def test_known_noncommit_is_clean_preintent_with_link_free_failure(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    noncommit_mode: str,
) -> None:
    original_rename = publication._rename_new_only
    original_reconcile = publication._reconcile_staging_rename
    intent_reconcile_calls = 0

    def fail_intent_before_move(parent: int, source: str, target: str) -> None:
        if target == harness.context_sha256:
            if noncommit_mode == "rename_return":
                return
            raise OSError(errno.EIO, "injected known-noncommit rename failure")
        original_rename(parent, source, target)

    def reconcile_with_one_uncertain_observation(
        parent: int,
        source: str,
        target: str,
        staging_fd: int,
    ) -> str:
        nonlocal intent_reconcile_calls
        if target == harness.context_sha256:
            intent_reconcile_calls += 1
            if intent_reconcile_calls == 1:
                if noncommit_mode == "reconcile_error":
                    raise OSError(errno.EIO, "injected first reconciliation failure")
                if noncommit_mode == "reconcile_uncertain":
                    return "uncertain"
        return original_reconcile(parent, source, target, staging_fd)

    monkeypatch.setattr(publication, "_rename_new_only", fail_intent_before_move)
    monkeypatch.setattr(
        publication,
        "_reconcile_staging_rename",
        reconcile_with_one_uncertain_observation,
    )
    with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError) as captured:
        publication.execute_and_publish_matched_v3_cpu_oci_build(harness.request)

    assert type(captured.value) is publication.ForagerMatchedV3CpuOciBuildPublicationError
    assert harness.authorization_calls == 0
    assert harness.executor_calls == 0
    assert not harness.intent_directory.exists()
    assert not any((harness.publication_root / "intents" / "sha256").iterdir())
    failure = publication.validate_published_matched_v3_cpu_oci_build_failure(
        harness.publication_root,
        expected_failure_receipt_sha256=harness.failure_digest_from(captured.value),
    )
    assert failure.phase == "pre_intent"
    receipt_raw = (failure.directory / publication._FAILURE_FILENAME).read_bytes()
    receipt = publication.parse_matched_v3_cpu_oci_build_failure_receipt(
        receipt_raw,
        expected_file_sha256=_sha(receipt_raw),
    )
    assert receipt["context_receipt_sha256"] is None
    assert receipt["intent_sha256"] is None
    assert receipt["plan_sha256"] is None
    assert intent_reconcile_calls >= 2


def test_interrupt_after_rename_return_reconciles_exact_visible_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    address = "6" * 64
    commit_state = publication._IntentCommitState()
    original_reconcile = publication._reconcile_staging_rename
    calls = 0

    def interrupt_first_reconciliation(
        parent: int,
        source: str,
        target: str,
        staging_fd: int,
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise KeyboardInterrupt("injected after rename return")
        return original_reconcile(parent, source, target, staging_fd)

    monkeypatch.setattr(
        publication,
        "_reconcile_staging_rename",
        interrupt_first_reconciliation,
    )
    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError):
            publication._publish_files(
                opened,
                category="intents",
                address=address,
                files={"intent.v1.json": b"{}\n"},
                intent=True,
                commit_state=commit_state,
            )

    assert calls >= 2
    assert commit_state.committed is True
    assert (root_path / "intents" / "sha256" / address).is_dir()


def test_snapshot_manifest_move_then_rename_error_is_visible_uncertainty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "measured-snapshot.v1.json"
    raw = b'{"snapshot":"fixture"}\n'
    original_rename = publication._rename_new_only

    def move_then_raise(parent: int, source: str, target: str) -> None:
        original_rename(parent, source, target)
        raise OSError(errno.EIO, "injected snapshot post-syscall failure")

    monkeypatch.setattr(publication, "_rename_new_only", move_then_raise)
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError):
        publication._write_snapshot_manifest_new_only(
            output,
            raw,
            expected_sha256=_sha(raw),
        )

    assert output.read_bytes() == raw
    assert stat.S_IMODE(output.stat().st_mode) == 0o400


@pytest.mark.parametrize(
    "noncommit_mode",
    ["rename_error", "rename_return", "reconcile_error", "reconcile_uncertain"],
)
def test_snapshot_manifest_known_noncommit_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    noncommit_mode: str,
) -> None:
    output = tmp_path / "measured-snapshot.v1.json"
    raw = b'{"snapshot":"fixture"}\n'
    original_reconcile = publication._reconcile_staging_rename
    reconcile_calls = 0

    def fail_before_move(_parent: int, _source: str, _target: str) -> None:
        if noncommit_mode == "rename_return":
            return
        raise OSError(errno.EIO, "injected known snapshot noncommit")

    def reconcile_with_one_uncertain_observation(
        parent: int,
        source: str,
        target: str,
        staging_fd: int,
    ) -> str:
        nonlocal reconcile_calls
        reconcile_calls += 1
        if reconcile_calls == 1:
            if noncommit_mode == "reconcile_error":
                raise OSError(errno.EIO, "injected first snapshot reconciliation failure")
            if noncommit_mode == "reconcile_uncertain":
                return "uncertain"
        return original_reconcile(parent, source, target, staging_fd)

    monkeypatch.setattr(publication, "_rename_new_only", fail_before_move)
    monkeypatch.setattr(
        publication,
        "_reconcile_staging_rename",
        reconcile_with_one_uncertain_observation,
    )
    with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError) as captured:
        publication._write_snapshot_manifest_new_only(
            output,
            raw,
            expected_sha256=_sha(raw),
        )

    assert type(captured.value) is publication.ForagerMatchedV3CpuOciBuildPublicationError
    assert not output.exists()
    assert not any(path.name.startswith("snapshot-staging-") for path in tmp_path.iterdir())
    assert reconcile_calls >= 2


def test_snapshot_manifest_postcommit_fsync_failure_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "measured-snapshot.v1.json"
    raw = b'{"snapshot":"fixture"}\n'
    original_rename = publication._rename_new_only
    original_fsync = publication.os.fsync
    committed = False

    def record_commit(parent: int, source: str, target: str) -> None:
        nonlocal committed
        original_rename(parent, source, target)
        committed = True

    def fail_postcommit_fsync(descriptor: int) -> None:
        if committed:
            raise OSError(errno.EIO, "injected postcommit directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(publication, "_rename_new_only", record_commit)
    monkeypatch.setattr(publication.os, "fsync", fail_postcommit_fsync)
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError) as captured:
        publication._write_snapshot_manifest_new_only(
            output,
            raw,
            expected_sha256=_sha(raw),
        )

    assert captured.value.image_state_uncertain is False
    assert output.read_bytes() == raw


def test_snapshot_manifest_postcommit_root_exit_failure_is_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "measured-snapshot.v1.json"
    raw = b'{"snapshot":"fixture"}\n'
    original_open_root = publication._open_root

    @contextmanager
    def fail_after_root_use(
        path: Path,
        *,
        label: str,
        mutable: bool,
    ) -> Iterator[Any]:
        with original_open_root(path, label=label, mutable=mutable) as opened:
            yield opened
            if label == "snapshot manifest output parent":
                raise OSError("injected snapshot root-exit failure")

    monkeypatch.setattr(publication, "_open_root", fail_after_root_use)
    with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError):
        publication._write_snapshot_manifest_new_only(
            output,
            raw,
            expected_sha256=_sha(raw),
        )

    assert output.read_bytes() == raw


def test_namespace_swap_before_rename_fails_without_visible_address(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    address = "1" * 64
    original_verify = publication._verify_publication_namespace
    calls = 0

    def swap_before_rename(
        root: Any,
        category: str,
        category_fd: int,
        namespace_fd: int,
    ) -> None:
        nonlocal calls
        calls += 1
        if category == "intents" and calls == 3:
            os.rename(
                "sha256",
                "sha256-displaced",
                src_dir_fd=category_fd,
                dst_dir_fd=category_fd,
            )
            os.mkdir("sha256", 0o700, dir_fd=category_fd)
        original_verify(root, category, category_fd, namespace_fd)

    monkeypatch.setattr(
        publication,
        "_verify_publication_namespace",
        swap_before_rename,
    )
    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError):
            publication._publish_files(
                opened,
                category="intents",
                address=address,
                files={"intent.v1.json": b"{}\n"},
                intent=True,
            )

    assert not (root_path / "intents" / "sha256" / address).exists()


def test_namespace_swap_after_rename_is_visible_state_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    address = "2" * 64
    original_rename = publication._rename_new_only

    def rename_then_swap(parent: int, source: str, target: str) -> None:
        original_rename(parent, source, target)
        category_fd = os.open("..", publication._directory_flags(), dir_fd=parent)
        try:
            os.rename(
                "sha256",
                "sha256-displaced",
                src_dir_fd=category_fd,
                dst_dir_fd=category_fd,
            )
            os.mkdir("sha256", 0o700, dir_fd=category_fd)
        finally:
            os.close(category_fd)

    monkeypatch.setattr(publication, "_rename_new_only", rename_then_swap)
    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError):
            publication._publish_files(
                opened,
                category="intents",
                address=address,
                files={"intent.v1.json": b"{}\n"},
                intent=True,
            )


def test_address_swap_after_named_open_is_visible_state_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    address = "3" * 64
    original_open = publication._open_directory_at
    swapped = False

    def open_then_swap(parent: int, name: str, *, label: str) -> int:
        nonlocal swapped
        descriptor = original_open(parent, name, label=label)
        if label == "published intents directory" and not swapped:
            swapped = True
            os.rename(
                name,
                "displaced-address",
                src_dir_fd=parent,
                dst_dir_fd=parent,
            )
            os.mkdir(name, 0o500, dir_fd=parent)
        return descriptor

    monkeypatch.setattr(publication, "_open_directory_at", open_then_swap)
    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        with pytest.raises(publication.MatchedV3CpuOciBuildPublicationStateUncertainError):
            publication._publish_files(
                opened,
                category="intents",
                address=address,
                files={"intent.v1.json": b"{}\n"},
                intent=True,
            )
    assert swapped is True


def _directory_metadata(*, uid: int, gid: int, mode: int) -> os.stat_result:
    return cast(
        os.stat_result,
        SimpleNamespace(
            st_uid=uid,
            st_gid=gid,
            st_mode=stat.S_IFDIR | mode,
        ),
    )


def test_anchored_chain_allows_legitimate_child_mutation_during_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "mutable-root"
    target.mkdir(mode=0o700)
    target_before = target.stat()
    original_open = publication.os.open
    mutated = False

    def open_then_mutate(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal mutated
        descriptor = original_open(path, flags, mode, dir_fd=dir_fd)
        opened = os.fstat(descriptor)
        if not mutated and (opened.st_dev, opened.st_ino) == (
            target_before.st_dev,
            target_before.st_ino,
        ):
            os.mkdir("legitimate-child", 0o700, dir_fd=descriptor)
            mutated = True
        return descriptor

    monkeypatch.setattr(publication.os, "open", open_then_mutate)
    chain = publication._open_anchored_directory_chain(
        target,
        label="concurrently mutated root",
    )
    try:
        chain.verify()
    finally:
        chain.close()

    assert mutated is True
    assert target.stat().st_nlink > target_before.st_nlink


def test_nested_directory_open_allows_legitimate_child_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_path = _prepared_publication_root(tmp_path)
    with publication._open_root(
        root_path,
        label="test publication root",
        mutable=True,
    ) as opened:
        category_fd = publication._open_directory_at(
            opened.descriptor,
            "intents",
            label="test intent category",
        )
        namespace_fd = -1
        try:
            target_before = os.stat(
                "sha256",
                dir_fd=category_fd,
                follow_symlinks=False,
            )
            original_fstat = publication.os.fstat
            mutated = False

            def fstat_then_mutate(descriptor: int) -> os.stat_result:
                nonlocal mutated
                metadata = original_fstat(descriptor)
                if not mutated and (metadata.st_dev, metadata.st_ino) == (
                    target_before.st_dev,
                    target_before.st_ino,
                ):
                    os.mkdir("legitimate-child", 0o700, dir_fd=descriptor)
                    mutated = True
                return metadata

            monkeypatch.setattr(publication.os, "fstat", fstat_then_mutate)
            namespace_fd = publication._open_directory_at(
                category_fd,
                "sha256",
                label="concurrently mutated sha256 namespace",
            )
            assert mutated is True
            assert original_fstat(namespace_fd).st_nlink > target_before.st_nlink
        finally:
            publication._close_descriptors(
                (
                    (namespace_fd, "test namespace"),
                    (category_fd, "test category"),
                ),
                primary=None,
            )


def test_group_writable_ancestor_is_universally_rejected(
    tmp_path: Path,
) -> None:
    group_writable = tmp_path / "group-writable"
    target = group_writable / "target"
    group_writable.mkdir(mode=0o700)
    group_writable.chmod(0o775)
    target.mkdir(mode=0o700)

    with pytest.raises(
        publication.ForagerMatchedV3CpuOciBuildPublicationError,
        match="insecure or non-directory component",
    ):
        publication._open_anchored_directory_chain(
            target,
            label="group-writable test root",
        )


def test_writable_ancestor_policy_is_universal_except_root_sticky() -> None:
    effective_uid = os.geteuid()
    effective_gid = os.getegid()
    assert not publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=effective_uid, gid=effective_gid, mode=0o775)
    )
    assert not publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=effective_uid, gid=effective_gid, mode=0o777)
    )
    assert not publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=effective_uid + 1, gid=effective_gid, mode=0o755)
    )
    assert not publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=0, gid=0, mode=0o777)
    )
    assert publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=0, gid=0, mode=0o1777)
    )
    assert publication._directory_component_metadata_is_secure(
        _directory_metadata(uid=effective_uid, gid=effective_gid, mode=0o700)
    )


def test_anchored_chain_still_rejects_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    target = real / "target"
    target.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(publication.ForagerMatchedV3CpuOciBuildPublicationError):
        publication._open_anchored_directory_chain(
            alias / "target",
            label="symlink-bearing root",
        )
