"""Fail-closed contracts for hidden-regime calibration readiness receipts."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import stat
import subprocess
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

import alberta_framework.evaluation.hidden_regime_calibration_readiness as readiness
from alberta_framework.evaluation.hidden_regime_calibration_readiness import (
    CERTIFICATION_SPECS,
    PreparedReadinessReceipt,
    ReadinessDraft,
    ReadinessError,
    VerifiedCertificationBundle,
    build_readiness_draft,
    canonical_json_bytes,
    canonical_sha256,
    execute_bound_calibration_worker,
    finalize_readiness_receipt,
    publish_readiness_receipt,
    require_validated_readiness_receipt,
    run_readiness_certifications,
    validate_published_readiness_receipt,
    validate_readiness_receipt,
)

pytestmark = pytest.mark.unit

_RUNNER_MODULE = "alberta_framework.evaluation.hidden_regime_factorial_calibration"
_REAL_SUBPROCESS_RUN = subprocess.run
_REAL_PROTOCOL_BINDING = readiness._protocol_binding
_SHA = "1" * 64


def _module_locator(module: str) -> Path:
    return Path(*module.split(".")).with_suffix(".py")


def _fake_repository(root: Path) -> Path:
    (root / "alberta_framework/evaluation").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "alberta_framework/__init__.py").write_text('"""fake package"""\n')
    (root / "alberta_framework/evaluation/__init__.py").write_text(
        '"""fake evaluation package"""\n'
    )
    for module in readiness._BASE_SOURCE_ROOT_MODULES:
        locator = root / _module_locator(module)
        if module == _RUNNER_MODULE:
            source = """\
\"\"\"Bound fake calibration runner.\"\"\"
import json
import os

def main(argv):
    print(json.dumps({
        \"argv\": list(argv),
        \"cwd_empty\": os.listdir(os.getcwd()) == [],
        \"origin\": __file__,
    }, sort_keys=True))
    return 0
"""
        else:
            source = f'"""Fake source root for {module}."""\n'
        locator.write_text(source, encoding="utf-8")

    functions_by_path: dict[Path, set[str]] = {}
    for spec in CERTIFICATION_SPECS:
        for node_id in spec.node_ids:
            locator_text, separator, function_name = node_id.partition("::")
            if not separator:
                function_name = "test_full_file_certification"
            functions_by_path.setdefault(Path(locator_text), set()).add(function_name)
    for relative, functions in functions_by_path.items():
        (root / relative).write_text(
            '"""Fake certification source."""\n\n'
            "import alberta_framework.evaluation.hidden_regime_factorial_calibration "
            "as bound_runner\n\n"
            + "\n\n".join(
                f"def {function_name}():\n    assert callable(bound_runner.main)"
                for function_name in sorted(functions)
            )
            + "\n",
            encoding="utf-8",
        )
    (root / "tests/conftest.py").write_text('"""Fake pytest configuration."""\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "readiness-fixture"\nversion = "0"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    return root


def _fake_runtime_identity() -> dict[str, object]:
    return {
        "schema": readiness.READINESS_RUNTIME_SCHEMA,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "hexversion": 1,
            "cache_tag": "cpython-312",
            "byteorder": "little",
            "executable_sha256": _SHA,
        },
        "platform": {
            "system": "Linux",
            "release": "test",
            "version_sha256": _SHA,
            "machine": "x86_64",
            "libc": ["glibc", "test"],
            "cpu_count": 1,
        },
        "dependencies": {
            "key_versions": {
                "alberta-framework": "test",
                "chex": "test",
                "jax": "test",
                "jaxlib": "test",
                "numpy": "test",
                "pytest": "test",
                "scipy": "test",
            },
            "installed_distribution_count": 1,
            "installed_distribution_inventory_sha256": _SHA,
        },
        "jax": {
            "default_backend": "cpu",
            "enable_x64": False,
            "config_sha256": _SHA,
            "devices": [
                {
                    "id": 0,
                    "process_index": 0,
                    "platform": "cpu",
                    "device_kind": "test",
                    "local_hardware_id": 0,
                }
            ],
        },
        "environment": [],
    }


def _fake_protocol_binding() -> dict[str, object]:
    manifests: list[object] = [
        {
            "name": "hidden-regime-calibration-a-v1",
            "use_partition": "calibration-only",
            "manifest_payload_sha256": "2" * 64,
        }
    ]
    return {
        "receipt_schema": readiness.CALIBRATION_READINESS_RECEIPT_SCHEMA,
        "design_schema": readiness.DESIGN_SCHEMA,
        "design_envelope_schema": readiness.DESIGN_ENVELOPE_SCHEMA,
        "protocol_status": readiness.PROTOCOL_STATUS,
        "protocol_payload_sha256": readiness.CALIBRATION_DESIGN_PAYLOAD_SHA256,
        "seed_snapshot_sha256": readiness.SEED_SNAPSHOT_SHA256,
        "manifest_bindings": manifests,
        "manifest_bindings_sha256": readiness._protocol_canonical_sha256(manifests),
        "recurrence_eligibility_sha256": "5" * 64,
        "gate_matrix_sha256": "6" * 64,
        "development_summary_schema": readiness.BOUND_DEVELOPMENT_SUMMARY_SCHEMA,
        "primitive_trace_schema": readiness.BOUND_PRIMITIVE_TRACE_SCHEMA,
        "consumed_calibration_namespace_sha256": "7" * 64,
        "matched_case_count": 240,
    }


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    return _fake_repository(tmp_path / "repository")


@pytest.fixture(autouse=True)
def stable_external_bindings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(readiness, "_build_runtime_identity", _fake_runtime_identity)
    monkeypatch.setattr(readiness, "_protocol_binding", _fake_protocol_binding)


def _successful_certification_run(
    calls: list[tuple[tuple[str, ...], Path]],
) -> Callable[..., subprocess.CompletedProcess[bytes]]:
    def run(
        command: tuple[str, ...],
        *,
        cwd: Path,
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        calls.append((tuple(command), Path(cwd)))
        assert command[1:3] == ("-I", "-c")
        assert Path(cwd).name == "source"
        assert (Path(cwd) / "alberta_framework").is_dir()
        return subprocess.CompletedProcess(command, 0, stdout=b"certified\n", stderr=b"")

    return run


def _prepare(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ReadinessDraft, VerifiedCertificationBundle, PreparedReadinessReceipt, list[object]]:
    draft = build_readiness_draft(repository_root=repository)
    calls: list[object] = []
    monkeypatch.setattr(readiness.subprocess, "run", _successful_certification_run(calls))
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
    )
    prepared = finalize_readiness_receipt(draft, bundle)
    return draft, bundle, prepared, calls


def test_draft_is_deterministic_nonexecuting_and_binds_exact_source_support(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("draft construction executed a subprocess")

    monkeypatch.setattr(readiness.subprocess, "run", forbidden)
    first = build_readiness_draft(repository_root=repository)
    second = build_readiness_draft(repository_root=repository)
    assert first.base_body == second.base_body
    assert first.source_archive == second.source_archive
    assert "certification_contract" not in first.base_body
    assert "authorization" not in first.base_body
    assert first.base_body["source_literal_outcome_guard"] == {
        "scope": "source_literals_only_not_managed_or_external_execution_history",
        "learner_outcome_constant": False,
        "ledger_all_false": True,
        "ledger_entry_count": 3,
        "execution_absence_attested": False,
    }
    governance = first.base_body["execution_governance"]
    assert governance["initial_started_record_count"] == 0
    assert governance["initial_completed_record_count"] == 0
    assert governance["initial_protected_record_count"] == 0
    assert governance["protected_execution_permitted"] is False
    assert "cannot prove non-execution" in governance["managed_boundary_scope"]
    assert "external clone" in first.base_body["claim_scope"]
    assert first.base_body["worker_execution"]["allowed_entrypoint_modes"] == [
        "--worker-case-v1",
        "--worker-preflight-v1",
    ]

    source = first.base_body["source_snapshot"]
    assert isinstance(source, dict)
    manifest = source["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["calibration_runner_module"] == _RUNNER_MODULE
    assert _RUNNER_MODULE in manifest["root_modules"]
    support = {item["locator"]: item["role"] for item in manifest["support_files"]}
    assert support["pyproject.toml"] == "dependency_lock"
    assert support["uv.lock"] == "dependency_lock"
    assert support["tests/test_hidden_regime_trace_audit.py"] == "certification_source"
    assert support["tests/test_hidden_regime_checkpoint.py"] == "certification_source"
    with zipfile.ZipFile(io.BytesIO(first.source_archive)) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert "alberta_framework/evaluation/hidden_regime_calibration_readiness.py" in (
            archive.namelist()
        )
        assert "alberta_framework/evaluation/hidden_regime_factorial_calibration.py" in (
            archive.namelist()
        )
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())


def test_live_protocol_binding_matches_final_frozen_digest_and_v5_v3_schemas() -> None:
    binding = _REAL_PROTOCOL_BINDING()
    assert binding["protocol_payload_sha256"] == (
        "735ceb533717e8b71c0159372b44041b2fd533ec14b62e78234de2c3552dd47d"
    )
    assert binding["seed_snapshot_sha256"] == readiness.SEED_SNAPSHOT_SHA256
    assert binding["development_summary_schema"].endswith(".development.v5")
    assert binding["primitive_trace_schema"].endswith(".primitive-trace.v3")


def test_missing_or_symlinked_mandatory_runner_fails_closed(
    repository: Path,
) -> None:
    runner = repository / _module_locator(_RUNNER_MODULE)
    original = runner.read_bytes()
    runner.unlink()
    with pytest.raises(ReadinessError, match="mandatory calibration runner"):
        build_readiness_draft(repository_root=repository)

    outside = repository.parent / "outside.py"
    outside.write_bytes(original)
    runner.symlink_to(outside)
    with pytest.raises(ReadinessError, match="not a regular file"):
        build_readiness_draft(repository_root=repository)


def test_certifications_are_derived_from_verified_extraction_and_sealed(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_readiness_draft(repository_root=repository)
    with pytest.raises(ReadinessError, match="explicit authorization"):
        run_readiness_certifications(
            draft,
            authorize_certification_execution=False,
        )

    calls: list[tuple[tuple[str, ...], Path]] = []
    monkeypatch.setattr(readiness.subprocess, "run", _successful_certification_run(calls))
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
    )
    assert len(calls) == len(CERTIFICATION_SPECS)
    assert all(repository not in cwd.parents and cwd != repository for _, cwd in calls)
    assert tuple(record["status"] for record in bundle.records) == ("passed",) * len(
        CERTIFICATION_SPECS
    )
    assert all(
        record["harness_sha256"] == readiness._CERTIFICATION_BOOTSTRAP_SHA256
        for record in bundle.records
    )
    prepared = finalize_readiness_receipt(draft, bundle)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    ).valid

    forged_records = [dict(record) for record in bundle.records]
    forged_records[0]["status"] = "failed"
    forged = dataclasses.replace(bundle, records=tuple(forged_records))
    with pytest.raises(ReadinessError, match="seal"):
        finalize_readiness_receipt(draft, forged)


def test_certification_harness_really_runs_snapshot_tests_without_checkout(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draft = build_readiness_draft(repository_root=repository)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    bundle = run_readiness_certifications(
        draft,
        authorize_certification_execution=True,
        timeout_seconds_per_group=30,
    )
    assert len(bundle.records) == len(CERTIFICATION_SPECS)
    assert all(record["status"] == "passed" for record in bundle.records)
    assert all(record["stdout"]["byte_size"] > 0 for record in bundle.records)


def test_in_memory_validator_rejects_body_and_archive_tampering(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    ).valid

    changed = copy.deepcopy(prepared.payload)
    changed["body"]["source_literal_outcome_guard"]["ledger_all_false"] = False
    changed["receipt_sha256"] = canonical_sha256(changed["body"])
    validation = validate_readiness_receipt(
        changed,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("source literal guard" in error for error in validation.errors)

    governance_tamper = copy.deepcopy(prepared.payload)
    governance_tamper["body"]["execution_governance"]["initial_started_record_count"] = 1
    governance_tamper["receipt_sha256"] = canonical_sha256(governance_tamper["body"])
    validation = validate_readiness_receipt(
        governance_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("pristine deterministic genesis" in error for error in validation.errors)

    worker_mode_tamper = copy.deepcopy(prepared.payload)
    worker_mode_tamper["body"]["worker_execution"]["allowed_entrypoint_modes"] = [
        "--worker-case-v1"
    ]
    worker_mode_tamper["receipt_sha256"] = canonical_sha256(worker_mode_tamper["body"])
    validation = validate_readiness_receipt(
        worker_mode_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("allowed worker entrypoint modes" in error for error in validation.errors)

    exit_code_type_tamper = copy.deepcopy(prepared.payload)
    exit_code_type_tamper["body"]["certification_contract"]["records"][0]["exit_code"] = False
    records = exit_code_type_tamper["body"]["certification_contract"]["records"]
    exit_code_type_tamper["body"]["certification_contract"]["records_sha256"] = canonical_sha256(
        records
    )
    exit_code_type_tamper["receipt_sha256"] = canonical_sha256(exit_code_type_tamper["body"])
    validation = validate_readiness_receipt(
        exit_code_type_tamper,
        prepared.source_archive,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("strict integer zero" in error for error in validation.errors)

    archive_tamper = bytearray(prepared.source_archive)
    archive_tamper[-1] ^= 1
    validation = validate_readiness_receipt(
        prepared.payload,
        bytes(archive_tamper),
        repository_root=repository,
    )
    assert not validation.valid
    assert any("archive digest" in error for error in validation.errors)


def test_current_source_drift_is_rejected_but_bound_archive_remains_self_valid(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    runner = repository / _module_locator(_RUNNER_MODULE)
    runner.write_text(runner.read_text(encoding="utf-8") + "\nDRIFT = True\n", encoding="utf-8")
    current = validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=True,
    )
    assert not current.valid
    assert any("source closure" in error for error in current.errors)
    assert validate_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
        recheck_current=False,
    ).valid


def test_runner_facing_validator_returns_only_bound_identities(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    validated = require_validated_readiness_receipt(
        prepared.payload,
        prepared.source_archive,
        repository_root=repository,
    )
    assert validated.payload == prepared.payload
    assert validated.receipt_sha256 == prepared.payload["receipt_sha256"]
    assert validated.source_archive_sha256 == readiness._sha256_bytes(prepared.source_archive)
    assert validated.calibration_runner_module == _RUNNER_MODULE
    assert (
        validated.execution_genesis_sha256
        == prepared.payload["body"]["execution_governance"]["genesis_sha256"]
    )

    bad = copy.deepcopy(prepared.payload)
    bad["body"]["authorization"]["protected_candidate_execution_permitted"] = True
    bad["receipt_sha256"] = canonical_sha256(bad["body"])
    with pytest.raises(ReadinessError, match="authorization policy"):
        require_validated_readiness_receipt(
            bad,
            prepared.source_archive,
            repository_root=repository,
        )


def test_publication_is_content_addressed_immutable_new_only_and_no_symlink_root(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "published"
    publication_root.mkdir()
    published = publish_readiness_receipt(
        prepared,
        publication_root,
        authorize_publication=True,
    )
    assert published.directory.name == prepared.payload["receipt_sha256"]
    assert stat.S_IMODE(published.directory.stat().st_mode) == 0o555
    assert stat.S_IMODE(published.receipt_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(published.source_archive_path.stat().st_mode) == 0o444
    assert published.receipt_path.read_bytes() == canonical_json_bytes(prepared.payload)
    assert validate_published_readiness_receipt(
        published.directory,
        repository_root=repository,
    ).valid
    with pytest.raises(FileExistsError, match="overwrite"):
        publish_readiness_receipt(
            prepared,
            publication_root,
            authorize_publication=True,
        )

    actual_root = tmp_path / "actual-root"
    actual_root.mkdir()
    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(actual_root, target_is_directory=True)
    with pytest.raises(ReadinessError, match="symlink"):
        publish_readiness_receipt(
            prepared,
            symlink_root,
            authorize_publication=True,
        )


def test_published_symlink_and_byte_tamper_are_rejected(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = publish_readiness_receipt(prepared, first_root, authorize_publication=True)
    external = tmp_path / "external.json"
    external.write_bytes(canonical_json_bytes(prepared.payload))
    external.chmod(0o444)
    first.directory.chmod(0o755)
    first.receipt_path.unlink()
    first.receipt_path.symlink_to(external)
    first.directory.chmod(0o555)
    validation = validate_published_readiness_receipt(
        first.directory,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("symlink" in error for error in validation.errors)

    second_root = tmp_path / "second"
    second_root.mkdir()
    second = publish_readiness_receipt(prepared, second_root, authorize_publication=True)
    second.directory.chmod(0o755)
    second.source_archive_path.chmod(0o644)
    tampered = bytearray(second.source_archive_path.read_bytes())
    tampered[-1] ^= 1
    second.source_archive_path.write_bytes(tampered)
    second.source_archive_path.chmod(0o444)
    second.directory.chmod(0o555)
    validation = validate_published_readiness_receipt(
        second.directory,
        repository_root=repository,
    )
    assert not validation.valid
    assert any("archive digest" in error for error in validation.errors)


def test_bound_worker_uses_empty_cwd_and_zipimport_snapshot(
    repository: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    publication_root = tmp_path / "worker"
    publication_root.mkdir()
    published = publish_readiness_receipt(prepared, publication_root, authorize_publication=True)
    monkeypatch.setattr(readiness.subprocess, "run", _REAL_SUBPROCESS_RUN)
    with pytest.raises(ReadinessError, match="explicit authorization"):
        execute_bound_calibration_worker(
            published.directory,
            authorize_calibration_execution=False,
        )
    with pytest.raises(ReadinessError, match="entrypoint mode"):
        execute_bound_calibration_worker(
            published.directory,
            ("--unsupported-worker-mode",),
            authorize_calibration_execution=True,
        )
    completed = execute_bound_calibration_worker(
        published.directory,
        ("--worker-preflight-v1", "7"),
        authorize_calibration_execution=True,
        timeout_seconds=30,
    )
    assert completed.returncode == 0, completed.stderr.decode()
    report = json.loads(completed.stdout)
    assert report["argv"] == ["--worker-preflight-v1", "7"]
    assert report["cwd_empty"] is True
    assert "/source.zip/alberta_framework/evaluation/" in report["origin"]


def test_receipt_json_is_canonical_and_digest_excludes_only_self_digest(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _draft, _bundle, prepared, _calls = _prepare(repository, monkeypatch)
    raw = canonical_json_bytes(prepared.payload)
    parsed = json.loads(raw)
    assert raw == canonical_json_bytes(parsed)
    assert parsed["receipt_sha256"] == canonical_sha256(parsed["body"])
    assert "receipt_sha256" not in parsed["body"]
    with pytest.raises(TypeError, match="unsupported JSON type"):
        canonical_json_bytes({"float": 1.0})
