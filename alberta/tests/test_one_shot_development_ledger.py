"""Contracts for the generic one-shot development execution ledger."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = (
    ROOT
    / "alberta_framework/evaluation/"
    "_one_shot_development_ledger.py"
)


def _load_primitive() -> Any:
    name = "_one_shot_development_ledger_unit_test"
    spec = importlib.util.spec_from_file_location(name, LEDGER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("ledger primitive spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ledger = _load_primitive()
pytestmark = pytest.mark.unit


def _spec() -> Any:
    return ledger.OneShotDevelopmentLedgerSpec(
        campaign="synthetic-future-utility-v3",
        development_root=0x12345678,
        development_root_hex="0x12345678",
        protocol_config_sha256="1" * 64,
        control_protocol_config_sha256="2" * 64,
        runtime_config_sha256="3" * 64,
        consumed_history_sha256="a" * 64,
        key_manifest_sha256="4" * 64,
        stream_sha256="5" * 64,
        cadence_bound_stream_sha256="6" * 64,
        source_envelope_sha256="7" * 64,
        execution_source_closure_sha256="8" * 64,
        bootstrap_sha256="b" * 64,
        ledger_primitive_sha256=hashlib.sha256(LEDGER_PATH.read_bytes()).hexdigest(),
        declared_loader_sha256="d" * 64,
        acknowledgement="consume synthetic root 0x12345678; no retry",
    )


def _directory(tmp_path: Path) -> Path:
    directory = tmp_path / "ledger"
    directory.mkdir()
    return directory.resolve()


def test_ledger_is_pure_stdlib_and_has_only_ledger_write_authority() -> None:
    tree = ast.parse(LEDGER_PATH.read_text(encoding="utf-8"))
    roots = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    roots |= {
        node.module.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert roots <= set(sys.stdlib_module_names) | {"__future__"}
    assert ledger.DEVELOPMENT_ONLY
    assert not ledger.ROOT_ISSUANCE_AUTHORIZED
    assert not ledger.PANEL_EXECUTION_AUTHORIZED
    assert not ledger.EXPERIMENT_OUTPUT_WRITES_ALLOWED
    assert ledger.LEDGER_WRITES_ONLY
    assert not ledger.EVIDENCE_AUTHORIZED
    assert not ledger.SCIENTIFIC_PROMOTION_ALLOWED
    assert not ledger.RETRY_OR_RECOVERY_AUTHORIZED
    assert ledger.CROSS_PROCESS_REPLAY_PREVENTED_ON_LOCAL_POSIX_FILESYSTEM


def test_spec_binds_the_executing_ledger_source_bytes() -> None:
    with pytest.raises(ValueError, match="executing source bytes"):
        dataclasses.replace(_spec(), ledger_primitive_sha256="c" * 64)


def test_successful_state_machine_is_append_only_and_self_validating(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()

    genesis = ledger.initialize_genesis(directory, spec)
    assert ledger.inspect_state(directory) == "issued-unused"
    assert ledger.validate_genesis(directory, spec) == genesis
    assert oct((directory / ledger.GENESIS_FILENAME).stat().st_mode & 0o777) == "0o444"

    started = ledger.consume_attempt(
        directory,
        spec,
        acknowledgement=spec.acknowledgement,
    )
    assert ledger.inspect_state(directory) == "consumed-pending"
    assert ledger.validate_started(directory, spec) == started
    assert started["attempt_consumed_before_evaluator_import"] is True
    assert started["attempts_consumed"] == 1

    terminal = ledger.record_terminal(
        directory,
        spec,
        status="completed",
        panel_completed=True,
        report_sha256="9" * 64,
    )
    assert ledger.inspect_state(directory) == "consumed-terminal"
    assert (
        ledger.validate_terminal(
            directory,
            spec,
            status="completed",
            panel_completed=True,
            report_sha256="9" * 64,
        )
        == terminal
    )
    assert terminal["scientific_promotion_allowed"] is False
    assert terminal["evidence_authorized"] is False


def test_wrong_acknowledgement_and_invalid_genesis_do_not_consume(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    ledger.initialize_genesis(directory, spec)

    with pytest.raises(ledger.LedgerError, match="acknowledgement"):
        ledger.consume_attempt(directory, spec, acknowledgement="almost")
    assert ledger.inspect_state(directory) == "issued-unused"

    genesis_path = directory / ledger.GENESIS_FILENAME
    os.chmod(genesis_path, 0o644)
    body = json.loads(genesis_path.read_text(encoding="ascii"))
    body["bindings"]["stream_sha256"] = "a" * 64
    genesis_path.write_text(ledger.canonical_json(body) + "\n", encoding="ascii")
    with pytest.raises(ledger.LedgerError, match="permissions|genesis"):
        ledger.consume_attempt(
            directory,
            spec,
            acknowledgement=spec.acknowledgement,
        )
    assert not (directory / ledger.STARTED_FILENAME).exists()


def test_genesis_requires_empty_directory_and_unknown_entries_fail_closed(
    tmp_path: Path,
) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    unknown = directory / "notes.txt"
    unknown.write_text("not part of the ledger\n", encoding="utf-8")

    with pytest.raises(ledger.LedgerError, match="otherwise-empty"):
        ledger.initialize_genesis(directory, spec)
    unknown.unlink()
    ledger.initialize_genesis(directory, spec)
    unknown.write_text("injected\n", encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="unexpected entries"):
        ledger.consume_attempt(
            directory,
            spec,
            acknowledgement=spec.acknowledgement,
        )
    assert not (directory / ledger.STARTED_FILENAME).exists()


def test_any_existing_started_file_permanently_blocks_replay(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    ledger.initialize_genesis(directory, spec)
    started_path = directory / ledger.STARTED_FILENAME
    started_path.write_bytes(b"{")

    assert ledger.inspect_state(directory) == "consumed-pending"
    with pytest.raises(ledger.AttemptAlreadyConsumedError, match="remains consumed"):
        ledger.consume_attempt(
            directory,
            spec,
            acknowledgement=spec.acknowledgement,
        )
    with pytest.raises(ledger.LedgerError, match="permissions|ASCII JSON"):
        ledger.validate_started(directory, spec)
    assert started_path.read_bytes() == b"{"


def test_exactly_one_process_can_create_started_record(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    ledger.initialize_genesis(directory, spec)
    program = textwrap.dedent(
        """
        import hashlib
        import importlib.util
        import sys
        from pathlib import Path

        module_path = Path(sys.argv[1])
        directory = Path(sys.argv[2])
        module_spec = importlib.util.spec_from_file_location("_isolated_ledger", module_path)
        assert module_spec is not None and module_spec.loader is not None
        primitive = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = primitive
        module_spec.loader.exec_module(primitive)
        campaign = primitive.OneShotDevelopmentLedgerSpec(
            campaign="synthetic-future-utility-v3",
            development_root=0x12345678,
            development_root_hex="0x12345678",
            protocol_config_sha256="1" * 64,
            control_protocol_config_sha256="2" * 64,
            runtime_config_sha256="3" * 64,
            consumed_history_sha256="a" * 64,
            key_manifest_sha256="4" * 64,
            stream_sha256="5" * 64,
            cadence_bound_stream_sha256="6" * 64,
            source_envelope_sha256="7" * 64,
            execution_source_closure_sha256="8" * 64,
            bootstrap_sha256="b" * 64,
            ledger_primitive_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
            declared_loader_sha256="d" * 64,
            acknowledgement="consume synthetic root 0x12345678; no retry",
        )
        try:
            primitive.consume_attempt(
                directory,
                campaign,
                acknowledgement=campaign.acknowledgement,
            )
        except primitive.AttemptAlreadyConsumedError:
            print("consumed")
        else:
            print("won")
        """
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-I", "-c", program, str(LEDGER_PATH), str(directory)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    outcomes: list[str] = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=15.0)
        assert process.returncode == 0, stdout + stderr
        outcomes.append(stdout.strip())

    assert outcomes.count("won") == 1
    assert outcomes.count("consumed") == 7
    assert ledger.validate_started(directory, spec) == ledger.started_record(spec)


def test_terminal_failure_is_exact_and_never_replaced(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    ledger.initialize_genesis(directory, spec)
    ledger.consume_attempt(directory, spec, acknowledgement=spec.acknowledgement)
    terminal = ledger.record_terminal(
        directory,
        spec,
        status="failed",
        panel_completed=False,
        failure_sha256="a" * 64,
    )

    assert terminal["report_sha256"] is None
    assert terminal["failure_sha256"] == "a" * 64
    assert ledger.validate_terminal(
        directory,
        spec,
        status="failed",
        panel_completed=False,
        failure_sha256="a" * 64,
    ) == terminal
    with pytest.raises(ledger.LedgerError, match="never replaced"):
        ledger.record_terminal(
            directory,
            spec,
            status="failed",
            panel_completed=True,
            failure_sha256="b" * 64,
        )


@pytest.mark.parametrize(
    ("status", "panel_completed", "report", "failure", "message"),
    [
        ("completed", False, "9" * 64, None, "completed panel"),
        ("completed", True, None, None, "report_sha256"),
        ("completed", True, "9" * 64, "a" * 64, "cannot contain failure"),
        ("failed", False, "9" * 64, None, "cannot contain report"),
        ("failed", False, None, None, "failure_sha256"),
    ],
)
def test_invalid_terminal_combinations_are_rejected(
    status: str,
    panel_completed: bool,
    report: str | None,
    failure: str | None,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ledger.terminal_record(
            _spec(),
            status=status,
            panel_completed=panel_completed,
            report_sha256=report,
            failure_sha256=failure,
        )


def test_symlinked_directory_and_records_are_rejected(tmp_path: Path) -> None:
    directory = _directory(tmp_path)
    spec = _spec()
    ledger.initialize_genesis(directory, spec)
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        ledger.inspect_state(alias)

    started = directory / ledger.STARTED_FILENAME
    target = tmp_path / "outside-started.json"
    target.write_text("{}\n", encoding="ascii")
    started.symlink_to(target)
    with pytest.raises(ledger.LedgerError, match="symlink"):
        ledger.consume_attempt(
            directory,
            spec,
            acknowledgement=spec.acknowledgement,
        )
