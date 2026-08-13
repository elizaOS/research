"""Tests for the file-path-only one-shot v3 bootstrap."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

ROOT = Path(__file__).parents[1]
BOOTSTRAP_PATH = (
    ROOT / "alberta_framework" / "evaluation" / "_compositional_future_utility_v3_bootstrap.py"
)

EXPECTED_CLOSURE = (
    "alberta_framework.core.compositional_features",
    "alberta_framework.core.future_utility",
    "alberta_framework.core.resource_manager",
    "alberta_framework.evaluation._compositional_future_utility_calibration_engine",
    "alberta_framework.evaluation._compositional_future_utility_state_gate",
    "alberta_framework.evaluation._compositional_future_utility_v3_evaluator",
    "alberta_framework.evaluation._compositional_future_utility_v3_report_gate",
    "alberta_framework.evaluation._compositional_future_utility_v3_reward_counts",
    "alberta_framework.evaluation.compositional_control_life_development",
    "alberta_framework.evaluation.compositional_discovery_development",
    "alberta_framework.evaluation.compositional_future_utility_calibration_v3_protocol",
    "alberta_framework.evaluation.compositional_future_utility_calibration_v3_source",
    "alberta_framework.evaluation.compositional_future_utility_panel_core",
    "alberta_framework.evaluation.generated_birth_identity_ledger",
    "alberta_framework.evaluation.generated_birth_identity_scrub_epoch",
    "alberta_framework.evaluation.generated_class_lifecycle_scrub",
    "alberta_framework.evaluation.generated_class_recurrence",
    "alberta_framework.evaluation.generated_expression_lineage",
    "alberta_framework.evaluation.generated_reacquisition_epoch",
)

LEDGER_SOURCE = ROOT / "alberta_framework" / "evaluation" / "_one_shot_development_ledger.py"
LOADER_SOURCE = (
    ROOT / "alberta_framework" / "evaluation" / "_compositional_future_utility_declared_loader.py"
)
PRODUCTION_LEDGER_DIRECTORY = (
    ROOT / "outputs" / "compositional_future_utility_calibration_v3" / "one_shot_ledger"
)

FAKE_EVALUATOR = "alberta_framework.evaluation._bootstrap_fake_evaluator"
FAKE_REPORT_GATE = "alberta_framework.evaluation._bootstrap_fake_report_gate"
FAKE_EXTRA = "alberta_framework.evaluation._bootstrap_fake_unused"
FAKE_STAGES = (
    "entry-preflight",
    "before-scan:fake-arm",
    "after-scan:fake-arm",
    "closure-postflight",
    "completion-postflight",
)

_SUBPROCESS_DRIVER = r"""
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import ModuleType

bootstrap_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
action = sys.argv[3]
module_spec = importlib.util.spec_from_file_location("_fake_campaign_bootstrap", bootstrap_path)
if module_spec is None or module_spec.loader is None:
    raise RuntimeError("bootstrap module spec failed")
bootstrap = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = bootstrap
module_spec.loader.exec_module(bootstrap)
raw = json.loads(config_path.read_text(encoding="utf-8"))
raw["repo_root"] = Path(raw["repo_root"])
raw["closure"] = tuple(tuple(record) for record in raw["closure"])
raw["authorization_stages"] = tuple(raw["authorization_stages"])
definition = bootstrap.BootstrapCampaign(**raw)

if action == "describe":
    value = bootstrap.describe_campaign(definition)
elif action == "describe-twice":
    before = sorted(name for name in sys.modules if name.startswith("_one_shot_v3_"))
    first = bootstrap.describe_campaign(definition)
    middle = sorted(name for name in sys.modules if name.startswith("_one_shot_v3_"))
    second = bootstrap.describe_campaign(definition)
    after = sorted(name for name in sys.modules if name.startswith("_one_shot_v3_"))
    value = {
        "equal": first == second,
        "private_before": before,
        "private_middle": middle,
        "private_after": after,
    }
elif action == "execute":
    outcome = bootstrap.execute_campaign(
        definition,
        acknowledgement=definition.acknowledgement,
    )
    private_modules = sorted(
        name for name in sys.modules if name.startswith("_one_shot_v3_")
    )
    if private_modules:
        raise RuntimeError(f"anonymous primitive modules leaked: {private_modules!r}")
    value = outcome.to_config()
elif action == "concurrent":
    owner = bootstrap.CampaignBootstrap(
        definition,
        acknowledgement=definition.acknowledgement,
    )
    values = []
    failures = []
    terminal_observations = []
    lock = threading.Lock()
    terminal_path = definition.repo_root / definition.ledger_relative_path / "terminal.json"
    def run():
        try:
            result = owner.run()
            with lock:
                values.append(result)
                terminal_observations.append(terminal_path.is_file())
        except BaseException as error:
            with lock:
                failures.append(repr(error))
    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"concurrent failures: {failures!r}")
    value = {
        "result_count": len(values),
        "unique_result_identities": len({id(item) for item in values}),
        "cached_identity": owner.completed_value() is values[0],
        "all_returned_after_terminal": all(terminal_observations),
        "unique_report_jsons": len({item.canonical_report_json for item in values}),
        "all_report_views_match": all(
            json.loads(item.canonical_report_json) == item.descriptive_report
            for item in values
        ),
    }
elif action == "recursive":
    owner = bootstrap.CampaignBootstrap(
        definition,
        acknowledgement=definition.acknowledgement,
    )
    owner._execute = lambda capability: owner.run()
    first = None
    second = None
    try:
        owner.run()
    except BaseException as error:
        first = f"{type(error).__name__}:{error}"
    try:
        owner.run()
    except BaseException as error:
        second = f"{type(error).__name__}:{error}"
    value = {"first": first, "second": second}
elif action == "preload":
    preloaded_name = sys.argv[4]
    sys.modules[preloaded_name] = ModuleType(preloaded_name)
    bootstrap.CampaignBootstrap(
        definition,
        acknowledgement=definition.acknowledgement,
    )
    raise AssertionError("preloaded runtime was accepted")
else:
    raise ValueError(f"unknown action: {action}")
print(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
"""


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


FAKE_REPORT_BODY: dict[str, object] = {
    "campaign": "isolated-fake-v3-campaign",
    "development_only": True,
    "panel": {
        "arm_records": [{"arm": "fake-arm", "endpoint": 0.75}],
        "panel_completed": True,
    },
    "schema": "alberta.fake-v3-descriptive-report.v1",
}
FAKE_REPORT_SHA256 = _canonical_sha256(FAKE_REPORT_BODY)
FAKE_REPORT: dict[str, object] = {
    **FAKE_REPORT_BODY,
    "report_sha256": FAKE_REPORT_SHA256,
}
FAKE_CANONICAL_REPORT_JSON = json.dumps(
    FAKE_REPORT,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)
FAKE_TAMPERED_REPORT = {
    **FAKE_REPORT,
    "panel": {
        "arm_records": [{"arm": "fake-arm", "endpoint": 0.5}],
        "panel_completed": True,
    },
}
FAKE_TAMPERED_CANONICAL_REPORT_JSON = json.dumps(
    FAKE_TAMPERED_REPORT,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
    allow_nan=False,
)


def _fake_evaluator_source(
    *,
    ledger_directory: Path,
    mode: str,
    mutation_path: Path | None = None,
) -> str:
    started_path = ledger_directory / "started.json"
    mutation = ""
    if mutation_path is not None:
        mutation = (
            f"Path({str(mutation_path)!r}).write_bytes("
            f"Path({str(mutation_path)!r}).read_bytes() + b'\\n# fake mutation\\n')"
        )
    return f"""\
from pathlib import Path
if not Path({str(started_path)!r}).is_file():
    raise RuntimeError("evaluator imported before started.json")
from dataclasses import dataclass
from alberta_framework.evaluation import _bootstrap_fake_report_gate as report_gate

MODE = {mode!r}
REPORT_SHA256 = {FAKE_REPORT_SHA256!r}
CANONICAL_REPORT_JSON = {FAKE_CANONICAL_REPORT_JSON!r}
TAMPERED_CANONICAL_REPORT_JSON = {FAKE_TAMPERED_CANONICAL_REPORT_JSON!r}

@dataclass(frozen=True, slots=True)
class Snapshot:
    panel_completed: bool
    stage: str
    scans_completed: int

class V3EvaluatorProgress:
    def __init__(self):
        self.panel_completed = False
        self.stage = "not-entered"
        self.scans_completed = 0
    def snapshot(self):
        return Snapshot(self.panel_completed, self.stage, self.scans_completed)

@dataclass(frozen=True, slots=True)
class Result:
    canonical_report_json: str
    report_sha256: str
    progress: Snapshot

def _require(authorizer, capability, bindings, stage):
    if authorizer(capability, stage, bindings) is not True:
        raise PermissionError("fake authorization failed at " + stage)

def evaluate_v3_operational_panel(
    *, attempt_capability, attempt_authorizer, expected_bindings, progress
):
    _require(attempt_authorizer, attempt_capability, expected_bindings, "entry-preflight")
    progress.stage = "entry-authorized"
    if MODE == "early-failure":
        raise RuntimeError("fake-early-failure")
    _require(attempt_authorizer, attempt_capability, expected_bindings, "before-scan:fake-arm")
    progress.stage = "scan-returned"
    _require(attempt_authorizer, attempt_capability, expected_bindings, "after-scan:fake-arm")
    progress.panel_completed = True
    progress.scans_completed = 1
    progress.stage = "panel-returned"
    if MODE == "late-failure":
        raise RuntimeError("fake-late-failure")
    if MODE == "undeclared-import":
        __import__("alberta_framework.evaluation.not_declared")
    if MODE == "displaced-finder":
        import sys
        sys.meta_path.insert(0, object())
    if MODE == "mutate-source":
        {mutation or "raise AssertionError('missing mutation path')"}
    _require(attempt_authorizer, attempt_capability, expected_bindings, "closure-postflight")
    _require(attempt_authorizer, attempt_capability, expected_bindings, "completion-postflight")
    canonical_report_json = CANONICAL_REPORT_JSON
    report_sha256 = REPORT_SHA256
    if MODE == "noncanonical-report":
        canonical_report_json = " " + canonical_report_json
    if MODE == "tampered-report-body":
        canonical_report_json = TAMPERED_CANONICAL_REPORT_JSON
    if MODE == "mismatched-report-sha256":
        report_sha256 = "0" * 64
    return Result(canonical_report_json, report_sha256, progress.snapshot())
"""


def _fake_report_gate_source() -> str:
    return """\
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ExpectedExecutionBindings:
    execution_source_closure_sha256: str
    bootstrap_sha256: str
    ledger_primitive_sha256: str
    declared_loader_sha256: str
    genesis_sha256: str
    started_sha256: str
"""


def _write_fake_repo(
    tmp_path: Path,
    *,
    mode: str = "success",
    include_unused: bool = False,
    mutation_target: str | None = None,
    mutation_path_override: Path | None = None,
) -> tuple[Path, Path, Path]:
    repo = (tmp_path / "fake_repo").resolve()
    evaluation = repo / "alberta_framework" / "evaluation"
    (repo / "alberta_framework" / "core").mkdir(parents=True)
    evaluation.mkdir(parents=True)
    primitive_path = evaluation / "_one_shot_development_ledger.py"
    loader_path = evaluation / "_compositional_future_utility_declared_loader.py"
    evaluator_path = repo / (FAKE_EVALUATOR.replace(".", "/") + ".py")
    report_gate_path = repo / (FAKE_REPORT_GATE.replace(".", "/") + ".py")
    shutil.copyfile(LEDGER_SOURCE, primitive_path)
    shutil.copyfile(LOADER_SOURCE, loader_path)
    report_gate_path.write_text(_fake_report_gate_source(), encoding="utf-8")
    ledger_directory = repo / "outputs" / "fake_campaign" / "one_shot_ledger"
    mutation_path = None
    if mutation_target == "report-gate":
        mutation_path = report_gate_path
    elif mutation_target == "loader":
        mutation_path = loader_path
    elif mutation_target == "ledger":
        mutation_path = primitive_path
    elif mutation_target == "bootstrap":
        if mutation_path_override is None:
            raise ValueError("bootstrap mutation requires a temporary override path")
        mutation_path = mutation_path_override
    evaluator_path.write_text(
        _fake_evaluator_source(
            ledger_directory=ledger_directory,
            mode=mode,
            mutation_path=mutation_path,
        ),
        encoding="utf-8",
    )
    closure = [
        (FAKE_EVALUATOR, FAKE_EVALUATOR.replace(".", "/") + ".py"),
        (FAKE_REPORT_GATE, FAKE_REPORT_GATE.replace(".", "/") + ".py"),
    ]
    if include_unused:
        unused_path = repo / (FAKE_EXTRA.replace(".", "/") + ".py")
        unused_path.write_text("VALUE = 1\n", encoding="utf-8")
        closure.append((FAKE_EXTRA, FAKE_EXTRA.replace(".", "/") + ".py"))
    closure.sort()
    root = 0x10203040
    config = {
        "campaign": "isolated-fake-v3-campaign",
        "development_root": root,
        "development_root_hex": f"0x{root:08X}",
        "protocol_config_sha256": _digest("fake-protocol"),
        "control_protocol_config_sha256": _digest("fake-control"),
        "runtime_config_sha256": _digest("fake-runtime"),
        "consumed_history_sha256": _digest("fake-history"),
        "key_manifest_sha256": _digest("fake-key-manifest"),
        "stream_sha256": _digest("fake-stream"),
        "cadence_bound_stream_sha256": _digest("fake-cadence-stream"),
        "source_envelope_sha256": _digest("fake-source-envelope"),
        "acknowledgement": "I acknowledge this isolated fake one-shot test campaign.",
        "repo_root": str(repo),
        "ledger_relative_path": "outputs/fake_campaign/one_shot_ledger",
        "ledger_primitive_relative_path": (
            "alberta_framework/evaluation/_one_shot_development_ledger.py"
        ),
        "declared_loader_relative_path": (
            "alberta_framework/evaluation/_compositional_future_utility_declared_loader.py"
        ),
        "closure": closure,
        "evaluator_module": FAKE_EVALUATOR,
        "report_gate_module": FAKE_REPORT_GATE,
        "authorization_stages": list(FAKE_STAGES),
    }
    config_path = tmp_path / "fake_config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    return repo, ledger_directory, config_path


def _invoke(
    config_path: Path,
    action: str,
    *extra: str,
    check: bool = True,
    bootstrap_path: Path = BOOTSTRAP_PATH,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _SUBPROCESS_DRIVER,
            str(bootstrap_path),
            str(config_path),
            action,
            *extra,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def _description(config_path: Path) -> dict[str, object]:
    process = _invoke(config_path, "describe")
    value = json.loads(process.stdout)
    assert type(value) is dict
    return value


def _issue_fake_genesis(ledger_directory: Path, description: dict[str, object]) -> bytes:
    ledger_directory.mkdir(parents=True)
    genesis = description["genesis_candidate"]
    payload = (
        json.dumps(genesis, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "ascii"
        )
        + b"\n"
    )
    path = ledger_directory / "genesis.json"
    path.write_bytes(payload)
    path.chmod(0o444)
    return payload


def test_bootstrap_is_pure_stdlib_and_pins_the_exact_production_closure() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom))
    assert not any(
        name == "alberta_framework"
        or name.startswith("alberta_framework.")
        or name == "jax"
        or name.startswith("jax.")
        for name in imports
    )

    module = compile(source, str(BOOTSTRAP_PATH), "exec", ast.PyCF_ONLY_AST)
    assert isinstance(module, ast.Module)
    closure = next(
        node.value
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PRODUCTION_CLOSURE_NAMES"
    )
    assert closure is not None
    assert ast.literal_eval(closure) == EXPECTED_CLOSURE
    assert "initialize_genesis" not in source


def test_exact_production_description_pins_every_binding_without_ledger_write() -> None:
    assert not PRODUCTION_LEDGER_DIRECTORY.exists()

    process = subprocess.run(
        [sys.executable, "-I", str(BOOTSTRAP_PATH.resolve()), "--describe-spec"],
        check=True,
        capture_output=True,
        text=True,
    )
    description = cast(dict[str, object], json.loads(process.stdout))

    assert not PRODUCTION_LEDGER_DIRECTORY.exists()
    assert description["schema"] == "alberta.one-shot-v3-bootstrap.description.v1"
    assert description["campaign"] == (
        "compositional-future-utility-calibration-v3-cadence-separated"
    )
    assert description["ledger_relative_path"] == (
        "outputs/compositional_future_utility_calibration_v3/one_shot_ledger"
    )
    ledger_spec = cast(dict[str, object], description["ledger_spec"])
    assert ledger_spec == {
        "campaign": "compositional-future-utility-calibration-v3-cadence-separated",
        "development_root": 317_707_403,
        "development_root_hex": "0x12EFD48B",
        "protocol_config_sha256": (
            "09b7d06ae720f1a2aeb167ae10e4dbde46dff5437659e431bfff79a8445dc16c"
        ),
        "control_protocol_config_sha256": (
            "208afe0b0b91603e1da73f4b87116259a814d2332bdb107102b403e81ce667ca"
        ),
        "runtime_config_sha256": (
            "48f769d8b53c652b7f6ab251ca31be74ada978af53f9e8e15d04ea6b538720b6"
        ),
        "consumed_history_sha256": (
            "0c61ae4ae11e1e1b056cb481a0c652e37ba7119af9d8b6a5516856e0798c58e6"
        ),
        "key_manifest_sha256": ("ae8ad5a84b6d8f1449e90e71925184ffef46b74edf1a231948475fcf0fe11fd5"),
        "stream_sha256": ("f8fdc3a73c06726686e1b285686219806401e2ff6179cb46ed14200d78bc3758"),
        "cadence_bound_stream_sha256": (
            "ac4447b3c86c2f53acf3731d9e6a2d0b39a8e2552b3968748295700e6cbdebf1"
        ),
        "source_envelope_sha256": (
            "25d10d556df131be2822adb2879720b0624fc4af873458a285ee8a7bfd9e6e41"
        ),
        "execution_source_closure_sha256": ledger_spec["execution_source_closure_sha256"],
        "bootstrap_sha256": hashlib.sha256(BOOTSTRAP_PATH.read_bytes()).hexdigest(),
        "ledger_primitive_sha256": hashlib.sha256(LEDGER_SOURCE.read_bytes()).hexdigest(),
        "declared_loader_sha256": hashlib.sha256(LOADER_SOURCE.read_bytes()).hexdigest(),
        "acknowledgement": (
            "I acknowledge that starting the sole nonpromoting v3 development attempt "
            "irreversibly consumes root 0x12EFD48B; no retry, recovery, winner selection, "
            "evidence promotion, or artifact output is authorized."
        ),
    }
    closure_value = description["execution_source_closure"]
    assert type(closure_value) is list
    closure = cast(list[dict[str, object]], closure_value)
    assert len(closure) == 19
    assert tuple(record["canonical_name"] for record in closure) == EXPECTED_CLOSURE
    for record in closure:
        relative_path = cast(str, record["relative_path"])
        assert relative_path == cast(str, record["canonical_name"]).replace(".", "/") + ".py"
        assert record["sha256"] == hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
    assert ledger_spec["execution_source_closure_sha256"] == _canonical_sha256(closure)
    genesis = cast(dict[str, object], description["genesis_candidate"])
    assert genesis["acknowledgement"] == ledger_spec["acknowledgement"]
    genesis_bindings = cast(dict[str, object], genesis["bindings"])
    for name in (
        "protocol_config_sha256",
        "control_protocol_config_sha256",
        "runtime_config_sha256",
        "consumed_history_sha256",
        "key_manifest_sha256",
        "stream_sha256",
        "cadence_bound_stream_sha256",
        "source_envelope_sha256",
        "execution_source_closure_sha256",
        "bootstrap_sha256",
        "ledger_primitive_sha256",
        "declared_loader_sha256",
    ):
        assert genesis_bindings[name] == ledger_spec[name]
    genesis_body = dict(genesis)
    genesis_sha256 = genesis_body.pop("genesis_sha256")
    assert genesis_sha256 == _canonical_sha256(genesis_body)


def test_fake_description_is_deterministic_relative_and_creates_nothing(
    tmp_path: Path,
) -> None:
    repo, ledger_directory, config_path = _write_fake_repo(tmp_path)
    assert not ledger_directory.exists()

    first = _description(config_path)
    second = _description(config_path)

    assert first == second
    assert not ledger_directory.exists()
    assert first["ledger_relative_path"] == "outputs/fake_campaign/one_shot_ledger"
    closure_value = first["execution_source_closure"]
    assert type(closure_value) is list
    closure_records = cast(list[dict[str, object]], closure_value)
    assert [record["canonical_name"] for record in closure_records] == [
        FAKE_EVALUATOR,
        FAKE_REPORT_GATE,
    ]
    assert all(
        not Path(cast(str, record["relative_path"])).is_absolute() for record in closure_records
    )
    assert str(repo) not in json.dumps(closure_records)
    genesis = cast(dict[str, object], first["genesis_candidate"])
    bindings = cast(dict[str, object], genesis["bindings"])
    ledger_spec = cast(dict[str, object], first["ledger_spec"])
    assert (
        bindings["execution_source_closure_sha256"]
        == ledger_spec["execution_source_closure_sha256"]
    )

    repeated = json.loads(_invoke(config_path, "describe-twice").stdout)
    assert repeated == {
        "equal": True,
        "private_after": [],
        "private_before": [],
        "private_middle": [],
    }


def test_missing_fake_ledger_is_rejected_without_creation(tmp_path: Path) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path)

    process = _invoke(config_path, "execute", check=False)

    assert process.returncode != 0
    assert not ledger_directory.exists()


def test_fake_success_consumes_before_import_and_closes_terminal(tmp_path: Path) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path)
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "execute")
    outcome = json.loads(process.stdout)
    terminal = json.loads((ledger_directory / "terminal.json").read_text(encoding="ascii"))

    assert process.stdout.count("\n") == 1
    assert process.stdout.count('"descriptive_report":') == 1
    assert process.stdout == (
        json.dumps(outcome, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    )
    assert (ledger_directory / "started.json").is_file()
    assert terminal["status"] == "completed"
    assert terminal["panel_completed"] is True
    assert terminal["report_sha256"] == FAKE_REPORT_SHA256
    assert "descriptive_report" not in terminal
    assert outcome["schema"] == "alberta.one-shot-v3-bootstrap.outcome.v2"
    assert outcome["descriptive_report"] == FAKE_REPORT
    assert outcome["report_sha256"] == terminal["report_sha256"]
    assert outcome["terminal_sha256"] == terminal["terminal_sha256"]
    assert set(ledger_directory.iterdir()) == {
        ledger_directory / "genesis.json",
        ledger_directory / "started.json",
        ledger_directory / "terminal.json",
    }
    assert all((path.stat().st_mode & 0o777) == 0o444 for path in ledger_directory.iterdir())


@pytest.mark.parametrize(
    ("mode", "expected_panel_completed", "message"),
    [
        ("early-failure", False, "fake-early-failure"),
        ("late-failure", True, "fake-late-failure"),
        ("undeclared-import", True, "undeclared first-party module import blocked"),
        ("displaced-finder", True, "fake authorization failed at closure-postflight"),
    ],
)
def test_fake_failures_close_once_and_reraise_original(
    tmp_path: Path,
    mode: str,
    expected_panel_completed: bool,
    message: str,
) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path, mode=mode)
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "execute", check=False)
    terminal_path = ledger_directory / "terminal.json"
    terminal_before_retry = terminal_path.read_bytes()

    assert process.returncode != 0
    assert process.stdout == ""
    assert message in process.stderr
    terminal = json.loads(terminal_before_retry)
    assert terminal["status"] == "failed"
    assert terminal["panel_completed"] is expected_panel_completed
    assert terminal["report_sha256"] is None
    assert len(terminal["failure_sha256"]) == 64
    if mode in {"early-failure", "late-failure"}:
        early = mode == "early-failure"
        progress = {
            "available": True,
            "panel_completed": not early,
            "stage": "entry-authorized" if early else "panel-returned",
            "scans_completed": 0 if early else 1,
        }
        expected_failure = {
            "schema": "alberta.one-shot-v3-bootstrap.failure.v1",
            "campaign": "isolated-fake-v3-campaign",
            "development_root": 0x10203040,
            "development_root_hex": "0x10203040",
            "bootstrap_stage": "evaluate-operational-panel",
            "failure_type": "RuntimeError",
            "failure_message": message,
            "panel_completed": not early,
            "progress": progress,
            "attempts_consumed": 1,
            "retry_or_recovery_authorized": False,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
        }
        assert terminal["failure_sha256"] == _canonical_sha256(expected_failure)

    retry = _invoke(config_path, "execute", check=False)
    assert retry.returncode != 0
    assert terminal_path.read_bytes() == terminal_before_retry


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("noncanonical-report", "canonical_report_json is not canonical JSON"),
        ("tampered-report-body", "canonical report SHA-256 does not reconstruct"),
        ("mismatched-report-sha256", "canonical report hash does not match report_sha256"),
    ],
)
def test_report_observability_preflight_rejects_tampering_before_success_terminal(
    tmp_path: Path,
    mode: str,
    message: str,
) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path, mode=mode)
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "execute", check=False)
    terminal_path = ledger_directory / "terminal.json"
    terminal_bytes = terminal_path.read_bytes()
    terminal = json.loads(terminal_bytes)

    assert process.returncode != 0
    assert process.stdout == ""
    assert message in process.stderr
    assert terminal["status"] == "failed"
    assert terminal["panel_completed"] is True
    assert terminal["report_sha256"] is None
    assert terminal["failure_sha256"] == _canonical_sha256(
        {
            "schema": "alberta.one-shot-v3-bootstrap.failure.v1",
            "campaign": "isolated-fake-v3-campaign",
            "development_root": 0x10203040,
            "development_root_hex": "0x10203040",
            "bootstrap_stage": "result-observability-preflight",
            "failure_type": "ValueError",
            "failure_message": message,
            "panel_completed": True,
            "progress": {
                "available": True,
                "panel_completed": True,
                "stage": "panel-returned",
                "scans_completed": 1,
            },
            "attempts_consumed": 1,
            "retry_or_recovery_authorized": False,
            "scientific_promotion_allowed": False,
            "evidence_authorized": False,
            "experiment_output_writes_allowed": False,
        }
    )
    assert set(ledger_directory.iterdir()) == {
        ledger_directory / "genesis.json",
        ledger_directory / "started.json",
        terminal_path,
    }

    retry = _invoke(config_path, "execute", check=False)
    assert retry.returncode != 0
    assert retry.stdout == ""
    assert terminal_path.read_bytes() == terminal_bytes


@pytest.mark.parametrize("mutation_target", ["report-gate", "loader", "ledger"])
def test_fake_post_bind_source_mutation_fails_closed(tmp_path: Path, mutation_target: str) -> None:
    _, ledger_directory, config_path = _write_fake_repo(
        tmp_path,
        mode="mutate-source",
        mutation_target=mutation_target,
    )
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "execute", check=False)
    terminal = json.loads((ledger_directory / "terminal.json").read_text(encoding="ascii"))

    assert process.returncode != 0
    assert "changed during the attempt" in process.stderr
    assert terminal["status"] == "failed"
    assert terminal["panel_completed"] is True


def test_temporary_bootstrap_copy_mutation_fails_closed(tmp_path: Path) -> None:
    bootstrap_copy = (tmp_path / "bootstrap_copy.py").resolve()
    shutil.copyfile(BOOTSTRAP_PATH, bootstrap_copy)
    _, ledger_directory, config_path = _write_fake_repo(
        tmp_path,
        mode="mutate-source",
        mutation_target="bootstrap",
        mutation_path_override=bootstrap_copy,
    )
    description = json.loads(_invoke(config_path, "describe", bootstrap_path=bootstrap_copy).stdout)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(
        config_path,
        "execute",
        check=False,
        bootstrap_path=bootstrap_copy,
    )
    terminal = json.loads((ledger_directory / "terminal.json").read_text(encoding="ascii"))

    assert process.returncode != 0
    assert "bootstrap source changed during the attempt" in process.stderr
    assert terminal["status"] == "failed"
    assert terminal["panel_completed"] is True


def test_declared_but_unused_fake_module_fails_exact_closure(tmp_path: Path) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path, include_unused=True)
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "execute", check=False)
    terminal = json.loads((ledger_directory / "terminal.json").read_text(encoding="ascii"))

    assert process.returncode != 0
    assert "declared execution closure mismatch" in process.stderr
    assert terminal["status"] == "failed"
    assert terminal["panel_completed"] is True


@pytest.mark.parametrize("preloaded_name", ["jaxlib", "alberta_framework.preloaded"])
def test_clean_process_rejection_leaves_fake_genesis_untouched(
    tmp_path: Path, preloaded_name: str
) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path)
    description = _description(config_path)
    genesis = _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "preload", preloaded_name, check=False)

    assert process.returncode != 0
    assert "bootstrap requires a clean process" in process.stderr
    assert (ledger_directory / "genesis.json").read_bytes() == genesis
    assert not (ledger_directory / "started.json").exists()
    assert not (ledger_directory / "terminal.json").exists()


def test_one_owner_seals_concurrent_waiters_before_return(tmp_path: Path) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path)
    description = _description(config_path)
    _issue_fake_genesis(ledger_directory, description)

    process = _invoke(config_path, "concurrent")
    result = json.loads(process.stdout)
    terminal = json.loads((ledger_directory / "terminal.json").read_text(encoding="ascii"))

    assert result == {
        "all_report_views_match": True,
        "all_returned_after_terminal": True,
        "cached_identity": True,
        "result_count": 8,
        "unique_result_identities": 1,
        "unique_report_jsons": 1,
    }
    assert terminal["status"] == "completed"
    assert terminal["panel_completed"] is True
    assert set(ledger_directory.iterdir()) == {
        ledger_directory / "genesis.json",
        ledger_directory / "started.json",
        ledger_directory / "terminal.json",
    }


def test_one_owner_rejects_recursion_and_seals_failure_without_ledger_write(
    tmp_path: Path,
) -> None:
    _, ledger_directory, config_path = _write_fake_repo(tmp_path)

    process = _invoke(config_path, "recursive")
    result = json.loads(process.stdout)

    assert "recursive process-attempt entry is forbidden" in result["first"]
    assert "process attempt is sealed after failure" in result["second"]
    assert not ledger_directory.exists()


def test_symlinked_fake_ledger_parent_is_rejected_before_consumption(
    tmp_path: Path,
) -> None:
    repo, ledger_directory, config_path = _write_fake_repo(tmp_path)
    description = _description(config_path)
    external = (tmp_path / "external_outputs").resolve()
    external_ledger = external / "fake_campaign" / "one_shot_ledger"
    _issue_fake_genesis(external_ledger, description)
    os.symlink(external, repo / "outputs", target_is_directory=True)

    process = _invoke(config_path, "execute", check=False)

    assert process.returncode != 0
    assert "path contains a symlink" in process.stderr
    assert not (ledger_directory / "started.json").exists()
    assert not (external_ledger / "started.json").exists()


def test_temporary_copy_rejects_nonexact_cli_entry_paths(tmp_path: Path) -> None:
    bootstrap_copy = (tmp_path / "bootstrap_copy.py").resolve()
    bootstrap_link = (tmp_path / "bootstrap_link.py").resolve()
    shutil.copyfile(BOOTSTRAP_PATH, bootstrap_copy)
    os.symlink(bootstrap_copy, bootstrap_link)
    commands = (
        [sys.executable, "-m", "bootstrap_copy", "--describe-spec"],
        [
            sys.executable,
            "-I",
            "-c",
            ("import runpy,sys;runpy.run_path(sys.argv[1],run_name='__main__')"),
            str(bootstrap_copy),
            "--describe-spec",
        ],
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import sys;from pathlib import Path;"
                "path=Path(sys.argv[1]);"
                "namespace={'__name__':'__main__','__file__':str(path),"
                "'__package__':None,'__spec__':None};"
                "exec(compile(path.read_bytes(),str(path),'exec'),namespace)"
            ),
            str(bootstrap_copy),
            "--describe-spec",
        ],
        [sys.executable, str(bootstrap_link), "--describe-spec"],
        [sys.executable, bootstrap_copy.name, "--describe-spec"],
    )

    for command in commands:
        process = subprocess.run(
            command,
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )
        assert process.returncode != 0, command
        assert "bootstrap CLI" in process.stderr or "bootstrap source path" in process.stderr

    assert not (tmp_path / "outputs").exists()
