"""Focused contracts for the pure-stdlib declared-source loader primitive."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = (
    ROOT
    / "alberta_framework/evaluation/"
    "_compositional_future_utility_declared_loader.py"
)


def _load_primitive() -> Any:
    name = "_future_utility_declared_loader_unit_test"
    spec = importlib.util.spec_from_file_location(name, LOADER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("declared loader spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


loader = _load_primitive()
pytestmark = pytest.mark.unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_repository(tmp_path: Path) -> Path:
    root = tmp_path.resolve()
    for relative in (
        "alberta_framework",
        "alberta_framework/core",
        "alberta_framework/evaluation",
    ):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text(
            "raise AssertionError('package initializer executed')\n",
            encoding="utf-8",
        )
    return root


_ISOLATED_PRELUDE = """
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

assert not any(
    name == "alberta_framework" or name.startswith("alberta_framework.")
    for name in sys.modules
)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
loader_path = Path(sys.argv[1])
repo_root = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("_isolated_declared_loader", loader_path)
assert spec is not None and spec.loader is not None
primitive = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = primitive
spec.loader.exec_module(primitive)
"""


def _run_isolated(root: Path, body: str, *extra_args: str) -> dict[str, object]:
    script = textwrap.dedent(_ISOLATED_PRELUDE + "\n" + body)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(LOADER_PATH), str(root), *extra_args],
        check=False,
        capture_output=True,
        text=True,
        timeout=15.0,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    output = completed.stdout.strip()
    return json.loads(output) if output else {}


def test_loader_is_pure_stdlib_and_has_no_protocol_or_scientific_authority() -> None:
    tree = ast.parse(LOADER_PATH.read_text(encoding="utf-8"))
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
    assert loader.DEVELOPMENT_ONLY
    assert not loader.ROOT_ISSUANCE_AUTHORIZED
    assert not loader.PROTOCOL_DECLARATION_AUTHORIZED
    assert not loader.SOURCE_GENERATION_AUTHORIZED
    assert not loader.PANEL_EXECUTION_AUTHORIZED
    assert not loader.RUNNER_AVAILABLE
    assert not loader.OUTPUT_WRITES_ALLOWED
    assert not loader.EVIDENCE_AUTHORIZED
    assert not loader.SCIENTIFIC_PROMOTION_ALLOWED
    assert not loader.CROSS_PROCESS_REPLAY_PREVENTED


def test_declared_leaf_and_dependency_load_without_package_initializers(
    tmp_path: Path,
) -> None:
    root = _fake_repository(tmp_path)
    helper = root / "alberta_framework/core/helper.py"
    helper.write_text("VALUE = 40\n", encoding="utf-8")
    leaf = root / "alberta_framework/evaluation/declared_leaf.py"
    leaf.write_text(
        "from alberta_framework.core.helper import VALUE\nRESULT = VALUE + 2\n",
        encoding="utf-8",
    )

    result = _run_isolated(
        root,
        """
helper = primitive.DeclaredModuleBinding(
    "alberta_framework.core.helper",
    "alberta_framework/core/helper.py",
    sys.argv[3],
)
leaf = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    sys.argv[4],
)
stubs = primitive.install_namespace_stubs(repo_root)
finder = primitive.DeclaredSourceFinder(repo_root, (leaf, helper))
assert all(getattr(module, "__path__") == [] for module in stubs)
with finder:
    assert sys.meta_path[0] is finder
    loaded = finder.load(leaf.canonical_name)
    assert loaded.RESULT == 42
    closure = primitive.validate_loaded_closure(finder)
assert finder not in sys.meta_path
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
print(json.dumps({"result": loaded.RESULT, "closure": closure}))
""",
        _sha256(helper),
        _sha256(leaf),
    )

    assert result["result"] == 42
    assert result["closure"] == [
        [
            "alberta_framework.core.helper",
            str(helper),
            _sha256(helper),
        ],
        [
            "alberta_framework.evaluation.declared_leaf",
            str(leaf),
            _sha256(leaf),
        ],
    ]


def test_undeclared_prototype_import_is_blocked_before_side_effect(
    tmp_path: Path,
) -> None:
    root = _fake_repository(tmp_path)
    sentinel = root / "prototype-executed.txt"
    prototype = root / "alberta_framework/core/prototype_agent.py"
    prototype.write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    leaf = root / "alberta_framework/evaluation/declared_leaf.py"
    leaf.write_text("import alberta_framework.core.prototype_agent\n", encoding="utf-8")

    result = _run_isolated(
        root,
        """
leaf = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    sys.argv[3],
)
primitive.install_namespace_stubs(repo_root)
finder = primitive.DeclaredSourceFinder(repo_root, (leaf,))
with finder:
    try:
        finder.load(leaf.canonical_name)
    except ModuleNotFoundError as error:
        assert "undeclared first-party module import blocked" in str(error)
    else:
        raise AssertionError("undeclared Prototype-like module was imported")
    assert finder.observed_execution_closure == finder.expected_closure
    try:
        primitive.validate_loaded_closure(finder)
    except RuntimeError as error:
        assert "missing or replaced" in str(error)
    else:
        raise AssertionError("failed import disappeared from postflight")
print(json.dumps({
    "sentinel_exists": Path(sys.argv[4]).exists(),
    "prototype_loaded": "alberta_framework.core.prototype_agent" in sys.modules,
}))
""",
        _sha256(leaf),
        str(sentinel),
    )

    assert result == {"sentinel_exists": False, "prototype_loaded": False}


def test_binding_hash_name_path_symlink_and_preload_validation_fail_closed(
    tmp_path: Path,
) -> None:
    root = _fake_repository(tmp_path)
    leaf = root / "alberta_framework/evaluation/declared_leaf.py"
    leaf.write_text("VALUE = 1\n", encoding="utf-8")
    linked = root / "alberta_framework/evaluation/linked_leaf.py"
    target = root / "linked-target.py"
    target.write_text("VALUE = 2\n", encoding="utf-8")
    linked.symlink_to(target)

    _run_isolated(
        root,
        """
def must_raise(error_type, fragment, callback):
    try:
        callback()
    except error_type as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"expected {error_type.__name__}: {fragment}")

must_raise(
    ValueError,
    "immediate core/evaluation",
    lambda: primitive.DeclaredModuleBinding(
        "alberta_framework.prototype_agent",
        "alberta_framework/prototype_agent.py",
        sys.argv[3],
    ),
)
must_raise(
    ValueError,
    "does not match",
    lambda: primitive.DeclaredModuleBinding(
        "alberta_framework.evaluation.declared_leaf",
        "alberta_framework/evaluation/wrong.py",
        sys.argv[3],
    ),
)
must_raise(
    ValueError,
    "repository-relative",
    lambda: primitive.DeclaredModuleBinding(
        "alberta_framework.evaluation.declared_leaf",
        "../declared_leaf.py",
        sys.argv[3],
    ),
)
must_raise(
    ValueError,
    "lowercase hexadecimal",
    lambda: primitive.DeclaredModuleBinding(
        "alberta_framework.evaluation.declared_leaf",
        "alberta_framework/evaluation/declared_leaf.py",
        "A" * 64,
    ),
)
binding = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    "0" * 64,
)
must_raise(
    ValueError,
    "digest mismatch",
    lambda: primitive.DeclaredSourceFinder(repo_root, (binding,)),
)
linked = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.linked_leaf",
    "alberta_framework/evaluation/linked_leaf.py",
    sys.argv[4],
)
must_raise(
    ValueError,
    "contains a symlink",
    lambda: primitive.DeclaredSourceFinder(repo_root, (linked,)),
)
valid = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    sys.argv[3],
)
must_raise(
    ValueError,
    "duplicate declared module name",
    lambda: primitive.DeclaredSourceFinder(repo_root, (valid, valid)),
)
sys.modules["alberta_framework.evaluation.preloaded"] = ModuleType(
    "alberta_framework.evaluation.preloaded"
)
must_raise(
    RuntimeError,
    "already loaded",
    lambda: primitive.install_namespace_stubs(repo_root),
)
print("{}")
""",
        _sha256(leaf),
        _sha256(target),
    )


def test_extra_closure_reload_and_delete_reimport_are_rejected(tmp_path: Path) -> None:
    root = _fake_repository(tmp_path)
    leaf = root / "alberta_framework/evaluation/declared_leaf.py"
    leaf.write_text("VALUE = 7\n", encoding="utf-8")

    result = _run_isolated(
        root,
        """
binding = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    sys.argv[3],
)
primitive.install_namespace_stubs(repo_root)
finder = primitive.DeclaredSourceFinder(repo_root, (binding,))
with finder:
    loaded = finder.load(binding.canonical_name)
    try:
        importlib.reload(loaded)
    except ImportError as error:
        assert "reload is forbidden" in str(error)
    else:
        raise AssertionError("declared module reload was accepted")
    del sys.modules[binding.canonical_name]
    try:
        finder.load(binding.canonical_name)
    except ImportError as error:
        assert "cannot be executed twice" in str(error)
    else:
        raise AssertionError("delete/reimport was accepted")
    sys.modules[binding.canonical_name] = loaded
    extra = ModuleType("alberta_framework.evaluation.extra")
    sys.modules[extra.__name__] = extra
    try:
        primitive.validate_loaded_closure(finder)
    except RuntimeError as error:
        assert "unexpected first-party loaded modules" in str(error)
    else:
        raise AssertionError("extra loaded closure was accepted")
print(json.dumps({"value": loaded.VALUE}))
""",
        _sha256(leaf),
    )

    assert result == {"value": 7}


def test_bound_bytes_execute_but_postflight_detects_source_mutation(tmp_path: Path) -> None:
    root = _fake_repository(tmp_path)
    leaf = root / "alberta_framework/evaluation/declared_leaf.py"
    leaf.write_text("VALUE = 'bound'\n", encoding="utf-8")

    result = _run_isolated(
        root,
        """
binding = primitive.DeclaredModuleBinding(
    "alberta_framework.evaluation.declared_leaf",
    "alberta_framework/evaluation/declared_leaf.py",
    sys.argv[3],
)
primitive.install_namespace_stubs(repo_root)
finder = primitive.DeclaredSourceFinder(repo_root, (binding,))
source = repo_root / binding.relative_path
source.write_text("VALUE = 'mutated'\\n", encoding="utf-8")
with finder:
    loaded = finder.load(binding.canonical_name)
    assert loaded.VALUE == "bound"
    try:
        primitive.validate_loaded_closure(finder)
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("post-bind source mutation passed postflight")
print(json.dumps({"value": loaded.VALUE}))
""",
        _sha256(leaf),
    )

    assert result == {"value": "bound"}


def test_process_attempt_has_no_implicit_execution_and_caches_one_success() -> None:
    calls = 0
    attempt: Any

    def builder(capability: object) -> str:
        nonlocal calls
        calls += 1
        assert attempt.authorizes(capability)
        return "only-value"

    attempt = loader.ProcessAttempt(builder)
    assert attempt.completed_value() is None
    assert calls == 0
    assert attempt.get() == "only-value"
    assert attempt.get() == "only-value"
    assert attempt.completed_value() == "only-value"
    assert calls == 1


def test_process_attempt_failure_consumes_and_seals_every_baseexception() -> None:
    calls = 0

    def builder(_capability: object) -> str:
        nonlocal calls
        calls += 1
        raise SystemExit("sealed")

    attempt = loader.ProcessAttempt(builder)
    with pytest.raises(SystemExit, match="sealed"):
        attempt.get()
    with pytest.raises(RuntimeError, match="sealed after failure") as captured:
        attempt.get()
    assert isinstance(captured.value.__cause__, SystemExit)
    assert calls == 1
    assert attempt.completed_value() is None


def test_process_attempt_rejects_recursive_entry_without_deadlock() -> None:
    attempt: Any

    def builder(_capability: object) -> str:
        return str(attempt.get())

    attempt = loader.ProcessAttempt(builder)
    with pytest.raises(RuntimeError, match="recursive process-attempt entry"):
        attempt.get()
    with pytest.raises(RuntimeError, match="sealed after failure"):
        attempt.get()


def test_process_attempt_concurrent_success_and_nonblocking_completed_value() -> None:
    calls = 0
    entered = threading.Event()
    release = threading.Event()

    def builder(_capability: object) -> str:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=5.0)
        return "shared"

    attempt = loader.ProcessAttempt(builder)
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(attempt.get) for _ in range(6)]
        assert entered.wait(timeout=5.0)
        started = time.monotonic()
        assert attempt.completed_value() is None
        assert time.monotonic() - started < 0.5
        release.set()
        assert [future.result(timeout=5.0) for future in futures] == ["shared"] * 6
    assert calls == 1
    assert attempt.completed_value() == "shared"
