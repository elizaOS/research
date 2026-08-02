from __future__ import annotations

import ast
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from eliza_robot.bridge.physical_execution import (
    UnsupervisedPhysicalControlError,
    reject_unsupervised_physical_motion,
)

ROOT = Path(__file__).resolve().parents[2]
_CONSTRUCTORS = {
    "AinexRemoteBackend",
    "AsimovRemoteBackend",
    "Ros1RosbridgeBackend",
    "RosBridgeBackend",
}
_SUPERVISED_SERVER_FILES = {
    ROOT / "eliza_robot" / "bridge" / "server.py",
}
_EXPECTED_QUARANTINED_CONSTRUCTORS = Counter(
    {
        "scripts/evidence_aruco_full_anchor_e2e.py:AinexRemoteBackend": 1,
        "scripts/evidence_final_e2e.py:AinexRemoteBackend": 1,
        "scripts/evidence_full_sysid.py:AinexRemoteBackend": 1,
        "scripts/evidence_per_joint_compensation.py:AinexRemoteBackend": 1,
        "scripts/evidence_real_robot_sweep.py:AinexRemoteBackend": 1,
        "scripts/evidence_real_robot_sysid.py:AinexRemoteBackend": 1,
        "scripts/evidence_sim_real_co_execution.py:AinexRemoteBackend": 1,
        "scripts/evidence_state_mirror_e2e.py:AinexRemoteBackend": 1,
        "scripts/evidence_text_to_action_calibrated_e2e.py:AinexRemoteBackend": 1,
        "scripts/evidence_text_to_action_e2e.py:AinexRemoteBackend": 1,
        "scripts/evidence_text_to_action_e2e.py:AsimovRemoteBackend": 1,
        "scripts/evidence_vlm_evaluation_e2e.py:AinexRemoteBackend": 1,
        "scripts/run_asimov1_real_agent.py:AsimovRemoteBackend": 1,
    }
)


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _is_explicit_simulation_constructor(node: ast.Call, name: str) -> bool:
    if name == "AsimovRemoteBackend":
        return any(
            keyword.arg == "mock"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        )
    if name not in {"Ros1RosbridgeBackend", "RosBridgeBackend"}:
        return False
    backend_arg = node.args[0] if node.args else None
    return isinstance(backend_arg, ast.Constant) and backend_arg.value == "ros_sim"


def _enclosing_function(tree: ast.AST, call: ast.Call) -> ast.FunctionDef | ast.AsyncFunctionDef:
    functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= call.lineno <= (node.end_lineno or node.lineno)
    ]
    assert functions, f"physical constructor at line {call.lineno} is not inside a function"
    return min(functions, key=lambda node: (node.end_lineno or node.lineno) - node.lineno)


def test_every_direct_physical_constructor_has_an_earlier_local_quarantine() -> None:
    audited: list[str] = []
    source_paths = [
        *(ROOT / "eliza_robot").rglob("*.py"),
        *(ROOT / "scripts").rglob("*.py"),
    ]
    for path in sorted(source_paths):
        if "tests" in path.parts or path in _SUPERVISED_SERVER_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name not in _CONSTRUCTORS or _is_explicit_simulation_constructor(node, name):
                continue
            function = _enclosing_function(tree, node)
            quarantine_lines = [
                child.lineno
                for child in ast.walk(function)
                if isinstance(child, ast.Call)
                and _call_name(child) == "reject_unsupervised_physical_motion"
            ]
            relative = path.relative_to(ROOT)
            assert any(line < node.lineno for line in quarantine_lines), (
                f"{relative}:{node.lineno} constructs {name} before a local quarantine"
            )
            audited.append(f"{relative}:{name}")

    assert Counter(audited) == _EXPECTED_QUARANTINED_CONSTRUCTORS


def _constructor_trap(calls: list[str]):
    class ConstructorTrap:
        def __init__(self, *_args, **_kwargs) -> None:
            calls.append("constructed")
            raise AssertionError("physical backend constructor ran")

    return ConstructorTrap


def _load_quarantined_async_function(
    path: Path,
    function_name: str,
    *,
    globals_: Mapping[str, object] | None = None,
) -> Callable[..., Any]:
    """Load one exact script function without importing its GPU/media stack.

    The CI-light lane deliberately omits OpenCV and MuJoCo. Compiling the
    function node from the real script keeps these safety tests executable
    there while the separate AST audit above verifies the function's source
    location and quarantine ordering.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == function_name
    ]
    assert len(matches) == 1, f"expected exactly one {function_name} in {path}"
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            matches[0],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {
        "reject_unsupervised_physical_motion": reject_unsupervised_physical_motion,
    }
    if globals_ is not None:
        namespace.update(globals_)
    exec(compile(module, str(path), "exec"), namespace)  # noqa: S102
    loaded = namespace[function_name]
    assert callable(loaded)
    return loaded


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["hiwonder-ainex", "asimov-1"])
async def test_text_to_action_builder_quarantines_before_remote_constructor(
    profile: str,
) -> None:
    calls: list[str] = []
    trap = _constructor_trap(calls)
    build_backend = _load_quarantined_async_function(
        ROOT / "scripts" / "evidence_text_to_action_e2e.py",
        "_build_backend",
        globals_={"AinexRemoteBackend": trap, "AsimovRemoteBackend": trap},
    )

    with pytest.raises(UnsupervisedPhysicalControlError, match="quarantined"):
        await build_backend(
            SimpleNamespace(profile=profile, no_real=False, host="unused", port=0)
        )

    assert calls == []


@pytest.mark.asyncio
async def test_calibrated_builder_quarantines_before_remote_constructor(
) -> None:
    calls: list[str] = []
    build_backend = _load_quarantined_async_function(
        ROOT / "scripts" / "evidence_text_to_action_calibrated_e2e.py",
        "_build_backend",
        globals_={"AinexRemoteBackend": _constructor_trap(calls)},
    )

    with pytest.raises(UnsupervisedPhysicalControlError, match="quarantined"):
        await build_backend(SimpleNamespace())

    assert calls == []


@pytest.mark.asyncio
async def test_aruco_helper_quarantines_before_remote_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    backend_module = ModuleType("eliza_robot.bridge.backends.ainex_remote")
    backend_module.AinexRemoteBackend = _constructor_trap(calls)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, backend_module.__name__, backend_module)
    try_connect_real = _load_quarantined_async_function(
        ROOT / "scripts" / "evidence_aruco_full_anchor_e2e.py",
        "_try_connect_real",
    )

    with pytest.raises(UnsupervisedPhysicalControlError, match="quarantined"):
        await try_connect_real("unused", 0)

    assert calls == []


@pytest.mark.asyncio
async def test_asimov_runner_quarantines_before_remote_constructor(
) -> None:
    calls: list[str] = []
    run_motion = _load_quarantined_async_function(
        ROOT / "scripts" / "run_asimov1_real_agent.py",
        "_run_motion",
        globals_={"AsimovRemoteBackend": _constructor_trap(calls)},
    )

    with pytest.raises(UnsupervisedPhysicalControlError, match="quarantined"):
        await run_motion(SimpleNamespace())

    assert calls == []
