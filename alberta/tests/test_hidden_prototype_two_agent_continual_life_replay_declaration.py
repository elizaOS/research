"""Pure-stdlib checks for the U0 replay declaration."""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = (
    ROOT
    / "alberta_framework/evaluation/"
    "hidden_prototype_two_agent_continual_life_replay_declaration.py"
)
U0 = (
    ROOT
    / "alberta_framework/evaluation/"
    "hidden_prototype_two_agent_continual_life_development.py"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_u0_replay_declaration_test", DECLARATION)
    if spec is None or spec.loader is None:
        raise RuntimeError("declaration module spec unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _return_keys(tree: ast.Module, name: str) -> set[str]:
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)
    dictionaries = [node.value for node in ast.walk(function)
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)]
    payload = max(dictionaries, key=lambda item: len(item.keys))
    return {cast(str, ast.literal_eval(key)) for key in payload.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def _parity_keys(tree: ast.Module) -> set[str]:
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_run_arm")
    value = next(node.value for node in ast.walk(function) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "parity"
                         for target in node.targets) and isinstance(node.value, ast.Dict))
    return {cast(str, ast.literal_eval(key)) for key in value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def _coverage(fields: dict[str, tuple[str, ...]]) -> set[str]:
    return {name for group in fields.values() for name in group}


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        elif key in result:
            raise AssertionError(f"overlapping expectation leaf: {key}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _report(module: ModuleType) -> dict[str, Any]:
    expected = _merge(module.OBSERVED_EXPECTATIONS, module.STATIC_EXPECTATIONS)
    arms = cast(dict[str, dict[str, Any]], expected.pop("runs_by_arm"))
    expected["runs"] = list(arms.values())
    return expected


class DeclarationTests(unittest.TestCase):
    module: ModuleType
    tree: ast.Module

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load()
        cls.tree = ast.parse(U0.read_text(encoding="utf-8"))

    def test_pure_stdlib_consumed_manifest_is_now_source_invalid(self) -> None:
        tree = ast.parse(DECLARATION.read_text(encoding="utf-8"))
        roots = {alias.name.split(".", 1)[0] for node in tree.body
                 if isinstance(node, ast.Import) for alias in node.names}
        roots |= {node.module.split(".", 1)[0] for node in tree.body
                  if isinstance(node, ast.ImportFrom) and node.module is not None}
        self.assertTrue(roots <= set(sys.stdlib_module_names) | {"__future__"})
        errors = self.module.validate_semantic_source_manifest(ROOT)
        self.assertEqual(len(errors), 1)
        self.assertTrue(any(
            "alberta_framework/core/prototype_agent.py" in error
            and "semantic source digest mismatch" in error
            for error in errors
        ))
        self.assertEqual(self.module.semantic_source_manifest_digest(),
                         self.module.SEMANTIC_SOURCE_MANIFEST_DIGEST)

    def test_coverage_partitions_are_exhaustive(self) -> None:
        self.assertEqual(_coverage(self.module.METRIC_COVERAGE),
                         _return_keys(self.tree, "_metrics"))
        self.assertEqual(_coverage(self.module.RESOURCE_COVERAGE),
                         _return_keys(self.tree, "_resources"))
        self.assertEqual(_coverage(self.module.WORK_COVERAGE),
                         _return_keys(self.tree, "_work"))
        self.assertEqual(_coverage(self.module.PARITY_COVERAGE), _parity_keys(self.tree))
        self.assertEqual(len(self.module.NEW_WORK_UNOBSERVED), 7)

    def test_matching_fields_cannot_override_consumed_source_invalidity(self) -> None:
        report = _report(self.module)
        result = self.module.compare_declared_replay(report, ROOT)
        self.assertFalse(result["declared_fields_exact"])
        self.assertFalse(result["source_manifest_valid"])
        self.assertEqual(result["mismatches"], ())
        self.assertEqual(result["coverage"], "partial")
        self.assertFalse(result["full_report_identity"])
        self.assertEqual(result["source_manifest_coverage"],
                         "selected-direct-files-not-transitive-closure")
        self.assertFalse(result["runtime_identity_bound"])
        self.assertEqual(result["conclusion"], "source-manifest-mismatch")
        runs = cast(list[dict[str, Any]], report["runs"])
        runs[0]["metrics"]["mean_horde_squared_error"] += 1.0
        mismatch = self.module.compare_declared_replay(report, ROOT)
        self.assertFalse(mismatch["declared_fields_exact"])
        self.assertEqual(
            mismatch["mismatches"],
            ("runs_by_arm.hidden_inferred_full.metrics.mean_horde_squared_error",),
        )
        self.assertEqual(mismatch["conclusion"], "source-manifest-mismatch")

    def test_unknowns_and_new_counters_are_explicit(self) -> None:
        declaration = self.module.PRE_RUN_DECLARATION
        self.assertFalse(declaration["full_report_identity_claim_allowed"])
        self.assertEqual(declaration["source_manifest_coverage"],
                         "selected-direct-files-not-transitive-closure")
        self.assertFalse(declaration["runtime_identity_bound"])
        self.assertIn("runs_by_arm.*.metrics.context_onehots_used",
                      declaration["unknown_report_paths"])
        self.assertEqual(set(self.module.NEW_WORK_UNOBSERVED),
                         set(self.module.WORK_COVERAGE["static-unobserved"]))


if __name__ == "__main__":
    unittest.main()
