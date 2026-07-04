"""Import smoke for the dependency-light eliza_robot surface.

Proves that the schema, profiles, bridge transport, trajectory-DB, and other
pure-Python modules load with only the small-wheel CI tier installed
(pydantic, pyyaml, numpy, websockets, Pillow) — no jax/mujoco/brax/torch. This
is the fast standalone signal that the extracted robot package is coherent on a
hosted runner; the heavy simulation/RL modules are exercised on GPU hosts, not
here. Run from robot/: `python scripts/ci_import_smoke.py`. Exits non-zero on
the first module that fails to import so CI fails loudly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Run standalone (`python scripts/ci_import_smoke.py`) without an editable
# install: put the robot package root on sys.path so `import eliza_robot`
# resolves the same way pytest's pyproject `pythonpath = ["."]` makes it.
_ROBOT_ROOT = Path(__file__).resolve().parent.parent
if str(_ROBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROBOT_ROOT))

# Modules that must import with only the CI-light dependency tier. Kept explicit
# (not glob-discovered) so a newly-added heavy import in one of these files is a
# visible CI failure to triage, not a silently-skipped module.
IMPORT_SAFE_MODULES = [
    "eliza_robot",
    "eliza_robot.profiles",
    "eliza_robot.profiles.schema",
    "eliza_robot.schema",
    "eliza_robot.schema.canonical",
    "eliza_robot.schema.embodied_context",
    "eliza_robot.schema.hyperscape_adapter",
    "eliza_robot.interfaces",
    "eliza_robot.bridge.types",
    "eliza_robot.bridge.protocol",
    "eliza_robot.bridge.validation",
    "eliza_robot.bridge.safety",
    "eliza_robot.bridge.trace_log",
    "eliza_robot.bridge.async_compat",
    "eliza_robot.bridge.perception",
    "eliza_robot.trajectory_db.models",
    "eliza_robot.trajectory_db.schema",
    "eliza_robot.trajectory_db.db",
    "eliza_robot.curriculum.goal_checker",
    "eliza_robot.curriculum.loader",
    "eliza_robot.erobot.spec",
    "eliza_robot.erobot.components",
    "eliza_robot.erobot.profile",
    "eliza_robot.erobot.bom",
    "eliza_robot.rl.skills.registry",
    "eliza_robot.perception.config",
    "eliza_robot.perception.entity_slots.slot_config",
]


def main() -> int:
    failures: list[tuple[str, str]] = []
    for name in IMPORT_SAFE_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — smoke test: report every failure, don't mask
            failures.append((name, f"{type(exc).__name__}: {exc}"))

    ok = len(IMPORT_SAFE_MODULES) - len(failures)
    print(f"import smoke: {ok}/{len(IMPORT_SAFE_MODULES)} modules imported")
    for name, err in failures:
        print(f"  FAIL {name} -> {err}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
