"""Pure-stdlib declared-source loading for bounded future-utility processes.

This module binds canonical first-party module names to exact source bytes and
loads them without executing ``alberta_framework`` package initializers.  It is
an import-integrity primitive, not a sandbox: arbitrary file reads, dynamic
``exec``, ``runpy``, native loading, and third-party runtime/compiler closure
remain outside its claim.

The module declares no protocol, namespace, root, key, stream, panel, writer,
selection rule, evidence path, or promotion authority.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path, PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Final, Literal, cast

DEVELOPMENT_ONLY: Final = True
ROOT_ISSUANCE_AUTHORIZED: Final = False
PROTOCOL_DECLARATION_AUTHORIZED: Final = False
SOURCE_GENERATION_AUTHORIZED: Final = False
PANEL_EXECUTION_AUTHORIZED: Final = False
RUNNER_AVAILABLE: Final = False
OUTPUT_WRITES_ALLOWED: Final = False
EVIDENCE_AUTHORIZED: Final = False
SCIENTIFIC_PROMOTION_ALLOWED: Final = False
CROSS_PROCESS_REPLAY_PREVENTED: Final = False

_FIRST_PARTY_ROOT: Final = "alberta_framework"
_NAMESPACE_NAMES: Final = (
    "alberta_framework",
    "alberta_framework.core",
    "alberta_framework.evaluation",
)
_DECLARED_LEAF_PREFIXES: Final = (
    "alberta_framework.core.",
    "alberta_framework.evaluation.",
)
_NAMESPACE_RELATIVE_PATHS: Final = MappingProxyType(
    {
        "alberta_framework": "alberta_framework",
        "alberta_framework.core": "alberta_framework/core",
        "alberta_framework.evaluation": "alberta_framework/evaluation",
    }
)
_NAMESPACE_MARKER = object()


def _is_first_party_name(name: str) -> bool:
    return name == _FIRST_PARTY_ROOT or name.startswith(_FIRST_PARTY_ROOT + ".")


def _canonical_relative_path(canonical_name: str) -> str:
    return canonical_name.replace(".", "/") + ".py"


@dataclasses.dataclass(frozen=True, slots=True)
class DeclaredModuleBinding:
    """One canonical module name bound to exact repository source bytes."""

    canonical_name: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.canonical_name) is not str or not self.canonical_name:
            raise TypeError("canonical_name must be a non-empty exact string")
        parts = self.canonical_name.split(".")
        if (
            len(parts) != 3
            or not all(part.isidentifier() for part in parts)
            or not any(
                self.canonical_name.startswith(prefix)
                for prefix in _DECLARED_LEAF_PREFIXES
            )
        ):
            raise ValueError(
                "canonical_name must be one immediate core/evaluation leaf module"
            )
        if type(self.relative_path) is not str or not self.relative_path:
            raise TypeError("relative_path must be a non-empty exact string")
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or path.as_posix() != self.relative_path
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in self.relative_path
        ):
            raise ValueError("relative_path must be a canonical repository-relative POSIX path")
        expected_path = _canonical_relative_path(self.canonical_name)
        if self.relative_path != expected_path:
            raise ValueError(
                "relative_path does not match the canonical module name: "
                f"{expected_path!r} required"
            )
        if type(self.sha256) is not str:
            raise TypeError("sha256 must be an exact string")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")


@dataclasses.dataclass(frozen=True, slots=True)
class _BoundSource:
    binding: DeclaredModuleBinding
    resolved_path: Path
    source_bytes: bytes


@dataclasses.dataclass(frozen=True, slots=True)
class _ExecutedSource:
    bound: _BoundSource
    module: ModuleType


def _validated_root(root: Path) -> Path:
    if not isinstance(root, Path):
        raise TypeError("root must be a pathlib.Path")
    if not root.is_absolute():
        raise ValueError("root must be absolute")
    if root.is_symlink():
        raise ValueError("root must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"root cannot be resolved: {root}") from error
    if resolved != root:
        raise ValueError("root must already be an exact resolved path")
    if not resolved.is_dir():
        raise ValueError("root must be a directory")
    return resolved


def _reject_relative_symlinks(root: Path, relative_parts: Sequence[str]) -> Path:
    candidate = root
    for part in relative_parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"declared path contains a symlink: {candidate}")
    return candidate


def _resolved_regular_file(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    candidate = _reject_relative_symlinks(root, relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"declared source does not exist: {relative_path}") from error
    if not resolved.is_relative_to(root):
        raise ValueError(f"declared source resolves outside root: {relative_path}")
    if resolved != candidate:
        raise ValueError(f"declared source path is not exact: {relative_path}")
    if not resolved.is_file():
        raise ValueError(f"declared source is not a regular file: {relative_path}")
    return resolved


def _resolved_directory(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    candidate = _reject_relative_symlinks(root, relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"namespace directory does not exist: {relative_path}") from error
    if not resolved.is_relative_to(root) or resolved != candidate or not resolved.is_dir():
        raise ValueError(f"namespace directory is not an exact in-root directory: {relative_path}")
    return resolved


def _bind_source(root: Path, binding: DeclaredModuleBinding) -> _BoundSource:
    resolved_path = _resolved_regular_file(root, binding.relative_path)
    source_bytes = resolved_path.read_bytes()
    observed = hashlib.sha256(source_bytes).hexdigest()
    if observed != binding.sha256:
        raise ValueError(
            f"declared source digest mismatch for {binding.canonical_name}: "
            f"{binding.sha256} != {observed}"
        )
    return _BoundSource(binding, resolved_path, source_bytes)


def install_namespace_stubs(root: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    """Install three empty namespace packages without executing ``__init__.py``.

    Every existing ``alberta_framework`` module is rejected because a
    ``sys.modules`` hit would bypass the declared-source finder.
    """

    resolved_root = _validated_root(root)
    preloaded = sorted(name for name in sys.modules if _is_first_party_name(name))
    if preloaded:
        raise RuntimeError("first-party modules are already loaded: " + ", ".join(preloaded))
    directories = {
        name: _resolved_directory(resolved_root, relative)
        for name, relative in _NAMESPACE_RELATIVE_PATHS.items()
    }
    modules: dict[str, ModuleType] = {}
    for name in _NAMESPACE_NAMES:
        module = ModuleType(name)
        spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = []
        module.__package__ = name
        module.__loader__ = None
        module.__spec__ = spec
        module.__path__ = []
        setattr(module, "__declared_namespace_marker__", _NAMESPACE_MARKER)
        setattr(module, "__declared_namespace_directory__", str(directories[name]))
        modules[name] = module
    try:
        for name, module in modules.items():
            sys.modules[name] = module
        setattr(modules["alberta_framework"], "core", modules["alberta_framework.core"])
        setattr(
            modules["alberta_framework"],
            "evaluation",
            modules["alberta_framework.evaluation"],
        )
    except BaseException:
        for name, module in modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise
    return cast(
        tuple[ModuleType, ModuleType, ModuleType],
        tuple(modules[name] for name in _NAMESPACE_NAMES),
    )


class _DeclaredSourceLoader(importlib.abc.Loader):
    """Execute the already-hashed bytes held by one declared finder."""

    def __init__(self, finder: DeclaredSourceFinder, bound: _BoundSource) -> None:
        self._finder = finder
        self._bound = bound

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        del spec
        return None

    def exec_module(self, module: ModuleType) -> None:
        binding = self._bound.binding
        if module.__name__ != binding.canonical_name:
            raise ImportError("declared loader received a module with the wrong canonical name")
        spec = module.__spec__
        if spec is None or spec.origin != str(self._bound.resolved_path):
            raise ImportError("declared module origin differs from its exact binding")
        self._finder._record_execution(self._bound, module)
        code = compile(
            self._bound.source_bytes,
            str(self._bound.resolved_path),
            "exec",
            dont_inherit=True,
        )
        exec(code, module.__dict__)


class DeclaredSourceFinder(importlib.abc.MetaPathFinder):
    """First-position finder that rejects every undeclared Alberta import."""

    def __init__(self, root: Path, bindings: Sequence[DeclaredModuleBinding]) -> None:
        resolved_root = _validated_root(root)
        if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
            raise TypeError("bindings must be a sequence of DeclaredModuleBinding values")
        declared = tuple(bindings)
        if not declared:
            raise ValueError("at least one declared module binding is required")
        if any(type(binding) is not DeclaredModuleBinding for binding in declared):
            raise TypeError("every binding must be an exact DeclaredModuleBinding")
        bound_by_name: dict[str, _BoundSource] = {}
        names: set[str] = set()
        paths: set[Path] = set()
        for binding in declared:
            if binding.canonical_name in names:
                raise ValueError(f"duplicate declared module name: {binding.canonical_name}")
            bound = _bind_source(resolved_root, binding)
            if bound.resolved_path in paths:
                raise ValueError(f"duplicate declared source path: {binding.relative_path}")
            names.add(binding.canonical_name)
            paths.add(bound.resolved_path)
            bound_by_name[binding.canonical_name] = bound
        self._root = resolved_root
        self._bound_by_name = MappingProxyType(bound_by_name)
        self._ledger_lock = threading.Lock()
        self._executed: dict[str, _ExecutedSource] = {}
        self._installed = False

    @property
    def expected_closure(self) -> tuple[tuple[str, str, str], ...]:
        """Return exact sorted ``(module, resolved source path, sha256)`` records."""

        return tuple(
            sorted(
                (
                    bound.binding.canonical_name,
                    str(bound.resolved_path),
                    bound.binding.sha256,
                )
                for bound in self._bound_by_name.values()
            )
        )

    @property
    def observed_execution_closure(self) -> tuple[tuple[str, str, str], ...]:
        """Return the monotonic sorted closure of loader executions so far."""

        with self._ledger_lock:
            return tuple(
                sorted(
                    (
                        executed.bound.binding.canonical_name,
                        str(executed.bound.resolved_path),
                        executed.bound.binding.sha256,
                    )
                    for executed in self._executed.values()
                )
            )

    def __enter__(self) -> DeclaredSourceFinder:
        if self._installed or self in sys.meta_path:
            raise RuntimeError("declared-source finder is already installed")
        sys.meta_path.insert(0, self)
        self._installed = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> Literal[False]:
        del exc_type, exc_value, traceback
        self.uninstall()
        return False

    def uninstall(self) -> None:
        """Remove only this finder instance from ``sys.meta_path``."""

        matches = [index for index, finder in enumerate(sys.meta_path) if finder is self]
        if len(matches) > 1:
            raise RuntimeError("declared-source finder appears more than once")
        if matches:
            del sys.meta_path[matches[0]]
        self._installed = False

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path
        if not _is_first_party_name(fullname):
            return None
        if not self._installed or not sys.meta_path or sys.meta_path[0] is not self:
            raise ImportError("declared-source finder must remain first in sys.meta_path")
        if target is not None:
            raise ImportError(f"reload is forbidden for declared module {fullname}")
        if fullname in _NAMESPACE_NAMES:
            raise ImportError(f"namespace stub was not preinstalled: {fullname}")
        bound = self._bound_by_name.get(fullname)
        if bound is None:
            raise ModuleNotFoundError(f"undeclared first-party module import blocked: {fullname}")
        with self._ledger_lock:
            if fullname in self._executed:
                raise ImportError(f"declared module cannot be executed twice: {fullname}")
        loader = _DeclaredSourceLoader(self, bound)
        spec = importlib.util.spec_from_file_location(
            fullname,
            bound.resolved_path,
            loader=loader,
        )
        if spec is None:
            raise ImportError(f"could not construct declared module spec: {fullname}")
        return spec

    def load(self, canonical_name: str) -> ModuleType:
        """Load one declared leaf while this finder is first and active."""

        if type(canonical_name) is not str:
            raise TypeError("canonical_name must be an exact string")
        if canonical_name not in self._bound_by_name:
            raise ValueError(f"module is not declared: {canonical_name}")
        if not self._installed or not sys.meta_path or sys.meta_path[0] is not self:
            raise RuntimeError("declared-source finder is not active in first position")
        return importlib.import_module(canonical_name)

    def _record_execution(self, bound: _BoundSource, module: ModuleType) -> None:
        name = bound.binding.canonical_name
        with self._ledger_lock:
            if name in self._executed:
                raise ImportError(f"declared module cannot be executed twice: {name}")
            self._executed[name] = _ExecutedSource(bound, module)


def _validate_namespace_stubs(root: Path) -> None:
    expected_modules: dict[str, ModuleType] = {}
    for name in _NAMESPACE_NAMES:
        module = sys.modules.get(name)
        if type(module) is not ModuleType:
            raise RuntimeError(f"declared namespace stub is missing or replaced: {name}")
        if getattr(module, "__declared_namespace_marker__", None) is not _NAMESPACE_MARKER:
            raise RuntimeError(f"declared namespace marker is invalid: {name}")
        spec = module.__spec__
        if (
            spec is None
            or spec.loader is not None
            or spec.submodule_search_locations != []
            or getattr(module, "__path__", None) != []
            or hasattr(module, "__file__")
        ):
            raise RuntimeError(f"declared namespace stub metadata is invalid: {name}")
        expected_directory = _resolved_directory(root, _NAMESPACE_RELATIVE_PATHS[name])
        if getattr(module, "__declared_namespace_directory__", None) != str(
            expected_directory
        ):
            raise RuntimeError(f"declared namespace directory is invalid: {name}")
        expected_modules[name] = module
    package = expected_modules["alberta_framework"]
    if (
        getattr(package, "core", None) is not expected_modules["alberta_framework.core"]
        or getattr(package, "evaluation", None)
        is not expected_modules["alberta_framework.evaluation"]
    ):
        raise RuntimeError("declared namespace parent/child identities are invalid")


def validate_loaded_closure(
    finder: DeclaredSourceFinder,
) -> tuple[tuple[str, str, str], ...]:
    """Require exact executed, loaded, resolved-path, and current-byte closure.

    The returned tuples are sorted ``(canonical module, resolved absolute
    source path, lowercase SHA-256)`` records.  Failed imports remain visible
    in the finder's monotonic ledger and therefore cannot pass this postflight.
    """

    if type(finder) is not DeclaredSourceFinder:
        raise TypeError("finder must be an exact DeclaredSourceFinder")
    _validate_namespace_stubs(finder._root)
    expected = finder.expected_closure
    observed = finder.observed_execution_closure
    if observed != expected:
        raise RuntimeError(
            f"declared execution closure mismatch: expected {expected!r}, observed {observed!r}"
        )
    allowed_names = {*_NAMESPACE_NAMES, *(record[0] for record in expected)}
    unexpected = sorted(
        name
        for name in sys.modules
        if _is_first_party_name(name) and name not in allowed_names
    )
    if unexpected:
        raise RuntimeError("unexpected first-party loaded modules: " + ", ".join(unexpected))
    with finder._ledger_lock:
        executed_snapshot = dict(finder._executed)
    for name, resolved_path, digest in expected:
        executed = executed_snapshot[name]
        if sys.modules.get(name) is not executed.module:
            raise RuntimeError(f"declared module is missing or replaced in sys.modules: {name}")
        module = executed.module
        spec = module.__spec__
        module_file = getattr(module, "__file__", None)
        if (
            spec is None
            or spec.origin != str(executed.bound.resolved_path)
            or module_file != str(executed.bound.resolved_path)
        ):
            raise RuntimeError(f"declared module origin changed after load: {name}")
        rebound = _bind_source(finder._root, executed.bound.binding)
        if (
            rebound.resolved_path != executed.bound.resolved_path
            or str(rebound.resolved_path) != resolved_path
            or rebound.binding.sha256 != digest
        ):
            raise RuntimeError(f"declared module binding changed after load: {name}")
    return expected


_AttemptState = Literal["never", "running", "success", "failed"]


class ProcessAttempt[T]:
    """Run one capability-bound builder once, consuming before builder entry."""

    def __init__(self, builder: Callable[[object], T]) -> None:
        if not callable(builder):
            raise TypeError("process-attempt builder must be callable")
        self._condition = threading.Condition()
        self._builder: Callable[[object], T] | None = builder
        self._state: _AttemptState = "never"
        self._owner_thread: int | None = None
        self._active_capability: object | None = None
        self._value: T | None = None
        self._failure: BaseException | None = None

    def get(self) -> T:
        """Return the sole success, waiting only for another caller's active attempt."""

        thread_id = threading.get_ident()
        with self._condition:
            while self._state == "running":
                if self._owner_thread == thread_id:
                    raise RuntimeError("recursive process-attempt entry is forbidden")
                self._condition.wait()
            if self._state == "success":
                return cast(T, self._value)
            if self._state == "failed":
                raise RuntimeError("the process attempt is sealed after failure") from self._failure
            builder = self._builder
            if builder is None:
                raise RuntimeError("the process attempt has no available builder")
            self._builder = None
            self._state = "running"
            self._owner_thread = thread_id
            capability = object()
            self._active_capability = capability
        try:
            value = builder(capability)
            if value is None:
                raise TypeError("process-attempt builder must return a non-None value")
        except BaseException as error:
            with self._condition:
                self._failure = error
                self._state = "failed"
                self._owner_thread = None
                self._active_capability = None
                self._condition.notify_all()
            raise
        with self._condition:
            self._value = value
            self._state = "success"
            self._owner_thread = None
            self._active_capability = None
            self._condition.notify_all()
        return value

    def authorizes(self, capability: object) -> bool:
        """Return whether ``capability`` is the active attempt's identity token."""

        with self._condition:
            return self._state == "running" and self._active_capability is capability

    def completed_value(self) -> T | None:
        """Return completed success without starting or waiting for an attempt."""

        if not self._condition.acquire(blocking=False):
            return None
        try:
            if self._state != "success":
                return None
            return cast(T, self._value)
        finally:
            self._condition.release()


__all__ = [
    "CROSS_PROCESS_REPLAY_PREVENTED",
    "DEVELOPMENT_ONLY",
    "EVIDENCE_AUTHORIZED",
    "OUTPUT_WRITES_ALLOWED",
    "PANEL_EXECUTION_AUTHORIZED",
    "PROTOCOL_DECLARATION_AUTHORIZED",
    "ROOT_ISSUANCE_AUTHORIZED",
    "RUNNER_AVAILABLE",
    "SCIENTIFIC_PROMOTION_ALLOWED",
    "SOURCE_GENERATION_AUTHORIZED",
    "DeclaredModuleBinding",
    "DeclaredSourceFinder",
    "ProcessAttempt",
    "install_namespace_stubs",
    "validate_loaded_closure",
]
