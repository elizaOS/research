"""Universal to_config/optimizer_from_config round-trip tests.

Every optimizer class shipped in ``core/optimizers.py``,
``core/baseline_optimizers.py``, and ``core/swift_td.py`` (discovered by
introspection, so newly added optimizers are picked up automatically)
must round-trip through ``optimizer_from_config``.
"""

import inspect

import pytest

from alberta_framework.core import baseline_optimizers, optimizers, swift_td
from alberta_framework.core.optimizers import Bounder, optimizer_from_config

_OPTIMIZER_MODULES = (optimizers, baseline_optimizers, swift_td)


def _optimizer_classes() -> list[type]:
    """Collect every concrete optimizer class defined in the swept modules.

    An "optimizer class" is any non-abstract class defining both
    ``to_config`` and ``init`` (this excludes ``Bounder`` subclasses, state
    dataclasses, and the abstract bases themselves).
    """
    classes: set[type] = set()
    for module in _OPTIMIZER_MODULES:
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if inspect.isabstract(cls):
                continue
            if issubclass(cls, Bounder):
                continue
            if callable(getattr(cls, "to_config", None)) and callable(
                getattr(cls, "init", None)
            ):
                classes.add(cls)
    return sorted(classes, key=lambda c: c.__name__)


_OPTIMIZER_CLASSES = _optimizer_classes()


def test_discovery_finds_expected_optimizers():
    """The introspection sweep must find at least the known optimizer set."""
    names = {cls.__name__ for cls in _OPTIMIZER_CLASSES}
    expected = {
        "LMS",
        "IDBD",
        "Autostep",
        "AutostepGTDLambda",
        "ObGD",
        "AdaGain",
        "Adam",
        "RMSprop",
        "NADALINE",
        "SwiftTD",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


@pytest.mark.parametrize(
    "cls", _OPTIMIZER_CLASSES, ids=[c.__name__ for c in _OPTIMIZER_CLASSES]
)
def test_round_trip_with_default_args(cls: type):
    """Every optimizer round-trips through optimizer_from_config."""
    opt = cls()
    config = opt.to_config()
    assert config["type"] == cls.__name__

    restored = optimizer_from_config(config)
    assert type(restored) is cls
    assert restored.to_config() == config


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown optimizer type"):
        optimizer_from_config({"type": "NotAnOptimizer"})
