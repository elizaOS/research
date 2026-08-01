"""Fail-closed contracts for stable physical actuator identities."""

from __future__ import annotations

from typing import Any

import pytest

from eliza_robot.bridge.backends.ainex_remote import AinexRemoteBackend
from eliza_robot.bridge.backends.asimov_remote import AsimovRemoteBackend
from eliza_robot.bridge.backends.base import (
    PHYSICAL_RESOURCE_ID_MAX_LENGTH,
    BridgeBackend,
    _physical_motion_authority_error,
    _supervised_motion_dispatch_authority,
    canonical_physical_resource_id,
)
from eliza_robot.bridge.backends.calibrated import CalibratedBackend
from eliza_robot.bridge.backends.dual_target import DualTargetBackend
from eliza_robot.bridge.backends.noise_injector import NoiseInjectorBackend
from eliza_robot.bridge.backends.ros_backend import RosBridgeBackend
from eliza_robot.bridge.backends.state_mirror import StateMirrorBackend
from eliza_robot.bridge.protocol import (
    CommandEnvelope,
    EventEnvelope,
    ResponseEnvelope,
    utc_now_iso,
)
from eliza_robot.bridge.types import JsonDict


def _command() -> CommandEnvelope:
    return CommandEnvelope(
        request_id="resource-identity-test",
        timestamp=utc_now_iso(),
        command="head.set",
        payload={"pan": 0.1, "tilt": 0.0, "duration": 0.2},
    )


class _Leaf(BridgeBackend):
    def __init__(self, resources: tuple[str, ...]) -> None:
        self._resources = resources

    @property
    def backend_name(self) -> str:
        return "resource-test-leaf"

    async def connect(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def handle_command(self, cmd: CommandEnvelope) -> ResponseEnvelope:
        if self._resources:
            authority_error = _physical_motion_authority_error(
                self.backend_name,
                self._resources[0],
                cmd,
            )
            if authority_error is not None:
                return authority_error
        return ResponseEnvelope(
            request_id=cmd.request_id,
            timestamp=utc_now_iso(),
            ok=True,
            backend=self.backend_name,
            message="ok",
            data={},
        )

    async def poll_events(self) -> list[EventEnvelope]:
        return []

    def capabilities(self) -> JsonDict:
        return {}

    def physical_motion_resources(self) -> tuple[str, ...]:
        return self._resources


class _ErasingWrapper(BridgeBackend):
    """An unsafe third-party wrapper that forgets to expose its leaf."""

    def __init__(self, inner: BridgeBackend) -> None:
        self._inner = inner

    @property
    def backend_name(self) -> str:
        return "erasing-wrapper"

    async def connect(self) -> None:
        await self._inner.connect()

    async def shutdown(self) -> None:
        await self._inner.shutdown()

    async def handle_command(self, cmd: CommandEnvelope) -> ResponseEnvelope:
        return await self._inner.handle_command(cmd)

    async def poll_events(self) -> list[EventEnvelope]:
        return await self._inner.poll_events()

    def capabilities(self) -> JsonDict:
        return self._inner.capabilities()


class _ForgingWrapper(_ErasingWrapper):
    def physical_motion_resources(self) -> tuple[str, ...]:
        return ("physical:forged-unit",)


def test_canonical_physical_resource_id_is_strict_and_non_normalizing() -> None:
    raw = "lab-a/robot-07:actuators"
    assert canonical_physical_resource_id(raw) == f"physical:{raw}"
    assert canonical_physical_resource_id("x" * PHYSICAL_RESOURCE_ID_MAX_LENGTH) == (
        "physical:" + "x" * PHYSICAL_RESOURCE_ID_MAX_LENGTH
    )


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        7,
        "",
        " leading",
        "trailing ",
        "internal space",
        "tab\tcharacter",
        "line\nbreak",
        "delete\x7f",
        "non-ascii-é",
        "x" * (PHYSICAL_RESOURCE_ID_MAX_LENGTH + 1),
    ],
)
def test_canonical_physical_resource_id_rejects_invalid_values(invalid: Any) -> None:
    with pytest.raises(ValueError, match="physical_resource_id"):
        canonical_physical_resource_id(invalid)


def test_physical_leaf_constructors_require_explicit_identity() -> None:
    with pytest.raises(TypeError, match="physical_resource_id"):
        AinexRemoteBackend()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="physical_resource_id"):
        RosBridgeBackend("ros_real")
    with pytest.raises(ValueError, match="physical_resource_id"):
        AsimovRemoteBackend(mock=False)

    assert RosBridgeBackend("ros_sim").physical_motion_resources() == ()
    assert AsimovRemoteBackend(mock=True).physical_motion_resources() == ()


def test_transport_aliases_share_only_the_explicit_physical_identity() -> None:
    expected = ("physical:lab-a/robot-07",)
    assert AinexRemoteBackend(
        host="robot.local",
        port=9090,
        physical_resource_id="lab-a/robot-07",
    ).physical_motion_resources() == expected
    assert AinexRemoteBackend(
        host="10.1.2.3",
        port=9191,
        physical_resource_id="lab-a/robot-07",
    ).physical_motion_resources() == expected
    assert RosBridgeBackend(
        "ros_real",
        physical_resource_id="lab-a/robot-07",
    ).physical_motion_resources() == expected
    assert AsimovRemoteBackend(
        mock=False,
        livekit_url="wss://first.invalid",
        physical_resource_id="lab-a/robot-07",
    ).physical_motion_resources() == expected
    assert AsimovRemoteBackend(
        mock=False,
        livekit_url="wss://second.invalid",
        physical_resource_id="lab-a/robot-07",
    ).physical_motion_resources() == expected


def test_builtin_wrappers_preserve_resources_and_dual_target_keeps_duplicates() -> None:
    first_resources = ("physical:first",)
    second_resources = ("physical:first", "physical:second")
    first = _Leaf(first_resources)
    second = _Leaf(second_resources)

    calibrated = CalibratedBackend(first, {})
    noisy = NoiseInjectorBackend(first, n_joints=1)
    mirror = StateMirrorBackend(first, real=object(), sim_env=object())

    assert calibrated.physical_motion_resources() is first_resources
    assert noisy.physical_motion_resources() is first_resources
    assert mirror.physical_motion_resources() is first_resources
    assert DualTargetBackend(first, second).physical_motion_resources() == (
        "physical:first",
        "physical:first",
        "physical:second",
    )


@pytest.mark.asyncio
async def test_erased_or_forged_wrapper_identity_cannot_authorize_the_leaf() -> None:
    leaf = _Leaf(("physical:actual-unit",))
    command = _command()

    erased = _ErasingWrapper(leaf)
    with _supervised_motion_dispatch_authority(
        command,
        erased.physical_motion_resources(),
    ):
        erased_response = await erased.handle_command(command)
    assert not erased_response.ok

    forged = _ForgingWrapper(leaf)
    with _supervised_motion_dispatch_authority(
        command,
        forged.physical_motion_resources(),
    ):
        forged_response = await forged.handle_command(command)
    assert not forged_response.ok


@pytest.mark.asyncio
async def test_physical_leaves_authorize_against_stored_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _command()
    expected = ("physical:actual-unit",)

    ainex = AinexRemoteBackend(physical_resource_id="actual-unit")
    ainex._ros = type("_ConnectedRos", (), {"is_connected": True})()  # noqa: SLF001
    monkeypatch.setattr(ainex, "_dispatch", lambda _cmd: None)
    monkeypatch.setattr(
        ainex,
        "physical_motion_resources",
        lambda: ("physical:forged-unit",),
    )
    with _supervised_motion_dispatch_authority(command, expected):
        assert (await ainex.handle_command(command)).ok

    ros = RosBridgeBackend("ros_real", physical_resource_id="actual-unit")
    ros._ready = True  # noqa: SLF001
    monkeypatch.setattr(ros, "_dispatch_blocking", lambda _cmd: None)
    monkeypatch.setattr(
        ros,
        "physical_motion_resources",
        lambda: ("physical:forged-unit",),
    )
    with _supervised_motion_dispatch_authority(command, expected):
        assert (await ros.handle_command(command)).ok

    asimov = AsimovRemoteBackend(
        mock=False,
        transport=object(),
        physical_resource_id="actual-unit",
    )
    monkeypatch.setattr(
        asimov,
        "physical_motion_resources",
        lambda: ("physical:forged-unit",),
    )
    with _supervised_motion_dispatch_authority(command, expected):
        assert (await asimov.handle_command(command)).ok
