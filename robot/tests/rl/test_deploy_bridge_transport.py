"""Authenticated transport contracts for unified-bridge deployment clients."""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, call

import pytest

from eliza_robot.rl.deploy.preflight_check import (
    BRIDGE_AUTH_TOKEN_ENV,
    BridgeTransportSecurityError,
    bridge_connect_options,
    check_bridge,
    policy_tick_joint_payload,
    print_results,
    redacted_bridge_url,
)


class _FakeWebSocket:
    def __init__(self) -> None:
        self._frames = iter(
            (
                '{"type":"event","event":"session.hello","backend":"ros_real","data":{}}',
                '{"type":"event","event":"telemetry.basic","backend":"ros_real",'
                '"data":{"battery_mv":8000,"imu_roll":0.0,'
                '"joint_positions":{"r_hip_pitch":0.0}}}',
            )
        )

    async def recv(self) -> str:
        return next(self._frames)

    async def close(self) -> None:
        return None


class _FakeConnectContext:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self._websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self._websocket

    async def __aexit__(self, *_args: object) -> None:
        return None


def test_physical_bridge_requires_loopback_and_environment_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BRIDGE_AUTH_TOKEN_ENV, raising=False)
    with pytest.raises(BridgeTransportSecurityError, match=BRIDGE_AUTH_TOKEN_ENV):
        bridge_connect_options("ws://127.0.0.1:9100", physical=True)

    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, "a" * 32)
    with pytest.raises(BridgeTransportSecurityError, match="must target loopback"):
        bridge_connect_options("wss://robot.example.invalid:9100", physical=True)


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://localhost:9100",
        "ws://127.0.0.1:9100",
        "wss://[::1]:9100",
    ],
)
def test_physical_bridge_uses_authorization_header_not_url(
    endpoint: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "physical-secret-" + "x" * 32
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, token)
    options = bridge_connect_options(endpoint, physical=True)
    assert options == {"additional_headers": {"Authorization": f"Bearer {token}"}}
    assert token not in endpoint


@pytest.mark.parametrize(
    "endpoint",
    [
        "ws://operator:LEAKME_7f91@127.0.0.1:9100",
        "ws://127.0.0.1:9100?token=LEAKME_7f91",
        "ws://127.0.0.1:9100#LEAKME_7f91",
    ],
)
def test_bridge_rejects_url_secret_channels(endpoint: str) -> None:
    with pytest.raises(BridgeTransportSecurityError) as raised:
        bridge_connect_options(endpoint, physical=False)
    assert "LEAKME_7f91" not in str(raised.value)


def test_simulation_preserves_unauthenticated_remote_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, "hardware-secret-" + "z" * 32)
    assert bridge_connect_options("ws://sim.example.invalid:9100/path", physical=False) == {}


def test_physical_bridge_rejects_nonroot_url_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, "a" * 32)
    with pytest.raises(BridgeTransportSecurityError, match="root websocket path"):
        bridge_connect_options("ws://localhost:9100/not-a-credential-channel", physical=True)


def test_invalid_bearer_is_rejected_without_echoing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "x" * 32 + "\r\ninjected: header"
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, secret)
    with pytest.raises(BridgeTransportSecurityError) as raised:
        bridge_connect_options("ws://localhost:9100", physical=True)
    assert secret not in str(raised.value)


def test_overlong_bearer_is_rejected_without_echoing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "x" * 4_097
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, secret)
    with pytest.raises(BridgeTransportSecurityError, match="at most 4096") as raised:
        bridge_connect_options("ws://localhost:9100", physical=True)
    assert secret not in str(raised.value)


def test_redacted_endpoint_never_contains_url_credentials_or_components() -> None:
    rendered = redacted_bridge_url(
        "ws://operator:secret@127.0.0.1:9100/secret-path?token=secret#secret"
    )
    assert rendered == "ws://127.0.0.1:9100"
    assert "secret" not in rendered


def test_policy_tick_joint_payload_uses_nested_action_and_seconds() -> None:
    assert policy_tick_joint_payload(
        {"r_hip_pitch": 0.125},
        duration_sec=0.05,
    ) == {
        "action": {
            "joint_positions": {"r_hip_pitch": 0.125},
            "duration": 0.05,
        }
    }


@pytest.mark.parametrize(
    ("positions", "duration"),
    [
        ({}, 0.05),
        ({"r_hip_pitch": float("nan")}, 0.05),
        ({"r_hip_pitch": 0.0}, 0.0),
        ({"r_hip_pitch": 0.0}, 50.0),
    ],
)
def test_policy_tick_joint_payload_rejects_unsafe_values(
    positions: dict[str, float],
    duration: float,
) -> None:
    with pytest.raises(ValueError):
        policy_tick_joint_payload(positions, duration_sec=duration)


@pytest.mark.asyncio
async def test_preflight_sends_bearer_header_and_consumes_unified_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "p" * 32
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, token)
    observed: dict[str, object] = {}

    def connect(uri: str, **kwargs: object) -> _FakeConnectContext:
        observed["uri"] = uri
        observed["kwargs"] = kwargs
        return _FakeConnectContext(_FakeWebSocket())

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    results = await check_bridge(
        "ws://127.0.0.1:9100",
        timeout=0.5,
        physical=True,
    )
    assert results["all_ok"] is True
    assert results["backend_type"] == "ros_real"
    assert observed == {
        "uri": "ws://127.0.0.1:9100",
        "kwargs": {
            "open_timeout": 0.5,
            "additional_headers": {"Authorization": f"Bearer {token}"},
        },
    }
    assert token not in repr(results)


def test_preflight_success_is_explicitly_non_authorizing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    print_results(
        {
            "bridge_reachable": True,
            "backend_type": "ros_real",
            "servo_count": 12,
            "imu_data": True,
            "battery_mv": 8_000,
            "battery_ok": True,
            "all_ok": True,
            "errors": [],
        }
    )
    output = capsys.readouterr().out
    assert "does not authorize physical motion" in output
    assert "safe to deploy" not in output


@pytest.mark.asyncio
async def test_preflight_fails_before_connect_when_physical_token_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BRIDGE_AUTH_TOKEN_ENV, raising=False)

    def connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe preflight attempted a connection")

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    results = await check_bridge("ws://localhost:9100", physical=True)
    assert results["bridge_reachable"] is False
    assert results["all_ok"] is False
    assert results["errors"] == [
        f"physical bridge requires {BRIDGE_AUTH_TOKEN_ENV}"
    ]


@pytest.mark.asyncio
async def test_walking_client_forwards_bearer_header_outside_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.rl.deploy.deploy_walking import DeployWalking

    token = "w" * 32
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, token)
    observed: dict[str, object] = {}

    def connect(uri: str, **kwargs: object) -> _FakeConnectContext:
        observed["uri"] = uri
        observed["kwargs"] = kwargs
        return _FakeConnectContext(_FakeWebSocket())

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    deployer = DeployWalking.__new__(DeployWalking)
    deployer.hz = 20.0
    deployer._send_command = AsyncMock()  # type: ignore[method-assign]
    deployer._recv_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"ok": False, "message": "expected test stop"}
    )
    await deployer.run_with_bridge("ws://localhost:9100", physical=True)

    assert observed == {
        "uri": "ws://localhost:9100",
        "kwargs": {
            "additional_headers": {"Authorization": f"Bearer {token}"},
        },
    }


@pytest.mark.asyncio
async def test_composite_client_forwards_bearer_header_outside_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.rl.deploy.deploy_composite import DeployComposite

    token = "c" * 32
    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, token)
    observed: dict[str, object] = {}

    async def connect(uri: str, **kwargs: object) -> _FakeWebSocket:
        observed["uri"] = uri
        observed["kwargs"] = kwargs
        return _FakeWebSocket()

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    deployer = DeployComposite.__new__(DeployComposite)
    deployer._stopped = False
    deployer.hz = 20.0
    deployer.task = "wave"
    deployer._send_command = AsyncMock()  # type: ignore[method-assign]
    deployer._recv_response = AsyncMock(  # type: ignore[method-assign]
        return_value={"ok": False, "message": "expected test stop"}
    )
    await deployer.run_with_bridge("ws://127.0.0.1:9100", physical=True)

    assert observed == {
        "uri": "ws://127.0.0.1:9100",
        "kwargs": {
            "additional_headers": {"Authorization": f"Bearer {token}"},
        },
    }


@pytest.mark.asyncio
async def test_walking_error_shutdown_sends_only_stop_and_logs_exact_rejection(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from eliza_robot.rl.deploy.deploy_walking import DeployWalking

    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, "w" * 32)
    websocket = _FakeWebSocket()

    def connect(_uri: str, **_kwargs: object) -> _FakeConnectContext:
        return _FakeConnectContext(websocket)

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    deployer = DeployWalking.__new__(DeployWalking)
    deployer.hz = 20.0
    deployer.duration = 5.0
    deployer.ramp_seconds = 1.0
    deployer.dry_run = False
    deployer._send_command = AsyncMock()  # type: ignore[method-assign]
    deployer._recv_response = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {"ok": True, "message": "started"},
            {"ok": False, "message": "fence denied", "request_id": "stop-1"},
        ]
    )
    deployer._control_loop = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("control failure")
    )

    with pytest.raises(RuntimeError, match="control failure"):
        await deployer.run_with_bridge("ws://localhost:9100", physical=True)

    assert deployer._send_command.await_args_list == [
        call(websocket, "policy.start", {"task": "deploy_walking", "hz": 20.0}),
        call(websocket, "policy.stop", {}),
    ]
    assert (
        'WARNING: policy.stop was not acknowledged: '
        '{"message":"fence denied","ok":false,"request_id":"stop-1"}'
        in capsys.readouterr().out
    )


@pytest.mark.asyncio
async def test_composite_fall_shutdown_stops_without_positive_motion_or_recovery(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from eliza_robot.rl.deploy.deploy_composite import DeployComposite

    monkeypatch.setenv(BRIDGE_AUTH_TOKEN_ENV, "c" * 32)
    websocket = _FakeWebSocket()

    async def connect(_uri: str, **_kwargs: object) -> _FakeWebSocket:
        return websocket

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    deployer = DeployComposite.__new__(DeployComposite)
    deployer._stopped = False
    deployer.hz = 20.0
    deployer.task = "wave"
    deployer.duration = 5.0
    deployer.ramp_seconds = 1.0
    deployer.dry_run = False
    deployer._send_command = AsyncMock()  # type: ignore[method-assign]
    deployer._recv_response = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            {"ok": True, "message": "started"},
            {"ok": True, "message": "stopped", "request_id": "stop-2"},
        ]
    )

    async def detect_fall(_ws: object) -> None:
        deployer._fell = True

    deployer._control_loop = AsyncMock(  # type: ignore[method-assign]
        side_effect=detect_fall
    )

    await deployer.run_with_bridge("ws://127.0.0.1:9100", physical=True)

    assert deployer._send_command.await_args_list == [
        call(
            websocket,
            "policy.start",
            {"task": "deploy_composite_wave", "hz": 20.0},
        ),
        call(websocket, "policy.stop", {}),
    ]
    output = capsys.readouterr().out
    assert (
        'policy.stop acknowledged: '
        '{"message":"stopped","ok":true,"request_id":"stop-2"}'
        in output
    )
    assert "Fall recovery disabled" in output


@pytest.mark.asyncio
async def test_composite_auto_recovery_is_quarantined_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.rl.deploy.deploy_composite import DeployComposite

    def connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fall-recovery quarantine attempted a connection")

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    deployer = DeployComposite.__new__(DeployComposite)
    with pytest.raises(RuntimeError, match="separate physical recovery supervisor"):
        await deployer.run_with_bridge(
            "ws://127.0.0.1:9100",
            auto_recover=True,
            physical=True,
        )


@pytest.mark.asyncio
async def test_legacy_raw_rosbridge_validation_stays_quarantined_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.bridge.physical_execution import UnsupervisedPhysicalControlError
    from eliza_robot.rl.deploy.validate_real import run_servo_step_response

    def connect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("quarantined raw ROSBridge path attempted a connection")

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    with pytest.raises(UnsupervisedPhysicalControlError, match="quarantined"):
        await run_servo_step_response()


@pytest.mark.asyncio
async def test_servo_ping_uses_passive_unified_telemetry_not_legacy_status_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from eliza_robot.rl.deploy.validate_real import run_validation

    observed: dict[str, object] = {}

    def connect(uri: str, **kwargs: object) -> _FakeConnectContext:
        observed["uri"] = uri
        observed["kwargs"] = kwargs
        return _FakeConnectContext(_FakeWebSocket())

    websocket_module = types.ModuleType("websockets")
    websocket_module.connect = connect  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "websockets", websocket_module)

    result = await run_validation(
        "ws://sim.example.invalid:9100",
        "servo_ping",
        duration=0.01,
        physical=False,
    )
    assert result["success"] is True
    assert len(result["log"]) == 1
    assert observed == {
        "uri": "ws://sim.example.invalid:9100",
        "kwargs": {},
    }
