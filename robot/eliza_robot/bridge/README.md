# AiNex Unified Bridge

This package supports two websocket API surfaces:

- `eliza_robot.bridge.server`: strict command-envelope protocol (`type=command`)
- `eliza_robot.bridge.rosbridge_server`: ROSBridge-compatible protocol (`op=publish|subscribe|call_service|...`)

The ROSBridge-compatible endpoint provides protocol parity for simulation only.
Physical control is available solely through the authenticated unified endpoint.

## Targets

- `ros_real`: real AiNex ROS1 stack
- `ros_sim`: ROS simulation stack
- `isaac`: Isaac-target adapter with ROSBridge-compatible control semantics
- `mock`: in-memory development backend

## Quick Start

Use the repository-managed Python environment from the robot package root.

Verify host runtime prerequisites:

```bash
./eliza_robot/bridge/scripts/verify_runtime_env.sh
```

### Command-Envelope Server (existing API)

Before starting any physical backend, provision both server-side values. The
resource ID is the stable raw inventory identity for this actuator set; it is
deployment-specific, non-secret, and intentionally has no example value here.

```bash
export ELIZA_ROBOT_BRIDGE_AUTH_TOKEN='<random-32-to-4096-character-visible-ASCII-secret>'
: "${ELIZA_ROBOT_PHYSICAL_RESOURCE_ID:?export the stable inventory identity before starting}"
PYTHONPATH=. python -m eliza_robot.bridge.server --backend ros_real --host 127.0.0.1 --port 9100
```

The bridge validates the raw resource ID as 1–128 visible ASCII characters and
adds the `physical:` namespace itself. Keep the same raw value when backend
aliases or multiple server runtimes in one process address the same actuators.
This identity coordinates process-local ownership; it does not provide a
cross-process lock.

`@elizaos/plugin-ainex` reads `ELIZA_AINEX_BRIDGE_AUTH_TOKEN`, not the server
variable above. Configure that plugin setting with the same secret value as
`ELIZA_ROBOT_BRIDGE_AUTH_TOKEN`. The plugin does not consume
`ELIZA_ROBOT_PHYSICAL_RESOURCE_ID`; that is bridge-host configuration, not a
client credential. Do not place either bearer-token variable in a websocket
URL or logs.

With explicit safety and trace logging:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.server \
  --backend ros_real \
  --host 127.0.0.1 \
  --port 9100 \
  --queue-size 256 \
  --max-commands-per-sec 30 \
  --deadman-timeout-sec 1.0 \
  --trace-log-path /tmp/ainex_bridge_trace.jsonl
```

You can also load safety/logging defaults from config:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.server \
  --backend ros_real \
  --host 127.0.0.1 \
  --port 9100 \
  --config eliza_robot/bridge/config/default_bridge_config.json
```

The checked-in config does not provide either deployment value; the physical
launch still requires both environment variables.

### ROSBridge-Compatible Server (protocol parity)

The ROSBridge-compatible endpoint intentionally does not expose `ros_real`.
Physical control must use the authenticated, loopback-only unified endpoint above.

### Direct-hardware tool quarantine

Legacy evidence and calibration programs that instantiate `AinexRemoteBackend`
or `AsimovRemoteBackend` directly are quarantined. They fail before connecting
to hardware or creating physical-run artifacts. Their simulation-only flags
remain available. A physical workflow must be implemented as an authenticated
command-envelope client so ownership, validation, deadman, telemetry, and stop
handling all pass through `MotionSafetySupervisor`.

## ROSBridge-Compatible Operations

Supported websocket ops:

- `subscribe` / `unsubscribe`
- `publish`
- `call_service`
- `advertise` / `unadvertise` (acknowledged)
- `set_level` (acknowledged)

### Key Topics

- `/app/set_walking_param` (`publish`)
- `/app/set_action` (`publish`)
- `/head_pan_controller/command` (`publish`)
- `/head_tilt_controller/command` (`publish`)
- `/ros_robot_controller/bus_servo/set_position` (`publish`)
- `/ros_robot_controller/bus_servo/set_state` (`publish`)
- `/walking/is_walking` (`subscribe`)
- `/ros_robot_controller/battery` (`subscribe`)
- `/imu` (`subscribe`)

### Key Services

- `/walking/command` (`call_service`)
- `/ros_robot_controller/bus_servo/get_position` (`call_service`)
- `/ros_robot_controller/bus_servo/get_state` (`call_service`)

## Startup Scripts

- `eliza_robot/bridge/scripts/start_rosbridge_real.sh`
- `eliza_robot/bridge/scripts/start_rosbridge_sim.sh`
- `eliza_robot/bridge/scripts/start_rosbridge_isaac.sh`
- `eliza_robot/bridge/scripts/start_bridge_real.sh`
- `eliza_robot/bridge/scripts/start_bridge_sim.sh`

Both real-robot script names fail before launch unless
`ELIZA_ROBOT_BRIDGE_AUTH_TOKEN` is 32–4096 visible ASCII characters and
`ELIZA_ROBOT_PHYSICAL_RESOURCE_ID` is nonempty. Despite its compatibility
filename, `start_rosbridge_real.sh` starts only the unified envelope endpoint;
it never exposes the physical backend through the ROSBridge-compatible API.

## Isaac Preparation

Use the runbook and URDF export helper:

- `docs/bridge/isaaclab_setup.md`
- `eliza_robot/bridge/scripts/prepare_ainex_urdf.sh`

## Safety + Scheduling

Command-envelope mode includes:

- queue-based command execution
- rate limiter (`--max-commands-per-sec`)
- deadman stop (`--deadman-timeout-sec`)

These controls and deployment settings do not constitute real-hardware
validation or promotion of the incomplete hard safety envelope.

ROSBridge-compatible mode focuses on wire compatibility and backend parity.

## Run Tests

Use the repository test commands documented in `CLAUDE.md`.

## Smoke Test (ROSBridge Mode)

```bash
PYTHONPATH=. python3 -m eliza_robot.bridge.tools.rosbridge_smoke --uri ws://127.0.0.1:9090
```

## Parity Check (ROSBridge Mode)

```bash
PYTHONPATH=. python3 -m eliza_robot.bridge.tools.rosbridge_parity \
  --left-uri ws://127.0.0.1:19091 \
  --right-uri ws://127.0.0.1:19092
```

## ROS Backend Integration Test (Docker)

This runs a real ROS1 runtime in a container, builds required AiNex message packages, launches a ROS harness, and validates the `ros_real` bridge backend end-to-end.

```bash
./eliza_robot/bridge/scripts/run_ros_container_integration_test.sh
```

## Full Validation Pass

```bash
./eliza_robot/bridge/scripts/run_all_checks.sh
```
