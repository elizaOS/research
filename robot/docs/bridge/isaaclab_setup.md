# Isaac Sim + IsaacLab Setup for AiNex

This runbook covers end-to-end setup: from prerequisites through asset pipeline to running a ROSBridge-compatible websocket endpoint for agent control.

## 1) Prerequisites

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| OS | Ubuntu 22.04 | Ubuntu 22.04 |
| GPU VRAM | 8 GB | 16 GB |
| NVIDIA Driver | 535+ | Latest |
| CUDA | 12.1+ | 12.4+ |
| Python | 3.10 | 3.10 |
| Isaac Sim | 4.2.0 | 4.5.0 |
| IsaacLab | 2.0.0 | 2.1.0 |

Version pins are tracked in `bridge/config/isaaclab_versions.json`.

Check prerequisites:

```bash
nvidia-smi              # GPU driver and VRAM
nvcc --version          # CUDA toolkit
python3 --version       # Python version
```

## 2) Environment Setup

```bash
./bridge/scripts/setup_isaac_env.sh
```

This creates a virtual environment and installs bridge dependencies. Follow the on-screen instructions for Isaac Sim and IsaacLab installation.

## 3) Export AiNex URDF from xacro

```bash
./bridge/scripts/prepare_ainex_urdf.sh
```

Generates:
- `bridge/generated/ainex.urdf` — standalone URDF with patched mesh paths
- `bridge/generated/meshes/` — copied STL mesh files

Source: `ros_ws_src/ainex_simulations/ainex_description/urdf/ainex.xacro`

## 4) Validate Robot Model

```bash
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.validate_model
```

Checks:
- All 24 revolute joints present with correct limits
- Link masses are physically reasonable
- Mesh references resolve
- Standing pose is valid

## 5) Convert URDF to USD

In the Isaac-enabled Python environment:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.convert_urdf_to_usd
```

Or validate only:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.convert_urdf_to_usd --validate-only
```

Output: `bridge/generated/ainex.usd`

## 6) Test IsaacLab Configuration

Dry-run (no Isaac Sim required):

```bash
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.run_sim --dry-run
```

Full simulation (requires Isaac Sim):

```bash
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.run_sim
PYTHONPATH=. python -m eliza_robot.bridge.isaaclab.run_sim --headless
```

## 7) Start Bridge

### Unified Launcher

```bash
# Isaac backend (default)
PYTHONPATH=. python -m eliza_robot.bridge.launch --target isaac

# Real robot (both deployment values must already be provisioned)
: "${ELIZA_ROBOT_BRIDGE_AUTH_TOKEN:?set a random 32-to-4096-character visible-ASCII token}"
: "${ELIZA_ROBOT_PHYSICAL_RESOURCE_ID:?set the stable inventory identity for this robot}"
PYTHONPATH=. python -m eliza_robot.bridge.launch --target real

# Gazebo simulation
PYTHONPATH=. python -m eliza_robot.bridge.launch --target sim

# Development mock
PYTHONPATH=. python -m eliza_robot.bridge.launch --target mock

# List all targets
PYTHONPATH=. python -m eliza_robot.bridge.launch --list-targets
```

### Convenience Scripts

```bash
./eliza_robot/bridge/scripts/start_rosbridge_isaac.sh   # Isaac backend
./eliza_robot/bridge/scripts/start_rosbridge_real.sh    # Real unified endpoint
./eliza_robot/bridge/scripts/start_rosbridge_sim.sh     # Gazebo sim
./eliza_robot/bridge/scripts/start_rosbridge_mock.sh    # Mock backend
```

The simulation scripts expose a ROSBridge-compatible websocket. The
compatibility-named real script requires both physical deployment variables
and starts only the authenticated, loopback unified endpoint on port 9100.

### Environment Overrides

| Variable | Description |
|----------|-------------|
| `AINEX_BRIDGE_HOST` | Listen host (default: 0.0.0.0) |
| `AINEX_ROSBRIDGE_PORT` | ROSBridge port (default: 9090) |
| `AINEX_ENVELOPE_PORT` | Command-envelope port (default: 9100) |
| `AINEX_PUBLISH_HZ` | Telemetry publish rate |
| `AINEX_MAX_CMD_SEC` | Rate limit (commands/sec) |
| `AINEX_DEADMAN_SEC` | Deadman timeout (seconds) |
| `ELIZA_ROBOT_BRIDGE_AUTH_TOKEN` | Bridge bearer secret; physical targets require 32–4096 visible ASCII characters |
| `ELIZA_ROBOT_PHYSICAL_RESOURCE_ID` | Stable raw inventory identity; required only by the physical bridge host |

The elizaOS plugin uses `ELIZA_AINEX_BRIDGE_AUTH_TOKEN`; configure it with the
same secret value as `ELIZA_ROBOT_BRIDGE_AUTH_TOKEN`. It does not read the
physical resource ID. The ID is process-local coordination metadata, not a
client credential or a cross-process lock. These settings do not establish
real-hardware validation.

## 8) Run Tests

```bash
# All unit and integration tests
PYTHONPATH=. python -m unittest discover -s bridge/tests -p "test_*.py"

# Specific test suites
PYTHONPATH=. python -m unittest bridge.tests.test_rosbridge_contract
PYTHONPATH=. python -m unittest bridge.tests.test_backend_parity
PYTHONPATH=. python -m unittest bridge.tests.test_isaac_backend
PYTHONPATH=. python -m unittest bridge.tests.test_joint_map
PYTHONPATH=. python -m unittest bridge.tests.test_ainex_cfg
PYTHONPATH=. python -m unittest bridge.tests.test_actions
PYTHONPATH=. python -m unittest bridge.tests.test_sim_state
```

Smoke test against a running endpoint:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.tools.rosbridge_smoke --uri ws://127.0.0.1:9090
```

Parity check between two endpoints:

```bash
PYTHONPATH=. python -m eliza_robot.bridge.tools.rosbridge_parity \
  --left-uri ws://127.0.0.1:9090 \
  --right-uri ws://127.0.0.1:9091
```

## 9) Endpoint Swap Acceptance Checklist

To verify "drop-in endpoint swap" between targets:

- [ ] Same websocket client connects to both `real` and `isaac` endpoints
- [ ] `subscribe` to `/ros_robot_controller/battery` returns data on both
- [ ] `call_service` to `/walking/command` with `start`/`stop` succeeds on both
- [ ] `publish` to `/app/set_walking_param` accepted on both
- [ ] `publish` to `/head_pan_controller/command` accepted on both
- [ ] `call_service` to `/ros_robot_controller/bus_servo/get_position` returns positions on both
- [ ] `publish` to `/ros_robot_controller/bus_servo/set_position` accepted on both
- [ ] `get_time` returns valid secs/nsecs on both
- [ ] `advertise` acknowledged on both
- [ ] Error responses preserve request IDs on both
- [ ] Rate limiting activates at configured threshold
- [ ] Deadman timeout issues auto-stop after inactivity

## 10) Network Topology

```
┌─────────────────┐     ws://host:9090     ┌──────────────────┐
│  ML Agent /      │ ──────────────────────▶ │  ROSBridge        │
│  Web Client      │ ◀────────────────────── │  Websocket Server │
└─────────────────┘     (bidirectional)     └────────┬─────────┘
                                                      │
                                             ┌────────┴─────────┐
                                             │  Target Router    │
                                             └──┬────┬────┬─────┘
                                                │    │    │
                               ┌────────────────┘    │    └───────────────┐
                               ▼                     ▼                    ▼
                        ┌──────────┐          ┌──────────┐         ┌──────────┐
                        │ Real     │          │ Gazebo   │         │ IsaacLab │
                        │ Robot    │          │ Sim      │         │ Sim      │
                        │ (ROS1)   │          │ (ROS1)   │         │ (USD)    │
                        └──────────┘          └──────────┘         └──────────┘
```
