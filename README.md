# elizaOS Research

Hardware, silicon, and embodiment research for [elizaOS](https://github.com/elizaOS/eliza)
— the open-source framework for building and deploying autonomous AI agents.

This repository holds the research-grade tracks that sit *below* and *beside* the
agent runtime: the custom AI phone SoC, the humanoid robot stack, the continual-RL
framework that trains its policies, and the elizaOS plugin that drives real
hardware. It is developed separately from the runtime so the main
[`elizaOS/eliza`](https://github.com/elizaOS/eliza) repo stays focused on the
agent framework, while this repo can carry large binary design artifacts (RTL,
CAD, meshes, checkpoints) without weighing that repo down.

## Contents

| Path              | What it is                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| [`chip/`](chip/)  | **Eliza E1 SoC** — CLI-first pre-tapeout scaffold for an open RISC-V AI phone chip: RTL, cocotb/formal verification, BSP, FPGA/package evidence, physical-design entry points. |
| [`robot/`](robot/) | **`@elizaos/robot`** — Python robotics stack (MuJoCo sim, Brax/MJX baselines, websocket bridge, perception, trajectory DB) with a thin TypeScript re-export surface. Profile-driven; first shipping profile is Hiwonder AiNex. |
| [`alberta/`](alberta/) | **Alberta Framework** — JAX implementation of [The Alberta Plan for AI Research](https://arxiv.org/abs/2208.11173); continual learning with no training phases or resets. Trains the robot policies. |
| [`plugin-ainex/`](plugin-ainex/) | **`@elizaos/plugin-ainex`** — elizaOS plugin that drives the Hiwonder AiNex humanoid (and compatible robots) through the robot bridge: locomotion, servo, action-group, and text-conditioned policy control. |
| [`docs/`](docs/)  | Chip track, robot embodiment doc, and TEE-native security design notes. |

## Relationship to elizaOS

This repo is mounted as a **git submodule** inside `elizaOS/eliza` at
`packages/research`. The elizaOS AI-phone OS build (`packages/os`) imports the
E1 chip device tree from `packages/research/chip/` via that submodule, so the
chip design stays reachable to the OS build without living in the runtime repo.

To work on this content from an eliza checkout (it is marked `update = none` in
eliza's `.gitmodules`, so routine `--recursive` checkouts skip its large payload;
`--checkout` opts in):

```bash
git submodule update --init --checkout packages/research
```

Or clone it standalone:

```bash
git clone https://github.com/elizaOS/research.git
```

## Per-track setup

Each directory carries its own `README.md` (and, where relevant, `AGENTS.md` /
`CLAUDE.md`) with build, test, and toolchain instructions. Start there:

- Silicon: [`chip/README.md`](chip/README.md)
- Robot: [`robot/README.md`](robot/README.md)
- Training framework: [`alberta/README.md`](alberta/README.md)
- Plugin: [`plugin-ainex/README.md`](plugin-ainex/README.md)

## License

MIT, matching elizaOS. Individual vendored components under `robot/vendor/` and
`alberta/` retain their upstream licenses (see the respective directories).
