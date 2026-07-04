# elizaOS Research — repository guide for agents

Hardware, silicon, and embodiment research for **elizaOS**. This repo holds the
tracks that sit below and beside the agent runtime — the E1 AI phone SoC, the
humanoid robot stack, the continual-RL training framework, and the elizaOS
plugin that drives real hardware. It is a **git submodule** of
[`elizaOS/eliza`](https://github.com/elizaOS/eliza), mounted at
`packages/research`; the eliza OS build consumes `chip/` from here.

`CLAUDE.md` and `AGENTS.md` at every level are **identical** — author
`CLAUDE.md`, then copy it to `AGENTS.md`. Read the directory-local `CLAUDE.md`
before working inside any track; this root file is the map.

## Naming

Write **elizaOS** (not `ElizaOS`). npm scope is `@elizaos/*`. In plain language,
say **Eliza agents**. **Never reference "Milady" or any downstream white-label
distribution anywhere in this repository** — this is an elizaOS project; keep
downstream product names out of it.

## Layout

```
chip/           Eliza E1 SoC — RTL, verification (cocotb/formal), BSP, PD, board/package artifacts
robot/          @elizaos/robot — Python robotics stack (MuJoCo/Brax/MJX, bridge, perception) + thin TS surface
alberta/        Alberta Framework — JAX continual-RL (The Alberta Plan); trains robot policies
plugin-ainex/   @elizaos/plugin-ainex — elizaOS plugin driving the Hiwonder AiNex humanoid via the robot bridge
docs/           chip/ track, robot.mdx embodiment doc, tee-native/ security design notes
.github/workflows/  CI: typescript · alberta · robot (light lane) · e1-chip-fast
                    (per-PR make lint typecheck) · e1-chip (heavy Docker regression,
                    workflow_dispatch + monthly cron)
```

## Toolchains (per track — see each directory's docs)

- **chip/** — Verilator, Yosys, cocotb, OpenROAD/OpenLane, KiCad; Docker tool image
  built by `e1-chip.yml`. Native builds are preferred on Linux x86_64.
- **robot/** — Python 3 + JAX/MuJoCo/Brax; profile-driven (`RobotProfileId`).
  Heavy logic in the `eliza_robot` Python package; TS `src/` is a thin surface.
- **alberta/** — Python 3.12+, JAX 0.4+. `pip install -e alberta/`.
- **plugin-ainex/** — TypeScript elizaOS plugin; `bun run build`. Depends on the
  `eliza_robot` bridge (`python -m eliza_robot.bridge.server`).

## Conventions

- **ESM only** for TypeScript; **logger over `print`/`console`** in shipped code.
- Large design artifacts (RTL nets, CAD `.step`, meshes `.STL`, checkpoints) are
  tracked directly — there is no git-lfs here. Regenerable build outputs
  (`build/`, `out/`, `vendor/mujoco_menagerie/`, `.tools/`) are gitignored; the
  golden snapshots that are tracked must stay reproducible from source.
- The chip CI (`e1-chip.yml`) carries a monthly AlphaChip checkpoint-blocker
  attestation gate — a deliberate human re-attestation, not something to
  auto-satisfy.

## Relationship to the eliza runtime

`@elizaos/plugin-ainex` is an elizaOS plugin: it registers actions/providers and
talks to the robot bridge. `@elizaos/robot` exposes a thin TS surface
(`RobotProfileId`, `ROBOT_PACKAGE_VERSION`) that the plugin imports. Nothing in
the eliza runtime imports the heavy Python here — the boundary is the bridge
websocket and the plugin. Keep it that way: the runtime depends on the plugin,
not on the research internals.
