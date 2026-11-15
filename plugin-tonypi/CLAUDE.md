# @elizaos/plugin-tonypi

Drives a Hiwonder TonyPi / TonyPi Pro via the stock on-robot JSON-RPC server
(`RPCServer.py` :9030). Same wire format as the bruhOS mobile app.

**bruhOS is progressive enhancement only** — never required. Stock TonyPi and
TonyPi Pro work with RPC alone. Optional `ELIZA_BRUHOS_URL` unlocks skill
actions when a bruhOS skill HTTP API is reachable.

## Purpose / role

Agent-facing TypeScript surface. Does **not** open `/dev/ttyAMA0` and does **not**
register an `eliza_robot` backend / `RobotProfile`. The robot's stock
`RPCServer.py` must already be running for locomotion.

bruhOS may separately call **into** Eliza over REST; that reverse path lives in
bruhOS, not here.

## Layout

```
src/
  index.ts              tonyPiPlugin export + autoEnable
  service.ts            TonyPiService — stock RPC + optional bruhOS probe
  rpc-client.ts         TonyPiRpcClient — JSON-RPC HTTP
  bruhos-client.ts      Optional BruhOsClient — /v1/capabilities, skills/run
  capabilities.ts       stock / pro / bruhos levels
  pro-modes.ts          LoadFunc catalog (voice_control=14, smart_chat=13, …)
  types.ts              Walk codes/groups + action allowlist
  actions/              TONYPI_* (+ LOAD_MODE Pro; RUN_SKILL when bruhOS up)
  providers/            TONYPI_ROBOT_STATE
bridge/
  README.md             Points at stock RPCServer (no custom sidecar)
```

## Commands

```bash
bun run --cwd plugin-tonypi typecheck
bun run --cwd plugin-tonypi test
```

## Config / env vars

| Var | Purpose |
|---|---|
| `ELIZA_TONYPI_RPC_URL` | `http://<host>:9030` (preferred) |
| `ELIZA_TONYPI_HOST` | Host only; port 9030 |
| `ELIZA_TONYPI_BRIDGE_URL` | Legacy alias; http(s) only |
| `ELIZA_BRUHOS_URL` | Optional bruhOS skill API (`http://<host>:9040`) |
| `ELIZA_BRUHOS_HOST` | Optional bruhOS host; port 9040 |
| `ELIZA_BRUHOS_AUTO` | `true`/`1` → probe bruhOS on TonyPi host:9040 |

## Conventions / gotchas

- Do **not** stop `RPCServer.py` — it owns the bus; we are a client.
- Do **not** require bruhOS for any stock action.
- Walk loops use named groups from ActionGroupDict (`back_fast` for code `2`).
- Stop calls `StopBusServo` + `StopActionGroup` + `RunAction("0")`.
- Action names are allowlisted; path characters rejected.
- `TONYPI_RUN_SKILL` validates only when bruhOS capabilities probe succeeds.
- Pro voice / multimodal AI = stock `LoadFunc` modes (14 voice, 13 chat), not bruhOS.
- Camera stream / raw WonderEcho mic ASR stay out of this plugin; agent uses modes + Eliza.
