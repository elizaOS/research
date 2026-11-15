# @elizaos/plugin-tonypi

elizaOS plugin that drives a **Hiwonder TonyPi / TonyPi Pro** through the stock
on-robot **JSON-RPC server** (`RPCServer.py`, port **9030**). Same transport as
the bruhOS mobile app.

**bruhOS is optional progressive enhancement** — this plugin never requires it.
A stock TonyPi or TonyPi Pro with only `RPCServer.py` is a complete target.

```
Eliza agent  →  @elizaos/plugin-tonypi  →  http://<robot>:9030  →  RPCServer.py  →  servos
                         │
                         └─(optional)→  http://<robot>:9040  →  bruhOS skills
```

Leave the robot's stock `TonyPi.py` / `RPCServer.py` running. Do not open the
servo UART from a second process.

## Progressive enhancement

| Layer | When | What you get |
|---|---|---|
| **stock** | `ELIZA_TONYPI_RPC_URL` / `HOST` + Heartbeat | Walk, stop, action groups, bus servos |
| **pro** | Same stock RPC (TonyPi Pro image / richer firmware) | Head, StandUp, hands, **LoadFunc modes** — vision + **voice / Smart Chat** (ChatGPT + voice module) |
| **bruhos** | Optional `ELIZA_BRUHOS_URL` answers `/v1/capabilities` | Skill actions (`detect`, `walk_toward`, `grab`, …) |

TonyPi Pro’s marketed voice / multimodal AI is **on-robot Functions** over the
same `:9030` RPC (`LoadFunc`), not a separate bruhOS dependency. AI modes need
the robot on Wi-Fi with an LLM key (WonderPi / `SetLLMKey`); the mic is the
Pro voice interaction module when fitted.

If bruhOS is down or unset, stock actions keep working and `TONYPI_RUN_SKILL`
stays inactive (`validate` → false).

### Reverse path (bruhOS → Eliza)

bruhOS is designed to call **into** an Eliza instance over REST (planner /
agent API). That direction is configured in bruhOS, not in this plugin. This
plugin is Eliza → robot only.

## Actions

| Action | Transport | Notes |
|---|---|---|
| `TONYPI_WALK_*` / turn / side-step | stock RPC `RunAction(..., 0)` | Loop until stop |
| `TONYPI_STOP` | stock RPC stop trio | Frame + group + walk-end |
| `TONYPI_STAND` / `SQUAT` / `WAVE` / `BOW` | stock RPC | |
| `TONYPI_STAND_UP` | stock RPC `StandUp` | Fall recovery |
| `TONYPI_SET_HEAD` | stock RPC `SetPWMServo` | pitch + yaw |
| `TONYPI_SET_SERVO` | stock RPC `SetBusServoPulse` | Ids 1..16 |
| `TONYPI_RUN_ACTION_GROUP` | stock RPC | Allowlisted names / dict codes |
| `TONYPI_LOAD_MODE` | stock RPC `LoadFunc` | Pro modes: `voice_control` (14), `smart_chat` (13), vision 2–12, AI 13–18 |
| `TONYPI_UNLOAD_MODE` | `StopFunc` + `UnloadFunc` | Leave a Pro mode |
| `TONYPI_RUN_SKILL` | **bruhOS only** | Options: `skill`, `args` / `target` |

### Provider

- **`TONYPI_ROBOT_STATE`** — stock RPC, enhancement level, bruhOS skills, last action

## Configuration

| Environment variable | Purpose |
|---|---|
| `ELIZA_TONYPI_RPC_URL` | Full URL, e.g. `http://192.168.149.1:9030` (preferred) |
| `ELIZA_TONYPI_HOST` | Host only; port defaults to `9030` |
| `ELIZA_TONYPI_BRIDGE_URL` | Legacy alias for RPC URL (`http(s)://` only) |
| `ELIZA_BRUHOS_URL` | Optional bruhOS skill API, e.g. `http://192.168.149.1:9040` |
| `ELIZA_BRUHOS_HOST` | Optional bruhOS host; port defaults to `9040` |
| `ELIZA_BRUHOS_AUTO` | `true`/`1` → probe `http://$ELIZA_TONYPI_HOST:9040` |

Auto-enables when any of those are set, or when `features.tonypi = true`.

### bruhOS HTTP contract (optional)

When present, bruhOS should expose:

- `GET /v1/capabilities` → `{ name, version, body?, skills? }`
- `GET /v1/skills` → `{ skills: [...] }`
- `GET /v1/observe` → compressed observation
- `POST /v1/skills/run` → `{ skill, args }` → `{ status, detail, data }`

## Related

- Stock robot tree: `RPCServer.py`, `ActionGroupDict.py`, `ActionGroups/*.d6a`
- bruhOS (`~/Sites/bruhOS`) — skill OS beside the body
- bruhos-app `src/lib/rpc.ts` — same stock JSON-RPC surface
- `@elizaos/plugin-ainex` — different robot (AiNex) over the `eliza_robot` bridge
