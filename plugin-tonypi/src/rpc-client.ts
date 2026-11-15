// JSON-RPC 2.0 client for the stock Hiwonder TonyPi RPCServer (port 9030).
// Wire format matches bruhos-app `src/lib/rpc.ts` and on-robot `RPCServer.py`.

export const DEFAULT_RPC_PORT = 9030;
/** Hiwonder AP-mode robot address. */
export const DEFAULT_AP_HOST = "192.168.149.1";

export class TonyPiRpcError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "TonyPiRpcError";
  }
}

export interface TonyPiRpcClientOptions {
  /** Base URL, e.g. `http://192.168.149.1:9030` or `http://127.0.0.1:9030`. */
  url: string;
  /** Default per-call timeout (ms). Default 4000. */
  timeoutMs?: number;
}

let _seq = 0;

export function redactedRpcUrl(rawUrl: string): string {
  try {
    const parsed = new URL(rawUrl);
    if (
      !["http:", "https:"].includes(parsed.protocol) ||
      parsed.hostname === ""
    ) {
      return "<invalid tonyPi rpc endpoint>";
    }
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return "<invalid tonyPi rpc endpoint>";
  }
}

function _normalizeRpcUrl(rawUrl: string): string {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error("TonyPi RPC URL must be a valid http:// or https:// URL");
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("TonyPi RPC URL must use http:// or https://");
  }
  if (parsed.username !== "" || parsed.password !== "") {
    throw new Error(
      "TonyPi RPC credentials must not be placed in URL userinfo",
    );
  }
  // RPCServer serves POST / ; strip path noise so we always hit the root.
  parsed.pathname = "/";
  parsed.search = "";
  parsed.hash = "";
  return `${parsed.toString().replace(/\/$/, "")}/`;
}

/** Build an RPC URL from host (and optional port). */
export function rpcUrlFromHost(
  host: string,
  port: number = DEFAULT_RPC_PORT,
): string {
  const trimmed = host.trim();
  if (trimmed === "") throw new Error("TonyPi host must not be empty");
  if (/^https?:\/\//i.test(trimmed)) return _normalizeRpcUrl(trimmed);
  return _normalizeRpcUrl(`http://${trimmed}:${port}`);
}

export class TonyPiRpcClient {
  readonly url: string;
  readonly timeoutMs: number;

  constructor(options: TonyPiRpcClientOptions) {
    this.url = _normalizeRpcUrl(options.url);
    this.timeoutMs = options.timeoutMs ?? 4000;
  }

  /** Low-level JSON-RPC call. Robot methods return `[ok, data, name]`. */
  async call<T = unknown>(
    method: string,
    params: unknown[] = [],
    timeoutMs?: number,
  ): Promise<T> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs ?? this.timeoutMs);
    try {
      const res = await fetch(this.url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          method,
          params,
          id: ++_seq,
        }),
        signal: ctrl.signal,
      });
      if (!res.ok) {
        throw new TonyPiRpcError(
          `HTTP ${res.status} from ${redactedRpcUrl(this.url)}`,
        );
      }
      const body = (await res.json()) as {
        error?: { message?: string };
        result?: unknown;
      };
      if (body.error) {
        throw new TonyPiRpcError(body.error.message ?? "RPC error");
      }
      const result = body.result;
      // Robot methods reply with [ok, data, name]; ok === false carries E0x.
      if (Array.isArray(result) && result[0] === false) {
        throw new TonyPiRpcError(String(result[1]));
      }
      return (Array.isArray(result) ? result[1] : result) as T;
    } catch (err) {
      if (err instanceof TonyPiRpcError) throw err;
      if ((err as Error).name === "AbortError") {
        throw new TonyPiRpcError(
          `RPC timeout (${timeoutMs ?? this.timeoutMs}ms) calling ${method}`,
        );
      }
      throw new TonyPiRpcError(
        `RPC ${method} failed: ${(err as Error).message}`,
      );
    } finally {
      clearTimeout(timer);
    }
  }

  heartbeat(): Promise<unknown> {
    return this.call("Heartbeat", [], 2500);
  }

  /**
   * Run a stock action group. `name` may be an ActionGroupDict code (`"1"`)
   * or a `.d6a` basename (`"go_forward"`). `times=0` loops until stop.
   */
  runAction(name: string, times = 1): Promise<unknown> {
    return this.call("RunAction", [name, times]);
  }

  /** End a looping walk the same way the Hiwonder / bruhOS apps do. */
  stopWalk(): Promise<unknown> {
    return this.call("RunAction", ["0", 1]);
  }

  /** Cancel the group loop (`stop_action_group`). */
  stopActionGroup(): Promise<unknown> {
    return this.call("StopActionGroup", ["stopActionGroup"]);
  }

  /** Cancel the in-frame action player (`stop_action`). */
  stopBusServo(): Promise<unknown> {
    return this.call("StopBusServo", ["stopAction"]);
  }

  /**
   * Full stop: frame cancel + group cancel + walk-end code.
   * Matches the two SDK stop flags plus the app's `RunAction('0')`.
   */
  async stopAll(): Promise<void> {
    const errors: string[] = [];
    for (const step of [
      () => this.stopBusServo(),
      () => this.stopActionGroup(),
      () => this.stopWalk(),
    ]) {
      try {
        await step();
      } catch (err) {
        errors.push((err as Error).message);
      }
    }
    if (errors.length === 3) {
      throw new TonyPiRpcError(`stop failed: ${errors.join("; ")}`);
    }
  }

  /**
   * `SetBusServoPulse(duration_ms, count, id, pulse, ...)`.
   * Ids 1..16 per RPCServer validation.
   */
  setBusServoPulse(
    durationMs: number,
    positions: Array<{ id: number; pulse: number }>,
  ): Promise<unknown> {
    if (positions.length === 0) {
      throw new TonyPiRpcError("SetBusServoPulse requires at least one servo");
    }
    const params: number[] = [durationMs, positions.length];
    for (const p of positions) {
      params.push(p.id, p.pulse);
    }
    return this.call("SetBusServoPulse", params);
  }

  standUp(): Promise<unknown> {
    return this.call("StandUp", []);
  }

  /**
   * Head look via PWM servos (pitch id=1, yaw id=2), same as bruhos-app.
   * Present on stock images that expose SetPWMServo.
   */
  setHead(pitch: number, yaw: number, durationMs = 120): Promise<unknown> {
    return this.call("SetPWMServo", [
      durationMs,
      2,
      1,
      Math.round(pitch),
      2,
      Math.round(yaw),
    ]);
  }

  /**
   * Soft Pro-feature probe: read-only NetWorkTest when present.
   * Returns false on any failure — stock robots stay usable.
   */
  async probeProFeatures(): Promise<boolean> {
    try {
      await this.call("NetWorkTest", [], 2500);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Load an on-robot Function by id (vision / AI / voice modes).
   * Ids match Functions/Running.py and bruhos-app MODES (e.g. 14 = Voice Control).
   */
  loadFunc(id: number): Promise<unknown> {
    return this.call("LoadFunc", [id], 12000);
  }

  unloadFunc(): Promise<unknown> {
    return this.call("UnloadFunc", [], 12000);
  }

  startFunc(): Promise<unknown> {
    return this.call("StartFunc", [], 12000);
  }

  stopFunc(): Promise<unknown> {
    return this.call("StopFunc", [], 12000);
  }

  /** Dance 1–4 with song; 0 stops. Stock RPC SingAndDance. */
  singAndDance(n: 0 | 1 | 2 | 3 | 4): Promise<unknown> {
    return this.call("SingAndDance", [n]);
  }

  /**
   * Color target for pick-capable vision modes (x, y in frame coords).
   * Optional progressive Pro RPC — unused on bare stock locomotion.
   */
  setPoint(id: number, x: number, y: number): Promise<unknown> {
    return this.call("SetPoint", [id, [x, y]]);
  }
}
