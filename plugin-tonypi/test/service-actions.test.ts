import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import type { IAgentRuntime } from "@elizaos/core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  loadModeAction,
  runSkillAction,
  stopAction,
  walkForwardAction,
  waveAction,
} from "../src/actions";
import { robotStateProvider } from "../src/providers";
import { TonyPiService } from "../src/service";

interface TestRuntimeState {
  service: TonyPiService | null;
  settings: Record<string, string>;
}

function buildRuntime(): IAgentRuntime {
  const state: TestRuntimeState = { service: null, settings: {} };
  const runtime: Partial<IAgentRuntime> = {
    agentId: "test-agent" as IAgentRuntime["agentId"],
    getSetting: (key: string) => state.settings[key] ?? null,
    getService: <T>(_serviceType: string) => state.service as unknown as T,
  };
  return Object.assign(runtime, { __state: state }) as IAgentRuntime;
}

async function buildRpcServer() {
  const received: Array<{ method: string; params: unknown[] }> = [];
  const server = createServer(
    async (req: IncomingMessage, res: ServerResponse) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
        method: string;
        params?: unknown[];
        id: number;
      };
      received.push({ method: body.method, params: body.params ?? [] });
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          jsonrpc: "2.0",
          id: body.id,
          result: [true, {}, body.method],
        }),
      );
    },
  );
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
  const port = (server.address() as AddressInfo).port;
  return {
    url: `http://127.0.0.1:${port}/`,
    received,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

describe("TonyPiService + actions (stock RPC)", () => {
  let runtime: IAgentRuntime;
  let server: Awaited<ReturnType<typeof buildRpcServer>>;

  beforeEach(async () => {
    runtime = buildRuntime();
    server = await buildRpcServer();
    const state = (runtime as unknown as { __state: TestRuntimeState }).__state;
    state.settings.ELIZA_TONYPI_RPC_URL = server.url;
    const service = await TonyPiService.start(runtime);
    state.service = service;
  });

  afterEach(async () => {
    const state = (runtime as unknown as { __state: TestRuntimeState }).__state;
    await state.service?.stop();
    await server.close();
  });

  it("connects with Heartbeat", () => {
    expect(server.received[0]?.method).toBe("Heartbeat");
    expect(
      (
        runtime as unknown as { __state: TestRuntimeState }
      ).__state.service?.isConnected(),
    ).toBe(true);
  });

  it("walk forward sends RunAction go_forward times=0", async () => {
    server.received.length = 0;
    const result = await walkForwardAction.handler?.(
      runtime,
      {} as never,
      undefined,
      undefined,
      async () => {},
    );
    expect(result?.success).toBe(true);
    expect(server.received).toEqual([
      { method: "RunAction", params: ["go_forward", 0] },
    ]);
  });

  it("stop sends StopBusServo + StopActionGroup + RunAction 0", async () => {
    server.received.length = 0;
    const result = await stopAction.handler?.(
      runtime,
      {} as never,
      undefined,
      undefined,
      async () => {},
    );
    expect(result?.success).toBe(true);
    expect(server.received.map((r) => r.method)).toEqual([
      "StopBusServo",
      "StopActionGroup",
      "RunAction",
    ]);
    expect(server.received[2]?.params).toEqual(["0", 1]);
  });

  it("wave plays once", async () => {
    server.received.length = 0;
    const result = await waveAction.handler?.(
      runtime,
      {} as never,
      undefined,
      undefined,
      async () => {},
    );
    expect(result?.success).toBe(true);
    expect(server.received[0]).toEqual({
      method: "RunAction",
      params: ["wave", 1],
    });
  });

  it("robot state provider reports last action", async () => {
    await walkForwardAction.handler?.(
      runtime,
      {} as never,
      undefined,
      undefined,
      async () => {},
    );
    const result = await robotStateProvider.get(
      runtime,
      {} as never,
      {} as never,
    );
    expect(result.values?.tonyPiConnected).toBe(true);
    expect(result.values?.tonyPiRunning).toBe(true);
    expect(result.values?.tonyPiAction).toBe("go_forward");
    expect(result.values?.tonyPiBruhOs).toBe(false);
    // Mock RPC answers NetWorkTest → "pro"; real stock-only robots stay "stock".
    expect(["stock", "pro"]).toContain(result.values?.tonyPiEnhancement);
  });

  it("run skill stays unavailable without bruhOS", async () => {
    const ok = await runSkillAction.validate?.(runtime, {} as never);
    expect(ok).toBe(false);
  });

  it("load voice control mode sends LoadFunc 14", async () => {
    server.received.length = 0;
    const result = await loadModeAction.handler?.(
      runtime,
      {} as never,
      undefined,
      { mode: "voice_control" },
      async () => {},
    );
    expect(result?.success).toBe(true);
    expect(server.received[0]).toEqual({
      method: "LoadFunc",
      params: [14],
    });
    expect(server.received.some((r) => r.method === "StartFunc")).toBe(true);
  });
});

describe("TonyPiService bruhOS progressive enhancement", () => {
  let runtime: IAgentRuntime;
  let rpc: Awaited<ReturnType<typeof buildRpcServer>>;
  let bruh: {
    url: string;
    close: () => Promise<void>;
  };

  beforeEach(async () => {
    runtime = buildRuntime();
    rpc = await buildRpcServer();
    const bruhServer = createServer(
      async (req: IncomingMessage, res: ServerResponse) => {
        const path = req.url?.split("?")[0] ?? "/";
        if (req.method === "GET" && path === "/v1/capabilities") {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              name: "bruhos",
              version: "0.1.0.dev0",
              body: "tonypi_pro",
              skills: [
                { skill: "detect" },
                { skill: "walk_toward" },
                { skill: "wave" },
              ],
            }),
          );
          return;
        }
        if (req.method === "POST" && path === "/v1/skills/run") {
          const chunks: Buffer[] = [];
          for await (const chunk of req) chunks.push(chunk as Buffer);
          const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
            skill: string;
          };
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              status: "ok",
              detail: `ran ${body.skill}`,
              data: {},
            }),
          );
          return;
        }
        res.writeHead(404).end();
      },
    );
    await new Promise<void>((r) =>
      bruhServer.listen(0, "127.0.0.1", () => r()),
    );
    const port = (bruhServer.address() as AddressInfo).port;
    bruh = {
      url: `http://127.0.0.1:${port}`,
      close: () => new Promise<void>((r) => bruhServer.close(() => r())),
    };

    const state = (runtime as unknown as { __state: TestRuntimeState }).__state;
    state.settings.ELIZA_TONYPI_RPC_URL = rpc.url;
    state.settings.ELIZA_BRUHOS_URL = bruh.url;
    const service = await TonyPiService.start(runtime);
    state.service = service;
  });

  afterEach(async () => {
    const state = (runtime as unknown as { __state: TestRuntimeState }).__state;
    await state.service?.stop();
    await rpc.close();
    await bruh.close();
  });

  it("probes bruhOS without breaking stock RPC", () => {
    const service = (runtime as unknown as { __state: TestRuntimeState })
      .__state.service;
    expect(service?.isConnected()).toBe(true);
    expect(service?.hasBruhOs()).toBe(true);
    expect(service?.getCapabilities().body).toBe("tonypi_pro");
    expect(service?.getState().enhancement).toBe("bruhos");
  });

  it("run skill works when bruhOS is present", async () => {
    expect(await runSkillAction.validate?.(runtime, {} as never)).toBe(true);
    const result = await runSkillAction.handler?.(
      runtime,
      {} as never,
      undefined,
      { skill: "walk_toward", target: "red_block" },
      async () => {},
    );
    expect(result?.success).toBe(true);
    expect(result?.text).toContain("walk_toward");
  });
});
