import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import {
  BruhOsClient,
  bruhOsUrlFromHost,
  redactedBruhOsUrl,
} from "../src/bruhos-client";

async function startBruhOsServer(
  routes: Record<
    string,
    (req: IncomingMessage, body: unknown) => unknown | Promise<unknown>
  >,
) {
  const received: Array<{ method: string; path: string; body: unknown }> = [];
  const server = createServer(
    async (req: IncomingMessage, res: ServerResponse) => {
      const chunks: Buffer[] = [];
      for await (const chunk of req) chunks.push(chunk as Buffer);
      const raw = Buffer.concat(chunks).toString("utf8");
      const body = raw === "" ? null : JSON.parse(raw);
      const path = req.url?.split("?")[0] ?? "/";
      received.push({ method: req.method ?? "GET", path, body });
      const key = `${req.method ?? "GET"} ${path}`;
      const handler = routes[key];
      if (!handler) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "not found" }));
        return;
      }
      try {
        const result = await handler(req, body);
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(result));
      } catch (err) {
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: (err as Error).message }));
      }
    },
  );
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", () => r()));
  const port = (server.address() as AddressInfo).port;
  return {
    url: `http://127.0.0.1:${port}`,
    received,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

describe("BruhOsClient", () => {
  const closers: Array<() => Promise<void>> = [];
  afterEach(async () => {
    while (closers.length) {
      await closers.pop()?.();
    }
  });

  it("normalizes host to bruhOS URL", () => {
    expect(bruhOsUrlFromHost("192.168.149.1")).toBe(
      "http://192.168.149.1:9040",
    );
    expect(redactedBruhOsUrl("http://192.168.149.1:9040/secret")).toBe(
      "http://192.168.149.1:9040",
    );
  });

  it("loads capabilities and runs a skill", async () => {
    const server = await startBruhOsServer({
      "GET /v1/capabilities": () => ({
        name: "bruhos",
        version: "0.1.0.dev0",
        body: "tonypi",
        skills: [{ skill: "wave", description: "Wave hello." }],
      }),
      "POST /v1/skills/run": (_req, body) => {
        const call = body as { skill: string; args: Record<string, unknown> };
        expect(call.skill).toBe("wave");
        return { status: "ok", detail: "waved", data: {} };
      },
    });
    closers.push(server.close);
    const client = new BruhOsClient({ url: server.url });
    const caps = await client.capabilities();
    expect(caps.body).toBe("tonypi");
    expect(caps.skills[0]?.skill).toBe("wave");
    const outcome = await client.runSkill("wave");
    expect(outcome.ok).toBe(true);
    expect(outcome.detail).toBe("waved");
  });

  it("observe returns compressed observation shape", async () => {
    const server = await startBruhOsServer({
      "GET /v1/observe": () => ({
        t: 1.5,
        frame: 3,
        objects: [{ label: "block", color: "red", bearing_deg: 10 }],
        imu: { fallen: false },
        line: { visible: false, offset: 0 },
        holding: null,
      }),
    });
    closers.push(server.close);
    const client = new BruhOsClient({ url: server.url });
    const obs = await client.observe();
    expect(obs.frame).toBe(3);
    expect(obs.objects).toHaveLength(1);
  });
});
