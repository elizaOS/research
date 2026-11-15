import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import {
  redactedRpcUrl,
  rpcUrlFromHost,
  TonyPiRpcClient,
  TonyPiRpcError,
} from "../src/rpc-client";

async function startRpcServer(
  handler: (method: string, params: unknown[]) => unknown,
) {
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
      let result: unknown;
      try {
        result = handler(body.method, body.params ?? []);
      } catch (err) {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            jsonrpc: "2.0",
            id: body.id,
            error: { message: (err as Error).message },
          }),
        );
        return;
      }
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ jsonrpc: "2.0", id: body.id, result }));
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

describe("TonyPiRpcClient", () => {
  const closers: Array<() => Promise<void>> = [];
  afterEach(async () => {
    while (closers.length) {
      await closers.pop()?.();
    }
  });

  it("normalizes host to RPC URL", () => {
    expect(rpcUrlFromHost("192.168.149.1")).toBe("http://192.168.149.1:9030/");
    expect(redactedRpcUrl("http://192.168.149.1:9030/secret")).toBe(
      "http://192.168.149.1:9030",
    );
  });

  it("calls RunAction and unwraps [ok, data, name]", async () => {
    const server = await startRpcServer((method, params) => {
      expect(method).toBe("RunAction");
      expect(params).toEqual(["go_forward", 0]);
      return [true, {}, "RunAction"];
    });
    closers.push(server.close);
    const client = new TonyPiRpcClient({ url: server.url });
    await client.runAction("go_forward", 0);
    expect(server.received[0]?.method).toBe("RunAction");
  });

  it("stopAll sends frame + group + walk-end", async () => {
    const server = await startRpcServer((method) => [true, {}, method]);
    closers.push(server.close);
    const client = new TonyPiRpcClient({ url: server.url });
    await client.stopAll();
    expect(server.received.map((r) => r.method)).toEqual([
      "StopBusServo",
      "StopActionGroup",
      "RunAction",
    ]);
    expect(server.received[2]?.params).toEqual(["0", 1]);
  });

  it("SetBusServoPulse packs duration, count, id/pulse pairs", async () => {
    const server = await startRpcServer((method, params) => {
      expect(method).toBe("SetBusServoPulse");
      expect(params).toEqual([500, 2, 1, 400, 2, 600]);
      return [true, {}, "SetBusServoPulse"];
    });
    closers.push(server.close);
    const client = new TonyPiRpcClient({ url: server.url });
    await client.setBusServoPulse(500, [
      { id: 1, pulse: 400 },
      { id: 2, pulse: 600 },
    ]);
  });

  it("surfaces robot [false, err] as TonyPiRpcError", async () => {
    const server = await startRpcServer(() => [false, "E01", "RunAction"]);
    closers.push(server.close);
    const client = new TonyPiRpcClient({ url: server.url });
    await expect(client.runAction("wave", 1)).rejects.toBeInstanceOf(
      TonyPiRpcError,
    );
  });
});
