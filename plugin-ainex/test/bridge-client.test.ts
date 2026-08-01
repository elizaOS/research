import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { WebSocketServer, type WebSocket as WsWebSocket } from "ws";
import { AinexBridgeClient, redactedBridgeUrl } from "../src/bridge-client";

interface BridgeServerHarness {
  url: string;
  close: () => Promise<void>;
  emitEvent: (event: string, data: Record<string, unknown>) => void;
  /** Authorization headers observed during websocket upgrades. */
  authorizationHeaders: Array<string | undefined>;
  /** Set the handler used to reply to incoming CommandEnvelopes. */
  onCommand: (
    fn: (
      command: string,
      payload: Record<string, unknown>,
      request_id: string,
    ) => {
      ok: boolean;
      message?: string;
      data?: Record<string, unknown>;
    } | null,
  ) => void;
  /** All commands the test server has received this run. */
  received: Array<{
    command: string;
    payload: Record<string, unknown>;
    preempt: boolean;
  }>;
}

async function startBridgeServer(): Promise<BridgeServerHarness> {
  const wss = new WebSocketServer({ host: "127.0.0.1", port: 0 });
  await new Promise<void>((resolve) => wss.once("listening", () => resolve()));
  const { port } = wss.address() as AddressInfo;
  const url = `ws://127.0.0.1:${port}`;
  const sockets: WsWebSocket[] = [];
  const received: BridgeServerHarness["received"] = [];
  const authorizationHeaders: Array<string | undefined> = [];

  let handler: BridgeServerHarness extends {
    onCommand: (fn: infer H) => unknown;
  }
    ? H
    : never = (() => ({ ok: true, message: "ok", data: {} })) as never;

  wss.on("connection", (socket, request) => {
    sockets.push(socket);
    authorizationHeaders.push(request.headers.authorization);
    socket.send(
      JSON.stringify({
        type: "event",
        event: "session.hello",
        timestamp: new Date().toISOString(),
        backend: "test",
        data: { capabilities: { walk_set: true } },
      }),
    );
    socket.on("message", (raw) => {
      const parsed = JSON.parse(raw.toString());
      if (parsed.type !== "command") return;
      received.push({
        command: parsed.command,
        payload: parsed.payload,
        preempt: parsed.preempt === true,
      });
      const reply = handler(parsed.command, parsed.payload, parsed.request_id);
      if (reply === null) {
        // Drop the response on the floor — used to test client-side timeouts.
        return;
      }
      socket.send(
        JSON.stringify({
          type: "response",
          request_id: parsed.request_id,
          timestamp: new Date().toISOString(),
          ok: reply.ok,
          backend: "test",
          message: reply.message ?? (reply.ok ? "ok" : "error"),
          data: reply.data ?? {},
        }),
      );
    });
  });

  return {
    url,
    received,
    authorizationHeaders,
    onCommand(fn) {
      handler = fn as typeof handler;
    },
    emitEvent(event, data) {
      const payload = JSON.stringify({
        type: "event",
        event,
        timestamp: new Date().toISOString(),
        backend: "test",
        data,
      });
      for (const s of sockets) s.send(payload);
    },
    async close() {
      for (const s of sockets) s.close();
      await new Promise<void>((resolve, reject) =>
        wss.close((err) => (err ? reject(err) : resolve())),
      );
    },
  };
}

describe("AinexBridgeClient", () => {
  let harness: BridgeServerHarness;

  beforeEach(async () => {
    harness = await startBridgeServer();
  });

  afterEach(async () => {
    await harness.close();
  });

  it("connects, sends a walk.set command, and resolves the response", async () => {
    const client = new AinexBridgeClient({ url: harness.url });
    await client.connect();
    expect(client.isConnected()).toBe(true);

    harness.onCommand((cmd, payload) => {
      if (cmd === "walk.set") {
        return { ok: true, message: "ok", data: { speed: payload.speed } };
      }
      return { ok: false, message: "unsupported" };
    });

    const response = await client.send("walk.set", {
      speed: 2,
      height: 0.036,
      x: 0.04,
      y: 0,
      yaw: 0,
    });

    expect(response.ok).toBe(true);
    expect(response.data.speed).toBe(2);
    expect(harness.received).toHaveLength(1);
    expect(harness.received[0]?.command).toBe("walk.set");
    expect(harness.received[0]?.payload.x).toBe(0.04);

    await client.disconnect();
  });

  it("sends a configured bridge token only in the Authorization header", async () => {
    const token = "bridge-test-token-that-is-at-least-32-chars";
    const client = new AinexBridgeClient({
      url: harness.url,
      authToken: token,
    });

    await client.connect();

    expect(harness.authorizationHeaders).toEqual([`Bearer ${token}`]);
    expect(client.url).not.toContain(token);
    await client.disconnect();
  });

  it("omits Authorization when no bridge token is configured", async () => {
    const client = new AinexBridgeClient({ url: harness.url });

    await client.connect();

    expect(harness.authorizationHeaders).toEqual([undefined]);
    await client.disconnect();
  });

  it("rejects auth tokens that cannot be represented as one safe header value", () => {
    expect(
      () =>
        new AinexBridgeClient({
          url: harness.url,
          authToken: "safe-prefix\r\nX-Injected: true",
        }),
    ).toThrow(/visible ASCII/);
  });

  it("enforces the physical bridge bearer length contract", () => {
    expect(
      () =>
        new AinexBridgeClient({
          url: harness.url,
          authToken: "too-short",
        }),
    ).toThrow(/32\.\.4096 visible ASCII/);
    expect(
      () =>
        new AinexBridgeClient({
          url: harness.url,
          authToken: "x".repeat(4097),
        }),
    ).toThrow(/32\.\.4096 visible ASCII/);
  });

  it("rejects bearer auth over a non-loopback plaintext websocket", () => {
    expect(
      () =>
        new AinexBridgeClient({
          url: "ws://192.0.2.10:9100",
          authToken: "bridge-test-token-that-is-at-least-32-chars",
        }),
    ).toThrow(/wss:\/\/ or a loopback ws:\/\//);
  });

  it("accepts bearer auth over wss and loopback IPv4 or IPv6", () => {
    const token = "bridge-test-token-that-is-at-least-32-chars";
    expect(
      new AinexBridgeClient({
        url: "wss://bridge.example.test",
        authToken: token,
      }),
    ).toBeDefined();
    expect(
      new AinexBridgeClient({ url: "ws://127.1.2.3:9100", authToken: token }),
    ).toBeDefined();
    expect(
      new AinexBridgeClient({ url: "ws://[::1]:9100", authToken: token }),
    ).toBeDefined();
  });

  it("rejects credentials embedded in websocket URL userinfo", () => {
    expect(
      () => new AinexBridgeClient({ url: "ws://user:secret@127.0.0.1:9100" }),
    ).toThrow(/must not be placed in URL userinfo/);
  });

  it("rejects query, fragment, and nonroot path channels for bearer auth", () => {
    const token = "bridge-test-token-that-is-at-least-32-chars";
    for (const url of [
      "ws://127.0.0.1:9100/?token=LEAKME_7f91",
      "ws://127.0.0.1:9100/#LEAKME_7f91",
      "ws://127.0.0.1:9100/LEAKME_7f91",
    ]) {
      expect(() => new AinexBridgeClient({ url, authToken: token })).toThrow(
        /query or fragment|root path/,
      );
    }
  });

  it("redacts URL credentials and components from diagnostics", async () => {
    const unsafe =
      "ws://operator:LEAKME_7f91@127.0.0.1:9100/LEAKME_7f91?token=LEAKME_7f91#LEAKME_7f91";
    expect(redactedBridgeUrl(unsafe)).toBe("ws://127.0.0.1:9100");

    const client = new AinexBridgeClient({
      url: "ws://127.0.0.1:9100/LEAKME_7f91?token=LEAKME_7f91",
    });
    await expect(client.send("profile.describe")).rejects.not.toThrow(
      /LEAKME_7f91/,
    );
  });

  it("dispatches event envelopes to per-event handlers", async () => {
    const client = new AinexBridgeClient({ url: harness.url });
    await client.connect();
    const events: Array<{ battery_mv: number }> = [];
    client.on("telemetry.basic", (env) => {
      events.push({ battery_mv: env.data.battery_mv as number });
    });

    harness.emitEvent("telemetry.basic", { battery_mv: 12000 });
    await new Promise((r) => setTimeout(r, 20));
    expect(events).toEqual([{ battery_mv: 12000 }]);

    await client.disconnect();
  });

  it("rejects sends that exceed the timeout window", async () => {
    const client = new AinexBridgeClient({
      url: harness.url,
      sendTimeoutMs: 80,
    });
    await client.connect();

    // Tell the harness to drop responses (return null) so the client's
    // sendTimeoutMs window expires.
    harness.onCommand(() => null);

    await expect(
      client.send("walk.set", { speed: 2, height: 0.036, x: 0, y: 0, yaw: 0 }),
    ).rejects.toThrow(/timeout/i);
    await client.disconnect();
  });

  it("rejects sends when disconnected", async () => {
    const client = new AinexBridgeClient({ url: harness.url });
    await client.connect();
    await client.disconnect();
    await expect(
      client.send("walk.command", { action: "stop" }),
    ).rejects.toThrow(/not connected/i);
  });
});
