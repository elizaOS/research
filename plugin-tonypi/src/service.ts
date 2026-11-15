// Owns stock RPC + optional bruhOS clients; caches last-command / capabilities.

import {
  type IAgentRuntime,
  logger,
  Service,
  type ServiceTypeName,
} from "@elizaos/core";
import {
  BruhOsClient,
  bruhOsUrlFromHost,
  redactedBruhOsUrl,
} from "./bruhos-client";
import {
  emptyCapabilities,
  enhancementLevel,
  type TonyPiCapabilities,
} from "./capabilities";
import { redactedRpcUrl, rpcUrlFromHost, TonyPiRpcClient } from "./rpc-client";
import type { TonyPiStateSnapshot } from "./types";

export class TonyPiService extends Service {
  static override serviceType: ServiceTypeName = "tonypi" as ServiceTypeName;
  override capabilityDescription =
    "Drives a Hiwonder TonyPi / TonyPi Pro via stock RPCServer :9030, with optional bruhOS skill enhancement.";

  private client: TonyPiRpcClient | null = null;
  private bruhos: BruhOsClient | null = null;
  private connected = false;
  private caps: TonyPiCapabilities = emptyCapabilities();
  private state: TonyPiStateSnapshot = {
    connected: false,
    lastAction: "",
    lastTimes: 0,
    runningHint: false,
    receivedAt: 0,
    enhancement: "stock",
    bruhos: false,
    pro: false,
  };

  static async start(runtime: IAgentRuntime): Promise<TonyPiService> {
    const service = new TonyPiService(runtime);
    await service._tryConnect();
    return service;
  }

  getClient(): TonyPiRpcClient | null {
    return this.client;
  }

  /** @deprecated Use getClient(). */
  getBridge(): TonyPiRpcClient | null {
    return this.client;
  }

  getBruhOs(): BruhOsClient | null {
    return this.bruhos;
  }

  getCapabilities(): TonyPiCapabilities {
    return { ...this.caps, bruhosSkills: [...this.caps.bruhosSkills] };
  }

  getState(): TonyPiStateSnapshot {
    return {
      ...this.state,
      connected: this.connected,
      enhancement: enhancementLevel(this.caps),
      bruhos: this.caps.bruhos,
      pro: this.caps.pro,
    };
  }

  isConnected(): boolean {
    return this.connected && this.client !== null;
  }

  /** True when the optional bruhOS skill API was probed successfully. */
  hasBruhOs(): boolean {
    return this.caps.bruhos && this.bruhos !== null;
  }

  noteAction(name: string, times: number, runningHint: boolean): void {
    this.state = {
      ...this.getState(),
      lastAction: name,
      lastTimes: times,
      runningHint,
      receivedAt: Date.now(),
    };
  }

  async stop(): Promise<void> {
    this.client = null;
    this.bruhos = null;
    this.connected = false;
    this.caps = emptyCapabilities();
    this.state = {
      ...this.state,
      connected: false,
      runningHint: false,
      enhancement: "stock",
      bruhos: false,
      pro: false,
      receivedAt: Date.now(),
    };
  }

  private _readSetting(key: string): string | null {
    const raw = this.runtime.getSetting(key);
    if (raw === null || raw === undefined) return null;
    if (typeof raw === "string") return raw === "" ? null : raw;
    return String(raw);
  }

  private _resolveRpcUrl(): string | null {
    const url =
      this._readSetting("ELIZA_TONYPI_RPC_URL") ??
      this._readSetting("ELIZA_TONYPI_BRIDGE_URL");
    if (url) {
      // Legacy ws:// sidecar URLs are no longer supported.
      if (/^wss?:\/\//i.test(url)) {
        throw new Error(
          "ELIZA_TONYPI_BRIDGE_URL/RPC_URL must be http(s):// to the stock RPCServer (port 9030), not ws://",
        );
      }
      return url;
    }
    const host = this._readSetting("ELIZA_TONYPI_HOST");
    if (host) return rpcUrlFromHost(host);
    return null;
  }

  private _resolveBruhOsUrl(): string | null {
    const url = this._readSetting("ELIZA_BRUHOS_URL");
    if (url) {
      if (/^wss?:\/\//i.test(url)) {
        throw new Error("ELIZA_BRUHOS_URL must be http(s)://, not ws://");
      }
      return url;
    }
    // Optional: same host as TonyPi, default bruhOS port.
    const host =
      this._readSetting("ELIZA_BRUHOS_HOST") ??
      this._readSetting("ELIZA_TONYPI_HOST");
    const auto = this._readSetting("ELIZA_BRUHOS_AUTO");
    if (host && (auto === "1" || auto === "true")) {
      return bruhOsUrlFromHost(host);
    }
    return null;
  }

  private async _tryConnect(): Promise<void> {
    await this._connectStockRpc();
    await this._connectBruhOs();
  }

  private async _connectStockRpc(): Promise<void> {
    let rawUrl: string | null;
    try {
      rawUrl = this._resolveRpcUrl();
    } catch (err) {
      logger.warn(`[TonyPiService] ${(err as Error).message}`);
      return;
    }
    if (!rawUrl) {
      logger.info(
        "[TonyPiService] no ELIZA_TONYPI_RPC_URL / ELIZA_TONYPI_HOST; stock RPC stays disconnected (bruhOS-only still possible)",
      );
      return;
    }
    const client = new TonyPiRpcClient({ url: rawUrl });
    const safeUrl = redactedRpcUrl(client.url);
    this.client = client;
    try {
      await client.heartbeat();
      this.connected = true;
      this.caps = { ...this.caps, stock: true };
      const pro = await client.probeProFeatures();
      this.caps = { ...this.caps, pro };
      this.state = {
        ...this.state,
        connected: true,
        enhancement: enhancementLevel(this.caps),
        pro,
        receivedAt: Date.now(),
      };
      logger.info(
        `[TonyPiService] stock RPC at ${safeUrl} (enhancement=${enhancementLevel(this.caps)})`,
      );
    } catch (err) {
      this.connected = false;
      this.caps = { ...this.caps, stock: false, pro: false };
      logger.warn(
        `[TonyPiService] RPC unreachable at ${safeUrl}: ${(err as Error).message}`,
      );
    }
  }

  private async _connectBruhOs(): Promise<void> {
    let rawUrl: string | null;
    try {
      rawUrl = this._resolveBruhOsUrl();
    } catch (err) {
      logger.warn(`[TonyPiService] ${(err as Error).message}`);
      return;
    }
    if (!rawUrl) {
      // Progressive enhancement: absence is normal on stock TonyPi / Pro.
      return;
    }
    const client = new BruhOsClient({ url: rawUrl });
    const safeUrl = redactedBruhOsUrl(client.url);
    try {
      const caps = await client.capabilities();
      this.bruhos = client;
      this.caps = {
        ...this.caps,
        bruhos: true,
        bruhosVersion: caps.version,
        bruhosSkills: caps.skills,
        body: caps.body === "unknown" ? this.caps.body : caps.body,
      };
      this.state = {
        ...this.state,
        enhancement: enhancementLevel(this.caps),
        bruhos: true,
        receivedAt: Date.now(),
      };
      logger.info(
        `[TonyPiService] bruhOS enhancement at ${safeUrl} (v=${caps.version || "?"}, skills=${caps.skills.length})`,
      );
    } catch (err) {
      this.bruhos = null;
      this.caps = {
        ...this.caps,
        bruhos: false,
        bruhosVersion: "",
        bruhosSkills: [],
      };
      logger.info(
        `[TonyPiService] bruhOS not available at ${safeUrl} (${(err as Error).message}); continuing on stock RPC only`,
      );
    }
  }
}
