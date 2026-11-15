// Shared utilities for plugin-tonypi actions (stock RPCServer JSON-RPC).

import type {
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
} from "@elizaos/core";
import type { TonyPiRpcClient } from "../rpc-client";
import { TonyPiService } from "../service";
import { assertSafeActionName } from "../types";

export function getService(runtime: IAgentRuntime): TonyPiService | null {
  return runtime.getService<TonyPiService>(TonyPiService.serviceType) ?? null;
}

export function getClient(runtime: IAgentRuntime): TonyPiRpcClient | null {
  return getService(runtime)?.getClient() ?? null;
}

export async function notConnected(
  callback: HandlerCallback | undefined,
  context: string,
): Promise<ActionResult> {
  const text = `TonyPi RPC is not connected (${context}); set ELIZA_TONYPI_RPC_URL=http://<robot>:9030 (stock RPCServer) and retry.`;
  await callback?.({ text });
  return { success: false, text };
}

export async function runAction(
  runtime: IAgentRuntime,
  callback: HandlerCallback | undefined,
  name: string,
  times: number,
  okText: string,
  context: string,
  opts?: { runningHint?: boolean },
): Promise<ActionResult> {
  const runningHint = opts?.runningHint ?? times === 0;
  const service = getService(runtime);
  const client = service?.getClient() ?? null;
  if (!client || !service?.isConnected()) {
    return notConnected(callback, context);
  }
  let safe: string;
  try {
    safe = assertSafeActionName(name);
  } catch (err) {
    const text = (err as Error).message;
    await callback?.({ text });
    return { success: false, text };
  }
  try {
    await client.runAction(safe, times);
  } catch (err) {
    const text = `TonyPi ${context} failed: ${(err as Error).message}`;
    await callback?.({ text });
    return { success: false, text };
  }
  service.noteAction(safe, times, runningHint);
  await callback?.({ text: okText });
  return { success: true, text: okText, data: { name: safe, times } };
}

export async function stopRobot(
  runtime: IAgentRuntime,
  callback: HandlerCallback | undefined,
): Promise<ActionResult> {
  const service = getService(runtime);
  const client = service?.getClient() ?? null;
  if (!client || !service?.isConnected()) {
    return notConnected(callback, "stop");
  }
  try {
    await client.stopAll();
  } catch (err) {
    const text = `TonyPi stop failed: ${(err as Error).message}`;
    await callback?.({ text });
    return { success: false, text };
  }
  service.noteAction("0", 1, false);
  const text = "TonyPi stopped.";
  await callback?.({ text });
  return { success: true, text };
}

export function getNumberOption(
  options: Record<string, unknown> | undefined,
  key: string,
  fallback: number,
): number {
  const value = options?.[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function getStringOption(
  options: Record<string, unknown> | undefined,
  key: string,
  fallback: string,
): string {
  const value = options?.[key];
  return typeof value === "string" && value !== "" ? value : fallback;
}
