import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { proModeById, resolveProModeId } from "../pro-modes";
import {
  getClient,
  getNumberOption,
  getService,
  getStringOption,
  notConnected,
} from "./_helpers";

/**
 * Progressive Pro: start an on-robot Function (vision / voice / AI).
 * Voice Control = 14, Smart Chat = 13 — stock RPC LoadFunc, no bruhOS required.
 */
export const loadModeAction: Action = {
  name: "TONYPI_LOAD_MODE",
  similes: [
    "LOAD_MODE",
    "START_VOICE",
    "VOICE_CONTROL",
    "SMART_CHAT",
    "LOAD_FUNC",
  ],
  description:
    "Load a TonyPi Pro on-robot mode via LoadFunc. Options: mode (id or slug, e.g. 14 / voice_control / smart_chat). AI/voice modes need Wi-Fi + LLM key on the robot.",
  examples: [],
  validate: async (runtime: IAgentRuntime, _message: Memory) => {
    return getService(runtime)?.isConnected() ?? false;
  },
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    const service = getService(runtime);
    const client = getClient(runtime);
    if (!client || !service?.isConnected()) {
      return notConnected(callback, "load mode");
    }
    const modeOpt = options?.mode ?? options?.id ?? options?.name;
    if (modeOpt === undefined || modeOpt === null || modeOpt === "") {
      const text =
        "TONYPI_LOAD_MODE requires options.mode (e.g. 14, 'voice_control', 'smart_chat').";
      await callback?.({ text });
      return { success: false, text };
    }
    let id: number;
    try {
      id = resolveProModeId(
        typeof modeOpt === "number" ? modeOpt : String(modeOpt),
      );
    } catch (err) {
      const text = (err as Error).message;
      await callback?.({ text });
      return { success: false, text };
    }
    const def = proModeById(id);
    try {
      await client.loadFunc(id);
      // WonderPi / bruhOS app usually StartFunc after LoadFunc for interactive modes.
      const autoStart = options?.start !== false;
      if (autoStart) {
        try {
          await client.startFunc();
        } catch {
          // Some images start on LoadFunc alone.
        }
      }
      if (def?.pick) {
        const x = getNumberOption(options, "x", Number.NaN);
        const y = getNumberOption(options, "y", Number.NaN);
        if (Number.isFinite(x) && Number.isFinite(y)) {
          await client.setPoint(id, x, y);
        }
      }
    } catch (err) {
      const text = `TonyPi load mode failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
    const label = def?.title ?? String(id);
    service.noteAction(`mode:${id}`, 1, true);
    const hint =
      def?.group === "ai"
        ? " (AI/voice: robot needs Wi-Fi + LLM key; mic may be the Pro voice module)"
        : "";
    const text = `TonyPi loaded mode ${id} (${label})${hint}.`;
    await callback?.({ text });
    return {
      success: true,
      text,
      data: { id, title: label, group: def?.group },
    };
  },
};

export const unloadModeAction: Action = {
  name: "TONYPI_UNLOAD_MODE",
  similes: ["UNLOAD_MODE", "STOP_MODE", "UNLOAD_FUNC", "STOP_VOICE"],
  description:
    "Stop and unload the current TonyPi Pro Function (StopFunc + UnloadFunc).",
  examples: [],
  validate: async (runtime: IAgentRuntime, _message: Memory) => {
    return getService(runtime)?.isConnected() ?? false;
  },
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    _options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    const service = getService(runtime);
    const client = getClient(runtime);
    if (!client || !service?.isConnected()) {
      return notConnected(callback, "unload mode");
    }
    try {
      try {
        await client.stopFunc();
      } catch {
        // ignore if nothing running
      }
      await client.unloadFunc();
    } catch (err) {
      const text = `TonyPi unload mode failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
    service.noteAction("mode:0", 1, false);
    const text = "TonyPi mode unloaded.";
    await callback?.({ text });
    return { success: true, text };
  },
};

/** Convenience: resolve slug helpers for tests / callers. */
export function modeSlugFromOptions(
  options: Record<string, unknown> | undefined,
): string {
  return getStringOption(options, "mode", "");
}
