import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import {
  getClient,
  getNumberOption,
  getService,
  notConnected,
} from "./_helpers";

/** Head look via SetPWMServo — progressive Pro/stock feature, not bruhOS. */
export const setHeadAction: Action = {
  name: "TONYPI_SET_HEAD",
  similes: ["LOOK", "TURN_HEAD", "SET_HEAD", "HEAD_LOOK"],
  description:
    "Point TonyPi head via SetPWMServo (pitch + yaw). Options: pitch, yaw, duration_ms (default 120). Stock RPC; no bruhOS required.",
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
      return notConnected(callback, "set head");
    }
    const pitch = getNumberOption(options, "pitch", Number.NaN);
    const yaw = getNumberOption(options, "yaw", Number.NaN);
    if (!Number.isFinite(pitch) || !Number.isFinite(yaw)) {
      const text = "TONYPI_SET_HEAD requires options.pitch and options.yaw.";
      await callback?.({ text });
      return { success: false, text };
    }
    const duration_ms = getNumberOption(options, "duration_ms", 120);
    try {
      await client.setHead(pitch, yaw, duration_ms);
    } catch (err) {
      const text = `TonyPi set head failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
    service.noteAction("set_head", 1, false);
    const text = `TonyPi head set (pitch=${pitch}, yaw=${yaw}).`;
    await callback?.({ text });
    return { success: true, text, data: { pitch, yaw } };
  },
};
