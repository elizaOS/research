import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { getClient, getService, notConnected } from "./_helpers";

/** Fall recovery via stock RPC `StandUp` (same as bruhos-app). Works without bruhOS. */
export const standUpAction: Action = {
  name: "TONYPI_STAND_UP",
  similes: ["GET_UP_FROM_FALL", "RECOVER_FALL", "STANDUP"],
  description:
    "Fall recovery via stock RPC StandUp (TonyPi / Pro). Does not require bruhOS.",
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
      return notConnected(callback, "stand up");
    }
    try {
      await client.standUp();
    } catch (err) {
      const text = `TonyPi stand up failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
    service.noteAction("stand_up", 1, false);
    const text = "TonyPi stood up.";
    await callback?.({ text });
    return { success: true, text };
  },
};
