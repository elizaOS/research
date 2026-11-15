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

interface ServoTarget {
  id: number;
  pulse: number;
}

function _parsePositions(
  options: Record<string, unknown> | undefined,
): ServoTarget[] | null {
  const positions = options?.positions;
  if (Array.isArray(positions)) {
    const out: ServoTarget[] = [];
    for (const item of positions) {
      if (!item || typeof item !== "object") continue;
      const row = item as { id?: unknown; pulse?: unknown; position?: unknown };
      const pulse = row.pulse ?? row.position;
      if (typeof row.id === "number" && typeof pulse === "number") {
        out.push({ id: row.id, pulse });
      }
    }
    return out.length === 0 ? null : out;
  }
  const id = options?.id;
  const pulse = options?.pulse ?? options?.position;
  if (typeof id === "number" && typeof pulse === "number") {
    return [{ id, pulse }];
  }
  return null;
}

export const setServoAction: Action = {
  name: "TONYPI_SET_SERVO",
  similes: ["SET_SERVO", "MOVE_SERVO", "MOVE_JOINT", "SET_JOINT"],
  description:
    "Drive TonyPi bus servos via SetBusServoPulse. Options: positions=[{id,pulse}], duration_ms (default 500). Ids 1..16.",
  examples: [],
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    const positions = _parsePositions(options);
    if (positions === null) {
      const text =
        "TONYPI_SET_SERVO requires options.positions=[{id,pulse}] or options.id+options.pulse.";
      await callback?.({ text });
      return { success: false, text };
    }
    const service = getService(runtime);
    const client = getClient(runtime);
    if (!client || !service?.isConnected()) {
      return notConnected(callback, "set servo");
    }
    const duration_ms = getNumberOption(options, "duration_ms", 500);
    try {
      await client.setBusServoPulse(duration_ms, positions);
    } catch (err) {
      const text = `TonyPi set servo failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
    const text = `TonyPi servos updated (${positions.length} joints).`;
    await callback?.({ text });
    return { success: true, text };
  },
};
