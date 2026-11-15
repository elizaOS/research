import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { runAction } from "./_helpers";

export const bowAction: Action = {
  name: "TONYPI_BOW",
  similes: ["BOW", "BOW_DOWN"],
  description: "Play TonyPi `bow` once via RunAction.",
  examples: [],
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    _options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    return runAction(runtime, callback, "bow", 1, "TonyPi is bowing.", "bow", {
      runningHint: false,
    });
  },
};
