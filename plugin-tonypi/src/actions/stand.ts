import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { runAction } from "./_helpers";

export const standAction: Action = {
  name: "TONYPI_STAND",
  similes: ["STAND", "STAND_UP", "GET_UP"],
  description: "Play TonyPi `stand` once via RunAction.",
  examples: [],
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    _options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    return runAction(
      runtime,
      callback,
      "stand",
      1,
      "TonyPi is standing.",
      "stand",
      { runningHint: false },
    );
  },
};
