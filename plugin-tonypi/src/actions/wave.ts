import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { runAction } from "./_helpers";

export const waveAction: Action = {
  name: "TONYPI_WAVE",
  similes: ["WAVE", "WAVE_HAND", "GREET", "SAY_HI"],
  description: "Play TonyPi `wave` once via RunAction.",
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
      "wave",
      1,
      "TonyPi is waving.",
      "wave",
      { runningHint: false },
    );
  },
};
