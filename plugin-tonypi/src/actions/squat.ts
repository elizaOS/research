import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { runAction } from "./_helpers";

export const squatAction: Action = {
  name: "TONYPI_SQUAT",
  similes: ["SQUAT", "CROUCH", "SIT"],
  description:
    "Play TonyPi `squat` once. There is no stock `sit` group; `sit_ups` is different.",
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
      "squat",
      1,
      "TonyPi is squatting.",
      "squat",
      { runningHint: false },
    );
  },
};
