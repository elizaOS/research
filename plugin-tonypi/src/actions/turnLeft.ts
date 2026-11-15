import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { WALK_GROUPS } from "../types";
import { runAction } from "./_helpers";

export const turnLeftAction: Action = {
  name: "TONYPI_TURN_LEFT",
  similes: ["TURN_LEFT", "ROTATE_LEFT", "SPIN_LEFT"],
  description: "Loop TonyPi `turn_left` via RunAction until TONYPI_STOP.",
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
      WALK_GROUPS.turnLeft,
      0,
      "TonyPi is turning left (looping turn_left).",
      "turn left",
    );
  },
};
