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

export const turnRightAction: Action = {
  name: "TONYPI_TURN_RIGHT",
  similes: ["TURN_RIGHT", "ROTATE_RIGHT", "SPIN_RIGHT"],
  description: "Loop TonyPi `turn_right` via RunAction until TONYPI_STOP.",
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
      WALK_GROUPS.turnRight,
      0,
      "TonyPi is turning right (looping turn_right).",
      "turn right",
    );
  },
};
