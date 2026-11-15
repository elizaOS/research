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

export const sideStepRightAction: Action = {
  name: "TONYPI_SIDE_STEP_RIGHT",
  similes: ["SIDE_STEP_RIGHT", "STRAFE_RIGHT", "STEP_RIGHT"],
  description: "Loop TonyPi `right_move_fast` via RunAction until TONYPI_STOP.",
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
      WALK_GROUPS.sideRight,
      0,
      "TonyPi is side-stepping right (looping right_move_fast).",
      "side step right",
    );
  },
};
