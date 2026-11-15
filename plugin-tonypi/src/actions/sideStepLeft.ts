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

export const sideStepLeftAction: Action = {
  name: "TONYPI_SIDE_STEP_LEFT",
  similes: ["SIDE_STEP_LEFT", "STRAFE_LEFT", "STEP_LEFT"],
  description: "Loop TonyPi `left_move_fast` via RunAction until TONYPI_STOP.",
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
      WALK_GROUPS.sideLeft,
      0,
      "TonyPi is side-stepping left (looping left_move_fast).",
      "side step left",
    );
  },
};
