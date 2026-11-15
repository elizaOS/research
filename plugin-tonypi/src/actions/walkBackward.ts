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

export const walkBackwardAction: Action = {
  name: "TONYPI_WALK_BACKWARD",
  similes: ["WALK_BACKWARD", "MOVE_BACKWARD", "GO_BACKWARD"],
  description:
    "Loop TonyPi `back_fast` (ActionGroupDict code 2) via RunAction until TONYPI_STOP.",
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
      WALK_GROUPS.back,
      0,
      "TonyPi is walking backward (looping back_fast).",
      "walk backward",
    );
  },
};
