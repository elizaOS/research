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

export const walkForwardAction: Action = {
  name: "TONYPI_WALK_FORWARD",
  similes: ["WALK_FORWARD", "MOVE_FORWARD", "GO_FORWARD"],
  description:
    "Loop TonyPi `go_forward` via stock RunAction until TONYPI_STOP. Recorded gait, not a velocity controller.",
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
      WALK_GROUPS.forward,
      0,
      "TonyPi is walking forward (looping go_forward).",
      "walk forward",
    );
  },
};
