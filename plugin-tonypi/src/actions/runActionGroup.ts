import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { getNumberOption, getStringOption, runAction } from "./_helpers";

export const runActionGroupAction: Action = {
  name: "TONYPI_RUN_ACTION_GROUP",
  similes: ["RUN_ACTION_GROUP", "PLAY_ACTION", "PLAY_ACTION_GROUP"],
  description:
    "Play an allowlisted TonyPi action group via RunAction. Options: name (required), times (0 loops; default 1).",
  examples: [],
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    const name = getStringOption(options, "name", "");
    if (name === "") {
      const text =
        "TONYPI_RUN_ACTION_GROUP requires options.name (e.g. 'wave').";
      await callback?.({ text });
      return { success: false, text };
    }
    const times = getNumberOption(options, "times", 1);
    return runAction(
      runtime,
      callback,
      name,
      times,
      `TonyPi is playing action group: ${name} (times=${times}).`,
      `run action group '${name}'`,
      { runningHint: times === 0 },
    );
  },
};
