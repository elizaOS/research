import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { stopRobot } from "./_helpers";

export const stopAction: Action = {
  name: "TONYPI_STOP",
  similes: ["STOP", "HALT", "FREEZE", "EMERGENCY_STOP"],
  description:
    "Stop TonyPi: StopBusServo + StopActionGroup + RunAction('0') against the stock RPCServer.",
  examples: [],
  validate: async (_runtime: IAgentRuntime, _message: Memory) => true,
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    _options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    return stopRobot(runtime, callback);
  },
};
