import type {
  IAgentRuntime,
  Memory,
  Provider,
  ProviderResult,
  State,
} from "@elizaos/core";
import { listProModes } from "../pro-modes";
import { TonyPiService } from "../service";

function _disconnectedResult(): ProviderResult {
  return {
    text: "(tonypi not connected)",
    values: { tonyPiConnected: false, tonyPiBruhOs: false },
    data: {},
  };
}

export const robotStateProvider: Provider = {
  name: "TONYPI_ROBOT_STATE",
  description:
    "TonyPi stock RPC reachability, optional bruhOS enhancement, and last commanded action.",
  get: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State,
  ): Promise<ProviderResult> => {
    const service = runtime.getService<TonyPiService>(
      TonyPiService.serviceType,
    );
    if (!service) return _disconnectedResult();

    const caps = service.getCapabilities();
    const snapshot = service.getState();
    const stockUp = service.isConnected();
    const bruhosUp = service.hasBruhOs();

    if (!stockUp && !bruhosUp) return _disconnectedResult();

    const skillNames = caps.bruhosSkills.map((s) => s.skill);
    const aiModes = listProModes("ai")
      .map((m) => `${m.id}:${m.title}`)
      .join(", ");
    const lines = [
      `stock_rpc: ${stockUp ? "yes" : "no"}`,
      `enhancement: ${snapshot.enhancement}`,
      `pro_rpc: ${caps.pro ? "yes" : "no"}`,
      `bruhos: ${bruhosUp ? "yes" : "no"}`,
      caps.bruhosVersion ? `bruhos_version: ${caps.bruhosVersion}` : null,
      caps.body !== "unknown" ? `body: ${caps.body}` : null,
      stockUp
        ? `pro_ai_modes: ${aiModes} (TONYPI_LOAD_MODE; voice=14 chat=13)`
        : null,
      skillNames.length > 0 ? `bruhos_skills: ${skillNames.join(", ")}` : null,
      `running_hint: ${snapshot.runningHint ? "yes" : "no"}`,
      `last_action: ${snapshot.lastAction === "" ? "(none)" : snapshot.lastAction}`,
      `last_times: ${snapshot.lastTimes}`,
    ].filter((line): line is string => line !== null);

    return {
      text: `TonyPi robot state:\n${lines.join("\n")}`,
      values: {
        tonyPiConnected: stockUp,
        tonyPiRunning: snapshot.runningHint,
        tonyPiAction: snapshot.lastAction,
        tonyPiEnhancement: snapshot.enhancement,
        tonyPiPro: caps.pro,
        tonyPiBruhOs: bruhosUp,
        tonyPiBody: caps.body,
      },
      data: { state: snapshot, capabilities: caps },
    };
  },
};
