import type { Plugin } from "@elizaos/core";
import { actions } from "./actions";
import { providers } from "./providers";
import { TonyPiService } from "./service";

export { actions } from "./actions";
export {
  BruhOsClient,
  BruhOsError,
  bruhOsUrlFromHost,
  DEFAULT_BRUHOS_PORT,
  redactedBruhOsUrl,
} from "./bruhos-client";
export {
  BRUHOS_SKILLS,
  emptyCapabilities,
  enhancementLevel,
} from "./capabilities";
export {
  listProModes,
  PRO_MODES,
  proModeById,
  resolveProModeId,
} from "./pro-modes";
export { providers } from "./providers";
export {
  DEFAULT_AP_HOST,
  DEFAULT_RPC_PORT,
  redactedRpcUrl,
  rpcUrlFromHost,
  TonyPiRpcClient,
  TonyPiRpcError,
} from "./rpc-client";
export { TonyPiService } from "./service";
export * from "./types";

export const tonyPiPlugin: Plugin = {
  name: "tonypi",
  description:
    "Drives Hiwonder TonyPi / TonyPi Pro via stock RPCServer :9030, with optional bruhOS skill enhancement.",
  services: [TonyPiService],
  providers,
  actions,
  autoEnable: {
    shouldEnable: (env, config) => {
      if (
        env?.ELIZA_TONYPI_RPC_URL ||
        env?.ELIZA_TONYPI_HOST ||
        env?.ELIZA_TONYPI_BRIDGE_URL ||
        env?.ELIZA_BRUHOS_URL ||
        env?.ELIZA_BRUHOS_HOST
      ) {
        return true;
      }
      const features = config?.features as Record<string, unknown> | undefined;
      const tonyPi = features?.tonypi;
      if (tonyPi === true) return true;
      if (
        typeof tonyPi === "object" &&
        tonyPi !== null &&
        (tonyPi as { enabled?: unknown }).enabled !== false
      ) {
        return Boolean((tonyPi as { enabled?: unknown }).enabled);
      }
      return false;
    },
  },
  init: async (_config, _runtime) => {},
  async dispose(runtime) {
    const svc = runtime.getService<TonyPiService>(TonyPiService.serviceType);
    await svc?.stop();
  },
};

export default tonyPiPlugin;
