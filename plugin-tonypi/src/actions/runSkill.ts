import type {
  Action,
  ActionResult,
  HandlerCallback,
  IAgentRuntime,
  Memory,
  State,
} from "@elizaos/core";
import { assertSafeBruhOsSkill } from "../capabilities";
import { getService, getStringOption } from "./_helpers";

/**
 * Progressive enhancement: run a bruhOS skill when the optional skill API is up.
 * Stock TonyPi / Pro without bruhOS: validate returns false (action stays hidden).
 */
export const runSkillAction: Action = {
  name: "TONYPI_RUN_SKILL",
  similes: [
    "BRUHOS_RUN_SKILL",
    "RUN_SKILL",
    "DETECT",
    "WALK_TOWARD",
    "GRAB",
    "FOLLOW_LINE",
  ],
  description:
    "Run a bruhOS skill (detect, walk_toward, grab, stand_up, …) when ELIZA_BRUHOS_URL is reachable. Options: skill (required), args (object). Not available on stock-only robots.",
  examples: [],
  validate: async (runtime: IAgentRuntime, _message: Memory) => {
    return getService(runtime)?.hasBruhOs() ?? false;
  },
  handler: async (
    runtime: IAgentRuntime,
    _message: Memory,
    _state: State | undefined,
    options: Record<string, unknown> | undefined,
    callback?: HandlerCallback,
  ): Promise<ActionResult> => {
    const service = getService(runtime);
    const client = service?.getBruhOs() ?? null;
    if (!service?.hasBruhOs() || !client) {
      const text =
        "bruhOS skill API is not available; stock RPC still works. Set ELIZA_BRUHOS_URL to enable skills.";
      await callback?.({ text });
      return { success: false, text };
    }
    const skillRaw = getStringOption(options, "skill", "");
    if (skillRaw === "") {
      const text =
        "TONYPI_RUN_SKILL requires options.skill (e.g. 'walk_toward').";
      await callback?.({ text });
      return { success: false, text };
    }
    let skill: string;
    try {
      skill = assertSafeBruhOsSkill(skillRaw);
    } catch (err) {
      const text = (err as Error).message;
      await callback?.({ text });
      return { success: false, text };
    }
    const argsRaw = options?.args;
    const args =
      argsRaw && typeof argsRaw === "object" && !Array.isArray(argsRaw)
        ? (argsRaw as Record<string, unknown>)
        : {};
    // Convenience: bare `target` option maps into args.target.
    if (typeof options?.target === "string" && args.target === undefined) {
      args.target = options.target;
    }
    try {
      const outcome = await client.runSkill(skill, args);
      service.noteAction(`skill:${skill}`, 1, false);
      const text = outcome.ok
        ? `bruhOS ${skill}: ${outcome.detail || "ok"}`
        : `bruhOS ${skill} ${outcome.status}: ${outcome.detail || "failed"}`;
      await callback?.({ text });
      return {
        success: outcome.ok,
        text,
        data: { skill, args, outcome },
      };
    } catch (err) {
      const text = `bruhOS skill failed: ${(err as Error).message}`;
      await callback?.({ text });
      return { success: false, text };
    }
  },
};
