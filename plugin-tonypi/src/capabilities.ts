/**
 * Progressive enhancement levels for plugin-tonypi.
 *
 * - stock  — Hiwonder RPCServer.py :9030 (TonyPi / TonyPi Pro). Always enough.
 * - pro    — richer stock RPC surface (head PWM, StandUp, modes) when present.
 * - bruhos — optional bruhOS skill runtime beside the robot. Never required.
 *
 * The plugin must not expect bruhOS. Stock robots keep working with RPC alone.
 * The reverse path (bruhOS → Eliza REST) is owned by bruhOS, not this plugin.
 */

export type TonyPiBodyKind = "tonypi" | "tonypi_pro" | "unknown";

export type TonyPiEnhancement = "stock" | "pro" | "bruhos";

/** Default bruhOS v0.1 skill whitelist (mirrors bruhos/skills/builtin.py). */
export const BRUHOS_SKILLS = [
  "detect",
  "look_at",
  "walk_toward",
  "follow_line",
  "grab",
  "release",
  "kick",
  "wave",
  "stand_up",
  "stop",
] as const;

export type BruhOsSkillName = (typeof BRUHOS_SKILLS)[number];

export interface BruhOsSkillInfo {
  skill: string;
  description?: string;
  args?: Record<string, unknown>;
}

export interface TonyPiCapabilities {
  /** Stock JSON-RPC Heartbeat succeeded. */
  stock: boolean;
  /**
   * Optional Pro-ish RPC methods available (read-only probe succeeded).
   * Never required; stock RunAction still works when false.
   */
  pro: boolean;
  /** Optional bruhOS skill HTTP API reachable. */
  bruhos: boolean;
  body: TonyPiBodyKind;
  bruhosVersion: string;
  bruhosSkills: BruhOsSkillInfo[];
}

export function emptyCapabilities(): TonyPiCapabilities {
  return {
    stock: false,
    pro: false,
    bruhos: false,
    body: "unknown",
    bruhosVersion: "",
    bruhosSkills: [],
  };
}

export function enhancementLevel(caps: TonyPiCapabilities): TonyPiEnhancement {
  if (caps.bruhos) return "bruhos";
  if (caps.pro) return "pro";
  return "stock";
}

export function assertSafeBruhOsSkill(name: string): string {
  const trimmed = name.trim();
  if (trimmed === "") {
    throw new Error("bruhOS skill name must be a non-empty string");
  }
  if (!/^[a-z][a-z0-9_]*$/.test(trimmed)) {
    throw new Error(`bruhOS skill name rejected: ${name}`);
  }
  return trimmed;
}
