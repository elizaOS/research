import type { TonyPiEnhancement } from "./capabilities";

/** Hold-to-walk ActionGroupDict codes (Hiwonder / bruhOS). */
export const WALK_CODES = {
  forward: "1",
  back: "2",
  sideLeft: "3",
  sideRight: "4",
  turnLeft: "7",
  turnRight: "8",
  stop: "0",
} as const;

/**
 * Named `.d6a` basenames for the walk codes above.
 * From on-robot ActionGroupDict.py (`2` → `back_fast`, not `back`).
 */
export const WALK_GROUPS = {
  forward: "go_forward",
  back: "back_fast",
  sideLeft: "left_move_fast",
  sideRight: "right_move_fast",
  turnLeft: "turn_left",
  turnRight: "turn_right",
} as const;

/** Stock / Pro action groups we allow through TONYPI_RUN_ACTION_GROUP. */
export const ALLOWED_ACTION_GROUPS: ReadonlySet<string> = new Set([
  "stand",
  "stand_slow",
  "go_forward",
  "go_forward_fast",
  "go_forward_one_step",
  "go_forward_one_small_step",
  "back",
  "back_fast",
  "back_one_step",
  "left_move",
  "left_move_fast",
  "right_move",
  "right_move_fast",
  "turn_left",
  "turn_right",
  "wave",
  "bow",
  "squat",
  "chest",
  "twist",
  "stepping",
  "push_ups",
  "sit_ups",
  "wing_chun",
  "left_uppercut",
  "right_uppercut",
  "left_kick",
  "right_kick",
  "left_shot",
  "left_shot_fast",
  "right_shot",
  "right_shot_fast",
  "stand_up_front",
  "stand_up_back",
  "jugong",
  "weightlifting",
  "move_up",
  "put_down",
  "grab_left",
  "grab_right",
  "grab_squat_left",
  "grab_squat_right",
  // TonyPi Pro hands / small steps (configs/tonypi_pro.toml + stock Pro packs)
  "hand_close",
  "hand_open",
  "turn_left_small_step",
  "turn_right_small_step",
  "creep_forward",
  "lie_down",
  "huibi",
]);

export function assertSafeActionName(name: string): string {
  if (name === "") {
    throw new Error("action name must be a non-empty string");
  }
  if (name.includes("/") || name.includes("\\") || name.includes("..")) {
    throw new Error(`action name rejected (path characters): ${name}`);
  }
  // Numeric ActionGroupDict codes ("0".."35") are fine.
  if (/^\d+$/.test(name)) return name;
  if (!ALLOWED_ACTION_GROUPS.has(name)) {
    throw new Error(`action name not in TonyPi allowlist: ${name}`);
  }
  return name;
}

export interface TonyPiStateSnapshot {
  connected: boolean;
  lastAction: string;
  lastTimes: number;
  runningHint: boolean;
  receivedAt: number;
  /** Highest enhancement layer currently available. */
  enhancement: TonyPiEnhancement;
  /** Optional bruhOS skill API probed successfully. */
  bruhos: boolean;
  /** Optional Pro-ish RPC methods probed successfully. */
  pro: boolean;
}
