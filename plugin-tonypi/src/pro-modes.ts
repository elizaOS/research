/**
 * TonyPi / TonyPi Pro on-robot Function ids (Functions/Running.py).
 * Same table as bruhos-app `src/lib/catalog.ts` MODES.
 *
 * Voice / multimodal AI (ChatGPT + voice module) live here as LoadFunc apps —
 * progressive Pro surface on stock RPC, not bruhOS-specific.
 */

export type ProModeGroup = "vision" | "ai";

export interface ProModeDef {
  id: number;
  title: string;
  group: ProModeGroup;
  blurb: string;
  /** Color / point pick in the WonderPi / bruhOS app UI. */
  pick?: boolean;
}

/** Allowlisted LoadFunc ids we will start from the agent. */
export const PRO_MODES: readonly ProModeDef[] = [
  {
    id: 2,
    group: "vision",
    title: "Auto Shooting",
    pick: true,
    blurb: "Find a ball of a picked color, line up, kick.",
  },
  {
    id: 3,
    group: "vision",
    title: "Gesture Control",
    blurb: "Steer with right index finger gestures.",
  },
  {
    id: 4,
    group: "vision",
    title: "Line Following",
    pick: true,
    blurb: "Follow a tape line of a picked color.",
  },
  {
    id: 5,
    group: "vision",
    title: "Object Tracking",
    pick: true,
    blurb: "Head and body follow a picked-color object.",
  },
  {
    id: 6,
    group: "vision",
    title: "Face Detection",
    blurb: "Scan, then wave on face detect.",
  },
  {
    id: 7,
    group: "vision",
    title: "Tag Recognition",
    blurb: "Tag cards trigger actions.",
  },
  {
    id: 8,
    group: "vision",
    title: "Smart Transport",
    blurb: "Pick up colored blocks and carry them.",
  },
  {
    id: 10,
    group: "vision",
    title: "Cross Hurdle",
    blurb: "Spot a blue hurdle and step over it.",
  },
  {
    id: 11,
    group: "vision",
    title: "Stairs",
    blurb: "Spot a red step and climb.",
  },
  {
    id: 12,
    group: "vision",
    title: "Color Sorting",
    blurb: "Recognize and sort colored blocks.",
  },
  // AI / voice — TonyPi Pro multimodal pack (ChatGPT + voice interaction module).
  {
    id: 13,
    group: "ai",
    title: "Smart Chat",
    blurb: "Wake word then talk; needs LLM key + Wi-Fi.",
  },
  {
    id: 14,
    group: "ai",
    title: "Voice Control",
    blurb: "Spoken commands → actions; needs LLM key + Wi-Fi.",
  },
  {
    id: 15,
    group: "ai",
    title: "AI Shooting",
    blurb: "Ask it to find and kick a ball.",
  },
  {
    id: 16,
    group: "ai",
    title: "Real-time Tracking",
    blurb: "Describe something; it follows.",
  },
  {
    id: 17,
    group: "ai",
    title: "Scene Understanding",
    blurb: "Ask what it sees (VLM).",
  },
  {
    id: 18,
    group: "ai",
    title: "Weather",
    blurb: "Ask about the weather.",
  },
] as const;

const BY_ID = new Map(PRO_MODES.map((m) => [m.id, m]));
const BY_TITLE = new Map(
  PRO_MODES.map((m) => [m.title.toLowerCase().replace(/\s+/g, "_"), m]),
);

export function proModeById(id: number): ProModeDef | undefined {
  return BY_ID.get(id);
}

/** Resolve a LoadFunc id from numeric id or title slug (e.g. "voice_control"). */
export function resolveProModeId(raw: string | number): number {
  if (typeof raw === "number" && Number.isInteger(raw)) {
    if (!BY_ID.has(raw)) {
      throw new Error(`TonyPi Pro mode id not allowlisted: ${raw}`);
    }
    return raw;
  }
  const text = String(raw).trim();
  if (/^\d+$/.test(text)) {
    const id = Number(text);
    if (!BY_ID.has(id)) {
      throw new Error(`TonyPi Pro mode id not allowlisted: ${id}`);
    }
    return id;
  }
  const slug = text.toLowerCase().replace(/\s+/g, "_");
  const hit =
    BY_TITLE.get(slug) ??
    PRO_MODES.find((m) => m.title.toLowerCase() === text.toLowerCase());
  if (!hit) {
    throw new Error(
      `TonyPi Pro mode not found: ${raw} (try id or slug, e.g. voice_control)`,
    );
  }
  return hit.id;
}

export function listProModes(group?: ProModeGroup): ProModeDef[] {
  return PRO_MODES.filter((m) => (group ? m.group === group : true)).map(
    (m) => ({ ...m }),
  );
}
