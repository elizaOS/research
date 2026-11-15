/**
 * Optional HTTP client for a bruhOS skill runtime beside the robot.
 *
 * Contract (progressive enhancement — absent → plugin stays on stock RPC):
 *   GET  /v1/capabilities  → { name, version, body?, skills? }
 *   GET  /v1/skills        → { skillpack_hash?, skills: [...] }
 *   GET  /v1/observe       → Observation dict (bruhos.schemas.obs)
 *   POST /v1/skills/run    → { skill, args } → Outcome dict (bruhos.schemas.skill)
 *
 * bruhOS may also call *into* an Eliza instance over REST (chat / agents).
 * That reverse path is configured in bruhOS, not here.
 */

import {
  assertSafeBruhOsSkill,
  type BruhOsSkillInfo,
  type TonyPiBodyKind,
} from "./capabilities";

export const DEFAULT_BRUHOS_PORT = 9040;

export class BruhOsError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BruhOsError";
  }
}

export interface BruhOsClientOptions {
  /** Base URL, e.g. `http://192.168.149.1:9040`. */
  url: string;
  timeoutMs?: number;
}

export interface BruhOsCapabilities {
  name: string;
  version: string;
  body: TonyPiBodyKind;
  skills: BruhOsSkillInfo[];
  skillpackHash: string;
}

export interface BruhOsOutcome {
  status: string;
  detail: string;
  data: Record<string, unknown>;
  ok: boolean;
}

export interface BruhOsObservation {
  t: number;
  frame: number;
  objects: unknown[];
  imu: Record<string, unknown>;
  line: Record<string, unknown>;
  holding: string | null;
}

export function redactedBruhOsUrl(rawUrl: string): string {
  try {
    const parsed = new URL(rawUrl);
    if (
      !["http:", "https:"].includes(parsed.protocol) ||
      parsed.hostname === ""
    ) {
      return "<invalid bruhos endpoint>";
    }
    return `${parsed.protocol}//${parsed.host}`;
  } catch {
    return "<invalid bruhos endpoint>";
  }
}

function _normalizeBaseUrl(rawUrl: string): string {
  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new Error("bruhOS URL must be a valid http:// or https:// URL");
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("bruhOS URL must use http:// or https://");
  }
  if (parsed.username !== "" || parsed.password !== "") {
    throw new Error("bruhOS credentials must not be placed in URL userinfo");
  }
  parsed.search = "";
  parsed.hash = "";
  const path = parsed.pathname.replace(/\/$/, "");
  parsed.pathname = path === "" ? "/" : path;
  return parsed.toString().replace(/\/$/, "");
}

/** Build a bruhOS base URL from host (default port 9040). */
export function bruhOsUrlFromHost(
  host: string,
  port: number = DEFAULT_BRUHOS_PORT,
): string {
  const trimmed = host.trim();
  if (trimmed === "") throw new Error("bruhOS host must not be empty");
  if (/^https?:\/\//i.test(trimmed)) return _normalizeBaseUrl(trimmed);
  return _normalizeBaseUrl(`http://${trimmed}:${port}`);
}

function _asSkillInfo(raw: unknown): BruhOsSkillInfo | null {
  if (typeof raw === "string" && raw !== "") return { skill: raw };
  if (raw && typeof raw === "object") {
    const obj = raw as Record<string, unknown>;
    const skill =
      (typeof obj.skill === "string" && obj.skill) ||
      (typeof obj.name === "string" && obj.name) ||
      "";
    if (skill === "") return null;
    return {
      skill,
      description:
        typeof obj.description === "string" ? obj.description : undefined,
      args:
        obj.args && typeof obj.args === "object"
          ? (obj.args as Record<string, unknown>)
          : undefined,
    };
  }
  return null;
}

export class BruhOsClient {
  readonly url: string;
  readonly timeoutMs: number;

  constructor(options: BruhOsClientOptions) {
    this.url = _normalizeBaseUrl(options.url);
    this.timeoutMs = options.timeoutMs ?? 8000;
  }

  private async _fetchJson(
    path: string,
    init?: RequestInit,
    timeoutMs?: number,
  ): Promise<unknown> {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs ?? this.timeoutMs);
    try {
      const res = await fetch(`${this.url}${path}`, {
        ...init,
        signal: ctrl.signal,
        headers: {
          Accept: "application/json",
          ...(init?.body ? { "Content-Type": "application/json" } : {}),
          ...init?.headers,
        },
      });
      if (!res.ok) {
        throw new BruhOsError(
          `HTTP ${res.status} from ${redactedBruhOsUrl(this.url)}${path}`,
        );
      }
      return (await res.json()) as unknown;
    } catch (err) {
      if (err instanceof BruhOsError) throw err;
      if ((err as Error).name === "AbortError") {
        throw new BruhOsError(
          `bruhOS timeout (${timeoutMs ?? this.timeoutMs}ms) ${path}`,
        );
      }
      throw new BruhOsError(`bruhOS ${path} failed: ${(err as Error).message}`);
    } finally {
      clearTimeout(timer);
    }
  }

  async capabilities(): Promise<BruhOsCapabilities> {
    const raw = (await this._fetchJson(
      "/v1/capabilities",
      undefined,
      2500,
    )) as Record<string, unknown> | null;
    if (!raw || typeof raw !== "object") {
      throw new BruhOsError("bruhOS capabilities response was empty");
    }
    const name = typeof raw.name === "string" ? raw.name : "bruhos";
    const version =
      typeof raw.version === "string"
        ? raw.version
        : typeof raw.bruhos_version === "string"
          ? raw.bruhos_version
          : "";
    const bodyRaw =
      typeof raw.body === "string"
        ? raw.body
        : typeof raw.body_name === "string"
          ? raw.body_name
          : "unknown";
    const body: TonyPiBodyKind =
      bodyRaw === "tonypi" || bodyRaw === "tonypi_pro" ? bodyRaw : "unknown";
    const skillList = Array.isArray(raw.skills) ? raw.skills : [];
    const skills = skillList
      .map(_asSkillInfo)
      .filter((s): s is BruhOsSkillInfo => s !== null);
    const skillpackHash =
      typeof raw.skillpack_hash === "string" ? raw.skillpack_hash : "";
    return { name, version, body, skills, skillpackHash };
  }

  async listSkills(): Promise<BruhOsSkillInfo[]> {
    const raw = (await this._fetchJson("/v1/skills")) as Record<
      string,
      unknown
    >;
    const list = Array.isArray(raw.skills) ? raw.skills : [];
    return list
      .map(_asSkillInfo)
      .filter((s): s is BruhOsSkillInfo => s !== null);
  }

  async observe(): Promise<BruhOsObservation> {
    const raw = (await this._fetchJson("/v1/observe")) as Record<
      string,
      unknown
    >;
    return {
      t: typeof raw.t === "number" ? raw.t : 0,
      frame: typeof raw.frame === "number" ? raw.frame : 0,
      objects: Array.isArray(raw.objects) ? raw.objects : [],
      imu:
        raw.imu && typeof raw.imu === "object"
          ? (raw.imu as Record<string, unknown>)
          : {},
      line:
        raw.line && typeof raw.line === "object"
          ? (raw.line as Record<string, unknown>)
          : {},
      holding: typeof raw.holding === "string" ? raw.holding : null,
    };
  }

  async runSkill(
    skill: string,
    args: Record<string, unknown> = {},
  ): Promise<BruhOsOutcome> {
    const safe = assertSafeBruhOsSkill(skill);
    const raw = (await this._fetchJson("/v1/skills/run", {
      method: "POST",
      body: JSON.stringify({ skill: safe, args }),
    })) as Record<string, unknown>;
    const status = typeof raw.status === "string" ? raw.status : "failed";
    const detail = typeof raw.detail === "string" ? raw.detail : "";
    const data =
      raw.data && typeof raw.data === "object"
        ? (raw.data as Record<string, unknown>)
        : {};
    return { status, detail, data, ok: status === "ok" };
  }
}
