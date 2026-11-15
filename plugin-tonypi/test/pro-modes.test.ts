import { describe, expect, it } from "vitest";
import { listProModes, proModeById, resolveProModeId } from "../src/pro-modes";

describe("Pro modes catalog", () => {
  it("resolves voice control and smart chat", () => {
    expect(resolveProModeId(14)).toBe(14);
    expect(resolveProModeId("voice_control")).toBe(14);
    expect(resolveProModeId("Smart Chat")).toBe(13);
    expect(proModeById(14)?.group).toBe("ai");
    expect(proModeById(13)?.title).toBe("Smart Chat");
  });

  it("lists AI modes including voice", () => {
    const ai = listProModes("ai");
    expect(ai.map((m) => m.id)).toEqual([13, 14, 15, 16, 17, 18]);
  });

  it("rejects unknown ids", () => {
    expect(() => resolveProModeId(99)).toThrow(/allowlisted/);
  });
});
