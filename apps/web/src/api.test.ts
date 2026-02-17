import { describe, expect, it, vi } from "vitest";
import { getMetaCycles } from "./api";

describe("api client", () => {
  it("parses cycles payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          latest_cycle: "2026-01-02T00:00:00",
          cycles: ["2026-01-02T00:00:00"],
          lead_hours: [0, 6]
        })
      })
    );
    const payload = await getMetaCycles();
    expect(payload.latest_cycle).toBe("2026-01-02T00:00:00");
    expect(payload.lead_hours).toEqual([0, 6]);
  });

  it("rejects invalid payload", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          latest_cycle: "2026-01-02T00:00:00",
          cycles: [],
          lead_hours: []
        })
      })
    );
    await expect(getMetaCycles()).rejects.toThrow("Invalid cycle metadata response");
  });
});
