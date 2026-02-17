import { describe, expect, it } from "vitest";
import { selectionReducer, tileRequestKey } from "./state";

describe("selectionReducer", () => {
  it("updates cycle and lead", () => {
    const initial = { cycle: "A", leadHour: 0 };
    const step1 = selectionReducer(initial, { type: "setCycle", cycle: "B" });
    const step2 = selectionReducer(step1, { type: "setLead", leadHour: 6 });
    expect(step2).toEqual({ cycle: "B", leadHour: 6 });
  });
});

describe("tileRequestKey", () => {
  it("changes with lead", () => {
    const state = { cycle: "2026-01-01T00:00:00", leadHour: 0 };
    const key0 = tileRequestKey(state, 2, 1, 1);
    const key1 = tileRequestKey({ ...state, leadHour: 12 }, 2, 1, 1);
    expect(key0).not.toEqual(key1);
  });
});
