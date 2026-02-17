import { describe, expect, it } from "vitest";
import { buildTemperatureTileLayer, temperatureToRgba, tileToRgba } from "./layers";
import type { TileResponse } from "./types";

describe("buildTemperatureTileLayer", () => {
  it("creates a deterministic layer id", () => {
    const layer = buildTemperatureTileLayer("2026-01-02T00:00:00", 12);
    expect(layer.id).toBe("temperature-tile-layer-2026-01-02T00:00:00-12");
    expect(layer.props.refinementStrategy).toBe("no-overlap");
  });

  it("renderSubLayers returns null for loading/empty tiles to prevent zero-dimension WebGL textures", () => {
    const layer = buildTemperatureTileLayer(
      "2026-01-02T00:00:00",
      12
    ) as import("@deck.gl/geo-layers").TileLayer<
      unknown,
      { renderSubLayers: (props: { data?: unknown }) => unknown }
    >;
    const renderSubLayers = layer.props.renderSubLayers!;
    expect(renderSubLayers({ data: undefined })).toBeNull();
    expect(renderSubLayers({ data: null })).toBeNull();
    expect(renderSubLayers({ data: {} })).toBeNull();
    expect(renderSubLayers({ data: { values: [] } })).toBeNull();
    expect(renderSubLayers({ data: { values: [[]] } })).toBeNull();
    expect(renderSubLayers({ data: { values: [[], []] } })).toBeNull();
  });
});

describe("temperatureToRgba", () => {
  it("maps cold temperatures to blueish colors", () => {
    const [r, , b] = temperatureToRgba(-20);
    expect(b).toBeGreaterThan(r);
  });

  it("maps warm temperatures to reddish colors", () => {
    const [r, , b] = temperatureToRgba(35);
    expect(r).toBeGreaterThan(b);
  });

  it("treats NaN as 0C for stable rendering", () => {
    const rgba = temperatureToRgba(Number.NaN);
    expect(rgba).toEqual(temperatureToRgba(0));
  });
});

describe("tileToRgba", () => {
  it("produces non-empty, non-all-black buffer for valid tile", () => {
    const tile: TileResponse = {
      product: "temperature_2m_member",
      cycle: "2026-01-01T00:00:00",
      lead_hour: 0,
      z: 10,
      x: 512,
      y: 512,
      bounds: { west: 0, south: 0, east: 1, north: 1 },
      units: "degC",
      latitudes: [0.5, 0],
      longitudes: [0, 0.5],
      values: [[0, 10], [20, 30]],
    };
    const { data, width, height } = tileToRgba(tile);
    expect(width).toBe(2);
    expect(height).toBe(2);
    expect(data.length).toBe(16);
    const hasNonBlack = Array.from(data).some((v, i) => i % 4 < 3 && v > 0);
    expect(hasNonBlack).toBe(true);
  });

  it("treats NaN as 0C color for stable pixels", () => {
    const tile = {
      product: "temperature_2m_member",
      cycle: "2026-01-01T00:00:00",
      lead_hour: 0,
      z: 10,
      x: 512,
      y: 512,
      bounds: { west: 0, south: 0, east: 1, north: 1 },
      units: "degC",
      latitudes: [0],
      longitudes: [0],
      values: [[Number.NaN]],
    } as TileResponse;
    const { data } = tileToRgba(tile);
    const expected = temperatureToRgba(0);
    expect(data[0]).toBe(expected[0]);
    expect(data[1]).toBe(expected[1]);
    expect(data[2]).toBe(expected[2]);
    expect(data[3]).toBe(expected[3]);
  });

  it("keeps original tile dimensions (no padding/stretch seams)", () => {
    const tile: TileResponse = {
      product: "temperature_2m_member",
      cycle: "2026-01-01T00:00:00",
      lead_hour: 0,
      z: 3,
      x: 0,
      y: 0,
      bounds: { west: -180, south: -90, east: 180, north: 90 },
      units: "degC",
      latitudes: [0, 1, 2],
      longitudes: [0, 1, 2, 3, 4],
      values: [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
      ],
    };
    const { data, width, height } = tileToRgba(tile);
    expect(width).toBe(5);
    expect(height).toBe(3);
    expect(data.length).toBe(5 * 3 * 4);
  });

  it("returns 0x0 for empty tiles and avoids creating invalid textures", () => {
    const emptyCases: TileResponse[] = [
      { ...baseTile(), values: [] },
      { ...baseTile(), values: [[]] },
      { ...baseTile(), values: [[], [], []] },
    ];
    for (const tile of emptyCases) {
      const result = tileToRgba(tile);
      expect(result.width).toBe(0);
      expect(result.height).toBe(0);
      expect(result.data.length).toBe(0);
    }
  });
});

function baseTile(): TileResponse {
  return {
    product: "temperature_2m_member",
    cycle: "2026-01-01T00:00:00",
    lead_hour: 0,
    z: 3,
    x: 0,
    y: 0,
    bounds: { west: -180, south: -90, east: 180, north: 90 },
    units: "degC",
    latitudes: [],
    longitudes: [],
    values: [],
  };
}
