import { BitmapLayer } from "@deck.gl/layers";
import { TileLayer } from "@deck.gl/geo-layers";
import type { Layer } from "@deck.gl/core";
import { getTemperatureTile } from "./api";
import type { TileResponse } from "./types";

export function temperatureToRgba(tempC: number): [number, number, number, number] {
  const safe = Number.isFinite(tempC) ? tempC : 0;
  const normalized = Math.max(0, Math.min(1, (safe + 30.0) / 70.0));

  // Piecewise interpolation for a standard cool->warm temperature ramp:
  // -30C blue -> 0C cyan -> 10C green -> 25C yellow -> 40C red
  if (normalized <= 0.4285714286) {
    const t = normalized / 0.4285714286;
    return [0, Math.round(255 * t), 255, 190];
  }
  if (normalized <= 0.5714285714) {
    const t = (normalized - 0.4285714286) / 0.1428571428;
    return [Math.round(255 * (1 - t)), 255, Math.round(255 * (1 - t)), 190];
  }
  if (normalized <= 0.7857142857) {
    const t = (normalized - 0.5714285714) / 0.2142857143;
    return [255, Math.round(255 * (1 - t * 0.4)), 0, 190];
  }
  const t = (normalized - 0.7857142857) / 0.2142857143;
  return [255, Math.round(153 * (1 - t)), 0, 190];
}

export function tileToRgba(tile: TileResponse): { data: Uint8ClampedArray; width: number; height: number } {
  const height = tile.values?.length ?? 0;
  const width = height > 0 && Array.isArray(tile.values[0]) ? tile.values[0].length : 0;
  if (width === 0 || height === 0) {
    return { data: new Uint8ClampedArray(0), width: 0, height: 0 };
  }
  const buffer = new Uint8ClampedArray(width * height * 4);
  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      const offset = (row * width + col) * 4;
      const [r, g, b, a] = temperatureToRgba(tile.values[row][col]);
      buffer[offset] = r;
      buffer[offset + 1] = g;
      buffer[offset + 2] = b;
      buffer[offset + 3] = a;
    }
  }
  return { data: buffer, width, height };
}

export function buildTemperatureTileLayer(cycle: string, leadHour: number): Layer {
  return new TileLayer({
    id: `temperature-tile-layer-${cycle}-${leadHour}`,
    minZoom: 0,
    maxZoom: 12,
    tileSize: 256,
    refinementStrategy: "no-overlap",
    getTileData: async ({ index }) => {
      const { z, x, y } = index;
      return getTemperatureTile(cycle, leadHour, z, x, y);
    },
    renderSubLayers: (props) => {
      const tile = props.data as TileResponse | undefined;
      if (
        !tile ||
        !Array.isArray(tile.values) ||
        tile.values.length === 0 ||
        !Array.isArray(tile.values[0]) ||
        tile.values[0].length === 0
      ) {
        return null;
      }
      const image = tileToRgba(tile);
      if (image.width === 0 || image.height === 0) {
        return null;
      }
      // Use ImageData so deck.gl gets width/height correctly; disable mipmaps to avoid
      // glTexStorage2D/glGenerateMipmap errors with small or non-power-of-2 textures
      const imageData = new ImageData(image.width, image.height);
      imageData.data.set(image.data);
      return new BitmapLayer(props, {
        id: `${props.id}-bitmap`,
        bounds: [tile.bounds.west, tile.bounds.south, tile.bounds.east, tile.bounds.north],
        image: imageData,
        textureParameters: {
          minFilter: "linear",
          magFilter: "linear",
          mipmapFilter: "none",
          addressModeU: "clamp-to-edge",
          addressModeV: "clamp-to-edge"
        }
      });
    }
  });
}
