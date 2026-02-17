import { expect, test } from "@playwright/test";

test("map loads and lead change triggers new tile request", async ({ page }) => {
  let tileRequestCount = 0;
  page.on("request", (request) => {
    const url = request.url();
    if (url.includes("/tiles/temperature_2m_member/")) {
      tileRequestCount += 1;
    }
  });

  await page.goto("/");
  await expect(page.getByTestId("map-container")).toBeVisible();
  await page.waitForTimeout(3000);
  const beforeLeadChange = tileRequestCount;

  const options = await page.locator("[data-testid='lead-select'] option").count();
  if (options > 1) {
    await page.getByTestId("lead-select").selectOption({ index: options - 1 });
  }
  await page.waitForTimeout(3000);

  expect(tileRequestCount).toBeGreaterThan(beforeLeadChange);

  const screenshot = await page.getByTestId("map-container").screenshot();
  expect(screenshot.length).toBeGreaterThan(1000);
});

test("rendered map pixels stay continuous across tile boundaries", async ({ page }) => {
  let interceptedTileRequests = 0;
  await page.route("https://demotiles.maplibre.org/style.json*", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        version: 8,
        name: "blank",
        sources: {},
        layers: [{ id: "background", type: "background", paint: { "background-color": "#000000" } }]
      })
    });
  });

  await page.route("**/tiles/temperature_2m_member/**", async (route) => {
    interceptedTileRequests += 1;
    const url = new URL(route.request().url());
    const parts = url.pathname.split("/");
    const z = Number(parts[parts.length - 3]);
    const x = Number(parts[parts.length - 2]);
    const y = Number(parts[parts.length - 1]);

    const n = 2 ** z;
    const west = (x / n) * 360 - 180;
    const east = ((x + 1) / n) * 360 - 180;
    const latFromY = (yy: number) => {
      const latRad = Math.atan(Math.sinh(Math.PI * (1 - (2 * yy) / n)));
      return (latRad * 180) / Math.PI;
    };
    const north = latFromY(y);
    const south = latFromY(y + 1);

    const width = 32;
    const height = 32;
    const longitudes = Array.from({ length: width }, (_, i) => west + ((east - west) * i) / (width - 1));
    const latitudes = Array.from({ length: height }, (_, i) => north + ((south - north) * i) / (height - 1));
    const values = latitudes.map((lat) => longitudes.map((lon) => 12 + lat * 0.05 + lon * 0.03));

    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        product: "temperature_2m_member",
        cycle: "2026-02-16T00:00:00",
        lead_hour: 0,
        z,
        x,
        y,
        bounds: { west, south, east, north },
        units: "degC",
        latitudes,
        longitudes,
        values
      })
    });
  });

  await page.goto("/");
  await expect(page.getByTestId("map-container")).toBeVisible();
  await page.waitForFunction(() => (window as unknown as { __tilesLoaded?: number }).__tilesLoaded === 1, null, {
    timeout: 1000
  }).catch(() => {});
  await page.waitForTimeout(4000);
  expect(interceptedTileRequests).toBeGreaterThan(0);

  const continuity = await page.evaluate(() => {
    const mapRoot = document.getElementById("map");
    if (!mapRoot) {
      return { ok: false, reason: "missing map root" };
    }
    const canvases = Array.from(mapRoot.querySelectorAll("canvas")) as HTMLCanvasElement[];
    if (canvases.length === 0) {
      return { ok: false, reason: "missing canvas" };
    }
    const canvas = canvases.reduce((a, b) => (a.width * a.height >= b.width * b.height ? a : b));
    if (canvas.width < 32 || canvas.height < 32) {
      return { ok: false, reason: "canvas too small" };
    }

    const copy = document.createElement("canvas");
    copy.width = canvas.width;
    copy.height = canvas.height;
    const ctx = copy.getContext("2d");
    if (!ctx) {
      return { ok: false, reason: "missing 2d context" };
    }
    ctx.drawImage(canvas, 0, 0);
    const data = ctx.getImageData(0, 0, copy.width, copy.height).data;

    const channelJump = (idxA: number, idxB: number): number => {
      const dr = Math.abs(data[idxA] - data[idxB]);
      const dg = Math.abs(data[idxA + 1] - data[idxB + 1]);
      const db = Math.abs(data[idxA + 2] - data[idxB + 2]);
      return dr + dg + db;
    };

    const verticalBoundaryMean: number[] = [];
    for (let x = 0; x < copy.width - 1; x += 1) {
      let acc = 0;
      for (let y = 0; y < copy.height; y += 1) {
        const a = (y * copy.width + x) * 4;
        const b = (y * copy.width + x + 1) * 4;
        acc += channelJump(a, b);
      }
      verticalBoundaryMean.push(acc / copy.height);
    }

    const horizontalBoundaryMean: number[] = [];
    for (let y = 0; y < copy.height - 1; y += 1) {
      let acc = 0;
      for (let x = 0; x < copy.width; x += 1) {
        const a = (y * copy.width + x) * 4;
        const b = ((y + 1) * copy.width + x) * 4;
        acc += channelJump(a, b);
      }
      horizontalBoundaryMean.push(acc / copy.width);
    }

    const median = (arr: number[]): number => {
      const sorted = [...arr].sort((a, b) => a - b);
      return sorted[Math.floor(sorted.length / 2)];
    };
    const percentile = (arr: number[], p: number): number => {
      const sorted = [...arr].sort((a, b) => a - b);
      const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
      return sorted[idx];
    };

    const vMedian = median(verticalBoundaryMean);
    const hMedian = median(horizontalBoundaryMean);
    const vP95 = percentile(verticalBoundaryMean, 0.95);
    const hP95 = percentile(horizontalBoundaryMean, 0.95);
    const vMax = Math.max(...verticalBoundaryMean);
    const hMax = Math.max(...horizontalBoundaryMean);

    return {
      ok: true,
      vMedian,
      hMedian,
      vP95,
      hP95,
      vMax,
      hMax,
      vRatio: vMedian > 0 ? vMax / vMedian : Infinity,
      hRatio: hMedian > 0 ? hMax / hMedian : Infinity
    };
  });

  expect(continuity.ok).toBe(true);
  // In a smooth synthetic field, tile seam lines should not produce giant jump spikes.
  // Use p95-relative thresholds to remain stable if large flat areas make median ~0.
  expect(continuity.vMax).toBeLessThanOrEqual(Math.max(20, continuity.vP95 * 2.5));
  expect(continuity.hMax).toBeLessThanOrEqual(Math.max(20, continuity.hP95 * 2.5));
});
