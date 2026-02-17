import { useEffect, useMemo, useRef, useState } from "react";
import { MapboxOverlay } from "@deck.gl/mapbox";
import type { Map as MapLibreMap } from "maplibre-gl";

/** maplibre-gl loaded from CDN (index.html) to avoid Vite/esbuild __publicField transformation */
const maplibregl = (window as unknown as { maplibregl: typeof import("maplibre-gl") }).maplibregl;
import type { Layer } from "@deck.gl/core";
import { getMetaCycles, getWindEnsembleTimeseries } from "./api";
import { buildTemperatureTileLayer } from "./layers";
import type { MetaCyclesResponse, TimeseriesWindEnsemblesResponse } from "./types";
import "./styles.css";

function ensemblePlotPopupContent(payload: TimeseriesWindEnsemblesResponse): string {
  const width = 360;
  const height = 220;
  const margin = { top: 18, right: 14, bottom: 30, left: 46 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const allValues = payload.values.flat();
  const minValue = Math.min(...allValues);
  const maxValue = Math.max(...allValues);
  const range = Math.max(0.001, maxValue - minValue);
  const leads = payload.lead_hours;
  const xFor = (idx: number) => margin.left + (idx / Math.max(1, leads.length - 1)) * plotW;
  const yFor = (value: number) => margin.top + plotH - ((value - minValue) / range) * plotH;

  const hGrid = Array.from({ length: 5 }, (_, i) => {
    const y = margin.top + (i / 4) * plotH;
    return `<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotW}" y2="${y}" stroke="#d0d0d0" stroke-width="1" />`;
  }).join("");
  const vGrid = Array.from({ length: Math.min(6, leads.length) }, (_, i) => {
    const x = margin.left + (i / Math.max(1, Math.min(6, leads.length) - 1)) * plotW;
    return `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotH}" stroke="#e0e0e0" stroke-width="1" />`;
  }).join("");
  const seriesLines = payload.values
    .map((series) => {
      const points = series.map((v, i) => `${xFor(i)},${yFor(v)}`).join(" ");
      return `<polyline points="${points}" fill="none" stroke="#1f77b4" stroke-opacity="0.25" stroke-width="1.2" />`;
    })
    .join("");
  const yMinText = minValue.toFixed(1);
  const yMaxText = maxValue.toFixed(1);
  const x0 = leads[0] ?? 0;
  const xN = leads[leads.length - 1] ?? 0;

  return `
    <div class="ensemble-popup">
      <div class="ensemble-popup-title">Wind Ensembles (${payload.units})</div>
      <div class="ensemble-popup-subtitle">
        Nearest grid point: ${payload.nearest_latitude.toFixed(2)}, ${payload.nearest_longitude.toFixed(2)}
      </div>
      <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
        ${hGrid}
        ${vGrid}
        <rect x="${margin.left}" y="${margin.top}" width="${plotW}" height="${plotH}" fill="none" stroke="#9a9a9a" stroke-width="1" />
        ${seriesLines}
        <text x="6" y="${margin.top + 4}" font-size="11" fill="#444">${yMaxText} ${payload.units}</text>
        <text x="6" y="${margin.top + plotH + 2}" font-size="11" fill="#444">${yMinText}</text>
        <text x="${margin.left}" y="${height - 8}" font-size="11" fill="#444">${x0}h</text>
        <text x="${margin.left + plotW - 20}" y="${height - 8}" font-size="11" fill="#444">${xN}h</text>
      </svg>
      <div class="ensemble-popup-footnote">${payload.ensemble_member_ids.length} ensemble members, units in knots</div>
    </div>
  `;
}

export default function App() {
  const mapRef = useRef<MapLibreMap | null>(null);
  const overlayRef = useRef<MapboxOverlay | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const popupRef = useRef<InstanceType<typeof maplibregl.Popup> | null>(null);
  const selectedCycleRef = useRef<string>("");

  const [meta, setMeta] = useState<MetaCyclesResponse | null>(null);
  const [selectedCycle, setSelectedCycle] = useState<string>("");
  const [selectedLead, setSelectedLead] = useState<number>(0);
  const [status, setStatus] = useState<string>("Loading metadata...");
  const [error, setError] = useState<string>("");

  const layer: Layer | null = useMemo(() => {
    if (!selectedCycle) {
      return null;
    }
    return buildTemperatureTileLayer(selectedCycle, selectedLead);
  }, [selectedCycle, selectedLead]);

  useEffect(() => {
    let cancelled = false;
    async function loadMeta() {
      try {
        const payload = await getMetaCycles();
        if (cancelled) {
          return;
        }
        setMeta(payload);
        setSelectedCycle(payload.latest_cycle);
        setSelectedLead(payload.lead_hours[0]);
        setStatus("Metadata loaded");
      } catch (err) {
        if (cancelled) {
          return;
        }
        setError((err as Error).message);
      }
    }
    void loadMeta();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    selectedCycleRef.current = selectedCycle;
  }, [selectedCycle]);

  useEffect(() => {
    if (mapRef.current || !mapContainerRef.current) {
      return;
    }
    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: "https://demotiles.maplibre.org/style.json",
      center: [-125.0, 35.0],
      zoom: 2
    });
    map.addControl(new maplibregl.NavigationControl(), "top-right");
    const overlay = new MapboxOverlay({ interleaved: true, layers: [] });
    map.addControl(overlay);
    mapRef.current = map;
    overlayRef.current = overlay;

    map.on("click", async (event) => {
      if (!selectedCycleRef.current) {
        return;
      }
      setStatus("Loading ensemble timeseries...");
      setError("");
      try {
        const ensembles = await getWindEnsembleTimeseries(
          selectedCycleRef.current,
          event.lngLat.lat,
          event.lngLat.lng
        );
        if (popupRef.current) {
          popupRef.current.remove();
        }
        popupRef.current = new maplibregl.Popup({ closeButton: true, closeOnClick: false, maxWidth: "420px" })
          .setLngLat(event.lngLat)
          .setHTML(ensemblePlotPopupContent(ensembles))
          .addTo(map);
        setStatus("Ensemble popup loaded");
      } catch (err) {
        setError((err as Error).message);
      }
    });

    return () => {
      popupRef.current?.remove();
      popupRef.current = null;
      overlay.finalize();
      map.remove();
      overlayRef.current = null;
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!overlayRef.current || !layer) {
      return;
    }
    overlayRef.current.setProps({ layers: [layer] });
    setStatus(`Rendering ${selectedLead}h lead`);
  }, [layer, selectedLead]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>ECMWF ENS Temperature MVP</h1>
        <div className="controls">
          <label>
            Cycle
            <select
              data-testid="cycle-select"
              value={selectedCycle}
              onChange={(event) => setSelectedCycle(event.target.value)}
            >
              {meta?.cycles.map((cycle) => (
                <option value={cycle} key={cycle}>
                  {cycle}
                </option>
              ))}
            </select>
          </label>
          <label>
            Lead
            <select
              data-testid="lead-select"
              value={selectedLead}
              onChange={(event) => setSelectedLead(Number(event.target.value))}
            >
              {meta?.lead_hours.map((lead) => (
                <option value={lead} key={lead}>
                  {lead}h
                </option>
              ))}
            </select>
          </label>
          <span data-testid="status">{status}</span>
        </div>
      </header>
      {error ? <div className="error-banner">{error}</div> : null}
      <div className="content">
        <div className="map-panel">
          <div ref={mapContainerRef} id="map" data-testid="map-container" />
        </div>
        <aside className="detail-panel" data-testid="detail-panel">
          <h2>Click details</h2>
          <p>Click anywhere on the map to open a popup with all ensemble wind traces.</p>
          <p>Popup chart includes nearest grid point, grid lines, and units in knots.</p>
        </aside>
      </div>
    </div>
  );
}
