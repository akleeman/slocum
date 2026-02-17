import type {
  MetaCyclesResponse,
  PointWindResponse,
  TileResponse,
  TimeseriesWindEnsemblesResponse,
  TimeseriesWindResponse
} from "./types";

const jsonHeaders = {
  Accept: "application/json"
};

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: jsonHeaders });
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${url}`);
  }
  return (await response.json()) as T;
}

export async function getMetaCycles(): Promise<MetaCyclesResponse> {
  const payload = await fetchJson<MetaCyclesResponse>("/meta/cycles");
  if (!payload.cycles?.length || !payload.lead_hours?.length) {
    throw new Error("Invalid cycle metadata response");
  }
  return payload;
}

export async function getTemperatureTile(
  cycle: string,
  leadHour: number,
  z: number,
  x: number,
  y: number
): Promise<TileResponse> {
  return fetchJson<TileResponse>(`/tiles/temperature_2m_member/${cycle}/${leadHour}/${z}/${x}/${y}`);
}

export async function getPointWind(
  cycle: string,
  leadHour: number,
  lat: number,
  lon: number
): Promise<PointWindResponse> {
  const query = new URLSearchParams({
    cycle,
    lead: String(leadHour),
    lat: String(lat),
    lon: String(lon)
  });
  return fetchJson<PointWindResponse>(`/point/wind?${query}`);
}

export async function getWindTimeseries(
  cycle: string,
  lat: number,
  lon: number
): Promise<TimeseriesWindResponse> {
  const query = new URLSearchParams({
    cycle,
    lat: String(lat),
    lon: String(lon)
  });
  return fetchJson<TimeseriesWindResponse>(`/timeseries/wind?${query}`);
}

export async function getWindEnsembleTimeseries(
  cycle: string,
  lat: number,
  lon: number
): Promise<TimeseriesWindEnsemblesResponse> {
  const query = new URLSearchParams({
    cycle,
    lat: String(lat),
    lon: String(lon)
  });
  return fetchJson<TimeseriesWindEnsemblesResponse>(`/timeseries/wind/ensembles?${query}`);
}
