export type MetaCyclesResponse = {
  latest_cycle: string;
  cycles: string[];
  lead_hours: number[];
};

export type TileResponse = {
  product: string;
  cycle: string;
  lead_hour: number;
  z: number;
  x: number;
  y: number;
  bounds: {
    west: number;
    south: number;
    east: number;
    north: number;
  };
  units: string;
  latitudes: number[];
  longitudes: number[];
  values: number[][];
};

export type PointWindResponse = {
  cycle: string;
  lead_hour: number;
  latitude: number;
  longitude: number;
  nearest_latitude: number;
  nearest_longitude: number;
  units: string;
  count: number;
  minimum: number;
  p10: number;
  p50: number;
  mean: number;
  p90: number;
  maximum: number;
};

export type TimeseriesWindResponse = {
  cycle: string;
  latitude: number;
  longitude: number;
  nearest_latitude: number;
  nearest_longitude: number;
  units: string;
  lead_hours: number[];
  p10: number[];
  p50: number[];
  mean: number[];
  p90: number[];
};

export type TimeseriesWindEnsemblesResponse = {
  cycle: string;
  latitude: number;
  longitude: number;
  nearest_latitude: number;
  nearest_longitude: number;
  units: string;
  lead_hours: number[];
  ensemble_member_ids: number[];
  values: number[][];
};
