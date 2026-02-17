from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import xarray as xr

from .config import Settings, get_settings
from .tilemath import TileBounds, stride_for_zoom


REQUIRED_DIMS = {"init_time", "lead_time", "ensemble_member", "latitude", "longitude"}
REQUIRED_VARS = {"wind_u_10m", "wind_v_10m", "temperature_2m"}


@dataclass(frozen=True)
class TileGrid:
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float]]
    units: str


@dataclass(frozen=True)
class WindPointSummary:
    nearest_latitude: float
    nearest_longitude: float
    minimum: float
    p10: float
    p50: float
    mean: float
    p90: float
    maximum: float
    count: int
    units: str


@dataclass(frozen=True)
class WindTimeseriesSummary:
    nearest_latitude: float
    nearest_longitude: float
    lead_hours: list[int]
    p10: list[float]
    p50: list[float]
    mean: list[float]
    p90: list[float]
    units: str


@dataclass(frozen=True)
class WindEnsembleTimeseriesSummary:
    nearest_latitude: float
    nearest_longitude: float
    lead_hours: list[int]
    ensemble_member_ids: list[int]
    values: list[list[float]]
    units: str


def _lead_to_hours(lead: np.timedelta64) -> int:
    return int(lead / np.timedelta64(1, "h"))


class ForecastDataAdapter:
    def __init__(self, dataset: xr.Dataset):
        self._dataset = dataset
        missing_dims = REQUIRED_DIMS.difference(dataset.sizes.keys())
        if missing_dims:
            missing_sorted = ", ".join(sorted(missing_dims))
            raise ValueError(f"Dataset missing required dimensions: {missing_sorted}")
        missing_vars = REQUIRED_VARS.difference(dataset.data_vars.keys())
        if missing_vars:
            missing_sorted = ", ".join(sorted(missing_vars))
            raise ValueError(f"Dataset missing required variables: {missing_sorted}")

    @property
    def dataset(self) -> xr.Dataset:
        return self._dataset

    def list_cycles(self, max_cycles: int) -> list[str]:
        cycles = self._dataset["init_time"].values
        items = [np.datetime_as_string(t, unit="s") for t in cycles]
        items.sort(reverse=True)
        return items[:max_cycles]

    def list_lead_hours(self) -> list[int]:
        leads = self._dataset["lead_time"].values
        return [_lead_to_hours(v) for v in leads]

    def ensure_cycle(self, cycle: str) -> np.datetime64:
        try:
            cycle_ts = np.datetime64(cycle)
        except ValueError as exc:
            raise ValueError(f"Invalid cycle timestamp: {cycle}") from exc
        available = self._dataset["init_time"].values
        if cycle_ts not in available:
            raise ValueError(f"Cycle not available: {cycle}")
        return cycle_ts

    def ensure_lead(self, lead_hour: int) -> np.timedelta64:
        lead = np.timedelta64(lead_hour, "h")
        available = self._dataset["lead_time"].values
        if lead not in available:
            raise ValueError(f"Lead not available: {lead_hour}h")
        return lead

    @staticmethod
    def _select_bbox(data_array: xr.DataArray, bounds: TileBounds) -> xr.DataArray:
        lats = data_array["latitude"]
        lons = data_array["longitude"]

        lat_slice = (
            slice(bounds.north, bounds.south)
            if bool((lats[0] > lats[-1]).item())
            else slice(bounds.south, bounds.north)
        )
        lat_selected = data_array.sel(latitude=lat_slice)

        if bounds.west <= bounds.east:
            return lat_selected.sel(longitude=slice(bounds.west, bounds.east))

        west_chunk = lat_selected.sel(longitude=slice(bounds.west, 180.0))
        east_chunk = lat_selected.sel(longitude=slice(-180.0, bounds.east))
        return xr.concat([west_chunk, east_chunk], dim="longitude")

    @staticmethod
    def _tile_resolution_for_zoom(z: int) -> int:
        # Keep low-zoom tiles small while preserving enough samples for smooth seams.
        stride = max(1, stride_for_zoom(z))
        base = get_settings().tile_target_px
        return max(2, int(np.ceil(base / stride)))

    def wind_speed_p50_tile(
        self,
        cycle: str,
        lead_hour: int,
        bounds: TileBounds,
        z: int,
    ) -> TileGrid:
        cycle_ts = self.ensure_cycle(cycle)
        lead = self.ensure_lead(lead_hour)

        ds_sel = self._dataset.sel(init_time=cycle_ts, lead_time=lead)
        u10 = ds_sel["wind_u_10m"]
        v10 = ds_sel["wind_v_10m"]
        # Apply spatial subset BEFORE quantile to avoid loading full 721x1440 grid
        u10_sub = self._select_bbox(u10, bounds)
        v10_sub = self._select_bbox(v10, bounds)
        speed = np.sqrt(u10_sub * u10_sub + v10_sub * v10_sub)
        p50 = speed.quantile(0.5, dim="ensemble_member")

        subset = p50
        stride = stride_for_zoom(z)
        if stride > 1:
            subset = subset.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))

        raw = subset.values.astype(np.float32)
        if np.ma.is_masked(raw):
            raw = np.ma.filled(raw, fill_value=np.nan)
        values = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return TileGrid(
            latitudes=[float(v) for v in subset["latitude"].values.tolist()],
            longitudes=[float(v) for v in subset["longitude"].values.tolist()],
            values=np.round(values, 3).tolist(),
            units="m/s",
        )

    def temperature_2m_member_tile(
        self,
        cycle: str,
        lead_hour: int,
        bounds: TileBounds,
        z: int,
    ) -> TileGrid:
        cycle_ts = self.ensure_cycle(cycle)
        lead = self.ensure_lead(lead_hour)

        ds_sel = self._dataset.sel(init_time=cycle_ts, lead_time=lead)
        t2m = ds_sel["temperature_2m"]
        if "ensemble_member" in t2m.dims:
            t2m = t2m.isel(ensemble_member=0)
        units_attr = str(t2m.attrs.get("units", "")).strip().lower()
        if units_attr in {"k", "kelvin", "degrees_kelvin", "degree_kelvin"}:
            temp_c = t2m - np.float32(273.15)
        else:
            temp_c = t2m

        # Sample directly on tile bounds so neighboring tiles share identical seam coordinates.
        # This avoids visible jumps caused by tile-local decimation phase differences.
        resolution = self._tile_resolution_for_zoom(z)
        lon_targets = np.linspace(bounds.west, bounds.east, resolution, dtype=np.float32)

        lat_desc = bool((temp_c["latitude"][0] > temp_c["latitude"][-1]).item())
        if lat_desc:
            interp_src = temp_c.sortby("latitude")
            lat_targets_asc = np.linspace(bounds.south, bounds.north, resolution, dtype=np.float32)
            subset = interp_src.interp(latitude=lat_targets_asc, longitude=lon_targets)
            subset = subset.isel(latitude=slice(None, None, -1))
        else:
            interp_src = temp_c
            lat_targets = np.linspace(bounds.south, bounds.north, resolution, dtype=np.float32)
            subset = interp_src.interp(latitude=lat_targets, longitude=lon_targets)

        raw = subset.values.astype(np.float32)
        if np.ma.is_masked(raw):
            raw = np.ma.filled(raw, fill_value=np.nan)
        values = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return TileGrid(
            latitudes=[float(v) for v in subset["latitude"].values.tolist()],
            longitudes=[float(v) for v in subset["longitude"].values.tolist()],
            values=np.round(values, 3).tolist(),
            units="degC",
        )

    def wind_point_summary(
        self,
        cycle: str,
        lead_hour: int,
        latitude: float,
        longitude: float,
    ) -> WindPointSummary:
        cycle_ts = self.ensure_cycle(cycle)
        lead = self.ensure_lead(lead_hour)
        ds_sel = self._dataset.sel(init_time=cycle_ts, lead_time=lead)
        u = ds_sel["wind_u_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        v = ds_sel["wind_v_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        speed = np.sqrt(u * u + v * v).values.astype(np.float32)
        return WindPointSummary(
            nearest_latitude=float(u["latitude"].item()),
            nearest_longitude=float(u["longitude"].item()),
            minimum=float(np.min(speed)),
            p10=float(np.quantile(speed, 0.10)),
            p50=float(np.quantile(speed, 0.50)),
            mean=float(np.mean(speed)),
            p90=float(np.quantile(speed, 0.90)),
            maximum=float(np.max(speed)),
            count=int(speed.shape[0]),
            units="m/s",
        )

    def wind_timeseries_summary(
        self,
        cycle: str,
        latitude: float,
        longitude: float,
    ) -> WindTimeseriesSummary:
        cycle_ts = self.ensure_cycle(cycle)
        ds_sel = self._dataset.sel(init_time=cycle_ts)
        u = ds_sel["wind_u_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        v = ds_sel["wind_v_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        speed = np.sqrt(u * u + v * v)
        p10 = speed.quantile(0.10, dim="ensemble_member").values.astype(np.float32)
        p50 = speed.quantile(0.50, dim="ensemble_member").values.astype(np.float32)
        mean = speed.mean(dim="ensemble_member").values.astype(np.float32)
        p90 = speed.quantile(0.90, dim="ensemble_member").values.astype(np.float32)
        lead_hours = [_lead_to_hours(v) for v in ds_sel["lead_time"].values]
        return WindTimeseriesSummary(
            nearest_latitude=float(u["latitude"].item()),
            nearest_longitude=float(u["longitude"].item()),
            lead_hours=lead_hours,
            p10=np.round(p10, 3).tolist(),
            p50=np.round(p50, 3).tolist(),
            mean=np.round(mean, 3).tolist(),
            p90=np.round(p90, 3).tolist(),
            units="m/s",
        )

    def wind_ensemble_timeseries_summary(
        self,
        cycle: str,
        latitude: float,
        longitude: float,
    ) -> WindEnsembleTimeseriesSummary:
        cycle_ts = self.ensure_cycle(cycle)
        ds_sel = self._dataset.sel(init_time=cycle_ts)
        u = ds_sel["wind_u_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        v = ds_sel["wind_v_10m"].sel(latitude=latitude, longitude=longitude, method="nearest")
        speed_ms = np.sqrt(u * u + v * v).transpose("ensemble_member", "lead_time").values.astype(np.float32)
        speed_kt = speed_ms * np.float32(1.9438445)
        lead_hours = [_lead_to_hours(vv) for vv in ds_sel["lead_time"].values]
        member_ids = [int(vv) for vv in ds_sel["ensemble_member"].values.tolist()]
        return WindEnsembleTimeseriesSummary(
            nearest_latitude=float(u["latitude"].item()),
            nearest_longitude=float(u["longitude"].item()),
            lead_hours=lead_hours,
            ensemble_member_ids=member_ids,
            values=np.round(speed_kt, 3).tolist(),
            units="kt",
        )


@lru_cache(maxsize=1)
def get_adapter() -> ForecastDataAdapter:
    import icechunk

    settings: Settings = get_settings()
    storage = icechunk.s3_storage(
        bucket=settings.dataset_bucket,
        prefix=settings.dataset_prefix,
        region=settings.dataset_region,
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session(settings.dataset_branch)
    dataset = xr.open_zarr(session.store, chunks="auto")
    return ForecastDataAdapter(dataset=dataset)
