from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from slocum_api.data_access import ForecastDataAdapter
from slocum_api.tilemath import TileBounds, tile_xyz_bounds


def test_adapter_requires_expected_dimensions(sample_dataset: xr.Dataset):
    broken = sample_dataset.drop_dims("lead_time")
    with pytest.raises(ValueError, match="required dimensions"):
        ForecastDataAdapter(broken)


def test_cycles_and_leads_are_reported(adapter: ForecastDataAdapter):
    cycles = adapter.list_cycles(max_cycles=2)
    leads = adapter.list_lead_hours()
    assert cycles == ["2026-01-02T00:00:00", "2026-01-01T00:00:00"]
    assert leads == [0, 6, 12]


def test_temperature_tile_returns_subset_grid(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-60.0, south=-30.0, east=40.0, north=45.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=8)
    assert tile.units == "degC"
    assert len(tile.latitudes) > 0
    assert len(tile.longitudes) > 0
    assert len(tile.values) == len(tile.latitudes)
    assert len(tile.values[0]) == len(tile.longitudes)


def test_temperature_tile_values_match_lat_lon_shape(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-60.0, south=-30.0, east=40.0, north=45.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=8)
    assert len(tile.values) == len(tile.latitudes)
    for row in tile.values:
        assert len(row) == len(tile.longitudes)


def test_temperature_tile_resolution_matches_zoom_sampling_policy(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-60.0, south=-30.0, east=40.0, north=45.0)
    low_zoom = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=0)
    high_zoom = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=10)
    assert len(low_zoom.latitudes) >= 2
    assert len(low_zoom.longitudes) >= 2
    assert len(high_zoom.latitudes) >= len(low_zoom.latitudes)
    assert len(high_zoom.longitudes) >= len(low_zoom.longitudes)


def test_temperature_tile_values_are_finite_and_non_nan(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-60.0, south=-30.0, east=40.0, north=45.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=8)
    values_arr = np.array(tile.values)
    assert not np.any(np.isnan(values_arr)), "Tile contains NaN values"
    assert np.all(np.isfinite(values_arr)), "Tile contains non-finite values"


def test_temperature_tile_values_in_plausible_range_celsius(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-60.0, south=-30.0, east=40.0, north=45.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=6, bounds=bounds, z=8)
    values_arr = np.array(tile.values)
    assert np.all(values_arr >= -120), "Tile contains implausibly low temperatures (<-120C)"
    assert np.all(values_arr <= 80), "Tile contains implausibly high temperatures (>80C)"


def test_point_and_timeseries_summaries(adapter: ForecastDataAdapter):
    point = adapter.wind_point_summary(
        cycle="2026-01-01T00:00:00",
        lead_hour=0,
        latitude=10.2,
        longitude=-35.4,
    )
    assert point.count == 5
    assert point.maximum >= point.p90 >= point.p50 >= point.p10 >= point.minimum

    series = adapter.wind_timeseries_summary(
        cycle="2026-01-01T00:00:00",
        latitude=10.2,
        longitude=-35.4,
    )
    assert series.lead_hours == [0, 6, 12]
    assert len(series.p50) == 3

    ensembles = adapter.wind_ensemble_timeseries_summary(
        cycle="2026-01-01T00:00:00",
        latitude=10.2,
        longitude=-35.4,
    )
    assert ensembles.units == "kt"
    assert ensembles.lead_hours == [0, 6, 12]
    assert len(ensembles.ensemble_member_ids) == 5
    assert len(ensembles.values) == 5
    assert all(len(member_series) == 3 for member_series in ensembles.values)


def test_temperature_tile_with_nan_dataset_replaces_nan_with_zero(dataset_with_nan: xr.Dataset):
    """When dataset contains NaN, tile values are replaced with 0 for serialization safety."""
    adapter = ForecastDataAdapter(dataset_with_nan)
    # Tile covering (0,0) - the cell with NaN in both ensembles
    bounds = TileBounds(west=-1.0, south=-1.0, east=1.0, north=1.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=0, bounds=bounds, z=8)
    values_arr = np.array(tile.values)
    assert not np.any(np.isnan(values_arr)), "NaN must be replaced before response"


def test_invalid_cycle_raises(adapter: ForecastDataAdapter):
    bounds = TileBounds(west=-20.0, south=-20.0, east=20.0, north=20.0)
    with pytest.raises(ValueError, match="Cycle not available"):
        adapter.temperature_2m_member_tile(cycle="2020-01-01T00:00:00", lead_hour=0, bounds=bounds, z=1)


def test_invalid_lead_raises(adapter: ForecastDataAdapter):
    with pytest.raises(ValueError, match="Lead not available"):
        adapter.wind_point_summary(
            cycle="2026-01-01T00:00:00",
            lead_hour=48,
            latitude=0.0,
            longitude=0.0,
        )


def test_temperature_tile_uses_single_member_zero():
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0, 1], dtype=np.int16)
    latitude = np.array([0.0], dtype=np.float32)
    longitude = np.array([0.0], dtype=np.float32)
    # Member 0: 0C, Member 1: 40C
    t = np.array([[[[[0.0]], [[40.0]]]]], dtype=np.float32)
    zeros = np.zeros_like(t, dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "wind_u_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "wind_v_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "temperature_2m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), t),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    adapter = ForecastDataAdapter(dataset)
    bounds = TileBounds(west=-1.0, south=-1.0, east=1.0, north=1.0)
    tile = adapter.temperature_2m_member_tile(cycle="2026-01-01T00:00:00", lead_hour=0, bounds=bounds, z=10)
    values = np.array(tile.values, dtype=np.float32)
    assert np.all(np.isfinite(values))
    assert np.max(np.abs(values)) < 1e-5


def test_adjacent_temperature_tiles_share_seam_values():
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0], dtype=np.int16)
    latitude = np.array([45.0, 0.0, -45.0], dtype=np.float32)
    longitude = np.array([-90.0, -45.0, 0.0, 45.0, 90.0], dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    # Deterministic field with continuous seam at 0 longitude.
    t_member0 = (lon_grid / 20.0) + (lat_grid / 100.0)
    t = t_member0[np.newaxis, np.newaxis, np.newaxis, :, :].astype(np.float32)
    zeros = np.zeros_like(t, dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "wind_u_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "wind_v_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "temperature_2m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), t),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    adapter = ForecastDataAdapter(dataset)

    left_bounds = tile_xyz_bounds(z=1, x=0, y=0)
    right_bounds = tile_xyz_bounds(z=1, x=1, y=0)
    # Use high z for stride=1 to preserve full test grid on both sides.
    left = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, left_bounds, z=10)
    right = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, right_bounds, z=10)

    left_lons = np.array(left.longitudes)
    right_lons = np.array(right.longitudes)
    seam_lons = np.intersect1d(np.round(left_lons, 6), np.round(right_lons, 6))
    assert seam_lons.size >= 1, "Expected at least one shared seam longitude between adjacent tiles"
    seam_lon = seam_lons[0]

    left_lats = np.array(left.latitudes)
    right_lats = np.array(right.latitudes)
    seam_lats = np.intersect1d(np.round(left_lats, 6), np.round(right_lats, 6))
    assert seam_lats.size >= 1, "Expected shared seam latitudes between adjacent tiles"

    left_col = int(np.where(np.round(left_lons, 6) == seam_lon)[0][0])
    right_col = int(np.where(np.round(right_lons, 6) == seam_lon)[0][0])

    left_values = np.array(left.values, dtype=np.float32)
    right_values = np.array(right.values, dtype=np.float32)
    for seam_lat in seam_lats:
        left_row = int(np.where(np.round(left_lats, 6) == seam_lat)[0][0])
        right_row = int(np.where(np.round(right_lats, 6) == seam_lat)[0][0])
        assert left_values[left_row, left_col] == pytest.approx(right_values[right_row, right_col], abs=2e-3)
        # Stronger check: near-seam slope should be continuous as well (no jump in first interior band)
        if left_col > 0 and right_col < (right_values.shape[1] - 1):
            left_slope = left_values[left_row, left_col] - left_values[left_row, left_col - 1]
            right_slope = right_values[right_row, right_col + 1] - right_values[right_row, right_col]
            assert left_slope == pytest.approx(right_slope, abs=2e-3)


def test_adjacent_temperature_tiles_share_vertical_seam_values():
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0], dtype=np.int16)
    # Include equator explicitly so y=0/y=1 seam at z=1 has shared latitude sample.
    latitude = np.array([80.0, 60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0, -80.0], dtype=np.float32)
    longitude = np.array([-170.0, -120.0, -60.0, 0.0, 60.0, 120.0, 170.0], dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    # Continuous field in both latitude and longitude.
    t_member0 = 10.0 + (lon_grid / 60.0) + (lat_grid / 50.0)
    t = t_member0[np.newaxis, np.newaxis, np.newaxis, :, :].astype(np.float32)
    zeros = np.zeros_like(t, dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "wind_u_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "wind_v_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "temperature_2m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), t),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    adapter = ForecastDataAdapter(dataset)

    north_bounds = tile_xyz_bounds(z=1, x=1, y=0)
    south_bounds = tile_xyz_bounds(z=1, x=1, y=1)
    north = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, north_bounds, z=10)
    south = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, south_bounds, z=10)

    north_lats = np.array(north.latitudes)
    south_lats = np.array(south.latitudes)
    seam_lats = np.intersect1d(np.round(north_lats, 6), np.round(south_lats, 6))
    assert seam_lats.size >= 1, "Expected at least one shared seam latitude between adjacent tiles"
    seam_lat = seam_lats[0]

    north_lons = np.array(north.longitudes)
    south_lons = np.array(south.longitudes)
    seam_lons = np.intersect1d(np.round(north_lons, 6), np.round(south_lons, 6))
    assert seam_lons.size >= 1, "Expected shared seam longitudes between adjacent tiles"

    north_row = int(np.where(np.round(north_lats, 6) == seam_lat)[0][0])
    south_row = int(np.where(np.round(south_lats, 6) == seam_lat)[0][0])
    north_values = np.array(north.values, dtype=np.float32)
    south_values = np.array(south.values, dtype=np.float32)
    for seam_lon in seam_lons:
        north_col = int(np.where(np.round(north_lons, 6) == seam_lon)[0][0])
        south_col = int(np.where(np.round(south_lons, 6) == seam_lon)[0][0])
        assert north_values[north_row, north_col] == pytest.approx(
            south_values[south_row, south_col], abs=1e-6
        )


def test_adjacent_tiles_without_shared_native_grid_points_are_still_seam_continuous():
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0], dtype=np.int16)
    # 0.25-degree grid that does not generally align with WebMercator tile seams (e.g., lon=45).
    latitude = np.linspace(85.0, -85.0, 681, dtype=np.float32)
    longitude = np.linspace(-180.0, 180.0, 1441, dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    t_member0 = 8.0 + (lon_grid * 0.02) + (lat_grid * 0.01)
    t = t_member0[np.newaxis, np.newaxis, np.newaxis, :, :].astype(np.float32)
    zeros = np.zeros_like(t, dtype=np.float32)
    dataset = xr.Dataset(
        data_vars={
            "wind_u_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "wind_v_10m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), zeros),
            "temperature_2m": (("init_time", "lead_time", "ensemble_member", "latitude", "longitude"), t),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    adapter = ForecastDataAdapter(dataset)

    left_bounds = tile_xyz_bounds(z=3, x=4, y=3)
    right_bounds = tile_xyz_bounds(z=3, x=5, y=3)
    left = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, left_bounds, z=3)
    right = adapter.temperature_2m_member_tile("2026-01-01T00:00:00", 0, right_bounds, z=3)

    # Interpolated tile strategy should include seam longitudes exactly in both tiles.
    assert left.longitudes[-1] == pytest.approx(right.longitudes[0], abs=1e-6)
    left_values = np.array(left.values, dtype=np.float32)
    right_values = np.array(right.values, dtype=np.float32)
    for row in range(min(left_values.shape[0], right_values.shape[0])):
        assert left_values[row, -1] == pytest.approx(right_values[row, 0], abs=1e-4)
