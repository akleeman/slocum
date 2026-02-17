from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from slocum_api.data_access import ForecastDataAdapter


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    init_time = np.array(["2026-01-01T00:00:00", "2026-01-02T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0, 6, 12], dtype="timedelta64[h]")
    ensemble_member = np.arange(5, dtype=np.int16)
    latitude = np.linspace(90.0, -90.0, 9, dtype=np.float32)
    longitude = np.linspace(-180.0, 180.0, 13, endpoint=False, dtype=np.float32)

    shape = (init_time.size, lead_time.size, ensemble_member.size, latitude.size, longitude.size)
    rng = np.random.default_rng(seed=1234)
    u = rng.normal(loc=3.0, scale=2.0, size=shape).astype(np.float32)
    v = rng.normal(loc=4.0, scale=2.0, size=shape).astype(np.float32)
    t = rng.normal(loc=15.0, scale=8.0, size=shape).astype(np.float32)

    return xr.Dataset(
        data_vars={
            "wind_u_10m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                u,
            ),
            "wind_v_10m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                v,
            ),
            "temperature_2m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                t,
            ),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )


@pytest.fixture
def adapter(sample_dataset: xr.Dataset) -> ForecastDataAdapter:
    return ForecastDataAdapter(sample_dataset)


@pytest.fixture
def dataset_with_nan() -> xr.Dataset:
    """Dataset with explicit NaN in some cells to test propagation to tiles."""
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0, 1], dtype=np.int16)
    latitude = np.array([0.0, -10.0], dtype=np.float32)
    longitude = np.array([0.0, 10.0], dtype=np.float32)

    # At (lat=0, lon=0): both ensemble members have NaN -> p50 will be NaN
    # Other cells have valid data
    u = np.array(
        [
            [
                [
                    [[float("nan"), 2.0], [2.0, 2.0]],
                    [[4.0, 4.0], [4.0, 4.0]],
                ],
            ]
        ],
        dtype=np.float32,
    )
    u[0, 0, 1, 0, 0] = float("nan")
    v = np.zeros_like(u, dtype=np.float32)
    t = np.full_like(u, fill_value=1.0, dtype=np.float32)
    t[0, 0, :, 0, 0] = float("nan")
    return xr.Dataset(
        data_vars={
            "wind_u_10m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                u,
            ),
            "wind_v_10m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                v,
            ),
            "temperature_2m": (
                ("init_time", "lead_time", "ensemble_member", "latitude", "longitude"),
                t,
            ),
        },
        coords={
            "init_time": init_time,
            "lead_time": lead_time,
            "ensemble_member": ensemble_member,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
