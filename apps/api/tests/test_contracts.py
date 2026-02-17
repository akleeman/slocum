from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi.testclient import TestClient

from slocum_api.app import create_app
from slocum_api.data_access import ForecastDataAdapter, get_adapter


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_json(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _contract_dataset() -> xr.Dataset:
    init_time = np.array(["2026-01-01T00:00:00"], dtype="datetime64[s]")
    lead_time = np.array([0], dtype="timedelta64[h]")
    ensemble_member = np.array([0, 1], dtype=np.int16)
    latitude = np.array([0.0, -10.0], dtype=np.float32)
    longitude = np.array([0.0, 10.0], dtype=np.float32)

    # p50 speed at (lat=0, lon=0) -> median([2, 4]) == 3
    u = np.array([[[[[2.0, 2.0], [2.0, 2.0]], [[4.0, 4.0], [4.0, 4.0]]]]], dtype=np.float32)
    v = np.zeros_like(u, dtype=np.float32)
    # member 0 temperature in Celsius
    t = np.array([[[[[1.0, 1.0], [1.0, 1.0]], [[20.0, 20.0], [20.0, 20.0]]]]], dtype=np.float32)
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


def test_meta_and_tile_contract_fixtures():
    adapter = ForecastDataAdapter(_contract_dataset())
    app = create_app()
    app.dependency_overrides[get_adapter] = lambda: adapter
    client = TestClient(app)

    meta_actual = client.get("/meta/cycles").json()
    meta_expected = _load_json("meta_cycles_expected.json")
    assert meta_actual == meta_expected

    tile_actual = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/0/10/512/512").json()
    assert tile_actual["product"] == "temperature_2m_member"
    assert tile_actual["cycle"] == "2026-01-01T00:00:00"
    assert tile_actual["lead_hour"] == 0
    assert tile_actual["units"] == "degC"
    values = np.array(tile_actual["values"], dtype=np.float32)
    assert values.size > 0
    assert np.all(np.isfinite(values))
    # Contract dataset has constant 1C for member 0.
    assert np.allclose(values, 1.0, atol=1e-5)
