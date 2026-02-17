from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from slocum_api.app import create_app
from slocum_api.data_access import ForecastDataAdapter, get_adapter


def _test_client(adapter: ForecastDataAdapter) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_adapter] = lambda: adapter
    return TestClient(app)


def test_meta_cycles(adapter: ForecastDataAdapter):
    client = _test_client(adapter)
    response = client.get("/meta/cycles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["latest_cycle"] == "2026-01-02T00:00:00"
    assert payload["lead_hours"] == [0, 6, 12]


def test_temperature_tile_returns_payload_and_cache_header(adapter: ForecastDataAdapter):
    client = _test_client(adapter)
    response = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/6/2/1/1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["product"] == "temperature_2m_member"
    assert payload["lead_hour"] == 6
    assert "Cache-Control" in response.headers
    assert len(payload["values"]) > 0


def test_temperature_tile_values_serializable_and_valid(adapter: ForecastDataAdapter):
    client = _test_client(adapter)
    response = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/6/2/1/1")
    assert response.status_code == 200
    payload = response.json()
    values = payload["values"]
    assert len(values) > 0
    for row in values:
        for cell in row:
            assert cell is not None, "Tile values must not contain null"
            assert isinstance(cell, (int, float)), "Tile values must be numbers"
            assert cell == cell, "Tile values must not be NaN"
            assert abs(cell) != float("inf"), "Tile values must be finite"


def test_temperature_tile_invalid_cycle_returns_400(adapter: ForecastDataAdapter):
    client = _test_client(adapter)
    response = client.get("/tiles/temperature_2m_member/2021-01-01T00:00:00/6/2/1/1")
    assert response.status_code == 400
    assert "Cycle not available" in response.json()["detail"]


def test_point_and_timeseries(adapter: ForecastDataAdapter):
    client = _test_client(adapter)
    point = client.get(
        "/point/wind",
        params={"cycle": "2026-01-01T00:00:00", "lead": 6, "lat": 20.0, "lon": 30.0},
    )
    assert point.status_code == 200
    assert point.json()["count"] == 5

    timeseries = client.get(
        "/timeseries/wind",
        params={"cycle": "2026-01-01T00:00:00", "lat": 20.0, "lon": 30.0},
    )
    assert timeseries.status_code == 200
    assert timeseries.json()["lead_hours"] == [0, 6, 12]

    ensemble_timeseries = client.get(
        "/timeseries/wind/ensembles",
        params={"cycle": "2026-01-01T00:00:00", "lat": 20.0, "lon": 30.0},
    )
    assert ensemble_timeseries.status_code == 200
    payload = ensemble_timeseries.json()
    assert payload["units"] == "kt"
    assert payload["lead_hours"] == [0, 6, 12]
    assert len(payload["ensemble_member_ids"]) == 5
    assert len(payload["values"]) == 5
    assert all(len(member_values) == 3 for member_values in payload["values"])


def test_temperature_tile_with_nan_dataset_succeeds(dataset_with_nan):
    """NaN in tile values is replaced with 0 before response, so serialization succeeds."""
    from slocum_api.data_access import ForecastDataAdapter

    adapter = ForecastDataAdapter(dataset_with_nan)
    client = _test_client(adapter)
    response = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/0/8/128/128")
    assert response.status_code == 200
    payload = response.json()
    assert all(cell is not None for row in payload["values"] for cell in row)


def test_transient_tile_error_retries_then_succeeds(adapter: ForecastDataAdapter, monkeypatch: pytest.MonkeyPatch):
    client = _test_client(adapter)
    call_count = {"value": 0}
    original = adapter.temperature_2m_member_tile

    def flaky(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("dispatch failure: connect timeout")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "temperature_2m_member_tile", flaky)
    response = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/6/2/1/1")
    assert response.status_code == 200
    assert call_count["value"] >= 2


def test_transient_tile_error_returns_503_after_retries(
    adapter: ForecastDataAdapter, monkeypatch: pytest.MonkeyPatch
):
    client = _test_client(adapter)

    def always_fails(*args, **kwargs):
        raise RuntimeError("dispatch failure: connect timeout")

    monkeypatch.setattr(adapter, "temperature_2m_member_tile", always_fails)
    response = client.get("/tiles/temperature_2m_member/2026-01-01T00:00:00/6/2/1/1")
    assert response.status_code == 503
    assert "Temporary data access timeout" in response.json()["detail"]
