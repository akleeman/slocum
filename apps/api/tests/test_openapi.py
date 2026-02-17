from __future__ import annotations

from fastapi.testclient import TestClient

from slocum_api.app import create_app


def test_openapi_contains_core_paths():
    app = create_app()
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    assert "/meta/cycles" in schema["paths"]
    assert "/tiles/temperature_2m_member/{cycle}/{lead_hour}/{z}/{x}/{y}" in schema["paths"]
    assert "/point/wind" in schema["paths"]
    assert "/timeseries/wind" in schema["paths"]
    assert "/timeseries/wind/ensembles" in schema["paths"]
