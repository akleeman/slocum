from __future__ import annotations

import logging
import time
from collections.abc import Callable

from fastapi import Depends, FastAPI, HTTPException, Query, Response

from .config import get_settings
from .data_access import ForecastDataAdapter, get_adapter
from .models import (
    MetaCyclesResponse,
    PointWindResponse,
    TileResponse,
    TimeseriesWindEnsemblesResponse,
    TimeseriesWindResponse,
)
from .tilemath import tile_xyz_bounds

LOGGER = logging.getLogger(__name__)


def _is_transient_storage_error(exc: Exception) -> bool:
    text = str(exc).lower()
    markers = (
        "icechunkerror",
        "dispatch failure",
        "timed out",
        "connect timeout",
        "connection reset",
        "temporary failure",
        "service unavailable",
    )
    return any(marker in text for marker in markers)


def _run_with_retries(operation: Callable[[], object]) -> object:
    settings = get_settings()
    attempts = settings.transient_read_retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except ValueError:
            raise
        except Exception as exc:
            if not _is_transient_storage_error(exc):
                raise
            if attempt == attempts:
                raise
            delay = settings.transient_backoff_seconds * attempt
            LOGGER.warning(
                "Transient storage error (attempt %s/%s), retrying in %.2fs: %s",
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)


def create_app() -> FastAPI:
    app = FastAPI(title="Slocum Forecast API", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/meta/cycles", response_model=MetaCyclesResponse)
    def list_cycles(adapter: ForecastDataAdapter = Depends(get_adapter)) -> MetaCyclesResponse:
        cycles = adapter.list_cycles(max_cycles=14)
        return MetaCyclesResponse(
            latest_cycle=cycles[0],
            cycles=cycles,
            lead_hours=adapter.list_lead_hours(),
        )

    @app.get("/tiles/temperature_2m_member/{cycle}/{lead_hour}/{z}/{x}/{y}", response_model=TileResponse)
    def temperature_tile(
        cycle: str,
        lead_hour: int,
        z: int,
        x: int,
        y: int,
        response: Response,
        adapter: ForecastDataAdapter = Depends(get_adapter),
    ) -> TileResponse:
        try:
            bounds = tile_xyz_bounds(z=z, x=x, y=y)
            tile = _run_with_retries(
                lambda: adapter.temperature_2m_member_tile(
                    cycle=cycle,
                    lead_hour=lead_hour,
                    bounds=bounds,
                    z=z,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if _is_transient_storage_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Temporary data access timeout. Please retry.",
                ) from exc
            raise

        response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=60"
        return TileResponse(
            product="temperature_2m_member",
            cycle=cycle,
            lead_hour=lead_hour,
            z=z,
            x=x,
            y=y,
            bounds={
                "west": bounds.west,
                "south": bounds.south,
                "east": bounds.east,
                "north": bounds.north,
            },
            units=tile.units,
            latitudes=tile.latitudes,
            longitudes=tile.longitudes,
            values=tile.values,
        )

    @app.get("/point/wind", response_model=PointWindResponse)
    def point_wind(
        cycle: str = Query(...),
        lead: int = Query(..., ge=0),
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        adapter: ForecastDataAdapter = Depends(get_adapter),
    ) -> PointWindResponse:
        try:
            item = _run_with_retries(
                lambda: adapter.wind_point_summary(
                    cycle=cycle,
                    lead_hour=lead,
                    latitude=lat,
                    longitude=lon,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if _is_transient_storage_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Temporary data access timeout. Please retry.",
                ) from exc
            raise
        return PointWindResponse(
            cycle=cycle,
            lead_hour=lead,
            latitude=lat,
            longitude=lon,
            nearest_latitude=item.nearest_latitude,
            nearest_longitude=item.nearest_longitude,
            units=item.units,
            count=item.count,
            minimum=item.minimum,
            p10=item.p10,
            p50=item.p50,
            mean=item.mean,
            p90=item.p90,
            maximum=item.maximum,
        )

    @app.get("/timeseries/wind", response_model=TimeseriesWindResponse)
    def timeseries_wind(
        cycle: str = Query(...),
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        adapter: ForecastDataAdapter = Depends(get_adapter),
    ) -> TimeseriesWindResponse:
        try:
            item = _run_with_retries(
                lambda: adapter.wind_timeseries_summary(
                    cycle=cycle,
                    latitude=lat,
                    longitude=lon,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if _is_transient_storage_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Temporary data access timeout. Please retry.",
                ) from exc
            raise
        return TimeseriesWindResponse(
            cycle=cycle,
            latitude=lat,
            longitude=lon,
            nearest_latitude=item.nearest_latitude,
            nearest_longitude=item.nearest_longitude,
            units=item.units,
            lead_hours=item.lead_hours,
            p10=item.p10,
            p50=item.p50,
            mean=item.mean,
            p90=item.p90,
        )

    @app.get("/timeseries/wind/ensembles", response_model=TimeseriesWindEnsemblesResponse)
    def timeseries_wind_ensembles(
        cycle: str = Query(...),
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        adapter: ForecastDataAdapter = Depends(get_adapter),
    ) -> TimeseriesWindEnsemblesResponse:
        try:
            item = _run_with_retries(
                lambda: adapter.wind_ensemble_timeseries_summary(
                    cycle=cycle,
                    latitude=lat,
                    longitude=lon,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            if _is_transient_storage_error(exc):
                raise HTTPException(
                    status_code=503,
                    detail="Temporary data access timeout. Please retry.",
                ) from exc
            raise
        return TimeseriesWindEnsemblesResponse(
            cycle=cycle,
            latitude=lat,
            longitude=lon,
            nearest_latitude=item.nearest_latitude,
            nearest_longitude=item.nearest_longitude,
            units=item.units,
            lead_hours=item.lead_hours,
            ensemble_member_ids=item.ensemble_member_ids,
            values=item.values,
        )

    return app
