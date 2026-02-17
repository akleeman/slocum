from __future__ import annotations

from pydantic import BaseModel, Field


class MetaCyclesResponse(BaseModel):
    latest_cycle: str
    cycles: list[str]
    lead_hours: list[int]


class TileResponse(BaseModel):
    product: str
    cycle: str
    lead_hour: int
    z: int
    x: int
    y: int
    bounds: dict[str, float]
    units: str
    latitudes: list[float]
    longitudes: list[float]
    values: list[list[float]] = Field(description="2D row-major data grid")


class PointWindResponse(BaseModel):
    cycle: str
    lead_hour: int
    latitude: float
    longitude: float
    nearest_latitude: float
    nearest_longitude: float
    units: str
    count: int
    minimum: float
    p10: float
    p50: float
    mean: float
    p90: float
    maximum: float


class TimeseriesWindResponse(BaseModel):
    cycle: str
    latitude: float
    longitude: float
    nearest_latitude: float
    nearest_longitude: float
    units: str
    lead_hours: list[int]
    p10: list[float]
    p50: list[float]
    mean: list[float]
    p90: list[float]


class TimeseriesWindEnsemblesResponse(BaseModel):
    cycle: str
    latitude: float
    longitude: float
    nearest_latitude: float
    nearest_longitude: float
    units: str
    lead_hours: list[int]
    ensemble_member_ids: list[int]
    values: list[list[float]] = Field(description="2D array [ensemble_member][lead_hour]")
