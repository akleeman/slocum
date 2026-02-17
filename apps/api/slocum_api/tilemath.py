from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TileBounds:
    west: float
    south: float
    east: float
    north: float


def tile_xyz_bounds(z: int, x: int, y: int) -> TileBounds:
    n = 2**z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0

    lat_top_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat_bottom_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 1) / n)))
    lat_top = math.degrees(lat_top_rad)
    lat_bottom = math.degrees(lat_bottom_rad)
    return TileBounds(west=lon_left, south=lat_bottom, east=lon_right, north=lat_top)


def stride_for_zoom(z: int) -> int:
    if z <= 2:
        return 24
    if z <= 4:
        return 12
    if z <= 6:
        return 6
    if z <= 8:
        return 3
    return 1
