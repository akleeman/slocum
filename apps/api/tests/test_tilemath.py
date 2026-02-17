from slocum_api.tilemath import stride_for_zoom, tile_xyz_bounds


def test_tile_xyz_bounds_world_tile():
    bounds = tile_xyz_bounds(z=0, x=0, y=0)
    assert round(bounds.west, 3) == -180.0
    assert round(bounds.east, 3) == 180.0
    assert bounds.north > 85.0
    assert bounds.south < -85.0


def test_stride_for_zoom_increases_for_lower_zoom():
    assert stride_for_zoom(10) == 1
    assert stride_for_zoom(6) == 6
    assert stride_for_zoom(3) == 12
    assert stride_for_zoom(0) == 24
