import numpy as np
import pyproj
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from sl.lib import tinylib, units

beaufort_in_knots = tinylib._beaufort_scale * 1.94384449
beaufort_colors = [
          '#0328b3',# dark blue
          '#42b1e5',# lighter blue
          '#a1eeff',# light blue
          '#60fd4b',# green
          '#1cea00',# yellow-green
          '#fbef36',# yellow
          '#fbc136',# orange
          '#ff4f02',# red
          '#d50c02',# darker-red
          '#ff00c0',# red-purple
          '#d925ac',# purple
          '#b30d8a',# dark purple
          # '#000000', # black
          ]
wind_cmap = plt.cm.colors.ListedColormap(beaufort_colors, 'beaufort_map')
wind_cmap.set_over('0.25')
wind_cmap.set_under('0.75')

wind_norm = plt.cm.colors.BoundaryNorm(beaufort_in_knots, wind_cmap.N)


def bounded_map(lons, lats):
    """
    Produces a basemap which overs the domain spanned by
    lons and lats.
    """
    max_lat = np.nanmax(lats)
    min_lat = np.nanmin(lats)
    max_lon = np.nanmax(lons)
    min_lon = np.nanmin(lons)
    # the mid points are used as the center of the domain
    mid_lat = 0.5 * (max_lat + min_lat)
    mid_lon = 0.5 * (max_lon + min_lon)
    # compute the height and width
    geod = pyproj.Geod(ellps='WGS84')
    height = geod.inv(mid_lon, max_lat,
                      mid_lon, min_lat, radians=False)[2]
    width = geod.inv(max_lon, mid_lat,
                    min_lon, mid_lat, radians=False)[2]
    # actually build the basemap
    bm = Basemap(projection='laea', resolution='i',
                 lat_0=mid_lat, lon_0=mid_lon,
                 width=width * 1.1, height=height * 1.1)
    bm.drawcoastlines()
    bm.drawcountries()
    return bm


def wind_forecast(fcst):
    ba = wind_barbs(fcst)
    plt.colorbar(ba[0])
    plt.show()


def wind_pcolormesh(fcst):
    """
    Produces a plot of wind speeds using pcolormesh
    """
    fcst = fcst.indexed_by(time=0)
    lons = np.mod(fcst['longitude'].data, 360)
    lats = fcst['latitude'].data
    # Here we build a shifted grid for pcolormesh, which would
    # otherwise have plotted wind colors with a misleading offset.
    assert np.unique(np.diff(lons)).size == 1
    assert np.unique(np.diff(lats)).size == 1
    lon_diff = np.unique(np.diff(lons)) / 2.
    grid_lons = np.concatenate([lons - lon_diff, lons[-1] + lon_diff])
    lat_diff = np.unique(np.diff(lats)) / 2.
    grid_lats = np.concatenate([lats - lat_diff, lats[-1] + lat_diff])
    grid_lons, grid_lats = np.meshgrid(grid_lons, grid_lats)

    m = bounded_map(lons, lats)
    grid_xs, grid_ys = m(grid_lons, grid_lats)
    wind_speed = units.convert_units(fcst['wind_speed'], 'knot')
    return m.pcolormesh(grid_xs, grid_ys, wind_speed.data)


def wind_barbs(fcst):
    """
    Produces a plot of wind speeds using barbs
    """
    fcst = fcst.indexed_by(time=0)
    lons = np.mod(fcst['longitude'].data, 360)
    lats = fcst['latitude'].data
    lons, lats = np.meshgrid(lons, lats)
    wind_speed = units.convert_units(fcst['wind_speed'], 'knot')
    uwnd = units.convert_units(fcst['uwnd'], 'knot')
    vwnd = units.convert_units(fcst['vwnd'], 'knot')

    m = bounded_map(lons, lats)
    xs, ys = m(lons, lats)
    return m.barbs(xs, ys, uwnd.data, vwnd.data, wind_speed.data,
                   norm=wind_norm, cmap=wind_cmap)

