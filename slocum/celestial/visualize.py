import pyproj
import numpy as np

from mpl_toolkits import basemap

from slocum.lib import units
from slocum.visualize.utils import bounding_box, axis_figure


def get_basemap(lons, lats, pad=0.1, lat_pad=None, lon_pad=None, **kwdargs):
    kwdargs['projection'] = kwdargs.get('projection', 'cyl')
    kwdargs['resolution'] = kwdargs.get('resolution', 'i')
    bm_args = bounding_box(lons, lats, pad=pad, lat_pad=lat_pad, lon_pad=lon_pad)
    bm_args.update(kwdargs)
    # explicitly specify axis, even if its just the gca.
    bm_args['ax'] = axis_figure(bm_args.get('ax', None))[0]
    m = basemap.Basemap(**bm_args)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents()
    m.ax.set_axis_bgcolor('#389090')

    
    lat_min = np.floor(bm_args['llcrnrlat'])
    lat_max = np.ceil(bm_args['urcrnrlat'])
    m.drawparallels(np.arange(lat_min, lat_max + 1),
                    labels=[1, 0, 0, 0])
    
    lon_min = np.floor(bm_args['llcrnrlon'])
    lon_max = np.ceil(bm_args['urcrnrlon'])
    m.drawmeridians(np.arange(lon_min, lon_max + 1),
                    labels=[0, 0, 0, 1], rotation=90)

    return m


def plot_line_of_position(lon, lat, z, bm=None):

    geod = pyproj.Geod(ellps="sphere")

    def line(x):
        dist_m = units.convert_scalar(x, 'nautical_mile', 'm')
        new_lon, new_lat, _ = geod.fwd(lon, lat, z + 90., dist_m)
        return new_lon, new_lat
    deltas = np.linspace(-30., 30., 101)

    line_lons, line_lats = map(np.array, zip(*[line(x) for x in deltas]))

    if bm is None:
        bm = get_basemap(line_lons, line_lats, pad=1.)

    xs, ys = bm(line_lons, line_lats)
    bm.plot(xs, ys)
    return bm
