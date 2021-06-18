import xarray as xra
import numpy as np
import matplotlib.pyplot as plt

from slocum.lib import angles
from slocum.query import variables
from slocum.query import utils as query_utils
from slocum.visualize import interactive, utils, velocity
from slocum.compression import compress, schemes


def test_data():
    ds = xra.Dataset()
    ds['time'] = ('time', np.arange(4),
                  {'units': 'hours since 2013-12-12 12:00:00'})
    ds['longitude'] = (('longitude'),
                       np.mod(np.arange(235., 240.) + 180, 360) - 180,
                       {'units': 'degrees east'})
    ds['latitude'] = ('latitude',
                      np.arange(35., 40.),
                      {'units': 'degrees north'})
    shape = tuple([ds.dims[x]
                   for x in ['time', 'longitude', 'latitude']])

    x, y = np.meshgrid(np.arange(-2, 3), np.arange(-2, 3))

    wind_mids = 0.5 * (variables.wind_bins[1:] +
                       variables.wind_bins[:-1])
    wind_speed = wind_mids[x * x + y * y]

    current_mids = 0.5 * (variables.current_bins[1:] +
                          variables.current_bins[:-1])
    current_speed = current_mids[x * x + y * y]

    dir = np.arctan2(y, x)

    current_speeds = np.empty(shape)
    wind_speeds = np.empty(shape)
    dirs = np.empty(shape)
    for i in range(ds.dims['time']):
        wind_speeds[i] = wind_speed
        current_speeds[i] = current_speed
        dirs[i] = dir + i * np.pi / 2

    uwnd, vwnd = angles.radial_to_vector(wind_speeds, dirs.copy(),
                                         orientation="from")
    ds['x_wind'] = (('time', 'longitude', 'latitude'),
                    uwnd, {'units': 'm/s'})
    ds['y_wind'] = (('time', 'longitude', 'latitude'),
                    vwnd, {'units': 'm/s'})

    ucurr, vcurr = angles.radial_to_vector(current_speeds, dirs.copy(),
                                           orientation="from")
    ds['sea_water_x_velocity'] = (('time', 'longitude', 'latitude'),
                    ucurr, {'units': 'm/s'})
    ds['sea_water_y_velocity'] = (('time', 'longitude', 'latitude'),
                    vcurr, {'units': 'm/s'})
    return xra.decode_cf(ds)


def main():
    fcst = test_data()
    fcst = compress.decompress_dataset(compress.compress_dataset(fcst))

    for vn in ['current', 'wind']:
        variable = query_utils.get_variable(vn)
        if isinstance(variable, schemes.VelocityVariable):
            interactive.InteractiveVelocity(fcst, variable)

    plt.show()


if __name__ == "__main__":
    import cProfile
    cProfile.runctx("main()", globals(), locals(), 'interactive.prof')
