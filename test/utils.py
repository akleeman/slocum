
import xarray as xra
import numpy as np

from slocum.query import variables
from slocum.compression import schemes


def create_data():
    ds = xray.Dataset()
    ds['time'] = ('time', np.arange(10),
                  {'units': 'hours since 2013-12-12 12:00:00'})
    ds['longitude'] = (('longitude'),
                       np.mod(np.arange(235., 240.) + 180, 360) - 180,
                       {'units': 'degrees east'})
    ds['latitude'] = ('latitude',
                      np.arange(35., 40.),
                      {'units': 'degrees north'})
    shape = tuple([ds.dims[x]
                   for x in ['time', 'longitude', 'latitude']])
    beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                                33., 40., 47., 55., 63., 75.]) / 1.94384449
    mids = 0.5 * (beaufort_scale[1:] + beaufort_scale[:-1])
    speeds = mids[np.random.randint(mids.size, size=10 * 5 * 5)]
    speeds = speeds.reshape(shape)

    dirs = np.linspace(-7 * np.pi / 8, np.pi, 16)
    dirs = dirs[np.random.randint(dirs.size, size=10 * 5 * 5)]
    dirs = dirs.reshape(shape)
    # the directions were chosen to be direction from
    uwnd = - speeds * np.sin(dirs)
    uwnd = uwnd.reshape(shape).astype(np.float32)
    vwnd = - speeds * np.cos(dirs)
    vwnd = vwnd.reshape(shape).astype(np.float32)

    ds['x_wind'] = (('time', 'longitude', 'latitude'),
                    uwnd, {'units': 'm/s'})
    ds['y_wind'] = (('time', 'longitude', 'latitude'),
                    vwnd, {'units': 'm/s'})

    ds = add_tiny_variable(variables.pressure, ds)

    return xray.decode_cf(ds)


def add_tiny_variable(variable, ds):
    # generates random data for a tiny variable such that all the
    # values fall in the middle of bins.  This way roundtrips
    # should come out equivalent.
    variable.bins
    mids = 0.5 * (variable.bins[1:] + variable.bins[:-1])
    data = mids[np.random.randint(mids.size, size=ds['x_wind'].size)]
    ds[variable.variable_name] = (ds['x_wind'].dims,
                                  data.reshape(ds['x_wind'].shape),
                                  {'units': variable.units})
    return ds


def create_ensemble_data():
    ds = xray.concat([create_data() for i in range(21)], dim='realization')
    return ds.transpose(*schemes._default_dim_order)


def create_gfs_data():
    ds = xray.Dataset()
    ds['time'] = ('time', np.arange(0, 120, 3),
                  {'units': 'hours since 2013-12-12 12:00:00'})
    ds['longitude'] = (('longitude'),
                       np.mod(np.arange(0., 360.) + 180, 360) - 180,
                       {'units': 'degrees east'})
    ds['latitude'] = ('latitude',
                      np.arange(65, -66, -1),
                      {'units': 'degrees north'})
    shape = tuple([ds.dims[x]
                   for x in ['time', 'longitude', 'latitude']])
    size = reduce(np.multiply, shape)
    beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                                33., 40., 47., 55., 63., 75.]) / 1.94384449
    mids = 0.5 * (beaufort_scale[1:] + beaufort_scale[:-1])
    speeds = mids[np.random.randint(mids.size, size=size)]
    speeds = speeds.reshape(shape)

    dirs = np.linspace(-7 * np.pi / 8, np.pi, 16)
    dirs = dirs[np.random.randint(dirs.size, size=size)]
    dirs = dirs.reshape(shape)
    # the directions were chosen to be direction from
    uwnd = - speeds * np.sin(dirs)
    uwnd = uwnd.reshape(shape).astype(np.float32)
    vwnd = - speeds * np.cos(dirs)
    vwnd = vwnd.reshape(shape).astype(np.float32)

    ds['ugrd10m'] = (('time', 'longitude', 'latitude'),
                     uwnd, {'units': 'm/s'})
    ds['vgrd10m'] = (('time', 'longitude', 'latitude'),
                     vwnd, {'units': 'm/s'})

    return xray.decode_cf(ds)
