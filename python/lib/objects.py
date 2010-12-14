import numpy as np

from collections import namedtuple

from lib import iterlib

LatLon = namedtuple('LatLon', 'lat lon')
Course = namedtuple('Course', 'loc speed bearing')
Leg = namedtuple('Leg', 'course time wind distance')

class Wind():
    def __init__(self, u=None, v=None, speed=None, dir=None):
        self.u = u
        self.v = v
        self.speed = np.sqrt(np.power(u, 2.) + np.power(v, 2.))
        # we need to use negative signs when computing the direction since
        # we actually want to know where the wind is coming from not where
        # the wind is headed
        self.dir = np.arctan2(-u, -v)

class LatLon():
    def __init__(self, lat, lon):
        self.lat = np.mod(lat + 90, 180) - 90
        self.lon = np.mod(lon, 360)

    def as_rad(self):
        return LatLon(np.deg2rad(self.lat), np.deg2rad(self.lon))

    def as_deg(self):
        return LatLon(np.rad2deg(self.lat), np.rad2deg(self.lon))

    def __add__(self, other):
        import pdb; pdb.set_trace()
        return LatLon(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other):
        return LatLon(self.lat - other.lat, self.lon - other.lon)


class DataField():
    def __init__(self, ndarray, dims):
        if any(x != len(y) for x, y in zip(ndarray.shape, dims)):
            raise ValueError("dim lengths and array shape must agree")
        self.data = ndarray
        self.dims = [np.array(x) for x in dims]

    def __call__(self, *points):
        """
        @returns: the function value obtained by linear interpolation
        @rtype: number
        @raise TypeError: if the number of arguments (C{len(points)})
            does not match the number of variables of the function
        @raise ValueError: if the evaluation point is outside of the
            domain of definition and no default value is defined
        """
        if len(points) != len(self.dims):
            raise TypeError('Wrong number of arguments')

        def get_nhbrs(point, dim):
            if not point:
                return None
            j = int(np.sum(dim <= point))
            if j == 0:
                msg = "%6.3f not in [%6.3f, %6.3f]" % (point, np.min(dim), np.max(dim))
                raise ValueError(msg)
            i = j - 1
            if dim[i] == point:
                return (i, j), 1.0
            elif j == len(dim):
                msg = "%6.3f not in [%6.3f, %6.3f]" % (point, np.min(dim), np.max(dim))
                raise ValueError(msg)
            else:
                alpha = np.abs(point - dim[j])/np.abs(dim[i] - dim[j])
                return (i, j), alpha

        nhbrs = map(get_nhbrs, points, self.dims)
        ret = self.data
        for i, nhbr in enumerate(nhbrs):
            ret = ret.take(nhbr[0], axis=i)
        ret = ret.T
        for i, x in enumerate(nhbrs):
            ret = np.dot(ret, [nhbr[1], 1.-nhbr[1]])
            if ret.ndim == 2:
                # for some reason np.dot doesn't always reduce the dimension
                # so we need to do it manually from time to time
                ret = ret[..., 0]
        return ret

def test():
    testfile = os.path.join(os.path.dirname(__file__), '../data/analysis_20091201_v11l30flk.nc')
    nc = NetCDFFile(filename=testfile, mode='r')
    dims = [nc.variables['lat'], nc.variables['lon']]
    uwnd = objects.DataField(ncdflib.wind_variable(nc.variables['uwnd'])[0], dims)
    # test interpolation
    val1 = uwnd(-73.625, 184.125)
    val2 = uwnd(-73.375, 184.125)
    val3 = uwnd(-73.5, 184.125)
    assert val3 == 0.5 * (val2 + val1)
