import numpy as np

from bisect import bisect


class Wind(object):
    """
    An object which holds wind data for a single location,
    allowing you to query the zonal (u) and meridional (v)
    wind speeds, the vector wind speed (speed) and vector
    wind direction (dir).
    """
    def __init__(self, u=None, v=None):
        self.u = u
        self.v = v
        self.speed = np.sqrt(np.power(u, 2.) + np.power(v, 2.))
        # we need to use negative signs when computing the direction since
        # we actually want to know where the wind is coming from not where
        # the wind is headed
        self.dir = np.arctan2(-u, -v)
        # Here we bin according to the standard cardinal directions
        bins = np.arange(-np.pi, np.pi, step=np.pi / 4)
        names = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW']
        ind = bisect(bins, self.dir)
        if ind == 0 or self.dir > np.pi:
            self.readable = '-'
        else:
            self.readable = names[ind - 1]


class LatLon(object):
    """
    An object for holding latitudes and longitudes which
    standardizes the ranges such that lat in [-90, 90]
    and lon in [0, 360].
    """
    def __init__(self, lat, lon):
        self.lat = np.mod(lat + 90, 180) - 90
        self.lon = np.mod(lon, 360)

    def copy(self):
        return LatLon(self.lat, self.lon)

    def __add__(self, other):
        return LatLon(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other):
        return LatLon(self.lat - other.lat, self.lon - other.lon)
