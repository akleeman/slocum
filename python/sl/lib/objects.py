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
        # 'S' is anything before first or after last bin
        bins = np.linspace(-15 * np.pi/16, 15 * np.pi/16, 16)
        names = ['S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N', 'NNE',
                'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S']
        self.readable = names[bisect(bins, self.dir)]

    def nautical_dir(self):
        """
        Returns self.dir mapped onto 0..2*pi range
        """
        if self.dir >= 0:
            return self.dir
        else:
            return 2 * np.pi + self.dir


class LatLon(object):
    """
    An object for holding latitudes and longitudes which
    standardizes the ranges such that lat in [-90, 90]
    and lon in [0, 360].
    """
    def __init__(self, lat, lon):
        assert -90. <= lat <= 90. # in case anyone is crazy enough to sail
                                  # there and causes floating-point issues
        self.lat = lat
        self.lon = np.mod(lon, 360)

    def nautical_latlon(self):
        """
        Returns a (lat, lon) tuple with lon in 'nautical' range of -180..180
        (OpenCPN doesn't handle waypoints with lon > 180 correctly).
        """
        if self.lon > 180: 
            return (self.lat, self.lon - 360)
        else:
            return (self.lat, self.lon)

    def copy(self):
        return LatLon(self.lat, self.lon)

    def __add__(self, other):
        return LatLon(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other):
        return LatLon(self.lat - other.lat, self.lon - other.lon)
