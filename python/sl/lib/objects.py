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
        # Note to Alex: I moved cardinal directions to the center of the
        # respective bins, so that you don't get 'NNE' for 1°, or 'SSE' for
        # 91°, etc.  Also added intermediate directions because I need these
        # for the wind arrows in the route forecast.  Replaced np.arange by
        # np.linspace (more robust for ranges bounded by floats/float steps),
        # and used bisect to find the correct index (your diffs logic was
        # probably smarter but bisect seemed easier?) 
        # Here we bin according to the standard cardinal directions
        # 'S' is anything before first or after last bin
        bins = np.linspace(-15 * np.pi/16, 15 * np.pi/16, 16)
        names = ['S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N', 'NNE',
                'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S']
        self.readable = names[bisect(bins, self.dir)]

    # Note to Alex: Added nautDir method to get wind direction as 0..2*pi for
    # display purposes
    def nautDir(self):
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
        # Note to Alex: Does lat require modulo? Current assigment doesn't
        # handle lat==90 correctly
        self.lat = np.mod(lat + 90, 180) - 90
        self.lon = np.mod(lon, 360)

    def nautLatLon(self):
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
