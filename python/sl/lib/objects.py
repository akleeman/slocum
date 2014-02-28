import numpy as np

from bisect import bisect


class NautAngle(float):

    """
    Subclasses float to represent a latitude or longitude angle.

    Angles will be normalized to fall into [-pi, pi[ such that longitudes are
    centered on the Greenwich meridian (east longitude positive, west
    negative) and latitudes are centered on the equator (north latitude
    positive, south negative).

    NautAngle uses radians internally but its __str__ and __repr__ methods
    provide humanly readable representations in degrees. The following
    initializations all result in the same value for a NautAngle object:

    >>> NautAngle(np.radians(-45))
    -45.0
    >>> NautAngle("-45")
    -45.0
    >>> NautAngle("45S")
    -45.0
    >>> NautAngle("s 45")
    -45.0

    In all four cases the internal value of the NautAngle object will be
    -pi/4:

    >>> np.sin(NautAngle("s 45"))
    -0.70710678118654746

    The numerical value of a NautAangle object in degrees is available as

    >>> NautAngle("s 45").degrees
    -45.0
    >>> type(NautAngle("s 45").degrees)
    numpy.float64

    NautAngle provides the following methods:
    -----------------------------------------

    distance_to(self, other)    - Returns a NautAngle object with the shortest
                                  angular distance between self and other.
                                  Positive if self is east/north of other,
                                  negative if self is west/south of other.
                                    >>> a = NautAngle("-45")
                                    >>> b = NautAngle("170")
                                    >>> a.distance_to(b)
                                    145.0
                                    >>> b.distance_to(a)
                                    -145.0

    is_east_of(self, other)     - True if self is east of other "the short way
                                  around"
    is_west_of(self, other)     - True if self is west of other "the short way
                                  around"

    is_north_of(self, other)
    is_south_od(self, other)    - Same logic as for east/west

                                  All four comparison methods return True is
                                  self and other have the same value.

    is_almost_equal(self, other, places=6)
                                - True if self is within places digits of other

    The following binary operators are defined for x [operator] y:

    __lt__      - True if x is west/south of y
    __le__      - True if x is west/south of or equal to y
    __eg__      - Same as x.is_almost_equal(y)
    __ne__      - Same as not x.is_almost_equal(y)
    __ge__      - True if x is east/north of or equal to y
    __gt__      - True if x is east/north of y

    __add__     - Returns (self = other) as a NautAngle object.
    __sub__     - Same as x.distance_to(y)

    """

    @staticmethod
    def normalize(rad):
        return np.mod(rad + np.pi, 2 * np.pi) - np.pi

    def __new__(cls, angle):
        try:
            strArg = isinstance(angle, basestring)  # Python 2
        except NameError:
            strArg = isinstance(angle, str)         # Python 3
        if strArg:
            name = 'N'
            for s in "NSEW":
                if s in angle.upper():
                    angle = angle.upper().replace(s, '')
                    name = s
                    break
            sign = -1 if name in "SW" else 1
            rad = np.radians(sign * float(angle))
        else:
            rad = angle

        return float.__new__(cls, NautAngle.normalize(rad))

    @property
    def degrees(self):
        return np.degrees(self.real)

    def __str__(self):
        whole = int(self.degrees)
        minutes = (abs(self.degrees) - abs(whole)) * 60
        return "%d %07.4f" % (whole, minutes)

    def __repr__(self):
        return str(np.degrees(self.real))

    def distance_to(self, other):
        """
        Returns a NautAngle object with the shortest angular distance between
        self and other. Positive if self is east of other, negative if self
        is west of other.
        """
        diff = self.real - NautAngle.normalize(other)
        return NautAngle(diff % (2 * np.pi))

    def is_east_of(self, other):
        """
        For longitude comparisons: Returns True if - at any given latitude -
        one has to travel east from other to self along the shortest route
        (e.g. 160 deg is east of 150 deg and -170 deg is east of 170 deg).
        False otherwise.
        """
        if self.distance_to(other).real > 0:
            return True
        else:
            return False

    def is_west_of(self, other):
        """
        For longitude comparisons: Returns True if - at any given latitude -
        one has to travel west from other to self along the shortest route
        (e.g. 20 deg is west of 150 deg and 170 deg is west of -170 deg).
        False otherwise.
        """
        if self.distance_to(other).real < 0:
            return True
        else:
            return False

    def is_almost_equal(self, other, places=6):
        """
        Returns True if self and other are equal to within places digits.
        """
        diff = self.real - NautAngle.normalize(other)
        return abs(diff) < 10**-places

    def is_north_of(self, other):
        dist = self.distance_to(other).real
        if dist > 0:
            return True
        elif dist < 0 and self.real > other.real:
            # 90,-90: dist wraps to -180
            return True
        else:
            return False

    def is_south_of(self, other):
        dist = self.distance_to(other).real
        if dist < 0:
            return True
        else:
            return False

    def __lt__(self, other):
        return self.is_west_of(other)

    def __gt__(self, other):
        return self.is_east_of(other)

    def __le__(self, other):
        return self.is_west_of(other) or self.is_almost_equal(other)

    def __eq__(self, other):
        return self.is_almost_equal(other)

    def __ne__(self, other):
        return not self.is_almost_equal(other)

    def __ge__(self, other):
        return self.is_east_of(other) or self.is_almost_equal(other)

    def __add__(self, other):
        return NautAngle(self.real + float(other))

    def __sub__(self, other):
        return self.distance_to(other)


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
    and lon in [-180, 180[.
    """
    def __init__(self, lat, lon):
        assert -90. <= lat <= 90. # in case anyone is crazy enough to sail
                                  # there and causes floating-point issues
        self.lat = lat
        self.lon = np.mod(lon + 180, 360) - 180

    def nautical_latlon(self):
        # TODO: eliminate, no longer needed
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
