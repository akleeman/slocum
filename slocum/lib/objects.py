from collections import namedtuple
from bisect import bisect

import numpy as np

from slocum.lib import units
from slocum.lib import conventions as conv

BoundingBox = namedtuple('BoundingBox', ['north', 'south', 'east', 'west'])
Position = namedtuple('Position', ['lat', 'lon'])

class NautAngle(float):

    """
    Subclasses float to represent a latitude or longitude angle.

    Angles will be normalized to fall into [-180, 180[ such that longitudes
    are centered on the Greenwich meridian (east longitude positive, west
    negative) and latitudes are centered on the equator (north latitude
    positive, south negative).

    NautAngle stores the angle in degrees and can be initialized by either a
    numerical value or a string. The following initializations all have the
    same result:

    >>> NautAngle(-45)
    -45.0
    >>> NautAngle("-45")
    -45.0
    >>> NautAngle("45S")
    -45.0
    >>> NautAngle("s 45")
    -45.0

    The value of a NautAangle object in radians is available as

    >>> NautAngle("s 45").radians
    -45.0


    NautAngle provides the following methods:
    -----------------------------------------

    distance_to(self, other)    - Returns a NautAngle object with the shortest
                                  angular distance between self and other.
                                  Positive east/north wards, negative the other
                                  way.
                                    >>> a = NautAngle(-45)
                                    >>> b = NautAngle(170)
                                    >>> a.distance_to(b)
                                    -145.0
                                    >>> b.distance_to(a)
                                    145.0

    is_east_of(self, other)     - True if self is east of other "the short way
                                  around"
    is_west_of(self, other)     - True if self is west of other "the short way
                                  around"

    is_north_of(self, other)
    is_south_od(self, other)    - Same logic as for east/west

                                  All four comparison methods return False is
                                  self and other have the same value.

    is_almost_equal(self, other, places=6)
                                - True if self is within places digits of other

    The following binary operators are defined for x [operator] y:

    __lt__      - True if x is west/south of y
    __le__      - True if x is west/south of or equal to y
    __eq__      - Same as x.is_almost_equal(y)
    __ne__      - Same as not x.is_almost_equal(y)
    __ge__      - True if x is east/north of or equal to y
    __gt__      - True if x is east/north of y

    __add__     - Returns (self + other) as a NautAngle object.
    __sub__     - Retruns x.distance_to(y) as a float if y is a NautAngle
                  object and as a NautAnlgel object is y is something else (int
                  or float)

    """

    @staticmethod
    def normalize(degrees):
        return np.mod(degrees + 180., 360.) - 180.

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
            degrees = sign * float(angle)
        else:
            degrees = angle

        return float.__new__(cls, NautAngle.normalize(degrees))

    @property
    def radians(self):
        return np.radians(self.real)

    def __str__(self):
        # keep simple so we can initialize a NautAngle object with
        # NautAngle(str(x)), even if we don't know whether x is a float or a
        # NautAngle object.
        # TODO: Change constructor to also work for strings like "ddd mm.mmm"
        #
        # whole = int(self.degrees)
        # minutes = (abs(self.degrees) - abs(whole)) * 60
        # return "%d %07.4f" % (whole, minutes)
        #
        return str(self.real)

    def __repr__(self):
        return str(self.real)

    def distance_to(self, other):
        """
        Returns a float value with the shortest angular distance between
        self and other. Positive if self is east of other, negative if self
        is west of other.
        """
        other = NautAngle.normalize(other)
        diff = NautAngle.normalize(other - self.real)
        if diff == -180 and other > self.real:
            return 180  # so distance from lat=-90 to lat=90 comes out correct
        else:
            return diff

    def is_east_of(self, other):
        """
        For longitude comparisons: Returns True if - at any given latitude -
        one has to travel east from other to self along the shortest route
        (e.g. 160 deg is east of 150 deg and -170 deg is east of 170 deg).
        False otherwise.
        """
        if self.distance_to(other).real < 0:
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
        if self.distance_to(other).real > 0:
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
        dist = self.distance_to(other)
        if dist < 0:
            return True
        elif dist > 0 and self.real > other.real:
            # 90,-90: dist wraps to -180
            return True
        else:
            return False

    def is_south_of(self, other):
        dist = self.distance_to(other).real
        if dist > 0:
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
        if isinstance(other, NautAngle):    # sum of two lats or lons
                                            # -> return a float
            return self.real + float(other)
        else:
            return NautAngle(self.real + float(other))

    def __sub__(self, other):
        if isinstance(other, NautAngle):    # difference between two lats or
                                            # lons -> return a float
            return -self.distance_to(other)
        else:                               # float or int subtracted from self
                                            # -> return a NautAngle object
            return NautAngle(-self.distance_to(other))

    def full_circle(self):
        """
        Returns angle as a float value in the [0, 360[ range.
        """
        return self.real % 360.

    def compass_dir(self):
        """
        Returns 'rounded' compass direction as a tuple (rounded angle, name)
        where name is a string 'N', 'NNE', 'NE', ... and rounded angle is the
        compass angle for the name in range [0..360[.

            >>> NautAngle(30).compass_dir()
            (22.5, 'NNE')
            >>> NautAngle(275).compass_dir()
            (270, 'W')
            >>> NautAngle(-85).compass_dir()
            (270, 'W')
        """
        # Bin according to the standard cardinal directions
        # 'S' is anything before first or after last bin
        bins = np.linspace(-15 * 180./16., 15 * 180./16., 16)
        names = ['S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'N', 'NNE',
                'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S']
        i = bisect(bins, self.real)
        j = (i-1) % 16
        i = i % 16
        if j > i:   # wrap around at S
            center_dir = np.average(np.mod([bins[i], bins[j]], 360.))
        else:
            center_dir = np.average(bins[j:i+1])

        return center_dir % 360., names[i]

    def named_str(self, kind, decimals=0, leading_0=False):
        """
        Returns angle as a sring, rounded to ``decimals`` decimals and with a
        'name' character appended, depending on the value of ``kind`` which
        must be either ``conv.LAT`` or ``conv.LON``.  If ``leading_0`` is
        True leading zeroes will be added as requied to get 2-digit
        latitudes and 3-digit longitudes.

        >>> NautAngle(-45.162).named_str(conv.LAT)
        '45S'
        >>> NautAngle(-45.162).named_str(conv.LON, decimals=1,
        ...                              leading_0=True)
        ...
        '045.2W'
        """
        if kind == conv.LAT:
            name = 'S' if self.real < 0 else 'N'
            width = 2
        elif kind == conv.LON:
            name = 'W' if self.real < 0 else 'E'
            width = 3
        else:
            raise(ValueError, "unknown kind: %s" % kind)

        if decimals:
            fmt = (".%df" % decimals)
            width += 1
        else:
            fmt = 'd'

        if leading_0:
            width += decimals
            fmt = "0%d%s" % (width, fmt)
        fmt = '%' + fmt + '%s'

        return fmt % (np.round(abs(self.real), decimals), name)



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
        __, name = NautAngle(np.degrees(self.dir)).compass_dir()
        self.readable = name

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
