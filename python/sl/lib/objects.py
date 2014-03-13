import numpy as np

from bisect import bisect
import re


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
    # _string_conversion maps a tuple (compiled regex, handler function) to
    # each string format that can be used to initialize a  NautAngle instance.
    # The handler function will receive the match object's groupdict as the
    # only argument and it needs to return the angle's value (unnormalized) as
    # a float prior to applying any sign encoded in a N/S/E/W pre-/post-fix.
    # All re assume that a N/S/E/W character indicatiing the sign has alread
    # been stripped out of the string and there is no leading or trailing
    # whitespace.
    _string_conversion = {}

    # as decimal degrees (ddd.ddd):
    def __read_deg_str(matchDict):
        sign = -1 if matchDict['sign'] == '-' else 1
        return float(matchDict['degrees']) * sign

    _string_conversion['deg'] = (
            re.compile(r'''(?P<sign>[+-]?)
                       [ ]*
                       (?P<degrees>[0-9]{1,3}(\Z|([.][0-9]*)\Z))
                       ''', re.VERBOSE),
            __read_deg_str)

    # as degrees and minutes (ddd mm.mmm):
    def __read_deg_min_str(matchDict):
        sign = -1 if matchDict['sign'] == '-' else 1
        return ((float(matchDict['degrees']) + float(matchDict['minutes'])
                / 60.) * sign)

    _string_conversion['deg_min'] = (
            re.compile(r'''(?P<sign>[+-]?)
                       [ ]*
                       (?P<degrees>[0-9]{1,3})
                       [ ]+
                       (?P<minutes>[0-5]?[0-9](\Z|([.][0-9]*)\Z))
                       ''', re.VERBOSE),
            __read_deg_min_str)

    # as degrees, minutes and seconds (ddd mm ss.sss):
    def __read_deg_min_sec_str(matchDict):
        sign = -1 if matchDict['sign'] == '-' else 1
        return ((float(matchDict['degrees']) + float(matchDict['minutes'])
                / 60. + float(matchDict['seconds']) / 3600.) * sign)

    _string_conversion['deg_min_sec'] = (
            re.compile(r'''(?P<sign>[+-]?)
                       [ ]*
                       (?P<degrees>[0-9]{1,3})
                       [ ]+
                       (?P<minutes>[0-5]?[0-9])
                       [ ]+
                       (?P<seconds>[0-5]?[0-9](\Z|([.][0-9]*\Z)))
                       ''', re.VERBOSE),
            __read_deg_min_sec_str)

    @staticmethod
    def normalize(degrees):
        return np.mod(degrees + 180., 360.) - 180.

    def __new__(cls, angle):
        try:
            strArg = isinstance(angle, basestring)  # Python 2
        except NameError:
            strArg = isinstance(angle, str)         # Python 3
        if strArg:
            _angle = angle
            name = 'N'
            for s in "NSEW":
                if s in angle.upper():
                    _angle = _angle.upper().replace(s, '').strip()
                    name = s
                    break
            sign = -1 if name in "SW" else 1
            for r in NautAngle._string_conversion.values():
                m = r[0].match(_angle)
                if m:
                    degrees = sign * r[1](m.groupdict())
                    break
            else:
                raise InvalidAngleFormat(
                        "Invalid string for initializing NautAngle "
                        "object: %s" % angle)
        else:
            degrees = angle

        return float.__new__(cls, NautAngle.normalize(degrees))

    @property
    def radians(self):
        return np.radians(self.real)

    # move to some prettyprint method if needed

    # def __str__(self):
    #     whole = int(self.real)
    #     minutes = (abs(self.real) - abs(whole)) * 60
    #     return "%d %07.4f" % (whole, minutes)

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
        return NautAngle(self.real + float(other))

    def __sub__(self, other):
        if isinstance(other, NautAngle):    # difference between two lats or
                                            # lons -> return a float
            return -self.distance_to(other)
        else:                               # float or int subtracted from self
                                            # -> return a NautAngle object
            return NautAngle(-self.distance_to(other))



class InvalidAngleFormat(Exception):
    """
    Indicates invalid string for intializing NautAngle object
    """


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
        assert -90. <= lat <= 90.  # in case anyone is crazy enough to sail
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
