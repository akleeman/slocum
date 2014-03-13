"""
Utilities for combining a route and a grib forecast into a set of waypoints
with forecast info along the route.

Provides the following classes (see individual doc strings for more details):

    Route           -   Reads a route as a series of waypoints from a file and
                        provides methods to advance the boats position along
                        that route.
    RouteForecast   -   Combines a Route object with a polyglot.Dataset
                        containing forecast data and provides methods to
                        extract forecast data along the route and export the
                        result as waypoints to a gpx file (can be opened on
                        OpenCPN as temorary layer).  Wind data will be encoded
                        in waypoint names.
    FcstWP          -   Convenience wrapper to store forecast data along
                        waypoints.

Utility functions (available at module level):

    interpol        -   Given a point's lat/lon and the values of some variable
                        at 2x2 grid points around the point interpol will
                        returns the variable value at the point (bilinear
                        interpolation).
    appWind         -   Calculates apparent wind angle and speed from true wind
                        direction and speed, plus boat's COG and SOG.
    trueWindAngle   -   Calculates true wind angle given true wind direction
                        and speed plus boat COG and SOG.
"""

import datetime as dt
from collections import namedtuple
import csv
import logging
from bisect import bisect, bisect_left, bisect_right
import xml.etree.ElementTree as ET
import numpy as np
from netCDF4 import num2date
from pyproj import Geod

from objects import Wind, NautAngle, Position, BoundingBox, LatLon

logger = logging.getLogger(__name__)
if not __name__ == '__main__':
    logger.addHandler(logging.NullHandler())

# Skeleton OpenCPN gpx file (used by exportFcstGPX):
GPX_WRAPPER = """<?xml version="1.0" encoding="utf-8" ?>
<gpx version="1.1" creator="OpenCPN" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd" xmlns:opencpn="http://www.opencpn.org">
</gpx>
"""

XML_HEADER = '<?xml version="1.0" encoding="utf-8" ?>'  # used by exportFcstGPX
                                                        # as first output line

# OpenCPN user icons for wind arrow way point symbols are assumed to named:
# W_ARROW_ICON_PREFIX + [cardinal direction], where cardinal direction is
# provided by Wind.readable
W_ARROW_ICON_PREFIX = 'warr_'


class Route(object):

    """
    Holds a route in list rtePts of LatLon objects and associated speeds (in
    m/s) to the next respective waypoint.  Also maintains a current position in
    curPos and two prevWP and nextWP indices that point to the elements of
    rtePts that bracket curPos.  Exports a method getPos(utc) that will return
    the position along the route at time utc as a LatLon object.

    The following format specifiers for the route input file are recognized by
    the constructor:

        'csv'
            csv file with lat, lon in decimal degrees, followed by speed in kn
            to the next waypoint; speed can be empty if avrgSpeed is passed to
            route constructor; if speed is provided for any leg it will
            overwrite avrgSpeed for that leg

    """
    # TODO: add parsers for different import formats

    RtePoint = namedtuple('RtePoint', ['wp', 'speed'])

    geod = Geod(ellps='WGS84')

    def __init__(self, ifh=None, inFmt=None, utcDept=None, avrgSpeed=None):
        """
        Constructs a route object from the data in file object ifh which must
        be in format inFmt (see class doc string for valid formats).  utcDept
        is a datetime object specifying departure from the first waypoint.  If
        avrgSpeed in knots is provided, it will be substituted for any missing
        leg speeds in the input.  If no ifh is provided the route object will
        be constructed with an empty rtePts list.  If no avrgSpeed > 0 is
        provided and the input file contains missing leg speeds an
        'InvalidInFileError' will be raised.  Note: longitudes in rtePts will
        be in range 0..360 (see objects.LatLon).  Use nautical_latlon() method
        of objects.LatLon to get a (lat, lon) tuple with lon in range
        -180..180.
        """
        self.utcDept = utcDept
        self.rtePts = []

        if ifh is None:     # in case we want to roll our own
            return

        if inFmt == 'csv':
            self.read_csv(ifh, avrgSpeed)
        else:
            raise (InvalidInFormatError,
                   "Invalid route file format specifier: %s" % inFmt)

        self.ur, self.ll = self.updateBBox()

        self.resetCurPos()
        self.updateUtcArrival()

    def updateBBox(self):
        """
        Determines bounding box for route based on current lat/lon values in
        self.rtePts, defined as ur and ll corners. Returns (ur, ll) tuple.  If
        self.ur and self.ll exist they will be updated with the new values.
        """
        ll = LatLon(90, 359.999999)

        ur = LatLon(-90, 0)

        for p in self.rtePts:
            if p.wp.lat < ll.lat: ll.lat = p.wp.lat
            if p.wp.lat > ur.lat: ur.lat = p.wp.lat
            if p.wp.lon < ll.lon: ll.lon = p.wp.lon
            if p.wp.lon > ur.lon: ur.lon = p.wp.lon

        if hasattr(self, 'ur'): self.ur = ur.copy()
        if hasattr(self, 'll'): self.ll = ll.copy()

        return ur, ll

    def updateUtcArrival(self):
        """
        Updates time of arrival based on self.rtePts and self.utcDept.
        Returns tuple (tdist, ttime) with total route distance in meter and
        total time en route in seconds.
        """
        if len(self.rtePts) < 2:
            self.utcArrival = self.utcDept
            return (0., 0.)

        ttime = 0.                          # total time in seconds
        tdist = 0.                          # total distance in meters
        for cur, nxt in zip(self.rtePts[:-1], self.rtePts[1:]):

            az12, az21, dist = Route.geod.inv(cur.wp.lon, cur.wp.lat,
                                              nxt.wp.lon, nxt.wp.lat)
            tdist += dist
            ttime += float(dist) / cur.speed

        self.utcArrival = self.utcDept + dt.timedelta(0, int(round(ttime)))

        return (tdist, ttime)

    def getPos(self, utc):
        """
        Returns a tuple (LatLon, course, speed) with the boat's position as a
        LatLon object, course over ground in deg true, and SOG in m/s along the
        route at time utc, a dateime object.  Does not update self.curPos.
        Raises TimeOverlapError if utc is before self.utcDept or after
        self.utcArrival of arrival.
        """
        if utc < self.utcDept:
            raise (TimeOverlapError,
                   "position requested for time before departure")

        self.updateUtcArrival()     # just in case some one messed with the
                                    # waypoint list
        if utc > self.utcArrival:
            raise TimeOverlapError, "position requested for time after arrival"

        deltaT = utc - self.utcDept
        origCurPos = self.curPos
        origWP = (self.prevWP, self.nextWP)
        self.curPos = self.rtePts[0].wp.copy()
        self.prevWP, self.nextWP = (0, 1)
        # TODO: If any exception from advanceCurPos is handled somewhere, the
        # handler needs to restore the orig values
        course, speed, deltaT = self.advanceCurPos(deltaT)
        tmpCurPos = self.curPos
        self.curPos = origCurPos
        self.prevWP, self.nextWP = origWP

        return tmpCurPos, course, speed

    def advanceCurPos(self, deltaT):
        """
        Advances self.curPos by deltaT (a datetime.timedelta object), based on
        the speeds in self.rtePts.  Returns a tuple (course, speed, time left
        over) with the boat's course in deg true, SOG in m/s and the 'unused'
        portion of deltaT (if any, in case the last WP was reached) as a
        timedelta object (0 if the new position is still before the last WP).
        """
        logger.debug(
            "advanceCurPos called with deltaT: %d sec, curPos: "
            "%.3f %.3f, prevWP: %d, nextWP: %d" %
            (deltaT.total_seconds(), self.curPos.lat, self.curPos.lon,
            self.prevWP, self.nextWP))

        if self.nextWP is None:     # we're already at the end of the route
            return None, None, deltaT
        elif deltaT.total_seconds() <= 0:  # utcDept coincides with
                                           # forecast time
            az12, __, __ = Route.geod.inv(self.curPos.lon,
                                    self.curPos.lat,
                                    self.rtePts[self.nextWP].wp.lon,
                                    self.rtePts[self.nextWP].wp.lat)
            logger.debug("exiting advanceCurPos with deltaT: % d sec, "
                    "curPos: % .3f % .3f, prevWP: % d, nextWP: % d" %
                        (deltaT.total_seconds(), self.curPos.lat,
                            self.curPos.lon, self.prevWP, self.nextWP))
            return az12, self.rtePts[self.prevWP].speed, deltaT

        while deltaT.total_seconds() > 0:

            az12, __, distToNext = Route.geod.inv(self.curPos.lon,
                                            self.curPos.lat,
                                            self.rtePts[self.nextWP].wp.lon,
                                            self.rtePts[self.nextWP].wp.lat)

            # TODO: agree on what's 'at the WP' (100m)?
            if distToNext < 100:
                self.curPos = self.rtePts[self.nextWP].wp.copy()
                self.__advancePrevNext()
                continue

            tToNext = dt.timedelta(0, distToNext /
                                   self.rtePts[self.prevWP].speed)

            if deltaT < tToNext:    # new curPos in this iteration
                dist = deltaT.total_seconds() * self.rtePts[self.prevWP].speed
                self.curPos.lon, self.curPos.lat, __ = (
                    Route.geod.fwd(self.curPos.lon, self.curPos.lat, az12,
                                   dist))
                deltaT = dt.timedelta(0)
                logger.debug("exiting advanceCurPos with deltaT: % d sec, "
                        "curPos: % .3f % .3f, prevWP: % d, nextWP: % d" %
                            (deltaT.total_seconds(), self.curPos.lat,
                                self.curPos.lon, self.prevWP, self.nextWP))
                return az12, self.rtePts[self.prevWP].speed, deltaT
            else:                   # move to next WP
                deltaT -= tToNext
                self.curPos = self.rtePts[self.nextWP].wp.copy()
                prevWP, nextWP = self.__advancePrevNext()
                if nextWP is None:
                    logger.debug("exiting advanceCurPos with deltaT: % d sec,"
                            " curPos: % .3f % .3f, prevWP: % d, nextWP: None"
                            % (deltaT.total_seconds(), self.curPos.lat,
                                     self.curPos.lon, self.prevWP))
                    return None, None, deltaT

    def resetCurPos(self):
        """
        Resets self.curPos to first wp and prevWP and nextWP to 0 and 1
        respectively (or nextWP = None if route has only one wp).
        """
        if len(self.rtePts) > 0:
            self.curPos = self.rtePts[0].wp.copy()
            self.prevWP = 0
            if len(self.rtePts) > 1:
                self.nextWP = 1
                if not self.utcDept is None:
                    self.updateUtcArrival()
                else:
                    self.utcArrive = self.utcDept
            else:
                self.nextWP = None
        else:
            self.curPos = None
            self.prevWP = None
            self.nextWP = None

    def __advancePrevNext(self):
        """
        Advances the prevWP and nextWP indices along self.rtePts.  If prevWP
        reaches the end of the route nextWP will be set to None.  Returns a
        (prevWP, nextWP) tuple.
        """
        if self.nextWP is not None:
            self.prevWP = self.nextWP
            if self.nextWP == len(self.rtePts) - 1:   # already at the end
                self.nextWP = None
            else:
                self.nextWP += 1
        return (self.prevWP, self.nextWP)

    def read_csv(self, ifh, avrgSpeed):
        """
        Reads route csv file object ifh with lat, lon (both in decimal degrees,
        - = S/W) and speed (in knots) to next wp as fields for each record.  No
        header line.  If the file contains records for which the speed field is
        empty or <= 0 avrgSpeed will be substituted.  If neither avrgSpeed nor
        the speed field in a record are > 0, InvalidInFileError will be raised.
        On return self.rtePts will be populated with waypoints and speeds
        (converted to  m/s) from ifh.
        """
        rte = csv.reader(ifh, skipinitialspace=True)
        for r in rte:
            if r[2] == '' or not float(r[2]) > 0:
                if not avrgSpeed > 0:
                    raise InvalidInFileError
                else:
                    r[2] = avrgSpeed
            try:
                speed = units.normalize_scalar(float(r[2]), 'knot')
                self.rtePts.append(Route.RtePoint(LatLon(float(r[0]),
                                   float(r[1])), speed))
            except:
                raise InvalidInFileError


class FcstWP(LatLon):

    """
    Convenience wrapper to store forecast data at waypoints. Subclasses LatLon
    and adds Wind as an attribute, plus utc (datetime object), cog (true, in
    radians), and sog (m/s).
    """
    def __init__(self, lat, lon, u, v, utc=None, cog=None, sog=None):
        LatLon.__init__(self, lat, lon)
        self.wind = Wind(u, v)
        self.utc = utc
        self.cog = cog
        self.sog = sog


class RouteForecast(object):

    """
    Initialized with a Route and a polyglot.Dataset object containing the
    forecast data, method genRteFcst of RouteForecast will generate a wind
    forecast along the Route for all forecast times provided in the Dataset
    that overlap with the travel time.  The result can be written to a gpx
    waypoint file wich can be displayed in OpenCPN as a temroary layer.
    """

    def __init__(self, rte, fcst):
        """
        rte:    rtefcst.Route object
        fcst:   polyglot.Dataset object
        """
        fTimes = num2date(fcst['time'].data[:],
                          fcst['time'].attributes['units'])
        # check that forecast times overlap travel time
        if fTimes[0] > rte.utcArrival or fTimes[-1] < rte.utcDept:
            raise (TimeOverlapError,
                   "Forecast times do not overlap with route times")

        # check that forecast area overlaps route
        if (rte.ur.lat < min(fcst['latitude'].data) or rte.ll.lat >
                max(fcst['latitude'].data)):
            raise (RegionOverlapError,
                   "Route latitudes outside of forecast region")
        if (rte.ur.lon < min(fcst['longitude'].data) or rte.ll.lon >
                max(fcst['longitude'].data)):
            raise (RegionOverlapError,
                   "Route longitudes outside of forecast region")

        self.rte = rte
        self.fcst = fcst
        self.fTimes = fTimes

    def genRteFcst(self):
        """
        Returns a list with FcstWP objects for the temporal and regional
        overlap between self.rte and self.fcst
        """
        # time index of first forecast >= utcDept:
        i_t0 = bisect_left(self.fTimes, self.rte.utcDept)
        # time index of first forecast after utcArrival:
        i_tn = bisect_right(self.fTimes, self.rte.utcArrival)

        logger.debug("genRteFcst: it_0, i_tn: %d, %d" % (i_t0, i_tn))

        self.rte.resetCurPos()
        rteFcst = []
        for i in range(i_t0, i_tn):

            if i == i_t0:  # advance position to first overlapping forecast time
                deltaT = self.fTimes[i] - self.rte.utcDept
            else:
                deltaT = self.fTimes[i] - self.fTimes[i - 1]

            logger.debug("genRteFcst: i, deltaT: %d, %s" % (i, deltaT))

            cog, sog = self.rte.advanceCurPos(deltaT)[:2]

            try:
                u, v = self.getCurPosFcst(i, ('uwnd', 'vwnd'))
            except PointNotInsideGrid:
                logger.debug("genRteFcst: PointNotInsideGrid exception caught "
                             "on i = %d" % i)
                continue

            # TODO: delete once standard wind speeds are m/s in units.py;
            #       or check for != 'm/s' and convert with unit.py dictionary
            if self.fcst['uwnd'].attributes['units'] == 'knot':
                u = u / KN_PER_MS
            if self.fcst['vwnd'].attributes['units'] == 'knot':
                v = v / KN_PER_MS

            rteFcst.append(FcstWP(self.rte.curPos.lat, self.rte.curPos.lon, u,
                                  v, self.fTimes[i], np.radians(cog), sog))

        return rteFcst

    def getCurPosFcst(self, timeIndex, fcstVars):
        """
        Returns interpolated forecast data for self.fTimes[timeIndex] at
        self.curPos .  fcstVars is a list or tuple of strings that denote the
        variables in self.fcst for which the values at self.curPos should be
        calculated (e.g. ['uwnd', 'vwnd']).
        Returns a tuple with the resulting forecast values for fcstVars in the
        same order as fcstVars.  Raises PointNotInsideGrid if self.curPos is
        not enclosed by forecast grid points.
        """
        lat = self.rte.curPos.lat
        lon = self.rte.curPos.lon

        # we assume lats are sorted but don't know if S -> N ('normal') or
        # N -> S ('reverse'):
        sortLat = self.fcst['latitude'].data.copy()
        if self.fcst['latitude'].data[0] > self.fcst['latitude'].data[-1]:
            sortLat.sort()  # sort S -> N
            i = -1 * bisect(sortLat, lat)
        else:
            i = bisect(sortLat, lat)
        j = bisect(self.fcst['longitude'].data, lon)

        if (i == 0 or j == 0 or abs(i) == len(self.fcst['latitude'].data) or
                                j == len(self.fcst['longitude'].data)):
            raise PointNotInsideGrid

        latVec = self.fcst['latitude'].data[i-1:i+1]
        lonVec = self.fcst['longitude'].data[j-1:j+1]

        logger.debug("curPos: %.2f %.2f, lat slice: %s, lon slice: %s" %
                     (lat, lon,  latVec, lonVec))

        out = []
        for fVar in fcstVars:
            box = self.fcst[fVar].data[timeIndex, i-1:i+1, j-1:j+1]

            logger.debug("\n%s-box:\n%s" % (fVar, box))

            z = interpol(box, latVec, lonVec, lat, lon)

            logger.debug("%s-result: %f" % (fVar, z))

            out.append(z)

        return tuple(out)

    def exportFcstGPX(self, gpxFile, windApparent=True, timeLabels=True):
        """
        Exports forecast along Route in OpenCPN's gpx XML-format.

            gpxFile         -   File object open for writing; must be closed by
                                calling function
            windApparent    -   If set to 'False' true wind speed and direction
                                will be used for waypoint names
            timeLables      -   If set to 'False' utc time labels will be
                                ommitted from waypoint names

        Returns number of waypoints written to gpxFile
        """
        rteFcst = self.genRteFcst()
        root = ET.fromstring(GPX_WRAPPER)
        for wpCount, fp in enumerate(rteFcst):
            lat, lon = fp.nautical_latlon()
            wp = ET.SubElement(root, 'wpt', attrib={'lat': "%.6f" % lat,
                          'lon': "%.6f" % lon})

            wp_time = ET.SubElement(wp, 'time')
            wp_time.text = fp.utc.replace(microsecond=0).isoformat() + 'Z'

            wp_name = ET.SubElement(wp, 'name')
            # construct name string:
            if windApparent:
                awa, wwSide, aws = appWind(fp.wind, fp.cog, fp.sog)
                wspeed = units.covert_from_default(aws, 'knot')
                nameStr = "%d%s@%d" % (round(np.degrees(awa)), wwSide,
                                        round(wspeed))
            else:
                wspeed = units.covert_from_default(fp.wind.speed, 'knot')
                nameStr = "%d@%d" % (round(np.degrees(fp.wind.nautical_dir())),
                                     round(wspeed))
            if timeLabels:
                nameStr += " %s" % fp.utc.strftime("%HZ%d%b")
            wp_name.text = nameStr

            wp_sym = ET.SubElement(wp, 'sym')
            wp_sym.text = W_ARROW_ICON_PREFIX + fp.wind.readable

            ET.SubElement(wp, 'type').text = 'WPT'
            wp_ext = ET.SubElement(wp, 'extensions')
            for ext in ['opencpn:viz', 'opencpn:viz_name']:
                ET.SubElement(wp_ext, ext).text = '1'

        xmlStr = ET.tostring(root, encoding='utf-8')
        # Etree adds some namespace prefix that confuses OpenCPN
        xmlStr = xmlStr.replace('ns0:', '').replace(':ns0=', '=')
        gpxFile.write("%s\n" % XML_HEADER)
        gpxFile.write(xmlStr)

        return wpCount


def interpol(box, latVec, lonVec, lat, lon):
    """
    Bilinear interpolation:
        box         - corner values as np 2x2 array, lats are the first
                      index, longs are second
        latVec      - corner latitudes as np array
        lonVec      - corner longitudes as np array
        lat, lon    - point for which to interpolate

    Returns interpolation result.
    """
    x = (lat - latVec[0]) / float(latVec[1] - latVec[0])
    y = (lon - lonVec[0]) / float(lonVec[1] - lonVec[0])

    return np.dot(np.dot(np.array([1 - x, x]), box), np.array([1 - y, y]))


def appWind(trueWind, cog, sog):
    """
    Returns a tuple (apparent wind angle in radians, wwSide, apparent wind speed
    in knots). 'wwSide' is a single character indicating the windward side: 'G'
    (green) if apparent wind is from stbd and 'R' (red) if apparent wind is
    from port.
    Arguments:
        trueWind    -   Wind object with true wind in
        cog         -   Course over ground in radians
        sog         -   speed over ground

    trueWind.speed and sog must be in the same unit. Return value will be in
    the same unit.
    """

    twa, wwSide = trueWindAngle(cog, trueWind.nautical_dir())
    aws = np.sqrt(sog**2 + trueWind.speed**2 + 2*sog*trueWind.speed*np.cos(twa))
    awa = np.arccos((sog**2 + aws**2 - trueWind.speed**2) / (2 * aws * sog))

    return awa, wwSide, aws


def trueWindAngle(cog, twd):
    """
    Returns a tuple with true wind angle in radians (0..pi, relative to boat
    COG) and the side the wind is from ('G' for stbd, 'R' for port).
    Arguments:
        cog     -   Boat's COG in radians
        twd     -   True wind direction i radians (0..2*pi)
    """
    x = cog - twd
    if 0 <= x < np.pi:
        twa = x
        wwSide = 'R'
    elif -np.pi <= x < 0:
        twa = -x
        wwSide = 'G'
    elif x >= np.pi:
        twa = 2*np.pi - x
        wwSide = 'G'
    elif x < -np.pi:
        twa = x + 2*np.pi
        wwSide = 'R'

    return twa, wwSide


class InvalidInFormatError(Exception):

    """
    Indicates missing or invalid format specifier for input file.
    """


class InvalidInFileError(Exception):

    """
    Indicates missing or invalid data in the input file.
    """


class TimeOverlapError(Exception):

    """
    Indicates that a time or timeframe does not fall into or overlap with
    another timeframe as required.
    """


class RegionOverlapError(Exception):

    """
    Indicates that two geographical regions do not overlap as required
    """


class PointNotInsideGrid(RegionOverlapError):

    """
    Lat/lon point for which data interpolation is requested is not enclosed by
    forecast grid.
    """


if __name__ == '__main__':

    import zlib
    import tinylib

    fmt = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
