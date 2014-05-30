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

import csv
import numpy as np
import logging
import xml.etree.ElementTree as ET

from bisect import bisect, bisect_left, bisect_right
from collections import namedtuple
from pyproj import Geod
import xray

import units
from objects import NautAngle, Position, BoundingBox
import conventions as conv

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

        'gpx'
            XML file with a route exported from OpenCPN. avrgSpeed argument
            must be provided to Route constructor.

    """
    # TODO: add parsers for different import formats

    RtePoint = namedtuple('RtePoint', ['lat', 'lon', 'speed'])

    geod = Geod(ellps='WGS84')

    def __init__(self, ifh=None, inFmt=None, utcDept=None, avrgSpeed=None):
        """
        Constructs a route object from the data in file object ifh. which must
        be in format inFmt (see class doc string for valid formats).

        Parameters
        ----------
        ifh : file like
            Input file with route data, open for reading. If no ifh is
            provided the route object will be constructed with an empty rtePts
            list.
        inFmt : string
            Format of route input file (see class doc string for valid
            formats).
        utcDept : np.datetime64 or dt.datetime
            Departure from the first waypoint.
        avrgSpeed : float
            Average boat speed over ground in knots. If provided, it will be
            substituted for any missing leg speeds in the input. If no
            avrgSpeed > 0 is provided and the input file contains missing leg
            speeds a ValueError will be raised.

        Note: longitudes in rtePts will be in range [-180, 180[ (see
        objects.NautAngle).  Use full_circle method of NautAngle to get an
        angle in the [0, 360[ range.
        """
        self.utcDept = np.datetime64(utcDept)
        self.rtePts = []

        if ifh is None:     # in case we want to roll our own
            return

        if inFmt == 'csv':
            self.read_csv(ifh, avrgSpeed)
        elif inFmt == 'gpx':
            self.read_gpx(ifh, avrgSpeed)
        else:
            raise (InvalidInFormatError,
                   "Invalid route file format specifier: %s" % inFmt)

        self.bbox = BoundingBox(*self.updateBBox())
        self.curPos = Position(*self.resetCurPos())
        self.updateUtcArrival()     # np.datetime64 object

    def updateBBox(self):
        """
        Determines bounding box for route based on current lat/lon values in
        self.rtePts. Returns a tuple with the north, south, east, west values
        (all as NautAngle objects). If hasattr(self, bbox) it will also update
        self.bbox.
        """
        north = south = self.rtePts[0].lat
        east = west = self.rtePts[0].lon

        for p in self.rtePts[1:]:
            if p.lat < south: south = p.lat
            if p.lat > north: north = p.lat
            if p.lon < west: west = p.lon
            if p.lon > east: east = p.lon

        assert east >= west

        if hasattr(self, 'bbox'):
            self.bbox = BoundingBox(north, south, east, west)

        return north, south, east, west

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
            __, __, dist = Route.geod.inv(cur.lon, cur.lat, nxt.lon, nxt.lat)
            tdist += dist
            speed_ms = units.convert_from_default(cur.speed, 'm/s')
            ttime += float(dist) / speed_ms
        self.utcArrival = self.utcDept + np.timedelta64(int(round(ttime)), 's')
        return (tdist, ttime)

    def getPos(self, utc):
        """
        Returns a tuple (position, course, speed) with the boat's position as a
        Position namedtuple with lat/lon as two NautAngle objects, course over
        ground in deg true, and SOG in m/s along the route at time utc, a
        np.datetime64 object.  Does not update self.curPos.  Raises
        TimeOverlapError if utc is before self.utcDept or after
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
        origPos = self.curPos
        origWP = (self.prevWP, self.nextWP)
        self.curPos = Position(*self.rtePts[0][:2])
        self.prevWP, self.nextWP = (0, 1)
        # TODO: If any exception from advanceCurPos is handled somewhere, the
        # handler needs to restore the orig values
        course, speed, deltaT = self.advanceCurPos(deltaT)
        tmpCurPos = self.curPos
        self.curPos = origPos
        self.prevWP, self.nextWP = origWP

        return tmpCurPos, course, speed

    def advanceCurPos(self, deltaT):
        """
        Advances self.curPos by deltaT (a np.timedelta64 object), based on
        the speeds in self.rtePts.  Returns a tuple (course, speed, time left
        over) with the boat's course in deg true, SOG in default units and the
        'unused' portion of deltaT (if any, in case the last WP was reached) as
        a np.timedelta64 object (0 if the new position is still before the last
        WP).
        """
        logger.debug(
            "advanceCurPos called with deltaT: sec %d, curPos: "
            "%.3f %.3f, prevWP: %d, nextWP: %d" %
            (units.total_seconds(deltaT), self.curPos.lat, self.curPos.lon,
            self.prevWP, self.nextWP))

        if self.nextWP is None:     # we're already at the end of the route
            return None, None, deltaT
        elif deltaT <= 0:  # utcDept coincides with
                           # forecast time
            az12, __, __ = Route.geod.inv(self.curPos.lon,
                                    self.curPos.lat,
                                    self.rtePts[self.nextWP].lon,
                                    self.rtePts[self.nextWP].lat)
            logger.debug("exiting advanceCurPos with deltaT: % d sec, "
                    "curPos: % .3f % .3f, prevWP: % d, nextWP: % d" %
                        (units.total_seconds(deltaT), self.curPos.lat,
                            self.curPos.lon, self.prevWP, self.nextWP))
            return az12, self.rtePts[self.prevWP].speed, deltaT

        while deltaT > 0:

            az12, __, distToNext = Route.geod.inv(self.curPos.lon,
                                            self.curPos.lat,
                                            self.rtePts[self.nextWP].lon,
                                            self.rtePts[self.nextWP].lat)

            # TODO: agree on what's 'at the WP' (100m)?
            if distToNext < 100:
                self.curPos = Position(*self.rtePts[self.nextWP][:2])
                self.__advancePrevNext()
                continue

            speed_ms = units.convert_from_default(
                    self.rtePts[self.prevWP].speed, 'm/s')
            tToNext = np.timedelta64(int(round(distToNext / speed_ms)), 's')

            if deltaT < tToNext:    # new curPos in this iteration
                dist = units.total_seconds(deltaT) * speed_ms
                lon, lat, __ = (
                        Route.geod.fwd(self.curPos.lon, self.curPos.lat, az12,
                            dist))
                self.curPos = Position(NautAngle(lat), NautAngle(lon))
                deltaT = np.timedelta64(0)
                logger.debug("exiting advanceCurPos with deltaT: % d sec, "
                        "curPos: % .3f % .3f, prevWP: % d, nextWP: % d" %
                            (units.total_seconds(deltaT), self.curPos.lat,
                                self.curPos.lon, self.prevWP, self.nextWP))
                return az12, self.rtePts[self.prevWP].speed, deltaT
            else:                   # move to next WP
                deltaT -= tToNext
                self.curPos = Position(*self.rtePts[self.nextWP][:2])
                prevWP, nextWP = self.__advancePrevNext()
                if nextWP is None:
                    logger.debug("exiting advanceCurPos with deltaT: % d sec,"
                            " curPos: % .3f % .3f, prevWP: % d, nextWP: None"
                            % (units.total_seconds(deltaT), self.curPos.lat,
                                     self.curPos.lon, self.prevWP))
                    return None, None, deltaT

    def resetCurPos(self):
        """
        Returns a tuple (lat, lon) with values from the first waypoint
        in the route. Resets prevWP and nextWP to 0 and 1 respectively (or
        nextWP = None if route has only one wp). If hasattr(self, 'curPos')
        self.curPos is also updated.
        """
        if len(self.rtePts) > 0:
            lat, lon = self.rtePts[0][:2]
            self.prevWP = 0
            if len(self.rtePts) > 1:
                self.nextWP = 1
                if self.utcDept:
                    self.updateUtcArrival()
                else:
                    self.utcArrive = self.utcDept
            else:
                self.nextWP = None
        else:
            lat = lon = None
            self.prevWP = self.nextWP = None

        if hasattr(self, 'curPos'):
            self.curPos = Position(lat, lon)

        return lat, lon

    def __advancePrevNext(self):
        """
        Advances the prevWP and nextWP indices along self.rtePts.  If prevWP
        reaches the end of the route nextWP will be set to None.  Returns a
        (prevWP, nextWP) tuple.
        """
        if self.nextWP:
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
        (converted to default units defined in units.py) from ifh.
        """
        if isinstance(ifh, file):
            fn = ifh.name
        else:
            fn = "route input"
        rte = csv.reader(ifh, skipinitialspace=True)
        for r in rte:
            if r[2] == '' or not float(r[2]) > 0:
                if avrgSpeed is None:
                    raise ValueError(
                            "Expected a non None average speed to deal with "
                            "missing speed value in %s." % fn)
                if not avrgSpeed > 0:
                    raise ValueError(
                            "Expected an average speed > 0 to deal with "
                            "missing speed value in %s." % fn)
                else:
                    r[2] = avrgSpeed
            try:
                speed, __ = units.normalize_scalar(float(r[2]), 'knot')
                self.rtePts.append(Route.RtePoint(NautAngle(r[0]),
                                   NautAngle(r[1]), speed))
            except:
                raise InvalidInFileError(
                        "Could not process line\n"
                        "%s\n"
                        "in %s" % (str(r), fn))

    def read_gpx(self, ifo, avrgSpeed):
        """
        Reads route from gpx XML file, iterating over rtept elements. Since
        OpenCPN route files don't include leg speed, avrgSpeed will be used for
        all legs.
        """
        # XML name space
        ns_string = 'http://www.topografix.com/GPX/1/1'

        if isinstance(ifo, file):
            fn = ifo.name
        else:
            fn = "route input"

        if avrgSpeed is None:
            raise ValueError(
                    "Expected a non None average speed to process %s as gpx."
                    % fn)
        if not avrgSpeed > 0:
            raise ValueError(
                    "Expected an average speed > 0 to process %s as gpx." % fn)

        norm_speed, __ = units.normalize_scalar(avrgSpeed, 'knot')
        root = ET.fromstring(ifo.read())
        for rp in root.iter("{%s}rtept" % ns_string):
            lat = NautAngle(rp.attrib['lat'])
            lon = NautAngle(rp.attrib['lon'])
            self.rtePts.append(Route.RtePoint(lat, lon, norm_speed))

    def queryDict(self, model=u'gfs', fc_type='gridded', grid_delta=(0.5, 0.5),
            fc_vars=[conv.PRESSURE, conv.PRECIP, conv.WIND]):
        """
        Returns a query dictionary constructed from Route.
        """
        qd = {}

        # borders around bbox in full degrees:
        east = round(self.bbox.east + 1)
        west = round(self.bbox.west - 1)
        north = min(90., round(self.bbox.north + 1))
        south = max(-90., round(self.bbox.south - 1))
        qd['domain'] = {'N': north, 'S': south, 'E': east, 'W': west}

        # TODO: determine fc times based on utcDept, utcArrival and latest
        # available forecast; will require rtefcst to move up to sl to get
        # access to poseidon functions.
        # For now we just take a week's worth of 3 hr intervals.
        qd['hours'] = range(0, 169, 3)

        qd['model'] = model
        qd['grid_delta'] = grid_delta
        qd['type'] = fc_type
        qd['vars'] = fc_vars
        qd['warnings'] = []

        return qd

class RouteForecast(object):

    """
    Initialized with a Route and a xray.Dataset object containing the
    forecast data, method genRteFcst of RouteForecast will generate a
    forecast along the Route for all forecast variables and the times provided
    in the Dataset that overlap with the travel time.  The result can be
    written to a gpx waypoint file wich can be displayed in OpenCPN as a
    tempoary layer.
    """

    # Convenience wrapper to store forecast data at waypoints.
    # lat, lon: NautAngle objects;
    # utc: np.datetime64; cog: float; sog: float in default units
    # vars: dict mapping vars names (e.g. 'precip', 'wind_speed') to their values
    FcstWP = namedtuple('FcstWP', ['lat', 'lon', 'utc', 'cog', 'sog', 'vars'])

    def __init__(self, rte, fcst):
        """
        rte:    rtefcst.Route object
        fcst:   xray.Dataset object
        """
        fTimes = xray.decode_cf_datetime(fcst['time'].values,
                fcst['time'].attrs['units'])   # array of np.datetime64
        # check that forecast times overlap travel time
        if (fTimes[0] > rte.utcArrival or
                fTimes[-1] < rte.utcDept):
            raise(TimeOverlapError,
                   "Forecast times do not overlap with route times")

        # check that forecast area overlaps route
        if (rte.bbox.north < NautAngle(min(fcst[conv.LAT].values)) or
                rte.bbox.south > NautAngle(max(fcst[conv.LAT].values))):
            raise(RegionOverlapError,
                   "Route latitudes outside of forecast region")
        fcLons = [NautAngle(a) for a in fcst[conv.LON].values]
        if (rte.bbox.east < min(fcLons) or rte.bbox.west > max(fcLons)):
            raise(RegionOverlapError,
                   "Route longitudes outside of forecast region")

        self.rte = rte
        self.fcst = fcst
        self.fTimes = fTimes

    def genRteFcst(self):
        """
        Returns a list with FcstWP named tuples for the temporal and regional
        overlap between self.rte and self.fcst.
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
                fcDict = self.getCurPosFcst(i)
            except PointNotInsideGrid:
                logger.debug("genRteFcst: PointNotInsideGrid exception caught "
                             "on i = %d" % i)
                continue

            rteFcst.append(RouteForecast.FcstWP(
                    self.rte.curPos.lat, self.rte.curPos.lon, self.fTimes[i],
                    cog, sog, fcDict))

        return rteFcst

    def getCurPosFcst(self, timeIndex):
        """
        Returns interpolated forecast data for self.fTimes[timeIndex] at
        self.curPos for all noncoordinate variables contained in the forecast.
        Returns a dictionary that maps variable names to the resulting forecast
        values.  Raises PointNotInsideGrid if self.curPos is not enclosed by
        forecast grid points.
        """
        lat = self.rte.curPos.lat
        lon = self.rte.curPos.lon

        # we assume lats are sorted but don't know if S -> N ('normal') or
        # N -> S ('reverse'):
        sortLat = self.fcst[conv.LAT].values.copy()
        if self.fcst[conv.LAT].values[0] > self.fcst[conv.LAT].values[-1]:
            sortLat.sort()  # sort S -> N
            i = -1 * bisect(sortLat, lat)
        else:
            i = bisect(sortLat, lat)
        j = bisect(self.fcst[conv.LON].values, lon)

        if (i == 0 or j == 0 or abs(i) == len(self.fcst[conv.LAT].values) or
                                j == len(self.fcst[conv.LON].values)):
            raise PointNotInsideGrid

        latVec = self.fcst[conv.LAT].values[i-1:i+1]
        lonVec = self.fcst[conv.LON].values[j-1:j+1]

        logger.debug("curPos: %.2f %.2f, lat slice: %s, lon slice: %s" %
                     (lat, lon,  latVec, lonVec))

        out = {}
        for fVar in self.fcst.noncoordinates.keys():
            box = self.fcst[fVar].values[timeIndex, i-1:i+1, j-1:j+1]

            logger.debug("\n%s-box:\n%s" % (fVar, box))

            z = interpol(box, latVec, lonVec, lat, lon)

            logger.debug("%s-result: %f" % (fVar, z))

            out[fVar], __ = units.normalize_scalar(
                    z, self.fcst[fVar].attrs[conv.UNITS])

        return out

    def exportFcstGPX(self, gpxFile, windApparent=True, timeLabels=True):
        """
        Exports forecast along Route in OpenCPN's gpx XML-format.

            Parameter
            ---------
            gpxFile : file like
                File object open for writing gpx output file; must be closed by
                calling function.
            windApparent : boolean
                If set to 'False' true wind speed and direction will be used
                for waypoint names.
            timeLables : boolean
                If set to 'False' utc time labels will be ommitted from
                waypoint names.

            Returns
            -------
            Number of waypoints written to gpxFile.
        """
        # TODO: change output data to ranges (or BF for wind)
        rteFcst = self.genRteFcst()
        root = ET.fromstring(GPX_WRAPPER)
        for wpCount, fp in enumerate(rteFcst):
            wp = ET.SubElement(root, 'wpt', attrib={'lat': "%.6f" % fp.lat,
                               'lon': "%.6f" % fp.lon})

            wp_time = ET.SubElement(wp, 'time')
            wp_time.text = fp.utc.astype('M8[us]').item().isoformat() + 'Z'

            wp_name = ET.SubElement(wp, 'name')
            # construct name string:
            if windApparent:
                awa, wwSide, aws = appWind(fp.vars[conv.WIND_DIR],
                        fp.vars[conv.WIND_SPEED], fp.cog, fp.sog)
                wSpeed = units.convert_from_default(aws, 'knot')
                wAngle = units.convert_from_default(awa, 'degrees')
                nameStr = "%d%s@%d" % (round(wAngle), wwSide, round(wSpeed))
            else:
                wSpeed = units.convert_from_default(fp.vars[conv.WIND_SPEED],
                        'knot')
                wAngle = units.convert_from_default(fp.vars[conv.WIND_DIR],
                        'degrees') % 360.
                nameStr = "%d@%d" % (round(wAngle), round(wSpeed))
            if timeLabels:
                nameStr += " %s" % (
                        fp.utc.astype('M8[us]').item().strftime("%HZ%d%b"))
            wp_name.text = nameStr

            ET.SubElement(wp, 'desc').text = RouteForecast.fcst_data2str(fp)

            wp_sym = ET.SubElement(wp, 'sym')
            twdDeg = units.convert_from_default(
                    fp.vars[conv.WIND_DIR], 'degrees')
            __, name = NautAngle(twdDeg).compass_dir()
            wp_sym.text = W_ARROW_ICON_PREFIX + name

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

    @staticmethod
    def fcst_data2str(fp):
        """
        Returns the forcast data in fp (a FcstWP named tuple) as a string.
        """
        separator = ' +++ '
        fcUnitDir = {conv.PRECIP: 'mm/hr',
                     conv.PRESSURE: 'hPa',
                     conv.WIND_SPEED: 'knot',
                     conv.WIND_DIR: 'deg'}

        out = "%s%sCOG: %d%sSOG: %.1f kn%sFORECAST DATA: " % (
                fp.utc.astype('M8[m]'), separator,
                units.convert_from_default(fp.cog, 'degrees'), separator,
                units.convert_from_default(fp.sog, 'knot'), separator)

        for key in fp.vars:
            if key in fcUnitDir:
                fcUnit = fcUnitDir[key]
                fcVal = units.convert_from_default(fp.vars[key], fcUnit)
            else:
                fcUnit = ''
                fcVal = fp.vars[key]
            out += "%s%s: %.1f %s" % (separator, key, fcVal, fcUnit)

        return out


def interpol(box, latVec, lonVec, lat, lon):
    """
    Bilinear interpolation.

    Parameters
    ----------
    box : np.array
        Corner values of the function to be interpolated as an np 2x2 array;
        lats are the first index, longs are second.
    latVec : np.array
        Corner latitudes as np array.
    lonVec : np.array
        Corner longitudes as np array.
    lat : float or NautAngle
        Latitude for which to interpolate.
    lon : float or NautAngle
        Longitude for which to interpolate.

    Returns
    -------
    Interpolation result as float.
    """
    x = (lat - latVec[0]) / float(latVec[1] - latVec[0])
    # convert lons to NautAngles to handle potential wrap around
    naLon0 = NautAngle(lonVec[0])
    naLon1 = NautAngle(lonVec[1])
    # difference between two NautAngles is a float
    y = (NautAngle(lon) - naLon0) / (naLon1 - naLon0)

    return np.dot(np.dot(np.array([1 - x, x]), box), np.array([1 - y, y]))


def appWind(twd, tws, cog, sog):
    """
    Calculate apparent wind speed and angle.

    Paratmeters
    -----------
    twd : float
        True wind direction as an angle from true North in default units,
        either equivalent to 0..360 deg or to -180..180 deg.
    tws : float
        True wind speed.
    cog : float
        Course over ground as an angle from true North in default units,
        either equivalent to 0..360 deg or to -180..180 deg.
    sog : float
        Speed over ground.

    tws and sog must be in the same unit. Return value for apparent wind
    speed will be in this same unit.

    Returns
    -------
    Tuple (apparent wind angle, wwSide, apparent wind speed).
    Apparent wind angle will be in default units.  'wwSide' is a single
    character indicating the windward side: 'G' (green) if apparent wind is
    from stbd and 'R' (red) if apparent wind is from port.
    """
    twa, wwSide = trueWindAngle(cog, twd)
    twa_rad = units.convert_from_default(twa, 'radians')
    aws = np.sqrt(sog**2 + tws**2 + 2*sog*tws*np.cos(twa_rad))
    awa = np.arccos((sog**2 + aws**2 - tws**2) / (2 * aws * sog))
    awa, __ = units.normalize_scalar(awa, 'radians')

    return awa, wwSide, aws


def trueWindAngle(cog, twd):
    """
    Calculate true wind angle (relative to boat heading) from true wind
    direction.

    Parameters
    ----------
    cog : float
        Boat's COG in default angle units, either equivalent to 0..360 deg or
        to -180..180 deg.
    twd : float
        True wind direction in default angle units, either equivalent to 0..360
        deg or to -180..180 deg.

    Returns
    -------
    Tuple with true wind angle in default angle units (equivalent to 0..180,
    relative to boat COG) and the side the wind is from ('G' for stbd, 'R' for
    port).
    """
    # convert to deg in 0..360 range (floats):
    cog = NautAngle(units.convert_from_default(cog, 'degrees')).full_circle()
    twd = NautAngle(units.convert_from_default(twd, 'degrees')).full_circle()

    x = cog - twd
    if 0 <= x < 180:
        twa = x
        wwSide = 'R'
    elif -180 <= x < 0:
        twa = -x
        wwSide = 'G'
    elif x >= 180:
        twa = 360. - x
        wwSide = 'G'
    elif x < -180:
        twa = x + 360.
        wwSide = 'R'

    twa, __ = units.normalize_scalar(twa, 'degrees')

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

    fmt = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=fmt)
