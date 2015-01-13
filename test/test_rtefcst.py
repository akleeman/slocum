import unittest
import StringIO
import datetime as dt
import logging
import zlib
import os.path
import pickle
import numpy as np

from slocum.lib import rtefcst
from slocum.lib import objects
from slocum.lib import tinylib
from slocum.lib import units

TEST_DIR = 'test'

# test csv with route data; total length 265072.4 meters
testCSV = """ -34.13648, 151.55222,
              -33.81644, 152.83151, 6
              -33.31621, 153.20315,
              -33.49928, 153.99383, """
# test gpx with route data, same waypoints as csv
testGPX = """<?xml version="1.0" encoding="utf-8" ?>
<gpx version="1.1" creator="OpenCPN" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.topografix.com/GPX/1/1" xmlns:gpxx="http://www.garmin.com/xmlschemas/GpxExtensions/v3" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd" xmlns:opencpn="http://www.opencpn.org">
    <rte>
        <name>Test Route</name>
        <extensions>
            <opencpn:start>Moreton Bay</opencpn:start>
            <opencpn:end>Somewhere</opencpn:end>
            <opencpn:viz>1</opencpn:viz>
            <opencpn:guid>5ee6c818-79cf-4b5f-b540-2c645b649a05</opencpn:guid>
        </extensions>
        <rtept lat="-34.13648" lon="151.55222">
            <time>2014-03-30T10:16:45Z</time>
            <name>001</name>
            <sym>diamond</sym>
            <type>WPT</type>
            <extensions>
                <opencpn:guid>6de4cc2a-b2c3-4b9c-831d-91fa08743541</opencpn:guid>
                <opencpn:viz>1</opencpn:viz>
                <opencpn:viz_name>0</opencpn:viz_name>
                <opencpn:auto_name>1</opencpn:auto_name>
            </extensions>
        </rtept>
        <rtept lat="-33.81644" lon="152.83151">
            <time>2014-03-30T10:16:46Z</time>
            <name>002</name>
            <sym>diamond</sym>
            <type>WPT</type>
            <extensions>
                <opencpn:guid>2d46f34e-55e7-4f36-a29f-899c6a571e6b</opencpn:guid>
                <opencpn:viz>1</opencpn:viz>
                <opencpn:viz_name>0</opencpn:viz_name>
                <opencpn:auto_name>1</opencpn:auto_name>
            </extensions>
        </rtept>
        <rtept lat="-33.31621" lon="153.20315">
            <time>2014-03-30T10:16:47Z</time>
            <name>003</name>
            <sym>diamond</sym>
            <type>WPT</type>
            <extensions>
                <opencpn:guid>297027ca-edea-4d9f-ba59-db9d1034a7d6</opencpn:guid>
                <opencpn:viz>1</opencpn:viz>
                <opencpn:viz_name>0</opencpn:viz_name>
                <opencpn:auto_name>1</opencpn:auto_name>
            </extensions>
        </rtept>
        <rtept lat="-33.49928" lon="153.99383">
            <time>2014-03-30T10:16:48Z</time>
            <name>004</name>
            <sym>diamond</sym>
            <type>WPT</type>
            <extensions>
                <opencpn:guid>657553b9-84e6-490f-88fe-346f3f3eefa6</opencpn:guid>
                <opencpn:viz>1</opencpn:viz>
                <opencpn:viz_name>0</opencpn:viz_name>
                <opencpn:auto_name>1</opencpn:auto_name>
            </extensions>
        </rtept>
    </rte>
</gpx>"""

# expected output to be produced by RouteForecast.exportFcstGPX():
expectedAppWindOutStr="""<?xml version="1.0" encoding="utf-8" ?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" creator="OpenCPN" version="1.1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
<wpt lat="-34.136480" lon="151.552220"><time>2014-02-02T18:00:00Z</time><name>35R@18 18Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-34.065615" lon="151.840972"><time>2014-02-02T21:00:00Z</time><name>30R@18 21Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.994072" lon="152.129242"><time>2014-02-03T00:00:00Z</time><name>31R@21 00Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.921853" lon="152.417026"><time>2014-02-03T03:00:00Z</time><name>29R@18 03Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.848963" lon="152.704323"><time>2014-02-03T06:00:00Z</time><name>31R@23 06Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.674463" lon="152.937515"><time>2014-02-03T09:00:00Z</time><name>1G@24 09Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.419218" lon="153.127040"><time>2014-02-03T12:00:00Z</time><name>1G@24 12Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.356415" lon="153.374661"><time>2014-02-03T15:00:00Z</time><name>59R@21 15Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.423253" lon="153.662415"><time>2014-02-03T18:00:00Z</time><name>59R@10 18Z03Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.489423" lon="153.950608"><time>2014-02-03T21:00:00Z</time><name>45R@11 21Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt></gpx>"""

expectedTrueWindOutStr="""<?xml version="1.0" encoding="utf-8" ?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" creator="OpenCPN" version="1.1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
<wpt lat="-34.136480" lon="151.552220"><time>2014-02-02T18:00:00Z</time><name>27@15 18Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-34.065615" lon="151.840972"><time>2014-02-02T21:00:00Z</time><name>33@14 21Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.994072" lon="152.129242"><time>2014-02-03T00:00:00Z</time><name>34@17 00Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.921853" lon="152.417026"><time>2014-02-03T03:00:00Z</time><name>35@14 03Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.848963" lon="152.704323"><time>2014-02-03T06:00:00Z</time><name>34@19 06Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.674463" lon="152.937515"><time>2014-02-03T09:00:00Z</time><name>34@19 09Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.419218" lon="153.127040"><time>2014-02-03T12:00:00Z</time><name>34@19 12Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.356415" lon="153.374661"><time>2014-02-03T15:00:00Z</time><name>34@19 15Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.423253" lon="153.662415"><time>2014-02-03T18:00:00Z</time><name>16@9 18Z03Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.489423" lon="153.950608"><time>2014-02-03T21:00:00Z</time><name>34@8 21Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt></gpx>"""

class UtilityTest(unittest.TestCase):

    def testInterpolDateline(self):
        box = np.array([[-5, 15], [10, -10]])
        latVec = np.array([10., -10.])
        lonVec = np.array([170., -170.])
        centerValue = rtefcst.interpol(box, latVec, lonVec, 0., 180.)
        self.assertAlmostEqual(centerValue, 2.5)

    def testTrueWindAngle(self):
        a_90, __ = units.normalize_scalar(90, 'degrees')
        a_105, __ = units.normalize_scalar(105, 'degrees')
        a_135, __ = units.normalize_scalar(135, 'degrees')
        a_165, __ = units.normalize_scalar(165, 'degrees')
        a_225, __ = units.normalize_scalar(225, 'degrees')
        a_330, __ = units.normalize_scalar(330, 'degrees')
        param = [{'cog': a_225, 'twd': a_135, 'twa': (a_90, 'R')},
                 {'cog': a_135, 'twd': a_225, 'twa': (a_90, 'G')},
                 {'cog': a_225, 'twd': a_330, 'twa': (a_105, 'G')},
                 {'cog': a_135, 'twd': a_330, 'twa': (a_165, 'R')}]

        for p in param:
            twa, wws = rtefcst.trueWindAngle(p['cog'], p['twd'])
            self.assertAlmostEqual(twa, p['twa'][0])
            self.assertEqual(wws, p['twa'][1])

    def testAppWind(self):
        awa, wws, aws = rtefcst.appWind(45, 10, 135, 5)
        self.assertAlmostEqual(awa, 63.43494882292201)
        self.assertAlmostEqual(aws, 11.180339887498949)
        self.assertEqual(wws, 'R')
        awa, wws, aws = rtefcst.appWind(225, 10, 135, 5)
        self.assertAlmostEqual(awa, 63.43494882292201)
        self.assertAlmostEqual(aws, 11.180339887498949)
        self.assertEqual(wws, 'G')

class RouteTest(unittest.TestCase):

    def testReadCSVInvalidFormatSpecifier(self):
        tDept = np.datetime64('2014-02-01T12:00Z')
        ifh = StringIO.StringIO(testCSV)
        kwargs = {'ifh': ifh, 'inFmt': 'xxx', 'utcDept': tDept}
        self.assertRaises(rtefcst.InvalidInFormatError, rtefcst.Route, **kwargs)
        ifh.close()

    def testReadCSVnoSpeed(self):
        tDept = np.datetime64('2014-02-01T12:00Z')
        ifh = StringIO.StringIO(testCSV)
        kwargs = {'ifh': ifh, 'inFmt': 'csv', 'utcDept': tDept}
        self.assertRaises(ValueError, rtefcst.Route, **kwargs)
        ifh.close()

    def testReadCSVok(self):
        tDept = np.datetime64('2014-02-01T12:00Z')
        ifh = StringIO.StringIO(testCSV)
        r = rtefcst.Route(ifh=ifh, inFmt='csv', utcDept=tDept, avrgSpeed=4.0)
        ifh.close()

        # check basic route set-up:
        self.assertEqual(len(r.rtePts), 4)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[0].lat)
        self.assertAlmostEqual(r.curPos.lon, r.rtePts[0].lon)
        self.assertAlmostEqual(r.bbox.north, r.rtePts[2].lat)
        self.assertAlmostEqual(r.bbox.east, r.rtePts[3].lon)
        self.assertAlmostEqual(r.bbox.south, r.rtePts[0].lat)
        self.assertAlmostEqual(r.bbox.west, r.rtePts[0].lon)

        d, t = r.updateUtcArrival()
        self.assertAlmostEqual(float(d) / t, 2.2419884)
        tArrival = tDept + np.timedelta64(int(round(t)), 's')
        deltaT = r.utcArrival - tArrival
        self.assertAlmostEqual(units.total_seconds(deltaT), 0, places=1)

        # advance to position before final wp:
        deltaT = np.timedelta64(30 * 3600, 's')
        course, speed, remT = r.advanceCurPos(deltaT)
        self.assertAlmostEqual(units.total_seconds(remT), 0, places=1)
        self.assertAlmostEqual(course, 105.650454424, places=4)
        self.assertAlmostEqual(speed, 2.05777778036, places=4)
        self.assertAlmostEqual(r.curPos.lat, -33.44927136, places=4)

        # advance to final wp with time left over:
        r.resetCurPos()
        deltaT = np.timedelta64(40 * 3600, 's')
        course, speed, remT = r.advanceCurPos(deltaT)
        self.assertAlmostEqual(units.total_seconds(remT), 25769.0, places=1)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[-1].lat)
        self.assertIsNone(course)
        self.assertIsNone(speed)

        # append waypoint and re-check arrival time and bbox
        r.rtePts.append(
                rtefcst.Route.RtePoint(objects.NautAngle(-34.3),
                    objects.NautAngle(154.3), 5))
        d, t = r.updateUtcArrival()
        self.assertAlmostEqual(d, 358293.980918, places=2)
        deltaT = r.utcArrival - r.utcDept
        self.assertAlmostEqual(units.total_seconds(deltaT), 163533.002444, places=1)
        r.updateBBox()
        self.assertEqual(r.bbox.north, r.rtePts[2].lat)
        self.assertEqual(r.bbox.east, r.rtePts[4].lon)
        self.assertEqual(r.bbox.south, r.rtePts[4].lat)
        self.assertEqual(r.bbox.west, r.rtePts[0].lon)

        # getPos:
        p, course, speed = r.getPos(np.datetime64('2014-02-02T12:00Z'))
        self.assertAlmostEqual(p.lat, -33.342428226, places=4)
        self.assertAlmostEqual(p.lon, 153.314862881, places=4)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[3].lat)
        self.assertAlmostEqual(course, 105.650454424, places=4)
        self.assertAlmostEqual(speed, 2.05777778036, places=4)
        self.assertRaises(rtefcst.TimeOverlapError, r.getPos,
                np.datetime64('2014-03-02T12:00Z'))

    def testReadGPXok(self):
        tDept = np.datetime64('2014-02-01T12:00Z')
        ifh = StringIO.StringIO(testGPX)
        r = rtefcst.Route(ifh=ifh, inFmt='gpx', utcDept=tDept, avrgSpeed=5.0)
        ifh.close()

        # check basic route set-up:
        self.assertEqual(len(r.rtePts), 4)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[0].lat)
        self.assertAlmostEqual(r.curPos.lon, r.rtePts[0].lon)
        self.assertAlmostEqual(r.bbox.north, r.rtePts[2].lat)
        self.assertAlmostEqual(r.bbox.east, r.rtePts[3].lon)
        self.assertAlmostEqual(r.bbox.south, r.rtePts[0].lat)
        self.assertAlmostEqual(r.bbox.west, r.rtePts[0].lon)

        d, t = r.updateUtcArrival()
        self.assertAlmostEqual(float(d) / t, 2.57222222545)
        tArrival = tDept + np.timedelta64(int(round(t)), 's')
        deltaT = r.utcArrival - tArrival
        self.assertAlmostEqual(units.total_seconds(deltaT), 0, places=1)

"""
class RteFcstTest(unittest.TestCase):

    with open(os.path.join(TEST_DIR, 'rtefcst_test.fcst'), 'rb') as wb:
        tinyfcst = zlib.decompress(wb.read())
    fcst = tinylib.from_beaufort(tinyfcst)

    tDept = dt.datetime(2014, 2, 2, 18, 0, 0)
    ifh = StringIO.StringIO(testCSV)
    rte = rtefcst.Route(ifh=ifh, inFmt='csv', utcDept=tDept, avrgSpeed=5.0)
    ifh.close()

    rf = rtefcst.RouteForecast(rte, fcst)

    def testPosFcst(self):
        RteFcstTest.rf.rte.resetCurPos()    # utcDept coincides with self.fTimes[6]
        u, v = RteFcstTest.rf.getCurPosFcst(6, ('uwnd', 'vwnd'))
        self.assertAlmostEqual(u, -6.658279, places=6)
        self.assertAlmostEqual(v, -13.079877, places=6)
        deltaT = dt.timedelta(0, 12 * 3600)     # advances 4 fcst times
        RteFcstTest.rf.rte.advanceCurPos(deltaT)
        u, v = RteFcstTest.rf.getCurPosFcst(10, ('uwnd', 'vwnd'))
        self.assertAlmostEqual(u, -10.278049, places=6)
        self.assertAlmostEqual(v, -15.382188, places=6)
        # move curPos outside fcst box and check for exception:
        RteFcstTest.rf.rte.curPos.lat = 10.
        self.assertRaises(rtefcst.PointNotInsideGrid,
                          RteFcstTest.rf.getCurPosFcst, 6, ('uwnd'))
        RteFcstTest.rf.rte.resetCurPos()

    def testExportGPXAppWind(self):
        ofh = StringIO.StringIO()
        RteFcstTest.rf.exportFcstGPX(ofh)
        outStr = ofh.getvalue()
        ofh.close()
        self.assertEqual(expectedAppWindOutStr, outStr)

    def testExportGPXTrueWind(self):
        ofh = StringIO.StringIO()
        RteFcstTest.rf.exportFcstGPX(ofh, windApparent=False)
        outStr = ofh.getvalue()
        ofh.close()
        self.assertEqual(expectedTrueWindOutStr, outStr)
"""
if __name__ == '__main__':

    unittest.main()
