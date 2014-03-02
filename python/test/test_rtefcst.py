import unittest
import StringIO
import datetime as dt
import logging
import zlib
import os.path
import pickle

from sl.lib import rtefcst
from sl.lib import objects
from sl.lib import tinylib

TEST_DIR = 'test'

# test csv with route data; total length 265072.4 meters
testCSV = """ -34.13648, 151.55222, 
              -33.81644, 152.83151, 6
              -33.31621, 153.20315, 
              -33.49928, 153.99383, """

# expected output to be produced by RouteForecast.exportFcstGPX(): 
expectedAppWindOutStr="""<?xml version="1.0" encoding="utf-8" ?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" creator="OpenCPN" version="1.1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
<wpt lat="-34.136480" lon="151.552220"><time>2014-02-02T18:00:00Z</time><name>35R@18 18Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-34.065615" lon="151.840972"><time>2014-02-02T21:00:00Z</time><name>30R@18 21Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.994072" lon="152.129242"><time>2014-02-03T00:00:00Z</time><name>31R@21 00Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.921853" lon="152.417026"><time>2014-02-03T03:00:00Z</time><name>29R@18 03Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.848963" lon="152.704323"><time>2014-02-03T06:00:00Z</time><name>31R@23 06Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.674463" lon="152.937515"><time>2014-02-03T09:00:00Z</time><name>1G@24 09Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.419218" lon="153.127040"><time>2014-02-03T12:00:00Z</time><name>1G@24 12Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.356415" lon="153.374661"><time>2014-02-03T15:00:00Z</time><name>59R@21 15Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.423253" lon="153.662415"><time>2014-02-03T18:00:00Z</time><name>59R@10 18Z03Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.489423" lon="153.950608"><time>2014-02-03T21:00:00Z</time><name>45R@11 21Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt></gpx>"""

expectedTrueWindOutStr="""<?xml version="1.0" encoding="utf-8" ?>
<gpx xmlns="http://www.topografix.com/GPX/1/1" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" creator="OpenCPN" version="1.1" xsi:schemaLocation="http://www.topografix.com/GPX/1/1 http://www.topografix.com/GPX/1/1/gpx.xsd">
<wpt lat="-34.136480" lon="151.552220"><time>2014-02-02T18:00:00Z</time><name>27@15 18Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-34.065615" lon="151.840972"><time>2014-02-02T21:00:00Z</time><name>33@14 21Z02Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.994072" lon="152.129242"><time>2014-02-03T00:00:00Z</time><name>34@17 00Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.921853" lon="152.417026"><time>2014-02-03T03:00:00Z</time><name>35@14 03Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.848963" lon="152.704323"><time>2014-02-03T06:00:00Z</time><name>34@19 06Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.674463" lon="152.937515"><time>2014-02-03T09:00:00Z</time><name>34@19 09Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.419218" lon="153.127040"><time>2014-02-03T12:00:00Z</time><name>34@19 12Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.356415" lon="153.374661"><time>2014-02-03T15:00:00Z</time><name>34@19 15Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.423253" lon="153.662415"><time>2014-02-03T18:00:00Z</time><name>16@9 18Z03Feb</name><sym>warr_NNE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt><wpt lat="-33.489423" lon="153.950608"><time>2014-02-03T21:00:00Z</time><name>34@8 21Z03Feb</name><sym>warr_NE</sym><type>WPT</type><extensions><opencpn:viz>1</opencpn:viz><opencpn:viz_name>1</opencpn:viz_name></extensions></wpt></gpx>"""

class RouteTest(unittest.TestCase):

    def testReadCSVInvalidFormatSpecifier(self):
        tDept = dt.datetime(2014, 2, 1, 12, 0, 0)
        ifh = StringIO.StringIO(testCSV)
        kwargs = {'ifh': ifh, 'inFmt': 'xxx', 'utcDept': tDept}
        self.assertRaises(rtefcst.InvalidInFormatError, rtefcst.Route, **kwargs)
        ifh.close()

    def testReadCSVnoSpeed(self):
        tDept = dt.datetime(2014, 2, 1, 12, 0, 0)
        ifh = StringIO.StringIO(testCSV)
        kwargs = {'ifh': ifh, 'inFmt': 'csv', 'utcDept': tDept}
        self.assertRaises(rtefcst.InvalidInFileError, rtefcst.Route, **kwargs)
        ifh.close()

    def testReadCSVok(self):
        tDept = dt.datetime(2014, 2, 1, 12, 0, 0)
        ifh = StringIO.StringIO(testCSV)
        r = rtefcst.Route(ifh=ifh, inFmt='csv', utcDept=tDept, avrgSpeed=4.0)
        ifh.close()

        # check basic route set-up:
        self.assertEqual(len(r.rtePts), 4)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[0].wp.lat)
        self.assertAlmostEqual(r.curPos.lon, r.rtePts[0].wp.lon)
        self.assertAlmostEqual(r.ur.lat, r.rtePts[2].wp.lat)
        self.assertAlmostEqual(r.ur.lon, r.rtePts[3].wp.lon)
        self.assertAlmostEqual(r.ll.lat, r.rtePts[0].wp.lat)
        self.assertAlmostEqual(r.ll.lon, r.rtePts[0].wp.lon)

        d, t = r.updateUtcArrival()
        self.assertAlmostEqual(float(d) / t, 2.2419884)
        tArrival = tDept + dt.timedelta(0, t)
        deltaT = r.utcArrival - tArrival
        self.assertAlmostEqual(deltaT.total_seconds(), 0, places=1)

        # advance to position before final wp: 
        deltaT = dt.timedelta(0, 30 * 3600)
        course, speed, remT = r.advanceCurPos(deltaT)
        self.assertAlmostEqual(remT.total_seconds(), 0, places=1)
        self.assertAlmostEqual(course, 105.650454424, places=6)
        self.assertAlmostEqual(speed, 2.05777778036)
        self.assertAlmostEqual(r.curPos.lat, -33.44927136)

        # advance to final wp with time left over:
        r.resetCurPos()
        deltaT = dt.timedelta(0, 40 * 3600)
        course, speed, remT = r.advanceCurPos(deltaT)
        self.assertAlmostEqual(remT.total_seconds(), 25769.0, places=1)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[-1].wp.lat)
        self.assertIsNone(course)
        self.assertIsNone(speed)

        # append waypoint and re-check arrival time and bbox
        r.rtePts.append(rtefcst.Route.RtePoint(objects.LatLon(-34.3, 154.3), 5))
        d, t = r.updateUtcArrival()
        self.assertAlmostEqual(d, 358293.980918, places=2)
        deltaT = r.utcArrival - r.utcDept
        self.assertAlmostEqual(deltaT.total_seconds(), 163533.002444, places=1)
        r.updateBBox()
        self.assertAlmostEqual(r.ur.lat, r.rtePts[2].wp.lat)
        self.assertAlmostEqual(r.ur.lon, r.rtePts[4].wp.lon)
        self.assertAlmostEqual(r.ll.lat, r.rtePts[4].wp.lat)
        self.assertAlmostEqual(r.ll.lon, r.rtePts[0].wp.lon)
        
        # getPos:
        p, course, speed = r.getPos(dt.datetime(2014, 2, 2, 12, 0, 0))
        self.assertAlmostEqual(p.lat, -33.342428226)
        self.assertAlmostEqual(p.lon, 153.314862881)
        self.assertAlmostEqual(r.curPos.lat, r.rtePts[3].wp.lat)
        self.assertAlmostEqual(course, 105.650454424, places=6)
        self.assertAlmostEqual(speed, 2.05777778036)
        self.assertRaises(rtefcst.TimeOverlapError, r.getPos, 
                dt.datetime(2014, 3, 2, 12, 0, 0))


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

if __name__ == '__main__':

    unittest.main()
