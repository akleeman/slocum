import numpy as np
import unittest

from lib import objects, navigation

class TestNavigation(unittest.TestCase):

    def test_apparent_wind(self):
        wind = objects.Wind(u=10., v=10.)
        course = objects.Course(None, 6., 0.)
        apparent_wind = navigation.apparent_wind(course, wind)
        self.assertAlmostEqual(apparent_wind.speed, np.sqrt(16. + 100.))
        self.assertAlmostEqual(apparent_wind.dir, np.arctan2(-10, -4))

    def test_rhumbline(self):
        start = objects.LatLon(18.5, -155) # hawaii
        end = objects.LatLon(-16.5, -175) # fiji
        bearing = navigation.rhumbline_bearing(start, end)
        distance = navigation.rhumbline_distance(start, end)
        rhumb = navigation.rhumbline_path(start, bearing)
        approx = rhumb(distance)

        print "start: ", start.lat, start.lon
        print "end  : ", end.lat, end.lon
        print "rhumb(distance): ", approx.lat, approx.lon
        self.fail()

if __name__ == '__main__':
    unittest.main()
