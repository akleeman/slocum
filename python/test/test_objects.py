import numpy as np
import unittest

from lib import objects

class TestObjects(unittest.TestCase):

    def test_wind_orientation(self):
        wind = objects.Wind(u=0, v=1)
        self.assertAlmostEqual(wind.dir, np.pi)
        self.assertAlmostEqual(wind.dir, np.pi)
        wind = objects.Wind(u=-1, v=0)
        self.assertAlmostEqual(wind.dir, np.pi/2)
        wind = objects.Wind(u=0, v=-1)
        self.assertAlmostEqual(wind.dir, 0.)
        wind = objects.Wind(u=1, v=0)
        self.assertAlmostEqual(wind.dir, -np.pi/2.)
        wind = objects.Wind(u=1, v=1)
        self.assertAlmostEqual(wind.dir, -3*np.pi/4)

    def test_wind_speed(self):
        wind = objects.Wind(u=1, v=1)
        self.assertAlmostEqual(wind.speed, np.sqrt(2.))

    def test_lat_lon(self):
        latlon = objects.LatLon(40., 135.)
        self.assertAlmostEqual(latlon.as_rad().lat, np.deg2rad(40.))
        self.assertAlmostEqual(latlon.as_rad().lon, np.deg2rad(135.))

if __name__ == '__main__':
    unittest.main()
