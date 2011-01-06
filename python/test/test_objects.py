import os
import numpy as np
import unittest

from lib import objects

_data_dir = os.path.join(os.path.dirname(__file__), '../../data/')

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

    def test_data_object_reconstruction(self):
        file = os.path.join(_data_dir, 'test/test-gefs.nc')
        obj = objects.DataObject(file, mode='r')

        expected = obj.variables['U-component_of_wind_height_above_ground'].data.copy()
        actual = expected.copy()
        actual.fill(np.nan)

        var = 'U-component_of_wind_height_above_ground'
        def fill(ndarray, dataobj):
            dim = dataobj.variables[var].dimensions[0]
            for i, x in enumerate(dataobj.iterator(dim=dim, vars=[var])):
                if x[1].variables[var].data.ndim > 1:
                    fill(ndarray[i, ...], x[1])
                else:
                    ndarray[i,:] = x[1].variables[var].data
        fill(actual, obj)
        self.assertTrue(np.all(expected == actual))

    def test_ccmp_data_field(self):
        testfile = os.path.join(_data_dir, 'test/test-ccmp.nc')
        testobjs = objects.DataObject(testfile, 'r')
        testobj = list(testobjs.iterator('time'))[0][1]
        uwnd = objects.DataField(testobj, 'uwnd')

        # test interpolation
        val1 = uwnd(-73.625, 184.125)
        val2 = uwnd(-73.375, 184.125)
        val3 = uwnd(-73.5, 184.125)
        self.assertAlmostEqual(val3, 0.5 * (val1 + val2))
        """
        time[0]=20091231.75 lat[2]=-77.875 lon[166]=41.625 uwnd[3046]=897 m/s

        from ncks we know the following:
        lat: -77.875
        lon: 41.625
        uwnd= 897 m/s
        uwnd:scale_factor = 0.003051944f ;
        uwnd:add_offset = 0.f ;
        """
        val_test = uwnd(-77.875, 41.625)
        self.assertAlmostEqual(val_test, (897 * 0.003051944 + 0.)*objects._speed['m/s'], 4)

    def test_gefs_data_field(self):
        testfile = os.path.join(_data_dir, 'test/test-gefs.nc')
        testobj = objects.DataObject(testfile, 'r')
        sliced = testobj.slice('ens', 10).slice('time1', 5).slice('height_above_ground1', 0)

        uwnd = objects.DataField(sliced, 'U-component_of_wind_height_above_ground')
        self.assertAlmostEqual(uwnd(17.0, 180.0), -4.91792631149, 4)

        [3, 12, 0, 10, 13]
        -5.5700002
        -10.827188800
        sliced = testobj.slice('ens', 3).slice('time1', 12).slice('height_above_ground1', 0)
        uwnd = objects.DataField(sliced, 'U-component_of_wind_height_above_ground')
        self.assertAlmostEqual(uwnd(10.0, 183.0), -10.827188800, 4)

if __name__ == '__main__':
    unittest.main()
