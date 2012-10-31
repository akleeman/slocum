import os
import numpy as np
import unittest

from sl.objects import objects

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

    def test_interpolate(self):
        obj = Data()
        obj.create_coordinate('dim1', np.arange(10))
        obj.create_coordinate('dim2', np.arange(20))
        obj.create_variable('var1',
                            ('dim1', 'dim2'),
                            data=np.random.normal(size=(10, 20)))

        val = obj.interpolate('var1', dim1=3.5, dim2=2.5)
        expected = np.mean(obj['var1'].data[3:5, 2:4])
        np.testing.assert_almost_equal(val, expected)

        val = obj.interpolate('var1', dim1=3, dim2=2.5)
        expected = np.mean(obj['var1'].data[3, 2:4])
        np.testing.assert_almost_equal(val, expected)

        val = obj.interpolate('var1', dim1=3.5, dim2=2)
        expected = np.mean(obj['var1'].data[3:5, 2])
        np.testing.assert_almost_equal(val, expected)

        val = obj.interpolate('var1', dim1=2.2, dim2=3.9)
        expected = np.dot(np.dot(obj['var1'].data[2:4, 3:5].T, [0.8, 0.2]), [0.1, 0.9])
        np.testing.assert_almost_equal(val, expected)

    def test_data_object_reconstruction(self):
        file = open(os.path.join(_data_dir, 'test/test-gefs.nc'), 'r')
        obj = objects.Data(file)

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
        testfile = open(os.path.join(_data_dir, 'test/test-ccmp.nc'), 'r')
        testobjs = objects.Data(testfile)
        testobj = list(testobjs.iterator('time'))[0][1]

        def uwnd(x, y):
            ret = testobj.interpolate(var='uwnd', lat=x, lon=y)
            assert ret.size == 1
            return np.asscalar(ret)

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
        testobj.variables['uwnd'] = objects.normalize_variable(testobj['uwnd'])
        val_test = uwnd(-77.875, 41.625)
        self.assertAlmostEqual(val_test, (897 * 0.003051944 + 0.)*objects._speed['m/s'], 4)

    def test_gefs_data_field(self):
        testfile = os.path.join(_data_dir, 'test/test-gefs.nc')
        testobj = objects.Data(testfile)

        val = testobj.interpolate(lat=34., lon=212.)
        self.assertAlmostEqual(np.asscalar(val['lat'].data), 34.)
        self.assertAlmostEqual(np.asscalar(val['lon'].data), 212.)
        uwnd = val['U-component_of_wind_height_above_ground']
        self.assertAlmostEqual(uwnd.data[3, 10, 0], -6.6599998)

        val = testobj.interpolate(lat=22., lon=210.)
        uwnd = val['U-component_of_wind_height_above_ground']
        self.assertAlmostEqual(uwnd.data[1, 7, 0],  -3.97, 4)

if __name__ == '__main__':
    unittest.main()
