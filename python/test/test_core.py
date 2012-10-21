import os
import numpy as np
import unittest

from wx.objects import core

class TestCore(unittest.TestCase):

    def setUp(self):
        self.data = core.Data()
        time = np.arange(100)
        self.data.create_coordinate('test_time', time)
        lats = np.arange(20., 30., 0.5)
        self.data.create_coordinate('test_lat', lats)
        lons = np.arange(180., 220.)
        self.data.create_coordinate('test_lon', lons)
        xlons, xlats = np.meshgrid(lons, lats)
        var_data = np.array([1.5*xlats + 0.5*xlons + t for t in time])
        self.data.create_variable('test_var',
                             dim=('test_time', 'test_lat', 'test_lon',),
                             data=var_data)

        self.data.create_dimension('lat_dim', lats.size)
        self.data.create_dimension('lon_dim', lons.size)
        self.data.create_variable('test_var_no_coords',
                             dim=('test_time', 'lat_dim', 'lon_dim',),
                             data=var_data)


    def test_interpolate(self):
        time = self.data['test_time'].data
        # exactly over a grid point
        tmp = self.data.interpolate(test_lat=21., test_lon=181.)
        expected = 1.5*21. + 0.5*181. + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "exactly over a grid")
        # middle of one dimension
        tmp = self.data.interpolate(test_lat=21.5, test_lon=181.)
        expected = 1.5*21.5 + 0.5*181. + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "middle of one dimension")
        # middle of a grid point
        tmp = self.data.interpolate(test_lat=21.5, test_lon=181.5)
        expected = 1.5*21.5 + 0.5*181.5 + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "middle of a grid point")
        # on the edge
        tmp = self.data.interpolate(test_lat=20., test_lon=181.)
        expected = 1.5*20. + 0.5*181. + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "on the edge of a grid")
        # in a corner
        tmp = self.data.interpolate(test_lat=20., test_lon=180.)
        expected = 1.5*20. + 0.5*180. + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "in the corner of the grid")
        # somewhere in the middle
        lt = np.random.uniform(20., 29.)
        ln = np.random.uniform(180., 199.)
        tmp = self.data.interpolate(test_lat=lt, test_lon=ln)
        expected = 1.5*lt + 0.5*ln + time
        np.testing.assert_array_almost_equal(expected,
                        np.squeeze(tmp['test_var'].data), 1e-8,
                        "somewhere in the middle")

    def test_interpolate_raises(self):

        self.assertRaises(ValueError,
                          self.data.interpolate, test_lat=19., test_lon=180., )
        self.assertRaises(ValueError,
                          self.data.interpolate, lat_dim=21., lon_dim=181.)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
    unittest.TextTestRunner(verbosity=2).run(suite)
