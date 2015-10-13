import unittest
import numpy as np

from slocum.lib import angles
from slocum.compression import schemes

import utils

def roundtrip(scheme, ds):
    coords = ds.copy()[ds.coords.keys()]
    return scheme.decompress(scheme.compress(ds), coords)

class CommonTests(object):

    def assertRequiredAlmostEqual(self, expected, actual):
        scheme = self.get_scheme()
        for vn in scheme.required_variables():
            np.testing.assert_array_almost_equal(expected[vn].values,
                                                 actual[vn].values,
                                                 decimal=4)
            self.assertEqual(expected[vn].attrs,
                             actual[vn].attrs)

    def test_normalize_idempotent(self):
        ds = self.get_data()
        scheme = self.get_scheme()
        first = scheme.normalize(ds)
        second = scheme.normalize(first)
        self.assertRequiredAlmostEqual(first, second)

    def test_roundtrip_idempotent(self):
        ds = self.get_data()
        scheme = self.get_scheme()
        first = roundtrip(scheme, ds)
        second = roundtrip(scheme, first)
        self.assertRequiredAlmostEqual(first, second)

    def test_roundtrip(self):
        ds = self.get_data()
        scheme = self.get_scheme()
        actual = roundtrip(scheme, ds)
        self.assertRequiredAlmostEqual(ds, actual)

class TinyVariableTest(unittest.TestCase, CommonTests):

    def get_scheme(self):
        return schemes.TinyVariable('great_white_shark_length',
                                    units='m',
                                    bins=np.arange(11))

    def get_data(self):
        ds = utils.create_data()
        bins = self.get_scheme().bins
        mids = 0.5 * (bins[1:] + bins[:-1])
        data = mids[np.random.randint(mids.size, size=ds['x_wind'].size)]
        ds['great_white_shark_length'] = (('time', 'longitude', 'latitude'),
                                          data.reshape(ds['x_wind'].shape),
                                          {'units': 'm'})

        return ds[['great_white_shark_length']]


class TinyDirectionTest(unittest.TestCase, CommonTests):

    def get_scheme(self):
        return schemes.TinyDirection('albatros_flight_direction')

    def get_data(self):
        ds = utils.create_data()
        mids = np.linspace(-7. / 8. * np.pi, np.pi, 16).astype(np.float32)
        data = mids[np.random.randint(mids.size, size=ds['x_wind'].size)]
        ds['albatros_flight_direction'] = (('time', 'longitude', 'latitude'),
                                           data.reshape(ds['x_wind'].shape),
                                           {'units': 'radians'})
        return ds[['albatros_flight_direction']]

    def test_maximum_error(self):
        from slocum.lib import angles
        ds = self.get_data()
        scheme = self.get_scheme()
        actual = roundtrip(scheme, ds)
        diff = angles.angle_diff(ds[scheme.variable_name].values,
                                 actual[scheme.variable_name].values)
        self.assertTrue(np.all(np.abs(diff) <= np.pi/16))


class VelocityVariableTest(unittest.TestCase, CommonTests):

    def get_scheme(self):
        beaufort_bins = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                                  33., 40., 47., 55., 63., 75.]) / 1.94384449
        return schemes.VelocityVariable('x_wind', 'y_wind', 'wind',
                                        speed_bins=beaufort_bins)

    def get_data(self):
        return utils.create_data()


class CombinedVariableTest(unittest.TestCase, CommonTests):

    def get_scheme(self):
        wave_direction = schemes.TinyDirection('sea_surface_wave_to_direction')
        wave_height = schemes.TinyVariable('sea_surface_wave_significant_height',
                                           units='m',
                                           bins=np.array([0., 0.2, 0.5, 0.75, 1., 1.5, 2.,
                                                          3., 4., 5., 6., 7., 8., 10., 15.]))
        wave = schemes.CombinedVariable([wave_height, wave_direction])
        return wave

    def get_data(self):
        scheme = self.get_scheme()
        ds = utils.create_data()
        for x in scheme.variables:
            ds = utils.add_tiny_variable(x, ds)
        return ds
