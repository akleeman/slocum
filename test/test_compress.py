import unittest
import numpy as np

from slocum.compression import compress

from utils import create_data

class CompressTest(unittest.TestCase):

    def test_version_assert(self):
        # create a forecast that looks like its from a newer version
        # and make sure an assertion is raised.
        ds = create_data()
        original_version = compress._VERSION
        compress._VERSION = np.array(compress._VERSION + 1, dtype=np.uint8)
        beaufort = compress.compress_dataset(ds)
        compress._VERSION = original_version
        self.assertRaises(ValueError,
                          lambda: compress.decompress_dataset(beaufort))

    def test_stringify(self):
        expected = 'THISISATEST'
        payload = compress._stringify('wind', expected)
        actual_name, actual_info, _ = compress._split_single_variable(payload)

        self.assertEqual(actual_name, 'wind')
        self.assertEqual(expected, actual_info)

    def test_compress_dataset(self):
        ds = create_data()
        compressed = compress.compress_dataset(ds)
        actual = compress.decompress_dataset(compressed)

        np.testing.assert_allclose(actual['x_wind'].values, ds['x_wind'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['y_wind'].values, ds['y_wind'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['air_pressure_at_sea_level'].values,
                                   ds['air_pressure_at_sea_level'].values,
                                   atol=1e-4, rtol=1e-4)
        # pass it through the system again, it should be idempotent.
        compressed = compress.compress_dataset(ds)
        actual = compress.decompress_dataset(compressed)
        np.testing.assert_allclose(actual['x_wind'].values, ds['x_wind'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['y_wind'].values, ds['y_wind'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['air_pressure_at_sea_level'].values,
                                   ds['air_pressure_at_sea_level'].values,
                                   atol=1e-4, rtol=1e-4)
