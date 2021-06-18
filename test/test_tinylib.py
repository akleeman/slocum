import sys
import xarray as xra
import numpy as np
import unittest
import warnings

from slocum.compression import tinylib

from utils import create_data

class TinylibTest(unittest.TestCase):

    def test_bool(self):
        np.random.seed(seed=1982)
        original_array = np.random.normal(size=100)
        tiny = tinylib.tiny_bool(original_array > 0.)
        recovered = tinylib.expand_bool(**tiny)
        self.assertFalse(np.any(np.logical_xor(original_array > 0., recovered)))

    def test_masked(self):
        # test packing 2 bit ints of odd length
        np.random.seed(seed=1982)
        shape = (30, 3)
        original_array = np.random.normal(size=shape)
        mask = np.random.uniform(size=shape) > 0.5
        tiny = tinylib.tiny_array(original_array, mask=mask)
        recovered = tinylib.expand_array(mask=mask, **tiny)
        self.assertEqual(np.sum(np.isfinite(recovered)), np.sum(mask))
        self.assertLess(np.max(np.abs(recovered[mask] - original_array[mask])),
                        np.max(np.diff(tiny['divs'])))

    def test_consistent(self):
        np.random.seed(seed=1982)
        dirs = np.random.uniform(-np.pi, np.pi,
                                 size=(3960,)).astype('float32')
        dir_bins = np.linspace(-15 * np.pi/16., 15 * np.pi/16., 16)
        tiny = tinylib.tiny_array(dirs, divs=dir_bins, wrap=True)
        tiny['wrap_val'] = np.pi
        recovered = tinylib.expand_array(**tiny)
        # cater for wrap-around case (dir < 0 mapped onto +np.pi)
        diffs = np.where((dirs < 0) & (recovered > 0), recovered + dirs,
                recovered - dirs)
        self.assertLessEqual(np.max(np.abs(diffs)),
                             np.pi / 16)

        original_array = np.random.normal(size=(30, 3))
        tiny = tinylib.tiny_array(original_array)
        recovered = tinylib.expand_array(**tiny)
        self.assertEqual(original_array.shape, recovered.shape)
        self.assertLessEqual(np.max(np.abs(original_array - recovered)),
                             np.max(np.diff(tiny['divs'])))

        precip = np.random.normal(size=3742).astype('float32')
        precip[precip < 0] = 0.
        tiny = tinylib.tiny_array(precip, divs=np.array([0., 1e-8, 0.5, 10.]))
        recovered = tinylib.expand_array(**tiny)
        self.assertEqual(precip.shape, recovered.shape)
        self.assertLessEqual(np.max(np.abs(precip - recovered)),
                             np.max(np.diff(tiny['divs'])))

    def test_tiny(self):
        bins = np.arange(11.)
        # all this data falls exactly on the mid points so it
        # should be preserved through a round trip
        linear = np.arange(10.) + 0.5
        tiny = tinylib.tiny_array(linear, bits=4, divs=bins)
        recovered = tinylib.expand_array(**tiny)
        np.testing.assert_array_equal(linear, recovered)
        # check idempotent
        tiny = tinylib.tiny_array(recovered, bits=4, divs=bins)
        recovered_again = tinylib.expand_array(**tiny)
        np.testing.assert_array_equal(recovered_again, recovered)

        # now we should get some that fall outside the divs
        # which should get rounded to the upper and lower limits.
        linear = np.arange(12.) - 0.5
        with warnings.catch_warnings(record=True) as w:
            tiny = tinylib.tiny_array(linear, bits=4, divs=bins)
            recovered = tinylib.expand_array(**tiny)
        assert len(w)

        np.testing.assert_array_equal(recovered[1:-1], linear[1:-1])
        self.assertTrue(recovered[0] < bins[0])
        self.assertTrue(recovered[-1] > bins[-1])
        # check idempotent
        tiny = tinylib.tiny_array(recovered, bits=4, divs=bins)
        recovered_again = tinylib.expand_array(**tiny)
        np.testing.assert_array_equal(recovered_again, recovered)

        # give it a shot with wrapping
        wrap_val = 10.5
        tiny = tinylib.tiny_array(linear, bits=4, divs=bins, wrap=True)
        recovered = tinylib.expand_array(wrap_val=wrap_val, **tiny)
        np.testing.assert_array_equal(recovered[1:-1], linear[1:-1])
        # anything less than zero or greater than 10 should end up 10.5
        self.assertEqual(recovered[0], wrap_val)
        self.assertEqual(recovered[-1], wrap_val)
        # check idempotent
        tiny = tinylib.tiny_array(recovered, bits=4, divs=bins, wrap=True)
        recovered_again = tinylib.expand_array(wrap_val=wrap_val, **tiny)
        np.testing.assert_array_equal(recovered_again, recovered)

    def test_pack_unpack(self):
        # test packing 2 bit ints of odd length
        orig = np.mod(np.arange(15), 4)
        packed = tinylib.pack_ints(orig)
        recovered = tinylib.unpack_ints(**packed)
        self.assertTrue(np.all(orig == recovered))

        # test packing 2 bit ints of odd length
        orig = np.mod(np.arange(15), 4).reshape((5, 3))
        packed = tinylib.pack_ints(orig)
        recovered = tinylib.unpack_ints(**packed)
        self.assertTrue(np.all(orig == recovered))

        # test packing a short series of 2 bit ints
        orig = np.array([3, 2, 1, 2])
        packed = tinylib.pack_ints(orig)
        recovered = tinylib.unpack_ints(**packed)
        self.assertTrue(np.all(orig == recovered))

        # test packing 4 bit ints with an odd length
        orig = np.arange(15)
        packed = tinylib.pack_ints(orig)
        recovered = tinylib.unpack_ints(**packed)
        self.assertTrue(np.all(orig == recovered))

        # test packing 4 bit ints with an even length
        orig = np.arange(16)
        packed = tinylib.pack_ints(orig)
        recovered = tinylib.unpack_ints(**packed)
        self.assertTrue(np.all(orig == recovered))

    def test_small(self):
        least_significant_digit = 2
        expected = np.random.normal(size=102).reshape(51, 2)
        small = tinylib.small_array(expected, least_significant_digit=2)
        actual = tinylib.expand_small_array(**small)
        actual = actual.reshape(expected.shape)
        np.testing.assert_allclose(actual, expected,
                                   rtol=1.,
                                   atol=np.power(10, -least_significant_digit))

    def test_small_time(self):
        ds = create_data()

        sm_time = tinylib.small_time(ds['time'])
        num_times, units = tinylib.expand_small_time(sm_time['packed_array'])
        actual = xray.Dataset({'time': ('time', num_times,
                                        {'units': units})})
        actual = xray.decode_cf(actual)
        self.assertTrue(np.all(actual['time'].values == ds['time'].values))
        self.assertTrue(units == ds['time'].encoding['units'])


if __name__ == "__main__":
    sys.exit(unittest.main())
