import sys
import xray
import numpy as np
import unittest

from slocum.lib import tinylib


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
        dirs = np.random.uniform(-np.pi, np.pi, size=(3960,)).astype('float32')
        dir_bins = tinylib._direction_bins

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
        ds = xray.Dataset()
        ds['time'] = (('time'), np.arange(10),
                      {'units': 'hours since 2013-12-12 12:00:00'})
        sm_time = tinylib.small_time(ds['time'])
        ret = tinylib.expand_small_time(**sm_time)
        self.assertTrue(np.all(ret[0] == ds['time'].values))
        self.assertTrue(ret[1] == ds['time'].attrs['units'])

    def test_beaufort(self):
        np.random.seed(1982)
        ds = xray.Dataset()

        ds['time'] = ('time', np.arange(10),
                      {'units': 'hours since 2013-12-12 12:00:00'})
        ds['longitude'] = (('longitude'),
                           np.mod(np.arange(235., 240.) + 180, 360) - 180,
                           {'units': 'degrees east'})
        ds['latitude'] = ('latitude',
                          np.arange(35., 40.),
                          {'units': 'degrees north'})

        shape = tuple([ds.dims[x]
                       for x in ['time', 'longitude', 'latitude']])
        mids = 0.5 * (tinylib._beaufort_scale[1:] +
                      tinylib._beaufort_scale[:-1])
        speeds = mids[np.random.randint(mids.size, size=10 * 5 * 5)]
        speeds = speeds.reshape(shape)
        dirs = tinylib._direction_bins + np.pi / 16
        dirs = dirs[np.random.randint(mids.size, size=10 * 5 * 5)]
        dirs = dirs.reshape(shape)
        uwnd = - speeds * np.sin(dirs)
        uwnd = uwnd.reshape(shape).astype(np.float32)
        vwnd = - speeds * np.cos(dirs)
        vwnd = vwnd.reshape(shape).astype(np.float32)

        ds['uwnd'] = (('time', 'longitude', 'latitude'),
                      uwnd, {'units': 'm/s'})
        ds['vwnd'] = (('time', 'longitude', 'latitude'),
                      vwnd, {'units': 'm/s'})

        beaufort = tinylib.to_beaufort(ds)
        actual = tinylib.from_beaufort(beaufort)
        np.testing.assert_allclose(actual['uwnd'].values, ds['uwnd'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['vwnd'].values, ds['vwnd'].values,
                                   atol=1e-4, rtol=1e-4)

        # now add precip and test everything
        mids = 0.5 * (tinylib._precip_scale[1:] + tinylib._precip_scale[:-1])
        precip = mids[np.random.randint(mids.size, size=10 * 5 * 5)]

        ds['precip'] = (('time', 'longitude', 'latitude'),
                        precip.reshape(shape),
                        {'units': 'kg.m-2.s-1'})
        beaufort = tinylib.to_beaufort(ds)
        actual = tinylib.from_beaufort(beaufort)
        np.testing.assert_allclose(actual['uwnd'].values, ds['uwnd'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['vwnd'].values, ds['vwnd'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['precip'].values, ds['precip'].values,
                                   atol=1e-4, rtol=1e-4)

        # add pressure and test everything
        mids = 0.5 * (tinylib._pressure_scale[1:] +
                      tinylib._pressure_scale[:-1])
        pres = mids[np.random.randint(mids.size, size=10 * 5 * 5)]
        ds['pressure'] = (('time', 'longitude', 'latitude'),
                           pres.reshape(shape),
                           {'units': 'Pa'})
        beaufort = tinylib.to_beaufort(ds)
        actual = tinylib.from_beaufort(beaufort)
        np.testing.assert_allclose(actual['uwnd'].values, ds['uwnd'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['vwnd'].values, ds['vwnd'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['precip'].values, ds['precip'].values,
                                   atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(actual['pressure'].values, ds['pressure'].values,
                                   atol=1e-4, rtol=1e-4)

if __name__ == "__main__":
    sys.exit(unittest.main())
