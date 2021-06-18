import xarray as xra
import unittest

import numpy as np

from slocum.query import subset

import utils


class SubsetTest(unittest.TestCase):

    def test_latitude_slicer(self):

        queries = [((10., -10.), 1.9, np.linspace(10., -10., 11)),
                   ((10., -10.), 2.1, np.linspace(10., -10., 11)),
                   ((30., 10.), 0.5, np.linspace(30., 10., 41)),
                   ((-10., -30.), 0.5, np.linspace(-10., -30., 41)),
                   ((10., -10.), 2.0, np.linspace(10., -10., 11)),]

        for (north, south), resol, expected in queries:
            query = {'domain': {'N': north, 'S': south,
                                'E': -150, 'W': -170},
                     'resolution': resol}

            lats = np.linspace(-90, 90., 361)
            slicer = subset.latitude_slicer(lats, query)
            np.testing.assert_array_equal(np.sort(expected), lats[slicer])

            lats = np.linspace(90, -90., 361)
            slicer = subset.latitude_slicer(lats, query)
            np.testing.assert_array_equal(np.sort(expected)[::-1], lats[slicer])

        lats = np.linspace(-90, 90, 361)
        # add an irregularly spaced grid
        lats[180] = 1.1
        self.assertRaises(Exception,
                          lambda: subset.latitude_slicer(lats, query))

    def test_longitude_slicer(self):

        queries = [((-10., 10.), 0.5, np.mod(np.linspace(-10, 10., 41), 360.)),
                   ((10., 30.), 0.5, np.linspace(10., 30., 41)),
                   ((10., 30.), 1.0, np.linspace(10., 30., 21)),
                   ((170., -170.), 0.5, np.linspace(170., 190., 41)),
                   ((170., -170.), 1.1, np.linspace(170., 190., 21)),
                   ]

        for (west, east), resol, expected in queries:
            query = {'domain': {'N': 10., 'S': -10.,
                                'E': east, 'W': west},
                     'resolution': resol}

            # Try when the longitudes are defined on 0. to 360.
            lons = np.linspace(0., 359.5, 720)
            slices = subset.longitude_slicer(lons, query)
            
            if isinstance(slices, list):
                sliced = np.concatenate([lons[s] for s in slices])
            else:
                sliced = lons[slices]

            np.testing.assert_array_equal(expected, sliced)

            # And again when they're defined on -180. to 180.
            lons_180 = np.mod(lons + 180., 360.) - 180.
            slices = subset.longitude_slicer(lons_180, query)

            if isinstance(slices, list):
                sliced = np.concatenate([lons[s] for s in slices])
            else:
                sliced = lons[slices]

            np.testing.assert_array_equal(expected, sliced)


        lons = np.linspace(0., 359.5, 720)
        # add an irregularly spaced grid
        lons[180] = 1.1
        self.assertRaises(Exception,
                          lambda: subset.longitude_slicer(lons, query))

    def test_time_slicer(self):

        queries = [(np.array([0., 24, 48]))
                   ]

        time = xray.Dataset()
        time['time'] = (('time', [0, 6, 12, 18, 24, 36, 48, 72, 96],
                        {'units': 'hours since 2011-01-01'}))
        time = xray.conventions.decode_cf_variable(time['time'].variable)

        for hours in queries:
            query = {'hours': hours}
            max_hours = int(max(hours))
            slicer = subset.time_slicer(time, query)
            actual = time.values[slicer][-1] - time.values[slicer][0]
            expected = np.timedelta64(max_hours, 'h')
            self.assertEqual(actual, expected)

    def test_subset(self):
        fcst = utils.create_data()

        query = {'hours': np.array([0., 2., 4., 6.]),
                 'domain': {'N': np.max(fcst['latitude'].values) - 1,
                            'S': np.min(fcst['latitude'].values) + 1,
                            'E': np.max(fcst['longitude'].values) - 1,
                            'W': np.min(fcst['longitude'].values) + 1},
                 'grid_delta': (1., 1.),
                 'variables': ['wind']}
        ss = subset.subset_dataset(fcst, query)

        np.testing.assert_array_equal(ss['longitude'].values,
                                      np.arange(query['domain']['W'],
                                                query['domain']['E'] + 1))
        np.testing.assert_array_equal(np.sort(ss['latitude'].values),
                                      np.arange(query['domain']['S'],
                                                query['domain']['N'] + 1))


    def test_forecast_containing_point(self):
        fcst = utils.create_data()
        lat = np.random.uniform(np.min(fcst['latitude'].values),
                                np.max(fcst['latitude'].values))
        lon = np.random.uniform(np.min(fcst['longitude'].values),
                                np.max(fcst['longitude'].values))
        query = {'location': {'latitude': lat, 'longitude': lon},
                   'model': 'gfs',
                   'type': 'spot',
                   'hours': np.linspace(0, 9, 3).astype('int'),
                   'variables': ['wind'],
                   'warnings': []}

        modified_query = subset.query_containing_point(query)
        ss = subset.subset_gridded_dataset(fcst, modified_query)

        self.assertTrue(np.any(lon >= ss['longitude'].values))
        self.assertTrue(np.any(lon <= ss['longitude'].values))
        self.assertTrue(np.any(lat >= ss['latitude'].values))
        self.assertTrue(np.any(lat <= ss['latitude'].values))

        # we should be able to pass the results through again and get the same thing.
        subset2 = subset.subset_gridded_dataset(ss, modified_query)
        self.assertTrue(subset2.equals(ss))


    def test_subset_spot_dataset(self):

        fcst = utils.create_data()
        times, units, cal = xray.conventions.encode_cf_datetime(fcst['time'])
        assert 'hours' in units

        def test_one_query(lon_slice, lat_slice, hour_slice):
            lon = np.mean(fcst['longitude'].values[lon_slice])
            lat = np.mean(fcst['latitude'].values[lat_slice])
            hours = times[hour_slice]
            query = {'location': {'latitude': lat, 'longitude': lon},
                       'model': 'gefs',
                       'type': 'spot',
                       'hours': hours,
                       'variables': ['wind']}

            ss = subset.subset_spot_dataset(fcst, query)

            assert fcst['x_wind'].dims == ('time', 'longitude', 'latitude')
            expected = np.mean(np.mean(fcst['x_wind'].values[hour_slice,
                                                             lon_slice,
                                                             lat_slice],
                                       axis=2),
                               axis=1)
            np.testing.assert_array_almost_equal(ss['x_wind'].values.reshape(-1),
                                                 expected)

            assert fcst['y_wind'].dims == ('time', 'longitude', 'latitude')
            expected = np.mean(np.mean(fcst['y_wind'].values[hour_slice,
                                                             lon_slice,
                                                             lat_slice],
                                       axis=2),
                               axis=1)
            np.testing.assert_array_almost_equal(ss['y_wind'].values.reshape(-1),
                                                 expected)

            np.testing.assert_array_equal(ss['latitude'].values, lat)
            np.testing.assert_array_equal(ss['longitude'].values, lon)

        # test a query with lat/lon in the middle of a grid.
        test_one_query(slice(0, 2),
                       slice(0, 2),
                       slice(0, None, 3))
        # and with the lat/lon exactly on a grid
        test_one_query(slice(1, 2),
                       slice(1, 2),
                       slice(1, None, 3))


if __name__ == "__main__":
    unittest.main()
