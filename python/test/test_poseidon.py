import unittest
import itertools

import numpy as np

from sl import poseidon
import xray
import datetime


class PoseidonTest(unittest.TestCase):

    def test_latitude_slicer(self):

        queries = [((10., -10.), 1.9, np.linspace(10., -10., 11)),
                   ((10., -10.), 2.1, np.linspace(10., -10., 11)),
                   ((30., 10.), 0.5, np.linspace(30., 10., 41)),
                   ((-10., -30.), 0.5, np.linspace(-10., -30., 41)),
                   ((10., -10.), 2.0, np.linspace(10., -10., 11)),]

        for (north, south), delta, expected in queries:
            query = {'domain': {'N': north, 'S': south,
                                'E': -150, 'W': -170},
                 'grid_delta': (delta, 0.5)}

            lats = np.linspace(-90, 90., 361)
            slicer = poseidon.latitude_slicer(lats, query)
            np.testing.assert_array_equal(expected, lats[slicer])

            lats = np.linspace(90, -90., 361)
            slicer = poseidon.latitude_slicer(lats, query)
            np.testing.assert_array_equal(expected, lats[slicer])

        lats = np.linspace(-90, 90, 361)
        # add an irregularly spaced grid
        lats[180] = 1.1
        self.assertRaises(Exception,
                          lambda: poseidon.latitude_slicer(lats, query))

    def test_longitude_slicer(self):

        queries = [((10., 30.), 0.5, np.linspace(10., 30., 41)),
                   ((10., 30.), 1.0, np.linspace(10., 30., 21)),
                   ((170., -170.), 0.5, np.linspace(170., 190., 41)),
                   ((170., -170.), 1.1, np.linspace(170., 190., 21)),
                   ]

        for (west, east), delta, expected in queries:
            query = {'domain': {'N': 10., 'S': -10.,
                                'E': east, 'W': west},
                 'grid_delta': (0.5, delta)}

            lons = np.linspace(0., 360., 721)
            slicer = poseidon.longitude_slicer(lons, query)
            np.testing.assert_array_equal(expected, lons[slicer])

        lons = np.linspace(0., 360., 721)
        # add an irregularly spaced grid
        lons[180] = 1.1
        self.assertRaises(Exception,
                          lambda: poseidon.longitude_slicer(lons, query))

        lons = np.linspace(0., 360., 721)
        query = {'domain': {'N': 10., 'S': -10.,
                            'E': 10., 'W': -10.},
                 'grid_delta': (0.5, delta)}
        self.assertRaises(Exception,
                          lambda: poseidon.longitude_slicer(lons, query))

    def test_time_slicer(self):

        queries = [(np.array([0., 24, 48]))
                   ]

        time = xray.XArray(('time',), [0, 6, 12, 18, 24, 36, 48, 72, 96],
                           attributes={'units': 'hours since 2011-01-01'})
        time = xray.conventions.decode_cf_variable(time)

        for hours in queries:
            query = {'hours': hours}
            max_hours = int(max(hours))
            slicer = poseidon.time_slicer(time, query)
            actual = time.data[slicer][-1] - time.data[slicer][0]
            expected = np.timedelta64(max_hours, 'h')
            self.assertEqual(actual, expected)

if __name__ == "__main__":
    unittest.main()
