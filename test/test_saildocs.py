import unittest
import itertools

import numpy as np

from slocum.query import saildocs, utils


class SaildocsTest(unittest.TestCase):

    def test_parse_grid(self):

        tests = [('0.5,0.5', (0.5, 0.5)),
                 ('1.0,0.5', (1.0, 0.5)),
                 ('2,2', (2., 2.))]

        for grid_str, expected in tests:
            actual = saildocs.parse_grid(grid_str)
#             self.assertTrue(len(warns) == 0)
            self.assertAlmostEqual(actual, expected, places=2)

        actual = saildocs.parse_grid(None)
        self.assertAlmostEqual(actual, (2., 2.), places=2)

        bad_queries = ['0.6,0.5,0.7', '0.5']

        for bad in bad_queries:
            self.assertRaises(utils.BadQuery,
                              lambda: saildocs.parse_grid(bad))


    def test_parse_forecast_request(self):

        tests = [('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind'],
                   }),
                 ('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   }),
                 (u'GFS:14S,20S,154W,146W\u015a0.5,0.5',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': [24., 48., 72.],
                   'variables': ['wind']
                   }),
                 ('GFS:14S,20S,154W,146W',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (2., 2.),
                   'hours': [24., 48., 72.],
                   'variables': ['wind']
                   }),
                 ('GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120| WIND,',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   })]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_forecast_request(request)
            self.assertDictEqual(actual, expected)

    def test_parse_spot_request(self):

        tests = [('spot:20S,154W|4,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'type': 'spot',
                   'send-image': False,
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot: 20S,154W |4 ,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'send-image': False,
                   'type': 'spot',
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot:20S,154W|4,3|wind|image',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'type': 'spot',
                   'send-image': True,
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_spot_request(request)
            np.testing.assert_array_equal(actual.pop('hours'),
                                          expected.pop('hours'))
            self.assertDictEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
