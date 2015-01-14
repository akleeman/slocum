import unittest
import itertools

import numpy as np

from slocum.lib import saildocs


class SaildocsTest(unittest.TestCase):

    def test_parse_hours(self):
        tests = [('0,3...12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3...,12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3,...12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3,...,12,24', [0, 3, 6, 9, 12, 24]),
                 ('0, 3,...12, 24', [0, 3, 6, 9, 12, 24]),
                 ('0,12...72', [0, 12, 24, 36, 48, 60, 72]),
                 ('6,12,18,24,36,48,72,96', [6, 12, 18, 24, 36, 48, 72, 96])]

        for hour_str, expected in tests:
            actual, warns = saildocs.parse_hours(hour_str)
            self.assertEqual(actual, expected)
            self.assertTrue(len(warns) == 0)

        actual, warns = saildocs.parse_hours(None)
        self.assertEqual(actual, [24., 48., 72.])
        self.assertTrue(len(warns) == 1)

        bad_queries = ['0,3...8', '12,24,36...82,92']
        for bad in bad_queries:
            self.assertRaises(saildocs.BadQuery,
                              lambda: list(saildocs.parse_hours(bad)))

    def test_parse_grid(self):

        tests = [('0.5,0.5', (0.5, 0.5)),
                 ('1.0,0.5', (1.0, 0.5)),
                 ('2,2', (2., 2.))]

        for grid_str, expected in tests:
            actual, warns = saildocs.parse_grid(grid_str)
            self.assertTrue(len(warns) == 0)
            self.assertAlmostEqual(actual, expected, places=2)

        actual, warns = saildocs.parse_grid(None)
        self.assertAlmostEqual(actual, (2., 2.), places=2)
        self.assertTrue(len(warns) == 1)

        bad_queries = ['0.6,0.5', '0.6,0.6', '0.5,0.6', '0.5']

        for bad in bad_queries:
            self.assertRaises(saildocs.BadQuery,
                              lambda: saildocs.parse_grid(bad))

    def test_parse_domain(self):

        tests = [('10N,10S,10E,20E', {'N': 10., 'S': -10.,
                                      'W': 10., 'E': 20.}),
                 ('10S,10N,20E,10E', {'N': 10., 'S': -10.,
                                      'W': 10., 'E': 20.}),
                 ('10N,10S,10W,20E', {'N': 10., 'S': -10.,
                                      'W': -10, 'E': 20.}),
                 ('10N,20N,10W,20E', {'N': 20., 'S': 10.,
                                      'W': -10, 'E': 20.}),
                 ('10N,20N,170W,170E', {'N': 20., 'S': 10.,
                                        'W': 170., 'E': -170.}),
                 ('10N,20N,170E,170W', {'N': 20., 'S': 10.,
                                        'W': 170., 'E': -170.}),
                 ('10N,20N,190E,170E', {'N': 20., 'S': 10.,
                                        'W': 170., 'E': -170.})
                 ]

        for domain_str, expected in tests:
            actual = saildocs.parse_domain(domain_str)
            self.assertEqual(expected, actual)

        bad_queries = ['100N,10S,10E,20E',
                       '10#,10S,10E,20E',
                       '10E,10S,10E,20E',
                       '10N,10S,10N,20E',
                       '10N,10E',
                       '10N,10S,10E,400E',
                       '10N,10,10E,20E']

        for bad in bad_queries:
            self.assertRaises(saildocs.BadQuery,
                              lambda: saildocs.parse_domain(bad))

    def test_validate_variables(self):

        supported = saildocs._supported_variables

        for order in range(2):
            for sup_vars in itertools.permutations(supported, order + 1):
                actual, warns = saildocs.validate_variables(sup_vars)
                self.assertSetEqual(set(sup_vars), set(actual))
                self.assertTrue(len(warns) == 0)

                unsup_vars = ['dorado density']
                unsup_vars.extend(sup_vars)
                actual, warns = saildocs.validate_variables(unsup_vars)
                self.assertSetEqual(set(sup_vars), set(actual))
                self.assertTrue(len(warns) == 1)
                self.assertTrue(unsup_vars[0] in warns[0])

        actual, warns = saildocs.validate_variables([' wind'])
        self.assertEqual(['wind'], actual)

        actual, warns = saildocs.validate_variables([' wind '])
        self.assertEqual(['wind'], actual)

        default, warns = saildocs.validate_variables([])
        self.assertEquals(default, ['wind'])
        self.assertTrue(len(warns) == 1)

    def test_validate_model(self):
        supported = saildocs._supported_models

        for sup_mod in supported:
            actual, warns = saildocs.validate_model(sup_mod)
            self.assertEqual(sup_mod, actual)
            self.assertTrue(len(warns) == 0)

        default, warns = saildocs.validate_model(None)
        self.assertEquals(default, 'gfs')
        self.assertTrue(len(warns) == 1)

    def test_parse_forecast_request(self):

        tests = [('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND',
                  {'domain': saildocs.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'vars': ['wind'],
                   'warnings': []}),
                 ('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120',
                  {'domain': saildocs.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'vars': ['wind'],
                   'warnings': ['No variable requested, defaulting to WIND']}),
                 (u'GFS:14S,20S,154W,146W\u015a0.5,0.5',
                  {'domain': saildocs.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': [24., 48., 72.],
                   'vars': ['wind'],
                   'warnings': ['No hours defined, using default of 24,48,72',
                                'No variable requested, defaulting to WIND']}),
                 ('GFS:14S,20S,154W,146W',
                  {'domain': saildocs.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (2., 2.),
                   'hours': [24., 48., 72.],
                   'vars': ['wind'],
                   'warnings': ['No grid size defined, defaulted to 2 degrees',
                                'No hours defined, using default of 24,48,72',
                                'No variable requested, defaulting to WIND']}),
                 ('GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120| WIND,',
                  {'domain': saildocs.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'grid_delta': (0.5, 0.5),
                   'hours': list(np.arange(41.) * 3),
                   'vars': ['wind'],
                   'warnings': []})]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_forecast_request(request)
            self.assertDictEqual(actual, expected)

    def test_parse_spot_request(self):

        tests = [('spot:20S,154W|4,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'type': 'spot',
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'vars': ['wind'],
                   'warnings': ['No variable requested, defaulting to WIND']}),
                 ('spot: 20S,154W |4 ,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'type': 'spot',
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'vars': ['wind'],
                   'warnings': ['No variable requested, defaulting to WIND']}),
                 ('spot:20S,154W|4,3|wind',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gfs',
                   'type': 'spot',
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'vars': ['wind'],
                   'warnings': []}),
                 ]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_spot_request(request)
            np.testing.assert_array_equal(actual.pop('hours'),
                                          expected.pop('hours'))
            self.assertDictEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
