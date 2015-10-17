import unittest
import warnings
import itertools

from slocum.query import utils

class QueryUtilsTest(unittest.TestCase):

    def test_parse_hours(self):
        tests = [('0,3...12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3...,12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3,...12,24', [0, 3, 6, 9, 12, 24]),
                 ('0,3,...,12,24', [0, 3, 6, 9, 12, 24]),
                 ('0, 3,...12, 24', [0, 3, 6, 9, 12, 24]),
                 ('0,12...72', [0, 12, 24, 36, 48, 60, 72]),
                 ('6,12,18,24,36,48,72,96', [6, 12, 18, 24, 36, 48, 72, 96])]

        for hour_str, expected in tests:
            actual = utils.parse_hours(hour_str)
            self.assertEqual(actual, expected)

        with warnings.catch_warnings(record=True) as w:
            actual = utils.parse_hours(None)
            self.assertEqual(actual, [24., 48., 72.])
        self.assertEqual(len(w), 1)

        bad_queries = ['0,3...8', '12,24,36...82,92']
        for bad in bad_queries:
            self.assertRaises(utils.BadQuery,
                              lambda: list(utils.parse_hours(bad)))

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
            actual = utils.parse_domain(domain_str)
            self.assertEqual(expected, actual)

        bad_queries = ['100N,10S,10E,20E',
                       '10#,10S,10E,20E',
                       '10E,10S,10E,20E',
                       '10N,10S,10N,20E',
                       '10N,10E',
                       '10N,10S,10E,400E',
                       '10N,10,10E,20E']

        for bad in bad_queries:
            self.assertRaises(utils.BadQuery,
                              lambda: utils.parse_domain(bad))

    def test_parse_grid(self):

        tests = [('0.5,0.5', 0.5),
                 ('native', None),
                 ('1.0,0.5', 1.),
                 ('2,2', 2.)]

        for resol_str, expected in tests:
            actual = utils.parse_resolution(resol_str)
            self.assertAlmostEqual(actual, expected, places=2)

        actual = utils.parse_resolution(None)
        self.assertAlmostEqual(actual, 2., places=2)

        bad_queries = ['0.6,0.5,0.7', ]

        for bad in bad_queries:
            self.assertRaises(utils.BadQuery,
                              lambda: utils.parse_resolution(bad))

    def test_validate_variables(self):

        supported = utils._variables.keys()

        for order in range(2):
            for sup_vars in itertools.permutations(supported, order + 1):
                actual = utils.validate_variables(sup_vars)
                self.assertSetEqual(set(sup_vars), set(actual))

                # add an unsupported variable and make sure it
                # gets filtered out.
                with warnings.catch_warnings(record=True) as w:
                    unsup_vars = ['doradodensity']
                    unsup_vars.extend(sup_vars)
                    actual = utils.validate_variables(unsup_vars)
                    self.assertSetEqual(set(sup_vars), set(actual))
                    self.assertEqual(len(w), 1)
                    self.assertIn(unsup_vars[0], w[0].message.message)

        actual = utils.validate_variables([' wind'])
        self.assertEqual(['wind'], actual)

        actual = utils.validate_variables([' wind '])
        self.assertEqual(['wind'], actual)

        with warnings.catch_warnings(record=True) as w:
            default = utils.validate_variables([])
            self.assertEquals(default, ['wind'])
        self.assertTrue(len(w) == 1)

    def test_validate_model(self):
        supported = utils._models.keys()

        with warnings.catch_warnings(record=True) as w:
            for sup_mod in supported:
                actual = utils.validate_model(sup_mod)
                self.assertEqual(sup_mod, actual)
        self.assertTrue(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            default = utils.validate_model(None)
            self.assertEquals(default, 'gfs')
        self.assertTrue(len(w) == 1)
