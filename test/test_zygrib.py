import unittest
import numpy as np

from slocum.query import zygrib

class ZygribTest(unittest.TestCase):

    def assertQueryEqual(self, q1, q2):
        np.testing.assert_array_equal(q1.pop('hours'),
                                      q2.pop('hours'))
        self.assertDictEqual(q1, q1)

    def test_parse_query(self):

        tests = [(
"""area: 40.1N,35.1N,120W,-125E
resol: 0.25
days: 3
hours: 3
waves: WW3-GLOBAL
meteo: GFS
WIND PRESS""",    {'domain': {'N': np.float64(40.1),
                              'S': np.float64(35.1),
                              'W': np.float64(-125.),
                              'E': np.float64(-120.)},
                 'model': 'gfs',
                 'type': 'gridded',
                 'resolution': 0.25,
                 'hours': np.arange(25.) * 3,
                 'variables': ['press', 'wind'],
                }),
        # This uses an aliased name for resolution
                 (
"""
area: 40.1N,35.1N,120W,-125E
grid: 0.25
days: 3
hours: 3
waves: WW3-GLOBAL
meteo: GFS
WIND PRESS
        """,    {'domain': {'N': np.float64(40.1),
                            'S': np.float64(35.1),
                            'W': np.float64(-125.),
                            'E': np.float64(-120.)},
                 'model': 'gfs',
                 'type': 'gridded',
                 'resolution': 0.25,
                 'hours': np.arange(25.) * 3,
                 'variables': ['press', 'wind'],
                }),
                 ]

        for query, expected in tests:
            actual = zygrib.parse_query(query)
            self.assertQueryEqual(actual, expected)
