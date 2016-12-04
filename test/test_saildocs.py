import unittest
import itertools

import numpy as np

from slocum.query import saildocs, utils


class SaildocsTest(unittest.TestCase):

    def test_parse_gridded_request(self):

        tests = [('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind'],
                   }),
                 ('GFS:14S,20S,154W,146W|0.5,0.5|0,3..120',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   }),
                 (u'GFS:14S,20S,154W,146W\u015a0.5,0.5',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': [24., 48., 72.],
                   'variables': ['wind']
                   }),
                 ('GFS:14S,20S,154W,146W',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': None,
                   'hours': [24., 48., 72.],
                   'variables': ['wind']
                   }),
                 ('GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120| WIND,',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   }),
                 ('GFS : 14S,20S,154W, 146W/  0.5 |0, 3.. 120| WIND,',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   }),
                 ('GFS : 14S,20S,154W, 146W/  native |0, 3.. 120| WIND,',
                  {'domain': utils.parse_domain('14S,20S,154W,146W'),
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': None,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind']
                   })]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_gridded_request(request)
            self.assertDictEqual(actual, expected)

    def test_parse_spot_request(self):

        tests = [('spot:20S,154W',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gefs',
                   'type': 'spot',
                   'send-image': False,
                   'hours': np.linspace(0, 120, 21).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot:20S,154W|4,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gefs',
                   'type': 'spot',
                   'send-image': False,
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot:20S,154W|4,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gefs',
                   'type': 'spot',
                   'send-image': False,
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot: 20S,154W |4 ,3',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gefs',
                   'send-image': False,
                   'type': 'spot',
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot:20S,154W|4,3|wind|image',
                  {'location': {'latitude': -20., 'longitude':-154.},
                   'model': 'gefs',
                   'type': 'spot',
                   'send-image': True,
                   'hours': np.linspace(0, 96, 33).astype('int'),
                   'variables': ['wind']
                   }),
                 ('spot:26S,32.9E|0,3..180|WIND',
                  {'location': {'latitude': -26., 'longitude': 32.9},
                   'model': 'gefs',
                   'type': 'spot',
                   'send-image': False,
                   'hours': np.linspace(0, 180, 61).astype('int'),
                   'variables': [u'wind']
                    }),
                 ]

        for request, expected in tests:
            self.maxDiff = 3000
            actual = saildocs.parse_spot_request(request)
            np.testing.assert_array_equal(actual.pop('hours'),
                                          expected.pop('hours'))
            actual_location = actual.pop('location')
            expected_location = expected.pop('location')
            np.testing.assert_allclose([actual_location['latitude'],
                                        actual_location['longitude']],
                                       [expected_location['latitude'],
                                        expected_location['longitude']])
            self.assertDictEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
