import os
import tempfile
import unittest
import numpy as np

from slocum.lib import emaillib
from slocum.query import request, grads, subset
from slocum.compression import compress

import utils

class RequestTest(unittest.TestCase):

    def setUp(self):
        # create a GFS like forecast on disk
        fcst = utils.create_gfs_data()
        _, fn = tempfile.mkstemp("_gfs.nc")
        fcst.to_netcdf(fn)
        self.temporary_fcst_path = fn
        unittest.TestCase.setUp(self)

    def tearDown(self):
        os.remove(self.temporary_fcst_path)
        unittest.TestCase.tearDown(self)

    def test_iterate_query_strings(self):

        tests = [('SEND GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120| WIND,',
                  ['send gfs : 14s,20s,154w, 146w/  0.5, 0.5 |0, 3.. 120| wind,']),
                 ('SEND GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120| WIND,\n'
                  'someotherline',
                  ['send gfs : 14s,20s,154w, 146w/  0.5, 0.5 |0, 3.. 120| wind,'])]

        for query_text, expected in tests:
            actual = list(request.iterate_query_strings(query_text))
            self.assertEqual(actual, expected)


    def test_parse_query_string(self):
        # this test only needs to make sure that the appropriate parser
        # is used and function, more thorough testing should be done in
        # each of the parsers (test_saildocs for example)
        tests = [('SEND GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND',
                  {'domain': {'N': -14.,
                              'S': -20.,
                              'E': -146.,
                              'W': -154.},
                   'model': 'gfs',
                   'type': 'gridded',
                   'resolution': 0.5,
                   'hours': list(np.arange(41.) * 3),
                   'variables': ['wind'],
                   })]
        for query_string, expected in tests:
            actual = request.parse_query_string(query_string)
            self.assertDictEqual(expected, actual)

    def test_gfs(self):
        query = {'hours': [0., 3., 9., 24.],
                 'model': 'gfs',
                 'domain': {'N': 39.,
                            'S': 35,
                            'E': -120,
                            'W': -124},
                 'grid_delta': (1., 1.),
                 'variables': ['wind']}

        model = grads.GFS()
        gfs = model.fetch(self.temporary_fcst_path)
        expected = subset.subset_dataset(gfs, query)
        compressed = request.process_query(query, url=self.temporary_fcst_path)
        actual = compress.decompress_dataset(compressed)

        np.testing.assert_array_almost_equal(actual['x_wind'].values,
                                             expected['x_wind'].values,
                                             4)
        np.testing.assert_array_almost_equal(actual['y_wind'].values,
                                             expected['y_wind'].values,
                                             4)

    def test_warnings_issued(self):
        query_str = "SEND GFS : 14S,20S,154W, 146W/  0.5, 0.5 |0, 3.. 120"
        response = request.process_single_query(query_str,
                                                reply_to="joshua@slocum.com",
                                                url=self.temporary_fcst_path)
        self.assertIn("defaulting to WIND", response.as_string())
