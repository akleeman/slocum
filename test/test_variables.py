import sys
import xray
import numpy as np
import unittest

from slocum.query import utils
from slocum.query import variables

from utils import create_ensemble_data


class VariablesTest(unittest.TestCase):

    def test_roundtrip(self):
        ds = create_ensemble_data()
        for vn in utils.available_variables(ds):
            v = utils.get_variable(vn)
            orig = ds.copy(deep=True)
            expected = v.decompress(v.compress(ds), ds)
            # make sure two passes yields the same result
            actual = v.decompress(v.compress(expected), ds)
            actual.identical(expected)