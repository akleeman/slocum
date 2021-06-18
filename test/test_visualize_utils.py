import xarray as xra
import unittest
import numpy as np

from slocum.visualize import utils


def create_domain():
    ds = xray.Dataset()
    ds['longitude'] = ('longitude', np.linspace(115., 145, 31),
                       {'units': 'degrees_east'})
    ds['latitude'] = ('latitude', np.linspace(-8, 8, 17),
                       {'units': 'degrees_north'})
    return ds

class TestVisualizeUtils(unittest.TestCase):

    def test_bounding_box(self):
        ds = create_domain()
        bbox = utils.bounding_box(ds, pad=0.)
        self.assertEqual(bbox['llcrnrlat'], -8.)
        self.assertEqual(bbox['urcrnrlat'], 8.)
        self.assertEqual(bbox['urcrnrlon'], 145.)
        self.assertEqual(bbox['llcrnrlon'], 115.)
        self.assertEqual(bbox['lon_0'], 130.)
        self.assertEqual(bbox['lat_0'], 0.)