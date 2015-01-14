from slocum.lib.objects import NautAngle
import numpy as np
import unittest


class NautAngleTest(unittest.TestCase):

    def test_init_with_float(self):
        a = NautAngle(315)
        self.assertAlmostEqual(a, -45)
        self.assertAlmostEqual(a.real, -45)
        self.assertAlmostEqual(a.radians, np.radians(-45))

    def test_init_with_string(self):
        a = NautAngle('-45')
        self.assertAlmostEqual(a.real, -45)
        a = NautAngle('45.5s')
        self.assertAlmostEqual(a.real, -45.5)
        a = NautAngle('W 45.')
        self.assertAlmostEqual(a.real, -45)
        a = NautAngle('N-45')
        self.assertAlmostEqual(a.real, -45)
        a = NautAngle(' -45 W')
        self.assertAlmostEqual(a.real, 45)

    def test_init_edge_cases(self):
        a = NautAngle('180')
        self.assertAlmostEqual(a.real, -180)
        a = NautAngle('-180')
        self.assertAlmostEqual(a.real, -180)
        a = NautAngle('0S')
        self.assertAlmostEqual(a.real, 0)

    def test_string_conversion(self):
        a = NautAngle('-30.3137')
        self.assertEqual(str(a), '-30.3137')
        self.assertEqual(repr(a), '-30.3137')

    def test_distance_to(self):
        a = NautAngle(-45)
        b = NautAngle(170)
        self.assertAlmostEqual(float(a.distance_to(b)), -145)
        self.assertAlmostEqual(float(b.distance_to(a)), 145)

    def test_distance_to_edge_cases(self):
        a = NautAngle(90)
        b = NautAngle(-90)
        self.assertEqual(float(a.distance_to(b)), -180)
        self.assertEqual(float(b.distance_to(a)), 180)

    def test_comparison_methods_lon(self):
        a = NautAngle(20)
        b = NautAngle(-170)
        self.assertFalse(a.is_east_of(b))
        self.assertTrue(a.is_west_of(b))
        self.assertTrue(b.is_east_of(a))
        self.assertFalse(b.is_west_of(a))

    def test_comparison_methods_lat(self):
        a = NautAngle(20)
        b = NautAngle(-30)
        self.assertFalse(a.is_south_of(b))
        self.assertTrue(a.is_north_of(b))
        self.assertTrue(b.is_south_of(a))
        self.assertFalse(b.is_north_of(a))

    def test_comparison_methods_edge_cases(self):
        a = NautAngle(-170)
        b = NautAngle(190)
        self.assertFalse(a.is_east_of(b))
        self.assertFalse(a.is_west_of(b))
        self.assertTrue(a.is_almost_equal(b))
        a = NautAngle(90)
        b = NautAngle(-90)
        self.assertTrue(a.is_north_of(b))
        self.assertTrue(b.is_south_of(a))

    def test_almost_equal(self):
        a = NautAngle('45')
        b = NautAngle('45.0001')
        self.assertFalse(a.is_almost_equal(b))
        self.assertTrue(a.is_almost_equal(b, places=4))
