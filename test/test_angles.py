import unittest
import itertools

import numpy as np

from slocum.lib import angles


class AnglesTest(unittest.TestCase):

    def test_angle_diff(self):

        pairs = [((180, 180), 0.),
                 ((181, 181), 0.),
                 ((-1, -1), 0.),
                 ((180, -180), 0.),
                 # note, wrapped values get sent to -180
                 ((180, 0), -180.),
                 ((0, 180), -180.),
                 ((360, 180), -180.),
                 ((361, 181), -180.),
                 ((179, 0), 179.),
                 ((0, 179), -179.),
                 ((360, 181), 179.),
                 ((1, 182), 179.),
                 ((181, 179), 2.),
                 ((179, 181), -2.),
                 ((361, 359), 2.),
                 ((-1, 1.), -2.),
                 ((np.pi - 0.1, 0., False), np.pi - 0.1),
                 ((np.pi - 0.1, -np.pi + 0.1, False), -0.2)]

        for args, expected in pairs:
            actual = angles.angle_diff(*args)
            self.assertAlmostEqual(expected,
                                   actual,
                                   places=6,
                                   msg=("actual: %f "
                                        "expected: %f "
                                        "args: %s"
                                        % (actual, expected,
                                           args)))


    def test_geographic_distance(self):
        actual = angles.geographic_distance(0., 0., 1., 0.)
        expected = 111194.87 # 60 nautical miles in kilometers
        self.assertAlmostEqual(actual, expected, 1)
        actual = angles.geographic_distance(0., 0.5, 0., -0.5)
        self.assertAlmostEqual(actual, expected, 1)
        actual = angles.geographic_distance(0., 0.5, 0., 0.5)
        self.assertAlmostEqual(actual, 0., 1)

    def test_vector_to_radial_roundtrip(self):
        # round trip some random values.
        for i in range(10):
            magnitude = np.random.uniform(0., 5.)
            direction = np.random.uniform(-np.pi, np.pi)
            vector = angles.radial_to_vector(magnitude, direction)
            radial = angles.vector_to_radial(*vector)
            self.assertAlmostEqual(radial[0], magnitude, 5)
            self.assertAlmostEqual(radial[1], direction, 5)


    def test_vector_radial(self):

        pairs = [((0., 1.), (1., 0.)),
                 ((1., 0.), (1., np.pi / 2)),
                 ((-1., 0.), (1., - np.pi / 2)),
                 ((0., -1.), (1., np.pi)),
                 ((3., 4.), (5., np.arccos(4. / 5.))),
                 ((0., 0.), (0., 0.))]

        for vector, radial in pairs:
            actual = angles.vector_to_radial(*vector)
            # convert from radial to vector and compare
            np.testing.assert_almost_equal(actual,
                                           radial,
                                           err_msg=("vector_to_radial\n"
                                                    "actual: %s "
                                                    "expected: %s "
                                                    "args: %s"
                                                    % (actual, radial,
                                                       vector)))
            # now do the same in reverse
            actual = angles.radial_to_vector(*radial)
            np.testing.assert_almost_equal(actual,
                                           vector,
                                           err_msg=("radial_to_vector\n"
                                                    "actual: %s "
                                                    "expected: %s "
                                                    "args: %s"
                                                    % (actual, vector,
                                                       radial)))
            # now try with direction indicating where the field is flowing from.
            actual = angles.vector_to_radial(*vector, orientation="from")
            # add pi to the expected value.  Note this is done in a way that
            # keeps the result in [-pi, pi).
            radial = (radial[0], np.mod(radial[1] + 2 * np.pi, 2 * np.pi) - np.pi)
            # convert from radial to vector and compare
            np.testing.assert_almost_equal(actual,
                                           radial,
                                           err_msg=("vector_to_radial\n"
                                                    "actual: %s "
                                                    "expected: %s "
                                                    "args: %s"
                                                    % (actual, radial,
                                                       vector)))
