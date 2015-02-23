import copy
import xray
import unittest
import numpy as np

from slocum.lib import conventions


class TestConventions(unittest.TestCase):

    def test_roundtrip_encode_datetime(self):
        units_to_test = ['hours since 2013-12-12T12:00:00Z',
                         'hours since 2013-12-12T12:00:00+00:00',
                         'hours since 2013-12-12T12:00:00']
        for units in units_to_test:
            original = xray.Variable(('time'), np.arange(10),
                                     {'units': units})

            as_datetimes = conventions.decode_cf_time_variable(copy.copy(original))
            # reencode datetime
            actual = conventions.encode_cf_datetime(as_datetimes.values,
                                                    as_datetimes.encoding['units'],
                                                    as_datetimes.encoding['calendar'])
            np.testing.assert_array_equal(original.values, actual[0])

            # reencode datetime twice to make sure theres not state change
            actual = conventions.encode_cf_datetime(as_datetimes.values,
                                                    as_datetimes.encoding['units'],
                                                    as_datetimes.encoding['calendar'])
            np.testing.assert_array_equal(original.values, actual[0])

    def test_encode_datetime(self):
        origin = '2013-12-12T12:00:00+00:00'
        num_dates = np.arange(10)
        dates = [np.datetime64(origin) + np.timedelta64(i, 'h')
                 for i in num_dates]
        original = xray.Variable(('time'), dates)
        original.encoding.update({'units': 'hours since %s' % origin})
        # reencode datetime
        actual = conventions.encode_cf_datetime(original.values,
                                                original.encoding['units'])
        np.testing.assert_array_equal(num_dates, actual[0])
        np.testing.assert_array_equal(original.encoding['units'], actual[1])

    def test_decode_datetime(self):
        origin = '2013-12-12T12:00:00+00:00'
        num_dates = np.arange(10)
        dates = [np.datetime64(origin) + np.timedelta64(i, 'h')
                 for i in num_dates]
        original = xray.Variable(('time'), num_dates)
        units = 'hours since %s' % origin
        original.attrs.update({'units': units})
        actual = conventions.decode_cf_datetime(num_dates, units)

        np.testing.assert_array_equal(dates, actual)
