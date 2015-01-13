import os
import xray
import zlib
import numpy as np
import base64
import unittest
import datetime
import functools
import contextlib

from slocum.lib import tinylib

from email import Parser

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')


class WindbreakerRegression(unittest.TestCase):

    @contextlib.contextmanager
    def get_windbreaker(self, email_store):
        from slocum import windbreaker

        def mock_send_email(*args, **kwdargs):
            email_store.append(functools.partial(lambda x: x, *args, **kwdargs))
        # save the actual send email method for the future
        send_email = windbreaker.emaillib.send_email
        # replace with our mock_send_email
        windbreaker.emaillib.send_email = mock_send_email
        # use our mocked out version of windbreaker
        yield windbreaker
        # then when done return emaillib to normal
        windbreaker.emaillib.send_email = send_email

    # test that a sailmail address gets a size check

    def test_query(self):

        emails = []

        with self.get_windbreaker(emails) as windbreaker:
            some_email = "foo@bar.com"
            query_str = 'send GFS:30s,40s,160w,175w|0.5,0.5|0,3..120|WIND'
            query = windbreaker.parse_query(query_str)
            windbreaker.respond_to_query(query, some_email,
                                         forecast_path='%s/test_gfs.nc' % _data_dir)
        assert len(emails) == 1
        parser = Parser.Parser()
        email = parser.parsestr(emails.pop().args[0].as_string())
        # make sure we are sending responses from ensembleweather
        self.assertEqual(email['From'], 'query@ensembleweather.com')
        # make sure we are responding to the right person
        self.assertEqual(email['To'], some_email)
        # here we pull out the expected attachment
        body, attach = email.get_payload()
        bytes = attach.get_payload()
        # inflate it back into an xray Dataset
        fcst = tinylib.from_beaufort(base64.b64decode(bytes))

        # make sure we're actually compressing the file.
        self.assertLessEqual(len(bytes), len(zlib.compress(fcst.dumps())))

        # make sure the lats are what we expect
        expected_lats = np.linspace(-30., -40., 21)
        np.testing.assert_array_equal(fcst['latitude'].values, expected_lats)
        self.assertEqual(fcst['latitude'].attrs['units'].lower(),
                         "degrees north")
        # and the lons
        expected_lons = -np.linspace(175., 160., 31)
        np.testing.assert_array_equal(fcst['longitude'].values, expected_lons)
        self.assertEqual(fcst['longitude'].attrs['units'].lower(),
                         "degrees east")
        # and the time
        expected_time = np.linspace(0, 120, 41)
        expected_time = xray.conventions.decode_cf_datetime(expected_time,
                                                            fcst['time'].encoding['units'])
        np.testing.assert_array_equal(fcst['time'].values, expected_time)
        self.assertIn("hours", fcst['time'].encoding['units'])

        # make sure the forecasts are recent enough
        time = xray.conventions.decode_cf_variable(fcst['time'])
        now = np.datetime64(datetime.datetime.now())
        self.assertLessEqual(now - time.values[0],
                              np.timedelta64(datetime.timedelta(days=1)),
                              "Forecasts are more than a day old.")

