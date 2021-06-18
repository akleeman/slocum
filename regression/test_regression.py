import os
import xarray as xra
import zlib
import numpy as np
import base64
import unittest
import datetime
import functools
import contextlib

from email import Parser

from slocum.lib import emaillib
from slocum.query import subset
from slocum.compression import compress

_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')


class TestRegression(unittest.TestCase):

    @contextlib.contextmanager
    def get_request(self, email_store):
        from slocum.query import request
        def mock_send_email(*args, **kwdargs):
            email_store.append(functools.partial(lambda x: x, *args, **kwdargs))
        # save the actual send email method for the future
        send_email = request.emaillib.send_email
        # replace with our mock_send_email
        request.emaillib.send_email = mock_send_email
        # use our mocked out version of request
        yield request
        # then when done return emaillib to normal
        request.emaillib.send_email = send_email

    def test_query(self):
        some_email = "foo@bar.com"
        msg = emaillib.create_email(to="query@ensembleweather.com",
                                    fr=some_email,
                                    body='send GFS:30s,35s,175e,175w|0.5,0.5|0,3..12|WIND')

        emails = []
        # use a mocked emaillib to make sure an email was sent
        with self.get_request(emails) as request:
            request.process_email(msg.as_string(),
                                  fail_hard=True)
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

        # inflate it back into an xra Dataset
        fcst = compress.decompress_dataset(base64.b64decode(bytes))

        # make sure we're actually compressing the file.
        self.assertLessEqual(len(bytes), len(zlib.compress(fcst.to_netcdf())))

        # make sure the lats are what we expect
        expected_lats = np.linspace(-35., -30., 11)
        np.testing.assert_array_equal(fcst['latitude'].values, expected_lats)
        self.assertEqual(fcst['latitude'].attrs['units'].lower(),
                         "degrees north")
        # and the lons
        expected_lons = np.mod(np.linspace(175., 185., 21) + 180, 360) - 180
        np.testing.assert_array_equal(fcst['longitude'].values, expected_lons)
        self.assertEqual(fcst['longitude'].attrs['units'].lower(),
                         "degrees east")
        # and the time
        expected_time = np.linspace(0, 12, 5)
        expected_time = xra.conventions.decode_cf_datetime(expected_time,
                                                            fcst['time'].encoding['units'])
        np.testing.assert_array_equal(fcst['time'].values, expected_time)
        self.assertIn("hours", fcst['time'].encoding['units'])


    def test_spot_forecast(self):
        some_email = "foo@bar.com"
        msg = emaillib.create_email(to="query@ensembleweather.com",
                                    fr=some_email,
                                    body='send spot:gefs:8.53S,115.54E|8,6|wind')

        emails = []
        # use a mocked emaillib to make sure an email was sent
        with self.get_request(emails) as request:
            request.process_email(msg.as_string(),
                                  fail_hard=True)
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

        # inflate it back into an xra Dataset
        fcst = compress.decompress_dataset(base64.b64decode(bytes))

        # make sure we're actually compressing the file.
        self.assertLessEqual(len(bytes), len(zlib.compress(fcst.to_netcdf())))

        # make sure the lats are what we expect
        np.testing.assert_array_almost_equal(fcst['latitude'].values,
                                             np.array([-8.53]), 3)
        self.assertEqual(fcst['latitude'].attrs['units'].lower(),
                         "degrees north")
        # and the lons
        np.testing.assert_array_almost_equal(fcst['longitude'].values,
                                             np.array([115.54]), 3)
        self.assertEqual(fcst['longitude'].attrs['units'].lower(),
                         "degrees east")
        # and the time
        expected_time = np.linspace(0, 192, 33)
        expected_time = xra.conventions.decode_cf_datetime(expected_time,
                                                            fcst['time'].encoding['units'])
        np.testing.assert_array_equal(fcst['time'].values, expected_time)
        self.assertIn("hours", fcst['time'].encoding['units'])

