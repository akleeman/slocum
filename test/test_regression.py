import xray
import zlib
import mock
import numpy as np
import base64
import pytest
import unittest
import functools
import contextlib

from email import Parser

from slocum.lib import emaillib
from slocum.query import request
from slocum.compression import compress


def mock_process_email(msg):
    """
    Takes the same information that request.process_email takes, but then
    returns the email that would have been sent.
    """
    emails = []
    with mock.patch('slocum.lib.emaillib.send_email') as mock_send_email:
        mock_send_email.return_value = None

        # This is a sneaky way of storing the arguments that were passed into
        # the function.
        def mock_send_email_function(*args, **kwdargs):
            emails.append(functools.partial(lambda x: x, *args, **kwdargs))

        mock_send_email.side_effect = mock_send_email_function
        request.process_email(msg.as_string(), fail_hard=True)

    assert len(emails) == 1

    parser = Parser.Parser()
    email = parser.parsestr(emails.pop().args[0].as_string())
    return email


@pytest.mark.regression
def test_query():
    some_email = "foo@bar.com"
    msg = emaillib.create_email(to="query@ensembleweather.com",
                                fr=some_email,
                                body='send GFS:30s,35s,175e,175w|0.5,0.5|0,3..12|WIND')
    email = mock_process_email(msg)

    # make sure we are sending responses from ensembleweather
    assert email['From'] == 'query@ensembleweather.com'

    # make sure we are responding to the right person
    assert email['To'] == some_email

    # here we pull out the expected attachment
    body, attach = email.get_payload()
    bytes = attach.get_payload()

    # inflate it back into an xray Dataset
    fcst = compress.decompress_dataset(base64.b64decode(bytes))

    # make sure we're actually compressing the file.
    assert len(bytes) <= len(zlib.compress(fcst.to_netcdf()))

    # make sure the lats are what we expect
    expected_lats = np.linspace(-35., -30., 11)
    np.testing.assert_array_equal(fcst['latitude'].values, expected_lats)
    assert fcst['latitude'].attrs['units'].lower() == "degrees north"
    # and the lons
    expected_lons = np.mod(np.linspace(175., 185., 21) + 180, 360) - 180
    np.testing.assert_array_equal(fcst['longitude'].values, expected_lons)
    assert fcst['longitude'].attrs['units'].lower() == "degrees east"
    # and the time
    expected_time = np.linspace(0, 12, 5)
    expected_time = xray.conventions.decode_cf_datetime(expected_time,
                                                        fcst['time'].encoding['units'])
    np.testing.assert_array_equal(fcst['time'].values, expected_time)
    assert "hours" in fcst['time'].encoding['units']


@pytest.mark.regression
def test_spot_forecast():
    some_email = "foo@bar.com"
    msg = emaillib.create_email(to="query@ensembleweather.com",
                                fr=some_email,
                                body='send spot:gefs:8.53S,115.54E|8,6|wind')

    email = mock_process_email(msg)

    # make sure we are sending responses from ensembleweather
    assert email['From'] == 'query@ensembleweather.com'

    # make sure we are responding to the right person
    assert email['To'] == some_email

    # here we pull out the expected attachment
    body, attach = email.get_payload()
    bytes = attach.get_payload()

    # inflate it back into an xray Dataset
    fcst = compress.decompress_dataset(base64.b64decode(bytes))

    # make sure we're actually compressing the file.
    assert len(bytes) <= len(zlib.compress(fcst.to_netcdf()))

    # make sure the lats are what we expect
    np.testing.assert_array_almost_equal(fcst['latitude'].values,
                                         np.array([-8.53]), 3)
    assert fcst['latitude'].attrs['units'].lower() == "degrees north"
    # and the lons
    np.testing.assert_array_almost_equal(fcst['longitude'].values,
                                         np.array([115.54]), 3)
    assert fcst['longitude'].attrs['units'].lower() == "degrees east"
    # and the time
    expected_time = np.linspace(0, 192, 33)
    expected_time = xray.conventions.decode_cf_datetime(expected_time,
                                                        fcst['time'].encoding['units'])
    np.testing.assert_array_equal(fcst['time'].values, expected_time)
    assert "hours" in fcst['time'].encoding['units']
