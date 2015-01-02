import os
import sys
import json
import xray
import zlib
import numpy as np
import base64
import logging
import datetime

from cStringIO import StringIO

from sl import poseidon
from sl.lib import conventions as conv, units
from sl.lib import objects, tinylib, saildocs, emaillib
from sl.lib.objects import NautAngle

_smtp_server = 'localhost'
_windbreaker_email = 'query@ensembleweather.com'

_email_body = """
This forecast brought to you by your friends on Saltbreaker.

Remember, always be skeptical of numerical forecasts (such as this).
For a full disclaimer visit www.ensembleweather.com."""

_info_body = """saildocs-like queries follow the format:

%(send_usage)s

For more info on saildocs queries check out:

    http://www.saildocs.com/gribinfo

or send an email to gribinfo@saildocs.com.
""" % {'send_usage': saildocs._send_usage}


def arg_closest(x, reference):
    """
    Find the closest element index in reference for each
    element of x.  This isn't the most efficient but
    allows nearest neighbors with arbitrary reference
    data.
    """
    return np.array([np.argmin(np.abs(reference - y)) for y in x])


def get_forecast(query, path=None):
    """
    Here we do some crude caching which allows the user to specify a path
    to a local file that holds the data instead of going through
    opendap/poseidon.
    """
    warnings = []
    if path and os.path.exists(path):
        fcst = poseidon.forecast(query, xray.open_dataset(path))
        warnings.append('Using cached forecasts (%s) which may be old.' % path)
    else:
        fcst = poseidon.forecast(query)
        if path is not None:
            fcst.dump(path)
    return fcst


def query_to_beaufort(query_string, forecast_path=None):
    """
    Takes a query string and returns the corresponding tiny forecast.
    """
    logging.debug(query_string)
    query = saildocs.parse_saildocs_query(query_string)
    # log the query so debugging others request failures will be easier.
    logging.debug(json.dumps(query))
    # Acquires a forecast corresponding to a query
    fcst = get_forecast(query, path=forecast_path)
    logging.debug('Obtained the forecast')
    compressed_forecast = tinylib.to_beaufort(fcst)
    logging.debug("Compressed Size: %d" % len(compressed_forecast))
    return compressed_forecast


def process_query(query_string, reply_to, forecast_path=None, output=None):
    """
    Takes an un-parsed query string, parses it, fetches the forecast
    compresses it and replies to the sender.

    Parameters
    ----------
    query_string : string
        An un-parsed query.
    reply_to : string
        The email address of the sender.  The compressed forecast is sent
        to this person.
    forecast_path : string
        The path to an optional cached forecast.
    output : file-like
        A file like object to which the compressed forecast is written
    """
    compressed_forecast = query_to_beaufort(query_string, forecast_path)
    logging.debug("Compressed Size: %d" % len(compressed_forecast))
    # Make sure the forecast file isn't too large for sailmail
    if 'sailmail' in reply_to and len(compressed_forecast) > 25000:
        raise saildocs.BadQuery("Forecast was too large (%d bytes) for sailmail!"
                       % len(compressed_forecast))
    forecast_attachment = StringIO(compressed_forecast)
    if output:
        logging.debug("dumping file to output")
        output.write(forecast_attachment.getvalue())
    # creates the new mime email
    file_fmt = '%Y-%m-%d_%H%m.fcst'
    filename = datetime.datetime.today().strftime(file_fmt)
    filename = '_'.join([query['type'], filename])
    weather_email = emaillib.create_email(reply_to, _windbreaker_email,
                              _email_body,
                              subject=query_string,
                              attachments={filename: forecast_attachment})
    logging.debug('Sending email to %s' % reply_to)
    emaillib.send_email(weather_email)
    logging.debug('Email sent.')


def windbreaker(mime_text, ncdf_weather=None, output=None, fail_hard=False):
    """
    Takes a mime_text email that contains one or several saildoc-like
    requests and replies to the sender with emails containing the
    desired compressed forecasts.
    """
    exceptions = None if fail_hard else Exception
    logging.debug('Wind Breaker')
    logging.debug(mime_text)
    email_body = emaillib.get_body(mime_text)
    reply_to = emaillib.get_reply_to(mime_text)
    logging.debug('Extracted email body: %s' % str(email_body))
    if len(email_body) != 1:
        emaillib.send_error(reply_to,
                            "Your email should contain only one body")
    # Turns a query string into a dict of params
    logging.debug("About to parse saildocs")
    # The set makes sure there aren't duplicate queries.
    queries = set(list(saildocs.iterate_queries(email_body[0])))
    # if there are no queries let the sender know
    if len(queries) == 0:
        emaillib.send_error(reply_to,
            'We were unable to find any forecast requests in your email.')
    for query_string in queries:
        try:
            process_query(query_string, reply_to=reply_to,
                          forecast_path=ncdf_weather,
                          output=output)
        except saildocs.BadQuery, e:
            # It would be nice to be able to try processing all queries
            # in an email even if some of them failed, but a hard fail on
            # the first error will make sure we never accidentally send
            # tons of error emails to the user.
            logging.error(e)
            emaillib.send_error('akleeman@gmail.com',
                                ('Bad query: %s.' % query_string), e,
                                reply_to)
            emaillib.send_error(reply_to,
                                ("Error processing '%s'.  If there were other " +
                                 "queries in the same email they won't be " +
                                 "processed.\n") % query_string, e)
            if fail_hard:
                raise
        except exceptions, e:
            logging.error(e)
            emaillib.send_error('akleeman@gmail.com',
                                ('Query %s just failed.' % query_string), e)
            emaillib.send_error(reply_to,
                                ("Error processing %s.  Alex just got an urgent"
                                 " e-mail, he's looking into the problem. "
                                 "If there were other " +
                                 "queries in the same email they won't be " +
                                 "processed.\n") % query_string, e)
            if fail_hard:
                raise


def spot_message(spot, out=sys.stdout):
    """
    Dumps the readble spot message held in forecast 'spot'
    to the file-like output.
    """
    assert conv.TIME in spot
    assert conv.LAT in spot
    assert conv.LON in spot

    variables = [conv.UWND, conv.VWND]
    variables = [v for v in variables if v in spot]

    scale_to_knots = units._speed[spot['uwnd'][2][conv.UNITS]]
    uwnd = spot[conv.UWND][1].reshape(-1) * scale_to_knots
    vwnd = spot[conv.VWND][1].reshape(-1) * scale_to_knots
    winds = [objects.Wind(u, v) for u, v in zip(uwnd, vwnd)]

    time_units = spot[conv.TIME][2][conv.UNITS]
    assert time_units.startswith('hours')
    ref_time = time_units.split('since')[1].strip()
    ref_time = datetime.datetime.strptime(ref_time, '%Y-%m-%d %H:%M:%S')
    dates = [ref_time + datetime.timedelta(hours=x) for x in spot[conv.TIME][1]]
    date_strings = [x.strftime('%Y-%m-%d %H:%M UTC') for x in dates]

    fmt = '%20s\t%7s%5s\t%s'
    beaufort_in_knots = tinylib._beaufort_scale * scale_to_knots

    speeds = np.array([w.speed for w in winds])
    forces = np.digitize(speeds, beaufort_in_knots)

    if 'pressure' in spot:
        pressures = np.digitize(spot['pressure'][1].reshape(-1),
                                tinylib._pressure_scale)
    else:
        pressures = np.full_like(speeds, 0.)

    def iter_lines():
        yield '%20s\t%9s\t%9s' % ('Date', ' Wind (Knots)', 'MSL Press (Pa)')
        for d, w, f, p in zip(date_strings, winds, forces, pressures):
            speeds = '%d-%d' % (beaufort_in_knots[f - 1],
                                beaufort_in_knots[f])
            press = '%d-%d' % (tinylib._pressure_scale[p - 1],
                               tinylib._pressure_scale[p])
            yield fmt % (d, speeds, w.readable, press)

    ref_time = time_units.split(' since ')[1]

    out.write("The forecast used for this SPOT forecast was run on %s UTC\n" %
              ref_time)
    out.write('\n'.join(iter_lines()))
    return out
