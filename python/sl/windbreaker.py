import os
import zlib
import numpy as np
import logging
import datetime

from cStringIO import StringIO

from xray import open_dataset, backends

from sl import poseidon
from sl.lib import conventions as conv
from sl.lib import objects, tinylib, saildocs, emaillib
from sl.lib.objects import NautAngle

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

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


def regular_grid(xmin, xmax, delta):
    """A convenient function around linspace"""
    # Note to Alex: The formula below has trouble with float values. For
    # example,
    #
    #   regular_grid(-9.9999999, -7., -0.5)
    #
    # results in
    #
    #   array([-9.99999999, -9.39999999, -8.79999999, -8.2       ,
    #          -7.6       , -7.        ])
    # and
    #
    #   regular_grid(0, 3.999999, 2)
    #
    # results in
    #
    #   array([ 0.      ,  3.999999])
    #
    # Have changed the calls to this function to use np.arange instead.
    #
    return np.linspace(xmin, xmax, (xmax - xmin) / delta + 1)


def arg_closest(x, reference):
    """
    Find the closest element index in reference for each
    element of x.  This isn't the most efficient but
    allows nearest neighbors with arbitrary reference
    data.
    """
    return np.array([np.argmin(np.abs(reference - y)) for y in x])


def get_forecast(query, path=None):
    warnings = []
    variables = {}
    if len(set(['wind']).intersection(query['vars'])):
        variables['u-component_of_wind_height_above_ground'] = conv.UWND
        variables['v-component_of_wind_height_above_ground'] = conv.VWND
    if len(set(['rain', 'precip']).intersection(query['vars'])):
        variables['Precipitation_rate_surface_Mixed_intervals_Average'] = conv.PRECIP
    if len(set(['press', 'pressure', 'mslp']).intersection(query['vars'])):
        variables['Pressure_reduced_to_MSL_msl'] = conv.PRESSURE

    north, south, east, west = [NautAngle(query['domain'][d])
                                for d in "NSEW"]

    # Here we do some crude caching which
    # allows the user to specify a path to a local
    # file that holds the data instead of going through
    # poseidon.
    if path and os.path.exists(path):
        fcst = open_dataset(path)
        warnings.append('Using cached forecasts (%s) which may be old.' % path)
    else:
        fcst = poseidon.gfs(north, south, east, west, variables=variables)
        if path is not None:
            fcst.dump(path)

    # extract all the closest latitudes
    step = query['grid_delta'][0]
    # adding step/2. to upper bound to avoid unintended truncations due to
    # floating point issues; carefull not to extend range beyond 180 though...
    scheibe_mehr = min(step / 2., (180. - (north - south)) / 2.)
    lats = np.arange(south, north + scheibe_mehr, step)
    lat_inds = arg_closest(lats, fcst['latitude'].data.values)
    # Notes to Alex:
    # [1] What if the query string has 'odd' fractional lat/lons
    # that are offset against the GFS grid (e.g. "send GFS:32.3S,36.7S, ...",
    # selected in the Airmail GUI selector)? -> changed threshold to step/2
    # [2] Why not return fcst wih any lats that overlapped and discard the
    # rest - ie. 'not np.any(...) <= threshold' raises the exceptions rather
    # than 'np.any(...) > thershold"? -> changed if clause
    #
    # if np.any((fcst['latitude'].data[lat_inds] - lats) > 0.05):
    if not np.any(abs(fcst['latitude'].data[lat_inds] - lats) <= step / 2.):
        raise saildocs.BadQuery("Requested latitudes not found in the forecast.")
    fcst = fcst.indexed_by(latitude=lat_inds)
    # extract all the closest longitudes
    # lon_range = np.mod(query['domain']['E'] - query['domain']['W'], 360)
    # lon_count = lon_range / query['grid_delta'][1] + 1
    # lons = query['domain']['W'] + np.arange(lon_count) * query['grid_delta'][1]
    # lons = np.mod(lons + 180., 360.) - 180.
    step = query['grid_delta'][1]
    scheibe_mehr = min(step / 2., (180. - (east - west)) / 2.)
    lons = [NautAngle(lon)
            for lon in np.arange(west, east + scheibe_mehr, step)]
    # fcst_lons = np.mod(fcst['longitude'].data.values + 180., 360.) - 180.
    fcst_lons = [NautAngle(lon)
                 for lon in fcst['longitude'].data.values]
    lon_inds = arg_closest(np.array(lons), np.array(fcst_lons))

    if not np.any(abs(fcst_lons[lon_inds] - lons) <= step / 2.):
        raise saildocs.BadQuery("Requested longitudes not found in the forecast.")
    fcst = fcst.indexed_by(longitude=lon_inds)
    # next step is parsing out the times
    # we assume that the forecast units are in hours
    # TODO: simplify - xray time dimension contains a datetime subclass
    dates = fcst['time'].data.to_pydatetime()
    ref_time = dates[0]
    query_times = np.array([ref_time + datetime.timedelta(hours=x)
                            for x in query['hours']])
    time_inds = arg_closest(query_times, dates)
    fcst = fcst.indexed_by(time=time_inds)
    if np.any((dates[time_inds] - query_times) >= datetime.timedelta(hours=1)):
        raise saildocs.BadQuery("Requested times not found in the forecast.")
    return fcst


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
    logger.debug(query_string)
    query = saildocs.parse_saildocs_query(query_string)
    # log the query so debugging others request failures will be easier.
    logger.debug("model: %s" % query['model'])
    domain_str = ','.join(':'.join([k, str(v)])
                          for k, v in query['domain'].iteritems())
    logger.debug("domain: %s" % domain_str)
    logger.debug("grid_delta: %s" % ','.join(map(str, query['grid_delta'])))
    logger.debug("hours: %s" % ','.join(map(str, query['hours'])))
    logger.debug("warnings: %s" % '\n'.join(query['warnings']))
    logger.debug("variables: %s" % ','.join(query['vars']))
    # Acquires a forecast corresponding to a query
    fcst = get_forecast(query, path=forecast_path)
    logger.debug('Obtained the forecast')
    tiny_fcst = tinylib.to_beaufort(fcst)
    compressed_forecast = zlib.compress(tiny_fcst, 9)
    logger.debug("Compressed Size: %d" % len(compressed_forecast))
    # Make sure the forecast file isn't too large for sailmail
    if 'sailmail' in reply_to and len(compressed_forecast) > 25000:
        raise saildocs.BadQuery("Forecast was too large (%d bytes) for sailmail!"
                       % len(compressed_forecast))
    forecast_attachment = StringIO(compressed_forecast)
    if output:
        logger.debug("dumping file to output")
        output.write(forecast_attachment.getvalue())
    # creates the new mime email
    file_fmt = 'windbreaker_%Y-%m-%d_%H%m.fcst'
    filename = datetime.datetime.today().strftime(file_fmt)
    weather_email = emaillib.create_email(reply_to, _windbreaker_email,
                              _email_body,
                              subject=query_string,
                              attachments={filename: forecast_attachment})
    logger.debug('Sending email to %s' % reply_to)
    emaillib.send_email(weather_email)
    logger.debug('Email sent.')


def windbreaker(mime_text, ncdf_weather=None, output=None):
    """
    Takes a mime_text email that contains one or several saildoc-like
    requests and replies to the sender with emails containing the
    desired compressed forecasts.
    """
    logger.debug('Wind Breaker')
    logger.debug(mime_text)
    email_body = emaillib.get_body(mime_text)
    reply_to = emaillib.get_reply_to(mime_text)
    logger.debug('Extracted email body: %s' % str(email_body))
    if len(email_body) != 1:
        emaillib.send_error(reply_to,
                            "Your email should contain only one body")
    # Turns a query string into a dict of params
    logger.debug("About to parse saildocs")
    queries = list(saildocs.iterate_queries(email_body[0]))
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
            emaillib.send_error(reply_to,
                                ("Error processing %s.  If there were other " +
                                 "queries in the same email they won't be " +
                                 "processed.\n") % query_string, e)
            raise
