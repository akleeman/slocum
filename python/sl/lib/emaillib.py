import os
import re
import zlib
import yaml
import numpy as np
import logging
import smtplib
import datetime

from email import Parser, mime, encoders
from email.mime import Multipart
from email.mime.text import MIMEText
from cStringIO import StringIO

from polyglot import Dataset

from sl import poseidon
from sl.lib import conventions as conv
from sl.lib import objects, tinylib

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_smtp_server = 'localhost'
_windbreaker_email = 'query@ensembleweather.com'
_email_body = """
This forecast brought to you by your friends on Saltbreaker.

Remember, always be skeptical of numerical forecasts (such as this).
For a full disclaimer visit www.ensembleweather.com."""


def args_from_email(email):
    body = get_body(email)
    assert len(body) == 1
    return body[0].rstrip().split(' ')


def get_reply_to(email):
    """
    Parses a mime email and returns the reply to address.
    If not reply to is explicitly specified the senders
    address is used.
    """
    parse = Parser.Parser()
    msg = parse.parsestr(email)
    if msg['Reply-To']:
        return msg['Reply-To']
    elif msg['From']:
        return msg['From']


def create_email(to, fr, body, subject=None, attachments=None):
    """
    Creates a multipart MIME email to 'to' and from 'fr'.  Both
    of which must be valid email addresses
    """
    msg = Multipart.MIMEMultipart()
    msg['Subject'] = subject or '(no subject)'
    msg['From'] = fr
    if isinstance(to, list):
        to = ','.join(to)
    msg['To'] = to
    body = MIMEText(body, 'plain')
    msg.attach(body)
    if attachments is not None:
        for attach_name, attach in attachments.iteritems():
            part = mime.base.MIMEBase('application', "octet-stream")
            part.set_payload(attach.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment; filename="%s"'
                            % attach_name)
            msg.attach(part)

    return msg


def send_email(mime_email):
    to = mime_email['To']
    fr = mime_email['From']
    s = smtplib.SMTP('localhost')
    server = smtplib.SMTP(_smtp_server)
    server.sendmail(fr, to, mime_email.as_string())
    s.quit()


def get_body(email):
    """
    Takes a MIME email and extract the (potentially multi part) body
    """
    parse = Parser.Parser()
    msg = parse.parsestr(email)

    def get_body(x):
        # get the text from messages of type 'text/plain'
        if x.get_content_maintype() == 'multipart':
            for y in x.get_payload():
                for z in get_body(y):
                    yield z
        elif x.get_content_type() == 'text/plain':
            yield x.get_payload().strip()

    return filter(len, get_body(msg))


def send_error(to, body, exception = None, fr = None):
    """
    Sends a simple email and logs at the same time
    """
    if not exception is None:
        body = '%s\n%s' % (body, str(exception))
    logger.debug(body)
    fr = fr or _windbreaker_email
    send_email(create_email(to, fr, body))


def parse_saildocs_hours(hours_str):
    # the hours query is a bit complex since it
    # involves interpreting the '..' as slices
    prev = 0
    for hr in hours_str.split(','):
        if '..' in hr:
            low, high = map(float, hr.split('..'))
            diff = low - prev
            if np.mod((high - low), diff):
                # the step size will not be integer valued
                # not sure how to handle that so for now just
                # yield the low and high value
                hours = [low, high]
            else:
                hours = np.linspace(low, high,
                                    num = ((high - low) / diff) + 1,
                                    endpoint=True)
        else:
            hours = map(float, [hr])
        for x in hours:
            prev = x
            yield x


def get_forecast(query, path=None):
    ll = objects.LatLon(query['lower'], query['left'])
    ur = objects.LatLon(query['upper'], query['right'])
    ur, ll = poseidon.ensure_corners(ur, ll, expand=False)
    vars = {}
    if 'wind' in query['vars']:
        vars['U-component_of_wind_height_above_ground'] = conv.UWND
        vars['V-component_of_wind_height_above_ground'] = conv.VWND
    else:
        raise ValueError("Currently only support wind forecasts")
#     if 'rain' in query['vars'] or 'precip' in query['vars']:
#         vars['Total_precipitation'] = conv.PRECIP
#     if 'cloud' in query['vars']:
#         vars['Total_cloud_cover'] = conv.CLOUD
#     if 'pressure' in query['vars']:
#         vars['Pressure'] = 'mslp'

    if path is not None and os.path.exists(path):
        fcst = Dataset(path)
    else:
        fcst = poseidon.gfs(ll, ur)
        if path is not None:
            fcst.dump(path)

    iter_hours = parse_saildocs_hours(query['hours'])

    assert fcst[conv.TIME].attributes[conv.UNITS].startswith('hour')

    def time_inds(hours):
        """determines which indices extract the required hours"""
        for hr in hours:
            inds = np.nonzero(fcst['time'].data[:] == float(hr))
            if len(inds) == 1 and inds[0].size == 1:
                # only yield an hour index if we know it exists
                yield inds[0][0]
    time_inds = list(time_inds(iter_hours))
    obj = fcst.take(time_inds, 'time')

    lats = fcst['latitude'].data[:]
    lat_resol = np.unique(np.diff(lats))
    if len(lat_resol) != 1:
        raise ValueError("Forecast has non-uniform latitudes")
    lat_resol = np.abs(lat_resol[0])
    lat_step = int(max(1, np.floor(query['grid_delta'] / lat_resol)))
    logger.debug("latitude - resol: %d, step: %d" % (lat_resol, lat_step))
    lat_inds = np.arange(lats.size)[::lat_step]

    lons = fcst['longitude'].data[:]
    lon_resol = np.unique(np.diff(lons))
    if len(lon_resol) != 1:
        raise ValueError("Forecast has non-uniform longitudes")
    lon_resol = np.abs(lon_resol[0])
    lon_step = int(max(1, np.floor(query['grid_delta'] / lon_resol)))
    logger.debug("latitude - resol: %d, step: %d" % (lat_resol, lat_step))
    lon_inds = np.arange(lons.size)[::lon_step]

    obj = obj.take(lat_inds, conv.LAT)
    obj = obj.take(lon_inds, conv.LON)
    if conv.ENSEMBLE in query:
        obj = obj.take(np.arange(query[conv.ENSEMBLE]), conv.ENSEMBLE)
    return obj


def windbreaker(mime_text, ncdf_weather=None,
                catchable_exceptions=None, output=None):
    """
    Takes a mime_text email that contains one or several saildoc-like
    requests and replies to the sender with emails containing the
    desired forecasts.
    """
    logger.debug('Wind Breaker')
    logger.debug(mime_text)
    email_body = get_body(mime_text)
    sender = get_reply_to(mime_text)
    logger.debug('Extracted email body: %s' % str(email_body))
    if len(email_body) != 1:
        send_error(sender, "Your email should contain only one body")
        return False
    # Turns a query string into a dict of params
    logger.debug("About to parse saildocs")
    logger.debug(str(email_body))
    try:
        queries = list(parse_saildocs(email_body[0]))
    except catchable_exceptions, e:
        send_error(sender,
                   'Failed to interpret your forecast requests', e)
        return False
    logger.debug("parsed the saildoc body")
    # if there are no queries let the sender know
    if len(queries) == 0:
        send_error(sender,
            'We were unable to find any forecast requests in your email.', e)
        return False
    success = True
    # for each query we send a seperate email
    for query in queries:
        logger.debug('Processing query %s', yaml.dump(query))
        # Aquires a forecast corresponding to a query
        fcst = get_forecast(query, path=ncdf_weather)
        logger.debug('Obtained the forecast:', str(fcst))
        # could use other extensions:
        # .grb, .grib  < 30kBytes
        # .bz2         < 5kBytes
        # .fcst        < 30kBytes
        # .gfcst       < 15kBytes
        try:
            tiny_fcst = tinylib.to_beaufort(fcst)
            compressed_forecast = zlib.compress(tiny_fcst, 9)
            logger.debug('Tinified the forecast')
        except catchable_exceptions, e:
            send_error(sender, "Failure compressing your email", e)
            success = False
            continue
        logger.debug("Compressed Size: %d" % len(compressed_forecast))
        # Make sure the forecast file isn't too large for sailmail
        if 'sailmail' in sender and len(compressed_forecast) > 25000:
            error_msg = ("Requested forecast was too large for sailmail! %d"
                         % len(compressed_forecast))
            send_error(sender, error_msg)
            success = False
            continue
        forecast_attachment = StringIO(compressed_forecast)
        if output:
            logger.debug("dumping file to output")
            output.write(forecast_attachment.getvalue())
        # creates the new mime email
        file_fmt = 'windbreaker_%Y-%m-%d_%H%m.fcst'
        filename = datetime.datetime.today().strftime(file_fmt)
        weather_email = create_email(sender, _windbreaker_email,
                              _email_body,
                              subject=email_body[0],
                              attachments={filename: forecast_attachment})
        logger.debug('Sending email to %s' % sender)
        send_email(weather_email)
        logger.debug('Email sent.')
    return success


def parse_saildocs(email_body):
    """
    Searches through an email body and yields individual saildocs queries
    """
    logger.debug("splitting lines:")
    lines = email_body.lower().split('\n')
    logger.debug('\n'.join(lines))
    matches = filter(lambda x: x,
                     [re.match('\s*(send\s.+)\s*', x) for x in lines])
    queries = [x.groups()[0] for x in matches]
    return (parse_saildocs_query(x) for x in queries)


def parse_saildocs_query(query):
    """
    Parses a saildocs string retrieving the forecast query params
    """
    query = query.strip()
    if len(query.split(' ', 2)) == 0:
        raise ValueError("expected at least one space")
    command, opts_args = query.split(' ', 1)
    opts_args = filter(len, opts_args.split(' '))
    if len(opts_args) > 1:
        options = opts_args[0]
        args = opts_args[1:]
    else:
        options = opts_args[0]
        args = None
    # subscribe and spot doesn't currently work
    if not command.lower() == 'send':
        raise ValueError("currently only supports the 'send' command")
    region, grid, hours_str, vars = options.split('|')
    vars = set([x.lower() for x in vars.split(',')])
    if not len(vars):
        vars = 'wind'
    # we only support the gfs model for now
    provider, corners = region.split(':')
    if not provider.lower() == 'gfs':
        raise ValueError("currently only supports the GFS model")
    # parse the corners of the forecast grid

    def floatify(latlon):
        """ Turns a latlon string into a float """
        sign = -2. * (latlon[-1].lower() in ['s', 'w']) + 1
        return float(latlon[:-1]) * sign
    upper, lower, left, right = map(floatify, corners.split(','))
    # determine the grid size
    grid_delta = set([np.abs(float(x)) for x in grid.split(',')])
    if len(grid_delta) > 1:
        raise ValueError("grid delta must be the same for lat/lon")
    grid_delta = grid_delta.pop()

    query_dict = {'upper': upper,
            'lower': lower,
            'left': left,
            'right': right,
            'grid_delta': grid_delta,
            'hours': hours_str,
            'start': None,
            'end': None,
            'vars': vars}

    if args:
        kwdargs = dict(x.lower().split('=') for x in args)
        if 'start' in kwdargs and 'end' in kwdargs:
            query_dict['start'] = objects.LatLon(*[floatify(x) for x in kwdargs['start'].split(',')])
            query_dict['end'] = objects.LatLon(*[floatify(x) for x in kwdargs['end'].split(',')])
        if 'ensembles' in kwdargs:
            query_dict['ensembles'] = int(kwdargs['ensembles'])

    return query_dict

if __name__ == "__main__":
    query = parse_saildocs_query('send GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND START=25,175')
    if not os.path.exists('test.nc'):
        fcst = get_forecast(query)
        fcst.dump('test.nc')
    else:
        fcst = Dataset('test.nc')
    tiny_fcst = tinylib.to_beaufort(fcst)
    compressed_forecast = zlib.compress(tiny_fcst, 9)
    with open('test.windbreaker', 'w') as f:
        f.write(compressed_forecast)

