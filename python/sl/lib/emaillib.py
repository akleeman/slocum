import os
import re
import sys
import zlib
import yaml
import numpy as np
import logging
import itertools

from email import Parser, mime, encoders
from email.mime import Multipart
from email.mime.text import MIMEText
from optparse import OptionParser
from cStringIO import StringIO

from sl import poseidon
from sl.lib import tinylib, datelib
from sl.objects import objects
import smtplib

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_windbreaker_email = 'wx@saltbreaker.com'

def clean_email_queue(queue_dir):
    """
    Removes any unprocessed emails if they are old
    """
    emails = [os.path.join(queue_dir, f) for f in os.listdir(queue_dir) if f.endswith('mime')]
    datelib.remove_old_files(emails, days_old=2)

def args_from_email(email):
    body = get_body(email)
    assert len(body) == 1
    return body[0].rstrip().split(' ')

def get_sender(email):
    parse = Parser.Parser()
    msg = parse.parsestr(email)
    if msg['Reply-To']:
        return msg['Reply-To']
    elif msg['From']:
        return msg['From']

def create_email(to, fr, body, subject=None, attachments=None):
    msg = Multipart.MIMEMultipart()
    msg['Subject'] = subject or '(no subject)'
    msg['From'] = fr
    if type(to) == type(list()):
        to = ','.join(to)
    msg['To'] = to
    body = MIMEText(body, 'plain')
    msg.attach(body)
    #msg.set_payload(body)
    if attachments:
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
    server = smtplib.SMTP('mail.saltbreaker.com', 26)
    server.login(_windbreaker_email, 'w3ath3r')
    server.sendmail(fr, to, mime_email.as_string())
    s.quit()

def mock_send_email(mime_email):
    to = mime_email['To']
    fr = mime_email['From']
    import pdb; pdb.set_trace()

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

def windbreaker(mime_text, ncdf_weather=None, catchable_exceptions=None, output=None):
    """
    Takes a mime_text email that contains one or several saildoc-like
    requests and replies to the sender with emails containing the
    desired forecasts.
    """
    logger.debug('Wind Breaker')
    email_body = get_body(mime_text)
    logger.debug('Extracted email body')
    if len(email_body) > 1:
        send_error(sender, "Your email contains more than one body")
        return False
    sender = get_sender(mime_text)
    # Turns a query string into a dict of params
    try:
        queries = list(parse_saildocs(email_body[0]))
    except catchable_exceptions, e:
        send_error(sender,
                   'Failed to interpret your forecast requests', e)
        return False
    # if there are no queries let the sender know
    if len(queries) == 0:
        send_error(sender,
                   'We were unable to find any forecast requests in your recent email.', e)
        return False
    success = True
    # for each query we send a seperate email
    for query in queries:
        logger.debug('Processing query %s', yaml.dump(query))
        # Aquires a forecast corresponding to a query
        try:
            obj = poseidon.email_forecast(query, path=ncdf_weather)
            logger.debug('Obtained the required forecast')
        except catchable_exceptions, e:
            send_error(sender,
                       'Failure getting your forecast from NOAA.', e)
            success = False
            continue
        # could use other extensions:
        # .grb, .grib  < 30kBytes
        # .bz2         < 5kBytes
        # .fcst        < 30kBytes
        # .gfcst       < 15kBytes
        try:
            tiny_fcst = tinylib.to_beaufort(obj, query['start'], query['end'])
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
        weather_email = create_email(sender, _windbreaker_email,
                                      'This forecast brought to you by your friends on Saltbreaker.',
                                      attachments={'windbreaker.fcst': forecast_attachment})
        logger.debug('Sending email to %s' % sender)
        send_email(weather_email)
        logger.debug('Email sent.')
    return success

def windbreaker_queue(queue_directory, ncdf_weather=None, catchable_exceptions=None, output=None):
        # remove any old forecasts so the data directory
    # doesn't get too large
    if not os.path.isdir(queue_directory):
        raise ValueError("%s is not a directory" % queue_directory)
    # remove old emails
    clean_email_queue(queue_directory)
    # clean out the forecast data directory
    poseidon.clean_data_dir()
    # create the processed/failed/processing directories
    processed_dir = os.path.join(queue_directory, 'processed')
    if not os.path.exists(processed_dir):
        os.mkdir(processed_dir)
    failed_dir = os.path.join(queue_directory, 'failed')
    if not os.path.exists(failed_dir):
        os.mkdir(failed_dir)
    processing_dir = os.path.join(queue_directory, 'processing')
    if not os.path.exists(processing_dir):
        os.mkdir(processing_dir)
    emails = [f for f in os.listdir(queue_directory) if f.endswith('.mime')]
    logger.debug("processing %d emails from %s" % (len(emails), queue_directory))
    for fn in emails:
        full_path = os.path.join(processing_dir, fn)
        # move the current file to the processing directory
        # that way two email_queue processes can't be working
        # on the same requests
        os.rename(os.path.join(queue_directory, fn), full_path)
        logger.debug("processing %s" % full_path)
        try:
            f = open(full_path, 'r')
            success = windbreaker(f.read(), ncdf_weather, catchable_exceptions, output)
            f.close()
        except catchable_exceptions, e:
            logger.debug(e)
            success = False
        # if processing the email failed in any way we move it to
        # the failed directory otherwise move to processed
        if not success:
            logger.debug("moving %s to failed" % full_path)
            os.rename(full_path, os.path.join(failed_dir, fn))
        else:
            logger.debug("moving %s to processed" % full_path)
            os.rename(full_path, os.path.join(processed_dir, fn))
    logger.debug("queue processing complete.")

def parse_saildocs(email_body):
    """
    Searches through an email body and yields individual saildocs queries
    """
    lines = email_body.lower().split('\n')
    matches = filter(lambda x : x,
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
    region, grid, hours_str, vars  = options.split('|')
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
            'start':None,
            'end':None,
            'vars':vars}

    if args:
        kwdargs = dict(x.lower().split('=') for x in args)
        if 'start' in kwdargs and 'end' in kwdargs:
            query_dict['start'] = objects.LatLon(*[floatify(x) for x in kwdargs['start'].split(',')])
            query_dict['end'] = objects.LatLon(*[floatify(x) for x in kwdargs['end'].split(',')])
        if 'ensembles' in kwdargs:
            query_dict['ensembles'] = int(kwdargs['ensembles'])

    return query_dict

def test_parse_saildocs():

    print list(parse_saildocs_hours('0,6,12,24..60,66..90,102,120'))
    import pdb; pdb.set_trace()

    print parse_saildocs_query('send GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND START=25,175')
    print parse_saildocs_query('send GFS:10S,42S,162E,144W|13,13|0,6,12,24..60,66..90,102,120|PRMSL,WIND,WAVES,RAIN')

if __name__ == "__main__":
    test_parse_saildocs()


#
#
#fimg = open('/home/kleeman/Desktop/test.jpg', 'rb')
#img = fimg.read()
#img_str = base64.b64encode(zlib.compress(img,9))
#
#def contents(x):
#    if x.get_content_maintype() == 'multipart':
#        return [contents(y) for y in x.get_payload()]
#    else:
#        if x.get_filename():
#            return x.get_payload(decode=True)
#        else:
#            return x.get_payload()
#
##EXAMPLE OF SENDING PICTURES IN AN EMAIL
## Import smtplib for the actual sending function
#import smtplib
#
## Here are the email package modules we'll need
#from email.mime.image import MIMEImage
#from email.mime.multipart import MIMEMultipart
#
#COMMASPACE = ', '
#
## Create the container (outer) email message.
#msg = MIMEMultipart()
#msg['Subject'] = 'Our family reunion'
## me == the sender's email address
## family = the list of all recipients' email addresses
#msg['From'] = me
#msg['To'] = COMMASPACE.join(family)
#msg.preamble = 'Our family reunion'
#
## Assume we know that the image files are all in PNG format
#for file in pngfiles:
#    # Open the files in binary mode.  Let the MIMEImage class automatically
#    # guess the specific image type.
#    fp = open(file, 'rb')
#    img = MIMEImage(fp.read())
#    fp.close()
#    msg.attach(img)
#
## Send the email via our own SMTP server.
#s = smtplib.SMTP()
#s.sendmail(me, family, msg.as_string())
#s.quit()
