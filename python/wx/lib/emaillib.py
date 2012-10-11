import re
import sys
import numpy as np
import itertools

from email import Parser, mime
from email.mime import Multipart
from optparse import OptionParser

from wx.objects import objects
import smtplib

def args_from_email(email):
    body = get_body(email)
    assert len(body) == 1
    return body[0].rstrip().split(' ')

def get_sender(email):
    return 'akleeman@gmail.com'

def create_email(to, fr, body, subject=None, attach=None):
    msg = Multipart.MIMEMultipart()
    msg['Subject'] = subject or '(no subject)'
    msg['From'] = fr
    if type(to) == type(list()):
        to = ','.join(to)
    msg['To'] = to
    msg.preamble = body

    part = mime.base.MIMEBase('application', "octet-stream")
    part.set_payload(open(attach,"rb").read())
    part.add_header('Content-Disposition', 'attachment; filename="%s"' % attach)
    msg.attach(part)   # msg is an instance of MIMEMultipart()
    return msg

def send_email(mime_email):
    to = mime_email['To']
    fr = mime_email['From']
    s = smtplib.SMTP('localhost')
    server = smtplib.SMTP('mail.saltbreaker.com', 26)
    server.login('wx@saltbreaker.com', 'w3ath3r')
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

def parse_saildocs(query_body):
    """
    Parses a saildocs string retrieving the forecast query params
    """
    query_body = query_body.strip()
    if not len(query_body.split(' ', 2)) == 2:
        raise ValueError("expected a single space")
    command, options = query_body.split(' ', 1)
    # subscribe and spot doesn't currently work
    if not command.lower() == 'send':
        raise ValueError("currently only supports the 'send' command")
    region, grid, hours_str, vars  = options.split('|')
    # we only support the gfs model for now
    provider, corners = region.split(':')
    if not provider.lower() == 'gfs':
        raise ValueError("currently only supports the GFS model")
    # parse the corners of the forecast grid
    def floatify(latlon):
        """ Turns a latlon string into a float """
        sign = -2. * (latlon[-1] in ['S', 'W']) + 1
        return float(latlon[:-1]) * sign
    upper, lower, left, right = map(floatify, corners.split(','))
    # determine the grid size
    grid_delta = set(map(float, grid.split(',')))
    if len(grid_delta) > 1:
        raise ValueError("grid delta must be the same for lat/lon")
    grid_delta = grid_delta.pop()
    # Now we determine the desired forecast times
    def hours_gen(hours_str):
        """Parses a saildocs string of desired hours"""
        discrete_hours = map(float, filter(len, re.split(',|\.', hours_str)))
        step = np.min(np.diff(discrete_hours))
        for hr in hours_str.split(','):
            if '..' in hr:
                low, high = map(float, hr.split('..'))
                hours = np.linspace(low, high,
                                    num=(high - low) / step + 1,
                                    endpoint=True)
                for x in hours:
                    yield x
            else:
                yield float(hr)

    return {'upper': upper,
            'lower': lower,
            'left': left,
            'right': right,
            'grid_delta': grid_delta,
            'hours': list(hours_gen(hours_str)),}

def test_parse_saildocs():

    print parse_saildocs('send GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|PRMSL,WIND,RAIN')
    print parse_saildocs('send GFS:10S,42S,162E,144W|13,13|0,6,12,24..60,66..90,102,120|PRMSL,WIND,WAVES,RAIN')

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
