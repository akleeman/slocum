import os
import logging
import smtplib

from email import Parser, mime, encoders
from email.mime import Multipart
from email.mime.text import MIMEText

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

_smtp_server = 'localhost'
_no_reply = 'noreply@ensembleweather.com'


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
    """
    Uses SMTP to actually send a MIME email.  The sender
    and recipient are parsed from the MIME object.
    """
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


def send_error(to, body, exception=None, fr=None):
    """
    Sends a simple email and logs at the same time
    """
    if not exception is None:
        body = '%s\n%s' % (body, str(exception))
    logger.debug(body)
    fr = fr or _no_reply
    send_email(create_email(to, fr, body))
