import json
import logging
import datetime
import tempfile
import retrying
import warnings

from io import BytesIO

from slocum.lib import emaillib
from slocum.compression import compress
from slocum.visualize import visualize

import utils
import subset
import zygrib
import saildocs

_query_email = 'query@ensembleweather.com'

_email_body = """
This forecast was brought to you by your friends on Saltbreaker.

Remember, always be skeptical of numerical forecasts (such as this).
For a full disclaimer visit www.ensembleweather.com."""


def iterate_query_strings(query_text):
    """
    Takes a  block of text, likely from an email, and iterates over
    queries in the text.
    """
    return saildocs.iterate_query_strings(query_text)


def parse_query_string(query_string):
    """
    Takeas a query string and parses it into a dictionary.
    """
    return saildocs.parse_saildocs_query(query_string)


def process_query(query, url=None):
    """
    Takes a query and returns the corresponding tiny forecast.
    """
    # log the query so debugging others request failures will be easier.
    logging.debug(json.dumps(query))
    # Acquires a forecast corresponding to a query
    fcst = get_forecast(query, url=url)
    compressed_forecast = compress.compress_dataset(fcst)
    logging.debug("Compressed Size: %d" % len(compressed_forecast))
    return compressed_forecast


# retry on DAP errors.
def is_recoverable_dap_error(e):
    return isinstance(e, RuntimeError) and "DAP" in e.msg


@retrying.retry(retry_on_exception=is_recoverable_dap_error,
                wait_random_min=100, wait_random_max=2000,
                stop_max_attempt_number=3)
def get_forecast(query, url=None):
    """
    Fetches the forecast from the model provider and returns
    the forecast subset to the query domain.
    """
    if url is not None:
        warnings.warn('Forecast was extracted from %s'
                      ' which may be out of date.'
                      % url)
    model = utils.get_model(query['model'])
    fcst = model.fetch(url=url)
    sub_fcst = subset.subset_dataset(fcst, query)
    return sub_fcst


def query_summary(query):
    """
    Takes a query and returns a summary
    """
    if query['type'] == 'spot':
        loc_string = '%(latitude)6.2fN,%(longitude)6.2fE' % query['location']
    elif query['type'] == 'gridded':
        loc_string = '%(S)6.2fN,%(N)6.2fN,%(E)6.2fE%(W)6.2f' % query['domain']

    time_string = '%d-%d Hours' % (min(query['hours']),
                                   max(query['hours']))
    summary = ' '.join([query['type'], query['model'], loc_string, time_string])
    return summary


def decide_extension(reply_to):
    """
    Some email services have strict rules on which extensions they can
    recieve.  For sailmail users (for example) the .fcst extension is
    the best choice, while for iridium users .fcst won't make it through
    their firewall, in which case something like .zip is better.
    """
    if 'iridium' in reply_to:
        return 'zip'
    else:
        return 'fcst'


def _response_to_query(query, reply_to, subject=None, url=None):
    """
    Takes a parsed query string fetches the forecast,
    compresses it and replies to the sender.

    Parameters
    ----------
    query : dict
        An dictionary query
    reply_to : string
        The email address of the sender.  The compressed forecast is sent
        to this person.
    subject : string (optional)
        The subject of the email that is sent. If none, a summary of
        the query is used.
    url : string (optional)
        A path to the forecast that should be used.  This is mostly used
        for debugging and testing.

    Returns
    ----------
    forecast_attachment : file-like
        A file-like object holding the forecast that was sent
    """
    compressed_forecast = process_query(query, url=url)
    logging.debug("Compressed Size: %d" % len(compressed_forecast))
    # create a file-like forecast attachment
    logging.debug('Obtained the forecast')

    file_fmt = '%Y-%m-%d_%H%m'
    filename = datetime.datetime.today().strftime(file_fmt)
    filename = '_'.join([query['type'], filename])
    if query.get('send-image', False):
        logging.debug('Sending an image of the forecast')
        fcst = compress.decompress_dataset(compressed_forecast)
        visualize.plot_spot(fcst)
        import matplotlib.pyplot as plt
        png = BytesIO()
        plt.savefig(png)
        png.seek(0)
        attachments = {'%s.png' % filename: png}
    else:
        logging.debug('Sending the compressed forecasts')
        forecast_attachment = BytesIO(compressed_forecast)
        ext = decide_extension(reply_to)
        attachment_name = '%s.%s' % (filename, ext)
        logging.debug("Attaching forecast as %s" % attachment_name)
        attachments = {attachment_name: forecast_attachment}
    # Make sure the forecast file isn't too large for sailmail
    if 'sailmail' in reply_to and len(compressed_forecast) > 25000:
        raise utils.BadQuery("Forecast was too large (%d bytes) for sailmail!"
                             % len(compressed_forecast))
    # creates the new mime email
    weather_email = emaillib.create_email(reply_to, _query_email,
                              _email_body,
                              subject=subject or query_summary(query),
                              attachments=attachments)
    return weather_email


def add_warnings_to_body(email, warns):
    if not len(warns):
        return
    body = [x for x in email.get_payload()
            if x.get_content_maintype() == 'text']
    assert len(body) == 1
    body = body[0]
    body_text = body.get_payload()

    warning_text = ("Warning, some issues were encountered while"
                    " fetching your forecast:\n\t-%s"
                    % '\n\t-'.join([str(w.message) for w in warns]))
    body.set_payload('%s\n\n%s' % (body_text, warning_text))


def process_single_query(query_string, reply_to, url=None):
    """
    Parses a query and creates a response which contains
    warnings that were encountered during processing.
    """
    with warnings.catch_warnings(record=True) as warns:
        query = parse_query_string(query_string)
        email = _response_to_query(query, reply_to=reply_to, url=url)
    # this adds a section with warnings in place.
    add_warnings_to_body(email, warns)
    return email


def process_email(mime_text, url=None,
                  fail_hard=False, log_input=True):
    """
    Takes a mime_text email that contains one or several
    requests and replies to the sender with emails containing the
    desired compressed forecasts.
    """
    exceptions = None if fail_hard else Exception
    if log_input:
        # here we store the input to a temp file so if it
        # fails its easier to repeat the error.
        _, tf = tempfile.mkstemp('query_email')
        with open(tf, 'w') as f:
            f.write(mime_text)
        logging.debug("cached input to %s" % tf)
    # pull information from the mime email text
    email_body = emaillib.get_body(mime_text)
    reply_to = emaillib.get_reply_to(mime_text)
    logging.debug('Extracted email body: %s' % str(email_body))
    if len(email_body) != 1:
        emaillib.send_error(reply_to,
                            "Your email should contain only one body")
    # Here we try parsing all the queries up front.  We'll parse
    # them again one by one later
    query_strings = list(iterate_query_strings(email_body[0]))
    n_queries = len(query_strings)
    # if there are no queries let the sender know
    if n_queries == 0:
        emaillib.send_error(reply_to,
            'We were unable to find any forecast requests in your email.')
    for query_string in query_strings:
        try:
            email = process_single_query(query_string, reply_to, url=url)
            emaillib.send_email(email)
        except utils.BadQuery, e:
            # These exceptions were the user's fault.
            logging.error(e)
            # It would be nice to be able to try processing all queries
            # in an email even if some of them failed, but a hard fail on
            # the first error will make sure we never accidentally send
            # tons of error emails to the user.
            emaillib.send_error(reply_to,
                                ("Bad query: '%s'.  If there were other "
                                 "queries in the same email they won't be "
                                 "processed.\n Query failed with error %s"
                                 % email_body[0], e))
            if fail_hard:
                raise
        except exceptions, e:
            # These exceptions are unexpected errors on our end.
            logging.error(e)
            emaillib.send_error(reply_to,
                                ("Error processing %s your e-mail forecast "
                                 "the admin has been notified and will look into "
                                 "the problem. If there were other "
                                 "queries in the same email they won't be "
                                 "processed.\n") % email_body[0], e)
            if fail_hard:
                raise
