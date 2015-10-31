import os
import re
import logging
import numpy as np

import itertools

import utils

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
_usage = """ The saildocs-like query format is:

    send model:lat0,lat1,lon0,lon1|[lat_delta,lon_delta]|[hours]|[variables]

arguments in brackets are optional.

For a better explanation send an email to info@ensembleweather.com
"""

_supported_commands = ['send']

_spot_usage = """
send spot:model:location|[days,interval]|[variables]

Parameters in brackets are optional, but the requests are order dependent
so, for example, these structures are also valid:

    ``send model:location``

    ``send model:location||variables``

model
    One of %(models)s

    The forecast model used, Defaults to ``gefs``.

location
    Latitude,Longitude

    Latitudes must include either an 'S' or 'N' suffix (ie, 10S or -10N)
    and Longitudes must include either an 'E' or 'W' suffix (ie 190E or 170W).
    The resulting forecast is for the nearest modeled grid point.  Slocum
    doesn't NOT interpolate between points.  Such interpolation tends to
    reduce extreme values in forecasts.

days,interval
    Two integers.  The number of days followed by the hourly interval.

    The first integer is the duration of the forecast in days, the second
    is the number of hours between forecast points.  For example, to get
    a forecast for the next four days every three days you would use
    ``4,3``.

variables
    A list variables. Available variables include: %(variables)s

    Default is 'WIND'

    The list of variables to be included in the forecast.  Each
    variable should be separated by a comma.
""" % {'models': ', '.join(['``%s``' % x for x in utils._models.keys()]),
       'variables': ', '.join(['``%s``' % x for x in utils._variables.keys()])}


_gridded_usage = """
    ``send model:region|[resolution]|[hours]|[variables]``

Parameters in brackets are optional, but the requests are order dependent
so, for example, these structures are also valid:

    ``send model:region``

    ``send model:region||hours``

    ``send model:region|resolution``

    ``send model:region|||variables``

model
    One of %(models)s

    The forecast model used, Defaults to ``gfs``.

region
    Bounding box given by ``lat0,lat1,lon0,lon1``.

    The region is required and is specified by giving the corners
    of the region in degrees of latitude
    and longitude that surrounds the desired forecast area.
    Latitudes must include either
    an 'S' or 'N' suffix (ie, 10S or -10N) and Longitudes must include
    either an 'E' or 'W' suffix (ie 190E or 170W). For example,
    '35S,45S,166E,179E' would cover most of New Zealand. The bounding box
    cannot span more than 180 degrees of longitude.

resolution

    Number of degrees or the word 'native'.

    Specifies the width of each grid in the desired forecast.  Default is
    'native'.

    For example, GFS has a native resolution of 0.5 degrees.
    By default this native resolution is used, but if you may want a
    more coarse grid if you are requesting a forecast for a larger area.
    Resolution values will be rounded to the nearest multiple of the
    native grid.

hours

    List of integer hours.

    Indicates which hours of the forecast are desired. Defaults to '24,48,72'.
    Hours should be comma separated and ellipses can be used to avoid
    unnecessarily long queries. When ellipsis are used the difference
    between the previous two values is continued until the end time.
    For example, '0,6,12,24,36,48,60,72' is equivalent to '0,6,12,24...72'.

variables

    A list variables. Available variables include: %(variables)s

    Default is 'WIND'

    The list of variables to be included in the forecast.  Each
    variable should be separated by a comma.
""" % {'models': ', '.join(['``%s``' % x for x in utils._models.keys()]),
       'variables': ', '.join(['``%s``' % x for x in utils._variables.keys()])}


def iterate_query_strings(query_text):
    """
    Searches through an email body and yields individual query strings

    Paramters
    ---------
    query_text : string
        Text containing queries, usually this is the body of an email that
        presumably contains some queries.

    Returns
    -------
    queries : generator
        A generator which yields individual query strings.
    """
    lines = query_text.lower().split('\n')
    for command in _supported_commands:
        matches = [re.match('\s*(%s\s.+)\s*' % command, x) for x in lines]
        for query in filter(None, matches):
            yield query.groups()[0]


def split_fields(request, k):
    """
    Saildoc fields can be separated by '|', '/' or '\u015a'
    """
    fields = re.split(u'\s*[\|\/\u015a]\s*', unicode(request.strip()))
    return list(itertools.chain(fields, [None] * k))[:k]


def parse_gridded_request(request):
    """
    Parses a request for a gridded forecast.
    """
    # takes the first 4 '|' separated fields, if fewer than
    # k exist the missing fields are replaced with None
    model_domain, grid_str, hours_str, variables = split_fields(request, 4)
    model, domain_str = model_domain.split(':')
    # parse the domain and make sure its ok
    domain = utils.parse_domain(domain_str)
    # parse the grid_string
    resol = utils.parse_resolution(grid_str)
    # parse the hours string
    hours = utils.parse_hours(hours_str)
    # check the variables
    if variables is None:
        variables = []
    else:
        variables = variables.split(',')
    variables = utils.validate_variables(variables)
    return {'type': 'gridded',
            'model': model.lower().strip(),
            'domain': domain,
            'resolution': resol,
            'hours': hours,
            'variables': variables}


def parse_spot_request(request):
    """
    parses a request for a spot forecast
    """
    model_domain, time_str, variables, image = split_fields(request, 4)
    spot, location_str = model_domain.split(':', 1)
    assert spot.lower() == 'spot'
    if ':' in location_str:
        model, location_str = location_str.split(':', 1)
        model = model.lower()
        import ipdb; ipdb.set_trace()
    else:
        model = 'gefs'
    location = utils.parse_location(location_str)
    # default to 4 days every three hours
    time_str = time_str or '5,6'
    hours = utils.parse_times(time_str)

    if variables is None:
        variables = []
    else:
        variables = variables.split(',')
    variables = utils.validate_variables(variables)

    send_image = image is not None

    return {'type': 'spot',
            'model': model,
            'location': location,
            'hours': hours,
            'variables': variables,
            'send-image': send_image}


def parse_send_request(body):
    """
    Parses the a saildoc-like send request and returns
    a dictionary of attributes from the query.
    """
    # the model and domain are colon separated.
    model_domain, = split_fields(body, 1)
    model, _ = model_domain.split(':', 1)
    # make sure the model exists
    model = utils.validate_model(model)
    if model == 'spot':
        return parse_spot_request(body)
    else:
        return parse_gridded_request(body)


def parse_saildocs_query(query_str):
    """
    Parses a saildocs string retrieving the forecast query dict.

    queries should look like:

        command model:lat0,lat1,lon0,lon1|lat_delta,lon_delta|times|variables

    For example:

        send GFS:14S,20S,154W,146W|0.5,0.5|0,3..120|WIND

    Means send a wind forecast from 14S to 20S, 154W to 146W at 0.5
    degree spacing for the next 120 hours every 3 hours.
    """
    # remove any trailing white space
    query_str = query_str.strip()
    command_split = query_str.split(' ', 1)
    if len(command_split) != 2:
        raise utils.BadQuery("Expected a space between the "
                             "command and the body")
    command, body = command_split
    # saildocs subscription requests can have additional options
    # such as time= and days=.  These are parsed here.
    opts_args = filter(len, body.split(' '))
    options_start = np.nonzero(['=' in x for x in opts_args])[0]
    if len(options_start):
        args = opts_args[:options_start[0]]
        opts = opts_args[options_start[0]:]
        warnings.warn("Additional options %s were ignored"
                      % ' '.join(opts))
    else:
        args = ''.join(opts_args)
        opts = None
    # Check if the command is supported
    if not command.lower() in _supported_commands:
        raise utils.BadQuery("Unsupported command %s, only %s are supported"
                             % (command.lower(), ','.join(_supported_commands)))

    if command.lower() == 'send':
        query = parse_send_request(args)
    elif command.lower() in ['sub', 'cancel']:
        raise utils.BadQuery("slocum doesn't support subscriptions, "
                             "use send instead of sub.")
    else:
        raise utils.BadQuery("Unknown command handler.  %s"
                             % command)
    return query
