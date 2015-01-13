import os
import re
import numpy as np
import logging

import itertools

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
_usage = """ The saildocs-like query format is:

send model:lat0,lat1,lon0,lon1|[lat_delta,lon_delta]|[hours]|[variables]

options in brackets are optional.

For a better explanation send an email to info@ensembleweather.com
"""

_supported_commands = ['send']
_supported_variables = ['wind', 'rain', 'press']
_supported_models = ['gfs', 'gefs']

_send_usage = """
SEND requests use the format:

send model:lat0,lat1,lon0,lon1|[lat_delta,lon_delta]|[hours]|[variables]

Parameters in brackets are optional, but the requests are order dependent
so, for example, the hours parameter must follow the second pipe character (|).

model : The forecast model used (%(models)s)

lat0,lat1,lon0,lon1 : These should be numbers specifying the latitudes
    and longitudes (in degrees) of the bounding box that surrounds the
    area where a forecast is desired.  Latitudes must include either
    an 'S' or 'N' suffix (ie, 10S or -10N) and Longitudes must include
    either an 'E' or 'W' suffix (ie 190E or 170W).  The bounding box
    cannot span more than 180 degrees of longitude.  For example,
    '35S,45S,166E,179E' would cover most of New Zealand.

lat_delta,lon_delta : These specify the coarseness of the resulting
    forecast grid in degrees.  GFS forecast's highest resolution is
    0.5 degrees, so the deltas should be multiples of 0.5.  If no
    grid deltas are specified the default of 2 degrees is used.

hours : Indicates which hours of the forecast are desired.  Hours
    should be comma separated and ellipses can be used to avoid
    unnecessarily long queries.  The default is '24,48,72'.  When
    ellipsis are used the difference between the previous two values
    is continued until the end time.  For example, '0,6,12,24,36,48,60,72'
    is equivalent to '0,6,12,24...72'.

variables : The list of variables to be included in the forecast.  Each
    variable should be separated by a comma.  Current choices for variables
    are: %(variables)s
""" % {'models': _supported_models,
       'variables': _supported_variables}


class BadQuery(BaseException):
    """
    An exception that allows us to catch bad queries and respond
    to the user with helpful information.
    """
    def __init__(self, message=''):
        super(self.__class__, self).__init__('\n'.join([message, _usage]))


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


def validate_variables(variables):
    """
    Makes sure the requested variables are supported and
    reverts to defaults.  Any unsupported variables are filtered
    out and associated warnings are returned.

    Parameters
    ----------
    variables : list of strings
        A list of requested variables.  If empty the default
        of 'WIND' is used.

    Returns
    -------
    variables : list of strings
        A list of the supported variables
    warnings : list of strings
        Any warnings
    """
    warnings = []
    variables = set(filter(len, [x.strip().lower() for x in variables]))
    if not len(variables):
        variables = set(['wind'])
        warnings.append('No variable requested, defaulting to WIND')
    if len(variables.difference(_supported_variables)):
        for x in variables.difference(_supported_variables):
            warnings.append('Variable "%s" is unsupported' % x)
    # The user requested a variable but its not supported, for
    # this we do a hard failure instead of defaulting to wind.
    if not len(variables.intersection(_supported_variables)):
        raise BadQuery("Unsupported variable(s) %s"
                       % (','.join(variables)))
    return list(variables.intersection(_supported_variables)), warnings


def validate_model(model):
    """
    Makes sure the model is supported, if not raise
    an error.  If no model is supplied we default to
    GFS.

    Parameters
    ----------
    mode : string
        The name of the desired model.

    Returns
    -------
        model : string
            The normalized or default model
        warnings : list of strings
            A list of warnings.
    """
    warnings = []
    if model is None:
        model = 'gfs'
        warnings.append('Using default model of %s' % model)
    model = model.lower().strip()
    if not model in _supported_models and model != 'spot':
        raise BadQuery("Unsupported model %s" % model)
    return model, warnings


def parse_domain(domain_str):
    """
    Parses the domain from a domain string that follows the
    saildocs format of:

    lat0,lat1,lon0,lon1

    each entry must include the direction N,S,E,W.  Longitudes
    are choosen such that the resulting bounding box of the
    domain doesn't exceed 180 degrees.  The latitudes are simply
    sorted to determine the upper and lower extend of the domain.

    Returns
    -------
    bounding_box : dict
        dictionary with entries N, S, E, W which indicate the
        northern-most, southern-most, eastern-most and western-most
        extents of the domain.
    """
    def floatify(latlon):
        """ Turns a latlon string into a float """
        sign = -2. * (latlon[-1].lower() in ['s', 'w']) + 1
        return float(latlon[:-1]) * sign
    corners = domain_str.strip().split(',')
    if not len(corners) == 4:
        raise BadQuery("Expected four comma seperated values "
                       "defining the corners of the domain.")

    is_lat = lambda x: x[-1].lower() in ['n', 's']
    lats = filter(is_lat, corners)
    if not len(lats) == 2:
        raise BadQuery("Expected two latitudes (determined by " +
                       "values ending in 'N' or 'S'")
    is_lon = lambda x: x[-1].lower() in ['e', 'w']
    lons = filter(is_lon, corners)
    if not len(lons) == 2:
        raise BadQuery("Expected two longitudes (determined by " +
                       "values ending in 'E' or 'W'")
    lats = np.array([floatify(x) for x in lats])
    lons = np.array([floatify(x) for x in lons])
    # make sure latitudes are in range.
    if np.any(lats > 90.) or np.any(lats < -90):
        raise BadQuery("Latitudes must be within -90 and 90, got %s" %
                       ','.join(map(str, lats)))
    # we let the user use either longitudes of 0 to 360
    # or -180 to 180, then convert to nautical (-180 to 180).
    if np.any(lons > 360.) or np.any(lons < -180.):
        raise BadQuery("Longitudes must be within -180 and 360, got %s" %
                       ','.join(map(str, lons)))
    # make sure lons end up in -180 to 180.
    lons = np.mod(lons + 180., 360.) - 180.

    lats = np.sort(lats)
    # chose the order for lons that minimizes the
    # angle.  This means people can't make requests
    # for more than 180 degrees at a time.  But that seems
    # more desirable than accidentally sending near global
    # files when the dateline is crossed.
    angular_diff = lambda x, y: np.mod(x - y, 360.)
    if angular_diff(*lons) > angular_diff(*reversed(lons)):
        lons = lons[::-1]

    domain = {'N': lats[1], 'S': lats[0],
              'W': lons[1], 'E': lons[0]}
    return domain


def parse_location(location_str):
    """
    Parses the location from a string that follows the
    saildocs format of:

    lat,lon or lon,lat

    each entry must include the direction N,S,E,W.

    Returns
    -------
    {'latitude': float, 'longitude': float}
    """
    def floatify(latlon):
        """ Turns a latlon string into a float """
        sign = -2. * (latlon[-1].lower() in ['s', 'w']) + 1
        return float(latlon[:-1]) * sign
    points = location_str.strip().split(',')
    if not len(points) == 2:
        raise BadQuery("Expected four comma seperated values "
                       "defining a single point.")

    is_lat = lambda x: x[-1].lower() in ['n', 's']
    lat = filter(is_lat, points)
    if not len(lat) == 1:
        raise BadQuery("Expected two latitudes (determined by " +
                       "values ending in 'N' or 'S'")
    is_lon = lambda x: x[-1].lower() in ['e', 'w']
    lon = filter(is_lon, points)
    if not len(lon) == 1:
        raise BadQuery("Expected two longitudes (determined by " +
                       "values ending in 'E' or 'W'")
    lat = floatify(lat[0])
    lon = floatify(lon[0])

    # make sure latitude is in range.
    if (lat > 90.) or (lat < -90):
        raise BadQuery("Latitude must be within -90 and 90, got %s" %
                       str(lat))
    # we let the user use either longitudes of 0 to 360
    # or -180 to 180, then convert to nautical (-180 to 180).
    if lon > 360. or lon < -180.:
        raise BadQuery("Longitudes must be within -180 and 360, got %s" %
                       str(lon))
    # make sure lons end up in -180 to 180.
    lon = np.mod(lon + 180., 360.) - 180.

    location = {'latitude': lat,
              'longitude': lon}
    return location


def parse_hours(hours_str):
    """
    Takes an hour string in the form:

        hour0,hour1,hour2...hourN

    and returns the expanded set of integer representations
    of the hour.  The ellipsis indicates that the previous
    hourly step should be extended till the last indicated
    hour.

    Parameters
    ----------
    hours_str : string or None
        String of requested valid times in hours.
        example: 0,24...96
        If None, the default of [24, 48, 72] is used.

    Returns
    -------
    hours : list of floats
        A list of the valid hours.
    warnings : list of strings
        A list of warnings
    """
    if hours_str is None:
        return [24., 48., 72.], ["No hours defined, using default of 24,48,72"]
    warnings = []
    # the hours query is a bit complex since it
    # involves interpreting the '..' as slices
    prev = 0
    # any . neighboring and , should be converted to a .
    # for example 0,3,...12 -> 0,3...12
    hours_str = re.sub('\.,|,\.', '.', hours_str)

    # yields valid hours
    def iter_hours():
        for hr in hours_str.split(','):
            if '..' in hr:
                # split on any set of at least two '.'s
                low, high = map(float, re.split('\.{2,}', hr))
                diff = low - prev
                if np.mod((high - low), diff):
                    raise BadQuery("Unsure how to parse the sequence %s" % hr)
                else:
                    hours = np.linspace(low, high,
                                        num=((high - low) / diff) + 1,
                                        endpoint=True)
            else:
                hours = map(float, [hr])
            for x in hours:
                prev = x
                yield x

    return list(iter_hours()), warnings


def parse_times(time_str):
    """
    spot forecasts use different formats for requesting forecast times.
    While that is unfortunate, for cross compatibility we do the same.

    Time strings take the form:

        days,interval

    So 4,3 would be every 3 hours for 4 days.
    """
    warnings = []
    days, interval = time_str.split(',')
    assert int(days) == float(days)
    days = int(days)
    assert int(interval) == float(interval)
    interval = int(interval)
    if interval < 3:
        warnings.append('Minimum interval is 3 hours')
    if days > 14:
        warnings.append('Maximum spot forecast period is 14 days')
    hours = np.arange(days * 24 + 1)[::interval]
    return hours.tolist(), warnings


def parse_grid(grid_str):
    """
    Parses a string representation of grid deltas, makes sure
    they are valid and then returns their floating point
    representations

    Parameters
    ----------
    grid_str : string or None
        If a string it must be of the format '%f,%f' corresponding
        to lat_diff,lon_diff.  If None, the default of 2 degrees
        is used.
    warnings : list of strings
        A list of warnings
    """
    # default to a grid size of 2 degrees, this is what
    # saildocs does.
    if grid_str is None:
        return (2., 2.), ['No grid size defined, defaulted to 2 degrees']
    # for now warnings isn't used.
    warnings = []
    parts = grid_str.split(',')
    if len(parts) != 2:
        raise BadQuery(("Expected grid deltas to be of form " +
                        "dlat,dlon the string %s doesn't follow " +
                        "those rules") % grid_str)
    lat_delta, lon_delta = grid_str.split(',')
    try:
        lat_delta = float(lat_delta)
        lon_delta = float(lon_delta)
    except:
        raise BadQuery("Expected grid deltas to be floating point numbers")

    if not int(2. * lat_delta) / 2. == lat_delta:
        raise BadQuery("lat delta must be a multiple of 0.5")
    if not int(2. * lon_delta) / 2. == lon_delta:
        raise BadQuery("lon delta must be a multiple of 0.5")

    return (np.abs(lat_delta), np.abs(lon_delta)), warnings


def split_fields(request, k):
    """
    Saildoc fields can be separated by '|', '/' or '\u015a'
    """
    fields = re.split(u'[\|\/\u015a]', unicode(request.strip()))
    return list(itertools.chain(fields, [None] * k))[:k]


def parse_forecast_request(request):
    """
    Parses a request for a gridded forecast.
    """
    warnings = []
    # takes the first 4 '|' separated fields, if fewer than
    # k exist the missing fields are replaced with None
    model_domain, grid_str, hours_str, variables = split_fields(request, 4)
    model, domain_str = model_domain.split(':')
    # parse the domain and make sure its ok
    domain = parse_domain(domain_str)
    # parse the grid_string
    grid_delta, grid_warnings = parse_grid(grid_str)
    warnings.extend(grid_warnings)
    # parse the hours string
    hours, hour_warnings = parse_hours(hours_str)
    warnings.extend(hour_warnings)
    # check the variables
    if variables is None:
        variables = []
    else:
        variables = variables.split(',')
    variables, var_warnings = validate_variables(variables)
    warnings.extend(var_warnings)
    return {'type': 'gridded',
            'model': model.lower().strip(),
            'domain': domain,
            'grid_delta': grid_delta,
            'hours': hours,
            'vars': variables,
            'warnings': warnings}


def parse_spot_request(request):
    """
    parses a request for a spot forecast
    """
    warnings = []
    model_domain, time_str, variables = split_fields(request, 3)
    spot, location_str = model_domain.split(':', 1)
    assert spot.lower() == 'spot'
    if ':' in location_str:
        model, location_str = location_str.split(':', 1)
        model = model.lower()
    else:
        model = 'gfs'
    location = parse_location(location_str)

    hours, time_warnings = parse_times(time_str)
    warnings.extend(time_warnings)

    if variables is None:
        variables = []
    else:
        variables = variables.split(',')
    variables, var_warnings = validate_variables(variables)
    warnings.extend(var_warnings)

    return {'type': 'spot',
            'model': model,
            'location': location,
            'hours': hours,
            'vars': variables,
            'warnings': warnings}


def parse_send_request(body):
    """
    Parses the a saildoc-like send request and returns
    a dictionary of attributes from the query.
    """
    # the model and domain are colon separated.
    model_domain, = split_fields(body, 1)
    model, _ = model_domain.split(':', 1)
    # make sure the model exists
    model, _ = validate_model(model)
    if model == 'spot':
        return parse_spot_request(body)
    else:
        return parse_forecast_request(body)


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
        raise BadQuery("Expected a space between the command and the body")
    command, body = command_split
    opts_args = filter(len, body.split(' '))
    if len(opts_args) > 1:
        args = opts_args[0]
        opts = opts_args[1:]
    else:
        args = opts_args[0]
        opts = None
    # Check if the command is supported
    if not command.lower() in _supported_commands:
        raise BadQuery("Unsupported command %s, only %s are supported"
                       % (command.lower(), ','.join(_supported_commands)))

    if command.lower() == 'send':
        query = parse_send_request(args)
    else:
        raise BadQuery("Unknown command handler.")
    return query
