import re
import warnings
import numpy as np

from collections import OrderedDict

from slocum.lib import angles

import grads
import variables

_models = {'gefs': grads.GEFS(),
           'gfs': grads.GFS(),
           'cmcens': grads.CMCENS(),
           'fens': grads.FENS(),
           'rtofs': grads.RTOFS()}

# Variable names should not have white space, it gets removed
# when doing lookups.
_variables = OrderedDict([('time', variables.time),
                          ('realization', variables.realization),
                          ('latitude', variables.Latitude()),
                          ('longitude', variables.Longitude()),
                          ('wind', variables.wind),
                          ('current', variables.current),
                          ('temperature', variables.temperature),
                          ('pressure', variables.pressure),
                          ('waveheight', variables.wave_height,),
                          ('wavedirection', variables.wave_direction),
                         ])

_aliases = {'gfsens': 'gefs',
            'fnens': 'fens',
            'fnmocens': 'fens',
            'press': 'pressure',
            'temp': 'temperature',
            'precip': 'precipitation',
            'wvsig': 'waveheight',
            }


class BadQuery(BaseException):
    pass


def contains_variable(fcst, variable):
    """
    Returns True if all the required variables are contained
    in the forecast.
    """
    return all(v in fcst for v in variable.required_variables())


def available_variables(fcst):
    """
    Returns a list of all the variables in a forecast.
    """
    return [k for k, v in _variables.iteritems()
            if contains_variable(fcst, v)]


def dealias(k):
    """
    A convenience function that applies aliasing if it exists.
    """
    # if the variable name is in the aliases lookup table
    # we use the alias, otherwise stick with the original
    return _aliases[k] if k in _aliases else k


def lookup(k, lut):
    """
    Looks up a key, k, in lookup table, lut.  If the key isn't
    found directly we try aliasing and removing whitespace.
    """
    k = k.lower()
    # if the key is directly in the lookup table return it.
    if k in lut:
        return lut[k]
    # otherwise dealias and see if we can find it.
    k = dealias(k)
    if k in lut:
        return lut[k]
    # finally try removing white space and search again.
    k = dealias(re.sub('[_\W]', '', k))
    if not k in lut:
        raise BadQuery("Unsupported value: %s" % k)
    return k


def get_model(model_name):
    """
    Lookup a model by name
    """
    return lookup(model_name, _models)


def get_variable(variable_name):
    """
    Lookup a variable by string name
    """
    return lookup(variable_name, _variables)


def get_variable_names(variable_names):
    """
    Return a list of all valid variables
    """
    def is_valid_variable(x):
        try:
            get_variable(x)
            return True
        except BadQuery, e:
            warnings.warn("Unsupported variable %s" % x)
            return False
    return [v for v in variable_names if is_valid_variable(v)]


def validate_variables(variables):
    """
    Makes sure the requested variables are supported and
    reverts to defaults.  Any unsupported variables are filtered
    out and warnings are issued.

    Parameters
    ----------
    variables : list of strings
        A list of requested variables.  If empty the default
        of 'WIND' is used.

    Returns
    -------
    variables : list of strings
        A list of the supported variables
    """
    variables = set(filter(len, [x.strip().lower() for x in variables]))
    if not len(variables):
        variables = set(['wind'])
        warnings.warn('No variable requested, defaulting to WIND')
    # fail softly if some of the requested variables don't exist.
    variables = get_variable_names(variables)
    # the user specifically requested variables but none of them
    # are valid so we fail hard.
    if not len(variables):
        raise BadQuery("Couldn't recognize any variables")
    return variables


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
    """
    if model is None:
        model = 'gfs'
        warnings.warn('Using default model of %s' % model)
    model = model.lower().strip()
    if not model in _models and model != 'spot':
        raise BadQuery("Unsupported model %s" % model)
    return model


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
    lons = angles.angle_sort(lons)
    lats = np.sort(lats)
    domain = {'N': lats[-1], 'S': lats[0],
              'W': lons[0], 'E': lons[-1]}
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
    """
    if hours_str is None:
        warnings.warn("No hours defined, using default of 24,48,72")
        return [24., 48., 72.]
    # the hours query is a bit complex since it
    # involves interpreting the '..' as slices
    # any . neighboring and , should be converted to a .
    # for example 0,3,...12 -> 0,3...12
    hours_str = re.sub('\.,|,\.', '.', hours_str)

    # yields valid hours
    def iter_hours():
        prev = 0
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

    return list(iter_hours())


def parse_times(time_str):
    """
    spot forecasts use different formats for requesting forecast times.
    While that is unfortunate, for cross compatibility we do the same.

    Time strings take the form:

        days,interval

    So 4,3 would be every 3 hours for 4 days.
    """
    days, interval = time_str.split(',')
    assert int(days) == float(days)
    days = int(days)
    assert int(interval) == float(interval)
    interval = int(interval)
    if interval < 3:
        warnings.warn('Minimum interval is 3 hours')
    if days > 14:
        warnings.warn('Maximum spot forecast period is 14 days')
    hours = np.arange(days * 24 + 1)[::interval]
    return hours.tolist()
