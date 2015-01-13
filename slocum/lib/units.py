import numpy as np
import logging
import datetime

import slocum.lib.conventions as conv

_precip_rate = {'mm/hr': 3600.,
                'kg.m-2.s-1': 1.,
                'kg m-2 s-1': 1.}

_length = {'m': 3.2808399,
           'ft': 1.0}

_speed = {'m/s': 1.94384449,
          'm s-1': 1.94384449,
          'knot': 1.0,
          'mph': 0.868976242}

_longitude = {'degrees_east': 1.,
              'degrees_west': -1.}

_latitude = {'degrees_north': 1.,
              'degrees_south': -1.}

_pressure = {'Pa': 1.,
             'hPa': 100.,
             'kPa': 1./1000.,
             'kg m-1 s-2': 1.,
             'atm': 1./101325.,
             'bar': 1e-5}

_angle = {'radians': 180./np.pi,
          'rad': 180./np.pi,
          'degrees': 1.,
          'deg': 1.}


def transform_longitude(x):
    return np.mod(x + 180, 360) - 180


def validate_angle(x):
    assert np.all(x >= -180.)
    assert np.all(x <= 180.)
    return x


def validate_positive(x):
    assert np.all(x > 0.)
    return x

_all_units = [(_speed, 'm/s', None),
              (_length, 'm', validate_positive),
              (_longitude, 'degrees_east', transform_longitude),
              (_latitude, 'degrees_north', validate_angle),
              (_precip_rate, 'kg.m-2.s-1', validate_positive),
              (_pressure, 'Pa', None),
              (_angle, 'degrees', None)]


def _convert(v, possible_units, cur_units, new_units, validate=None):
    if cur_units == new_units and validate is None:
        logging.debug("units %s and %s are the same, skipping conversion"
                      % (cur_units, new_units))
        return v
    assert v.values.dtype == np.float32
    data = v.values
    mult = (possible_units[cur_units] / possible_units[new_units])
    data = data * mult
    if validate is not None:
        data = validate(data)
    v.values[...] = data
    v.attrs[conv.UNITS] = new_units
    return (v.dims, data, v.attrs)


def convert_units(v, new_units):
    # convert the units
    if conv.UNITS in v.attrs:
        cur_units = v.attrs[conv.UNITS]
        for (possible_units, _, _) in _all_units:
            if cur_units in possible_units:
                return _convert(v, possible_units, cur_units, new_units)
    else:
        raise ValueError("No units found so convertion doesn't make sense")


def normalize_units(v):
    """
    Inspects a variables units and converts them to
    the default unit.  If no units are found nothing
    is done.  All modifications to the data are
    done in place
    """
    # convert the units
    if conv.UNITS in v.attrs:
        cur_units = v.attrs[conv.UNITS]
        for (possible_units, default, validate) in _all_units:
            if cur_units in possible_units:
                return _convert(v, possible_units, cur_units, default,
                                validate=validate)
                break
    return v


def normalize_variables(dataset):
    """
    Iterates over all variables in a dataset and normalizes their units.
    """
    for vn, v in dataset.iteritems():
        dataset[vn] = normalize_units(v)
    return dataset


def convert_array(v, cur_units, new_units):
    """
    Converts an array from cur_units to new_units.  Same as convert_units but
    can be used if v has no 'attributes' attribute (e.g. spot forecasts).
    Conversion is done in place.
    """
    for (possible_units, _, _) in _all_units:
        if cur_units in possible_units:
            mult = (possible_units[cur_units] / possible_units[new_units])
            v[:] *= mult
            return v
    else:
        raise ValueError("No units found so convertion doesn't make sense")


def convert_scalar(s, cur_units, new_units):
    """
    Converts a single scalar from cur_units to new_units.
    """
    for (possible_units, __, __) in _all_units:
        if cur_units in possible_units:
            s *= (possible_units[cur_units] / possible_units[new_units])
            return s
    else:
        raise ValueError("Current unit '%s' not found in %s" % (cur_units,
                         __file__))


def default_units(cur_units):
    """
    Returns a string with the default units for cur_units or None if no
    matching default units are found.
    """
    for (possible_units, default, __) in _all_units:
        if cur_units in possible_units:
            return default
    else:
        return None


def  normalize_scalar(s, cur_units):
    """
    Converts scalar from cur_units to default units. Returns a tuple
    (normalized value, default unit). Raises ValueError if no matching default
    units are found for cur_units.
    """
    default = default_units(cur_units)
    if default:
        return convert_scalar(s, cur_units, default), default
    else:
        raise ValueError("No matching default unit found for '%s'" % cur_units)


def convert_from_default(s, new_units):
    """
    Converts scalar s from default units to new_units and returns the
    result.  Raises ValueError if no matching default unit is found for
    new_units.
    """
    default = default_units(new_units)
    if default:
        return convert_scalar(s, default, new_units)

    else:
        raise ValueError("No matching default unit found for '%s'" %
                new_units)


def total_seconds(dt):
    if isinstance(dt, np.timedelta64):
        return dt.astype('m8[us]').astype(datetime.datetime).total_seconds()
    elif isinstance(dt, datetime.timedelta):
        return dt.total_seconds()
    else:
        raise ValueError("expected timedelta or np.timedelta64")

