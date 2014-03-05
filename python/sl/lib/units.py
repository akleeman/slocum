import numpy as np

import sl.lib.conventions as conv

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
             'kPa': 1./1000.,
             'kg m-1 s-2': 1.,
             'atm': 1./101325.,
             'bar': 1e-5}


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
              (_pressure, 'Pa', None)]


def _convert(v, possible_units, cur_units, new_units, validate=None):
    if cur_units == new_units:
        return v
    assert v.data.dtype == np.float32
    mult = (possible_units[cur_units] / possible_units[new_units])
    v.data[:] *= mult
    if validate is not None:
        v.data[:] = validate(v.data[:])
    v.attributes[conv.UNITS] = new_units
    return v


def convert_units(v, new_units):
    # convert the units
    if conv.UNITS in v.attributes:
        cur_units = v.attributes[conv.UNITS]
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
    if conv.UNITS in v.attributes:
        cur_units = v.attributes[conv.UNITS]
        for (possible_units, default, validate) in _all_units:
            if cur_units in possible_units:
                _convert(v, possible_units, cur_units, default,
                         validate=validate)
                break
    return v


def normalize_variables(dataset):
    """
    Iterates over all variables in a dataset and normalizes their units.
    """
    for _, v in dataset.variables.iteritems():
        normalize_units(v)
    return dataset
