import numpy as np

import sl.lib.conventions as conv

_length = {'m': 3.2808399,
           'ft': 1.0}
_length_unit = 'm'

_speed = {'m/s': 1.94384449,
          'm s-1': 1.94384449,
          'knot': 1.0,
          'mph': 0.868976242}
_speed_unit = 'm/s'

_longitude = {'degrees_east': 1.,
              'degrees_west': -1.}
_longitude_unit = 'degrees_east'

_latitude = {'degrees_north': 1.,
              'degrees_south': -1.}
_latitude_unit = 'degrees_north'

_variables = {'10 metre U wind component': conv.UWND,
              '10 metre V wind component': conv.VWND,
              'U-component_of_wind_height_above_ground': conv.UWND,
              'U-component of wind': conv.UWND,
              'V-component_of_wind_height_above_ground': conv.VWND,
              'V-component of wind': conv.VWND,
              'Mean sea level pressure': conv.PRESSURE,
              'Significant height of combined wind waves and swell': 'combined_sea_height',
              'Signific.height,combined wind waves+swell': 'combined_sea_height',
              'Direction of wind waves': 'wave_dir',
              'Significant height of wind waves':  'wave_height',
              'Mean period of wind waves':  'wave_period',
              'Direction of swell waves':  'swell_dir',
              'Significant height of swell waves':  'swell_height',
              'Mean period of swell waves':  'swell_period',
              'Primary wave direction':  'primary_wave_dir',
              'Primary wave mean period':  'primary_wave_period',
              'Secondary wave direction':  'secondary_wave_dir',
              'Secondary wave mean period':  'secondary_wave_period',
              }


def transform_longitude(x):
    return np.mod(x + 180, 360) - 180


def validate_angle(x):
    assert np.all(x >= -180.)
    assert np.all(x <= 180.)
    return x


def validate_length(x):
    assert np.all(x > 0.)
    return x

_all_units = [(_speed, 'm/s', None),
              (_length, 'm', validate_length),
              (_longitude, 'degrees_east', transform_longitude),
              (_latitude, 'degrees_north', validate_angle)]


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
        for (possible_units, default, validate) in _all_units:
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
        # TODO: is it fair to assume that only time will have units with 'since'?
        if 'since' in cur_units:
            # force time to have units of hours
            assert cur_units.startswith('hours')
    return v


def normalize_variables(dataset):
    for k, v in dataset.variables.iteritems():
        print k
        normalize_units(v)
    return dataset