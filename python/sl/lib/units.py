import numpy as np

import sl.lib.conventions as conv

_length = {'m': 3.2808399,
           'ft': 1.0}
_length_unit = 'ft'

_speed = {'m/s': 1.94384449,
          'm s-1': 1.94384449,
          'knot': 1.0,
          'mph': 0.868976242}
_speed_unit = 'knot'

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


def normalize_units(v):
    """
    Inspects a variables units and converts them to
    the default unit.  If no units are found nothing
    is done.  All modifications to the data are
    done in place
    """
    assert v.data.dtype == np.float32
    # convert the units
    if conv.UNITS in v.attributes:
        units = v.attributes[conv.UNITS]
        if units in _speed:
            mult = (_speed[units] / _speed[_speed_unit])
            v.attributes[conv.UNITS] = _speed_unit
        elif units in _length:
            mult = (_length[units] / _length[_length_unit])
            v.attributes[conv.UNITS] = _length_unit
        v.data[:] *= mult
    return v
