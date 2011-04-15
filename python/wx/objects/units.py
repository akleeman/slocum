from wx.objects import core

_length = {'m':3.2808399,
           'ft':1.0}
_length_unit = 'ft'

_speed = {'m/s':1.94384449,
          'm s-1':1.94384449,
          'knot':1.0,
          'mph':0.868976242}
_speed_unit = 'knot'

_variables = {'U-component_of_wind_height_above_ground':'uwnd',
              'U-component of wind':'uwnd',
              'V-component_of_wind_height_above_ground':'vwnd',
              'V-component of wind':'vwnd',
              'Mean sea level pressure':'mslp',
              'Significant height of combined wind waves and swell':'combined_sea_height',
              'Signific.height,combined wind waves+swell':'combined_sea_height',
              'Direction of wind waves':'wave_dir',
              'Significant height of wind waves':'wave_height',
              'Mean period of wind waves':'wave_period',
              'Direction of swell waves':'swell_dir',
              'Significant height of swell waves':'swell_height',
              'Mean period of swell waves':'swell_period',
              'Primary wave direction':'primary_wave_dir',
              'Primary wave mean period':'primary_wave_period',
              'Secondary wave direction':'secondary_wave_dir',
              'Secondary wave mean period':'secondary_wave_period',
              }

def from_udunits(units, x):
    try:
        coards.from_udunits(units, x)
    except:
        raise
#        now = datetime.datetime.strptime(units, 'hour since %Y-%m-%dT%H:%M:%SZ')
#        return now + datetime.timedelta(seconds=int(3600*x))

def normalize_variable(var):
    """
    Scales, shifts and converts the variable to standard units
    """
    # normalizing only makes sense for numeric types
    if var.data.dtype.kind == 'S':
        return var

    # scale the variable if needed
    var = core.Variable(var.dimensions,
                        data=var.data.astype('f'),
                        attributes=var.attributes.copy())
    if hasattr(var, 'scale_factor'):
        scale = var.scale_factor
        delattr(var, 'scale_factor')
    else:
        scale = 1.0

    # shift the variable if needed
    if hasattr(var, 'add_offset'):
        offset = var.add_offset
        delattr(var, 'add_offset')
    else:
        offset = 0.0

    # convert the units
    if hasattr(var, 'units'):
        if var.units in _speed:
            mult = (_speed[var.units] / _speed[_speed_unit])
            var.units = _speed_unit
        if var.units in _length:
            mult = (_length[var.units] / _length[_length_unit])
            var.units = _length_unit
    else:
        mult = 1.0

    var.data[:] = mult * (var.data[:] * scale + offset)
    return var

def normalize_data(obj):
    # normalizes each variable and then renames them according to _variables
    out = core.Data()
    out.update_attributes(obj.attributes)

    for d, length in obj.dimensions.iteritems():
        out.create_dimension(_variables[d] if d in _variables else d, length)
    for name, var in obj.variables.iteritems():
        var = normalize_variable(var)
        out.add_variable(_variables[name] if name in _variables else name, var)
    return out