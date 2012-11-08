import re
import coards
import sl.objects.conventions as conv

from sl.objects import core

_length = {'m':3.2808399,
           'ft':1.0}
_length_unit = 'ft'

_speed = {'m/s':1.94384449,
          'm s-1':1.94384449,
          'knot':1.0,
          'mph':0.868976242}
_speed_unit = 'knot'

_variables = {'10 metre U wind component':'uwnd',
              '10 metre V wind component':'vwnd',
              'U-component_of_wind_height_above_ground':'uwnd',
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

def from_udunits(x, units, tzinfo=None):
    return coards.from_udunits(x, units).replace(tzinfo=tzinfo)

def normalize_variable(v):
    """
    Scales, shifts and converts the variable to standard units
    """
    # normalizing only makes sense for numeric types
    if v.data.dtype.kind == 'S':
        return v

    # scale the variable if needed
    var = core.Variable(v.dimensions,
                        data=v.data.astype('f'))
                        #attributes=v.attributes.copy())
    if hasattr(v, 'scale_factor'):
        scale = v.scale_factor
        #delattr(var, 'scale_factor')
    else:
        scale = 1.0

    # shift the variable if needed
    if hasattr(v, 'add_offset'):
        offset = v.add_offset
        #delattr(var, 'add_offset')
    else:
        offset = 0.0

    # convert the units
    if conv.UNITS in v.attributes:
        units = v.attributes[conv.UNITS]
        if units in _speed:
            mult = (_speed[units] / _speed[_speed_unit])
            var.attributes[conv.UNITS] = _speed_unit
        elif units in _length:
            mult = (_length[units] / _length[_length_unit])
            var.attributes[conv.UNITS] = _length_unit
        else:
            var.attributes[conv.UNITS] = units
            mult = 1.0
    else:
        mult = 1.0

    var.data[:] = mult * (var.data[:] * scale + offset)
    return var

def normalize_time_units(units):
    try:
        from_udunits(0, units)
    except:
        #A capital T surrounded on both sides by two digits gets removed
        units = re.sub('(\d{2})T(\d{2})', '\\1 \\2', units)
        units = re.sub('(\d{2})Z$', '\\1', units)
        from_udunits(0, units)
    return units

def normalize_data(obj):
    # normalizes each variable and then renames them according to _variables
    out = core.Data()
    timevars = [x for x in obj.variables.keys() if 'time' in x]
    if len(timevars) > 1:
        timesets = [set(x for x in obj[v].data) for v in timevars]
        shared_times = reduce(lambda x, y: x.intersection(y), timesets)
        for timevar in timevars:
            alternate_inds = [i for i, x in enumerate(obj[timevar].data)
                              if x in shared_times]
            obj = obj.take(alternate_inds, timevar)
    #out.update_attributes(obj.attributes)
    for d, length in obj.dimensions.iteritems():
        out.create_dimension(_variables[d] if d in _variables else d, length)
    for name, var in obj.variables.iteritems():
        var = normalize_variable(var)
        attr = dict(var.attributes.iteritems())
        timevars = [x for x in var.dimensions if 'time' in x]
        normtime = lambda x : conv.TIME if conv.TIME in x else x
        dimensions = tuple(map(normtime, var.dimensions))
        if len(timevars) > 1:
            raise ValueError("expected a single time")
        if name == conv.TIME:
            attr[conv.UNITS] = normalize_time_units(var.attributes[conv.UNITS])
        if 'time' in name and not name == conv.TIME:
            continue
        out.create_variable(_variables[name] if name in _variables else name,
                            dim=dimensions,
                            data=var.data,
                            attributes=attr)
    out.__dict__['record_dimension'] = obj.record_dimension
    return out