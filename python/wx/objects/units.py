
_speed = {'m/s':1.94384449,
          'm s-1':1.94384449,
          'knot':1.0,
          'mph':0.868976242}

_speed_unit = 'knot'


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

    var = copy_variable(var, data=var.data.astype('f'))
    if hasattr(var, 'scale_factor'):
        scale = var.scale_factor
        delattr(var, 'scale_factor')
    else:
        scale = 1.0

    if hasattr(var, 'add_offset'):
        offset = var.add_offset
        delattr(var, 'add_offset')
    else:
        offset = 0.0

    if hasattr(var, 'units') and var.units in _speed:
        mult = (_speed[var.units] / _speed[_speed_unit])
        var.units = _speed_unit
    else:
        mult = 1.0

    var.data[:] = mult * (var.data[:] * scale + offset)
    return var