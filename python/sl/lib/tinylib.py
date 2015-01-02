import os
import zlib
import base64
import logging
import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger(os.path.basename(__file__))

try:
    import xray
    _has_xray = True
except:
    logger.warn("xray not found, only spot forecasts will be available.")
    _has_xray = False

from collections import OrderedDict

import sl.lib.conventions as conv

from sl.lib import objects, units

# the beaufort scale in m/s
_beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                            33., 40., 47., 55., 63., 75.]) / 1.94384449
# ensemble wind speed spread
_ws_spread_scale = _beaufort_scale
# precipitation scale in kg.m-2.s-1
_precip_scale = np.array([1e-8, 1., 5.]) / 3600.
# 'S' sits between _direction_bins[-1] and _direction_bins[0]
_direction_bins = np.linspace(-15 * np.pi/16., 15 * np.pi/16., 16)
# this pressure scale was derived by taking all the MSL pressures
# from a forecast run and computing the quantiles (then rounding to
# more friendly numbers).  We might find that we can add precision
# by focusing the bins around pressures expected in sailing waters
# rather than globally ... but for now this should work.
_pressure_scale = np.array([97500., 99000., 99750,
                            100500., 100700., 100850.,
                            101000., 101150., 101350.,
                            101600., 101900., 102150.,
                            102500., 103100., 104000.])

_variables = {conv.WIND_SPEED: {'dtype': np.float32,
                                'dims': (conv.TIME, conv.LAT, conv.LON),
                                'divs': _beaufort_scale,
                                'bits': 4,
                                'attributes': {conv.UNITS: 'm/s'}},
              conv.WIND_DIR: {'dtype': np.float32,
                              'dims': (conv.TIME, conv.LAT, conv.LON),
                              'divs': _direction_bins,
                              'bits': 4,
                              'attributes': {conv.UNITS: 'radians'}},
              # The 'long_name' and 'n' attributes below are not the cleanest
              # solution (set in enslib) but otherwise we have to add
              # attributes to the payload before shipping
              conv.ENS_SPREAD_WS: {'dtype': np.float32,
                      'dims': (conv.TIME, conv.LAT, conv.LON),
                      'divs': _ws_spread_scale,
                      'bits': 4,
                      'attributes': {conv.UNITS: 'm/s',
                                    'long_name':
                                        'Mean of top n (ens - gfs) deltas',
                                    'n': 2}},
              conv.PRECIP: {'dtype': np.float32,
                              'dims': (conv.TIME, conv.LAT, conv.LON),
                              'divs': _precip_scale,
                              'bits': 2,
                              'attributes': {conv.UNITS: 'kg.m-2.s-1'}},
              conv.PRESSURE: {'dtype': np.float32,
                              'dims': (conv.TIME, conv.LAT, conv.LON),
                              'divs': _pressure_scale,
                              'bits': 4,
                              'attributes': {conv.UNITS: 'Pa'}},
              conv.LON: {'dtype': np.float64,
                         'dims': (conv.LON, ),
                         'least_significant_digit': 2,
                         'attributes': {conv.UNITS: 'degrees east'}},
              conv.LAT: {'dtype': np.float64,
                         'dims': (conv.LAT, ),
                         'least_significant_digit': 2,
                         'attributes': {conv.UNITS: 'degrees north'}},
              conv.TIME: {'dtype': np.int64,
                          'dims': (conv.TIME, ),
                          'least_significant_digit': 0},
              }

# shorthand for check_beaufort, to_beaufort
_units = {k: _variables[k]['attributes'][conv.UNITS] for k in _variables
        if 'attributes' in _variables[k] and conv.UNITS in
        _variables[k]['attributes']}

_variable_order = [conv.TIME, conv.LAT, conv.LON,
                   conv.WIND_SPEED, conv.WIND_DIR,
                   conv.ENS_SPREAD_WS,
                   conv.PRECIP, conv.PRESSURE]


def pack_ints(arr, req_bits=None):
    """
    Takes an array of integers and returns a dictionary
    holding the packed data.

    Parameters
    ----------
    arr : np.ndarray
        An array of integers which are to be packed.  The maximum
        value of the array is used to determine the number of
        required bits (unless otherwise specified).
    req_bits : int (optional)
        The number of bits required to losslessly store arr.
    """
    assert np.all(np.isfinite(arr))
    # there has got to be a better way to check if an array
    # holds integers, but this does this trick
    try:
        np.iinfo(arr.dtype)
    except:
        raise ValueError("pack_ints requires an integer array as input")
    if np.any(arr < 0):
        raise ValueError("expected all values of arr to be non negative")
    # we assume that these integer arrays are unsigned.
    assert np.min(arr) >= 0
    # the number of bits required to store the largest number in arr
    # TODO: we could think of better packing schemes ... but not now.
    req_bits = req_bits or np.ceil(np.log2(np.max(arr) + 1))
    assert int(req_bits) == req_bits
    req_bits = int(req_bits)
    if req_bits >= 8:
        raise ValueError("why not just use uint8 or bigger?")
    # we pack several sub 8 bit into one 'out_bits' bit integer.
    # will it ever make sense to use more than 8?
    out_bits = 8
    if np.mod(out_bits, req_bits):
        print "output: %d    required: %d" % (out_bits, req_bits)
        raise ValueError("bit size in the output type must be a " +
                         "multiple of the num bits")
    vals_per_int = out_bits / req_bits
    # packed_size is the number of required elements in the output array
    packed_size = np.ceil(float(arr.size) * req_bits / out_bits)
    output = np.zeros(shape=(packed_size,), dtype=np.uint8)
    # iterate over each input element
    for i, x in enumerate(arr.flatten()):
        # determine which index
        ind = np.floor(i / vals_per_int)
        if np.mod(i, vals_per_int):
            output[ind] = output[ind] << req_bits
        output[ind] += x
    packed_array = {'packed_array': output.tostring(),
                    'bits': req_bits,
                    'shape': arr.shape,
                    'dtype': str(arr.dtype)}
    return packed_array


def unpack_ints(packed_array, bits, shape, dtype=None):
    """
    Takes a packed array and expands it to its original size such that

    np.all(original_array == unpack_ints(**pack_ints(original_array)))

    Parameters
    ----------
    packed_array : np.ndarray
        An integer array of packed data.
    bits : int
        The number of bits used to pack each encoded int
    shape : tuple of ints
        The resulting shape of the array
    dtype : numpy.dtype
        The output data type

    Returns
    -------
    unpacked : np.ndarray
        The unpacked values held in packed_array
    """
    # the dtype here is different than the final one.  At this point
    # all the data is stored in packed bytes.
    packed_array = np.fromstring(packed_array, dtype=np.uint8)
    info = np.iinfo(packed_array.dtype)
    enc_bits = info.bits
    if np.mod(enc_bits, bits):
        raise ValueError("the bit encoding doesn't line up")
    # how many values are in each int?
    vals_per_int = enc_bits / bits
    size = reduce(lambda x, y: x * y, shape)
    # the masks are used for logical AND comparisons to retrieve the values
    masks = [(2 ** bits - 1) << i * bits for i in range(vals_per_int)]

    def iter_vals():
        # iterate over array values until we've generated enough values to
        # fill the original array
        cnt = 0
        for x in packed_array:
            # nvals is the number of values contained in the next packed int
            nvals = min(size - cnt, vals_per_int)
            # by comparing the int x to each of the masks and shifting we
            # recover the packed values.  Note that this will always generate
            # 'vals_per_int' values but sometimes we only need a few which is
            # why we select out the first nvals
            reversed_vals = [(x & y) >> j * bits for j, y in enumerate(masks)]
            for v in reversed(reversed_vals[:nvals]):
                yield v
            cnt += nvals
    # recreate the original array
    return np.array([x for x in iter_vals()], dtype=dtype).reshape(shape)


def tiny_array(arr, bits=None, divs=None, mask=None, wrap=False):
    """
    A convenience wrapper around  tiny_masked and tiny_unmasked which decides
    which method to use based on the mask argument.  If wrap == True it is
    assumed that the binning continues from the last back to the first bin
    (i.e. the '0' bin sits between the last and first bounding value in divs as
    is the case for angles).
    """
    assert np.all(np.isfinite(arr))
    if mask is None:
        return tiny_unmasked(arr, bits=bits, divs=divs, wrap=wrap)
    else:
        return tiny_masked(arr, mask=mask, bits=bits, divs=divs, wrap=wrap)


def tiny_unmasked(arr, bits=None, divs=None, wrap=False):
    """
    Bins the values in arr by using dividers (divs).  The result
    is a set of integers indicating which bin each value in arr belongs
    to.  These bin indicators are then stored as packed integers (provided
    the number of bins can be stored in 4 or less bits.

    Parameters
    ----------
    arr : np.ndarray
        An array of values that are going to be lossy compressed by binning
    bits : int
        The number of bits that should be used to hold arr
    divs : np.ndarray
        The dividers (ie, edges) of the bins, should be length bins
    wrap : boolean
        If wrap == True it is assumed that the binning continues from the last
        back to the first bin (i.e. the '0' bin sits between the last and first
        bounding value in divs as is the case for angles).

    Returns
    -------
    tiny : dict
        A dictionary holding all the required arguments to expand the array
    """
    if divs is None:
        bits = bits or 4
        n = np.power(2., bits)
        lower = np.min(arr)
        upper = np.max(arr)
        # n is the number of 'levels' that can be represented'
        divs = np.linspace(lower, upper, n).astype(np.float)
    else:
        bits = bits or np.ceil(np.log2(divs.size))
    if int(np.log2(bits)) != np.log2(bits):
        raise ValueError("bits must be a power of two")
    # it doesn't make sense to store anything larger than this
    assert bits <= 4
    n = np.power(2., bits)
    # for each element of the array, count how many divs are less than the elem
    # note that a zero after shifting means that the value was less than all div
    # and a value of n means it was larger than the nth div.
    if not wrap:
        bins = np.maximum(0, np.digitize(arr.reshape(-1), divs) - 1)
    else:
        bins = np.digitize(arr.reshape(-1), divs) % len(divs)
    tiny = pack_ints(bins, bits)
    tiny['divs'] = divs
    tiny['shape'] = arr.shape
    tiny['dtype'] = str(arr.dtype)
    return tiny


def tiny_masked(arr, mask=None, bits=None, divs=None, wrap=False):
    """
    Creates a tiny array with a mask.  Only the values
    that are not masked are packed.  The mask is not
    stored, so must be persisted to expansion.

    See tiny_unmasked for more details.

    Parameters
    ----------
    arr : np.ndarray
        The full precision array that is to be packed.
    mask : np.ndarray
        A masked which is applied to arr, via arr[mask]
    bits : int
        The number of bit used to store arr.
    wrap : boolean
        If wrap == True it is assumed that the binning continues from the last
        back to the first bin (i.e. the '0' bin sits between the last and first
        bounding value in divs as is the case for angles).

    Returns
    -------
    tiny : dict
        A dictionary holding all the required arguments to
        expand the array. Note however that in order to expand
        the tiny array the same mask used to create it must
        be used.
    """
    if mask is None:
        return tiny_array(arr, bits, divs, wrap=wrap)
    masked = arr[mask].flatten()
    ta = tiny_unmasked(masked, bits=bits, divs=divs, wrap=wrap)
    ta['shape'] = arr.shape
    ta['masked'] = True
    return ta


def tiny_bool(arr, mask=None):
    """
    A convenience wrapper around tiny_array to be used for boolean arrays.
    """
    assert np.all(np.isfinite(arr))
    if not arr.dtype == 'bool':
        raise ValueError("expected a boolean valued array")
    return tiny_array(arr, bits=1, divs=np.array([0., 1.]), mask=mask)


def expand_bool(packed_array, shape, **kwdargs):
    """
    A convenience wrapper around exapnd_array to be used for boolean arrays.
    """
    return expand_array(packed_array, bits=1, shape=shape,
                        divs=np.array([False, True]),
                        dtype=None, masked=False, mask=None)


def expand_array(packed_array, bits, shape, divs, dtype=None,
                 masked=False, mask=None, wrap_val=None):
    """
    A convenience function which decides how to expand based on
    the masked flag.
    """
    if masked:
        return expand_masked(mask, packed_array, bits, shape, divs, dtype,
                wrap_val=wrap_val)
    else:
        return expand_unmasked(packed_array, bits, shape, divs, dtype,
                wrap_val=wrap_val)


def expand_unmasked(packed_array, bits, shape, divs, dtype=None,
                    wrap_val=None):
    """
    Expands a tiny array to the original data type, but with a loss of
    information.  The original data, which has been binned, is returned as
    the value at the middle of each bin. If wrap != None it is assumed that
    the binning wraps (e.g. for angles) and that wrap_val is the center
    value between the last and first bounding value in divs.

    Typical usage:

    original_array = np.random.normal(size=(30, 3))
    tiny = tiny_array(original_array)
    recovered = expand_unmasked(**tiny)
    """
    dtype = dtype or divs.dtype
    ndivs = divs.size
    bins = unpack_ints(packed_array, bits, shape)
    if dtype == np.dtype('bool'):
        return bins.astype('bool')
    if wrap_val:
        upper = bins
        lower = (bins-1) % len(divs)
        averages = np.where(bins > 0, 0.5 * (divs[lower] + divs[upper]),
                            wrap_val)
    else:
        lower_bins = bins
        upper_bins = np.minimum(lower_bins + 1, ndivs - 1)
        averages = 0.5 * (divs[lower_bins] + divs[upper_bins])

    return averages.astype(dtype).reshape(shape)


def expand_masked(mask, packed_array, bits, shape, divs, dtype=None,
                  wrap_val=None):
    """
    Expands a masked tiny array by filling any masked values with nans
    """
    masked_shape = (np.sum(mask),)
    masked = expand_array(packed_array, bits, masked_shape, divs, dtype,
                          wrap_val)
    ret = np.empty(shape)
    ret.fill(np.nan)
    ret[mask] = masked
    return ret


def small_array(arr, least_significant_digit):
    assert np.all(np.isfinite(arr))
    data = np.round(arr * np.power(10, least_significant_digit))
    return {'packed_array': zlib.compress(data.tostring(), 9),
            'dtype': arr.dtype,
            'least_significant_digit': least_significant_digit}


def expand_small_array(packed_array, dtype, least_significant_digit):
    arr = np.fromstring(zlib.decompress(packed_array), dtype=dtype)
    arr = arr / np.power(10, least_significant_digit)
    return arr


def small_time(time_var):
    time_var = xray.conventions.encode_cf_variable(time_var)
    assert time_var.attrs[conv.UNITS].lower().startswith('hour')
    origin = xray.conventions.decode_cf_datetime([0],
                                                 time_var.attrs[conv.UNITS],
                                                 calendar='standard')
    origin = pd.to_datetime(origin[0])
    diffs = np.diff(np.concatenate([[0], time_var.values[:]]))
    np.testing.assert_array_equal(diffs.astype('int'), diffs)
    fromordinal = datetime.datetime.fromordinal(origin.toordinal())
    seconds = int(datetime.timedelta.total_seconds(origin - fromordinal))
    augmented = np.concatenate([[origin.toordinal(), seconds],
                                diffs.astype(_variables[conv.TIME]['dtype'])])
    return small_array(augmented, least_significant_digit=0)


def expand_small_time(packed_array, dtype, least_significant_digit):
    augmented = expand_small_array(packed_array, dtype,
                                   least_significant_digit)
    origin = datetime.datetime.fromordinal(augmented[0])
    origin += datetime.timedelta(seconds=int(augmented[1]))
    times = np.cumsum(augmented[2:])
    units = origin.strftime('hours since %Y-%m-%d %H:%M:%S')
    return times, units


def check_beaufort(obj):

    if conv.UWND in obj:
        units.convert_units(obj[conv.UWND], _units[conv.WIND_SPEED])
        # we need both UWND and VWND to do anything with wind
        assert conv.VWND in obj
        units.convert_units(obj[conv.VWND], _units[conv.WIND_SPEED])
        # double check
        assert obj[conv.UWND].attrs[conv.UNITS] == _units[conv.WIND_SPEED]
        assert obj[conv.VWND].attrs[conv.UNITS] == _units[conv.WIND_SPEED]

    if conv.ENS_SPREAD_WS in obj:
        units.convert_units(obj[conv.ENS_SPREAD_WS],
                _units[conv.ENS_SPREAD_WS])
        # double check
        assert (obj[conv.ENS_SPREAD_WS].attrs[conv.UNITS] ==
                _units[conv.ENS_SPREAD_WS])

    # make sure latitudes are in degrees and are on the correct scale
    assert 'degrees' in obj[conv.LAT].attrs[conv.UNITS]
    assert np.min(np.asarray(obj[conv.LAT].values)) >= -90
    assert np.max(np.asarray(obj[conv.LAT].values)) <= 90
    # make sure longitudes are in degrees and are on the correct scale
    assert 'degrees' in obj[conv.LON].attrs[conv.UNITS]
    obj[conv.LON].values[:] = np.mod(obj[conv.LON].values + 180., 360) - 180.
    assert obj[conv.UWND].shape == obj[conv.VWND].shape

    if conv.PRECIP in obj:
        units.convert_units(obj[conv.PRECIP], _units[conv.PRECIP])


def to_beaufort_single(obj):
    """
    Takes an object holding wind and precip, cloud or pressure
    variables and compresses it by converting zonal and meridional
    winds to wind speed and direction, then compressing to
    beaufort scales and second order cardinal directions.

    Parameters
    ----------
    """
    # first we make sure all the data is in the expected units
    check_beaufort(obj)
    uwnd = obj[conv.UWND].values
    vwnd = obj[conv.VWND].values
    # keep this ordered so the coordinates get written (and read) first
    encoded_variables = OrderedDict()
    encoded_variables[conv.TIME] = small_time(obj[conv.TIME])['packed_array']
    for v in [conv.LAT, conv.LON]:
        small = small_array(np.asarray(obj[v].values).astype(_variables[v]['dtype']),
                            _variables[v]['least_significant_digit'])
        encoded_variables[v] = small['packed_array']
    # convert the wind speeds to a beaufort scale and store them
    wind = [objects.Wind(*x) for x in zip(uwnd[:].flatten(), vwnd[:].flatten())]
    speeds = np.array([x.speed for x in wind]).reshape(uwnd.shape)
    assert obj[conv.UWND].attrs[conv.UNITS] == _units[conv.WIND_SPEED]
    assert obj[conv.VWND].attrs[conv.UNITS] == _units[conv.WIND_SPEED]
    speeds = speeds.astype(_variables[conv.WIND_SPEED]['dtype'])
    tiny_wind = tiny_array(speeds, bits=_variables[conv.WIND_SPEED]['bits'],
            divs=_beaufort_scale)
    encoded_variables[conv.WIND_SPEED] = tiny_wind['packed_array']

    # convert the direction to cardinal directions and store them
    directions = np.array([x.dir for x in wind]).reshape(uwnd.shape)
    directions.astype(_variables[conv.WIND_DIR]['dtype'])
    tiny_direction = tiny_array(directions,
            bits=_variables[conv.WIND_DIR]['bits'], divs=_direction_bins,
            wrap=True)
    encoded_variables[conv.WIND_DIR] = tiny_direction['packed_array']

    if conv.ENS_SPREAD_WS in obj:
        tiny_ws_spread = tiny_array(obj[conv.ENS_SPREAD_WS].values,
                bits=_variables[conv.ENS_SPREAD_WS]['bits'],
                divs=_ws_spread_scale)
        encoded_variables[conv.ENS_SPREAD_WS] = tiny_ws_spread['packed_array']

    if conv.PRECIP in obj:
        tiny_precip = tiny_array(obj[conv.PRECIP].values,
                bits=_variables[conv.PRECIP]['bits'], divs=_precip_scale)
        encoded_variables[conv.PRECIP] = tiny_precip['packed_array']

    if conv.PRESSURE in obj:
        tiny_pres = tiny_array(obj[conv.PRESSURE].values,
                bits=_variables[conv.PRESSURE]['bits'], divs=_pressure_scale)
        encoded_variables[conv.PRESSURE] = tiny_pres['packed_array']

    def stringify(vname, packed):
        vid = _variable_order.index(vname)
        l = np.array(len(packed), dtype=np.uint16)
        # make sure the length fits in 16 bits
        assert len(packed) == l
        l0 = (l >> 8) & 0xff
        l1 = l & 0xff
        header = np.array([vid, l0, l1], dtype=np.uint8).tostring()
        return ''.join([header, packed])

    payload = ''.join(stringify(k, v) for k, v in encoded_variables.iteritems())
    logger.debug("compression ratio: %f" %
                 (len(zlib.compress(payload, 9)) / float(len(payload))))
    return payload


def unstring_beaufort(payload):
    payload = zlib.decompress(payload)
    while len(payload):
        vid, l0, l1 = np.fromstring(payload[:3], dtype=np.uint8)
        vlen = (l0 << 8) + l1
        packed = payload[3:(vlen + 3)]
        payload = payload[(vlen + 3):]
        vname = _variable_order[vid]
        info = _variables[vname]
        info['packed_array'] = packed
        yield vname, info


def beaufort_to_dict(payload):
    variables = list(unstring_beaufort(payload))
    out = {}
    for vname, info in variables:
        if vname == conv.TIME:
            data, time_units = expand_small_time(
                    info['packed_array'], info['dtype'],
                    info['least_significant_digit'])
            info['attributes'] = {conv.UNITS: time_units}
            out[vname] = ((vname), data, info.get('attributes', None))
        elif vname in [conv.LAT, conv.LON]:
            data = expand_small_array(info['packed_array'],
                                     info['dtype'],
                                     info['least_significant_digit'])
            out[vname] = (vname, data, info.get('attributes', None))
        elif vname == conv.WIND_DIR:
            shape = [out[d][1].size for d in info['dims']]
            data = expand_array(info['packed_array'],
                                 bits=info['bits'],
                                 shape=shape,
                                 divs=info['divs'],
                                 dtype=info['dtype'],
                                 wrap_val=np.pi)
            out[vname] = (info['dims'], data,
                          info.get('attributes', None))
        else:
            shape = [out[d][1].size for d in info['dims']]
            data = expand_array(info['packed_array'],
                                 bits=info['bits'],
                                 shape=shape,
                                 divs=info['divs'],
                                 dtype=info['dtype'])
            out[vname] = (info['dims'], data,
                          info.get('attributes', None))
    dims = out[conv.WIND_SPEED][0]
    vwnd = -np.cos(out[conv.WIND_DIR][1]) * out[conv.WIND_SPEED][1]
    uwnd = -np.sin(out[conv.WIND_DIR][1]) * out[conv.WIND_SPEED][1]
    out[conv.UWND] = (dims, uwnd, {conv.UNITS: 'm/s'})
    out[conv.VWND] = (dims, vwnd, {conv.UNITS: 'm/s'})
    return out


def from_beaufort_single(payload):
    variables = beaufort_to_dict(payload)
    out = xray.Dataset(variables)
    out[conv.TIME] = xray.conventions.decode_cf_variable(out[conv.TIME])
    return units.normalize_variables(out)


def to_beaufort(fcst):
    if conv.ENSEMBLE in fcst.dims:
        tinys = [to_beaufort(fcst.isel(**{conv.ENSEMBLE: i}))
                 for i in range(fcst.dims[conv.ENSEMBLE])]
        tiny_fcst = '\t'.join([base64.b64encode(x) for x in tinys])
        tiny_fcst = 'ENS:%s' % tiny_fcst
    else:
        tiny_fcst = to_beaufort_single(fcst)
    return zlib.compress(tiny_fcst, 9)


def from_beaufort(payload):
    payload = zlib.decompress(payload)
    if payload.startswith('ENS'):
        payload = payload[3:]
        fcst = [base64.b64decode(x) for x in payload.split('\t')]
        fcst = xray.concat([from_beaufort_single(f) for f in fcst],
                           conv.ENSEMBLE)
    else:
        fcst = from_beaufort_single(payload)
    return fcst
