import os
import copy
import xray
import zlib
import numpy as np
import netCDF4
import logging
import datetime

from bisect import bisect
from collections import OrderedDict

import sl.lib.conventions as conv

from sl.lib import objects, units

from xray import Dataset

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

# the beaufort scale in m/s
_beaufort_scale = np.array([0., 1., 3., 6., 10., 16., 21., 27.,
                            33., 40., 47., 55., 63., 75.]) * 1.94384449
# precipitation scale in kg.m-2.s-1
_precip_scale = np.array([1e-8, 1., 5.]) / 3600.
_direction_bins = np.arange(-np.pi, np.pi, step=np.pi / 8)
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
                                'attributes': {conv.UNITS: 'knot'}},
              conv.WIND_DIR: {'dtype': np.float32,
                              'dims': (conv.TIME, conv.LAT, conv.LON),
                              'divs': _direction_bins,
                              'bits': 4,
                              'attributes': {conv.UNITS: 'radians'}},
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


def tiny_array(arr, bits=None, divs=None, mask=None):
    """
    A convenience wrapper around  tiny_masked and tiny_unmasked
    which decides which method to use based on the mask argument
    """
    if mask is None:
        return tiny_unmasked(arr, bits=bits, divs=divs)
    else:
        return tiny_masked(arr, mask=mask, bits=bits, divs=divs)


def tiny_unmasked(arr, bits=None, divs=None):
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
    # this certainly not the fastest implementation but should do.
    # note that a zero now means that the value was less than all div
    # and a value of n means it was larger than the nth div.
    bins = np.maximum(1, np.minimum(n, np.array([bisect(divs, y)
                                                 for y in arr.flatten()])))
    bins = bins.astype(np.uint8)
    tiny = pack_ints(bins - 1, bits)
    tiny['divs'] = divs
    tiny['shape'] = arr.shape
    tiny['dtype'] = str(arr.dtype)
    return tiny


def tiny_masked(arr, mask=None, bits=None, divs=None):
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

    Returns
    -------
    tiny : dict
        A dictionary holding all the required arguments to
        expand the array. Note however that in order to expand
        the tiny array the same mask used to create it must
        be used.
    """
    if mask is None:
        return tiny_array(arr, bits, divs)
    masked = arr[mask].flatten()
    ta = tiny_unmasked(masked, bits=bits, divs=divs)
    ta['shape'] = arr.shape
    ta['masked'] = True
    return ta


def tiny_bool(arr, mask=None):
    """
    A convenience wrapper around tiny_array to be used for boolean arrays.
    """
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
                 masked=False, mask=None):
    """
    A convenience function which decides how to expand based on
    the masked flag.
    """
    if masked:
        return expand_masked(mask, packed_array, bits, shape, divs, dtype)
    else:
        return expand_unmasked(packed_array, bits, shape, divs, dtype)


def expand_unmasked(packed_array, bits, shape, divs, dtype=None):
    """
    Expands a tiny array to the original data type, but with
    a loss of information.  The original data, which has been
    binned, is returned as the value at the middle of each bin.


    Typical usage:

    original_array = np.random.normal(size=(30, 3))
    tiny = tiny_array(original_array)
    recovered = expand_unmasked(**tiny)
    """
    dtype = dtype or divs.dtype
    ndivs = divs.size
    lower_bins = unpack_ints(packed_array, bits, shape)
    if dtype == np.dtype('bool'):
        return lower_bins.astype('bool')
    upper_bins = np.minimum(lower_bins + 1, ndivs - 1)
    averages = 0.5 * (divs[lower_bins] + divs[upper_bins])
    return averages.astype(dtype).reshape(shape)


def expand_masked(mask, packed_array, bits, shape, divs, dtype=None):
    """
    Expands a masked tiny array by filling any masked values with nans
    """
    masked_shape = (np.sum(mask),)
    masked = expand_array(packed_array, bits, masked_shape, divs, dtype)
    ret = np.empty(shape)
    ret.fill(np.nan)
    ret[mask] = masked
    return ret


def small_array(arr, least_significant_digit):
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
    assert time_var.attributes[conv.UNITS].startswith('hours')
    origin = netCDF4.num2date([0],
                              time_var.attributes[conv.UNITS],
                              calendar='standard')[0]
    diffs = np.diff(np.concatenate([[0], time_var.data[:]]))
    fromordinal = datetime.datetime.fromordinal(origin.toordinal())
    seconds = int(datetime.timedelta.total_seconds(origin - fromordinal))
    augmented = np.concatenate([[origin.toordinal(),
                                 seconds],
                                diffs])
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
    if conv.UWND in obj.variables:
        units.convert_units(obj[conv.UWND], 'm/s')
        # we need both UWND and VWND to do anything with wind
        assert conv.VWND in obj.variables
        units.convert_units(obj[conv.VWND], 'm/s')
        # double check
        assert obj[conv.UWND].attributes[conv.UNITS] == 'm/s'
        assert obj[conv.VWND].attributes[conv.UNITS] == 'm/s'
    # make sure latitudes are in degrees and are on the correct scale
    assert 'degrees' in obj[conv.LAT].attributes[conv.UNITS]
    assert np.min(obj[conv.LAT]) >= -90
    assert np.max(obj[conv.LAT]) <= 90
    # make sure longitudes are in degrees and are on the correct scale
    assert 'degrees' in obj[conv.LON].attributes[conv.UNITS]
    assert np.min(obj[conv.LON]) >= -180
    assert np.max(obj[conv.LON]) <= 180
    assert obj[conv.UWND].shape == obj[conv.VWND].shape

    if conv.PRECIP in obj.variables:
        units.convert_units(obj[conv.PRECIP], 'kg.m-2.s-1')

_variable_order = [conv.TIME, conv.LAT, conv.LON,
                   conv.WIND_SPEED, conv.WIND_DIR,
                   conv.PRECIP, conv.PRESSURE]


def to_beaufort(obj):
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
    obj = copy.deepcopy(obj)
    uwnd = obj[conv.UWND].data
    vwnd = obj[conv.VWND].data
    # keep this ordered so the coordinates get written (and read) first
    encoded_variables = OrderedDict()

    encoded_variables[conv.TIME] = small_time(obj[conv.TIME])['packed_array']
    for v in [conv.LAT, conv.LON]:
        small = small_array(np.asarray(obj[v].data).astype(_variables[v]['dtype']),
                            _variables[v]['least_significant_digit'])
        encoded_variables[v] = small['packed_array']

    # convert the wind speeds to a beaufort scale and store them
    wind = [objects.Wind(*x) for x in zip(uwnd[:].flatten(), vwnd[:].flatten())]
    speeds = np.array([x.speed for x in wind]).reshape(uwnd.shape)
    speeds = speeds.astype(_variables[conv.WIND_SPEED]['dtype'])
    tiny_wind = tiny_array(speeds, bits=4, divs=_beaufort_scale)
    encoded_variables[conv.WIND_SPEED] = tiny_wind['packed_array']

    # convert the direction to cardinal directions and store them
    directions = np.array([x.dir for x in wind]).reshape(uwnd.shape)
    directions.astype(_variables[conv.WIND_DIR]['dtype'])
    tiny_direction = tiny_array(directions, bits=4,
                                divs=_direction_bins)
    encoded_variables[conv.WIND_DIR] = tiny_direction['packed_array']

    if conv.PRECIP in obj.variables:
        tiny_precip = tiny_array(obj[conv.PRECIP].data, bits=2.,
                                 divs=_precip_scale)
        encoded_variables[conv.PRECIP] = tiny_precip['packed_array']

    if conv.PRESSURE in obj.variables:
        tiny_pres = tiny_array(obj[conv.PRESSURE].data, bits=4,
                               divs=_pressure_scale)
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
    return zlib.compress(payload, 9)


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


def from_beaufort(payload):
    variables = list(unstring_beaufort(payload))

    out = Dataset()
    for vname, info in variables:
        if vname == conv.TIME:
            data, time_units = expand_small_time(info['packed_array'],
                                     info['dtype'],
                                     info['least_significant_digit'])
            info['attributes'] = {conv.UNITS: time_units}
            out[vname] = ((vname), data, info.get('attributes', None))
        elif vname in [conv.LAT, conv.LON]:
            data = expand_small_array(info['packed_array'],
                                     info['dtype'],
                                     info['least_significant_digit'])
            out[vname] = (vname, data, info.get('attributes', None))
        else:
            shape = [out.dimensions[d] for d in info['dims']]
            data = expand_array(info['packed_array'],
                                 bits=info['bits'],
                                 shape=shape,
                                 divs=info['divs'],
                                 dtype=info['dtype'])
            out[vname] = (info['dims'], data,
                          info.get('attributes', None))

    dims = out[conv.WIND_SPEED].dimensions
    vwnd = -np.cos(out[conv.WIND_DIR].data) * out[conv.WIND_SPEED].data
    uwnd = -np.sin(out[conv.WIND_DIR].data) * out[conv.WIND_SPEED].data
    out[conv.UWND] = (dims, uwnd, {conv.UNITS: 'm/s'})
    out[conv.VWND] = (dims, vwnd, {conv.UNITS: 'm/s'})
    return units.normalize_variables(out)

