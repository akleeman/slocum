import os
import xarray as xra
import zlib
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from functools import reduce

logger = logging.getLogger(os.path.basename(__file__))

from collections import OrderedDict


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

    Returns
    ----------
    packed_array : dict
        A dictionary containing a key 'packed_array' which contains
        a character string holding a packed version of 'arr', as well
        as various other key/values used to reconstruct the original
        array shape/dtype.
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
    req_bits = req_bits or np.ceil(np.log2(np.max(arr) + 1))
    assert int(req_bits) == req_bits
    req_bits = int(req_bits)
    # arr is stored base 0 using req_bits, so we make sure req_bit
    # can actually hold all the data!
    assert np.max(arr) < np.power(2, req_bits)
    if req_bits >= 8:
        raise ValueError("why not just use uint8 or bigger?")
    # we pack several sub 8 bit into one 'out_bits' bit integer.
    # will it ever make sense to use more than 8?
    out_bits = 8
    if np.mod(out_bits, req_bits):
        print("output: %d    required: %d" % (out_bits, req_bits))
        raise ValueError("bit size in the output type must be a " +
                         "multiple of the num bits")
    vals_per_int = out_bits / req_bits
    # packed_size is the number of required elements in the output array
    packed_size = int(np.ceil(float(arr.size) * req_bits / out_bits))
    output = np.zeros(shape=(packed_size,), dtype=np.uint8)
    # iterate over each input element
    for i, x in enumerate(arr.flatten()):
        # determine which index
        ind = int(np.floor(i / vals_per_int))
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
    data = np.array([x for x in iter_vals()], dtype=dtype)
    return data.reshape(shape)


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
        # add a pad of 0.1% on either side of the divs
        epsilon = np.maximum(1e-3 * (upper - lower), 1e-6)
        # n is the number of 'levels' that can be represented'
        divs = np.linspace(lower - epsilon,
                           upper + epsilon,
                           n).astype(np.float)
    else:
        bits = bits or np.ceil(np.log2(divs.size))
    if int(np.log2(bits)) != np.log2(bits):
        raise ValueError("bits must be a power of two")
    # it doesn't make sense to store anything larger than this using tinylib
    assert bits <= 4
    # make sure that bin dividers are monotonically increasing and that
    # bins are not of size zero.
    assert np.all(np.diff(divs) > 0.)
    n = np.power(2., bits)
    if not wrap:
        assert divs.size - 1 <= n
        # for each element of the array, count how many divs are less
        # than the element note that a zero after shifting means that
        # the value was less than all div and a value of n means it was
        # larger than the largest div.
        bins = np.digitize(arr.reshape(-1), divs)
        if np.any(bins == 0):
            warnings.warn("Some values were too small to encode!")
        if np.any(bins == divs.size):
            warnings.warn("Some values were too large to encode!")
    else:
        # With wrapping it's ok to have values that fall outside
        # the divs, in that case the resulting decoded value will
        # take on a wrapped value.
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
    # we need to shift the bins.  We want zeros to get 'wrapped' to zero
    # and ones to fall in the first (and only) bin.
    return tiny_array(arr, bits=1, divs=np.array([0., 1.]) + 1e-6,
                      mask=mask, wrap=True)


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
    if wrap_val is not None:
        upper = bins
        lower = (bins-1) % len(divs)
        # any bins that are zero mean they fell on the wrap value
        # all others are as before
        averages = np.where(bins > 0,
                            0.5 * (divs[lower] + divs[upper]),
                            wrap_val)
    else:
        upper_bins = bins
        lower_bins = np.maximum(upper_bins - 1, 0)
        upper_bins = np.minimum(upper_bins, ndivs - 1)
        averages = 0.5 * (divs[lower_bins] + divs[upper_bins])
        # if any values fell outside the divs we set them to be
        # slightly larger than div boundaries. Ideally to avoid
        # this the divs should be designed such that no values
        # fall beyond their range.
        epsilon = np.maximum(0.5 * np.min(np.diff(divs)), 1e-6)
        if np.any(bins == 0):
            averages[bins == 0] = np.min(divs) - epsilon
            warnings.warn("Some values were too small to decode!")
        if np.any(bins == ndivs):
            averages[bins == ndivs] = np.max(divs) + epsilon
            warnings.warn("Some values were too large to decode!")

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
    """
    Creates a small array.  This is done by rounding to a least
    significant digit and then zlib compressing te array.

    See Also: expand_small_array
    """
    assert np.all(np.isfinite(arr))
    data = np.round(arr * np.power(10, least_significant_digit))
    data = data.astype(arr.dtype)
    return {'packed_array': zlib.compress(data.tostring(), 9),
            'dtype': arr.dtype,
            'least_significant_digit': least_significant_digit}


def expand_small_array(packed_array, dtype, least_significant_digit):
    """
    Takes a the output from small_array and reconstructs a the original
    (rounded) array
    """
    arr = np.fromstring(zlib.decompress(packed_array), dtype=dtype)
    arr = arr / np.power(10, least_significant_digit)
    return arr


def small_time(time_var):
    """
    This packs the time variable by taking advantage of the fact that
    time is monotonically increasing.  It first converts the starting
    time to ordinal + seconds, then stores the incremental differences
    using small_array().
    """
    if np.issubdtype(time_var.dtype, np.datetime64):
        time_var = xra.conventions.encode_cf_variable(time_var)
    else:
        # if the time_var does not have datetime values we want
        # to make sure it is an encoded datetime.
        assert np.issubdtype(time_var.dtype, np.int)
        assert 'since' in time_var.attrs.get('units', '')

    assert time_var.attrs['units'].lower().startswith('hour')
    origin = xra.conventions.decode_cf_datetime([0],
                                                 time_var.attrs['units'])[0]
    origin = pd.to_datetime(origin)
    # diffs should be integer valued
    diffs = np.diff(np.concatenate([[0], time_var.values[:]]))
    diffs = np.round(diffs, 6)
    np.testing.assert_array_equal(diffs.astype('int32'), diffs)
    diffs = diffs.astype(np.int32)
    # we use from_ordinal since thats what expansion will use.
    # this way if the roundtrip to_ordinal doesn't work we'll
    # still have correct times.
    fromordinal = datetime.datetime.fromordinal(origin.toordinal())
    seconds = np.int32(datetime.timedelta.total_seconds(origin - fromordinal))
    augmented = np.concatenate([[origin.toordinal(), seconds],
                                diffs]).astype(np.int32)
    return small_array(augmented, least_significant_digit=0)


def expand_small_time(packed_array):
    """
    Expands a small_time encoded time array.
    """
    augmented = expand_small_array(packed_array, dtype=np.int32,
                                   least_significant_digit=0)
    origin = datetime.datetime.fromordinal(augmented[0])
    # there is a very strange bug that happens here if
    # we don't convert augmented to an int, timedelta must
    # make some assumptions on the dtype.
    origin += datetime.timedelta(seconds=int(augmented[1]))
    times = np.cumsum(augmented[2:])
    units = origin.strftime('hours since %Y-%m-%d %H:%M:%S')
    return times, units


def small_trival_variable(var):
    """
    All that matters when storing trival coordinates, coordinates equivalent
    to np.arange(n), is the length n.
    """
    # make sure var isn't so large that it doesn't fit in two bytes
    assert var.size == np.array(var.size, np.int16)
    assert np.all(var.values == np.arange(var.size))
    # return a string encoded byte
    return np.array(var.size, np.int16).tostring()

def expand_trival_variable(packed_array, ):
    """
    Expands a trival variable.
    """
    # determine the size of the array
    n = np.asscalar(np.fromstring(packed_array, dtype=np.int16))
    return np.arange(n)
