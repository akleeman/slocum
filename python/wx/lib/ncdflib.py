"""Module with constants and utility functions for reading netCDF-3 files

http://www.unidata.ucar.edu/software/netcdf/docs/netcdf/Classic-Format-Spec.html
"""

import coards
import pickle
import struct
import unicodedata
import numpy as np

from wb.lib import datelib
import wb.datetime.ttypes as tdatetime

MAGIC         = 'CDF'
_31BYTE       = '\x01' # version_byte = 1
_64BYTE       = '\x02' # version_byte = 2
NULL          = '\x00'
ZERO          = '\x00\x00\x00\x00'
ABSENT        = ZERO * 2
NC_DIMENSION  = '\x00\x00\x00\x0a'
NC_VARIABLE   = '\x00\x00\x00\x0b'
NC_ATTRIBUTE  = '\x00\x00\x00\x0c'
STREAMING     = '\xff\xff\xff\xff'
NC_BYTE       = '\x00\x00\x00\x01'
NC_CHAR       = '\x00\x00\x00\x02'
NC_SHORT      = '\x00\x00\x00\x03'
# netCDF-3 only supports 32-bit integers
NC_INT        = '\x00\x00\x00\x04'
NC_FLOAT      = '\x00\x00\x00\x05'
NC_DOUBLE     = '\x00\x00\x00\x06'
FILL_BYTE     = '\x81'
FILL_CHAR     = '\x00'
FILL_SHORT    = '\x80\x01'
FILL_INT      = '\x80\x00\x00\x01'
FILL_FLOAT    = '\x7c\xf0\x00\x00'
FILL_DOUBLE   = '\x47\x9e\x00\x00\x00\x00'
NC_BYTE_ORDER = '>' # netCDF-3 uses network byte order
NC_WORD_LEN   = 4 # netCDF-3 has 4-byte alignment (with some exceptions)

# Map between netCDF type and numpy dtype and vice versa. Due to a bug
# in the __hash__() method of numpy dtype objects (fixed in development
# release of numpy), we need to explicitly match byteorder for dict
# lookups to succeed. Here we normalize to native byte order.
#
# NC_CHAR is a special case because netCDF represents strings as
# character arrays. When NC_CHAR is encountered as the type of an
# attribute value, this TYPEMAP is not consulted and the data is read
# as a string. However, when NC_CHAR is encountered as the type of a
# variable, then the data is read is a numpy array of 1-char elements
# (equivalently, length-1 raw "strings"). There is no support for numpy
# arrays of multi-character strings.
TYPEMAP = {
        NC_BYTE: np.dtype('int8').newbyteorder('='),
        NC_CHAR: np.dtype('c').newbyteorder('='),
        NC_SHORT: np.dtype('int16').newbyteorder('='),
        NC_INT: np.dtype('int32').newbyteorder('='),
        NC_FLOAT: np.dtype('float32').newbyteorder('='),
        NC_DOUBLE: np.dtype('float64').newbyteorder('='),
        }
for k in TYPEMAP.keys():
    TYPEMAP[TYPEMAP[k]] = k

FILLMAP = {
        # values used as wb conventions for fill values; TODO move this map
        # to a conventions file
        # -999 commonly used as a missing weather value (impossible measurement)
        np.dtype('int8'): -1,
        np.dtype('c'): "\x00",
        np.dtype('int16'): -999,
        np.dtype('int32'): -999,
        np.dtype('int64'): -999,
        np.dtype('float32'): np.nan,
        np.dtype('float64'): np.nan,
        np.dtype('bool'): False
        }

# Special characters that are permitted in netCDF names except in the
# 0th position of the string
_specialchars = '_.@+- !"#$%&\()*,:;<=>?[]^`{|}~'

# The following are reserved names in CDL and may not be used as names of
# variables, dimension, attributes
_reserved_names = set([
        'byte',
        'char',
        'short',
        'ushort',
        'int',
        'uint',
        'int64',
        'uint64',
        'float'
        'real',
        'double',
        'bool',
        'string',
        ])

def unpickle_variable(var):
    """
    Unpickles an array of objects stored using core.Data.create_pickled_variable

    In order to store an array of objects in a netcdf file it needs to be
    serialized using pickle and saved as a matrix of characters where each
    row corresponds to the pickled string (via pickle.dumps).

    Parameters
    """
    if var.data.ndim != 2:
        raise pickle.UnpicklingError("var.data is not a matrix" +
                    " and in turn is likely not a pickled 1-d array")
    # if the data doesn't hold a pickled variable, pickle will raise an exception
    return [pickle.loads(x.tostring().rstrip('\x00')) for x in var.data]

def round_num_bytes(n):
    """Given positive 32-bit integer n, return the smallest int greater than or
    equal to n that is evenly divisible by the block size NC_WORD_LEN"""
    if not isinstance(n, int):
        return TypeError, "n must be int"
    if not (n >= 0):
        raise ValueError, "n must be a non-negative integer"
    return (n + ((NC_WORD_LEN - (n % NC_WORD_LEN)) % NC_WORD_LEN))

def _isalnumMUTF8(c):
    """Return True if the given UTF-8 encoded character is alphanumeric
    or multibyte.

    Input is not checked!
    """
    return (c.isalnum() or (len(c.encode('utf-8')) > 1))

def is_valid_name(s):
    """Test whether an object can be validly converted to a netCDF
    dimension, variable or attribute name

    Earlier versions of the netCDF C-library reference implementation
    enforced a more restricted set of characters in creating new names,
    but permitted reading names containing arbitrary bytes. This
    specification extends the permitted characters in names to include
    multi-byte UTF-8 encoded Unicode and additional printing characters
    from the US-ASCII alphabet. The first character of a name must be
    alphanumeric, a multi-byte UTF-8 character, or '_' (reserved for
    special names with meaning to implementations, such as the
    "_FillValue" attribute). Subsequent characters may also include
    printing special characters, except for '/' which is not allowed in
    names. Names that have trailing space characters are also not
    permitted.
    """
    if not isinstance(s, basestring):
        return False
    if not isinstance(s, unicode):
        s = unicode(s, 'utf-8')
    num_bytes = len(s.encode('utf-8'))
    return ((unicodedata.normalize('NFC', s) == s) and
            (s not in _reserved_names) and
            (num_bytes >= 0) and
            ('/' not in s) and
            (s[-1] != ' ') and
            (_isalnumMUTF8(s[0]) or (s[0] == '_')) and
            all(map(lambda c: _isalnumMUTF8(c) or (c in _specialchars), s))
            )

def unpack_string(nelems, bytestring):
    if not isinstance(nelems, int):
        raise TypeError, "nelems must be int"
    if not (nelems >= 0):
        raise ValueError, "nelems must be a non-negative integer"
    if not isinstance(bytestring, str):
        raise TypeError, "bytestring must be a raw string"
    if (len(bytestring) != round_num_bytes(nelems)) or\
            not all(NULL == b for b in bytestring[nelems:]):
        raise ValueError, ("bytestring must be padded to the nearest " +
                "4-byte boundary with trailing null bytes")
    s = unicodedata.normalize('NFC', bytestring[:nelems].decode('utf-8'))
    # preferentially return str when possible
    try:
        return str(s)
    except UnicodeError:
        return s

def read_string(f):
    """Convert a sequence of bytes from a netCDF file that represent a
      dimension/variable/attribute name to a Python string."""
    nelems = read_int(f)
    # Read up to the next 4-byte boundary
    bytestring = f.read(round_num_bytes(nelems))
    return unpack_string(nelems, bytestring)

def pack_string(s):
    if not isinstance(s, unicode):
        s = unicode(s, 'utf-8')
    s = unicodedata.normalize('NFC', s)
    bytestring = s.encode('utf-8')
    # Get the number of bytes
    nelems = len(bytestring)
    # Pad to next 4-byte boundary
    bytestring += NULL * (round_num_bytes(nelems) - nelems)
    return (nelems, bytestring)

def write_string(f, s):
    """Convert a Python str or unicode to a sequence of bytes that can
    be understood by netCDF and write to file."""
    (nelems, bytestring) = pack_string(s)
    write_int(f, nelems)
    f.write(bytestring)

def unpack_int(bytestring):
    if not isinstance(bytestring, str):
        raise TypeError, "bytestring must be a raw string"
    if len(bytestring) != 4:
        raise ValueError, "bytestring must be a 4-byte sequence"
    return struct.unpack(NC_BYTE_ORDER + '1i', bytestring)[0]

def read_int(f):
    """Convert a 4-byte sequence from a netCDF file to a 32-bit int"""
    bytestring = f.read(4)
    return unpack_int(bytestring)

def pack_int(i):
    if not isinstance(i, int):
        raise TypeError, "i must be an int"
    try:
        # size of Python int is platform-dependent, so we need to check that
        # this int really fits in a 4-byte sequence
        bytestring = struct.pack(NC_BYTE_ORDER + '1i', i)
        assert i == unpack_int(bytestring)
    except:
        raise ValueError, "Can not encode as a netCDF 32-bit integer"
    return bytestring

def write_int(f, i):
    """Convert a 32-bit int to a 4-byte sequence that is understood by
    netCDF and write to file"""
    bytestring = pack_int(i)
    f.write(bytestring)

def unpack_int64(bytestring):
    if not isinstance(bytestring, str):
        raise TypeError, "bytestring must be a raw string"
    if len(bytestring) != 8:
        raise ValueError, "bytestring must be a 8-byte sequence"
    return struct.unpack(NC_BYTE_ORDER + '1q', bytestring)[0]

def read_int64(f):
    """Convert a 8-byte sequence from a netCDF file to a 64-bit int"""
    bytestring = f.read(8)
    return unpack_int64(bytestring)

def pack_int64(i):
    try:
        # size of Python int is platform-dependent, so we need to check that
        # this int really fits in a 8-byte sequence
        bytestring = struct.pack(NC_BYTE_ORDER + '1q', i)
        assert i == unpack_int64(bytestring)
    except:
        raise ValueError, "Can not encode as a netCDF 64-bit integer"
    return bytestring

def write_int64(f, i):
    """Convert a 64-bit int to a 8-byte sequence that is understood by
    netCDF and write to file"""
    bytestring = pack_int64(i)
    f.write(bytestring)

def from_udunits(units, index=0):
    return coards.from_udunits(index, units).replace(tzinfo=None)

def char_stack(arr, axis=-1):
    """Reduce a character array to an array of fixed-width strings

    Given an N-dimensional numpy array whose elements are 1-character
    (1-byte) strings, return a (N-1)-dimensional array whose elements
    are fixed-width concatenations of characters along a specified
    axis. This function is useful for converting a netCDF-style
    character array to a more user-friendly string array.

    Parameters
    ----------
    arr : array
        A numpy array whose data type is 1-element string (i.e.,
        arr.dtype.char == 'c' and arr.dtype.itemsize == 1).
    axis : int, optional
        The axis along which the concatenation should be done. If no
        axis argument is provided, the concatenation is done along the
        last dimension of arr.

    Returns
    -------
    out : array
        An array with one less dimension than arr whose dtype is string
        of length arr.shape[axis].
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError, "arr must be a numpy ndarray"
    if arr.dtype != np.dtype('S1'):
        raise TypeError, "dtype of arr must be numpy.dtype('|S1')"
    if not arr.shape:
        raise ValueError, "arr must not be a scalar (degenerate) array"
    out_shape = list(arr.shape)
    if axis < 0:
        axis = arr.ndim + axis
    strlen = out_shape.pop(axis)
    if strlen < 1:
        raise ValueError, "arr has zero length along the specified axis"
    f = lambda x: ''.join(x.tostring())
    out = np.apply_along_axis(f, axis=axis, arr=arr)
    if out.dtype != np.dtype('|S%d' % (strlen)):
        raise RuntimeError, "Character array concatenation failed"
    return out

def char_unstack(arr):
    """Expand an array of fixed-width strings to a character array

    Given an N-dimensional numpy array whose elements are strings of
    length K, return a (N+1)-dimensional array whose last dimension is
    of length K and whose elements are single characters. This is
    useful for converting a numpy array of strings into a netCDF-style
    character array.

    Parameters
    ----------
    arr : array
        A numpy array of strings (i.e., arr.dtype.char == 'c')

    Returns
    -------
    out : array
        An array with one more dimension than arr whose dtype is 'S1'
        and whose length along the last dimension is arr.dtype.itemsize
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError, "arr must be a numpy ndarray"
    if not arr.dtype.char in ['c', 'S']:
        raise TypeError, "arr must be an array of strings"
    str_len = arr.dtype.itemsize
    out_shape = list(arr.shape) + [str_len]
    out = np.array(list(arr.tostring())).reshape(out_shape)
    return out
