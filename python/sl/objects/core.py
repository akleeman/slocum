"""Core objects to represent data and models

The Data class and Variable class in this module are heavily reworked versions
of code from the PuPyNeRe netCDF reader:

    http://dealmeida.net/hg/pupynere/file/b95f566d84af/pupynere.py

pupynere is released under the MIT license for unlimited use.
"""

import csv
import copy
import logging
import itertools
import numpy as np
from os import SEEK_END
import cPickle as pickle
from cPickle import HIGHEST_PROTOCOL
from operator import mul, or_
from cStringIO import StringIO

from sl.lib import ncdflib, datelib
from sl.lib.collections import OrderedDict, FrozenOrderedDict
import sl.objects.conventions as conv

ENSURE_VALID = True

def _prettyprint(x, numchars):
    """Given an object x, call x.__str__() and format the returned
    string so that it is numchars long, padding with trailing spaces or
    truncating with ellipses as necessary"""
    s = str(x).rstrip(ncdflib.NULL)
    if len(s) <= numchars:
        return s + ' ' * (numchars - len(s))
    else:
        return s[:(numchars - 3)] + '...'


class AttributesDict(OrderedDict):
    """A subclass of OrderedDict whose __setitem__ method automatically
    checks and converts values to be valid netCDF attributes
    """
    def __init__(self, *args, **kwds):
        OrderedDict.__init__(self, *args, **kwds)

    def __setitem__(self, key, value):
        if ENSURE_VALID and not ncdflib.is_valid_name(key):
                raise ValueError("Not a valid attribute name")
        # Strings get special handling because netCDF treats them as
        # character arrays. Everything else gets coerced to a numpy
        # vector. netCDF treats scalars as 1-element vectors. Arrays of
        # non-numeric type are not allowed.
        if isinstance(value, basestring):
            try:
                ncdflib.pack_string(value)
            except:
                raise ValueError("Not a valid value for a netCDF attribute")
        else:
            try:
                value = ncdflib.coerce_type(np.atleast_1d(np.asarray(value)))
            except:
                import pdb; pdb.set_trace()
                raise ValueError("Not a valid value for a netCDF attribute")
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be vectors " +
                        "(1-dimensional)")
            value = ncdflib.coerce_type(value)
            if str(value.dtype) not in ncdflib.TYPEMAP:
                # A plain string attribute is okay, but an array of
                # string objects is not okay!
                raise ValueError("Can not convert to a valid netCDF type")
        OrderedDict.__setitem__(self, key, value)

    def copy(self):
        """The copy method of the superclass simply calls the constructor,
        which in turn calls the update method, which in turns calls
        __setitem__. This subclass implementation bypasses the expensive
        validation in __setitem__ for a substantial speedup."""
        obj = self.__class__()
        for (attr, value) in self.iteritems():
            OrderedDict.__setitem__(obj, attr, copy.copy(value))
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        return self.copy()

    def update(self, other=None, **kwargs):
        """Set multiple attributes with a mapping object or an iterable of
        key/value pairs"""
        # Capture arguments in an OrderedDict
        args_dict = OrderedDict(other, **kwargs)
        try:
            # Attempt __setitem__
            for (attr, value) in args_dict.iteritems():
                self.__setitem__(attr, value)
        except:
            # A plain string attribute is okay, but an array of
            # string objects is not okay!
            raise ValueError("Can not convert to a valid netCDF type")
            # Clean up so that we don't end up in a partial state
            for (attr, value) in args_dict.iteritems():
                if self.__contains__(attr):
                    self.__delitem__(attr)
            # Re-raise
            raise

    def __eq__(self, other):
        if not set(self.keys()) == set(other.keys()):
            return False
        for (key, value) in self.iteritems():
            if value.__class__ != other[key].__class__:
                return False
            if isinstance(value, basestring):
                if value != other[key]:
                    return False
            else:
                if value.tostring() != other[key].tostring():
                    return False
        return True


class Data(object):
    """A class to organize numpy array data in a self-describing,
    netCDF-compatible format

    Core attributes:

    Variables are accessed by dictionary-like behavior.
    """

    def __init__(self, ncdf=None):
        """Initializes a data object with contents read from specified
        netCDF file.

        Parameters
        ----------
        ncdf : str or file_like, optional
            If None (default), then an empty data object is created.
            If str, ncdf must be a sequence of raw bytes that
            constitute a netCDF-3 file. Otherwise, ncdf must be an
            object with a read() method whose contents are a
            netCDF-3 file.
        """
        # The __setattr__ method of the base object class is used to
        # bypass the overloaded __setattr__ method
        object.__setattr__(self, '_version_byte', ncdflib._64BYTE)
        # self.attributes is AttributesDict
        object.__setattr__(self, 'attributes', AttributesDict())
        # self.variables and self.dimensions are FrozenOrderedDict.
        # These dictionaries should not be directly modified by the
        # user; use the create_dimension and create_variable methods.
        object.__setattr__(self, 'dimensions', FrozenOrderedDict())
        object.__setattr__(self, 'record_dimension', None)
        object.__setattr__(self, 'variables', FrozenOrderedDict())
        if ncdf is not None:
            if isinstance(ncdf, str):
                self._loads(ncdf)
            else:
                self._load(ncdf)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr__ method to bypass."""
        raise AttributeError("__setattr__ is disabled")

    def __delattr__(self, attr):
        raise AttributeError("__delattr__is disabled")

    def __getitem__(self, name):
        """obj[varname] returns the variable"""
        if name not in self.variables:
            raise KeyError("'%s' variable does not exist" % (str(name)))
        return self.variables[name]

    def __contains__(self, v):
        return v in self.variables

    def iterkeys(self):
        """keys() and other dict-like behavior is implemented for
        consistency with overloaded __getitem__"""
        return self.variables.__iter__()

    def itervalues(self):
        for v in self.variables.itervalues():
            yield v

    def iteritems(self):
        for k in self.variables:
            yield (k, self.variables[k])

    def keys(self):
        return self.variables.keys()

    def items(self):
        return self.variables.items()

    def values(self):
        return self.variables.values()

    def __len__(self):
        """For convenience, __len__ is overloaded to return the number
        of records along the unlimited dimension. If there are no
        record variables, then the number of records is undefined (i.e.
        without data, we have no idea how long the unlimited
        dimension). Python requires len() to return integer, so we
        can't return None here. Instead we raise an exception. Thanks
        to bzimmer for the idea."""
        if (self.record_dimension is None) or \
            (self.dimensions[self.record_dimension] is None):
            raise TypeError("Overloaded __len__ method only works for " +
                    "objects with a record dimension and at least one " +
                    "record variable")
        else:
            return self.dimensions[self.record_dimension]

    def __iter__(self):
        """If there is a record dimension, __iter__ is overloaded in a
        manner analogoues to __len__"""
        if self.record_dimension is None:
            import pdb; pdb.set_trace()
            raise TypeError("Overloaded __iter__ method only works for " +
                    "objects with a record dimension")
        else:
            return self.iterator(dim=self.record_dimension)

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        # Create a new Data instance
        obj = self.__class__()
        # Copy dimensions
        object.__setattr__(obj, 'dimensions', self.dimensions)
        # Copy variables
        object.__setattr__(obj, 'variables', self.variables)
        # Copy attributes
        object.__setattr__(obj, 'attributes', self.attributes)
        # Copy record_dimension
        object.__setattr__(obj, 'record_dimension', self.record_dimension)
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        # Create a new Data instance
        obj = self.__class__()
        # Copy dimensions
        object.__setattr__(obj, 'dimensions', self.dimensions.copy())
        # Copy variables
        object.__setattr__(obj, 'variables', copy.deepcopy(self.variables))
        # Copy attributes
        object.__setattr__(obj, 'attributes', self.attributes.copy())
        # Copy record_dimension
        object.__setattr__(obj, 'record_dimension',
                           copy.deepcopy(self.record_dimension))
        return obj

    def __eq__(self, other):
        if not isinstance(other, Data):
            return False
        if dict(self.dimensions) != dict(other.dimensions):
            return False
        if not dict(self.variables) == dict(other.variables):
            return False
        if not self.attributes == other.attributes:
            return False
        if not self.record_dimension == other.record_dimension:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """Human-readable summary in 70ish columns"""
        lines = []
        lines.append('\n' + self.__class__.__name__)
        # Print dimensions
        lines.append('\ndimensions:')
        if self.dimensions:
            lines.append('  %s| %s' %
                    (_prettyprint('name', 16),
                    _prettyprint('length', 8)))
            lines.append(' =%s==%s' %
                    ('=' * 16, '=' * 8))
            for (name, length) in self.dimensions.iteritems():
                if name == self.record_dimension:
                    lines.append(' *%s| %s' %
                            (_prettyprint(name, 16),
                            _prettyprint(length, 8)))
                else:
                    lines.append('  %s| %s' %
                            (_prettyprint(name, 16),
                            _prettyprint(length, 8)))
        else:
            lines.append('  None')
        # Print variables
        lines.append('\nvariables:')
        if self.variables:
            lines.append('  %s| %s| %s| %s' %
                    (_prettyprint('name', 16),
                    _prettyprint('dtype', 8),
                    _prettyprint('shape', 16),
                    _prettyprint('dimensions', 22)))
            lines.append(' =%s==%s==%s==%s' %
                    ('=' * 16, '=' * 8, '=' * 16, '=' * 22))
            for (name, var) in self.variables.iteritems():
                if self.record_dimension in var.dimensions:
                    lines.append(' *%s| %s| %s| %s' %
                            (_prettyprint(name, 16),
                            _prettyprint(var.dtype, 8),
                            _prettyprint(var.shape, 16),
                            _prettyprint(var.dimensions, 22)))
                else:
                    lines.append('  %s| %s| %s| %s' %
                            (_prettyprint(name, 16),
                            _prettyprint(var.dtype, 8),
                            _prettyprint(var.shape, 16),
                            _prettyprint(var.dimensions, 22)))
        else:
            lines.append('  None')
        # Print attributes
        lines.append('\nattributes:')
        if self.attributes:
            lines.append('  %s| %s' %
                    (_prettyprint('name', 16),
                    _prettyprint('value', 50)))
            lines.append(' =%s==%s' %
                    ('=' * 16, '=' * 50))
            for (attr, val) in self.attributes.iteritems():
                lines.append('  %s| %s' %
                        (_prettyprint(attr, 16),
                        _prettyprint(val, 50)))
        else:
            lines.append('  None')
        lines.append('')
        return '\n'.join(lines)

    @property
    def coordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return FrozenOrderedDict([(dim, length)
                for (dim, length) in self.dimensions.iteritems()
                if (dim in self.variables) and
                (self.variables[dim].data.ndim == 1) and
                (self.variables[dim].dimensions == (dim,))
                ])

    @property
    def noncoordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return FrozenOrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.coordinates])

    def _load(self, f):
        """Populate an initialized, empty object from data read from
        netCDF file. This method is non-public because we don't want to
        overwrite existing data by calling it with a non-empty
        object."""

        # Check magic bytes
        magic = f.read(len(ncdflib.MAGIC))
        if magic != ncdflib.MAGIC:
            raise ValueError("Not a valid netCDF-3 file")
        # Check version_byte
        _version_byte = f.read(len(ncdflib.FILL_BYTE))
        if _version_byte not in [ncdflib._31BYTE, ncdflib._64BYTE]:
            raise ValueError("netCDF file header does not have a " +
                    "recognized version_byte")
        object.__setattr__(self, '_version_byte', _version_byte)
        numrecs = ncdflib.read_int(f)
        # Read dimensions and add them. The create_dimension method handles
        # the error-checking for us.
        dimensions = self._read_dim_array(f)
        for (name, length) in dimensions.iteritems():
            self.create_dimension(name, length)
        # Read global attributes
        attributes = self._read_att_array(f)
        self.attributes.update(attributes)
        # Read variables
        self._read_variables(f)
        # If there is a record dimension, then we need to verify that the value
        # numrecs reported in the file header matches the actual number of
        # records (unless the header value is STREAMING, which means that we
        # have to read the records to know how many there are)
        if (not (self.record_dimension is not None) ^
                (numrecs == ncdflib.unpack_int(ncdflib.ZERO))) or\
                ((self.record_dimension is not None) and
                (numrecs != ncdflib.unpack_int(ncdflib.STREAMING)) and
                (numrecs != self.dimensions[self.record_dimension])):
            raise ValueError("netCDF file has a numrecs header field that " +
                    "is inconsistent with dimensions")

    def _loads(self, s):
        """Read data from netCDF format as a string of bytes"""
        buf = StringIO(s)
        self._load(buf)
        buf.close()

    def dump(self, ncdf):
        """Write current data contents to file in netCDF format

        Parameters
        ----------
        ncdf : file-like object
            Must have write method
        """
        # Write magic header bytes
        ncdf.write(ncdflib.MAGIC)
        # Write version byte
        ncdf.write(self._version_byte)
        # Write number of records.
        if self.record_dimension is None:
            ncdf.write(ncdflib.ZERO)
        else:
            ncdflib.write_int(ncdf,
                self.dimensions[self.record_dimension] or 0)
        # Write dimensions
        self._write_dim_array(ncdf)
        # Write global attributes
        self._write_att_array(ncdf, self.attributes)
        # Write variables
        self._write_variables(ncdf)

    def dumps(self):
        """Return current data contents in netCDF format as a string of
        bytes"""
        buf = StringIO()
        self.dump(buf)
        s = buf.getvalue()
        buf.close()
        return s

    def _read_dim_array(self, f):
        """Read dim_array from a netCDF file and return as a
        FrozenOrderedDict"""
        dim_header = f.read(len(ncdflib.NC_DIMENSION))
        num_dim = ncdflib.read_int(f)
        if (dim_header not in [ncdflib.NC_DIMENSION, ncdflib.ZERO]) or\
                ((dim_header == ncdflib.ZERO) and (num_dim != 0)) or\
                (num_dim < 0):
            raise ValueError("dimensions header is invalid")
        dimensions = FrozenOrderedDict()
        for dim in xrange(num_dim):
            name = ncdflib.read_string(f)
            length = ncdflib.read_int(f)
            # This length may not be the true length of the dimension
            # if this dimension is the record dimension. netCDF
            # indicates the record dimension with zero length even when
            # the number of records is non-zero. We replace this zero
            # with None to indicate that this slot is reserved for the
            # number of records.
            if not length:
                length = None
            # Use the __setitem__ method of the superclass
            super(type(dimensions), dimensions).__setitem__(name, length)
        return dimensions

    def _write_dim_array(self, f):
        """Write dim_array of a netCDF file"""
        if self.dimensions:
            f.write(ncdflib.NC_DIMENSION)
            ncdflib.write_int(f, len(self.dimensions))
            for (name, length) in self.dimensions.iteritems():
                ncdflib.write_string(f, name)
                if name == self.record_dimension:
                    # netCDF indicates the record dimension with zero
                    # length even when the number of records is non-zero
                    f.write(ncdflib.ZERO)
                else:
                    ncdflib.write_int(f, length)
        else:
            f.write(ncdflib.ABSENT)

    def _read_att_array(self, f):
        """Read att_array from a netCDF file and return as AttributesDict"""
        attr_header = f.read(len(ncdflib.NC_ATTRIBUTE))
        num_attr = ncdflib.read_int(f)
        if (attr_header not in [ncdflib.NC_ATTRIBUTE, ncdflib.ZERO]) or\
                ((attr_header == ncdflib.ZERO) and (num_attr != 0)) or\
                (num_attr < 0):
            raise ValueError, "atttributes header is invalid"
        attributes = AttributesDict()
        for attr in xrange(num_attr):
            name = ncdflib.read_string(f)
            # The next 4 bytes tell us the data type of this attribute
            nc_type = f.read(4)
            if nc_type == ncdflib.NC_CHAR:
                # strings get special treatment
                value = ncdflib.read_string(f)
            else:
                # Look up corresponding numpy dtype
                dtype = np.dtype(ncdflib.TYPEMAP[nc_type])
                # Read values and convert to numpy vector. Bytes are read
                # with network endianness, and then the resulting numpy
                # array is converted to native endianness. This is
                # expensive but worthwhile because Cython only supports native
                # endianness and because it's difficult to keep track of
                # non-native byte order.
                nelems = ncdflib.read_int(f)
                num_bytes = nelems * dtype.itemsize
                value = np.fromstring(f.read(num_bytes),
                        dtype=dtype.newbyteorder(ncdflib.NC_BYTE_ORDER))
                if value.dtype.byteorder != '=':
                    value = value.byteswap().newbyteorder('=')
                # If necessary, continue reading past the padding bytes
                f.read(ncdflib.round_num_bytes(num_bytes) - num_bytes)
            attributes[name] = value
        return attributes

    def _write_att_array(self, f, attributes):
        """Write att_array of a netCDF file"""
        if attributes:
            f.write(ncdflib.NC_ATTRIBUTE)
            ncdflib.write_int(f, len(attributes))
            for (name, value) in attributes.iteritems():
                ncdflib.write_string(f, name)
                if not (isinstance(value, basestring) or
                        isinstance(value, np.ndarray)):
                    raise ValueError("bad attribute type: type(%s) = %s" %
                                     (name, type(value)))
                if isinstance(value, basestring):
                    f.write(ncdflib.NC_CHAR)
                    ncdflib.write_string(f, value)
                elif isinstance(value, np.ndarray):
                    if value.ndim != 1:
                        raise ValueError(
                                "numpy array attributes must be 1-dimensional")
                    f.write(ncdflib.TYPEMAP[str(value.dtype)])
                    ncdflib.write_int(f, value.size)
                    # Write in network byte order
                    if value.dtype.byteorder != ncdflib.NC_BYTE_ORDER:
                        bytestring = value.byteswap().tostring()
                    else:
                        bytestring = value.tostring()
                    f.write(bytestring)
                    # Write any necessary padding bytes
                    size_diff = ncdflib.round_num_bytes(len(bytestring)) - \
                            len(bytestring)
                    padding_bytes = ncdflib.NULL * size_diff
                    f.write(padding_bytes)
                else:
                    raise ValueError, ("attribute value must be either " +
                            "string or 1-dimensional numeric array")
        else:
            f.write(ncdflib.ABSENT)

    def _read_variables(self, f):
        """Read the var_array and data block of a netCDF file."""
        var_header = f.read(len(ncdflib.NC_VARIABLE))
        num_var = ncdflib.read_int(f)
        if (var_header not in [ncdflib.NC_VARIABLE, ncdflib.ZERO]) or\
                ((var_header == ncdflib.ZERO) and (num_var != 0)) or\
                (num_var < 0):
            raise ValueError, "dimensions header is invalid"
        # We need to keep track of the order in which variables appear
        # in var_metadata
        var_metadata = OrderedDict()
        for var in xrange(num_var):
            name = ncdflib.read_string(f)
            num_dims = ncdflib.read_int(f)
            dimids = [ncdflib.read_int(f) for i in xrange(num_dims)]
            attributes = self._read_att_array(f)
            nc_type = f.read(4)
            vsize = ncdflib.read_int(f)
            if self._version_byte == ncdflib._31BYTE:
                begin = ncdflib.read_int(f)
            elif self._version_byte == ncdflib._64BYTE:
                begin = ncdflib.read_int64(f)
            else:
                raise RuntimeError, "self._version_byte is not recognized"
            var_metadata[name] = (dimids, attributes, nc_type, vsize, begin)
        # We use a list to track record variables because their data are
        # interleaved and the number of records is not determined until the
        # file is scanned to EOF
        recs = list()
        for var in var_metadata:
            (dimids, attributes, nc_type, vsize, begin) = var_metadata[var]
            if dimids:
                (var_dims, var_shape) = zip(*[self.dimensions.items()[i]
                        for i in dimids])
            else:
                (var_dims, var_shape) = ((), ())
            if self.record_dimension in var_dims:
                if var_dims[0] != self.record_dimension:
                    raise ValueError(
                        "Record variables must have the record dimension " +
                        "as their 0th dimension")
                recs.append(var)
        # Determine the size of each record and the total number of records
        if any([k is None for k in self.dimensions.values()]):
            if recs:
                # For a record variable, vsize is the size of a single
                # record of this variable (includes padding bytes if there
                # is more than one record variable). recsize is sum of
                # vsize over all record variables
                recsize = sum([var_metadata[var][3] for var in recs])
                # The data block for record variables extends to the end of
                # the file. The 'begin' offsets for the record variables
                # are staggered through the first record.
                recbegin = min([var_metadata[var][4] for var in recs])
                f.seek(recbegin)
                f.seek(0, SEEK_END)
                recbytes = f.tell() - recbegin
                if recbytes % recsize != 0:
                    raise ValueError, "Record data is misaligned"
                numrecs = recbytes / recsize
                # netCDF stores record variables as having zero length
                # along the record dimension. Here we replace this sentinel
                # zero with the real number of records.
                super(type(self.dimensions), self.dimensions).__setitem__(
                    self.record_dimension, numrecs)
            else:
                super(type(self.dimensions), self.dimensions).__setitem__(
                    self.record_dimension, None)
        # The order in which variables are created is important for file
        # consistency. We need to create variables in exactly the same order as
        # they appear in the netCDF file so that the dump() method reproduces
        # the original order.
        for var in var_metadata:
            (dimids, attributes, nc_type, vsize, begin) = var_metadata[var]
            if dimids:
                (var_dims, var_shape) = zip(*[self.dimensions.items()[i]
                        for i in dimids])
            else:
                (var_dims, var_shape) = ((), ())
            # Convert var_shape to a list so that we can set elements.
            # This is necessary because record variables have None in
            # place of the true number of records.
            var_shape = list(var_shape)
            dtype = np.dtype(ncdflib.TYPEMAP[nc_type])
            if self.record_dimension in var_dims:
                if var_dims[0] != self.record_dimension:
                    raise ValueError("Not valid netCDF: the 0th dimension " +
                                     "dimension of each record variable " +
                                     "must be the unlimited dimension")
                if var_shape[0] != numrecs:
                    raise ValueError("Record variable shape does not match " +
                                     "number of records")
                slice_size = reduce(mul, var_shape[1:], 1) * dtype.itemsize
                if len(recs) > 1:
                    if vsize != ncdflib.round_num_bytes(slice_size):
                        raise ValueError, ("vsize does not match expected " +
                                "size given data type and dimension")
                    # Append bytes to a string buffer. The bytes for
                    # multiple record variables are block-interleaved
                    # in each record, and we want to pull out the
                    # desired bytes so that numpy.fromstring can read
                    # contiguous bytes.
                    stringbuf = StringIO()
                    for i in xrange(numrecs):
                        f.seek(begin + i * recsize)
                        stringbuf.write(f.read(slice_size))
                    data = np.fromstring(stringbuf.getvalue(),
                            dtype=dtype.newbyteorder(ncdflib.NC_BYTE_ORDER))
                else:
                    if vsize != slice_size:
                        raise ValueError, ("vsize does not match expected " +
                                "size given data type and dimension")
                    f.seek(begin)
                    data = np.fromstring(f.read(),
                            dtype=dtype.newbyteorder(ncdflib.NC_BYTE_ORDER))
            else:
                # Read the data for this non-record variable
                num_bytes = reduce(mul, var_shape, 1) * dtype.itemsize
                if vsize != ncdflib.round_num_bytes(num_bytes):
                    raise ValueError("data type and shape of variable are " +
                            "inconsistent with vsize specified in header")
                f.seek(begin)
                data = np.fromstring(f.read(num_bytes),
                        dtype=dtype.newbyteorder(ncdflib.NC_BYTE_ORDER))
                # If necessary, continue reading past the padding bytes
                f.read(vsize - num_bytes)
            # Reshape flat vector to desired shape
            data = data.reshape(var_shape)
            # Convert to native endianness
            if data.dtype.byteorder != '=':
                data = data.byteswap().newbyteorder('=')
            # Create new variable
            self.create_variable(var, dim=var_dims, data=data,
                                 attributes=attributes)

    def _write_variables(self, f):
        """Write the var_array and data block of a netCDF file."""
        if self.variables:
            # We need to know the total number of record variables.
            # This is importnat for byte alignment.
            num_rec_vars = sum([
                    self.record_dimension in self.variables[v].dimensions
                    for v in self.variables])
            # begin_field_offsets stores the file offset for the 'begin' field
            # of each variable header so that we can seek back and write this
            # value as we determine where the data is written to the
            # file. We could precompute the begin offsets (which would
            # enable write to file-like objects that have no seek or
            # tell methods), but this is tricky and I don't like
            # pointer arithmetic.
            begin_field_offsets = dict()
            f.write(ncdflib.NC_VARIABLE)
            ncdflib.write_int(f, len(self.variables))
            for (name, v) in self.variables.iteritems():
                ncdflib.write_string(f, name)
                ncdflib.write_int(f, v.ndim)
                for d in v.dimensions:
                    # dimid
                    ncdflib.write_int(f, self.dimensions.keys().index(d))
                self._write_att_array(f, v.attributes)
                # Write nc_type enum.
                f.write(ncdflib.TYPEMAP[str(v.dtype)])
                # Write vsize. This is different depending on whether the
                # variable is a record or non-record variable.
                if self.record_dimension in v.dimensions:
                    if num_rec_vars > 1:
                        # When there is more than one record variable,
                        # the vsize of each record variable is the size
                        # of a single record slice  + padding
                        vsize = ncdflib.round_num_bytes(
                                v.data.dtype.itemsize *
                                reduce(mul, v.data.shape[1:], 1))
                    else:
                        # When there is exactly one record variable,
                        # vsize is unpadded.
                        vsize = v.data.dtype.itemsize * \
                                reduce(mul, v.data.shape[1:], 1)
                else:
                    # For non-record variables, vsize is the size of the entire
                    # data block + padding
                    vsize = ncdflib.round_num_bytes(
                            v.data.dtype.itemsize * v.data.size)
                ncdflib.write_int(f, vsize)
                # Fill in a placeholder value here; we will need to replace the
                # values later once we have calculated the proper offsets
                begin_field_offsets[name] = f.tell()
                if self._version_byte == ncdflib._31BYTE:
                    ncdflib.write_int(f, 0)
                elif self._version_byte == ncdflib._64BYTE:
                    ncdflib.write_int64(f, 0)
                else:
                    raise RuntimeError("version_byte is not recognized")
            # Check to make sure we are where we think we are in the file. The
            # data block comes immediately after the 'begin' field of the last
            # variable in var_array
            if (self._version_byte == ncdflib._31BYTE) and (
                    max(begin_field_offsets.values()) != f.tell() - 4):
                raise RuntimeError("error in calculating file offset")
            if (self._version_byte == ncdflib._64BYTE) and (
                    max(begin_field_offsets.values()) != f.tell() - 8):
                raise RuntimeError("error in calculating file offset")
            # Now that we have written (almost all of) the var_array, we can
            # start writing the data block. As we write the data, we go back
            # and fill in the correct 'begin' values in var_array
            # The non-record variables are written first
            for (name, v) in self.variables.iteritems():
                if self.record_dimension in v.dimensions:
                    continue
                # Write the file offset for the beginning of this data block
                begin = f.tell()
                f.seek(begin_field_offsets[name])
                if self._version_byte == ncdflib._31BYTE:
                    try:
                        ncdflib.write_int(f, begin)
                    except:
                        import pdb; pdb.set_trace()
                elif self._version_byte == ncdflib._64BYTE:
                    ncdflib.write_int64(f, begin)
                else:
                    raise RuntimeError("version_byte is not recognized")
                # Seek back to begin and starting writing
                f.seek(begin)
                data = v.data
                if data.dtype.byteorder != ncdflib.NC_BYTE_ORDER:
                    bytestring = data.byteswap().tostring()
                else:
                    bytestring = data.tostring()
                f.write(bytestring)
                vsize = ncdflib.round_num_bytes(len(bytestring))
                f.write(ncdflib.NULL * (vsize - len(bytestring)))
            # The rest of the file contains data for the record variables.
            recs = [name for (name, v) in self.variables.iteritems()
                    if self.record_dimension in v.dimensions]
            # Start writing records
            begin = f.tell()# Remember this position in the file!
            if len(recs) > 1:
                # Write the file offset for the first record of each record
                # variable while simultaneously calculating the total record
                # size over all record variables and determining the
                # padding for each variable. recsize must be
                # incremented *after* writing the 'begin' offset.
                padding_bytes = dict()
                recsize = 0
                for var in recs:
                    dtype = self.variables[var].data.dtype.newbyteorder('=')
                    slice_size = dtype.itemsize * \
                            reduce(mul, self.variables[var].data.shape[1:], 1)
                    # Record for each variable is padded to next 4-byte
                    # boundary
                    vsize = ncdflib.round_num_bytes(slice_size)
                    # Determine padding for this variable
                    nc_type = ncdflib.TYPEMAP[str(dtype)]
                    padding_bytes[var] = ncdflib.NULL * (vsize - slice_size)
                    # Write the begin position *before* incrementing recsize
                    f.seek(begin_field_offsets[var])
                    if self._version_byte == ncdflib._31BYTE:
                        ncdflib.write_int(f, begin + recsize)
                    elif self._version_byte == ncdflib._64BYTE:
                        ncdflib.write_int64(f, begin + recsize)
                    else:
                        raise RuntimeError, "version_byte is not recognized"
                    recsize += vsize
                # Now that we are finished writing the 'begin' offsets,
                # we need to return the file pointer to the start of
                # the record data block.
                f.seek(begin)
                # Write each multi-variable record. Because all records
                # are the same size, we can repeatedly write and read
                # records with the same string buffer while maintaining
                # record alignment.
                stringbuf = StringIO()
                if (self.record_dimension is not None) and \
                    (self.dimensions[self.record_dimension] is not None):
                    for i in xrange(self.dimensions[self.record_dimension]):
                        stringbuf.seek(0)
                        for var in recs:
                            data = self.variables[var].data[i, ...]
                            if data.dtype.byteorder != ncdflib.NC_BYTE_ORDER:
                                bytestring = data.byteswap().tostring()
                            else:
                                bytestring = data.tostring()
                            stringbuf.write(bytestring)
                            # Add any padding bytes if necessary
                            stringbuf.write(padding_bytes[var])
                        # Once we're done appending to the string
                        # buffer, dump its contents to file. Each time
                        # we do this, one more record is appended to
                        # the end of the file
                        f.write(stringbuf.getvalue())
            elif len(recs) == 1:
                # Write the file offset for the beginning of the record data
                # block
                f.seek(begin_field_offsets[name])
                if self._version_byte == ncdflib._31BYTE:
                    ncdflib.write_int(f, begin)
                elif self._version_byte == ncdflib._64BYTE:
                    ncdflib.write_int64(f, begin)
                else:
                    raise RuntimeError("version_byte is not recognized")
                # Now that we are finished writing the 'begin' offsets,
                # we need to return the file pointer to the start of
                # the record data block.
                f.seek(begin)
                data = v.data
                if data.dtype.byteorder != ncdflib.NC_BYTE_ORDER:
                    bytestring = data.byteswap().tostring()
                else:
                    bytestring = data.tostring()
                f.write(bytestring)
                # No padding if there is only one record variable
            else:
                # EOF
                pass
        else:
            f.write(ncdflib.ABSENT)

    def create_dimension(self, name, length):
        """Create a new dimension.

        Parameters
        ----------
        name : string
            The name of the new dimension. An exception will be raised if the
            object already has a dimension with this name. name must satisfy
            netCDF-3 naming rules.
        length : int or None
            The length of the new dimension; must be non-negative and
            representable as a signed 32-bit integer. If None, the new
            dimension is unlimited, and its length is not determined until a
            variable is defined on it. An exception will be raised if you
            attempt to create another unlimited dimension when the object
            already has an unlimited dimension.
        """
        if ENSURE_VALID and not ncdflib.is_valid_name(name):
            raise ValueError("Not a valid dimension name")
        if name in self.dimensions:
            raise ValueError("Dimension named '%s' already exists" % name)
        if not length:
            # netCDF-3 only allows one unlimited dimension.
            if self.record_dimension is not None:
                raise ValueError("Only one unlimited dimension is allowed")
        else:
            if not isinstance(length, int):
                raise TypeError("Dimension length must be int")
            try:
                assert length >= 0
                ncdflib.pack_int(length)
            except:
                raise ValueError("Length of non-record dimension must be a " +
                        "positive-valued signed 32-bit integer")
        # Use the __setitem__ method of the superclass
        super(type(self.dimensions), self.dimensions).__setitem__(name, length)
        if not length:
            object.__setattr__(self, 'record_dimension', name)

    def create_variable(self, name, dim, dtype=None, data=None,
            attributes=None):
        """Create a new variable.

        Parameters
        ----------
        name : string
            The name of the new variable. An exception will be raised
            if the object already has a variable with this name. name
            must satisfy netCDF-3 naming rules. If name equals the name
            of a dimension, then the new variable is treated as a
            coordinate variable and must be 1-dimensional.
        dtype: numpy.dtype or similar
            The data type of the new variable. If this argument is not a
            numpy.dtype, an attempt will be made to convert it for you.
        dim : tuple
            The dimensions of the new variable. Elements must be dimensions of
            the object.
        data : numpy.ndarray or None, optional
            Data to populate the new variable. If None (default), then
            an empty numpy array is allocated with the appropriate
            shape and dtype. If data contains int64 integers, it will
            be coerced to int32 (for the sake of netCDF compatibility),
            and an exception will be raised if this coercion is not
            safe.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created variable.
        """
        if name in self.variables:
            raise ValueError("Variable named '%s' already exists" % (name))
        if not all([(d in self.dimensions) for d in dim]):
            bad = [d for d in dim if (d not in self.dimensions)]
            raise ValueError("the following dim(s) are not valid " +
                    "dimensions of this object: %s" % bad)
        record = (self.record_dimension is not None) and \
                 (self.record_dimension in dim)
        # Check that the record dimension does not appear anywhere other
        # than dim[0].
        if record and (list(reversed(dim)).index(self.record_dimension) !=
                       len(dim) - 1):
            raise ValueError("Only the 0th dimension of a variable is " +
                             "allowed to be the record dimension")
        if data is None:
            # Check that data is compatible with the length of the
            # record dimension
            if record and self.dimensions[self.record_dimension] is None:
                raise ValueError(
                    "data can not be None when creating the first record " +
                    "variable in an object")
            shape = tuple([
                self.dimensions[d] if d != self.record_dimension
                else self.dimensions[self.record_dimension]
                for d in dim])
        else:
            data = np.asarray(data)
            shape = data.shape
            if record:
                numrecs = self.dimensions[self.record_dimension]
                if (numrecs is not None) and (shape[0] != numrecs):
                    raise ValueError(
                        "data length along the record dimension is "
                        "inconsistent with the number of records in the " +
                        "other record variables")
        if (name in self.dimensions) and ((data is None) or (data.ndim != 1)):
            msg = ("A coordinate variable must be defined with " +
                    "1-dimensional data: %s") % name
            if ENSURE_VALID:
                raise ValueError(msg)
            else:
                logging.warn(msg)

        super(type(self.variables), self.variables).__setitem__(name,
                Variable(dim=dim, data=data, dtype=dtype, shape=shape,
                attributes=attributes))
        # Update self.dimensions if this new variable is the first
        # record variable
        if record and self.dimensions[self.record_dimension] is None:
            super(type(self.dimensions), self.dimensions).__setitem__(
                self.record_dimension, data.shape[0])
        return self.variables[name]

    def add_variable(self, name, variable):
        """A convenience function for adding a variable from one object to
        another.

        Parameters:
        name : string - The name under which the variable will be added
        variable : core.Variable - The variable to be added. If the desired
            action is to add a copy of the variable be sure to do so before
            passing it to this function.
        """
        # any error checking should be taken care of by create_variable
        self.create_variable(name,
                             dim=variable.dimensions,
                             data=variable.data,
                             attributes=variable.attributes)
        return self.variables[name]

    def delete_variable(self, name):
        """Delete a variable. Dimensions on which the variable is
        defined are not affected.

        Parameters
        ----------
        name : string
            The name of the variable to be deleted. An exception will
            be raised if there is no variable with this name.
        """
        if name not in self.variables:
            raise ValueError("Object does not have a variable '%s'" %
                    (str(name)))
        else:
            super(type(self.variables), self.variables).__delitem__(name)


    def create_coordinate(self, name, data, record=False, attributes=None):
        """Create a new dimension and a corresponding coordinate variable.

        This method combines the create_dimension and create_variable methods
        for the common case when the variable is a 1-dimensional coordinate
        variable with the same name as the dimension.

        Parameters
        ----------
        name : string
            The name of the new dimension and variable. An exception
            will be raised if the object already has a dimension or
            variable with this name. name must satisfy netCDF-3 naming
            rules.
        data : array_like
            The coordinate values along this dimension; must be
            1-dimensional.  The dtype of data is the dtype of the new
            coordinate variable, and the size of data is the length of
            the new dimension. If data contains int64 integers, it will
            be coerced to int32 (for the sake of netCDF compatibility),
            and an exception will be raised if this coercion is not
            safe.
        record : bool, optional
            A flag to indicate whether the new dimension is unlimited.
            An exception will be raised if you attempt to create
            another unlimited dimension when the object already has an
            unlimited dimension. The default is False.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.

        Returns
        -------
        var : Variable
            Reference to the newly created coordinate variable.
        """
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError("data must be 1-dimensional (vector)")
        if not isinstance(record, bool):
            raise TypeError("record argument must be bool")
        if record and self.record_dimension is not None:
            raise ValueError("Only one unlimited dimension is allowed")
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        old_dims = {'record_dimension': self.record_dimension,
                    'dimensions': self.dimensions.copy(),
                   }
        if record:
            self.create_dimension(name, None)
        else:
            self.create_dimension(name, data.size)
        try:
            self.create_variable(name=name,
                    dim=(name,),
                    data=data,
                    attributes=attributes)
        except:
            # Restore previous state
            for k in old_dims:
                object.__setattr__(self, k, old_dims[k])
            raise
        return self.variables[name]

    def add_coordinate(self, name, coordinate, record=False):
        """A convenience function for adding a coordinate from one object to
        another.

        Parameters:
        name : string - The name under which the variable will be added
        coordinate : core.Variable - The coordinate to be added.  If the desired
            action is to add a copy of the coordinate be sure to do so before
            passing it to this function.
        """
        # any error checking should be taken care of by create_variable
        # TODO record should be defaulted to None and discovered in the
        # variable if not otherwise specified
        self.create_coordinate(name=name,
                               data=coordinate.data,
                               record=record,
                               attributes=coordinate.attributes)
        return self.variables[name]

    def create_string_variable(self, name, dim, data, attributes=None,
                               strlen=None):
        """
        Creates a new variable from an ndarray of variable-length
        strings by storing them as an array characters.

        Parameters
        ----------
        name : string
            The name of the new variable. An exception will be raised
            if the object already has a variable with this name. name
            must satisfy netCDF-3 naming rules.
        dim : tuple
            The dimensions of the new variable. An extra dimension
            whose name is name + '_strlen' is created to accommodate
            the character data.
        data : iterable
            Data to populate the new variable; must consist of strings
            of characters in the allowed netCDF-3 character set. data
            is stored as a character array.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.
        strlen : int or None, optional
            Padding depth of the character array. Any strings with fewer than
            strlen characters are right-padded with null bytes. An exception is
            raised if any of the elements in data has length greater than
            strlen. If None (default), then strlen equals the length of the
            longest element in data.
        """
        if name in self.dimensions:
            raise ValueError(
                    "Character array can not be a coordinate variable")
        data = np.asarray(data)
        if data.dtype.char != 'S':
            raise TypeError("data must contain string elements")
        char_data = ncdflib.char_unstack(data, axis= -1)
        if strlen is not None:
            strlendiff = strlen - data.dtype.itemsize
            if strlendiff < 0:
                raise ValueError("strlen is smaller than the length of " +
                        "the longest element in data")
            elif strlendiff > 0:
                shp = tuple(list(data.shape) + [strlendiff])
                padding = np.empty(shp, dtype='|S1')
                padding.fill(ncdflib.NULL)
                char_data = np.concatenate((char_data, padding), axis= -1)
        if not char_data.shape[-1]:
            raise ValueError("String array must have at least one element " +
                    "with non-zero length")
        # we need to create a dimension for the string length
        strlen_name = conv.strlen(name)
        if strlen_name in dim:
            raise ValueError("'%s' is a reserved dimension name" %
                    (strlen_name))
        if strlen_name in self.dimensions:
            raise ValueError(("'%s' is not a valid variable name " +
                    "because there is already a '%s' dimension") %
                    (name, strlen_name))
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        old_dims = {'dimensions': self.dimensions.copy(),
                   }
        self.create_dimension(strlen_name, char_data.shape[-1])
        try:
            self.create_variable(name=name,
                    dim=tuple(list(dim) + [strlen_name]),
                    dtype=char_data.dtype,
                    data=char_data,
                    attributes=attributes)
        except:
            # Restore previous state
            for k in old_dims:
                object.__setattr__(self, k, old_dims[k])
            raise
        return self.variables[name]

    def unpack_string_variable(self, name):
        """Unpacks an array of strings stored as a char array under a variable
        created using the create_string_variable() method.

        Returns a numpy array of strings.
        """
        chars = self.variables[name].data
        extra_dim = conv.strlen(name)
        if extra_dim not in self.dimensions:
            raise ValueError("object does not have a '%s' dimension" %
                    (extra_dim))
        if extra_dim != self.variables[name].dimensions[-1]:
            raise ValueError(
                    "variable '%s' does not have a trailing '%s' dimension" %
                    (name, extra_dim))
        return ncdflib.char_stack(chars, axis= -1)

    def create_pickled_variable(self, name, dim, data, attributes=None):
        """
        Creates a new variable from an ndarray of pickle-able objects by
        serializing them and storing the strings as a matrix of characters.

        Parameters
        ----------
        name : string
            The name of the new variable. An exception will be raised
            if the object already has a variable with this name. name
            must satisfy netCDF-3 naming rules.
            coordinate variable and must be 1-dimensional.
        dim : tuple
            The dimensions of the new variable.
        data : iterable
            Data to populate the new variable. Each element of data is
            pickled and stored as a string.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.
        """
        # pickle the objects and store them in a char matrix, if the object
        # is not pickle-able it will be caught here.
        data = np.asarray(data)
        strings = [pickle.dumps(d, HIGHEST_PROTOCOL) for d in data.flat]
        # Pack into a numpy array and reshape
        strings = np.array(strings,
                dtype=("|S%d" % (max([len(s) for s in strings]))))
        strings.shape = data.shape
        try:
            return self.create_string_variable(name=name,
                    dim=dim, data=strings, attributes=attributes)
        except:
            raise

    def unpickle_variable(self, name):
        """Unpickles an array of pickled objects stored as a char array under a
        variable created using the create_pickled_variable() method.

        Returns a numpy array of unpickled objects.
        """
        chars = self.variables[name].data
        extra_dim = conv.strlen(name)
        if extra_dim not in self.dimensions:
            raise ValueError("object does not have a '%s' dimension" %
                    (extra_dim))
        if extra_dim != self.variables[name].dimensions[-1]:
            raise ValueError(
                    "variable '%s' does not have a trailing '%s' dimension" %
                    (name, extra_dim))
        strings = ncdflib.char_stack(chars, axis= -1)
        out = np.empty(strings.shape, dtype=object)
        for (i, s) in np.ndenumerate(strings):
            out[i] = pickle.loads(s)
        return out

    def view(self, s, dim=None):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string, optional
            The dimension to slice along. If multiple dimensions of a
            variable equal dim (e.g. a correlation matrix), then that
            variable is sliced only along its first matching dimension.
            If None (default), then the object is sliced along its
            unlimited dimension; an exception is raised if the object
            does not have an unlimited dimension.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes, dimensions,
            variable names and variable attributes as the original.
            Variables that are not defined along the specified
            dimensions are viewed in their entirety. Variables that are
            defined along the specified dimension have their data
            contents taken along the specified dimension.

            Care must be taken since modifying (most) values in the returned
            object will result in modification to the parent object.

        See Also
        --------
        numpy.take
        Variable.take
        """
        if dim is None:
            if self.record_dimension is None:
                raise ValueError(
                    "dim can not be None unless object has a record dimension")
            else:
                dim = self.record_dimension
        if not isinstance(s, slice):
            raise IndexError("view requires a slice argument")
        # Create a new object
        obj = self.__class__()
        # Create views onto the variables and infer the new dimension length
        new_length = self.dimensions[dim]
        for (name, var) in self.variables.iteritems():
            if dim in var.dimensions:
                super(type(obj.variables), obj.variables).__setitem__(
                    name, var.view(s, dim))
                axis = list(var.dimensions).index(dim)
                new_length = obj.variables[name].data.shape[axis]
            else:
                super(type(obj.variables), obj.variables).__setitem__(
                    name, var)
        # Hard write the dimensions, skipping validation
        object.__setattr__(obj, 'record_dimension', self.record_dimension)
        object.__setattr__(obj, 'dimensions', self.dimensions.copy())
        super(type(obj.dimensions), obj.dimensions).__setitem__(
            dim, new_length)
        if (dim != obj.record_dimension) and (obj.dimensions[dim] == 0):
            raise IndexError(
                "view would result in a non-record dimension of length zero")
        # Reference to the attributes, this intentionally does not copy.
        object.__setattr__(obj, 'attributes', self.attributes)
        return obj

    def take(self, indices, dim=None):
        """Return a new object whose contents are taken from the
        current object along a specified dimension

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract. indices must be compatible
            with the ndarray.take() method.
        dim : string, optional
            The dimension to slice along. If multiple dimensions of a
            variable equal dim (e.g. a correlation matrix), then that
            variable is sliced only along its first matching dimension.
            If None (default), then the object is sliced along its
            unlimited dimension; an exception is raised if the object
            does not have an unlimited dimension.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes, dimensions,
            variable names and variable attributes as the original.
            Variables that are not defined along the specified
            dimensions are copied in their entirety. Variables that are
            defined along the specified dimension have their data
            contents taken along the specified dimension.

        See Also
        --------
        numpy.take
        Variable.take
        """
        if dim is None:
            if self.record_dimension is None:
                raise ValueError(
                    "dim can not be None unless object has a record dimension")
            else:
                dim = self.record_dimension
        # Create a new object
        obj = self.__class__()
        # Create fancy-indexed variables and infer the new dimension length
        new_length = self.dimensions[dim]
        for (name, var) in self.variables.iteritems():
            if dim in var.dimensions:
                super(type(obj.variables), obj.variables).__setitem__(
                        name, var.take(indices, dim))
                new_length = obj.variables[name].data.shape[
                    list(var.dimensions).index(dim)]
            else:
                super(type(obj.variables), obj.variables).__setitem__(
                        name, copy.deepcopy(var))
        # Hard write the dimensions, skipping validation
        object.__setattr__(obj, 'record_dimension', self.record_dimension)
        object.__setattr__(obj, 'dimensions', self.dimensions.copy())
        super(type(obj.dimensions), obj.dimensions).__setitem__(
            dim, new_length)
        if (dim != obj.record_dimension) and (obj.dimensions[dim] == 0):
            raise IndexError(
                "take would result in a non-record dimension of length zero")
        # Copy attributes
        object.__setattr__(obj, 'attributes', self.attributes.copy())
        return obj

    def renamed(self, name_dict):
        """
        Returns a copy of the current object with variables and dimensions
        reanmed according to the arguments passed via **kwds

        Parameters
        ----------
        name_dict : dict-like
            Dictionary-like object whose keys are current variable
            names and whose values are new names.
        """
        for name in self.dimensions.iterkeys():
            if name in self.variables and not name in self.coordinates:
                raise ValueError("Renaming assumes that only coordinates " +
                                 "have both a dimension and variable under " +
                                 "the same name.  In this case it appears %s " +
                                 "has a dim and var but is not a coordinate"
                                 % name)

        new_names = dict((name, name)
                for name, d in self.dimensions.iteritems())
        new_names.update(dict((name, name)
                for name, v in self.variables.iteritems()))

        for k, v in name_dict.iteritems():
            if not k in new_names:
                raise ValueError("Cannot rename %s because it does not exist" % k)
        new_names.update(name_dict)

        obj = self.__class__()
        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in self.dimensions.iteritems():
            obj.create_dimension(new_names[name], length)
        object.__setattr__(obj, 'record_dimension', self.record_dimension)
        # a variable is only added if it doesn't currently exist, otherwise
        # and exception is thrown
        for (name, v) in self.variables.iteritems():
            obj.create_variable(new_names[name],
                                tuple([new_names[d] for d in v.dimensions]),
                                data=v.data.copy(),
                                attributes=v.attributes.copy())
        # update the root attributes
        obj.attributes.update(self.attributes.copy())
        return obj

    def update(self, other):
        """
        An update method (simular to dict.update) for data objects whereby each
        dimension, variable and attribute from 'other' is updated in the current
        object.  Note however that because Data object attributes are often
        write protected an exception will be raised if an attempt to overwrite
        any variables is made.
        """
        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in other.dimensions.iteritems():
            if not name in self.dimensions:
                self.create_dimension(name, length)
            else:
                cur_length = self.dimensions[name]
                if cur_length is None:
                    cur_length = self[self.record_dimension].data.size
                if length != cur_length:
                    raise ValueError("inconsistent dimension lengths for " +
                                     "dim: %s , %s != %s" %
                                     (name, length, cur_length))
        # a variable is only added if it doesn't currently exist, otherwise
        # and exception is thrown
        for (name, v) in other.variables.iteritems():
            if not name in self.variables:
                self.create_variable(name,
                                     v.dimensions,
                                     data=v.data.copy(),
                                     attributes=v.attributes.copy())
            else:
                if self[name].dimensions != other[name].dimensions:
                    raise ValueError("%s has different dimensions cur:%s new:%s"
                                     % (name, str(self[name].dimensions),
                                        str(other[name].dimensions)))
                if (self.variables[name].data.tostring() !=
                    other.variables[name].data.tostring()):
                    raise ValueError("%s has different data" % name)
                self[name].attributes.update(other[name].attributes)
        # update the root attributes
        self.attributes.update(other.attributes)

    def select(self, var):
        """Return a new object that contains the specified variables,
        along with the dimensions on which those variables are defined
        and corresponding coordinate variables.

        Parameters
        ----------
        var : bounded sequence of strings
            The variables to include in the returned object.

        Returns
        -------
        obj : Data object
            The returned object has the same attributes as the
            original. A dimension is included if at least one of the
            specified variables is defined along that dimension.
            Coordinate variables (1-dimensional variables with the same
            name as a dimension) that correspond to an included
            dimension are also included. All other variables are
            dropped.
        """
        if isinstance(var, basestring):
            var = [var]
        if not (hasattr(var, '__iter__') and hasattr(var, '__len__')):
            raise TypeError("var must be a bounded sequence")
        if not all((v in self.variables for v in var)):
            raise KeyError(
                "One or more of the specified variables does not exist")
        # Create a new Data instance
        obj = self.__class__()
        # Copy relevant dimensions
        dim = reduce(or_, [set(self.variables[v].dimensions) for v in var])
        # Create dimensions in the same order as they appear in self.dimension
        for d in self.dimensions:
            if d in dim:
                if d == self.record_dimension:
                    obj.create_dimension(d, None)
                else:
                    obj.create_dimension(d, self.dimensions[d])

        # Also include any coordinate variables defined on the relevant
        # dimensions
        for (name, v) in self.variables.iteritems():
            if (name in var) or ((name in dim) and (v.dimensions == (name,))):
                obj.create_variable(name=name,
                        dim=v.dimensions,
                        data=v.data.copy(),
                        attributes=v.attributes.copy())
        # Copy attributes
        object.__setattr__(obj, 'attributes', self.attributes.copy())
        return obj

    def iterator(self, dim=None, views=False):
        """Iterator along a data dimension

        Return an iterator yielding (coordinate, data_object) pairs
        that are singleton along the specified dimension

        Parameters
        ----------
        dim : string, optional
            The dimension along which you want to iterate. If None
            (default), then the iterator operates along the record
            dimension; if there is no record dimension, an exception
            will be raised.
        views : boolean, optional
            If True, the iterator will give views of the data along
            the dimension, otherwise copies.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued
            coordinate variables and data objects. The yielded data
            objects contain *copies* onto the underlying numpy arrays of
            the original data object. If the data object does not have
            a coordinate variable with the same name as the specified
            dimension, then the returned coordinate value is None. If
            multiple dimensions of a variable equal dim (e.g. a
            correlation matrix), then that variable is iterated along
            the first matching dimension.

        Examples
        --------
        >>> d = Data()
        >>> d.create_coordinate(name='x', data=numpy.arange(10))
        >>> d.create_coordinate(name='y', data=numpy.arange(20))
        >>> print d

        dimensions:
          name            | length
         ===========================
          x               | 10
          y               | 20

        variables:
          name            | dtype   | shape           | dimensions
         =====================================================================
          x               | int32   | (10,)           | ('x',)
          y               | int32   | (20,)           | ('y',)

        attributes:
          None

        >>> i = d.iterator(dim='x')
        >>> (a, b) = i.next()
        >>> print a

        dtype:
          int32

        dimensions:
          name            | length
         ===========================
          x               | 1

        attributes:
          None

        >>> print b

        dimensions:
          name            | length
         ===========================
          x               | 1
          y               | 20

        variables:
          name            | dtype   | shape           | dimensions
         =====================================================================
          x               | int32   | (1,)            | ('x',)
          y               | int32   | (20,)           | ('y',)

        attributes:
          None

        """
        # Determine the size of the dim we're about to iterate over
        n = self.dimensions[dim]
        # Iterate over the object
        if dim in self.coordinates:
            coord = self.variables[dim]
            if views:
                for i in xrange(n):
                    s = slice(i, i + 1)
                    yield (coord.view(s, dim=dim),
                           self.view(s, dim=dim))
            else:
                for i in xrange(n):
                    indices = np.array([i])
                    yield (coord.take(indices, dim=dim),
                           self.take(indices, dim=dim))
        else:
            if views:
                for i in xrange(n):
                    yield (None, self.view(slice(i, i + 1), dim=dim))
            else:
                for i in xrange(n):
                    yield (None, self.take(np.array([i]), dim=dim))

    def group_by(self, dim, keyfunc):
        """
        Similar to itertools.group_by, this function iterates over slices
        along dim yielding takes of all the slices which share common keys.

        keys are defined by by keyfunc which must take a core.Data object
        as input and yield a hashable key.

        Parameters
        ---------
        dim : string
            the dimension to group over
        keyfunc : callable
            a function taking slices and yielding corresponding keys

        Returns
        ---------
        a generator yielding : (key, obj.take(inds_with_same_key, dim)
        """
        def getkey(x):
            # apply keyfunc to the slice (not the current dim value)
            # x looks like : (ind, (dim, slice))
            return keyfunc(x[1][1])
        # here we only want views since we're simply getting the indices
        iter = sorted(enumerate(self.iterator(dim, views=True)), key=getkey)
        for k, group in itertools.groupby(iter, getkey):
            # at this point group holds all of the (enumerated) individual
            # slices with the same key, but what we really want is this
            # merged into one object, rather than merging all the
            # slices we can simply keep track of the indices and then
            # do a take
            inds = [x[0] for x in group]
            yield (k, self.take(inds, dim=dim))

    def iterarray(self, var, dim=None):
        """Iterator along a data dimension returning the corresponding slices
        of the underlying data of a varaible.

        Return an iterator yielding (scalar, ndarray) pairs that are singleton
        along the specified dimension.  While iterator is more general, this
        method has less overhead and in turn should be considerably faster.

        Parameters
        ----------
        var : string
            The variable over which you want to iterate.

        dim : string, optional
            The dimension along which you want to iterate. If None
            (default), then the iterator operates along the record
            dimension; if there is no record dimension, an exception
            will be raised.

        Returns
        -------
        it : iterator
            The returned iterator yields pairs of scalar-valued
            and ndarray objects. The yielded data objects contain *views*
            onto the underlying numpy arrays of the original data object.

        Examples
        --------
        >>> d = Data()
        >>> d.create_coordinate(name='t', data=numpy.arange(5))
        >>> d.create_dimension(name='h', length=3)
        >>> d.create_variable(name='x', dim=('t', 'h'),\
        ...     data=numpy.random.random((10, 3,)))
        >>> print d['x'].data
        [[ 0.33499995  0.47606901  0.41334325]
         [ 0.20229308  0.73693437  0.97451746]
         [ 0.40020704  0.29763575  0.85588908]
         [ 0.44114434  0.79233816  0.59115313]
         [ 0.18583972  0.55084889  0.95478946]]
        >>> i = d.iterarray(var='x', dim='t')
        >>> (a, b) = i.next()
        >>> print a
        0
        >>> print b
        [[ 0.33499995  0.47606901  0.41334325]]
        """
        # Get a reference to the underlying ndarray for the desired variable
        # and build a list of slice objects
        data = self.variables[var].data
        dim = dim or self.record_dimension
        axis = list(self.variables[var].dimensions).index(dim)
        slicer = [slice(None)] * data.ndim
        # Determine the size of the dim we're about to iterate over
        n = self.dimensions[dim]
        # Iterate over dim returning views of the variable.
        if dim in self.coordinates:
            coord = self.variables[dim].data
            for i in xrange(n):
                slicer[axis] = slice(i, i + 1)
                yield (coord[i], data[slicer])
        else:
            for i in xrange(n):
                slicer[axis] = slice(i, i + 1)
                yield (None, data[slicer])

    def csv(self, var, out=None):
        """
        Writes the variable 'var' to csv using the I/O object out

        Parameters
        ----------
        var : string
            The variable to write to csv.
        out : file-like (optional)
            The file-like object to write the csv to.  Defaults to a new
            StringIO object.

        Returns:
        -------
        out : file-like
            The object that the csv was written to.
        """
        if self[var].data.ndim != 2:
            raise ValueError("csv currently only supports 2 dimensional vars")
        # default to writing to a StringIO object
        out = out or StringIO()
        # get the row/col dimensions from the variable
        row_dim, col_dim = self[var].dimensions
        # if the column dimension is a coordinate we include the
        # coordinate values in the first row.  other wise just index it
        if col_dim in self.coordinates:
            col_var = self[col_dim]
            is_time = (col_dim == conv.TIME)
            col_data = datelib.from_udvar(col_var) if is_time else col_var.data
            header = col_data
        else:
            is_record = (col_dim == self.record_dimension)
            dim_len = self.dimensions[col_dim]
            header = np.arange(dim_len)
        header = [str(x) for x in header]
        # write the first line
        w = csv.writer(out)
        w.writerow(list(itertools.chain([''], [str(x) for x in header])))
        # if the row dimensions is a coordinate we use its values other-
        # wise we just index them.
        if row_dim in self.coordinates:
            row_var = self[row_dim]
            is_time = (row_dim == conv.TIME)
            row_data = datelib.from_udvar(row_var) if is_time else row_var.data
            row_labels = [str(x) for x in row_data]
        else:
            is_record = (row_dim == self.record_dimension)
            dim_len = self.dimensions[row_dim]
            row_labels = '' * self.dimensions[dim_len]
        row_labels = [str(x) for x in row_labels]
        # write each row to csv
        for r, data in zip(row_labels, self[var].data):
            w.writerow(list(itertools.chain([r], [str(x) for x in data])))
        return out

    def sort_coordinate(self, coord):
        if not coord in self.coordinates:
            raise ValueError("%s is not a coordinate" % coord)
        data = self[coord].data
        if not data.ndim == 1:
            raise ValueError("coordinate must be unidimensional")
        inds = np.argsort(data)
        def sort_var(x):
            axis = x.dimensions.index(coord)
            x.data[:] = x.data.take(inds, axis=axis)
        [sort_var(v) for v in self.variables.values() if coord in v.dimensions]

    def interpolate(self, vars=None, fast=False, **points):

        if not all([k in self.coordinates for k in points.keys()]):
            raise ValueError("interpolation only works on coordinates")

        from bisect import bisect
        def neighbors(coord, val):
            if not self[coord].data.ndim == 1:
                raise ValueError("coordinate has more than one dimension")
            data = self[coord].data
            j = bisect(data, val)
            if j == 0 or j == data.size:
                raise ValueError("value of %6.3f is outside the range of %s"
                                 % (val, coord))
            i = j - 1
            if data[i] == val:
                return [i, i], 1.0
            else:
                alpha = np.abs(val - data[j]) / np.abs(data[i] - data[j])
                assert alpha <= 1. and alpha >= 0.
                return [i, j], alpha

        def weight(obj, inds, coord, w):
            assert len(inds) == 2
            views = [obj.view(slice(x, x + 1), dim=coord) for x in inds]
            if vars is None:
                vs = [v for v in obj.variables.keys() if coord in obj[v].dimensions]
            else:
                vs = vars
            for v in vs:
                views[0][v].data[:] *= w
                views[0][v].data[:] += (1. - w) * views[1][v].data[:]
            return views[0]

        def closest(obj, inds, coord, w):
            assert len(inds) == 2
            import pdb; pdb.set_trace()
            views = [obj.view(x, dim=coord) for x in inds]
            if vars is None:
                vs = [v for v in obj.variables.keys() if coord in obj[v].dimensions]
            else:
                vs = vars
            for v in vs:
                views[0][v].data[:] *= w
                views[0][v].data[:] += (1. - w) * views[1][v].data[:]
            return views[0]

        nhbrs = [(k, neighbors(k, v)) for k, v in points.iteritems()]
        obj = self
        for coord, nhbr in nhbrs:
            if fast:
                closest_ind = nhbr[0][nhbr[1] > 0.5]
                obj = obj.view(slice(closest_ind, closest_ind + 1),
                               dim=coord)
            else:
                obj = obj.take(nhbr[0], dim=coord)

        if not fast:
            for coord, nhbr in nhbrs:
                obj = weight(obj, [0, 1], coord, nhbr[1])
        return obj

    def interp(self, var, **points):
        # ASSUMPTION, ALL COORDINATES ARE SORTED
        if len(points) != len(self[var].dimensions):
            msg = "interpolation requires specifying all dimensions"
            msg = msg + " required: %s -- given:%s"
            raise ValueError(msg % (str(self[var].dimensions), str(points.keys())))
        # TODO remove this restriction
        # assert self[var].data.ndim <= 2

        from bisect import bisect
        def neighbors(coord, val):
            if not self[coord].data.ndim == 1:
                raise ValueError("coordinate has more than one dimension")
            data = self[coord].data
            j = bisect(data, val)
            if j == 0 or j == data.size:
                raise ValueError("value of %6.3f is outside the range of %s"
                                 % (val, coord))
            i = j - 1
            if data[i] == val:
                return [i, i], 1.0
            else:
                alpha = np.abs(val - data[j]) / np.abs(data[i] - data[j])
                assert alpha <= 1. and alpha >= 0.
                return [i, j], alpha

        iter = [(k, points[k]) for k in self[var].dimensions]
        nhbrs = [neighbors(k, v) for k, v in iter]
        ret = self[var].data.copy()
        for i, nhbr in reversed(list(enumerate(nhbrs))):
            ret = ret.take(nhbr[0], axis=i)
        ret = ret.T
        for inds, alpha in nhbrs:
            ndim = ret.ndim
            ret = np.dot(ret, [alpha, 1. - alpha])
            if ret.ndim == ndim:
                raise ValueError("dot did not reduce the dimension")
                # for some reason np.dot doesn't always reduce the dimension
                # so we need to do it manually from time to time
                ret = ret[..., 0]
        return ret

    def squeeze(self, dimension):
        """
        Squeezes dimensions of length 1, removing them for an object
        """
        # Create a new Data instance
        obj = self.__class__()
        if self.dimensions[dimension] != 1:
            raise ValueError(("Can only squeeze along dimensions with" +
                             "length one, %s has length %d") %
                             (dimension, self.dimensions[dimension]))

        # Copy dimensions
        for (name, length) in self.dimensions.iteritems():
            if not name == dimension:
                obj.create_dimension(name, length)
        # Copy variables
        for (name, var) in self.variables.iteritems():
            if not name == dimension:
                dims = list(var.dimensions)
                data = var.data.copy()
                if dimension in dims:
                    shape = list(var.data.shape)
                    index = dims.index(dimension)
                    shape.pop(index)
                    dims.pop(index)
                    data = data.reshape(shape)
                obj.create_variable(name=name,
                        dim=tuple(dims),
                        data=data,
                        attributes=var.attributes.copy())
        # Copy attributes
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.attributes.__setattr__(attr, self.attributes[attr].copy())
            else:
                obj.attributes.__setattr__(attr, self.attributes[attr])
        return obj

class Variable(object):
    """A class that wraps metadata around a numpy ndarray"""

    def __init__(self, dim, data=None, dtype=None, shape=None,
                 attributes=None):
        """Initializes a variable object with contents read from
        specified netCDF file.

        Parameters
        ----------
        dim : tuple
            Each element of dim must be a string that is a valid name
            in netCDF. Elements of dim do not necessarily have to be
            unique (e.g., the rows and columns of a covariance matrix).
            If dim is the empty tuple (), then the variable represents
            a scalar.
        data : array_like, optional
            A numpy ndarray or an array-like object that can be
            converted to ndarray. If None (default), the variable is
            initialized with an arbitrarily-populated array whose shape
            and dtype are specified by the other arguments. The number
            of dimensions of data must equal the length of dim, and the
            ith element of dim is assigned to the ith dimension of
            data. If data contains int64 integers, it will be coerced
            to int32 (for the sake of netCDF compatibility), and an
            exception will be raised if this coercion is not safe.
        dtype : dtype_like or None, optional
            A numpy dtype object, or an object that can be coerced to a
            dtype. If the data argument is not None, then this
            argument must equal the dtype of data. If None (default), then
            dtype is inferred from data.
        shape : tuple, optional
            Tuple of integers that specifies the shape of the data. If
            the data argument is not None, then this argument must
            equal the shape of data. shape must be a tuple of the same
            length as dim.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.
        """
        if attributes is None:
            attributes = AttributesDict()
        # Check dim
        if not isinstance(dim, tuple):
            raise TypeError("dim must be a tuple")
        if not all([ncdflib.is_valid_name(d) for d in dim]):
            bad = [d for d in dim if not ncdflib.is_valid_name(d)]
            raise ValueError("the following dim(s) are not valid " +
                             "dimensions of this object: %s" % bad)
        # Check shape
        if shape is not None:
            if not isinstance(shape, tuple):
                raise TypeError("shape must be a tuple")
            try:
                # checking for >= 0 allows empty data sets
                assert all([ncdflib.pack_int(s) and (s >= 0) for s in shape])
            except:
                raise ValueError("shape tuple must contain positive values " +
                        "that can be represented as 32-bit signed integers")
        # Convert dtype
        if dtype is not None and not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        if data is None:
            if dtype is None:
                raise ValueError("must provide dtype if data is None")
            if shape is None:
                raise ValueError("must provide shape if data is None")
            data = np.empty(shape, dtype=dtype)
            # Check the fillval
            if conv.FILLVALUE not in attributes:
                attributes[conv.FILLVALUE] = ncdflib.FILLMAP[dtype]
            data.fill(attributes[conv.FILLVALUE])
        data = np.asarray(data)
        # Check dtype
        if (dtype is not None) and (data.dtype != dtype):
            raise ValueError("dtype and data arguments are incompatible" +
                             "data: %s, dtype:%s" % (data.dtype, dtype))
        # Coerce dtype to be netCDF-3 compatible, or raise an exception if this
        # coercion is not possible without losing information
        data = ncdflib.coerce_type(data)
        if shape is None:
            shape = data.shape
        # Check data shape
        for (i, s) in enumerate(shape):
            if data.shape[i] != shape[i]:
                raise ValueError(("the %d^th data dimension (%s) is %d but the" +
                                  " input shape suggests it should be %d")
                                  % (i, dim[i], data.shape[i], int(s)))
        if data.shape != shape:
            raise ValueError("data array shape %s does not match the shape %s"
                             % (str(data.shape), str(shape)))
        if len(dim) != data.ndim:
            raise ValueError("number of dimensions don't match dim:%d data:%d"
                             % (len(dim), data.ndim))
        object.__setattr__(self, 'dimensions', dim)
        object.__setattr__(self, 'data', data)
        object.__setattr__(self, 'attributes', AttributesDict(attributes))

    def __getattr__(self, attr):
        """__getattr__ is overloaded to selectively expose some of the
        attributes of the underlying numpy array"""
        if attr in self.__dict__:
            return self.__dict__[attr]
        elif hasattr(self.data, attr) and (attr in
                set(['dtype', 'shape', 'size', 'ndim', 'nbytes',
                'flat', '__iter__', 'view'])):
            return self.data.__getattribute__(attr)
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr__ method to bypass."""
        raise AttributeError, "Object is tamper-proof"

    def __delattr__(self, attr):
        raise AttributeError, "Object is tamper-proof"

    def __getitem__(self, index):
        """__getitem__ is overloaded to access the underlying numpy data"""
        return self.data[index]

    def __setitem__(self, index, data):
        """__setitem__ is overloaded to access the underlying numpy data"""
        self.data[index] = data

    def __len__(self):
        """__len__ is overloaded to access the underlying numpy data"""
        return self.data.__len__()

    def __copy__(self):
        """
        Returns a shallow copy of the current object.
        """
        # Create the simplest possible dummy object and then overwrite it
        obj = self.__class__(dim=(), data=0)
        object.__setattr__(obj, 'dimensions', self.dimensions)
        object.__setattr__(obj, 'data', self.data)
        object.__setattr__(obj, 'attributes', self.attributes)
        return obj

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        # Create the simplest possible dummy object and then overwrite it
        obj = self.__class__(dim=(), data=0)
        # tuples are immutable
        object.__setattr__(obj, 'dimensions', self.dimensions)
        object.__setattr__(obj, 'data', self.data.copy())
        object.__setattr__(obj, 'attributes', self.attributes.copy())
        return obj

    def __eq__(self, other):
        if self.dimensions != other.dimensions or \
           (self.data.tostring() != other.data.tostring()):
            return False
        if not self.attributes == other.attributes:
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        """Human-readable summary in 70ish columns"""
        lines = []
        # Print dtype
        lines.append('\n' + self.__class__.__name__)
        lines.append('\ndtype:')
        lines.append('  %s' % (self.dtype))
        # Print dimensions
        lines.append('\ndimensions:')
        if self.dimensions:
            lines.append('  %s| %s' %
                    (_prettyprint('name', 16),
                    _prettyprint('length', 8)))
            lines.append(' =%s==%s' %
                    ('=' * 16, '=' * 8))
            for (i, name) in enumerate(self.dimensions):
                lines.append('  %s| %s' %
                        (_prettyprint(name, 16),
                        _prettyprint(self.shape[i], 8)))
        else:
            lines.append('  None')
        # Print attributes
        lines.append('\nattributes:')
        if self.attributes:
            lines.append('  %s| %s' %
                    (_prettyprint('name', 16),
                    _prettyprint('value', 50)))
            lines.append(' =%s==%s' %
                    ('=' * 16, '=' * 50))
            for (attr, val) in self.attributes.iteritems():
                lines.append('  %s| %s' %
                        (_prettyprint(attr, 16),
                        _prettyprint(val, 50)))
        else:
            lines.append('  None')
        lines.append('')
        return '\n'.join(lines)

    def view(self, s, dim):
        """Return a new Variable object whose contents are a view of the object
        sliced along a specified dimension.

        Parameters
        ----------
        s : slice
            The slice representing the range of the values to extract.
        dim : string
            The dimension to slice along. If multiple dimensions equal
            dim (e.g. a correlation matrix), then the slicing is done
            only along the first matching dimension.

        Returns
        -------
        obj : Variable object
            The returned object has the same attributes and dimensions
            as the original. Data contents are taken along the
            specified dimension.  Care must be taken since modifying (most)
            values in the returned object will result in modification to the
            parent object.

        See Also
        --------
        take
        """
        # When dim appears repeatededly in self.dimensions, using the index()
        # method gives us only the first one, which is the desired behavior
        axis = list(self.dimensions).index(dim)
        slices = [slice(None)] * self.data.ndim
        slices[axis] = s
        # Shallow copy
        obj = copy.copy(self)
        object.__setattr__(obj, 'data', self.data[slices])
        return obj

    def take(self, indices, dim):
        """Return a new Variable object whose contents are sliced from
        the current object along a specified dimension

        Parameters
        ----------
        indices : array_like
            The indices of the values to extract. indices must be compatible
            with the ndarray.take() method.
        dim : string
            The dimension to slice along. If multiple dimensions equal
            dim (e.g. a correlation matrix), then the slicing is done
            only along the first matching dimension.

        Returns
        -------
        obj : Variable object
            The returned object has the same attributes and dimensions
            as the original. Data contents are taken along the
            specified dimension.

        See Also
        --------
        numpy.take
        """
        indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError('indices should have a single dimension')

        # When dim appears repeatededly in self.dimensions, using the index()
        # method gives us only the first one, which is the desired behavior
        axis = list(self.dimensions).index(dim)
        # Deep copy
        obj = copy.deepcopy(self)
        object.__setattr__(obj, 'data', self.data.take(indices, axis=axis))
        return obj
