"""Core objects to represent data and models

The Data class and Variable class in this module are heavily reworked versions
of code from the PuPyNeRe netCDF reader:

    http://dealmeida.net/hg/pupynere/file/b95f566d84af/pupynere.py

pupynere is released under the MIT license for unlimited use.
"""

from __future__ import with_statement

import copy
import numpy as np
import pickle
import logging
from os import SEEK_END
from operator import mul, or_
from cStringIO import StringIO
from wx.lib import ncdflib
from wx.lib.collections import OrderedDict, SafeOrderedDict


def _prettyprint(x, numchars):
    """Given an object x, call x.__str__() and format the returned
    string so that it is numchars long, padding with trailing spaces or
    truncating with ellipses as necessary"""
    s = str(x).rstrip(ncdflib.NULL)
    if len(s) <= numchars:
        return s + ' ' * (numchars - len(s))
    else:
        return s[:(numchars - 3)] + '...'

def _coerce_type(arr):
    """Coerce a numeric data type to a type that is compatible with
    netCDF-3

    netCDF-3 can not handle 64-bit integers, but on most platforms
    Python integers are int64. To work around this discrepancy, this
    helper function coerces int64 arrays to int32. An exception is
    raised if this coercion is not safe.

    netCDF-3 can not handle booleans, but booleans can be trivially
    (albeit wastefully) represented as bytes. To work around this
    discrepancy, this helper function coerces bool arrays to int8.
    """
    if arr.dtype.newbyteorder('=') == np.dtype('int64'):
        cast_arr = arr.astype(
                np.dtype('int32').newbyteorder(arr.dtype.byteorder))
        if not (cast_arr == arr).all():
            raise ValueError("array contains integer values that " +
                    "are not representable as 32-bit signed integers")
        return cast_arr
    elif arr.dtype.newbyteorder('=') == np.dtype('bool'):
        cast_arr = arr.astype(
                np.dtype('int8').newbyteorder(arr.dtype.byteorder))
        return cast_arr
    else:
        return arr

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
        ncdf : string or file_like, optional
            If None (default), then an empty data object is created. If
            string, ncdffile is interpreted as a path on the file
            system and an attempt will be made to read from that file.
            Otherwise, ncdffile must be a file-like object that has
            read, seek and tell methods.
        """
        # The __setattr__ method of the base object class is used to
        # bypass the overloaded __setattr__ method
        object.__setattr__(self, '_version_byte', ncdflib._64BYTE)
        object.__setattr__(self, 'numrecs', None)
        object.__setattr__(self, 'record_dimension', None)
        # self.variables and self.dimensions are SafeOrderedDict. This
        # class provides two key advantages over basic dict:
        # (1) (key, value) elements are ordered. This ensures that
        #     data is written to netCDF file in exactly the same
        #     order that it was read from file.
        # (2) Once a (key, value) element is added, it can not be
        #     deleted or overwritten (although the value, if mutable,
        #     can be modified in place). This helps to maintain data
        #     consistency.
        #
        # self.attributes is OrderedDict, which preserves order but
        # does not prevent overwriting or deletion of values.
        object.__setattr__(self, 'dimensions', SafeOrderedDict())
        object.__setattr__(self, 'attributes', OrderedDict())
        object.__setattr__(self, 'variables', SafeOrderedDict())
        if ncdf is not None:
            if isinstance(ncdf, basestring):
                with open(ncdf, 'rb') as f:
                    self._from_file(f)
            else:
                self._from_file(ncdf)

    def __setattr__(self, attr, value):
        """"__setattr__ is overloaded to prevent operations that could
        cause loss of data consistency. If you really intend to update
        dir(self), use the self.__dict__.update method or the
        super(type(a), self).__setattr method to bypass."""
        raise AttributeError("Object is tamper-proof. " +
                             "so things like obj.foo = bar wont work")

    def __delattr__(self, attr):
        raise AttributeError("Object is tamper-proof")

    def __getitem__(self, name):
        """obj[varname] returns the variable"""
        if name not in self.variables:
            raise KeyError("variable not found")
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
        if self.numrecs is None:
            raise TypeError("Overloaded __len__ method only works for " +
                    "objects with a record dimension and at least one " +
                    "record variable")
        else:
            return self.numrecs

    def __iter__(self):
        """If there is a record dimension, __iter__ is overloaded in a
        manner analogoues to __len__"""
        if self.record_dimension is None:
            raise TypeError("Overloaded __iter__ method only works for " +
                    "objects with a record dimension")
        else:
            return self.iterator(dim=self.record_dimension)

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        # Create a new Data instance
        obj = self.__class__()
        # Copy dimensions
        for (name, length) in self.dimensions.iteritems():
            obj.create_dimension(name, length)
        # Copy variables
        for (name, var) in self.variables.iteritems():
            obj.create_variable(name=name,
                    dim=var.dimensions,
                    data=var.data.copy(),
                    attributes=var.attributes.copy())
        # Copy attributes
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

    def __eq__(self, other):
        if not isinstance(other, Data):
            return False
        if dict(self.dimensions) != dict(other.dimensions):
            return False
        if not dict(self.variables) == dict(other.variables):
            return False
        if not set(self.attributes.keys()) == set(other.attributes.keys()):
            return False

        for key in self.attributes:
            if not np.all(numpylib.naneq(self.attributes[key],
                                         other.attributes[key])):
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
                            _prettyprint(self.numrecs, 8)))
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
        return OrderedDict([(dim, length)
                for (dim, length) in self.dimensions.iteritems()
                if (dim in self.variables) and
                (self.variables[dim].data.ndim == 1) and
                (self.variables[dim].dimensions == (dim,))
                ])

    @property
    def noncoordinates(self):
        # A coordinate variable is a 1-dimensional variable with the
        # same name as its dimension
        return OrderedDict([(name, v)
                for (name, v) in self.variables.iteritems()
                if name not in self.coordinates])

    def has_attribute(self, attr):
        return attr in self.attributes

    def get_attribute(self, attr):
        """Get the value of a user-defined attribute"""
        if not self.has_attribute(attr):
            raise KeyError("attribute not found")
        return self.attributes[attr]

    def set_attribute(self, attr, value):
        """Set the value of a user-defined attribute"""
        if not ncdflib.is_valid_name(attr):
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
                value = np.atleast_1d(np.asarray(value))
            except:
                raise ValueError("Not a valid value for a netCDF attribute")
            if value.ndim > 1:
                raise ValueError("netCDF attributes must be vectors " +
                        "(1-dimensional)")
            value = _coerce_type(value)
            # Coerce dtype to native byte order for hashing/dictionary lookup
            if value.dtype.newbyteorder('=') not in ncdflib.TYPEMAP:
                # A plain string attribute is okay, but an array of
                # string objects is not okay!
                raise ValueError("Can not convert to a valid netCDF type")
        self.attributes[attr] = value

    def del_attribute(self, attr):
        if not self.has_attribute(attr):
            raise KeyError("attribute not found")
        del self.attributes[attr]

    def update_attributes(self, *args, **kwargs):
        """Set multiple user-defined attributes with a mapping object
        or an iterable of key/value pairs"""
        # Try it on a dummy copy first so that we don't end up in a
        # partial state
        dummy = self.__class__()
        try:
            attributes = OrderedDict(*args, **kwargs)
            for (k, v) in attributes.iteritems():
                # set_attribute method does checks and type conversions
                dummy.set_attribute(k, v)
            # If we can set the requested attributes without any
            # complaints, then we can apply it to the real object.
            del dummy
            for (k, v) in attributes.iteritems():
                self.set_attribute(k, v)
        except:
            raise

    def clear_attributes(self, **kwargs):
        """Delete all user-defined attributes"""
        object.__setattr__(self, 'attributes', OrderedDict())

    def _from_file(self, f):
        """Populate an initialized, empty object from data read from
        netCDF file. This method is non-public because we don't want to
        overwrite existing data by calling it with a non-empty
        object."""
        # Check magic bytes
        magic = f.read(len(ncdflib.MAGIC))
        if magic != ncdflib.MAGIC:
            raise ValueError, "Not a valid netCDF-3 file"
        # Check version_byte
        _version_byte = f.read(len(ncdflib.FILL_BYTE))
        if _version_byte not in [ncdflib._31BYTE, ncdflib._64BYTE]:
            raise ValueError, ("netCDF file header does not have a " +
                    "recognized version_byte")
        object.__setattr__(self, '_version_byte', _version_byte)
        numrecs = ncdflib.read_int(f)
        # Read dimensions and add them. The create_dimension method handles the
        # error-checking for us.
        dimensions = self._read_dim_array(f)
        for (name, length) in dimensions.iteritems():
            if length == 0:
                self.create_dimension(name, None)
            else:
                self.create_dimension(name, length)
        # Read global attributes
        attributes = self._read_att_array(f)
        # Use the update_attributes method because it does
        # error-checking and type conversion
        self.update_attributes(attributes)
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
                (numrecs != self.numrecs)):
            raise ValueError("netCDF file has a numrecs header field that " +
                    "is inconsistent with dimensions")

    def to_file(self, ncdf):
        """Write current data contents to file in netCDF format

        Parameters
        ----------
        ncdf : file-like object
            Must have write, seek and tell methods.
        """
        # Write magic header bytes
        ncdf.write(ncdflib.MAGIC)
        # Write version byte
        ncdf.write(self._version_byte)
        # Write number of records.
        if self.numrecs is None:
            ncdf.write(ncdflib.ZERO)
        else:
            ncdflib.write_int(ncdf, self.numrecs)
        # Write dimensions
        self._write_dim_array(ncdf)
        # Write global attributes
        self._write_att_array(ncdf, self.attributes)
        # Write variables
        self._write_variables(ncdf)

    def _loads(self, s):
        """Read data from netCDF format as a string of bytes"""
        buf = StringIO()
        buf.write(s)
        buf.seek(0)
        self._from_file(buf)
        buf.close()

    def dumps(self):
        """Return current data contents in netCDF format as a string of
        bytes"""
        buf = StringIO()
        self.to_file(buf)
        s = buf.getvalue()
        buf.close()
        return s

    def _read_dim_array(self, f):
        """Read dim_array from a netCDF file and return as a
        SafeOrderedDict"""
        dim_header = f.read(len(ncdflib.NC_DIMENSION))
        num_dim = ncdflib.read_int(f)
        if (dim_header not in [ncdflib.NC_DIMENSION, ncdflib.ZERO]) or\
                ((dim_header == ncdflib.ZERO) and (num_dim != 0)) or\
                (num_dim < 0):
            raise ValueError, "dimensions header is invalid"
        dimensions = SafeOrderedDict()
        for dim in xrange(num_dim):
            name = ncdflib.read_string(f)
            length = ncdflib.read_int(f)
            dimensions[name] = length
        return dimensions

    def _write_dim_array(self, f):
        """Write dim_array of a netCDF file"""
        if self.dimensions:
            f.write(ncdflib.NC_DIMENSION)
            ncdflib.write_int(f, len(self.dimensions))
            for (name, length) in self.dimensions.iteritems():
                ncdflib.write_string(f, name)
                if length is None:
                    f.write(ncdflib.ZERO)
                else:
                    ncdflib.write_int(f, length)
        else:
            f.write(ncdflib.ABSENT)

    def _read_att_array(self, f):
        """Read att_array from a netCDF file and return as OrderedDict"""
        attr_header = f.read(len(ncdflib.NC_ATTRIBUTE))
        num_attr = ncdflib.read_int(f)
        if (attr_header not in [ncdflib.NC_ATTRIBUTE, ncdflib.ZERO]) or\
                ((attr_header == ncdflib.ZERO) and (num_attr != 0)) or\
                (num_attr < 0):
            raise ValueError, "atttributes header is invalid"
        attributes = OrderedDict()
        for attr in xrange(num_attr):
            name = ncdflib.read_string(f)
            # The next 4 bytes tell us the data type of this attribute
            nc_type = f.read(4)
            if nc_type == ncdflib.NC_CHAR:
                # strings get special treatment
                value = ncdflib.read_string(f)
            else:
                # Look up corresponding numpy dtype
                dtype = ncdflib.TYPEMAP[nc_type]
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
                if value.dtype != value.dtype.newbyteorder('='):
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
                        raise ValueError, (
                                "numpy array attributes must be 1-dimensional")
                    # Coerce dtype to native byte order for hashing/dictionary
                    # lookup
                    f.write(ncdflib.TYPEMAP[value.dtype.newbyteorder('=')])
                    ncdflib.write_int(f, value.size)
                    # Write in network byte order
                    if value.dtype !=\
                            value.dtype.newbyteorder(ncdflib.NC_BYTE_ORDER):
                        bytestring = value.byteswap().tostring()
                    else:
                        bytestring = value.tostring()
                    f.write(bytestring)
                    # Write any necessary padding bytes
                    size_diff = ncdflib.round_num_bytes(len(bytestring)) -\
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
        var_order = []
        for var in xrange(num_var):
            name = ncdflib.read_string(f)
            num_dims = ncdflib.read_int(f)
            dimids = [ncdflib.read_int(f) for i in xrange(num_dims)]
            pos = f.tell()
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
                (dimensions, shape) = zip(*[self.dimensions.items()[i]
                        for i in dimids])
            else:
                (dimensions, shape) = ((), ())
            if shape and (shape[0] is None):
                # Add this record variable to the list; the variable will be
                # created later
                recs.append(var)
        # Determine the size of each record and the total number of records
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
        # The order in which variables are created is important! We
        # need to create variables in exactly the same order as they
        # appear in vatt_array, because the to_file() method will write
        # the variables to vatt_array in the order of creation.
        for var in var_metadata:
            (dimids, attributes, nc_type, vsize, begin) = var_metadata[var]
            if dimids:
                (dimensions, shape) = zip(*[self.dimensions.items()[i]
                        for i in dimids])
            else:
                (dimensions, shape) = ((), ())
            # Convert shape to list so that we can set elements
            # (necessary for record variables)
            shape = list(shape)
            dtype = ncdflib.TYPEMAP[nc_type]
            if var in recs:
                if shape[0] is not None:
                    raise ValueError, ("Not valid netCDF: the 0th " +
                            "dimension of each record must be the " +
                            "unlimited dimension")
                shape[0] = numrecs
                slice_size = reduce(mul, shape[1:], 1) * dtype.itemsize
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
                num_bytes = reduce(mul, shape, 1) * dtype.itemsize
                if vsize != ncdflib.round_num_bytes(num_bytes):
                    raise ValueError("data type and shape of variable are " +
                            "inconsistent with vsize specified in header")
                f.seek(begin)
                data = np.fromstring(f.read(num_bytes),
                        dtype=dtype.newbyteorder(ncdflib.NC_BYTE_ORDER))
                # If necessary, continue reading past the padding bytes
                f.read(vsize - num_bytes)
            # Reshape flat vector to desired shape
            data = data.reshape(shape)
            # Convert to native endianness
            if data.dtype != data.dtype.newbyteorder('='):
                data = data.byteswap().newbyteorder('=')
            # Create new variable
            self.create_variable(var, dimensions,
                    data=data, attributes=attributes)

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
                # Write nc_type enum. Coerce dtype to native byte order for
                # hashing/dictionary lookup
                f.write(ncdflib.TYPEMAP[v.dtype.newbyteorder('=')])
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
                        vsize = v.data.dtype.itemsize *\
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
                    ncdflib.write_int(f, begin)
                elif self._version_byte == ncdflib._64BYTE:
                    ncdflib.write_int64(f, begin)
                else:
                    raise RuntimeError("version_byte is not recognized")
                # Seek back to begin and starting writing
                f.seek(begin)
                data = v.data
                if data.dtype != data.dtype.newbyteorder(
                        ncdflib.NC_BYTE_ORDER):
                    bytestring = data.byteswap().tostring()
                else:
                    bytestring = data.tostring()
                f.write(bytestring)
                vsize = ncdflib.round_num_bytes(len(bytestring))
                # Coerce dtype to native byte order for hashing/dictionary
                # lookup
                f.write(ncdflib.NULL * (vsize - len(bytestring)))
            # The rest of the file contains data for the record variables.
            recs = [name for (name, v) in self.variables.iteritems()
                    if self.record_dimension in v.dimensions]
            # Start writing records
            begin = f.tell() # Remember this position in the file!
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
                    slice_size = dtype.itemsize *\
                            reduce(mul, self.variables[var].data.shape[1:], 1)
                    # Record for each variable is padded to next 4-byte
                    # boundary
                    vsize = ncdflib.round_num_bytes(slice_size)
                    # Determine padding for this variable
                    nc_type = ncdflib.TYPEMAP[dtype]
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
                # Write each multi-variable record. We use a string buffer to
                # assemble the contents of each record; because all records are
                # the same size, we can repeatedly overwrite and getvalue() the
                # same string buffer
                stringbuf = StringIO()
                for i in xrange(self.numrecs):
                    stringbuf.seek(0)
                    for var in recs:
                        data = self.variables[var].data[i, ...]
                        if data.dtype != data.dtype.newbyteorder(
                                ncdflib.NC_BYTE_ORDER):
                            bytestring = data.byteswap().tostring()
                        else:
                            bytestring = data.tostring()
                        stringbuf.write(bytestring)
                        # Add any padding bytes if necessary
                        stringbuf.write(padding_bytes[var])
                    # Once we're done appending to the string buffer, dump its
                    # contents to file. Each time we do this, one more record
                    # is appended to the end of the file
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
                if data.dtype != data.dtype.newbyteorder(
                        ncdflib.NC_BYTE_ORDER):
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
            The length of the new dimension; must be greater than zero and
            representable as a signed 32-bit integer. If None, the new
            dimension is unlimited, and its length is not determined until a
            variable is defined on it. An exception will be raised if you
            attempt to create another unlimited dimension when the object
            already has an unlimited dimension.
        """
        if not isinstance(name, basestring):
            raise TypeError("Dimension name must be a non-empty string")
        if not ncdflib.is_valid_name(name):
            raise ValueError("Not a valid dimension name")
        if name in self.dimensions:
            raise ValueError("Dimension named '%s' already exists" % name)
        if length is None:
            # netCDF-3 only allows one unlimited dimension.
            if self.record_dimension is not None:
                raise ValueError("Only one unlimited dimension is allowed")
        else:
            if not isinstance(length, int):
                raise TypeError("Dimension length must be int")
            try:
                assert length > 0
                ncdflib.pack_int(length)
            except:
                raise ValueError("Length of non-record dimension must be a " +
                        "positive-valued signed 32-bit integer")
        self.dimensions[name] = length
        if length is None:
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
        if not ncdflib.is_valid_name(name):
            raise ValueError("Not a valid variable name: %s", str(name))
        if name in self.variables:
            raise ValueError("Variable named '%s' already exists" % (name))
        if not isinstance(dim, tuple):
            raise TypeError('dim must be a tuple')
        if not all([isinstance(d, basestring) and (d in self.dimensions)
                for d in dim]):
            bad = [d for d in dim if not isinstance(d, basestring) or
                                     (d not in self.dimensions)]
            raise ValueError("the following dim(s) are not valid " +
                    "dimensions of this object: %s" % bad)
        if attributes is None:
            attributes = OrderedDict()

        record = (self.record_dimension in dim)
        # Check that the record dimension does not appear anywhere
        # other than dim[0]. This is tricky!
        if record and (list(reversed(dim)).index(self.record_dimension) !=
                len(dim) - 1):
            raise ValueError, ("only the 0th dimension of a variable is " +
                    "allowed to be unlimited")

        # Check dtype
        if dtype is not None:
            if not isinstance(dtype, np.dtype):
                try:
                    dtype = np.dtype(dtype)
                except:
                    TypeError, "Can not convert dtype to a numpy dtype object"
            # byteorder must be normalized (here we choose native byte
            # order as the standard) for hashing and dict lookup to work
            if dtype.newbyteorder('=') not in ncdflib.TYPEMAP:
                raise TypeError("dtype %s is not compatible with netCDF" %
                        (str(dtype)))
        if data is None:
            if dtype is None:
                raise ValueError("must provide dtype if data is None")
            # If this is the first record variable to be created, then
            # data must be provided to determine self.numrecs
            if record and (self.numrecs is None):
                raise ValueError("must provide data when creating " +
                        "the first record variable of an object")
            # Check the fillval
            if not '_FillValue' in attributes:
                attributes['_FillValue'] = ncdflib.FILLMAP[dtype]
            data = np.empty(
                    tuple(self.numrecs if d == self.record_dimension
                    else self.dimensions[d] for d in dim),
                    dtype=dtype)
            data.fill(attributes['_FillValue'])

        # np.asarray will turn just about anything into an array (objects,
        # string, even np.asarray(np) works!  In turn the hope is that
        # any arrays that consist of invalid dtypes will have been caught
        # above.
        data = _coerce_type(np.asarray(data))
        if (dtype is not None) and (data.dtype != dtype):
            raise ValueError, "dtype and data arguments are incompatible"

        # Check that data is compatible with the length of the record
        # dimension
        if record:
            if self.numrecs is None:
                # If no other record variables exist, then the length
                # of this new variable sets the number of records for
                # the entire object
                try:
                    assert data.shape[0] >= 0
                    ncdflib.pack_int(data.shape[0])
                except:
                    raise ValueError, ("record length must be a " +
                            "non-negative signed 32-bit integer")
            else:
                # Check that the length of data along the unlimited dimension
                # matches the lengths of other record variables.
                if data.shape[0] != self.numrecs:
                    raise ValueError, ("data length along the unlimited " +
                            "dimension is inconsistent with the number of " +
                            "records in the other record variables")
        # Check that data is compatible with the lengths of the non-record
        # dimensions
        if not len(dim) == data.ndim:
            raise ValueError("number of dimensions don't match dim:%d data:%d"
                             % (len(dim), data.ndim))
        for (i, d) in enumerate(dim):
            if d == self.record_dimension:
                continue
            if data.shape[i] != self.dimensions[d]:
                raise ValueError, ("data array shape does not match the " +
                        "length of dimension '%s'") % (d)
        # Check that data is 1-dimensional if name matches a dimension name
#        if (name in self.dimensions) and (data.ndim != 1):
#            raise ValueError, "coordinate variables must be 1-dimensional"
        self.variables[name] = Variable(
                dim=dim, data=data, attributes=attributes)
        # Update self.numrecs if this new variable is the first
        # created record variable
        if (self.record_dimension in self.variables[name].dimensions) and\
                (self.numrecs is None):
            object.__setattr__(self, 'numrecs', self.variables[name].shape[0])
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
        if not ncdflib.is_valid_name(name):
            raise ValueError, "Not a valid variable name"
        if name in self.dimensions:
            raise ValueError, "Dimension named '%s' already exists" % (name)
        if name in self.variables:
            raise ValueError, "Variable named '%s' already exists" % (name)
        data = _coerce_type(np.asarray(data))
        if data.ndim != 1:
            raise ValueError, "data must be 1-dimensional (vector)"
        if data.dtype.newbyteorder('=') not in ncdflib.TYPEMAP:
            raise TypeError, ("Can not store data type %s" % (str(data.dtype)))
        if not isinstance(record, bool):
            raise TypeError, "record argument must be bool"
        if record and self.record_dimension is not None:
            raise ValueError, "Only one unlimited dimension is allowed"
        # We need to be cleanly roll back the effects of
        # create_dimension if create_variable fails, otherwise we will
        # end up in a partial state.
        old_dims = {
                'record_dimension': self.record_dimension,
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
        self.create_coordinate(name=name,
                               data=coordinate.data,
                               record=record,
                               attributes=coordinate.attributes)

    def create_pickled_variable(self, name, dim, data, attributes=None):
        """
        Creates a new variable from an ndarray of pickle-able objects by
        serializing them and storing the strings as a matrix of characters.

        Parameters
        ----------
        name : string
            The name of the new variable. An exception will be raised
            if the object already has a variable with this name. name
            must satisfy netCDF-3 naming rules. If name equals the name
            of a dimension, then the new variable is treated as a
            coordinate variable and must be 1-dimensional.
        dim : tuple
            The dimension of the new variable. Elements must be dimensions of
            the object, and due to current limitations in our pickling structure
            dim must be length one.
        data : iterable
            Data to populate the new variable. Each iterate of data is
            pickled and stored as a string.
        attributes : dict_like or None, optional
            Attributes to assign to the new variable. Attribute names
            must be unique and must satisfy netCDF-3 naming rules. If
            None (default), an empty attribute dictionary is
            initialized.
        """
        logging.info("TODO: have object pickling happen internally")
        assert len(dim) == 1
        # pickle the objects and store them in a char matrix, if the object
        # is not pickle-able it will be caught here.
        strings = [pickle.dumps(x) for x in data]
        max_len = max(len(x) for x in strings)
        data = np.zeros((len(strings), max_len), dtype='c')
        for i, s in enumerate(strings):
            data[i,:len(s)] = s

        # we need to create a dimension for the string length
        strlen_name = "%s_strlen" % name
        self.create_dimension(strlen_name, data.shape[1])
        self.create_variable(name,
                             (dim[0], strlen_name),
                             dtype='c',
                             data = data,
                             attributes=attributes)

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
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

    def view(self, ind, dim=None):
        """Return a new object whose contents are a view of a slice from the
        current object along a specified dimension

        Parameters
        ----------
        ind : integer
            The index of the values to extract. Unlike 'take' the
            possible values of ind are restricted to those that can be
            represented by a slice object.  This is an artifact of
            numpy's designation between a slice and a fancy slice
            (which returns a copy not a view).  If ind is an integer it
            is converted to an appropriate length-1 slice.
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
                raise ValueError, ("object does not have a record dimension " +
                                   "to default to")
            dim = self.record_dimension
        if not isinstance(dim, basestring):
            raise TypeError, "dim must be a dimension name (string)"
        if dim not in self.dimensions:
            raise ValueError, "Object does not have a dimension '%s'" % (dim)
        # Convert ind to a numpy array of integer indices. This step
        # can be computationally expensive, but it simplifies
        # error-checking and code maintenance
        if dim == self.record_dimension:
            if self.numrecs is None:
                raise ValueError, ("Can not take data along the record " +
                        "dimension because the number of records is " +
                        "undetermined, this implies the record dimension" +
                        "has no data.")

        # this should be already enforced elsewhere but just in case
        if not self.variables[dim].data.ndim == 1:
            raise ValueError, "Coordinate is not 1-dimensional"

        if isinstance(ind, int):
            # This type test against int (not a more general integer-like base
            # class that includes long int) works because these Data objects are
            # restricted to dimension lengths are less than 2 ** 31
            ind = slice(ind, ind+1)

        if not isinstance(ind, slice):
            raise IndexError("view requires ind to be a slice or integer")
        new_size = np.arange(self.variables[dim].data.size)[ind].size
        if new_size == 0:
            raise IndexError("view would result in an empty coordinate")

        # Create a new Data instance
        obj = Data()
        # Copy dimensions, modifying dim
        for (name, length) in self.dimensions.iteritems():
            if (length is not None) and (name == dim):
                obj.create_dimension(name, new_size)
            else:
                obj.create_dimension(name, length)
        # Create the views of variables
        for (name, var) in self.variables.iteritems():
            if dim in var.dimensions:
                obj.variables[name] = var.view(dim, ind)
            else:
                obj.variables[name] = var
        # Map the attributes, this intentionally does not copy.
        object.__setattr__(obj, 'attributes', self.attributes)
        return obj

    def take(self, ind, dim=None):
        """Return a new object whose contents are sliced from the
        current object along a specified dimension

        Parameters
        ----------
        ind : array_like
            The indices of the values to extract. ind is interpreted
            acccording to numpy conventions; i.e., slices and boolean
            indices work.
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
                raise ValueError("object does not have an unlimited " +
                        "dimension; a non-None dimension must be specified")
            dim = self.record_dimension
        if not isinstance(dim, basestring):
            raise TypeError("dim must be a dimension name (string)")
        if dim not in self.dimensions:
            raise ValueError("Object does not have a dimension '%s'" % dim)
        if (dim == self.record_dimension) and (self.numrecs is None):
           raise ValueError("Can not take data along the record " +
                        "dimension because the number of records is " +
                        "undetermined, this implies the record dimension" +
                        "has no data.")
        if dim == self.record_dimension:
            dim_length = self.numrecs
        else:
            dim_length = self.dimensions[dim]
        try:
            ind = np.arange(dim_length)[ind]
        except:
            # this logs so we know where the cause of improper indexing, but
            # also raises the original exception
            msg = "ind is not a valid index into dimension '%s'" % (dim)
            logging.error(msg)
            raise
        # Create a new Data instance
        obj = self.__class__()
        # Copy dimensions, modifying dim
        for (name, length) in self.dimensions.iteritems():
            if (length is not None) and (name == dim):
                obj.create_dimension(name, ind.size)
            else:
                obj.create_dimension(name, length)
        # Copy takes of variables
        for (name, var) in self.variables.iteritems():
            if dim in var.dimensions:
                obj.variables[name] = var.take(dim, ind)
            else:
                obj.variables[name] = copy.deepcopy(var)
        # Maintain consistency of obj.numrecs. Ordinarily, the
        # create_variable method does this housekeeping automatically,
        # but we skipped this by directly assigning to obj.variables
        if dim == self.record_dimension:
            object.__setattr__(obj, 'numrecs', ind.size)
        else:
            object.__setattr__(obj, 'numrecs', self.numrecs)
        # Copy attributes
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

    def renamed(self, **kwds):
        """
        Returns a copy of the current object with variables and dimensions
        reanmed according to the arguments passed via **kwds

        Parameters
        ----------
        **kwds : arbitrary named arguments
            kwds should be in the form: old_name='new_name'
        """
#        for name, d in self.dimensions.iteritems():
#            if name in self.variables and not name in self.coordinates:
#                raise ValueError(("Renaming assumes that only coordinates " +
#                                 "have both a dimension and variable under " +
#                                 "the same name.  In this case it appears %s " +
#                                 "has a dim and var but is not a coordinate")
#                                 % name)

        new_names = dict((name, name) for name, d in self.dimensions.iteritems())
        new_names.update(dict((name, name) for name, v in self.variables.iteritems()))

        for k, v in kwds.iteritems():
            if not k in new_names:
                raise ValueError("Cannot rename %s because it does not exist" % k)
        new_names.update(kwds)

        obj = self.__class__()
        # if a dimension is a new one it gets added, if the dimension already
        # exists we confirm that they are identical (or throw an exception)
        for (name, length) in self.dimensions.iteritems():
            obj.create_dimension(new_names[name], length)
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
                length = length or other.numrecs
                cur_length = self.dimensions[name] or self.numrecs
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
                                     % name, self[name].dimensions, other[name].dimensions)
                if not np.all(numpylib.naneq(self[name].data, other[name].data)):
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
            raise TypeError, "var must be a bounded sequence"
        if not all([isinstance(v, basestring) and (v in self.variables)
                for v in var]):
            raise ValueError, ("each variable name in var must belong to " +
                    "the object's variables dictionary")
        # Create a new Data instance
        obj = Data()
        # Copy relevant dimensions
        dim = reduce(or_, [set(self.variables[v].dimensions) for v in var])
        # Create dimensions in the same order as they appear in self.dimension
        for d in self.dimensions:
            if d in dim:
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
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

    def iterator(self, dim=None):
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
        if dim is None:
            if self.record_dimension is None:
                raise ValueError("object has no record dimension")
            if self.numrecs is None:
                raise ValueError("Can not iterate over record dimension " +
                        "because it there are no record values. (numrecs == 0)")
            dim = self.record_dimension
        if dim not in self.dimensions:
            raise ValueError("dimension is not found")

        coord = (dim in self.coordinates)
        # iterate along dim returning takes at each iterate
        for i in range(self.variables[dim].data.size):
            obj = self.take([i], dim)
            if coord:
                yield (obj[dim], obj)
            else:
                yield (None, obj)

    def iterarray(self, var, dim=None):
        """Iterator along a data dimension returning the coresponding slices
        of a variable.

        Return an iterator yielding (scalar, ndarray) pairs
        that are singleton along the specified dimension.  While iterator is
        more general this method has less overhead and in turn should be
        considerably faster.

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
        if dim is None:
            if self.record_dimension is None:
                raise ValueError, "object has no record dimension"
            if self.numrecs is None:
                raise ValueError, ("Can not iterate over record dimension " +
                        "because it there are no record values. (numrecs == 0)")
            dim = self.record_dimension
        if dim not in self.dimensions:
            raise ValueError, "dimension %s is not found" % dim

        dim_axis = list(self.variables[var].dimensions).index(dim)
        slicer = [slice(None)] * self.variables[var].data.ndim

        # We determine the size of the dim we're about to iterate over by
        # first checking to see if it has a corresponding variable.  If it does
        # the variables length is used, otherwise we infer the length from
        # the dimensions (or numrecs)
        coord = (dim in self.coordinates)
        if coord:
            # this should be enforced elsewhere, but just in case
            assert self.variables[dim].data.ndim == 1
            n = self.variables[dim].data.size
        else:
            # the dimension is not a coordinate so we need to infer the dim
            # size from the dimensions (or numrecs)
            if dim is self.record_dimension:
                n = self.numrecs
            else:
                n = self.dimensions[dim]
        # iterate over dim returning
        for i in xrange(n):
            slicer[dim_axis] = slice(i, i+1, None)
            v = self.variables[var].data[slicer]
            if coord:
                yield self.variables[dim].data[i], v
            else:
                yield i, v

    def interpolate(self, var, **points):
        # ASSUMPTION, ALL COORDINATES ARE SORTED
        if len(points) != len(self[var].dimensions):
            msg = "interpolation requires specifying all dimensions"
            msg = msg + " required: %s -- given:%s"
            raise ValueError(msg % (str(self[var].dimensions), str(points.keys())))
        # TODO remove this restriction
        #assert self[var].data.ndim <= 2

        from bisect import bisect
        def neighbors(coord, val):
            assert self[coord].data.ndim == 1
            data = self[coord].data
            j = bisect(data, val)
            if j == 0 or j == data.size:
                raise ValueError("value of %6.3f is outside the range of %s"
                                 % (val, coord))
            i = j-1
            if data[i] == val:
                return [i, i], 1.0
            else:
                alpha = np.abs(val - data[j])/np.abs(data[i] - data[j])
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
            ret = np.dot(ret, [alpha, 1.-alpha])
            if ret.ndim == ndim:
                raise ValueError("dot did not reduce the dimension")
                # for some reason np.dot doesn't always reduce the dimension
                # so we need to do it manually from time to time
                ret = ret[..., 0]
        return ret


class Variable(object):
    """A class that wraps metadata around a numpy ndarray"""

    def __init__(self, dim, data=None, dtype=None, shape=None, attributes=None):
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
        dtype : dtype_like, optional
            A numpy dtype object, or an object that can be coerced to a
            dtype.  If the data argument is not None, then this
            argument must equal the dtype of data. If the data argument
            is None and dtype is None, then dtype is the numpy default
            (typically 64-bit float).
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
        # initialize attributes
        if attributes is None:
            attributes = OrderedDict()
        # Check dim
        if not isinstance(dim, tuple):
            raise TypeError("dim must be a tuple")
        if not all(map(ncdflib.is_valid_name, dim)):
            raise ValueError(
                    "elements of dim must be valid netCDF dimension names")
        # Check dtype
        if dtype is not None:
            if not isinstance(dtype, np.dtype):
                try:
                    dtype = np.dtype(dtype)
                except:
                    TypeError, "Can not convert dtype to a numpy dtype object"
            # byteorder must be normalized (here we choose native byte
            # order as the standard) for hashing and dict lookup to work
            if dtype.newbyteorder('=') not in ncdflib.TYPEMAP:
                raise TypeError("dtype %s is not compatible with netCDF" %
                        (str(dtype)))
        # Check shape
        if shape is not None:
            if not isinstance(shape, tuple):
                raise TypeError, "shape must be a tuple"
            if len(shape) != len(dim):
                raise ValueError, "shape and dim must have the same length"
            try:
                assert all([ncdflib.pack_int(s) and (s > 0) for s in shape])
            except:
                raise ValueError("shape tuple must contain positive " +
                        "values that can be represented as "
                        "32-bit signed integers")
        # Check data
        if data is None:
            if (shape is None) or (dtype is None):
                raise ValueError(
                        "shape and dtype must be specified if data is None")
            # Populate with np.empty if no data is provided
            if not '_FillValue' in attributes:
                attributes['_FillValue'] = ncdflib.FILLMAP[dtype]
            data = np.empty(shape, dtype=dtype)
            data.fill(attributes['_FillValue'])
        try:
            data = np.asarray(data)
        except:
            raise TypeError, "data must be a numpy array or array-like"
        data = _coerce_type(data)
        if (shape is not None) and (data.shape != shape):
            raise ValueError, "shape and data arguments are incompatible"
        if (dtype is not None) and (data.dtype != dtype):
            raise ValueError, "dtype and data arguments are incompatible"
        if len(dim) != data.ndim:
            raise ValueError, "dim and data arguments are incompatible"
        # Initialize attributes
        object.__setattr__(self, 'dimensions', dim)
        object.__setattr__(self, 'data', data)
        # self.attributes is an OrderedDict() to ensure consistent
        # order of writing to file
        object.__setattr__(self, 'attributes', OrderedDict())
        if attributes is not None:
            if not isinstance(attributes, OrderedDict):
                attributes = OrderedDict(attributes)
            for (attr, value) in attributes.iteritems():
                self.set_attribute(attr, value)

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
        super(type(a), self).__setattr method to bypass."""
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

    def __deepcopy__(self, memo=None):
        """
        Returns a deep copy of the current object.

        memo does nothing but is required for compatability with copy.deepcopy
        """
        obj = self.__class__(dim=self.dimensions,
                data=self.data.copy())
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

    def __eq__(self, other):
        if self.dimensions != other.dimensions or not np.all(
          numpylib.naneq(self.data, other.data)):
            return False
        if not set(self.attributes.keys()) == set(other.attributes.keys()):
            return False
        for key in self.attributes:
            if not np.all(numpylib.naneq(self.attributes[key],
                                         other.attributes[key])):
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

    def has_attribute(self, attr):
        return attr in self.attributes

    def get_attribute(self, attr):
        """Get the value of a user-defined attribute"""
        if not self.has_attribute(attr):
            raise KeyError, "attribute not found"
        return self.attributes[attr]

    def set_attribute(self, attr, value):
        """Set the value of a user-defined attribute"""
        if not ncdflib.is_valid_name(attr):
            raise ValueError, "Not a valid attribute name"
        # Strings get special handling because netCDF treats them as
        # character arrays. Everything else gets coerced to a numpy
        # vector. netCDF treats scalars as 1-element vectors. Arrays of
        # non-numeric type are not allowed.
        if isinstance(value, basestring):
            try:
                ncdflib.pack_string(value)
            except:
                raise ValueError, (
                        "Not a valid string value for a netCDF attribute")
        else:
            try:
                value = np.atleast_1d(np.asarray(value))
            except:
                raise ValueError, (
                        "Not a valid vector value for a netCDF attribute")
            if value.ndim > 1:
                raise ValueError, ("netCDF attributes must be vectors " +
                        "(1-dimensional)")
            value = _coerce_type(value)
            # Coerce dtype to native byte order for hashing/dictionary lookup
            if value.dtype.newbyteorder('=') not in ncdflib.TYPEMAP:
                # A plain string attribute is okay, but an array of
                # string objects is not okay!
                raise ValueError, "Can not convert to a valid netCDF type"
        self.attributes[attr] = value

    def del_attribute(self, attr):
        if not self.has_attribute(attr):
            raise KeyError, "attribute not found"
        del self.attributes[attr]

    def update_attributes(self, *args, **kwargs):
        """Set multiple user-defined attributes with a mapping object
        or an iterable of key/value pairs"""
        # Try it on a dummy copy first so that we don't end up in a
        # partial state
        dummy = self.__class__(
               dim=('dummy',), data=np.array([0]), attributes=self.attributes.copy())
        try:
            attributes = OrderedDict(*args, **kwargs)
            for (k, v) in attributes.iteritems():
                # set_attribute method does checks and type conversions
                dummy.set_attribute(k, v)
            del dummy
            # If we can set the requested attributes without any
            # complaints, then we can apply it to the real object.
            for (k, v) in attributes.iteritems():
                self.set_attribute(k, v)
        except:
            raise

    def clear_attributes(self, **kwargs):
        """Delete all user-defined attributes"""
        object.__setattr__(self, 'attributes', OrderedDict())

    def view(self, dim, ind):
        """Return a new Variable object whose contents are a view of the object
        sliced along a specified dimension.

        Parameters
        ----------
        ind : slice
            The index of the values to extract.  Unlike 'take' the possible
            values of ind are restricted to those that can be represented by
            a slice object.  This is an artifact of numpy's designation between
            a slice and a fancy slice (which returns a copy not a view).  If ind
            is an integer it is converted to an appropriate slice.
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
        if not isinstance(dim, basestring):
            raise TypeError("dim must be a dimension name (string)")
        if dim not in self.dimensions:
            raise ValueError("Object does not have a dimension '%s'" % dim)
        # When dim appears repeatededly in self.dimensions, using the index()
        # method gives us only the first one, which is the desired behavior
        axis = list(self.dimensions).index(dim)
        if isinstance(ind, int):
            ind = slice(ind, ind+1)
        elif not isinstance(ind, slice):
            raise IndexError("view requires ind to be a slice or integer")

        # create a list of null slices, then fill in the appropriate axis
        slices = [ind if i == axis else slice(None)
                  for i in range(self.data.ndim)]
        # apply the slice and copy attributes
        obj = Variable(dim=self.dimensions, data=self.data[slices])
        object.__setattr__(obj, 'attributes', self.attributes)
        return obj

    def take(self, dim, ind):
        """Return a new Variable object whose contents are sliced from
        the current object along a specified dimension

        Parameters
        ----------
        ind : array_like
            The indices of the values to extract. ind is interpreted
            acccording to numpy conventions; i.e., slices and boolean
            indices work.
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
        if not isinstance(dim, basestring):
            raise TypeError, "dim must be a dimension name (string)"
        if dim not in self.dimensions:
            raise ValueError, "Object does not have a dimension '%s'" % (dim)
        # When dim appears repeatededly in self.dimensions, using the index()
        # method gives us only the first one, which is the desired behavior
        axis = list(self.dimensions).index(dim)
        obj = Variable(dim=self.dimensions,
                data=self.data.take(ind, axis=axis))
        for attr in self.attributes:
            # Attribute values are either numpy vectors or strings; the
            # former have a copy() method, while the latter are
            # immutable
            if hasattr(self.attributes[attr], 'copy'):
                obj.set_attribute(attr, self.attributes[attr].copy())
            else:
                obj.set_attribute(attr, self.attributes[attr])
        return obj

if __name__ == "__main__":
    obj = Data()
    obj.create_coordinate('dim1', np.arange(10))
    obj.create_coordinate('dim2', np.arange(20))
    obj.create_variable('var1',
                        ('dim1', 'dim2'),
                        data=np.random.normal(size=(10, 20)))

    val = obj.interpolate('var1', dim1=3.5, dim2=2.5)
    expected = np.mean(obj['var1'].data[3:5, 2:4])
    np.testing.assert_almost_equal(val, expected)

    val = obj.interpolate('var1', dim1=3, dim2=2.5)
    expected = np.mean(obj['var1'].data[3, 2:4])
    np.testing.assert_almost_equal(val, expected)

    val = obj.interpolate('var1', dim1=3.5, dim2=2)
    expected = np.mean(obj['var1'].data[3:5, 2])
    np.testing.assert_almost_equal(val, expected)

    val = obj.interpolate('var1', dim1=2.2, dim2=3.9)
    expected = np.dot(np.dot(obj['var1'].data[2:4, 3:5].T, [0.8, 0.2]), [0.1, 0.9])
    np.testing.assert_almost_equal(val, expected)
