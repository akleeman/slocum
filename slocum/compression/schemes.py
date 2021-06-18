"""
schemes.py

This module handles the various variable definitions that are used
to compress and expand variables.
"""
import xarray as xra
import logging
import warnings
import itertools
import numpy as np

from . import tinylib

from slocum.lib import units, angles

_default_dim_order = ('time', 'realization', 'longitude', 'latitude')

def infer_shape(dim_order, coords):
    """
    Takes an assumed dimension order and a set of known coordinates
    and infers the corresponding dims, shape.
    """
    def iter_available():
        for d in dim_order:
            if d in coords:
                assert coords[d].ndim == 1
                yield d, coords[d].size

    pairs = list(iter_available())
    return (tuple([x for x, _ in pairs]),
            tuple([y for _, y in pairs]))


def swap_dims(var, dim_order):
    """
    A convenience function that transposes the data to meet the
    expectations of the compression algorithms.
    """
    if any(d not in dim_order for d in var.dims):
        unexpected = [d for d in var.dims if d not in dim_order]
        raise ValueError("Unexpected dimension(s) %s in %s"
                         % (','.join(unexpected), var.name))
    var = var.transpose(*[d for d in dim_order if d in var.dims])
    return var


class AbstractVariable(object):
    """
    An abstract class that mostly exists just to make the API obvious.
    """
    attrs = {}
    dtype = np.float32
    dim_order = _default_dim_order

    # this needs to be defined
    variable_name = None

    def required_variables(self):
        return [self.variable_name]

    def compress(self, ds):
        # should pull any variables out of the dataset, ds, and compress
        raise NotImplementedError("Define compress()")

    def decompress(self, compressed, coords):
        # extracts any variables out of compressed.  coords contains any
        # coordinates that have already been decompressed which help
        # determine the shape of the resulting data.
        raise NotImplementedError("Define decompress()")


class TrivialCoordinate(AbstractVariable):
    """
    Holds a trivial variable (ie, a coordinate that is equivalent
    to np.arange(n)).
    """
    def __init__(self, variable_name):
        self.variable_name = variable_name

    def compress(self, ds):
        compressed = tinylib.small_trival_variable(ds[self.variable_name])
        return compressed

    def decompress(self, compressed, coords):
        data = tinylib.expand_trival_variable(compressed)
        ds = coords.copy(deep=True)
        ds[self.variable_name] = (self.variable_name, data, self.attrs)
        return ds


class SmallVariable(AbstractVariable):
    """
    Small Variables are compressed by rounding to the nearest
    least significant digit, then using zlib to compress the
    data.
    """
    # These need to be set in implementing classes
    units = None
    variable_name = None

    def __init__(self, variable_name, units,
                 least_significant_digit=2):
        self.variable_name = variable_name
        self.units = units
        self.least_significant_digit = least_significant_digit

    def normalize(self, ds):
        ds = ds.copy(deep=True)
        ds[self.variable_name] = swap_dims(ds[self.variable_name],
                                           self.dim_order)
        # for now this only works for coordinates
        assert ds[self.variable_name].attrs['units'] == 'units'
        return ds

    def compress(self, ds):
        ds = self.normalize(ds)
        data = ds[self.variable_name].values.astype(self.dtype)
        small = tinylib.small_array(data, self.least_significant_digit)
        logging.debug("compressed %d values to %d bytes" %
                      (data.size, len(small['packed_array'])))
        return small['packed_array']

    def decompress(self, compressed, coords):
        # infer the shape of the resulting data using any data
        # that has already been decompressed (coords)
        if self.variable_name in self.dim_order:
            dims = (self.variable_name, )
            shape = -1
        else:
            dims, shape = infer_shape(self.dim_order, coords)
        data = tinylib.expand_small_array(compressed,
                                          self.dtype,
                                          self.least_significant_digit)
        attrs = self.attrs.copy()
        attrs.update({'units': self.units})
        # for now this only works with coordinates
        ds = coords.copy(deep=True)
        ds[self.variable_name] = (dims, data.reshape(shape), attrs)
        return ds


class SmallCoordinate(SmallVariable):

    def __init__(self, *args, **kwdargs):
        super(SmallCoordinate, self).__init__(*args, **kwdargs)
        self.dim_order = [self.variable_name]


class TinyVariable(AbstractVariable):
    """
    This compression scheme bins the variable into, typically, 16 bins
    then compresses the data into 4 bits.
    """
    bits = 4
    dim_order = _default_dim_order

    # These need to be set in implementing classes
    variable_name = None
    units = None
    bins = None

    def __init__(self, variable_name, units, bins,
                 wrap=False, wrap_val=None):
        self.variable_name = variable_name
        self.units = units
        self.bins = bins
        if wrap:
            assert bins.size <= 2 ** self.bits
            # if wrap is True make sure wrap_val is set
            assert wrap_val is not None
        else:
            assert bins.size < 2 ** self.bits
        self.wrap = wrap
        self.wrap_val = wrap_val

    def normalize(self, ds):
        ds = ds.copy(deep=True)
        units.convert_units(ds[self.variable_name], self.units)
        # make sure modification in place happened
        assert ds[self.variable_name].attrs['units'] == self.units
        ds[self.variable_name] = swap_dims(ds[self.variable_name],
                                           self.dim_order)

        return ds

    def compress(self, ds, bins=None):
        bins = self.bins if bins is None else bins
        ds = self.normalize(ds)
        # convert the wind speeds to a beaufort scale and store them
        tiny = tinylib.tiny_array(ds[self.variable_name].values,
                                  bits=self.bits,
                                  divs=bins,
                                  wrap=self.wrap)
        # return a dictionary of packed strings
        logging.debug("compressed %d values to %d bytes" %
                      (ds[self.variable_name].size,
                       len(tiny['packed_array'])))
        return tiny['packed_array']

    def decompress(self, compressed, coords, bins=None):
        bins = self.bins if bins is None else bins
        # infer the shape of the resulting data using any data
        # that has already been decompressed (coords)
        dims, shape = infer_shape(self.dim_order, coords)
        # expand the speed and direction
        data = tinylib.expand_array(compressed,
                                    shape=shape,
                                    bits=self.bits,
                                    divs=bins,
                                    dtype=self.dtype,
                                    wrap_val=self.wrap_val)
        # store the result in a dataset and return
        ds = coords.copy(deep=True)
        ds[self.variable_name] = (dims, data, {'units': self.units})
        return ds


class TinyDirection(TinyVariable):
    """
    Stores a direction variable using 4 bits.
    """
    def __init__(self, variable_name):
        # 'S' sits between _direction_bins[-1] and _direction_bins[0]
        bins = np.linspace(-15 * np.pi/16., 15 * np.pi/16., 16)
        super(TinyDirection, self).__init__(variable_name,
                                            units='radians',
                                            bins=bins,
                                            wrap=True,
                                            wrap_val=np.pi)


class TimeCoordinate(AbstractVariable):
    """
    Compress a time variable by creating an augmented array that
    starts with the reference time (in ordinal) then holds diffs,
    since these are often the same the result compresses well.
    """
    variable_name = 'time'

    def __init__(self, variable_name):
        self.variable_name = variable_name

    def normalize(self, ds):
        # this will issue a warning about overwriting data types
        # that we don't care about.
        with warnings.catch_warnings(record=True):
            ds = xra.decode_cf(ds.copy(deep=True))
        return ds

    def compress(self, ds):
        ds = self.normalize(ds)
        tiny = tinylib.small_time(ds[self.variable_name])
        logging.debug("compressed %d values to %d bytes" %
                      (ds[self.variable_name].size,
                       len(tiny['packed_array'])))
        return tiny['packed_array']

    def decompress(self, compressed, coords):
        data, units = tinylib.expand_small_time(compressed)
        ds = coords.copy(deep=True)
        # for now this only works with coordinates
        attrs = self.attrs.copy()
        attrs['units'] = units
        ds[self.variable_name] = (self.variable_name, data, attrs)
        return xra.decode_cf(ds)


class CombinedVariable(AbstractVariable):

    def __init__(self, variables):
        self.variables = variables

    def required_variables(self):
        return itertools.chain(*[x.required_variables()
                                 for x in self.variables])

    def normalize(self, ds):
        # this performs variable specific normalization
        for v in self.variables:
            if v.variable_name in ds:
                ds = v.normalize(ds)
        return ds

    def compress(self, ds):
        individual = [x.compress(ds) for x in self.variables]
        # for now we assume that each variable is the same
        # size once compressed
        assert len(set([len(x) for x in individual])) == 1
        return ''.join(individual)

    def decompress(self, compressed, coords):
        out = coords.copy(deep=True)
        n = len(compressed) / float(len(self.variables))
        assert int(n) == n
        n = int(n)
        parts = list(map(''.join, list(zip(*[iter(compressed)] * n))))
        for x, p in zip(self.variables, parts):
            out.update(x.decompress(p, coords))
        return self.normalize(out)


class VelocityVariable(AbstractVariable):
    """
    Some variables such as wind and current are velocities so actually
    require two different compression schemes, one for speed and one
    for direction.  This class handles that case.
    """
    dtype = np.float32
    bits = 4

    def __init__(self, u_name, v_name, variable_name, speed_bins,
                 speed_bin_names=None, units='m/s',
                 direction_orientation='from'):
        self.u_name = u_name
        self.v_name = v_name
        self.variable_name = variable_name
        self.units = units
        self.speed_bins = speed_bins
        # the bin names should name the center values, so
        # should have length one shorter than the bins.
        if speed_bin_names is not None:
            assert len(speed_bin_names) == speed_bins.size - 1
        self.speed_bin_names = speed_bin_names
        self.speed_name = '%s_speed' % self.variable_name
        assert direction_orientation in ['from', 'to']
        self.direction_name = '%s_%s_direction' % (self.variable_name,
                                                   direction_orientation)
        self.direction_orientation = direction_orientation

        self.speed = TinyVariable(self.speed_name, units, self.speed_bins)
        self.direction = TinyDirection(self.direction_name)
        # these aren't used for compression but they're nice to
        # have around.
        self.uvel = TinyVariable(self.u_name, units, self.speed_bins)
        self.vvel = TinyVariable(self.v_name, units, self.speed_bins)

    def required_variables(self):
        return [self.u_name, self.v_name]

    def normalize(self, ds):
        # avoid modification in place
        ds = ds.copy(deep=True)

        # this performs variable specific normalization
        for v in [self.speed, self.direction, self.uvel, self.vvel]:
            if v.variable_name in ds:
                ds = v.normalize(ds)

        # decide if u,v should be converted to speed direction or vice versa
        if self.u_name in ds and self.speed_name not in ds:
            assert self.u_name in ds
            assert self.v_name in ds
            assert self.speed_name not in ds
            assert self.direction_name not in ds
            # convert to magnitudes
            speed, direction = angles.vector_to_radial(ds[self.u_name],
                                       ds[self.v_name],
                                       orientation=self.direction_orientation)
            ds[self.speed_name] = speed
            assert (ds[self.u_name].attrs['units'] ==
                    ds[self.v_name].attrs['units'])
            ds[self.speed_name].attrs['units'] = ds[self.u_name].attrs['units']
            ds[self.direction_name] = direction
            ds[self.direction_name].attrs['units'] = 'radians'
        elif self.u_name not in ds and self.speed_name in ds:
            uwnd, vwnd = angles.radial_to_vector(ds[self.speed_name],
                                        ds[self.direction_name],
                                        orientation=self.direction_orientation)
            ds[self.u_name] = uwnd
            ds[self.u_name].attrs['units'] = ds[self.speed_name].attrs['units']
            ds[self.v_name] = vwnd
            ds[self.v_name].attrs['units'] = ds[self.speed_name].attrs['units']
        elif self.u_name not in ds and self.speed_name not in ds:
            raise ValueError("Couldn't find %s or %s" %
                             (self.u_name, self.speed_name))
        else:
            # Found both vector and radial representations so we
            # check for consistency.
            speed, direction = angles.vector_to_radial(ds[self.u_name].values,
                                        ds[self.v_name].values,
                                        orientation=self.direction_orientation)
            # MaskedVelocity uses fill values, so here we only compare non-nan
            # speeds.
            finite = np.isfinite(speed)
            np.testing.assert_almost_equal(speed[finite],
                                           ds[self.speed_name].values[finite],
                                           decimal=4,
                                           err_msg=("stored and derived radial "
                                                    " representation don't match"),
                                           verbose=True)

        return ds

    def compress(self, ds, speed_bins=None):
        ds = self.normalize(ds)
        # convert the wind speeds to a beaufort scale and store them
        tiny_speed = self.speed.compress(ds, bins=speed_bins)
        tiny_direction = self.direction.compress(ds)
        tiny = ''.join([tiny_speed, tiny_direction])
        # return a dictionary of packed strings
        logging.debug("compressed %d values to %d bytes" %
                      (ds[self.speed_name].size * 2,
                       len(tiny)))
        return tiny

    def decompress(self, compressed, coords,
                   speed_bins=None):
        n = len(compressed) / 2.
        assert int(n) == n
        n = int(n)
        tiny_speed = compressed[:n]
        tiny_direction = compressed[n:]
        # expand the speed and direction
        speed = self.speed.decompress(tiny_speed, coords, bins=speed_bins)
        directions = self.direction.decompress(tiny_direction, coords)
        # store the result in a dataset and return
        out = coords.copy(deep=True)
        out[self.speed_name] = speed[self.speed_name]
        out[self.direction_name] = directions[self.direction_name]
        return self.normalize(out)


class MaskedVelocity(VelocityVariable):

        def __init__(self, *args, **kwdargs):
            super(MaskedVelocity, self).__init__(*args, **kwdargs)
            # otherwise near zero values could be considered NaNs
            assert np.min(self.speed_bins) == 0.
            self.mask_value = -1.
            self.masked_speed_bins = np.concatenate([[self.mask_value],
                                                     self.speed_bins])

        def compress(self, ds):
            ds = self.normalize(ds)
            # tinylib can't handle NaNs so we set them to negative
            # numbers which can be unpacked after decompression.
            mask = np.isnan(ds[self.speed_name].values)
            ds[self.speed_name].values[mask] = self.mask_value
            # this doesn't matter, it just can't be nan
            ds[self.direction_name].values[mask] = 0.
            # then run compression as before.  Note that normalize
            # gets run twice so make sure it is idempotent.
            super_compress = super(MaskedVelocity, self).compress
            return super_compress(ds, speed_bins=self.masked_speed_bins)


        def decompress(self, compressed, coords):
            super_decompress = super(MaskedVelocity, self).decompress
            ds = super_decompress(compressed, coords,
                                  speed_bins=self.masked_speed_bins)
            mask = ds[self.speed_name].values < 0.
            ds[self.speed_name].values[mask] = np.nan
            ds[self.direction_name].values[mask] = np.nan
            ds[self.u_name].values[mask] = np.nan
            ds[self.v_name].values[mask] = np.nan
            return ds
