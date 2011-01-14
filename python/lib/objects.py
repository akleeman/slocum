import numpy as np
import coards
import datetime
import cStringIO

from lib import pupynere, collections

LatLon = collections.namedtuple('LatLon', 'lat lon')
Course = collections.namedtuple('Course', 'loc speed bearing heading')
Leg = collections.namedtuple('Leg', 'course time wind distance rel_wind_dir etc')

_speed = {'m/s':1.94384449,
          'm s-1':1.94384449,
          'knot':1.0,
          'mph':0.868976242}

_speed_unit = 'knot'

def make_udunits(units):
    'hour since %Y-%m-%dT%H:%M:%SZ'

def from_udunits(units, x):
    try:
        coards.from_udunits(units, x)
    except:
        now = datetime.datetime.strptime(units, 'hour since %Y-%m-%dT%H:%M:%SZ')
        return now + datetime.timedelta(seconds=int(3600*x))

class Wind(object):
    def __init__(self, u=None, v=None, speed=None, dir=None):
        self.u = u
        self.v = v
        self.speed = np.sqrt(np.power(u, 2.) + np.power(v, 2.))
        # we need to use negative signs when computing the direction since
        # we actually want to know where the wind is coming from not where
        # the wind is headed
        self.dir = np.arctan2(-u, -v)

        d = self.dir - np.pi/8
        if d > 3*np.pi/4:
            self.readable = 'S'
        elif d > np.pi/2:
            self.readable = 'SE'
        elif d > np.pi/4:
            self.readable = 'E'
        elif d > 0:
            self.readable = 'NE'
        elif d > -np.pi/4:
            self.readable = 'N'
        elif d > -np.pi/2:
            self.readable = 'NW'
        elif d > -3*np.pi/4:
            self.readable = 'W'
        elif d > -np.pi:
            self.readable = 'SW'
        else:
            self.readable = '-'

class LatLon(object):
    def __init__(self, lat, lon):
        self.lat = np.mod(lat + 90, 180) - 90
        self.lon = np.mod(lon, 360)

    def copy(self):
        return LatLon(self.lat, self.lon)

    def as_rad(self):
        return LatLon(np.deg2rad(self.lat), np.deg2rad(self.lon))

    def as_deg(self):
        return LatLon(np.rad2deg(self.lat), np.rad2deg(self.lon))

    def __add__(self, other):
        return LatLon(self.lat + other.lat, self.lon + other.lon)

    def __sub__(self, other):
        return LatLon(self.lat - other.lat, self.lon - other.lon)

class DataObject(pupynere.netcdf_file):

    def __getitem__(self, index):
        return self.variables[index]

    def __close__(self):
        pass

    def iterator(self, dim=None, vars=None):
        """
        Iterates over variable 'var' along dimension 'dim' returning a tuple of
        (dimension_value, slice_along_dimension).

        var - The variable to iterate over.  Default behavior is to iterate over
            the unique non-dimensional variable, failing if there is more than one.
        dim - The dimension to iterator along.  Defaults to the record dimension
        """
        if not dim:
            dim = [x for x in self.dimensions if self[x].isrec]
            assert len(dim) == 1
            dim = dim[0]

        assert dim in self.variables
        for i in range(len(self[dim].data)):
            d = slice_variable(self.variables[dim], dim, i)
            obj = self.slice(dim, i, vars=vars)
            yield d, obj

    def slice(self, dim, ind, vars=None):
        """
        dim - a dimension to slice along
        ind - the index along dimension dim you want to keep
        vars - the variables to retain in the slice (defaults to all)

        returns a data object for which all variables containing 'dim' have been
        sliced.
        """
        assert dim in self.dimensions
        dims = [d for d in self.dimensions if not d == dim]
        obj = self.select(vars=[], dims=dims)
        if not vars:
            vars = self.variables.keys()
        else:
            assert isinstance(vars, list)
        for v in vars:
            if dim in self.variables[v].dimensions:
                obj.variables[v] = slice_variable(self.variables[v], dim, ind)
            else:
                obj.variables[v] = copy_variable(self.variables[v])
        return obj

    def select(self, vars, dims=None):
        """
        Creates a new data object containing copies of the variables 'vars'
        """
        obj = cStringIO.StringIO()
        obj.close()
        ret = DataObject(obj, mode='w')
        vars = set(vars)
        for d in self.dimensions.keys():
            if d in self.variables:
                vars.add(d)
            ret.createDimension(d, self.dimensions[d])
        for v in vars:
            ret.variables[v] = copy_variable(self.variables[v])
        ret._attributes = self._attributes.copy()
        return ret

    def rename(self, var, name):
        """
        Renames a variable 'var' to 'name' replacing any occurrence in dimensions
        """
        assert var in self.variables
        self.variables[name] = self.variables.pop(var)

        if var in self.dimensions:
            self.dimensions[name] = self.dimensions.pop(var)
            self._dims[self._dims.index(var)] = name
            for v in self.variables.values():
                if var in v.dimensions:
                    ind = list(v.dimensions).index(var)
                    dims = list(v.dimensions)
                    dims[ind] = name
                    v.dimensions = tuple(dims)

def copy_variable(var, data=None, dimensions=None):
    if data is None:
        data = var.data.copy()
    typecode = var._typecode
    shape = data.shape
    if dimensions is None:
        dimensions = var.dimensions
    attributes = var._attributes.copy()
    return pupynere.netcdf_variable(data, typecode, shape, dimensions, attributes)

def slice_variable(var, dim, ind):
    """
    var - a netcdf_variable
    dim - the dimension along which you would like to select index 'ind'
    ind - index to select

    Returns a new netcdf_variable with dimensions that were reduced along the
    given dimension.
    """
    dim_ind = list(var.dimensions).index(dim)
    assert dim_ind >= 0
    slicer = len(var.dimensions) * [slice(None, None, None)]
    slicer[dim_ind] = ind

    data = var.data[slicer]
    typecode = var._typecode
    shape = data.shape
    dimensions = [x for x in var.dimensions if not x == dim]
    attributes = var._attributes
    try:
        attributes.pop('dimensions')
    except:
        pass
    return pupynere.netcdf_variable(data, typecode, shape, dimensions, attributes)

def normalize_variable(var):
    """
    Scales, shifts and converts the variable to standard units
    """
    var = copy_variable(var, data = var.data.astype('d'))
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

    if var.units in _speed:
        mult = (_speed[var.units] / _speed[_speed_unit])
        var.units = _speed_unit
    else:
        mult = 1.0

    var.data[:] = mult * (var.data[:] * scale + offset)

    return var

class DataField(object):
    def __init__(self, data_obj, var, dims=None, units=None):
        var = data_obj.variables[var]
        if not dims:
            dims = var.dimensions
        dims = [data_obj.variables[x].data for x in dims]

        self.var = normalize_variable(var)
        self.dims = dims
        self.data = self.var.data

        for i, dim in enumerate(self.dims):
            inds = np.argsort(dim)
            self.dims[i] = dim[inds]
            self.data = self.data.take(inds, axis=i)

        assert self.data.ndim == 2 # we only support two dimensions for now
        if np.any([x != y.shape[0] for x, y in zip(self.data.shape, self.dims)]):
            import pdb; pdb.set_trace()
            raise ValueError("dim lengths and array shape must agree")

    def __call__(self, *points):
        """
        @returns: the function value obtained by linear interpolation
        @rtype: number
        @raise TypeError: if the number of arguments (C{len(points)})
            does not match the number of variables of the function
        @raise ValueError: if the evaluation point is outside of the
            domain of definition and no default value is defined
        """
        if len(points) != len(self.dims):
            raise TypeError('Wrong number of arguments')
        assert self.data.ndim <= 2
        def get_nhbrs(point, dim):
            if not point:
                return None
            j = int(np.sum(dim <= point))
            if j == 0:
                msg = "%6.3f not in [%6.3f, %6.3f]" % (point, np.min(dim), np.max(dim))
                raise ValueError(msg)
            i = j - 1
            if dim[i] == point:
                return (i, j), 1.0
            elif j == len(dim):
                msg = "%6.3f not in [%6.3f, %6.3f]" % (point, np.min(dim), np.max(dim))
                raise ValueError(msg)
            else:
                alpha = np.abs(point - dim[j])/np.abs(dim[i] - dim[j])
                assert alpha <= 1
                return (i, j), alpha

        nhbrs = map(get_nhbrs, points, self.dims)
        ret = self.data.copy()
        for i, nhbr in enumerate(nhbrs):
            ret = ret.take(nhbr[0], axis=i)
        ret = ret.T
        for i, nhbr in enumerate(nhbrs):
            ret = np.dot(ret, [nhbr[1], 1.-nhbr[1]])
            if ret.ndim == 2:
                # for some reason np.dot doesn't always reduce the dimension
                # so we need to do it manually from time to time
                ret = ret[..., 0]
        return ret
