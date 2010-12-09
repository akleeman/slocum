import coards
import numpy as np

from Scientific.IO.NetCDF import _NetCDFFile

from lib import iterlib

# conversion from speed in a given unit to speed in knots
_speed = {'m/s':1.94384449,
          'knot':1.0,
          'mph':0.868976242}

def scale_variable(x, units=None):
    scale = x.scale_factor if hasattr(x, 'scale_factor') else 1.0
    offset = x.add_offset if hasattr(x, 'add_offset') else 0.0
    return np.array(x) * scale + offset

def wind_variable(x, units=None):
    var = scale_variable(x)
    if units:
        var *= (_speed[x.units] / _speed[units])
    return var

class NcdfInterpolator():

    def __init__(self, filename, axesnames, variablename):
        from Scientific.IO.NetCDF import NetCDFFile
        self.file = NetCDFFile(filename, 'r')
        self.var = self.file.variables[variablename]
        self.axesnames = list(axesnames)
        self.axes = dict((x, np.array(self.file.variables[x])) for x in axesnames)
        self.fixed = [x for x in self.var.dimensions if not x in axesnames]
        self.data = scale_variable(self.var)

    def __call__(self, *points):
        """
        @returns: the function value obtained by linear interpolation
        @rtype: number
        @raise TypeError: if the number of arguments (C{len(points)})
            does not match the number of variables of the function
        @raise ValueError: if the evaluation point is outside of the
            domain of definition and no default value is defined
        """
        if len(points) != len(self.axesnames):
            raise TypeError('Wrong number of arguments')
        if len(points) == 1:
            # Fast Pyrex implementation for the important special case
            # of a function of one variable with all arrays of type double.
            period = self.period[0]
            if period is None: period = 0.
            try:
                return _interpolate(points[0], self.axes[self.axesnames[0]],
                                    self.values, period)
            except:
                # Run the Python version if anything goes wrong
                pass

        def get_slice(point, axisname):
            axis = self.axes[axisname]
            ind = int(np.sum(axis <= point))
            if axis[ind] == point:
                return axisname, [ind]
            elif ind == len(axis):
                msg = "%6.3f not in [%6.3f, %6.3f]" % (point, np.min(axis), np.max(axis))
                raise ValueError(msg)
            else:
                return axisname, [ind, ind + 1]

        slices = dict((d, np.arange(len(d))) for d in self.fixed)
        slices.update(map(get_slice, points, self.axes))
        indices = [slices[x] for x in self.var.dimensions]
        shape = [x if y in self.fixed else 1 for x, y in zip(self.data.shape, self.var.dimensions)]
        ret = np.zeros(shape)

        def interp(data, slices, fixed_dims=list()):
            if fixed_dims:
                dim = fixed_dims.pop(0)

        fixed_dims = [i for i, x in enumerate(self.var.dimensions) if x in self.fixed]

        for i, inds in enumerate(indices):
            is_fixed

        import pdb; pdb.set_trace()

        values = self.values[slices]
        for item in neighbours:
            weight = item[1]
            values = (1.-weight)*values[0]+weight*values[1]
        return values