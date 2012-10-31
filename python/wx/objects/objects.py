import numpy as np
import coards
import logging
import datetime
import cStringIO

from bisect import bisect

import wx.objects.conventions as conv

from wx.lib import pupynere, collections, iterlib, numpylib
from wx.objects import core

Course = collections.namedtuple('Course', 'loc speed bearing heading')
Leg = collections.namedtuple('Leg', 'course time wind distance rel_wind_dir wx')

Data = core.Data
Variable = core.Variable

class Wind(object):
    def __init__(self, u=None, v=None):
        self.u = u
        self.v = v
        self.speed = np.sqrt(np.power(u, 2.) + np.power(v, 2.))
        # we need to use negative signs when computing the direction since
        # we actually want to know where the wind is coming from not where
        # the wind is headed
        self.dir = np.arctan2(-u, -v)

        bins = np.arange(-np.pi, np.pi, step=np.pi/4)
        names = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW']
        ind = bisect(bins, self.dir)
        if ind == 0 or self.dir > np.pi:
            self.readable = '-'
        else:
            self.readable = names[ind - 1]

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
        if any(x != y.shape[0] for x, y in zip(self.data.shape, self.dims)):
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

def merge(objects, dim):
    """
    Merges several objects by expanding along dimension 'dim'.  Ie, two objects
    containing the same structure of data but at two separate time steps can be
    merged resulting in a single object containing both time steps.

    Parameters:
    objects : iterable of core.Data
        An iterator of objects to be merged.  Each object must have the same
        variable names and each variable which does not have dim as a dimension
        must be identical across all iterates.
    dim :
        The dimension along which the objects should be merged.

    Example:

    obj = core.Data()
    obj.create_coordinate('time', np.arange(100), record=True)
    obj.create_coordinate('ensemble', np.arange(1000))
    obj.create_variable('test', ('time', 'ensemble'),
            data=np.random.normal(size=(100,1000)))
    merged = merge([v for k, v in obj.iterator('ensemble')],
                       dim='ensemble')
    print merged == obj # True
    """
    def dim_len(x):
        return x.dimensions[dim] if x.dimensions[dim] is not None else 0

    def slice_finder(x):
        s = [None] * len(x.dimensions)
        if dim in x.dimensions:
            i = list(x.dimensions).index(dim)
        else:
            i = None
        return (i, [slice(None)] * len(x.dimensions))

    objects = list(objects)
    # create the merged object by taking the correct number of slices
    # this will allocate each ndarray upfront and in turn speed things up a bit
    m = objects[0].take([0] * sum(map(dim_len, objects)), dim=dim)
    # get a list of all the shared variables and a dictionary for slicing
    to_merge = [name for name, v in m.variables.iteritems()
                if dim in v.dimensions]
    slices_to_merge = dict([(name, slice_finder(m[name]))
            for name in to_merge])
    # make sure that all fixed variables are identical
    fixed = [name for name, v in m.variables.iteritems()
             if dim not in v.dimensions]
    if not all(all(m[x] == obj[x] for x in fixed) for obj in objects):
        for x in fixed:
            if not all(m[x] == obj[x] for obj in objects):
                raise ValueError("variable %s is not identical across objects"
                                 % x)
    i = 0 # index along dim
    for obj in objects:
        if set(m.variables.keys()) != set(obj.variables.keys()):
            # TODO: generalize to remove this requirement
            raise ValueError("All objects must have the same variables")
        # here we iterate through each variable in our current obj and
        # populate corresponding slices of the variables in m
        n = dim_len(obj)
        s = slice(i, i + n)
        for name, v in obj.variables.iteritems():
            if name in to_merge:
                (j, slice_list) = slices_to_merge[name]
                slice_list[j] = s
                try:
                    m[name].data[slice_list] = v.data
                    # This overwriting of attributes is ugly, but
                    # checking attribute equality isn't great either
                    m[name].attributes.update(v.attributes)
                except:
                    raise ValueError(
                            "objects do not have consistent dimensions")
            elif (m[name] != v) or (dim in v.dimensions):
                raise ValueError("objects are incompatible for concatenation")
        i += n # advance to next slice
    for name in to_merge:
        attr_dict = core.AttributesDict(reduce(collections.intersect_dicts,
                                    [obj[name].attributes for obj in objects]))
        object.__setattr__(m[name], 'attributes', attr_dict)
    return m

def normalize_ensemble_passages(passages):
    """
    Takes a set of passages and merges them, creating an ensemble
    dimension.
    """
    passages = list(passages)
    most_steps = max([x.dimensions[conv.STEP] for x in passages])
    nensembles = len(passages)
    def normalize(passages):
        for i, psg in enumerate(passages):
            obj = core.Data()
            obj.create_coordinate(conv.STEP, np.arange(most_steps))
            obj.create_coordinate(conv.ENSEMBLE, np.array([i]))
            nsteps = min([np.sum(np.isfinite(v)) for v in psg.variables.values()
                          if conv.STEP in v.dimensions])
            obj.create_variable(conv.NUM_STEPS,
                                dim=(conv.ENSEMBLE,),
                                data=np.array([nsteps]))
            variables = (var for var in psg.variables.keys()
                         if not var in [conv.STEP, conv.ENSEMBLE, conv.NUM_STEPS])
            for var in variables:
                obj.create_variable(var,
                                    dim=(conv.STEP, conv.ENSEMBLE),
                                    data=numpylib.nans((most_steps, 1)),
                                    attributes=psg[var].attributes)
                obj[var].data[:nsteps, 0] = psg[var].data[:nsteps].flatten()
            yield obj
    return merge(normalize(passages), conv.ENSEMBLE)

def intersect_ensemble_passages(ensemble_passages):
    """
    Returns an ensemble passage object consisting of only the passages that intersect
    """
    def take_common(x, dim, common):
        return x.take(np.sort([np.nonzero(x[dim].data == y)[0][0] for y in common]), conv.STEP)
    passages = [y for x, y in ensemble_passages.iterator(conv.ENSEMBLE)]
    iterlats = (x[conv.LAT].data[:x[conv.NUM_STEPS].data[0]].flatten() for x in passages)
    common_lats = reduce(np.intersect1d, iterlats)
    if not common_lats.size:
        raise ValueError("passages have no overlapping lat lons")
    # paired passages now consists of only similar lats
    paired_passages = [take_common(x, conv.LAT, common_lats) for x in passages]
    # find common longitudes
    iterlons = (x[conv.LON].data[:x[conv.NUM_STEPS].data[0]].flatten() for x in paired_passages)
    common_lons = reduce(np.intersect1d, iterlons)
    if not len(common_lons):
        raise ValueError("passages have no overlapping lat lons")
    # paired passages now consists of similar lats and lons
    paired_passages = [take_common(x, conv.LON, common_lons) for x in passages]
    return normalize_ensemble_passages(paired_passages)

#class Trajectory():
#    path = {}
#
#    def append(self, time, **vals):
#        self.path[time] = vals


    #SEE:  https://cf-pcmdi.llnl.gov/trac/wiki/PointObservationConventions




#    If there is only one trajectory in the file, then a Trajectory dataset follows the same rules above as a Point dataset, except that it must have the global attribute
#
# :cdm_datatype = "Trajectory";
#
#If there are multiple trajectories in the same file, then the trajectories are identified through the trajectory dimension. Use the global attribute
#
#  :trajectoryDimension = "dimName";
#
#to name the trajectory dimension. If there is no such attribute, then there must be a dimension named trajectory.
#
#All Variables with the trajectory dimension.as their outer dimension are considered trajectory Variables, containing information about the trajectory. The number of trajectories in the file will then be the length of the trajectory dimension.
#
#The latitude, longitude, altitude, and time variables must all be observation variables, i.e. have the observation dimension as their outer dimension. There must also exist a trajectory id variable and optionally a trajectory description variable. The trajectory ids must be unique within the file. These can be identified in two ways:
#
#   1. Trajectory variables explicitly named trajectory_id and trajectory_description.
#   2. Global attributes trajectory_id, and trajectory_description whose values are the names of the trajectory id and trajectory description variables.
#
#The observations must be associated with their corresponding trajectory using linked lists, contiguous lists, or multidimensional structures.



#netcdf trajectoryData {
# dimensions:
#   trajectory = 11;
#   record = UNLIMITED;
#
# variables:
#   int trajectory(trajectory); // some kind of id
#   int firstObs(trajectory);
#   int numObs(trajectory);
#
#   int nextObs(record);
#   int trajectoryIndex(record);
#   int time_observation(record);
#   float latitude(record);
#   float longitude(record);
#   int depth(record);
#
#   float obs_data1(record);
#   int obs_data2(record);
#   int obs_data3(record);
#   ...
#}
#
#Contiguous lists look just like linked lists except that you dont need the nextObs variable to store the link, and of course, you have to store the observations contiguously.
