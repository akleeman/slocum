import numpy as np
import coards
import logging
import datetime
import cStringIO

from bisect import bisect

from wx.lib import pupynere, collections, iterlib
from wx.objects import core

Course = collections.namedtuple('Course', 'loc speed bearing heading')
Leg = collections.namedtuple('Leg', 'course time wind distance rel_wind_dir wx')

Data = core.Data
Variable = core.Variable

class Wind(object):
    def __init__(self, u=None, v=None, speed=None, dir=None):
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