import numpy as np
import coards
import logging
import datetime
import cStringIO

from bisect import bisect

from wx.lib import pupynere, collect, iterlib

Course = collect.namedtuple('Course', 'loc speed bearing heading')
Leg = collect.namedtuple('Leg', 'course time wind distance rel_wind_dir wx')

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
        if ind > 0 or self.dir > np.pi:
            self.readable = names[ind - 1]
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