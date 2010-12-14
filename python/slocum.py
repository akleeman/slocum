#!/usr/bin/python2.6
"""
Some notes:

This cdo command creates a new ncdf file with the maximum value for a day
cdo timselmax,4 analysis_20091201_v11l30flk.nc analysis_20091201_v11l30flk_max.nc

This cdo command creates a new ncdf file with the mean value for a day
cdo timselmean,4 analysis_20091201_v11l30flk.nc analysis_20091201_v11l30flk_max.nc

get ensemble forecasts from:
  ftp://ftp.cdc.noaa.gov/Datasets.other/map/ENS/
or better yet:
  ftp://polar.ncep.noaa.gov/pub/waves/
the file nww3.all contains all sorts of great stuff

hurricane tracks can be downloaded from:
  http://www.nationalatlas.gov/atlasftp.html
see basemap examples for how to plot them

REQUIRES:
numpy
scipy
coards
basemap
matplotlib
scientific

TODO:
- Take into account leeway/waves to create a heading
- Make sure rhumbline calculations are accurate
"""

import os
import sys
import numpy as np
import coards
import logging
import datetime

from matplotlib import pyplot as plt
from Scientific.IO.NetCDF import NetCDFFile
from mpl_toolkits.basemap import Basemap

from lib import objects, ncdflib, navigation

MAX_BOAT_SPEED = 6
MAX_POINTING = 0.3490658503988659 # 30/180*pi
MIN_WIND_SPEED = 3
REQ_WIND_SPEED = 15
MAX_WIND_SPEED = 40
SECONDS_IN_HOUR = 3600

def directional_max_speed(deg_off_wind):
    if deg_off_wind > MAX_POINTING:
        return MAX_BOAT_SPEED
    else:
        return (0.5 + 0.5* deg_off_wind / MAX_POINTING) * MAX_BOAT_SPEED

def boat_speed(wind, bearing):
    deg_off_wind = relative_wind(wind, bearing)
    if wind.speed > MIN_WIND_SPEED and wind.speed <= REQ_WIND_SPEED:
        speed = np.sqrt((wind.speed - MIN_WIND_SPEED) / (REQ_WIND_SPEED - MIN_WIND_SPEED))
        return directional_max_speed(deg_off_wind) * speed
    elif wind.speed > MIN_WIND_SPEED and wind.speed <= MAX_WIND_SPEED:
        return directional_max_speed(deg_off_wind)
    else:
        return 0

def simulate_passage(waypoints):
    testfile = os.path.join(os.path.dirname(__file__), '../data/analysis_20091201_v11l30flk.nc')
    nc = NetCDFFile(filename=testfile, mode='r')
    timevar = nc.variables['time']
    time = [coards.from_udunits(t, timevar.units) for t in np.array(timevar)]
    delta_t = time[1] - time[0]
    dims = [nc.variables['lat'], nc.variables['lon']]
    uwnd = objects.DataField(ncdflib.wind_variable(nc.variables['uwnd'])[0], dims)
    vwnd = objects.DataField(ncdflib.wind_variable(nc.variables['vwnd'])[0], dims)

    now = time[0]
    here = waypoints.pop(0)
    dt = delta_t
    for destination in waypoints:
        while not here == destination:
            # determine the bearing (following a rhumbline) between here and the end
            bearing = navigation.rhumbline_bearing(here, destination)
            # get the wind and use that to compute the boat speed
            wind = objects.Wind(uwnd(here.lat, here.lon), vwnd(here.lat, here.lon))
            deg_off_wind = relative_wind(wind, bearing)
            speed = max(boat_speed(wind, bearing), 0.1)
            # given our speed how far can we go in one timestep?
            distance = speed * float(dt.seconds / SECONDS_IN_HOUR)
            remaining = navigation.rhumbline_distance(here, destination)
            #import pdb; pdb.set_trace()
            if distance > remaining:
                here = destination
                required_time = int(dt.seconds * remaining / distance)
                now = now + datetime.timedelta(seconds=required_time)
                distance = remaining
                dt = delta_t - datetime.timedelta(seconds=required_time)
            else:
                # and once we know how far, where does that put us in terms of lat long
                here = navigation.rhumbline_path(here, bearing)(distance)
                now = now + dt
                dt = delta_t
            yield objects.Leg(here, now, wind, speed, distance, bearing, deg_off_wind)

def optimal_passage(start, end, resol=50):
    c1 = objects.LatLon(start.lat, end.lon)
    c2 = objects.LatLon(end.lat, start.lon)

    def get_time(x):
        passage = list(simulate_passage([start, x, end]))
        passage_time = (passage[-1].time - passage[0].time)
        return passage_time

    waypoints = [objects.LatLon(x*c1.lat + (1.-x)*c2.lat, x*c1.lon + (1.-x)*c2.lon) for x in np.arange(0., 1., step=1./resol)]
    times = [(x, get_time(x)) for x in waypoints]
    plt.plot([x[1].seconds for x in times])
    plt.show()
    optimal_waypoint = waypoints[np.argmin([x[1].seconds for x in times])]
    return simulate_passage([start, optimal_waypoint, end])

def main():
    start = objects.LatLon(18.5, -155) # hawaii
    mid = objects.LatLon(18.5, -175) # waypoint
    end = objects.LatLon(-16.5, -175) # fiji
    passage = list(optimal_passage(start, end))
    plot_passage(passage)
    import pdb; pdb.set_trace()
    return 0

if __name__ == "__main__":
    sys.exit(main())
