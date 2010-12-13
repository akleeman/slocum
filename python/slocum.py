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
    deg_off_wind = np.abs(wind.dir - bearing)
    if deg_off_wind > np.pi: deg_off_wind = 2*np.pi - deg_off_wind
    if wind.knots > MIN_WIND_SPEED and wind.knots <= REQ_WIND_SPEED:
        speed = np.sqrt((wind.knots - MIN_WIND_SPEED) / (REQ_WIND_SPEED - MIN_WIND_SPEED))
        return directional_max_speed(deg_off_wind) * speed
    elif wind.knots > MIN_WIND_SPEED and wind.knots <= MAX_WIND_SPEED:
        return directional_max_speed(deg_off_wind)
    else:
        return 0

def passage(waypoints):
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
    for destination in waypoints:
        while not here == destination:
            # determine the bearing (following a rhumbline) between here and the end
            bearing = navigation.rhumbline_bearing(here, destination)
            # get the wind and use that to compute the boat speed
            wind = objects.Wind(uwnd(here.lat, here.lon), vwnd(here.lat, here.lon))
            speed = boat_speed(wind, bearing)
            # given our speed how far can we go in one timestep?
            distance = speed * (delta_t.seconds / SECONDS_IN_HOUR)
            remaining = navigation.rhumbline_distance(here, destination)
            if distance > remaining:
                here = destination
                now = now + datetime.timedelta(seconds=delta_t.seconds * remaining / distance)
                distance = remaining
            else:
                # and once we know how far, where does that put us in terms of lat long
                here = navigation.rhumbline_path(here, bearing)(distance)
                now = now + delta_t
            yield here, now, wind, speed, distance

def main():

    start = objects.LatLon(18.5, -155) # hawaii
    end = objects.LatLon(-16.5, -175) # fiji

    trip = list(passage(start, end))
    print trip[len(trip) - 1][1] - trip[0][1]

    mid = objects.LatLon(0.5*(start.lat + end.lat), 0.5*(start.lon + end.lon))
    m = Basemap(projection='ortho',lon_0=mid.lon,lat_0=mid.lat,resolution='l')
    #m = Basemap(projection='ortho',lon_0=mid.lon,lat_0=mid.lat,resolution=None)
#    m = Basemap(projection='robin',
#                llcrnrlon=llcrnrlon,llcrnrlat=llcrnrlat,
#                urcrnrlon=urcrnrlon,urcrnrlat=urcrnrlat,
#                lat_0=0.5*(llcrnrlat+urcrnrlat),
#                lon_0=0.5*(llcrnrlon + urcrnrlon))
    lons = [x[0].lon for x in trip]
    lats = [x[0].lat for x in trip]
    x,y = m(lons,lats)
    # draw colored markers.
    # use zorder=10 to make sure markers are drawn last.
    # (otherwise they are covered up when continents are filled)
    m.scatter(x,y,10,edgecolors='none',zorder=10)

    # map with continents drawn and filled.
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcountries()
    # draw parallels and meridians.
    m.drawparallels(np.arange(-90.,120.,30.))
    m.drawmeridians(np.arange(0.,420.,60.))
    m.drawmapboundary(fill_color='aqua')
    plt.show()
    return 0

if __name__ == "__main__":
    sys.exit(main())
