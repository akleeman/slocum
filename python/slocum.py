#!/usr/bin/python2.6
import os
import sys
import numpy as np

from matplotlib import pyplot as plt
from Scientific.IO.NetCDF import NetCDFFile

from lib import objects, ncdflib, navigation

def wind_speed(u, v):
    return np.sqrt(np.power(u, 2.) + np.power(v, 2.))

def wind_dir(u, v):
    return np.arctan(u/v)

def directional_speed(deg_off_wind):
    max_pointing = 0.3490658503988659 # 20/180*pi
    if deg_off_wind > max_pointing:
        return 6
    else:
        return 3 + 3* deg_off_wind / max_pointing

def leg(uwind, vwind, heading, distance):
    wind = objects.Wind(uwind, vwind, wind_speed(uwind, vwind), wind_dir(uwind, vwind))
    boat_speed = directional_speed(np.abs(wind.dir - heading))
    time = distance / boat_speed
    return (wind, time)

def main():
    testfile = os.path.join(os.path.dirname(__file__), '../data/analysis_20091201_v11l30flk.nc')
    nc = NetCDFFile(filename=testfile, mode='r')

    dims = [nc.variables['lat'], nc.variables['lon']]

    uwnd = objects.DataField(ncdflib.wind_variable(nc.variables['uwnd'])[0], dims)
    vwnd = objects.DataField(ncdflib.wind_variable(nc.variables['vwnd'])[0], dims)

    pts = 100
    start = objects.LatLon(18.5, -155) # hawaii
    end = objects.LatLon(-16.5, -175) # fiji

    diff = end - start

    bearing = navigation.bearing(start, end)
    distance = navigation.rhumbline_distance(start, end)
    rhumb = navigation.rhumbline_path(start, bearing)
    step = distance/pts
    points = map(rhumb, np.arange(0., distance + step, step=distance/pts))

    uwinds = [vwnd(p.lat, p.lon) for p in points]
    vwinds = [uwnd(p.lat, p.lon) for p in points]
    legs = [leg(u, v, bearing, step) for u, v in zip(uwinds, vwinds)]

    plt.plot([x[0].knots for x in legs])
    plt.show()

    val1 = obj(-73.625, 184.125)
    val2 = obj(-73.375, 184.125)
    val3 = obj(-73.5, 184.125)

    assert val3 == 0.5 * (val2 + val1)
    # from ncks
    #time[0]=200880 lat[19]=-73.625 lon[736]=184.125 uwnd[28096]=-1241 m/s
    #assert val1 == -1241

    return 0

if __name__ == "__main__":
    sys.exit(main())